# =========================================================
# Diffusion Priors Pretraining (Lat-Act / Src-Act / Tgt-Act)
# - Standalone script to pretrain three denoisers as "priors" for ACTIONS
# - Matches logging / directory style used by train_align.py
# - Uses DDPM by default (configurable); EDM-style sigma schedule optional
# - Normalization is applied ONLY inside the prior models; encoder/decoder spaces unchanged
# - Saves: models/*.pt (+EMA), stats/normalization.yml, stats/schedule.yml, stats/dataset.yml, params.yml, tb/
# =========================================================

import os
import math
import time
import json
import pathlib
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

from tqdm import trange

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ruamel.yaml import YAML

import utils  # must exist in your repo (same as align code)
import replay_buffer  # must exist in your repo (same as align code)

# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def infer_action_dim(env) -> int:
    if hasattr(env, "action_space") and getattr(env.action_space, "shape", None) is not None:
        return int(env.action_space.shape[0])
    if hasattr(env, "action_spec"):
        spec = env.action_spec
        spec = spec() if callable(spec) else spec
        low, high = spec
        return int(np.prod(low.shape))
    if hasattr(env, "action_dim"):
        return int(env.action_dim)
    if hasattr(env, "robots") and len(getattr(env, "robots", [])) > 0:
        ctrl = getattr(env.robots[0], "controller", None)
        if ctrl is not None and hasattr(ctrl, "control_dim"):
            return int(ctrl.control_dim)
    raise AttributeError("Cannot infer action dimension for this env.")

def infer_action_shape(env) -> tuple:
    return (infer_action_dim(env),)

def safe_reset(env, max_tries=50, reseed=True, sleep=0.0):
    """Robosuite reset with randomization error handling (same style as train_align)."""
    try:
        from robosuite.utils.errors import RandomizationError
    except Exception:
        class RandomizationError(Exception):
            pass
    import time as _t

    tries = 0
    while True:
        try:
            return env.reset()
        except RandomizationError:
            tries += 1
            if tries >= max_tries:
                raise
            if reseed and hasattr(env, "seed"):
                env.seed(int(_t.time() * 1e6) % (2**31 - 1))
            if sleep > 0:
                _t.sleep(sleep)


def build_buffer_dir(root: str, task: str, robot: str, controller: str) -> str:
    return str(pathlib.Path(root) / task / robot / controller)


# ---------------------------------------------------------
# Time / Noise Schedules
# ---------------------------------------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal (improved DDPM). Returns betas (T,)."""
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 1e-8, 0.999)
    return betas


def make_ddpm_schedule(T: int, beta_type: str = "cosine") -> Dict[str, torch.Tensor]:
    if beta_type == "cosine":
        betas = cosine_beta_schedule(T)
    else:
        # fallback: linear
        betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
    }


# ---------------------------------------------------------
# Positional / Time Embedding
# ---------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        # t is in [0, 1] or arbitrary scalar; we map to [0, 1] range embeddings
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device)
        )
        args = t[..., None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb


# ---------------------------------------------------------
# Small MLP Denoiser (Noise Predictor)
# ---------------------------------------------------------
class Denoiser(nn.Module):
    """
    Input: x (B, Dx), optional cond c (B, Dc), time embedding (B, Dt)
    Output: predicted noise eps_hat (same shape as x)
    Internally, this model expects *normalized* inputs. Normalization is performed
    by the wrapper (PriorModel) using stored mu/std.
    """
    def __init__(self, x_dim: int, c_dim: int, t_dim: int, hidden: int = 256, n_layers: int = 3):
        super().__init__()
        in_dim = x_dim + (c_dim if c_dim > 0 else 0) + t_dim
        # utils.build_mlp 的旧签名是位置参数: (in_dim, out_dim, n_layers, hidden_dim, ...)
        self.net = utils.build_mlp(
            in_dim,        # input_dim
            x_dim,         # output_dim
            n_layers,      # number of layers
            hidden,        # hidden width
            activation='relu',
            output_activation='identity',
            batch_norm=False,
        )

    def forward(self, x_in: torch.Tensor):
        return self.net(x_in)


class PriorModel(nn.Module):
    """
    Wraps Denoiser with: normalization for x and c, time embedding, and optional condition.
    Exposes a uniform forward(input_x, cond_c, t_idx_or_sigma) that returns predicted noise.
    - mode='ddpm': t is integer index in [0, T-1] or float in [0,1]; we embed t/T.
    - mode='edm':  t is sigma (float), we embed log(sigma).
    """
    def __init__(self, x_dim: int, c_dim: int, hidden: int, n_layers: int,
                 use_condition: bool, time_embed_dim: int,
                 mode: str = 'ddpm'):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim if use_condition else 0
        self.use_condition = use_condition
        self.mode = mode
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.denoiser = Denoiser(x_dim=x_dim, c_dim=self.c_dim, t_dim=time_embed_dim,
            hidden=hidden, n_layers=n_layers)
        # Buffers for normalization
        self.register_buffer('mu_x', torch.zeros(x_dim))
        self.register_buffer('std_x', torch.ones(x_dim))
        self.register_buffer('mu_c', torch.zeros(self.c_dim if self.c_dim>0 else 1))
        self.register_buffer('std_c', torch.ones(self.c_dim if self.c_dim>0 else 1))

    def set_norm(self, mu_x: np.ndarray, std_x: np.ndarray,
                 mu_c: Optional[np.ndarray] = None, std_c: Optional[np.ndarray] = None):
        self.mu_x = torch.as_tensor(mu_x, dtype=torch.float32, device=self.mu_x.device)
        self.std_x = torch.as_tensor(std_x, dtype=torch.float32, device=self.std_x.device)
        if self.use_condition and self.c_dim > 0 and (mu_c is not None) and (std_c is not None):
            self.mu_c = torch.as_tensor(mu_c, dtype=torch.float32, device=self.mu_x.device)
            self.std_c = torch.as_tensor(std_c, dtype=torch.float32, device=self.mu_x.device)

    def _norm_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu_x) / (self.std_x + 1e-8)

    def _norm_c(self, c: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if (not self.use_condition) or (self.c_dim == 0) or (c is None):
            return None
        return (c - self.mu_c) / (self.std_c + 1e-8)

    def _embed_t(self, t_value: torch.Tensor) -> torch.Tensor:
        # t_value is float scalar per-batch item; for ddpm use t/T in [0,1]; for edm use log(sigma)
        return self.time_embed(t_value)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor], t_value: torch.Tensor) -> torch.Tensor:
        x_n = self._norm_x(x)
        if self.use_condition and self.c_dim > 0:
            c_n = self._norm_c(c)
            assert c_n is not None
            inp = torch.cat([x_n, c_n, self._embed_t(t_value)], dim=-1)
        else:
            inp = torch.cat([x_n, self._embed_t(t_value)], dim=-1)
        return self.denoiser(inp)


# ---------------------------------------------------------
# EMA Helper
# ---------------------------------------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.m = model
        self.shadow: Dict[str, torch.Tensor] = {}
        self.decay = decay
        self._init()

    def _init(self):
        for n, p in self.m.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    @torch.no_grad()
    def update(self):
        for n, p in self.m.named_parameters():
            if (not p.requires_grad) or (n not in self.shadow):
                continue
            self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, target: nn.Module):
        for n, p in target.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])


# ---------------------------------------------------------
# Data plumbing (ReplayBuffer-based as in align)
# ---------------------------------------------------------
@dataclass
class DomainDims:
    robot_obs_dim: int
    cond_dim: int
    act_dim: int


def probe_dims(env_name: str, robot: str, controller: str, env_kwargs: Dict[str, Any],
               robot_obs_keys, cond_keys) -> DomainDims:
    probe_env = utils.make_robosuite_env(
        env_name,
        robots=robot,
        controller_type=controller,
        **env_kwargs,
    )
    probe = safe_reset(probe_env)
    rob = np.concatenate([probe[k] for k in robot_obs_keys])
    cond = np.concatenate([probe[k] for k in cond_keys])  # 多键拼接
    act_dim = infer_action_dim(probe_env)
    return DomainDims(robot_obs_dim=rob.shape[0], cond_dim=cond.shape[0], act_dim=act_dim)


def build_env_for_buffer(env_name: str, robot: str, controller: str, env_kwargs: Dict[str, Any],
                         robot_obs_keys, cond_keys):
    # This env exposes observation space = concat(robot_obs_keys + cond_keys)
    obs_keys = list(robot_obs_keys) + list(cond_keys)
    env = utils.make(
        env_name,
        robots=robot,
        controller_type=controller,
        obs_keys=obs_keys,
        seed=0,
        **env_kwargs,
    )
    return env, obs_keys


def build_replay_buffer(buffer_dir: str, env, obs_keys, device: torch.device, batch_size: int) -> replay_buffer.ReplayBuffer:
    buf = replay_buffer.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=int(1e7),
        batch_size=batch_size,
        device=device,
    )
    demos = utils.load_episodes(pathlib.Path(buffer_dir), obs_keys)
    buf.add_rollouts(demos)
    return buf


# ---------------------------------------------------------
# Training Loop per-prior
# ---------------------------------------------------------
@dataclass
class TrainCfg:
    steps: int
    batch_size: int
    lr: float
    n_layers: int
    hidden_dim: int
    ema_decay: float
    schedule_type: str  # 'ddpm' or 'edm'
    T: int              # for ddpm
    beta: str           # 'cosine' or 'linear' (ddpm)
    loss_weight: str    # 'sigma2' or 'snr' (ddpm)


def ddpm_loss_weight(alpha_cumprod_t: torch.Tensor, betas_t: torch.Tensor, mode: str) -> torch.Tensor:
    sigma2 = 1.0 - alpha_cumprod_t
    if mode == 'sigma2':
        return sigma2
    elif mode == 'snr':
        snr = alpha_cumprod_t / torch.clamp(sigma2, min=1e-8)
        return 1.0 / torch.clamp(snr, min=1e-8)
    else:
        return torch.ones_like(alpha_cumprod_t)


def train_one_prior(
    name: str,
    prior: PriorModel,
    make_batch_fn,  # returns (x, c)
    writer: SummaryWriter,
    device: torch.device,
    out_dir: pathlib.Path,
    cfg: TrainCfg,
    log_freq: int,
    save_interval: int,
    val_sampler=None,
):
    print(f"[stage] start training {name}")
    prior = prior.to(device)
    prior.train()
    opt = torch.optim.Adam(prior.parameters(), lr=cfg.lr)
    ema = EMA(prior, decay=cfg.ema_decay)
    prior_ema = PriorModel(prior.x_dim, prior.c_dim, cfg.hidden_dim, cfg.n_layers,
                           prior.use_condition, prior.time_embed.dim, mode=cfg.schedule_type).to(device)
    ema.copy_to(prior_ema)

    # schedule
    if cfg.schedule_type == 'ddpm':
        sched = make_ddpm_schedule(cfg.T, cfg.beta)
        betas = sched['betas'].to(device)
        alphas = sched['alphas'].to(device)
        alphas_cumprod = sched['alphas_cumprod'].to(device)
        T = cfg.T
    else:
        sigma_min, sigma_max = 0.01, 1.2
        T = None

    global_step = 0
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_ckpt(tag: str):
        torch.save(prior.state_dict(), out_dir / f"{name}.pt")
        ema.copy_to(prior_ema)
        torch.save(prior_ema.state_dict(), out_dir / f"{name}_ema.pt")

    pbar = trange(cfg.steps, desc=f"{name}", dynamic_ncols=True)
    running_loss = None

    for _ in pbar:
        x, c = make_batch_fn()
        if cfg.schedule_type == 'ddpm':
            t_idx = torch.randint(0, T, (x.shape[0],), device=device)
            a_bar_t = alphas_cumprod[t_idx]
            a_bar_t_sqrt = torch.sqrt(a_bar_t)
            one_minus_a_bar_t_sqrt = torch.sqrt(1.0 - a_bar_t)
            noise = torch.randn_like(x)
            x_t = a_bar_t_sqrt.unsqueeze(-1) * x + one_minus_a_bar_t_sqrt.unsqueeze(-1) * noise
            t_float = (t_idx.to(torch.float32) + 0.5) / float(T)
            eps_hat = prior(x_t, c, t_float)
            if cfg.loss_weight in ('sigma2','snr'):
                w = ddpm_loss_weight(a_bar_t, betas[t_idx], cfg.loss_weight).unsqueeze(-1)
                loss = F.mse_loss(eps_hat, noise, reduction='none')
                loss = (w * loss).mean()
            else:
                loss = F.mse_loss(eps_hat, noise)
        else:
            u = torch.rand(x.shape[0], device=device)
            sigma = torch.exp(torch.log(torch.tensor(0.01, device=device)) * (1-u) +
                              torch.log(torch.tensor(1.2, device=device)) * u)
            noise = torch.randn_like(x)
            x_noisy = x + sigma.unsqueeze(-1) * noise
            t_val = torch.log(sigma + 1e-8)
            eps_hat = prior(x_noisy, c, t_val)
            loss = (sigma.unsqueeze(-1)**2 * (eps_hat - noise)**2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update()

        running_loss = loss.item() if running_loss is None else (0.98*running_loss + 0.02*loss.item())
        pbar.set_postfix_str(f"loss={running_loss:.6f}")

        if (global_step % log_freq) == 0:
            writer.add_scalar(f"train/{name}/loss", loss.item(), global_step)

        if (global_step % save_interval) == 0 and global_step > 0:
            save_ckpt(tag=f"step_{global_step}")

        global_step += 1

    save_ckpt(tag="final")
    print(f"[stage] finished training {name}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='YAML config path')
    return p.parse_args()


def main():
    args = parse_args()
    yaml = YAML(typ='safe')
    params = yaml.load(open(args.config))

    # ------------------------------
    # Device & seeds
    # ------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    seed = int(params.get('seed', 0))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # ------------------------------
    # Build dataset paths from config (auto-discovery)
    # ------------------------------
    expert_folder = params.get('expert_folder', 'human_demonstrations')
    task_name = params.get('task_name', params.get('env_name', 'Task'))

    src_robot = params['src_env']['robot']
    tgt_robot = params['tgt_env']['robot']
    controller = params['src_env']['controller_type']  # assume same for tgt per your config
    _raw_cond = params.get('cond_keys', params.get('cond_key', 'robot0_eef_pos'))
    cond_keys = _raw_cond if isinstance(_raw_cond, (list, tuple)) else [_raw_cond]

    src_buffer_dir = build_buffer_dir(expert_folder, task_name, src_robot, controller)
    tgt_buffer_dir = build_buffer_dir(expert_folder, task_name, tgt_robot, controller)

    # ------------------------------
    # Logging dirs (align style)
    # ------------------------------
    logdir_prefix = pathlib.Path(params.get('logdir_prefix') or pathlib.Path(__file__).parent)
    data_path = logdir_prefix / 'logs' / time.strftime("%m.%d.%Y")
    logdir = '_'.join([
        time.strftime("%H-%M-%S"),
        params['env_name'],
        src_robot,
        params['src_env']['controller_type'],
        tgt_robot,
        params['tgt_env']['controller_type'],
        params.get('suffix', 'diffusion_pretrain')
    ])
    logdir = data_path / logdir
    logdir.mkdir(parents=True, exist_ok=True)

    # dump params snapshot
    import yaml as pyyaml
    with open(logdir / 'params.yml', 'w') as fp:
        pyyaml.safe_dump(params, fp, sort_keys=False)

    model_dir = logdir / 'models'
    stats_dir = logdir / 'stats'
    tb_dir = logdir / 'tb'
    cache_dir = logdir / 'caches'
    model_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(tb_dir))

    # ------------------------------
    # Probe dynamic dims per-domain
    # ------------------------------
    env_kwargs = params.get('env_kwargs', {})
    src_dims = probe_dims(params['env_name'], src_robot, params['src_env']['controller_type'], env_kwargs,
                          params['src_env']['robot_obs_keys'], cond_keys)
    tgt_dims = probe_dims(params['env_name'], tgt_robot, params['tgt_env']['controller_type'], env_kwargs,
                          params['tgt_env']['robot_obs_keys'], cond_keys)

    # 输出 cond key 与维度（按你的要求）
    print(f"[config] cond_keys={cond_keys}, src_cond_dim={src_dims.cond_dim}, tgt_cond_dim={tgt_dims.cond_dim}")

    # ------------------------------
    # Build envs exposing robot_obs + cond for buffers
    # ------------------------------
    src_env, src_obs_keys = build_env_for_buffer(params['env_name'], src_robot, params['src_env']['controller_type'],
                                                 env_kwargs, params['src_env']['robot_obs_keys'], cond_keys)
    tgt_env, tgt_obs_keys = build_env_for_buffer(params['env_name'], tgt_robot, params['tgt_env']['controller_type'],
                                                 env_kwargs, params['tgt_env']['robot_obs_keys'], cond_keys)

    # ------------------------------
    # Build replay buffers (as in align)
    # ------------------------------
    batch_size = int(params.get('diffusion', {}).get('latent', {}).get('batch_size', params.get('batch_size', 4096)))
    src_buffer = build_replay_buffer(src_buffer_dir, src_env, src_obs_keys, device, batch_size)
    tgt_buffer = build_replay_buffer(tgt_buffer_dir, tgt_env, tgt_obs_keys, device, batch_size)

    # ------------------------------
    # Load source ACT encoder (A_src) to produce latent z_a
    # （这里：A_src 的输入 = robot_obs (cos/sin 拼好) + action_去夹爪）
    # ------------------------------
    lat_act_dim = int(params['lat_act_dim'])  # 供后续 prior_lat_act 使用

    src_act_arm_dim = src_dims.act_dim - 1
    tgt_act_arm_dim = tgt_dims.act_dim - 1
    print(f"[dims] src: robot_obs_dim={src_dims.robot_obs_dim}, act_dim={src_dims.act_dim}, act_arm_dim={src_act_arm_dim}")
    print(f"[dims] tgt: robot_obs_dim={tgt_dims.robot_obs_dim}, act_dim={tgt_dims.act_dim}, act_arm_dim={tgt_act_arm_dim}")

    src_model_dir = pathlib.Path(params['src_model_dir'])
    print("[diffusion] building A_src (local ActEncoder) ...")
    class ActEncoder(nn.Module):
        def __init__(self, in_dim, out_dim, n_layers=3, hidden=256, batch_norm=False):
            super().__init__()
            self.net = utils.build_mlp(
                in_dim, out_dim, n_layers, hidden,
                activation='relu', output_activation='tanh', batch_norm=batch_norm
            )
        def forward(self, x):
            return self.net(x)

    # A_src 输入 = robot_obs_dim + act_arm_dim
    A_src_in_dim = src_dims.robot_obs_dim + src_act_arm_dim
    A_src = ActEncoder(
        in_dim=A_src_in_dim,
        out_dim=lat_act_dim,
        n_layers=int(params.get('n_layers', 3)),
        hidden=int(params.get('hidden_dim', 256)),
        batch_norm=False
    ).to(device)

    print("[diffusion] loading act_enc.pt ...")
    ckpt_act = torch.load(pathlib.Path(src_model_dir) / "act_enc.pt", map_location=device)
    # === 修正 state_dict key 前缀 ===
    new_sd = {f"net.{k}": v for k, v in ckpt_act.items()}
    A_src.load_state_dict(new_sd, strict=True)

    for p in A_src.parameters():
        p.requires_grad = False
    A_src.eval()
    print(f"[diffusion] A_src loaded and frozen. (in_dim={A_src_in_dim}, out_dim={lat_act_dim})")

    # ------------------------------
    # Read stats passes from config
    # ------------------------------
    diff_cfg = params.get('diffusion', {})
    stats_passes  = int(diff_cfg.get('stats_passes', 200))
    latent_passes = int(diff_cfg.get('latent_passes', 200))
    print(f"[stats] stats_passes={stats_passes}, latent_passes={latent_passes}")

    # ------------------------------
    # Collect stats: mu/std for src_act / tgt_act / latent_act & cond
    #   - act: 去夹爪（arm 维度）
    #   - latent_act: A_src(concat(robot_obs, act_arm))
    # ------------------------------
    def iter_buffer_act_and_c(buf: replay_buffer.ReplayBuffer, robot_dim: int, cond_dim: int, act_arm_dim: int, passes: int) -> Tuple[np.ndarray, np.ndarray]:
        As = []
        Cs = []
        print("[stage] collecting act(wo_gripper)/cond stats ...")
        for _ in trange(passes, desc="act+cond stats", dynamic_ncols=True):
            with torch.no_grad():
                obs, act, _, _, _ = buf.sample()
                a_arm = act[:, :act_arm_dim]
                if cond_dim > 0:
                    c = obs[:, robot_dim:robot_dim+cond_dim]
                else:
                    c = torch.empty((obs.shape[0], 0), device=obs.device)
                As.append(a_arm.cpu().numpy())
                Cs.append(c.cpu().numpy())
        A = np.concatenate(As, axis=0)
        C = np.concatenate(Cs, axis=0) if Cs and Cs[0].shape[1] > 0 else np.zeros((A.shape[0], 0), dtype=np.float32)
        print("[stage] act/cond stats done.")
        return A, C

    def iter_latent_act_stats(buf: replay_buffer.ReplayBuffer, robot_dim: int, act_arm_dim: int, encoder: nn.Module, passes: int) -> np.ndarray:
        ZAs = []
        print("[stage] collecting latent(a) stats (first forward may be slow due to CUDA init) ...")
        for _ in trange(passes, desc="latent act stats", dynamic_ncols=True):
            with torch.no_grad():
                obs, act, _, _, _ = buf.sample()
                x_robot = obs[:, :robot_dim]
                a_arm   = act[:, :act_arm_dim]
                enc_in  = torch.cat([x_robot, a_arm], dim=-1)
                za = encoder(enc_in)
                ZAs.append(za.cpu().numpy())
        ZA = np.concatenate(ZAs, axis=0)
        print("[stage] latent(a) stats done.")
        return ZA

    # src act & cond stats（去夹爪）
    As, Cs_src = iter_buffer_act_and_c(src_buffer, src_dims.robot_obs_dim, src_dims.cond_dim, src_act_arm_dim, stats_passes)
    mu_act_src, std_act_src = As.mean(axis=0), As.std(axis=0) + 1e-8
    if Cs_src.shape[1] > 0:
        mu_c_src, std_c_src = Cs_src.mean(axis=0), Cs_src.std(axis=0) + 1e-8
    else:
        mu_c_src, std_c_src = np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)

    # tgt act & cond stats（去夹爪）
    At, Cs_tgt = iter_buffer_act_and_c(tgt_buffer, tgt_dims.robot_obs_dim, tgt_dims.cond_dim, tgt_act_arm_dim, stats_passes)
    mu_act_tgt, std_act_tgt = At.mean(axis=0), At.std(axis=0) + 1e-8
    if Cs_tgt.shape[1] > 0:
        mu_c_tgt, std_c_tgt = Cs_tgt.mean(axis=0), Cs_tgt.std(axis=0) + 1e-8
    else:
        mu_c_tgt, std_c_tgt = np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)

    # latent stats over source domain: pass [robot_obs, action_wo_gripper] through A_src
    ZA = iter_latent_act_stats(src_buffer, src_dims.robot_obs_dim, src_act_arm_dim, A_src, latent_passes)
    mu_za, std_za = ZA.mean(axis=0), ZA.std(axis=0) + 1e-8

    # Save stats files (only act-related + cond)
    normalization = {
        'cond_src':   {'key': cond_keys, 'mu': mu_c_src.tolist(), 'std': std_c_src.tolist()},
        'cond_tgt':   {'key': cond_keys, 'mu': mu_c_tgt.tolist(), 'std': std_c_tgt.tolist()},
        'src_act':    {'mu': mu_act_src.tolist(), 'std': std_act_src.tolist()},
        'tgt_act':    {'mu': mu_act_tgt.tolist(), 'std': std_act_tgt.tolist()},
        'latent_act': {'mu': mu_za.tolist(),     'std': std_za.tolist()},
    }
    with open(stats_dir / 'normalization.yml', 'w') as fp:
        yaml_out = YAML()
        yaml_out.default_flow_style = False
        yaml_out.dump(normalization, fp)
    print(f"[save] normalization -> {stats_dir / 'normalization.yml'}")

    dataset_meta = {
        'task_name': task_name,
        'expert_folder': expert_folder,
        'src_buffer_dir': src_buffer_dir,
        'tgt_buffer_dir': tgt_buffer_dir,
        'src_robot': src_robot,
        'tgt_robot': tgt_robot,
        'controller_type': controller,
        'cond_key': cond_keys,
        'cond_dim': int(src_dims.cond_dim),
        'src_act_dim': int(src_dims.act_dim),
        'tgt_act_dim': int(tgt_dims.act_dim),
        'src_act_arm_dim': int(src_act_arm_dim),   # 记录 arm 维度
        'tgt_act_arm_dim': int(tgt_act_arm_dim),
        'lat_act_dim': int(lat_act_dim),
    }
    with open(stats_dir / 'dataset.yml', 'w') as fp:
        YAML().dump(dataset_meta, fp)
    print(f"[save] dataset meta -> {stats_dir / 'dataset.yml'}")

    # schedule snapshot (DDPM defaults)
    lat_cfg = diff_cfg.get('latent', {})
    schedule_type = lat_cfg.get('schedule', {}).get('type', 'ddpm')
    T = int(lat_cfg.get('schedule', {}).get('T', 128))
    beta = str(lat_cfg.get('schedule', {}).get('beta', 'cosine'))
    loss_weight = str(lat_cfg.get('schedule', {}).get('loss_weight', 'sigma2'))
    schedule_meta = {
        'type': schedule_type,
        'T': T,
        'beta': beta,
        'loss_weight': loss_weight,
    }
    with open(stats_dir / 'schedule.yml', 'w') as fp:
        YAML().dump(schedule_meta, fp)
    print(f"[save] schedule -> {stats_dir / 'schedule.yml'}")

    # ------------------------------
    # Build three ACTION priors (with normalization baked in)
    #  - latent(action): x_dim = lat_act_dim
    #  - src/tgt action: x_dim = act_arm_dim（去夹爪）
    # ------------------------------
    use_condition = bool(diff_cfg.get('use_condition', True))
    n_layers = int(lat_cfg.get('n_layers', params.get('n_layers', 3)))
    hidden_dim = int(lat_cfg.get('hidden_dim', params.get('hidden_dim', 256)))
    ema_decay = float(lat_cfg.get('ema_decay', 0.999))
    steps = int(lat_cfg.get('steps', params.get('tgt_align_timesteps', 300000)))
    bs = int(lat_cfg.get('batch_size', params.get('batch_size', 4096)))
    lr = float(lat_cfg.get('lr', params.get('lr', 3e-4)))
    log_freq = int(diff_cfg.get('log_freq', params.get('log_freq', 1000)))
    save_interval = int(diff_cfg.get('save_interval', params.get('evaluation', {}).get('save_interval', 20000)))

    time_embed_dim = 128

    prior_lat_act = PriorModel(x_dim=lat_act_dim, c_dim=src_dims.cond_dim, hidden=hidden_dim,
                               n_layers=n_layers, use_condition=use_condition,
                               time_embed_dim=time_embed_dim, mode=schedule_type).to(device)
    prior_lat_act.set_norm(mu_za, std_za, mu_c_src, std_c_src)

    prior_src_act = PriorModel(x_dim=src_act_arm_dim, c_dim=src_dims.cond_dim,
                               hidden=hidden_dim, n_layers=n_layers,
                               use_condition=use_condition, time_embed_dim=time_embed_dim,
                               mode=schedule_type).to(device)
    prior_src_act.set_norm(mu_act_src, std_act_src, mu_c_src, std_c_src)

    prior_tgt_act = PriorModel(x_dim=tgt_act_arm_dim, c_dim=tgt_dims.cond_dim,
                               hidden=hidden_dim, n_layers=n_layers,
                               use_condition=use_condition, time_embed_dim=time_embed_dim,
                               mode=schedule_type).to(device)
    prior_tgt_act.set_norm(mu_act_tgt, std_act_tgt, mu_c_tgt, std_c_tgt)

    # unified train cfg
    train_cfg = TrainCfg(
        steps=steps,
        batch_size=bs,
        lr=lr,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        ema_decay=ema_decay,
        schedule_type=schedule_type,
        T=T,
        beta=beta,
        loss_weight=loss_weight,
    )

    # ------------------------------
    # Build batch makers (ACTIONS)
    #  - latent: enc(concat(robot_obs, act_arm))
    #  - src/tgt: act_arm
    # ------------------------------
    def make_batch_lat_act():
        obs, act, _, _, _ = src_buffer.sample()
        x_robot = obs[:, :src_dims.robot_obs_dim]
        a_arm   = act[:, :src_act_arm_dim]
        c       = obs[:, src_dims.robot_obs_dim:src_dims.robot_obs_dim+src_dims.cond_dim] if src_dims.cond_dim > 0 else None
        with torch.no_grad():
            enc_in = torch.cat([x_robot, a_arm], dim=-1)
            za = A_src(enc_in)
        return za, c

    def make_batch_src_act():
        obs, act, _, _, _ = src_buffer.sample()
        a_arm = act[:, :src_act_arm_dim]
        c     = obs[:, src_dims.robot_obs_dim:src_dims.robot_obs_dim+src_dims.cond_dim] if src_dims.cond_dim > 0 else None
        return a_arm, c

    def make_batch_tgt_act():
        obs, act, _, _, _ = tgt_buffer.sample()
        a_arm = act[:, :tgt_act_arm_dim]
        c     = obs[:, tgt_dims.robot_obs_dim:tgt_dims.robot_obs_dim+tgt_dims.cond_dim] if tgt_dims.cond_dim > 0 else None
        return a_arm, c

    # ------------------------------
    # Train three ACTION priors (sequentially)
    # ------------------------------
    train_one_prior(
        name='score_lat_act',
        prior=prior_lat_act,
        make_batch_fn=make_batch_lat_act,
        writer=writer,
        device=device,
        out_dir=model_dir,
        cfg=train_cfg,
        log_freq=log_freq,
        save_interval=save_interval,
    )

    train_one_prior(
        name='score_src_act',
        prior=prior_src_act,
        make_batch_fn=make_batch_src_act,
        writer=writer,
        device=device,
        out_dir=model_dir,
        cfg=train_cfg,
        log_freq=log_freq,
        save_interval=save_interval,
    )

    train_one_prior(
        name='score_tgt_act',
        prior=prior_tgt_act,
        make_batch_fn=make_batch_tgt_act,
        writer=writer,
        device=device,
        out_dir=model_dir,
        cfg=train_cfg,
        log_freq=log_freq,
        save_interval=save_interval,
    )

    writer.close()
    print(f"[DONE] Saved models & stats under: {str(logdir)}")


if __name__ == '__main__':
    main()