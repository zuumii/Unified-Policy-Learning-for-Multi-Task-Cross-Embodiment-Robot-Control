# =========================================================
# Diffusion Priors Pretraining (Lat-Act / Src-Act / Tgt-Act)
# Condition = [EEF position p(3), EEF linear velocity v(3)]
# - qdot 直接用“关节动作（臂部维）”代替（不做维度检测）
# - p,v 在线计算，不落盘；把 [p,v] 追加到 obs / next_obs 尾部
# - 训练流程与日志、统计保持不变
# =========================================================

import os
import math
import time
import json
import pathlib
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ruamel.yaml import YAML

import utils
import replay_buffer

import mujoco as mj
from tqdm import tqdm, trange

# ---- constants ----
C_KEY = "cond_eef_pv"   # EEF position (3) + linear velocity (3)
C_DIM = 6               # 3 (p) + 3 (v)


# ---------------------------------------------------------
# Schedules / Embeddings / Denoiser / Prior
# ---------------------------------------------------------
def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
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


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device))
        args = t[..., None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb


class Denoiser(nn.Module):
    def __init__(self, x_dim: int, c_dim: int, t_dim: int, hidden: int = 256, n_layers: int = 3):
        super().__init__()
        in_dim = x_dim + (c_dim if c_dim > 0 else 0) + t_dim
        self.net = utils.build_mlp(
            in_dim, x_dim, n_layers, hidden,
            activation='relu', output_activation='identity', batch_norm=False
        )

    def forward(self, x_in: torch.Tensor):
        return self.net(x_in)


class PriorModel(nn.Module):
    def __init__(self, x_dim: int, c_dim: int, hidden: int, n_layers: int,
                 use_condition: bool, time_embed_dim: int, mode: str = 'ddpm'):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim if use_condition else 0
        self.use_condition = use_condition
        self.mode = mode
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.denoiser = Denoiser(x_dim=x_dim, c_dim=self.c_dim, t_dim=time_embed_dim,
                                 hidden=hidden, n_layers=n_layers)
        self.register_buffer('mu_x', torch.zeros(x_dim))
        self.register_buffer('std_x', torch.ones(x_dim))
        self.register_buffer('mu_c', torch.zeros(self.c_dim if self.c_dim > 0 else 1))
        self.register_buffer('std_c', torch.ones(self.c_dim if self.c_dim > 0 else 1))

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
# Replay-buffer plumbing（按你原仓库风格）
# ---------------------------------------------------------
@dataclass
class DomainDims:
    robot_obs_dim: int
    cond_dim: int
    act_dim: int


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


def probe_dims(env_name: str, robot: str, controller: str, env_kwargs: Dict[str, Any],
               robot_obs_keys: List[str], cond_keys: List[str]) -> DomainDims:
    probe_env = utils.make_robosuite_env(
        env_name, robots=robot, controller_type=controller, **env_kwargs
    )
    probe = probe_env.reset()
    rob = np.concatenate([probe[k] for k in robot_obs_keys])
    cond_dim = (C_DIM if len(cond_keys) > 0 else 0)
    act_dim = infer_action_dim(probe_env)
    return DomainDims(robot_obs_dim=int(rob.shape[0]), cond_dim=cond_dim, act_dim=act_dim)


def build_env_for_buffer(env_name: str, robot: str, controller: str, env_kwargs: Dict[str, Any],
                         robot_obs_keys: List[str], cond_keys_ignored: List[str]):
    # 在线增广，所以此处只让实时环境暴露 robot_obs_keys（不包含 condition 尾部）
    obs_keys = list(robot_obs_keys)
    env = utils.make(
        env_name,
        robots=robot,
        controller_type=controller,
        obs_keys=obs_keys,
        seed=0,
        **env_kwargs,
    )
    return env, obs_keys


def build_replay_buffer_from_episodes(episodes_aug, device: torch.device, batch_size: int):
    # 以第一个 episode 推断形状
    obs_shape = (episodes_aug[0]['obs'].shape[1],)
    act_shape = (episodes_aug[0]['action'].shape[1],)
    buf = replay_buffer.ReplayBuffer(
        obs_shape=obs_shape,
        action_shape=act_shape,
        capacity=int(1e7),
        batch_size=batch_size,
        device=device,
    )
    buf.add_rollouts(episodes_aug)
    return buf


# ---------------------------------------------------------
# p,v 在线计算相关：Jp(q) 与动作当作 qdot
# ---------------------------------------------------------
def _atan2_recover_q(cos_arr: np.ndarray, sin_arr: np.ndarray) -> np.ndarray:
    assert cos_arr.shape == sin_arr.shape, "cos/sin shape mismatch"
    return np.arctan2(sin_arr, cos_arr)  # [T, nq]


def _set_qpos_and_forward(env, q: np.ndarray):
    # 用 robosuite 的 forward，避免直接 mj_forward 的 dtype/contiguous 陷阱
    data = env.sim.data
    n = q.shape[-1]
    data.qpos[:n] = q.astype(np.float64)
    env.sim.forward()


def _eef_site_id(env) -> int:
    model = getattr(env.sim.model, "_model", env.sim.model)
    site_type = int(mj.mjtObj.mjOBJ_SITE)
    for name in ("gripper0_eef", "gripper0_grip_site", "robot0_eef"):
        sid = mj.mj_name2id(model, site_type, name)
        if sid != -1:
            return sid
    for i in range(int(model.nsite)):
        nm = mj.mj_id2name(model, site_type, i) or ""
        if "eef" in nm or "grip" in nm:
            return i
    return 0


def _eef_lin_jacobian(env, site_id: int) -> np.ndarray:
    """返回 EEF 线速度雅可比 Jp ∈ R^{3×nv}"""
    model = getattr(env.sim.model, "_model", env.sim.model)  # MjModel
    data = getattr(env.sim.data, "_data", env.sim.data)      # MjData
    nv = int(model.nv)
    jacp = np.zeros((3, nv), dtype=np.float64, order="C")
    jacr = np.zeros((3, nv), dtype=np.float64, order="C")
    mj.mj_jacSite(model, data, jacp, jacr, int(site_id))
    return jacp


def _eef_site_pos(env, site_id: int) -> np.ndarray:
    """返回当前 q 下 EEF 的世界坐标位置 p ∈ R^3"""
    data = getattr(env.sim.data, "_data", env.sim.data)
    return np.array(data.site_xpos[site_id][:3], dtype=np.float64)


def precompute_augmented_in_memory_using_actions(
    task_name: str,
    robots: str,
    controller: str,
    root_dir: str,
    robot_obs_keys: List[str],
    env_kwargs: dict,
    act_arm_dim: int,
    capacity: Optional[int] = None,
):
    """
    读取 root_dir/<task>/<robot>/<controller> 原始 *.npz，
    在线用 action(臂部维) 当作 qdot 计算：
      p = site_xpos(q)
      v = Jp(q) @ qdot
    把 [p,v] 追加到 obs/next_obs 尾部，不落盘，直接返回增广后的 episodes 列表。
    """
    in_dir = pathlib.Path(root_dir) / task_name / robots / controller

    # 1) 只为拿 mj 模型/数据：robosuite 原生 env（不包 GymWrapper）
    base_env = utils.make_robosuite_env(
        env_name=task_name,
        robots=robots,
        controller_type=controller,
        render=False,
        offscreen_render=False,
        **(env_kwargs or {})
    )
    site_id = _eef_site_id(base_env)

    # 2) 读原始 episodes（obs_keys 只给 robot_obs_keys；cos/sin/action 在 ep 原键里）
    episodes = utils.load_episodes(in_dir, obs_keys=list(robot_obs_keys), lat_obs_keys=None, capacity=capacity)

    out_eps = []
    for ep in tqdm(episodes, desc=f"[{robots}] augment with p,v (in-memory)"):
        # 恢复 q
        cos = ep['robot0_joint_pos_cos']    # [L_raw, nq]
        sin = ep['robot0_joint_pos_sin']    # [L_raw, nq]
        q = _atan2_recover_q(cos, sin)      # [L_raw, nq]

        action = ep['action']               # [L_raw-1, D_act] （通常转移长度）
        a_arm = action[:, :act_arm_dim]     # [L_raw-1, nq]（按要求，不做检测）

        # 用 q[:-1] 与 action 对齐，让条件 c[t] 对应 obs[t]、action[t]
        T_trans = action.shape[0]
        q_for = q[:T_trans]                 # [T_trans, nq]

        # 在线求 p,v，长度 == T_trans
        P = np.zeros((T_trans, 3), dtype=np.float32)
        V = np.zeros((T_trans, 3), dtype=np.float32)
        for t in range(T_trans):
            _set_qpos_and_forward(base_env, q_for[t])
            p = _eef_site_pos(base_env, site_id)                 # (3,)
            Jp = _eef_lin_jacobian(base_env, site_id)            # (3, nv)
            v = (Jp[:, :q_for.shape[1]] @ a_arm[t]).astype(np.float64)
            P[t] = p.astype(np.float32)
            V[t] = v.astype(np.float32)

        # 统一 T
        T = min(T_trans, ep['obs'].shape[0], ep['next_obs'].shape[0])

        # 对齐 curr/next：next 使用 1-step shift，末尾 copy
        p_curr = P[:T]
        v_curr = V[:T]
        if T > 1:
            p_next = np.vstack([P[1:T], P[T-1:T]])
            v_next = np.vstack([V[1:T], V[T-1:T]])
        else:
            p_next = p_curr.copy()
            v_next = v_curr.copy()

        # 条件 = [p, v]（顺序固定）
        c_curr = np.concatenate([p_curr, v_curr], axis=-1)   # [T, 6]
        c_next = np.concatenate([p_next, v_next], axis=-1)   # [T, 6]

        obs_aug = np.concatenate([ep['obs'][:T], c_curr], axis=-1)
        next_obs_aug = np.concatenate([ep['next_obs'][:T], c_next], axis=-1)

        new_ep = dict(ep)
        new_ep['obs'] = obs_aug
        new_ep['next_obs'] = next_obs_aug
        new_ep[C_KEY] = c_curr  # 可选：保留调试
        new_ep['action'] = action[:T]
        out_eps.append(new_ep)

    return out_eps


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
@dataclass
class TrainCfg:
    steps: int
    batch_size: int
    lr: float
    n_layers: int
    hidden_dim: int
    ema_decay: float
    schedule_type: str
    T: int
    beta: str
    loss_weight: str


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

    if cfg.schedule_type == 'ddpm':
        sched = make_ddpm_schedule(cfg.T, cfg.beta)
        betas = sched['betas'].to(device)
        alphas = sched['alphas'].to(device)
        alphas_cumprod = sched['alphas_cumprod'].to(device)
        T = cfg.T
    else:
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
            if cfg.loss_weight in ('sigma2', 'snr'):
                w = ddpm_loss_weight(a_bar_t, None, cfg.loss_weight).unsqueeze(-1)
                loss = F.mse_loss(eps_hat, noise, reduction='none')
                loss = (w * loss).mean()
            else:
                loss = F.mse_loss(eps_hat, noise)
        else:
            u = torch.rand(x.shape[0], device=device)
            sigma_min, sigma_max = 0.01, 1.2
            sigma = torch.exp(torch.log(torch.tensor(sigma_min, device=device)) * (1 - u) +
                              torch.log(torch.tensor(sigma_max, device=device)) * u)
            noise = torch.randn_like(x)
            x_noisy = x + sigma.unsqueeze(-1) * noise
            t_val = torch.log(sigma + 1e-8)
            eps_hat = prior(x_noisy, c, t_val)
            loss = (sigma.unsqueeze(-1) ** 2 * (eps_hat - noise) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update()

        running_loss = loss.item() if running_loss is None else (0.98 * running_loss + 0.02 * loss.item())
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    seed = int(params.get('seed', 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    expert_folder = params.get('expert_folder', 'human_demonstrations')
    task_name = params.get('task_name', params.get('env_name', 'Task'))

    src_robot = params['src_env']['robot']
    tgt_robot = params['tgt_env']['robot']
    env_kwargs = params.get('env_kwargs', {})

    # --- Logging dirs（与 align 同风格） ---
    logdir_prefix = pathlib.Path(params.get('logdir_prefix') or pathlib.Path(__file__).parent)
    data_path = logdir_prefix / 'logs' / time.strftime("%m.%d.%Y")
    logdir = '_'.join([
        time.strftime("%H-%M-%S"),
        params['env_name'],
        src_robot, params['src_env']['controller_type'],
        tgt_robot, params['tgt_env']['controller_type'],
        params.get('suffix', 'diffusion_pretrain')
    ])
    logdir = data_path / logdir
    logdir.mkdir(parents=True, exist_ok=True)

    # snapshot params
    import yaml as pyyaml
    with open(logdir / 'params.yml', 'w') as fp:
        pyyaml.safe_dump(params, fp, sort_keys=False)

    model_dir = logdir / 'models'
    stats_dir = logdir / 'stats'
    tb_dir = logdir / 'tb'
    model_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(tb_dir))

    # --- 维度探测（仅记录，不用于切片；cond 固定为 C_DIM=6） ---
    src_dims = probe_dims(params['env_name'], src_robot, params['src_env']['controller_type'], env_kwargs,
                          params['src_env']['robot_obs_keys'], [C_KEY])
    tgt_dims = probe_dims(params['env_name'], tgt_robot, params['tgt_env']['controller_type'], env_kwargs,
                          params['tgt_env']['robot_obs_keys'], [C_KEY])

    # --- 为推断 action_dim、构造 A_src 输入维度，需要 GymWrapper 版本的 env ---
    src_env, _ = build_env_for_buffer(
        params['env_name'],
        src_robot,
        params['src_env']['controller_type'],
        params['src_env'].get('env_kwargs', {}),
        params['src_env']['robot_obs_keys'],
        []
    )
    tgt_env, _ = build_env_for_buffer(
        params['env_name'],
        tgt_robot,
        params['tgt_env']['controller_type'],
        params['tgt_env'].get('env_kwargs', {}),
        params['tgt_env']['robot_obs_keys'],
        []
    )

    # --- 关节动作维度（臂部），按既有约定（总控维 - 1 去掉夹爪） ---
    src_act_arm_dim = infer_action_dim(src_env) - 1
    tgt_act_arm_dim = infer_action_dim(tgt_env) - 1

    print("[prep] augmenting in-memory with p (EEF), v = Jp(q) * action (SRC/TGT) ...")
    episodes_src_aug = precompute_augmented_in_memory_using_actions(
        task_name=task_name,
        robots=src_robot,
        controller=params['src_env']['controller_type'],
        root_dir=expert_folder,
        robot_obs_keys=params['src_env']['robot_obs_keys'],
        env_kwargs=env_kwargs,
        act_arm_dim=src_act_arm_dim,
        capacity=None,
    )
    episodes_tgt_aug = precompute_augmented_in_memory_using_actions(
        task_name=task_name,
        robots=tgt_robot,
        controller=params['tgt_env']['controller_type'],
        root_dir=expert_folder,
        robot_obs_keys=params['tgt_env']['robot_obs_keys'],
        env_kwargs=env_kwargs,
        act_arm_dim=tgt_act_arm_dim,
        capacity=None,
    )

    # --- 用增广后的 episodes 构建 buffers（obs 已包含 [p,v] 在尾部）
    diff_cfg = params.get('diffusion', {})
    batch_size = int(diff_cfg.get('latent', {}).get('batch_size', params.get('batch_size', 4096)))
    src_buffer = build_replay_buffer_from_episodes(episodes_src_aug, device, batch_size)
    tgt_buffer = build_replay_buffer_from_episodes(episodes_tgt_aug, device, batch_size)

    # --- Load A_src (ActEncoder) ---
    lat_act_dim = int(params['lat_act_dim'])

    class ActEncoder(nn.Module):
        def __init__(self, in_dim, out_dim, n_layers=3, hidden=256, batch_norm=False):
            super().__init__()
            self.net = utils.build_mlp(
                in_dim, out_dim, n_layers, hidden,
                activation='relu', output_activation='tanh', batch_norm=batch_norm
            )

        def forward(self, x):
            return self.net(x)

    A_src_in_dim = src_dims.robot_obs_dim + src_act_arm_dim
    A_src = ActEncoder(
        in_dim=A_src_in_dim, out_dim=lat_act_dim,
        n_layers=int(params.get('n_layers', 3)),
        hidden=int(params.get('hidden_dim', 256)), batch_norm=False
    ).to(device)

    ckpt_act = torch.load(pathlib.Path(params['src_model_dir']) / "act_enc.pt", map_location=device)
    new_sd = {f"net.{k}": v for k, v in ckpt_act.items()}
    A_src.load_state_dict(new_sd, strict=True)
    for p in A_src.parameters():
        p.requires_grad = False
    A_src.eval()

    # --- Stats (act/src/tgt + latent) ---
    stats_passes = int(diff_cfg.get('stats_passes', 200))
    latent_passes = int(diff_cfg.get('latent_passes', 200))

    def iter_buffer_act_and_c(buf: replay_buffer.ReplayBuffer, robot_dim: int, act_arm_dim: int, passes: int) -> Tuple[np.ndarray, np.ndarray]:
        As, Cs = [], []
        for _ in trange(passes, desc="act+cond stats", dynamic_ncols=True):
            with torch.no_grad():
                obs, act, _, _, _ = buf.sample()
                a_arm = act[:, :act_arm_dim]
                c = obs[:, -C_DIM:]  # 6 维 [p,v]
                As.append(a_arm.cpu().numpy())
                Cs.append(c.cpu().numpy())
        A = np.concatenate(As, axis=0)
        C = np.concatenate(Cs, axis=0)
        return A, C

    def iter_latent_act_stats(buf: replay_buffer.ReplayBuffer, robot_dim: int, act_arm_dim: int, encoder: nn.Module, passes: int) -> np.ndarray:
        ZAs = []
        for _ in trange(passes, desc="latent act stats", dynamic_ncols=True):
            with torch.no_grad():
                obs, act, _, _, _ = buf.sample()
                x_robot = obs[:, :robot_dim]
                a_arm = act[:, :act_arm_dim]
                enc_in = torch.cat([x_robot, a_arm], dim=-1)
                za = encoder(enc_in)
                ZAs.append(za.cpu().numpy())
        return np.concatenate(ZAs, axis=0)

    As, Cs_src = iter_buffer_act_and_c(src_buffer, src_dims.robot_obs_dim, src_act_arm_dim, stats_passes)
    mu_act_src, std_act_src = As.mean(axis=0), As.std(axis=0) + 1e-8
    mu_c_src, std_c_src = Cs_src.mean(axis=0), Cs_src.std(axis=0) + 1e-8

    At, Cs_tgt = iter_buffer_act_and_c(tgt_buffer, tgt_dims.robot_obs_dim, tgt_act_arm_dim, stats_passes)
    mu_act_tgt, std_act_tgt = At.mean(axis=0), At.std(axis=0) + 1e-8
    mu_c_tgt, std_c_tgt = Cs_tgt.mean(axis=0), Cs_tgt.std(axis=0) + 1e-8

    ZA = iter_latent_act_stats(src_buffer, src_dims.robot_obs_dim, src_act_arm_dim, A_src, latent_passes)
    mu_za, std_za = ZA.mean(axis=0), ZA.std(axis=0) + 1e-8

    normalization = {
        'cond_src':   {'key': C_KEY, 'mu': mu_c_src.tolist(), 'std': std_c_src.tolist()},
        'cond_tgt':   {'key': C_KEY, 'mu': mu_c_tgt.tolist(), 'std': std_c_tgt.tolist()},
        'src_act':    {'mu': mu_act_src.tolist(), 'std': std_act_src.tolist()},
        'tgt_act':    {'mu': mu_act_tgt.tolist(), 'std': std_act_tgt.tolist()},
        'latent_act': {'mu': mu_za.tolist(),     'std': std_za.tolist()},
    }
    with open(stats_dir / 'normalization.yml', 'w') as fp:
        YAML().dump(normalization, fp)

    lat_cfg = diff_cfg.get('latent', {})
    schedule_type = lat_cfg.get('schedule', {}).get('type', 'ddpm')
    T = int(lat_cfg.get('schedule', {}).get('T', 128))
    beta = str(lat_cfg.get('schedule', {}).get('beta', 'cosine'))
    loss_weight = str(lat_cfg.get('schedule', {}).get('loss_weight', 'sigma2'))
    schedule_meta = {'type': schedule_type, 'T': T, 'beta': beta, 'loss_weight': loss_weight}
    with open(stats_dir / 'schedule.yml', 'w') as fp:
        YAML().dump(schedule_meta, fp)

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

    prior_lat_act = PriorModel(x_dim=lat_act_dim, c_dim=C_DIM, hidden=hidden_dim,
                               n_layers=n_layers, use_condition=use_condition,
                               time_embed_dim=time_embed_dim, mode=schedule_type).to(device)
    prior_lat_act.set_norm(mu_za, std_za, mu_c_src, std_c_src)

    prior_src_act = PriorModel(x_dim=src_act_arm_dim, c_dim=C_DIM,
                               hidden=hidden_dim, n_layers=n_layers,
                               use_condition=use_condition, time_embed_dim=time_embed_dim,
                               mode=schedule_type).to(device)
    prior_src_act.set_norm(mu_act_src, std_act_src, mu_c_src, std_c_src)

    prior_tgt_act = PriorModel(x_dim=tgt_act_arm_dim, c_dim=C_DIM,
                               hidden=hidden_dim, n_layers=n_layers,
                               use_condition=use_condition, time_embed_dim=time_embed_dim,
                               mode=schedule_type).to(device)
    prior_tgt_act.set_norm(mu_act_tgt, std_act_tgt, mu_c_tgt, std_c_tgt)

    train_cfg = TrainCfg(
        steps=steps, batch_size=bs, lr=lr, n_layers=n_layers, hidden_dim=hidden_dim,
        ema_decay=ema_decay, schedule_type=schedule_type, T=T, beta=beta, loss_weight=loss_weight
    )

    def make_batch_lat_act():
        obs, act, _, _, _ = src_buffer.sample()
        x_robot = obs[:, :src_dims.robot_obs_dim]
        a_arm = act[:, :src_act_arm_dim]
        c = obs[:, -C_DIM:]  # [p,v]
        with torch.no_grad():
            enc_in = torch.cat([x_robot, a_arm], dim=-1)
            za = A_src(enc_in)
        return za, c

    def make_batch_src_act():
        obs, act, _, _, _ = src_buffer.sample()
        a_arm = act[:, :src_act_arm_dim]
        c = obs[:, -C_DIM:]  # [p,v]
        return a_arm, c

    def make_batch_tgt_act():
        obs, act, _, _, _ = tgt_buffer.sample()
        a_arm = act[:, :tgt_act_arm_dim]
        c = obs[:, -C_DIM:]  # [p,v]
        return a_arm, c

    train_one_prior('score_lat_act', prior_lat_act, make_batch_lat_act, writer, device, model_dir, train_cfg, log_freq, save_interval)
    train_one_prior('score_src_act', prior_src_act, make_batch_src_act, writer, device, model_dir, train_cfg, log_freq, save_interval)
    train_one_prior('score_tgt_act', prior_tgt_act, make_batch_tgt_act, writer, device, model_dir, train_cfg, log_freq, save_interval)

    writer.close()
    print(f"[DONE] Saved models & stats under: {str(logdir)}")


if __name__ == '__main__':
    main()