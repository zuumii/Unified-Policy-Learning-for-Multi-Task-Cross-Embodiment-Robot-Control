# align.py
# =========================================================
# 原始 align 结构 + 全新 SEW 集成 + 新增 Diffusion 先验（与 GAN 并存）
# - 保持原判别器/循环一致/动力学损失不变（默认权重同原来）
# - 新增：三路 Diffusion 先验（lat/src/tgt），权重/开关由 config 控制
# - 新增：严格维度校验（x_dim/c_dim 与预训练一致）
# - 先验 forward 有梯度；其参数冻结；梯度仅回到 E_tgt/D_tgt（生成器侧）
# - 提供 state_dict()/load_state_dict()（不变）
# =========================================================
import pathlib
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from td3 import Actor

def _infer_denoiser_arch_from_state(sd: dict, prefix: str = "denoiser.net.") -> tuple[int, int, int, int]:
    """
    从 ckpt 的 state_dict 推断 MLP 结构（仅限 utils.build_mlp 这种 Linear-Act-...-Linear 的顺序）：
      - 返回: in_dim, out_dim, n_hidden_layers (不含输出层), hidden_dim
    约定：Linear 层权重名形如  denoiser.net.{idx}.weight  且 idx 递增，最后一个是输出层。
    """
    lin_idxs = []
    for k in sd.keys():
        if k.startswith(prefix) and k.endswith(".weight"):
            try:
                lin_idxs.append(int(k.split(".")[2]))
            except Exception:
                pass
    lin_idxs = sorted(set(lin_idxs))
    if not lin_idxs:
        raise RuntimeError("[diff.prior] ckpt 中未发现线性层权重（denoiser.net.*.weight）")

    shapes = [sd[f"{prefix}{i}.weight"].shape for i in lin_idxs]  # [(out, in), ...]
    in_dim = int(shapes[0][1])
    out_dim = int(shapes[-1][0])

    # 隐藏层数 = 线性层数 - 1（去掉最后的输出层）
    n_hidden_layers = len(shapes) - 1
    # 取第一个隐藏层的 out_dim 作为 hidden_dim（如果不一致也以第一个为准）
    hidden_dim = int(shapes[0][0]) if n_hidden_layers >= 1 else out_dim
    return in_dim, out_dim, n_hidden_layers, hidden_dim


def _rebuild_prior_to_match_ckpt(
    old_prior, ckpt_sd: dict, *,
    x_dim: int, c_dim: int, t_dim: int,  # 锁死 IO 维度
    mode: str,
) -> "PriorModel":
    """用 ckpt 推断到的隐藏规模重建 PriorModel（保持 x/c/t 维与当前一致），再加载权重。"""
    in_dim_ckpt, out_dim_ckpt, n_hidden_layers, hidden_dim = _infer_denoiser_arch_from_state(ckpt_sd)

    # 期望的输入维（必须匹配 ckpt 首层 in_features）
    expected_in = x_dim + (c_dim if c_dim > 0 else 0) + t_dim
    if in_dim_ckpt != expected_in:
        raise RuntimeError(
            f"[diff.prior] ckpt 输入维不匹配：ckpt_in={in_dim_ckpt}, expected={expected_in} "
            f"(x={x_dim}, c={c_dim}, t={t_dim}).\n"
            f"请检查 cond.keys 顺序/维度是否与预训练先验一致，或把 config.cond.strict_dim_check 设为 false 后自行确认。"
        )
    if out_dim_ckpt != x_dim:
        raise RuntimeError(
            f"[diff.prior] ckpt 输出维不匹配：ckpt_out={out_dim_ckpt}, expected={x_dim} （x_dim 锁死）。"
        )

    # 用推断到的规模重建
    rebuilt = PriorModel(
        x_dim=x_dim,
        c_dim=c_dim,
        hidden=hidden_dim,
        n_layers=n_hidden_layers,
        use_condition=(c_dim > 0),
        time_embed_dim=t_dim,   # 这里传的是 t-embedding 维度（与你训练时一致）
        mode=mode,
    ).to(next(old_prior.parameters()).device)

    # 加载（这次就能 strict=True）
    rebuilt.load_state_dict(ckpt_sd, strict=True)
    return rebuilt
# -----------------------------
# 基础 Agent（与原版保持一致）
# -----------------------------
class Agent:
    """Base class for adaptation"""
    def __init__(self, obs_dims, act_dims, device):
        self.obs_dim = obs_dims['obs_dim']
        self.robot_obs_dim = obs_dims['robot_obs_dim']
        self.obj_obs_dim = obs_dims['obj_obs_dim']
        self.act_dim = act_dims['act_dim']
        self.device = device
        self.batch_norm = False
        self.modules = []
        self.expl_noise = 0.1  # 用于 sample_action 时的一致性

        assert self.obs_dim == self.robot_obs_dim + self.obj_obs_dim

    def eval_mode(self):
        for m in self.modules:
            m.eval()

    def train_mode(self):
        for m in self.modules:
            m.train()

    def freeze(self):
        for m in self.modules:
            for p in m.parameters():
                p.requires_grad = False

    # ====== 从 ckpt 推断 Actor 结构（便于兼容历史权重）======
    @staticmethod
    def _infer_actor_arch_from_state_dict(sd: dict):
        items = []
        for k, v in sd.items():
            if k.startswith("trunk.") and k.endswith(".weight"):
                parts = k.split(".")
                if len(parts) >= 3:
                    try:
                        idx = int(parts[1])
                    except Exception:
                        continue
                    items.append((idx, tuple(v.shape)))
        items.sort(key=lambda x: x[0])
        if not items:
            raise RuntimeError("actor state_dict has no trunk.*.weight keys")

        first_shape = items[0][1]
        last_shape  = items[-1][1]
        in_dim = int(first_shape[1])
        hidden_dim = int(first_shape[0])
        out_dim = int(last_shape[0])
        n_linear = len(items)
        n_hidden_layers = n_linear - 1
        return in_dim, out_dim, n_hidden_layers, hidden_dim

    def _rebuild_actor_to_match_ckpt(self, actor_sd: dict):
        in_dim, out_dim, n_layers, hidden_dim = self._infer_actor_arch_from_state_dict(actor_sd)
        print(f"[align] Rebuild Actor to match ckpt: in={in_dim}, out={out_dim}, "
              f"hidden={hidden_dim}, layers={n_layers}")
        old_actor = getattr(self, "actor", None)
        self.actor = Actor(in_dim, out_dim, n_layers, hidden_dim).to(self.device)
        if hasattr(self, "modules") and old_actor is not None:
            self.modules = [self.actor if (m is old_actor) else m for m in self.modules]


class ObsAgent(Agent):
    def __init__(self, obs_dims, act_dims, device,
                 n_layers=3, hidden_dim=256,
                 actor_n_layers=None, actor_hidden_dim=None):
        super().__init__(obs_dims, act_dims, device)
        self.lat_obs_dim = obs_dims['lat_obs_dim']

        # enc/dec/dynamics：保持与老 ckpt 对齐（不放大）
        self.obs_enc = utils.build_mlp(self.robot_obs_dim, self.lat_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(self.device)
        self.obs_dec = utils.build_mlp(self.lat_obs_dim, self.robot_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='identity', batch_norm=self.batch_norm).to(self.device)
        self.inv_dyn = utils.build_mlp(self.lat_obs_dim*2, self.act_dim-1, n_layers, hidden_dim, 
            activation='relu', output_activation='identity', batch_norm=self.batch_norm).to(self.device)
        self.fwd_dyn = utils.build_mlp(self.lat_obs_dim+self.act_dim-1, self.lat_obs_dim, n_layers, hidden_dim, 
            activation='relu', output_activation='identity', batch_norm=self.batch_norm).to(self.device)

        # 这里先建一个“占位”的 actor，ObsActAgent 会覆盖为 latent 头
        a_n = actor_n_layers or n_layers
        a_h = actor_hidden_dim or hidden_dim
        self.actor = Actor(self.lat_obs_dim+self.obj_obs_dim, self.act_dim, a_n, a_h).to(self.device)

        self.modules = [self.obs_enc, self.obs_dec, self.inv_dyn, self.fwd_dyn, self.actor]

    def save(self, model_dir):
        torch.save(self.actor.state_dict(), f'{model_dir}/actor.pt')
        torch.save(self.obs_enc.state_dict(), f'{model_dir}/obs_enc.pt')        
        torch.save(self.obs_dec.state_dict(), f'{model_dir}/obs_dec.pt')
        torch.save(self.inv_dyn.state_dict(), f'{model_dir}/inv_dyn.pt')
        torch.save(self.fwd_dyn.state_dict(), f'{model_dir}/fwd_dyn.pt')

    def load(self, model_dir):
        model_dir = pathlib.Path(model_dir)

        def _safe_load(m, fname):
            sd = torch.load(model_dir / fname, map_location=self.device)
            try:
                m.load_state_dict(sd, strict=True)
            except RuntimeError as e:
                print(f"[align.load] {fname} strict=True failed -> fallback strict=False: {e}")
                m.load_state_dict(sd, strict=False)

        _safe_load(self.obs_enc, "obs_enc.pt")
        _safe_load(self.obs_dec, "obs_dec.pt")
        _safe_load(self.fwd_dyn, "fwd_dyn.pt")
        _safe_load(self.inv_dyn, "inv_dyn.pt")

        actor_sd = torch.load(model_dir / "actor.pt", map_location=self.device)
        try:
            self.actor.load_state_dict(actor_sd, strict=True)
        except RuntimeError as e:
            print(f"[align.load] actor strict=True failed -> rebuild from ckpt: {e}")
            self._rebuild_actor_to_match_ckpt(actor_sd)
            self.actor.load_state_dict(actor_sd, strict=True)

    def load_actor(self, model_dir):
        model_dir = pathlib.Path(model_dir)
        actor_sd = torch.load(model_dir / "actor.pt", map_location=self.device)
        try:
            self.actor.load_state_dict(actor_sd, strict=True)
        except RuntimeError as e:
            print(f"[align.load_actor] strict=True failed -> rebuild from ckpt: {e}")
            self._rebuild_actor_to_match_ckpt(actor_sd)
        self.actor.load_state_dict(actor_sd, strict=True)
        for p in self.actor.parameters():
            p.requires_grad = False

    def sample_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
            robot_obs, obj_obs = obs[:, :self.robot_obs_dim], obs[:, self.robot_obs_dim:]
            lat_obs = self.obs_enc(robot_obs)
            act = self.actor(torch.cat([lat_obs, obj_obs], dim=-1))

        act = act.cpu().data.numpy().flatten()
        if not deterministic:
            act += np.random.normal(0, self.expl_noise, size=act.shape[0])
            act = np.clip(act, -1, 1)
        return act  


class ObsActAgent(ObsAgent):
    def __init__(self, obs_dims, act_dims, device,
                 n_layers=3, hidden_dim=256,
                 actor_n_layers=None, actor_hidden_dim=None):
        # 先建 enc/dec/dyn 和“占位” actor
        super().__init__(obs_dims, act_dims, device,
                         n_layers=n_layers, hidden_dim=hidden_dim,
                         actor_n_layers=actor_n_layers, actor_hidden_dim=actor_hidden_dim)

        self.lat_act_dim = act_dims['lat_act_dim']

        # ===== 动作分支（保持与旧版一致）=====
        self.act_enc = utils.build_mlp(
            self.robot_obs_dim + self.act_dim - 1, self.lat_act_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh', batch_norm=self.batch_norm
        ).to(device)
        self.act_dec = utils.build_mlp(
            self.robot_obs_dim + self.lat_act_dim, self.act_dim - 1, n_layers, hidden_dim,
            activation='relu', output_activation='tanh', batch_norm=self.batch_norm
        ).to(device)
        self.inv_dyn = utils.build_mlp(
            self.lat_obs_dim * 2, self.lat_act_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh', batch_norm=self.batch_norm
        ).to(device)
        self.fwd_dyn = utils.build_mlp(
            self.lat_obs_dim + self.lat_act_dim, self.lat_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh', batch_norm=self.batch_norm
        ).to(device)

        # ===== 关键修复：actor 改为 “latent 动作头”（lat_act_dim + 1）=====
        a_n = actor_n_layers or n_layers
        a_h = actor_hidden_dim or hidden_dim
        old_actor = self.actor
        self.actor = Actor(self.lat_obs_dim + self.obj_obs_dim, self.lat_act_dim + 1, a_n, a_h).to(device)
        # 替换 modules 里的旧 actor
        self.modules = [
            self.obs_enc, self.obs_dec,
            self.inv_dyn, self.fwd_dyn,
            self.actor,
            self.act_enc, self.act_dec,
        ]

    def save(self, model_dir):
        super().save(model_dir)
        torch.save(self.act_enc.state_dict(), f'{model_dir}/act_enc.pt')        
        torch.save(self.act_dec.state_dict(), f'{model_dir}/act_dec.pt')

    def load(self, model_dir):
        model_dir = pathlib.Path(model_dir)
        super().load(model_dir)

        def _safe_load(m, fname):
            sd = torch.load(model_dir / fname, map_location=self.device)
            try:
                m.load_state_dict(sd, strict=True)
            except RuntimeError as e:
                print(f"[align.load] {fname} strict=True failed -> fallback strict=False: {e}")
                m.load_state_dict(sd, strict=False)

        _safe_load(self.act_enc, "act_enc.pt")
        _safe_load(self.act_dec, "act_dec.pt")

    def sample_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
            robot_obs, obj_obs = obs[:, :self.robot_obs_dim], obs[:, self.robot_obs_dim:]
            lat_obs = self.obs_enc(robot_obs)
            lat_act = self.actor(torch.cat([lat_obs, obj_obs], dim=-1))
            lat_act, gripper_act = lat_act[:, :-1], lat_act[:, -1].reshape(-1, 1)
            act = self.act_dec(torch.cat([robot_obs, lat_act], dim=-1))
            act = torch.cat([act, gripper_act], dim=-1)
        
        act = act.cpu().data.numpy().flatten()
        if not deterministic:
            act += np.random.normal(0, self.expl_noise, size=act.shape[0])
            act = np.clip(act, -1, 1)

        return act


# -----------------------------
# SEW 预测器（冻结）与适配器（保持不变）
# -----------------------------
class SEWPredictor(nn.Module):
    """
    统一加载 SEW 角预测小网络：
    - 若 path 是 TorchScript => torch.jit.load
    - 否则：普通 PyTorch => torch.load，然后尝试以下几种 key：
        * checkpoint["model_state_dict"] / ["state_dict"] / ["model"] / 直接是 state_dict
    支持标准化 mean/std（可为 None）
    """
    def __init__(self, path, device, in_dim, out_dim=1, hidden=256, mean=None, std=None):
        super().__init__()
        self.device = device
        self.mean = None if mean is None else torch.as_tensor(mean, dtype=torch.float32, device=device)
        self.std  = None if std  is None else torch.as_tensor(std,  dtype=torch.float32, device=device)

        # 模型结构（注意 utils.build_mlp 的参数顺序）
        self.model = utils.build_mlp(in_dim, out_dim, 3, hidden, activation='relu',
                                     output_activation='identity', batch_norm=False).to(device)

        # 尝试加载
        ok = False
        try:
            # 先试 TorchScript
            ts = torch.jit.load(path, map_location=device)
            class _WrapTS(nn.Module):
                def __init__(self, tsmod): 
                    super().__init__(); 
                    self.ts = tsmod
                def forward(self, x): 
                    return self.ts(x)
            self.model = _WrapTS(ts).to(device)
            ok = True
        except Exception:
            # 回退到普通 checkpoint
            ckpt = torch.load(path, map_location=device)
            if isinstance(ckpt, dict):
                for k in ['model_state_dict', 'state_dict', 'model']:
                    if k in ckpt and isinstance(ckpt[k], dict):
                        self.model.load_state_dict(ckpt[k], strict=False)
                        ok = True
                        break
                if not ok:
                    try:
                        self.model.load_state_dict(ckpt, strict=False)
                        ok = True
                    except Exception:
                        pass
            else:
                ok = False

        if not ok:
            raise RuntimeError(
                f"Failed to load SEW predictor from {path}. "
                f"Expect TorchScript file or a checkpoint with a state_dict."
            )

    @torch.no_grad()
    def forward(self, x):
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-8)
        y = self.model(x)
        return y


def _slice_robot_obs(obs: torch.Tensor, robot_dim: int) -> torch.Tensor:
    # 从 obs = [robot_obs | obj_obs] 中切出 robot_obs
    return obs[:, :robot_dim]


def _robot_obs_to_q(robot_obs: torch.Tensor, dof: int) -> torch.Tensor:
    # robot_obs = [cos(q[0:dof]), sin(q[0:dof]), ...] -> q
    D = robot_obs.shape[1]
    if D >= 2 * dof:
        cos = robot_obs[:, :dof]
        sin = robot_obs[:, dof:2*dof]
        return torch.atan2(sin, cos)
    return robot_obs[:, :dof]


# =========================================================
# 新增：Diffusion 先验（与预训练脚本一致的轻量运行版）
# =========================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        # 和预训练一致：1..1000 指数频率
        freqs = torch.exp(torch.linspace(0.0, np.log(1000.0), half, device=t.device))
        args = t[..., None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb


class Denoiser(nn.Module):
    def __init__(self, x_dim: int, c_dim: int, t_dim: int, hidden: int = 256, n_layers: int = 3):
        super().__init__()
        in_dim = x_dim + (c_dim if c_dim > 0 else 0) + t_dim
        # utils.build_mlp 参数顺序：in, out, n_layers, hidden
        self.net = utils.build_mlp(
            in_dim, x_dim, n_layers, hidden,
            activation='relu', output_activation='identity', batch_norm=False
        )

    def forward(self, x_in: torch.Tensor):
        return self.net(x_in)


class PriorModel(nn.Module):
    """与预训练脚本保持参数/缓冲一致，能直接 load_state_dict"""
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
        # 归一化参数与预训练一致的 buffer 名
        self.register_buffer('mu_x', torch.zeros(x_dim))
        self.register_buffer('std_x', torch.ones(x_dim))
        self.register_buffer('mu_c', torch.zeros(self.c_dim if self.c_dim>0 else 1))
        self.register_buffer('std_c', torch.ones(self.c_dim if self.c_dim>0 else 1))

    def _norm_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu_x) / (self.std_x + 1e-8)

    def _norm_c(self, c: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if (not self.use_condition) or (self.c_dim == 0) or (c is None):
            return None
        return (c - self.mu_c) / (self.std_c + 1e-8)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor], t_value: torch.Tensor) -> torch.Tensor:
        x_n = self._norm_x(x)
        if self.use_condition and self.c_dim > 0:
            c_n = self._norm_c(c)
            assert c_n is not None
            inp = torch.cat([x_n, c_n, self.time_embed(t_value)], dim=-1)
        else:
            inp = torch.cat([x_n, self.time_embed(t_value)], dim=-1)
        return self.denoiser(inp)


def _cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 1e-8, 0.999)
    return betas


def _make_ddpm_schedule(T: int, beta_type: str = "cosine") -> Dict[str, torch.Tensor]:
    if beta_type == "cosine":
        betas = _cosine_beta_schedule(T)
    else:
        betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {"betas": betas, "alphas": alphas, "alphas_cumprod": alphas_cumprod}


def _ddpm_weight(alpha_cumprod_t: torch.Tensor, mode: str) -> torch.Tensor:
    # 'sigma2' | 'snr' | 'none'
    if mode == 'sigma2':
        sigma2 = 1.0 - alpha_cumprod_t
        return sigma2
    elif mode == 'snr':
        sigma2 = torch.clamp(1.0 - alpha_cumprod_t, min=1e-8)
        snr = alpha_cumprod_t / sigma2
        return 1.0 / torch.clamp(snr, min=1e-8)
    else:
        return torch.ones_like(alpha_cumprod_t)


class DiffusionPriors:
    """打包三套先验 + 调度；仅 forward 计算噪声回归损失；参数冻结"""
    def __init__(self, device: torch.device):
        self.device = device
        self.enabled = False
        self.lambda_lat = 1.0
        self.lambda_src = 1.0
        self.lambda_tgt = 1.0
        self.t_embed = 'ddpm'

        self.lat_prior = None
        self.src_prior = None
        self.tgt_prior = None

        # 调度
        self.T = 128
        self.beta = 'cosine'
        self.loss_weight = 'sigma2'
        self.alphas_cumprod = None
        self.betas = None

        # 统计信息仅用于日志
        self.cond_dim = 0
        self.cond_keys = []

    def load_from_dir(self, cfg: Dict[str, Any], cond_dim: int, src_robot_dim: int, tgt_robot_dim: int, lat_dim: int):
        """
        cfg = {
          "priors": { "dir": ..., "use_ema": true, "schedule_from_stats": bool?, "t_embed": "ddpm"/"edm" },
          "enabled": bool,
          "lambda": { "lat":.., "src":.., "tgt":.. },
          "schedule_override": { "use_override": bool, "T":.., "beta":.., "loss_weight":.. },
          "cond_dim": int,
          "cond_keys": [..]
        }
        """
        self.enabled = bool(cfg.get("enabled", False))
        self.lambda_lat = float(((cfg.get("lambda") or {}).get("lat", 1.0)))
        self.lambda_src = float(((cfg.get("lambda") or {}).get("src", 1.0)))
        self.lambda_tgt = float(((cfg.get("lambda") or {}).get("tgt", 1.0)))
        self.cond_dim = int(cfg.get("cond_dim", 0))
        self.cond_keys = list(cfg.get("cond_keys", []))
        if cond_dim != self.cond_dim:
            # train_align 已计算 cond_dim；这里只做一致性提示，不中断
            print(f"[diffusion] warn: cond_dim from align={cond_dim}, bundle={self.cond_dim}")

        pri_cfg = cfg.get("priors", {}) or {}
        root = pathlib.Path(pri_cfg.get("dir", ""))
        if self.enabled:
            if (not root.exists()) or (not (root / "models").exists()):
                raise RuntimeError(f"[diffusion] priors.dir 不存在或无 models 子目录: {root}")
        use_ema = bool(pri_cfg.get("use_ema", True))
        self.t_embed = str(pri_cfg.get("t_embed", "ddpm"))

        # --- 调度：优先 override；否则读 stats/schedule.yml；最后默认 ---
        sched_ov = (cfg.get("schedule_override") or {})
        use_ov = bool(sched_ov.get("use_override", False))
        if use_ov:
            self.T = int(sched_ov.get("T", 128))
            self.beta = str(sched_ov.get("beta", "cosine"))
            self.loss_weight = str(sched_ov.get("loss_weight", "sigma2"))
        else:
            # 尝试从 stats/schedule.yml 读取
            try:
                from ruamel.yaml import YAML
                sched = YAML(typ='safe').load(open(root / "stats" / "schedule.yml"))
                self.T = int(sched.get("T", 128))
                self.beta = str(sched.get("beta", "cosine"))
                self.loss_weight = str(sched.get("loss_weight", "sigma2"))
            except Exception:
                self.T, self.beta, self.loss_weight = 128, "cosine", "sigma2"

        # 预构建 ddpm 调度（edm 暂时复用 ddpm t_embed；采样噪声相同）
        s = _make_ddpm_schedule(self.T, self.beta)
        self.betas = s["betas"].to(self.device)
        self.alphas_cumprod = s["alphas_cumprod"].to(self.device)

        if not self.enabled:
            print("[diffusion] disabled by config.")
            return

        # 载入三套先验（路径固定：models/score_*.pt 或 *_ema.pt）
        def _load_one(name: str, x_dim: int):
            fname = f"{name}_ema.pt" if use_ema else f"{name}.pt"
            path = root / "models" / fname
            if not path.exists():
                raise RuntimeError(f"[diffusion] 模型不存在: {path}")

            prior = PriorModel(x_dim=x_dim, c_dim=self.cond_dim, hidden=512, n_layers=4,
                            use_condition=(self.cond_dim > 0),
                            time_embed_dim=128, mode=self.t_embed).to(self.device)
            sd = torch.load(path, map_location=self.device)
            try:
                prior.load_state_dict(sd, strict=True)
            except RuntimeError:
                # 严格加载失败 → 按 ckpt 的层数/宽度重建（IO 维锁死）
                prior = _rebuild_prior_to_match_ckpt(
                    prior, sd, x_dim=x_dim, c_dim=self.cond_dim, t_dim=128, mode=self.t_embed
                )
            for p in prior.parameters():
                p.requires_grad = False
            prior.eval()
            return prior

        self.lat_prior = _load_one("score_lat", lat_dim)
        self.src_prior = _load_one("score_src_obs", src_robot_dim)
        self.tgt_prior = _load_one("score_tgt_obs", tgt_robot_dim)
        
        # === 从 stats/normalization.yml 读取训练期的归一化，并写回每个 prior 的 buffer ===
        try:
            from ruamel.yaml import YAML as _YAML
            norm_path = root / "stats" / "normalization.yml"
            norm = _YAML(typ='safe').load(open(norm_path, "r"))
        except Exception as e:
            raise RuntimeError(f"[diffusion] 读取 normalization.yml 失败: {e}")

        def _to_tensor(x, like: torch.Tensor):
            t = torch.as_tensor(x, dtype=like.dtype, device=like.device)
            return t

        # 取出各块统计
        latent_mu = norm["latent"]["mu"]; latent_std = norm["latent"]["std"]
        src_mu    = norm["src_obs"]["mu"]; src_std    = norm["src_obs"]["std"]
        tgt_mu    = norm["tgt_obs"]["mu"]; tgt_std    = norm["tgt_obs"]["std"]

        cond_src_mu = norm["cond_src"]["mu"]; cond_src_std = norm["cond_src"]["std"]
        cond_tgt_mu = norm["cond_tgt"]["mu"]; cond_tgt_std = norm["cond_tgt"]["std"]

        # --- 形状强校验 ---
        if len(latent_mu) != lat_dim or len(latent_std) != lat_dim:
            raise RuntimeError(f"[diffusion] latent 统计长度不等于 lat_dim: "
                               f"{len(latent_mu)}/{len(latent_std)} vs {lat_dim}")
        if len(src_mu) != src_robot_dim or len(src_std) != src_robot_dim:
            raise RuntimeError(f"[diffusion] src_obs 统计长度不等于 src_robot_dim: "
                               f"{len(src_mu)}/{len(src_std)} vs {src_robot_dim}")
        if len(tgt_mu) != tgt_robot_dim or len(tgt_std) != tgt_robot_dim:
            raise RuntimeError(f"[diffusion] tgt_obs 统计长度不等于 tgt_robot_dim: "
                               f"{len(tgt_mu)}/{len(tgt_std)} vs {tgt_robot_dim}")
        if self.cond_dim > 0:
            if len(cond_src_mu) != self.cond_dim or len(cond_src_std) != self.cond_dim:
                raise RuntimeError(f"[diffusion] cond_src 统计长度不等于 cond_dim: "
                                   f"{len(cond_src_mu)}/{len(cond_src_std)} vs {self.cond_dim}")
            if len(cond_tgt_mu) != self.cond_dim or len(cond_tgt_std) != self.cond_dim:
                raise RuntimeError(f"[diffusion] cond_tgt 统计长度不等于 cond_dim: "
                                   f"{len(cond_tgt_mu)}/{len(cond_tgt_std)} vs {self.cond_dim}")

        # --- 写回三个 prior 的 buffer ---
        # score_lat: x -> latent, c -> cond_src
        self.lat_prior.mu_x.data.copy_(_to_tensor(latent_mu, self.lat_prior.mu_x))
        self.lat_prior.std_x.data.copy_(_to_tensor(latent_std, self.lat_prior.std_x))
        if self.lat_prior.c_dim > 0:
            self.lat_prior.mu_c.data.copy_(_to_tensor(cond_src_mu, self.lat_prior.mu_c))
            self.lat_prior.std_c.data.copy_(_to_tensor(cond_src_std, self.lat_prior.std_c))

        # score_src_obs: x -> src_obs, c -> cond_src
        self.src_prior.mu_x.data.copy_(_to_tensor(src_mu, self.src_prior.mu_x))
        self.src_prior.std_x.data.copy_(_to_tensor(src_std, self.src_prior.std_x))
        if self.src_prior.c_dim > 0:
            self.src_prior.mu_c.data.copy_(_to_tensor(cond_src_mu, self.src_prior.mu_c))
            self.src_prior.std_c.data.copy_(_to_tensor(cond_src_std, self.src_prior.std_c))

        # score_tgt_obs: x -> tgt_obs, c -> cond_tgt
        self.tgt_prior.mu_x.data.copy_(_to_tensor(tgt_mu, self.tgt_prior.mu_x))
        self.tgt_prior.std_x.data.copy_(_to_tensor(tgt_std, self.tgt_prior.std_x))
        if self.tgt_prior.c_dim > 0:
            self.tgt_prior.mu_c.data.copy_(_to_tensor(cond_tgt_mu, self.tgt_prior.mu_c))
            self.tgt_prior.std_c.data.copy_(_to_tensor(cond_tgt_std, self.tgt_prior.std_c))

        # 打印确认（均值只看个大概）
        def _peek(name, p):
            print(f"[diffusion:{name}] x_dim={p.x_dim}, c_dim={p.c_dim}, "
                  f"mu_x.mean={p.mu_x.mean().item():.4g}, std_x.mean={p.std_x.mean().item():.4g}, "
                  f"mu_c.mean={(p.mu_c.mean().item() if p.c_dim>0 else float('nan')):.4g}, "
                  f"std_c.mean={(p.std_c.mean().item() if p.c_dim>0 else float('nan')):.4g}")
        _peek("lat", self.lat_prior)
        _peek("src", self.src_prior)
        _peek("tgt", self.tgt_prior)           

        # 维度强校验
        if self.lat_prior.c_dim != self.cond_dim or \
           self.src_prior.c_dim != self.cond_dim or \
           self.tgt_prior.c_dim != self.cond_dim:
            raise RuntimeError(f"[diffusion] c_dim 不匹配：prior(c_dim)≠ cond_dim={self.cond_dim}")

        print(f"[diffusion] loaded priors from {root} | use_ema={use_ema} | T={self.T}, beta={self.beta}, "
              f"loss_weight={self.loss_weight}, t_embed={self.t_embed}, cond_dim={self.cond_dim}, keys={self.cond_keys}")

    def _loss_one(self, prior: PriorModel, x: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        """DDPM 噪声回归损失"""
        if prior is None:
            return torch.tensor(0.0, device=self.device)
        B = x.shape[0]
        T = self.T
        t_idx = torch.randint(0, T, (B,), device=self.device)
        a_bar_t = self.alphas_cumprod[t_idx]
        noise = torch.randn_like(x)
        x_t = a_bar_t.sqrt().unsqueeze(-1) * x + (1.0 - a_bar_t).sqrt().unsqueeze(-1) * noise
        t_float = (t_idx.to(torch.float32) + 0.5) / float(T)
        eps_hat = prior(x_t, c, t_float)
        w = _ddpm_weight(a_bar_t, self.loss_weight).unsqueeze(-1)
        return (w * (eps_hat - noise) ** 2).mean()

    def loss_lat(self, z_tgt: torch.Tensor, c_tgt: Optional[torch.Tensor]) -> torch.Tensor:
        return self._loss_one(self.lat_prior, z_tgt, c_tgt)

    def loss_src(self, fake_src_robot: torch.Tensor, c_tgt: Optional[torch.Tensor]) -> torch.Tensor:
        return self._loss_one(self.src_prior, fake_src_robot, c_tgt)

    def loss_tgt(self, fake_tgt_robot: torch.Tensor, c_tgt: Optional[torch.Tensor]) -> torch.Tensor:
        return self._loss_one(self.tgt_prior, fake_tgt_robot, c_tgt)

class ActionDiffusionPriors:
    """动作分支的三套先验（lat_act / src_act / tgt_act），与 DiffusionPriors 完全平行"""
    def __init__(self, device: torch.device):
        self.device = device
        self.enabled = False
        self.lambda_lat = 1.0
        self.lambda_src = 1.0
        self.lambda_tgt = 1.0
        self.t_embed = 'ddpm'

        self.lat_prior = None
        self.src_prior = None
        self.tgt_prior = None

        # 调度
        self.T = 128
        self.beta = 'cosine'
        self.loss_weight = 'sigma2'
        self.alphas_cumprod = None
        self.betas = None

        # 统计信息仅用于日志
        self.cond_dim = 0
        self.cond_keys = []

    def load_from_dir(self, cfg: Dict[str, Any],
                      cond_dim: int, src_act_dim: int, tgt_act_dim: int, lat_act_dim: int):
        """
        cfg = {
          "priors": { "dir": ..., "use_ema": true, "t_embed": "ddpm"/"edm" },
          "enabled": bool,
          "lambda": { "lat":.., "src":.., "tgt":.. },
          "schedule_override": { "use_override": bool, "T":.., "beta":.., "loss_weight":.. },
          "cond_dim": int,
          "cond_keys": [..]
        }
        """
        self.enabled = bool(cfg.get("enabled", False))
        self.lambda_lat = float(((cfg.get("lambda") or {}).get("lat", 1.0)))
        self.lambda_src = float(((cfg.get("lambda") or {}).get("src", 1.0)))
        self.lambda_tgt = float(((cfg.get("lambda") or {}).get("tgt", 1.0)))
        self.cond_dim = int(cfg.get("cond_dim", 0))
        self.cond_keys = list(cfg.get("cond_keys", []))
        if cond_dim != self.cond_dim:
            print(f"[diffusion-act] warn: cond_dim from align={cond_dim}, bundle={self.cond_dim}")

        pri_cfg = cfg.get("priors", {}) or {}
        root = pathlib.Path(pri_cfg.get("dir", ""))
        if self.enabled:
            if (not root.exists()) or (not (root / "models").exists()):
                raise RuntimeError(f"[diffusion-act] priors.dir 不存在或无 models 子目录: {root}")
        use_ema = bool(pri_cfg.get("use_ema", True))
        self.t_embed = str(pri_cfg.get("t_embed", "ddpm"))

        # 调度
        sched_ov = (cfg.get("schedule_override") or {})
        use_ov = bool(sched_ov.get("use_override", False))
        if use_ov:
            self.T = int(sched_ov.get("T", 128))
            self.beta = str(sched_ov.get("beta", "cosine"))
            self.loss_weight = str(sched_ov.get("loss_weight", "sigma2"))
        else:
            try:
                from ruamel.yaml import YAML
                sched = YAML(typ='safe').load(open(root / "stats" / "schedule.yml"))
                self.T = int(sched.get("T", 128))
                self.beta = str(sched.get("beta", "cosine"))
                self.loss_weight = str(sched.get("loss_weight", "sigma2"))
            except Exception:
                self.T, self.beta, self.loss_weight = 128, "cosine", "sigma2"

        s = _make_ddpm_schedule(self.T, self.beta)
        self.betas = s["betas"].to(self.device)
        self.alphas_cumprod = s["alphas_cumprod"].to(self.device)

        if not self.enabled:
            print("[diffusion-act] disabled by config.")
            return

        # 载入三套先验（动作）
        def _load_one(name: str, x_dim: int):
            fname = f"{name}_ema.pt" if use_ema else f"{name}.pt"
            path = root / "models" / fname
            if not path.exists():
                raise RuntimeError(f"[diffusion-act] 模型不存在: {path}")
            prior = PriorModel(x_dim=x_dim, c_dim=self.cond_dim, hidden=512, n_layers=4,
                               use_condition=(self.cond_dim > 0),
                               time_embed_dim=128, mode=self.t_embed).to(self.device)
            sd = torch.load(path, map_location=self.device)
            try:
                prior.load_state_dict(sd, strict=True)
            except RuntimeError as e:
                # 回退：根据 ckpt 动态重建（保证 IO 维一致）
                prior = _rebuild_prior_to_match_ckpt(
                    prior, sd, x_dim=x_dim, c_dim=self.cond_dim, t_dim=128, mode=self.t_embed
                )
            for p in prior.parameters():
                p.requires_grad = False
            prior.eval()
            return prior

        self.lat_prior = _load_one("score_lat_act", lat_act_dim)
        self.src_prior = _load_one("score_src_act", src_act_dim)
        self.tgt_prior = _load_one("score_tgt_act", tgt_act_dim)

        # 写 normalization
        try:
            from ruamel.yaml import YAML as _YAML
            norm_path = root / "stats" / "normalization.yml"
            norm = _YAML(typ='safe').load(open(norm_path, "r"))
        except Exception as e:
            raise RuntimeError(f"[diffusion-act] 读取 normalization.yml 失败: {e}")

        def _to_tensor(x, like: torch.Tensor):
            return torch.as_tensor(x, dtype=like.dtype, device=like.device)

        latent_mu = norm["latent_act"]["mu"]; latent_std = norm["latent_act"]["std"]
        src_mu    = norm["src_act"]["mu"];    src_std  = norm["src_act"]["std"]
        tgt_mu    = norm["tgt_act"]["mu"];    tgt_std  = norm["tgt_act"]["std"]

        cond_src_mu = norm["cond_src"]["mu"]; cond_src_std = norm["cond_src"]["std"]
        cond_tgt_mu = norm["cond_tgt"]["mu"]; cond_tgt_std = norm["cond_tgt"]["std"]

        # 形状强校验
        if len(latent_mu) != lat_act_dim or len(latent_std) != lat_act_dim:
            raise RuntimeError(f"[diffusion-act] latent_act 统计长度不等于 lat_act_dim: "
                               f"{len(latent_mu)}/{len(latent_std)} vs {lat_act_dim}")
        if len(src_mu) != src_act_dim or len(src_std) != src_act_dim:
            raise RuntimeError(f"[diffusion-act] src_act 统计长度不等于 src_act_dim: "
                               f"{len(src_mu)}/{len(src_std)} vs {src_act_dim}")
        if len(tgt_mu) != tgt_act_dim or len(tgt_std) != tgt_act_dim:
            raise RuntimeError(f"[diffusion-act] tgt_act 统计长度不等于 tgt_act_dim: "
                               f"{len(tgt_mu)}/{len(tgt_std)} vs {tgt_act_dim}")
        if self.cond_dim > 0:
            if len(cond_src_mu) != self.cond_dim or len(cond_src_std) != self.cond_dim:
                raise RuntimeError(f"[diffusion-act] cond_src 统计长度不等于 cond_dim: "
                                   f"{len(cond_src_mu)}/{len(cond_src_std)} vs {self.cond_dim}")
            if len(cond_tgt_mu) != self.cond_dim or len(cond_tgt_std) != self.cond_dim:
                raise RuntimeError(f"[diffusion-act] cond_tgt 统计长度不等于 cond_dim: "
                                   f"{len(cond_tgt_mu)}/{len(cond_tgt_std)} vs {self.cond_dim}")

        # 回写 buffer
        self.lat_prior.mu_x.data.copy_(_to_tensor(latent_mu, self.lat_prior.mu_x))
        self.lat_prior.std_x.data.copy_(_to_tensor(latent_std, self.lat_prior.std_x))
        if self.lat_prior.c_dim > 0:
            self.lat_prior.mu_c.data.copy_(_to_tensor(cond_src_mu, self.lat_prior.mu_c))
            self.lat_prior.std_c.data.copy_(_to_tensor(cond_src_std, self.lat_prior.std_c))

        self.src_prior.mu_x.data.copy_(_to_tensor(src_mu, self.src_prior.mu_x))
        self.src_prior.std_x.data.copy_(_to_tensor(src_std, self.src_prior.std_x))
        if self.src_prior.c_dim > 0:
            self.src_prior.mu_c.data.copy_(_to_tensor(cond_src_mu, self.src_prior.mu_c))
            self.src_prior.std_c.data.copy_(_to_tensor(cond_src_std, self.src_prior.std_c))

        self.tgt_prior.mu_x.data.copy_(_to_tensor(tgt_mu, self.tgt_prior.mu_x))
        self.tgt_prior.std_x.data.copy_(_to_tensor(tgt_std, self.tgt_prior.std_x))
        if self.tgt_prior.c_dim > 0:
            self.tgt_prior.mu_c.data.copy_(_to_tensor(cond_tgt_mu, self.tgt_prior.mu_c))
            self.tgt_prior.std_c.data.copy_(_to_tensor(cond_tgt_std, self.tgt_prior.std_c))

        def _peek(name, p):
            print(f"[diffusion-act:{name}] x_dim={p.x_dim}, c_dim={p.c_dim}, "
                  f"mu_x.mean={p.mu_x.mean().item():.4g}, std_x.mean={p.std_x.mean().item():.4g}, "
                  f"mu_c.mean={(p.mu_c.mean().item() if p.c_dim>0 else float('nan')):.4g}, "
                  f"std_c.mean={(p.std_c.mean().item() if p.c_dim>0 else float('nan')):.4g}")
        _peek("lat_act", self.lat_prior)
        _peek("src_act", self.src_prior)
        _peek("tgt_act", self.tgt_prior)

        if self.lat_prior.c_dim != self.cond_dim or \
           self.src_prior.c_dim != self.cond_dim or \
           self.tgt_prior.c_dim != self.cond_dim:
            raise RuntimeError(f"[diffusion-act] c_dim 不匹配：prior(c_dim)≠ cond_dim={self.cond_dim}")

        print(f"[diffusion-act] loaded priors from {root} | use_ema={use_ema} | "
              f"T={self.T}, beta={self.beta}, loss_weight={self.loss_weight}, "
              f"t_embed={self.t_embed}, cond_dim={self.cond_dim}, keys={self.cond_keys}")

    # 复用与 obs 相同的 DDPM 噪声回归损失
    def _loss_one(self, prior: PriorModel, x: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        if prior is None:
            return torch.tensor(0.0, device=self.device)
        B = x.shape[0]
        T = self.T
        t_idx = torch.randint(0, T, (B,), device=self.device)
        a_bar_t = self.alphas_cumprod[t_idx]
        noise = torch.randn_like(x)
        x_t = a_bar_t.sqrt().unsqueeze(-1) * x + (1.0 - a_bar_t).sqrt().unsqueeze(-1) * noise
        t_float = (t_idx.to(torch.float32) + 0.5) / float(T)
        eps_hat = prior(x_t, c, t_float)
        w = _ddpm_weight(a_bar_t, self.loss_weight).unsqueeze(-1)
        return (w * (eps_hat - noise) ** 2).mean()

    def loss_lat(self, z_a_tgt: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        return self._loss_one(self.lat_prior, z_a_tgt, c)

    def loss_src(self, fake_src_a: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        return self._loss_one(self.src_prior, fake_src_a, c)

    def loss_tgt(self, real_tgt_a: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        return self._loss_one(self.tgt_prior, real_tgt_a, c)



# -----------------------------
# 对齐器：Observation 版 + ObsAct 版
# -----------------------------
class ObsAligner:
    def __init__(
        self, 
        src_agent, 
        tgt_agent, 
        device, 
        n_layers=3, 
        hidden_dim=256,
        lr=3e-4,
        lmbd_gp=10,
        log_freq=1000,
        # ===== 新增：对齐权重（显式传入，不改变默认行为）=====
        gan_lambda: Optional[Dict[str, Any]] = None,
        diffusion_bundle_cfg: Optional[Dict[str, Any]] = None,
        # ===== 新增：cycle/dynamics 权重（保持原默认值）=====
        cycle_lambda: float = 10.0,
        dynamics_lambda: float = 10.0,
        # ===== 新增/旧：SEW 相关 =====
        sew_cfg: Optional[Dict[str, Any]] = None,
        loss_huber_delta: Optional[float] = None,  # 若不为 None，用 SmoothL1
    ):
        self.device = device
        self.lmbd_gp = lmbd_gp
        self.lmbd_cyc = cycle_lambda
        self.lmbd_dyn = dynamics_lambda
        self.log_freq = log_freq

        self.src_obs_enc = src_agent.obs_enc
        self.src_obs_dec = src_agent.obs_dec
        self.tgt_obs_enc = tgt_agent.obs_enc
        self.tgt_obs_dec = tgt_agent.obs_dec
        self.fwd_dyn = src_agent.fwd_dyn
        self.inv_dyn = src_agent.inv_dyn

        assert src_agent.lat_obs_dim == tgt_agent.lat_obs_dim
        self.lat_obs_dim = src_agent.lat_obs_dim
        self.src_obs_dim = src_agent.robot_obs_dim
        self.tgt_obs_dim = tgt_agent.robot_obs_dim

        self.lat_disc = utils.build_mlp(self.lat_obs_dim, 1, n_layers, hidden_dim,
            activation='leaky_relu', output_activation='identity').to(self.device)
        self.src_disc = utils.build_mlp(self.src_obs_dim, 1, n_layers, hidden_dim,
            activation='leaky_relu', output_activation='identity').to(self.device)
        self.tgt_disc = utils.build_mlp(self.tgt_obs_dim, 1, n_layers, hidden_dim,
            activation='leaky_relu', output_activation='identity').to(self.device)

        # Optimizers（仅本脚本训练的模块有优化器）
        self.tgt_obs_enc_opt = torch.optim.Adam(self.tgt_obs_enc.parameters(), lr=lr)
        self.tgt_obs_dec_opt = torch.optim.Adam(self.tgt_obs_dec.parameters(), lr=lr)
        self.lat_disc_opt = torch.optim.Adam(self.lat_disc.parameters(), lr=lr)
        self.src_disc_opt = torch.optim.Adam(self.src_disc.parameters(), lr=lr)
        self.tgt_disc_opt = torch.optim.Adam(self.tgt_disc.parameters(), lr=lr)

        # =========== GAN 开关/权重 ===========
        gan_lambda = gan_lambda or {}
        self.gan_enabled = bool(gan_lambda.get("enabled", True))
        self.gan_lmbd_lat = float(gan_lambda.get("lat", 1.0))
        self.gan_lmbd_src = float(gan_lambda.get("src", 1.0))
        self.gan_lmbd_tgt = float(gan_lambda.get("tgt", 1.0))
        


        # =========== Diffusion 先验 ===========
        self.priors: Optional[DiffusionPriors] = None
        if diffusion_bundle_cfg is not None:
            self.priors = DiffusionPriors(self.device)
            # 注意：lat/src/tgt 维度从 agent 中拿，cond_dim 由 train_align 传入
            self.priors.load_from_dir(
                cfg=diffusion_bundle_cfg,
                cond_dim=int(diffusion_bundle_cfg.get("cond_dim", 0)),
                src_robot_dim=self.src_obs_dim,
                tgt_robot_dim=self.tgt_obs_dim,
                lat_dim=self.lat_obs_dim
            )

        # =========== SEW 相关（保持原逻辑） ===========
        self.sew_enabled = False
        self.sew_lambda = 0.0
        self.sew_ratio_cross = 0.7
        self.sew_ratio_self = 0.3
        self.sew_every = 1
        self.sew_max_batch = 256
        self.sew_train_all = False
        self.loss_huber_delta = loss_huber_delta

        self.sew_src_pred: Optional[SEWPredictor] = None
        self.sew_tgt_pred: Optional[SEWPredictor] = None
        self.sew_input_mode = "robot_obs"   # "robot_obs" | "q"
        self.src_dof = self.src_obs_dim     # 仅在 "q" 模式下有意义
        self.tgt_dof = self.tgt_obs_dim
        self.src_qpos_slice = None          # [start, length]，仅 "q" 模式下应用
        self.tgt_qpos_slice = None

        if sew_cfg is not None and sew_cfg.get("enabled", False):
            self._init_sew_from_cfg(sew_cfg)

    # --------- SEW 初始化 ----------
    def _init_sew_from_cfg(self, cfg: Dict[str, Any]):
        self.sew_enabled = True
        self.sew_lambda = float(cfg.get("lambda", 0.0))
        self.sew_ratio_cross = float(cfg.get("ratio_cross", 0.7))
        self.sew_ratio_self  = float(cfg.get("ratio_self", 0.3))
        self.sew_every = int(cfg.get("every", 1))
        self.sew_max_batch = int(cfg.get("max_batch", 256))
        self.sew_train_all = bool(cfg.get("train_all", False))
        self.sew_input_mode = cfg.get("input", "robot_obs")

        # 读取 dof / slice（仅在 "q" 模式下有效）
        self.src_dof = int(cfg.get("src_dof", self.src_obs_dim))
        self.tgt_dof = int(cfg.get("tgt_dof", self.tgt_obs_dim))
        self.src_qpos_slice = cfg.get("src_qpos_slice", None)  # e.g., [0, 6]
        self.tgt_qpos_slice = cfg.get("tgt_qpos_slice", None)

        # 计算各域 predictor 的输入维
        def _infer_in_dim(which: str) -> int:
            if self.sew_input_mode == "q":
                if which == "src":
                    if self.src_qpos_slice is not None:
                        return int(self.src_qpos_slice[1])
                    return self.src_dof
                else:
                    if self.tgt_qpos_slice is not None:
                        return int(self.tgt_qpos_slice[1])
                    return self.tgt_dof
            else:
                # robot_obs 模式：直接用对应域的 robot_obs 维度
                return self.src_obs_dim if which == "src" else self.tgt_obs_dim

        # 加载两个预测器
        def _load_predictor(block: Dict[str, Any], which: str) -> SEWPredictor:
            path = block["path"]
            mean, std = None, None
            norm = block.get("norm", None)
            if norm is not None:
                mean = torch.tensor(norm["mean"], dtype=torch.float32, device=self.device)
                std  = torch.tensor(norm["std"],  dtype=torch.float32, device=self.device)
            in_dim = _infer_in_dim(which)
            return SEWPredictor(path, device=self.device, in_dim=in_dim, out_dim=1, hidden=256,
                                mean=mean, std=std)

        self.sew_src_pred = _load_predictor(cfg["src_predictor"], "src")
        self.sew_tgt_pred = _load_predictor(cfg["tgt_predictor"], "tgt")

        print(f"[SEW] enabled | λ={self.sew_lambda}, cross={self.sew_ratio_cross}, self={self.sew_ratio_self}, "
              f"every={self.sew_every}, max_batch={self.sew_max_batch}, train_all={self.sew_train_all}, "
              f"input={self.sew_input_mode}, src_dof={self.src_dof}, tgt_dof={self.tgt_dof}, "
              f"src_qpos_slice={self.src_qpos_slice}, tgt_qpos_slice={self.tgt_qpos_slice}")

    # --------- 对抗的梯度惩罚 ----------
    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        alpha = torch.rand((real_samples.size(0), 1)).to(self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones(real_samples.shape[0], 1, requires_grad=False, device=self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # --------- 判别器更新（保持原逻辑） ----------
    def update_disc(self, src_obs, src_act, tgt_obs, tgt_act, L, step):
        fake_lat_obs = self.tgt_obs_enc(tgt_obs).detach()
        real_lat_obs = self.src_obs_enc(src_obs).detach()
        lat_disc_loss = self.lat_disc(fake_lat_obs).mean() - self.lat_disc(real_lat_obs).mean()

        fake_src_obs = self.src_obs_dec(fake_lat_obs).detach()
        src_disc_loss = self.src_disc(fake_src_obs).mean() - self.src_disc(src_obs).mean()
        
        real_tgt_obs = self.tgt_obs_dec(self.src_obs_enc(src_obs)).detach()
        tgt_disc_loss = self.tgt_disc(real_tgt_obs).mean() - self.tgt_disc(tgt_obs).mean()

        lat_gp = self.compute_gradient_penalty(self.lat_disc, real_lat_obs, fake_lat_obs)
        src_gp = self.compute_gradient_penalty(self.src_disc, src_obs, fake_src_obs)
        tgt_gp = self.compute_gradient_penalty(self.tgt_disc, tgt_obs, real_tgt_obs)

        disc_loss = lat_disc_loss + src_disc_loss + tgt_disc_loss + \
            self.lmbd_gp * (lat_gp + src_gp + tgt_gp)

        self.lat_disc_opt.zero_grad()
        self.src_disc_opt.zero_grad()
        self.tgt_disc_opt.zero_grad()
        disc_loss.backward()
        self.lat_disc_opt.step()
        self.src_disc_opt.step()
        self.tgt_disc_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_lat_disc/lat_disc_loss', lat_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/lat_gp', lat_gp.item(), step)
            L.add_scalar('train_lat_disc/src_disc_loss', src_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/src_gp', src_gp.item(), step)
            L.add_scalar('train_lat_disc/tgt_disc_loss', tgt_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/tgt_gp', tgt_gp.item(), step)

    # --------- 生成器更新（原版 + Diffusion + SEW） ----------
    def update_gen(self, src_obs, src_act, src_next_obs, tgt_obs, tgt_act, tgt_next_obs, L, step,
                   c_src: Optional[torch.Tensor] = None, c_tgt: Optional[torch.Tensor] = None):
        # --- GAN 主损失（与原版一致）---
        fake_lat_obs = self.tgt_obs_enc(tgt_obs)
        lat_gen_loss = -self.lat_disc(fake_lat_obs).mean()

        fake_src_obs = self.src_obs_dec(fake_lat_obs)
        src_gen_loss = -self.src_disc(fake_src_obs).mean()

        real_tgt_obs = self.tgt_obs_dec(self.src_obs_enc(src_obs))
        tgt_gen_loss = -self.tgt_disc(real_tgt_obs).mean()

        gen_loss = 0.0
        if self.gan_enabled:
            gen_loss = (self.gan_lmbd_lat * lat_gen_loss +
                        self.gan_lmbd_src * src_gen_loss +
                        self.gan_lmbd_tgt * tgt_gen_loss)

        # --- Cycle（与原版一致）---
        pred_src_obs = self.src_obs_dec(self.tgt_obs_enc(real_tgt_obs))
        pred_tgt_obs = self.tgt_obs_dec(self.src_obs_enc(fake_src_obs))
        cycle_loss = F.l1_loss(pred_src_obs, src_obs) + F.l1_loss(pred_tgt_obs, tgt_obs)

        # --- Dynamics（与原版一致）---
        fake_lat_next_obs = self.tgt_obs_enc(tgt_next_obs)
        # inverse: (z, z') -> a_wo_g
        pred_act = self.inv_dyn(torch.cat([fake_lat_obs, fake_lat_next_obs], dim=-1))
        inv_loss = F.mse_loss(pred_act, tgt_act)
        # forward: (z, a_wo_g) -> z'
        pred_lat_next_obs = self.fwd_dyn(torch.cat([fake_lat_obs, tgt_act], dim=-1))
        fwd_loss = F.mse_loss(pred_lat_next_obs, fake_lat_next_obs)

        # --- 新增：Diffusion 先验（支持三路 + 权重）---
        diff_loss_total = torch.tensor(0.0, device=self.device)
        if (self.priors is not None) and self.priors.enabled:
            # z_tgt
            L_lat = self.priors.loss_lat(fake_lat_obs, c_tgt)
            # fake_src on src-robot space
            L_src = self.priors.loss_src(fake_src_obs, c_tgt)
            # real tgt-robot（最稳）
            L_tgt = self.priors.loss_tgt(real_tgt_obs, c_src)
            diff_loss_total = (self.priors.lambda_lat * L_lat +
                               self.priors.lambda_src * L_src +
                               self.priors.lambda_tgt * L_tgt)

            if step % self.log_freq == 0:
                L.add_scalar('train_prior/lat', L_lat.item(), step)
                L.add_scalar('train_prior/src', L_src.item(), step)
                L.add_scalar('train_prior/tgt', L_tgt.item(), step)
                L.add_scalar('train_prior/total', diff_loss_total.item(), step)

        # 总损失（保持原名项权重不变，再加 diffusion）
        loss = gen_loss + self.lmbd_cyc * cycle_loss + self.lmbd_dyn * (inv_loss + fwd_loss) + diff_loss_total

        self.tgt_obs_enc_opt.zero_grad()
        self.tgt_obs_dec_opt.zero_grad()
        loss.backward()
        self.tgt_obs_enc_opt.step()
        self.tgt_obs_dec_opt.step()

        if step % self.log_freq == 0:
            if self.gan_enabled:
                L.add_scalar('train_lat_gen/lat_gen_loss', lat_gen_loss.item(), step)
                L.add_scalar('train_lat_gen/src_gen_loss', src_gen_loss.item(), step)
                L.add_scalar('train_lat_gen/tgt_gen_loss', tgt_gen_loss.item(), step)
            L.add_scalar('train_lat_gen/inv_loss', inv_loss.item(), step)
            L.add_scalar('train_lat_gen/fwd_loss', fwd_loss.item(), step)
            L.add_scalar('train_lat_gen/cycle_loss', cycle_loss.item(), step)

    # --- 可选的 Huber 损失（稳定些） ---
    def _sew_loss_fn(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_huber_delta is None:
            return F.mse_loss(pred, target)
        return F.smooth_l1_loss(pred, target, beta=self.loss_huber_delta)

    # ----------------- 供训练脚本保存/恢复（含优化器） -----------------
    def state_dict(self) -> Dict[str, Any]:
        d = {
            "tgt_obs_enc": self.tgt_obs_enc.state_dict(),
            "tgt_obs_dec": self.tgt_obs_dec.state_dict(),
            "lat_disc": self.lat_disc.state_dict(),
            "src_disc": self.src_disc.state_dict(),
            "tgt_disc": self.tgt_disc.state_dict(),
            "opt": {
                "tgt_obs_enc_opt": self.tgt_obs_enc_opt.state_dict(),
                "tgt_obs_dec_opt": self.tgt_obs_dec_opt.state_dict(),
                "lat_disc_opt": self.lat_disc_opt.state_dict(),
                "src_disc_opt": self.src_disc_opt.state_dict(),
                "tgt_disc_opt": self.tgt_disc_opt.state_dict(),
            },
            # 记录一下开关与权重（便于 resume 观察），先验权重与开关
            "gan": {
                "enabled": self.gan_enabled,
                "lambda_lat": self.gan_lmbd_lat,
                "lambda_src": self.gan_lmbd_src,
                "lambda_tgt": self.gan_lmbd_tgt,
            },
            "priors_meta": {
                "enabled": (self.priors is not None and self.priors.enabled),
                "lambda_lat": (0.0 if self.priors is None else self.priors.lambda_lat),
                "lambda_src": (0.0 if self.priors is None else self.priors.lambda_src),
                "lambda_tgt": (0.0 if self.priors is None else self.priors.lambda_tgt),
            }
        }
        return d

    def load_state_dict(self, sd: Dict[str, Any], strict: bool=True):
        self.tgt_obs_enc.load_state_dict(sd["tgt_obs_enc"], strict=strict)
        self.tgt_obs_dec.load_state_dict(sd["tgt_obs_dec"], strict=strict)
        self.lat_disc.load_state_dict(sd["lat_disc"], strict=strict)
        self.src_disc.load_state_dict(sd["src_disc"], strict=strict)
        self.tgt_disc.load_state_dict(sd["tgt_disc"], strict=strict)

        # 优化器
        if "opt" in sd:
            self.tgt_obs_enc_opt.load_state_dict(sd["opt"]["tgt_obs_enc_opt"])
            self.tgt_obs_dec_opt.load_state_dict(sd["opt"]["tgt_obs_dec_opt"])
            self.lat_disc_opt.load_state_dict(sd["opt"]["lat_disc_opt"])
            self.src_disc_opt.load_state_dict(sd["opt"]["src_disc_opt"])
            self.tgt_disc_opt.load_state_dict(sd["opt"]["tgt_disc_opt"])

        # 读回开关/权重仅用于日志参考（不强制覆盖当前 config）
        # （如果你想严格复现，亦可在此覆盖）


class ObsActAligner(ObsAligner):
    def __init__(
        self, 
        src_agent, 
        tgt_agent, 
        device, 
        n_layers=3, 
        hidden_dim=256,
        lr=3e-4,
        lmbd_gp=10,
        log_freq=1000,
        gan_lambda: Optional[Dict[str, Any]] = None,
        diffusion_bundle_cfg: Optional[Dict[str, Any]] = None,
        diffusion_act_bundle_cfg: Optional[Dict[str, Any]] = None, 
        cycle_lambda: float = 10.0,
        dynamics_lambda: float = 10.0,
        sew_cfg: Optional[Dict[str, Any]] = None,
        loss_huber_delta: Optional[float] = None,
    ):
        super().__init__(src_agent, tgt_agent, device, n_layers=n_layers, 
            hidden_dim=hidden_dim, lr=lr, lmbd_gp=lmbd_gp, log_freq=log_freq,
            gan_lambda=gan_lambda, diffusion_bundle_cfg=diffusion_bundle_cfg,
            cycle_lambda=cycle_lambda, dynamics_lambda=dynamics_lambda,
            sew_cfg=sew_cfg, loss_huber_delta=loss_huber_delta)

        assert src_agent.lat_act_dim == tgt_agent.lat_act_dim
        self.lat_act_dim = src_agent.lat_act_dim
        self.src_act_dim = src_agent.act_dim - 1
        self.tgt_act_dim = tgt_agent.act_dim - 1

        self.src_act_enc = src_agent.act_enc 
        self.src_act_dec = src_agent.act_dec
        self.tgt_act_enc = tgt_agent.act_enc 
        self.tgt_act_dec = tgt_agent.act_dec
        
        # =========== 动作分支 Diffusion 先验 ===========
        self.priors_act: Optional[ActionDiffusionPriors] = None
        if diffusion_act_bundle_cfg is not None:
            self.priors_act = ActionDiffusionPriors(self.device)
            self.priors_act.load_from_dir(
                cfg=diffusion_act_bundle_cfg,
                cond_dim=int(diffusion_act_bundle_cfg.get("cond_dim", 0)),
                src_act_dim=self.src_act_dim,
                tgt_act_dim=self.tgt_act_dim,
                lat_act_dim=self.lat_act_dim
            )

        if self.priors_act is None:
            print("[DEBUG] Diffusion ACT priors = None (disabled)")
        else:
            print(f"[DEBUG] Diffusion-ACT enabled = {self.priors_act.enabled}, "
                  f"λ = ({self.priors_act.lambda_lat}, {self.priors_act.lambda_src}, {self.priors_act.lambda_tgt}), "
                  f"cond_dim = {self.priors_act.cond_dim}")        

        # 判别器（拼接 z 和 z_a）
        self.lat_disc = utils.build_mlp(self.lat_obs_dim + self.lat_act_dim, 1, n_layers, 
            hidden_dim, activation='leaky_relu', output_activation='identity').to(self.device)
        self.src_disc = utils.build_mlp(self.src_obs_dim + self.src_act_dim, 1, n_layers, 
            hidden_dim, activation='leaky_relu', output_activation='identity').to(self.device)
        self.tgt_disc = utils.build_mlp(self.tgt_obs_dim + self.tgt_act_dim, 1, n_layers, 
            hidden_dim, activation='leaky_relu', output_activation='identity').to(self.device)
        print(f"[DEBUG] GAN enabled = {self.gan_enabled}, "
            f"GAN λ = ({self.gan_lmbd_lat}, {self.gan_lmbd_src}, {self.gan_lmbd_tgt})")

        if self.priors is None:
            print("[DEBUG] Diffusion priors = None (disabled)")
        else:
            print(f"[DEBUG] Diffusion enabled = {self.priors.enabled}, "
                f"λ = ({self.priors.lambda_lat}, {self.priors.lambda_src}, {self.priors.lambda_tgt}), "
                f"cond_dim = {self.priors.cond_dim}")        
        # 目标侧动作分支也参与优化
        self.tgt_act_enc_opt = torch.optim.Adam(self.tgt_act_enc.parameters(), lr=lr)
        self.tgt_act_dec_opt = torch.optim.Adam(self.tgt_act_dec.parameters(), lr=lr)
        self.lat_disc_opt = torch.optim.Adam(self.lat_disc.parameters(), lr=lr)
        self.src_disc_opt = torch.optim.Adam(self.src_disc.parameters(), lr=lr)
        self.tgt_disc_opt = torch.optim.Adam(self.tgt_disc.parameters(), lr=lr)
        
        

    def update_disc(self, src_obs, src_act, tgt_obs, tgt_act, L, step):
        # 关 GAN：完全跳过 D 更新与日志
        if not self.gan_enabled:
            # if step % self.log_freq == 0:
            #     # 这行也可以注释掉，免得输出太多
            #     # print(f"[DEBUG][update_disc] GAN disabled -> skip D update at step={step}")
            return
        fake_lat_obs = self.tgt_obs_enc(tgt_obs).detach()
        fake_lat_act = self.tgt_act_enc(torch.cat([tgt_obs, tgt_act], dim=-1)).detach()
        real_lat_obs = self.src_obs_enc(src_obs).detach()
        real_lat_act = self.src_act_enc(torch.cat([src_obs, src_act], dim=-1)).detach()

        fake_lat_input = torch.cat([fake_lat_obs, fake_lat_act], dim=-1)
        real_lat_input = torch.cat([real_lat_obs, real_lat_act], dim=-1)
        lat_disc_loss = self.lat_disc(fake_lat_input).mean() - self.lat_disc(real_lat_input).mean()

        fake_src_obs = self.src_obs_dec(fake_lat_obs).detach()
        fake_src_act = self.src_act_dec(torch.cat([fake_src_obs, fake_lat_act], dim=-1)).detach()
        fake_src_input = torch.cat([fake_src_obs, fake_src_act], dim=-1)
        src_input = torch.cat([src_obs, src_act], dim=-1)
        src_disc_loss = self.src_disc(fake_src_input).mean() - self.src_disc(src_input).mean()

        real_tgt_obs = self.tgt_obs_dec(real_lat_obs).detach()
        real_tgt_act = self.tgt_act_dec(torch.cat([real_tgt_obs, real_lat_act], dim=-1)).detach()
        real_tgt_input = torch.cat([real_tgt_obs, real_tgt_act], dim=-1)
        tgt_input = torch.cat([tgt_obs, tgt_act], dim=-1)
        tgt_disc_loss = self.tgt_disc(real_tgt_input).mean() - self.tgt_disc(tgt_input).mean()

        lat_gp = self.compute_gradient_penalty(self.lat_disc, real_lat_input, fake_lat_input)
        src_gp = self.compute_gradient_penalty(self.src_disc, src_input, fake_src_input)
        tgt_gp = self.compute_gradient_penalty(self.tgt_disc, tgt_input, real_tgt_input)

        disc_loss = lat_disc_loss + src_disc_loss + tgt_disc_loss + self.lmbd_gp * (lat_gp + src_gp + tgt_gp)

        self.lat_disc_opt.zero_grad()
        self.src_disc_opt.zero_grad()
        self.tgt_disc_opt.zero_grad()
        disc_loss.backward()
        self.lat_disc_opt.step()
        self.src_disc_opt.step()
        self.tgt_disc_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_lat_disc/lat_disc_loss', lat_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/lat_gp', lat_gp.item(), step)
            L.add_scalar('train_lat_disc/src_disc_loss', src_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/src_gp', src_gp.item(), step)
            L.add_scalar('train_lat_disc/tgt_disc_loss', tgt_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/tgt_gp', tgt_gp.item(), step)

    def update_gen(self, src_obs, src_act, src_next_obs, tgt_obs, tgt_act, tgt_next_obs, L, step,
                   c_src: Optional[torch.Tensor] = None, c_tgt: Optional[torch.Tensor] = None,
                   c_src_act: Optional[torch.Tensor] = None, c_tgt_act: Optional[torch.Tensor] = None):
        
        # --- GAN 主损失 ---
        fake_lat_obs = self.tgt_obs_enc(tgt_obs)
        fake_lat_act = self.tgt_act_enc(torch.cat([tgt_obs, tgt_act], dim=-1))
        real_lat_obs = self.src_obs_enc(src_obs)
        real_lat_act = self.src_act_enc(torch.cat([src_obs, src_act], dim=-1))

        fake_lat_input = torch.cat([fake_lat_obs, fake_lat_act], dim=-1)
        real_lat_input = torch.cat([real_lat_obs, real_lat_act], dim=-1)
        lat_gen_loss = -self.lat_disc(fake_lat_input).mean()

        fake_src_obs = self.src_obs_dec(fake_lat_obs)
        fake_src_act = self.src_act_dec(torch.cat([fake_src_obs, fake_lat_act], dim=-1))
        fake_src_input = torch.cat([fake_src_obs, fake_src_act], dim=-1)
        src_input = torch.cat([src_obs, src_act], dim=-1)
        src_gen_loss = -self.src_disc(fake_src_input).mean()

        real_tgt_obs = self.tgt_obs_dec(real_lat_obs)
        real_tgt_act = self.tgt_act_dec(torch.cat([real_tgt_obs, real_lat_act], dim=-1))
        real_tgt_input = torch.cat([real_tgt_obs, real_tgt_act], dim=-1)
        tgt_input = torch.cat([tgt_obs, tgt_act], dim=-1)
        tgt_gen_loss = -self.tgt_disc(real_tgt_input).mean()

        gen_loss = 0.0
        if self.gan_enabled:
            gen_loss = (self.gan_lmbd_lat * lat_gen_loss +
                        self.gan_lmbd_src * src_gen_loss +
                        self.gan_lmbd_tgt * tgt_gen_loss)

        # --- Cycle 一致性 ---
        fake_lat_obs_1 = self.src_obs_enc(fake_src_obs)
        fake_lat_act_1 = self.src_act_enc(torch.cat([fake_src_obs, fake_src_act], dim=-1))
        pred_tgt_obs = self.tgt_obs_dec(fake_lat_obs_1)
        pred_tgt_act = self.tgt_act_dec(torch.cat([pred_tgt_obs, fake_lat_act_1], dim=-1))
        tgt_obs_cycle_loss = F.l1_loss(pred_tgt_obs, tgt_obs)
        tgt_act_cycle_loss = F.l1_loss(pred_tgt_act, tgt_act)

        real_lat_obs_1 = self.tgt_obs_enc(real_tgt_obs)
        real_lat_act_1 = self.tgt_act_enc(torch.cat([real_tgt_obs, real_tgt_act], dim=-1))
        pred_src_obs = self.src_obs_dec(real_lat_obs_1)
        pred_src_act = self.src_act_dec(torch.cat([pred_src_obs, real_lat_act_1], dim=-1))
        src_obs_cycle_loss = F.l1_loss(pred_src_obs, src_obs)
        src_act_cycle_loss = F.l1_loss(pred_src_act, src_act)
        cycle_loss = tgt_obs_cycle_loss + tgt_act_cycle_loss + src_obs_cycle_loss + src_act_cycle_loss

        # --- Dynamics ---
        fake_lat_next_obs = self.tgt_obs_enc(tgt_next_obs)
        pred_lat_act = self.inv_dyn(torch.cat([fake_lat_obs, fake_lat_next_obs], dim=-1))
        pred_act = self.tgt_act_dec(torch.cat([tgt_obs, pred_lat_act], dim=-1))
        inv_loss = F.mse_loss(pred_act, tgt_act)
        pred_lat_next_obs = self.fwd_dyn(torch.cat([fake_lat_obs, fake_lat_act], dim=-1))
        fwd_loss = F.mse_loss(pred_lat_next_obs, fake_lat_next_obs)

        # --- Diffusion 先验（与 Obs 对齐器一致） ---
        diff_loss_total = torch.tensor(0.0, device=self.device)
        if (self.priors is not None) and self.priors.enabled:
            L_lat = self.priors.loss_lat(fake_lat_obs, c_tgt)
            L_src = self.priors.loss_src(fake_src_obs, c_tgt)
            L_tgt = self.priors.loss_tgt(real_tgt_obs, c_src)
            diff_loss_total = (self.priors.lambda_lat * L_lat +
                               self.priors.lambda_src * L_src +
                               self.priors.lambda_tgt * L_tgt)
            if step % self.log_freq == 0:
                L.add_scalar('train_prior/lat', L_lat.item(), step)
                L.add_scalar('train_prior/src', L_src.item(), step)
                L.add_scalar('train_prior/tgt', L_tgt.item(), step)
                L.add_scalar('train_prior/total', diff_loss_total.item(), step)

        # --- 动作分支 Diffusion 先验（与 obs 规则一致） ---
        diff_act_loss_total = torch.tensor(0.0, device=self.device)
        if (self.priors_act is not None) and self.priors_act.enabled:
            # 三路：latent-act / src-act / tgt-act
            L_lat_a = self.priors_act.loss_lat(fake_lat_act, c_tgt_act)
            L_src_a = self.priors_act.loss_src(fake_src_act, c_tgt_act)
            L_tgt_a = self.priors_act.loss_tgt(real_tgt_act, c_src_act)
            diff_act_loss_total = (self.priors_act.lambda_lat * L_lat_a +
                                   self.priors_act.lambda_src * L_src_a +
                                   self.priors_act.lambda_tgt * L_tgt_a)
            if step % self.log_freq == 0:
                L.add_scalar('train_prior_act/lat',   L_lat_a.item(), step)
                L.add_scalar('train_prior_act/src',   L_src_a.item(), step)
                L.add_scalar('train_prior_act/tgt',   L_tgt_a.item(), step)
                L.add_scalar('train_prior_act/total', diff_act_loss_total.item(), step)

        # 总损失（追加 act 先验）
        loss = gen_loss + self.lmbd_cyc * cycle_loss + self.lmbd_dyn * (inv_loss + fwd_loss) \
             + diff_loss_total + diff_act_loss_total


        self.tgt_obs_enc_opt.zero_grad()
        self.tgt_obs_dec_opt.zero_grad()
        self.tgt_act_enc_opt.zero_grad()
        self.tgt_act_dec_opt.zero_grad()
        loss.backward()
        self.tgt_obs_enc_opt.step()
        self.tgt_obs_dec_opt.step()
        self.tgt_act_enc_opt.step()
        self.tgt_act_dec_opt.step()

        if step % self.log_freq == 0:
            if self.gan_enabled:
                L.add_scalar('train_lat_gen/lat_gen_loss', lat_gen_loss.item(), step)
                L.add_scalar('train_lat_gen/src_gen_loss', src_gen_loss.item(), step)
                L.add_scalar('train_lat_gen/tgt_gen_loss', tgt_gen_loss.item(), step)
            L.add_scalar('train_lat_gen/inv_loss', inv_loss.item(), step)
            L.add_scalar('train_lat_gen/fwd_loss', fwd_loss.item(), step)
            L.add_scalar('train_lat_gen/cycle_loss', cycle_loss.item(), step)

    # 追加 act 分支优化器/权重到保存
    def state_dict(self) -> Dict[str, Any]:
        base = super().state_dict()
        base.update({
            "tgt_act_enc": self.tgt_act_enc.state_dict(),
            "tgt_act_dec": self.tgt_act_dec.state_dict(),
        })
        base["opt"].update({
            "tgt_act_enc_opt": self.tgt_act_enc_opt.state_dict(),
            "tgt_act_dec_opt": self.tgt_act_dec_opt.state_dict(),
        })
                # 仅记录开关与 λ，行为与 obs 先验保持一致（用于日志参考）
        base["priors_act_meta"] = {
            "enabled": (self.priors_act is not None and self.priors_act.enabled),
            "lambda_lat": (0.0 if self.priors_act is None else self.priors_act.lambda_lat),
            "lambda_src": (0.0 if self.priors_act is None else self.priors_act.lambda_src),
            "lambda_tgt": (0.0 if self.priors_act is None else self.priors_act.lambda_tgt),
        }
        return base
    
    

    def load_state_dict(self, sd: Dict[str, Any], strict: bool=True):
        super().load_state_dict(sd, strict=strict)
        self.tgt_act_enc.load_state_dict(sd["tgt_act_enc"], strict=strict)
        self.tgt_act_dec.load_state_dict(sd["tgt_act_dec"], strict=strict)
        if "opt" in sd:
            self.tgt_act_enc_opt.load_state_dict(sd["opt"]["tgt_act_enc_opt"])
            self.tgt_act_dec_opt.load_state_dict(sd["opt"]["tgt_act_dec_opt"])