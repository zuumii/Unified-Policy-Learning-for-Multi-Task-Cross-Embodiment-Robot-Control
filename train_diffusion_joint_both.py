# =========================================================
# Diffusion Priors Pretraining (Lat-Act / Src-Joint / Tgt-Joint)
# 方案B：统一“obs+act”
# - src_joint/tgt_joint: x = concat([robot_obs, a_arm])
# - latent_act         : x = concat([z_obs_src, z_act_src])
# 条件 cond：最小化配置
# - config.diffusion.cond.keys: [list of keys]
#   * 键 == "eef_action_vel" -> 在线计算 v = Jp(q) @ a_arm （追加到 obs/next_obs 尾部）
#   * 其他键：若 demos 中存在则直接取；不存在则忽略
#   * 最终若一个条件键也没拼上 -> use_condition=False
# 其余训练流程（DDPM/EDM/EMA/日志/保存）与原脚本保持一致
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
from tqdm import tqdm, trange

import utils                 # 需与 align 相同仓库内
import replay_buffer         # 需与 align 相同仓库内

# 尝试导入 mujoco（robosuite 后端）
try:
    import mujoco as mj
except Exception as _e:
    mj = None

# ----------------------------
# 常量与小工具
# ----------------------------
C_KEY_SENTINEL = "eef_action_vel"    # 触发在线计算 v = Jp(q) @ a_arm
# EEF_STACK_KEYS = ["robot0_eef_pos", "robot0_eef_lin_vel"]  # 常见现成键（仅示例，不强制）

DEFAULT_EEF_SITE_CANDIDATES = (
    "gripper0_eef", "gripper0_grip_site", "robot0_eef", "robot0_grip_site"
)

# ----------------------------
# DDPM 调度
# ----------------------------
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

# ----------------------------
# 时间嵌入
# ----------------------------
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

# ----------------------------
# 小 MLP 噪声预测器
# ----------------------------
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

# ----------------------------
# 先验包裹：含归一化 & 时间嵌入 & 条件分支
# ----------------------------
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
        # 归一化缓冲
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

# ----------------------------
# EMA
# ----------------------------
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

# ----------------------------
# Env / Buffer 构造与维度探测
# ----------------------------
@dataclass
class DomainDims:
    robot_obs_dim: int
    act_dim: int

def infer_action_dim(env) -> int:
    # 兼容不同 env API
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
    raise AttributeError("Cannot infer action dimension from env.")

def probe_robot_obs_dim(env_name: str, robot: str, controller: str, env_kwargs: Dict[str, Any],
                        robot_obs_keys: List[str]) -> int:
    probe_env = utils.make_robosuite_env(
        env_name, robots=robot, controller_type=controller, **(env_kwargs or {})
    )
    obs = probe_env.reset()
    rob = np.concatenate([obs[k] for k in robot_obs_keys])
    return int(rob.shape[0])

def build_env_for_buffer(env_name: str, robot: str, controller: str,
                         env_kwargs: Dict[str, Any], robot_obs_keys: List[str]):
    obs_keys = list(robot_obs_keys)
    env = utils.make(
        env_name, robots=robot, controller_type=controller,
        obs_keys=obs_keys, seed=0, **(env_kwargs or {})
    )
    return env, obs_keys

def build_replay_buffer_from_episodes(episodes_aug, device: torch.device, batch_size: int):
    # 用第一条 episode 判定形状
    if len(episodes_aug) == 0:
        raise ValueError(f"No episodes found")
    obs_shape = (episodes_aug[0]['obs'].shape[1],)
    act_shape = (episodes_aug[0]['action'].shape[1],)
    buf = replay_buffer.ReplayBuffer(
        obs_shape=obs_shape, action_shape=act_shape,
        capacity=int(1e7), batch_size=batch_size, device=device
    )
    buf.add_rollouts(episodes_aug)
    return buf

# ----------------------------
# EEF 相关：q 恢复 / site / Jp / p
# ----------------------------
def _atan2_recover_q(cos_arr: np.ndarray, sin_arr: np.ndarray) -> np.ndarray:
    assert cos_arr.shape == sin_arr.shape, "cos/sin shape mismatch"
    return np.arctan2(sin_arr, cos_arr)  # [T, nq]

def _eef_site_id(env) -> int:
    if mj is None:
        raise RuntimeError("MuJoCo not available for EEF Jacobian.")
    model = getattr(env.sim.model, "_model", env.sim.model)
    sid = -1
    # 常见命名优先
    for name in DEFAULT_EEF_SITE_CANDIDATES:
        try:
            s = mj.mj_name2id(model, int(mj.mjtObj.mjOBJ_SITE), name)
            if s != -1:
                return int(s)
        except Exception:
            pass
    # 回退：找包含 eef/grip 的 site
    try:
        nsite = int(model.nsite)
        for i in range(nsite):
            nm = mj.mj_id2name(model, int(mj.mjtObj.mjOBJ_SITE), i) or ""
            if ("eef" in nm) or ("grip" in nm):
                sid = i
                break
    except Exception:
        sid = -1
    if sid == -1:
        # 最终兜底
        return 0
    return int(sid)

def _set_qpos_and_forward(env, q: np.ndarray):
    data = env.sim.data
    n = q.shape[-1]
    data.qpos[:n] = q.astype(np.float64)
    env.sim.forward()

def _eef_lin_jacobian(env, site_id: int) -> np.ndarray:
    model = getattr(env.sim.model, "_model", env.sim.model)
    data = getattr(env.sim.data, "_data", env.sim.data)
    nv = int(model.nv)
    jacp = np.zeros((3, nv), dtype=np.float64, order="C")
    jacr = np.zeros((3, nv), dtype=np.float64, order="C")
    mj.mj_jacSite(model, data, jacp, jacr, int(site_id))
    return jacp

def _eef_site_pos(env, site_id: int) -> np.ndarray:
    data = getattr(env.sim.data, "_data", env.sim.data)
    return np.array(data.site_xpos[int(site_id)][:3], dtype=np.float64)

# ----------------------------
# 条件键解析与“内存增广”
# - 只在 keys 中出现 "eef_action_vel" 时在线计算 v=Jp(q)@a_arm；
# - 其余 keys 在 demos 存在则直接取；缺失就忽略；
# - 拼接顺序 == keys 顺序；若最终拼接维度==0，则训练时走无条件。
# ----------------------------
@dataclass
class CondPlan:
    keys: List[str]                  # 期望键序
    include_eef_action_vel: bool     # 是否包含特殊键
    final_dim: int                   # 追加后条件总维度
    appended_order: List[str]        # 实际被追加的键（缺失键已剔除）
    arm_dim_src: int                 # 用于 src 域 a_arm 维度
    arm_dim_tgt: int                 # 用于 tgt 域 a_arm 维度

def plan_condition(
    params: Dict[str, Any],
    src_env, tgt_env,
    episodes_src: List[Dict[str, np.ndarray]],
    episodes_tgt: List[Dict[str, np.ndarray]],
) -> CondPlan:
    # 配置最小化：可能不存在 diffusion.cond 或 cond.keys
    cond_cfg = params.get("diffusion", {}).get("cond", {}) or {}
    req_keys: List[str] = list(cond_cfg.get("keys", []) or [])
    include_eef = any(k == C_KEY_SENTINEL for k in req_keys)

    # 估计臂部维（默认 action_dim-1；<=0 时回退为全维并告警）
    def _arm_dim_from_env(env):
        ad = infer_action_dim(env)
        arm_dim = ad - 1
        if arm_dim <= 0:
            print(f"[warn] inferred arm_dim={arm_dim} (<=0). Fallback to full action dim={ad}.")
            arm_dim = ad
        return int(arm_dim)

    arm_dim_src = _arm_dim_from_env(src_env)
    arm_dim_tgt = _arm_dim_from_env(tgt_env)

    # 计算实际能追加的维度（逐键粗判定维数：eef_action_vel=3；其他键从首条 episode 推断维数）
    appended = []
    dim_sum = 0

    def _probe_dim_from_episode(eps_list: List[Dict[str, np.ndarray]], key: str) -> Optional[int]:
        for ep in eps_list:
            if key in ep:
                arr = ep[key]
                if isinstance(arr, np.ndarray):
                    last_dim = int(arr.shape[-1])
                    return last_dim
        return None

    for k in req_keys:
        if k == C_KEY_SENTINEL:
            appended.append(k)
            dim_sum += 3
        else:
            d_src = _probe_dim_from_episode(episodes_src, k)
            d_tgt = _probe_dim_from_episode(episodes_tgt, k)
            d = d_src if (d_src is not None) else d_tgt
            if d is None:
                print(f"[cond] key '{k}' not found in demos. It will be ignored.")
                continue
            appended.append(k)
            dim_sum += int(d)

    print(f"[cond] requested keys = {req_keys}")
    print(f"[cond] appended keys  = {appended} (dim={dim_sum})")
    return CondPlan(
        keys=req_keys,
        include_eef_action_vel=include_eef,
        final_dim=dim_sum,
        appended_order=appended,
        arm_dim_src=arm_dim_src,
        arm_dim_tgt=arm_dim_tgt,
    )

def augment_episodes_with_condition(
    task_name: str,
    robots: str,
    controller: str,
    root_dir: str,
    robot_obs_keys: List[str],
    env_kwargs: dict,
    cond_plan: CondPlan,
    arm_dim: int
) -> List[Dict[str, np.ndarray]]:
    """
    加载原始 episodes，并按 cond_plan 将条件列（按 appended_order 顺序）追加进 obs/next_obs 尾部。
    - 对于普通键：若 ep 中存在则 slice 对齐 T 后直接拼；
    - 对于 eef_action_vel：在线计算 v = Jp(q) @ a_arm（仅 3 维）并在 appended_order 的“该位置”拼接；
    - 若最终 appended_order 为空，则不改动 obs 形状。
    """
    in_dir = pathlib.Path(root_dir) / task_name / robots / controller

    # 用 robosuite 原生 env（非 GymWrapper）拿 mujoco 模型（仅在需要 eef_action_vel 时用）
    base_env = utils.make_robosuite_env(
        env_name=task_name, robots=robots, controller_type=controller,
        render=False, offscreen_render=False, **(env_kwargs or {})
    )
    site_id = _eef_site_id(base_env) if (cond_plan.include_eef_action_vel and (mj is not None)) else None

    # 需要把普通条件键一并加载（不含 sentinel），以及 eef_action_vel 计算所需的关节位姿键
    extra_keys = [k for k in cond_plan.appended_order if k != C_KEY_SENTINEL]
    joint_keys = []
    if cond_plan.include_eef_action_vel and (mj is not None):
        # 只加 cos/sin，不加 robot0_joint_pos（很多数据不包含）
        joint_keys = ['robot0_joint_pos_cos', 'robot0_joint_pos_sin']

    # 去重后作为 obs_keys 传给 loader，避免 KeyError
    load_keys = list(dict.fromkeys(list(robot_obs_keys) + extra_keys + joint_keys))
    episodes = utils.load_episodes(in_dir, obs_keys=load_keys, lat_obs_keys=None, capacity=None)

    out_eps = []
    for ep in tqdm(episodes, desc=f"[{robots}] build condition (in-memory)"):
        # ====== 1) 时间长度对齐基准 ======
        # 注意：下面不再直接使用 ep['obs'] 作为前半段；只用它来拿长度。
        obs = ep['obs']
        next_obs = ep['next_obs']
        T_obs = obs.shape[0]
        T_next = next_obs.shape[0]
        action = ep['action']               # [T_trans, D_act]
        T_trans = action.shape[0]
        T = min(T_obs, T_next, T_trans)
        if T <= 0:
            continue

        # ====== 2+3) 严格按 appended_order 构建 cond_cols（保持顺序） ======
        cond_cols = []
        for k in cond_plan.appended_order:
            if k == C_KEY_SENTINEL:
                # 在“这个位置”计算并追加 v = Jp(q) @ a_arm（保证顺序）
                if mj is None:
                    raise RuntimeError("MuJoCo not available for eef_action_vel.")
                if ('robot0_joint_pos_cos' in ep) and ('robot0_joint_pos_sin' in ep):
                    q = _atan2_recover_q(ep['robot0_joint_pos_cos'], ep['robot0_joint_pos_sin'])
                elif 'robot0_joint_pos' in ep:
                    q = ep['robot0_joint_pos']
                else:
                    raise KeyError(
                        f"Need joint positions (cos/sin or joint_pos) for {C_KEY_SENTINEL}, but not found in episode keys."
                    )
                a_arm = action[:T, :arm_dim]
                q_for = q[:T]

                V = np.zeros((T, 3), dtype=np.float32)
                for t in range(T):
                    _set_qpos_and_forward(base_env, q_for[t])
                    Jp = _eef_lin_jacobian(base_env, site_id)   # (3, nv)
                    v = (Jp[:, :q_for.shape[1]] @ a_arm[t]).astype(np.float64)
                    V[t] = v.astype(np.float32)
                cond_cols.append(V)  # eef_action_vel 被放在 appended_order 指定的位置
            else:
                # 普通键：若它本就在 base（robot_obs_keys）中，则跳过尾部，避免 base+tail 重复
                if k in robot_obs_keys:
                    # 如需强制“既在 base 又在尾部”请去掉本 continue
                    # print(f"[cond] key '{k}' already in base; skip tail.")
                    continue
                if k in ep:
                    cond_cols.append(ep[k][:T])
                else:
                    # 缺失的普通键直接忽略（此键在 plan 时已提示）
                    pass

        # （可选）调试：看一下每列维度与顺序
        # if cond_cols:
        #     print(f"[debug] cond order={cond_plan.appended_order} -> shapes={[c.shape[-1] for c in cond_cols]} (sum={sum(c.shape[-1] for c in cond_cols)})")

        # ====== 4) 用“纯 base（robot_obs_keys）”重建前半段，再 + cond 尾部 ======
        # 先组纯 base（严格按 robot_obs_keys），避免 loader 把别的键塞进 obs 前半段/尾部
        base_cols = []
        for key in robot_obs_keys:
            if key not in ep:
                raise KeyError(f"[base] missing key '{key}' in episode")
            base_cols.append(ep[key][:T])
        base_obs = np.concatenate(base_cols, axis=-1)

        # base_next 用 1-step shift（保证与 base_obs 同维）
        if T > 1:
            base_next = np.vstack([base_obs[1:T], base_obs[T-1:T]])
        else:
            base_next = base_obs.copy()

        if len(cond_cols) > 0:
            c_curr = np.concatenate(cond_cols, axis=-1)              # [T, C_dim]
            c_next = (np.vstack([c_curr[1:T], c_curr[T-1:T]]) if T > 1 else c_curr.copy())
            new_obs = np.concatenate([base_obs, c_curr], axis=-1)    # 纯 base + 条件尾部
            new_next_obs = np.concatenate([base_next, c_next], axis=-1)
        else:
            new_obs = base_obs
            new_next_obs = base_next

        new_ep = dict(ep)
        new_ep['obs'] = new_obs
        new_ep['next_obs'] = new_next_obs
        new_ep['action'] = action[:T]
        out_eps.append(new_ep)

    return out_eps

# ----------------------------
# 训练
# ----------------------------
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

def ddpm_loss_weight(alpha_cumprod_t: torch.Tensor, mode: str) -> torch.Tensor:
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
    make_batch_fn,  # -> (x, c)
    writer: SummaryWriter,
    device: torch.device,
    out_dir: pathlib.Path,
    cfg: TrainCfg,
    log_freq: int,
    save_interval: int,
):
    print(f"[stage] start training {name}")
    prior = prior.to(device).train()
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
        T = None  # EDM 或其他策略留作扩展

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
            a_bar_sqrt = torch.sqrt(a_bar_t)
            one_minus_sqrt = torch.sqrt(1.0 - a_bar_t)
            noise = torch.randn_like(x)
            x_t = a_bar_sqrt.unsqueeze(-1) * x + one_minus_sqrt.unsqueeze(-1) * noise
            t_float = (t_idx.to(torch.float32) + 0.5) / float(T)
            eps_hat = prior(x_t, c, t_float)
            if cfg.loss_weight in ('sigma2', 'snr'):
                w = ddpm_loss_weight(a_bar_t, cfg.loss_weight).unsqueeze(-1)
                loss = (w * (eps_hat - noise) ** 2).mean()
            else:
                loss = F.mse_loss(eps_hat, noise)
        else:
            # 预留：EDM 风格
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

# ----------------------------
# 入口
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='YAML config path')
    return p.parse_args()

def main():
    args = parse_args()
    yaml = YAML(typ='safe')
    params = yaml.load(open(args.config))

    # 设备与种子
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    seed = int(params.get('seed', 0))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # 基本路径
    expert_folder = params.get('expert_folder', 'human_demonstrations')
    # task_name = params.get('task_name', params.get('env_name', 'Task'))
    PRIMARY_TASK_FOR_LOG = params.get('env_name', 'Lift')  # 仅用于命名/日志
    TASKS = ["Lift", "Reach"]  # 强制固定为这两个任务

    src_robot = params['src_env']['robot']
    tgt_robot = params['tgt_env']['robot']
    src_ctrl = params['src_env']['controller_type']
    tgt_ctrl = params['tgt_env']['controller_type']
    env_kwargs = params.get('env_kwargs', {})

    # 日志目录（与你的 align 风格一致）
    logdir_prefix = pathlib.Path(params.get('logdir_prefix') or pathlib.Path(__file__).parent)
    data_path = logdir_prefix / 'logs' / time.strftime("%m.%d.%Y")
    logdir = '_'.join([
        time.strftime("%H-%M-%S"),
        params['env_name'],
        src_robot, src_ctrl,
        tgt_robot, tgt_ctrl,
        params.get('suffix', 'diffusion_pretrain_b')
    ])
    logdir = data_path / logdir
    logdir.mkdir(parents=True, exist_ok=True)

    # 保存 config 快照
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

    # 构造两个用于 buffer 的 env（GymWrapper 版）
    src_env, _ = build_env_for_buffer(
        params['env_name'], src_robot, src_ctrl,
        params['src_env'].get('env_kwargs', {}), params['src_env']['robot_obs_keys']
    )
    tgt_env, _ = build_env_for_buffer(
        params['env_name'], tgt_robot, tgt_ctrl,
        params['tgt_env'].get('env_kwargs', {}), params['tgt_env']['robot_obs_keys']
    )

    # 探测 robot_obs_dim（用于切片）
    src_robot_obs_dim = probe_robot_obs_dim(
        params['env_name'], src_robot, src_ctrl,
        env_kwargs, params['src_env']['robot_obs_keys']
    )
    tgt_robot_obs_dim = probe_robot_obs_dim(
        params['env_name'], tgt_robot, tgt_ctrl,
        env_kwargs, params['tgt_env']['robot_obs_keys']
    )

    # 原始 episodes（用于 cond 维度计划）
    # src_in_dir = pathlib.Path(expert_folder) / task_name / src_robot / src_ctrl
    # tgt_in_dir = pathlib.Path(expert_folder) / task_name / tgt_robot / tgt_ctrl
    # episodes_src_raw = utils.load_episodes(src_in_dir, obs_keys=list(params['src_env']['robot_obs_keys']),
    #                                        lat_obs_keys=None, capacity=None)
    # episodes_tgt_raw = utils.load_episodes(tgt_in_dir, obs_keys=list(params['tgt_env']['robot_obs_keys']),
    #                                        lat_obs_keys=None, capacity=None)
    
    episodes_src_raw, episodes_tgt_raw = [], []

    for tn in TASKS:
        src_in_dir = pathlib.Path(expert_folder) / tn / src_robot / src_ctrl
        tgt_in_dir = pathlib.Path(expert_folder) / tn / tgt_robot / tgt_ctrl

        if src_in_dir.exists():
            eps_src = utils.load_episodes(
                src_in_dir,
                obs_keys=list(params['src_env']['robot_obs_keys']),
                lat_obs_keys=None, capacity=None
            )
            episodes_src_raw.extend(eps_src)
        else:
            print(f"[warn] SRC dataset not found for task '{tn}': {src_in_dir}")

        if tgt_in_dir.exists():
            eps_tgt = utils.load_episodes(
                tgt_in_dir,
                obs_keys=list(params['tgt_env']['robot_obs_keys']),
                lat_obs_keys=None, capacity=None
            )
            episodes_tgt_raw.extend(eps_tgt)
        else:
            print(f"[warn] TGT dataset not found for task '{tn}': {tgt_in_dir}")

    if len(episodes_src_raw) == 0:
        raise RuntimeError(f"No SRC episodes found under tasks {TASKS}.")
    if len(episodes_tgt_raw) == 0:
        raise RuntimeError(f"No TGT episodes found under tasks {TASKS}.")

    # 条件计划（keys 解析 + eef_action_vel 触发规则 + 维度汇总）
    cond_plan = plan_condition(params, src_env, tgt_env, episodes_src_raw, episodes_tgt_raw)

    # 内存增广：按 cond_plan 追加条件列（若最终无条件列则不改变 obs 形状）
    # print("[prep] building augmented episodes (SRC) ...")
    # episodes_src_aug = augment_episodes_with_condition(
    #     task_name=task_name, robots=src_robot, controller=src_ctrl,
    #     root_dir=expert_folder, robot_obs_keys=params['src_env']['robot_obs_keys'],
    #     env_kwargs=env_kwargs, cond_plan=cond_plan, arm_dim=cond_plan.arm_dim_src
    # )
    # print("[prep] building augmented episodes (TGT) ...")
    # episodes_tgt_aug = augment_episodes_with_condition(
    #     task_name=task_name, robots=tgt_robot, controller=tgt_ctrl,
    #     root_dir=expert_folder, robot_obs_keys=params['tgt_env']['robot_obs_keys'],
    #     env_kwargs=env_kwargs, cond_plan=cond_plan, arm_dim=cond_plan.arm_dim_tgt
    # )
    print("[prep] building augmented episodes (SRC+TGT across Lift/Reach) ...")
    episodes_src_aug, episodes_tgt_aug = [], []

    for tn in TASKS:
        # SRC
        src_dir = pathlib.Path(expert_folder) / tn / src_robot / src_ctrl
        if src_dir.exists():
            eps_src_aug = augment_episodes_with_condition(
                task_name=tn, robots=src_robot, controller=src_ctrl,
                root_dir=expert_folder, robot_obs_keys=params['src_env']['robot_obs_keys'],
                env_kwargs=env_kwargs, cond_plan=cond_plan, arm_dim=cond_plan.arm_dim_src
            )
            episodes_src_aug.extend(eps_src_aug)
        # TGT
        tgt_dir = pathlib.Path(expert_folder) / tn / tgt_robot / tgt_ctrl
        if tgt_dir.exists():
            eps_tgt_aug = augment_episodes_with_condition(
                task_name=tn, robots=tgt_robot, controller=tgt_ctrl,
                root_dir=expert_folder, robot_obs_keys=params['tgt_env']['robot_obs_keys'],
                env_kwargs=env_kwargs, cond_plan=cond_plan, arm_dim=cond_plan.arm_dim_tgt
            )
            episodes_tgt_aug.extend(eps_tgt_aug)

    if len(episodes_src_aug) == 0:
        raise RuntimeError(f"No SRC augmented episodes after processing tasks {TASKS}.")
    if len(episodes_tgt_aug) == 0:
        raise RuntimeError(f"No TGT augmented episodes after processing tasks {TASKS}.")

    # 构造回放缓冲
    diff_cfg = params.get('diffusion', {})
    lat_cfg = diff_cfg.get('latent', {})
    batch_size = int(lat_cfg.get('batch_size', params.get('batch_size', 4096)))
    def _read_prior_cfg(block: str):
        b = diff_cfg.get(block, {}) or {}
        sch = b.get('schedule', {}) or {}
        return {
            'n_layers':   int(b.get('n_layers',   params.get('n_layers',   3))),
            'hidden_dim': int(b.get('hidden_dim', params.get('hidden_dim', 256))),
            'ema_decay':  float(b.get('ema_decay', 0.999)),
            'steps':      int(b.get('steps',      params.get('tgt_align_timesteps', 300000))),
            'batch_size': int(b.get('batch_size', params.get('batch_size', 4096))),
            'lr':         float(b.get('lr',       params.get('lr', 3e-4))),
            'schedule': {
                'type':       sch.get('type', 'ddpm'),
                'T':          int(sch.get('T', 128)),
                'beta':       str(sch.get('beta', 'cosine')),
                'loss_weight':str(sch.get('loss_weight', 'sigma2')),
            }
        }


    cfg_lat = _read_prior_cfg('latent_act')
    cfg_src = _read_prior_cfg('src_act')
    cfg_tgt = _read_prior_cfg('tgt_act')

    print("[cfg] latent_act:", cfg_lat)
    print("[cfg] src_act   :", cfg_src)
    print("[cfg] tgt_act   :", cfg_tgt)
        
    
    # 统一 buffer 的 batch，取三路最大值
    buf_bs = max(cfg_lat['batch_size'], cfg_src['batch_size'], cfg_tgt['batch_size'])

    src_buffer = build_replay_buffer_from_episodes(episodes_src_aug, device, buf_bs)
    tgt_buffer = build_replay_buffer_from_episodes(episodes_tgt_aug, device, buf_bs)

    # 三路各自的 TrainCfg
    train_cfg_lat = TrainCfg(
        steps=cfg_lat['steps'], batch_size=cfg_lat['batch_size'], lr=cfg_lat['lr'],
        n_layers=cfg_lat['n_layers'], hidden_dim=cfg_lat['hidden_dim'], ema_decay=cfg_lat['ema_decay'],
        schedule_type=cfg_lat['schedule']['type'], T=cfg_lat['schedule']['T'],
        beta=cfg_lat['schedule']['beta'], loss_weight=cfg_lat['schedule']['loss_weight'],
    )
    train_cfg_src = TrainCfg(
        steps=cfg_src['steps'], batch_size=cfg_src['batch_size'], lr=cfg_src['lr'],
        n_layers=cfg_src['n_layers'], hidden_dim=cfg_src['hidden_dim'], ema_decay=cfg_src['ema_decay'],
        schedule_type=cfg_src['schedule']['type'], T=cfg_src['schedule']['T'],
        beta=cfg_src['schedule']['beta'], loss_weight=cfg_src['schedule']['loss_weight'],
    )
    train_cfg_tgt = TrainCfg(
        steps=cfg_tgt['steps'], batch_size=cfg_tgt['batch_size'], lr=cfg_tgt['lr'],
        n_layers=cfg_tgt['n_layers'], hidden_dim=cfg_tgt['hidden_dim'], ema_decay=cfg_tgt['ema_decay'],
        schedule_type=cfg_tgt['schedule']['type'], T=cfg_tgt['schedule']['T'],
        beta=cfg_tgt['schedule']['beta'], loss_weight=cfg_tgt['schedule']['loss_weight'],
    )
    # ---------------------------
    # 加载 E_src / A_src 编码器（冻结）
    # ---------------------------
    lat_obs_dim = int(params['lat_obs_dim'])
    lat_act_dim = int(params['lat_act_dim'])

    class ObsEncoder(nn.Module):
        def __init__(self, in_dim, out_dim, n_layers=3, hidden=256, batch_norm=False):
            super().__init__()
            self.net = utils.build_mlp(
                in_dim, out_dim, n_layers, hidden,
                activation='relu', output_activation='identity', batch_norm=batch_norm
            )
        def forward(self, x): return self.net(x)

    class ActEncoder(nn.Module):
        def __init__(self, in_dim, out_dim, n_layers=3, hidden=256, batch_norm=False):
            super().__init__()
            self.net = utils.build_mlp(
                in_dim, out_dim, n_layers, hidden,
                activation='relu', output_activation='tanh', batch_norm=batch_norm
            )
        def forward(self, x): return self.net(x)

    # E_src
    E_src_in_dim = src_robot_obs_dim
    E_src = ObsEncoder(
        in_dim=E_src_in_dim, out_dim=lat_obs_dim,
        n_layers=int(params.get('n_layers', 3)),
        hidden=int(params.get('hidden_dim', 256)), batch_norm=False
    ).to(device)
    ckpt_obs = torch.load(pathlib.Path(params['src_model_dir']) / "obs_enc.pt", map_location=device)
    # 兼容 "module." 前缀 或 直接权重
    try:
        E_src.load_state_dict(ckpt_obs, strict=True)
    except Exception:
        new_sd = {}
        for k, v in ckpt_obs.items():
            kk = k if k.startswith("net.") else f"net.{k}"
            new_sd[kk] = v
        E_src.load_state_dict(new_sd, strict=False)
    for p in E_src.parameters(): p.requires_grad = False
    E_src.eval()

    # A_src
    # A_src 的输入 = concat([robot_obs_src, a_arm_src])
    src_action_dim = infer_action_dim(src_env)
    src_arm_dim = cond_plan.arm_dim_src
    A_src_in_dim = src_robot_obs_dim + src_arm_dim

    A_src = ActEncoder(
        in_dim=A_src_in_dim, out_dim=lat_act_dim,
        n_layers=int(params.get('n_layers', 3)),
        hidden=int(params.get('hidden_dim', 256)), batch_norm=False
    ).to(device)
    ckpt_act = torch.load(pathlib.Path(params['src_model_dir']) / "act_enc.pt", map_location=device)
    try:
        A_src.load_state_dict(ckpt_act, strict=True)
    except Exception:
        new_sd = {}
        for k, v in ckpt_act.items():
            kk = k if k.startswith("net.") else f"net.{k}"
            new_sd[kk] = v
        A_src.load_state_dict(new_sd, strict=False)
    for p in A_src.parameters(): p.requires_grad = False
    A_src.eval()

    # ---------------------------
    # 统计（每域 x 的 mu/std；条件 c 的 mu/std）
    # ---------------------------
    stats_passes = int(diff_cfg.get('stats_passes', 200))
    latent_passes = int(diff_cfg.get('latent_passes', 200))

    # —— 在增广后，根据“obs 尾部”实际增加的列数动态判定 cond_dim —— 
    cond_dim_src = int(episodes_src_aug[0]['obs'].shape[1] - src_robot_obs_dim)
    # print(episodes_src_aug[0]['obs'].shape[1])
    # print(src_robot_obs_dim)
    # print(cond_dim_src)
    cond_dim_tgt = int(episodes_tgt_aug[0]['obs'].shape[1] - tgt_robot_obs_dim)
    if cond_dim_src != cond_dim_tgt:
        print(f"[warn] cond_dim mismatch: src={cond_dim_src}, tgt={cond_dim_tgt} — using src")
    cond_dim = cond_dim_src
    use_condition = (cond_dim > 0)

    def iter_stats_src_joint(buf: replay_buffer.ReplayBuffer) -> np.ndarray:
        Xs = []
        for _ in trange(stats_passes, desc="stats: src_joint", dynamic_ncols=True):
            with torch.no_grad():
                obs, act, _, _, _ = buf.sample()
                rob = obs[:, :src_robot_obs_dim]
                a_arm = act[:, :src_arm_dim]
                x = torch.cat([rob, a_arm], dim=-1)
                Xs.append(x.cpu().numpy())
        return np.concatenate(Xs, axis=0)

    def iter_stats_tgt_joint(buf: replay_buffer.ReplayBuffer) -> np.ndarray:
        Xs = []
        tgt_arm_dim = cond_plan.arm_dim_tgt
        for _ in trange(stats_passes, desc="stats: tgt_joint", dynamic_ncols=True):
            with torch.no_grad():
                obs, act, _, _, _ = buf.sample()
                rob = obs[:, :tgt_robot_obs_dim]
                a_arm = act[:, :tgt_arm_dim]
                x = torch.cat([rob, a_arm], dim=-1)
                Xs.append(x.cpu().numpy())
        return np.concatenate(Xs, axis=0)

    def iter_stats_latent(buf: replay_buffer.ReplayBuffer) -> np.ndarray:
        Xs = []
        for _ in trange(latent_passes, desc="stats: latent_act", dynamic_ncols=True):
            with torch.no_grad():
                obs, act, _, _, _ = buf.sample()
                rob = obs[:, :src_robot_obs_dim]
                a_arm = act[:, :src_arm_dim]
                zobs = E_src(rob)
                zact = A_src(torch.cat([rob, a_arm], dim=-1))
                x = torch.cat([zobs, zact], dim=-1)
                Xs.append(x.cpu().numpy())
        return np.concatenate(Xs, axis=0)

    def iter_stats_cond(buf: replay_buffer.ReplayBuffer, robot_dim: int) -> Optional[np.ndarray]:
        if cond_dim <= 0:
            return None
        Cs = []
        for _ in trange(stats_passes, desc="stats: cond", dynamic_ncols=True):
            with torch.no_grad():
                obs, _, _, _, _ = buf.sample()
                c = obs[:, -cond_dim:]
                Cs.append(c.cpu().numpy())
        return np.concatenate(Cs, axis=0)

    # x 统计
    X_src = iter_stats_src_joint(src_buffer)
    mu_x_src, std_x_src = X_src.mean(axis=0), X_src.std(axis=0) + 1e-8

    X_tgt = iter_stats_tgt_joint(tgt_buffer)
    mu_x_tgt, std_x_tgt = X_tgt.mean(axis=0), X_tgt.std(axis=0) + 1e-8

    X_lat = iter_stats_latent(src_buffer)
    mu_x_lat, std_x_lat = X_lat.mean(axis=0), X_lat.std(axis=0) + 1e-8

    # cond 统计（源/目标各一份；latent 用源域）
    if cond_dim > 0:
        C_src = iter_stats_cond(src_buffer, src_robot_obs_dim)
        mu_c_src = C_src.mean(axis=0); std_c_src = C_src.std(axis=0) + 1e-8

        C_tgt = iter_stats_cond(tgt_buffer, tgt_robot_obs_dim)
        mu_c_tgt = C_tgt.mean(axis=0); std_c_tgt = C_tgt.std(axis=0) + 1e-8
    else:
        mu_c_src = std_c_src = mu_c_tgt = std_c_tgt = None

    # 保存 normalization / dataset / schedule
    normalization = {
        'src_joint': {'x_dim': int(X_src.shape[1]), 'mu': mu_x_src.tolist(), 'std': std_x_src.tolist()},
        'tgt_joint': {'x_dim': int(X_tgt.shape[1]), 'mu': mu_x_tgt.tolist(), 'std': std_x_tgt.tolist()},
        'latent_act': {'x_dim': int(X_lat.shape[1]), 'mu': mu_x_lat.tolist(), 'std': std_x_lat.tolist()},
        'cond': {
            'dim': int(cond_dim),
            'appended_order': cond_plan.appended_order,
            'mu_src': (mu_c_src.tolist() if mu_c_src is not None else None),
            'std_src': (std_c_src.tolist() if std_c_src is not None else None),
            'mu_tgt': (mu_c_tgt.tolist() if mu_c_tgt is not None else None),
            'std_tgt': (std_c_tgt.tolist() if std_c_tgt is not None else None),
        },
        'arm_dim': {'src': int(cond_plan.arm_dim_src), 'tgt': int(cond_plan.arm_dim_tgt)}
    }
    with open(stats_dir / 'normalization.yml', 'w') as fp:
        YAML().dump(normalization, fp)
    print(f"[save] normalization -> {stats_dir / 'normalization.yml'}")

    dataset_meta = {
        'task_name': 'Lift',
        'expert_folder': expert_folder,
        'src_robot': src_robot, 'tgt_robot': tgt_robot,
        'src_ctrl': src_ctrl, 'tgt_ctrl': tgt_ctrl,
        'src_robot_obs_dim': int(src_robot_obs_dim),
        'tgt_robot_obs_dim': int(tgt_robot_obs_dim),
        'cond_keys_requested': params.get('diffusion', {}).get('cond', {}).get('keys', []),
        'cond_appended_order': cond_plan.appended_order,
        'cond_dim_final': int(cond_dim),
    }
    with open(stats_dir / 'dataset.yml', 'w') as fp:
        YAML().dump(dataset_meta, fp)
    print(f"[save] dataset -> {stats_dir / 'dataset.yml'}")

    # ---- resolve schedules for three branches & save ----
    diff_cfg = params.get('diffusion', {})
    sched_defaults = diff_cfg.get('schedule_defaults', {}) or {}   # e.g. {type,T,beta,loss_weight}

    def _resolve_branch_schedule(branch_name: str):
        """
        合并逻辑：
        1) 先用 schedule_defaults 作为基线
        2) 再用 diffusion.<branch>.schedule 覆盖
        """
        bcfg = diff_cfg.get(branch_name, {}) or {}
        bsch = (bcfg.get('schedule', {}) or {}).copy()

        # 合并：defaults -> branch.schedule
        merged = dict(sched_defaults)
        merged.update(bsch)

        stype = str(merged.get('type', 'ddpm'))
        TT = int(merged.get('T', 128))
        beta_ = str(merged.get('beta', 'cosine'))
        lw = str(merged.get('loss_weight', 'sigma2'))
        return {"type": stype, "T": TT, "beta": beta_, "loss_weight": lw}

    sched_lat = _resolve_branch_schedule('latent_act')
    sched_src = _resolve_branch_schedule('src_act')
    sched_tgt = _resolve_branch_schedule('tgt_act')

    # 你后面训练构造 TrainCfg 时，也分别使用对应分支的 schedule：
    #   schedule_type = sched_lat["type"] / sched_src["type"] / sched_tgt["type"]
    #   T             = sched_lat["T"]   / ...
    #   beta          = sched_lat["beta"]/ ...
    #   loss_weight   = sched_lat["loss_weight"] / ...

    # 统一写入 stats/schedule.yml，方便核对
    schedule_meta = {
        "defaults": sched_defaults,
        "latent_act": sched_lat,
        "src_act": sched_src,
        "tgt_act": sched_tgt,
    }
    with open(stats_dir / 'schedule.yml', 'w') as fp:
        YAML().dump(schedule_meta, fp)
    print(f"[save] schedule -> {stats_dir / 'schedule.yml'}")

    # ---------------------------
    # 构建三路 Prior
    # ---------------------------
    # ---------------------------
    # 构建三路 Prior（分别使用各自 cfg_*）
    # ---------------------------
    use_condition = (cond_dim > 0)
    time_embed_dim = 128
    log_freq = int(params.get('diffusion', {}).get('log_freq', params.get('log_freq', 1000)))
    save_interval = int(params.get('diffusion', {}).get('save_interval', params.get('evaluation', {}).get('save_interval', 20000)))

    xdim_src = mu_x_src.shape[0]
    xdim_tgt = mu_x_tgt.shape[0]
    xdim_lat = mu_x_lat.shape[0]

    prior_src = PriorModel(
        x_dim=xdim_src, c_dim=cond_dim,
        hidden=cfg_src['hidden_dim'], n_layers=cfg_src['n_layers'],
        use_condition=use_condition, time_embed_dim=time_embed_dim,
        mode=cfg_src['schedule']['type']
    ).to(device)
    prior_src.set_norm(mu_x_src, std_x_src, mu_c_src, std_c_src)

    prior_tgt = PriorModel(
        x_dim=xdim_tgt, c_dim=cond_dim,
        hidden=cfg_tgt['hidden_dim'], n_layers=cfg_tgt['n_layers'],
        use_condition=use_condition, time_embed_dim=time_embed_dim,
        mode=cfg_tgt['schedule']['type']
    ).to(device)
    prior_tgt.set_norm(mu_x_tgt, std_x_tgt, mu_c_tgt, std_c_tgt)

    prior_lat = PriorModel(
        x_dim=xdim_lat, c_dim=cond_dim,
        hidden=cfg_lat['hidden_dim'], n_layers=cfg_lat['n_layers'],
        use_condition=use_condition, time_embed_dim=time_embed_dim,
        mode=cfg_lat['schedule']['type']
    ).to(device)
    prior_lat.set_norm(mu_x_lat, std_x_lat, mu_c_src, std_c_src)

    # ---------------------------
    # 三路 batch 构造
    # ---------------------------
    def _slice_to(bs, *tensors):
        if tensors[0].shape[0] == bs:
            return tensors
        idx = slice(0, bs)
        return tuple(t[idx] for t in tensors)

    def make_batch_src_joint():
        obs, act, _, _, _ = src_buffer.sample()           # shape: [buf_bs, ...]
        rob = obs[:, :src_robot_obs_dim]
        a_arm = act[:, :cond_plan.arm_dim_src]
        x = torch.cat([rob, a_arm], dim=-1)
        c = None if cond_dim == 0 else obs[:, -cond_dim:]
        x, c = _slice_to(train_cfg_src.batch_size, x, c if c is not None else x)
        if cond_dim == 0: c = None
        return x, c

    def make_batch_tgt_joint():
        obs, act, _, _, _ = tgt_buffer.sample()
        rob = obs[:, :tgt_robot_obs_dim]
        a_arm = act[:, :cond_plan.arm_dim_tgt]
        x = torch.cat([rob, a_arm], dim=-1)
        c = None if cond_dim == 0 else obs[:, -cond_dim:]
        x, c = _slice_to(train_cfg_tgt.batch_size, x, c if c is not None else x)
        if cond_dim == 0: c = None
        return x, c

    def make_batch_lat_act():
        obs, act, _, _, _ = src_buffer.sample()
        rob = obs[:, :src_robot_obs_dim]
        a_arm = act[:, :cond_plan.arm_dim_src]
        with torch.no_grad():
            zobs = E_src(rob)
            zact = A_src(torch.cat([rob, a_arm], dim=-1))
        x = torch.cat([zobs, zact], dim=-1)
        c = None if cond_dim == 0 else obs[:, -cond_dim:]
        x, c = _slice_to(train_cfg_lat.batch_size, x, c if c is not None else x)
        if cond_dim == 0: c = None
        return x, c

    # ---------------------------
    # 训练：三个域（顺序可随意）
    # ---------------------------
    train_one_prior('score_src_joint', prior_src, make_batch_src_joint,
                    writer, device, model_dir, train_cfg_src, log_freq, save_interval)
    train_one_prior('score_tgt_joint', prior_tgt, make_batch_tgt_joint,
                    writer, device, model_dir, train_cfg_tgt, log_freq, save_interval)
    train_one_prior('score_lat_act',  prior_lat, make_batch_lat_act,
                    writer, device, model_dir, train_cfg_lat, log_freq, save_interval)

    writer.close()
    print(f"[DONE] Saved models & stats under: {str(logdir)}")


if __name__ == '__main__':
    main()