# =========================================================
# Flow Matching Priors Pretraining (Lat-Act / Src-Act / Tgt-Act)
# Condition = [EEF position p(3), EEF linear velocity v(3)]  => 6D
# - qdot 直接用“关节动作（臂部维）”代替（不做维度检测）
# - p,v 在线计算，不落盘；把 [p,v] 追加到 obs / next_obs 尾部
# - 统计、归一化、训练循环与日志/保存目录风格保持不变
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

from tqdm import trange, tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ruamel.yaml import YAML

import utils  # 必须存在（与你 align 代码同仓）
import replay_buffer  # 必须存在（与你 align 代码同仓）

import mujoco as mj

# ---- constants ----
C_KEY = "cond_eef_pv"  # 仅用于 stats/normalization.yml 的 key 标记
C_DIM = 6              # EEF position p(3) + EEF linear velocity v(3)


# ---------------------------------------------------------
# Time Embedding (for t in [0, 1])
# ---------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        # t ∈ [0, 1], map to sinusoidal embeddings
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
# Small MLP (kept name "Denoiser" for minimal intrusion)
# 在 Flow Matching 中预测的是 velocity field v_hat，而非噪声。
# ---------------------------------------------------------
class Denoiser(nn.Module):
    """
    Input: x (B, Dx), optional cond c (B, Dc), time embedding (B, Dt)
    Output: predicted velocity v_hat (same shape as x)
    内部假设：输入已经在 wrapper 里做过标准化（mu/std）
    """
    def __init__(self, x_dim: int, c_dim: int, t_dim: int, hidden: int = 256, n_layers: int = 3):
        super().__init__()
        in_dim = x_dim + (c_dim if c_dim > 0 else 0) + t_dim
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
    包装 MLP：内含 x/c 的标准化、时间嵌入与（可选）条件。
    统一 forward(x, c, t) -> v_hat
    """
    def __init__(self, x_dim: int, c_dim: int, hidden: int, n_layers: int,
                 use_condition: bool, time_embed_dim: int):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim if use_condition else 0
        self.use_condition = use_condition
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.denoiser = Denoiser(x_dim=x_dim, c_dim=self.c_dim, t_dim=time_embed_dim,
                                 hidden=hidden, n_layers=n_layers)
        # Buffers for normalization
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
        # t_value is float scalar per-batch item in [0,1]
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
    if hasattr(env, "robots") and len(getattr(env.robots, [])) > 0:
        ctrl = getattr(env.robots[0], "controller", None)
        if ctrl is not None and hasattr(ctrl, "control_dim"):
            return int(ctrl.control_dim)
    raise AttributeError("Cannot infer action dimension for this env.")


def probe_dims(env_name: str, robot: str, controller: str, env_kwargs: Dict[str, Any],
               robot_obs_keys: List[str]) -> DomainDims:
    """
    仅用来探测 robot_obs_dim / act_dim；cond_dim 固定为 C_DIM（=6）
    """
    probe_env = utils.make_robosuite_env(
        env_name,
        robots=robot,
        controller_type=controller,
        **env_kwargs,
    )
    probe = probe_env.reset()
    rob = np.concatenate([probe[k] for k in robot_obs_keys])
    act_dim = infer_action_dim(probe_env)
    return DomainDims(robot_obs_dim=int(rob.shape[0]), cond_dim=C_DIM, act_dim=act_dim)


def build_env_for_buffer(env_name: str, robot: str, controller: str, env_kwargs: Dict[str, Any],
                         robot_obs_keys: List[str]):
    """
    在线增广 [p,v]，因此构建 buffer 环境时只暴露 robot_obs_keys，不包含 cond。
    """
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


def build_replay_buffer_from_episodes(episodes_aug, device: torch.device, batch_size: int) -> replay_buffer.ReplayBuffer:
    # 以第一个 episode 推断形状（obs 已包含 [p,v] 在尾部）
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
# [p,v] 在线计算相关：Jp(q) 与动作当作 qdot
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
    # 常见候选名
    for name in ("gripper0_eef", "gripper0_grip_site", "robot0_eef"):
        sid = mj.mj_name2id(model, site_type, name)
        if sid != -1:
            return sid
    # 兜底扫描
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


def _compute_pv_from_actions(env, q_traj: np.ndarray, a_arm_traj: np.ndarray, site_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    q_traj:     [T, nq]
    a_arm_traj: [T, nq]  # 直接把“关节动作（臂部维）”当作 qdot
    return:     (P, V)  # P: [T, 3] EEF 位置, V: [T, 3] EEF 线速度
    """
    T, nq = q_traj.shape
    P = np.zeros((T, 3), dtype=np.float32)
    V = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        _set_qpos_and_forward(env, q_traj[t])
        p = _eef_site_pos(env, site_id)                   # (3,)
        Jp = _eef_lin_jacobian(env, site_id)              # (3, nv)
        v = (Jp[:, :nq] @ a_arm_traj[t]).astype(np.float64)
        P[t] = p.astype(np.float32)
        V[t] = v.astype(np.float32)
    return P, V


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
      v = Jp(q)@qdot
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
    for ep in tqdm(episodes, desc=f"[{robots}] augment with [p, v] (in-memory)"):
        # 恢复 q
        cos = ep['robot0_joint_pos_cos']    # [L_raw, nq]
        sin = ep['robot0_joint_pos_sin']    # [L_raw, nq]
        q = _atan2_recover_q(cos, sin)      # [L_raw, nq]

        action = ep['action']               # [L_raw-1, D_act] （通常转移长度）
        assert act_arm_dim > 0, "act_arm_dim 应为正（一般为 action_dim - 1）"
        a_arm = action[:, :act_arm_dim]     # [L_raw-1, nq]（按约定，不做检测）

        # 用 q[:-1] 与 action 对齐，让 c[t]=[p,v] 对应 obs[t]、action[t]
        T_trans = action.shape[0]
        q_for = q[:T_trans]                 # [T_trans, nq]

        # 在线求 p,v，长度 == T_trans
        P_full, V_full = _compute_pv_from_actions(base_env, q_for, a_arm, site_id)  # [T_trans,3], [T_trans,3]

        # 构造 obs_aug / next_obs_aug，三者统一用 T = min(...)
        T = min(T_trans, ep['obs'].shape[0], ep['next_obs'].shape[0])

        # 追加 [p,v]（obs 用 [p,v][t]；next_obs 用 [p,v][t+1]，末尾复制兜底）
        p_curr = P_full[:T]
        v_curr = V_full[:T]
        if T > 1:
            p_next = np.vstack([P_full[1:T], P_full[T-1:T]])
            v_next = np.vstack([V_full[1:T], V_full[T-1:T]])
        else:
            p_next = p_curr.copy()
            v_next = v_curr.copy()

        c_curr = np.concatenate([p_curr, v_curr], axis=-1)  # [T, 6]
        c_next = np.concatenate([p_next, v_next], axis=-1)  # [T, 6]

        obs_aug = np.concatenate([ep['obs'][:T], c_curr], axis=-1)
        next_obs_aug = np.concatenate([ep['next_obs'][:T], c_next], axis=-1)

        new_ep = dict(ep)
        new_ep['obs'] = obs_aug
        new_ep['next_obs'] = next_obs_aug
        new_ep[C_KEY] = c_curr  # 可选：便于调试
        new_ep['action'] = action[:T]
        out_eps.append(new_ep)

    return out_eps


# ---------------------------------------------------------
# Training Loop per-prior (Flow Matching)
# ---------------------------------------------------------
@dataclass
class TrainCfg:
    steps: int
    batch_size: int
    lr: float
    n_layers: int
    hidden_dim: int
    ema_decay: float
    # 训练 path 配置（FM 线性路径）
    t_min: float = 0.0
    t_max: float = 1.0


def train_one_prior(
    name: str,
    prior: PriorModel,
    make_batch_fn,  # returns (x0, c)
    writer: SummaryWriter,
    device: torch.device,
    out_dir: pathlib.Path,
    cfg: TrainCfg,
    log_freq: int,
    save_interval: int,
    val_sampler=None,
):
    """
    标准 Flow Matching（线性路径）：
      x_t = (1 - t) * x0 + t * x1, t ~ U(0,1), x1 ~ N(0, I)
      目标速度：u_t = x1 - x0 （常量）
      预测 v_hat(x_t, c, t)，最小化 ||v_hat - u_t||^2
    """
    print(f"[stage] start training {name}")
    prior = prior.to(device)
    prior.train()
    opt = torch.optim.Adam(prior.parameters(), lr=cfg.lr)
    ema = EMA(prior, decay=cfg.ema_decay)
    prior_ema = PriorModel(prior.x_dim, prior.c_dim, cfg.hidden_dim, cfg.n_layers,
                           prior.use_condition, prior.time_embed.dim).to(device)
    ema.copy_to(prior_ema)

    global_step = 0
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_ckpt(tag: str):
        torch.save(prior.state_dict(), out_dir / f"{name}.pt")
        ema.copy_to(prior_ema)
        torch.save(prior_ema.state_dict(), out_dir / f"{name}_ema.pt")

    pbar = trange(cfg.steps, desc=f"{name}", dynamic_ncols=True)
    running_loss = None

    for _ in pbar:
        # ----- Batch -----
        x0, c = make_batch_fn()                        # clean sample
        B = x0.shape[0]

        t = torch.full((B,), 0.5, device=device, dtype=x0.dtype)
        # endpoint x1 ~ N(0, I)
        x1 = torch.randn_like(x0)
        # linear path
        x_t = (1.0 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * x1
        # target velocity (constant along t for linear path)
        u_t = x1 - x0

        # ----- Forward -----
        v_hat = prior(x_t, c, t)                       # predict velocity field

        # ----- Loss -----
        loss = F.mse_loss(v_hat, u_t)

        # ----- Opt/EMA/Logs -----
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

    # final save
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
    # ------------------------------
    # 读取配置 & 设备
    # ------------------------------
    args = parse_args()
    yaml = YAML(typ='safe')
    params = yaml.load(open(args.config))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    seed = int(params.get('seed', 0))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # ------------------------------
    # 数据位置 & 任务/机器人信息
    # ------------------------------
    expert_folder = params.get('expert_folder', 'human_demonstrations')
    task_name = params.get('task_name', params.get('env_name', 'Task'))

    src_robot = params['src_env']['robot']
    tgt_robot = params['tgt_env']['robot']
    controller_src = params['src_env']['controller_type']
    controller_tgt = params['tgt_env']['controller_type']
    env_kwargs = params.get('env_kwargs', {})

    # ------------------------------
    # Logging dirs（与 align 同风格）
    # ------------------------------
    logdir_prefix = pathlib.Path(params.get('logdir_prefix') or pathlib.Path(__file__).parent)
    data_path = logdir_prefix / 'logs' / time.strftime("%m.%d.%Y")
    logdir = '_'.join([
        time.strftime("%H-%M-%S"),
        params['env_name'],
        src_robot, controller_src,
        tgt_robot, controller_tgt,
        params.get('suffix', 'flowmatching_pretrain_act_pv')
    ])
    logdir = data_path / logdir
    logdir.mkdir(parents=True, exist_ok=True)

    # 保存 params 快照
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
    # 动态维度探测（cond_dim 固定为 C_DIM）
    # ------------------------------
    src_dims = probe_dims(params['env_name'], src_robot, controller_src, env_kwargs,
                          params['src_env']['robot_obs_keys'])
    tgt_dims = probe_dims(params['env_name'], tgt_robot, controller_tgt, env_kwargs,
                          params['tgt_env']['robot_obs_keys'])

    # ------------------------------
    # 为推断 action_dim、构造 A_src 输入维度，需要 GymWrapper 版本的 env
    # ------------------------------
    src_env, _ = build_env_for_buffer(
        params['env_name'],
        src_robot,
        controller_src,
        params['src_env'].get('env_kwargs', {}),
        params['src_env']['robot_obs_keys'],
    )

    tgt_env, _ = build_env_for_buffer(
        params['env_name'],
        tgt_robot,
        controller_tgt,
        params['tgt_env'].get('env_kwargs', {}),
        params['tgt_env']['robot_obs_keys'],
    )

    # --- 关节动作维度（臂部），按你的既有约定（总控维 - 1 去掉夹爪） ---
    src_act_arm_dim = infer_action_dim(src_env) - 1
    tgt_act_arm_dim = infer_action_dim(tgt_env) - 1
    assert src_act_arm_dim > 0 and tgt_act_arm_dim > 0, "act_arm_dim 必须为正（一般为 action_dim-1）"

    print("[prep] augmenting in-memory with [p (EEF), v = Jp(q) * action] (SRC/TGT) ...")
    episodes_src_aug = precompute_augmented_in_memory_using_actions(
        task_name=task_name,
        robots=src_robot,
        controller=controller_src,
        root_dir=expert_folder,
        robot_obs_keys=params['src_env']['robot_obs_keys'],
        env_kwargs=env_kwargs,
        act_arm_dim=src_act_arm_dim,
        capacity=None,
    )
    episodes_tgt_aug = precompute_augmented_in_memory_using_actions(
        task_name=task_name,
        robots=tgt_robot,
        controller=controller_tgt,
        root_dir=expert_folder,
        robot_obs_keys=params['tgt_env']['robot_obs_keys'],
        env_kwargs=env_kwargs,
        act_arm_dim=tgt_act_arm_dim,
        capacity=None,
    )

    # ------------------------------
    # 用增广后的 episodes 构建 buffers（obs 已包含 [p,v] 在尾部）
    # ------------------------------
    fm_cfg = params.get('flowmatching', {})
    batch_size = int(fm_cfg.get('latent', {}).get('batch_size', params.get('batch_size', 4096)))
    src_buffer = build_replay_buffer_from_episodes(episodes_src_aug, device, batch_size)
    tgt_buffer = build_replay_buffer_from_episodes(episodes_tgt_aug, device, batch_size)

    # ------------------------------
    # Load source ACT encoder (A_src) to produce latent z_a
    #  A_src 输入 = concat(robot_obs_src, act_arm_src)
    # ------------------------------
    lat_act_dim = int(params['lat_act_dim'])  # 用于 prior_lat_act

    class ActEncoder(nn.Module):
        def __init__(self, in_dim, out_dim, n_layers=3, hidden=256, batch_norm=False):
            super().__init__()
            self.net = utils.build_mlp(
                in_dim, out_dim, n_layers, hidden,
                activation='relu', output_activation='tanh', batch_norm=batch_norm
            )
        def forward(self, x):
            return self.net(x)

    # A_src 输入 = robot_obs_dim + act_arm_dim（源域）
    A_src_in_dim = src_dims.robot_obs_dim + src_act_arm_dim
    A_src = ActEncoder(
        in_dim=A_src_in_dim,
        out_dim=lat_act_dim,
        n_layers=int(params.get('n_layers', 3)),
        hidden=int(params.get('hidden_dim', 256)),
        batch_norm=False
    ).to(device)

    print("[flowmatching] loading act_enc.pt ...")
    ckpt_act = torch.load(pathlib.Path(params['src_model_dir']) / "act_enc.pt", map_location=device)
    # 修正 state_dict key 前缀（与训练时保存格式对齐）
    new_sd = {f"net.{k}": v for k, v in ckpt_act.items()}
    A_src.load_state_dict(new_sd, strict=True)
    for p in A_src.parameters():
        p.requires_grad = False
    A_src.eval()
    print(f"[flowmatching] A_src loaded and frozen. (in_dim={A_src_in_dim}, out_dim={lat_act_dim})")

    # ------------------------------
    # 统计：mu/std for src_act / tgt_act / latent_act & cond([p,v])
    # ------------------------------
    stats_passes  = int(fm_cfg.get('stats_passes', 200))
    latent_passes = int(fm_cfg.get('latent_passes', 200))
    print(f"[stats] stats_passes={stats_passes}, latent_passes={latent_passes}")

    def iter_buffer_act_and_c(buf: replay_buffer.ReplayBuffer, robot_dim: int, act_arm_dim: int, passes: int) -> Tuple[np.ndarray, np.ndarray]:
        As, Cs = [], []
        print("[stage] collecting act(wo_gripper)/cond([p,v]) stats ...")
        for _ in trange(passes, desc="act+cond stats", dynamic_ncols=True):
            with torch.no_grad():
                obs, act, _, _, _ = buf.sample()
                a_arm = act[:, :act_arm_dim]
                c = obs[:, -C_DIM:]  # 固定从尾部取 [p,v]
                As.append(a_arm.cpu().numpy())
                Cs.append(c.cpu().numpy())
        A = np.concatenate(As, axis=0)
        C = np.concatenate(Cs, axis=0)
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
        return ZA

    # src act & cond([p,v]) stats
    As, Cs_src = iter_buffer_act_and_c(src_buffer, src_dims.robot_obs_dim, src_act_arm_dim, stats_passes)
    mu_act_src, std_act_src = As.mean(axis=0), As.std(axis=0) + 1e-8
    mu_c_src, std_c_src = Cs_src.mean(axis=0), Cs_src.std(axis=0) + 1e-8

    # tgt act & cond([p,v]) stats
    At, Cs_tgt = iter_buffer_act_and_c(tgt_buffer, tgt_dims.robot_obs_dim, tgt_act_arm_dim, stats_passes)
    mu_act_tgt, std_act_tgt = At.mean(axis=0), At.std(axis=0) + 1e-8
    mu_c_tgt, std_c_tgt = Cs_tgt.mean(axis=0), Cs_tgt.std(axis=0) + 1e-8

    # latent stats over source domain
    ZA = iter_latent_act_stats(src_buffer, src_dims.robot_obs_dim, src_act_arm_dim, A_src, latent_passes)
    mu_za, std_za = ZA.mean(axis=0), ZA.std(axis=0) + 1e-8

    # Save stats files
    normalization = {
        'cond_src':   {'key': C_KEY, 'mu': mu_c_src.tolist(), 'std': std_c_src.tolist()},
        'cond_tgt':   {'key': C_KEY, 'mu': mu_c_tgt.tolist(), 'std': std_c_tgt.tolist()},
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
        'src_robot': src_robot,
        'tgt_robot': tgt_robot,
        'controller_type_src': controller_src,
        'controller_type_tgt': controller_tgt,
        'cond_key': C_KEY,
        'cond_dim_src': int(C_DIM),
        'cond_dim_tgt': int(C_DIM),
        'src_act_dim': int(src_dims.act_dim),
        'tgt_act_dim': int(tgt_dims.act_dim),
        'src_act_arm_dim': int(src_act_arm_dim),
        'tgt_act_arm_dim': int(tgt_act_arm_dim),
        'lat_act_dim': int(lat_act_dim),
        'robot_obs_dim_src': int(src_dims.robot_obs_dim),
        'robot_obs_dim_tgt': int(tgt_dims.robot_obs_dim),
    }
    with open(stats_dir / 'dataset.yml', 'w') as fp:
        YAML().dump(dataset_meta, fp)
    print(f"[save] dataset meta -> {stats_dir / 'dataset.yml'}")

    # schedule snapshot（FM 线性路径设定）
    path_meta = fm_cfg.get('path_defaults', {
        'type': 'flowmatching',
        'path': 'linear',
        't_sampling': 'uniform_0_1',
        'prior_endpoint': 'standard_normal',
    })
    schedule_meta = {
        'type': str(path_meta.get('type', 'flowmatching')),
        'path': str(path_meta.get('path', 'linear')),
        't_sampling': str(path_meta.get('t_sampling', 'uniform_0_1')),
        'prior_endpoint': str(path_meta.get('prior_endpoint', 'standard_normal')),
    }
    with open(stats_dir / 'schedule.yml', 'w') as fp:
        YAML().dump(schedule_meta, fp)
    print(f"[save] schedule -> {stats_dir / 'schedule.yml'}")

    # ------------------------------
    # Build three ACTION priors (with normalization baked in)
    #  - latent(action): x_dim = lat_act_dim, c_dim = C_DIM
    #  - src action:     x_dim = src_act_arm_dim, c_dim = C_DIM
    #  - tgt action:     x_dim = tgt_act_arm_dim, c_dim = C_DIM
    # ------------------------------
    use_condition = bool(fm_cfg.get('use_condition', True))

    lat_cfg = fm_cfg.get('latent', {})
    n_layers = int(lat_cfg.get('n_layers', params.get('n_layers', 3)))
    hidden_dim = int(lat_cfg.get('hidden_dim', params.get('hidden_dim', 256)))
    ema_decay = float(lat_cfg.get('ema_decay', 0.999))
    steps = int(lat_cfg.get('steps', params.get('tgt_align_timesteps', 300000)))
    bs = int(lat_cfg.get('batch_size', params.get('batch_size', 4096)))
    lr = float(lat_cfg.get('lr', params.get('lr', 3e-4)))

    log_freq = int(fm_cfg.get('log_freq', params.get('log_freq', 1000)))
    save_interval = int(fm_cfg.get('save_interval', params.get('evaluation', {}).get('save_interval', 20000)))
    time_embed_dim = 128

    prior_lat_act = PriorModel(x_dim=lat_act_dim, c_dim=C_DIM, hidden=hidden_dim,
                               n_layers=n_layers, use_condition=use_condition,
                               time_embed_dim=time_embed_dim).to(device)
    prior_lat_act.set_norm(mu_za, std_za, mu_c_src, std_c_src)

    prior_src_act = PriorModel(x_dim=src_act_arm_dim, c_dim=C_DIM,
                               hidden=hidden_dim, n_layers=n_layers,
                               use_condition=use_condition, time_embed_dim=time_embed_dim).to(device)
    prior_src_act.set_norm(mu_act_src, std_act_src, mu_c_src, std_c_src)

    prior_tgt_act = PriorModel(x_dim=tgt_act_arm_dim, c_dim=C_DIM,
                               hidden=hidden_dim, n_layers=n_layers,
                               use_condition=use_condition, time_embed_dim=time_embed_dim).to(device)
    prior_tgt_act.set_norm(mu_act_tgt, std_act_tgt, mu_c_tgt, std_c_tgt)

    # 统一训练配置
    train_cfg = TrainCfg(
        steps=steps,
        batch_size=bs,
        lr=lr,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        ema_decay=ema_decay,
        t_min=0.0,
        t_max=1.0,
    )

    # ------------------------------
    # Build batch makers (ACTIONS)
    #  - latent: za = A_src(concat(robot_obs_src, act_arm_src))
    #  - src/tgt: act_arm
    #  c = obs[:, -C_DIM:]
    # ------------------------------
    def make_batch_lat_act():
        obs, act, _, _, _ = src_buffer.sample()
        x_robot = obs[:, :src_dims.robot_obs_dim]
        a_arm   = act[:, :src_act_arm_dim]
        c       = obs[:, -C_DIM:]
        with torch.no_grad():
            enc_in = torch.cat([x_robot, a_arm], dim=-1)
            za = A_src(enc_in)
        return za, c

    def make_batch_src_act():
        obs, act, _, _, _ = src_buffer.sample()
        a_arm = act[:, :src_act_arm_dim]
        c     = obs[:, -C_DIM:]
        return a_arm, c

    def make_batch_tgt_act():
        obs, act, _, _, _ = tgt_buffer.sample()
        a_arm = act[:, :tgt_act_arm_dim]
        c     = obs[:, -C_DIM:]
        return a_arm, c

    # ------------------------------
    # 训练三路 ACTION priors（顺序执行）
    # ------------------------------
    train_one_prior(
        name='flow_lat_act',
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
        name='flow_src_act',
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
        name='flow_tgt_act',
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