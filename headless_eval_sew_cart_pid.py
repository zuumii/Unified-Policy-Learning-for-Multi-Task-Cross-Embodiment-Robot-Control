# headless_eval_sew_cart_pid.py
# 评测已训练好的“笛卡尔”策略（网络输出 7 维：6D 任务速度 + 抓手；SEW 通道用 PD 控制）。
# - 与原评测一致：保存视频、统计 action、尊重 render_camera
# - 可配置 SEW 几何并在线构建 θ 对关节的雅可比（仅在启用时参与）
# - 关键：在加载前“重建 7 维 Actor”，保证能加载你现有的 7 维 checkpoint
# - 兼容 dm-mujoco API（mj_name2id 等），并正确获取机器人底座位姿

import pathlib
import argparse
import json
import time
import gc
import warnings
import random
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from ruamel.yaml import YAML
import imageio

import utils

# 复用已有几何原语
from bc_cartsew_aug import (
    CartSEWAugAgent as Agent,
    _unwrap_mj, _arm_vel_cols, _arm_qpos_addrs, _arm_joint_list,
    _eef_spatial_jacobian, _analytic_point_jac,
    _stereo_theta_sigma_torch
)

# ----------------- 小工具 -----------------
def _get(d, k, default):
    return d[k] if isinstance(d, dict) and k in d else default

def _ensure_offscreen_context(sim):
    from robosuite.utils.binding_utils import MjRenderContextOffscreen
    if getattr(sim, "_render_context_offscreen", None) is not None:
        return
    try:
        sim._render_context_offscreen = MjRenderContextOffscreen(sim, device_id=0)
    except TypeError:
        sim._render_context_offscreen = MjRenderContextOffscreen(sim)

def _auto_pick_camera(env, params):
    # 严格尊重 config
    if isinstance(params, dict) and params.get('render_camera'):
        return params['render_camera']
    names = list(getattr(env, "camera_names", []) or [])
    prefs = ['frontview', 'agentview', 'birdview', 'sideview', 'topview']
    for n in prefs:
        if n in names:
            return n
    return names[0] if names else None

def _infer_key_shapes(env_probe, obs_keys):
    raw = env_probe.reset()
    shapes = {}
    for k in obs_keys:
        v = np.asarray(raw[k]).reshape(-1)
        shapes[k] = int(v.shape[0])
    return shapes

def _split_from_concat(obs_keys, obs_vec, key_shapes):
    out = {}; i = 0
    for k in obs_keys:
        n = key_shapes[k]
        out[k] = obs_vec[i:i+n]
        i += n
    return out

# ==== MuJoCo name/id 兼容包装（支持 dm-mujoco 与旧接口）====
def _mj_body_name2id(m, name: str) -> int:
    import mujoco as mj
    try:
        return int(mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, name))
    except Exception:
        fn = getattr(m, "body_name2id", None)
        if fn is None:
            raise
        return int(fn(name))

def _mj_body_id2name(m, bid: int) -> str:
    import mujoco as mj
    try:
        return mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(bid))
    except Exception:
        fn = getattr(m, "body_id2name", None)
        if fn is None:
            return f"body[{int(bid)}]"
        return fn(int(bid))

def _mj_site_name2id(m, name: str) -> int:
    import mujoco as mj
    try:
        return int(mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, name))
    except Exception:
        fn = getattr(m, "site_name2id", None)
        if fn is None:
            raise
        return int(fn(name))

def _mj_joint_name2id(m, name: str) -> int:
    import mujoco as mj
    try:
        return int(mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, name))
    except Exception:
        fn = getattr(m, "jnt_name2id", None)
        if fn is None:
            raise
        return int(fn(name))

# ---- 机器人底座世界位姿（robosuite 机器人）----
def _get_robot_base_pose(env):
    """
    返回 (pos_w, rot_w3x3, base_body_name)
    优先从 robot_model 读取 base 名；回退 'robot0_base'；再回退 body0
    """
    robot = env.robots[0]
    base_name = None
    if hasattr(robot, "robot_model") and robot.robot_model is not None:
        for attr in ("base_body", "root_body", "base_link", "base_name", "base_body_name"):
            if hasattr(robot.robot_model, attr):
                val = getattr(robot.robot_model, attr)
                if isinstance(val, str) and len(val) > 0:
                    base_name = val
                    break
    if not base_name:
        base_name = "robot0_base"

    m, d = _unwrap_mj(env)
    try:
        bid = _mj_body_name2id(m, base_name)
    except Exception:
        bid = 0
        try:
            base_name = _mj_body_id2name(m, 0)
        except Exception:
            base_name = "body0"

    pos_w = np.array(d.xpos[bid]).copy()
    rot_w = np.array(d.xmat[bid]).reshape(3, 3).copy()
    return pos_w, rot_w, base_name

def _q_from_obs(d, aux_joint_keys):
    if "robot0_joint_pos" in d:
        return np.asarray(d["robot0_joint_pos"], dtype=np.float32).reshape(-1)
    cos_key = next((k for k in aux_joint_keys if k.endswith("_cos") and "joint_pos" in k), None)
    sin_key = next((k for k in aux_joint_keys if k.endswith("_sin") and "joint_pos" in k), None)
    if cos_key and sin_key and (cos_key in d) and (sin_key in d):
        cosv = np.asarray(d[cos_key]).reshape(-1)
        sinv = np.asarray(d[sin_key]).reshape(-1)
        return np.arctan2(sinv, cosv).astype(np.float32)
    raise KeyError("需要 robot0_joint_pos 或 joint_pos_cos/sin 以还原 q")

def _x_task_from_keys(d, x_task_keys):
    parts = [np.asarray(d[k], dtype=np.float32).reshape(-1) for k in x_task_keys]
    return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float32)

def _set_global_seeds(seed):
    np.random.seed(seed); random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _wrap_to_pi(e):
    return ((e + np.pi) % (2*np.pi)) - np.pi

# ----------------- SEW 几何解析 + θ梯度 + J6 -----------------
def _resolve_body_site_world(env, typ: str, name: str, offset_xyz) -> np.ndarray:
    m, d = _unwrap_mj(env)
    off = np.asarray(offset_xyz, dtype=np.float64).reshape(3,)
    if typ == "body":
        bid = _mj_body_name2id(m, name)
        R = d.xmat[bid].reshape(3,3); t = d.xpos[bid]
        return (t + R @ off).copy()
    elif typ == "site":
        sid = _mj_site_name2id(m, name)
        R = d.site_xmat[sid].reshape(3,3); t = d.site_xpos[sid]
        return (t + R @ off).copy()
    else:
        raise ValueError(f"Unsupported type {typ}")

def _resolve_S(env, cfg_S: Dict[str,Any]) -> np.ndarray:
    typ = str(cfg_S.get("type", "base")).lower()
    off = np.asarray(cfg_S.get("offset",[0.,0.,0.]), dtype=np.float64).reshape(3,)
    if typ == "base":
        t0, R0, _ = _get_robot_base_pose(env)
        return (t0 + R0 @ off).copy()
    elif typ in ("body","site"):
        name = cfg_S.get("name","")
        if not name: raise ValueError(f"S.{typ} requires a 'name'")
        return _resolve_body_site_world(env, typ, name, off)
    elif typ == "fixed":
        pos = np.asarray(cfg_S.get("pos_world",[0.,0.,0.]), dtype=np.float64).reshape(3,)
        return (pos + off).copy()
    else:
        raise ValueError(f"S.type={typ} not supported")

def _resolve_joint_point(env, joint_sel: Dict[str,Any]) -> np.ndarray:
    sel = str(joint_sel.get("select","by_index")).lower()
    point = str(joint_sel.get("point","joint_pos"))
    off = np.asarray(joint_sel.get("offset",[0.,0.,0.]), dtype=np.float64).reshape(3,)
    m, d = _unwrap_mj(env)

    if sel == "by_index":
        idx = int(joint_sel.get("index", 0))
        arm_joints = _arm_joint_list(env)
        if idx < 0 or idx >= len(arm_joints):
            raise IndexError(f"E/W index out of range: {idx}")
        jid = int(arm_joints[idx]["jid"])
    elif sel == "by_name":
        nm = joint_sel.get("name","")
        if not nm: raise ValueError("E/W.select=by_name requires 'name'")
        jid = int(_mj_joint_name2id(m, nm))
    else:
        raise ValueError(f"E/W.select={sel} not supported")

    bid = int(m.jnt_bodyid[jid])

    if point.startswith("site:"):
        sname = point.split(":",1)[1]
        return _resolve_body_site_world(env, "site", sname, off)
    if point.startswith("body:"):
        bname = point.split(":",1)[1]
        return _resolve_body_site_world(env, "body", bname, off)

    p_local = np.array(m.jnt_pos[jid], dtype=np.float64).reshape(3,) + off
    R = d.xmat[bid].reshape(3,3); t = d.xpos[bid]
    return (t + R @ p_local).copy()

def _resolve_ref_dir_world(env, sew_geom: Dict[str,Any]) -> np.ndarray:
    if "ref_dir_world" in sew_geom:
        return np.asarray(sew_geom["ref_dir_world"], dtype=np.float64).reshape(3,)
    if "ref_dir_local" in sew_geom:
        loc = sew_geom["ref_dir_local"]
        vloc = np.asarray(loc.get("vec",[0.,0.,1.]), dtype=np.float64).reshape(3,)
        typ = str(loc.get("type","base")).lower()
        if typ == "base":
            _, Rb, _ = _get_robot_base_pose(env)
            return (Rb @ vloc).copy()
        elif typ in ("body","site"):
            name = loc.get("name","")
            if not name: raise ValueError("ref_dir_local requires name")
            m, d = _unwrap_mj(env)
            if typ == "body":
                bid = _mj_body_name2id(m, name); R = d.xmat[bid].reshape(3,3)
            else:
                sid = _mj_site_name2id(m, name); R = d.site_xmat[sid].reshape(3,3)
            return (R @ vloc).copy()
        else:
            raise ValueError(f"ref_dir_local.type={typ} not supported")
    return np.array([0.,0.,1.], dtype=np.float64)

def _theta_grads_numpy(S, E, W, ref_dir=(0.,0.,1.)):
    S_t = torch.tensor(S, dtype=torch.float64, requires_grad=True).view(1,3)
    E_t = torch.tensor(E, dtype=torch.float64, requires_grad=True).view(1,3)
    W_t = torch.tensor(W, dtype=torch.float64, requires_grad=True).view(1,3)
    _, theta = _stereo_theta_sigma_torch(S_t.float(), E_t.float(), W_t.float(), ref_dir=tuple(ref_dir))
    gS, gE, gW = torch.autograd.grad(theta, [S_t, E_t, W_t], retain_graph=False, create_graph=False)
    return gE.detach().cpu().numpy(), gW.detach().cpu().numpy()

def build_J6(env, q: np.ndarray) -> np.ndarray:
    """与训练时一致：只取末端位姿在当前 eef 点的 6×n 雅可比 [Jp; Jr]"""
    import mujoco as mj
    m, d = _unwrap_mj(env)
    qpos_addrs = _arm_qpos_addrs(env); vel_cols = _arm_vel_cols(env)
    n = len(qpos_addrs)

    d.qpos[qpos_addrs] = np.asarray(q, dtype=np.float64).reshape(n,)
    mj.mj_forward(m, d)

    # 取 eef 的世界点和 body id（与你原来的逻辑一致）
    robot = env.robots[0]
    sid = getattr(robot, "eef_site_id", None)
    if sid is not None:
        tip_pos = np.array(d.site_xpos[sid]).copy()
        tip_bid = int(m.site_bodyid[sid])
    else:
        eef_name = getattr(robot, "eef_name", None)
        if eef_name is not None:
            bid = _mj_body_name2id(m, eef_name)
            tip_pos = np.array(d.xpos[bid]).copy()
            tip_bid = bid
        else:
            tip_pos = np.zeros(3); tip_bid = 0

    Jp, Jr = _eef_spatial_jacobian(env.sim, tip_pos, tip_bid, vel_cols)  # (3,n),(3,n)
    J6 = np.vstack([Jp, Jr]).astype(np.float64)  # (6,n)
    return J6

def _theta_now(env, S, E, W, ref_dir) -> float:
    with torch.no_grad():
        _, th = _stereo_theta_sigma_torch(
            torch.tensor(S, dtype=torch.float32).view(1,3),
            torch.tensor(E, dtype=torch.float32).view(1,3),
            torch.tensor(W, dtype=torch.float32).view(1,3),
            ref_dir=tuple(np.asarray(ref_dir, dtype=np.float32).tolist())
        )
    return float(th.item())

# ----------------- “7维头”本地 Actor（只在 eval 内使用） -----------------
class LocalActor7(nn.Module):
    def __init__(self, in_dim, n_layers=3, hidden_dim=256):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)]
            d = hidden_dim
        layers += [nn.Linear(d, 7), nn.Tanh()]  # 7 维：6D 任务 + 1D 抓手
        self.net = nn.Sequential(*layers)
        self.net_depth = n_layers
        self.hidden_dim = hidden_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# ----------------- PolicyRunner -----------------
class CartSEWPolicyRunner:
    def __init__(self, agent: Agent, env, sew_geom_cfg: Dict[str,Any], sew_ctrl_cfg: Dict[str,Any], inv_eval_cfg: Dict[str,Any]):
        self.agent = agent
        self.env = env
        self.sew_geom = sew_geom_cfg or {}
        self.sew_ctrl = sew_ctrl_cfg or {}
        self.inv_eval = inv_eval_cfg or {}

        # 护栏参数
        self.rate_limit_delta = float(self.inv_eval.get("rate_limit_delta", getattr(agent, "rate_limit_delta", 0.10)))
        self.ema_alpha        = float(self.inv_eval.get("ema_alpha",        getattr(agent, "ema_alpha",        0.5)))
        self.k_null           = float(self.inv_eval.get("k_null",           getattr(agent, "k_null",           0.0)))
        self.w_pos            = float(self.inv_eval.get("w_pos",            getattr(agent, "w_pos_eval",       5.0)))
        self.w_ori            = float(self.inv_eval.get("w_ori",            getattr(agent, "w_ori_eval",       1.0)))
        self.w_sew            = float(self.inv_eval.get("w_sew",            getattr(agent, "w_sew_eval",       0.2)))
        self.cond_thresh      = float(self.inv_eval.get("cond_sing_thresh_eval", getattr(agent, "cond_sing_thresh_eval", 2000.0)))
        self.smin_thresh      = float(self.inv_eval.get("smin_thresh_eval",      getattr(agent, "smin_thresh_eval", 1e-3)))
        self.trunc_svd_eval   = bool(self.inv_eval.get("trunc_svd_eval",         getattr(agent, "trunc_svd_eval", True)))
        self.rcond            = float(self.inv_eval.get("rcond", 1e-4))
        self.dls_lambda       = float(self.inv_eval.get("dls_lambda", getattr(agent, "dls_lambda", 1e-3)))
        self.sew_enable       = bool(self.sew_ctrl.get("enable", True))

        # PD
        self.kp   = float(self.sew_ctrl.get("kp", 2.0))
        self.kd   = float(self.sew_ctrl.get("kd", 0.1))
        self.dtheta_max = float(self.sew_ctrl.get("dtheta_max", 1.0))
        self.theta_ref_mode = str(self.sew_ctrl.get("theta_ref_mode", "angle")).lower()
        self.theta_ref = float(self.sew_ctrl.get("theta_ref", 0.0))
        self.cos_ref = float(self.sew_ctrl.get("cos_ref", 1.0))
        self.bias = float(self.sew_ctrl.get("bias", 0.0))

        self.dt = 1.0 / float(getattr(env, "control_freq", 20))
        self.prev_cmd = None
        self.e_prev = 0.0

    def reset(self):
        self.prev_cmd = None
        self.e_prev = 0.0
        if hasattr(self.agent, "reset_eval_smoothing"):
            try: self.agent.reset_eval_smoothing()
            except Exception: pass

    def _theta_ref_from_mode(self, theta_now: float) -> float:
        if self.theta_ref_mode == "angle":
            return self.theta_ref
        if self.theta_ref_mode == "cos":
            c = np.clip(self.cos_ref, -1.0, 1.0)
            a = np.arccos(c)
            candidates = np.array([ a, -a, 2*np.pi - a, a - 2*np.pi ], dtype=np.float64)
            dists = np.abs(_wrap_to_pi(candidates - theta_now))
            th_star = candidates[np.argmin(dists)]
            return float(th_star + self.bias)
        return self.theta_ref

    def _pd_theta_dot(self, theta_now: float) -> float:
        theta_ref = self._theta_ref_from_mode(theta_now)
        e = _wrap_to_pi(theta_ref - theta_now)
        de = (e - self.e_prev) / self.dt
        theta_dot = self.kp * e + self.kd * de
        theta_dot = float(np.clip(theta_dot, -self.dtheta_max, self.dtheta_max))
        self.e_prev = e
        return theta_dot

    def step(self, x_task: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        # 0) 网络前向：与训练一致（6D + grip）
        x_t = torch.tensor(x_task, dtype=torch.float32, device=self.agent.device).view(1, -1)
        with torch.no_grad():
            out = self.agent.actor(x_t)
        y6 = out[0, :6].detach().cpu().numpy()
        g_cmd = float(out[0, 6].item())

        # 1) 主任务反解：只用 J(6×n)，与训练完全对齐
        J6 = build_J6(self.env, q)
        U, S, Vt = np.linalg.svd(J6, full_matrices=False)
        rcond = (self.rcond if self.rcond is not None else 1e-4)
        thr = (S.max() if S.size > 0 else 1.0) * rcond
        Sinv = np.where(S > thr, 1.0 / S, 0.0)
        J6_pinv = (Vt.T * Sinv) @ U.T
        qdot_task = J6_pinv @ y6

        # 主任务奇异度
        try:
            smin = float(S[-1]) if S.size > 0 else 0.0
            smax = float(S[0])  if S.size > 0 else 0.0
            cond = smax / (smin + 1e-12) if smin > 1e-12 else np.inf
        except Exception:
            smin, cond = 0.0, np.inf

        # 2) SEW PD：产生 theta_dot_cmd；以及诊断量
        S_world = _resolve_S(self.env, self.sew_geom.get("S", {"type": "base", "offset": [0, 0, 0]}))
        E_sel = self.sew_geom.get("E_joint", {"select": "by_index", "index": 3})
        W_sel = self.sew_geom.get("W_joint", {"select": "by_index", "index": 6})
        E_world = _resolve_joint_point(self.env, E_sel)
        W_world = _resolve_joint_point(self.env, W_sel)
        ref_dir = _resolve_ref_dir_world(self.env, self.sew_geom)

        theta_now = _theta_now(self.env, S_world, E_world, W_world, ref_dir)
        theta_dot_cmd = self._pd_theta_dot(theta_now) if self.sew_enable else 0.0

        # 诊断：投影与基
        eps = 1e-12
        u = (W_world - S_world)
        u_norm = np.linalg.norm(u)
        if u_norm > eps:
            u = u / u_norm
        else:
            u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        P = np.eye(3, dtype=np.float64) - np.outer(u, u)

        v = P @ (E_world - S_world)
        nv = float(np.linalg.norm(v))
        v_hat = v / (nv + eps)

        r = P @ ref_dir
        nr = float(np.linalg.norm(r))
        if nr < 1e-9:
            alt = np.array([1., 0., 0.], dtype=np.float64)
            r = P @ alt
            nr = float(np.linalg.norm(r))
        e1 = r / (nr + eps)
        e2 = np.cross(u, e1)
        e2n = float(np.linalg.norm(e2))
        if e2n > eps:
            e2 = e2 / e2n
        else:
            e2 = np.array([0., 1., 0.], dtype=np.float64)
        x_ = float(v_hat.dot(e1))
        y_ = float(v_hat.dot(e2))
        theta_check = float(np.arctan2(y_, x_))

        # === 3) 计算 J_theta(1×n)（θ 对关节的雅可比） ===
        m, d = _unwrap_mj(self.env)
        vel_cols = _arm_vel_cols(self.env)

        def _jid_from_sel(sel):
            ss = str(sel.get("select", "by_index")).lower()
            if ss == "by_index":
                idx = int(sel.get("index", 0))
                arm = _arm_joint_list(self.env)
                if idx < 0 or idx >= len(arm):
                    raise IndexError(f"joint index out of range: {idx}")
                return int(arm[idx]["jid"])
            nm = sel.get("name", "")
            return int(m.jnt_name2id(nm))

        jid_E = _jid_from_sel(E_sel)
        jid_W = _jid_from_sel(W_sel)

        # θ 对 E/W 的梯度；E/W 的位置雅可比（3×n）
        gE, gW = _theta_grads_numpy(S_world, E_world, W_world, ref_dir=ref_dir)   # (1,3),(1,3)
        J_E = _analytic_point_jac(self.env.sim, jid_E, vel_cols)                  # (3,n)
        J_W = _analytic_point_jac(self.env.sim, jid_W, vel_cols)                  # (3,n)
        J_theta = (gE @ J_E + gW @ J_W).reshape(1, -1)                            # (1,n)

        # ——— 可选：有限差分数值校验 J_theta（debug）———
        fd_rel_err = None
        if bool(self.inv_eval.get("debug_theta_fd", False)):
            import mujoco as mj
            qpos_addrs = _arm_qpos_addrs(self.env)

            # 同步 q
            d.qpos[qpos_addrs] = np.asarray(q, dtype=np.float64)
            mj.mj_forward(m, d)

            theta0 = _theta_now(self.env, S_world, E_world, W_world, ref_dir)
            eps_fd = float(self.inv_eval.get("fd_eps", 1e-5))
            JT_fd = np.zeros_like(J_theta)

            def _E_W_world_at_current_q():
                def _point_from_sel(sel):
                    ss = str(sel.get("select", "by_index")).lower()
                    if ss == "by_index":
                        idx = int(sel.get("index", 0))
                        arm = _arm_joint_list(self.env)
                        jid = int(arm[idx]["jid"])
                    else:
                        jid = int(m.jnt_name2id(sel.get("name","")))
                    bid = int(m.jnt_bodyid[jid])
                    p_local = np.array(m.jnt_pos[jid], dtype=np.float64).reshape(3,)
                    Rb = d.xmat[bid].reshape(3,3); tb = d.xpos[bid]
                    return (tb + Rb @ p_local).copy()
                return _point_from_sel(E_sel), _point_from_sel(W_sel)

            for j in range(J_theta.shape[1]):
                d.qpos[qpos_addrs[j]] += eps_fd
                mj.mj_forward(m, d)
                E_eps, W_eps = _E_W_world_at_current_q()
                theta1 = _theta_now(self.env, S_world, E_eps, W_eps, ref_dir)
                JT_fd[0, j] = (theta1 - theta0) / eps_fd
                d.qpos[qpos_addrs[j]] -= eps_fd
            num = float(np.linalg.norm(JT_fd - J_theta))
            den = float(np.linalg.norm(J_theta) + 1e-12)
            fd_rel_err = num / den

        # === 4) 纯零空间注入（不影响末端 6D 任务速度） ===
        # 主任务的零空间投影矩阵
        N = np.eye(J6_pinv.shape[0], dtype=np.float64) - (J6_pinv @ J6)

        # θ 的最小范数方向：按 J_theta 的最小二乘（只有一行）
        eps_s = 1e-12
        Jt_norm2 = float((J_theta @ J_theta.T).ravel()[0]) if J_theta.size else 0.0
        if self.sew_enable and (Jt_norm2 > 1e-10):
            qdot_sew_min = (J_theta.T * (theta_dot_cmd / (Jt_norm2 + eps_s))).reshape(-1)  # (n,)
        else:
            qdot_sew_min = np.zeros_like(qdot_task)

        # —— 投影效率：eta = (J_theta N J_theta^T) / (J_theta J_theta^T) —— 
        num_eta = float((J_theta @ N @ J_theta.T).ravel()[0]) if J_theta.size else 0.0
        den_eta = float((J_theta @ J_theta.T).ravel()[0]) if J_theta.size else 0.0
        eta = float(num_eta / (den_eta + 1e-12)) if den_eta > 0.0 else 0.0
        eta = float(np.clip(eta, 0.0, 1.0))

        # alpha 决策（enable 才允许注入）
        alpha = float(self.sew_ctrl.get("alpha", 0.5))
        if (not self.sew_enable) or (Jt_norm2 <= 1e-10) or (not np.isfinite(cond)) or (smin < self.smin_thresh):
            alpha = 0.0

        # 最终关节速度：主任务 +（零空间里的）SEW
        qdot = qdot_task + alpha * (N @ qdot_sew_min)

        # —— 提取当前末端位置（世界系 xyz）以记录 —— 
        # 和 build_J6 使用一致的 eef 点
        robot = self.env.robots[0]
        sid = getattr(robot, "eef_site_id", None)
        if sid is not None:
            eef_pos_w = np.array(d.site_xpos[sid]).reshape(3).copy()
        else:
            eef_name = getattr(robot, "eef_name", None)
            if eef_name is not None:
                bid_eef = m.body_name2id(eef_name)
                eef_pos_w = np.array(d.xpos[bid_eef]).reshape(3).copy()
            else:
                eef_pos_w = np.zeros(3, dtype=np.float64)

        # —— 记录这一步参与 SEW 计算的几何 & 诊断 —— 
        sew_points = {
            "frame": "world",
            "S": np.asarray(S_world, dtype=np.float64).tolist(),
            "E": np.asarray(E_world, dtype=np.float64).tolist(),
            "W": np.asarray(W_world, dtype=np.float64).tolist(),
            "ref_dir": np.asarray(ref_dir, dtype=np.float64).tolist(),
        }

        sew_diag = {
            "W_minus_S_norm": float(np.linalg.norm(W_world - S_world)),
            "P_E_minus_S_norm": nv,
            "P_ref_norm": nr,
            "u": u.tolist(),
            "e1": e1.tolist(),
            "e2": e2.tolist(),
            "v_hat": v_hat.tolist(),
            "x": x_,
            "y": y_,
            "theta_check": theta_check,
        }
        if fd_rel_err is not None:
            sew_diag["fd_rel_err"] = float(fd_rel_err)

        # 5) 映射到动作域 + slew-rate + EMA
        cmd_scale = self.agent.cmd_scale.detach().cpu().numpy().reshape(-1)
        cmd = np.clip(qdot / (cmd_scale + 1e-12), -1.0, 1.0)

        if self.prev_cmd is None:
            self.prev_cmd = cmd.copy()
        delta = self.rate_limit_delta
        cmd = np.clip(cmd, self.prev_cmd - delta, self.prev_cmd + delta)
        alpha_ema = self.ema_alpha
        cmd = alpha_ema * cmd + (1.0 - alpha_ema) * self.prev_cmd
        self.prev_cmd = cmd.copy()

        # 记录底座姿态（用于复查 S=base 的取值）
        base_t, base_R, base_name = _get_robot_base_pose(self.env)

        act = np.concatenate([cmd.reshape(-1), np.array([g_cmd], dtype=np.float32)], axis=0)
        info = {
            # SEW 角与控制
            "theta_now": float(theta_now),
            "theta_dot_cmd": float(theta_dot_cmd),
            "Jtheta_norm2": float(Jt_norm2),

            # 主任务奇异度
            "smin": float(smin),
            "cond": float(cond),

            # SEW 投影效率
            "sew_eff": {
                "eta": float(eta),
                "num": float(num_eta),
                "den": float(den_eta),
            },

            # 末端位姿（仅位置，世界系 xyz）
            "eef": {
                "pos_world": eef_pos_w.tolist(),
                "task_twist_cmd": y6.tolist(),  # 6D 任务速度命令（可用于对齐分析）
            },

            # 几何与诊断
            "sew_points": sew_points,
            "sew_diag": sew_diag,

            # 底座姿态
            "base_pose": {
                "name": base_name,
                "t_world": base_t.tolist(),
                "R_world_rowmajor": base_R.reshape(-1).tolist(),
            },

            # 关节选择记录
            "sew_joints": {
                "E_joint": E_sel,
                "W_joint": W_sel,
            },

            # 是否启用 SEW 通道（确认 yml 生效）
            "sew_enable": bool(self.sew_enable),
            "alpha_used": float(alpha),
        }
        return act.astype(np.float32), info

# ----------------- 主程序 -----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='eval config file path')
    return p.parse_args()

def main():
    warnings.filterwarnings("ignore", message="Box bound precision lowered by casting to float32")

    args = parse_args()
    yaml = YAML(typ='safe')
    params = yaml.load(open(args.config, 'r'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 组装 obs_keys_env：x_task_keys + aux_joint_keys
    robot_obs_keys = params['robot_obs_keys']
    obj_obs_keys   = params.get('obj_obs_keys', [])
    x_task_keys    = robot_obs_keys + obj_obs_keys
    aux_joint_keys = params.get('aux_joint_keys', [])
    seen = set()
    obs_keys_env = [k for k in (x_task_keys + aux_joint_keys) if not (k in seen or seen.add(k))]

    # 探针
    env_probe = utils.make_robosuite_env(
        params['env_name'], robots=params['robots'],
        controller_type=params['controller_type'],
        **params['env_kwargs'],
    )
    key_shapes = _infer_key_shapes(env_probe, obs_keys_env)

    # 维度
    obs_dim_cart_in = sum(key_shapes.get(k, 0) for k in x_task_keys)
    if "robot0_joint_pos" in key_shapes:
        dof_arm = key_shapes["robot0_joint_pos"]
    elif ("robot0_joint_pos_cos" in key_shapes) and ("robot0_joint_pos_sin" in key_shapes):
        dof_arm = key_shapes["robot0_joint_pos_cos"]
    else:
        raise KeyError("需要 robot0_joint_pos 或 robot0_joint_pos_cos/sin 之一以还原 q")

    # 环境
    seed_eval = int(params.get('seed', 42)) + 100
    _set_global_seeds(seed_eval)
    env = utils.make(
        params['env_name'], robots=params['robots'],
        controller_type=params['controller_type'],
        obs_keys=obs_keys_env, seed=seed_eval, render=False,
        **params['env_kwargs'],
    )

    # 渲染
    camera_name = _auto_pick_camera(env, params)
    vcfg   = params.get('video', {}) if isinstance(params.get('video', {}), dict) else {}
    width  = int(_get(vcfg, 'width', 640))
    height = int(_get(vcfg, 'height', 480))
    fps    = int(_get(vcfg, 'fps', 20))

    # Agent：构造后，用“本地 7 维头”替换再加载
    model_dir = pathlib.Path(params['model_dir']).resolve()
    try:
        agent = Agent(
            x_task_dim=obs_dim_cart_in,
            dof_arm=dof_arm,
            device=device,
            n_layers=params.get('actor_n_layers', params.get('n_layers', 3)),
            hidden_dim=params.get('actor_hidden_dim', params.get('hidden_dim', 256)),
            lr=params.get('lr', 3e-4),
        )
    except TypeError:
        agent = Agent(x_task_dim=obs_dim_cart_in, dof_arm=dof_arm, device=device)

    # 关键：重建 7 维 Actor 头（避免 ckpt 维度不匹配）
    n_layers = params.get('actor_n_layers', params.get('n_layers', 3))
    hidden_dim = params.get('actor_hidden_dim', params.get('hidden_dim', 256))
    agent.actor = LocalActor7(obs_dim_cart_in, n_layers=n_layers, hidden_dim=hidden_dim).to(device)

    agent.load(model_dir)
    agent.attach_env(env._env) if hasattr(env, "_env") else agent.attach_env(env)

    # 可选 neutral
    try:
        probe_obs = env_probe.reset()
        if "robot0_joint_pos" in probe_obs:
            q0 = np.asarray(probe_obs["robot0_joint_pos"], dtype=np.float32)
        else:
            cosv = probe_obs["robot0_joint_pos_cos"]; sinv = probe_obs["robot0_joint_pos_sin"]
            q0 = np.arctan2(np.asarray(sinv), np.asarray(cosv)).astype(np.float32)
        if hasattr(agent, "set_q_neutral"):
            agent.set_q_neutral(q0)
    except Exception:
        pass

    # PolicyRunner
    sew_geom = params.get("sew_geom", {
        "S": {"type":"base","offset":[0,0,0]},
        "E_joint": {"select":"by_index","index":3,"point":"joint_pos","offset":[0,0,0]},
        "W_joint": {"select":"by_index","index":6,"point":"joint_pos","offset":[0,0,0]},
        "ref_dir_world": [0.0, 0.0, 1.0],
    })
    sew_ctrl = params.get("sew_ctrl", {
        "enable": True,
        "mode":"pd", "kp":2.0, "kd":0.1, "dtheta_max":1.0,
        "theta_ref_mode":"angle", "theta_ref":0.0
    })
    inv_eval = params.get("inverse_eval", {
        "rcond": 1e-4,
        "cond_sing_thresh_eval": 2000.0,
        "smin_thresh_eval": 1e-3,
        "trunc_svd_eval": True,
        "rate_limit_delta": 0.10,
        "ema_alpha": 0.5,
        "k_null": 0.0,
        "w_pos": 5.0, "w_ori": 1.0, "w_sew": 0.2,
        # 可配 SEW 权重（参与最小二乘的第7行），默认 2.0
        "w_theta": 2.0,
        # "dls_lambda": 1e-3,
    })

    runner = CartSEWPolicyRunner(agent, env._env if hasattr(env, "_env") else env, sew_geom, sew_ctrl, inv_eval)

    # 结果目录 & 元信息
    run_dir = (model_dir.parent / f"eval_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": str(pathlib.Path(args.config).resolve()),
        "env_name": params['env_name'],
        "robots": params['robots'],
        "controller_type": params['controller_type'],
        "seed": seed_eval,
        "camera": camera_name or "free_camera",
        "width": width, "height": height, "fps": fps,
        "num_episodes": int(params.get('num_episodes', 10)),
        "model_dir": str(model_dir), "headless": True,
        "obs_keys_env": obs_keys_env,
        "x_task_keys": x_task_keys,
        "aux_joint_keys": aux_joint_keys,
        "sew_geom": sew_geom,
        "sew_ctrl": sew_ctrl,
        "inverse_eval": inv_eval,
    }
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 评测 + 录像
    returns, global_actions = [], []
    num_episodes = int(params.get('num_episodes', 10))

    try:
        for i in range(num_episodes):
            obs = env.reset()
            _ensure_offscreen_context(env.sim)
            runner.reset()

            done = False
            ep_ret = 0.0

            ep_dir = run_dir / f"ep_{i:03d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            tmp_mp4 = ep_dir / f"ep_{i:03d}_tmp.mp4"
            writer = imageio.get_writer(tmp_mp4.as_posix(), fps=fps)

            def grab():
                frame = env.sim.render(camera_name=camera_name, width=width, height=height)
                return np.flipud(frame)

            writer.append_data(grab())

            ep_actions = []
            ep_infos = []

            while not done:
                d = _split_from_concat(obs_keys_env, obs, key_shapes)
                q = _q_from_obs(d, aux_joint_keys)
                x_task = _x_task_from_keys(d, x_task_keys)

                action, info = runner.step(x_task, q)
                ep_actions.append(action.copy()); ep_infos.append(info)

                obs, reward, done, _ = env.step(action)
                ep_ret += float(reward)
                writer.append_data(grab())

            writer.close()

            acts = np.asarray(ep_actions, dtype=np.float32)
            np.save(ep_dir / "actions.npy", acts)
            with open(ep_dir / "sew_metrics.json", "w", encoding="utf-8") as f:
                json.dump(ep_infos, f, ensure_ascii=False, indent=2)

            act_min = acts.min(axis=0).tolist(); act_max = acts.max(axis=0).tolist()
            act_mean = acts.mean(axis=0).tolist(); act_std  = acts.std(axis=0).tolist()

            thr_hi = 0.98; thr_09 = 0.90
            abs_acts = np.abs(acts)
            sat_hi_overall = float((abs_acts >= thr_hi).sum() / abs_acts.size)
            sat_hi_per_dim = (abs_acts >= thr_hi).sum(axis=0) / acts.shape[0]
            sat_09_overall = float((abs_acts >= thr_09).sum() / abs_acts.size)
            sat_09_per_dim = (abs_acts >= thr_09).sum(axis=0) / acts.shape[0]

            bins = np.linspace(-1.0, 1.0, 41, dtype=np.float32)
            hist_counts = []
            for ddd in range(acts.shape[1]):
                h, _ = np.histogram(acts[:, ddd], bins=bins)
                hist_counts.append(h.astype(np.int32))
            hist_counts = np.stack(hist_counts, axis=0)
            np.save(ep_dir / "action_hist_counts.npy", hist_counts)
            np.save(ep_dir / "action_hist_bins.npy", bins)

            with open(ep_dir / "action_stats.json", "w", encoding="utf-8") as f:
                json.dump({
                    "return": float(ep_ret),
                    "steps": int(acts.shape[0]),
                    "min": act_min,
                    "max": act_max,
                    "mean": act_mean,
                    "std": act_std,
                    "thr_0.90_overall": sat_09_overall,
                    "thr_0.90_per_dim": sat_09_per_dim.tolist(),
                    "thr_0.98_overall": sat_hi_overall,
                    "thr_0.98_per_dim": sat_hi_per_dim.tolist(),
                    "hist_bins": int(len(bins) - 1)
                }, f, ensure_ascii=False, indent=2)

            final_name = f"ep_{i:03d}_return_{ep_ret:.1f}.mp4"
            (ep_dir / final_name).write_bytes(tmp_mp4.read_bytes())
            tmp_mp4.unlink()

            with open(ep_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump({"episode": i, "return": float(ep_ret), "video": final_name},
                          f, ensure_ascii=False, indent=2)

            returns.append(ep_ret); global_actions.append(acts)

            grip_idx = acts.shape[1] - 1
            print(
                f"[EvalStats ep {i:03d}] ret={ep_ret:.1f} | "
                f"|a|≥0.90 overall={sat_09_overall*100:.1f}% (grip={sat_09_per_dim[grip_idx]*100:.1f}%) | "
                f"|a|≥0.98 overall={sat_hi_overall*100:.1f}% (grip={sat_hi_per_dim[grip_idx]*100:.1f}%) | "
                f"min={np.round(acts.min(0),3)} | max={np.round(acts.max(0),3)}"
            )
            print(f"[Eval] ep {i:03d} | return {ep_ret:.3f} | saved: {ep_dir/final_name}")

        # 全局统计
        if global_actions:
            all_actions = np.concatenate(global_actions, axis=0)
            g_abs = np.abs(all_actions)
            g_stats = {
                "min": all_actions.min(axis=0).tolist(),
                "max": all_actions.max(axis=0).tolist(),
                "mean": all_actions.mean(axis=0).tolist(),
                "std": all_actions.std(axis=0).tolist(),
                "thr_0.90_overall": float((g_abs >= 0.90).sum() / g_abs.size),
                "thr_0.90_per_dim": ((g_abs >= 0.90).sum(axis=0) / all_actions.shape[0]).tolist(),
                "thr_0.98_overall": float((g_abs >= 0.98).sum() / g_abs.size),
                "thr_0.98_per_dim": ((g_abs >= 0.98).sum(axis=0) / all_actions.shape[0]).tolist(),
            }
            with open(run_dir / "global_action_stats.json", "w", encoding="utf-8") as f:
                json.dump(g_stats, f, ensure_ascii=False, indent=2)
            print("[EvalStats Global]", json.dumps(g_stats))

    finally:
        try:
            if getattr(env.sim, "_render_context_offscreen", None) is not None:
                try: env.sim._render_context_offscreen.gl_ctx.free()
                except Exception: pass
                env.sim._render_context_offscreen = None
        except Exception:
            pass
        try: env.close()
        except Exception: pass
        del env
        gc.collect()

    if returns:
        print(f"[Eval] avg return over {len(returns)} eps : {np.mean(returns):.2f} ± {np.std(returns):.2f}")

if __name__ == '__main__':
    main()