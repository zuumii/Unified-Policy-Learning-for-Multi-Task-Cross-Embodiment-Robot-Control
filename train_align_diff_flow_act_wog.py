# train_align.py
# =========================================================
# 离线对齐训练（原流程 + 加固，兼容两种续训保存法）：
# - 形状探测：robosuite 原生 env + safe_reset
# - 续训：推荐 align_state.pt（aligner: 权重+优化器+步数）；可选附加分文件保存
# - 与 SEW 无关逻辑不变；SEW 配置透传给 Aligner（sew_cfg）
# - 新增：Diffusion 先验（与 GAN 并存，权重/开关由 config 控制）
# - 新增：cond.keys（可为多个键，按顺序拼接），严格维度校验与传递
# =========================================================
import pathlib
import argparse
import time
from ruamel.yaml import YAML

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F

import utils
import replay_buffer
from align_diff_flow_act_wog import ObsActAgent as Agent
from align_diff_flow_act_wog import ObsActAligner as Aligner
# 需要 mujoco 的雅可比
import mujoco as mj

def _unwrap_mj_sim(sim):
    """
    robosuite 在不同后端上会包一层 MjModel/MjData。
    这里把 sim.model / sim.data 解包成 mujoco._structs.MjModel / MjData。
    """
    m_wrapped = getattr(sim, "model", None)
    d_wrapped = getattr(sim, "data", None)

    # 常见字段名：model/_model or data/_data
    m = getattr(m_wrapped, "_model", getattr(m_wrapped, "model", m_wrapped))
    d = getattr(d_wrapped, "_data",  getattr(d_wrapped, "data",  d_wrapped))
    return m, d


def _get_arm_vel_indexes(env, robot_id=0):
    """
    取“臂”的 qvel 索引列表（优先用 robosuite 暴露的 _ref_joint_vel_indexes；退化到 qpos 索引）。
    """
    robot = env.robots[robot_id]
    vel_idx = getattr(robot, "_ref_joint_vel_indexes", None)
    if vel_idx is None:
        # 有些版本没有单独的 vel 索引；通常 qpos/ qvel 顺序一致，退化用 qpos 段代替
        vel_idx = getattr(robot, "_ref_joint_pos_indexes", None)
    if vel_idx is None:
        vel_idx = getattr(robot, "joint_indexes", None)
    if vel_idx is None:
        raise RuntimeError("无法定位该机器人的 qvel 段索引。")
    return np.asarray(vel_idx).ravel().astype(int)

def _eef_jacobians(env, q, robot_id=0):
    """
    写入一帧关节角 q，forward，然后用 mujoco 求 EEF 的雅可比：
      返回 (Jp, Jr)，形状 (3, dof_arm)，分别是线速度与角速度对关节速度的雅可比。
    """
    # 写 q 并 forward
    _set_qpos_and_forward(env, q, robot_id=robot_id)

    sim = env.sim
    model, data = _unwrap_mj_sim(sim)  # ← 关键：解包成 mujoco 原生结构

    # 找 EEF site id：优先用 robosuite 的属性
    robot = env.robots[robot_id]
    eef_sid = getattr(robot, "eef_site_id", None)
    if eef_sid is None:
        eef_name = getattr(robot, "eef_site_name", None)
        if eef_name is not None:
            # 这里用 wrapper 的 name2id 取 id 即可，id 在底层模型中相同
            eef_sid = sim.model.site_name2id(eef_name)
        else:
            candidates = [i for i, n in enumerate(sim.model.site_names) if ("eef" in n.lower())]
            if not candidates:
                raise RuntimeError("未能定位 EEF site。")
            eef_sid = candidates[0]

    # 构造 C 连续的 float64 Jacobian 缓冲
    nv = int(model.nv)
    jacp = np.zeros((3, nv), dtype=np.float64, order="C")
    jacr = np.zeros((3, nv), dtype=np.float64, order="C")

    # 用解包后的 model/data 计算雅可比
    mj.mj_jacSite(model, data, jacp, jacr, int(eef_sid))

    # 只切出“臂”的列
    vel_idx = _get_arm_vel_indexes(env, robot_id=robot_id)  # 来自 wrapper，但索引与底层一致
    Jp_arm = jacp[:, vel_idx].astype(np.float32, copy=False)
    Jr_arm = jacr[:, vel_idx].astype(np.float32, copy=False)
    return Jp_arm, Jr_arm

def _set_qpos_and_forward(env, q, robot_id=0):
    """
    把一帧关节角 q 写进 env 对应机器人，然后 forward。
    仅设置“臂”的关节（不要包含抓手），q: (dof,)
    """
    robot = env.robots[robot_id]
    # 优先走官方接口；不行就直接写 qpos
    for meth in ["set_joint_positions", "set_robot_joint_positions", "set_joint_pos"]:
        fn = getattr(robot, meth, None)
        if callable(fn):
            try:
                fn(np.asarray(q, dtype=float))
                env.sim.forward()
                return
            except Exception:
                pass
    idx = getattr(robot, "_ref_joint_pos_indexes", None)
    if idx is None:
        idx = getattr(robot, "joint_indexes", None)
    if idx is None:
        raise RuntimeError("无法定位该机器人的 qpos 段索引。")
    env.sim.data.qpos[np.asarray(idx).ravel().astype(int)] = np.asarray(q, dtype=float)
    env.sim.forward()

def _read_eef_world_pos(env, robot_id=0, obs_key=None):
    """
    从观测图读取世界系 EEF 坐标（与你训练时 key 对齐）
    """
    if obs_key is None:
        obs_key = f"robot{robot_id}_eef_pos"
    obs = None
    for meth in ["_get_observation", "_get_observations", "get_observation", "get_observations"]:
        fn = getattr(env, meth, None)
        if callable(fn):
            try:
                obs = fn(force_update=True)
            except TypeError:
                obs = fn()
            break
    if not (isinstance(obs, dict) and (obs_key in obs)):
        raise RuntimeError(f"观测中未找到键：{obs_key}")
    return np.asarray(obs[obs_key], dtype=float).reshape(3)

# ---------- 安全 reset（防 robosuite RandomizationError） ----------
def safe_reset(env, max_tries=50, reseed=True, sleep=0.0):
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='train config file path')
    return p.parse_args()


def _find_latest_step_dir(models_dir: pathlib.Path):
    cands = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith('step_')]
    if not cands:
        return None
    cands.sort(key=lambda d: int(d.name.split('_')[-1]))
    return cands[-1]


# ======== SEW 外来模型结构解析工具（仅用于尺寸校验与打印） ========
def _load_state_dict_flexible(model_path: str):
    """兼容直接 state_dict 或 {'state_dict': ...}"""
    obj = torch.load(model_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError("SEW 模型文件格式不支持：既不是 state_dict，也不是包含 'state_dict' 的字典。")

def _extract_linear_indices(keys):
    idxs = []
    for k in keys:
        if k.startswith("net.") and k.endswith(".weight"):
            try:
                idxs.append(int(k.split(".")[1]))
            except Exception:
                pass
    return sorted(set(idxs))

def _infer_mlp_spec_from_state(state):
    """
    推断：
      - in_dim: 首层 Linear 的 in_features
      - out_dims: 每个 Linear 的 out_features 列表；末层应为 1
      - depth: 线性层数（与 out_dims 长度相同）
    假设结构：Linear, ReLU, Linear, ReLU, ..., Linear, Tanh
    """
    keys = list(state.keys())
    lin_idxs = _extract_linear_indices(keys)
    if not lin_idxs:
        raise RuntimeError("SEW 模型：未发现 Linear 层权重（形如 'net.{i}.weight'）。")
    w0 = state[f"net.{lin_idxs[0]}.weight"]
    if w0.ndim != 2:
        raise RuntimeError("SEW 模型：首层 Linear 权重不是二维。")
    in_dim = int(w0.shape[1])

    out_dims = []
    prev = in_dim
    for li in lin_idxs:
        w = state[f"net.{li}.weight"]
        if w.ndim != 2:
            raise RuntimeError(f"SEW 模型：net.{li}.weight 不是二维。")
        out_dim = int(w.shape[0])
        in_feat = int(w.shape[1])
        if in_feat != prev:
            raise RuntimeError(
                f"SEW 模型层尺寸不一致：net.{li}.weight in_features={in_feat} 与上一层 out={prev} 不匹配。"
            )
        out_dims.append(out_dim)
        prev = out_dim
    return {"in_dim": in_dim, "out_dims": out_dims, "depth": len(out_dims)}
# ===========================================================


def main():
    args = parse_args()
    yaml = YAML(typ='safe')
    params = yaml.load(open(args.config))

    # ------------------------------
    # logging dirs
    # ------------------------------
    logdir_prefix = pathlib.Path(params.get('logdir_prefix') or pathlib.Path(__file__).parent)
    data_path = logdir_prefix / 'logs' / time.strftime("%m.%d.%Y")
    logdir = '_'.join([
        time.strftime("%H-%M-%S"),
        params['env_name'],
        params['src_env']['robot'],
        params['src_env']['controller_type'],
        params['tgt_env']['robot'],
        params['tgt_env']['controller_type'],
        params.get('suffix', 'align')
    ])
    logdir = data_path / logdir
    logdir.mkdir(parents=True, exist_ok=True)

    
    params['logdir'] = str(logdir)
    print(params)

    # dump params
    import yaml as pyyaml
    with open(logdir / 'params.yml', 'w') as fp:
        pyyaml.safe_dump(params, fp, sort_keys=False)

    model_dir = logdir / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    params['model_dir'] = str(model_dir)
    src_model_dir = pathlib.Path(params['src_model_dir'])   # 局部变量用于后续加载
    params['src_model_dir'] = str(src_model_dir)            # 回写为 str 以便 safe_dump

    logger = SummaryWriter(log_dir=params['logdir'])

    # ------------------------------
    # env + dims
    # ------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # === 解析 cond：obs 与 act 可不同 ===
    cond_obs_cfg = params.get('cond', {}) or {}
    cond_act_cfg = params.get('cond_act', {}) or {}  # 允许没有；没有则默认与 obs 相同

    cond_obs_keys = list(cond_obs_cfg.get('keys', []) or [])
    if not isinstance(cond_obs_keys, list):
        raise RuntimeError("config.cond.keys 必须是 list。")

    # 动作条件键：若未提供则复用 obs 的键
    if 'keys' in cond_act_cfg and cond_act_cfg['keys'] is not None:
        cond_act_keys = list(cond_act_cfg.get('keys') or [])
        if not isinstance(cond_act_keys, list):
            raise RuntimeError("config.cond_act.keys 必须是 list。")
    else:
        cond_act_keys = list(cond_obs_keys)

    # 严格检查开关：两边各自可设，未给则复用 cond.strict_dim_check
    strict_dim_check_obs = bool(cond_obs_cfg.get('strict_dim_check', True))
    strict_dim_check_act = bool(cond_act_cfg.get('strict_dim_check', strict_dim_check_obs))

    # 训练阶段需要把两套 cond 键都喂进环境
    merged_cond_keys = []
    seen = set()
    for k in cond_obs_keys + cond_act_keys:
        if k not in seen:
            merged_cond_keys.append(k)
            seen.add(k)

    src_train_obs_keys = list(params['src_env']['robot_obs_keys']) + merged_cond_keys
    tgt_train_obs_keys = list(params['tgt_env']['robot_obs_keys']) + merged_cond_keys

    src_env = utils.make(
        params['env_name'],
        robots=params['src_env']['robot'],
        controller_type=params['src_env']['controller_type'],
        obs_keys=src_train_obs_keys,
        seed=params['seed'],
        **params['env_kwargs'],
    )
    tgt_env = utils.make(
        params['env_name'],
        robots=params['tgt_env']['robot'],
        controller_type=params['tgt_env']['controller_type'],
        obs_keys=tgt_train_obs_keys,
        seed=params['seed'],
        **params['env_kwargs'],
    )

    # 评估用 env（robot+obj 观测）（保持不变）
    src_eval_env = utils.make(
        params['env_name'],
        robots=params['src_env']['robot'],
        controller_type=params['src_env']['controller_type'],
        obs_keys=params['src_env']['robot_obs_keys'] + params['src_env']['obj_obs_keys'],
        seed=params['seed'],
        **params['env_kwargs'],
    )
    tgt_eval_env = utils.make(
        params['env_name'],
        robots=params['tgt_env']['robot'],
        controller_type=params['tgt_env']['controller_type'],
        obs_keys=params['tgt_env']['robot_obs_keys'] + params['tgt_env']['obj_obs_keys'],
        seed=params['seed'],
        **params['env_kwargs'],
    )

    # ✅ 用原生 robosuite env 探测 shape（reset 返回 dict）
    src_probe_env = utils.make_robosuite_env(
        params['env_name'],
        robots=params['src_env']['robot'],
        controller_type=params['src_env']['controller_type'],
        **params['env_kwargs'],
    )
    tgt_probe_env = utils.make_robosuite_env(
        params['env_name'],
        robots=params['tgt_env']['robot'],
        controller_type=params['tgt_env']['controller_type'],
        **params['env_kwargs'],
    )
    src_probe = safe_reset(src_probe_env, max_tries=50, reseed=True)
    tgt_probe = safe_reset(tgt_probe_env, max_tries=50, reseed=True)

    # 训练时 robot_obs_dim（不含 cond）：
    src_robot_obs_shape = np.concatenate([src_probe[k] for k in params['src_env']['robot_obs_keys']]).shape
    tgt_robot_obs_shape = np.concatenate([tgt_probe[k] for k in params['tgt_env']['robot_obs_keys']]).shape

    # === cond 维度：分别计算 obs-cond 与 act-cond ===
    def _shape_of_keys(probe_dict, keys):
        if not keys:
            return 0
        try:
            return int(np.concatenate([probe_dict[k] for k in keys]).shape[0])
        except KeyError as e:
            raise RuntimeError(f"cond.keys 包含不存在的键：{e}")

    cond_obs_dim_src = _shape_of_keys(src_probe, cond_obs_keys)
    cond_obs_dim_tgt = _shape_of_keys(tgt_probe, cond_obs_keys)
    if strict_dim_check_obs and cond_obs_dim_src != cond_obs_dim_tgt:
        raise RuntimeError(f"[cond(obs)] 维度在 src/tgt 不一致：src={cond_obs_dim_src}, tgt={cond_obs_dim_tgt}")

    cond_act_dim_src = _shape_of_keys(src_probe, cond_act_keys)
    cond_act_dim_tgt = _shape_of_keys(tgt_probe, cond_act_keys)
    if strict_dim_check_act and cond_act_dim_src != cond_act_dim_tgt:
        raise RuntimeError(f"[cond(act)] 维度在 src/tgt 不一致：src={cond_act_dim_src}, tgt={cond_act_dim_tgt}")

    cond_obs_dim = cond_obs_dim_src  # 用作对齐器 obs 先验的 cond_dim
    cond_act_dim = cond_act_dim_src  # 用作对齐器 act 先验的 cond_dim
    
    # === 为 merged_cond_keys 建索引映射（按 env.observation 的拼接顺序） ===
    # 训练阶段每条观测的结构： [ robot_obs | merged_cond_keys逐段拼接 ]
    # 我们需要从中抽出两份：c_obs（按 cond_obs_keys 顺序）与 c_act（按 cond_act_keys 顺序）

    def _build_key_slices(probe_dict, robot_keys, merged_keys):
        # robot 段长度
        robot_len = int(np.concatenate([probe_dict[k] for k in robot_keys]).shape[0])
        # 为 merged_cond_keys 计算每个 key 的长度与在 merged-cond 段内的偏移
        lens = {}
        for k in merged_keys:
            val = np.asarray(probe_dict[k])
            lens[k] = int(val.reshape(-1).shape[0])  # 容错：允许 (3,), (3,1), ...
        # 计算 key -> (start, end)（相对于整条观测；即 robot 段之后）
        slices = {}
        offset = robot_len
        for k in merged_keys:
            l = lens[k]
            slices[k] = (offset, offset + l)
            offset += l
        return robot_len, slices  # robot_len 用于分出 robot 段

    src_robot_len, src_cond_slices = _build_key_slices(src_probe, params['src_env']['robot_obs_keys'], merged_cond_keys)
    tgt_robot_len, tgt_cond_slices = _build_key_slices(tgt_probe, params['tgt_env']['robot_obs_keys'], merged_cond_keys)

    def _gather_cond(obs_full: torch.Tensor, robot_len: int, cond_slices: dict, keys: list):
        """按 keys 顺序拼接对应段"""
        if not keys:
            return None
        parts = []
        for k in keys:
            s, e = cond_slices[k]
            parts.append(obs_full[:, s:e])
        return torch.cat(parts, dim=1)

    # 评估 shape（与原逻辑）
    src_obj_obs_shape   = np.concatenate([src_probe[k] for k in params['src_env']['obj_obs_keys']]).shape
    tgt_obj_obs_shape   = np.concatenate([tgt_probe[k] for k in params['tgt_env']['obj_obs_keys']]).shape
    assert src_obj_obs_shape[0] == tgt_obj_obs_shape[0]

    # dims（与原逻辑）
    src_obs_dims = {
        'robot_obs_dim': src_robot_obs_shape[0],
        'obs_dim': src_robot_obs_shape[0] + src_obj_obs_shape[0],
        'lat_obs_dim': params['lat_obs_dim'],
        'obj_obs_dim': src_obj_obs_shape[0],
    }
    src_act_dims = {
        'act_dim': src_eval_env.action_space.shape[0],
        'lat_act_dim': params['lat_act_dim'],
    }
    tgt_obs_dims = {
        'robot_obs_dim': tgt_robot_obs_shape[0],
        'obs_dim': tgt_robot_obs_shape[0] + tgt_obj_obs_shape[0],
        'lat_obs_dim': params['lat_obs_dim'],
        'obj_obs_dim': tgt_obj_obs_shape[0],
    }
    tgt_act_dims = {
        'act_dim': tgt_eval_env.action_space.shape[0],
        'lat_act_dim': params['lat_act_dim'],
    }

    # ------------------------------
    # Agents
    # ------------------------------
    # === PATCH START: Agents + grip_head init/load ===
    # ------------------------------
    # Agents
    # ------------------------------
    src_agent = Agent(src_obs_dims, src_act_dims, device)
    src_agent.load(params['src_model_dir'])  # 从源目录加载（冻结）
    src_agent.freeze()

    tgt_agent = Agent(tgt_obs_dims, tgt_act_dims, device)

    # 读取 grip 配置
    grip_cfg = params.get("grip_head", {}) or {}
    grip_enabled      = bool(grip_cfg.get("enabled", False))
    grip_trainable    = bool(grip_cfg.get("trainable", False))
    grip_lr           = float(grip_cfg.get("lr", 1e-3))
    grip_w_src        = float(grip_cfg.get("w_src", 1.0))
    grip_w_tgt        = float(grip_cfg.get("w_tgt", 1.0))
    grip_load_from_src= bool(grip_cfg.get("load_from_src", False))
    grip_bs           = int(grip_cfg.get("batch_size", params['batch_size']))
    grip_dataset_max  = int(grip_cfg.get("dataset_max_steps", 0))  # 0 表示不用下采样

    # 目标侧先构建夹爪头（是否可训练由 trainable 控制）
    tgt_agent.init_grip_head(
        enabled=grip_enabled,
        trainable=grip_trainable,
        obj_dim=tgt_obs_dims['obj_obs_dim'],
        hidden=256, n_layers=3
    )
    if grip_enabled and grip_load_from_src:
        # 若源目录下有 grip_head.pt，则尝试严格加载；失败打印后跳过
        tgt_agent.try_load_grip_head(params['src_model_dir'])

    # 目标侧冷启动：actor 从源目录加载并冻结（内部强制 out_dim=lat_act_dim）
    tgt_agent.load_actor(params['src_model_dir'])

    # 若需要训练夹爪头，准备优化器
    grip_opt = None
    if grip_enabled and grip_trainable and (tgt_agent.grip_head is not None):
        grip_opt = torch.optim.Adam(tgt_agent.grip_head.parameters(), lr=grip_lr)
    # === PATCH END ===

    # ------------------------------
    # Replay buffers（注意训练 env 带 cond） 
    # ------------------------------
    src_buffer = replay_buffer.ReplayBuffer(
        obs_shape=src_env.observation_space.shape,
        action_shape=src_env.action_space.shape,
        capacity=int(1e7),
        batch_size=params['batch_size'],
        device=device
    )
    demo_paths = utils.load_episodes(pathlib.Path(params['src_buffer']),
                                     src_train_obs_keys)  # ← 带 cond
    src_buffer.add_rollouts(demo_paths)

    tgt_buffer = replay_buffer.ReplayBuffer(
        obs_shape=tgt_env.observation_space.shape,
        action_shape=tgt_env.action_space.shape,
        capacity=int(1e7),
        batch_size=params['batch_size'],
        device=device
    )
    demo_paths = utils.load_episodes(pathlib.Path(params['tgt_buffer']),
                                     tgt_train_obs_keys)  # ← 带 cond
    tgt_buffer.add_rollouts(demo_paths)
    
    
    # === PATCH START: build obj->grip BC datasets ===
    def _concat_keys_from_episode(ep: dict, keys: list) -> np.ndarray:
        """将 episode 中的多个键按顺序拼接（逐步），返回形状 (T, sum_dim)"""
        parts = [ep[k] for k in keys]
        parts = [p.reshape(p.shape[0], -1) for p in parts]
        return np.concatenate(parts, axis=1)

    def _extract_obs_and_actions(ep: dict, want_keys: list):
        """
        支持 4 种 episode 结构：
        A) {'observations': {k: (T, d_k), ...}, 'actions': (T, A)}
        B) {'obs':          {k: (T, d_k), ...}, 'actions'/'action'/'act': (T, A)}
        C) {'observations': (T, D) or 'obs': (T, D)}  +  obs_keys 在
        - ep['obs_keys'] 或 ep['observation_keys'] 或 ep['meta']['obs_keys']
        D) 扁平：{k: (T, d_k), ..., 'actions': (T, A)}  —— 所有观测键直接在顶层

        返回：
        obs_dict: dict(key -> (T, d_k))
        actions : (T, A)
        """
        # -------- 取 actions ----------
        actions = None
        for akey in ["actions", "action", "act"]:
            if akey in ep:
                actions = ep[akey]
                break
        if actions is None:
            # 有些导出把动作放在 ep['data']['actions'] 或类似
            data_blk = ep.get("data")
            if isinstance(data_blk, dict):
                for akey in ["actions", "action", "act"]:
                    if akey in data_blk:
                        actions = data_blk[akey]
                        break
        if actions is None:
            raise KeyError(f"episode missing actions; available top-level keys: {list(ep.keys())}")

        # -------- 取 observations ----------
        # 优先 A / B：字典
        for okey in ["observations", "obs"]:
            if okey in ep and isinstance(ep[okey], dict):
                return ep[okey], actions

        # D) 扁平：所有 want_keys 直接在顶层
        if all(k in ep for k in want_keys):
            obs_dict = {k: ep[k] for k in want_keys}
            return obs_dict, actions

        # C) 紧凑矩阵 + obs_keys
        def _from_compact(mat_key):
            mat = ep.get(mat_key, None)
            if mat is None or not isinstance(mat, (np.ndarray,)):
                return None
            # 尝试找到 obs_keys
            key_candidates = [
                ep.get("obs_keys"),
                ep.get("observation_keys"),
                (ep.get("meta") or {}).get("obs_keys") if isinstance(ep.get("meta"), dict) else None,
            ]
            obs_keys = None
            for cand in key_candidates:
                if isinstance(cand, (list, tuple)) and all(isinstance(x, str) for x in cand):
                    obs_keys = list(cand)
                    break
            if obs_keys is None:
                return None  # 无法映射列 -> 键名

            # 构建列切片：每个键一段（要求 utils.load_episodes 当时是按 keys 逐段拼的）
            # 如果没有每段维度信息，就假设每个 key 是 1~N 维且顺序与 obs_keys 一致，按“尽可能等分/累加长度”处理
            # 更可靠的做法：在 ep 里找 'obs_slices' / 'obs_shape_map'
            shape_map = ep.get("obs_shape_map") or ep.get("observation_shape_map")
            slices = {}
            if isinstance(shape_map, dict):
                offset = 0
                for k in obs_keys:
                    d = int(shape_map[k])
                    slices[k] = (offset, offset + d)
                    offset += d
            else:
                # fallback：等长或不等长无法确定 —— 只能放弃
                return None

            # 只取我们需要的 want_keys，按切片从矩阵里抠出来
            out = {}
            T = mat.shape[0]
            for k in want_keys:
                if k not in slices:
                    raise KeyError(f"compact obs has no key '{k}'. available keys = {obs_keys}")
                s, e = slices[k]
                out[k] = mat[:, s:e].reshape(T, e - s)
            return out

        compact = _from_compact("observations") or _from_compact("obs")
        if compact is not None:
            return compact, actions

        # 兜底：打印一次结构，方便你对照
        raise KeyError(
            "episode format not supported.\n"
            f"top-level keys = {list(ep.keys())}\n"
            "Expect one of: dict observations/obs; flat keys with all obj_keys; or compact obs + obs_keys/obs_shape_map."
        )


    def _build_obj_grip_dataset(demo_list: list, obj_keys: list, max_steps: int = 0):
        """
        从 demo_list 构建 (X, y):
        X: 拼接的 obj_obs，形状 (N, sum_dim)
        y: gripper 动作，形状 (N, 1)
        - 逐 episode 做步数对齐，自动处理 off-by-one：
            * T_obs == T_act + 1 -> 使用 obs[:-1] 与 actions[:]
            * T_act == T_obs + 1 -> 使用 obs[:] 与 actions[:-1]
            * 其他 -> 截断到 min(T_obs, T_act)
        - 若不同 obj_keys 的长度不一致，先统一到它们的最短长度。
        - 下采样使用无放回抽样，避免越界。
        """
        if not isinstance(demo_list, (list, tuple)) or len(demo_list) == 0:
            raise ValueError("demo_list 为空或类型不对")

        X_list, y_list = [], []
        warn_printed = 0

        for i, ep in enumerate(demo_list):
            # 取出 obs 字典和 actions（已兼容多种 episode 结构的 _extract_obs_and_actions）
            obs_dict, actions = _extract_obs_and_actions(ep, obj_keys)
            actions = np.asarray(actions)
            if actions.ndim == 1:
                actions = actions.reshape(-1, 1)
            T_act = actions.shape[0]

            # ---- 统一 obj_keys 的长度到其最短值 ----
            key_arrays = []
            T_obs_each = []
            for k in obj_keys:
                arr = np.asarray(obs_dict[k])
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                key_arrays.append(arr)
                T_obs_each.append(arr.shape[0])

            T_obs_keys_min = min(T_obs_each)
            if any(t != T_obs_keys_min for t in T_obs_each):
                if warn_printed < 3:
                    print(f"[obj->grip][warn] obj_keys steps mismatch in one ep: {T_obs_each} -> use min={T_obs_keys_min}")
                    warn_printed += 1
                # 截断每个 key 到最短长度
                key_arrays = [a[:T_obs_keys_min] for a in key_arrays]

            T_obs = T_obs_keys_min

            # ---- 与动作步数对齐（处理 off-by-one）----
            # 先决定本 episode 使用的有效步数与索引切片
            if T_obs == T_act:
                use_T = T_obs
                sl_obs = slice(0, use_T)
                sl_act = slice(0, use_T)
            elif T_obs == T_act + 1:
                # 观测多 1 帧：去掉最后一帧观测
                use_T = T_act
                sl_obs = slice(0, use_T)      # obs[:-1]
                sl_act = slice(0, use_T)      # act[:]
                if warn_printed < 3:
                    print(f"[obj->grip][info] obs one-longer than actions (obs={T_obs}, act={T_act}) -> drop last obs")
                    warn_printed += 1
            elif T_act == T_obs + 1:
                # 动作多 1 帧：去掉最后一帧动作
                use_T = T_obs
                sl_obs = slice(0, use_T)      # obs[:]
                sl_act = slice(0, use_T)      # act[:-1]
                if warn_printed < 3:
                    print(f"[obj->grip][info] actions one-longer than obs (obs={T_obs}, act={T_act}) -> drop last action")
                    warn_printed += 1
            else:
                # 差距更大：保守起见截断到 min
                use_T = min(T_obs, T_act)
                sl_obs = slice(0, use_T)
                sl_act = slice(0, use_T)
                if warn_printed < 3:
                    print(f"[obj->grip][warn] large step mismatch (obs={T_obs}, act={T_act}) -> truncate to {use_T}")
                    warn_printed += 1

            if use_T <= 0:
                # 安全跳过异常 episode
                continue

            # ---- 形成本 episode 的 X / y ----
            parts = [a[sl_obs].reshape(use_T, -1) for a in key_arrays]
            X = np.concatenate(parts, axis=1)                # (use_T, sum_dim)

            a = actions[sl_act].reshape(use_T, -1)           # (use_T, A)
            if a.shape[1] < 1:
                raise ValueError(f"actions last dim < 1, got {a.shape}")
            y = a[:, -1:].copy()                             # 取最后一维作为 gripper

            X_list.append(X)
            y_list.append(y)

        # ---- 跨 episode 拼接 ----
        if len(X_list) == 0:
            raise ValueError("未收集到任何样本，请检查数据/keys")

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        N = int(X.shape[0])
        if N != int(y.shape[0]):
            raise ValueError(f"X and y length mismatch after concat: X={N}, y={y.shape[0]}")
        if N == 0:
            raise ValueError("Empty dataset after concatenation")

        # ---- 防御式下采样（无放回，绝不越界）----
        if (max_steps is not None) and (max_steps > 0):
            keep = min(int(max_steps), N)
            if keep < N:
                idx = np.random.choice(N, size=keep, replace=False).astype(np.int64)
                X = X[idx]
                y = y[idx]

        return X.astype(np.float32), y.astype(np.float32)

    def _sample_obj_batch(X: np.ndarray, Y: np.ndarray, bs: int):
        idx = np.random.randint(0, X.shape[0], size=(bs,))
        xb = torch.as_tensor(X[idx], dtype=torch.float32)
        yb = torch.as_tensor(Y[idx], dtype=torch.float32)
        return xb, yb

    src_obj_keys = params['src_env']['obj_obs_keys']
    tgt_obj_keys = params['tgt_env']['obj_obs_keys']

    # 从各自 demo 构建 obj->grip 监督集（不影响 align 的 replay buffers）
    # 从各自 demo 构建 obj->grip 监督集（不影响 align 的 replay buffers）
    src_obj_eps = utils.load_episodes(pathlib.Path(params['src_buffer']), src_obj_keys)
    tgt_obj_eps = utils.load_episodes(pathlib.Path(params['tgt_buffer']), tgt_obj_keys)

    src_obj_X, src_grip_y = _build_obj_grip_dataset(
        src_obj_eps, src_obj_keys, max_steps=grip_dataset_max
    )
    tgt_obj_X, tgt_grip_y = _build_obj_grip_dataset(
        tgt_obj_eps, tgt_obj_keys, max_steps=grip_dataset_max
    )
    # === PATCH END ===    

    # ------------------------------
    # （更新）SEW 外来模型解析与严格校验（无需 config 指定 dof/slice）
    # ------------------------------
    sew_cfg = params.get("sew", None)  # 没有或 enabled=false -> Aligner 内部不启用 SEW
    if isinstance(sew_cfg, dict) and sew_cfg.get("enabled", False):
        src_n = int(src_robot_obs_shape[0] // 2)
        tgt_n = int(tgt_robot_obs_shape[0] // 2)
        exp_src = max(src_n - 1, 1)
        exp_tgt = max(tgt_n - 1, 1)

        def _check_predictor(path: str, side: str, expected_in_dim: int):
            state = _load_state_dict_flexible(path)
            arch = _infer_mlp_spec_from_state(state)
            print(f"[SEW][{side}] External predictor -> in_dim={arch['in_dim']}, depth={arch['depth']}, out_dims={arch['out_dims']}")
            if arch["in_dim"] != expected_in_dim:
                raise RuntimeError(
                    f"[SEW][{side}] 预测器输入维度不匹配：arch.in_dim={arch['in_dim']}，"
                    f"期望 in_dim = n-1 = {expected_in_dim}（n={expected_in_dim+1}）"
                )

        # 新式：分别校验 src/tgt
        src_pred = (sew_cfg.get("src_predictor") or {})
        tgt_pred = (sew_cfg.get("tgt_predictor") or {})
        if isinstance(src_pred, dict) and src_pred.get("path"):
            _check_predictor(src_pred["path"], "src", exp_src)
        if isinstance(tgt_pred, dict) and tgt_pred.get("path"):
            _check_predictor(tgt_pred["path"], "tgt", exp_tgt)

        # 旧式：单路径允许匹配任一侧
        single_path = ((sew_cfg.get("model") or {}).get("path") if isinstance(sew_cfg.get("model"), dict) else None)
        if single_path:
            state = _load_state_dict_flexible(single_path)
            arch = _infer_mlp_spec_from_state(state)
            ok_src = (arch["in_dim"] == exp_src)
            ok_tgt = (arch["in_dim"] == exp_tgt)
            print(f"[SEW][single] External predictor -> in_dim={arch['in_dim']}, depth={arch['depth']}, out_dims={arch['out_dims']}")
            if not (ok_src or ok_tgt):
                raise RuntimeError(
                    "[SEW] 外来模型输入维度不匹配："
                    f"arch.in_dim={arch['in_dim']}, src 期望={exp_src}（n={src_n}），tgt 期望={exp_tgt}（n={tgt_n}）"
                )
            use_side = "src" if ok_src else "tgt"
            print(f"[SEW][single] Input dim matches {use_side} side.")

    # ------------------------------
    # Aligner（新增：把 diffusion 与 cond 传入）
    # ------------------------------
    align_loss_cfg = params.get("align_loss", {}) or {}
    gan_cfg = (align_loss_cfg.get("gan") or {})
    diff_cfg = (align_loss_cfg.get("diffusion") or {})

    # diffusion_priors（来自预训练目录）
    diff_priors_cfg = params.get("diffusion_priors", {}) or {}
    # 打包传给 Aligner；Aligner 内部完成读取/校验/加载
    diffusion_bundle_cfg = {
        "priors": diff_priors_cfg,
        "enabled": bool(diff_cfg.get("enabled", False)),
        "lambda": {
            "lat": float(((diff_cfg.get("lambda") or {}).get("lat", 1.0))),
            "src": float(((diff_cfg.get("lambda") or {}).get("src", 1.0))),
            "tgt": float(((diff_cfg.get("lambda") or {}).get("tgt", 1.0))),
        },
        "schedule_override": {
            "use_override": bool(((diff_cfg.get("schedule_override") or {}).get("use_override", False))),
            "T": int(((diff_cfg.get("schedule_override") or {}).get("T", 128))),
            "beta": str(((diff_cfg.get("schedule_override") or {}).get("beta", "cosine"))),
            "loss_weight": str(((diff_cfg.get("schedule_override") or {}).get("loss_weight", "sigma2"))),
        },
        "cond_dim": int(cond_obs_dim),         # ← 观测先验用 obs 的 cond 维度
        "cond_keys": list(cond_obs_keys),      # （仅日志）
    }
    diff_act_priors_cfg = params.get("diffusion_priors_act", {}) or {}   # ← 补这行
    diff_act_cfg = (align_loss_cfg.get("diffusion_act") or {})    

    diffusion_act_bundle_cfg = {
        "priors": diff_act_priors_cfg,
        "enabled": bool(diff_act_cfg.get("enabled", False)),
        "lambda": {
            "lat": float(((diff_act_cfg.get("lambda") or {}).get("lat", 1.0))),
            "src": float(((diff_act_cfg.get("lambda") or {}).get("src", 1.0))),
            "tgt": float(((diff_act_cfg.get("lambda") or {}).get("tgt", 1.0))),
        },
        "schedule_override": {
            "use_override": bool(((diff_act_cfg.get("schedule_override") or {}).get("use_override", False))),
            "T": int(((diff_act_cfg.get("schedule_override") or {}).get("T", 128))),
            "beta": str(((diff_act_cfg.get("schedule_override") or {}).get("beta", "cosine"))),
            "loss_weight": str(((diff_act_cfg.get("schedule_override") or {}).get("loss_weight", "sigma2"))),
        },
        "cond_dim": int(cond_act_dim),         # ← 动作先验用 act 的 cond 维度
        "cond_keys": list(cond_act_keys),      # （仅日志）
    }

    # ============ NEW: Flow Matching Priors（来自预训练目录） ============
    flow_priors_cfg = params.get("flow_priors", {}) or {}
    flow_cfg = (align_loss_cfg.get("flow") or {})

    flow_bundle_cfg = {
        "priors": flow_priors_cfg,                          # e.g. {"dir": "...", "use_ema": true, ...}
        "enabled": bool(flow_cfg.get("enabled", False)),
        "lambda": {
            "lat": float(((flow_cfg.get("lambda") or {}).get("lat", 1.0))),
            "src": float(((flow_cfg.get("lambda") or {}).get("src", 1.0))),
            "tgt": float(((flow_cfg.get("lambda") or {}).get("tgt", 1.0))),
        },
        # Flow 的采样 / 加权策略（与 Diffusion 的 schedule 概念对应）
        "t_sampling": str(flow_cfg.get("t_sampling", "uniform")),   # uniform / logit_normal / ...
        "loss_weight": str(flow_cfg.get("loss_weight", "unit")),    # unit / sigma2 / ...
        "cond_dim": int(cond_obs_dim),            # obs 通道的 cond 维度
        "cond_keys": list(cond_obs_keys),
    }

    flow_act_priors_cfg = params.get("flow_priors_act", {}) or {}
    flow_act_cfg = (align_loss_cfg.get("flow_act") or {})

    flow_act_bundle_cfg = {
        "priors": flow_act_priors_cfg,
        "enabled": bool(flow_act_cfg.get("enabled", False)),
        "lambda": {
            "lat": float(((flow_act_cfg.get("lambda") or {}).get("lat", 1.0))),
            "src": float(((flow_act_cfg.get("lambda") or {}).get("src", 1.0))),
            "tgt": float(((flow_act_cfg.get("lambda") or {}).get("tgt", 1.0))),
        },
        "t_sampling": str(flow_act_cfg.get("t_sampling", "uniform")),
        "loss_weight": str(flow_act_cfg.get("loss_weight", "unit")),
        "cond_dim": int(cond_act_dim),            # act 通道的 cond 维度
        "cond_keys": list(cond_act_keys),
    }
    
    # ------- 现在四个 bundle 都有了，再写 overview + dump -------
    params['_bundles_overview'] = {
        "diffusion": {
            "enabled": diffusion_bundle_cfg["enabled"],
            "lambda": diffusion_bundle_cfg["lambda"],
            "cond_dim": diffusion_bundle_cfg["cond_dim"],
            "cond_keys": diffusion_bundle_cfg.get("cond_keys", []),
        },
        "diffusion_act": {
            "enabled": diffusion_act_bundle_cfg["enabled"],
            "lambda": diffusion_act_bundle_cfg["lambda"],
            "cond_dim": diffusion_act_bundle_cfg["cond_dim"],
            "cond_keys": diffusion_act_bundle_cfg.get("cond_keys", []),
        },
        "flow": {
            "enabled": flow_bundle_cfg["enabled"],
            "lambda": flow_bundle_cfg["lambda"],
            "cond_dim": flow_bundle_cfg["cond_dim"],
            "cond_keys": flow_bundle_cfg.get("cond_keys", []),
        },
        "flow_act": {
            "enabled": flow_act_bundle_cfg["enabled"],
            "lambda": flow_act_bundle_cfg["lambda"],
            "cond_dim": flow_act_bundle_cfg["cond_dim"],
            "cond_keys": flow_act_bundle_cfg.get("cond_keys", []),
        },
    }

    # dump params（此时包含 logdir、bundles_overview 等关键信息）
    with open(pathlib.Path(params['logdir']) / 'params.yml', 'w') as fp:
        pyyaml.safe_dump(params, fp, sort_keys=False)    
    
    aligner = Aligner(
        src_agent, tgt_agent, device,
        n_layers=params.get("n_layers", 3),
        hidden_dim=params.get("hidden_dim", 256),
        lr=params.get("lr", 3e-4),
        lmbd_gp=params.get("lmbd_gp", 10),
        log_freq=params.get("log_freq", 1000),
        sew_cfg=sew_cfg,
        loss_huber_delta=params.get("sew_huber_delta", None),
        # 判别器权重
        gan_lambda={
            "lat": float(((gan_cfg.get("lambda") or {}).get("lat", 1.0))),
            "src": float(((gan_cfg.get("lambda") or {}).get("src", 1.0))),
            "tgt": float(((gan_cfg.get("lambda") or {}).get("tgt", 1.0))),
            "enabled": bool(gan_cfg.get("enabled", True)),
        },
        # 观测通道先验（Diffusion + Flow）
        diffusion_bundle_cfg=diffusion_bundle_cfg,
        flow_bundle_cfg=flow_bundle_cfg,                     # ← NEW
        # 动作通道先验（Diffusion + Flow）
        diffusion_act_bundle_cfg=diffusion_act_bundle_cfg,
        flow_act_bundle_cfg=flow_act_bundle_cfg,             # ← NEW
        # 其他损失
        cycle_lambda=float(((align_loss_cfg.get("cycle") or {}).get("lambda", 10.0))),
        dynamics_lambda=float(((align_loss_cfg.get("dynamics") or {}).get("lambda", 10.0))),
    )

    # ------------------------------
    # 续训（与原逻辑相同）
    # ------------------------------
    start_step = 0
    resume_cfg = params.get("resume", {"enabled": False})
    if resume_cfg.get("enabled", False):
        resume_dir = pathlib.Path(resume_cfg["dir"])
        ckpt_dir = resume_dir if resume_dir.name.startswith("step_") else _find_latest_step_dir(resume_dir)
        if ckpt_dir is not None and ckpt_dir.exists():
            # 目标侧模块
            try:
                tgt_agent.load(ckpt_dir)
                print(f"[Resume] Loaded tgt agent modules from {ckpt_dir}")
            except Exception as e:
                print(f"[Resume] tgt_agent.load failed: {e}")

            # 推荐路径：align_state.pt
            align_state_path = ckpt_dir / "align_state.pt"
            if align_state_path.exists():
                pkg = torch.load(align_state_path, map_location=device)
                try:
                    aligner.load_state_dict(pkg["aligner"], strict=False)
                    start_step = int(pkg.get("step", 0)) + 1
                    print(f"[Resume] Loaded align_state.pt (start_step={start_step})")
                    # === PATCH START: resume grip_opt ===
                    opt_pkg = pkg.get("opt", None)
                    if (opt_pkg is not None) and (grip_enabled and grip_trainable and (grip_opt is not None)):
                        sd = opt_pkg.get("grip_opt", None)
                        if sd is not None:
                            try:
                                grip_opt.load_state_dict(sd)
                                print("[Resume] restored grip_opt state")
                            except Exception as e:
                                print(f"[Resume] grip_opt restore failed: {e}")
                    # === PATCH END ===                    
                    
                except Exception as e:
                    print(f"[Resume] align_state.pt load failed: {e}")
                    

            else:
                # 兼容路径：分文件权重 + optim（用于排查）
                def _try_load(m, fname):
                    p = ckpt_dir / fname
                    if p.exists():
                        sd = torch.load(p, map_location=device)
                        m.load_state_dict(sd, strict=False)
                        print(f"[Resume] loaded {fname}")
                _try_load(aligner.lat_disc, "lat_disc.pt")
                _try_load(aligner.src_disc, "src_disc.pt")
                _try_load(aligner.tgt_disc, "tgt_disc.pt")
                optim_path = ckpt_dir / "align_optim.pt"
                if optim_path.exists():
                    pkg = torch.load(optim_path, map_location=device)
                    opt_sd = pkg.get("optim", {})
                    for name, sd in opt_sd.items():
                        opt = getattr(aligner, name, None)
                        if opt is not None:
                            opt.load_state_dict(sd)
                            print(f"[Resume] restored optimizer: {name}")
                    try:
                        start_step = int(pkg.get("step", 0)) + 1
                    except Exception:
                        pass
                    print(f"[Resume] compat start_step = {start_step}")
        else:
            print(f"[Resume] No valid checkpoint found under {resume_dir}; start from 0.")

    # 初始评估（可选）
    if start_step == 0:
        eval_eps = params.get("evaluation", {}).get("episodes", 4)
        tgt_agent.eval_mode()
        utils.evaluate(tgt_eval_env, tgt_agent, eval_eps, logger, step=0)
        tgt_agent.train_mode()

    # ------------------------------
    # 训练循环（注意：buffer.obs 带 cond；aligner.update_* 仍喂 robot_obs）
    # ------------------------------
    total_steps = params['tgt_align_timesteps']
    end_step = start_step + total_steps

    # 用于从 obs 切片 robot_obs 与 cond
    def _split_obs(obs: torch.Tensor, robot_dim: int, cond_dim: int):
        if cond_dim > 0:
            robot = obs[:, :robot_dim]
            cond = obs[:, robot_dim:robot_dim + cond_dim]
        else:
            robot = obs
            cond = None
        return robot, cond

    for step in range(start_step, end_step):
        # 判别器：多步（与原逻辑一致，传 robot_obs）
        for _ in range(5):
            src_obs_full, src_act, _, src_next_obs_full, _ = src_buffer.sample()
            tgt_obs_full, tgt_act, _, tgt_next_obs_full, _ = tgt_buffer.sample()

            src_obs       = src_obs_full[:, :src_robot_len]
            src_next_obs  = src_next_obs_full[:, :src_robot_len]
            tgt_obs       = tgt_obs_full[:, :tgt_robot_len]
            tgt_next_obs  = tgt_next_obs_full[:, :tgt_robot_len]

            src_act = src_act[:, :-1]  # 去掉夹爪
            tgt_act = tgt_act[:, :-1]
            aligner.update_disc(src_obs, src_act, tgt_obs, tgt_act, logger, step)

        # 生成器：一步（与原逻辑一致 + 传 obs-cond 与 act-cond 给 diffusion）
        src_obs_full, src_act, _, src_next_obs_full, _ = src_buffer.sample()
        tgt_obs_full, tgt_act, _, tgt_next_obs_full, _ = tgt_buffer.sample()

        # robot 段
        src_obs = src_obs_full[:, :src_robot_len]
        src_next_obs = src_next_obs_full[:, :src_robot_len]
        tgt_obs = tgt_obs_full[:, :tgt_robot_len]
        tgt_next_obs = tgt_next_obs_full[:, :tgt_robot_len]

        # cond(obs)：按 cond_obs_keys 顺序拼出来
        c_src_obs = _gather_cond(src_obs_full, src_robot_len, src_cond_slices, cond_obs_keys)
        c_tgt_obs = _gather_cond(tgt_obs_full, tgt_robot_len, tgt_cond_slices, cond_obs_keys)

        # cond(act)：可能与 obs-cond 不同 key 列表
        c_src_act = _gather_cond(src_obs_full, src_robot_len, src_cond_slices, cond_act_keys)
        c_tgt_act = _gather_cond(tgt_obs_full, tgt_robot_len, tgt_cond_slices, cond_act_keys)

        # 去掉夹爪
        src_act = src_act[:, :-1]
        tgt_act = tgt_act[:, :-1]

        aligner.update_gen(src_obs, src_act, src_next_obs,
                           tgt_obs, tgt_act, tgt_next_obs,
                           logger, step,
                           c_src=c_src_obs, c_tgt=c_tgt_obs,
                           c_src_act=c_src_act, c_tgt_act=c_tgt_act)

        # PRINT_EVERY = 1000

        # for step in range(start_step, end_step):
        #     # ... 你的训练步骤 ...

        #     if step % PRINT_EVERY == 0:
        #         def short(t, k=5):
        #             return t[0, :k].detach().float().cpu().numpy() if t is not None else None

        #         def batch_stats(t):
        #             if t is None:
        #                 return None
        #             x = t.detach().float()
        #             mean = x.mean().item()
        #             var  = x.var(unbiased=False).item()  # 使用总体方差；若想样本方差设 unbiased=True
        #             std  = var ** 0.5
        #             return mean, std, var

        #         # 样本前几个维度
        #         print(f"\n[step {step}] samples (first 5 dims of the first item):")
        #         print("  src_obs[0][:5] =", short(src_obs))
        #         print("  c_src[0][:5]   =", short(c_src))
        #         print("  tgt_obs[0][:5] =", short(tgt_obs))
        #         print("  c_tgt[0][:5]   =", short(c_tgt))

        #         # 批次统计
        #         s_stats = batch_stats(src_obs)
        #         cs_stats = batch_stats(c_src)
        #         t_stats = batch_stats(tgt_obs)
        #         ct_stats = batch_stats(c_tgt)

        #         def fmt(name, st):
        #             if st is None:
        #                 print(f"  {name} mean/std/var = None")
        #             else:
        #                 m, s, v = st
        #                 print(f"  {name} mean/std/var = {m:.6f} / {s:.6f} / {v:.6f}")

        #         fmt("src_obs", s_stats)
        #         fmt("c_src  ", cs_stats)
        #         fmt("tgt_obs", t_stats)
        #         fmt("c_tgt  ", ct_stats)
        
        # === PATCH START: independent gripper BC step ===
        if (grip_enabled and grip_trainable and (tgt_agent.grip_head is not None) and (grip_opt is not None)):
            xb_s, yb_s = _sample_obj_batch(src_obj_X, src_grip_y, grip_bs)
            xb_t, yb_t = _sample_obj_batch(tgt_obj_X, tgt_grip_y, grip_bs)

            xb_s = xb_s.to(device); yb_s = yb_s.to(device)
            xb_t = xb_t.to(device); yb_t = yb_t.to(device)

            pred_s = tgt_agent.grip_head(xb_s)
            pred_t = tgt_agent.grip_head(xb_t)

            loss_s = F.mse_loss(pred_s, yb_s)
            loss_t = F.mse_loss(pred_t, yb_t)

            grip_loss = grip_w_src * loss_s + grip_w_tgt * loss_t
            grip_opt.zero_grad()
            grip_loss.backward()
            grip_opt.step()

            if step % params.get('log_freq', 1000) == 0:
                logger.add_scalar('train_grip/src_loss', loss_s.item(), step)
                logger.add_scalar('train_grip/tgt_loss', loss_t.item(), step)
                logger.add_scalar('train_grip/total', (loss_s + loss_t).item(), step)
        # === PATCH END ===        
      
        
        # 评估与保存（与原逻辑一致）
        if step % params['evaluation']['interval'] == 0:
            eval_eps = params.get("evaluation", {}).get("episodes", 4)
            tgt_agent.eval_mode()
            utils.evaluate(tgt_eval_env, tgt_agent, eval_eps, logger, step)
            
            # ======= 对齐度评估：跨域->解码->FK->世界坐标对比 =======
            with torch.no_grad():
                # 1) 从各自 buffer 采一批（只用 robot_obs），或直接复用一批生成器训练刚采的 batch
                # 各抽一批（用各自 buffer 的默认 batch_size）
                s_obs_full, _, _, _, _ = src_buffer.sample()
                t_obs_full, _, _, _, _ = tgt_buffer.sample()

                # 用实际返回的 batch 形状对齐 B（一般两边相等）
                B = min(s_obs_full.shape[0], t_obs_full.shape[0])
                if s_obs_full.shape[0] != B:
                    s_obs_full = s_obs_full[:B]
                if t_obs_full.shape[0] != B:
                    t_obs_full = t_obs_full[:B]
                # 切 robot_obs（你前面已有的函数）
                s_robot = s_obs_full[:, :src_robot_len]
                t_robot = t_obs_full[:, :tgt_robot_len]
                # 2) 跨域：tgt->src，src->tgt（潜空间 E/D）
                z_t = tgt_agent.obs_enc(t_robot)
                s_hat = src_agent.obs_dec(z_t)                # 形状：src 的 robot_obs
                z_s = src_agent.obs_enc(s_robot)
                t_hat = tgt_agent.obs_dec(z_s)                # 形状：tgt 的 robot_obs

                # 3) cos/sin -> q（只取“臂”的 dof；不含抓手）
                def _robot_obs_to_qnp(x: torch.Tensor, dof: int):
                    x = x.detach().cpu().numpy()
                    cosv = x[:, :dof]; sinv = x[:, dof:2*dof]
                    return np.arctan2(sinv, cosv).astype(np.float32)

                # 推断臂的 dof：robot_obs 的 cos/sin 对数
                src_arm_dof = src_robot_obs_shape[0] // 2
                tgt_arm_dof = tgt_robot_obs_shape[0] // 2
                q_src_hat   = _robot_obs_to_qnp(s_hat, src_arm_dof)   # (B, src_dof)
                q_tgt_hat   = _robot_obs_to_qnp(t_hat, tgt_arm_dof)
                q_src_real  = _robot_obs_to_qnp(s_robot, src_arm_dof)
                q_tgt_real  = _robot_obs_to_qnp(t_robot, tgt_arm_dof)

                # 4) 用“方法 A”：把 q 写进 probe env -> forward -> 观测里读 robot0_eef_pos
                #    （逐样本 FK；频率低，评估期可接受）
                p_src_from_tgt = []
                p_src_real     = []
                for i in range(q_src_hat.shape[0]):
                    _set_qpos_and_forward(src_probe_env, q_src_hat[i], robot_id=0)
                    p_src_from_tgt.append(_read_eef_world_pos(src_probe_env, robot_id=0, obs_key="robot0_eef_pos"))
                    _set_qpos_and_forward(src_probe_env, q_src_real[i], robot_id=0)
                    p_src_real.append(_read_eef_world_pos(src_probe_env, robot_id=0, obs_key="robot0_eef_pos"))
                p_src_from_tgt = np.asarray(p_src_from_tgt, dtype=float)  # (B,3)
                p_src_real     = np.asarray(p_src_real, dtype=float)

                p_tgt_from_src = []
                p_tgt_real     = []
                for i in range(q_tgt_hat.shape[0]):
                    _set_qpos_and_forward(tgt_probe_env, q_tgt_hat[i], robot_id=0)
                    p_tgt_from_src.append(_read_eef_world_pos(tgt_probe_env, robot_id=0, obs_key="robot0_eef_pos"))
                    _set_qpos_and_forward(tgt_probe_env, q_tgt_real[i], robot_id=0)
                    p_tgt_real.append(_read_eef_world_pos(tgt_probe_env, robot_id=0, obs_key="robot0_eef_pos"))
                p_tgt_from_src = np.asarray(p_tgt_from_src, dtype=float)
                p_tgt_real     = np.asarray(p_tgt_real, dtype=float)

                # 5) 误差：世界系直接对比
                err_tgt2src = np.linalg.norm(p_src_from_tgt - p_tgt_real, axis=1)   # (B,)
                err_src2tgt = np.linalg.norm(p_tgt_from_src - p_src_real, axis=1)

                # 6) 写 TensorBoard
                logger.add_scalar('align_eval/tgt2src_l2_mean', float(err_tgt2src.mean()), step)
                logger.add_scalar('align_eval/tgt2src_l2_std',  float(err_tgt2src.std()),  step)
                logger.add_scalar('align_eval/src2tgt_l2_mean', float(err_src2tgt.mean()), step)
                logger.add_scalar('align_eval/src2tgt_l2_std',  float(err_src2tgt.std()),  step)            

            # ======= 速度对齐评估：基于雅可比的线/角速度一致性 =======
            # 思路：
            #   src->tgt：把 (s_robot, s_act) 经跨域得到 (t_hat, a_t_hat)，
            #              用 tgt 雅可比算 (v_hat_t, w_hat_t)；
            #              对照是 src 自身 (v_src, w_src)。
            #   tgt->src：对称操作。
            with torch.no_grad():
                # --- 重新采一批，确保拿到 action ---
                s_obs_full, s_act_full, _, _, _ = src_buffer.sample()
                t_obs_full, t_act_full, _, _, _ = tgt_buffer.sample()

                # 统一 batch 尺寸
                B = min(s_obs_full.shape[0], t_obs_full.shape[0])
                s_obs_full = s_obs_full[:B]; s_act_full = s_act_full[:B]
                t_obs_full = t_obs_full[:B]; t_act_full = t_act_full[:B]

                # 切 robot 段与动作（去夹爪）
                s_robot = s_obs_full[:, :src_robot_obs_shape[0]]
                t_robot = t_obs_full[:, :tgt_robot_obs_shape[0]]
                s_act   = s_act_full[:, :-1]
                t_act   = t_act_full[:, :-1]

                # ---------- src -> tgt ----------
                # 观测跨域
                z_s   = src_agent.obs_enc(s_robot)
                t_hat = tgt_agent.obs_dec(z_s)                 # tgt 侧 robot_obs
                # 动作跨域（lat_act 通过 src_act_enc -> tgt_act_dec）
                z_as      = src_agent.act_enc(torch.cat([s_robot, s_act], dim=-1))
                a_t_hat   = tgt_agent.act_dec(torch.cat([t_hat, z_as], dim=-1))

                # 逐样本：J @ a
                src2tgt_v_err = []
                src2tgt_w_err = []
                # 推断各臂 DOF
                src_arm_dof = src_robot_obs_shape[0] // 2
                tgt_arm_dof = tgt_robot_obs_shape[0] // 2

                def _robot_obs_to_qnp(x: torch.Tensor, dof: int):
                    x = x.detach().cpu().numpy()
                    cosv = x[:, :dof]; sinv = x[:, dof:2*dof]
                    return np.arctan2(sinv, cosv).astype(np.float32)

                q_s_real  = _robot_obs_to_qnp(s_robot, src_arm_dof)
                q_t_hat   = _robot_obs_to_qnp(t_hat,  tgt_arm_dof)

                a_s_np    = s_act.detach().cpu().numpy().astype(np.float32)
                a_t_hat_np= a_t_hat.detach().cpu().numpy().astype(np.float32)

                for i in range(B):
                    # tgt 端预测速度
                    Jp_t, Jr_t = _eef_jacobians(tgt_probe_env, q_t_hat[i], robot_id=0)
                    v_hat_t = (Jp_t @ a_t_hat_np[i])
                    w_hat_t = (Jr_t @ a_t_hat_np[i])

                    # src 端对照速度（源域自身）
                    Jp_s, Jr_s = _eef_jacobians(src_probe_env, q_s_real[i], robot_id=0)
                    v_src = (Jp_s @ a_s_np[i])
                    w_src = (Jr_s @ a_s_np[i])

                    src2tgt_v_err.append(np.linalg.norm(v_hat_t - v_src))
                    src2tgt_w_err.append(np.linalg.norm(w_hat_t - w_src))

                src2tgt_v_err = np.asarray(src2tgt_v_err, dtype=np.float32)
                src2tgt_w_err = np.asarray(src2tgt_w_err, dtype=np.float32)

                logger.add_scalar('align_eval/vel_lin_src2tgt_l2_mean', float(src2tgt_v_err.mean()), step)
                logger.add_scalar('align_eval/vel_lin_src2tgt_l2_std',  float(src2tgt_v_err.std()),  step)
                logger.add_scalar('align_eval/vel_ang_src2tgt_l2_mean', float(src2tgt_w_err.mean()), step)
                logger.add_scalar('align_eval/vel_ang_src2tgt_l2_std',  float(src2tgt_w_err.std()),  step)

                # ---------- tgt -> src ----------
                z_t   = tgt_agent.obs_enc(t_robot)
                s_hat = src_agent.obs_dec(z_t)                 # src 侧 robot_obs
                z_at      = tgt_agent.act_enc(torch.cat([t_robot, t_act], dim=-1))
                a_s_hat   = src_agent.act_dec(torch.cat([s_hat, z_at], dim=-1))

                q_t_real  = _robot_obs_to_qnp(t_robot, tgt_arm_dof)
                q_s_hat   = _robot_obs_to_qnp(s_hat,  src_arm_dof)

                a_t_np    = t_act.detach().cpu().numpy().astype(np.float32)
                a_s_hat_np= a_s_hat.detach().cpu().numpy().astype(np.float32)

                tgt2src_v_err = []
                tgt2src_w_err = []

                for i in range(B):
                    # src 端预测速度
                    Jp_s, Jr_s = _eef_jacobians(src_probe_env, q_s_hat[i], robot_id=0)
                    v_hat_s = (Jp_s @ a_s_hat_np[i])
                    w_hat_s = (Jr_s @ a_s_hat_np[i])

                    # tgt 端对照速度（目标域自身）
                    Jp_t, Jr_t = _eef_jacobians(tgt_probe_env, q_t_real[i], robot_id=0)
                    v_tgt = (Jp_t @ a_t_np[i])
                    w_tgt = (Jr_t @ a_t_np[i])

                    tgt2src_v_err.append(np.linalg.norm(v_hat_s - v_tgt))
                    tgt2src_w_err.append(np.linalg.norm(w_hat_s - w_tgt))

                tgt2src_v_err = np.asarray(tgt2src_v_err, dtype=np.float32)
                tgt2src_w_err = np.asarray(tgt2src_w_err, dtype=np.float32)

                logger.add_scalar('align_eval/vel_lin_tgt2src_l2_mean', float(tgt2src_v_err.mean()), step)
                logger.add_scalar('align_eval/vel_lin_tgt2src_l2_std',  float(tgt2src_v_err.std()),  step)
                logger.add_scalar('align_eval/vel_ang_tgt2src_l2_mean', float(tgt2src_w_err.mean()), step)
                logger.add_scalar('align_eval/vel_ang_tgt2src_l2_std',  float(tgt2src_w_err.std()),  step)            
            
            tgt_agent.train_mode()

            print(f"[Save] step {step}")
            step_dir = model_dir / f"step_{step:07d}"
            step_dir.mkdir(parents=True, exist_ok=True)

            # 1) 目标侧模块（enc/dec/act_enc/dec 等）：per-file 保存（与 Agent.load 对齐）
            tgt_agent.save(step_dir)

            # 2) 推荐：aligner 的整体训练态（含优化器）—— 真正一键续训
            # === PATCH START: save grip_opt into align_state ===
            align_state = {
                "aligner": aligner.state_dict(),
                "step": step,
            }
            if (grip_enabled and grip_trainable and (grip_opt is not None)):
                align_state.setdefault("opt", {})["grip_opt"] = grip_opt.state_dict()
            torch.save(align_state, step_dir / "align_state.pt")
            # === PATCH END ===
            # 3) 可选分文件（不改）
            if params.get("save_extra_split", False):
                torch.save(aligner.lat_disc.state_dict(), step_dir / "lat_disc.pt")
                torch.save(aligner.src_disc.state_dict(), step_dir / "src_disc.pt")
                torch.save(aligner.tgt_disc.state_dict(), step_dir / "tgt_disc.pt")
                optim_state = {}
                for name in [
                    "tgt_obs_enc_opt", "tgt_obs_dec_opt",
                    "lat_disc_opt", "src_disc_opt", "tgt_disc_opt",
                    "tgt_act_enc_opt", "tgt_act_dec_opt",
                ]:
                    opt = getattr(aligner, name, None)
                    if opt is not None:
                        optim_state[name] = opt.state_dict()
                torch.save({"optim": optim_state, "step": step}, step_dir / "align_optim.pt")

    logger.close()


if __name__ == '__main__':
    main()