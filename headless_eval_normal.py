# eval_unified.py
# 统一评测脚本（cart / joint）
# - cart 分支：x_task = robot_obs_keys + obj_obs_keys；另外仅用于还原 q 的 aux_joint_keys 不进网络
# - joint 分支：输入网络的是完整 obs（robot_obs_keys + obj_obs_keys），不需要 aux_joint_keys

import pathlib
import argparse
import json
import time
import gc
import warnings
import random

import numpy as np
import torch
from ruamel.yaml import YAML
import imageio

# 两类 Agent
from bc_cart import CartSEWAugAgent as CartAgent
from bc_jointspace import BCObsActAgent   as JointAgent

import utils


# ----------------- 小工具 -----------------
def _get(d, k, default):
    return d[k] if isinstance(d, dict) and k in d else default

def _ensure_offscreen_context(sim):
    # 适配 robosuite 的 offscreen 渲染
    from robosuite.utils.binding_utils import MjRenderContextOffscreen
    if getattr(sim, "_render_context_offscreen", None) is not None:
        return
    try:
        sim._render_context_offscreen = MjRenderContextOffscreen(sim, device_id=0)
    except TypeError:
        sim._render_context_offscreen = MjRenderContextOffscreen(sim)

def _auto_pick_camera(env, params):
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

def _q_from_obs(d, aux_joint_keys):
    """
    只给 cart 用：优先用 robot0_joint_pos；否则从 cos/sin 还原。
    aux_joint_keys 只在 cart 分支传入，不会在 joint 分支调用。
    """
    if "robot0_joint_pos" in d:
        return np.asarray(d["robot0_joint_pos"], dtype=np.float32).reshape(-1)
    # 在 aux_joint_keys 里找 cos/sin
    cos_key = next((k for k in aux_joint_keys if k.endswith("_cos") and "joint_pos" in k), None)
    sin_key = next((k for k in aux_joint_keys if k.endswith("_sin") and "joint_pos" in k), None)
    if cos_key and sin_key and (cos_key in d) and (sin_key in d):
        cosv = np.asarray(d[cos_key]).reshape(-1)
        sinv = np.asarray(d[sin_key]).reshape(-1)
        return np.arctan2(sinv, cosv).astype(np.float32)
    raise KeyError("无法还原 q：需要 robot0_joint_pos 或 joint_pos_cos/sin")

def _x_task_from_keys(d, x_task_keys):
    parts = [np.asarray(d[k], dtype=np.float32).reshape(-1) for k in x_task_keys]
    return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float32)

def _set_global_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _detect_model_type(model_dir_str: str, override: str = None):
    """
    依据 model_dir 文本或覆盖参数判断：
      - 'cart'：笛卡尔
      - 'joint'：关节
    """
    if override:
        ov = override.strip().lower()
        if ov in ('cart', 'joint'):
            return ov
    s = model_dir_str.upper()
    if 'CART' in s:
        return 'cart'
    if 'JOINT' in s:
        return 'joint'
    return 'joint'  # 兜底


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

    # ===== 先判断模型类型（决定是否使用 aux_joint_keys）=====
    model_dir = pathlib.Path(params['model_dir']).resolve()
    model_type = _detect_model_type(str(model_dir), params.get('model_type', None))
    print(f"[Eval] model_dir={model_dir}")
    print(f"[Eval] detected model_type={model_type}")

    # ===== 组装 obs_keys_env =====
    robot_obs_keys = params['robot_obs_keys']                 # e.g. ['robot0_eef_pos', 'robot0_eef_quat']
    obj_obs_keys   = params.get('obj_obs_keys', [])           # e.g. ['target_to_robot0_eef_pos', 'target_pos']

    if model_type == 'cart':
        # cart：actor 输入是 x_task_keys（机器人末端位姿 + 目标等）
        x_task_keys    = robot_obs_keys + obj_obs_keys
        aux_joint_keys = params.get('aux_joint_keys', [])     # 仅用于还原 q，不进入 actor
        # 环境/数据真正需要的键：x_task + aux_joint（去重保序）
        seen = set()
        obs_keys_env = [k for k in (x_task_keys + aux_joint_keys) if not (k in seen or seen.add(k))]
    else:
        # joint：没有 aux_joint_keys，actor 输入就是完整 obs
        x_task_keys    = robot_obs_keys + obj_obs_keys
        aux_joint_keys = []
        seen = set()
        obs_keys_env = [k for k in (robot_obs_keys + obj_obs_keys) if not (k in seen or seen.add(k))]

    # ===== 观测维度探针 =====
    env_probe = utils.make_robosuite_env(
        params['env_name'],
        robots=params['robots'],
        controller_type=params['controller_type'],
        **params['env_kwargs'],
    )
    key_shapes = _infer_key_shapes(env_probe, obs_keys_env)

    # 供统计/agent 构造使用的维度
    obs_dim_cart_in = sum(key_shapes.get(k, 0) for k in x_task_keys)        # 仅 cart actor 的输入维度
    obs_dim_joint   = sum(key_shapes.get(k, 0) for k in (robot_obs_keys + obj_obs_keys))  # joint actor 输入维度

    if model_type == 'cart':
        # 推断臂 DoF（优先 robot0_joint_pos，否则 cos/sin）
        if "robot0_joint_pos" in key_shapes:
            dof_arm = key_shapes["robot0_joint_pos"]
        elif ("robot0_joint_pos_cos" in key_shapes) and ("robot0_joint_pos_sin" in key_shapes):
            dof_arm = key_shapes["robot0_joint_pos_cos"]
        else:
            raise KeyError("cart 模型需要 robot0_joint_pos 或 robot0_joint_pos_cos/sin 之一以还原 q")
    else:
        dof_arm = None  # joint 分支不使用

    # ===== 创建评测环境（按 obs_keys_env）=====
    seed_eval = int(params.get('seed', 42)) + 100
    _set_global_seeds(seed_eval)
    env = utils.make(
        params['env_name'],
        robots=params['robots'],
        controller_type=params['controller_type'],
        obs_keys=obs_keys_env,
        seed=seed_eval,
        render=False,
        **params['env_kwargs'],
    )

    # 渲染参数
    camera_name = _auto_pick_camera(env, params)
    vcfg   = params.get('video', {}) if isinstance(params.get('video', {}), dict) else {}
    width  = int(_get(vcfg, 'width', 640))
    height = int(_get(vcfg, 'height', 480))
    fps    = int(_get(vcfg, 'fps', 20))

    # ===== 构造并加载 Agent + policy_step =====
    if model_type == 'cart':
        # --- 笛卡尔 agent（输入 x_task，动作通过 J⁺ 映射到关节，最后拼抓手）---
        # 评测只需前向；用最小参数集合 + 兼容回退，防止不同版本构造函数不匹配
        try:
            agent = CartAgent(
                x_task_dim=obs_dim_cart_in,
                dof_arm=dof_arm,
                device=device,
                n_layers=params.get('actor_n_layers', params.get('n_layers', 3)),
                hidden_dim=params.get('actor_hidden_dim', params.get('hidden_dim', 256)),
                lr=params.get('lr', 3e-4),
            )
        except TypeError:
            # 进一步精简（极简构造）
            agent = CartAgent(
                x_task_dim=obs_dim_cart_in,
                dof_arm=dof_arm,
                device=device,
            )

        agent.load(model_dir)
        # 在线几何：attach 原始 env（utils.make 可能包了 wrapper）
        agent.attach_env(env._env) if hasattr(env, "_env") else agent.attach_env(env)

        def policy_step(obs_vec):
            d = _split_from_concat(obs_keys_env, obs_vec, key_shapes)
            q = _q_from_obs(d, aux_joint_keys)               # 仅 cart 用
            x_task = _x_task_from_keys(d, x_task_keys)       # e.g. eef_pos + eef_quat + targets ...
            act = agent.infer_action(x_task, q)              # (n+1,)
            return np.clip(act, -1.0, 1.0).astype(np.float32)

        reset_smoothing = hasattr(agent, "reset_eval_smoothing")

    else:
        # --- 关节空间 agent（输入完整 obs，输出 joints raw + grip logit（内部处理））---
        act_dim = env.action_space.shape[0]
        obs_dims = {
            'obs_dim':       obs_dim_joint,
            'robot_obs_dim': sum(key_shapes.get(k, 0) for k in robot_obs_keys),
            'obj_obs_dim':   sum(key_shapes.get(k, 0) for k in obj_obs_keys),
            'lat_obs_dim':   params.get('lat_obs_dim', 0),
        }
        act_dims = {
            'act_dim': act_dim,
            'lat_act_dim': params.get('lat_act_dim', 0),
        }
        agent = JointAgent(
            obs_dims, act_dims, device,
            n_layers=params.get('n_layers', 3),
            hidden_dim=params.get('hidden_dim', 256),
            lr=params.get('lr', 3e-4),
            actor_n_layers=params.get('actor_n_layers'),
            actor_hidden_dim=params.get('actor_hidden_dim'),
            grip_loss_weight=params.get('grip_loss_weight', 5.0),
            label_smoothing=params.get('label_smoothing', 0.0),
            auto_balance_gripper=params.get('auto_balance_gripper', True),
            use_action_ema=params.get('use_action_ema', True),
            ema_alpha=params.get('ema_alpha', 0.8),
            use_grip_hysteresis=params.get('use_grip_hysteresis', True),
            grip_prob_high=params.get('grip_prob_high', 0.70),
            grip_prob_low=params.get('grip_prob_low', 0.30),
            grip_cooldown_steps=params.get('grip_cooldown_steps', 4),
            enable_input_norm=params.get('enable_input_norm', False),
            huber_delta=params.get('huber_delta', 1.0),
        )
        agent.load(model_dir)

        def policy_step(obs_vec):
            return agent.sample_action(obs_vec, deterministic=True)

        reset_smoothing = hasattr(agent, "reset_eval_smoothing")

    # ===== 结果目录 & 元信息 =====
    run_dir = (model_dir.parent / f"eval_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_type": model_type,
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
    }
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ===== 循环评测 + 录像 + 统计 =====
    returns, global_actions = [], []
    num_episodes = int(params.get('num_episodes', 10))

    try:
        for i in range(num_episodes):
            obs = env.reset()
            _ensure_offscreen_context(env.sim)
            if reset_smoothing:
                try: agent.reset_eval_smoothing()
                except Exception: pass

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

            while not done:
                action = policy_step(obs)
                ep_actions.append(action.copy())
                obs, reward, done, _ = env.step(action)
                ep_ret += float(reward)
                writer.append_data(grab())

            writer.close()

            # 保存动作与统计
            acts = np.asarray(ep_actions, dtype=np.float32)  # [T, act_dim]
            np.save(ep_dir / "actions.npy", acts)

            act_min = acts.min(axis=0).tolist()
            act_max = acts.max(axis=0).tolist()
            act_mean = acts.mean(axis=0).tolist()
            act_std  = acts.std(axis=0).tolist()

            thr_hi = 0.98; thr_09 = 0.90
            abs_acts = np.abs(acts)
            sat_hi_overall = float((abs_acts >= thr_hi).sum() / abs_acts.size)
            sat_hi_per_dim = (abs_acts >= thr_hi).sum(axis=0) / acts.shape[0]
            sat_09_overall = float((abs_acts >= thr_09).sum() / abs_acts.size)
            sat_09_per_dim = (abs_acts >= thr_09).sum(axis=0) / acts.shape[0]

            bins = np.linspace(-1.0, 1.0, 41, dtype=np.float32)
            hist_counts = []
            for d in range(acts.shape[1]):
                h, _ = np.histogram(acts[:, d], bins=bins)
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

            grip_idx = acts.shape[1] - 1
            print(
                f"[EvalStats ep {i:03d}] "
                f"ret={ep_ret:.1f} | "
                f"|a|≥0.90 overall={sat_09_overall*100:.1f}% (grip={sat_09_per_dim[grip_idx]*100:.1f}%) | "
                f"|a|≥0.98 overall={sat_hi_overall*100:.1f}% (grip={sat_hi_per_dim[grip_idx]*100:.1f}%) | "
                f"min={np.round(acts.min(0),3)} | max={np.round(acts.max(0),3)}"
            )

            global_actions.append(acts)

            final_name = f"ep_{i:03d}_return_{ep_ret:.1f}.mp4"
            (ep_dir / final_name).write_bytes(tmp_mp4.read_bytes())
            tmp_mp4.unlink()

            with open(ep_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump({"episode": i, "return": float(ep_ret), "video": final_name, "model_type": model_type},
                          f, ensure_ascii=False, indent=2)

            returns.append(ep_ret)
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
        # 清理
        try:
            if getattr(env.sim, "_render_context_offscreen", None) is not None:
                try:
                    env.sim._render_context_offscreen.gl_ctx.free()
                except Exception:
                    pass
                env.sim._render_context_offscreen = None
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
        del env
        gc.collect()

    if returns:
        print(f"[Eval] avg return over {len(returns)} eps : {np.mean(returns):.2f} ± {np.std(returns):.2f}")


if __name__ == '__main__':
    main()