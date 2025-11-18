# headless_eval_cartsew.py
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

import utils
from bc_cartsew_aug import CartSEWAugAgent as Agent  # 使用结构化 Agent


# ----------------- 与训练侧保持一致的小工具 -----------------
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
    """
    取相机名的优先级：
    1) params['evaluation']['video']['render_camera']
    2) params['video']['render_camera']
    3) params['render_camera']   # 顶层（你现在的配置）
    4) 自动从 env.camera_names 里按偏好挑一个
    """
    # 1) evaluation.video.render_camera
    if isinstance(params.get('evaluation'), dict):
        vcfg = params['evaluation'].get('video', {})
        if isinstance(vcfg, dict) and vcfg.get('render_camera'):
            return vcfg['render_camera']

    # 2) video.render_camera
    if isinstance(params.get('video'), dict) and params['video'].get('render_camera'):
        return params['video']['render_camera']

    # 3) 顶层 render_camera
    if params.get('render_camera'):
        return params['render_camera']

    # 4) 自动选择
    names = list(getattr(env, "camera_names", []) or [])
    prefs = ['frontview', 'agentview', 'birdview', 'sideview', 'topview']
    for n in prefs:
        if n in names:
            return n
    return names[0] if names else None

def _infer_key_shapes(env_probe, obs_keys):
    raw = env_probe.reset()  # dict
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

def _q_from_cos_sin(d):
    cosv = np.asarray(d["robot0_joint_pos_cos"]).reshape(-1)
    sinv = np.asarray(d["robot0_joint_pos_sin"]).reshape(-1)
    return np.arctan2(sinv, cosv).astype(np.float32)

def _x_task_from_obj(d, obj_keys):
    parts = []
    for k in obj_keys:
        if k in d:
            parts.append(np.asarray(d[k], dtype=np.float32).reshape(-1))
    if parts:
        return np.concatenate(parts, axis=0)
    return np.zeros((0,), dtype=np.float32)

def _set_global_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------- 主程序 -----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='eval config file path (与训练相同格式)')
    return p.parse_args()

def main():
    warnings.filterwarnings("ignore", message="Box bound precision lowered by casting to float32")

    args = parse_args()
    yaml = YAML(typ='safe')
    params = yaml.load(open(args.config, 'r'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ===== 1) 观测维度探针 =====
    env_probe = utils.make_robosuite_env(
        params['env_name'],
        robots=params['robots'],
        controller_type=params['controller_type'],
        **params['env_kwargs'],
    )
    obs0 = env_probe.reset()
    obs_keys = params['robot_obs_keys'] + params['obj_obs_keys']
    key_shapes = _infer_key_shapes(env_probe, obs_keys)

    robot_obs_shape = np.concatenate([obs0[k] for k in params['robot_obs_keys']]).shape
    obj_obs_shape   = np.concatenate([obs0[k] for k in params['obj_obs_keys']]).shape

    # ===== 2) 评测环境 =====
    seed_eval = int(params.get('seed', 42)) + 100
    _set_global_seeds(seed_eval)
    env = utils.make(
        params['env_name'],
        robots=params['robots'],
        controller_type=params['controller_type'],
        obs_keys=obs_keys,
        seed=seed_eval,
        render=False,
        **params['env_kwargs'],
    )

    # ===== 3) 渲染参数（严格从 evaluation.video 读取） =====
    evaluation_cfg = params.get('evaluation', {}) if isinstance(params.get('evaluation', {}), dict) else {}
    vcfg   = evaluation_cfg.get('video', {}) if isinstance(evaluation_cfg.get('video', {}), dict) else {}
    enable_video = bool(_get(vcfg, 'enable', True))
    camera_name  = _auto_pick_camera(env, params)
    width  = int(_get(vcfg, 'width', 640))
    height = int(_get(vcfg, 'height', 480))
    fps    = int(_get(vcfg, 'fps', 20))

    # ===== 4) Agent：接口严格匹配你当前的 Agent 定义 =====
    x_task_dim = obj_obs_shape[0]
    dof_arm = obs0['robot0_joint_pos_cos'].shape[0]
    inv_cfg = params.get('inverse', {}) if isinstance(params.get('inverse', {}), dict) else {}
    feat    = params.get('feature', {}) if isinstance(params.get('feature', {}), dict) else {}

    agent = Agent(
        x_task_dim=x_task_dim,
        dof_arm=dof_arm,
        device=device,
        n_layers=params.get('actor_n_layers', params.get('n_layers', 3)),
        hidden_dim=params.get('actor_hidden_dim', params.get('hidden_dim', 256)),
        lr=params.get('lr', 3e-4),
        dls_lambda=float(inv_cfg.get('dls_lambda', 1e-3)),
        lam_max=float(inv_cfg.get('lam_max', 1e-1)),
        cond_thresh=float(inv_cfg.get('cond_thresh', 50.0)),
        E_idx=int(feat.get('E_idx', 4)),
        W_idx=int(feat.get('W_idx', 6)),
        huber_delta=float(params.get('huber_delta', 1.0)),
        w_grip=float(params.get('w_grip', 1.0)),
        grad_clip=float(params.get('grad_clip', 1.0)),
        use_batched_torch=bool(inv_cfg.get('use_batched_torch', False)),
    )

    # 载入权重
    model_dir = pathlib.Path(params['model_dir']).resolve()
    agent.load(model_dir)

    # 在线几何需要 robosuite 原生 env
    agent.attach_env(env._env) if hasattr(env, "_env") else agent.attach_env(env)

    #（可选）把 probe 的初始 q 作为 neutral，用于你后续加的护栏
    if hasattr(agent, "set_q_neutral"):
        try:
            q0 = _q_from_cos_sin({
                "robot0_joint_pos_cos": obs0["robot0_joint_pos_cos"],
                "robot0_joint_pos_sin": obs0["robot0_joint_pos_sin"],
            })
            agent.set_q_neutral(q0)
        except Exception:
            pass

    # ===== 5) 结果目录 & 元信息 =====
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
    }
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ===== 6) 循环评测 + 录像 =====
    num_episodes = int(params.get('num_episodes', 10))
    returns = []

    try:
        for i in range(num_episodes):
            obs = env.reset()
            if enable_video:
                _ensure_offscreen_context(env.sim)

            # 若 Agent 有平滑复位接口则调用（你新 Agent 里是有的）
            if hasattr(agent, "reset_eval_smoothing"):
                agent.reset_eval_smoothing()

            done = False
            ep_ret = 0.0

            if enable_video:
                ep_dir = run_dir / f"ep_{i:03d}"
                ep_dir.mkdir(parents=True, exist_ok=True)
                tmp_mp4 = ep_dir / f"ep_{i:03d}_tmp.mp4"
                writer = imageio.get_writer(tmp_mp4.as_posix(), fps=fps)

                def grab():
                    frame = env.sim.render(camera_name=camera_name, width=width, height=height)
                    return np.flipud(frame)
                writer.append_data(grab())

            while not done:
                # 完全按训练侧拆观测
                d = _split_from_concat(obs_keys, obs, key_shapes)
                q = _q_from_cos_sin(d)                           # (n,)
                x_task = _x_task_from_obj(d, params['obj_obs_keys'])  # (Dx,)

                act = agent.infer_action(x_task, q)              # (n+1,)
                act = np.clip(act, -1.0, 1.0).astype(np.float32)

                obs, reward, done, _ = env.step(act)
                ep_ret += float(reward)

                if enable_video:
                    writer.append_data(grab())

            if enable_video:
                writer.close()
                final_name = f"ep_{i:03d}_return_{ep_ret:.1f}.mp4"
                (ep_dir / final_name).write_bytes(tmp_mp4.read_bytes())
                tmp_mp4.unlink()
                with open(ep_dir / "metrics.json", "w", encoding="utf-8") as f:
                    json.dump({"episode": i, "return": float(ep_ret), "video": final_name},
                              f, ensure_ascii=False, indent=2)

            returns.append(ep_ret)
            print(f"[Eval] ep {i:03d} | return {ep_ret:.3f}")

    finally:
        # 清理离屏上下文
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