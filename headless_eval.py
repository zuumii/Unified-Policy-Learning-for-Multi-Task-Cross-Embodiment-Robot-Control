# headless_eval.py
import pathlib
import argparse
import json
import time
import gc
import warnings

import numpy as np
import torch
from ruamel.yaml import YAML

from align import ObsActAgent as Agent
import utils


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='train config file path')
    return p.parse_args()


def _get(d, k, default):
    return d[k] if isinstance(d, dict) and k in d else default


def _ensure_offscreen_context(sim):
    # 适配 mujoco-py 分支：手动建立离屏上下文（供 sim.render 使用）
    from robosuite.utils.binding_utils import MjRenderContextOffscreen
    if getattr(sim, "_render_context_offscreen", None) is not None:
        return
    try:
        sim._render_context_offscreen = MjRenderContextOffscreen(sim, device_id=0)
    except TypeError:
        sim._render_context_offscreen = MjRenderContextOffscreen(sim)


def _auto_pick_camera(env, params):
    # 1) 如果 yml 显式给了 render_camera，直接用
    if isinstance(params, dict) and params.get('render_camera'):
        return params['render_camera']
    # 2) 环境可用相机名
    names = list(getattr(env, "camera_names", []) or [])
    # 3) 偏好顺序：很多交互默认其实是 frontview（Reach 类任务尤常见）
    prefs = ['frontview', 'agentview', 'birdview', 'sideview', 'topview']
    for n in prefs:
        if n in names:
            return n
    # 4) 兜底：有就取第一个，没有就 None（自由相机，不建议）
    return names[0] if names else None


def main():
    # 可选：压掉 gym 的 float32 提示，不影响功能
    warnings.filterwarnings("ignore", message="Box bound precision lowered by casting to float32")

    args = parse_args()
    yaml = YAML(typ='safe')
    params = yaml.load(open(args.config, 'r'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 用原生 robosuite env 拿观测维度（保持你的流程）
    env_rs = utils.make_robosuite_env(
        params['env_name'],
        robots=params['robots'],
        controller_type=params['controller_type'],
        **params['env_kwargs'],
    )
    _obs0 = env_rs.reset()
    robot_obs_shape = np.concatenate([_obs0[k] for k in params['robot_obs_keys']]).shape
    obj_obs_shape   = np.concatenate([_obs0[k] for k in params['obj_obs_keys']]).shape

    # 评测环境（headless；不传渲染参数，避免和 utils 内部冲突）
    env = utils.make(
        params['env_name'],
        robots=params['robots'],
        controller_type=params['controller_type'],
        obs_keys=params['robot_obs_keys'] + params['obj_obs_keys'],
        seed=params['seed'] + 100,
        render=False,
        **params['env_kwargs'],
    )

    # 自动挑与交互一致的相机（可被 yml: render_camera 覆盖）
    camera_name = _auto_pick_camera(env, params)

    # 视频参数（可选 yml: video.width/height/fps）
    vcfg   = params.get('video', {}) if isinstance(params.get('video', {}), dict) else {}
    width  = int(_get(vcfg, 'width', 640))
    height = int(_get(vcfg, 'height', 480))
    fps    = int(_get(vcfg, 'fps', 20))

    # Agent
    obs_dims = {
        'robot_obs_dim': robot_obs_shape[0],
        'obs_dim': robot_obs_shape[0] + obj_obs_shape[0],
        'lat_obs_dim': params['lat_obs_dim'],
        'obj_obs_dim': obj_obs_shape[0],
    }
    act_dims = {
        'act_dim': env.action_space.shape[0],
        'lat_act_dim': params['lat_act_dim'],
    }
    agent = Agent(obs_dims, act_dims, device)
    agent.load(pathlib.Path(params['model_dir']))

    # 输出目录（以模型目录为根）
    model_dir = pathlib.Path(params['model_dir']).resolve()
    run_dir = (model_dir.parent / f"eval_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # 元信息
    meta = {
        "config": str(pathlib.Path(args.config).resolve()),
        "env_name": params['env_name'],
        "robots": params['robots'],
        "controller_type": params['controller_type'],
        "seed": params['seed'] + 100,
        "camera": camera_name or "free_camera",
        "width": width, "height": height, "fps": fps,
        "num_episodes": params['num_episodes'],
        "model_dir": str(model_dir), "headless": True,
    }
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    import imageio

    try:
        for i in range(params['num_episodes']):
            obs = env.reset()

            # 建立离屏上下文
            _ensure_offscreen_context(env.sim)

            done = False
            ep_ret = 0.0

            # 每集目录
            ep_dir = run_dir / f"ep_{i:03d}"
            ep_dir.mkdir(parents=True, exist_ok=True)

            tmp_mp4 = ep_dir / f"ep_{i:03d}_tmp.mp4"
            writer = imageio.get_writer(tmp_mp4.as_posix(), fps=fps)

            # --- 小修正：mujoco 离屏帧需要垂直翻转 ---
            def grab():
                frame = env.sim.render(camera_name=camera_name, width=width, height=height)
                return np.flipud(frame)  # ← 关键：解决“倒的”问题

            # reset 帧
            writer.append_data(grab())

            while not done:
                action = agent.sample_action(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                ep_ret += reward
                writer.append_data(grab())

            writer.close()

            final_name = f"ep_{i:03d}_return_{ep_ret:.1f}.mp4"
            (ep_dir / final_name).write_bytes(tmp_mp4.read_bytes())
            tmp_mp4.unlink()

            with open(ep_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump({"episode": i, "return": float(ep_ret), "video": final_name},
                          f, ensure_ascii=False, indent=2)

            print(f"[Eval] ep {i:03d} | return {ep_ret:.3f} | saved: {ep_dir/final_name}")
    finally:
        # 尽量干净地释放，减少 EGL 析构告警
        try:
            if getattr(env.sim, "_render_context_offscreen", None) is not None:
                # robosuite 的 MjRenderContext 有 gl_ctx.free()
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


if __name__ == '__main__':
    main()