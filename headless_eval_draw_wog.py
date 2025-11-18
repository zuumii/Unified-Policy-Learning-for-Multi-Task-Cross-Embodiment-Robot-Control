# headless_eval_bc.py
import pathlib
import argparse
import json
import time
import gc
import warnings
import shutil

import numpy as np
import torch
from ruamel.yaml import YAML

import utils
from bc_wog import BCObsActAgent as Agent
from bc_wog import Actor as BCActor  # 仅在“自适配 actor 规模”时用到


# =========================
#   边界默认值与小工具
# =========================
DEFAULT_BOUND_MIN = np.array([-0.3, -0.4, 0.8], dtype=np.float32)
DEFAULT_BOUND_MAX = np.array([0.3, 0.4, 1.25], dtype=np.float32)
DEFAULT_BOUND_KEY = "robot0_eef_pos"


def within_bounds(vec: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> bool:
    """逐元素区间判断：bmin <= vec <= bmax"""
    return (vec >= bmin).all() and (vec <= bmax).all()


def fetch_obs_dict(env, force_update: bool = False) -> dict:
    """Robosuite 原生观测字典"""
    return env._get_observations(force_update=force_update)


def fetch_pos(env, key: str, force_update: bool = False) -> np.ndarray:
    """从 robosuite 环境取指定键向量（如 robot0_eef_pos）"""
    obs_dict = fetch_obs_dict(env, force_update=force_update)
    if key not in obs_dict:
        avail = ", ".join(sorted(list(obs_dict.keys())))
        raise RuntimeError(f"[Eval] 边界键 '{key}' 不存在。可用键有：{avail}")
    return np.array(obs_dict[key], dtype=np.float32)


# [TSNE] 扁平化观测：按 keys 顺序拼接为一个向量（用于 obs_flat 与 z_obs 编码）
def _flatten_by_keys(obs_dict: dict, keys: list) -> np.ndarray:
    return np.concatenate([np.asarray(obs_dict[k], dtype=np.float32) for k in keys], axis=0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="train config file path")

    # === 边界相关 CLI，可覆盖默认值 ===
    p.add_argument("--bound-min", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="Eval 边界下界，默认与数据生成一致")
    p.add_argument("--bound-max", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="Eval 边界上界，默认与数据生成一致")
    p.add_argument("--bound-key", type=str, default=DEFAULT_BOUND_KEY,
                   help=f"用于边界判断的 obs 键名（默认 {DEFAULT_BOUND_KEY}）")
    return p.parse_args()


def _get(d, k, default):
    return d[k] if isinstance(d, dict) and k in d else default


def _ensure_offscreen_context(sim):
    """适配 mujoco-py 分支：手动建立离屏上下文"""
    from robosuite.utils.binding_utils import MjRenderContextOffscreen
    if getattr(sim, "_render_context_offscreen", None) is not None:
        return
    try:
        sim._render_context_offscreen = MjRenderContextOffscreen(sim, device_id=0)
    except TypeError:
        sim._render_context_offscreen = MjRenderContextOffscreen(sim)


def _auto_pick_camera(env, params):
    if isinstance(params, dict) and params.get("render_camera"):
        return params["render_camera"]
    names = list(getattr(env, "camera_names", []) or [])
    prefs = ["frontview", "agentview", "birdview", "sideview", "topview"]
    for n in prefs:
        if n in names:
            return n
    return names[0] if names else None


# =========================
#  Actor 自适配：按 ckpt 重建
# =========================
def _infer_actor_arch_from_state_dict(sd: dict):
    items = []
    for k, v in sd.items():
        if k.startswith("trunk.") and k.endswith(".weight") and isinstance(v, torch.Tensor):
            parts = k.split(".")
            if len(parts) >= 3:
                try:
                    idx = int(parts[1])
                except Exception:
                    continue
                items.append((idx, tuple(v.shape)))
    items.sort(key=lambda x: x[0])
    if not items:
        raise RuntimeError("actor state_dict 缺少 trunk.*.weight")

    first_shape = items[0][1]
    last_shape = items[-1][1]
    in_dim = int(first_shape[1])
    hidden_dim = int(first_shape[0])
    out_dim = int(last_shape[0])
    n_linear = len(items)
    n_hidden = n_linear - 1
    return in_dim, out_dim, n_hidden, hidden_dim


def _safe_load_(module, path, device, strict=True, name=""):
    sd = torch.load(path, map_location=device)
    try:
        module.load_state_dict(sd, strict=strict)
        return True
    except Exception as e:
        if strict:
            print(f"[eval.load] {name or path} strict=True 失败，尝试 strict=False：{e}")
            try:
                module.load_state_dict(sd, strict=False)
                return True
            except Exception as e2:
                print(f"[eval.load] {name or path} strict=False 仍失败：{e2}")
                return False
        else:
            print(f"[eval.load] {name or path} strict=False 失败：{e}")
            return False


def _load_agent_weights_with_actor_rebuild(agent, model_dir: pathlib.Path,
                                           device: torch.device,
                                           lat_obs_dim: int, obj_obs_dim: int, lat_act_dim: int):
    try:
        agent.load(model_dir.as_posix())
        print("[eval.load] 直接加载 agent.load(...) 成功。")
        return
    except Exception as e:
        print(f"[eval.load] agent.load 严格加载失败，将进入 actor 自适配流程：{e}")

    pairs = [
        ("obs_enc", "obs_enc.pt"),
        ("obs_dec", "obs_dec.pt"),
        ("act_enc", "act_enc.pt"),
        ("act_dec", "act_dec.pt"),
        ("inv_dyn", "inv_dyn.pt"),
        ("fwd_dyn", "fwd_dyn.pt"),
    ]
    for attr, fname in pairs:
        if hasattr(agent, attr):
            p = model_dir / fname
            if p.exists():
                _safe_load_(getattr(agent, attr), p.as_posix(), device, strict=True, name=fname)

    if hasattr(agent, "grip_head"):
        p = model_dir / "grip_head.pt"
        if p.exists():
            _safe_load_(agent.grip_head, p.as_posix(), device, strict=True, name="grip_head.pt")

    actor_ckpt = model_dir / "actor.pt"
    if not actor_ckpt.exists():
        raise FileNotFoundError(f"[eval.load] {actor_ckpt} 不存在。")
    actor_sd = torch.load(actor_ckpt.as_posix(), map_location=device)
    in_dim_ckpt, out_dim_ckpt, n_layers_ckpt, hidden_ckpt = _infer_actor_arch_from_state_dict(actor_sd)
    in_dim_expect = int(lat_obs_dim + obj_obs_dim)
    out_dim_expect = int(lat_act_dim)
    if in_dim_ckpt != in_dim_expect or out_dim_ckpt != out_dim_expect:
        raise RuntimeError("actor 输入/输出维不匹配")
    agent.actor = BCActor(in_dim_expect, out_dim_expect, n_layers_ckpt, hidden_ckpt).to(device)
    agent.actor.load_state_dict(actor_sd, strict=True)
    print("[eval.load] actor 重建并严格加载成功。")


# [TSNE] 推断编码器首层输入维度（兼容 nn.Sequential / 自定义模块）
def _infer_first_linear_in_features(module: torch.nn.Module):
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            return int(m.in_features)
    return None


def main():
    warnings.filterwarnings("ignore", message="Box bound precision lowered by casting to float32")

    args = parse_args()
    yaml = YAML(typ="safe")
    params = yaml.load(open(args.config, "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_rs = utils.make_robosuite_env(
        params["env_name"],
        robots=params["robots"],
        controller_type=params["controller_type"],
        **params["env_kwargs"],
    )
    _obs0 = env_rs.reset()
    robot_obs_shape = np.concatenate([_obs0[k] for k in params["robot_obs_keys"]]).shape
    obj_obs_shape = np.concatenate([_obs0[k] for k in params["obj_obs_keys"]]).shape

    env = utils.make(
        params["env_name"],
        robots=params["robots"],
        controller_type=params["controller_type"],
        obs_keys=params["robot_obs_keys"] + params["obj_obs_keys"],
        seed=params["seed"] + 100,
        render=False,
        **params["env_kwargs"],
    )
    camera_name = _auto_pick_camera(env, params)

    # [TSNE] 三组 keys：给 obs_enc 用 robot_keys；保存 obs_flat 用 all_keys
    robot_keys = params["robot_obs_keys"]
    obj_keys = params["obj_obs_keys"]
    obs_keys_all = robot_keys + obj_keys  # 说明用，不强依赖

    vcfg = params.get("video", {}) if isinstance(params.get("video", {}), dict) else {}
    width = int(_get(vcfg, "width", 640))
    height = int(_get(vcfg, "height", 480))
    fps = int(_get(vcfg, "fps", 20))

    obs_dims = {
        "robot_obs_dim": robot_obs_shape[0],
        "obs_dim": robot_obs_shape[0] + obj_obs_shape[0],
        "lat_obs_dim": params["lat_obs_dim"],
        "obj_obs_dim": obj_obs_shape[0],
    }
    act_dims = {
        "act_dim": env.action_space.shape[0],
        "lat_act_dim": params["lat_act_dim"],
    }
    agent = Agent(
        obs_dims,
        act_dims,
        device,
        n_layers=params.get("n_layers", 3),
        hidden_dim=params.get("hidden_dim", 256),
        actor_n_layers=params.get("actor_n_layers"),
        actor_hidden_dim=params.get("actor_hidden_dim"),
    )
    agent.expl_noise = 0.0
    if hasattr(agent, "use_external_gripper"):
        agent.use_external_gripper = True

    model_dir = pathlib.Path(params["model_dir"]).resolve()
    _load_agent_weights_with_actor_rebuild(
        agent,
        model_dir,
        device,
        lat_obs_dim=params["lat_obs_dim"],
        obj_obs_dim=obj_obs_shape[0],
        lat_act_dim=params["lat_act_dim"],
    )

    run_dir = (model_dir.parent / f"eval_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # [TSNE] 轨迹保存目录：eval_xxx/traj_npz
    traj_root = (run_dir / "traj_npz").resolve()
    traj_root.mkdir(parents=True, exist_ok=True)

    bmin = np.array(args.bound_min, dtype=np.float32) if args.bound_min else DEFAULT_BOUND_MIN
    bmax = np.array(args.bound_max, dtype=np.float32) if args.bound_max else DEFAULT_BOUND_MAX
    bkey = args.bound_key or DEFAULT_BOUND_KEY

    meta = {
        "config": str(pathlib.Path(args.config).resolve()),
        "env_name": params["env_name"],
        "robots": params["robots"],
        "controller_type": params["controller_type"],
        "seed": params["seed"] + 100,
        "camera": camera_name or "free_camera",
        "width": width,
        "height": height,
        "fps": fps,
        "num_episodes": int(params["num_episodes"]),
        "model_dir": str(model_dir),
        "headless": True,
        "bound_min": bmin.tolist(),
        "bound_max": bmax.tolist(),
        "bound_key": bkey,
    }
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    import imageio

    valid_idx, trial_idx, discarded_cnt = 0, 0, 0
    returns = []

    # Reach 任务的最后一帧距离统计（仅 env_name == "Reach" 时记录）
    reach_final_dists = []
    is_reach_task = (str(params["env_name"]).lower() == "reach")

    try:
        target_eps = int(params["num_episodes"])
        while valid_idx < target_eps:
            trial_idx += 1
            obs = env.reset()
            _ensure_offscreen_context(env.sim)
            pos0 = fetch_pos(env, bkey, force_update=True)
            if not within_bounds(pos0, bmin, bmax):
                discarded_cnt += 1
                print(f"[Eval] trial {trial_idx} out-of-bounds at reset -> discarded | pos={pos0}")
                continue

            done, ep_ret, episode_discarded = False, 0.0, False
            ep_dir = run_dir / f"ep_{valid_idx:03d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            tmp_mp4 = ep_dir / f"ep_{valid_idx:03d}_tmp.mp4"
            writer = imageio.get_writer(tmp_mp4.as_posix(), fps=fps)

            # [TSNE] 轨迹缓存（含 a_arm）
            traj = {"t": [], "obs_flat": [], "a_env": [], "a_arm": [], "obs_act": [],
                    "z_obs": [], "z_act": [], "z_obs_act": []}
            t = 0
            zobs_dim_warned = False  # robot_obs 维度不符，仅告警一次
            zact_warned = False      # act_enc 编码失败，仅告警一次

            def grab():
                frame = env.sim.render(camera_name=camera_name, width=width, height=height)
                return np.flipud(frame)

            writer.append_data(grab())
            try:
                while not done:
                    # ===== 1) 获取观测（拆分 try，避免连坐清空） =====
                    full_obs = fetch_obs_dict(env, force_update=False)

                    robot_obs_flat = None
                    obj_obs_flat = None

                    try:
                        robot_obs_flat = _flatten_by_keys(full_obs, robot_keys)
                    except Exception as e:
                        if not zobs_dim_warned:
                            print(f"[TSNE][warn] flatten robot_obs failed: {e}")
                            zobs_dim_warned = True

                    try:
                        obj_obs_flat = _flatten_by_keys(full_obs, obj_keys)
                    except Exception:
                        obj_obs_flat = None

                    if (robot_obs_flat is not None) and (obj_obs_flat is not None):
                        obs_flat = np.concatenate([robot_obs_flat, obj_obs_flat], axis=0)
                    else:
                        obs_flat = np.asarray(obs, dtype=np.float32)

                    # 只用 robot_obs -> obs_enc
                    z_obs_vec = None
                    if hasattr(agent, "obs_enc") and (robot_obs_flat is not None):
                        if robot_obs_flat.shape[0] == obs_dims["robot_obs_dim"]:
                            with torch.no_grad():
                                obs_t = torch.from_numpy(robot_obs_flat).to(device).unsqueeze(0).float()
                                z_obs_vec = agent.obs_enc(obs_t).cpu().numpy()[0]
                        else:
                            if not zobs_dim_warned:
                                print(f"[TSNE][warn] robot_obs_dim mismatch: got {robot_obs_flat.shape[0]}, "
                                      f"expected {obs_dims['robot_obs_dim']}. Skip z_obs for this step.")
                                zobs_dim_warned = True

                    # 记录时间与观测
                    traj["t"].append(t)
                    traj["obs_flat"].append(obs_flat.astype(np.float32))
                    if z_obs_vec is not None:
                        traj["z_obs"].append(z_obs_vec.astype(np.float32))

                    # ===== 2) 采样动作（环境仍用完整 action）=====
                    action = agent.sample_action(obs, deterministic=True).astype(np.float32)
                    a_env = action
                    # 去掉最后一维（夹爪）作为 a_arm（若存在）
                    a_arm = a_env[:-1] if a_env.shape[0] >= 2 else a_env

                    # ===== 3) z_act：只用 a_arm 参与候选 =====
                    z_act_vec = None
                    if hasattr(agent, "act_enc"):
                        cand_list = []
                        if isinstance(obs_flat, np.ndarray):
                            cand_list.append(("obs+act_arm", np.concatenate([obs_flat, a_arm], axis=0)))
                        if robot_obs_flat is not None:
                            cand_list.append(("robot+act_arm", np.concatenate([robot_obs_flat, a_arm], axis=0)))
                        cand_list.append(("act_arm", a_arm))

                        exp_in = _infer_first_linear_in_features(agent.act_enc)
                        tried = []
                        picked = None

                        for name, x in cand_list:
                            if exp_in is not None and x.shape[0] != exp_in:
                                continue
                            try:
                                with torch.no_grad():
                                    a_t = torch.from_numpy(x).to(device).unsqueeze(0).float()
                                    z_tmp = agent.act_enc(a_t).cpu().numpy()[0]
                                z_act_vec = z_tmp
                                picked = (name, x.shape[0])
                                break
                            except Exception as e:
                                tried.append((name, x.shape[0], str(e)))
                                continue

                        if z_act_vec is None and not zact_warned:
                            cand_dims = [(n, (None if not isinstance(x, np.ndarray) else x.shape[0])) for n, x in cand_list]
                            print(f"[TSNE][warn] act_enc encoding skipped; expected_in={exp_in}, "
                                  f"cand_dims={cand_dims}, tried={[(n,d) for n,d,_ in tried]}")
                            zact_warned = True
                        # 调试成功路径可临时打开：
                        # else:
                        #     print(f"[TSNE] act_enc picked: {picked}")

                    # 记录 a_env / a_arm / obs_act(使用 a_arm) / 以及可能的 z_act 与 z_obs_act
                    traj["a_env"].append(a_env)
                    traj["a_arm"].append(a_arm)
                    obs_act_vec = np.concatenate([obs_flat, a_arm], axis=0)
                    traj["obs_act"].append(obs_act_vec.astype(np.float32))

                    if z_act_vec is not None:
                        traj["z_act"].append(z_act_vec.astype(np.float32))
                        if z_obs_vec is not None:
                            z_obs_act_vec = np.concatenate([z_obs_vec.astype(np.float32), z_act_vec.astype(np.float32)], axis=0)
                            traj["z_obs_act"].append(z_obs_act_vec)

                    # ===== 4) 环境交互（原逻辑不变） =====
                    obs, reward, done, _ = env.step(a_env)
                    ep_ret += reward
                    pos = fetch_pos(env, bkey, force_update=False)
                    if not within_bounds(pos, bmin, bmax):
                        discarded_cnt += 1
                        episode_discarded = True
                        print(f"[Eval] trial {trial_idx} out-of-bounds at step -> discarded | pos={pos}")
                        break
                    writer.append_data(grab())

                    t += 1
            finally:
                if episode_discarded:
                    try:
                        writer.close()
                    except Exception:
                        pass
                    try:
                        shutil.rmtree(ep_dir, ignore_errors=True)
                    except Exception:
                        pass

            if episode_discarded:
                continue

            # === Reach: 记录最后一帧 target_to_robot0_eef_pos 的欧氏距离 ===
            if is_reach_task:
                last_obs_dict = fetch_obs_dict(env, force_update=True)
                if "target_to_robot0_eef_pos" in last_obs_dict:
                    vec = np.array(last_obs_dict["target_to_robot0_eef_pos"], dtype=np.float32)
                    final_dist = float(np.linalg.norm(vec))
                else:
                    if ("target_pos" in last_obs_dict) and ("robot0_eef_pos" in last_obs_dict):
                        tpos = np.array(last_obs_dict["target_pos"], dtype=np.float32)
                        eefp = np.array(last_obs_dict["robot0_eef_pos"], dtype=np.float32)
                        final_dist = float(np.linalg.norm(tpos - eefp))
                    else:
                        final_dist = float("nan")
                        print("[Eval][Reach] 缺少 target_to_robot0_eef_pos（以及兜底键），无法计算最终距离。")
                reach_final_dists.append(final_dist)

            writer.close()
            final_name = f"ep_{valid_idx:03d}_return_{ep_ret:.1f}.mp4"
            (ep_dir / final_name).write_bytes(tmp_mp4.read_bytes())
            tmp_mp4.unlink(missing_ok=True)

            # [TSNE] 保存轨迹到 eval_xxx/traj_npz/ep_XXX.npz（仅保存未丢弃 episode）
            traj_np = {k: np.asarray(v, dtype=np.float32) for k, v in traj.items() if len(v) > 0}
            traj_np["__episode__"] = np.array(valid_idx, dtype=np.int32)
            traj_np["__env_name__"] = np.array(params["env_name"])
            traj_np["__robots__"] = np.array(json.dumps(params["robots"]))
            np.savez_compressed((traj_root / f"ep_{valid_idx:03d}.npz").as_posix(), **traj_np)
            print(f"[TSNE] saved traj_npz/ep_{valid_idx:03d}.npz | keys={list(traj_np.keys())}")

            # 原有 metrics.json
            with open(ep_dir / "metrics.json", "w", encoding="utf-8") as f:
                out = {"episode": valid_idx, "return": float(ep_ret), "video": final_name}
                if is_reach_task and len(reach_final_dists) == len(returns) + 1:
                    out["reach_final_dist"] = reach_final_dists[-1]
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"[Eval] ep {valid_idx:03d} | return {ep_ret:.3f} | saved: {ep_dir/final_name}")

            returns.append(ep_ret)
            valid_idx += 1

        if returns:
            mean_ret = float(np.mean(returns))
            std_ret = float(np.std(returns))
            print(f"\n[Eval Summary] valid={valid_idx}/{target_eps}, discarded={discarded_cnt}")
            print(f"[Eval Summary] mean_return={mean_ret:.3f}, std_return={std_ret:.3f}")

            summary = {
                "valid": valid_idx,
                "discarded": discarded_cnt,
                "mean_return": mean_ret,
                "std_return": std_ret,
            }

            # Reach 任务的最后距离：平均值 + 方差（总体方差 np.var）
            if is_reach_task and len(reach_final_dists) > 0:
                clean = np.array([d for d in reach_final_dists if np.isfinite(d)], dtype=np.float32)
                if clean.size > 0:
                    reach_mean = float(np.mean(clean))
                    reach_var = float(np.var(clean))
                    print(f"[Eval Summary][Reach] final_dist_mean={reach_mean:.6f}, final_dist_var={reach_var:.6f}")
                    summary["reach_final_dist_mean"] = reach_mean
                    summary["reach_final_dist_var"] = reach_var
                else:
                    print("[Eval Summary][Reach] 无法计算最终距离（有效样本为空）。")
                    summary["reach_final_dist_mean"] = None
                    summary["reach_final_dist_var"] = None

            with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print()
        else:
            print(f"[Eval Summary] 全部被丢弃，未产生有效 episode。")

    finally:
        try:
            if getattr(env.sim, "_render_context_offscreen", None) is not None:
                env.sim._render_context_offscreen.gl_ctx.free()
                env.sim._render_context_offscreen = None
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
        del env
        gc.collect()


if __name__ == "__main__":
    main()