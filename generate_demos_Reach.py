# -*- coding: utf-8 -*-
import pathlib
import datetime
import uuid
import io
import argparse

import numpy as np
import torch  # 保留，不影响

import utils
from human_policy import ReachPolicy, LiftPolicy, BaseHumanPolicy, PickPlacePolicy

# 可选：Stack（有就用，没有也不报错）
try:
    from human_policy import StackPolicy
except Exception:
    StackPolicy = None

np.set_printoptions(precision=4, suppress=True)

# 工作空间边界（越界剔除）
BOUND_MIN = np.array([-0.3, -0.4, 0.8])
BOUND_MAX = np.array([0.3, 0.4, 1.2])


def eplen(episode):
    return len(episode["action"])


def save_episode(directory, episode):
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    identifier = str(uuid.uuid4().hex)
    length = eplen(episode)
    filename = directory / f"{timestamp}-{identifier}-{length}.npz"
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        with filename.open("wb") as f2:
            f2.write(f1.read())
    return filename


def sample_episodes(env, policy, directory, num_episodes=1, policy_obs_keys=None, render=False):
    """
    纯 policy→env 采集一批 episode；保持环境 obs 的键集合稳定写盘
    """
    episodes_saved = 0
    while episodes_saved < num_episodes:
        obs = env.reset()

        if isinstance(policy, BaseHumanPolicy):
            policy.reset()

        obs_keys = list(obs.keys())
        done = False
        episode = {k: [obs[k]] for k in obs_keys}
        episode["action"] = []
        episode["reward"] = []

        while not done:
            if policy_obs_keys is not None:
                policy_obs = np.concatenate([np.array(obs[k]) for k in policy_obs_keys])
            else:
                policy_obs = obs
            action, _ = policy.predict(policy_obs)
            obs, rew, done, info = env.step(action)

            if render:
                env.render()

            for k in obs_keys:
                episode[k].append(obs[k])
            episode["action"].append(action)
            episode["reward"].append(rew)

        print(f"[sample_episodes] Return: {np.sum(episode['reward']):.2f}")
        save_episode(directory, episode)
        episodes_saved += 1
    env.close()


# ======== 动作统计工具（按需开启） ========
def _print_episode_action_stats(acts: np.ndarray, prefix: str = ""):
    """
    acts: [T, act_dim]，一条 JV episode 的动作
    打印 |a|>=0.90 / 0.98 的总体与逐维比例，以及 min/max/mean/std
    """
    if acts.size == 0:
        print(f"{prefix}[Stats] empty actions")
        return
    thr_09, thr_98 = 0.90, 0.98
    T, A = acts.shape
    abs_acts = np.abs(acts)

    ep_min = acts.min(axis=0)
    ep_max = acts.max(axis=0)
    ep_mean = acts.mean(axis=0)
    ep_std = acts.std(axis=0)

    ge09_per_dim = (abs_acts >= thr_09).sum(axis=0) / T
    ge98_per_dim = (abs_acts >= thr_98).sum(axis=0) / T
    ge09_overall = float((abs_acts >= thr_09).sum() / (T * A))
    ge98_overall = float((abs_acts >= thr_98).sum() / (T * A))

    print(
        f"{prefix}[Stats] |a|>=0.90 overall={ge09_overall*100:.1f}% | "
        f"|a|>=0.98 overall={ge98_overall*100:.1f}%\n"
        f"        per-dim >=0.90: {np.round(ge09_per_dim*100, 1)}%\n"
        f"        per-dim >=0.98: {np.round(ge98_per_dim*100, 1)}%\n"
        f"        min:  {np.round(ep_min, 3)}\n"
        f"        max:  {np.round(ep_max, 3)}\n"
        f"        mean: {np.round(ep_mean, 3)}\n"
        f"        std:  {np.round(ep_std, 3)}"
    )


def collect_human_episode(env, policy, render=False, return_joints=False, reset_target=False):
    """
    在 OSC 控制器下采一条人/脚本演示；可返回关节轨迹用于 JV 回放
    """
    if reset_target and hasattr(env, "reset_target"):
        obs = env.reset_target()
    else:
        obs = env.reset()
    policy.reset()

    # Reach 可能没有 target_pos；有就检查
    if "target_pos" in obs:
        assert (obs["target_pos"] >= BOUND_MIN).all() and (obs["target_pos"] <= BOUND_MAX).all()

    # 记录 mujoco 状态（用于 JV 回放）
    if return_joints:
        task_xml = env.sim.model.get_xml()
        task_init_state = np.array(env.sim.get_state().flatten())

    desired_jps, gripper_actions = [], []

    obs_keys = list(obs.keys())
    done = False
    episode = {k: [obs[k]] for k in obs_keys}
    episode["action"] = []
    episode["reward"] = []

    while not done:
        action, action_info = policy.predict(obs)
        obs, rew, done, info = env.step(action)

        # 越界过滤（存在该键才检查，兼容所有任务）
        if ("robot0_eef_pos" in obs and
            ((obs["robot0_eef_pos"] < BOUND_MIN).any() or (obs["robot0_eef_pos"] > BOUND_MAX).any())):
            print(f"Human demo out of bounds at {obs['robot0_eef_pos']}")
            return (None, None) if return_joints else None

        if render:
            env.render()

        desired_jps.append(env.robots[0]._joint_positions)
        gripper_actions.append(action[-1])

        for k in obs_keys:
            episode[k].append(obs[k])
        episode["action"].append(action)
        episode["reward"].append(rew)

    print(f"[collect_human_episode] Return: {np.sum(episode['reward']):.2f}")

    # 某些 env 没有 _check_success；缺失时默认通过
    ok = env._check_success() if getattr(env, "_check_success", None) else True
    if ok:
        if return_joints:
            return episode, {"task_info": [task_xml, task_init_state],
                             "joint_info": [desired_jps, gripper_actions]}
        else:
            return episode
    return (None, None) if return_joints else None


def collect_random_episode(env, render=False, return_joints=False):
    """
    随机策略采集一条（调试用）
    """
    obs = env.reset()

    if return_joints:
        task_xml = env.sim.model.get_xml()
        task_init_state = np.array(env.sim.get_state().flatten())

    desired_jps, gripper_actions = [], []

    obs_keys = list(obs.keys())
    done = False
    episode = {k: [obs[k]] for k in obs_keys}
    episode["action"] = []
    episode["reward"] = []

    while not done:
        action = np.random.uniform(low=-1, high=1, size=env.action_dim)
        action[0] += 0.04
        action = np.clip(action, -1, 1)
        obs, rew, done, info = env.step(action)

        if ("robot0_eef_pos" in obs and
            ((obs["robot0_eef_pos"] < BOUND_MIN).any() or (obs["robot0_eef_pos"] > BOUND_MAX).any())):
            return (None, None) if return_joints else None

        if render:
            env.render()

        desired_jps.append(env.robots[0]._joint_positions)
        gripper_actions.append(action[-1])

        for k in obs_keys:
            episode[k].append(obs[k])
        episode["action"].append(action)
        episode["reward"].append(rew)

    if return_joints:
        return episode, {"task_info": [task_xml, task_init_state],
                         "joint_info": [desired_jps, gripper_actions]}
    return episode


def osc_to_jv(env_name, robots, num_episodes=64, render=False, print_action_stats=False, min_return=50):
    """
    生成 JV 演示；print_action_stats=True 可在每条 JV 轨迹后打印动作统计
    """
    controller_type = "OSC_POSE"

    # ---- 环境 & 策略 ----
    if env_name == "Reach":
        policy_cls = ReachPolicy
        env_kwargs = {"horizon": 100, "table_full_size": (0.6, 0.6, 0.05)}
    elif env_name == "Lift":
        policy_cls = LiftPolicy
        env_kwargs = {"horizon": 200, "use_touch_obs": True, "table_offset": (0, 0, 0.908)}
    elif env_name == "PickPlaceBread":
        policy_cls = PickPlacePolicy
        env_kwargs = {"horizon": 200, "use_touch_obs": True,
                      "bin1_pos": (-0.05, -0.25, 0.90), "bin2_pos": (-0.05, 0.28, 0.90)}
    elif env_name == "Stack":
        if StackPolicy is None:
            raise ValueError("StackPolicy not available. Please provide human_policy.StackPolicy.")
        policy_cls = StackPolicy
        env_kwargs = {"horizon": 200, "use_touch_obs": True, "table_offset": (0, 0, 0.908)}
    else:
        raise ValueError(f"Unknown env_name: {env_name}")

    if robots == "Panda":
        env_kwargs["gripper_types"] = "PandaTouchGripper"
    elif robots == "Sawyer":
        env_kwargs["gripper_types"] = "PandaTouchGripper"
    elif robots in ("IIWA", "iiwa7", "iiwa14"):
        env_kwargs["gripper_types"] = "PandaTouchGripper"
    elif robots == "UR5e":
        env_kwargs["gripper_types"] = "PandaTouchGripper"

    env = utils.make_robosuite_env(env_name, robots, controller_type, render=render, **env_kwargs)
    policy = policy_cls(env)

    jv_env = utils.make_robosuite_env(
        env_name, robots=robots, controller_type="JOINT_VELOCITY", render=render, **env_kwargs
    )

    directory = pathlib.Path(f"./human_demonstrations/{env_name}/{robots}/JOINT_VELOCITY")
    directory.mkdir(parents=True, exist_ok=True)

    episodes_saved = 0
    while episodes_saved < num_episodes:

        reset_target = False if (episodes_saved % 5 == 0) else True

        episode = None
        while episode is None:
            episode, ep_info = collect_human_episode(
                env, policy, render=render, return_joints=True, reset_target=reset_target
            )

        task_xml, task_init_state = ep_info["task_info"]
        desired_jps, gripper_actions = ep_info["joint_info"]

        jv_env.reset()
        jv_env.reset_from_xml_string(task_xml)
        jv_env.sim.reset()
        jv_env.sim.set_state_from_flattened(task_init_state)
        jv_env.sim.forward()

        obs = jv_env._get_observations(force_update=True)
        obs_keys = list(obs.keys())
        ep = {k: [obs[k]] for k in obs_keys}
        ep["action"] = []
        ep["reward"] = []

        for next_jp, gripper_action in zip(desired_jps, gripper_actions):
            action = np.zeros(jv_env.robots[0].dof, dtype=np.float32)
            action[-1] = gripper_action
            err = next_jp - jv_env.robots[0]._joint_positions

            kp = 20
            action[:-1] = np.clip(err * kp, -1, 1)

            obs, rew, done, info = jv_env.step(action)

            if ("robot0_eef_pos" in obs and
                ((obs["robot0_eef_pos"] < BOUND_MIN).any() or (obs["robot0_eef_pos"] > BOUND_MAX).any())):
                print(f"Joint vel episode out of bounds at {obs['robot0_eef_pos']}")
                break

            if render:
                jv_env.render()

            for k in obs_keys:
                ep[k].append(obs[k])
            ep["action"].append(action)
            ep["reward"].append(rew)

            if done:
                break

        ep_return = float(np.sum(ep["reward"]))
        print(f"[osc_to_jv] Episode {episodes_saved} return: {ep_return:.2f}")

        if print_action_stats and len(ep["action"]) > 0:
            acts_np = np.asarray(ep["action"], dtype=np.float32)
            _print_episode_action_stats(acts_np, prefix=f"[{env_name} ep {episodes_saved:03d}] ")

        if ep_return > min_return:
            save_episode(directory, ep)
            episodes_saved += 1

    env.close()
    jv_env.close()


def check_data():
    import replay_buffer
    buffer_1 = replay_buffer.ReplayBuffer(
        obs_shape=(3,),
        action_shape=(8,),
        capacity=1000000,
        batch_size=256,
        device="cpu",
    )
    buffer_2 = replay_buffer.ReplayBuffer(
        obs_shape=(3,),
        action_shape=(8,),
        capacity=1000000,
        batch_size=256,
        device="cpu",
    )

    obs_keys = ["robot0_eef_pos"]
    demo_dir = pathlib.Path("human_demonstrations/Reach/Panda/JOINT_VELOCITY")
    demo_paths_1 = utils.load_episodes(demo_dir, obs_keys)
    buffer_1.add_rollouts(demo_paths_1)

    demo_dir = pathlib.Path("human_demonstrations/Reach/Sawyer/JOINT_VELOCITY")
    demo_paths_2 = utils.load_episodes(demo_dir, obs_keys)
    buffer_2.add_rollouts(demo_paths_2)

    obs_1 = buffer_1.obses[: buffer_1.idx]
    obs_2 = buffer_2.obses[: buffer_2.idx]

    print(obs_1.mean(0), obs_1.std(0), np.amin(obs_1, axis=0), np.amax(obs_1, axis=0))
    print(obs_2.mean(0), obs_2.std(0), np.amin(obs_2, axis=0), np.amax(obs_2, axis=0))


def probe_obs(env_name="Lift", robots="Panda", controller_type="JOINT_VELOCITY",
              steps=20, render=False, **env_kwargs):
    """
    快速打印当前任务可用 obs 键、形状与范围
    """
    import utils as _utils
    env = _utils.make_robosuite_env(env_name, robots, controller_type, render=render, **env_kwargs)
    obs = env.reset()
    keys = list(obs.keys())
    print(f"[{env_name}] 可用 obs 键（{len(keys)}个）:", keys)
    stats = {k: {"shape": np.array(obs[k]).shape, "mins": [], "maxs": []} for k in keys}
    for _ in range(steps):
        a = np.zeros(env.action_dim)
        obs, _, done, _ = env.step(a)
        for k, v in obs.items():
            v = np.array(v)
            stats[k]["mins"].append(v.min())
            stats[k]["maxs"].append(v.max())
        if done:
            obs = env.reset()
    for k, v in stats.items():
        print(f"{k:28s} shape={v['shape']}  min≈{np.min(v['mins']): .3f}  max≈{np.max(v['maxs']): .3f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Reach", help="Robosuite task")
    parser.add_argument("--robot", type=str, default="Panda", help="Robot to generate demonstrations")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of demonstration episodes to generate")
    parser.add_argument("--render", action="store_true", help="Render while collecting")
    parser.add_argument("--print_action_stats", action="store_true", help="Print per-episode JV action statistics")
    parser.add_argument("--min_return", type=float, default=50.0, help="Only save episodes with return > this value")
    args = parser.parse_args()

    # 可选：探测观测键范围（便于调参）
    probe_kwargs = {}
    if args.env_name == "Reach":
        probe_kwargs.update(dict(horizon=80, table_full_size=[0.6, 0.6, 0.05]))
    elif args.env_name in ("Lift", "Stack"):
        probe_kwargs.update(dict(table_offset=(0, 0, 0.908)))

    if args.robot == "Panda":
        probe_kwargs.update(dict(gripper_types="PandaTouchGripper"))
    elif args.robot == "Sawyer":
        probe_kwargs.update(dict(gripper_types="PandaTouchGripper"))
    elif args.robot in ("IIWA", "iiwa7", "iiwa14"):
        probe_kwargs.update(dict(gripper_types="PandaTouchGripper"))
    elif args.robot == "UR5e":
        probe_kwargs.update(dict(gripper_types="PandaTouchGripper"))

    probe_obs(
        env_name=args.env_name,
        robots=args.robot,
        controller_type="JOINT_VELOCITY",
        steps=30,
        render=False,
        **probe_kwargs,
    )

    # 生成 JV 演示
    osc_to_jv(
        env_name=args.env_name,
        robots=args.robot,
        num_episodes=args.num_episodes,
        render=args.render,
        print_action_stats=args.print_action_stats,
        min_return=args.min_return,
    )