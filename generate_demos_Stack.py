# -*- coding: utf-8 -*-
"""
demo_generator_all.py
- 支持 Reach / Lift / PickPlaceBread / Stack
- 使用 human_policy 中的策略（包括 StackPolicy）
- 维持键集合锁定与对齐（同一轮生成期间键一致）
- 支持 OSC→JV 回放，生成 JOINT_VELOCITY 演示
"""
import pathlib
import datetime
import uuid
import io
import argparse

import numpy as np
import torch  # 保留不影响

import utils
from human_policy import ReachPolicy, LiftPolicy, BaseHumanPolicy, PickPlacePolicy, StackPolicy
import inspect
   
# 1) 导入 robosuite 的环境类（注意模块路径）
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.reach import Reach
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.environments.manipulation.stack import Stack

print("Lift kwargs:\n", inspect.signature(Lift.__init__))
print("Reach kwargs:\n", inspect.signature(Reach.__init__))
print("PickPlace kwargs:\n", inspect.signature(PickPlace.__init__))
print("Stack kwargs:\n", inspect.signature(Stack.__init__))

np.set_printoptions(precision=4, suppress=True)

# 操作空间边界（越界剔除）
BOUND_MIN = np.array([-0.3, -0.4, 0.8])
BOUND_MAX = np.array([0.3, 0.4, 1.25])


def eplen(episode):
    return len(episode["action"])


# ========= 键集合锁定与对齐 =========
def _init_key_profile(obs):
    """基于首次 reset 的 obs，锁定键集合与每个键的 shape；返回稳定排序 key_order。"""
    profile = {}
    for k, v in obs.items():
        arr = np.array(v)
        profile[k] = arr.shape
    key_order = sorted(profile.keys())
    return key_order, profile


def _align_obs(obs, key_order, key_shapes):
    """将当前 obs 对齐到固定 key_order；缺失键用 0 填充到固定 shape。"""
    aligned = {}
    for k in key_order:
        if k in obs:
            v = np.array(obs[k])
            aligned[k] = v
        else:
            aligned[k] = np.zeros(key_shapes[k], dtype=np.float32)
    return aligned


def _append_step(episode, obs_aligned, action, reward):
    for k, v in obs_aligned.items():
        episode[k].append(v)
    episode["action"].append(action)
    episode["reward"].append(reward)


# ========= 通用采样管线 =========
def sample_episodes(env, policy, directory, num_episodes=1, policy_obs_keys=None, render=False):
    """按固定键集合采样，适用于“纯 policy → env”的最小管线。"""
    episodes_saved = 0
    while episodes_saved < num_episodes:
        obs = env.reset()

        if isinstance(policy, BaseHumanPolicy) or hasattr(policy, "reset"):
            policy.reset()

        # 锁定键集合与形状
        key_order, key_shapes = _init_key_profile(obs)
        obs_aligned = _align_obs(obs, key_order, key_shapes)

        done = False
        episode = {k: [obs_aligned[k]] for k in key_order}
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

            obs_aligned = _align_obs(obs, key_order, key_shapes)
            _append_step(episode, obs_aligned, action, rew)

        print(f"[sample_episodes] Return: {np.sum(episode['reward']):.2f}")
        save_episode(directory, episode)
        episodes_saved += 1
    env.close()


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


# ========= OSC 直接采集（人/脚本策略） =========
def save_osc_episodes(num_episodes=64, render=False):
    """
    生成 OSC 演示并保存（示例入口，内部固定 env/robot，可按需改成参数化）
    """
    env_name = "Reach"   # 可改： "Lift" / "PickPlaceBread" / "Stack"
    controller_type = "OSC_POSE"
    robots = "Panda"     # 仅支持： "Panda" / "Sawyer" / "xArm6" / "UR5e"

    if env_name == "Reach":
        policy_cls = ReachPolicy
        env_kwargs = {"horizon": 100, "table_full_size": (0.6, 0.6, 0.00)}
    elif env_name == "Lift":
        policy_cls = LiftPolicy
        env_kwargs = {"horizon": 200, "use_touch_obs": True}
        if robots == "Panda":
            env_kwargs["gripper_types"] = "PandaTouchGripper"
        elif robots == "Sawyer":
            env_kwargs["gripper_types"] = "PandaTouchGripper"
        elif robots == "xArm6":
            env_kwargs["gripper_types"] = "PandaTouchGripper" 
        elif robots == "UR5e":
            env_kwargs["gripper_types"] = "PandaTouchGripper"

    elif env_name == "PickPlaceBread":
        policy_cls = PickPlacePolicy
        env_kwargs = {
            "horizon": 500,
            "use_touch_obs": True,
            "bin1_pos": (-0.05, -0.25, 0.90),
            "bin2_pos": (-0.05, 0.28, 0.90),
        }
        if robots == "Panda":
            env_kwargs["gripper_types"] = "PandaTouchGripper"
        elif robots == "Sawyer":
            env_kwargs["gripper_types"] = "PandaTouchGripper"
        elif robots == "xArm6":
            env_kwargs["gripper_types"] = "PandaTouchGripper" 
        elif robots == "UR5e":
            env_kwargs["gripper_types"] = "PandaTouchGripper"
    elif env_name == "Stack":
        policy_cls = StackPolicy
        env_kwargs = {"horizon": 300, "use_touch_obs": True, "table_offset": (0, 0, 0.908)}
        if robots == "Panda":
            env_kwargs["gripper_types"] = "PandaTouchGripper"
        elif robots == "Sawyer":
            env_kwargs["gripper_types"] = "PandaTouchGripper"
        elif robots == "xArm6":
            env_kwargs["gripper_types"] = "PandaTouchGripper" 
        elif robots == "UR5e":
            env_kwargs["gripper_types"] = "PandaTouchGripper"
    else:
        raise ValueError(f"Unsupported env_name for save_osc_episodes(): {env_name}")

    env = utils.make_robosuite_env(env_name, robots, controller_type, render=render, **env_kwargs)
    policy = policy_cls(env)

    directory = pathlib.Path(f"./human_demonstrations/{env_name}/{robots}/{controller_type}")
    directory.mkdir(parents=True, exist_ok=True)

    episodes_saved = 0
    while episodes_saved < num_episodes:
        episode = collect_human_episode(env, policy, render=render)
        if episode is None:
            continue
        else:
            print(f"Episode return: {np.sum(episode['reward']):.2f}")
            save_episode(directory, episode)
            episodes_saved += 1


def collect_human_episode(env, policy, render=False, return_joints=False, reset_target=False):
    """人/脚本策略在 OSC 控制器下采一条完整 episode；支持返回关节轨迹用于 OSC→JV 回放。"""
    if reset_target and hasattr(env, "reset_target"):
        obs = env.reset_target()
    else:
        obs = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()

    # 锁定键集合与形状（一次 episode 内保持一致）
    key_order, key_shapes = _init_key_profile(obs)
    obs_aligned = _align_obs(obs, key_order, key_shapes)

    if "target_pos" in obs_aligned:
        assert (obs_aligned["target_pos"] >= BOUND_MIN).all() and (obs_aligned["target_pos"] <= BOUND_MAX).all()

    if return_joints:
        task_xml = env.sim.model.get_xml()
        task_init_state = np.array(env.sim.get_state().flatten())

    desired_jps, gripper_actions = [], []

    done = False
    episode = {k: [obs_aligned[k]] for k in key_order}
    episode["action"] = []
    episode["reward"] = []

    while not done:
        action, action_info = policy.predict(obs)
        obs, rew, done, info = env.step(action)

        # 对齐后再做越界检查
        obs_aligned = _align_obs(obs, key_order, key_shapes)
        if (obs_aligned["robot0_eef_pos"] < BOUND_MIN).any() or (obs_aligned["robot0_eef_pos"] > BOUND_MAX).any():
            print(f"[collect_human_episode] Out of bounds at {obs_aligned['robot0_eef_pos']}")
            return (None, None) if return_joints else None

        if render:
            env.render()

        desired_jps.append(env.robots[0]._joint_positions)
        # 动作可能无抓取通道时兜底
        gripper_actions.append(action[-1] if action.shape[-1] > 0 else 0.0)

        _append_step(episode, obs_aligned, action, rew)

    print(f"[collect_human_episode] Return: {np.sum(episode['reward']):.2f}")
    if hasattr(env, "_check_success") and env._check_success():
        if return_joints:
            return episode, {"task_info": [task_xml, task_init_state], "joint_info": [desired_jps, gripper_actions]}
        else:
            return episode
    else:
        return (None, None) if return_joints else None


def collect_random_episode(env, render=False, return_joints=False):
    """随机策略采集一条（调试用）。"""
    obs = env.reset()

    # 锁定键集合与形状
    key_order, key_shapes = _init_key_profile(obs)
    obs_aligned = _align_obs(obs, key_order, key_shapes)

    if return_joints:
        task_xml = env.sim.model.get_xml()
        task_init_state = np.array(env.sim.get_state().flatten())

    desired_jps, gripper_actions = [], []

    done = False
    episode = {k: [obs_aligned[k]] for k in key_order}
    episode["action"] = []
    episode["reward"] = []

    while not done:
        action = np.random.uniform(low=-1, high=1, size=env.action_dim)
        action[0] += 0.04
        action = np.clip(action, -1, 1)
        obs, rew, done, info = env.step(action)

        obs_aligned = _align_obs(obs, key_order, key_shapes)

        if (obs_aligned["robot0_eef_pos"] < BOUND_MIN).any() or (obs_aligned["robot0_eef_pos"] > BOUND_MAX).any():
            return (None, None) if return_joints else None

        if render:
            env.render()

        desired_jps.append(env.robots[0]._joint_positions)
        gripper_actions.append(action[-1])

        _append_step(episode, obs_aligned, action, rew)

    if return_joints:
        return episode, {"task_info": [task_xml, task_init_state], "joint_info": [desired_jps, gripper_actions]}
    else:
        return episode


# ========= OSC → JV 回放 =========
def osc_to_jv(env_name, robots, num_episodes=64, render=False):
    controller_type = "OSC_POSE"

    if env_name == "Reach":
        policy_cls = ReachPolicy
        env_kwargs = {"horizon": 100, "table_full_size": (0.6, 0.6, 0.05)}
    elif env_name == "Lift":
        policy_cls = LiftPolicy
        env_kwargs = {"horizon": 200, "use_touch_obs": True, "table_offset": (0, 0, 0.908)}
    elif env_name == "PickPlaceBread":
        policy_cls = PickPlacePolicy
        env_kwargs = {
            "horizon": 200,
            "use_touch_obs": True,
            "bin1_pos": (-0.05, -0.25, 0.90),
            "bin2_pos": (-0.05, 0.28, 0.90),
        }
    elif env_name == "Stack":
        policy_cls = StackPolicy
        env_kwargs = {"horizon": 200, "use_touch_obs": True, "table_offset": (0, 0, 0.908)}
    else:
        raise ValueError(f"Unsupported env_name for osc_to_jv(): {env_name}")

    # 仅支持四种机器人↔夹爪（默认不再出现 xArm）
    if robots == "Panda":
        env_kwargs.setdefault("gripper_types", "PandaTouchGripper")
    elif robots == "Sawyer":
        env_kwargs.setdefault("gripper_types", "PandaTouchGripper")
    elif robots in ("xArm6", "xArm67", "xArm614"):
        env_kwargs.setdefault("gripper_types", "PandaTouchGripper")
    elif robots == "UR5e":
        env_kwargs.setdefault("gripper_types", "PandaTouchGripper")

    # 源环境（OSC）与目标环境（JV）
    env = utils.make_robosuite_env(env_name, robots, controller_type, render=render, **env_kwargs)
    policy = policy_cls(env)

    jv_env = utils.make_robosuite_env(
        env_name, robots=robots, controller_type="JOINT_VELOCITY", render=render, **env_kwargs
    )

    directory = pathlib.Path(f"./human_demonstrations/{env_name}/{robots}/JOINT_VELOCITY")
    directory.mkdir(parents=True, exist_ok=True)

    episodes_saved = 0
    while episodes_saved < num_episodes:
        reset_target = False
        episode = None
        ep_info = None

        # 采一条 OSC 演示（带回关节轨迹与初始状态）
        while episode is None:
            episode, ep_info = collect_human_episode(
                env, policy, render=render, return_joints=True, reset_target=reset_target
            )

        task_xml, task_init_state = ep_info["task_info"]
        desired_jps, gripper_actions = ep_info["joint_info"]

        # JV 环境复位到与 OSC 一致的初始状态
        jv_env.reset()
        jv_env.reset_from_xml_string(task_xml)
        jv_env.sim.reset()
        jv_env.sim.set_state_from_flattened(task_init_state)
        jv_env.sim.forward()

        obs = jv_env._get_observations(force_update=True)
        key_order, key_shapes = _init_key_profile(obs)
        obs_aligned = _align_obs(obs, key_order, key_shapes)

        episode = {k: [obs_aligned[k]] for k in key_order}
        episode["action"] = []
        episode["reward"] = []

        for i, (next_jp, gripper_action) in enumerate(zip(desired_jps, gripper_actions)):
            action = np.zeros(jv_env.robots[0].dof)
            action[-1] = gripper_action
            err = next_jp - jv_env.robots[0]._joint_positions

            kp = 20
            action[:-1] = np.clip(err * kp, -1, 1)

            obs, rew, done, info = jv_env.step(action)
            obs_aligned = _align_obs(obs, key_order, key_shapes)

            if (obs_aligned["robot0_eef_pos"] < BOUND_MIN).any() or (obs_aligned["robot0_eef_pos"] > BOUND_MAX).any():
                print(f"[osc_to_jv] Out of bounds at {obs_aligned['robot0_eef_pos']}")
                break

            if render:
                jv_env.render()

            _append_step(episode, obs_aligned, action, rew)

            if done:
                break

        print(f"[osc_to_jv] Episode {episodes_saved} return: {np.sum(episode['reward']):.2f}")

        # === 保存条件：回报阈值 > 110（按你的要求）===
        if np.sum(episode["reward"]) > 110:
            save_episode(directory, episode)
            episodes_saved += 1

    env.close()
    jv_env.close()


# ========= Debug/Probe 工具 =========
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


def probe_obs(env_name="Stack", robots="Panda", controller_type="JOINT_VELOCITY", steps=20, render=False, **env_kwargs):
    """快速打印当前任务可用 obs 键、形状与范围。"""
    env = utils.make_robosuite_env(env_name, robots, controller_type, render=render, **env_kwargs)
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


# ========= CLI =========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Stack",
                        help="Robosuite task (Reach / Lift / PickPlaceBread / Stack)")
    parser.add_argument("--robot", type=str, default="Panda", help="Robot to generate demonstrations")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of JV episodes to generate")
    parser.add_argument("--render", action="store_true", help="Enable on-screen rendering")
    args = parser.parse_args()

    # 探测可用键（可按需注释）
    common_kwargs = {"use_touch_obs": True}
    if args.env_name in ("Lift", "Stack"):
        common_kwargs["table_offset"] = (0, 0, 0.908)
    if args.robot == "Panda":
        common_kwargs["gripper_types"] = "PandaTouchGripper"
    elif args.robot == "Sawyer":
        common_kwargs["gripper_types"] = "PandaTouchGripper"
    elif args.robot in ("xArm6", "xArm67", "xArm614"):
        common_kwargs["gripper_types"] = "PandaTouchGripper"
    elif args.robot == "UR5e":
        common_kwargs["gripper_types"] = "PandaTouchGripper"

    probe_obs(args.env_name, args.robot, "JOINT_VELOCITY", steps=20, render=False, **common_kwargs)

    # 生成 JV 演示（由 OSC 轨迹回放得到）
    osc_to_jv(args.env_name, args.robot, args.num_episodes, render=args.render)