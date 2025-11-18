# train_rl_reuse.py
import os
import time
import pathlib
import argparse
from ruamel.yaml import YAML

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import utils
import replay_buffer
from td3_fixed import TD3Agent, TD3ObsAgent, TD3ObsActAgent

# ===== 安全 reset（仅为避免 robosuite 随机摆放失败抛 RandomizationError）=====
try:
    from robosuite.utils.errors import RandomizationError
except Exception:
    class RandomizationError(Exception):
        pass

def safe_reset(env, max_tries=50, reseed=True, sleep=0.0):
    """对 env.reset() 加壳，摆放失败时重试；必要时轻扰动随机种子。"""
    tries = 0
    while True:
        try:
            return env.reset()
        except RandomizationError:
            tries += 1
            if tries >= max_tries:
                raise
            if reseed and hasattr(env, "seed"):
                env.seed(int(time.time() * 1e6) % (2**31 - 1))
            if sleep > 0:
                time.sleep(sleep)
            continue


# ---------- helpers ----------
class _NoOpOptim:
    def zero_grad(self): pass
    def step(self): pass

def _load_state_if_exists(module, path, map_location='cpu', strict=False):
    if os.path.isfile(path):
        sd = torch.load(path, map_location=map_location)
        module.load_state_dict(sd, strict=strict)
        return True
    return False

def reuse_and_freeze_modules(agent, ckpt_dir, device):
    """
    读取并冻结：obs_enc / obs_dec / act_enc / act_dec / inv_dyn / fwd_dyn（按存在与否自适应）。
    同时把相关 target（obs_enc_target, act_dec_target）同步为相同权重并冻结。
    禁用自监督 dyn_cons 支路与这些模块的优化器。
    """
    if not ckpt_dir:
        print("[reuse] no ckpt dir provided, skip.")
        return
    print(f"[reuse] load from: {ckpt_dir}")

    loaded = {}

    def _try(module_name, filename, strict=False):
        if hasattr(agent, module_name):
            ok = _load_state_if_exists(getattr(agent, module_name),
                                       os.path.join(ckpt_dir, filename),
                                       map_location=device, strict=strict)
            loaded[module_name] = ok

    # 1) 加载
    _try('obs_enc', 'obs_enc.pt', strict=False)
    _try('obs_dec', 'obs_dec.pt', strict=False)
    _try('act_enc', 'act_enc.pt', strict=False)
    _try('act_dec', 'act_dec.pt', strict=False)
    _try('inv_dyn', 'inv_dyn.pt', strict=False)
    _try('fwd_dyn', 'fwd_dyn.pt', strict=False)

    print("[reuse] loaded:", {k: bool(v) for k, v in loaded.items()})

    # 2) 同步 target
    if hasattr(agent, 'obs_enc_target') and hasattr(agent, 'obs_enc') and loaded.get('obs_enc', False):
        agent.obs_enc_target.load_state_dict(agent.obs_enc.state_dict(), strict=False)
    if hasattr(agent, 'act_dec_target') and hasattr(agent, 'act_dec') and loaded.get('act_dec', False):
        agent.act_dec_target.load_state_dict(agent.act_dec.state_dict(), strict=False)

    # 3) 冻结主模块 + target
    def _freeze(m):
        if m is None: return
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()

    for name in ['obs_enc', 'obs_dec', 'act_enc', 'act_dec', 'inv_dyn', 'fwd_dyn',
                 'obs_enc_target', 'act_dec_target']:
        if hasattr(agent, name):
            _freeze(getattr(agent, name))
            print(f"[reuse] froze agent.{name}")

    # 4) 禁用自监督分支 + 替换优化器
    if hasattr(agent, 'update_dyn_cons'):
        agent.update_dyn_cons = lambda *args, **kwargs: None
    if hasattr(agent, 'dyn_cons_update_freq'):
        agent.dyn_cons_update_freq = 10**12  # 基本不触发

    for opt_name in ['obs_enc_opt', 'obs_dec_opt', 'act_enc_opt', 'act_dec_opt', 'inv_dyn_opt', 'fwd_dyn_opt']:
        if hasattr(agent, opt_name):
            setattr(agent, opt_name, _NoOpOptim())
            print(f"[reuse] replaced {opt_name} with NoOpOptim")

    # 5) 打印可训练参数
    def ntrainable(mod):
        return sum(p.numel() for p in mod.parameters() if p.requires_grad)
    report = {}
    for name in ['actor','critic','obs_enc','obs_dec','act_enc','act_dec','inv_dyn','fwd_dyn']:
        if hasattr(agent, name):
            report[name] = ntrainable(getattr(agent, name))
    print("[reuse] trainable params:", report)


# ---------- SEW predictor 动态结构工具（用于外来模型） ----------
def load_state_dict_flexible(model_path: str):
    """兼容直接 state_dict 或 {'state_dict': ...}"""
    obj = torch.load(model_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError("不支持的模型文件格式：既不是 state_dict，也不是包含 'state_dict' 的字典。")

def _extract_layer_indices_from_keys(state_keys):
    """从 'net.{i}.weight' 提取 i，排序去重，仅统计 Linear 层"""
    idxs = []
    for k in state_keys:
        if k.startswith("net.") and k.endswith(".weight"):
            try:
                idxs.append(int(k.split(".")[1]))
            except Exception:
                pass
    return sorted(set(idxs))

def infer_mlp_spec_from_state(state):
    """
    从 state_dict 反推：
      - in_dim（首层 Linear 的 in_features）
      - linear_out_dims（每个 Linear 的 out_features 列表；末层应为 1）
    假设结构：net.0 Linear, net.1 ReLU, net.2 Linear, ..., net.N Linear, net.N+1 Tanh
    """
    keys = list(state.keys())
    linear_idxs = _extract_layer_indices_from_keys(keys)
    if not linear_idxs:
        raise RuntimeError("未找到任何 Linear 层权重（形如 'net.{i}.weight'）。")

    w0 = state[f"net.{linear_idxs[0]}.weight"]
    if w0.ndim != 2:
        raise RuntimeError("首层 Linear weight 不是二维。")
    in_dim = int(w0.shape[1])

    linear_out_dims = []
    prev_out = in_dim
    for li in linear_idxs:
        w = state[f"net.{li}.weight"]
        if w.ndim != 2:
            raise RuntimeError(f"net.{li}.weight 不是二维。")
        out_dim = int(w.shape[0])
        if int(w.shape[1]) != prev_out:
            raise RuntimeError(
                f"层维度不一致：net.{li}.weight 的 in_features={int(w.shape[1])} 与上一层输出 {prev_out} 不匹配。"
            )
        linear_out_dims.append(out_dim)
        prev_out = out_dim

    return in_dim, linear_out_dims

class _SequentialSew(nn.Module):
    """
    按给定 spec 构建：Linear(+ReLU)* (L-1) + Linear(+Tanh)
    linear_out_dims: List[int] —— 每个 Linear 的输出维度（最后一个应为 1）
    """
    def __init__(self, in_dim: int, linear_out_dims):
        super().__init__()
        layers = []
        d_in = int(in_dim)
        L = len(linear_out_dims)
        assert L >= 1, "网络至少需要一层 Linear"
        for i, d_out in enumerate(linear_out_dims):
            layers.append(nn.Linear(d_in, int(d_out)))
            if i < L - 1:
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.Tanh())
            d_in = int(d_out)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def build_sew_predictor_from_spec(state):
    in_dim, linear_out_dims = infer_mlp_spec_from_state(state)
    model = _SequentialSew(in_dim, linear_out_dims)
    model.load_state_dict(state, strict=True)
    arch = {"in_dim": in_dim, "depth": len(linear_out_dims), "out_dims": linear_out_dims}
    return model, arch


# ---------- SEW predictor （与 BC 对齐的极简结构；保留以兼容，但外来模型分支不会用到它） ----------
class SewPredictor(nn.Module):
    """
    输入：q_angles[:-1] ；输出：cos(theta_sew) ∈ [-1,1]（tanh）
    """
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Tanh()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------- main ----------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='train config file path')
    return parser.parse_args()

def main():
    args = parse_args()
    yaml = YAML(typ='safe')
    params = yaml.load(open(args.config))

    # ------------------------------
    # logging dirs
    # ------------------------------
    demo_dir = None
    if params.get('expert_folder') is not None:
        demo_dir = (pathlib.Path(params['expert_folder']) /
                    params['env_name'] / params['robots'] / params['controller_type']).resolve()

    if params.get('logdir_prefix') is None:
        logdir_prefix = pathlib.Path(__file__).parent
    else:
        logdir_prefix = pathlib.Path(params['logdir_prefix'])
    data_path = (logdir_prefix / 'logs' / time.strftime("%m.%d.%Y")).resolve()
    logdir = '_'.join([
        time.strftime("%H-%M-%S"),
        params['env_name'],
        params['robots'],
        params['controller_type'],
        params.get('suffix', '')
    ])
    logdir = (data_path / logdir).resolve()
    params['logdir'] = str(logdir)
    print("======== Train Params ========")
    print(params)

    logdir.mkdir(parents=True, exist_ok=True)
    import yaml as pyyaml
    with open(logdir / 'params.yml', 'w') as fp:
        pyyaml.safe_dump(params, fp, sort_keys=False)

    model_dir = (logdir / 'models').resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    replay_buffer_dir = (logdir / 'replay_buffer').resolve()
    if params.get('save_buffer', False):
        replay_buffer_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # env + dims
    # ------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # probe 原始 robosuite obs，得到各子向量长度
    env_probe = utils.make_robosuite_env(
        params['env_name'],
        robots=params['robots'],
        controller_type=params['controller_type'],
        **params['env_kwargs']
    )
    obs0 = safe_reset(env_probe, max_tries=50, reseed=True)  # ✅ 安全 reset
    robot_obs_shape = np.concatenate([obs0[k] for k in params['robot_obs_keys']]).shape
    obj_obs_shape = np.concatenate([obs0[k] for k in params['obj_obs_keys']]).shape

    params['obs_keys'] = params['robot_obs_keys'] + params['obj_obs_keys']

    # 训练 env（按 obs_keys 打包）
    env = utils.make(
        params['env_name'],
        robots=params['robots'],
        controller_type=params['controller_type'],
        obs_keys=params['obs_keys'],
        seed=params['seed'],
        **params['env_kwargs'],
    )
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    print(f"Environment observation space shape {obs_shape}")
    print(f"Environment action space shape {act_shape}")

    eval_env = utils.make(
        params['env_name'],
        robots=params['robots'],
        controller_type=params['controller_type'],
        obs_keys=params['obs_keys'],
        seed=params['seed'] + 100,
        **params['env_kwargs'],
    )

    logger = SummaryWriter(log_dir=params['logdir'])

    # ------------------------------
    # build agent
    # ------------------------------
    # 支持两种字段名：'agent' 或 'agent_type'
    agent_name = (params.get('agent') or params.get('agent_type') or 'obsact').lower()
    agent_cls_map = {
        'plain': TD3Agent,
        'td3': TD3Agent,
        'obs': TD3ObsAgent,
        'td3_obs': TD3ObsAgent,
        'obsact': TD3ObsActAgent,
        'td3_obs_act': TD3ObsActAgent,
        'td3_obsact': TD3ObsActAgent,
    }
    agent_cls = agent_cls_map.get(agent_name, TD3ObsActAgent)

    # 维度
    obs_dims = {
        'obs_dim': obs_shape[0],
        'robot_obs_dim': robot_obs_shape[0],
        'obj_obs_dim': obj_obs_shape[0],
        'lat_obs_dim': params.get('lat_obs_dim', 0),
    }
    act_dims = {
        'act_dim': act_shape[0],
        'lat_act_dim': params.get('lat_act_dim', 0),
    }

    # 统一的超参（传到 TD3*Agent）
    agent_kwargs = dict(
        device=device,
        discount=params.get('discount', 0.99),
        tau=params.get('tau', 0.005),
        policy_noise=params.get('policy_noise', 0.2),
        noise_clip=params.get('noise_clip', 0.5),
        expl_noise=params.get('expl_noise', 0.1),
        n_layers=params.get('n_layers', 3),         # backbone 层数（enc/dec/critic）
        hidden_dim=params.get('hidden_dim', 256),    # backbone 宽度
        lr=params.get('lr', 3e-4),
        # 只放大 Actor
        actor_hidden_dim=params.get('actor_hidden_dim', None),
        actor_n_layers=params.get('actor_n_layers', None),
    )

    # 实例化：plain 版传标量维度，其余传 dict
    if agent_cls is TD3Agent:
        agent = agent_cls(obs_shape[0], act_shape[0], **agent_kwargs)
    else:
        # 确保有 lat_* 维度
        if obs_dims['lat_obs_dim'] == 0 and agent_cls is not TD3Agent:
            raise ValueError("lat_obs_dim is required for obs/obsact agents")
        if agent_cls is TD3ObsActAgent and act_dims['lat_act_dim'] == 0:
            raise ValueError("lat_act_dim is required for obsact agent")
        agent = agent_cls(obs_dims, act_dims, **agent_kwargs)

    # 复用并冻结（若给了 pretrained_from）
    reuse_and_freeze_modules(agent, params.get('pretrained_from'), device)

    # ------------------------------
    # attach SEW predictor to agent（仅 Obs / ObsAct，plain 不动）
    # ------------------------------
    sew_cfg = params.get('sew', {}) or {}
    if hasattr(agent, 'robot_obs_dim') and bool(sew_cfg.get('enabled', False)):
        n = int(agent.robot_obs_dim // 2)
        in_dim_expected = max(n - 1, 1)
        hidden = int(((sew_cfg.get('model') or {}).get('hidden', 256)))
        model_path = (sew_cfg.get('model') or {}).get('path', None)

        if model_path and os.path.isfile(model_path):
            # === 外来模型：自动解析结构 + 严格检查 in_dim ===
            state = load_state_dict_flexible(model_path)
            sew_model_dyn, arch = build_sew_predictor_from_spec(state)

            if int(arch["in_dim"]) != int(in_dim_expected):
                raise RuntimeError(
                    f"[SEW] 外来模型输入维度不匹配：arch.in_dim={arch['in_dim']}，预期 in_dim=n-1={in_dim_expected}（n={n}）。\n"
                    f"请使用与当前 robot_obs 关节数一致训练得到的模型，或切换到对应的 Robot/观测配置。"
                )

            print(f"[SEW] External model arch -> in_dim={arch['in_dim']}, depth={arch['depth']}, out_dims={arch['out_dims']}")

            sew_model = sew_model_dyn.to(device)
            for p in sew_model.parameters():
                p.requires_grad_(False)
            sew_model.eval()

            # 将 SEW 参数挂到 agent（td3_fixed 里默认有同名属性并设了安全默认值）
            agent.sew_enabled = True
            agent.sew_lambda = float(sew_cfg.get('lambda', 0.0))
            agent.sew_action_scale = float(sew_cfg.get('action_scale', 1.0))
            hz = float(sew_cfg.get('hz', 20.0))
            agent.sew_dt = 1.0 / max(hz, 1e-6)
            agent.sew_predictor = sew_model
            print(f"[SEW] enabled. model={model_path} | λ={agent.sew_lambda} | dt={agent.sew_dt:.6f} | scale={agent.sew_action_scale}")

        else:
            print("[SEW] enabled in config but no valid model path given -> SEW disabled.")
            agent.sew_enabled = False
            agent.sew_predictor = None
    else:
        print("[SEW] disabled or agent has no robot_obs_dim (plain TD3).")

    # ------------------------------
    # replay buffer
    # ------------------------------
    rb = replay_buffer.ReplayBuffer(
        obs_shape=obs_shape,
        action_shape=act_shape,
        capacity=params.get('replay_capacity', 2_000_000),
        batch_size=params['batch_size'],
        device=device
    )
    if demo_dir is not None and demo_dir.exists():
        demo_paths = utils.load_episodes(demo_dir, params['obs_keys'])
        rb.add_rollouts(demo_paths)
        print(f"[buffer] loaded demos from {demo_dir}")

    # ------------------------------
    # train loop
    # ------------------------------
    episode, episode_reward, done = 0, 0.0, True
    start_time = time.time()
    obs = safe_reset(env, max_tries=50, reseed=True)  # ✅ 安全 reset

    for step in range(params['total_timesteps']):
        # eval & save
        if step % params['evaluation']['interval'] == 0:
            print(f"[Step {step}] Evaluating...")
            utils.evaluate(eval_env, agent, 4, logger, step)

        if step % params['evaluation']['save_interval'] == 0:
            print(f"[Step {step}] Saving model...")
            step_dir = model_dir / f"step_{step:07d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            agent.save(step_dir)
            if params.get('save_buffer', False):
                rb.save(replay_buffer_dir)

        # act
        action = agent.sample_action(obs)  # 默认带探索噪声
        next_obs, reward, done, info = env.step(action)

        # not_done（避免把超时截断当成真正终止）
        timeout = False
        if isinstance(info, dict):
            timeout = bool(
                info.get("TimeLimit.truncated", False) or
                info.get("TimeLimit.truncation", False) or
                info.get("truncated", False)
            )
        not_done = 0.0 if (done and not timeout) else 1.0

        episode_reward += float(reward)
        rb.add(obs, action, reward, next_obs, not_done)

        # update
        agent.update(rb, logger, step)

        # episode end
        if done:
            logger.add_scalar('train/episode_reward', episode_reward, step)
            logger.add_scalar('train/duration', time.time() - start_time, step)
            start_time = time.time()
            logger.add_scalar('train/episode', episode, step)
            logger.flush()

            obs = safe_reset(env, max_tries=50, reseed=True)  # ✅ 安全 reset
            episode_reward = 0.0
            episode += 1
        else:
            obs = next_obs

    logger.close()


if __name__ == '__main__':
    main()