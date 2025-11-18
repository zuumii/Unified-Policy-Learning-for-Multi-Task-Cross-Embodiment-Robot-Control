# train_bc_reuse.py
import argparse
import time
import pathlib
from ruamel.yaml import YAML

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import utils
import replay_buffer
from bc import BCObsActAgent as Agent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='train config file path', required=True)
    return parser.parse_args()


def maybe_load_and_freeze(agent, ckpt_dir, device):
    """
    从 ckpt_dir 加载并冻结除 actor 以外的模块：
      obs_enc / obs_dec / act_enc / act_dec / inv_dyn / fwd_dyn  （存在则加载）
    冻结：requires_grad=False + eval()
    禁用：把自监督分支 update_dyn_cons 置空，并把 dyn_cons_update_freq 设很大
    """
    if ckpt_dir is None or str(ckpt_dir).lower() in ["none", "null", ""]:
        print("[pretrained] no checkpoint dir provided, skip loading.")
        return
    ckpt = pathlib.Path(ckpt_dir).resolve()
    print(f"[pretrained] load from: {ckpt}")

    def _load(module_name: str, filename: str, also_target: bool = False):
        if not hasattr(agent, module_name):
            return False
        f = ckpt / filename
        if not f.exists():
            print(f"[pretrained] {filename} not found, skip.")
            return False
        sd = torch.load(str(f), map_location=device)
        try:
            getattr(agent, module_name).load_state_dict(sd, strict=True)
            print(f"[pretrained] loaded {filename} into agent.{module_name}")
        except Exception as e:
            getattr(agent, module_name).load_state_dict(sd, strict=False)
            print(f"[pretrained] loaded {filename} with strict=False into agent.{module_name} ({e})")
        if also_target:
            tgt_name = module_name + "_target"
            if hasattr(agent, tgt_name):
                getattr(agent, tgt_name).load_state_dict(getattr(agent, module_name).state_dict(), strict=False)
                print(f"[pretrained] synced agent.{tgt_name} from agent.{module_name}")
        return True

    # 只加载非 actor 模块（按存在与否自适应）
    _load("obs_enc", "obs_enc.pt", also_target=False)
    _load("obs_dec", "obs_dec.pt", also_target=False)
    _load("inv_dyn", "inv_dyn.pt", also_target=False)
    _load("fwd_dyn", "fwd_dyn.pt", also_target=False)
    _load("act_enc", "act_enc.pt", also_target=False)
    _load("act_dec", "act_dec.pt", also_target=False)

    # 冻结这些模块（训练中不会更新它们）
    for name in ["obs_enc", "obs_dec", "inv_dyn", "fwd_dyn", "act_enc", "act_dec"]:
        if hasattr(agent, name):
            for p in getattr(agent, name).parameters():
                p.requires_grad_(False)
            getattr(agent, name).eval()
            print(f"[pretrained] froze agent.{name}")

    # 让这些模块的 optimizer 失效（即使被调用也不更新）
    for opt_name in ["obs_enc_opt", "obs_dec_opt", "inv_dyn_opt", "fwd_dyn_opt", "act_enc_opt", "act_dec_opt"]:
        if hasattr(agent, opt_name):
            for g in getattr(agent, opt_name).param_groups:
                g["lr"] = 0.0
            print(f"[pretrained] set lr=0 for {opt_name}")


    # 打印可训练参数做个 sanity check
    def _ntrainable(m):
        return sum(p.numel() for p in m.parameters() if getattr(p, "requires_grad", False))
    report = {}
    for name in ["actor", "obs_enc", "obs_dec", "act_enc", "act_dec", "inv_dyn", "fwd_dyn"]:
        if hasattr(agent, name):
            report[name] = _ntrainable(getattr(agent, name))
    print("[pretrained] trainable params:", report)


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
    obs0 = env_probe.reset()
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
    obs_dims = {
        'obs_dim': obs_shape[0],
        'robot_obs_dim': robot_obs_shape[0],
        'obj_obs_dim': obj_obs_shape[0],
        'lat_obs_dim': params['lat_obs_dim'],
    }
    act_dims = {
        'act_dim': act_shape[0],
        'lat_act_dim': params['lat_act_dim'],
    }

    # enc/dec/dynamics 用旧规模（和 ckpt 对齐），只放大 actor
    agent = Agent(
        obs_dims,
        act_dims,
        device,
        n_layers=params.get('n_layers', 3),
        hidden_dim=params.get('hidden_dim', 256),
        actor_n_layers=params.get('actor_n_layers'),
        actor_hidden_dim=params.get('actor_hidden_dim'),
    )

    # ------------------------------
    # replay buffer + demos
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
    else:
        print("[buffer] no demos loaded.")

    # ------------------------------
    # 载入并冻结（若给了 pretrained_from）
    # ------------------------------
    maybe_load_and_freeze(agent, params.get("pretrained_from", None), device)
    
    # —— 评估走确定性 ——（二选一）
    # A) 如果 utils.evaluate 支持 deterministic 参数：在调用处传 deterministic=True（见第 2 条）
    # B) 否则：在 train 脚本里给 agent 打个 predict 钩子，确保评估走确定性动作
    if not hasattr(agent, "predict"):
        agent.predict = lambda obs: (agent.sample_action(obs, deterministic=True), None)

    # —— BC 路径禁噪 ——（训练/采样默认不加噪；尤其抓手维绝不加噪）
    agent.expl_noise = 0.0

    # —— 抓手维 BC 权重（可调 2.0〜3.0）& 是否做不平衡修正（默认关）——
    if hasattr(agent, "bc_w_grip"):
        agent.bc_w_grip = 2.0
    if hasattr(agent, "bc_balance_gripper"):
        agent.bc_balance_gripper = False  # 若发现抓手开/关严重不平衡可改 True

    agent.dyn_cons_update_freq = params.get('dyn_cons_update_freq', getattr(agent, 'dyn_cons_update_freq', 1))

    # ☆ 冻结确认：只有 actor 仍可训练
    def _ntrainable(m):
        return sum(p.numel() for p in m.parameters() if getattr(p, "requires_grad", False))
    report = {}
    for name in ['actor', 'obs_enc', 'obs_dec', 'act_enc', 'act_dec', 'inv_dyn', 'fwd_dyn']:
        if hasattr(agent, name):
            report[name] = _ntrainable(getattr(agent, name))
    print("[freeze-check]", report)

    # ------------------------------
    # train loop
    # ------------------------------
    episode, episode_reward, done = 0, 0.0, True
    start_time = time.time()
    obs = env.reset()

    for step in range(params['total_timesteps']):
        # eval & save
        if step % params['evaluation']['interval'] == 0:
            print(f"[Step {step}] Evaluating...")
            agent.eval_mode()
            utils.evaluate(eval_env, agent, 4, logger, step)
            agent.train_mode()

        if step % params['evaluation']['save_interval'] == 0:
            print(f"[Step {step}] Saving model...")
            step_dir = model_dir / f"step_{step:07d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            agent.save(step_dir)

        # 这里只训练 BC，用数据更新
        agent.update(rb, logger, step)

    logger.close()


if __name__ == '__main__':
    main()