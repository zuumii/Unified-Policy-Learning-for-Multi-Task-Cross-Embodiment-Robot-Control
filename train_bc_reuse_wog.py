import argparse
import time
import pathlib
from ruamel.yaml import YAML

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import utils
import replay_buffer
from bc_wog import BCObsActAgent as Agent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='train config file path', required=True)
    return parser.parse_args()


def maybe_load_and_freeze(agent, ckpt_dir, device, pretrained_cfg: dict = None, gripper_cfg: dict = None):
    """
    从 ckpt_dir 加载并冻结除 actor 以外的模块：
      obs_enc / obs_dec / act_enc / act_dec / inv_dyn / fwd_dyn （若存在）
    夹爪头 grip_head 是否加载/是否冻结由 pretrained_cfg 控制：
      - pretrained_cfg.load_grip_head: bool
      - pretrained_cfg.freeze_grip_head: bool
    """
    pretrained_cfg = pretrained_cfg or {}
    load_grip = bool(pretrained_cfg.get("load_grip_head", True))
    freeze_grip = bool(pretrained_cfg.get("freeze_grip_head", False))

    if ckpt_dir is None or str(ckpt_dir).lower() in ["none", "null", ""]:
        print("[pretrained] no checkpoint dir provided, skip loading.")
        # 如果不从 ckpt 加载，又想训练 grip_head，则确保有优化器
        if getattr(agent, "grip_head", None) is not None and hasattr(agent, "grip_opt") is False:
            lr = float((gripper_cfg or {}).get("lr", 1e-3))
            agent.grip_opt = torch.optim.Adam(agent.grip_head.parameters(), lr=lr)
            print(f"[pretrained] created grip_opt from scratch, lr={lr}")
        return

    ckpt = pathlib.Path(ckpt_dir).resolve()
    print(f"[pretrained] load from: {ckpt}")

    def _load(module_name: str, filename: str, strict: bool = True):
        if not hasattr(agent, module_name):
            return False
        f = ckpt / filename
        if not f.exists():
            print(f"[pretrained] {filename} not found, skip.")
            return False
        sd = torch.load(str(f), map_location=device)
        try:
            getattr(agent, module_name).load_state_dict(sd, strict=strict)
            print(f"[pretrained] loaded {filename} into agent.{module_name} (strict={strict})")
        except Exception as e:
            # 宽松兜底
            getattr(agent, module_name).load_state_dict(sd, strict=False)
            print(f"[pretrained] loaded {filename} with strict=False into agent.{module_name} ({e})")
        return True

    # —— 加载非 actor 模块并冻结 —— 
    for name, file in [
        ("obs_enc", "obs_enc.pt"),
        ("obs_dec", "obs_dec.pt"),
        ("inv_dyn", "inv_dyn.pt"),
        ("fwd_dyn", "fwd_dyn.pt"),
        ("act_enc", "act_enc.pt"),
        ("act_dec", "act_dec.pt"),
    ]:
        ok = _load(name, file, strict=True)
        if ok:
            for p in getattr(agent, name).parameters():
                p.requires_grad_(False)
            getattr(agent, name).eval()
            print(f"[pretrained] froze agent.{name}")

    # —— 夹爪头：是否加载由开关决定 —— 
    if load_grip and hasattr(agent, "grip_head") and agent.grip_head is not None:
        _load("grip_head", "grip_head.pt", strict=False)
        if freeze_grip:
            for p in agent.grip_head.parameters():
                p.requires_grad_(False)
            agent.grip_head.eval()
            print("[pretrained] froze agent.grip_head")
        else:
            # 确保有优化器
            if not hasattr(agent, "grip_opt") or agent.grip_opt is None:
                lr = float((gripper_cfg or {}).get("lr", 1e-3))
                agent.grip_opt = torch.optim.Adam(agent.grip_head.parameters(), lr=lr)
                print(f"[pretrained] created grip_opt for loaded grip_head, lr={lr}")
    else:
        print("[pretrained] grip_head not loaded (by config or missing). "
              "It will be trained from scratch if enabled.")

    # —— 让这些模块的 optimizer 失效（若外部误创建） —— 
    for opt_name in ["obs_enc_opt", "obs_dec_opt", "inv_dyn_opt", "fwd_dyn_opt", "act_enc_opt", "act_dec_opt"]:
        if hasattr(agent, opt_name) and getattr(agent, opt_name) is not None:
            for g in getattr(agent, opt_name).param_groups:
                g["lr"] = 0.0
            print(f"[pretrained] set lr=0 for {opt_name}")

    # —— 关闭自监督分支 —— 
    if hasattr(agent, "update_dyn_cons"):
        agent.update_dyn_cons = lambda *args, **kwargs: None
    if hasattr(agent, "dyn_cons_update_freq"):
        agent.dyn_cons_update_freq = 10**12

    # —— 打印可训练参数 —— 
    def _ntrainable(m):
        return sum(p.numel() for p in m.parameters() if getattr(p, "requires_grad", False))
    report = {}
    for name in ["actor", "grip_head", "obs_enc", "obs_dec", "act_enc", "act_dec", "inv_dyn", "fwd_dyn"]:
        if hasattr(agent, name) and getattr(agent, name) is not None:
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
    pre_cfg = (params.get("pretrained") or {})
    grip_cfg = (params.get("gripper") or {})
    maybe_load_and_freeze(agent, params.get("pretrained_from", None), device,
                        pretrained_cfg=pre_cfg, gripper_cfg=grip_cfg)
    
    # —— 评估走确定性 ——（向下兼容）
    if not hasattr(agent, "predict"):
        agent.predict = lambda obs: (agent.sample_action(obs, deterministic=True), None)

    # —— BC 路径禁噪 ——（训练/采样默认不加噪；尤其抓手维绝不加噪）
    agent.expl_noise = 0.0

    # —— gripper 配置：默认 lr=0.001, update_every=1, 启用外接夹爪 —— 
    grip_cfg = (params.get("gripper") or {})
    agent.use_external_gripper = bool(grip_cfg.get("enabled", True))
    # 若存在 grip_opt，设置 lr（没有就跳过）
    if hasattr(agent, "grip_opt"):
        lr = float(grip_cfg.get("lr", 1e-3))
        for g in agent.grip_opt.param_groups:
            g["lr"] = lr
        agent.grip_update_every = int(grip_cfg.get("update_every", 1))
        print(f"[gripper] enabled={agent.use_external_gripper}, lr={lr}, update_every={agent.grip_update_every}")

    # —— 若 enc/dec/dyn 已被冻结：自监督分支频率已在 maybe_load_and_freeze 中被抬高 —— 

    # ☆ 冻结确认：应只有 actor 与 grip_head 可训练（除非你不加载/不冻结）
    def _ntrainable(m):
        return sum(p.numel() for p in m.parameters() if getattr(p, "requires_grad", False))
    report = {}
    for name in ['actor', 'grip_head', 'obs_enc', 'obs_dec', 'act_enc', 'act_dec', 'inv_dyn', 'fwd_dyn']:
        if hasattr(agent, name):
            report[name] = _ntrainable(getattr(agent, name))
    print("[freeze-check]", report)

    # ------------------------------
    # train loop
    # ------------------------------
    start_time = time.time()

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

        # 这里只训练 BC
        agent.update(rb, logger, step)

    logger.close()


if __name__ == '__main__':
    main()