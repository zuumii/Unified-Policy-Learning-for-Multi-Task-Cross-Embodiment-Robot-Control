import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_layers, hidden_dim):
        super().__init__()
        self.trunk = utils.build_mlp(
            state_dim, action_dim,
            n_layers, hidden_dim,
            activation='relu',
            output_activation='tanh'
        )
        self.apply(utils.weight_init)

    def forward(self, state):
        return self.trunk(state)


class BCAgent:
    """Behavior cloning (plain)."""
    def __init__(
        self,
        obs_dims,
        act_dims,
        device,
        n_layers=3,
        hidden_dim=256,
        lr=3e-4,
        # 仅用于放大 Actor 的规模；不影响 enc/dec/dynamics
        actor_n_layers=None,
        actor_hidden_dim=None,
    ):
        self.log_freq = 1000
        # BC 路径禁噪；如需开启，仅对非抓手维加入极小噪声
        self.expl_noise = 0.0
        self.device = device
        self.dyn_cons_update_freq = 1

        self.obs_dim = obs_dims['obs_dim']
        self.act_dim = act_dims['act_dim']

        # 抓手维加权（不改接口：作为内部默认）
        self.bc_w_grip = 2.0
        self.bc_balance_gripper = False  # 若需类别不平衡修正，改为 True

        # 只对 Actor 使用独立规模
        a_n = actor_n_layers or n_layers
        a_h = actor_hidden_dim or hidden_dim

        self.actor = Actor(self.obs_dim, self.act_dim, a_n, a_h).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.modules = [self.actor]

    def _split_arm_grip(self, x: torch.Tensor):
        """将动作按 [手臂连续维, 抓手维] 切分；最后一维视为抓手。"""
        arm, grip = x[:, :-1], x[:, -1].unsqueeze(-1)
        return arm, grip

    def sample_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
            act = self.actor(obs)

        act = act.cpu().data.numpy().flatten()

        # BC 路径评估/采样默认禁噪；如需调试开启，仅对非抓手维加噪
        if (not deterministic) and (self.expl_noise is not None) and (self.expl_noise > 0.0):
            noise = np.zeros_like(act)
            if act.shape[0] >= 2:  # 仅对非抓手维加噪
                noise[:-1] = np.random.normal(0, self.expl_noise, size=act.shape[0]-1)
            act = np.clip(act + noise, -1, 1)
        return act

    def predict(self, obs):
        """供评估使用的确定性接口。"""
        return self.sample_action(obs, deterministic=True), None

    def update_actor(self, obs, act, L, step):
        pred_act = self.actor(obs)

        # 抓手加权的 BC 损失（不改接口）
        pred_arm, pred_grip = self._split_arm_grip(pred_act)
        act_arm, act_grip = self._split_arm_grip(act)

        arm_mse = F.mse_loss(pred_arm, act_arm)
        grip_mse = F.mse_loss(pred_grip, act_grip)

        # 可选：抓手类别不平衡修正（默认关闭）
        if self.bc_balance_gripper:
            with torch.no_grad():
                # 以 0 作为开/合阈值的简单近似（动作已 tanh 归一）
                open_ratio = (act_grip > 0).float().mean().clamp(1e-4, 1 - 1e-4)
                weight_open = (1.0 - open_ratio)
                weight_close = open_ratio
                w = torch.where(act_grip > 0, weight_open, weight_close)
            grip_mse = ((pred_grip - act_grip) ** 2 * w).mean()

        loss = arm_mse + self.bc_w_grip * grip_mse

        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_actor/actor_loss', loss.item(), step)
            L.add_scalar('train_actor/mse_arm', arm_mse.item(), step)
            L.add_scalar('train_actor/mse_gripper', grip_mse.item(), step)
            L.add_scalar('train_actor/pi_sq', (pred_act ** 2).mean().item(), step)

    def update(self, replay_buffer, L, step):
        obs, act, rew, next_obs, not_done = replay_buffer.sample()
        if step % self.log_freq == 0:
            L.add_scalar('train/batch_reward', rew.mean().item(), step)
        self.update_actor(obs, act, L, step)

    def save(self, model_dir):
        torch.save(self.actor.state_dict(), f'{model_dir}/actor.pt')

    def load(self, model_dir):
        self.actor.load_state_dict(torch.load(f'{model_dir}/actor.pt'))

    def eval_mode(self):
        for m in self.modules:
            m.eval()

    def train_mode(self):
        for m in self.modules:
            m.train()


class BCObsAgent(BCAgent):
    """
    BC with observation encoder + self-supervised dyn constraints.
    """
    def __init__(
        self,
        obs_dims,
        act_dims,
        device,
        n_layers=3,
        hidden_dim=256,
        lr=3e-4,
        # 仅用于放大 Actor 的规模；不影响 enc/dec/dynamics
        actor_n_layers=None,
        actor_hidden_dim=None,
    ):
        super().__init__(
            obs_dims, act_dims, device,
            n_layers=n_layers, hidden_dim=hidden_dim, lr=lr,
            actor_n_layers=actor_n_layers, actor_hidden_dim=actor_hidden_dim
        )
        self.batch_norm = False

        self.robot_obs_dim = obs_dims['robot_obs_dim']
        self.obj_obs_dim = obs_dims['obj_obs_dim']
        self.lat_obs_dim = obs_dims['lat_obs_dim']

        # enc/dec/dynamics 按原规模，确保可加载旧 ckpt
        self.obs_enc = utils.build_mlp(
            self.robot_obs_dim, self.lat_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh',
            batch_norm=self.batch_norm
        ).to(device)
        self.obs_enc_opt = torch.optim.Adam(self.obs_enc.parameters(), lr=3e-4)

        self.obs_dec = utils.build_mlp(
            self.lat_obs_dim, self.robot_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='identity',
            batch_norm=self.batch_norm
        ).to(device)
        self.obs_dec_opt = torch.optim.Adam(self.obs_dec.parameters(), lr=3e-4)

        self.inv_dyn = utils.build_mlp(
            self.lat_obs_dim * 2, self.act_dim - 1, n_layers, hidden_dim,
            activation='relu', output_activation='identity',
            batch_norm=self.batch_norm
        ).to(device)
        self.inv_dyn_opt = torch.optim.Adam(self.inv_dyn.parameters(), lr=3e-4)

        self.fwd_dyn = utils.build_mlp(
            self.lat_obs_dim + self.act_dim - 1, self.lat_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='identity',
            batch_norm=self.batch_norm
        ).to(device)
        self.fwd_dyn_opt = torch.optim.Adam(self.fwd_dyn.parameters(), lr=3e-4)

        # 只对 Actor 使用独立规模
        a_n = actor_n_layers or n_layers
        a_h = actor_hidden_dim or hidden_dim
        self.actor = Actor(self.lat_obs_dim + self.obj_obs_dim, self.act_dim, a_n, a_h).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.modules = [self.obs_enc, self.obs_dec, self.inv_dyn, self.fwd_dyn, self.actor]

    def sample_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
            robot_obs, obj_obs = obs[:, :self.robot_obs_dim], obs[:, self.robot_obs_dim:]
            lat_obs = self.obs_enc(robot_obs)
            act = self.actor(torch.cat([lat_obs, obj_obs], dim=-1))

        act = act.cpu().data.numpy().flatten()
        if (not deterministic) and (self.expl_noise is not None) and (self.expl_noise > 0.0):
            noise = np.zeros_like(act)
            if act.shape[0] >= 2:
                noise[:-1] = np.random.normal(0, self.expl_noise, size=act.shape[0]-1)
            act = np.clip(act + noise, -1, 1)
        return act

    def predict(self, obs):
        return self.sample_action(obs, deterministic=True), None

    def update_actor(self, obs, act, L, step):
        robot_obs, obj_obs = obs[:, :self.robot_obs_dim], obs[:, self.robot_obs_dim:]
        lat_obs = self.obs_enc(robot_obs)
        pred_act = self.actor(torch.cat([lat_obs, obj_obs], dim=-1))

        pred_arm, pred_grip = self._split_arm_grip(pred_act)
        act_arm, act_grip = self._split_arm_grip(act)

        arm_mse = F.mse_loss(pred_arm, act_arm)
        grip_mse = F.mse_loss(pred_grip, act_grip)

        if self.bc_balance_gripper:
            with torch.no_grad():
                open_ratio = (act_grip > 0).float().mean().clamp(1e-4, 1 - 1e-4)
                weight_open = (1.0 - open_ratio)
                weight_close = open_ratio
                w = torch.where(act_grip > 0, weight_open, weight_close)
            grip_mse = ((pred_grip - act_grip) ** 2 * w).mean()

        loss = arm_mse + self.bc_w_grip * grip_mse

        self.actor_opt.zero_grad()
        self.obs_enc_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        self.obs_enc_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_actor/actor_loss', loss.item(), step)
            L.add_scalar('train_actor/mse_arm', arm_mse.item(), step)
            L.add_scalar('train_actor/mse_gripper', grip_mse.item(), step)
            L.add_scalar('train_actor/lat_obs_sq', (lat_obs ** 2).mean().item(), step)
            L.add_scalar('train_actor/pi_sq', (pred_act ** 2).mean().item(), step)

    def update_dyn_cons(self, obs, act, next_obs, L, step):
        robot_obs, robot_next_obs = obs[:, :self.robot_obs_dim], next_obs[:, :self.robot_obs_dim]
        act_wo_g = act[:, :-1]
        lat_obs = self.obs_enc(robot_obs)
        lat_next_obs = self.obs_enc(robot_next_obs)

        pred_act = self.inv_dyn(torch.cat([lat_obs, lat_next_obs], dim=-1))
        inv_loss = F.mse_loss(pred_act, act_wo_g)

        pred_next_obs = self.fwd_dyn(torch.cat([lat_obs, act_wo_g], dim=-1))
        fwd_loss = F.mse_loss(pred_next_obs, lat_next_obs)

        recon_robot_obs = self.obs_dec(lat_obs)
        recon_loss = F.mse_loss(recon_robot_obs, robot_obs)

        loss = fwd_loss + inv_loss * 10 + recon_loss

        self.obs_enc_opt.zero_grad()
        self.obs_dec_opt.zero_grad()
        self.inv_dyn_opt.zero_grad()
        self.fwd_dyn_opt.zero_grad()
        loss.backward()
        self.obs_enc_opt.step()
        self.obs_dec_opt.step()
        self.inv_dyn_opt.step()
        self.fwd_dyn_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_dyn_cons/inv_loss', inv_loss.item(), step)
            L.add_scalar('train_dyn_cons/fwd_loss', fwd_loss.item(), step)
            L.add_scalar('train_dyn_cons/recon_loss', recon_loss.item(), step)
            L.add_scalar('train_dyn_cons/lat_obs_sq', (lat_obs**2).mean().item(), step)
            L.add_scalar('train_dyn_cons/pred_act_sq', (pred_act**2).mean().item(), step)

    def update(self, replay_buffer, L, step):
        obs, act, rew, next_obs, not_done = replay_buffer.sample()
        if step % self.log_freq == 0:
            L.add_scalar('train/batch_reward', rew.mean().item(), step)
        if step % self.dyn_cons_update_freq == 0:
            self.update_dyn_cons(obs, act, next_obs, L, step)
        self.update_actor(obs, act, L, step)

    def save(self, model_dir):
        super().save(model_dir)
        torch.save(self.obs_enc.state_dict(), f'{model_dir}/obs_enc.pt')
        torch.save(self.obs_dec.state_dict(), f'{model_dir}/obs_dec.pt')
        torch.save(self.inv_dyn.state_dict(), f'{model_dir}/inv_dyn.pt')
        torch.save(self.fwd_dyn.state_dict(), f'{model_dir}/fwd_dyn.pt')

    def load(self, model_dir):
        super().load(model_dir)
        self.obs_enc.load_state_dict(torch.load(f'{model_dir}/obs_enc.pt'))
        self.obs_dec.load_state_dict(torch.load(f'{model_dir}/obs_dec.pt'))
        self.inv_dyn.load_state_dict(torch.load(f'{model_dir}/inv_dyn.pt'))
        self.fwd_dyn.load_state_dict(torch.load(f'{model_dir}/fwd_dyn.pt'))


class BCObsActAgent(BCObsAgent):
    """
    BC with observation & action latents + self-supervised constraints.
    """
    def __init__(
        self,
        obs_dims,
        act_dims,
        device,
        n_layers=3,
        hidden_dim=256,
        lr=3e-4,
        # 仅用于放大 Actor 的规模；不影响 enc/dec/dynamics
        actor_n_layers=None,
        actor_hidden_dim=None,
    ):
        super().__init__(
            obs_dims, act_dims, device,
            n_layers=n_layers, hidden_dim=hidden_dim, lr=lr,
            actor_n_layers=actor_n_layers, actor_hidden_dim=actor_hidden_dim
        )

        self.lat_act_dim = act_dims['lat_act_dim']

        # enc/dec/dynamics 仍按原规模，保持与 ckpt 对齐
        self.obs_enc = utils.build_mlp(
            self.robot_obs_dim, self.lat_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh',
            batch_norm=self.batch_norm
        ).to(device)
        self.obs_enc_opt = torch.optim.Adam(self.obs_enc.parameters(), lr=lr)

        self.obs_dec = utils.build_mlp(
            self.lat_obs_dim, self.robot_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='identity',
            batch_norm=self.batch_norm
        ).to(device)
        self.obs_dec_opt = torch.optim.Adam(self.obs_dec.parameters(), lr=lr)

        self.act_enc = utils.build_mlp(
            self.robot_obs_dim + self.act_dim - 1, self.lat_act_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh',
            batch_norm=self.batch_norm
        ).to(device)
        self.act_enc_opt = torch.optim.Adam(self.act_enc.parameters(), lr=lr)

        self.act_dec = utils.build_mlp(
            self.robot_obs_dim + self.lat_act_dim, self.act_dim - 1, n_layers, hidden_dim,
            activation='relu', output_activation='tanh',
            batch_norm=self.batch_norm
        ).to(device)
        self.act_dec_opt = torch.optim.Adam(self.act_dec.parameters(), lr=lr)

        self.inv_dyn = utils.build_mlp(
            self.lat_obs_dim * 2, self.lat_act_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh',
            batch_norm=self.batch_norm
        ).to(device)
        self.inv_dyn_opt = torch.optim.Adam(self.inv_dyn.parameters(), lr=lr)

        self.fwd_dyn = utils.build_mlp(
            self.lat_obs_dim + self.lat_act_dim, self.lat_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh',
            batch_norm=self.batch_norm
        ).to(device)
        self.fwd_dyn_opt = torch.optim.Adam(self.fwd_dyn.parameters(), lr=lr)

        # 只对 Actor 使用独立规模；输出为 (lat_act_dim + 1)
        a_n = actor_n_layers or n_layers
        a_h = actor_hidden_dim or hidden_dim
        self.actor = Actor(
            self.lat_obs_dim + self.obj_obs_dim,
            self.lat_act_dim + 1,
            a_n, a_h
        ).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.modules = [
            self.obs_enc, self.obs_dec, self.act_enc,
            self.act_dec, self.inv_dyn, self.fwd_dyn, self.actor
        ]

    def sample_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
            robot_obs, obj_obs = obs[:, :self.robot_obs_dim], obs[:, self.robot_obs_dim:]
            lat_obs = self.obs_enc(robot_obs)
            lat_act_all = self.actor(torch.cat([lat_obs, obj_obs], dim=-1))
            lat_act, gripper_act = lat_act_all[:, :-1], lat_act_all[:, -1].unsqueeze(-1)
            act_wo_g = self.act_dec(torch.cat([robot_obs, lat_act], dim=-1))
            act = torch.cat([act_wo_g, gripper_act], dim=-1)

        act = act.cpu().data.numpy().flatten()
        if (not deterministic) and (self.expl_noise is not None) and (self.expl_noise > 0.0):
            noise = np.zeros_like(act)
            if act.shape[0] >= 2:
                noise[:-1] = np.random.normal(0, self.expl_noise, size=act.shape[0]-1)
            act = np.clip(act + noise, -1, 1)
        return act

    def predict(self, obs):
        return self.sample_action(obs, deterministic=True), None

    def update_actor(self, obs, act, L, step):
        robot_obs, obj_obs = obs[:, :self.robot_obs_dim], obs[:, self.robot_obs_dim:]
        lat_obs = self.obs_enc(robot_obs)
        lat_act_all = self.actor(torch.cat([lat_obs, obj_obs], dim=-1))
        lat_act, gripper_act = lat_act_all[:, :-1], lat_act_all[:, -1].unsqueeze(-1)
        pred_act_wo_g = self.act_dec(torch.cat([robot_obs, lat_act], dim=-1))
        pred_act = torch.cat([pred_act_wo_g, gripper_act], dim=-1)

        # 抓手加权的 BC 损失
        pred_arm, pred_grip = self._split_arm_grip(pred_act)
        act_arm, act_grip = self._split_arm_grip(act)

        arm_mse = F.mse_loss(pred_arm, act_arm)
        grip_mse = F.mse_loss(pred_grip, act_grip)

        if self.bc_balance_gripper:
            with torch.no_grad():
                open_ratio = (act_grip > 0).float().mean().clamp(1e-4, 1 - 1e-4)
                weight_open = (1.0 - open_ratio)
                weight_close = open_ratio
                w = torch.where(act_grip > 0, weight_open, weight_close)
            grip_mse = ((pred_grip - act_grip) ** 2 * w).mean()

        loss = arm_mse + self.bc_w_grip * grip_mse

        self.actor_opt.zero_grad()
        self.obs_enc_opt.zero_grad()
        self.act_dec_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        self.obs_enc_opt.step()
        self.act_dec_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_actor/actor_loss', loss.item(), step)
            L.add_scalar('train_actor/mse_arm', arm_mse.item(), step)
            L.add_scalar('train_actor/mse_gripper', grip_mse.item(), step)
            L.add_scalar('train_actor/lat_obs_sq', (lat_obs ** 2).mean().item(), step)
            L.add_scalar('train_actor/lat_act_sq', (lat_act ** 2).mean().item(), step)
            L.add_scalar('train_actor/pi_sq', (pred_act ** 2).mean().item(), step)

    def update_dyn_cons(self, obs, act, next_obs, L, step):
        robot_obs, robot_next_obs = obs[:, :self.robot_obs_dim], next_obs[:, :self.robot_obs_dim]
        act_wo_g = act[:, :-1]

        lat_obs = self.obs_enc(robot_obs)
        lat_next_obs = self.obs_enc(robot_next_obs)

        # inverse: (z, z') -> z_a -> decode -> a_wo_g
        pred_lat_act = self.inv_dyn(torch.cat([lat_obs, lat_next_obs], dim=-1))
        pred_act_wo_g = self.act_dec(torch.cat([robot_obs, pred_lat_act], dim=-1))
        inv_loss = F.mse_loss(pred_act_wo_g, act_wo_g)

        # forward: (z, z_a) -> z'
        lat_act = self.act_enc(torch.cat([robot_obs, act_wo_g], dim=-1))
        pred_next_z = self.fwd_dyn(torch.cat([lat_obs, lat_act], dim=-1))
        fwd_loss = F.mse_loss(pred_next_z, lat_next_obs)

        # recon
        recon_robot = self.obs_dec(lat_obs)
        recon_obs_loss = F.mse_loss(recon_robot, robot_obs)
        recon_act = self.act_dec(torch.cat([robot_obs, lat_act], dim=-1))
        recon_act_loss = F.mse_loss(recon_act, act_wo_g)

        loss = fwd_loss + inv_loss + recon_obs_loss + recon_act_loss

        self.obs_enc_opt.zero_grad()
        self.obs_dec_opt.zero_grad()
        self.act_enc_opt.zero_grad()
        self.act_dec_opt.zero_grad()
        self.inv_dyn_opt.zero_grad()
        self.fwd_dyn_opt.zero_grad()
        loss.backward()
        self.obs_enc_opt.step()
        self.obs_dec_opt.step()
        self.act_enc_opt.step()
        self.act_dec_opt.step()
        self.inv_dyn_opt.step()
        self.fwd_dyn_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_dyn_cons/inv_loss', inv_loss.item(), step)
            L.add_scalar('train_dyn_cons/fwd_loss', fwd_loss.item(), step)
            L.add_scalar('train_dyn_cons/recon_obs_loss', recon_obs_loss.item(), step)
            L.add_scalar('train_dyn_cons/recon_act_loss', recon_act_loss.item(), step)
            L.add_scalar('train_dyn_cons/lat_obs_sq', (lat_obs**2).mean().item(), step)
            L.add_scalar('train_dyn_cons/lat_act_sq', (lat_act**2).mean().item(), step)
            L.add_scalar('train_dyn_cons/pred_act_sq', (pred_act_wo_g**2).mean().item(), step)

    def update(self, replay_buffer, L, step):
        obs, act, rew, next_obs, not_done = replay_buffer.sample()
        if step % self.log_freq == 0:
            L.add_scalar('train/batch_reward', rew.mean().item(), step)
        if step % self.dyn_cons_update_freq == 0:
            self.update_dyn_cons(obs, act, next_obs, L, step)
        self.update_actor(obs, act, L, step)

    def save(self, model_dir):
        super().save(model_dir)
        torch.save(self.act_enc.state_dict(), f'{model_dir}/act_enc.pt')
        torch.save(self.act_dec.state_dict(), f'{model_dir}/act_dec.pt')

    def load(self, model_dir):
        super().load(model_dir)
        self.act_enc.load_state_dict(torch.load(f'{model_dir}/act_enc.pt'))
        self.act_dec.load_state_dict(torch.load(f'{model_dir}/act_dec.pt'))