# td3_fixed.py — TD3 with SEW representation alignment (behavior term removed)

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


# -----------------------------
# Basic building blocks
# -----------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_layers, hidden_dim):
        super().__init__()
        self.trunk = utils.build_mlp(
            state_dim, action_dim,
            n_layers, hidden_dim,
            activation='relu',
            output_activation='tanh',
            batch_norm=False  # Actor 一般不做 BN
        )
        self.apply(utils.weight_init)

    def forward(self, state):
        return self.trunk(state)


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, state_dim, action_dim, n_layers, hidden_dim):
        super().__init__()
        self.trunk = utils.build_mlp(
            state_dim + action_dim, 1, n_layers, hidden_dim,
            activation='relu',
            output_activation='identity',
            batch_norm=False  # Critic 中也不建议 BN
        )
        self.apply(utils.weight_init)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        return self.trunk(state_action)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_layers, hidden_dim):
        super().__init__()
        self.Q1 = QFunction(state_dim, action_dim, n_layers, hidden_dim)
        self.Q2 = QFunction(state_dim, action_dim, n_layers, hidden_dim)

    def forward(self, state, action):
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        return q1, q2


# -----------------------------
# Helpers
# -----------------------------
def _robotobs_to_q(robot_obs: torch.Tensor):
    """
    robot_obs: (B, 2n) = [cos(n), sin(n)]
    return:
      q:           (B, n)
      q_no_last:   (B, n-1)
    """
    B, D = robot_obs.shape
    assert D % 2 == 0, "robot_obs must be [cos(n), sin(n)]"
    n = D // 2
    cos = robot_obs[:, :n]
    sin = robot_obs[:, n:2*n]
    q = torch.atan2(sin, cos)
    return q, q[:, :-1]


# -----------------------------
# Plain TD3
# -----------------------------
class TD3Agent:
    """TD3 algorithm (plain)."""
    def __init__(
        self,
        obs_dim,
        act_dim,
        device,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        expl_noise=0.1,
        n_layers=3,
        hidden_dim=256,
        lr=3e-4,
        # 仅用于放大 Actor 的规模；不影响 Critic
        actor_n_layers=None,
        actor_hidden_dim=None,
    ):
        self.device = device
        self.tau = tau
        self.discount = discount
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.expl_noise = expl_noise
        self.log_freq = 1000

        self.critic_update_freq = 1
        self.actor_update_freq = 2

        # 只对 Actor 使用独立规模
        a_n = actor_n_layers or n_layers
        a_h = actor_hidden_dim or hidden_dim

        self.actor = Actor(obs_dim, act_dim, a_n, a_h).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(obs_dim, act_dim, n_layers, hidden_dim).to(device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Target networks
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        for p in self.actor_target.parameters():
            p.requires_grad_(False)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # ---- SEW 默认属性（plain 版不使用）----
        self.sew_enabled = False
        self.sew_lambda = 0.0
        self.sew_predictor = None

    # ----- utils -----
    @staticmethod
    def _soft_update(src: nn.Module, tgt: nn.Module, tau: float):
        for p, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    # ----- acting -----
    def sample_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
            act = self.actor(obs)
            act = act.cpu().data.numpy().flatten()
        if not deterministic:
            act += np.random.normal(0, self.expl_noise, size=act.shape[0])
            act = np.clip(act, -1, 1)
        return act

    def predict(self, obs):
        return self.sample_action(obs, deterministic=True), None

    # ----- learning -----
    def update_actor_critic(self, obs, act, reward, next_obs, not_done, L, step):
        # --- Critic ---
        if self.critic_update_freq > 0 and step % self.critic_update_freq == 0:
            with torch.no_grad():
                noise = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_act = (self.actor_target(next_obs) + noise).clamp(-1, 1)

                target_Q1, target_Q2 = self.critic_target(next_obs, next_act)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q

            current_Q1, current_Q2 = self.critic(obs, act)
            # Huber 损失替换 MSE
            critic_loss = F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)

            if step % self.log_freq == 0:
                L.add_scalar('train_critic/critic_loss', critic_loss.item(), step)
                L.add_scalar('train_critic/Q1', current_Q1.mean().item(), step)
                L.add_scalar('train_critic/Q2', current_Q2.mean().item(), step)
                L.add_scalar('train_critic/target_Q', target_Q.mean().item(), step)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            # Critic 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
            self.critic_opt.step()

        # --- Actor + soft update targets ---
        if self.actor_update_freq > 0 and step % self.actor_update_freq == 0:
            pi = self.actor(obs)
            actor_loss = -self.critic(obs, pi)[0].mean()

            if step % self.log_freq == 0:
                L.add_scalar('train_actor/actor_loss', actor_loss.item(), step)
                L.add_scalar('train_actor/pi_norm', (pi**2).mean().item(), step)

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Soft update all targets
            self._soft_update(self.critic, self.critic_target, self.tau)
            self._soft_update(self.actor, self.actor_target, self.tau)

    def update(self, replay_buffer, L, step):
        obs, act, rew, next_obs, not_done = replay_buffer.sample()
        if step % self.log_freq == 0:
            L.add_scalar('train/batch_reward', rew.mean().item(), step)
        self.update_actor_critic(obs, act, rew, next_obs, not_done, L, step)

    # ----- io -----
    def save(self, model_dir):
        torch.save(self.critic.state_dict(), f'{model_dir}/critic.pt')
        torch.save(self.actor.state_dict(), f'{model_dir}/actor.pt')

    def load(self, model_dir):
        self.critic.load_state_dict(torch.load(f'{model_dir}/critic.pt'))
        self.critic_target.load_state_dict(torch.load(f'{model_dir}/critic.pt'))
        self.actor.load_state_dict(torch.load(f'{model_dir}/actor.pt'))
        self.actor_target.load_state_dict(torch.load(f'{model_dir}/actor.pt'))


# -----------------------------
# TD3 + observation latent (with SEW repr alignment)
# -----------------------------
class TD3ObsAgent(TD3Agent):
    """TD3 with observation encoder + self-supervised constraints + SEW repr-only regularization."""
    def __init__(
        self,
        obs_dims,
        act_dims,
        device,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        expl_noise=0.1,
        n_layers=3,
        hidden_dim=256,
        lr=3e-4,
        # 仅用于放大 Actor 的规模；不影响 enc/dec/dynamics/critic
        actor_n_layers=None,
        actor_hidden_dim=None,
        # SEW
        sew_cfg=None,
        sew_predictor=None,
    ):
        # Hyper
        self.device = device
        self.tau = tau
        self.discount = discount
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.expl_noise = expl_noise

        self.critic_update_freq = 1
        self.actor_update_freq = 2
        self.dyn_cons_update_freq = 1
        self.log_freq = 1000

        # dims
        self.obs_dim = obs_dims['obs_dim']
        self.robot_obs_dim = obs_dims['robot_obs_dim']
        self.obj_obs_dim = obs_dims['obj_obs_dim']
        self.lat_obs_dim = obs_dims['lat_obs_dim']
        self.act_dim = act_dims['act_dim']

        # ---- modules (no BatchNorm in RL by default) ----
        self.obs_enc = utils.build_mlp(
            self.robot_obs_dim, self.lat_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh', batch_norm=False
        ).to(device)
        self.obs_enc_opt = torch.optim.Adam(self.obs_enc.parameters(), lr=lr)

        self.obs_dec = utils.build_mlp(
            self.lat_obs_dim, self.robot_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='identity', batch_norm=False
        ).to(device)
        self.obs_dec_opt = torch.optim.Adam(self.obs_dec.parameters(), lr=lr)

        self.inv_dyn = utils.build_mlp(
            self.lat_obs_dim * 2, self.act_dim - 1, n_layers, hidden_dim,
            activation='relu', output_activation='identity', batch_norm=False
        ).to(device)
        self.inv_dyn_opt = torch.optim.Adam(self.inv_dyn.parameters(), lr=lr)

        self.fwd_dyn = utils.build_mlp(
            self.lat_obs_dim + self.act_dim - 1, self.lat_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='identity', batch_norm=False
        ).to(device)
        self.fwd_dyn_opt = torch.optim.Adam(self.fwd_dyn.parameters(), lr=lr)

        # 只对 Actor 使用独立规模
        a_n = actor_n_layers or n_layers
        a_h = actor_hidden_dim or hidden_dim
        self.actor = Actor(self.lat_obs_dim + self.obj_obs_dim, self.act_dim, a_n, a_h).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Critic on raw obs + real actions
        self.critic = Critic(self.obs_dim, self.act_dim, n_layers, hidden_dim).to(device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # ---- targets ----
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.obs_enc_target = copy.deepcopy(self.obs_enc).to(device)

        for m in [self.actor_target, self.critic_target, self.obs_enc_target]:
            for p in m.parameters():
                p.requires_grad_(False)

        # ---- SEW 配置（repr-only）----
        sew_cfg = sew_cfg or {}
        self.sew_enabled = bool(sew_cfg.get('enabled', False))
        self.sew_lambda = float(sew_cfg.get('lambda', 0.0))
        self.sew_predictor = (sew_predictor.to(device)
                              if (self.sew_enabled and sew_predictor is not None) else None)

    # ----- acting -----
    def sample_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
            robot_obs, obj_obs = obs[:, :self.robot_obs_dim], obs[:, self.robot_obs_dim:]
            lat_obs = self.obs_enc(robot_obs)
            act = self.actor(torch.cat([lat_obs, obj_obs], dim=-1))
            act = act.cpu().data.numpy().flatten()
        if not deterministic:
            noise = np.random.normal(0, self.expl_noise, size=act.shape[0])
            # 抓手维度噪声置零（最后一维）
            if noise.shape[0] >= 1:
                noise[-1] = np.random.normal(0, 0.05)
            act += noise
            act = np.clip(act, -1, 1)
        return act

    # ----- SEW repr loss -----
    def _sew_repr_loss(self, robot_obs: torch.Tensor) -> torch.Tensor:
        if (not self.sew_enabled) or (self.sew_predictor is None):
            return torch.tensor(0.0, device=self.device)
        # enc/dec 重建
        z = self.obs_enc(robot_obs)
        recon = self.obs_dec(z)
        # 角度
        q, q_less1 = _robotobs_to_q(robot_obs)
        qh, qh_less1 = _robotobs_to_q(recon)
        # cos(theta) 对齐
        cos_ref = self.sew_predictor(q_less1)
        cos_hat = self.sew_predictor(qh_less1)
        return F.mse_loss(cos_hat, cos_ref)

    # ----- learning -----
    def update_actor_critic(self, obs, act, reward, next_obs, not_done, L, step):
        # --- Critic ---
        if self.critic_update_freq > 0 and step % self.critic_update_freq == 0:
            with torch.no_grad():
                noise = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                # 抓手维度不做 target smoothing（最后一维）
                if noise.shape[1] >= 1:
                    noise[:, -1] = 0.0

                # target path: use ONLY target enc + target actor
                self.obs_enc_target.eval()
                robot_next, obj_next = next_obs[:, :self.robot_obs_dim], next_obs[:, self.robot_obs_dim:]
                lat_next = self.obs_enc_target(robot_next)
                next_act = self.actor_target(torch.cat([lat_next, obj_next], dim=-1))
                next_act = (next_act + noise).clamp(-1, 1)

                target_Q1, target_Q2 = self.critic_target(next_obs, next_act)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q

            current_Q1, current_Q2 = self.critic(obs, act)
            # Huber 损失替换 MSE
            critic_loss = F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)

            if step % self.log_freq == 0:
                L.add_scalar('train_critic/critic_loss', critic_loss.item(), step)
                L.add_scalar('train_critic/Q1', current_Q1.mean().item(), step)
                L.add_scalar('train_critic/Q2', current_Q2.mean().item(), step)
                L.add_scalar('train_critic/target_Q', target_Q.mean().item(), step)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            # Critic 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
            self.critic_opt.step()

        # --- Actor + enc/dec + soft updates ---
        if self.actor_update_freq > 0 and step % self.actor_update_freq == 0:
            robot_obs, obj_obs = obs[:, :self.robot_obs_dim], obs[:, self.robot_obs_dim:]
            lat_obs = self.obs_enc(robot_obs)
            pi = self.actor(torch.cat([lat_obs, obj_obs], dim=-1))

            actor_loss = -self.critic(obs, pi)[0].mean()

            # SEW 重建表征项（无行为项）
            L_sew_repr = self._sew_repr_loss(robot_obs)
            actor_loss = actor_loss + self.sew_lambda * L_sew_repr

            if step % self.log_freq == 0:
                L.add_scalar('train_actor/actor_loss', actor_loss.item(), step)
                if (self.sew_enabled and self.sew_predictor is not None):
                    L.add_scalar('train_sew/repr_loss', L_sew_repr.item(), step)
                    L.add_scalar('train_sew/repr_loss_weighted', (self.sew_lambda * L_sew_repr).item(), step)
                L.add_scalar('train_actor/lat_obs_norm_sq', (lat_obs**2).mean().item(), step)
                L.add_scalar('train_actor/pi_norm_sq', (pi**2).mean().item(), step)

            self.actor_opt.zero_grad()
            self.obs_enc_opt.zero_grad()
            self.obs_dec_opt.zero_grad()  # 关键：让重建项反向到 decoder
            actor_loss.backward()
            self.actor_opt.step()
            self.obs_enc_opt.step()
            self.obs_dec_opt.step()

            # soft update all targets (including obs_enc_target)
            self._soft_update(self.critic, self.critic_target, self.tau)
            self._soft_update(self.actor, self.actor_target, self.tau)
            self._soft_update(self.obs_enc, self.obs_enc_target, self.tau)

    def update_dyn_cons(self, obs, act, next_obs, L, step):
        robot_obs, robot_next_obs = obs[:, :self.robot_obs_dim], next_obs[:, :self.robot_obs_dim]
        act_wo_grip = act[:, :-1]

        lat_obs = self.obs_enc(robot_obs)
        lat_next_obs = self.obs_enc(robot_next_obs)

        # inverse: predict action (wo gripper) from (z, z')
        pred_act = self.inv_dyn(torch.cat([lat_obs, lat_next_obs], dim=-1))
        inv_loss = F.mse_loss(pred_act, act_wo_grip)

        # forward: predict z' from (z, a)
        pred_next_z = self.fwd_dyn(torch.cat([lat_obs, act_wo_grip], dim=-1))
        fwd_loss = F.mse_loss(pred_next_z, lat_next_obs)

        # reconstruct robot obs from z
        recon_robot_obs = self.obs_dec(lat_obs)
        recon_loss = F.mse_loss(recon_robot_obs, robot_obs)

        loss = fwd_loss + 3.0 * inv_loss + recon_loss

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
        if self.dyn_cons_update_freq == 1 or (step % self.dyn_cons_update_freq == 0):
            self.update_dyn_cons(obs, act, next_obs, L, step)
        self.update_actor_critic(obs, act, rew, next_obs, not_done, L, step)

    # ----- io -----
    def save(self, model_dir):
        torch.save(self.critic.state_dict(), f'{model_dir}/critic.pt')
        torch.save(self.actor.state_dict(), f'{model_dir}/actor.pt')
        torch.save(self.obs_enc.state_dict(), f'{model_dir}/obs_enc.pt')
        torch.save(self.obs_dec.state_dict(), f'{model_dir}/obs_dec.pt')
        torch.save(self.inv_dyn.state_dict(), f'{model_dir}/inv_dyn.pt')
        torch.save(self.fwd_dyn.state_dict(), f'{model_dir}/fwd_dyn.pt')

    def load(self, model_dir):
        self.critic.load_state_dict(torch.load(f'{model_dir}/critic.pt'))
        self.critic_target.load_state_dict(torch.load(f'{model_dir}/critic.pt'))
        self.actor.load_state_dict(torch.load(f'{model_dir}/actor.pt'))
        self.actor_target.load_state_dict(torch.load(f'{model_dir}/actor.pt'))

        self.obs_enc.load_state_dict(torch.load(f'{model_dir}/obs_enc.pt'))
        self.obs_enc_target.load_state_dict(torch.load(f'{model_dir}/obs_enc.pt'))
        self.obs_dec.load_state_dict(torch.load(f'{model_dir}/obs_dec.pt'))
        self.inv_dyn.load_state_dict(torch.load(f'{model_dir}/inv_dyn.pt'))
        self.fwd_dyn.load_state_dict(torch.load(f'{model_dir}/fwd_dyn.pt'))


# -----------------------------
# TD3 + observation & action latents (with SEW repr alignment)
# -----------------------------
class TD3ObsActAgent(TD3Agent):
    """TD3 with observation & action latents + self-supervised constraints + SEW repr-only regularization."""
    def __init__(
        self,
        obs_dims,
        act_dims,
        device,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        expl_noise=0.1,
        n_layers=3,
        hidden_dim=256,
        lr=3e-4,
        # 仅用于放大 Actor 的规模；不影响 enc/dec/dynamics/critic
        actor_n_layers=None,
        actor_hidden_dim=None,
        # SEW
        sew_cfg=None,
        sew_predictor=None,
    ):
        # Hyper
        self.device = device
        self.tau = tau
        self.discount = discount
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.expl_noise = expl_noise

        self.critic_update_freq = 1
        self.actor_update_freq = 2
        self.dyn_cons_update_freq = 1
        self.log_freq = 1000

        # dims
        self.obs_dim = obs_dims['obs_dim']
        self.robot_obs_dim = obs_dims['robot_obs_dim']
        self.obj_obs_dim = obs_dims['obj_obs_dim']
        self.lat_obs_dim = obs_dims['lat_obs_dim']

        self.act_dim = act_dims['act_dim']
        self.lat_act_dim = act_dims['lat_act_dim']

        # ---- modules (no BN) ----
        self.obs_enc = utils.build_mlp(
            self.robot_obs_dim, self.lat_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh', batch_norm=False
        ).to(device)
        self.obs_enc_opt = torch.optim.Adam(self.obs_enc.parameters(), lr=lr)

        self.obs_dec = utils.build_mlp(
            self.lat_obs_dim, self.robot_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='identity', batch_norm=False
        ).to(device)
        self.obs_dec_opt = torch.optim.Adam(self.obs_dec.parameters(), lr=lr)

        self.act_enc = utils.build_mlp(
            self.robot_obs_dim + self.act_dim - 1, self.lat_act_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh', batch_norm=False
        ).to(device)
        self.act_enc_opt = torch.optim.Adam(self.act_enc.parameters(), lr=lr)

        self.act_dec = utils.build_mlp(
            self.robot_obs_dim + self.lat_act_dim, self.act_dim - 1, n_layers, hidden_dim,
            activation='relu', output_activation='tanh', batch_norm=False
        ).to(device)
        self.act_dec_opt = torch.optim.Adam(self.act_dec.parameters(), lr=lr)

        self.inv_dyn = utils.build_mlp(
            self.lat_obs_dim * 2, self.lat_act_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh', batch_norm=False
        ).to(device)
        self.inv_dyn_opt = torch.optim.Adam(self.inv_dyn.parameters(), lr=lr)

        self.fwd_dyn = utils.build_mlp(
            self.lat_obs_dim + self.lat_act_dim, self.lat_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh', batch_norm=False
        ).to(device)
        self.fwd_dyn_opt = torch.optim.Adam(self.fwd_dyn.parameters(), lr=lr)

        # 只对 Actor 使用独立规模；输出为 (lat_act_dim + 1)
        a_n = actor_n_layers or n_layers
        a_h = actor_hidden_dim or hidden_dim
        self.actor = Actor(self.lat_obs_dim + self.obj_obs_dim, self.lat_act_dim + 1, a_n, a_h).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Critic on raw obs + real actions
        self.critic = Critic(self.obs_dim, self.act_dim, n_layers, hidden_dim).to(device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # ---- targets ----
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.obs_enc_target = copy.deepcopy(self.obs_enc).to(device)
        self.act_dec_target = copy.deepcopy(self.act_dec).to(device)

        for m in [self.actor_target, self.critic_target, self.obs_enc_target, self.act_dec_target]:
            for p in m.parameters():
                p.requires_grad_(False)

        # ---- SEW 配置（repr-only）----
        sew_cfg = sew_cfg or {}
        self.sew_enabled = bool(sew_cfg.get('enabled', False))
        self.sew_lambda = float(sew_cfg.get('lambda', 0.0))
        self.sew_predictor = (sew_predictor.to(device)
                              if (self.sew_enabled and sew_predictor is not None) else None)

        # ---- latent action 正则（目标约 0.3）----
        self._lat_act_sq_target = 0.3
        self._lat_act_sq_weight = 0.05  # 温和权重，避免盖过 Q 信号

    # ----- acting -----
    # def sample_action(self, obs, deterministic=False):
    #     with torch.no_grad():
    #         obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
    #         robot_obs, obj_obs = obs[:, :self.robot_obs_dim], obs[:, self.robot_obs_dim:]
    #         lat_obs = self.obs_enc(robot_obs)
    #         lat_act_all = self.actor(torch.cat([lat_obs, obj_obs], dim=-1))
    #         lat_act, gripper_act = lat_act_all[:, :-1], lat_act_all[:, -1].reshape(-1, 1)
    #         act_wo_g = self.act_dec(torch.cat([robot_obs, lat_act], dim=-1))
    #         act = torch.cat([act_wo_g, gripper_act], dim=-1)
    #         act = act.cpu().data.numpy().flatten()
    #     if not deterministic:
    #         # 基础：每维 = self.expl_noise
    #         noise = np.full(act.shape[0], self.expl_noise, dtype=np.float32)

    #         # 关节分层：大臂小，腕部大（你可按机器人改索引）
    #         # 假设 7DoF：0,1,2=大臂；3=过渡；4,5,6=腕部；7=夹爪
    #         if act.shape[0] >= 8:
    #             noise[0:3] *= 1.0    # 肩/肘/腰：= base
    #             noise[3]    *= 1.2   # 过渡关节：稍大
    #             noise[4:7]  *= 1.8   # 腕部：更大一点
    #             noise[7]     = 0.8   # 夹爪：大噪声用于“开关”探索
    #         else:
    #             # 兜底：未知维度时，末两维稍大、最后一维当作夹爪
    #             if act.shape[0] >= 2:
    #                 noise[-2] *= 1.5
    #             noise[-1] = 0.8

    #         noise = np.random.randn(act.shape[0]) * noise
    #         act += noise
    #         act = np.clip(act, -1, 1)
    #     return act
    
    def sample_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
            robot_obs, obj_obs = obs[:, :self.robot_obs_dim], obs[:, self.robot_obs_dim:]
            lat_obs = self.obs_enc(robot_obs)
            lat_act_all = self.actor(torch.cat([lat_obs, obj_obs], dim=-1))
            lat_act, gripper_act = lat_act_all[:, :-1], lat_act_all[:, -1].reshape(-1, 1)
            pi_wo_g = self.act_dec(torch.cat([robot_obs, lat_act], dim=-1))
            act = torch.cat([pi_wo_g, gripper_act], dim=-1)
            act = act.cpu().data.numpy().flatten()

        if not deterministic:
            # ✅ 和老版对齐：所有维度 N(0, expl_noise^2)
            act += np.random.normal(0, self.expl_noise, size=act.shape[0])
            act = np.clip(act, -1, 1)

        return act

    # ----- SEW repr loss -----
    def _sew_repr_loss(self, robot_obs: torch.Tensor) -> torch.Tensor:
        if (not self.sew_enabled) or (self.sew_predictor is None):
            return torch.tensor(0.0, device=self.device)
        z = self.obs_enc(robot_obs)
        recon = self.obs_dec(z)
        q, q_less1 = _robotobs_to_q(robot_obs)
        qh, qh_less1 = _robotobs_to_q(recon)
        cos_ref = self.sew_predictor(q_less1)
        cos_hat = self.sew_predictor(qh_less1)
        return F.mse_loss(cos_hat, cos_ref)

    # ----- learning -----
    def update_actor_critic(self, obs, act, reward, next_obs, not_done, L, step):
        # --- Critic ---
        # --- Critic ---
        if self.critic_update_freq > 0 and step % self.critic_update_freq == 0:
            with torch.no_grad():
                # ✅ 和老版对齐：所有维度都有 target policy smoothing 噪声
                noise = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

                self.obs_enc_target.eval()
                self.act_dec_target.eval()

                robot_next, obj_next = next_obs[:, :self.robot_obs_dim], next_obs[:, self.robot_obs_dim:]
                lat_next = self.obs_enc_target(robot_next)
                next_lat_all = self.actor_target(torch.cat([lat_next, obj_next], dim=-1))
                next_lat, next_grip = next_lat_all[:, :-1], next_lat_all[:, -1].reshape(-1, 1)
                next_wo_g = self.act_dec_target(torch.cat([robot_next, next_lat], dim=-1))
                next_act = torch.cat([next_wo_g, next_grip], dim=-1)
                next_act = (next_act + noise).clamp(-1, 1)

                target_Q1, target_Q2 = self.critic_target(next_obs, next_act)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q

            current_Q1, current_Q2 = self.critic(obs, act)
            # Huber 损失替换 MSE
            critic_loss = F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)

            if step % self.log_freq == 0:
                L.add_scalar('train_critic/critic_loss', critic_loss.item(), step)
                L.add_scalar('train_critic/Q1', current_Q1.mean().item(), step)
                L.add_scalar('train_critic/Q2', current_Q2.mean().item(), step)
                L.add_scalar('train_critic/target_Q', target_Q.mean().item(), step)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            # Critic 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
            self.critic_opt.step()

        # --- Actor + obs_enc + act_dec + soft updates ---
        if self.actor_update_freq > 0 and step % self.actor_update_freq == 0:
            robot_obs, obj_obs = obs[:, :self.robot_obs_dim], obs[:, self.robot_obs_dim:]
            lat_obs = self.obs_enc(robot_obs)
            lat_act_all = self.actor(torch.cat([lat_obs, obj_obs], dim=-1))
            lat_act, gripper_act = lat_act_all[:, :-1], lat_act_all[:, -1].reshape(-1, 1)
            pi_wo_g = self.act_dec(torch.cat([robot_obs, lat_act], dim=-1))
            pi = torch.cat([pi_wo_g, gripper_act], dim=-1)

            actor_loss = -self.critic(obs, pi)[0].mean()

            # latent action 正则：将 (lat_act**2).mean() 牵引到 ~0.3
            lat_act_sq = (lat_act**2).mean()
            lat_reg = (lat_act_sq - self._lat_act_sq_target) ** 2
            actor_loss = actor_loss + self._lat_act_sq_weight * lat_reg

            # SEW 重建表征项（无行为项）
            L_sew_repr = self._sew_repr_loss(robot_obs)
            actor_loss = actor_loss + self.sew_lambda * L_sew_repr

            if step % self.log_freq == 0:
                L.add_scalar('train_actor/actor_loss', actor_loss.item(), step)
                L.add_scalar('train_actor/lat_act_sq', lat_act_sq.item(), step)
                L.add_scalar('train_actor/lat_reg', (self._lat_act_sq_weight * lat_reg).item(), step)
                if (self.sew_enabled and self.sew_predictor is not None):
                    L.add_scalar('train_sew/repr_loss', L_sew_repr.item(), step)
                    L.add_scalar('train_sew/repr_loss_weighted', (self.sew_lambda * L_sew_repr).item(), step)
                L.add_scalar('train_actor/lat_obs_sq', (lat_obs**2).mean().item(), step)
                L.add_scalar('train_actor/pi_sq', (pi**2).mean().item(), step)

            self.actor_opt.zero_grad()
            self.obs_enc_opt.zero_grad()
            self.obs_dec_opt.zero_grad()  # 关键：让重建项反向到 decoder
            self.act_dec_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            self.obs_enc_opt.step()
            self.obs_dec_opt.step()
            self.act_dec_opt.step()

            # Soft update all targets
            self._soft_update(self.critic, self.critic_target, self.tau)
            self._soft_update(self.actor, self.actor_target, self.tau)
            self._soft_update(self.obs_enc, self.obs_enc_target, self.tau)
            self._soft_update(self.act_dec, self.act_dec_target, self.tau)

    def update_dyn_cons(self, obs, act, next_obs, L, step):
        robot_obs, robot_next_obs = obs[:, :self.robot_obs_dim], next_obs[:, :self.robot_obs_dim]
        act_wo_g = act[:, :-1]

        lat_obs = self.obs_enc(robot_obs)
        lat_next_obs = self.obs_enc(robot_next_obs)

        # inverse: (z, z') -> z_a -> decode -> a
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
        if self.dyn_cons_update_freq == 1 or (step % self.dyn_cons_update_freq == 0):
            self.update_dyn_cons(obs, act, next_obs, L, step)
        self.update_actor_critic(obs, act, rew, next_obs, not_done, L, step)

    # ----- io -----
    def save(self, model_dir):
        torch.save(self.critic.state_dict(), f'{model_dir}/critic.pt')
        torch.save(self.actor.state_dict(), f'{model_dir}/actor.pt')
        torch.save(self.obs_enc.state_dict(), f'{model_dir}/obs_enc.pt')
        torch.save(self.obs_dec.state_dict(), f'{model_dir}/obs_dec.pt')
        torch.save(self.act_enc.state_dict(), f'{model_dir}/act_enc.pt')
        torch.save(self.act_dec.state_dict(), f'{model_dir}/act_dec.pt')
        torch.save(self.inv_dyn.state_dict(), f'{model_dir}/inv_dyn.pt')
        torch.save(self.fwd_dyn.state_dict(), f'{model_dir}/fwd_dyn.pt')

    def load(self, model_dir):
        self.critic.load_state_dict(torch.load(f'{model_dir}/critic.pt'))
        self.critic_target.load_state_dict(torch.load(f'{model_dir}/critic.pt'))
        self.actor.load_state_dict(torch.load(f'{model_dir}/actor.pt'))
        self.actor_target.load_state_dict(torch.load(f'{model_dir}/actor.pt'))

        self.obs_enc.load_state_dict(torch.load(f'{model_dir}/obs_enc.pt'))
        self.obs_enc_target.load_state_dict(torch.load(f'{model_dir}/obs_enc.pt'))
        self.obs_dec.load_state_dict(torch.load(f'{model_dir}/obs_dec.pt'))

        self.act_enc.load_state_dict(torch.load(f'{model_dir}/act_enc.pt'))
        self.act_dec.load_state_dict(torch.load(f'{model_dir}/act_dec.pt'))
        self.act_dec_target.load_state_dict(torch.load(f'{model_dir}/act_dec.pt'))

        self.inv_dyn.load_state_dict(torch.load(f'{model_dir}/inv_dyn.pt'))
        self.fwd_dyn.load_state_dict(torch.load(f'{model_dir}/fwd_dyn.pt'))