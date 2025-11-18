# align.py
# =========================================================
# 原始 align 结构 + 全新 SEW 集成（两预测器，生成器端：交叉一致/循环一致）
# - 保持原判别器/循环一致/动力学损失不变
# - 不再使用 MuJoCo 几何求 SEW；改为两个冻结的 SEW 预测器
# - 仅目标侧生成器（E_tgt/D_tgt/[+ act_enc/act_dec]）参与 SEW 的梯度
# - 提供 state_dict()/load_state_dict() 以支持“保存优化器 + 续训”
# =========================================================
import pathlib
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from td3 import Actor


# -----------------------------
# 基础 Agent（与原版保持一致）
# -----------------------------
class Agent:
    """Base class for adaptation"""
    def __init__(self, obs_dims, act_dims, device):
        self.obs_dim = obs_dims['obs_dim']
        self.robot_obs_dim = obs_dims['robot_obs_dim']
        self.obj_obs_dim = obs_dims['obj_obs_dim']
        self.act_dim = act_dims['act_dim']
        self.device = device
        self.batch_norm = False
        self.modules = []
        self.expl_noise = 0.1  # 用于 sample_action 时的一致性

        assert self.obs_dim == self.robot_obs_dim + self.obj_obs_dim

    def eval_mode(self):
        for m in self.modules:
            m.eval()

    def train_mode(self):
        for m in self.modules:
            m.train()

    def freeze(self):
        for m in self.modules:
            for p in m.parameters():
                p.requires_grad = False

    # ====== 从 ckpt 推断 Actor 结构（便于兼容历史权重）======
    @staticmethod
    def _infer_actor_arch_from_state_dict(sd: dict):
        items = []
        for k, v in sd.items():
            if k.startswith("trunk.") and k.endswith(".weight"):
                parts = k.split(".")
                if len(parts) >= 3:
                    try:
                        idx = int(parts[1])
                    except Exception:
                        continue
                    items.append((idx, tuple(v.shape)))
        items.sort(key=lambda x: x[0])
        if not items:
            raise RuntimeError("actor state_dict has no trunk.*.weight keys")

        first_shape = items[0][1]
        last_shape  = items[-1][1]
        in_dim = int(first_shape[1])
        hidden_dim = int(first_shape[0])
        out_dim = int(last_shape[0])
        n_linear = len(items)
        n_hidden_layers = n_linear - 1
        return in_dim, out_dim, n_hidden_layers, hidden_dim

    def _rebuild_actor_to_match_ckpt(self, actor_sd: dict):
        in_dim, out_dim, n_layers, hidden_dim = self._infer_actor_arch_from_state_dict(actor_sd)
        print(f"[align] Rebuild Actor to match ckpt: in={in_dim}, out={out_dim}, "
              f"hidden={hidden_dim}, layers={n_layers}")
        old_actor = getattr(self, "actor", None)
        self.actor = Actor(in_dim, out_dim, n_layers, hidden_dim).to(self.device)
        if hasattr(self, "modules") and old_actor is not None:
            self.modules = [self.actor if (m is old_actor) else m for m in self.modules]


class ObsAgent(Agent):
    def __init__(self, obs_dims, act_dims, device,
                 n_layers=3, hidden_dim=256,
                 actor_n_layers=None, actor_hidden_dim=None):
        super().__init__(obs_dims, act_dims, device)
        self.lat_obs_dim = obs_dims['lat_obs_dim']

        # enc/dec/dynamics：保持与老 ckpt 对齐（不放大）
        self.obs_enc = utils.build_mlp(self.robot_obs_dim, self.lat_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(self.device)
        self.obs_dec = utils.build_mlp(self.lat_obs_dim, self.robot_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='identity', batch_norm=self.batch_norm).to(self.device)
        self.inv_dyn = utils.build_mlp(self.lat_obs_dim*2, self.act_dim-1, n_layers, hidden_dim, 
            activation='relu', output_activation='identity', batch_norm=self.batch_norm).to(self.device)
        self.fwd_dyn = utils.build_mlp(self.lat_obs_dim+self.act_dim-1, self.lat_obs_dim, n_layers, hidden_dim, 
            activation='relu', output_activation='identity', batch_norm=self.batch_norm).to(self.device)

        # 这里先建一个“占位”的 actor，ObsActAgent 会覆盖为 latent 头
        a_n = actor_n_layers or n_layers
        a_h = actor_hidden_dim or hidden_dim
        self.actor = Actor(self.lat_obs_dim+self.obj_obs_dim, self.act_dim, a_n, a_h).to(self.device)

        self.modules = [self.obs_enc, self.obs_dec, self.inv_dyn, self.fwd_dyn, self.actor]

    def save(self, model_dir):
        torch.save(self.actor.state_dict(), f'{model_dir}/actor.pt')
        torch.save(self.obs_enc.state_dict(), f'{model_dir}/obs_enc.pt')        
        torch.save(self.obs_dec.state_dict(), f'{model_dir}/obs_dec.pt')
        torch.save(self.inv_dyn.state_dict(), f'{model_dir}/inv_dyn.pt')
        torch.save(self.fwd_dyn.state_dict(), f'{model_dir}/fwd_dyn.pt')

    def load(self, model_dir):
        model_dir = pathlib.Path(model_dir)

        def _safe_load(m, fname):
            sd = torch.load(model_dir / fname, map_location=self.device)
            try:
                m.load_state_dict(sd, strict=True)
            except RuntimeError as e:
                print(f"[align.load] {fname} strict=True failed -> fallback strict=False: {e}")
                m.load_state_dict(sd, strict=False)

        _safe_load(self.obs_enc, "obs_enc.pt")
        _safe_load(self.obs_dec, "obs_dec.pt")
        _safe_load(self.fwd_dyn, "fwd_dyn.pt")
        _safe_load(self.inv_dyn, "inv_dyn.pt")

        actor_sd = torch.load(model_dir / "actor.pt", map_location=self.device)
        try:
            self.actor.load_state_dict(actor_sd, strict=True)
        except RuntimeError as e:
            print(f"[align.load] actor strict=True failed -> rebuild from ckpt: {e}")
            self._rebuild_actor_to_match_ckpt(actor_sd)
            self.actor.load_state_dict(actor_sd, strict=True)

    def load_actor(self, model_dir):
        model_dir = pathlib.Path(model_dir)
        actor_sd = torch.load(model_dir / "actor.pt", map_location=self.device)
        try:
            self.actor.load_state_dict(actor_sd, strict=True)
        except RuntimeError as e:
            print(f"[align.load_actor] strict=True failed -> rebuild from ckpt: {e}")
            self._rebuild_actor_to_match_ckpt(actor_sd)
            self.actor.load_state_dict(actor_sd, strict=True)
        for p in self.actor.parameters():
            p.requires_grad = False

    def sample_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
            robot_obs, obj_obs = obs[:, :self.robot_obs_dim], obs[:, self.robot_obs_dim:]
            lat_obs = self.obs_enc(robot_obs)
            act = self.actor(torch.cat([lat_obs, obj_obs], dim=-1))

        act = act.cpu().data.numpy().flatten()
        if not deterministic:
            act += np.random.normal(0, self.expl_noise, size=act.shape[0])
            act = np.clip(act, -1, 1)
        return act  


class ObsActAgent(ObsAgent):
    def __init__(self, obs_dims, act_dims, device,
                 n_layers=3, hidden_dim=256,
                 actor_n_layers=None, actor_hidden_dim=None):
        # 先建 enc/dec/dyn 和“占位” actor
        super().__init__(obs_dims, act_dims, device,
                         n_layers=n_layers, hidden_dim=hidden_dim,
                         actor_n_layers=actor_n_layers, actor_hidden_dim=actor_hidden_dim)

        self.lat_act_dim = act_dims['lat_act_dim']

        # ===== 动作分支（保持与旧版一致）=====
        self.act_enc = utils.build_mlp(
            self.robot_obs_dim + self.act_dim - 1, self.lat_act_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh', batch_norm=self.batch_norm
        ).to(device)
        self.act_dec = utils.build_mlp(
            self.robot_obs_dim + self.lat_act_dim, self.act_dim - 1, n_layers, hidden_dim,
            activation='relu', output_activation='tanh', batch_norm=self.batch_norm
        ).to(device)
        self.inv_dyn = utils.build_mlp(
            self.lat_obs_dim * 2, self.lat_act_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh', batch_norm=self.batch_norm
        ).to(device)
        self.fwd_dyn = utils.build_mlp(
            self.lat_obs_dim + self.lat_act_dim, self.lat_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh', batch_norm=self.batch_norm
        ).to(device)

        # ===== 关键修复：actor 改为 “latent 动作头”（lat_act_dim + 1）=====
        a_n = actor_n_layers or n_layers
        a_h = actor_hidden_dim or hidden_dim
        old_actor = self.actor
        self.actor = Actor(self.lat_obs_dim + self.obj_obs_dim, self.lat_act_dim + 1, a_n, a_h).to(device)
        # 替换 modules 里的旧 actor
        self.modules = [self.actor if (m is old_actor) else m for m in self.modules]

        # 把新增分支加入 modules
        self.modules += [self.act_enc, self.act_dec]

    def save(self, model_dir):
        super().save(model_dir)
        torch.save(self.act_enc.state_dict(), f'{model_dir}/act_enc.pt')        
        torch.save(self.act_dec.state_dict(), f'{model_dir}/act_dec.pt')

    def load(self, model_dir):
        model_dir = pathlib.Path(model_dir)
        super().load(model_dir)

        def _safe_load(m, fname):
            sd = torch.load(model_dir / fname, map_location=self.device)
            try:
                m.load_state_dict(sd, strict=True)
            except RuntimeError as e:
                print(f"[align.load] {fname} strict=True failed -> fallback strict=False: {e}")
                m.load_state_dict(sd, strict=False)

        _safe_load(self.act_enc, "act_enc.pt")
        _safe_load(self.act_dec, "act_dec.pt")

    def sample_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
            robot_obs, obj_obs = obs[:, :self.robot_obs_dim], obs[:, self.robot_obs_dim:]
            lat_obs = self.obs_enc(robot_obs)
            lat_act = self.actor(torch.cat([lat_obs, obj_obs], dim=-1))
            lat_act, gripper_act = lat_act[:, :-1], lat_act[:, -1].reshape(-1, 1)
            act = self.act_dec(torch.cat([robot_obs, lat_act], dim=-1))
            act = torch.cat([act, gripper_act], dim=-1)
        
        act = act.cpu().data.numpy().flatten()
        if not deterministic:
            act += np.random.normal(0, self.expl_noise, size=act.shape[0])
            act = np.clip(act, -1, 1)

        return act


# -----------------------------
# SEW 预测器（冻结）与适配器
# -----------------------------
class SEWPredictor(nn.Module):
    """
    统一加载 SEW 角预测小网络：
    - 若 path 是 TorchScript => torch.jit.load
    - 否则：普通 PyTorch => torch.load，然后尝试以下几种 key：
        * checkpoint["model_state_dict"] / ["state_dict"] / ["model"] / 直接是 state_dict
    支持标准化 mean/std（可为 None）
    """
    def __init__(self, path, device, in_dim, out_dim=1, hidden=256, mean=None, std=None):
        super().__init__()
        self.device = device
        self.mean = None if mean is None else torch.as_tensor(mean, dtype=torch.float32, device=device)
        self.std  = None if std  is None else torch.as_tensor(std,  dtype=torch.float32, device=device)

        # 模型结构（注意 utils.build_mlp 的参数顺序）
        self.model = utils.build_mlp(in_dim, out_dim, 3, hidden, activation='relu',
                                     output_activation='identity', batch_norm=False).to(device)

        # 尝试加载
        ok = False
        try:
            # 先试 TorchScript
            ts = torch.jit.load(path, map_location=device)
            class _WrapTS(nn.Module):
                def __init__(self, tsmod): 
                    super().__init__(); 
                    self.ts = tsmod
                def forward(self, x): 
                    return self.ts(x)
            self.model = _WrapTS(ts).to(device)
            ok = True
        except Exception:
            # 回退到普通 checkpoint
            ckpt = torch.load(path, map_location=device)
            if isinstance(ckpt, dict):
                for k in ['model_state_dict', 'state_dict', 'model']:
                    if k in ckpt and isinstance(ckpt[k], dict):
                        self.model.load_state_dict(ckpt[k], strict=False)
                        ok = True
                        break
                if not ok:
                    try:
                        self.model.load_state_dict(ckpt, strict=False)
                        ok = True
                    except Exception:
                        pass
            else:
                ok = False

        if not ok:
            raise RuntimeError(
                f"Failed to load SEW predictor from {path}. "
                f"Expect TorchScript file or a checkpoint with a state_dict."
            )

    @torch.no_grad()
    def forward(self, x):
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-8)
        y = self.model(x)
        return y


def _slice_robot_obs(obs: torch.Tensor, robot_dim: int) -> torch.Tensor:
    # 从 obs = [robot_obs | obj_obs] 中切出 robot_obs
    return obs[:, :robot_dim]


def _robot_obs_to_q(robot_obs: torch.Tensor, dof: int) -> torch.Tensor:
    # robot_obs = [cos(q[0:dof]), sin(q[0:dof]), ...] -> q
    D = robot_obs.shape[1]
    if D >= 2 * dof:
        cos = robot_obs[:, :dof]
        sin = robot_obs[:, dof:2*dof]
        return torch.atan2(sin, cos)
    return robot_obs[:, :dof]


# -----------------------------
# 对齐器：Observation 版 + ObsAct 版
# -----------------------------
class ObsAligner:
    def __init__(
        self, 
        src_agent, 
        tgt_agent, 
        device, 
        n_layers=3, 
        hidden_dim=256,
        lr=3e-4,
        lmbd_gp=10,
        log_freq=1000,
        # ===== 新增：SEW 预测器配置 =====
        sew_cfg: Optional[Dict[str, Any]] = None,
        # ===== 训练细节 =====
        loss_huber_delta: Optional[float] = None,  # 若不为 None，用 SmoothL1
    ):
        self.device = device
        self.lmbd_gp = lmbd_gp
        self.lmbd_cyc = 10
        self.lmbd_dyn = 10
        self.log_freq = log_freq

        self.src_obs_enc = src_agent.obs_enc
        self.src_obs_dec = src_agent.obs_dec
        self.tgt_obs_enc = tgt_agent.obs_enc
        self.tgt_obs_dec = tgt_agent.obs_dec
        self.fwd_dyn = src_agent.fwd_dyn
        self.inv_dyn = src_agent.inv_dyn

        assert src_agent.lat_obs_dim == tgt_agent.lat_obs_dim
        self.lat_obs_dim = src_agent.lat_obs_dim
        self.src_obs_dim = src_agent.robot_obs_dim
        self.tgt_obs_dim = tgt_agent.robot_obs_dim

        self.lat_disc = utils.build_mlp(self.lat_obs_dim, 1, n_layers, hidden_dim,
            activation='leaky_relu', output_activation='identity').to(self.device)
        self.src_disc = utils.build_mlp(self.src_obs_dim, 1, n_layers, hidden_dim,
            activation='leaky_relu', output_activation='identity').to(self.device)
        self.tgt_disc = utils.build_mlp(self.tgt_obs_dim, 1, n_layers, hidden_dim,
            activation='leaky_relu', output_activation='identity').to(self.device)

        # Optimizers（仅本脚本训练的模块有优化器）
        self.tgt_obs_enc_opt = torch.optim.Adam(self.tgt_obs_enc.parameters(), lr=lr)
        self.tgt_obs_dec_opt = torch.optim.Adam(self.tgt_obs_dec.parameters(), lr=lr)
        self.lat_disc_opt = torch.optim.Adam(self.lat_disc.parameters(), lr=lr)
        self.src_disc_opt = torch.optim.Adam(self.src_disc.parameters(), lr=lr)
        self.tgt_disc_opt = torch.optim.Adam(self.tgt_disc.parameters(), lr=lr)

        # =========== SEW 相关（全新方案） ===========
        self.sew_enabled = False
        self.sew_lambda = 0.0
        self.sew_ratio_cross = 0.7
        self.sew_ratio_self = 0.3
        self.sew_every = 1
        self.sew_max_batch = 256
        self.sew_train_all = False
        self.loss_huber_delta = loss_huber_delta

        self.sew_src_pred: Optional[SEWPredictor] = None
        self.sew_tgt_pred: Optional[SEWPredictor] = None
        self.sew_input_mode = "robot_obs"   # "robot_obs" | "q"
        self.src_dof = self.src_obs_dim     # 仅在 "q" 模式下有意义
        self.tgt_dof = self.tgt_obs_dim
        self.src_qpos_slice = None          # [start, length]，仅 "q" 模式下应用
        self.tgt_qpos_slice = None

        if sew_cfg is not None and sew_cfg.get("enabled", False):
            self._init_sew_from_cfg(sew_cfg)

    # --------- SEW 初始化 ----------
    def _init_sew_from_cfg(self, cfg: Dict[str, Any]):
        self.sew_enabled = True
        self.sew_lambda = float(cfg.get("lambda", 0.0))
        self.sew_ratio_cross = float(cfg.get("ratio_cross", 0.7))
        self.sew_ratio_self  = float(cfg.get("ratio_self", 0.3))
        self.sew_every = int(cfg.get("every", 1))
        self.sew_max_batch = int(cfg.get("max_batch", 256))
        self.sew_train_all = bool(cfg.get("train_all", False))
        self.sew_input_mode = cfg.get("input", "robot_obs")

        # 读取 dof / slice（仅在 "q" 模式下有效）
        self.src_dof = int(cfg.get("src_dof", self.src_obs_dim))
        self.tgt_dof = int(cfg.get("tgt_dof", self.tgt_obs_dim))
        self.src_qpos_slice = cfg.get("src_qpos_slice", None)  # e.g., [0, 6]
        self.tgt_qpos_slice = cfg.get("tgt_qpos_slice", None)

        # 计算各域 predictor 的输入维
        def _infer_in_dim(which: str) -> int:
            if self.sew_input_mode == "q":
                if which == "src":
                    if self.src_qpos_slice is not None:
                        return int(self.src_qpos_slice[1])
                    return self.src_dof
                else:
                    if self.tgt_qpos_slice is not None:
                        return int(self.tgt_qpos_slice[1])
                    return self.tgt_dof
            else:
                # robot_obs 模式：直接用对应域的 robot_obs 维度
                return self.src_obs_dim if which == "src" else self.tgt_obs_dim

        # 加载两个预测器
        def _load_predictor(block: Dict[str, Any], which: str) -> SEWPredictor:
            path = block["path"]
            mean, std = None, None
            norm = block.get("norm", None)
            if norm is not None:
                mean = torch.tensor(norm["mean"], dtype=torch.float32, device=self.device)
                std  = torch.tensor(norm["std"],  dtype=torch.float32, device=self.device)
            in_dim = _infer_in_dim(which)
            return SEWPredictor(path, device=self.device, in_dim=in_dim, out_dim=1, hidden=256,
                                mean=mean, std=std)

        self.sew_src_pred = _load_predictor(cfg["src_predictor"], "src")
        self.sew_tgt_pred = _load_predictor(cfg["tgt_predictor"], "tgt")

        print(f"[SEW] enabled | λ={self.sew_lambda}, cross={self.sew_ratio_cross}, self={self.sew_ratio_self}, "
              f"every={self.sew_every}, max_batch={self.sew_max_batch}, train_all={self.sew_train_all}, "
              f"input={self.sew_input_mode}, src_dof={self.src_dof}, tgt_dof={self.tgt_dof}, "
              f"src_qpos_slice={self.src_qpos_slice}, tgt_qpos_slice={self.tgt_qpos_slice}")

    # --------- 对抗的梯度惩罚 ----------
    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        alpha = torch.rand((real_samples.size(0), 1)).to(self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones(real_samples.shape[0], 1, requires_grad=False, device=self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # --------- 判别器更新（保持原逻辑） ----------
    def update_disc(self, src_obs, src_act, tgt_obs, tgt_act, L, step):
        fake_lat_obs = self.tgt_obs_enc(tgt_obs).detach()
        real_lat_obs = self.src_obs_enc(src_obs).detach()
        lat_disc_loss = self.lat_disc(fake_lat_obs).mean() - self.lat_disc(real_lat_obs).mean()

        fake_src_obs = self.src_obs_dec(fake_lat_obs).detach()
        src_disc_loss = self.src_disc(fake_src_obs).mean() - self.src_disc(src_obs).mean()
        
        real_tgt_obs = self.tgt_obs_dec(self.src_obs_enc(src_obs)).detach()
        tgt_disc_loss = self.tgt_disc(real_tgt_obs).mean() - self.tgt_disc(tgt_obs).mean()

        lat_gp = self.compute_gradient_penalty(self.lat_disc, real_lat_obs, fake_lat_obs)
        src_gp = self.compute_gradient_penalty(self.src_disc, src_obs, fake_src_obs)
        tgt_gp = self.compute_gradient_penalty(self.tgt_disc, tgt_obs, real_tgt_obs)

        disc_loss = lat_disc_loss + src_disc_loss + tgt_disc_loss + \
            self.lmbd_gp * (lat_gp + src_gp + tgt_gp)

        self.lat_disc_opt.zero_grad()
        self.src_disc_opt.zero_grad()
        self.tgt_disc_opt.zero_grad()
        disc_loss.backward()
        self.lat_disc_opt.step()
        self.src_disc_opt.step()
        self.tgt_disc_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_lat_disc/lat_disc_loss', lat_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/lat_gp', lat_gp.item(), step)
            L.add_scalar('train_lat_disc/src_disc_loss', src_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/src_gp', src_gp.item(), step)
            L.add_scalar('train_lat_disc/tgt_disc_loss', tgt_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/tgt_gp', tgt_gp.item(), step)

    # --------- 生成器更新（原版 + SEW：交叉一致 + 循环一致） ----------
    def update_gen(self, src_obs, src_act, src_next_obs, tgt_obs, tgt_act, tgt_next_obs, L, step):
        # --- GAN 主损失（与原版一致）---
        fake_lat_obs = self.tgt_obs_enc(tgt_obs)
        lat_gen_loss = -self.lat_disc(fake_lat_obs).mean()

        fake_src_obs = self.src_obs_dec(fake_lat_obs)
        src_gen_loss = -self.src_disc(fake_src_obs).mean()

        real_tgt_obs = self.tgt_obs_dec(self.src_obs_enc(src_obs))
        tgt_gen_loss = -self.tgt_disc(real_tgt_obs).mean()

        gen_loss = lat_gen_loss + src_gen_loss + tgt_gen_loss

        # --- Cycle（与原版一致）---
        pred_src_obs = self.src_obs_dec(self.tgt_obs_enc(real_tgt_obs))
        pred_tgt_obs = self.tgt_obs_dec(self.src_obs_enc(fake_src_obs))
        cycle_loss = F.l1_loss(pred_src_obs, src_obs) + F.l1_loss(pred_tgt_obs, tgt_obs)

        # --- Dynamics（与原版一致）---
        fake_lat_next_obs = self.tgt_obs_enc(tgt_next_obs)
        # inverse: (z, z') -> a_wo_g
        pred_act = self.inv_dyn(torch.cat([fake_lat_obs, fake_lat_next_obs], dim=-1))
        inv_loss = F.mse_loss(pred_act, tgt_act)
        # forward: (z, a_wo_g) -> z'
        pred_lat_next_obs = self.fwd_dyn(torch.cat([fake_lat_obs, tgt_act], dim=-1))
        fwd_loss = F.mse_loss(pred_lat_next_obs, fake_lat_next_obs)

        # --- 新增：SEW 交叉一致 + 循环一致 ---
        sew_loss = torch.tensor(0.0, device=self.device)
        if self.sew_enabled and self.sew_lambda > 0.0:
            if (step % max(1, self.sew_every) == 0) or self.sew_train_all:
                B = src_obs.shape[0]
                if (not self.sew_train_all) and (B > self.sew_max_batch):
                    idx = torch.randperm(B, device=src_obs.device)[:self.sew_max_batch]
                    src_obs_sew = src_obs[idx]
                    tgt_obs_sew = tgt_obs[idx]
                    real_tgt_obs_sew = real_tgt_obs[idx]
                    pred_tgt_obs_sew = pred_tgt_obs[idx]
                else:
                    src_obs_sew = src_obs
                    tgt_obs_sew = tgt_obs
                    real_tgt_obs_sew = real_tgt_obs
                    pred_tgt_obs_sew = pred_tgt_obs

                # 输入准备：robot_obs 或 q（含 qpos_slice）
                def _prep_in(obs_tensor, which: str):
                    rob = _slice_robot_obs(
                        obs_tensor,
                        self.src_obs_dim if which == "src" else self.tgt_obs_dim
                    )
                    if self.sew_input_mode == "q":
                        dof = self.src_dof if which == "src" else self.tgt_dof
                        q = _robot_obs_to_q(rob, dof)
                        sl = self.src_qpos_slice if which == "src" else self.tgt_qpos_slice
                        if sl is not None:
                            start, length = int(sl[0]), int(sl[1])
                            return q[:, start:start+length]
                        return q
                    return rob

                # Cross（最重要）：σ_tgt(D_tgt(E_src(x_src))) ≈ σ_src(x_src)
                with torch.no_grad():
                    teacher_src = _prep_in(src_obs_sew, "src")
                    sigma_src = self.sew_src_pred(teacher_src)
                student_src2tgt = _prep_in(real_tgt_obs_sew, "tgt")
                sigma_tgt_from_s = self.sew_tgt_pred(student_src2tgt)  # 保留梯度
                L_cross = self._sew_loss_fn(sigma_tgt_from_s, sigma_src)

                # Self（辅助）：σ_tgt(pred_tgt_obs) ≈ σ_tgt(tgt_obs)
                with torch.no_grad():
                    teacher_tgt = _prep_in(tgt_obs_sew, "tgt")
                    sigma_tgt = self.sew_tgt_pred(teacher_tgt)
                student_cycle = _prep_in(pred_tgt_obs_sew, "tgt")
                sigma_tgt_pred = self.sew_tgt_pred(student_cycle)      # 保留梯度
                L_self = self._sew_loss_fn(sigma_tgt_pred, sigma_tgt)

                sew_loss = self.sew_ratio_cross * L_cross + self.sew_ratio_self * L_self

        loss = gen_loss + self.lmbd_cyc * cycle_loss + self.lmbd_dyn * (inv_loss + fwd_loss) + self.sew_lambda * sew_loss

        self.tgt_obs_enc_opt.zero_grad()
        self.tgt_obs_dec_opt.zero_grad()
        loss.backward()
        self.tgt_obs_enc_opt.step()
        self.tgt_obs_dec_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_lat_gen/lat_gen_loss', lat_gen_loss.item(), step)
            L.add_scalar('train_lat_gen/src_gen_loss', src_gen_loss.item(), step)
            L.add_scalar('train_lat_gen/tgt_gen_loss', tgt_gen_loss.item(), step)
            L.add_scalar('train_lat_gen/inv_loss', inv_loss.item(), step)
            L.add_scalar('train_lat_gen/fwd_loss', fwd_loss.item(), step)
            L.add_scalar('train_lat_gen/cycle_loss', cycle_loss.item(), step)
            if self.sew_enabled and ((step % max(1, self.sew_every) == 0) or self.sew_train_all):
                L.add_scalar('train_lat_gen/sew_loss', sew_loss.item(), step)

    # --- 可选的 Huber 损失（稳定些） ---
    def _sew_loss_fn(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_huber_delta is None:
            return F.mse_loss(pred, target)
        return F.smooth_l1_loss(pred, target, beta=self.loss_huber_delta)

    # ----------------- 供训练脚本保存/恢复（含优化器） -----------------
    def state_dict(self) -> Dict[str, Any]:
        d = {
            "tgt_obs_enc": self.tgt_obs_enc.state_dict(),
            "tgt_obs_dec": self.tgt_obs_dec.state_dict(),
            "lat_disc": self.lat_disc.state_dict(),
            "src_disc": self.src_disc.state_dict(),
            "tgt_disc": self.tgt_disc.state_dict(),
            "opt": {
                "tgt_obs_enc_opt": self.tgt_obs_enc_opt.state_dict(),
                "tgt_obs_dec_opt": self.tgt_obs_dec_opt.state_dict(),
                "lat_disc_opt": self.lat_disc_opt.state_dict(),
                "src_disc_opt": self.src_disc_opt.state_dict(),
                "tgt_disc_opt": self.tgt_disc_opt.state_dict(),
            },
            "sew_cfg": {
                "enabled": self.sew_enabled,
                "lambda": self.sew_lambda,
                "ratio_cross": self.sew_ratio_cross,
                "ratio_self": self.sew_ratio_self,
                "every": self.sew_every,
                "max_batch": self.sew_max_batch,
                "train_all": self.sew_train_all,
                "input": self.sew_input_mode,
                "src_dof": self.src_dof,
                "tgt_dof": self.tgt_dof,
                "src_qpos_slice": self.src_qpos_slice,
                "tgt_qpos_slice": self.tgt_qpos_slice,
            },
        }
        return d

    def load_state_dict(self, sd: Dict[str, Any], strict: bool=True):
        self.tgt_obs_enc.load_state_dict(sd["tgt_obs_enc"], strict=strict)
        self.tgt_obs_dec.load_state_dict(sd["tgt_obs_dec"], strict=strict)
        self.lat_disc.load_state_dict(sd["lat_disc"], strict=strict)
        self.src_disc.load_state_dict(sd["src_disc"], strict=strict)
        self.tgt_disc.load_state_dict(sd["tgt_disc"], strict=strict)

        # 优化器
        if "opt" in sd:
            self.tgt_obs_enc_opt.load_state_dict(sd["opt"]["tgt_obs_enc_opt"])
            self.tgt_obs_dec_opt.load_state_dict(sd["opt"]["tgt_obs_dec_opt"])
            self.lat_disc_opt.load_state_dict(sd["opt"]["lat_disc_opt"])
            self.src_disc_opt.load_state_dict(sd["opt"]["src_disc_opt"])
            self.tgt_disc_opt.load_state_dict(sd["opt"]["tgt_disc_opt"])

        # SEW cfg（仅用于日志/一致性，预测器由外部 cfg 控制加载）
        if "sew_cfg" in sd:
            cfg = sd["sew_cfg"]
            self.sew_enabled = bool(cfg.get("enabled", self.sew_enabled))
            self.sew_lambda = float(cfg.get("lambda", self.sew_lambda))
            self.sew_ratio_cross = float(cfg.get("ratio_cross", self.sew_ratio_cross))
            self.sew_ratio_self  = float(cfg.get("ratio_self",  self.sew_ratio_self))
            self.sew_every = int(cfg.get("every", self.sew_every))
            self.sew_max_batch = int(cfg.get("max_batch", self.sew_max_batch))
            self.sew_train_all = bool(cfg.get("train_all", self.sew_train_all))
            self.sew_input_mode = cfg.get("input", self.sew_input_mode)
            self.src_dof = int(cfg.get("src_dof", self.src_dof))
            self.tgt_dof = int(cfg.get("tgt_dof", self.tgt_dof))
            self.src_qpos_slice = cfg.get("src_qpos_slice", self.src_qpos_slice)
            self.tgt_qpos_slice = cfg.get("tgt_qpos_slice", self.tgt_qpos_slice)


class ObsActAligner(ObsAligner):
    def __init__(
        self, 
        src_agent, 
        tgt_agent, 
        device, 
        n_layers=3, 
        hidden_dim=256,
        lr=3e-4,
        lmbd_gp=10,
        log_freq=1000,
        sew_cfg: Optional[Dict[str, Any]] = None,
        loss_huber_delta: Optional[float] = None,
    ):
        super().__init__(src_agent, tgt_agent, device, n_layers=n_layers, 
            hidden_dim=hidden_dim, lr=lr, lmbd_gp=lmbd_gp, log_freq=log_freq,
            sew_cfg=sew_cfg, loss_huber_delta=loss_huber_delta)

        assert src_agent.lat_act_dim == tgt_agent.lat_act_dim
        self.lat_act_dim = src_agent.lat_act_dim
        self.src_act_dim = src_agent.act_dim - 1
        self.tgt_act_dim = tgt_agent.act_dim - 1

        self.src_act_enc = src_agent.act_enc 
        self.src_act_dec = src_agent.act_dec
        self.tgt_act_enc = tgt_agent.act_enc 
        self.tgt_act_dec = tgt_agent.act_dec

        # 判别器（拼接 z 和 z_a）
        self.lat_disc = utils.build_mlp(self.lat_obs_dim + self.lat_act_dim, 1, n_layers, 
            hidden_dim, activation='leaky_relu', output_activation='identity').to(self.device)
        self.src_disc = utils.build_mlp(self.src_obs_dim + self.src_act_dim, 1, n_layers, 
            hidden_dim, activation='leaky_relu', output_activation='identity').to(self.device)
        self.tgt_disc = utils.build_mlp(self.tgt_obs_dim + self.tgt_act_dim, 1, n_layers, 
            hidden_dim, activation='leaky_relu', output_activation='identity').to(self.device)

        # 目标侧动作分支也参与优化
        self.tgt_act_enc_opt = torch.optim.Adam(self.tgt_act_enc.parameters(), lr=lr)
        self.tgt_act_dec_opt = torch.optim.Adam(self.tgt_act_dec.parameters(), lr=lr)
        self.lat_disc_opt = torch.optim.Adam(self.lat_disc.parameters(), lr=lr)
        self.src_disc_opt = torch.optim.Adam(self.src_disc.parameters(), lr=lr)
        self.tgt_disc_opt = torch.optim.Adam(self.tgt_disc.parameters(), lr=lr)

    def update_disc(self, src_obs, src_act, tgt_obs, tgt_act, L, step):
        fake_lat_obs = self.tgt_obs_enc(tgt_obs).detach()
        fake_lat_act = self.tgt_act_enc(torch.cat([tgt_obs, tgt_act], dim=-1)).detach()
        real_lat_obs = self.src_obs_enc(src_obs).detach()
        real_lat_act = self.src_act_enc(torch.cat([src_obs, src_act], dim=-1)).detach()

        fake_lat_input = torch.cat([fake_lat_obs, fake_lat_act], dim=-1)
        real_lat_input = torch.cat([real_lat_obs, real_lat_act], dim=-1)
        lat_disc_loss = self.lat_disc(fake_lat_input).mean() - self.lat_disc(real_lat_input).mean()

        fake_src_obs = self.src_obs_dec(fake_lat_obs).detach()
        fake_src_act = self.src_act_dec(torch.cat([fake_src_obs, fake_lat_act], dim=-1)).detach()
        fake_src_input = torch.cat([fake_src_obs, fake_src_act], dim=-1)
        src_input = torch.cat([src_obs, src_act], dim=-1)
        src_disc_loss = self.src_disc(fake_src_input).mean() - self.src_disc(src_input).mean()

        real_tgt_obs = self.tgt_obs_dec(real_lat_obs).detach()
        real_tgt_act = self.tgt_act_dec(torch.cat([real_tgt_obs, real_lat_act], dim=-1)).detach()
        real_tgt_input = torch.cat([real_tgt_obs, real_tgt_act], dim=-1)
        tgt_input = torch.cat([tgt_obs, tgt_act], dim=-1)
        tgt_disc_loss = self.tgt_disc(real_tgt_input).mean() - self.tgt_disc(tgt_input).mean()

        lat_gp = self.compute_gradient_penalty(self.lat_disc, real_lat_input, fake_lat_input)
        src_gp = self.compute_gradient_penalty(self.src_disc, src_input, fake_src_input)
        tgt_gp = self.compute_gradient_penalty(self.tgt_disc, tgt_input, real_tgt_input)

        disc_loss = lat_disc_loss + src_disc_loss + tgt_disc_loss + self.lmbd_gp * (lat_gp + src_gp + tgt_gp)

        self.lat_disc_opt.zero_grad()
        self.src_disc_opt.zero_grad()
        self.tgt_disc_opt.zero_grad()
        disc_loss.backward()
        self.lat_disc_opt.step()
        self.src_disc_opt.step()
        self.tgt_disc_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_lat_disc/lat_disc_loss', lat_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/lat_gp', lat_gp.item(), step)
            L.add_scalar('train_lat_disc/src_disc_loss', src_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/src_gp', src_gp.item(), step)
            L.add_scalar('train_lat_disc/tgt_disc_loss', tgt_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/tgt_gp', tgt_gp.item(), step)

    def update_gen(self, src_obs, src_act, src_next_obs, tgt_obs, tgt_act, tgt_next_obs, L, step):
        # --- GAN 主损失 ---
        fake_lat_obs = self.tgt_obs_enc(tgt_obs)
        fake_lat_act = self.tgt_act_enc(torch.cat([tgt_obs, tgt_act], dim=-1))
        real_lat_obs = self.src_obs_enc(src_obs)
        real_lat_act = self.src_act_enc(torch.cat([src_obs, src_act], dim=-1))

        fake_lat_input = torch.cat([fake_lat_obs, fake_lat_act], dim=-1)
        real_lat_input = torch.cat([real_lat_obs, real_lat_act], dim=-1)
        lat_gen_loss = -self.lat_disc(fake_lat_input).mean()

        fake_src_obs = self.src_obs_dec(fake_lat_obs)
        fake_src_act = self.src_act_dec(torch.cat([fake_src_obs, fake_lat_act], dim=-1))
        fake_src_input = torch.cat([fake_src_obs, fake_src_act], dim=-1)
        src_input = torch.cat([src_obs, src_act], dim=-1)
        src_gen_loss = -self.src_disc(fake_src_input).mean()

        real_tgt_obs = self.tgt_obs_dec(real_lat_obs)
        real_tgt_act = self.tgt_act_dec(torch.cat([real_tgt_obs, real_lat_act], dim=-1))
        real_tgt_input = torch.cat([real_tgt_obs, real_tgt_act], dim=-1)
        tgt_input = torch.cat([tgt_obs, tgt_act], dim=-1)
        tgt_gen_loss = -self.tgt_disc(real_tgt_input).mean()

        gen_loss = lat_gen_loss + src_gen_loss + tgt_gen_loss

        # --- Cycle 一致性 ---
        fake_lat_obs_1 = self.src_obs_enc(fake_src_obs)
        fake_lat_act_1 = self.src_act_enc(torch.cat([fake_src_obs, fake_src_act], dim=-1))
        pred_tgt_obs = self.tgt_obs_dec(fake_lat_obs_1)
        pred_tgt_act = self.tgt_act_dec(torch.cat([pred_tgt_obs, fake_lat_act_1], dim=-1))
        tgt_obs_cycle_loss = F.l1_loss(pred_tgt_obs, tgt_obs)
        tgt_act_cycle_loss = F.l1_loss(pred_tgt_act, tgt_act)

        real_lat_obs_1 = self.tgt_obs_enc(real_tgt_obs)
        real_lat_act_1 = self.tgt_act_enc(torch.cat([real_tgt_obs, real_tgt_act], dim=-1))
        pred_src_obs = self.src_obs_dec(real_lat_obs_1)
        pred_src_act = self.src_act_dec(torch.cat([pred_src_obs, real_lat_act_1], dim=-1))
        src_obs_cycle_loss = F.l1_loss(pred_src_obs, src_obs)
        src_act_cycle_loss = F.l1_loss(pred_src_act, src_act)
        cycle_loss = tgt_obs_cycle_loss + tgt_act_cycle_loss + src_obs_cycle_loss + src_act_cycle_loss

        # --- Dynamics ---
        fake_lat_next_obs = self.tgt_obs_enc(tgt_next_obs)
        pred_lat_act = self.inv_dyn(torch.cat([fake_lat_obs, fake_lat_next_obs], dim=-1))
        pred_act = self.tgt_act_dec(torch.cat([tgt_obs, pred_lat_act], dim=-1))
        inv_loss = F.mse_loss(pred_act, tgt_act)
        pred_lat_next_obs = self.fwd_dyn(torch.cat([fake_lat_obs, fake_lat_act], dim=-1))
        fwd_loss = F.mse_loss(pred_lat_next_obs, fake_lat_next_obs)

        # --- SEW：交叉一致 + 循环一致 ---
        sew_loss = torch.tensor(0.0, device=self.device)
        if self.sew_enabled and self.sew_lambda > 0.0:
            if (step % max(1, self.sew_every) == 0) or self.sew_train_all:
                B = src_obs.shape[0]
                if (not self.sew_train_all) and (B > self.sew_max_batch):
                    idx = torch.randperm(B, device=src_obs.device)[:self.sew_max_batch]
                    src_obs_sew = src_obs[idx]
                    tgt_obs_sew = tgt_obs[idx]
                    real_tgt_obs_sew = real_tgt_obs[idx]
                    pred_tgt_obs_sew = pred_tgt_obs[idx]
                else:
                    src_obs_sew = src_obs
                    tgt_obs_sew = tgt_obs
                    real_tgt_obs_sew = real_tgt_obs
                    pred_tgt_obs_sew = pred_tgt_obs

                def _prep_in(obs_tensor, which: str):
                    rob = _slice_robot_obs(
                        obs_tensor,
                        self.src_obs_dim if which == "src" else self.tgt_obs_dim
                    )
                    if self.sew_input_mode == "q":
                        dof = self.src_dof if which == "src" else self.tgt_dof
                        q = _robot_obs_to_q(rob, dof)
                        sl = self.src_qpos_slice if which == "src" else self.tgt_qpos_slice
                        if sl is not None:
                            start, length = int(sl[0]), int(sl[1])
                            return q[:, start:start+length]
                        return q
                    return rob

                with torch.no_grad():
                    teacher_src = _prep_in(src_obs_sew, "src")
                    sigma_src = self.sew_src_pred(teacher_src)
                student_src2tgt = _prep_in(real_tgt_obs_sew, "tgt")
                sigma_tgt_from_s = self.sew_tgt_pred(student_src2tgt)
                L_cross = self._sew_loss_fn(sigma_tgt_from_s, sigma_src)

                with torch.no_grad():
                    teacher_tgt = _prep_in(tgt_obs_sew, "tgt")
                    sigma_tgt = self.sew_tgt_pred(teacher_tgt)
                student_cycle = _prep_in(pred_tgt_obs_sew, "tgt")
                sigma_tgt_pred = self.sew_tgt_pred(student_cycle)
                L_self = self._sew_loss_fn(sigma_tgt_pred, sigma_tgt)

                sew_loss = self.sew_ratio_cross * L_cross + self.sew_ratio_self * L_self

        loss = gen_loss + self.lmbd_cyc * cycle_loss + self.lmbd_dyn * (inv_loss + fwd_loss) + self.sew_lambda * sew_loss

        self.tgt_obs_enc_opt.zero_grad()
        self.tgt_obs_dec_opt.zero_grad()
        self.tgt_act_enc_opt.zero_grad()
        self.tgt_act_dec_opt.zero_grad()
        loss.backward()
        self.tgt_obs_enc_opt.step()
        self.tgt_obs_dec_opt.step()
        self.tgt_act_enc_opt.step()
        self.tgt_act_dec_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_lat_gen/lat_gen_loss', lat_gen_loss.item(), step)
            L.add_scalar('train_lat_gen/src_gen_loss', src_gen_loss.item(), step)
            L.add_scalar('train_lat_gen/tgt_gen_loss', tgt_gen_loss.item(), step)
            L.add_scalar('train_lat_gen/inv_loss', inv_loss.item(), step)
            L.add_scalar('train_lat_gen/fwd_loss', fwd_loss.item(), step)
            L.add_scalar('train_lat_gen/cycle_loss', cycle_loss.item(), step)
            if self.sew_enabled and ((step % max(1, self.sew_every) == 0) or self.sew_train_all):
                L.add_scalar('train_lat_gen/sew_loss', sew_loss.item(), step)

    # 追加 act 分支优化器/权重到保存
    def state_dict(self) -> Dict[str, Any]:
        base = super().state_dict()
        base.update({
            "tgt_act_enc": self.tgt_act_enc.state_dict(),
            "tgt_act_dec": self.tgt_act_dec.state_dict(),
        })
        base["opt"].update({
            "tgt_act_enc_opt": self.tgt_act_enc_opt.state_dict(),
            "tgt_act_dec_opt": self.tgt_act_dec_opt.state_dict(),
        })
        return base

    def load_state_dict(self, sd: Dict[str, Any], strict: bool=True):
        super().load_state_dict(sd, strict=strict)
        self.tgt_act_enc.load_state_dict(sd["tgt_act_enc"], strict=strict)
        self.tgt_act_dec.load_state_dict(sd["tgt_act_dec"], strict=strict)
        if "opt" in sd:
            self.tgt_act_enc_opt.load_state_dict(sd["opt"]["tgt_act_enc_opt"])
            self.tgt_act_dec_opt.load_state_dict(sd["opt"]["tgt_act_dec_opt"])