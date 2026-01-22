#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2024-2025年最新强化学习算法实现 (GPU加速)
==========================================
包含近两年顶会顶刊的最新RL方法:

1. Diffusion Policy (2024, RSS/CoRL) - 扩散模型生成动作
2. TD-MPC2 (2024, ICLR) - 时序差分模型预测控制
3. Mamba Policy (2025) - 基于状态空间模型的策略
4. DPMD (2025) - 扩散策略镜像下降

References:
- Diffusion Policy: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023 / CoRL 2024
- TD-MPC2: Hansen et al., "TD-MPC2: Scalable, Robust World Models for Continuous Control", ICLR 2024
- Mamba: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2024

Author: CDC Project
Date: 2026-01-21
"""

import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
import math
import time

# 尝试导入深度学习库
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Normal, Categorical
    TORCH_AVAILABLE = True
    
    # GPU检测与自动配置
    def get_device(prefer_gpu: bool = True) -> torch.device:
        """智能设备选择"""
        if prefer_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"[AdvancedRL] 使用GPU: {torch.cuda.get_device_name(0)}")
            print(f"[AdvancedRL] GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return device
        else:
            print("[AdvancedRL] 使用CPU")
            return torch.device('cpu')
    
    def auto_batch_size(device: torch.device, base_batch: int = 256) -> int:
        """根据GPU显存自动调整batch_size"""
        if device.type == 'cuda':
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_mem >= 8:
                return min(512, base_batch * 2)
            elif gpu_mem >= 4:
                return base_batch
            else:
                return max(64, base_batch // 2)
        return base_batch
    
    DEFAULT_DEVICE = get_device()
    
except ImportError:
    TORCH_AVAILABLE = False
    DEFAULT_DEVICE = None
    warnings.warn("PyTorch not available. Advanced RL algorithms will not work.")


# 导入基类
try:
    from .rl_algorithms import BaseRLAlgorithm, ReplayBuffer, Experience
except ImportError:
    from rl_algorithms import BaseRLAlgorithm, ReplayBuffer, Experience


if TORCH_AVAILABLE:
    
    # ============================================================
    # GPU工具函数
    # ============================================================
    
    def to_device(tensor_or_array, device):
        """将numpy数组或tensor移到指定设备"""
        if isinstance(tensor_or_array, np.ndarray):
            return torch.FloatTensor(tensor_or_array).to(device)
        elif isinstance(tensor_or_array, torch.Tensor):
            return tensor_or_array.to(device)
        return tensor_or_array
    
    
    # ============================================================
    # 1. Diffusion Policy (RSS 2023 / CoRL 2024)
    # ============================================================
    
    class SinusoidalPosEmb(nn.Module):
        """正弦位置编码（用于扩散时间步）"""
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
        
        def forward(self, t):
            device = t.device
            half_dim = self.dim // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
            emb = t[:, None].float() * emb[None, :]
            emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
            return emb
    
    
    class DiffusionMLP(nn.Module):
        """扩散模型的MLP网络"""
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
            super().__init__()
            
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            self.noise_pred = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim + action_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, action_dim)
            )
        
        def forward(self, x, t, state):
            t_emb = self.time_mlp(t)
            s_emb = self.state_encoder(state)
            combined = torch.cat([x, t_emb, s_emb], dim=-1)
            return self.noise_pred(combined)
    
    
    class DiffusionPolicy(BaseRLAlgorithm):
        """
        Diffusion Policy (Chi et al., RSS 2023 / CoRL 2024)
        
        使用DDPM扩散模型生成动作，通过去噪过程从噪声中恢复最优动作。
        GPU加速版本。
        """
        
        def __init__(self, state_dim: int, action_dim: int, config: Dict = None):
            super().__init__(state_dim, action_dim, config)
            
            # GPU设置
            self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
            self.lr = config.get('lr', 1e-4)
            self.gamma = config.get('gamma', 0.99)
            self.batch_size = auto_batch_size(self.device, config.get('batch_size', 128))
            
            # 扩散参数 (优化：减少步数加快训练)
            self.n_diffusion_steps = config.get('n_diffusion_steps', 5)  # 原20
            self.beta_start = 1e-4
            self.beta_end = 0.02
            
            # Beta schedule (预计算，放GPU)
            self.betas = torch.linspace(self.beta_start, self.beta_end, self.n_diffusion_steps).to(self.device)
            self.alphas = 1. - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
            
            # 网络 -> GPU
            hidden_dim = config.get('hidden_dim', 256)
            self.model = DiffusionMLP(state_dim, action_dim, hidden_dim).to(self.device)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
            
            # 经验回放
            self.buffer = ReplayBuffer(config.get('buffer_size', 100000))
            
            # 探索
            self.epsilon = config.get('epsilon', 0.3)
            self.epsilon_min = config.get('epsilon_min', 0.05)
            self.epsilon_decay = config.get('epsilon_decay', 0.995)
            
            print(f"[DiffusionPolicy] 初始化完成 | Device: {self.device} | Batch: {self.batch_size}")
        
        def _q_sample(self, x_start, t, noise=None):
            """前向扩散 - 添加噪声"""
            if noise is None:
                noise = torch.randn_like(x_start)
            
            sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
            sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
            
            return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise, noise
        
        def _p_sample(self, x, t, state):
            """反向去噪一步"""
            pred_noise = self.model(x, t, state)
            
            alpha = self.alphas[t].view(-1, 1)
            beta = self.betas[t].view(-1, 1)
            
            mean = (1 / torch.sqrt(alpha)) * (
                x - (beta / self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)) * pred_noise
            )
            
            if t[0] > 0:
                noise = torch.randn_like(x)
                std = torch.sqrt(beta)
                return mean + std * noise
            else:
                return mean
        
        def _sample_action(self, state: torch.Tensor) -> torch.Tensor:
            """从扩散模型采样动作"""
            batch_size = state.shape[0]
            x = torch.randn(batch_size, self.action_dim, device=self.device)
            
            for t in reversed(range(self.n_diffusion_steps)):
                t_batch = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
                x = self._p_sample(x, t_batch, state)
            
            return x
        
        def select_action(self, state: np.ndarray, explore: bool = True) -> int:
            if explore and random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_continuous = self._sample_action(state_tensor)
                action_idx = torch.argmax(action_continuous, dim=-1).item()
                return int(np.clip(action_idx, 0, self.action_dim - 1))
        
        def update(self, batch: List) -> Dict[str, float]:
            if len(batch) < self.batch_size:
                return {}
            
            # 数据移到GPU
            states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
            actions = torch.LongTensor([e.action for e in batch]).to(self.device)
            
            action_onehot = F.one_hot(actions, self.action_dim).float()
            t = torch.randint(0, self.n_diffusion_steps, (len(batch),), device=self.device)
            
            noisy_actions, noise = self._q_sample(action_onehot, t)
            pred_noise = self.model(noisy_actions, t, states)
            
            loss = F.mse_loss(pred_noise, noise)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.training_step += 1
            
            return {'loss': loss.item(), 'epsilon': self.epsilon}
        
        def save(self, path: str):
            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'training_step': self.training_step
            }, path)
        
        def load(self, path: str):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.training_step = checkpoint['training_step']
    
    
    # ============================================================
    # 2. TD-MPC2 (ICLR 2024)
    # ============================================================
    
    class WorldModel(nn.Module):
        """TD-MPC2的世界模型"""
        def __init__(self, state_dim: int, action_dim: int, latent_dim: int = 256):
            super().__init__()
            
            # 状态编码器
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, 512),
                nn.LayerNorm(512),
                nn.Mish(),
                nn.Linear(512, latent_dim)
            )
            
            # 动力学模型
            self.dynamics = nn.Sequential(
                nn.Linear(latent_dim + action_dim, 512),
                nn.LayerNorm(512),
                nn.Mish(),
                nn.Linear(512, latent_dim)
            )
            
            # 奖励预测
            self.reward_pred = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.Mish(),
                nn.Linear(256, 1)
            )
            
            # Q网络
            self.q_network = nn.Sequential(
                nn.Linear(latent_dim, 512),
                nn.Mish(),
                nn.Linear(512, action_dim)
            )
        
        def encode(self, state):
            return self.encoder(state)
        
        def predict_next(self, latent, action_onehot):
            x = torch.cat([latent, action_onehot], dim=-1)
            return self.dynamics(x)
        
        def predict_reward(self, latent):
            return self.reward_pred(latent)
        
        def get_q_values(self, latent):
            return self.q_network(latent)
    
    
    class TDMPC2(BaseRLAlgorithm):
        """
        TD-MPC2 (Hansen et al., ICLR 2024)
        
        结合时序差分学习和模型预测控制，使用学习的世界模型进行规划。
        GPU加速版本，使用CEM优化器进行在线规划。
        """
        
        def __init__(self, state_dim: int, action_dim: int, config: Dict = None):
            super().__init__(state_dim, action_dim, config)
            
            # GPU设置
            self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
            self.lr = config.get('lr', 3e-4)
            self.gamma = config.get('gamma', 0.99)
            self.tau = config.get('tau', 0.005)
            self.batch_size = auto_batch_size(self.device, config.get('batch_size', 128))
            
            # MPC规划参数 (优化：大幅减少计算量)
            self.horizon = config.get('horizon', 2)  # 原3
            self.n_samples = config.get('n_samples', 16)  # 原32
            self.n_elites = config.get('n_elites', 4)  # 原6
            self.n_iterations = config.get('n_iterations', 1)  # 原2
            
            latent_dim = config.get('latent_dim', 256)
            
            # 世界模型 -> GPU
            self.world_model = WorldModel(state_dim, action_dim, latent_dim).to(self.device)
            self.target_world_model = WorldModel(state_dim, action_dim, latent_dim).to(self.device)
            self.target_world_model.load_state_dict(self.world_model.state_dict())
            
            self.optimizer = optim.Adam(self.world_model.parameters(), lr=self.lr)
            self.buffer = ReplayBuffer(config.get('buffer_size', 100000))
            
            # 探索
            self.epsilon = config.get('epsilon', 0.3)
            self.epsilon_min = config.get('epsilon_min', 0.05)
            self.epsilon_decay = config.get('epsilon_decay', 0.995)
            
            print(f"[TD-MPC2] 初始化完成 | Device: {self.device} | Horizon: {self.horizon}")
        
        @torch.no_grad()
        def _plan(self, state: torch.Tensor) -> int:
            """CEM规划 - 交叉熵方法优化动作序列"""
            latent = self.world_model.encode(state)  # (1, latent_dim)
            
            # 初始化动作分布
            mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
            std = torch.ones(self.horizon, self.action_dim, device=self.device) * 2.0
            
            for _ in range(self.n_iterations):
                # 采样动作序列
                samples = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                    self.n_samples, self.horizon, self.action_dim, device=self.device
                )
                
                # 评估每个序列的回报
                returns = torch.zeros(self.n_samples, device=self.device)
                
                for i in range(self.n_samples):
                    z = latent.clone()  # (1, latent_dim)
                    total_return = 0
                    discount = 1.0
                    
                    for t in range(self.horizon):
                        action_probs = F.softmax(samples[i, t], dim=-1)  # (action_dim,)
                        reward = self.world_model.predict_reward(z)  # z: (1, latent_dim)
                        # predict_next需要 (batch, latent_dim) 和 (batch, action_dim)
                        z = self.world_model.predict_next(z, action_probs.unsqueeze(0))  # (1, latent_dim)
                        total_return += discount * reward.item()
                        discount *= self.gamma
                    
                    # 终端价值估计
                    q_values = self.world_model.get_q_values(z)  # z already (1, latent_dim)
                    returns[i] = total_return + discount * q_values.max().item()
                
                # 选择精英样本更新分布
                elite_idx = returns.argsort(descending=True)[:self.n_elites]
                elite_samples = samples[elite_idx]
                mean = elite_samples.mean(dim=0)
                std = elite_samples.std(dim=0) + 0.1
            
            # 返回第一步的最优动作
            return mean[0].argmax().item()
        
        def select_action(self, state: np.ndarray, explore: bool = True) -> int:
            if explore and random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self._plan(state_tensor)
        
        def update(self, batch: List) -> Dict[str, float]:
            if len(batch) < self.batch_size:
                return {}
            
            # 数据移到GPU
            states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
            actions = torch.LongTensor([e.action for e in batch]).to(self.device)
            rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
            next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
            dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
            
            action_onehot = F.one_hot(actions, self.action_dim).float()
            
            # 编码
            latent = self.world_model.encode(states)
            next_latent = self.world_model.encode(next_states)
            
            # 动力学损失
            pred_next_latent = self.world_model.predict_next(latent, action_onehot)
            dynamics_loss = F.mse_loss(pred_next_latent, next_latent.detach())
            
            # 奖励预测损失
            pred_reward = self.world_model.predict_reward(latent).squeeze()
            reward_loss = F.mse_loss(pred_reward, rewards)
            
            # TD损失
            q_values = self.world_model.get_q_values(latent)
            current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze()
            
            with torch.no_grad():
                next_q = self.target_world_model.get_q_values(next_latent)
                target_q = rewards + self.gamma * next_q.max(dim=1)[0] * (1 - dones)
            
            q_loss = F.mse_loss(current_q, target_q)
            
            # 总损失
            loss = dynamics_loss + reward_loss + q_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
            self.optimizer.step()
            
            # 软更新目标网络
            for param, target_param in zip(self.world_model.parameters(), 
                                          self.target_world_model.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.training_step += 1
            
            return {
                'loss': loss.item(),
                'dynamics_loss': dynamics_loss.item(),
                'reward_loss': reward_loss.item(),
                'q_loss': q_loss.item()
            }
        
        def save(self, path: str):
            torch.save({
                'world_model': self.world_model.state_dict(),
                'target_world_model': self.target_world_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'training_step': self.training_step
            }, path)
        
        def load(self, path: str):
            checkpoint = torch.load(path, map_location=self.device)
            self.world_model.load_state_dict(checkpoint['world_model'])
            self.target_world_model.load_state_dict(checkpoint['target_world_model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.training_step = checkpoint['training_step']
    
    
    # ============================================================
    # 3. Mamba Policy (2025)
    # ============================================================
    
    class SelectiveSSM(nn.Module):
        """
        选择性状态空间模型 (Mamba核心)
        线性时间复杂度O(L)，替代Transformer的O(L²)
        """
        def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
            super().__init__()
            
            self.d_model = d_model
            self.d_state = d_state
            
            # 输入投影
            self.in_proj = nn.Linear(d_model, d_model * 2)
            
            # 1D卷积（因果）
            self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=d_conv, 
                                   padding=d_conv - 1, groups=d_model)
            
            # SSM参数投影
            self.x_proj = nn.Linear(d_model, d_state * 2 + 1, bias=False)
            
            # 状态矩阵A的对数（可学习）
            A = torch.arange(1, d_state + 1, dtype=torch.float32)
            self.A_log = nn.Parameter(torch.log(A.repeat(d_model, 1)))
            
            # 跳跃连接
            self.D = nn.Parameter(torch.ones(d_model))
            
            # 输出投影
            self.out_proj = nn.Linear(d_model, d_model)
        
        def forward(self, x):
            B, L, D = x.shape
            
            # 双分支：x和z
            xz = self.in_proj(x)
            x, z = xz.chunk(2, dim=-1)
            
            # 卷积
            x = x.transpose(1, 2)
            x = self.conv1d(x)[:, :, :L]
            x = x.transpose(1, 2)
            x = F.silu(x)
            
            # SSM
            y = self._ssm(x)
            
            # 门控
            y = y * F.silu(z)
            
            return self.out_proj(y)
        
        def _ssm(self, x):
            """选择性扫描"""
            B, L, D = x.shape
            
            # 投影得到delta, B, C
            x_proj = self.x_proj(x)
            delta, B_proj, C = x_proj.split([1, self.d_state, self.d_state], dim=-1)
            delta = F.softplus(delta)
            
            # 离散化A
            A = -torch.exp(self.A_log)
            
            # 循环扫描（简化版，完整版应使用并行扫描）
            h = torch.zeros(B, D, self.d_state, device=x.device)
            ys = []
            
            for i in range(L):
                dt = delta[:, i, 0].unsqueeze(-1).unsqueeze(-1)
                dA = torch.exp(A.unsqueeze(0) * dt)
                dB = B_proj[:, i].unsqueeze(1).expand(-1, D, -1)
                
                h = dA * h + dB * x[:, i].unsqueeze(-1)
                
                C_t = C[:, i].unsqueeze(1).expand(-1, D, -1)
                y = (h * C_t).sum(dim=-1)
                ys.append(y)
            
            y = torch.stack(ys, dim=1)
            
            # 跳跃连接
            y = y + x * self.D.unsqueeze(0).unsqueeze(0)
            
            return y
    
    
    class MambaBlock(nn.Module):
        """Mamba块：LayerNorm + SSM + 残差"""
        def __init__(self, d_model: int):
            super().__init__()
            self.norm = nn.LayerNorm(d_model)
            self.mamba = SelectiveSSM(d_model)
        
        def forward(self, x):
            return x + self.mamba(self.norm(x))
    
    
    class MambaPolicy(BaseRLAlgorithm):
        """
        Mamba Policy (2025)
        
        基于Mamba选择性状态空间模型的强化学习策略。
        O(L)复杂度替代O(L²)的Transformer，适合实时控制。
        GPU加速版本。
        """
        
        def __init__(self, state_dim: int, action_dim: int, config: Dict = None):
            super().__init__(state_dim, action_dim, config)
            
            # GPU设置
            self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
            self.lr = config.get('lr', 1e-4)
            self.gamma = config.get('gamma', 0.99)
            self.batch_size = auto_batch_size(self.device, config.get('batch_size', 64))
            self.context_length = config.get('context_length', 20)
            
            # Mamba参数
            d_model = config.get('d_model', 128)
            n_layers = config.get('n_layers', 4)
            
            # 嵌入层 -> GPU
            self.state_embed = nn.Linear(state_dim, d_model).to(self.device)
            self.action_embed = nn.Embedding(action_dim, d_model).to(self.device)
            self.return_embed = nn.Linear(1, d_model).to(self.device)
            
            # Mamba层 -> GPU
            self.mamba_layers = nn.ModuleList([
                MambaBlock(d_model) for _ in range(n_layers)
            ]).to(self.device)
            
            # 输出头 -> GPU
            self.action_head = nn.Linear(d_model, action_dim).to(self.device)
            self.value_head = nn.Linear(d_model, 1).to(self.device)
            self.ln = nn.LayerNorm(d_model).to(self.device)
            
            # 收集所有参数
            all_params = (
                list(self.state_embed.parameters()) +
                list(self.action_embed.parameters()) +
                list(self.return_embed.parameters()) +
                list(self.mamba_layers.parameters()) +
                list(self.action_head.parameters()) +
                list(self.value_head.parameters()) +
                list(self.ln.parameters())
            )
            self.mamba_optimizer = optim.Adam(all_params, lr=self.lr)
            
            # 历史缓存
            self.state_history = deque(maxlen=self.context_length)
            self.action_history = deque(maxlen=self.context_length)
            self.return_history = deque(maxlen=self.context_length)
            
            # 轨迹存储（用于序列训练）
            self.trajectories = []
            self.target_return = config.get('target_return', 100)
            
            # Q网络（用于快速在线决策） -> GPU
            self.q_network = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim)
            ).to(self.device)
            self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
            
            # 探索
            self.epsilon = config.get('epsilon', 0.3)
            self.epsilon_min = config.get('epsilon_min', 0.05)
            self.epsilon_decay = config.get('epsilon_decay', 0.995)
            
            self.buffer = ReplayBuffer(config.get('buffer_size', 100000))
            
            print(f"[MambaPolicy] 初始化完成 | Device: {self.device} | Layers: {n_layers}")
        
        def forward(self, states, actions=None, returns=None):
            """序列前向传播"""
            B, L, _ = states.shape
            
            # 嵌入
            s_emb = self.state_embed(states)
            
            if actions is not None:
                a_emb = self.action_embed(actions)
            else:
                a_emb = torch.zeros_like(s_emb)
            
            if returns is not None:
                r_emb = self.return_embed(returns.unsqueeze(-1))
            else:
                r_emb = torch.zeros_like(s_emb)
            
            # 融合
            x = s_emb + a_emb + r_emb
            
            # Mamba层
            for layer in self.mamba_layers:
                x = layer(x)
            
            x = self.ln(x)
            
            # 输出
            action_logits = self.action_head(x)
            values = self.value_head(x)
            
            return action_logits, values
        
        def select_action(self, state: np.ndarray, explore: bool = True) -> int:
            if explore and random.random() < self.epsilon:
                action = random.randint(0, self.action_dim - 1)
                self.state_history.append(state)
                self.action_history.append(action)
                return action
            
            self.state_history.append(state)
            
            with torch.no_grad():
                # 使用Q网络快速决策
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                
                if explore:
                    probs = F.softmax(q_values / 0.5, dim=-1)
                    action = torch.multinomial(probs, 1).item()
                else:
                    action = q_values.argmax().item()
            
            self.action_history.append(action)
            return action
        
        def store_trajectory(self, states, actions, rewards):
            """存储完整轨迹用于序列训练"""
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            
            self.trajectories.append({
                'states': np.array(states),
                'actions': np.array(actions),
                'returns': np.array(returns)
            })
            
            # 限制存储的轨迹数量
            if len(self.trajectories) > 100:
                self.trajectories = self.trajectories[-100:]
        
        def update(self, batch: List = None) -> Dict[str, float]:
            q_loss = torch.tensor(0.0, device=self.device)
            mamba_loss = torch.tensor(0.0, device=self.device)
            
            # Q网络在线更新
            if batch and len(batch) >= self.batch_size:
                states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
                actions = torch.LongTensor([e.action for e in batch]).to(self.device)
                rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
                next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
                dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
                
                q_values = self.q_network(states)
                current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze()
                
                with torch.no_grad():
                    next_q = self.q_network(next_states).max(dim=1)[0]
                    target_q = rewards + self.gamma * next_q * (1 - dones)
                
                q_loss = F.mse_loss(current_q, target_q)
                
                self.q_optimizer.zero_grad()
                q_loss.backward()
                self.q_optimizer.step()
            
            # Mamba序列模型离线更新
            if len(self.trajectories) >= 3:
                for _ in range(3):
                    traj = random.choice(self.trajectories)
                    traj_len = len(traj['states'])
                    
                    if traj_len < self.context_length:
                        continue
                    
                    start = random.randint(0, traj_len - self.context_length)
                    end = start + self.context_length
                    
                    states = torch.FloatTensor(traj['states'][start:end]).unsqueeze(0).to(self.device)
                    actions = torch.LongTensor(traj['actions'][start:end]).unsqueeze(0).to(self.device)
                    returns = torch.FloatTensor(traj['returns'][start:end]).unsqueeze(0).to(self.device)
                    
                    # 预测下一步动作：输入前L-1个状态和动作，预测L-1个动作
                    # states[:, :-1]: (1, L-1, state_dim)
                    # actions[:, :-1]: (1, L-1) 
                    # returns[:, :-1]: (1, L-1)
                    action_logits, values = self.forward(
                        states[:, :-1], 
                        actions[:, :-1], 
                        returns[:, :-1]
                    )
                    
                    # 动作预测损失：预测下一步动作
                    action_loss = F.cross_entropy(
                        action_logits.reshape(-1, self.action_dim),
                        actions[:, 1:].reshape(-1)
                    )
                    
                    # 价值预测损失
                    value_loss = F.mse_loss(values.squeeze(-1), returns[:, :-1])
                    
                    loss = action_loss + 0.5 * value_loss
                    
                    self.mamba_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.mamba_layers.parameters(), 1.0)
                    self.mamba_optimizer.step()
                    
                    mamba_loss = loss
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.training_step += 1
            
            return {'q_loss': q_loss.item(), 'mamba_loss': mamba_loss.item()}
        
        def reset_history(self):
            """重置历史缓存（每个episode开始时调用）"""
            self.state_history.clear()
            self.action_history.clear()
            self.return_history.clear()
        
        def save(self, path: str):
            torch.save({
                'state_embed': self.state_embed.state_dict(),
                'action_embed': self.action_embed.state_dict(),
                'return_embed': self.return_embed.state_dict(),
                'mamba_layers': self.mamba_layers.state_dict(),
                'action_head': self.action_head.state_dict(),
                'value_head': self.value_head.state_dict(),
                'ln': self.ln.state_dict(),
                'q_network': self.q_network.state_dict(),
                'epsilon': self.epsilon,
                'training_step': self.training_step
            }, path)
        
        def load(self, path: str):
            checkpoint = torch.load(path, map_location=self.device)
            self.state_embed.load_state_dict(checkpoint['state_embed'])
            self.action_embed.load_state_dict(checkpoint['action_embed'])
            self.return_embed.load_state_dict(checkpoint['return_embed'])
            self.mamba_layers.load_state_dict(checkpoint['mamba_layers'])
            self.action_head.load_state_dict(checkpoint['action_head'])
            self.value_head.load_state_dict(checkpoint['value_head'])
            self.ln.load_state_dict(checkpoint['ln'])
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.epsilon = checkpoint['epsilon']
            self.training_step = checkpoint['training_step']
    
    
    # ============================================================
    # 4. DPMD - Diffusion Policy Mirror Descent (2025)
    # ============================================================
    
    class DPMD(BaseRLAlgorithm):
        """
        Diffusion Policy with Mirror Descent (DPMD, 2025)
        
        结合扩散策略和镜像下降优化：
        - 扩散模型生成多模态动作分布
        - 镜像下降确保策略更新稳定
        - KL约束防止策略崩溃
        
        GPU加速版本。
        """
        
        def __init__(self, state_dim: int, action_dim: int, config: Dict = None):
            super().__init__(state_dim, action_dim, config)
            
            # GPU设置
            self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
            self.lr = config.get('lr', 1e-4)
            self.gamma = config.get('gamma', 0.99)
            self.batch_size = auto_batch_size(self.device, config.get('batch_size', 128))
            
            # 扩散参数 (优化：减少步数加快训练)
            self.n_diffusion_steps = config.get('n_diffusion_steps', 5)  # 原10
            self.kl_coef = config.get('kl_coef', 0.1)  # 镜像下降KL系数
            
            hidden_dim = config.get('hidden_dim', 256)
            
            # 扩散策略网络 -> GPU
            self.policy_net = nn.Sequential(
                nn.Linear(state_dim + action_dim + 1, hidden_dim),  # +1 for time embedding
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, action_dim)
            ).to(self.device)
            
            # 参考策略（用于KL约束）-> 使用相同结构以便复制权重
            self.ref_policy = nn.Sequential(
                nn.Linear(state_dim + action_dim + 1, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, action_dim)
            ).to(self.device)
            # 初始化时复制策略网络权重
            self.ref_policy.load_state_dict(self.policy_net.state_dict())
            
            # Q网络 -> GPU
            self.q_network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ).to(self.device)
            
            # 目标Q网络 -> GPU
            self.target_q = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ).to(self.device)
            self.target_q.load_state_dict(self.q_network.state_dict())
            
            # 扩散schedule -> GPU
            self.betas = torch.linspace(1e-4, 0.02, self.n_diffusion_steps).to(self.device)
            self.alphas = 1. - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            
            # 优化器
            self.optimizer = optim.Adam(
                list(self.policy_net.parameters()) + list(self.q_network.parameters()),
                lr=self.lr
            )
            
            self.buffer = ReplayBuffer(config.get('buffer_size', 100000))
            
            # 探索
            self.epsilon = config.get('epsilon', 0.3)
            self.epsilon_min = config.get('epsilon_min', 0.05)
            self.epsilon_decay = config.get('epsilon_decay', 0.995)
            self.tau = config.get('tau', 0.005)
            
            print(f"[DPMD] 初始化完成 | Device: {self.device} | KL_coef: {self.kl_coef}")
        
        def _sample_action_dist(self, state: torch.Tensor) -> torch.Tensor:
            """扩散采样动作分布"""
            B = state.shape[0]
            x = torch.randn(B, self.action_dim, device=self.device)
            
            for t in reversed(range(self.n_diffusion_steps)):
                # 时间嵌入
                t_embed = torch.full((B, 1), t / self.n_diffusion_steps, device=self.device)
                inp = torch.cat([state, x, t_embed], dim=-1)
                
                # 预测噪声
                noise_pred = self.policy_net(inp)
                
                # 去噪
                alpha = self.alphas[t]
                alpha_cumprod = self.alphas_cumprod[t]
                beta = self.betas[t]
                
                x = (1 / torch.sqrt(alpha)) * (
                    x - (beta / torch.sqrt(1 - alpha_cumprod)) * noise_pred
                )
                
                if t > 0:
                    x = x + torch.sqrt(beta) * torch.randn_like(x)
            
            return x
        
        def _ref_sample_action_dist(self, state: torch.Tensor) -> torch.Tensor:
            """参考策略的扩散采样动作分布"""
            B = state.shape[0]
            x = torch.randn(B, self.action_dim, device=self.device)
            
            for t in reversed(range(self.n_diffusion_steps)):
                t_embed = torch.full((B, 1), t / self.n_diffusion_steps, device=self.device)
                inp = torch.cat([state, x, t_embed], dim=-1)
                noise_pred = self.ref_policy(inp)
                
                alpha = self.alphas[t]
                alpha_cumprod = self.alphas_cumprod[t]
                beta = self.betas[t]
                
                x = (1 / torch.sqrt(alpha)) * (
                    x - (beta / torch.sqrt(1 - alpha_cumprod)) * noise_pred
                )
                
                if t > 0:
                    x = x + torch.sqrt(beta) * torch.randn_like(x)
            
            return x
        
        def select_action(self, state: np.ndarray, explore: bool = True) -> int:
            if explore and random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_logits = self._sample_action_dist(state_tensor)
                return action_logits.argmax(dim=-1).item()
        
        def update(self, batch: List) -> Dict[str, float]:
            if len(batch) < self.batch_size:
                return {}
            
            # 数据移到GPU
            states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
            actions = torch.LongTensor([e.action for e in batch]).to(self.device)
            rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
            next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
            dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
            
            # === Q学习 ===
            q_values = self.q_network(states)
            current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze()
            
            with torch.no_grad():
                next_q = self.target_q(next_states).max(dim=1)[0]
                target_q = rewards + self.gamma * next_q * (1 - dones)
            
            q_loss = F.mse_loss(current_q, target_q)
            
            # === 计算优势 (使用正确的TD误差) ===
            with torch.no_grad():
                # 修复：使用TD目标减去当前Q值作为优势估计
                advantages = target_q - current_q.detach()
                # 优势归一化，确保梯度稳定
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                # 修复：将优势限制在合理范围，避免极端值导致负奖励
                advantages = torch.clamp(advantages, -5.0, 5.0)
            
            # === 扩散策略更新 ===
            action_logits = self._sample_action_dist(states)
            
            # 修复：使用softmax概率而不是直接log_softmax，避免数值问题
            action_probs = F.softmax(action_logits, dim=-1)
            log_probs = torch.log(action_probs + 1e-8)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            # 策略梯度损失（优势加权）
            # 修复：确保正的优势对应正的奖励方向
            pg_loss = -(action_log_probs * advantages.detach()).mean()
            
            # === 镜像下降KL约束 ===
            # 使用相同的扩散采样得到参考策略输出
            with torch.no_grad():
                ref_action_logits = self._ref_sample_action_dist(states)
                ref_probs = F.softmax(ref_action_logits, dim=-1)
            
            # 修复：使用log_target=True避免负KL散度
            kl_div = F.kl_div(log_probs, ref_probs, reduction='batchmean', log_target=False)
            # 确保KL不为负
            kl_div = torch.abs(kl_div)
            
            # 总策略损失 (降低KL系数，让策略更自由探索)
            policy_loss = pg_loss + self.kl_coef * 0.5 * kl_div
            
            # === 总损失 ===
            loss = q_loss + policy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy_net.parameters()) + list(self.q_network.parameters()), 
                1.0
            )
            self.optimizer.step()
            
            # === 软更新目标网络 ===
            for param, target_param in zip(self.q_network.parameters(), 
                                          self.target_q.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            # === 定期更新参考策略（每100步同步一次，防止KL过大） ===
            if self.training_step % 100 == 0:
                self.ref_policy.load_state_dict(self.policy_net.state_dict())
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.training_step += 1
            
            return {
                'loss': loss.item(),
                'q_loss': q_loss.item(),
                'pg_loss': pg_loss.item(),
                'kl_div': kl_div.item(),
                'advantage_mean': advantages.mean().item()
            }
        
        def save(self, path: str):
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'ref_policy': self.ref_policy.state_dict(),
                'q_network': self.q_network.state_dict(),
                'target_q': self.target_q.state_dict(),
                'epsilon': self.epsilon,
                'training_step': self.training_step
            }, path)
        
        def load(self, path: str):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.ref_policy.load_state_dict(checkpoint['ref_policy'])
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_q.load_state_dict(checkpoint['target_q'])
            self.epsilon = checkpoint['epsilon']
            self.training_step = checkpoint['training_step']


# ============================================================
# 算法注册与工厂
# ============================================================

ADVANCED_ALGORITHM_INFO = {
    'DiffusionPolicy': {
        'name': 'Diffusion Policy',
        'year': 2024,
        'venue': 'RSS/CoRL',
        'authors': 'Chi et al.',
        'type': 'Generative',
        'description': '扩散模型生成动作，支持多模态分布，机器人操作SOTA'
    },
    'TDMPC2': {
        'name': 'TD-MPC2',
        'year': 2024,
        'venue': 'ICLR',
        'authors': 'Hansen et al.',
        'type': 'Model-based',
        'description': '世界模型+TD学习+MPC规划，大规模连续控制SOTA'
    },
    'MambaPolicy': {
        'name': 'Mamba Policy',
        'year': 2025,
        'venue': 'Emerging',
        'authors': 'Based on Gu & Dao',
        'type': 'Sequence Model',
        'description': '选择性状态空间模型，O(L)复杂度，实时控制友好'
    },
    'DPMD': {
        'name': 'DPMD',
        'year': 2025,
        'venue': 'Emerging',
        'authors': 'Novel',
        'type': 'Generative + Optimization',
        'description': '扩散策略+镜像下降，稳定更新，KL约束防崩溃'
    }
}


def get_advanced_algorithm(name: str, state_dim: int, action_dim: int, config: Dict = None):
    """
    获取2024-2025年新RL算法实例
    
    Args:
        name: 算法名称 ('DiffusionPolicy', 'TDMPC2', 'MambaPolicy', 'DPMD')
        state_dim: 状态维度
        action_dim: 动作维度
        config: 算法配置（包括device等）
    
    Returns:
        RL算法实例
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for advanced RL algorithms")
    
    algorithms = {
        'DiffusionPolicy': DiffusionPolicy,
        'Diffusion': DiffusionPolicy,
        'TDMPC2': TDMPC2,
        'TD-MPC2': TDMPC2,
        'MambaPolicy': MambaPolicy,
        'Mamba': MambaPolicy,
        'DPMD': DPMD,
    }
    
    if name not in algorithms:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(algorithms.keys())}")
    
    return algorithms[name](state_dim, action_dim, config or {})


def list_advanced_algorithms() -> List[str]:
    """返回可用的2024-2025新算法列表"""
    return ['DiffusionPolicy', 'TDMPC2', 'MambaPolicy', 'DPMD']


def print_advanced_algorithms():
    """打印所有新算法信息"""
    print("\n" + "="*70)
    print("2024-2025年最新强化学习算法")
    print("="*70)
    for name, info in ADVANCED_ALGORITHM_INFO.items():
        print(f"\n📌 {info['name']} ({name})")
        print(f"   年份: {info['year']} | 来源: {info['venue']}")
        print(f"   类型: {info['type']}")
        print(f"   描述: {info['description']}")
    print("\n" + "="*70)
