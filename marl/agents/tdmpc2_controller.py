"""
TD-MPC2 控制智能体
==================
结合模型预测控制和时序差分学习的容错控制
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class TDMPC2Config:
    """TD-MPC2配置"""
    # 状态和动作维度
    obs_dim: int = 40
    action_dim: int = 3  # timing_offset, fuel_adj, protection_level
    
    # 模型架构
    hidden_dim: int = 256
    latent_dim: int = 64
    n_ensemble: int = 3  # 世界模型集成数量
    
    # 规划参数
    horizon: int = 5      # 规划视野
    n_samples: int = 512  # MPPI采样数
    n_elites: int = 64    # 精英样本数
    temperature: float = 0.5
    
    # 学习率
    lr_world: float = 3e-4
    lr_policy: float = 3e-4
    lr_value: float = 3e-4
    
    # 其他
    gamma: float = 0.99
    tau: float = 0.005  # 目标网络软更新


class WorldModel(nn.Module):
    """
    世界模型：预测下一状态和奖励
    
    使用集成方法提高不确定性估计
    """
    
    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int, hidden_dim: int, n_ensemble: int):
        super().__init__()
        
        self.n_ensemble = n_ensemble
        self.latent_dim = latent_dim
        
        # 状态编码器
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # 动态模型集成
        self.dynamics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, latent_dim)
            ) for _ in range(n_ensemble)
        ])
        
        # 奖励预测器集成
        self.reward_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + action_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(n_ensemble)
        ])
        
        # 状态解码器（用于辅助损失）
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
    
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """编码观测到潜在空间"""
        return self.encoder(obs)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """解码潜在状态"""
        return self.decoder(latent)
    
    def predict(
        self, 
        latent: torch.Tensor, 
        action: torch.Tensor,
        ensemble_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测下一状态和奖励
        
        如果ensemble_idx为None，返回集成平均
        """
        sa = torch.cat([latent, action], dim=-1)
        
        if ensemble_idx is not None:
            next_latent = latent + self.dynamics[ensemble_idx](sa)  # 残差连接
            reward = self.reward_predictors[ensemble_idx](sa)
        else:
            # 集成平均
            next_latents = []
            rewards = []
            for i in range(self.n_ensemble):
                next_latents.append(latent + self.dynamics[i](sa))
                rewards.append(self.reward_predictors[i](sa))
            
            next_latent = torch.stack(next_latents, dim=0).mean(dim=0)
            reward = torch.stack(rewards, dim=0).mean(dim=0)
        
        return next_latent, reward
    
    def predict_trajectory(
        self,
        latent: torch.Tensor,
        actions: torch.Tensor,
        horizon: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测轨迹
        
        Args:
            latent: 初始潜在状态 [batch, latent_dim]
            actions: 动作序列 [batch, horizon, action_dim]
            
        Returns:
            latents: 潜在状态序列 [batch, horizon+1, latent_dim]
            rewards: 奖励序列 [batch, horizon]
        """
        batch_size = latent.shape[0]
        latents = [latent]
        rewards = []
        
        for t in range(horizon):
            next_latent, reward = self.predict(latents[-1], actions[:, t])
            latents.append(next_latent)
            rewards.append(reward)
        
        latents = torch.stack(latents, dim=1)
        rewards = torch.cat(rewards, dim=-1)
        
        return latents, rewards


class TwinQNetwork(nn.Module):
    """双Q网络"""
    
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        
        self.q1 = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([latent, action], dim=-1)
        return self.q1(sa), self.q2(sa)
    
    def q_min(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(latent, action)
        return torch.min(q1, q2)


class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(
        self, 
        latent_dim: int, 
        action_dim: int, 
        hidden_dim: int,
        timing_range: Tuple[float, float] = (-5.0, 5.0),
        fuel_range: Tuple[float, float] = (0.85, 1.15)
    ):
        super().__init__()
        
        self.timing_range = timing_range
        self.fuel_range = fuel_range
        
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # 连续动作头（timing, fuel）
        self.continuous_mean = nn.Linear(hidden_dim, 2)
        self.continuous_log_std = nn.Parameter(torch.zeros(2))
        
        # 离散动作头（protection level）
        self.discrete_head = nn.Linear(hidden_dim, 4)
    
    def forward(
        self, 
        latent: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action: [batch, 3] (timing, fuel, protection)
            log_prob: 动作对数概率
        """
        features = self.trunk(latent)
        
        # 连续动作
        mean = self.continuous_mean(features)
        std = self.continuous_log_std.exp()
        
        if deterministic:
            continuous = mean
            cont_log_prob = torch.zeros_like(mean[:, 0])
        else:
            dist = torch.distributions.Normal(mean, std)
            continuous = dist.rsample()
            cont_log_prob = dist.log_prob(continuous).sum(dim=-1)
        
        # 缩放到范围
        timing = torch.tanh(continuous[:, 0:1]) * (self.timing_range[1] - self.timing_range[0]) / 2
        fuel = torch.sigmoid(continuous[:, 1:2]) * (self.fuel_range[1] - self.fuel_range[0]) + self.fuel_range[0]
        
        # 离散动作
        protection_logits = self.discrete_head(features)
        protection_probs = F.softmax(protection_logits, dim=-1)
        
        if deterministic:
            protection = protection_probs.argmax(dim=-1, keepdim=True).float()
            disc_log_prob = torch.zeros_like(protection[:, 0])
        else:
            protection_dist = torch.distributions.Categorical(protection_probs)
            protection = protection_dist.sample().unsqueeze(-1).float()
            disc_log_prob = protection_dist.log_prob(protection.squeeze(-1).long())
        
        action = torch.cat([timing, fuel, protection], dim=-1)
        log_prob = cont_log_prob + disc_log_prob
        
        return action, log_prob


class TDMPC2Controller:
    """
    TD-MPC2控制器
    
    结合模型预测控制（MPC）和时序差分（TD）学习
    """
    
    def __init__(self, config: TDMPC2Config, device: str = 'cpu'):
        self.config = config
        self.device = torch.device(device)
        
        # 初始化网络
        self.world_model = WorldModel(
            config.obs_dim, config.action_dim, 
            config.latent_dim, config.hidden_dim, config.n_ensemble
        ).to(self.device)
        
        self.q_network = TwinQNetwork(
            config.latent_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        
        self.q_target = TwinQNetwork(
            config.latent_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        
        self.policy = PolicyNetwork(
            config.latent_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        
        # 优化器
        self.world_optimizer = torch.optim.Adam(
            self.world_model.parameters(), lr=config.lr_world
        )
        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=config.lr_value
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.lr_policy
        )
        
        # MPPI参数
        self.action_mean = torch.zeros(config.horizon, config.action_dim).to(self.device)
    
    def act(
        self, 
        obs: np.ndarray, 
        diagnosis: Optional[Dict] = None,
        use_planning: bool = True
    ) -> Dict:
        """
        选择动作
        
        Args:
            obs: 观测
            diagnosis: 诊断结果（可选）
            use_planning: 是否使用MPC规划
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            latent = self.world_model.encode(obs_tensor)
            
            if use_planning:
                action = self._mppi_planning(latent, diagnosis)
            else:
                action, _ = self.policy(latent, deterministic=True)
                action = action[0]
        
        return {
            'timing_offset': action[0].item(),
            'fuel_adj': action[1].item(),
            'protection_level': int(action[2].item())
        }
    
    def _mppi_planning(
        self, 
        latent: torch.Tensor,
        diagnosis: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Model Predictive Path Integral (MPPI) 规划
        """
        config = self.config
        batch_latent = latent.expand(config.n_samples, -1)
        
        # 初始化动作序列
        actions_mean = self.action_mean.unsqueeze(0).expand(config.n_samples, -1, -1)
        noise = torch.randn_like(actions_mean) * 0.5
        
        # 根据诊断结果调整采样范围
        if diagnosis and diagnosis.get('fault_type', 0) > 0:
            # 故障状态下增加探索
            noise *= 1.5
        
        actions = actions_mean + noise
        
        # 限制动作范围
        actions[:, :, 0] = torch.clamp(actions[:, :, 0], -5.0, 5.0)  # timing
        actions[:, :, 1] = torch.clamp(actions[:, :, 1], 0.85, 1.15)  # fuel
        actions[:, :, 2] = torch.clamp(actions[:, :, 2], 0, 3).round()  # protection
        
        # 滚动预测
        latents, rewards = self.world_model.predict_trajectory(
            batch_latent, actions, config.horizon
        )
        
        # 终端价值
        final_latent = latents[:, -1]
        final_action, _ = self.policy(final_latent)
        terminal_value = self.q_target.q_min(final_latent, final_action)
        
        # 计算累积回报
        discounts = config.gamma ** torch.arange(config.horizon).float().to(self.device)
        cumulative_rewards = (rewards * discounts).sum(dim=-1, keepdim=True)
        total_value = cumulative_rewards + (config.gamma ** config.horizon) * terminal_value
        
        # MPPI加权
        weights = F.softmax(total_value / config.temperature, dim=0)
        
        # 加权平均最优动作序列
        # weights: [n_samples, 1], actions: [n_samples, horizon, action_dim]
        weights_expanded = weights.unsqueeze(-1)  # [n_samples, 1, 1]
        optimal_actions = (weights_expanded * actions).sum(dim=0)  # [horizon, action_dim]
        
        # 更新滚动均值
        self.action_mean = torch.cat([
            optimal_actions[1:], 
            optimal_actions[-1:].clone()
        ], dim=0)
        
        return optimal_actions[0]  # 返回第一个动作
    
    def update_world_model(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor
    ) -> Dict[str, float]:
        """更新世界模型"""
        # 编码
        latent = self.world_model.encode(obs)
        next_latent_target = self.world_model.encode(next_obs)
        
        # 预测
        losses = []
        for i in range(self.config.n_ensemble):
            next_latent_pred, reward_pred = self.world_model.predict(latent, action, i)
            
            # 动态损失
            dynamics_loss = F.mse_loss(next_latent_pred, next_latent_target.detach())
            
            # 奖励损失
            reward_loss = F.mse_loss(reward_pred.squeeze(), reward)
            
            losses.append(dynamics_loss + reward_loss)
        
        # 重建损失
        recon_loss = F.mse_loss(self.world_model.decode(latent), obs)
        
        total_loss = sum(losses) / self.config.n_ensemble + 0.1 * recon_loss
        
        self.world_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.world_optimizer.step()
        
        return {'world_loss': total_loss.item()}
    
    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor
    ) -> Dict[str, float]:
        """更新Q网络"""
        with torch.no_grad():
            latent = self.world_model.encode(obs)
            next_latent = self.world_model.encode(next_obs)
            
            next_action, _ = self.policy(next_latent)
            target_q = self.q_target.q_min(next_latent, next_action)
            target = reward.unsqueeze(-1) + self.config.gamma * (1 - done.unsqueeze(-1)) * target_q
        
        latent = self.world_model.encode(obs)
        q1, q2 = self.q_network(latent, action)
        
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        
        self.q_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.q_optimizer.step()
        
        # 软更新目标网络
        for param, target_param in zip(self.q_network.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        
        return {'critic_loss': critic_loss.item()}
    
    def update_policy(self, obs: torch.Tensor) -> Dict[str, float]:
        """更新策略网络"""
        latent = self.world_model.encode(obs).detach()
        action, log_prob = self.policy(latent)
        
        q_value = self.q_network.q_min(latent, action)
        policy_loss = -q_value.mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        return {'policy_loss': policy_loss.item()}
    
    def save(self, path: str):
        torch.save({
            'world_model': self.world_model.state_dict(),
            'q_network': self.q_network.state_dict(),
            'q_target': self.q_target.state_dict(),
            'policy': self.policy.state_dict(),
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.q_target.load_state_dict(checkpoint['q_target'])
        self.policy.load_state_dict(checkpoint['policy'])
