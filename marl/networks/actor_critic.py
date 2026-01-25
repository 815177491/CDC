"""
Actor-Critic神经网络
====================
支持诊断智能体和控制智能体的网络结构
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Tuple, Dict, Optional


class DiagnosticNetwork(nn.Module):
    """
    诊断智能体网络
    
    输入: 观测特征（含控制历史）
    输出: 故障分类概率 + 严重程度估计 + 置信度
    """
    
    def __init__(
        self,
        obs_dim: int,
        n_fault_types: int = 4,
        hidden_dim: int = 128,
        use_lstm: bool = True,
        use_internal_critic: bool = True
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.n_fault_types = n_fault_types
        self.use_lstm = use_lstm
        self.use_internal_critic = use_internal_critic
        self.hidden_dim = hidden_dim
        
        # 特征编码器
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 可选LSTM用于时序建模
        if use_lstm:
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.hidden_state = None
        
        # Actor头 - 故障分类
        self.fault_type_head = nn.Linear(hidden_dim, n_fault_types)
        
        # Actor头 - 严重程度（Beta分布参数）
        self.severity_alpha = nn.Linear(hidden_dim, 1)
        self.severity_beta = nn.Linear(hidden_dim, 1)
        
        # Actor头 - 置信度（Beta分布参数）
        self.confidence_alpha = nn.Linear(hidden_dim, 1)
        self.confidence_beta = nn.Linear(hidden_dim, 1)
        
        # Critic头（可选，使用 SharedCritic 时可禁用以减少冗余参数）
        if use_internal_critic:
            self.critic = nn.Linear(hidden_dim, 1)
        else:
            self.critic = None
    
    def reset_hidden(self, batch_size: int = 1):
        """重置LSTM隐状态"""
        if self.use_lstm:
            device = next(self.parameters()).device
            self.hidden_state = (
                torch.zeros(1, batch_size, 128).to(device),
                torch.zeros(1, batch_size, 128).to(device)
            )
    
    def forward(
        self, 
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs: 观测 [batch, obs_dim]
            deterministic: 是否确定性输出
            
        Returns:
            actions: {'fault_type', 'severity', 'confidence'}
            log_probs: 动作对数概率
            value: 状态价值估计
        """
        # 特征编码
        features = self.encoder(obs)
        
        # LSTM处理
        if self.use_lstm and len(obs.shape) == 3:
            features, self.hidden_state = self.lstm(features, self.hidden_state)
            features = features[:, -1, :]  # 取最后时刻
        
        # 故障分类
        fault_logits = self.fault_type_head(features)
        fault_probs = F.softmax(fault_logits, dim=-1)
        
        if deterministic:
            fault_type = fault_probs.argmax(dim=-1)
            fault_log_prob = torch.log(fault_probs.gather(1, fault_type.unsqueeze(-1)) + 1e-8)
        else:
            fault_dist = Categorical(fault_probs)
            fault_type = fault_dist.sample()
            fault_log_prob = fault_dist.log_prob(fault_type).unsqueeze(-1)
        
        # 严重程度（使用Beta分布，输出在[0,1]）
        sev_alpha = F.softplus(self.severity_alpha(features)) + 1
        sev_beta = F.softplus(self.severity_beta(features)) + 1
        
        if deterministic:
            severity = sev_alpha / (sev_alpha + sev_beta)  # Beta分布均值
            sev_log_prob = torch.zeros_like(severity)
        else:
            # 使用重参数化近似
            severity = torch.distributions.Beta(sev_alpha, sev_beta).rsample()
            sev_log_prob = torch.distributions.Beta(sev_alpha, sev_beta).log_prob(severity.clamp(1e-6, 1-1e-6))
        
        # 置信度
        conf_alpha = F.softplus(self.confidence_alpha(features)) + 1
        conf_beta = F.softplus(self.confidence_beta(features)) + 1
        
        if deterministic:
            confidence = conf_alpha / (conf_alpha + conf_beta)
            conf_log_prob = torch.zeros_like(confidence)
        else:
            confidence = torch.distributions.Beta(conf_alpha, conf_beta).rsample()
            conf_log_prob = torch.distributions.Beta(conf_alpha, conf_beta).log_prob(confidence.clamp(1e-6, 1-1e-6))
        
        # 汇总
        actions = {
            'fault_type': fault_type,
            'severity': severity,
            'confidence': confidence
        }
        
        log_probs = fault_log_prob + sev_log_prob + conf_log_prob
        
        # 内部 Critic 或返回 None
        if self.critic is not None:
            value = self.critic(features)
        else:
            value = torch.zeros(features.shape[0], 1, device=features.device)
        
        return actions, log_probs, value
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估给定动作的概率和价值"""
        features = self.encoder(obs)
        
        # 故障分类
        fault_logits = self.fault_type_head(features)
        fault_probs = F.softmax(fault_logits, dim=-1)
        fault_dist = Categorical(fault_probs)
        fault_log_prob = fault_dist.log_prob(actions['fault_type'])
        fault_entropy = fault_dist.entropy()
        
        # 严重程度
        sev_alpha = F.softplus(self.severity_alpha(features)) + 1
        sev_beta = F.softplus(self.severity_beta(features)) + 1
        sev_dist = torch.distributions.Beta(sev_alpha.squeeze(), sev_beta.squeeze())
        sev_log_prob = sev_dist.log_prob(actions['severity'].squeeze().clamp(1e-6, 1-1e-6))
        
        # 置信度
        conf_alpha = F.softplus(self.confidence_alpha(features)) + 1
        conf_beta = F.softplus(self.confidence_beta(features)) + 1
        conf_dist = torch.distributions.Beta(conf_alpha.squeeze(), conf_beta.squeeze())
        conf_log_prob = conf_dist.log_prob(actions['confidence'].squeeze().clamp(1e-6, 1-1e-6))
        
        log_probs = fault_log_prob + sev_log_prob + conf_log_prob
        entropy = fault_entropy
        
        if self.critic is not None:
            value = self.critic(features)
        else:
            value = torch.zeros(features.shape[0], 1, device=features.device)
        
        return log_probs, entropy, value


class ControlNetwork(nn.Module):
    """
    控制智能体网络
    
    输入: 观测特征 + 诊断结果
    输出: 正时补偿(连续) + 燃油调整(连续) + 保护级别(离散)
    """
    
    def __init__(
        self,
        obs_dim: int,
        timing_range: Tuple[float, float] = (-5.0, 5.0),
        fuel_range: Tuple[float, float] = (0.85, 1.15),
        n_protection_levels: int = 4,
        hidden_dim: int = 128,
        use_internal_critic: bool = True
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.timing_range = timing_range
        self.fuel_range = fuel_range
        self.n_protection_levels = n_protection_levels
        self.use_internal_critic = use_internal_critic
        
        # 特征编码器
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor头 - 正时补偿（高斯分布）
        self.timing_mean = nn.Linear(hidden_dim, 1)
        self.timing_log_std = nn.Parameter(torch.zeros(1))
        
        # Actor头 - 燃油调整（高斯分布）
        self.fuel_mean = nn.Linear(hidden_dim, 1)
        self.fuel_log_std = nn.Parameter(torch.zeros(1))
        
        # Actor头 - 保护级别（分类）
        self.protection_head = nn.Linear(hidden_dim, n_protection_levels)
        
        # Critic头（可选）
        if use_internal_critic:
            self.critic = nn.Linear(hidden_dim, 1)
        else:
            self.critic = None
    
    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """前向传播"""
        features = self.encoder(obs)
        
        # 正时补偿
        timing_mean = self.timing_mean(features)
        timing_std = self.timing_log_std.exp()
        
        if deterministic:
            timing_offset = timing_mean
            timing_log_prob = torch.zeros_like(timing_offset)
        else:
            timing_dist = Normal(timing_mean, timing_std)
            timing_offset = timing_dist.rsample()
            timing_log_prob = timing_dist.log_prob(timing_offset)
        
        # 限制范围
        timing_offset = torch.clamp(timing_offset, self.timing_range[0], self.timing_range[1])
        
        # 燃油调整
        fuel_mean = self.fuel_mean(features)
        fuel_std = self.fuel_log_std.exp()
        
        if deterministic:
            fuel_adj = fuel_mean
            fuel_log_prob = torch.zeros_like(fuel_adj)
        else:
            fuel_dist = Normal(fuel_mean, fuel_std)
            fuel_adj = fuel_dist.rsample()
            fuel_log_prob = fuel_dist.log_prob(fuel_adj)
        
        # 限制范围
        fuel_adj = torch.clamp(fuel_adj, self.fuel_range[0], self.fuel_range[1])
        
        # 保护级别
        protection_logits = self.protection_head(features)
        protection_probs = F.softmax(protection_logits, dim=-1)
        
        if deterministic:
            protection = protection_probs.argmax(dim=-1)
            protection_log_prob = torch.log(protection_probs.gather(1, protection.unsqueeze(-1)) + 1e-8)
        else:
            protection_dist = Categorical(protection_probs)
            protection = protection_dist.sample()
            protection_log_prob = protection_dist.log_prob(protection).unsqueeze(-1)
        
        actions = {
            'timing_offset': timing_offset,
            'fuel_adj': fuel_adj,
            'protection_level': protection
        }
        
        log_probs = timing_log_prob + fuel_log_prob + protection_log_prob
        
        if self.critic is not None:
            value = self.critic(features)
        else:
            value = torch.zeros(features.shape[0], 1, device=features.device)
        
        return actions, log_probs.sum(dim=-1, keepdim=True), value
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估动作"""
        features = self.encoder(obs)
        
        # 正时
        timing_mean = self.timing_mean(features)
        timing_std = self.timing_log_std.exp()
        timing_dist = Normal(timing_mean.squeeze(), timing_std)
        timing_log_prob = timing_dist.log_prob(actions['timing_offset'].squeeze())
        
        # 燃油
        fuel_mean = self.fuel_mean(features)
        fuel_std = self.fuel_log_std.exp()
        fuel_dist = Normal(fuel_mean.squeeze(), fuel_std)
        fuel_log_prob = fuel_dist.log_prob(actions['fuel_adj'].squeeze())
        
        # 保护级别
        protection_logits = self.protection_head(features)
        protection_probs = F.softmax(protection_logits, dim=-1)
        protection_dist = Categorical(protection_probs)
        protection_log_prob = protection_dist.log_prob(actions['protection_level'])
        protection_entropy = protection_dist.entropy()
        
        log_probs = timing_log_prob + fuel_log_prob + protection_log_prob
        entropy = protection_entropy
        
        if self.critic is not None:
            value = self.critic(features)
        else:
            value = torch.zeros(features.shape[0], 1, device=features.device)
        
        return log_probs, entropy, value


class ActorCritic(nn.Module):
    """统一的Actor-Critic接口"""
    
    def __init__(self, agent_type: str, obs_dim: int, **kwargs):
        super().__init__()
        
        if agent_type == 'diagnostic':
            self.network = DiagnosticNetwork(obs_dim, **kwargs)
        elif agent_type == 'control':
            self.network = ControlNetwork(obs_dim, **kwargs)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        self.agent_type = agent_type
    
    def forward(self, obs, deterministic=False):
        return self.network(obs, deterministic)
    
    def evaluate_actions(self, obs, actions):
        return self.network.evaluate_actions(obs, actions)
