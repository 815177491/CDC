"""
MAPPO共享Critic网络
===================
集中式训练，分布式执行（CTDE）
"""

import torch
import torch.nn as nn
from typing import Tuple


class SharedCritic(nn.Module):
    """
    MAPPO共享Critic
    
    输入: 所有智能体的联合观测（全局状态）
    输出: 各智能体的状态价值
    
    这是MAPPO区别于IPPO的核心：
    - Actor使用局部观测做决策
    - Critic使用全局观测评估价值
    """
    
    def __init__(
        self,
        diag_obs_dim: int,
        ctrl_obs_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.diag_obs_dim = diag_obs_dim
        self.ctrl_obs_dim = ctrl_obs_dim
        total_dim = diag_obs_dim + ctrl_obs_dim
        
        # 共享特征编码器
        self.encoder = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 各智能体的价值头
        self.value_diag = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.value_ctrl = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        obs_diag: torch.Tensor,
        obs_ctrl: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs_diag: 诊断智能体观测 [batch, diag_obs_dim]
            obs_ctrl: 控制智能体观测 [batch, ctrl_obs_dim]
            
        Returns:
            value_diag: 诊断智能体状态价值 [batch, 1]
            value_ctrl: 控制智能体状态价值 [batch, 1]
        """
        # 拼接全局观测
        joint_obs = torch.cat([obs_diag, obs_ctrl], dim=-1)
        
        # 共享特征
        features = self.encoder(joint_obs)
        
        # 各智能体价值
        value_diag = self.value_diag(features)
        value_ctrl = self.value_ctrl(features)
        
        return value_diag, value_ctrl
    
    def get_values(
        self,
        obs_diag: torch.Tensor,
        obs_ctrl: torch.Tensor
    ) -> Tuple[float, float]:
        """获取价值（用于rollout收集）"""
        with torch.no_grad():
            v_diag, v_ctrl = self.forward(obs_diag, obs_ctrl)
        return v_diag.item(), v_ctrl.item()
