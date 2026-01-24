"""
诊断智能体
==========
基于观测和控制历史进行故障识别
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any

from ..networks import DiagnosticNetwork


class DiagnosticAgent:
    """
    诊断智能体
    
    职责：基于观测序列和控制动作历史，识别故障类型和严重程度
    """
    
    def __init__(
        self,
        obs_dim: int,
        n_fault_types: int = 4,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        device: str = 'cpu'
    ):
        """
        初始化诊断智能体
        
        Args:
            obs_dim: 观测维度
            n_fault_types: 故障类型数量
            hidden_dim: 隐层维度
            lr: 学习率
            device: 计算设备
        """
        self.obs_dim = obs_dim
        self.n_fault_types = n_fault_types
        self.device = torch.device(device)
        
        # 创建网络
        self.network = DiagnosticNetwork(
            obs_dim=obs_dim,
            n_fault_types=n_fault_types,
            hidden_dim=hidden_dim,
            use_lstm=False  # 简化版不使用LSTM
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
    
    def act(
        self, 
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Dict[str, Any]:
        """
        选择动作
        
        Args:
            obs: 观测向量
            deterministic: 是否确定性输出
            
        Returns:
            动作字典 {'fault_type', 'severity', 'confidence', 'log_prob', 'value'}
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            actions, log_probs, value = self.network(obs_tensor, deterministic)
        
        return {
            'fault_type': actions['fault_type'].item(),
            'severity': actions['severity'].cpu().numpy().flatten()[0],
            'confidence': actions['confidence'].cpu().numpy().flatten()[0],
            'log_prob': log_probs.cpu().numpy().flatten()[0],
            'value': value.item()
        }
    
    def get_value(self, obs: np.ndarray) -> float:
        """获取状态价值"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            _, _, value = self.network(obs_tensor)
        return value.item()
    
    def update(
        self,
        obs_batch: torch.Tensor,
        action_batch: Dict[str, torch.Tensor],
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5
    ) -> Dict[str, float]:
        """
        PPO更新
        
        Returns:
            损失字典
        """
        # 评估动作
        new_log_probs, entropy, values = self.network.evaluate_actions(obs_batch, action_batch)
        
        # PPO损失
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值损失
        value_loss = nn.functional.mse_loss(values.squeeze(), returns)
        
        # 熵奖励
        entropy_loss = -entropy.mean()
        
        # 总损失
        loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
        
        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item()
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
