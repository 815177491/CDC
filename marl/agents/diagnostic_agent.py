"""
诊断智能体
==========
基于观测和控制历史进行故障识别，支持并行双向决策架构
支持LSTM时序建模（Phase 4）

Author: CDC Project
Date: 2025-01-13
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Dict, Optional, Tuple, Any

from ..networks import DiagnosticNetwork


class DiagnosticAgent:
    """
    诊断智能体（并行双向决策架构）
    
    职责：基于观测序列和控制动作历史，识别故障类型和严重程度
    支持两阶段决策：
        1. encode + get_intent: 提取特征并生成意图向量发送给控制体
        2. get_action(peer_intent): 接收控制体意图后生成最终诊断动作
    
    LSTM模式：维护观测序列窗口，encode时将历史观测打包为3D张量
    """
    
    def __init__(
        self,
        obs_dim: int,
        n_fault_types: int = 4,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        intent_dim: int = 16,
        use_lstm: bool = False,
        seq_len: int = 8,
        device: str = 'cpu'
    ):
        """
        初始化诊断智能体
        
        Args:
            obs_dim: 观测维度
            n_fault_types: 故障类型数量
            hidden_dim: 隐层维度
            lr: 学习率
            intent_dim: 意图向量维度
            use_lstm: 是否启用LSTM时序建模
            seq_len: LSTM序列窗口长度
            device: 计算设备
        """
        self.obs_dim = obs_dim
        self.n_fault_types = n_fault_types
        self.device = torch.device(device)
        self.intent_dim = intent_dim
        self.use_lstm = use_lstm
        self.seq_len = seq_len
        
        # 创建网络
        self.network = DiagnosticNetwork(
            obs_dim=obs_dim,
            n_fault_types=n_fault_types,
            hidden_dim=hidden_dim,
            use_lstm=use_lstm,
            intent_dim=intent_dim
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        # LSTM序列窗口：累积历史观测用于时序建模
        if use_lstm:
            self._obs_buffer = deque(maxlen=seq_len)
    
    def reset_sequence(self):
        """重置LSTM序列状态（episode边界处调用）"""
        if self.use_lstm:
            self._obs_buffer.clear()
            self.network.reset_hidden(batch_size=1)
    
    def _prepare_obs_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """
        准备观测张量，LSTM模式返回3D序列张量
        
        Args:
            obs: 当前观测向量 [obs_dim,]
            
        Returns:
            LSTM模式: [1, seq_len, obs_dim]
            普通模式: [1, obs_dim]
        """
        if self.use_lstm:
            self._obs_buffer.append(obs.copy())
            # 用零填充不足seq_len的部分
            padded = list(self._obs_buffer)
            while len(padded) < self.seq_len:
                padded.insert(0, np.zeros_like(obs))
            seq = np.stack(padded, axis=0)  # [seq_len, obs_dim]
            return torch.FloatTensor(seq).unsqueeze(0).to(self.device)  # [1, seq_len, obs_dim]
        else:
            return torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # [1, obs_dim]
    
    def act(
        self, 
        obs: np.ndarray,
        deterministic: bool = False,
        peer_intent: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        选择动作（兼容单阶段与两阶段调用）
        
        Args:
            obs: 观测向量
            deterministic: 是否确定性输出
            peer_intent: 对方意图向量（可选，用于两阶段时直接传入）
            
        Returns:
            动作字典 {'fault_type', 'severity', 'confidence', 'log_prob', 'value'}
        """
        with torch.no_grad():
            obs_tensor = self._prepare_obs_tensor(obs)
            actions, log_probs, value = self.network(obs_tensor, deterministic, peer_intent)
        
        return {
            'fault_type': actions['fault_type'].item(),
            'severity': actions['severity'].cpu().numpy().flatten()[0],
            'confidence': actions['confidence'].cpu().numpy().flatten()[0],
            'log_prob': log_probs.cpu().numpy().flatten()[0],
            'value': value.item()
        }
    
    def encode_and_intent(self, obs: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        两阶段决策 - Phase 1: 特征提取和意图生成
        
        Args:
            obs: 观测向量
            
        Returns:
            (features, intent): 编码特征和意图向量
        """
        with torch.no_grad():
            obs_tensor = self._prepare_obs_tensor(obs)
            features = self.network.encode(obs_tensor)
            intent = self.network.get_intent(features)
        return features, intent
    
    def act_with_intent(
        self,
        features: torch.Tensor,
        peer_intent: torch.Tensor,
        deterministic: bool = False
    ) -> Dict[str, Any]:
        """
        两阶段决策 - Phase 2: 接收对方意图后生成动作
        
        Args:
            features: Phase 1 编码特征
            peer_intent: 对方意图向量
            deterministic: 是否确定性输出
            
        Returns:
            动作字典
        """
        with torch.no_grad():
            actions, log_probs, value = self.network.get_action(features, peer_intent, deterministic)
        
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
            obs_tensor = self._prepare_obs_tensor(obs)
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
        value_coef: float = 0.5,
        peer_intent_batch: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        PPO更新（支持意图融合）
        
        Args:
            peer_intent_batch: 对方意图向量批次 [batch, intent_dim]
            
        Returns:
            损失字典
        """
        # 评估动作
        new_log_probs, entropy, values = self.network.evaluate_actions(
            obs_batch, action_batch, peer_intent=peer_intent_batch
        )
        
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
