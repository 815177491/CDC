"""
经验回放缓冲区
==============
存储双智能体的轨迹数据
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Transition:
    """单步转移数据"""
    obs_diag: np.ndarray
    obs_ctrl: np.ndarray
    action_diag: Dict
    action_ctrl: Dict
    log_prob_diag: float
    log_prob_ctrl: float
    reward_diag: float
    reward_ctrl: float
    value_diag: float
    value_ctrl: float
    done: bool


class RolloutBuffer:
    """
    双智能体经验回放缓冲区
    
    存储一个episode的完整轨迹用于PPO更新
    """
    
    def __init__(self, buffer_size: int = 2048, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        初始化缓冲区
        
        Args:
            buffer_size: 缓冲区大小
            gamma: 折扣因子
            gae_lambda: GAE参数
        """
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.clear()
    
    def clear(self):
        """清空缓冲区"""
        # 诊断智能体数据
        self.obs_diag: List[np.ndarray] = []
        self.actions_diag: List[Dict] = []
        self.log_probs_diag: List[float] = []
        self.rewards_diag: List[float] = []
        self.values_diag: List[float] = []
        
        # 控制智能体数据
        self.obs_ctrl: List[np.ndarray] = []
        self.actions_ctrl: List[Dict] = []
        self.log_probs_ctrl: List[float] = []
        self.rewards_ctrl: List[float] = []
        self.values_ctrl: List[float] = []
        
        # 共享数据
        self.dones: List[bool] = []
        
        # 计算的优势和回报
        self.advantages_diag: Optional[np.ndarray] = None
        self.returns_diag: Optional[np.ndarray] = None
        self.advantages_ctrl: Optional[np.ndarray] = None
        self.returns_ctrl: Optional[np.ndarray] = None
        
        self.ptr = 0
    
    def add(
        self,
        obs_diag: np.ndarray,
        obs_ctrl: np.ndarray,
        action_diag: Dict,
        action_ctrl: Dict,
        log_prob_diag: float,
        log_prob_ctrl: float,
        reward_diag: float,
        reward_ctrl: float,
        value_diag: float,
        value_ctrl: float,
        done: bool
    ):
        """添加一步转移"""
        self.obs_diag.append(obs_diag.copy())
        self.obs_ctrl.append(obs_ctrl.copy())
        self.actions_diag.append(action_diag)
        self.actions_ctrl.append(action_ctrl)
        self.log_probs_diag.append(log_prob_diag)
        self.log_probs_ctrl.append(log_prob_ctrl)
        self.rewards_diag.append(reward_diag)
        self.rewards_ctrl.append(reward_ctrl)
        self.values_diag.append(value_diag)
        self.values_ctrl.append(value_ctrl)
        self.dones.append(done)
        
        self.ptr += 1
    
    def compute_returns_and_advantages(
        self,
        last_value_diag: float,
        last_value_ctrl: float
    ):
        """
        计算GAE优势和回报
        """
        n = len(self.rewards_diag)
        
        # 诊断智能体
        self.advantages_diag = np.zeros(n, dtype=np.float32)
        self.returns_diag = np.zeros(n, dtype=np.float32)
        
        last_gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value_diag
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values_diag[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])
            
            delta = self.rewards_diag[t] + self.gamma * next_value * next_non_terminal - self.values_diag[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages_diag[t] = last_gae
        
        self.returns_diag = self.advantages_diag + np.array(self.values_diag)
        
        # 控制智能体
        self.advantages_ctrl = np.zeros(n, dtype=np.float32)
        self.returns_ctrl = np.zeros(n, dtype=np.float32)
        
        last_gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value_ctrl
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values_ctrl[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])
            
            delta = self.rewards_ctrl[t] + self.gamma * next_value * next_non_terminal - self.values_ctrl[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages_ctrl[t] = last_gae
        
        self.returns_ctrl = self.advantages_ctrl + np.array(self.values_ctrl)
        
        # 标准化优势
        self.advantages_diag = (self.advantages_diag - self.advantages_diag.mean()) / (self.advantages_diag.std() + 1e-8)
        self.advantages_ctrl = (self.advantages_ctrl - self.advantages_ctrl.mean()) / (self.advantages_ctrl.std() + 1e-8)
    
    def get_batches(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[Dict, Dict]:
        """
        生成训练批次
        
        Returns:
            (diag_batch, ctrl_batch)
        """
        n = len(self.obs_diag)
        indices = np.random.permutation(n)
        
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]
            
            # 诊断智能体批次
            diag_batch = {
                'obs': torch.FloatTensor(np.array([self.obs_diag[i] for i in batch_indices])).to(device),
                'actions': {
                    'fault_type': torch.LongTensor([self.actions_diag[i]['fault_type'] for i in batch_indices]).to(device),
                    'severity': torch.FloatTensor([self.actions_diag[i]['severity'] for i in batch_indices]).to(device),
                    'confidence': torch.FloatTensor([self.actions_diag[i]['confidence'] for i in batch_indices]).to(device)
                },
                'log_probs': torch.FloatTensor([self.log_probs_diag[i] for i in batch_indices]).to(device),
                'advantages': torch.FloatTensor(self.advantages_diag[batch_indices]).to(device),
                'returns': torch.FloatTensor(self.returns_diag[batch_indices]).to(device)
            }
            
            # 控制智能体批次
            ctrl_batch = {
                'obs': torch.FloatTensor(np.array([self.obs_ctrl[i] for i in batch_indices])).to(device),
                'actions': {
                    'timing_offset': torch.FloatTensor([self.actions_ctrl[i]['timing_offset'] for i in batch_indices]).to(device),
                    'fuel_adj': torch.FloatTensor([self.actions_ctrl[i]['fuel_adj'] for i in batch_indices]).to(device),
                    'protection_level': torch.LongTensor([self.actions_ctrl[i]['protection_level'] for i in batch_indices]).to(device)
                },
                'log_probs': torch.FloatTensor([self.log_probs_ctrl[i] for i in batch_indices]).to(device),
                'advantages': torch.FloatTensor(self.advantages_ctrl[batch_indices]).to(device),
                'returns': torch.FloatTensor(self.returns_ctrl[batch_indices]).to(device)
            }
            
            yield diag_batch, ctrl_batch
    
    def __len__(self):
        return len(self.obs_diag)
