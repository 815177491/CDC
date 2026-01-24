#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多智能体强化学习算法
====================
实现MAPPO和QMIX等多智能体专用算法，用于双智能体（诊断+控制）的联合训练

算法:
1. MAPPO (Multi-Agent PPO): 集中式Critic + 分布式Actor
2. QMIX: 值分解方法，单调性约束下的联合Q函数分解

适用于:
- 诊断智能体 + 控制智能体的协同训练
- 共享全局状态信息
- 保持各自策略独立性

References:
- MAPPO: Yu et al., "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games", NeurIPS 2021
- QMIX: Rashid et al., "QMIX: Monotonic Value Function Factorisation for DMARL", ICML 2018
- VDN: Sunehag et al., "Value-Decomposition Networks For Cooperative MARL", AAMAS 2018

Author: CDC Project
Date: 2026-01-24
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
import warnings

# 尝试导入PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    warnings.warn("PyTorch not available. Multi-Agent algorithms will not work.")


# ============================================================
# 多智能体经验
# ============================================================

@dataclass
class MultiAgentExperience:
    """多智能体经验"""
    # 诊断智能体
    diag_state: np.ndarray
    diag_residual_seq: np.ndarray
    diag_action: int
    diag_reward: float
    
    # 控制智能体
    ctrl_state: np.ndarray
    ctrl_action: int
    ctrl_reward: float
    
    # 可选字段 (带默认值)
    diag_log_prob: float = 0.0
    diag_value: float = 0.0
    ctrl_log_prob: float = 0.0
    ctrl_value: float = 0.0
    global_state: np.ndarray = field(default_factory=lambda: np.array([]))
    next_global_state: np.ndarray = field(default_factory=lambda: np.array([]))
    done: bool = False
    
    # 用于QMIX的联合奖励
    joint_reward: float = 0.0


class MultiAgentReplayBuffer:
    """多智能体经验回放缓冲区"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: MultiAgentExperience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[MultiAgentExperience]:
        import random
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def clear(self):
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


# ============================================================
# 基类
# ============================================================

class BaseMultiAgentAlgorithm(ABC):
    """多智能体算法基类"""
    
    def __init__(self, 
                 diag_state_dim: int,
                 diag_action_dim: int,
                 ctrl_state_dim: int,
                 ctrl_action_dim: int,
                 config: Dict = None):
        self.diag_state_dim = diag_state_dim
        self.diag_action_dim = diag_action_dim
        self.ctrl_state_dim = ctrl_state_dim
        self.ctrl_action_dim = ctrl_action_dim
        self.config = config or {}
        
        self.training_step = 0
        
        if TORCH_AVAILABLE:
            device_str = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            self.device = torch.device(device_str)
        else:
            self.device = None
    
    @abstractmethod
    def select_actions(self, 
                      diag_state: np.ndarray,
                      diag_residual_seq: np.ndarray,
                      ctrl_state: np.ndarray,
                      explore: bool = True) -> Tuple[int, int]:
        """选择两个智能体的动作"""
        pass
    
    @abstractmethod
    def update(self, batch: List[MultiAgentExperience]) -> Dict[str, float]:
        """更新网络"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """保存模型"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """加载模型"""
        pass


if TORCH_AVAILABLE:
    
    # ============================================================
    # MAPPO网络组件
    # ============================================================
    
    class MAPPOActorNetwork(nn.Module):
        """
        MAPPO Actor网络 (分布式)
        每个智能体独立的策略网络
        """
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
            super().__init__()
            
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            
            self._init_weights()
        
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.01)
                    nn.init.constant_(m.bias, 0)
        
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return self.network(state)
        
        def get_action(self, state: torch.Tensor, 
                      deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            logits = self.forward(state)
            probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                action = probs.argmax(dim=-1)
                log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-8).squeeze(-1)
            else:
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            return action, log_prob, probs
        
        def evaluate_action(self, state: torch.Tensor, 
                           action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            logits = self.forward(state)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            return log_prob, entropy
    
    
    class MAPPODiagnosisActor(nn.Module):
        """
        诊断智能体的MAPPO Actor (带残差序列处理)
        """
        
        def __init__(self, state_dim: int, action_dim: int, 
                     hidden_dim: int = 256, sequence_len: int = 10):
            super().__init__()
            
            # 残差序列编码器
            self.residual_encoder = nn.Sequential(
                nn.Conv1d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            )
            
            # 融合后的策略网络
            fusion_dim = 64 + state_dim
            self.policy = nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            
            self._init_weights()
        
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.01)
                    nn.init.constant_(m.bias, 0)
        
        def forward(self, state: torch.Tensor, 
                   residual_seq: torch.Tensor) -> torch.Tensor:
            residual_feat = self.residual_encoder(residual_seq)
            combined = torch.cat([residual_feat, state], dim=-1)
            return self.policy(combined)
        
        def get_action(self, state: torch.Tensor,
                      residual_seq: torch.Tensor,
                      deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            logits = self.forward(state, residual_seq)
            probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                action = probs.argmax(dim=-1)
                log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-8).squeeze(-1)
            else:
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            return action, log_prob, probs
        
        def evaluate_action(self, state: torch.Tensor,
                           residual_seq: torch.Tensor,
                           action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            logits = self.forward(state, residual_seq)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            return log_prob, entropy
    
    
    class MAPPOCriticNetwork(nn.Module):
        """
        MAPPO Critic网络 (集中式)
        接收全局状态，输出状态价值
        """
        
        def __init__(self, global_state_dim: int, hidden_dim: int = 256):
            super().__init__()
            
            self.network = nn.Sequential(
                nn.Linear(global_state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            
            self._init_weights()
        
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    nn.init.constant_(m.bias, 0)
        
        def forward(self, global_state: torch.Tensor) -> torch.Tensor:
            return self.network(global_state)
    
    
    # ============================================================
    # MAPPO算法
    # ============================================================
    
    class MAPPO(BaseMultiAgentAlgorithm):
        """
        Multi-Agent Proximal Policy Optimization (MAPPO)
        
        特点:
        - 集中式训练，分布式执行 (CTDE)
        - 每个智能体有独立的Actor
        - 共享一个Critic (使用全局状态)
        - PPO-Clip目标函数
        
        适用于柴油机诊-控协同:
        - 诊断Actor: 输入残差序列，输出故障诊断
        - 控制Actor: 输入系统状态，输出VIT/燃油调整
        - Critic: 输入全局状态，评估联合价值
        """
        
        def __init__(self,
                     diag_state_dim: int,
                     diag_action_dim: int,
                     ctrl_state_dim: int,
                     ctrl_action_dim: int,
                     config: Dict = None):
            super().__init__(diag_state_dim, diag_action_dim, 
                           ctrl_state_dim, ctrl_action_dim, config)
            
            # 超参数
            self.lr_actor = self.config.get('lr_actor', 3e-4)
            self.lr_critic = self.config.get('lr_critic', 3e-4)
            self.gamma = self.config.get('gamma', 0.99)
            self.gae_lambda = self.config.get('gae_lambda', 0.95)
            self.clip_epsilon = self.config.get('clip_epsilon', 0.2)
            self.entropy_coef = self.config.get('entropy_coef', 0.01)
            self.value_coef = self.config.get('value_coef', 0.5)
            self.max_grad_norm = self.config.get('max_grad_norm', 0.5)
            self.ppo_epochs = self.config.get('ppo_epochs', 10)
            self.batch_size = self.config.get('batch_size', 64)
            self.sequence_len = self.config.get('sequence_len', 10)
            
            hidden_dim = self.config.get('hidden_dim', 256)
            
            # 全局状态维度 = 诊断状态 + 控制状态 + 残差特征
            self.global_state_dim = diag_state_dim + ctrl_state_dim + 64  # 64是残差特征维度
            
            # 网络
            self.diag_actor = MAPPODiagnosisActor(
                diag_state_dim, diag_action_dim, hidden_dim, self.sequence_len
            ).to(self.device)
            
            self.ctrl_actor = MAPPOActorNetwork(
                ctrl_state_dim, ctrl_action_dim, hidden_dim
            ).to(self.device)
            
            self.critic = MAPPOCriticNetwork(
                self.global_state_dim, hidden_dim
            ).to(self.device)
            
            # 共享残差编码器 (用于Critic)
            self.shared_residual_encoder = nn.Sequential(
                nn.Conv1d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            ).to(self.device)
            
            # 优化器
            self.diag_actor_optimizer = optim.Adam(
                self.diag_actor.parameters(), lr=self.lr_actor
            )
            self.ctrl_actor_optimizer = optim.Adam(
                self.ctrl_actor.parameters(), lr=self.lr_actor
            )
            self.critic_optimizer = optim.Adam(
                list(self.critic.parameters()) + 
                list(self.shared_residual_encoder.parameters()),
                lr=self.lr_critic
            )
            
            # 轨迹缓存
            self.trajectory = []
            
            print(f"[MAPPO] 初始化完成")
            print(f"  - 诊断动作空间: {diag_action_dim}")
            print(f"  - 控制动作空间: {ctrl_action_dim}")
            print(f"  - 全局状态维度: {self.global_state_dim}")
            print(f"  - 设备: {self.device}")
        
        def _encode_global_state(self, diag_state: torch.Tensor,
                                ctrl_state: torch.Tensor,
                                residual_seq: torch.Tensor) -> torch.Tensor:
            """编码全局状态"""
            residual_feat = self.shared_residual_encoder(residual_seq)
            return torch.cat([diag_state, ctrl_state, residual_feat], dim=-1)
        
        def select_actions(self,
                          diag_state: np.ndarray,
                          diag_residual_seq: np.ndarray,
                          ctrl_state: np.ndarray,
                          explore: bool = True) -> Tuple[int, int, Dict]:
            """
            选择两个智能体的动作
            
            Returns:
                diag_action: 诊断动作
                ctrl_action: 控制动作
                info: 包含log_prob和value的字典
            """
            with torch.no_grad():
                diag_state_t = torch.FloatTensor(diag_state).unsqueeze(0).to(self.device)
                residual_seq_t = torch.FloatTensor(diag_residual_seq).unsqueeze(0).to(self.device)
                ctrl_state_t = torch.FloatTensor(ctrl_state).unsqueeze(0).to(self.device)
                
                # 诊断动作
                diag_action, diag_log_prob, _ = self.diag_actor.get_action(
                    diag_state_t, residual_seq_t, deterministic=not explore
                )
                
                # 控制动作
                ctrl_action, ctrl_log_prob, _ = self.ctrl_actor.get_action(
                    ctrl_state_t, deterministic=not explore
                )
                
                # 全局价值
                global_state = self._encode_global_state(
                    diag_state_t, ctrl_state_t, residual_seq_t
                )
                value = self.critic(global_state)
                
                return (
                    diag_action.item(),
                    ctrl_action.item(),
                    {
                        'diag_log_prob': diag_log_prob.item(),
                        'ctrl_log_prob': ctrl_log_prob.item(),
                        'value': value.item()
                    }
                )
        
        def store_transition(self, experience: MultiAgentExperience):
            """存储转换"""
            self.trajectory.append(experience)
        
        def compute_gae(self, rewards: List[float], values: List[float], 
                       dones: List[bool], next_value: float) -> Tuple[List, List]:
            """计算GAE优势估计"""
            advantages = []
            returns = []
            gae = 0
            
            values = values + [next_value]
            
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + values[t])
            
            return advantages, returns
        
        def update(self, batch: List[MultiAgentExperience] = None) -> Dict[str, float]:
            """
            执行MAPPO更新
            """
            if len(self.trajectory) == 0:
                return {}
            
            # 使用trajectory而不是batch
            experiences = self.trajectory
            
            # 提取数据
            diag_states = torch.FloatTensor(
                np.array([e.diag_state for e in experiences])
            ).to(self.device)
            diag_residual_seqs = torch.FloatTensor(
                np.array([e.diag_residual_seq for e in experiences])
            ).to(self.device)
            diag_actions = torch.LongTensor(
                [e.diag_action for e in experiences]
            ).to(self.device)
            diag_old_log_probs = torch.FloatTensor(
                [e.diag_log_prob for e in experiences]
            ).to(self.device)
            
            ctrl_states = torch.FloatTensor(
                np.array([e.ctrl_state for e in experiences])
            ).to(self.device)
            ctrl_actions = torch.LongTensor(
                [e.ctrl_action for e in experiences]
            ).to(self.device)
            ctrl_old_log_probs = torch.FloatTensor(
                [e.ctrl_log_prob for e in experiences]
            ).to(self.device)
            
            # 使用联合奖励
            rewards = [e.joint_reward for e in experiences]
            values = [e.diag_value for e in experiences]  # 使用共享Critic的值
            dones = [e.done for e in experiences]
            
            # 计算GAE
            with torch.no_grad():
                last_exp = experiences[-1]
                last_diag_state = torch.FloatTensor(last_exp.diag_state).unsqueeze(0).to(self.device)
                last_ctrl_state = torch.FloatTensor(last_exp.ctrl_state).unsqueeze(0).to(self.device)
                last_residual_seq = torch.FloatTensor(last_exp.diag_residual_seq).unsqueeze(0).to(self.device)
                last_global_state = self._encode_global_state(
                    last_diag_state, last_ctrl_state, last_residual_seq
                )
                next_value = self.critic(last_global_state).item()
            
            advantages, returns = self.compute_gae(rewards, values, dones, next_value)
            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
            
            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO更新
            total_diag_policy_loss = 0
            total_ctrl_policy_loss = 0
            total_value_loss = 0
            total_diag_entropy = 0
            total_ctrl_entropy = 0
            n_updates = 0
            
            for _ in range(self.ppo_epochs):
                indices = torch.randperm(len(experiences))
                
                for start in range(0, len(experiences), self.batch_size):
                    end = min(start + self.batch_size, len(experiences))
                    batch_indices = indices[start:end]
                    
                    # 批次数据
                    b_diag_states = diag_states[batch_indices]
                    b_diag_residual_seqs = diag_residual_seqs[batch_indices]
                    b_diag_actions = diag_actions[batch_indices]
                    b_diag_old_log_probs = diag_old_log_probs[batch_indices]
                    
                    b_ctrl_states = ctrl_states[batch_indices]
                    b_ctrl_actions = ctrl_actions[batch_indices]
                    b_ctrl_old_log_probs = ctrl_old_log_probs[batch_indices]
                    
                    b_advantages = advantages[batch_indices]
                    b_returns = returns[batch_indices]
                    
                    # 诊断Actor更新
                    diag_log_probs, diag_entropy = self.diag_actor.evaluate_action(
                        b_diag_states, b_diag_residual_seqs, b_diag_actions
                    )
                    diag_ratio = torch.exp(diag_log_probs - b_diag_old_log_probs)
                    diag_surr1 = diag_ratio * b_advantages
                    diag_surr2 = torch.clamp(diag_ratio, 1 - self.clip_epsilon, 
                                            1 + self.clip_epsilon) * b_advantages
                    diag_policy_loss = -torch.min(diag_surr1, diag_surr2).mean()
                    
                    self.diag_actor_optimizer.zero_grad()
                    (diag_policy_loss - self.entropy_coef * diag_entropy.mean()).backward()
                    torch.nn.utils.clip_grad_norm_(self.diag_actor.parameters(), self.max_grad_norm)
                    self.diag_actor_optimizer.step()
                    
                    # 控制Actor更新
                    ctrl_log_probs, ctrl_entropy = self.ctrl_actor.evaluate_action(
                        b_ctrl_states, b_ctrl_actions
                    )
                    ctrl_ratio = torch.exp(ctrl_log_probs - b_ctrl_old_log_probs)
                    ctrl_surr1 = ctrl_ratio * b_advantages
                    ctrl_surr2 = torch.clamp(ctrl_ratio, 1 - self.clip_epsilon,
                                            1 + self.clip_epsilon) * b_advantages
                    ctrl_policy_loss = -torch.min(ctrl_surr1, ctrl_surr2).mean()
                    
                    self.ctrl_actor_optimizer.zero_grad()
                    (ctrl_policy_loss - self.entropy_coef * ctrl_entropy.mean()).backward()
                    torch.nn.utils.clip_grad_norm_(self.ctrl_actor.parameters(), self.max_grad_norm)
                    self.ctrl_actor_optimizer.step()
                    
                    # Critic更新
                    global_states = self._encode_global_state(
                        b_diag_states, b_ctrl_states, b_diag_residual_seqs
                    )
                    values = self.critic(global_states).squeeze()
                    value_loss = F.mse_loss(values, b_returns)
                    
                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()
                    
                    total_diag_policy_loss += diag_policy_loss.item()
                    total_ctrl_policy_loss += ctrl_policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_diag_entropy += diag_entropy.mean().item()
                    total_ctrl_entropy += ctrl_entropy.mean().item()
                    n_updates += 1
            
            # 清空轨迹
            self.trajectory.clear()
            self.training_step += 1
            
            return {
                'diag_policy_loss': total_diag_policy_loss / n_updates,
                'ctrl_policy_loss': total_ctrl_policy_loss / n_updates,
                'value_loss': total_value_loss / n_updates,
                'diag_entropy': total_diag_entropy / n_updates,
                'ctrl_entropy': total_ctrl_entropy / n_updates
            }
        
        def save(self, path: str):
            """保存模型"""
            torch.save({
                'diag_actor': self.diag_actor.state_dict(),
                'ctrl_actor': self.ctrl_actor.state_dict(),
                'critic': self.critic.state_dict(),
                'shared_residual_encoder': self.shared_residual_encoder.state_dict(),
                'training_step': self.training_step,
                'config': self.config
            }, path)
            print(f"[MAPPO] 模型已保存: {path}")
        
        def load(self, path: str):
            """加载模型"""
            checkpoint = torch.load(path, map_location=self.device)
            self.diag_actor.load_state_dict(checkpoint['diag_actor'])
            self.ctrl_actor.load_state_dict(checkpoint['ctrl_actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.shared_residual_encoder.load_state_dict(checkpoint['shared_residual_encoder'])
            self.training_step = checkpoint['training_step']
            print(f"[MAPPO] 模型已加载: {path}")
    
    
    # ============================================================
    # QMIX网络组件
    # ============================================================
    
    class QMixingNetwork(nn.Module):
        """
        QMIX混合网络
        使用超网络生成混合权重，确保联合Q函数对个体Q值的单调性
        """
        
        def __init__(self, n_agents: int, state_dim: int, 
                     hidden_dim: int = 64, hypernet_dim: int = 64):
            super().__init__()
            
            self.n_agents = n_agents
            self.hidden_dim = hidden_dim
            
            # 超网络1: 生成第一层权重
            self.hyper_w1 = nn.Sequential(
                nn.Linear(state_dim, hypernet_dim),
                nn.ReLU(),
                nn.Linear(hypernet_dim, n_agents * hidden_dim)
            )
            
            # 超网络2: 生成第二层权重
            self.hyper_w2 = nn.Sequential(
                nn.Linear(state_dim, hypernet_dim),
                nn.ReLU(),
                nn.Linear(hypernet_dim, hidden_dim)
            )
            
            # 偏置超网络
            self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
            self.hyper_b2 = nn.Sequential(
                nn.Linear(state_dim, hypernet_dim),
                nn.ReLU(),
                nn.Linear(hypernet_dim, 1)
            )
        
        def forward(self, agent_qs: torch.Tensor, 
                   state: torch.Tensor) -> torch.Tensor:
            """
            Args:
                agent_qs: 各智能体Q值 [batch, n_agents]
                state: 全局状态 [batch, state_dim]
            
            Returns:
                联合Q值 [batch, 1]
            """
            batch_size = agent_qs.shape[0]
            
            # 生成权重 (使用abs确保非负，保证单调性)
            w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.n_agents, self.hidden_dim)
            b1 = self.hyper_b1(state).view(batch_size, 1, self.hidden_dim)
            
            w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.hidden_dim, 1)
            b2 = self.hyper_b2(state).view(batch_size, 1, 1)
            
            # 前向传播
            hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
            q_tot = torch.bmm(hidden, w2) + b2
            
            return q_tot.squeeze(-1).squeeze(-1)
    
    
    class AgentQNetwork(nn.Module):
        """
        单个智能体的Q网络
        """
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
            super().__init__()
            
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return self.network(state)
    
    
    class DiagnosisQNetwork(nn.Module):
        """
        诊断智能体Q网络 (带残差序列处理)
        """
        
        def __init__(self, state_dim: int, action_dim: int, 
                     hidden_dim: int = 128, sequence_len: int = 10):
            super().__init__()
            
            self.residual_encoder = nn.Sequential(
                nn.Conv1d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            )
            
            fusion_dim = 64 + state_dim
            self.q_net = nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        
        def forward(self, state: torch.Tensor, 
                   residual_seq: torch.Tensor) -> torch.Tensor:
            residual_feat = self.residual_encoder(residual_seq)
            combined = torch.cat([residual_feat, state], dim=-1)
            return self.q_net(combined)
    
    
    # ============================================================
    # QMIX算法
    # ============================================================
    
    class QMIX(BaseMultiAgentAlgorithm):
        """
        QMIX: 值分解多智能体强化学习
        
        特点:
        - 每个智能体学习自己的Q函数
        - 混合网络将个体Q值组合为联合Q值
        - 单调性约束确保IGM (Individual-Global-Max)
        - 支持离线训练
        
        适用于:
        - 需要协调但无需完全集中式训练的场景
        - 可以使用经验回放
        """
        
        def __init__(self,
                     diag_state_dim: int,
                     diag_action_dim: int,
                     ctrl_state_dim: int,
                     ctrl_action_dim: int,
                     config: Dict = None):
            super().__init__(diag_state_dim, diag_action_dim,
                           ctrl_state_dim, ctrl_action_dim, config)
            
            # 超参数
            self.lr = self.config.get('lr', 3e-4)
            self.gamma = self.config.get('gamma', 0.99)
            self.tau = self.config.get('tau', 0.005)
            self.batch_size = self.config.get('batch_size', 128)
            self.buffer_size = self.config.get('buffer_size', 100000)
            self.epsilon = self.config.get('epsilon', 1.0)
            self.epsilon_min = self.config.get('epsilon_min', 0.05)
            self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
            self.sequence_len = self.config.get('sequence_len', 10)
            
            hidden_dim = self.config.get('hidden_dim', 128)
            
            # 全局状态维度
            self.global_state_dim = diag_state_dim + ctrl_state_dim + 64
            
            # 智能体Q网络
            self.diag_q = DiagnosisQNetwork(
                diag_state_dim, diag_action_dim, hidden_dim, self.sequence_len
            ).to(self.device)
            self.diag_q_target = DiagnosisQNetwork(
                diag_state_dim, diag_action_dim, hidden_dim, self.sequence_len
            ).to(self.device)
            self.diag_q_target.load_state_dict(self.diag_q.state_dict())
            
            self.ctrl_q = AgentQNetwork(
                ctrl_state_dim, ctrl_action_dim, hidden_dim
            ).to(self.device)
            self.ctrl_q_target = AgentQNetwork(
                ctrl_state_dim, ctrl_action_dim, hidden_dim
            ).to(self.device)
            self.ctrl_q_target.load_state_dict(self.ctrl_q.state_dict())
            
            # 混合网络
            self.mixer = QMixingNetwork(
                n_agents=2, state_dim=self.global_state_dim, hidden_dim=64
            ).to(self.device)
            self.mixer_target = QMixingNetwork(
                n_agents=2, state_dim=self.global_state_dim, hidden_dim=64
            ).to(self.device)
            self.mixer_target.load_state_dict(self.mixer.state_dict())
            
            # 残差编码器 (用于全局状态)
            self.residual_encoder = nn.Sequential(
                nn.Conv1d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            ).to(self.device)
            
            # 优化器
            self.optimizer = optim.Adam(
                list(self.diag_q.parameters()) +
                list(self.ctrl_q.parameters()) +
                list(self.mixer.parameters()) +
                list(self.residual_encoder.parameters()),
                lr=self.lr
            )
            
            # 经验回放
            self.buffer = MultiAgentReplayBuffer(self.buffer_size)
            
            print(f"[QMIX] 初始化完成")
            print(f"  - 诊断动作空间: {diag_action_dim}")
            print(f"  - 控制动作空间: {ctrl_action_dim}")
            print(f"  - 全局状态维度: {self.global_state_dim}")
            print(f"  - 设备: {self.device}")
        
        def _encode_global_state(self, diag_state: torch.Tensor,
                                ctrl_state: torch.Tensor,
                                residual_seq: torch.Tensor) -> torch.Tensor:
            """编码全局状态"""
            residual_feat = self.residual_encoder(residual_seq)
            return torch.cat([diag_state, ctrl_state, residual_feat], dim=-1)
        
        def select_actions(self,
                          diag_state: np.ndarray,
                          diag_residual_seq: np.ndarray,
                          ctrl_state: np.ndarray,
                          explore: bool = True) -> Tuple[int, int, Dict]:
            """选择动作 (epsilon-greedy)"""
            import random
            
            with torch.no_grad():
                diag_state_t = torch.FloatTensor(diag_state).unsqueeze(0).to(self.device)
                residual_seq_t = torch.FloatTensor(diag_residual_seq).unsqueeze(0).to(self.device)
                ctrl_state_t = torch.FloatTensor(ctrl_state).unsqueeze(0).to(self.device)
                
                # 诊断动作
                if explore and random.random() < self.epsilon:
                    diag_action = random.randint(0, self.diag_action_dim - 1)
                else:
                    diag_q_values = self.diag_q(diag_state_t, residual_seq_t)
                    diag_action = diag_q_values.argmax(dim=-1).item()
                
                # 控制动作
                if explore and random.random() < self.epsilon:
                    ctrl_action = random.randint(0, self.ctrl_action_dim - 1)
                else:
                    ctrl_q_values = self.ctrl_q(ctrl_state_t)
                    ctrl_action = ctrl_q_values.argmax(dim=-1).item()
                
                return diag_action, ctrl_action, {}
        
        def store_experience(self, experience: MultiAgentExperience):
            """存储经验"""
            self.buffer.push(experience)
        
        def update(self, batch: List[MultiAgentExperience] = None) -> Dict[str, float]:
            """执行QMIX更新"""
            if len(self.buffer) < self.batch_size:
                return {}
            
            # 采样batch
            experiences = self.buffer.sample(self.batch_size)
            
            # 转换为tensor
            diag_states = torch.FloatTensor(
                np.array([e.diag_state for e in experiences])
            ).to(self.device)
            diag_residual_seqs = torch.FloatTensor(
                np.array([e.diag_residual_seq for e in experiences])
            ).to(self.device)
            diag_actions = torch.LongTensor(
                [e.diag_action for e in experiences]
            ).to(self.device)
            
            ctrl_states = torch.FloatTensor(
                np.array([e.ctrl_state for e in experiences])
            ).to(self.device)
            ctrl_actions = torch.LongTensor(
                [e.ctrl_action for e in experiences]
            ).to(self.device)
            
            rewards = torch.FloatTensor(
                [e.joint_reward for e in experiences]
            ).to(self.device)
            
            # 下一状态 (简化：使用当前状态)
            next_diag_states = diag_states
            next_diag_residual_seqs = diag_residual_seqs
            next_ctrl_states = ctrl_states
            
            dones = torch.FloatTensor(
                [e.done for e in experiences]
            ).to(self.device)
            
            # 当前Q值
            diag_q_values = self.diag_q(diag_states, diag_residual_seqs)
            diag_q = diag_q_values.gather(1, diag_actions.unsqueeze(1)).squeeze()
            
            ctrl_q_values = self.ctrl_q(ctrl_states)
            ctrl_q = ctrl_q_values.gather(1, ctrl_actions.unsqueeze(1)).squeeze()
            
            # 组合Q值
            agent_qs = torch.stack([diag_q, ctrl_q], dim=1)
            global_states = self._encode_global_state(
                diag_states, ctrl_states, diag_residual_seqs
            )
            q_tot = self.mixer(agent_qs, global_states)
            
            # 目标Q值
            with torch.no_grad():
                next_diag_q_values = self.diag_q_target(
                    next_diag_states, next_diag_residual_seqs
                )
                next_diag_q = next_diag_q_values.max(dim=-1)[0]
                
                next_ctrl_q_values = self.ctrl_q_target(next_ctrl_states)
                next_ctrl_q = next_ctrl_q_values.max(dim=-1)[0]
                
                next_agent_qs = torch.stack([next_diag_q, next_ctrl_q], dim=1)
                next_global_states = self._encode_global_state(
                    next_diag_states, next_ctrl_states, next_diag_residual_seqs
                )
                next_q_tot = self.mixer_target(next_agent_qs, next_global_states)
                
                target_q_tot = rewards + self.gamma * (1 - dones) * next_q_tot
            
            # 损失
            loss = F.mse_loss(q_tot, target_q_tot)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.diag_q.parameters()) +
                list(self.ctrl_q.parameters()) +
                list(self.mixer.parameters()),
                1.0
            )
            self.optimizer.step()
            
            # 软更新目标网络
            for param, target_param in zip(self.diag_q.parameters(), 
                                          self.diag_q_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            
            for param, target_param in zip(self.ctrl_q.parameters(),
                                          self.ctrl_q_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            
            for param, target_param in zip(self.mixer.parameters(),
                                          self.mixer_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            
            # 衰减epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            self.training_step += 1
            
            return {
                'loss': loss.item(),
                'q_tot': q_tot.mean().item(),
                'epsilon': self.epsilon
            }
        
        def save(self, path: str):
            """保存模型"""
            torch.save({
                'diag_q': self.diag_q.state_dict(),
                'ctrl_q': self.ctrl_q.state_dict(),
                'mixer': self.mixer.state_dict(),
                'residual_encoder': self.residual_encoder.state_dict(),
                'epsilon': self.epsilon,
                'training_step': self.training_step,
                'config': self.config
            }, path)
            print(f"[QMIX] 模型已保存: {path}")
        
        def load(self, path: str):
            """加载模型"""
            checkpoint = torch.load(path, map_location=self.device)
            self.diag_q.load_state_dict(checkpoint['diag_q'])
            self.ctrl_q.load_state_dict(checkpoint['ctrl_q'])
            self.mixer.load_state_dict(checkpoint['mixer'])
            self.residual_encoder.load_state_dict(checkpoint['residual_encoder'])
            self.epsilon = checkpoint['epsilon']
            self.training_step = checkpoint['training_step']
            
            # 同步目标网络
            self.diag_q_target.load_state_dict(self.diag_q.state_dict())
            self.ctrl_q_target.load_state_dict(self.ctrl_q.state_dict())
            self.mixer_target.load_state_dict(self.mixer.state_dict())
            
            print(f"[QMIX] 模型已加载: {path}")


# ============================================================
# 工厂函数
# ============================================================

def get_multi_agent_algorithm(name: str,
                             diag_state_dim: int,
                             diag_action_dim: int,
                             ctrl_state_dim: int,
                             ctrl_action_dim: int,
                             config: Dict = None) -> BaseMultiAgentAlgorithm:
    """
    获取多智能体算法实例
    
    Args:
        name: 算法名称 ('MAPPO' 或 'QMIX')
        diag_state_dim: 诊断状态维度
        diag_action_dim: 诊断动作维度
        ctrl_state_dim: 控制状态维度
        ctrl_action_dim: 控制动作维度
        config: 配置参数
    
    Returns:
        算法实例
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    algorithms = {
        'MAPPO': MAPPO,
        'QMIX': QMIX
    }
    
    if name not in algorithms:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(algorithms.keys())}")
    
    return algorithms[name](
        diag_state_dim, diag_action_dim,
        ctrl_state_dim, ctrl_action_dim,
        config
    )


if __name__ == "__main__":
    # 测试MAPPO
    print("=" * 60)
    print("测试 MAPPO")
    print("=" * 60)
    
    mappo = get_multi_agent_algorithm(
        'MAPPO',
        diag_state_dim=12,
        diag_action_dim=20,
        ctrl_state_dim=10,
        ctrl_action_dim=45
    )
    
    # 模拟选择动作
    diag_state = np.random.randn(12).astype(np.float32)
    diag_residual_seq = np.random.randn(3, 10).astype(np.float32)
    ctrl_state = np.random.randn(10).astype(np.float32)
    
    diag_action, ctrl_action, info = mappo.select_actions(
        diag_state, diag_residual_seq, ctrl_state
    )
    print(f"诊断动作: {diag_action}, 控制动作: {ctrl_action}")
    print(f"Info: {info}")
    
    print("\n" + "=" * 60)
    print("测试 QMIX")
    print("=" * 60)
    
    qmix = get_multi_agent_algorithm(
        'QMIX',
        diag_state_dim=12,
        diag_action_dim=20,
        ctrl_state_dim=10,
        ctrl_action_dim=45
    )
    
    diag_action, ctrl_action, info = qmix.select_actions(
        diag_state, diag_residual_seq, ctrl_state
    )
    print(f"诊断动作: {diag_action}, 控制动作: {ctrl_action}")
