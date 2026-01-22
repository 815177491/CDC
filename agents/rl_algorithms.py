#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多种强化学习算法实现
====================
包含近年顶会顶刊的新RL方法:

1. DQN (2015, Nature) - 基线方法
2. Dueling DQN (2016, ICML) - 价值分解
3. PPO (2017, OpenAI) - 策略梯度，工业控制常用
4. SAC (2018, ICML) - 最大熵RL，探索性好
5. TD3 (2018, ICML) - 双延迟DDPG
6. Decision Transformer (2021, NeurIPS) - 序列建模方法
7. IQL (2022, ICLR) - 隐式Q学习，离线RL新方法

Author: CDC Project
Date: 2026-01-21
References:
- DQN: Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015
- Dueling DQN: Wang et al., "Dueling Network Architectures for Deep RL", ICML 2016
- PPO: Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017
- SAC: Haarnoja et al., "Soft Actor-Critic", ICML 2018
- TD3: Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic", ICML 2018
- Decision Transformer: Chen et al., "Decision Transformer", NeurIPS 2021
- IQL: Kostrikov et al., "Offline RL with Implicit Q-Learning", ICLR 2022
"""

import numpy as np
from collections import deque
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

# 尝试导入深度学习库
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Normal, Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. RL algorithms will not work.")


# ============================================================
# 基础组件
# ============================================================

@dataclass
class Experience:
    """经验元组"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """优先级经验回放 (用于Rainbow DQN等)"""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
    
    def push(self, state, action, reward, next_state, done, priority=None):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if priority is None:
            priority = max_priority
        
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) == 0:
            return [], [], []
        
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[i] for i in indices]
        
        # 重要性采样权重
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-6) ** self.alpha
    
    def __len__(self):
        return len(self.buffer)


# ============================================================
# 基类
# ============================================================

class BaseRLAlgorithm(ABC):
    """RL算法基类 (支持GPU加速)"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        self.training_step = 0
        self.episode_rewards = []
        
        # GPU支持
        if TORCH_AVAILABLE:
            device_str = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            self.device = torch.device(device_str)
        else:
            self.device = None
    
    @abstractmethod
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """选择动作"""
        pass
    
    @abstractmethod
    def update(self, batch: List[Experience]) -> Dict[str, float]:
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
    
    def get_name(self) -> str:
        return self.__class__.__name__


if TORCH_AVAILABLE:
    # ============================================================
    # 1. DQN (基线)
    # ============================================================
    
    class DQNNetwork(nn.Module):
        """标准DQN网络"""
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
            super().__init__()
            
            layers = []
            prev_dim = state_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, action_dim))
            
            self.network = nn.Sequential(*layers)
            self._init_weights()
        
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            return self.network(x)
    
    
    class DQN(BaseRLAlgorithm):
        """
        Deep Q-Network (Mnih et al., Nature 2015)
        基线方法，使用经验回放和目标网络
        """
        
        def __init__(self, state_dim: int, action_dim: int, config: Dict = None):
            super().__init__(state_dim, action_dim, config)
            
            # 超参数
            self.lr = config.get('lr', 1e-3)
            self.gamma = config.get('gamma', 0.99)
            self.epsilon = config.get('epsilon', 1.0)
            self.epsilon_min = config.get('epsilon_min', 0.05)
            self.epsilon_decay = config.get('epsilon_decay', 0.995)
            self.target_update_freq = config.get('target_update_freq', 100)
            self.batch_size = config.get('batch_size', 64)
            
            # 网络
            self.q_network = DQNNetwork(state_dim, action_dim)
            self.target_network = DQNNetwork(state_dim, action_dim)
            self.target_network.load_state_dict(self.q_network.state_dict())
            
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
            self.buffer = ReplayBuffer(config.get('buffer_size', 100000))
        
        def select_action(self, state: np.ndarray, explore: bool = True) -> int:
            if explore and random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()
        
        def update(self, batch: List[Experience]) -> Dict[str, float]:
            if len(batch) < self.batch_size:
                return {}
            
            states = torch.FloatTensor(np.array([e.state for e in batch]))
            actions = torch.LongTensor([e.action for e in batch])
            rewards = torch.FloatTensor([e.reward for e in batch])
            next_states = torch.FloatTensor(np.array([e.next_state for e in batch]))
            dones = torch.FloatTensor([e.done for e in batch])
            
            # 当前Q值
            current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # 目标Q值
            with torch.no_grad():
                next_q = self.target_network(next_states).max(dim=1)[0]
                target_q = rewards + self.gamma * next_q * (1 - dones)
            
            # 损失
            loss = F.mse_loss(current_q.squeeze(), target_q)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()
            
            # 更新目标网络
            self.training_step += 1
            if self.training_step % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            # 衰减探索率
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            return {'loss': loss.item(), 'epsilon': self.epsilon, 'q_mean': current_q.mean().item()}
        
        def save(self, path: str):
            torch.save({
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'training_step': self.training_step
            }, path)
        
        def load(self, path: str):
            checkpoint = torch.load(path)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.training_step = checkpoint['training_step']
    
    
    # ============================================================
    # 2. Dueling DQN (ICML 2016)
    # ============================================================
    
    class DuelingDQNNetwork(nn.Module):
        """
        Dueling DQN网络 - 分离状态价值和优势函数
        Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        """
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
            super().__init__()
            
            # 共享特征层
            self.feature = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            
            # 状态价值流 V(s)
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            
            # 优势函数流 A(s,a)
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim)
            )
        
        def forward(self, x):
            features = self.feature(x)
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            # Q = V + A - mean(A)
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
            return q_values
    
    
    class DuelingDQN(DQN):
        """
        Dueling DQN (Wang et al., ICML 2016)
        通过分离V(s)和A(s,a)提高学习效率
        """
        
        def __init__(self, state_dim: int, action_dim: int, config: Dict = None):
            BaseRLAlgorithm.__init__(self, state_dim, action_dim, config)
            
            self.lr = config.get('lr', 1e-3)
            self.gamma = config.get('gamma', 0.99)
            self.epsilon = config.get('epsilon', 1.0)
            self.epsilon_min = config.get('epsilon_min', 0.05)
            self.epsilon_decay = config.get('epsilon_decay', 0.995)
            self.target_update_freq = config.get('target_update_freq', 100)
            self.batch_size = config.get('batch_size', 64)
            
            # 使用Dueling架构
            self.q_network = DuelingDQNNetwork(state_dim, action_dim)
            self.target_network = DuelingDQNNetwork(state_dim, action_dim)
            self.target_network.load_state_dict(self.q_network.state_dict())
            
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
            self.buffer = ReplayBuffer(config.get('buffer_size', 100000))
    
    
    # ============================================================
    # 3. PPO (OpenAI 2017) - 策略梯度方法
    # ============================================================
    
    class ActorCriticNetwork(nn.Module):
        """Actor-Critic共享网络"""
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
            super().__init__()
            
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            )
            
            self.actor = nn.Linear(hidden_dim, action_dim)
            self.critic = nn.Linear(hidden_dim, 1)
        
        def forward(self, x):
            features = self.shared(x)
            return features
        
        def get_action_probs(self, x):
            features = self.shared(x)
            logits = self.actor(features)
            return F.softmax(logits, dim=-1)
        
        def get_value(self, x):
            features = self.shared(x)
            return self.critic(features)
        
        def evaluate(self, states, actions):
            features = self.shared(states)
            logits = self.actor(features)
            values = self.critic(features)
            
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            
            action_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            return action_log_probs, values.squeeze(), entropy
    
    
    class PPO(BaseRLAlgorithm):
        """
        Proximal Policy Optimization (Schulman et al., 2017)
        使用clip目标函数，稳定性好，工业控制常用
        """
        
        def __init__(self, state_dim: int, action_dim: int, config: Dict = None):
            super().__init__(state_dim, action_dim, config)
            
            self.lr = config.get('lr', 3e-4)
            self.gamma = config.get('gamma', 0.99)
            self.gae_lambda = config.get('gae_lambda', 0.95)
            self.clip_epsilon = config.get('clip_epsilon', 0.2)
            self.entropy_coef = config.get('entropy_coef', 0.01)
            self.value_coef = config.get('value_coef', 0.5)
            self.max_grad_norm = config.get('max_grad_norm', 0.5)
            self.ppo_epochs = config.get('ppo_epochs', 10)
            self.batch_size = config.get('batch_size', 64)
            
            self.network = ActorCriticNetwork(state_dim, action_dim)
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
            
            # 轨迹缓存
            self.states = []
            self.actions = []
            self.rewards = []
            self.values = []
            self.log_probs = []
            self.dones = []
        
        def select_action(self, state: np.ndarray, explore: bool = True) -> int:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                probs = self.network.get_action_probs(state_tensor)
                
                if explore:
                    dist = Categorical(probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    value = self.network.get_value(state_tensor)
                    
                    self.log_probs.append(log_prob.item())
                    self.values.append(value.item())
                    
                    return action.item()
                else:
                    return probs.argmax(dim=1).item()
        
        def store_transition(self, state, action, reward, done):
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
        
        def compute_gae(self, next_value: float) -> Tuple[List, List]:
            """计算GAE优势估计"""
            advantages = []
            returns = []
            gae = 0
            
            values = self.values + [next_value]
            
            for t in reversed(range(len(self.rewards))):
                delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + values[t])
            
            return advantages, returns
        
        def update(self, batch: List[Experience] = None) -> Dict[str, float]:
            if len(self.states) == 0:
                return {}
            
            # 计算GAE
            with torch.no_grad():
                last_state = torch.FloatTensor(self.states[-1]).unsqueeze(0)
                next_value = self.network.get_value(last_state).item()
            
            advantages, returns = self.compute_gae(next_value)
            
            # 转换为张量
            states = torch.FloatTensor(np.array(self.states))
            actions = torch.LongTensor(self.actions)
            old_log_probs = torch.FloatTensor(self.log_probs)
            advantages = torch.FloatTensor(advantages)
            returns = torch.FloatTensor(returns)
            
            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            total_loss = 0
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            
            # 多轮PPO更新
            for _ in range(self.ppo_epochs):
                # 随机打乱
                indices = torch.randperm(len(states))
                
                for start in range(0, len(states), self.batch_size):
                    end = start + self.batch_size
                    batch_indices = indices[start:end]
                    
                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]
                    
                    # 评估当前策略
                    log_probs, values, entropy = self.network.evaluate(batch_states, batch_actions)
                    
                    # 策略损失 (PPO-Clip)
                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # 价值损失
                    value_loss = F.mse_loss(values, batch_returns)
                    
                    # 总损失
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()
            
            # 清空缓存
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.values.clear()
            self.log_probs.clear()
            self.dones.clear()
            
            self.training_step += 1
            
            n_updates = self.ppo_epochs * (len(states) // self.batch_size + 1)
            return {
                'loss': total_loss / n_updates,
                'policy_loss': total_policy_loss / n_updates,
                'value_loss': total_value_loss / n_updates,
                'entropy': total_entropy / n_updates
            }
        
        def save(self, path: str):
            torch.save({
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'training_step': self.training_step
            }, path)
        
        def load(self, path: str):
            checkpoint = torch.load(path)
            self.network.load_state_dict(checkpoint['network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.training_step = checkpoint['training_step']
    
    
    # ============================================================
    # 4. SAC (ICML 2018) - 最大熵强化学习
    # ============================================================
    
    class SACNetwork(nn.Module):
        """SAC网络 - 包含双Q网络和策略网络"""
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
            super().__init__()
            
            # 策略网络 (输出动作概率)
            self.policy = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            
            # 双Q网络
            self.q1 = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            
            self.q2 = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        
        def get_action(self, state, deterministic=False):
            logits = self.policy(state)
            probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                dist = Categorical(probs)
                action = dist.sample()
            
            # 计算log概率
            log_probs = F.log_softmax(logits, dim=-1)
            
            return action, probs, log_probs
        
        def get_q_values(self, state):
            return self.q1(state), self.q2(state)
    
    
    class SAC(BaseRLAlgorithm):
        """
        Soft Actor-Critic (Haarnoja et al., ICML 2018)
        最大熵框架，自动温度调节，探索性好
        适合连续控制任务 (GPU加速版本)
        """
        
        def __init__(self, state_dim: int, action_dim: int, config: Dict = None):
            super().__init__(state_dim, action_dim, config)
            
            self.lr = config.get('lr', 3e-4)
            self.gamma = config.get('gamma', 0.99)
            self.tau = config.get('tau', 0.005)  # 软更新系数
            self.batch_size = config.get('batch_size', 256)
            self.target_entropy = -np.log(1.0 / action_dim) * 0.98  # 目标熵
            
            # 网络 -> GPU
            self.network = SACNetwork(state_dim, action_dim).to(self.device)
            self.target_network = SACNetwork(state_dim, action_dim).to(self.device)
            self.target_network.load_state_dict(self.network.state_dict())
            
            # 自动温度参数 -> GPU
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            
            # 优化器
            self.policy_optimizer = optim.Adam(self.network.policy.parameters(), lr=self.lr)
            self.q_optimizer = optim.Adam(
                list(self.network.q1.parameters()) + list(self.network.q2.parameters()), 
                lr=self.lr
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
            
            self.buffer = ReplayBuffer(config.get('buffer_size', 100000))
        
        def select_action(self, state: np.ndarray, explore: bool = True) -> int:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, _, _ = self.network.get_action(state_tensor, deterministic=not explore)
                return action.item()
        
        def update(self, batch: List[Experience]) -> Dict[str, float]:
            if len(batch) < self.batch_size:
                return {}
            
            states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
            actions = torch.LongTensor([e.action for e in batch]).to(self.device)
            rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
            next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
            dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
            
            # 更新Q网络
            with torch.no_grad():
                next_action, next_probs, next_log_probs = self.network.get_action(next_states)
                next_q1, next_q2 = self.target_network.get_q_values(next_states)
                next_q = torch.min(next_q1, next_q2)
                # V(s') = E[Q(s',a') - alpha * log pi(a'|s')]
                next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=-1)
                target_q = rewards + self.gamma * (1 - dones) * next_v
            
            current_q1, current_q2 = self.network.get_q_values(states)
            current_q1 = current_q1.gather(1, actions.unsqueeze(1)).squeeze()
            current_q2 = current_q2.gather(1, actions.unsqueeze(1)).squeeze()
            
            q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()
            
            # 更新策略网络
            _, probs, log_probs = self.network.get_action(states)
            q1, q2 = self.network.get_q_values(states)
            min_q = torch.min(q1, q2)
            
            # 策略损失: E[alpha * log pi - Q]
            policy_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=-1).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # 更新温度参数
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            
            # 软更新目标网络
            for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            self.training_step += 1
            
            return {
                'q_loss': q_loss.item(),
                'policy_loss': policy_loss.item(),
                'alpha': self.alpha,
                'entropy': entropy.item()
            }
        
        def save(self, path: str):
            torch.save({
                'network': self.network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'log_alpha': self.log_alpha,
                'training_step': self.training_step
            }, path)
        
        def load(self, path: str):
            checkpoint = torch.load(path)
            self.network.load_state_dict(checkpoint['network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp().item()
            self.training_step = checkpoint['training_step']
    
    
    # ============================================================
    # 5. TD3 (ICML 2018) - 双延迟深度确定性策略梯度
    # ============================================================
    
    class TD3(BaseRLAlgorithm):
        """
        Twin Delayed DDPG (Fujimoto et al., ICML 2018)
        通过双Q网络、延迟策略更新、目标策略平滑来减少过估计
        虽然原始用于连续动作，这里适配离散动作
        """
        
        def __init__(self, state_dim: int, action_dim: int, config: Dict = None):
            super().__init__(state_dim, action_dim, config)
            
            self.lr = config.get('lr', 3e-4)
            self.gamma = config.get('gamma', 0.99)
            self.tau = config.get('tau', 0.005)
            self.policy_delay = config.get('policy_delay', 2)  # 延迟更新策略
            self.policy_noise = config.get('policy_noise', 0.2)  # 目标策略噪声
            self.noise_clip = config.get('noise_clip', 0.5)
            self.batch_size = config.get('batch_size', 256)
            self.epsilon = config.get('epsilon', 1.0)
            self.epsilon_min = config.get('epsilon_min', 0.05)
            self.epsilon_decay = config.get('epsilon_decay', 0.995)
            
            hidden_dim = config.get('hidden_dim', 256)
            
            # 策略网络
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            self.actor_target = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            self.actor_target.load_state_dict(self.actor.state_dict())
            
            # 双Q网络
            self.critic1 = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            self.critic2 = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            self.critic1_target = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            self.critic2_target = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            self.critic1_target.load_state_dict(self.critic1.state_dict())
            self.critic2_target.load_state_dict(self.critic2.state_dict())
            
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
            self.critic_optimizer = optim.Adam(
                list(self.critic1.parameters()) + list(self.critic2.parameters()), 
                lr=self.lr
            )
            
            self.buffer = ReplayBuffer(config.get('buffer_size', 100000))
        
        def select_action(self, state: np.ndarray, explore: bool = True) -> int:
            if explore and random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = F.softmax(self.actor(state_tensor), dim=-1)
                return action_probs.argmax(dim=1).item()
        
        def update(self, batch: List[Experience]) -> Dict[str, float]:
            if len(batch) < self.batch_size:
                return {}
            
            states = torch.FloatTensor(np.array([e.state for e in batch]))
            actions = torch.LongTensor([e.action for e in batch])
            rewards = torch.FloatTensor([e.reward for e in batch])
            next_states = torch.FloatTensor(np.array([e.next_state for e in batch]))
            dones = torch.FloatTensor([e.done for e in batch])
            
            # 计算目标Q值
            with torch.no_grad():
                # 目标策略平滑 (加噪声)
                target_probs = F.softmax(self.actor_target(next_states), dim=-1)
                noise = torch.clamp(
                    torch.randn_like(target_probs) * self.policy_noise,
                    -self.noise_clip, self.noise_clip
                )
                target_probs = F.softmax(target_probs + noise, dim=-1)
                
                # 双Q取最小
                target_q1 = self.critic1_target(next_states)
                target_q2 = self.critic2_target(next_states)
                target_q = torch.min(target_q1, target_q2)
                target_v = (target_probs * target_q).sum(dim=-1)
                target_q_value = rewards + self.gamma * (1 - dones) * target_v
            
            # 当前Q值
            current_q1 = self.critic1(states).gather(1, actions.unsqueeze(1)).squeeze()
            current_q2 = self.critic2(states).gather(1, actions.unsqueeze(1)).squeeze()
            
            # Critic损失
            critic_loss = F.mse_loss(current_q1, target_q_value) + F.mse_loss(current_q2, target_q_value)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            self.training_step += 1
            actor_loss = torch.tensor(0.0)
            
            # 延迟更新策略
            if self.training_step % self.policy_delay == 0:
                action_probs = F.softmax(self.actor(states), dim=-1)
                q_values = self.critic1(states)
                actor_loss = -(action_probs * q_values).sum(dim=-1).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # 软更新目标网络
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            # 衰减探索率
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            return {
                'critic_loss': critic_loss.item(),
                'actor_loss': actor_loss.item(),
                'epsilon': self.epsilon
            }
        
        def save(self, path: str):
            torch.save({
                'actor': self.actor.state_dict(),
                'critic1': self.critic1.state_dict(),
                'critic2': self.critic2.state_dict(),
                'training_step': self.training_step,
                'epsilon': self.epsilon
            }, path)
        
        def load(self, path: str):
            checkpoint = torch.load(path)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic1.load_state_dict(checkpoint['critic1'])
            self.critic2.load_state_dict(checkpoint['critic2'])
            self.training_step = checkpoint['training_step']
            self.epsilon = checkpoint['epsilon']
    
    
    # ============================================================
    # 6. Decision Transformer (NeurIPS 2021)
    # ============================================================
    
    class DecisionTransformer(BaseRLAlgorithm):
        """
        Decision Transformer (Chen et al., NeurIPS 2021)
        将RL问题转化为序列建模问题，使用Transformer架构
        离线RL方法，从历史数据学习
        """
        
        def __init__(self, state_dim: int, action_dim: int, config: Dict = None):
            super().__init__(state_dim, action_dim, config)
            
            self.context_length = config.get('context_length', 20)
            self.n_heads = config.get('n_heads', 4)
            self.n_layers = config.get('n_layers', 3)
            self.embed_dim = config.get('embed_dim', 128)
            self.lr = config.get('lr', 1e-4)
            self.batch_size = config.get('batch_size', 64)
            
            # 嵌入层
            self.state_embed = nn.Linear(state_dim, self.embed_dim)
            self.action_embed = nn.Embedding(action_dim, self.embed_dim)
            self.return_embed = nn.Linear(1, self.embed_dim)
            self.timestep_embed = nn.Embedding(1000, self.embed_dim)
            
            # 位置编码
            self.pos_embed = nn.Parameter(torch.zeros(1, 3 * self.context_length, self.embed_dim))
            
            # Transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim, 
                nhead=self.n_heads,
                dim_feedforward=4 * self.embed_dim,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
            
            # 预测头
            self.action_head = nn.Linear(self.embed_dim, action_dim)
            
            # LayerNorm
            self.ln = nn.LayerNorm(self.embed_dim)
            
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
            
            # 历史序列
            self.state_history = deque(maxlen=self.context_length)
            self.action_history = deque(maxlen=self.context_length)
            self.return_history = deque(maxlen=self.context_length)
            self.timestep_history = deque(maxlen=self.context_length)
            self.target_return = config.get('target_return', 100)  # 期望回报
            
            # 离线数据
            self.trajectories = []
        
        def parameters(self):
            return list(self.state_embed.parameters()) + \
                   list(self.action_embed.parameters()) + \
                   list(self.return_embed.parameters()) + \
                   list(self.timestep_embed.parameters()) + \
                   [self.pos_embed] + \
                   list(self.transformer.parameters()) + \
                   list(self.action_head.parameters()) + \
                   list(self.ln.parameters())
        
        def _get_embedding(self, states, actions, returns, timesteps):
            batch_size, seq_len = states.shape[0], states.shape[1]
            
            state_embeds = self.state_embed(states)
            action_embeds = self.action_embed(actions)
            return_embeds = self.return_embed(returns.unsqueeze(-1))
            time_embeds = self.timestep_embed(timesteps)
            
            # 交错排列: [R1, s1, a1, R2, s2, a2, ...]
            stacked = torch.stack([return_embeds, state_embeds, action_embeds], dim=2)
            stacked = stacked.reshape(batch_size, 3 * seq_len, self.embed_dim)
            
            # 加时间嵌入和位置编码
            stacked = stacked + self.pos_embed[:, :3*seq_len, :]
            
            return stacked
        
        def select_action(self, state: np.ndarray, explore: bool = True) -> int:
            # 维护历史
            self.state_history.append(state)
            
            if len(self.state_history) < 1:
                return random.randint(0, self.action_dim - 1)
            
            with torch.no_grad():
                # 构建序列
                states = torch.FloatTensor(np.array(list(self.state_history))).unsqueeze(0)
                
                # 填充动作历史
                if len(self.action_history) < len(self.state_history):
                    self.action_history.append(0)
                actions = torch.LongTensor(list(self.action_history)).unsqueeze(0)
                
                # 返回值 (从目标回报开始递减)
                n = len(self.return_history)
                if n == 0:
                    self.return_history.append(self.target_return)
                returns = torch.FloatTensor(list(self.return_history)).unsqueeze(0)
                
                # 时间步
                timesteps = torch.arange(len(self.state_history)).unsqueeze(0)
                
                # 前向传播
                embeds = self._get_embedding(states, actions, returns, timesteps)
                embeds = self.ln(embeds)
                hidden = self.transformer(embeds)
                
                # 取最后一个状态位置的输出预测动作
                # 状态在位置 1, 4, 7, ... (3n+1)
                state_positions = torch.arange(0, hidden.shape[1], 3) + 1
                state_hidden = hidden[:, state_positions[-1], :]
                
                logits = self.action_head(state_hidden)
                
                if explore:
                    probs = F.softmax(logits / 0.5, dim=-1)  # 温度采样
                    action = Categorical(probs).sample().item()
                else:
                    action = logits.argmax(dim=-1).item()
                
                self.action_history.append(action)
                
                return action
        
        def store_trajectory(self, states, actions, rewards):
            """存储完整轨迹用于离线训练"""
            # 计算return-to-go
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + 0.99 * R
                returns.insert(0, R)
            
            self.trajectories.append({
                'states': np.array(states),
                'actions': np.array(actions),
                'returns': np.array(returns),
                'timesteps': np.arange(len(states))
            })
        
        def update(self, batch: List[Experience] = None) -> Dict[str, float]:
            """离线训练"""
            if len(self.trajectories) < 5:
                return {}
            
            total_loss = 0
            n_updates = 0
            
            for _ in range(10):  # 每次训练多轮
                # 随机采样轨迹片段
                traj = random.choice(self.trajectories)
                traj_len = len(traj['states'])
                
                if traj_len < self.context_length:
                    continue
                
                # 随机起始点
                start = random.randint(0, traj_len - self.context_length)
                end = start + self.context_length
                
                states = torch.FloatTensor(traj['states'][start:end]).unsqueeze(0)
                actions = torch.LongTensor(traj['actions'][start:end]).unsqueeze(0)
                returns = torch.FloatTensor(traj['returns'][start:end]).unsqueeze(0)
                timesteps = torch.LongTensor(traj['timesteps'][start:end]).unsqueeze(0)
                
                # 前向传播
                embeds = self._get_embedding(states, actions, returns, timesteps)
                embeds = self.ln(embeds)
                hidden = self.transformer(embeds)
                
                # 在每个状态位置预测动作
                state_positions = torch.arange(1, hidden.shape[1], 3)
                state_hidden = hidden[:, state_positions, :]
                
                logits = self.action_head(state_hidden)
                
                # 交叉熵损失
                loss = F.cross_entropy(logits.reshape(-1, self.action_dim), actions.reshape(-1))
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                n_updates += 1
            
            self.training_step += 1
            
            return {'loss': total_loss / max(n_updates, 1)}
        
        def reset_history(self):
            """重置历史（新episode开始时）"""
            self.state_history.clear()
            self.action_history.clear()
            self.return_history.clear()
            self.timestep_history.clear()
        
        def save(self, path: str):
            torch.save({
                'state_embed': self.state_embed.state_dict(),
                'action_embed': self.action_embed.state_dict(),
                'return_embed': self.return_embed.state_dict(),
                'timestep_embed': self.timestep_embed.state_dict(),
                'pos_embed': self.pos_embed,
                'transformer': self.transformer.state_dict(),
                'action_head': self.action_head.state_dict(),
                'ln': self.ln.state_dict(),
                'training_step': self.training_step
            }, path)
        
        def load(self, path: str):
            checkpoint = torch.load(path)
            self.state_embed.load_state_dict(checkpoint['state_embed'])
            self.action_embed.load_state_dict(checkpoint['action_embed'])
            self.return_embed.load_state_dict(checkpoint['return_embed'])
            self.timestep_embed.load_state_dict(checkpoint['timestep_embed'])
            self.pos_embed = checkpoint['pos_embed']
            self.transformer.load_state_dict(checkpoint['transformer'])
            self.action_head.load_state_dict(checkpoint['action_head'])
            self.ln.load_state_dict(checkpoint['ln'])
            self.training_step = checkpoint['training_step']
    
    
    # ============================================================
    # 7. IQL - Implicit Q-Learning (ICLR 2022)
    # ============================================================
    
    class IQL(BaseRLAlgorithm):
        """
        Implicit Q-Learning (Kostrikov et al., ICLR 2022)
        离线RL新方法，通过期望分位数回归避免OOD动作
        不需要显式约束策略，只用数据中的动作
        """
        
        def __init__(self, state_dim: int, action_dim: int, config: Dict = None):
            super().__init__(state_dim, action_dim, config)
            
            self.lr = config.get('lr', 3e-4)
            self.gamma = config.get('gamma', 0.99)
            self.tau = config.get('tau', 0.005)
            self.beta = config.get('beta', 3.0)  # AWR温度
            self.expectile = config.get('expectile', 0.7)  # 期望分位数
            self.batch_size = config.get('batch_size', 256)
            self.epsilon = config.get('epsilon', 0.1)
            
            hidden_dim = config.get('hidden_dim', 256)
            
            # Q网络 (双Q)
            self.q1 = nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            self.q2 = nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            self.q1_target = nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            self.q2_target = nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q2.state_dict())
            
            # V网络
            self.v = nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            
            # 策略网络
            self.policy = nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            
            self.q_optimizer = optim.Adam(
                list(self.q1.parameters()) + list(self.q2.parameters()), 
                lr=self.lr
            )
            self.v_optimizer = optim.Adam(self.v.parameters(), lr=self.lr)
            self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
            
            self.buffer = ReplayBuffer(config.get('buffer_size', 100000))
        
        def expectile_loss(self, diff, expectile):
            """期望分位数损失"""
            weight = torch.where(diff > 0, expectile, 1 - expectile)
            return (weight * (diff ** 2)).mean()
        
        def select_action(self, state: np.ndarray, explore: bool = True) -> int:
            if explore and random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                logits = self.policy(state_tensor)
                return logits.argmax(dim=1).item()
        
        def update(self, batch: List[Experience]) -> Dict[str, float]:
            if len(batch) < self.batch_size:
                return {}
            
            states = torch.FloatTensor(np.array([e.state for e in batch]))
            actions = torch.LongTensor([e.action for e in batch])
            rewards = torch.FloatTensor([e.reward for e in batch])
            next_states = torch.FloatTensor(np.array([e.next_state for e in batch]))
            dones = torch.FloatTensor([e.done for e in batch])
            
            # 更新V网络 (使用期望分位数回归)
            with torch.no_grad():
                q1 = self.q1_target(states).gather(1, actions.unsqueeze(1)).squeeze()
                q2 = self.q2_target(states).gather(1, actions.unsqueeze(1)).squeeze()
                target_q = torch.min(q1, q2)
            
            v = self.v(states).squeeze()
            v_loss = self.expectile_loss(target_q - v, self.expectile)
            
            self.v_optimizer.zero_grad()
            v_loss.backward()
            self.v_optimizer.step()
            
            # 更新Q网络
            with torch.no_grad():
                next_v = self.v(next_states).squeeze()
                target_q_value = rewards + self.gamma * (1 - dones) * next_v
            
            current_q1 = self.q1(states).gather(1, actions.unsqueeze(1)).squeeze()
            current_q2 = self.q2(states).gather(1, actions.unsqueeze(1)).squeeze()
            q_loss = F.mse_loss(current_q1, target_q_value) + F.mse_loss(current_q2, target_q_value)
            
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()
            
            # 更新策略 (AWR风格)
            with torch.no_grad():
                q1 = self.q1(states).gather(1, actions.unsqueeze(1)).squeeze()
                q2 = self.q2(states).gather(1, actions.unsqueeze(1)).squeeze()
                q = torch.min(q1, q2)
                v = self.v(states).squeeze()
                advantage = q - v
                weight = torch.exp(self.beta * advantage)
                weight = torch.clamp(weight, max=100.0)
            
            logits = self.policy(states)
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
            policy_loss = -(weight * action_log_probs).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # 软更新目标网络
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            self.training_step += 1
            
            return {
                'v_loss': v_loss.item(),
                'q_loss': q_loss.item(),
                'policy_loss': policy_loss.item(),
                'advantage_mean': advantage.mean().item()
            }
        
        def save(self, path: str):
            torch.save({
                'q1': self.q1.state_dict(),
                'q2': self.q2.state_dict(),
                'v': self.v.state_dict(),
                'policy': self.policy.state_dict(),
                'training_step': self.training_step
            }, path)
        
        def load(self, path: str):
            checkpoint = torch.load(path)
            self.q1.load_state_dict(checkpoint['q1'])
            self.q2.load_state_dict(checkpoint['q2'])
            self.v.load_state_dict(checkpoint['v'])
            self.policy.load_state_dict(checkpoint['policy'])
            self.training_step = checkpoint['training_step']


# ============================================================
# 算法工厂
# ============================================================

def get_algorithm(name: str, state_dim: int, action_dim: int, config: Dict = None):
    """
    根据名称获取RL算法实例
    
    Args:
        name: 算法名称 ('DQN', 'DuelingDQN', 'PPO', 'SAC', 'TD3', 'DecisionTransformer', 'IQL')
        state_dim: 状态维度
        action_dim: 动作维度
        config: 算法配置
    
    Returns:
        RL算法实例
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for RL algorithms")
    
    algorithms = {
        'DQN': DQN,
        'DuelingDQN': DuelingDQN,
        'PPO': PPO,
        'SAC': SAC,
        'TD3': TD3,
        'DecisionTransformer': DecisionTransformer,
        'DT': DecisionTransformer,
        'IQL': IQL,
    }
    
    if name not in algorithms:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(algorithms.keys())}")
    
    return algorithms[name](state_dim, action_dim, config or {})


def list_algorithms() -> List[str]:
    """返回可用的算法列表"""
    return ['DQN', 'DuelingDQN', 'PPO', 'SAC', 'TD3', 'DecisionTransformer', 'IQL']


# 算法信息
ALGORITHM_INFO = {
    'DQN': {
        'name': 'Deep Q-Network',
        'year': 2015,
        'venue': 'Nature',
        'authors': 'Mnih et al.',
        'type': 'Value-based',
        'description': '基线方法，使用经验回放和目标网络'
    },
    'DuelingDQN': {
        'name': 'Dueling DQN',
        'year': 2016,
        'venue': 'ICML',
        'authors': 'Wang et al.',
        'type': 'Value-based',
        'description': '分离状态价值和优势函数，提高学习效率'
    },
    'PPO': {
        'name': 'Proximal Policy Optimization',
        'year': 2017,
        'venue': 'OpenAI',
        'authors': 'Schulman et al.',
        'type': 'Policy Gradient',
        'description': 'Clip目标函数，稳定性好，工业控制常用'
    },
    'SAC': {
        'name': 'Soft Actor-Critic',
        'year': 2018,
        'venue': 'ICML',
        'authors': 'Haarnoja et al.',
        'type': 'Actor-Critic',
        'description': '最大熵框架，自动温度调节，探索性好'
    },
    'TD3': {
        'name': 'Twin Delayed DDPG',
        'year': 2018,
        'venue': 'ICML',
        'authors': 'Fujimoto et al.',
        'type': 'Actor-Critic',
        'description': '双Q网络+延迟更新+目标平滑，减少过估计'
    },
    'DecisionTransformer': {
        'name': 'Decision Transformer',
        'year': 2021,
        'venue': 'NeurIPS',
        'authors': 'Chen et al.',
        'type': 'Sequence Modeling',
        'description': '将RL转化为序列建模，使用Transformer架构'
    },
    'IQL': {
        'name': 'Implicit Q-Learning',
        'year': 2022,
        'venue': 'ICLR',
        'authors': 'Kostrikov et al.',
        'type': 'Offline RL',
        'description': '期望分位数回归，无需显式策略约束'
    }
}
