#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于SAC的强化学习诊断智能体
============================
使用Soft Actor-Critic算法进行故障诊断决策

特点:
1. 状态空间: 残差序列 + 历史特征 + 系统状态
2. 动作空间: 5种故障类型 × 4个置信度级别 = 20个离散动作
3. 奖励函数: 诊断准确率 + 检测延迟惩罚 + 下游控制效果反馈

算法: SAC (Haarnoja et al., ICML 2018)
- 最大熵框架，自动温度调节
- 探索性好，适合不确定性较高的诊断任务
- 离散动作版本

Author: CDC Project
Date: 2026-01-24
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
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
    warnings.warn("PyTorch not available. RL Diagnosis Agent will not work.")

import sys
sys.path.append('..')
from diagnosis.fault_injector import FaultType


# ============================================================
# 诊断动作定义
# ============================================================

class DiagnosisConfidence(Enum):
    """诊断置信度级别"""
    LOW = 0       # 低置信度 (0.25)
    MEDIUM = 1    # 中置信度 (0.50)
    HIGH = 2      # 高置信度 (0.75)
    CERTAIN = 3   # 确定 (1.00)


@dataclass
class DiagnosisAction:
    """诊断动作"""
    fault_type: FaultType
    confidence_level: DiagnosisConfidence
    action_index: int
    
    @property
    def confidence_value(self) -> float:
        """获取置信度数值"""
        return (self.confidence_level.value + 1) * 0.25


@dataclass
class RLDiagnosisResult:
    """RL诊断结果"""
    timestamp: float
    fault_detected: bool
    predicted_fault_type: FaultType
    confidence: float
    action_index: int
    residuals: Dict[str, float] = field(default_factory=dict)
    diagnosis_state: str = "HEALTHY"
    is_correct: Optional[bool] = None  # 与ground truth比较的结果


# ============================================================
# 诊断SAC网络
# ============================================================

if TORCH_AVAILABLE:
    
    class DiagnosisSACNetwork(nn.Module):
        """
        诊断SAC网络
        
        包含:
        - 特征提取器: 处理残差序列
        - 策略网络: 输出诊断动作概率
        - 双Q网络: 评估动作价值
        """
        
        def __init__(self, state_dim: int, action_dim: int, 
                     hidden_dim: int = 256, sequence_len: int = 10):
            super().__init__()
            
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.sequence_len = sequence_len
            
            # 残差序列特征提取 (使用1D卷积)
            self.residual_encoder = nn.Sequential(
                nn.Conv1d(3, 32, kernel_size=3, padding=1),  # 3个残差通道
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),  # 全局平均池化
                nn.Flatten()
            )
            
            # 状态特征融合
            # 64 (残差特征) + state_dim (当前状态)
            fusion_dim = 64 + state_dim
            
            # 策略网络
            self.policy = nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, action_dim)
            )
            
            # 双Q网络
            self.q1 = nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            
            self.q2 = nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            
            self._init_weights()
        
        def _init_weights(self):
            """初始化网络权重"""
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        def encode_features(self, state: torch.Tensor, 
                           residual_sequence: torch.Tensor) -> torch.Tensor:
            """
            编码特征
            
            Args:
                state: 当前状态 [batch, state_dim]
                residual_sequence: 残差序列 [batch, 3, seq_len]
            
            Returns:
                融合特征 [batch, fusion_dim]
            """
            # 残差序列特征
            residual_feat = self.residual_encoder(residual_sequence)
            
            # 融合
            return torch.cat([residual_feat, state], dim=-1)
        
        def get_action(self, state: torch.Tensor, 
                      residual_sequence: torch.Tensor,
                      deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            获取动作
            
            Returns:
                action: 动作索引
                probs: 动作概率
                log_probs: 对数概率
            """
            features = self.encode_features(state, residual_sequence)
            logits = self.policy(features)
            probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                dist = Categorical(probs)
                action = dist.sample()
            
            log_probs = F.log_softmax(logits, dim=-1)
            
            return action, probs, log_probs
        
        def get_q_values(self, state: torch.Tensor,
                        residual_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """获取Q值"""
            features = self.encode_features(state, residual_sequence)
            return self.q1(features), self.q2(features)


# ============================================================
# RL诊断智能体
# ============================================================

class RLDiagnosisAgent:
    """
    基于SAC的强化学习诊断智能体
    
    状态空间 (12维 + 残差序列):
    - Pmax, Pcomp, Texh (归一化测量值)
    - r_Pmax, r_Pcomp, r_Texh (当前残差)
    - Pmax变化率, Texh变化率
    - 当前VIT, 当前燃油系数
    - 时间步归一化
    - 上一次诊断动作编码
    - 残差序列 [3, sequence_len] (历史残差)
    
    动作空间 (20维离散):
    - 5种故障类型: NONE, INJECTION_TIMING, CYLINDER_LEAK, FUEL_DEGRADATION, INJECTOR_DRIFT
    - 4种置信度: LOW, MEDIUM, HIGH, CERTAIN
    - 组合: 5 × 4 = 20
    
    奖励函数:
    - 诊断准确率奖励: +1 正确, -1 错误
    - 检测延迟惩罚: -0.1 × 延迟步数
    - 置信度校准: 正确时高置信度加分, 错误时高置信度减分
    - 下游控制效果: 正确诊断后控制改善给予奖励
    """
    
    # 故障类型列表 (与FaultType对应)
    FAULT_TYPES = [
        FaultType.NONE,
        FaultType.INJECTION_TIMING,
        FaultType.CYLINDER_LEAK,
        FaultType.FUEL_DEGRADATION,
        FaultType.INJECTOR_DRIFT
    ]
    
    # 置信度级别
    CONFIDENCE_LEVELS = [
        DiagnosisConfidence.LOW,
        DiagnosisConfidence.MEDIUM,
        DiagnosisConfidence.HIGH,
        DiagnosisConfidence.CERTAIN
    ]
    
    def __init__(self, config: Dict = None):
        """
        初始化RL诊断智能体
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 维度定义
        self.state_dim = self.config.get('state_dim', 12)
        self.sequence_len = self.config.get('sequence_len', 10)
        self.n_fault_types = len(self.FAULT_TYPES)
        self.n_confidence_levels = len(self.CONFIDENCE_LEVELS)
        self.action_dim = self.n_fault_types * self.n_confidence_levels  # 20
        
        # 设备
        if TORCH_AVAILABLE:
            device_str = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            self.device = torch.device(device_str)
        else:
            self.device = None
        
        # SAC超参数
        self.lr = self.config.get('lr', 3e-4)
        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.005)
        self.batch_size = self.config.get('batch_size', 128)
        self.buffer_size = self.config.get('buffer_size', 100000)
        
        # 奖励权重
        self.accuracy_weight = self.config.get('accuracy_weight', 1.0)
        self.delay_penalty = self.config.get('delay_penalty', 0.1)
        self.confidence_weight = self.config.get('confidence_weight', 0.2)
        self.control_feedback_weight = self.config.get('control_feedback_weight', 0.3)
        
        # 历史记录
        self.residual_history = deque(maxlen=self.sequence_len)
        self.last_action = 0
        self.last_diagnosis_result = None
        self.fault_onset_step = None  # 故障开始的步数
        self.current_step = 0
        self.first_correct_detection_step = None
        
        # 前一步的观测值 (用于计算变化率)
        self.prev_observation = None
        
        # 初始化网络
        if TORCH_AVAILABLE:
            self._init_networks()
        
        # 经验回放
        self.replay_buffer = []
        
        # 训练统计
        self.training_step = 0
        self.episode_rewards = []
        self.detection_delays = []
        self.accuracy_history = []
        
        print(f"[RLDiagnosisAgent] 初始化完成")
        print(f"  - 状态维度: {self.state_dim}")
        print(f"  - 动作维度: {self.action_dim} ({self.n_fault_types}故障 × {self.n_confidence_levels}置信度)")
        print(f"  - 序列长度: {self.sequence_len}")
        print(f"  - 设备: {self.device}")
    
    def _init_networks(self):
        """初始化SAC网络"""
        # 主网络
        self.network = DiagnosisSACNetwork(
            self.state_dim, self.action_dim,
            hidden_dim=self.config.get('hidden_dim', 256),
            sequence_len=self.sequence_len
        ).to(self.device)
        
        # 目标网络
        self.target_network = DiagnosisSACNetwork(
            self.state_dim, self.action_dim,
            hidden_dim=self.config.get('hidden_dim', 256),
            sequence_len=self.sequence_len
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # 自动温度参数
        self.target_entropy = -np.log(1.0 / self.action_dim) * 0.98
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        
        # 优化器
        self.policy_optimizer = optim.Adam(
            self.network.policy.parameters(), lr=self.lr
        )
        self.q_optimizer = optim.Adam(
            list(self.network.q1.parameters()) + 
            list(self.network.q2.parameters()) +
            list(self.network.residual_encoder.parameters()),
            lr=self.lr
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
    
    def reset(self):
        """重置智能体状态"""
        self.residual_history.clear()
        self.last_action = 0
        self.last_diagnosis_result = None
        self.fault_onset_step = None
        self.current_step = 0
        self.first_correct_detection_step = None
        self.prev_observation = None
        
        # 初始化残差历史为零
        for _ in range(self.sequence_len):
            self.residual_history.append([0.0, 0.0, 0.0])
    
    def encode_state(self, observation: Dict[str, Any],
                    residuals: Dict[str, float],
                    current_vit: float = 0.0,
                    current_fuel: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        编码状态
        
        Args:
            observation: 测量值字典
            residuals: 残差字典
            current_vit: 当前VIT值
            current_fuel: 当前燃油系数
        
        Returns:
            state: 状态向量 [state_dim]
            residual_seq: 残差序列 [3, sequence_len]
        """
        # 提取并归一化测量值
        Pmax = observation.get('Pmax', 170) / 200.0
        Pcomp = observation.get('Pcomp', 150) / 200.0
        Texh = observation.get('Texh', 350) / 500.0
        
        # 当前残差
        r_Pmax = residuals.get('Pmax', 0)
        r_Pcomp = residuals.get('Pcomp', 0)
        r_Texh = residuals.get('Texh', 0)
        
        # 更新残差历史
        self.residual_history.append([r_Pmax, r_Pcomp, r_Texh])
        
        # 计算变化率
        if self.prev_observation is not None:
            dPmax = (observation.get('Pmax', 170) - self.prev_observation.get('Pmax', 170)) / 200.0
            dTexh = (observation.get('Texh', 350) - self.prev_observation.get('Texh', 350)) / 500.0
        else:
            dPmax = 0.0
            dTexh = 0.0
        self.prev_observation = observation.copy()
        
        # 归一化VIT和燃油
        vit_norm = (current_vit + 8) / 12.0  # [-8, 4] -> [0, 1]
        fuel_norm = (current_fuel - 0.7) / 0.3  # [0.7, 1.0] -> [0, 1]
        
        # 时间步归一化
        time_norm = min(self.current_step / 200.0, 1.0)
        
        # 上一次诊断动作编码
        last_action_norm = self.last_action / self.action_dim
        
        # 构建状态向量
        state = np.array([
            Pmax, Pcomp, Texh,
            r_Pmax, r_Pcomp, r_Texh,
            dPmax, dTexh,
            vit_norm, fuel_norm,
            time_norm, last_action_norm
        ], dtype=np.float32)
        
        # 构建残差序列 [3, sequence_len]
        residual_seq = np.array(list(self.residual_history), dtype=np.float32).T
        
        return state, residual_seq
    
    def decode_action(self, action_idx: int) -> DiagnosisAction:
        """
        解码动作索引
        
        Args:
            action_idx: 动作索引 [0, 19]
        
        Returns:
            DiagnosisAction: 诊断动作
        """
        fault_idx = action_idx // self.n_confidence_levels
        conf_idx = action_idx % self.n_confidence_levels
        
        return DiagnosisAction(
            fault_type=self.FAULT_TYPES[fault_idx],
            confidence_level=self.CONFIDENCE_LEVELS[conf_idx],
            action_index=action_idx
        )
    
    def encode_action(self, fault_type: FaultType, 
                     confidence: DiagnosisConfidence) -> int:
        """
        编码动作
        
        Args:
            fault_type: 故障类型
            confidence: 置信度级别
        
        Returns:
            action_idx: 动作索引
        """
        fault_idx = self.FAULT_TYPES.index(fault_type)
        conf_idx = confidence.value
        return fault_idx * self.n_confidence_levels + conf_idx
    
    def select_action(self, state: np.ndarray, 
                     residual_seq: np.ndarray,
                     explore: bool = True) -> int:
        """
        选择动作
        
        Args:
            state: 状态向量
            residual_seq: 残差序列
            explore: 是否探索
        
        Returns:
            action_idx: 动作索引
        """
        if not TORCH_AVAILABLE:
            return 0
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            seq_t = torch.FloatTensor(residual_seq).unsqueeze(0).to(self.device)
            
            action, _, _ = self.network.get_action(
                state_t, seq_t, deterministic=not explore
            )
            return action.item()
    
    def diagnose(self, observation: Dict[str, Any],
                residuals: Dict[str, float],
                current_vit: float = 0.0,
                current_fuel: float = 1.0,
                ground_truth: Optional[FaultType] = None,
                explore: bool = True) -> RLDiagnosisResult:
        """
        执行诊断
        
        Args:
            observation: 测量值
            residuals: 残差
            current_vit: 当前VIT
            current_fuel: 当前燃油
            ground_truth: 真实故障类型 (用于奖励计算)
            explore: 是否探索
        
        Returns:
            RLDiagnosisResult: 诊断结果
        """
        self.current_step += 1
        
        # 编码状态
        state, residual_seq = self.encode_state(
            observation, residuals, current_vit, current_fuel
        )
        
        # 选择动作
        action_idx = self.select_action(state, residual_seq, explore)
        diagnosis_action = self.decode_action(action_idx)
        
        # 更新记录
        self.last_action = action_idx
        
        # 判断是否检测到故障
        fault_detected = diagnosis_action.fault_type != FaultType.NONE
        
        # 确定诊断状态
        if not fault_detected:
            diagnosis_state = "HEALTHY"
        else:
            conf = diagnosis_action.confidence_value
            if conf <= 0.25:
                diagnosis_state = "WARNING"
            elif conf <= 0.75:
                diagnosis_state = "FAULT"
            else:
                diagnosis_state = "CRITICAL"
        
        # 检查诊断是否正确
        is_correct = None
        if ground_truth is not None:
            is_correct = (diagnosis_action.fault_type == ground_truth)
            
            # 记录首次正确检测
            if is_correct and ground_truth != FaultType.NONE:
                if self.first_correct_detection_step is None:
                    self.first_correct_detection_step = self.current_step
        
        result = RLDiagnosisResult(
            timestamp=self.current_step,
            fault_detected=fault_detected,
            predicted_fault_type=diagnosis_action.fault_type,
            confidence=diagnosis_action.confidence_value,
            action_index=action_idx,
            residuals=residuals.copy(),
            diagnosis_state=diagnosis_state,
            is_correct=is_correct
        )
        
        self.last_diagnosis_result = result
        return result
    
    def compute_reward(self, 
                      ground_truth: FaultType,
                      predicted: FaultType,
                      confidence: float,
                      control_improvement: float = 0.0,
                      fault_onset_step: Optional[int] = None) -> float:
        """
        计算奖励
        
        Args:
            ground_truth: 真实故障类型
            predicted: 预测故障类型
            confidence: 置信度
            control_improvement: 下游控制改善量 (正值表示改善)
            fault_onset_step: 故障开始步数
        
        Returns:
            reward: 总奖励
        """
        reward = 0.0
        
        # 1. 诊断准确率奖励
        is_correct = (predicted == ground_truth)
        if is_correct:
            reward += self.accuracy_weight * 1.0
        else:
            reward -= self.accuracy_weight * 1.0
        
        # 2. 置信度校准
        if is_correct:
            # 正确诊断时，高置信度加分
            reward += self.confidence_weight * confidence
        else:
            # 错误诊断时，高置信度减分
            reward -= self.confidence_weight * confidence * 2.0
        
        # 3. 检测延迟惩罚
        if ground_truth != FaultType.NONE and fault_onset_step is not None:
            delay = self.current_step - fault_onset_step
            if is_correct and self.first_correct_detection_step is not None:
                # 首次正确检测的延迟
                detection_delay = self.first_correct_detection_step - fault_onset_step
                reward -= self.delay_penalty * detection_delay
        
        # 4. 下游控制效果反馈
        if is_correct and control_improvement > 0:
            reward += self.control_feedback_weight * control_improvement
        
        # 5. 误报惩罚
        if ground_truth == FaultType.NONE and predicted != FaultType.NONE:
            reward -= 0.5  # 误报惩罚
        
        # 6. 漏报惩罚
        if ground_truth != FaultType.NONE and predicted == FaultType.NONE:
            reward -= 0.8  # 漏报惩罚更重
        
        return reward
    
    def store_experience(self, state: np.ndarray, residual_seq: np.ndarray,
                        action: int, reward: float,
                        next_state: np.ndarray, next_residual_seq: np.ndarray,
                        done: bool):
        """存储经验"""
        experience = {
            'state': state,
            'residual_seq': residual_seq,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'next_residual_seq': next_residual_seq,
            'done': done
        }
        
        self.replay_buffer.append(experience)
        
        # 限制缓冲区大小
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
    
    def update(self) -> Dict[str, float]:
        """
        执行SAC更新
        
        Returns:
            训练指标字典
        """
        if not TORCH_AVAILABLE or len(self.replay_buffer) < self.batch_size:
            return {}
        
        # 采样batch
        import random
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # 转换为tensor
        states = torch.FloatTensor(np.array([e['state'] for e in batch])).to(self.device)
        residual_seqs = torch.FloatTensor(np.array([e['residual_seq'] for e in batch])).to(self.device)
        actions = torch.LongTensor([e['action'] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e['reward'] for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e['next_state'] for e in batch])).to(self.device)
        next_residual_seqs = torch.FloatTensor(np.array([e['next_residual_seq'] for e in batch])).to(self.device)
        dones = torch.FloatTensor([e['done'] for e in batch]).to(self.device)
        
        # 更新Q网络
        with torch.no_grad():
            next_action, next_probs, next_log_probs = self.network.get_action(
                next_states, next_residual_seqs
            )
            next_q1, next_q2 = self.target_network.get_q_values(
                next_states, next_residual_seqs
            )
            next_q = torch.min(next_q1, next_q2)
            # V(s') = E[Q(s',a') - alpha * log pi(a'|s')]
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=-1)
            target_q = rewards + self.gamma * (1 - dones) * next_v
        
        current_q1, current_q2 = self.network.get_q_values(states, residual_seqs)
        current_q1 = current_q1.gather(1, actions.unsqueeze(1)).squeeze()
        current_q2 = current_q2.gather(1, actions.unsqueeze(1)).squeeze()
        
        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.q_optimizer.step()
        
        # 更新策略网络
        _, probs, log_probs = self.network.get_action(states, residual_seqs)
        q1, q2 = self.network.get_q_values(states, residual_seqs)
        min_q = torch.min(q1, q2)
        
        policy_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=-1).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # 更新温度参数
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()
        
        # 软更新目标网络
        for param, target_param in zip(self.network.parameters(), 
                                       self.target_network.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        
        self.training_step += 1
        
        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha,
            'entropy': entropy.item()
        }
    
    def get_detection_delay(self) -> Optional[int]:
        """获取检测延迟 (如果有首次正确检测)"""
        if self.first_correct_detection_step is not None and self.fault_onset_step is not None:
            return self.first_correct_detection_step - self.fault_onset_step
        return None
    
    def save(self, path: str):
        """保存模型"""
        if not TORCH_AVAILABLE:
            return
        
        torch.save({
            'network': self.network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'log_alpha': self.log_alpha,
            'training_step': self.training_step,
            'config': self.config
        }, path)
        print(f"[RLDiagnosisAgent] 模型已保存: {path}")
    
    def load(self, path: str):
        """加载模型"""
        if not TORCH_AVAILABLE:
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp().item()
        self.training_step = checkpoint['training_step']
        print(f"[RLDiagnosisAgent] 模型已加载: {path}")


# ============================================================
# 工厂函数
# ============================================================

def create_rl_diagnosis_agent(config: Dict = None) -> RLDiagnosisAgent:
    """
    创建RL诊断智能体
    
    Args:
        config: 配置参数
    
    Returns:
        RLDiagnosisAgent实例
    """
    default_config = {
        'state_dim': 12,
        'sequence_len': 10,
        'hidden_dim': 256,
        'lr': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'batch_size': 128,
        'buffer_size': 100000,
        'accuracy_weight': 1.0,
        'delay_penalty': 0.1,
        'confidence_weight': 0.2,
        'control_feedback_weight': 0.3
    }
    
    if config:
        default_config.update(config)
    
    return RLDiagnosisAgent(default_config)


if __name__ == "__main__":
    # 测试
    agent = create_rl_diagnosis_agent()
    agent.reset()
    
    # 模拟观测
    observation = {'Pmax': 175, 'Pcomp': 155, 'Texh': 360}
    residuals = {'Pmax': 0.02, 'Pcomp': 0.01, 'Texh': 0.03}
    
    # 执行诊断
    result = agent.diagnose(
        observation, residuals,
        ground_truth=FaultType.INJECTION_TIMING
    )
    
    print(f"诊断结果: {result.predicted_fault_type.name}")
    print(f"置信度: {result.confidence:.2f}")
    print(f"正确性: {result.is_correct}")
