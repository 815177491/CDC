#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多算法控制智能体
================
支持多种强化学习算法的控制智能体，可灵活切换算法

支持的算法:
- DQN (基线)
- Dueling DQN
- PPO
- SAC
- TD3
- Decision Transformer (NeurIPS 2021)
- IQL (ICLR 2022)

Author: CDC Project
Date: 2026-01-21
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Union
from enum import Enum, auto
from collections import deque
import warnings
import os

from .base_agent import Agent, AgentMessage, MessageType

# 尝试导入RL算法库
try:
    from .rl_algorithms import (
        get_algorithm, list_algorithms, ALGORITHM_INFO,
        BaseRLAlgorithm, TORCH_AVAILABLE
    )
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    TORCH_AVAILABLE = False
    warnings.warn("RL algorithms not available. Using PID controller only.")

import sys
sys.path.append('..')

try:
    from diagnosis.fault_injector import FaultType
except ImportError:
    class FaultType(Enum):
        INJECTOR_DEGRADATION = auto()
        COMPRESSION_LOSS = auto()
        EXHAUST_RESTRICTION = auto()


class ControlMode(Enum):
    """控制模式"""
    NORMAL = auto()
    FAULT_TOLERANT = auto()
    DEGRADED = auto()
    EMERGENCY = auto()


@dataclass
class ControlAction:
    """控制动作"""
    timestamp: float
    mode: ControlMode
    vit_adjustment: float = 0.0
    fuel_adjustment: float = 0.0
    speed_target: float = 0.0
    power_limit: float = 1.0
    cylinder_mask: List[bool] = field(default_factory=list)
    message: str = ""
    action_source: str = "PID"


class MultiAlgorithmController:
    """
    多算法控制器
    支持动态切换不同的RL算法
    """
    
    def __init__(
        self,
        state_dim: int = 10,
        n_vit_actions: int = 9,
        n_fuel_actions: int = 5,
        algorithm: str = 'DQN',
        device: str = 'cpu',
        config: Dict = None
    ):
        """
        Args:
            state_dim: 状态维度
            n_vit_actions: VIT离散动作数
            n_fuel_actions: 燃油离散动作数
            algorithm: 算法名称
            device: 计算设备
            config: 算法配置
        """
        self.state_dim = state_dim
        self.n_vit_actions = n_vit_actions
        self.n_fuel_actions = n_fuel_actions
        self.action_dim = n_vit_actions * n_fuel_actions
        self.device = device
        self.algorithm_name = algorithm
        
        # 动作映射
        self.vit_actions = np.linspace(-8, 4, n_vit_actions)
        self.fuel_actions = np.linspace(0.7, 1.0, n_fuel_actions)
        
        # 算法配置
        self.config = config or {
            'lr': 1e-3,
            'gamma': 0.99,
            'batch_size': 64,
            'buffer_size': 50000,
            'epsilon': 1.0,
            'epsilon_min': 0.05,
            'epsilon_decay': 0.995
        }
        
        # 初始化RL算法
        self.agent = None
        self.is_trained = False
        
        if RL_AVAILABLE and TORCH_AVAILABLE:
            try:
                self.agent = get_algorithm(algorithm, state_dim, self.action_dim, self.config)
                print(f"[MultiAlgorithmController] 使用算法: {algorithm}")
                info = ALGORITHM_INFO.get(algorithm, {})
                print(f"  来源: {info.get('venue', 'N/A')} {info.get('year', '')}")
                print(f"  类型: {info.get('type', 'N/A')}")
            except Exception as e:
                warnings.warn(f"Failed to initialize {algorithm}: {e}. Falling back to PID.")
                self.agent = None
        
        # PID兜底控制器
        self.pid_vit = PIDController(kp=0.5, ki=0.1, kd=0.05)
        self.pid_fuel = PIDController(kp=0.3, ki=0.05, kd=0.02)
        
        # 训练统计
        self.episode_rewards = []
        self.training_losses = []
        self.current_episode_reward = 0
    
    def switch_algorithm(self, new_algorithm: str, preserve_experience: bool = True):
        """
        切换RL算法
        
        Args:
            new_algorithm: 新算法名称
            preserve_experience: 是否保留经验回放
        """
        if not RL_AVAILABLE:
            warnings.warn("RL not available")
            return
        
        old_buffer = None
        if preserve_experience and self.agent and hasattr(self.agent, 'buffer'):
            old_buffer = self.agent.buffer
        
        try:
            self.agent = get_algorithm(new_algorithm, self.state_dim, self.action_dim, self.config)
            self.algorithm_name = new_algorithm
            
            if preserve_experience and old_buffer and hasattr(self.agent, 'buffer'):
                self.agent.buffer = old_buffer
            
            print(f"[MultiAlgorithmController] 切换到算法: {new_algorithm}")
            
        except Exception as e:
            warnings.warn(f"Failed to switch to {new_algorithm}: {e}")
    
    def encode_state(
        self,
        observation: Dict[str, Any],
        diagnosis_result: Any = None,
        current_vit: float = 0.0,
        current_fuel: float = 1.0
    ) -> np.ndarray:
        """编码状态向量"""
        Pmax = observation.get('Pmax', 170) / 200.0
        Pcomp = observation.get('Pcomp', 150) / 200.0
        Texh = observation.get('Texh', 350) / 500.0
        
        residuals = {}
        if diagnosis_result and hasattr(diagnosis_result, 'residuals'):
            residuals = diagnosis_result.residuals or {}
        
        r_Pmax = residuals.get('Pmax', 0) / 10.0
        r_Pcomp = residuals.get('Pcomp', 0) / 10.0
        r_Texh = residuals.get('Texh', 0) / 50.0
        
        fault_type_enc = 0.0
        mode_enc = 0.0
        
        if diagnosis_result:
            if hasattr(diagnosis_result, 'fault_detected') and diagnosis_result.fault_detected:
                if hasattr(diagnosis_result, 'fault_type') and diagnosis_result.fault_type:
                    fault_type_enc = hash(str(diagnosis_result.fault_type)) % 10 / 10.0
            
            if hasattr(diagnosis_result, 'diagnosis_state'):
                mode_map = {'HEALTHY': 0, 'WARNING': 0.33, 'FAULT': 0.66, 'CRITICAL': 1.0}
                state_name = getattr(diagnosis_result.diagnosis_state, 'name', 'HEALTHY')
                mode_enc = mode_map.get(state_name, 0)
        
        vit_norm = (current_vit + 8) / 12.0
        fuel_norm = (current_fuel - 0.7) / 0.3
        
        state = np.array([
            Pmax, Pcomp, Texh,
            np.clip(r_Pmax, -1, 1),
            np.clip(r_Pcomp, -1, 1),
            np.clip(r_Texh, -1, 1),
            fault_type_enc,
            mode_enc,
            np.clip(vit_norm, 0, 1),
            np.clip(fuel_norm, 0, 1)
        ], dtype=np.float32)
        
        return state
    
    def decode_action(self, action_idx: int) -> Tuple[float, float]:
        """解码动作索引为VIT和燃油调整"""
        vit_idx = action_idx // self.n_fuel_actions
        fuel_idx = action_idx % self.n_fuel_actions
        
        vit_adj = self.vit_actions[np.clip(vit_idx, 0, len(self.vit_actions)-1)]
        fuel_adj = self.fuel_actions[np.clip(fuel_idx, 0, len(self.fuel_actions)-1)]
        
        return float(vit_adj), float(fuel_adj)
    
    def select_action(
        self,
        state: np.ndarray,
        explore: bool = True
    ) -> Tuple[int, float, float]:
        """
        选择动作
        
        Returns:
            action_idx: 动作索引
            vit_adjustment: VIT调整
            fuel_adjustment: 燃油调整
        """
        if self.agent is not None:
            action_idx = self.agent.select_action(state, explore=explore)
            vit_adj, fuel_adj = self.decode_action(action_idx)
            return action_idx, vit_adj, fuel_adj
        else:
            return 0, 0.0, 1.0
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """存储经验"""
        if self.agent is not None:
            if hasattr(self.agent, 'buffer'):
                self.agent.buffer.push(state, action, reward, next_state, done)
            
            # PPO需要特殊处理
            if self.algorithm_name == 'PPO' and hasattr(self.agent, 'store_transition'):
                self.agent.store_transition(state, action, reward, done)
        
        self.current_episode_reward += reward
    
    def update(self) -> Dict[str, float]:
        """更新网络"""
        if self.agent is None:
            return {}
        
        # 根据算法类型选择更新方式
        if self.algorithm_name == 'PPO':
            return self.agent.update()
        elif hasattr(self.agent, 'buffer'):
            batch_size = self.config.get('batch_size', 64)
            if len(self.agent.buffer) >= batch_size:
                batch = self.agent.buffer.sample(batch_size)
                metrics = self.agent.update(batch)
                if 'loss' in metrics:
                    self.training_losses.append(metrics['loss'])
                return metrics
        
        return {}
    
    def end_episode(self):
        """结束episode"""
        self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0
        
        # Decision Transformer需要在episode结束时更新
        if self.algorithm_name in ['DecisionTransformer', 'DT']:
            if hasattr(self.agent, 'update'):
                self.agent.update()
    
    def save(self, path: str):
        """保存模型"""
        if self.agent is not None:
            self.agent.save(path)
            print(f"模型已保存: {path}")
    
    def load(self, path: str):
        """加载模型"""
        if self.agent is not None and os.path.exists(path):
            self.agent.load(path)
            self.is_trained = True
            print(f"模型已加载: {path}")
    
    def get_algorithm_info(self) -> Dict:
        """获取当前算法信息"""
        return ALGORITHM_INFO.get(self.algorithm_name, {})
    
    @staticmethod
    def available_algorithms() -> List[str]:
        """获取可用算法列表"""
        if RL_AVAILABLE:
            return list_algorithms()
        return []


class PIDController:
    """简单PID控制器"""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.integral = 0.0
        self.prev_error = 0.0
    
    def compute(self, error: float, dt: float = 0.1) -> float:
        self.integral += error * dt
        self.integral = np.clip(self.integral, -10, 10)
        
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return output
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


class AdaptiveMultiAlgorithmControlAgent(Agent):
    """
    多算法自适应控制智能体
    
    Features:
    - 支持7种RL算法动态切换
    - 安全约束层
    - PID兜底控制
    - 在线学习
    """
    
    def __init__(
        self,
        agent_id: str = "control_agent",
        algorithm: str = 'SAC',
        n_cylinders: int = 6,
        config: Dict = None
    ):
        super().__init__(agent_id)
        
        self.n_cylinders = n_cylinders
        self.algorithm = algorithm
        
        # 初始化多算法控制器
        self.controller = MultiAlgorithmController(
            state_dim=10,
            n_vit_actions=9,
            n_fuel_actions=5,
            algorithm=algorithm,
            config=config
        )
        
        # 安全约束
        self.safety_constraints = SafetyConstraints(
            Pmax_max=180.0,
            Pmax_min=140.0,
            vit_max=6.0,
            vit_min=-10.0,
            fuel_max=1.05,
            fuel_min=0.6
        )
        
        # 控制状态
        self.current_mode = ControlMode.NORMAL
        self.current_vit = 0.0
        self.current_fuel = 1.0
        
        # 历史记录
        self.action_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=1000)
        
        # 在线学习开关
        self.online_learning = True
    
    def process_observation(
        self,
        observation: Dict[str, Any],
        diagnosis_result: Any = None
    ) -> ControlAction:
        """
        处理观测生成控制动作
        """
        timestamp = observation.get('timestamp', 0.0)
        
        # 编码状态
        state = self.controller.encode_state(
            observation, diagnosis_result,
            self.current_vit, self.current_fuel
        )
        
        # 确定控制模式
        self._update_control_mode(diagnosis_result)
        
        # 选择动作
        action_idx, vit_adj, fuel_adj = self.controller.select_action(
            state, explore=self.online_learning
        )
        
        # 应用安全约束
        vit_adj, fuel_adj = self.safety_constraints.apply(
            vit_adj, fuel_adj,
            observation.get('Pmax', 160),
            self.current_mode
        )
        
        # 更新当前控制值
        self.current_vit = vit_adj
        self.current_fuel = fuel_adj
        
        # 构造控制动作
        action = ControlAction(
            timestamp=timestamp,
            mode=self.current_mode,
            vit_adjustment=vit_adj,
            fuel_adjustment=fuel_adj,
            action_source=self.algorithm
        )
        
        self.action_history.append(action)
        
        return action
    
    def _update_control_mode(self, diagnosis_result: Any):
        """更新控制模式"""
        if diagnosis_result is None:
            self.current_mode = ControlMode.NORMAL
            return
        
        if hasattr(diagnosis_result, 'diagnosis_state'):
            state_name = getattr(diagnosis_result.diagnosis_state, 'name', 'HEALTHY')
            mode_map = {
                'HEALTHY': ControlMode.NORMAL,
                'WARNING': ControlMode.FAULT_TOLERANT,
                'FAULT': ControlMode.FAULT_TOLERANT,
                'CRITICAL': ControlMode.EMERGENCY
            }
            self.current_mode = mode_map.get(state_name, ControlMode.NORMAL)
    
    def provide_feedback(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """提供反馈用于在线学习"""
        if self.online_learning:
            self.controller.store_transition(state, action, reward, next_state, done)
            self.controller.update()
            self.reward_history.append(reward)
    
    def switch_algorithm(self, new_algorithm: str):
        """切换算法"""
        self.controller.switch_algorithm(new_algorithm)
        self.algorithm = new_algorithm
    
    def save_model(self, path: str):
        """保存模型"""
        self.controller.save(path)
    
    def load_model(self, path: str):
        """加载模型"""
        self.controller.load(path)
    
    def get_training_stats(self) -> Dict:
        """获取训练统计"""
        return {
            'algorithm': self.algorithm,
            'episode_rewards': list(self.controller.episode_rewards),
            'training_losses': self.controller.training_losses[-100:],
            'recent_rewards': list(self.reward_history)[-100:],
            'is_trained': self.controller.is_trained
        }


@dataclass
class SafetyConstraints:
    """安全约束层"""
    Pmax_max: float = 180.0
    Pmax_min: float = 140.0
    vit_max: float = 6.0
    vit_min: float = -10.0
    fuel_max: float = 1.05
    fuel_min: float = 0.6
    
    def apply(
        self,
        vit: float,
        fuel: float,
        current_Pmax: float,
        mode: ControlMode
    ) -> Tuple[float, float]:
        """应用安全约束"""
        # 基本约束
        vit = np.clip(vit, self.vit_min, self.vit_max)
        fuel = np.clip(fuel, self.fuel_min, self.fuel_max)
        
        # Pmax保护
        if current_Pmax > self.Pmax_max - 5:
            vit = min(vit, 0)  # 只允许减少
            fuel = min(fuel, 0.9)
        
        if current_Pmax < self.Pmax_min + 5:
            vit = max(vit, 0)  # 只允许增加
        
        # 紧急模式限制
        if mode == ControlMode.EMERGENCY:
            fuel = min(fuel, 0.7)
            vit = np.clip(vit, -5, 2)
        
        return vit, fuel


# 便捷工厂函数
def create_control_agent(
    algorithm: str = 'SAC',
    n_cylinders: int = 6,
    config: Dict = None
) -> AdaptiveMultiAlgorithmControlAgent:
    """
    创建控制智能体
    
    Args:
        algorithm: 算法名称 ('DQN', 'DuelingDQN', 'PPO', 'SAC', 'TD3', 'DecisionTransformer', 'IQL')
        n_cylinders: 气缸数
        config: 配置参数
    
    Returns:
        控制智能体实例
    """
    return AdaptiveMultiAlgorithmControlAgent(
        agent_id="control_agent",
        algorithm=algorithm,
        n_cylinders=n_cylinders,
        config=config
    )


# 导出算法信息
def print_available_algorithms():
    """打印可用算法信息"""
    print("\n可用的强化学习算法:")
    print("=" * 60)
    
    if not RL_AVAILABLE:
        print("  (需要安装PyTorch)")
        return
    
    for name in list_algorithms():
        info = ALGORITHM_INFO.get(name, {})
        print(f"\n  {name} ({info.get('name', '')})")
        print(f"    来源: {info.get('venue', 'N/A')} {info.get('year', '')}")
        print(f"    类型: {info.get('type', 'N/A')}")
        print(f"    简介: {info.get('description', 'N/A')}")
    
    print("\n" + "=" * 60)
