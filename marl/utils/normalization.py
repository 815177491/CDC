"""
归一化工具模块
==============
提供观测和奖励的动态归一化
"""

import numpy as np
from typing import Dict, Tuple, Optional


class RunningMeanStd:
    """
    动态统计量追踪器
    
    使用Welford算法在线计算均值和方差
    """
    
    def __init__(self, shape: Tuple = (), epsilon: float = 1e-4):
        """
        初始化
        
        Args:
            shape: 统计量形状
            epsilon: 防止除零的小常数
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, arr: np.ndarray):
        """
        更新统计量
        
        Args:
            arr: 新观测数据 [batch, *shape] 或 [*shape]
        """
        if arr.ndim == len(self.mean.shape):
            # 单个样本
            arr = arr.reshape(1, *arr.shape)
        
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int):
        """使用并行算法合并统计量"""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        # 更新均值
        self.mean = self.mean + delta * batch_count / total_count
        
        # 更新方差（使用并行方差合并公式）
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        
        self.count = total_count
    
    def normalize(self, arr: np.ndarray) -> np.ndarray:
        """标准化数据"""
        return (arr - self.mean) / np.sqrt(self.var + 1e-8)
    
    def denormalize(self, arr: np.ndarray) -> np.ndarray:
        """反标准化数据"""
        return arr * np.sqrt(self.var + 1e-8) + self.mean
    
    def save(self) -> Dict:
        """保存状态"""
        return {
            'mean': self.mean.copy(),
            'var': self.var.copy(),
            'count': self.count
        }
    
    def load(self, state: Dict):
        """加载状态"""
        self.mean = state['mean']
        self.var = state['var']
        self.count = state['count']


class ObsNormalizer:
    """
    观测归一化器
    
    对输入观测进行动态标准化，并限制范围
    """
    
    def __init__(self, obs_shape: Tuple[int, ...], clip: float = 10.0):
        """
        初始化
        
        Args:
            obs_shape: 观测向量形状
            clip: 归一化后的裁剪范围
        """
        self.rms = RunningMeanStd(shape=obs_shape)
        self.clip = clip
    
    def normalize(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        """
        归一化观测
        
        Args:
            obs: 原始观测
            update: 是否更新统计量（训练时True，评估时False）
            
        Returns:
            归一化后的观测
        """
        if update:
            self.rms.update(obs)
        
        normalized = self.rms.normalize(obs)
        return np.clip(normalized, -self.clip, self.clip)
    
    def save(self) -> Dict:
        """保存状态"""
        return self.rms.save()
    
    def load(self, state: Dict):
        """加载状态"""
        self.rms.load(state)


class RewardNormalizer:
    """
    奖励归一化器
    
    对奖励进行动态标准化，缓解不同任务奖励尺度差异
    """
    
    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):
        """
        初始化
        
        Args:
            gamma: 折扣因子，用于计算回报的运行方差
            epsilon: 防止除零
        """
        self.rms = RunningMeanStd(shape=())
        self.gamma = gamma
        self.epsilon = epsilon
        self.returns = 0.0  # 累积折扣回报
    
    def normalize(self, reward: float, done: bool = False) -> float:
        """
        归一化单步奖励
        
        使用回报的标准差进行缩放（不减均值，保持奖励符号）
        
        Args:
            reward: 原始奖励
            done: 是否为episode结束
            
        Returns:
            归一化后的奖励
        """
        # 更新累积回报
        self.returns = self.returns * self.gamma + reward
        self.rms.update(np.array([self.returns]))
        
        # 仅除以标准差，不减均值
        normalized = reward / np.sqrt(self.rms.var + self.epsilon)
        
        # episode结束时重置累积回报
        if done:
            self.returns = 0.0
        
        return float(normalized)
    
    def save(self) -> Dict:
        """保存状态"""
        return {
            'rms': self.rms.save(),
            'returns': self.returns
        }
    
    def load(self, state: Dict):
        """加载状态"""
        self.rms.load(state['rms'])
        self.returns = state['returns']
