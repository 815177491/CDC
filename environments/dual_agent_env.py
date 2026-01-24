#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双智能体仿真环境
================
支持诊断智能体和控制智能体的联合训练

特点:
1. 返回ground truth故障标签
2. 随机多样的故障注入（不同发生时间、不同严重程度）
3. 提供诊断和控制的独立观测接口
4. 支持噪声注入用于鲁棒性测试
5. 支持混合故障场景

Author: CDC Project
Date: 2026-01-24
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Union
from enum import Enum
import random

import sys
sys.path.append('..')

from diagnosis.fault_injector import FaultType, FaultProfile, FaultInjector
from engine.engine_model import MarineEngine0D


# ============================================================
# 环境配置
# ============================================================

@dataclass
class EnvConfig:
    """环境配置"""
    # 时间设置
    max_steps: int = 200
    dt: float = 0.5  # 时间步长 (秒)
    
    # 故障注入设置
    fault_onset_range: Tuple[float, float] = (0.0, 0.5)  # 故障发生时间范围 (相对于max_steps)
    fault_severity_range: Tuple[float, float] = (0.3, 1.0)  # 故障严重程度范围
    fault_ramp_range: Tuple[float, float] = (0.0, 5.0)  # 故障发展时间范围 (秒)
    fault_probability: float = 0.8  # 每个episode发生故障的概率
    multi_fault_probability: float = 0.1  # 同时发生多个故障的概率
    
    # 安全限制
    Pmax_limit: float = 190.0  # bar
    Pmax_target: float = 170.0  # bar
    
    # 噪声设置
    sensor_noise_std: float = 0.0  # 传感器噪声标准差 (用于测试)
    
    # 工况设置
    speed_range: Tuple[float, float] = (80.0, 120.0)  # 转速范围 (rpm)
    load_range: Tuple[float, float] = (0.5, 1.0)  # 负荷范围
    
    # 控制动作空间
    n_vit_actions: int = 9
    n_fuel_actions: int = 5
    
    # 诊断动作空间
    n_fault_types: int = 5
    n_confidence_levels: int = 4


@dataclass
class DualAgentObservation:
    """双智能体观测"""
    # 诊断观测
    diag_state: np.ndarray  # 诊断状态向量
    diag_residual_seq: np.ndarray  # 残差序列 [3, seq_len]
    
    # 控制观测
    ctrl_state: np.ndarray  # 控制状态向量
    
    # 原始测量值
    measurements: Dict[str, float]
    residuals: Dict[str, float]
    
    # 全局状态
    global_state: np.ndarray


@dataclass
class DualAgentInfo:
    """双智能体附加信息"""
    # Ground Truth
    ground_truth_fault: FaultType
    fault_severity: float
    fault_onset_step: Optional[int]
    is_fault_active: bool
    
    # 系统状态
    Pmax: float
    Pmax_violation: bool
    current_vit: float
    current_fuel: float
    
    # 诊断反馈
    diagnosis_correct: Optional[bool]
    detection_delay: Optional[int]
    
    # 控制反馈
    control_improvement: float
    
    # 其他
    step: int
    done: bool
    truncated: bool


# ============================================================
# 双智能体环境
# ============================================================

class DualAgentEngineEnv:
    """
    双智能体柴油机仿真环境
    
    支持:
    - 诊断智能体 (RL-based): 故障检测与分类
    - 控制智能体 (RL-based): VIT和燃油调整
    - 随机故障注入
    - Ground truth故障标签
    - 协同奖励计算
    """
    
    # 故障类型列表 (排除TURBOCHARGER_FOULING，因为需要特殊处理)
    FAULT_TYPES = [
        FaultType.NONE,
        FaultType.INJECTION_TIMING,
        FaultType.CYLINDER_LEAK,
        FaultType.FUEL_DEGRADATION,
        FaultType.INJECTOR_DRIFT
    ]
    
    def __init__(self, config: EnvConfig = None, engine: MarineEngine0D = None):
        """
        初始化环境
        
        Args:
            config: 环境配置
            engine: 发动机模型实例（可选）
        """
        self.config = config or EnvConfig()
        
        # 初始化发动机模型
        if engine is not None:
            self.engine = engine
        else:
            self.engine = self._create_default_engine()
        
        # 故障注入器
        self.fault_injector = FaultInjector(self.engine)
        
        # 状态维度
        self.diag_state_dim = 12
        self.ctrl_state_dim = 10
        self.sequence_len = 10
        
        # 动作维度
        self.diag_action_dim = self.config.n_fault_types * self.config.n_confidence_levels  # 20
        self.ctrl_action_dim = self.config.n_vit_actions * self.config.n_fuel_actions  # 45
        
        # VIT和燃油动作映射
        self.vit_actions = np.linspace(-8, 4, self.config.n_vit_actions)
        self.fuel_actions = np.linspace(0.7, 1.0, self.config.n_fuel_actions)
        
        # 状态变量
        self.current_step = 0
        self.current_vit = 0.0
        self.current_fuel = 1.0
        self.current_speed = 100.0  # rpm
        self.current_load = 0.75
        
        # 故障状态
        self.ground_truth_fault = FaultType.NONE
        self.fault_severity = 0.0
        self.fault_onset_step = None
        self.first_correct_detection_step = None
        
        # 历史记录
        self.residual_history = deque(maxlen=self.sequence_len)
        self.Pmax_history = deque(maxlen=20)
        self.prev_measurements = None
        self.last_diagnosis_action = 0
        
        # 基准值 (健康状态)
        self.baseline_Pmax = 170.0
        self.baseline_Pcomp = 150.0
        self.baseline_Texh = 350.0
        
        print(f"[DualAgentEngineEnv] 初始化完成")
        print(f"  - 诊断状态维度: {self.diag_state_dim}")
        print(f"  - 控制状态维度: {self.ctrl_state_dim}")
        print(f"  - 诊断动作维度: {self.diag_action_dim}")
        print(f"  - 控制动作维度: {self.ctrl_action_dim}")
    
    def _create_default_engine(self) -> MarineEngine0D:
        """创建默认发动机模型"""
        try:
            from engine.engine_model import MarineEngine0D
            engine = MarineEngine0D()
            return engine
        except Exception as e:
            print(f"[警告] 无法创建发动机模型: {e}")
            return None
    
    def reset(self, 
             fault_type: Optional[FaultType] = None,
             fault_severity: Optional[float] = None,
             fault_onset: Optional[float] = None,
             speed: Optional[float] = None,
             load: Optional[float] = None,
             noise_std: Optional[float] = None) -> Tuple[DualAgentObservation, DualAgentInfo]:
        """
        重置环境
        
        Args:
            fault_type: 指定故障类型 (None则随机)
            fault_severity: 指定故障严重程度 (None则随机)
            fault_onset: 指定故障发生时间比例 (None则随机)
            speed: 指定转速 (None则随机)
            load: 指定负荷 (None则随机)
            noise_std: 传感器噪声标准差 (None则使用默认)
        
        Returns:
            observation: 初始观测
            info: 初始信息
        """
        self.current_step = 0
        self.current_vit = 0.0
        self.current_fuel = 1.0
        self.first_correct_detection_step = None
        self.last_diagnosis_action = 0
        
        # 清空历史
        self.residual_history.clear()
        self.Pmax_history.clear()
        self.prev_measurements = None
        
        # 初始化残差历史
        for _ in range(self.sequence_len):
            self.residual_history.append([0.0, 0.0, 0.0])
        
        # 随机工况
        if speed is None:
            self.current_speed = random.uniform(*self.config.speed_range)
        else:
            self.current_speed = speed
        
        if load is None:
            self.current_load = random.uniform(*self.config.load_range)
        else:
            self.current_load = load
        
        # 设置噪声
        if noise_std is not None:
            self.config.sensor_noise_std = noise_std
        
        # 清除之前的故障
        self.fault_injector.clear_all_faults()
        
        # 随机注入故障
        self._inject_random_fault(fault_type, fault_severity, fault_onset)
        
        # 更新基准值
        self._update_baseline()
        
        # 获取初始观测
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _inject_random_fault(self, 
                            fault_type: Optional[FaultType] = None,
                            fault_severity: Optional[float] = None,
                            fault_onset: Optional[float] = None):
        """随机注入故障"""
        # 决定是否注入故障
        if random.random() > self.config.fault_probability:
            self.ground_truth_fault = FaultType.NONE
            self.fault_severity = 0.0
            self.fault_onset_step = None
            return
        
        # 选择故障类型
        if fault_type is None:
            # 随机选择 (排除NONE)
            fault_type = random.choice(self.FAULT_TYPES[1:])
        self.ground_truth_fault = fault_type
        
        # 确定严重程度
        if fault_severity is None:
            self.fault_severity = random.uniform(*self.config.fault_severity_range)
        else:
            self.fault_severity = fault_severity
        
        # 确定发生时间
        if fault_onset is None:
            onset_ratio = random.uniform(*self.config.fault_onset_range)
        else:
            onset_ratio = fault_onset
        self.fault_onset_step = int(onset_ratio * self.config.max_steps)
        
        # 确定发展时间
        ramp_time = random.uniform(*self.config.fault_ramp_range)
        
        # 创建故障配置
        fault_profile = FaultProfile(
            fault_type=fault_type,
            severity=self.fault_severity,
            onset_time=self.fault_onset_step * self.config.dt,
            ramp_time=ramp_time,
            description=f"Random {fault_type.name} fault"
        )
        
        # 注入故障
        self.fault_injector.inject_fault(fault_profile)
        
        # 检查是否注入多个故障
        if random.random() < self.config.multi_fault_probability:
            # 选择另一个故障类型
            available_faults = [f for f in self.FAULT_TYPES[1:] if f != fault_type]
            if available_faults:
                second_fault = random.choice(available_faults)
                second_severity = random.uniform(0.2, 0.6)  # 第二个故障通常较轻
                second_onset = random.uniform(onset_ratio, min(onset_ratio + 0.2, 1.0))
                
                second_profile = FaultProfile(
                    fault_type=second_fault,
                    severity=second_severity,
                    onset_time=second_onset * self.config.max_steps * self.config.dt,
                    ramp_time=random.uniform(0, 3),
                    description=f"Secondary {second_fault.name} fault"
                )
                self.fault_injector.inject_fault(second_profile)
    
    def _update_baseline(self):
        """更新基准值"""
        # 基于当前工况计算基准值
        speed_factor = self.current_speed / 100.0
        load_factor = self.current_load
        
        self.baseline_Pmax = 170.0 * load_factor * (0.8 + 0.2 * speed_factor)
        self.baseline_Pcomp = 150.0 * load_factor * (0.8 + 0.2 * speed_factor)
        self.baseline_Texh = 350.0 * (0.7 + 0.3 * load_factor) * (0.9 + 0.1 * speed_factor)
    
    def step(self, diag_action: int, ctrl_action: int) -> Tuple[
        DualAgentObservation, float, float, bool, DualAgentInfo
    ]:
        """
        执行一步
        
        Args:
            diag_action: 诊断动作索引
            ctrl_action: 控制动作索引
        
        Returns:
            observation: 新观测
            diag_reward: 诊断奖励
            ctrl_reward: 控制奖励
            done: 是否结束
            info: 附加信息
        """
        self.current_step += 1
        current_time = self.current_step * self.config.dt
        
        # 解码控制动作
        vit_idx = ctrl_action // self.config.n_fuel_actions
        fuel_idx = ctrl_action % self.config.n_fuel_actions
        self.current_vit = self.vit_actions[vit_idx]
        self.current_fuel = self.fuel_actions[fuel_idx]
        
        # 应用故障效果
        self.fault_injector.apply_faults(current_time)
        
        # 模拟发动机响应
        measurements = self._simulate_engine_response()
        
        # 计算残差
        residuals = self._compute_residuals(measurements)
        
        # 更新历史
        self.residual_history.append([
            residuals['Pmax'], residuals['Pcomp'], residuals['Texh']
        ])
        self.Pmax_history.append(measurements['Pmax'])
        
        # 解码诊断动作
        diag_fault_idx = diag_action // self.config.n_confidence_levels
        diag_conf_idx = diag_action % self.config.n_confidence_levels
        predicted_fault = self.FAULT_TYPES[diag_fault_idx]
        confidence = (diag_conf_idx + 1) * 0.25
        
        # 判断诊断是否正确
        diagnosis_correct = (predicted_fault == self.ground_truth_fault)
        
        # 记录首次正确检测
        if diagnosis_correct and self.ground_truth_fault != FaultType.NONE:
            if self.first_correct_detection_step is None:
                self.first_correct_detection_step = self.current_step
        
        # 计算检测延迟
        detection_delay = None
        if self.first_correct_detection_step is not None and self.fault_onset_step is not None:
            detection_delay = self.first_correct_detection_step - self.fault_onset_step
        
        # 计算控制改善
        control_improvement = self._compute_control_improvement()
        
        # 计算奖励
        diag_reward = self._compute_diagnosis_reward(
            predicted_fault, confidence, diagnosis_correct, 
            detection_delay, control_improvement
        )
        ctrl_reward = self._compute_control_reward(measurements)
        
        # 更新状态
        self.prev_measurements = measurements
        self.last_diagnosis_action = diag_action
        
        # 检查是否结束
        done = self.current_step >= self.config.max_steps
        truncated = measurements['Pmax'] > self.config.Pmax_limit + 20  # 严重超限
        
        # 获取观测和信息
        observation = self._get_observation()
        info = self._get_info()
        info.diagnosis_correct = diagnosis_correct
        info.detection_delay = detection_delay
        info.control_improvement = control_improvement
        info.done = done
        info.truncated = truncated
        
        return observation, diag_reward, ctrl_reward, done or truncated, info
    
    def _simulate_engine_response(self) -> Dict[str, float]:
        """模拟发动机响应"""
        # 基础值
        base_Pmax = self.baseline_Pmax
        base_Pcomp = self.baseline_Pcomp
        base_Texh = self.baseline_Texh
        
        # VIT效果
        vit_effect_Pmax = -self.current_vit * 2.0  # VIT提前会降低Pmax
        vit_effect_Texh = self.current_vit * 5.0   # VIT提前会升高排温
        
        # 燃油效果
        fuel_effect_Pmax = (self.current_fuel - 1.0) * 30.0  # 削减燃油降低Pmax
        fuel_effect_Texh = (self.current_fuel - 1.0) * 20.0
        
        # 故障效果
        fault_effect = self._get_fault_effects()
        
        # 计算最终值
        Pmax = base_Pmax + vit_effect_Pmax + fuel_effect_Pmax + fault_effect['Pmax']
        Pcomp = base_Pcomp + fault_effect['Pcomp']
        Texh = base_Texh + vit_effect_Texh + fuel_effect_Texh + fault_effect['Texh']
        
        # 添加传感器噪声
        if self.config.sensor_noise_std > 0:
            Pmax += np.random.normal(0, self.config.sensor_noise_std * 5)
            Pcomp += np.random.normal(0, self.config.sensor_noise_std * 5)
            Texh += np.random.normal(0, self.config.sensor_noise_std * 10)
        
        return {
            'Pmax': max(0, Pmax),
            'Pcomp': max(0, Pcomp),
            'Texh': max(200, Texh)
        }
    
    def _get_fault_effects(self) -> Dict[str, float]:
        """获取故障效果"""
        effects = {'Pmax': 0.0, 'Pcomp': 0.0, 'Texh': 0.0}
        
        current_time = self.current_step * self.config.dt
        
        for fault_type, profile in self.fault_injector.active_faults.items():
            magnitude = profile.get_magnitude(current_time)
            
            if magnitude <= 0:
                continue
            
            if fault_type == FaultType.INJECTION_TIMING:
                # 喷油正时偏差
                effects['Pmax'] += magnitude * 15.0  # 正时提前增加Pmax
                effects['Texh'] -= magnitude * 20.0
            
            elif fault_type == FaultType.CYLINDER_LEAK:
                # 气缸泄漏
                effects['Pmax'] -= magnitude * 20.0
                effects['Pcomp'] -= magnitude * 15.0
                effects['Texh'] += magnitude * 30.0
            
            elif fault_type == FaultType.FUEL_DEGRADATION:
                # 燃油系统故障
                effects['Pmax'] -= magnitude * 10.0
                effects['Texh'] -= magnitude * 15.0
            
            elif fault_type == FaultType.INJECTOR_DRIFT:
                # 喷油器漂移
                effects['Pmax'] -= magnitude * 12.0
                effects['Pcomp'] -= magnitude * 8.0
        
        return effects
    
    def _compute_residuals(self, measurements: Dict[str, float]) -> Dict[str, float]:
        """计算残差"""
        residuals = {
            'Pmax': (measurements['Pmax'] - self.baseline_Pmax) / self.baseline_Pmax,
            'Pcomp': (measurements['Pcomp'] - self.baseline_Pcomp) / self.baseline_Pcomp,
            'Texh': (measurements['Texh'] - self.baseline_Texh) / self.baseline_Texh
        }
        return residuals
    
    def _compute_control_improvement(self) -> float:
        """计算控制改善量"""
        if len(self.Pmax_history) < 5:
            return 0.0
        
        # 比较最近几步的Pmax与之前的
        recent_Pmax = list(self.Pmax_history)[-3:]
        earlier_Pmax = list(self.Pmax_history)[:-3]
        
        if not earlier_Pmax:
            return 0.0
        
        recent_avg = np.mean(recent_Pmax)
        earlier_avg = np.mean(earlier_Pmax)
        
        # 目标是接近Pmax_target
        target = self.config.Pmax_target
        
        earlier_deviation = abs(earlier_avg - target)
        recent_deviation = abs(recent_avg - target)
        
        improvement = (earlier_deviation - recent_deviation) / target
        
        return improvement
    
    def _compute_diagnosis_reward(self, predicted_fault: FaultType,
                                  confidence: float,
                                  is_correct: bool,
                                  detection_delay: Optional[int],
                                  control_improvement: float) -> float:
        """计算诊断奖励"""
        reward = 0.0
        
        # 1. 诊断准确性奖励
        if is_correct:
            reward += 1.0
        else:
            reward -= 1.0
        
        # 2. 置信度校准
        if is_correct:
            reward += 0.2 * confidence  # 正确时高置信度加分
        else:
            reward -= 0.4 * confidence  # 错误时高置信度减分更多
        
        # 3. 检测延迟惩罚
        if detection_delay is not None and detection_delay > 0:
            delay_penalty = min(0.5, detection_delay * 0.05)
            reward -= delay_penalty
        
        # 4. 下游控制效果反馈
        if is_correct and control_improvement > 0:
            reward += 0.3 * control_improvement
        
        # 5. 特殊惩罚
        # 误报
        if self.ground_truth_fault == FaultType.NONE and predicted_fault != FaultType.NONE:
            reward -= 0.5
        
        # 漏报
        if self.ground_truth_fault != FaultType.NONE and predicted_fault == FaultType.NONE:
            # 故障已发生但未检测
            if self.fault_onset_step is not None and self.current_step > self.fault_onset_step:
                reward -= 0.8
        
        return reward
    
    def _compute_control_reward(self, measurements: Dict[str, float]) -> float:
        """计算控制奖励"""
        reward = 0.0
        Pmax = measurements['Pmax']
        
        # 1. 安全奖励
        if Pmax <= self.config.Pmax_limit:
            # 越接近目标越好
            deviation = abs(Pmax - self.config.Pmax_target) / self.config.Pmax_target
            reward += 1.0 - deviation
        else:
            # 超限惩罚
            overshoot = (Pmax - self.config.Pmax_limit) / self.config.Pmax_limit
            reward -= 5.0 * overshoot
        
        # 2. 效率惩罚
        fuel_penalty = (1.0 - self.current_fuel) * 0.5
        vit_penalty = abs(self.current_vit) * 0.02
        reward -= (fuel_penalty + vit_penalty)
        
        # 3. 平稳控制奖励
        if len(self.Pmax_history) >= 3:
            recent = list(self.Pmax_history)[-3:]
            volatility = np.std(recent) / self.config.Pmax_target
            reward -= volatility * 0.5
        
        return reward
    
    def _get_observation(self) -> DualAgentObservation:
        """获取观测"""
        # 获取当前测量值
        if self.prev_measurements is None:
            measurements = {
                'Pmax': self.baseline_Pmax,
                'Pcomp': self.baseline_Pcomp,
                'Texh': self.baseline_Texh
            }
        else:
            measurements = self.prev_measurements
        
        # 计算残差
        residuals = self._compute_residuals(measurements)
        
        # 计算变化率
        if self.prev_measurements is not None and len(self.Pmax_history) >= 2:
            dPmax = (self.Pmax_history[-1] - self.Pmax_history[-2]) / self.baseline_Pmax
            dTexh = 0.0  # 简化
        else:
            dPmax = 0.0
            dTexh = 0.0
        
        # 诊断状态 (12维)
        diag_state = np.array([
            measurements['Pmax'] / 200.0,
            measurements['Pcomp'] / 200.0,
            measurements['Texh'] / 500.0,
            residuals['Pmax'],
            residuals['Pcomp'],
            residuals['Texh'],
            dPmax,
            dTexh,
            (self.current_vit + 8) / 12.0,
            (self.current_fuel - 0.7) / 0.3,
            min(self.current_step / self.config.max_steps, 1.0),
            self.last_diagnosis_action / self.diag_action_dim
        ], dtype=np.float32)
        
        # 残差序列
        diag_residual_seq = np.array(list(self.residual_history), dtype=np.float32).T
        
        # 控制状态 (10维)
        # 故障类型编码
        fault_enc = self.FAULT_TYPES.index(self.ground_truth_fault) / len(self.FAULT_TYPES) \
                   if self.is_fault_active() else 0.0
        
        # 模式编码
        if self.ground_truth_fault == FaultType.NONE:
            mode_enc = 0.0
        elif self.fault_severity < 0.3:
            mode_enc = 0.33
        elif self.fault_severity < 0.7:
            mode_enc = 0.66
        else:
            mode_enc = 1.0
        
        ctrl_state = np.array([
            measurements['Pmax'] / 200.0,
            measurements['Pcomp'] / 200.0,
            measurements['Texh'] / 500.0,
            residuals['Pmax'],
            residuals['Pcomp'],
            residuals['Texh'],
            fault_enc,
            mode_enc,
            (self.current_vit + 8) / 12.0,
            (self.current_fuel - 0.7) / 0.3
        ], dtype=np.float32)
        
        # 全局状态
        global_state = np.concatenate([diag_state, ctrl_state])
        
        return DualAgentObservation(
            diag_state=diag_state,
            diag_residual_seq=diag_residual_seq,
            ctrl_state=ctrl_state,
            measurements=measurements,
            residuals=residuals,
            global_state=global_state
        )
    
    def _get_info(self) -> DualAgentInfo:
        """获取信息"""
        measurements = self.prev_measurements or {
            'Pmax': self.baseline_Pmax,
            'Pcomp': self.baseline_Pcomp,
            'Texh': self.baseline_Texh
        }
        
        return DualAgentInfo(
            ground_truth_fault=self.ground_truth_fault,
            fault_severity=self.fault_severity,
            fault_onset_step=self.fault_onset_step,
            is_fault_active=self.is_fault_active(),
            Pmax=measurements['Pmax'],
            Pmax_violation=measurements['Pmax'] > self.config.Pmax_limit,
            current_vit=self.current_vit,
            current_fuel=self.current_fuel,
            diagnosis_correct=None,
            detection_delay=None,
            control_improvement=0.0,
            step=self.current_step,
            done=False,
            truncated=False
        )
    
    def is_fault_active(self) -> bool:
        """检查故障是否已激活"""
        if self.ground_truth_fault == FaultType.NONE:
            return False
        if self.fault_onset_step is None:
            return False
        return self.current_step >= self.fault_onset_step
    
    def get_joint_reward(self, diag_reward: float, ctrl_reward: float,
                        diagnosis_correct: bool,
                        diag_weight: float = 0.4,
                        ctrl_weight: float = 0.4,
                        coop_weight: float = 0.2) -> float:
        """
        计算联合奖励
        
        Args:
            diag_reward: 诊断奖励
            ctrl_reward: 控制奖励
            diagnosis_correct: 诊断是否正确
            diag_weight: 诊断奖励权重
            ctrl_weight: 控制奖励权重
            coop_weight: 协同奖励权重
        
        Returns:
            联合奖励
        """
        # 基础加权
        joint = diag_weight * diag_reward + ctrl_weight * ctrl_reward
        
        # 协同奖励：诊断正确且控制效果好时给予额外奖励
        if diagnosis_correct and ctrl_reward > 0:
            joint += coop_weight * (ctrl_reward * 0.5)
        
        # 协同惩罚：诊断错误导致控制失效
        if not diagnosis_correct and ctrl_reward < 0:
            joint -= coop_weight * abs(ctrl_reward) * 0.3
        
        return joint
    
    def decode_ctrl_action(self, action_idx: int) -> Tuple[float, float]:
        """解码控制动作"""
        vit_idx = action_idx // self.config.n_fuel_actions
        fuel_idx = action_idx % self.config.n_fuel_actions
        return self.vit_actions[vit_idx], self.fuel_actions[fuel_idx]
    
    def decode_diag_action(self, action_idx: int) -> Tuple[FaultType, float]:
        """解码诊断动作"""
        fault_idx = action_idx // self.config.n_confidence_levels
        conf_idx = action_idx % self.config.n_confidence_levels
        return self.FAULT_TYPES[fault_idx], (conf_idx + 1) * 0.25


# ============================================================
# 工厂函数
# ============================================================

def create_dual_agent_env(config: Dict = None, 
                         engine: MarineEngine0D = None) -> DualAgentEngineEnv:
    """
    创建双智能体环境
    
    Args:
        config: 配置参数字典
        engine: 发动机模型实例
    
    Returns:
        DualAgentEngineEnv实例
    """
    env_config = EnvConfig()
    
    if config:
        for key, value in config.items():
            if hasattr(env_config, key):
                setattr(env_config, key, value)
    
    return DualAgentEngineEnv(env_config, engine)


if __name__ == "__main__":
    # 测试环境
    print("=" * 60)
    print("测试双智能体环境")
    print("=" * 60)
    
    env = create_dual_agent_env()
    
    # 重置
    obs, info = env.reset()
    print(f"Ground Truth: {info.ground_truth_fault.name}")
    print(f"故障严重程度: {info.fault_severity:.2f}")
    print(f"故障发生步数: {info.fault_onset_step}")
    print(f"诊断状态维度: {obs.diag_state.shape}")
    print(f"残差序列维度: {obs.diag_residual_seq.shape}")
    print(f"控制状态维度: {obs.ctrl_state.shape}")
    
    # 运行几步
    print("\n运行10步:")
    for i in range(10):
        diag_action = np.random.randint(env.diag_action_dim)
        ctrl_action = np.random.randint(env.ctrl_action_dim)
        
        obs, diag_r, ctrl_r, done, info = env.step(diag_action, ctrl_action)
        
        # 解码动作
        pred_fault, conf = env.decode_diag_action(diag_action)
        vit, fuel = env.decode_ctrl_action(ctrl_action)
        
        joint_r = env.get_joint_reward(diag_r, ctrl_r, info.diagnosis_correct or False)
        
        print(f"Step {i+1}: Pmax={info.Pmax:.1f}, "
              f"诊断={pred_fault.name}({conf:.0%}), "
              f"VIT={vit:.1f}, Fuel={fuel:.2f}, "
              f"R_diag={diag_r:.2f}, R_ctrl={ctrl_r:.2f}, R_joint={joint_r:.2f}")
        
        if done:
            break
    
    print("\n测试完成!")
