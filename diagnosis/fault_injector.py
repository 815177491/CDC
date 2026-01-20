"""
故障注入模块
============
在模型中模拟各类故障模式
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


class FaultType(Enum):
    """故障类型枚举"""
    NONE = auto()                    # 无故障
    INJECTION_TIMING = auto()        # 喷油正时偏差
    CYLINDER_LEAK = auto()           # 气缸泄漏
    FUEL_DEGRADATION = auto()        # 燃油系统故障
    INJECTOR_DRIFT = auto()          # 喷油器漂移
    TURBOCHARGER_FOULING = auto()    # 增压器积碳


@dataclass
class FaultProfile:
    """故障特征描述"""
    fault_type: FaultType
    severity: float  # 故障严重程度 (0-1)
    onset_time: float  # 故障发生时刻 [s]
    ramp_time: float  # 故障发展时间 [s]
    description: str = ""
    
    def get_magnitude(self, current_time: float) -> float:
        """
        获取当前时刻的故障幅值
        
        支持阶跃和斜坡故障模式
        """
        if current_time < self.onset_time:
            return 0.0
        
        elapsed = current_time - self.onset_time
        
        if self.ramp_time <= 0:
            # 阶跃故障
            return self.severity
        else:
            # 斜坡故障
            progress = min(elapsed / self.ramp_time, 1.0)
            return self.severity * progress


class FaultInjector:
    """
    故障注入器
    
    用于在发动机模型中模拟各类故障模式
    """
    
    def __init__(self, engine):
        """
        初始化故障注入器
        
        Args:
            engine: MarineEngine0D实例
        """
        self.engine = engine
        self.active_faults: Dict[FaultType, FaultProfile] = {}
        self.fault_history: list = []
        
        # 保存原始参数用于恢复
        self.original_timing: float = engine.combustion.injection_timing
        self.original_fuel_factor: float = 1.0
    
    def inject_fault(self, fault_profile: FaultProfile):
        """
        注入故障
        
        Args:
            fault_profile: 故障特征描述
        """
        self.active_faults[fault_profile.fault_type] = fault_profile
        self.fault_history.append({
            'type': fault_profile.fault_type,
            'onset_time': fault_profile.onset_time,
            'severity': fault_profile.severity
        })
    
    def apply_faults(self, current_time: float):
        """
        应用所有激活的故障
        
        根据当前时刻计算故障幅值并修改模型参数
        
        Args:
            current_time: 当前仿真时间 [s]
        """
        for fault_type, profile in self.active_faults.items():
            magnitude = profile.get_magnitude(current_time)
            
            if magnitude <= 0:
                continue
            
            if fault_type == FaultType.INJECTION_TIMING:
                # 喷油正时偏差: 正值表示提前
                timing_offset = magnitude * 5.0  # 最大偏差5度
                self.engine.combustion.injection_timing = (
                    self.original_timing + timing_offset
                )
            
            elif fault_type == FaultType.CYLINDER_LEAK:
                # 气缸泄漏: magnitude直接作为泄漏因子
                self.engine.solver.leak_factor = magnitude * 0.1  # 最大10%泄漏
            
            elif fault_type == FaultType.FUEL_DEGRADATION:
                # 燃油衰减: 1-magnitude的燃油可用
                self.engine.solver.fuel_degradation = 1.0 - magnitude * 0.3
            
            elif fault_type == FaultType.INJECTOR_DRIFT:
                # 喷油器漂移: 同时影响正时和喷油量
                timing_drift = magnitude * 2.0
                fuel_drift = 1.0 - magnitude * 0.15
                self.engine.combustion.injection_timing = (
                    self.original_timing + timing_drift
                )
                self.engine.solver.fuel_degradation = fuel_drift
            
            elif fault_type == FaultType.TURBOCHARGER_FOULING:
                # 增压器积碳: 降低扫气压力
                # 注意: 这需要在工况设置时考虑
                pass
    
    def clear_all_faults(self):
        """清除所有故障并恢复原始参数"""
        self.active_faults.clear()
        
        # 恢复原始参数
        self.engine.combustion.injection_timing = self.original_timing
        self.engine.solver.leak_factor = 0.0
        self.engine.solver.fuel_degradation = 1.0
    
    def clear_fault(self, fault_type: FaultType):
        """清除指定类型的故障"""
        if fault_type in self.active_faults:
            del self.active_faults[fault_type]
    
    def get_active_faults(self) -> Dict[FaultType, float]:
        """获取当前激活的故障及其严重程度"""
        return {ft: fp.severity for ft, fp in self.active_faults.items()}
    
    def create_timing_fault(self, offset_deg: float, 
                            onset_time: float = 0.0,
                            ramp_time: float = 0.0) -> FaultProfile:
        """
        创建喷油正时故障
        
        Args:
            offset_deg: 正时偏差 [deg], 正值=提前
            onset_time: 故障发生时刻 [s]
            ramp_time: 发展时间 [s]
        """
        severity = offset_deg / 5.0  # 归一化到0-1
        return FaultProfile(
            fault_type=FaultType.INJECTION_TIMING,
            severity=min(abs(severity), 1.0) * np.sign(severity),
            onset_time=onset_time,
            ramp_time=ramp_time,
            description=f"喷油正时偏差 {offset_deg:+.1f}°"
        )
    
    def create_leak_fault(self, leak_percent: float,
                          onset_time: float = 0.0,
                          ramp_time: float = 0.0) -> FaultProfile:
        """
        创建气缸泄漏故障
        
        Args:
            leak_percent: 泄漏百分比 [%]
        """
        severity = leak_percent / 10.0  # 10%对应severity=1
        return FaultProfile(
            fault_type=FaultType.CYLINDER_LEAK,
            severity=min(severity, 1.0),
            onset_time=onset_time,
            ramp_time=ramp_time,
            description=f"气缸泄漏 {leak_percent:.1f}%"
        )
    
    def create_fuel_fault(self, degradation_percent: float,
                          onset_time: float = 0.0,
                          ramp_time: float = 0.0) -> FaultProfile:
        """
        创建燃油系统故障
        
        Args:
            degradation_percent: 燃油衰减百分比 [%]
        """
        severity = degradation_percent / 30.0  # 30%对应severity=1
        return FaultProfile(
            fault_type=FaultType.FUEL_DEGRADATION,
            severity=min(severity, 1.0),
            onset_time=onset_time,
            ramp_time=ramp_time,
            description=f"燃油衰减 {degradation_percent:.1f}%"
        )
