"""
协同控制器
==========
主动容错控制器 (Active Fault-Tolerant Controller)
实现故障诊断与控制的闭环协同
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum, auto
from collections import deque

import sys
sys.path.append('..')

from diagnosis import FaultDiagnoser, FaultType, DiagnosisResult


class ControlMode(Enum):
    """控制模式"""
    NORMAL = auto()        # 正常模式
    FAULT_TOLERANT = auto()  # 容错模式
    DEGRADED = auto()      # 降级模式
    EMERGENCY = auto()     # 紧急模式


@dataclass
class ControlAction:
    """控制动作"""
    timestamp: float
    mode: ControlMode
    vit_adjustment: float = 0.0      # VIT调整量 [deg]
    fuel_adjustment: float = 0.0     # 燃油调整量 [%]
    speed_target: float = 0.0        # 目标转速 [rpm]
    power_limit: float = 1.0         # 功率限制系数 (0-1)
    cylinder_mask: List[bool] = field(default_factory=list)  # 各缸使能状态
    message: str = ""


class SynergyController:
    """
    控诊协同控制器
    
    功能:
    1. 场景A: Pmax越限协同控制
       - 检测到Pmax超过安全限值时, 协同调整VIT和燃油
    2. 场景B: 能效衰退补偿控制
       - 检测到某缸做功不足时, 在健康缸重新分配负荷
    
    控制策略:
    - 优先调整VIT (低成本)
    - 其次调整燃油 (影响功率)
    - 最后降速 (严重情况)
    """
    
    def __init__(self, engine, diagnoser: FaultDiagnoser):
        """
        初始化协同控制器
        
        Args:
            engine: MarineEngine0D实例
            diagnoser: 故障诊断器
        """
        self.engine = engine
        self.diagnoser = diagnoser
        
        # 控制参数
        self.n_cylinders = engine.geometry.n_cylinders
        
        # Pmax安全限值
        self.Pmax_limit = 190.0        # bar, 硬限值
        self.Pmax_warning = 180.0      # bar, 预警值
        self.Pmax_target = 170.0       # bar, 目标值
        
        # VIT调整范围
        self.vit_min = -8.0    # deg (滞后)
        self.vit_max = 4.0     # deg (提前)
        self.vit_current = 0.0
        
        # 燃油调整范围
        self.fuel_min = 0.7    # 最低70%
        self.fuel_max = 1.0    # 最高100%
        self.fuel_current = 1.0
        
        # 各缸负荷分配
        self.cylinder_load = [1.0] * self.n_cylinders  # 归一化负荷
        self.cylinder_healthy = [True] * self.n_cylinders  # 健康状态
        
        # PID控制器参数 (用于Pmax控制)
        self.Kp_vit = 0.5      # VIT比例增益
        self.Ki_vit = 0.05     # VIT积分增益
        self.Kd_vit = 0.1      # VIT微分增益
        
        self.Kp_fuel = 0.3     # 燃油比例增益
        self.Ki_fuel = 0.02
        
        # 积分项
        self.vit_integral = 0.0
        self.fuel_integral = 0.0
        self.last_error = 0.0
        
        # 当前控制模式
        self.mode = ControlMode.NORMAL
        
        # 控制历史
        self.control_history: List[ControlAction] = []
        
        # 性能指标
        self.metrics = {
            'Pmax_violations': 0,
            'control_interventions': 0,
            'fuel_saved': 0.0,
            'power_loss': 0.0,
        }
    
    def update(self, Y_measured: Dict[str, float], 
               timestamp: float = 0.0) -> ControlAction:
        """
        主控制更新循环
        
        Args:
            Y_measured: 实测值 {'Pmax': ..., 'Pcomp': ..., 'Texh': ...}
            timestamp: 时间戳
            
        Returns:
            action: 控制动作
        """
        # 执行诊断
        diagnosis = self.diagnoser.diagnose(Y_measured, timestamp)
        
        # 根据诊断结果确定控制模式
        self._update_mode(diagnosis, Y_measured)
        
        # 根据模式执行控制策略
        if self.mode == ControlMode.NORMAL:
            action = self._normal_control(Y_measured, timestamp)
        elif self.mode == ControlMode.FAULT_TOLERANT:
            action = self._fault_tolerant_control(diagnosis, Y_measured, timestamp)
        elif self.mode == ControlMode.DEGRADED:
            action = self._degraded_control(diagnosis, Y_measured, timestamp)
        else:  # EMERGENCY
            action = self._emergency_control(Y_measured, timestamp)
        
        # 记录历史
        self.control_history.append(action)
        
        return action
    
    def _update_mode(self, diagnosis: DiagnosisResult, Y_measured: Dict):
        """根据诊断结果更新控制模式"""
        Pmax = Y_measured.get('Pmax', 0)
        
        # 紧急模式: Pmax严重超限
        if Pmax > self.Pmax_limit:
            self.mode = ControlMode.EMERGENCY
            self.metrics['Pmax_violations'] += 1
            return
        
        # 容错模式: 检测到故障
        if diagnosis.fault_detected:
            if diagnosis.fault_type in [FaultType.INJECTION_TIMING, 
                                         FaultType.FUEL_DEGRADATION]:
                self.mode = ControlMode.FAULT_TOLERANT
            elif diagnosis.fault_type == FaultType.CYLINDER_LEAK:
                self.mode = ControlMode.DEGRADED
            return
        
        # Pmax预警
        if Pmax > self.Pmax_warning:
            self.mode = ControlMode.FAULT_TOLERANT
            return
        
        # 正常模式
        self.mode = ControlMode.NORMAL
    
    def _normal_control(self, Y_measured: Dict, timestamp: float) -> ControlAction:
        """正常模式控制 - 维持当前状态"""
        return ControlAction(
            timestamp=timestamp,
            mode=ControlMode.NORMAL,
            vit_adjustment=self.vit_current,
            fuel_adjustment=0.0,
            power_limit=1.0,
            cylinder_mask=[True] * self.n_cylinders,
            message="正常运行"
        )
    
    def _fault_tolerant_control(self, diagnosis: DiagnosisResult,
                                  Y_measured: Dict, 
                                  timestamp: float) -> ControlAction:
        """
        场景A: Pmax越限协同控制
        
        策略:
        1. 优先推迟喷油正时 (VIT-) 降低Pmax
        2. 若VIT调整不足, 则减少燃油
        """
        Pmax = Y_measured.get('Pmax', self.Pmax_target)
        
        # 计算Pmax误差
        error = Pmax - self.Pmax_target
        
        # VIT调整 (PID)
        self.vit_integral += error * 0.1  # 假设dt=0.1s
        derivative = error - self.last_error
        self.last_error = error
        
        vit_delta = -(self.Kp_vit * error + 
                      self.Ki_vit * self.vit_integral +
                      self.Kd_vit * derivative)
        
        # 限幅
        new_vit = np.clip(self.vit_current + vit_delta, self.vit_min, self.vit_max)
        
        # 如果VIT已饱和且Pmax仍高, 则调整燃油
        fuel_delta = 0.0
        if new_vit <= self.vit_min and error > 5:  # 5 bar余量
            self.fuel_integral += error * 0.1
            fuel_delta = -(self.Kp_fuel * error + self.Ki_fuel * self.fuel_integral)
            fuel_delta = np.clip(fuel_delta, -0.1, 0)  # 最大减10%
        
        new_fuel = np.clip(self.fuel_current + fuel_delta, self.fuel_min, self.fuel_max)
        
        # 更新状态
        self.vit_current = new_vit
        self.fuel_current = new_fuel
        
        self.metrics['control_interventions'] += 1
        if fuel_delta < 0:
            self.metrics['fuel_saved'] += abs(fuel_delta)
        
        # 应用到发动机
        self.engine.set_injection_timing(self.engine.calibrated_params.get(
            'injection_timing', 2.0) + new_vit)
        
        message = f"Pmax控制: VIT={new_vit:+.1f}°"
        if fuel_delta != 0:
            message += f", 燃油={new_fuel*100:.0f}%"
        
        return ControlAction(
            timestamp=timestamp,
            mode=ControlMode.FAULT_TOLERANT,
            vit_adjustment=new_vit,
            fuel_adjustment=(new_fuel - 1.0) * 100,
            power_limit=new_fuel,
            cylinder_mask=[True] * self.n_cylinders,
            message=message
        )
    
    def _degraded_control(self, diagnosis: DiagnosisResult,
                           Y_measured: Dict,
                           timestamp: float) -> ControlAction:
        """
        场景B: 能效衰退补偿控制
        
        策略:
        1. 识别故障缸
        2. 在健康缸重新分配负荷
        3. 维持总功率或接受部分功率损失
        """
        # 假设根据残差识别故障缸 (这里简化为第1缸)
        faulty_cylinder = 0  # 0-indexed
        
        # 更新健康状态
        self.cylinder_healthy[faulty_cylinder] = False
        
        # 计算负荷重分配
        healthy_count = sum(self.cylinder_healthy)
        if healthy_count == 0:
            return self._emergency_control(Y_measured, timestamp)
        
        # 故障缸负荷为0, 健康缸平均分担
        fault_load = self.cylinder_load[faulty_cylinder]
        extra_per_cyl = fault_load / healthy_count
        
        for i in range(self.n_cylinders):
            if self.cylinder_healthy[i]:
                # 每缸最多增加15%负荷
                self.cylinder_load[i] = min(1.0 + extra_per_cyl, 1.15)
            else:
                self.cylinder_load[i] = 0.0
        
        # 如果无法完全补偿, 记录功率损失
        total_capacity = sum(self.cylinder_load)
        power_loss = max(0, 1.0 - total_capacity / self.n_cylinders)
        self.metrics['power_loss'] += power_loss
        
        message = f"降级运行: 缸{faulty_cylinder+1}故障, " \
                  f"健康缸负荷+{extra_per_cyl*100:.0f}%"
        
        return ControlAction(
            timestamp=timestamp,
            mode=ControlMode.DEGRADED,
            vit_adjustment=self.vit_current,
            fuel_adjustment=0.0,
            power_limit=1.0 - power_loss,
            cylinder_mask=self.cylinder_healthy.copy(),
            message=message
        )
    
    def _emergency_control(self, Y_measured: Dict, 
                           timestamp: float) -> ControlAction:
        """紧急模式控制 - 快速降功保护"""
        # 立即最大程度推迟正时
        self.vit_current = self.vit_min
        
        # 快速减少燃油
        self.fuel_current = max(self.fuel_current - 0.1, self.fuel_min)
        
        message = f"⚠️ 紧急保护: VIT={self.vit_current}°, " \
                  f"燃油={self.fuel_current*100:.0f}%"
        
        return ControlAction(
            timestamp=timestamp,
            mode=ControlMode.EMERGENCY,
            vit_adjustment=self.vit_current,
            fuel_adjustment=(self.fuel_current - 1.0) * 100,
            power_limit=self.fuel_current,
            cylinder_mask=[True] * self.n_cylinders,
            message=message
        )
    
    def reset(self):
        """重置控制器状态"""
        self.vit_current = 0.0
        self.fuel_current = 1.0
        self.vit_integral = 0.0
        self.fuel_integral = 0.0
        self.last_error = 0.0
        self.mode = ControlMode.NORMAL
        self.cylinder_load = [1.0] * self.n_cylinders
        self.cylinder_healthy = [True] * self.n_cylinders
        self.control_history.clear()
    
    def get_performance_summary(self) -> Dict:
        """获取控制器性能汇总"""
        return {
            'total_interventions': self.metrics['control_interventions'],
            'pmax_violations': self.metrics['Pmax_violations'],
            'avg_vit_adjustment': np.mean([a.vit_adjustment 
                                           for a in self.control_history]) 
                                  if self.control_history else 0,
            'avg_fuel_reduction': np.mean([a.fuel_adjustment 
                                           for a in self.control_history])
                                  if self.control_history else 0,
            'total_power_loss': self.metrics['power_loss'],
        }
    
    def simulate_synergy_response(self, 
                                   fault_profile,
                                   base_condition,
                                   duration: float = 100.0,
                                   dt: float = 1.0) -> Dict:
        """
        仿真协同控制响应
        
        Args:
            fault_profile: 故障特征
            base_condition: 基准工况
            duration: 仿真时长 [s]
            dt: 时间步长 [s]
            
        Returns:
            response: 响应数据
        """
        from diagnosis import FaultInjector
        
        # 初始化
        self.reset()
        injector = FaultInjector(self.engine)
        injector.inject_fault(fault_profile)
        
        # 存储结果
        times = []
        Pmax_open = []     # 无控制
        Pmax_synergy = []  # 协同控制
        Pmax_baseline = [] # 基准线
        vit_history = []
        fuel_history = []
        
        # 获取基准Pmax
        self.engine.run_cycle(base_condition)
        baseline_Pmax = self.engine.get_pmax()
        
        # 时间循环
        for t in np.arange(0, duration, dt):
            times.append(t)
            Pmax_baseline.append(baseline_Pmax)
            
            # 应用故障
            injector.apply_faults(t)
            
            # 无控制情况
            self.engine.run_cycle(base_condition)
            Pmax_open.append(self.engine.get_pmax())
            
            # 协同控制情况
            Y_measured = {
                'Pmax': self.engine.get_pmax(),
                'Pcomp': self.engine.get_pcomp(),
                'Texh': self.engine.get_exhaust_temp()
            }
            
            action = self.update(Y_measured, t)
            vit_history.append(action.vit_adjustment)
            fuel_history.append(action.fuel_adjustment)
            
            # 重新运行带控制的循环
            # (简化: 假设VIT调整直接影响Pmax)
            Pmax_controlled = Pmax_open[-1] - action.vit_adjustment * 2  # 简化模型
            Pmax_synergy.append(Pmax_controlled)
        
        # 清理
        injector.clear_all_faults()
        self.reset()
        
        return {
            'time': np.array(times),
            'Pmax_baseline': np.array(Pmax_baseline),
            'Pmax_open_loop': np.array(Pmax_open),
            'Pmax_synergy': np.array(Pmax_synergy),
            'vit_adjustment': np.array(vit_history),
            'fuel_adjustment': np.array(fuel_history),
        }
