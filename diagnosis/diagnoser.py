"""
故障诊断器
==========
基于残差分析的故障检测与识别
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
from collections import deque

from .fault_injector import FaultType


@dataclass
class DiagnosisResult:
    """诊断结果"""
    timestamp: float
    fault_detected: bool
    fault_type: FaultType
    confidence: float  # 置信度 (0-1)
    residuals: Dict[str, float]
    recommendation: str = ""


class DiagnosisState(Enum):
    """诊断状态"""
    HEALTHY = "healthy"
    WARNING = "warning"
    FAULT = "fault"
    CRITICAL = "critical"


class FaultDiagnoser:
    """
    故障诊断器
    
    采用残差分析法进行故障检测与识别:
    - 计算实际输出与模型预测的残差
    - 基于阈值和残差方向识别故障类型
    
    残差定义:
        r = Y_real - Y_model
    
    故障特征矩阵 (残差方向):
        | 故障类型     | r_Pmax | r_Pcomp | r_Texh |
        |-------------|--------|---------|--------|
        | 正时提前    |   +    |    0    |   -    |
        | 正时滞后    |   -    |    0    |   +    |
        | 气缸泄漏    |   -    |    -    |   +    |
        | 燃油不足    |   -    |    0    |   -    |
    """
    
    def __init__(self, engine):
        """
        初始化诊断器
        
        Args:
            engine: MarineEngine0D实例 (健康模型)
        """
        self.engine = engine
        
        # 残差阈值 (相对值)
        self.thresholds = {
            'Pmax': 0.03,      # 3% Pmax偏差触发警告
            'Pcomp': 0.02,     # 2% Pcomp偏差
            'Texh': 0.05,      # 5% 排温偏差
        }
        
        # 临界阈值
        self.critical_thresholds = {
            'Pmax': 0.08,      # 8% 临界
            'Pcomp': 0.05,
            'Texh': 0.10,
        }
        
        # 绝对安全限值
        self.safety_limits = {
            'Pmax': 190.0,     # bar, 最大爆压安全限
            'Texh': 450.0,     # °C, 最大排温
        }
        
        # 残差历史 (滑动窗口)
        self.window_size = 10
        self.residual_history: Dict[str, deque] = {
            'Pmax': deque(maxlen=self.window_size),
            'Pcomp': deque(maxlen=self.window_size),
            'Texh': deque(maxlen=self.window_size),
        }
        
        # 诊断历史
        self.diagnosis_history: List[DiagnosisResult] = []
        
        # 当前状态
        self.current_state = DiagnosisState.HEALTHY
        
        # 故障特征矩阵 (用于故障分类)
        # [r_Pmax方向, r_Pcomp方向, r_Texh方向]
        # +1 = 正残差 (实际>模型), -1 = 负残差, 0 = 无显著变化
        self.fault_signatures = {
            FaultType.INJECTION_TIMING: {
                'early': [+1, 0, -1],    # 正时提前
                'late': [-1, 0, +1],     # 正时滞后
            },
            FaultType.CYLINDER_LEAK: [-1, -1, +1],
            FaultType.FUEL_DEGRADATION: [-1, 0, -1],
            FaultType.INJECTOR_DRIFT: [-1, -1, 0],
        }
    
    def compute_residuals(self, Y_measured: Dict[str, float]) -> Dict[str, float]:
        """
        计算残差
        
        Args:
            Y_measured: 实测值 {'Pmax': ..., 'Pcomp': ..., 'Texh': ...}
            
        Returns:
            residuals: 残差字典 (相对值)
        """
        # 获取模型预测值
        Pmax_model = self.engine.get_pmax()
        Pcomp_model = self.engine.get_pcomp()
        Texh_model = self.engine.get_exhaust_temp()
        
        # 计算相对残差
        residuals = {}
        
        if 'Pmax' in Y_measured and Pmax_model > 0:
            residuals['Pmax'] = (Y_measured['Pmax'] - Pmax_model) / Pmax_model
        else:
            residuals['Pmax'] = 0.0
        
        if 'Pcomp' in Y_measured and Pcomp_model > 0:
            residuals['Pcomp'] = (Y_measured['Pcomp'] - Pcomp_model) / Pcomp_model
        else:
            residuals['Pcomp'] = 0.0
        
        if 'Texh' in Y_measured and Texh_model > 0:
            residuals['Texh'] = (Y_measured['Texh'] - Texh_model) / (Texh_model + 273.15)
        else:
            residuals['Texh'] = 0.0
        
        # 更新历史
        for key, value in residuals.items():
            if key in self.residual_history:
                self.residual_history[key].append(value)
        
        return residuals
    
    def get_smoothed_residuals(self) -> Dict[str, float]:
        """获取平滑后的残差 (滑动平均)"""
        smoothed = {}
        for key, history in self.residual_history.items():
            if len(history) > 0:
                smoothed[key] = np.mean(list(history))
            else:
                smoothed[key] = 0.0
        return smoothed
    
    def check_thresholds(self, residuals: Dict[str, float]) -> DiagnosisState:
        """
        检查残差是否超过阈值
        
        Returns:
            state: 诊断状态
        """
        max_severity = 0.0
        
        for key, r in residuals.items():
            abs_r = abs(r)
            threshold = self.thresholds.get(key, 0.05)
            critical = self.critical_thresholds.get(key, 0.10)
            
            if abs_r >= critical:
                max_severity = max(max_severity, 2.0)
            elif abs_r >= threshold:
                max_severity = max(max_severity, 1.0)
        
        if max_severity >= 2.0:
            return DiagnosisState.CRITICAL
        elif max_severity >= 1.0:
            return DiagnosisState.FAULT
        else:
            return DiagnosisState.HEALTHY
    
    def check_safety_limits(self, Y_measured: Dict[str, float]) -> Tuple[bool, str]:
        """
        检查安全限值
        
        Returns:
            (violation, message): 是否超限, 描述信息
        """
        violations = []
        
        if 'Pmax' in Y_measured:
            if Y_measured['Pmax'] > self.safety_limits['Pmax']:
                violations.append(
                    f"Pmax={Y_measured['Pmax']:.1f}bar 超过限值 "
                    f"{self.safety_limits['Pmax']}bar"
                )
        
        if 'Texh' in Y_measured:
            if Y_measured['Texh'] > self.safety_limits['Texh']:
                violations.append(
                    f"排温={Y_measured['Texh']:.0f}°C 超过限值 "
                    f"{self.safety_limits['Texh']}°C"
                )
        
        if violations:
            return True, "; ".join(violations)
        return False, ""
    
    def classify_fault(self, residuals: Dict[str, float]) -> Tuple[FaultType, float]:
        """
        基于残差方向分类故障类型
        
        Args:
            residuals: 残差字典
            
        Returns:
            (fault_type, confidence): 故障类型, 置信度
        """
        # 提取残差符号向量
        def get_sign(r, threshold=0.01):
            if r > threshold:
                return +1
            elif r < -threshold:
                return -1
            return 0
        
        r_vec = [
            get_sign(residuals.get('Pmax', 0)),
            get_sign(residuals.get('Pcomp', 0)),
            get_sign(residuals.get('Texh', 0))
        ]
        
        # 计算与各故障特征的匹配度
        best_match = FaultType.NONE
        best_score = 0.0
        
        for fault_type, signature in self.fault_signatures.items():
            if isinstance(signature, dict):
                # 处理有子类型的故障 (如正时提前/滞后)
                for sub_type, sig in signature.items():
                    score = self._match_signature(r_vec, sig)
                    if score > best_score:
                        best_score = score
                        best_match = fault_type
            else:
                score = self._match_signature(r_vec, signature)
                if score > best_score:
                    best_score = score
                    best_match = fault_type
        
        # 置信度: 匹配分数 + 残差幅值
        residual_magnitude = np.mean([abs(r) for r in residuals.values()])
        confidence = min(best_score * (1 + residual_magnitude * 5), 1.0)
        
        return best_match, confidence
    
    def _match_signature(self, r_vec: List[int], signature: List[int]) -> float:
        """计算残差向量与故障特征的匹配分数"""
        matches = sum(1 for r, s in zip(r_vec, signature) if r == s and s != 0)
        non_zero = sum(1 for s in signature if s != 0)
        
        if non_zero == 0:
            return 0.0
        return matches / non_zero
    
    def diagnose(self, Y_measured: Dict[str, float], 
                 timestamp: float = 0.0) -> DiagnosisResult:
        """
        执行故障诊断
        
        Args:
            Y_measured: 实测值
            timestamp: 时间戳
            
        Returns:
            result: 诊断结果
        """
        # 计算残差
        residuals = self.compute_residuals(Y_measured)
        
        # 检查安全限值
        safety_violation, safety_msg = self.check_safety_limits(Y_measured)
        
        # 检查阈值
        state = self.check_thresholds(residuals)
        
        # 故障分类
        fault_type, confidence = self.classify_fault(residuals)
        
        # 判断是否检测到故障
        fault_detected = (state != DiagnosisState.HEALTHY) or safety_violation
        
        # 生成建议
        recommendation = self._generate_recommendation(
            fault_type, state, safety_violation, safety_msg
        )
        
        result = DiagnosisResult(
            timestamp=timestamp,
            fault_detected=fault_detected,
            fault_type=fault_type if fault_detected else FaultType.NONE,
            confidence=confidence,
            residuals=residuals,
            recommendation=recommendation
        )
        
        # 更新状态
        self.current_state = DiagnosisState.CRITICAL if safety_violation else state
        self.diagnosis_history.append(result)
        
        return result
    
    def _generate_recommendation(self, fault_type: FaultType,
                                   state: DiagnosisState,
                                   safety_violation: bool,
                                   safety_msg: str) -> str:
        """生成处理建议"""
        if safety_violation:
            return f"⚠️ 安全警告: {safety_msg}. 建议立即降功或启动协同控制!"
        
        if state == DiagnosisState.CRITICAL:
            return f"🔴 临界故障: 检测到{fault_type.name}. 建议立即采取容错措施."
        
        if state == DiagnosisState.FAULT:
            recommendations = {
                FaultType.INJECTION_TIMING: "调整VIT(可变喷油正时)进行补偿",
                FaultType.CYLINDER_LEAK: "安排停机检修活塞环和缸套",
                FaultType.FUEL_DEGRADATION: "检查喷油器和燃油滤清器",
                FaultType.INJECTOR_DRIFT: "重新校准喷油器或更换",
            }
            action = recommendations.get(fault_type, "进一步排查故障原因")
            return f"🟡 故障警告: 检测到{fault_type.name}. 建议: {action}"
        
        return "✅ 系统正常运行"
    
    def get_state_indicator(self) -> Tuple[str, str]:
        """
        获取状态指示
        
        Returns:
            (color, text): 指示灯颜色, 状态文本
        """
        state_map = {
            DiagnosisState.HEALTHY: ('green', '正常'),
            DiagnosisState.WARNING: ('yellow', '警告'),
            DiagnosisState.FAULT: ('orange', '故障'),
            DiagnosisState.CRITICAL: ('red', '临界'),
        }
        return state_map.get(self.current_state, ('gray', '未知'))
    
    def reset(self):
        """重置诊断器状态"""
        for history in self.residual_history.values():
            history.clear()
        self.diagnosis_history.clear()
        self.current_state = DiagnosisState.HEALTHY
