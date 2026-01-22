"""
诊断智能体
==========
基于KAN+PINN混合诊断器的故障诊断智能体

算法说明:
- KAN (MIT 2024): 主诊断器，可学习激活函数，可解释性强
- PINN (2024): 辅助诊断器，物理信息网络，嵌入热力学约束
- 融合策略: 投票机制 (KAN权重60% + PINN权重40%)

创新点:
1. KAN可解释诊断: 自动提取符号化诊断规则
2. PINN物理验证: 利用压缩/燃烧/能量方程验证诊断结果
3. 投票融合: 两种方法一致时置信度高，不一致时加权决策
4. 自适应阈值学习: 基于在线统计动态更新诊断阈值
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
from collections import deque
import warnings

from .base_agent import Agent, AgentMessage, MessageType

# 尝试导入sklearn，如果失败则使用简化版本
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available, using rule-based classifier only")

import sys
sys.path.append('..')
from diagnosis.fault_injector import FaultType


class DiagnosisState(Enum):
    """诊断状态"""
    HEALTHY = "healthy"
    WARNING = "warning"
    FAULT = "fault"
    CRITICAL = "critical"


@dataclass
class DiagnosisResult:
    """诊断结果"""
    timestamp: float
    fault_detected: bool
    fault_type: FaultType
    confidence: float
    residuals: Dict[str, float]
    predicted_trend: Dict[str, float] = field(default_factory=dict)
    recommendation: str = ""
    diagnosis_state: DiagnosisState = DiagnosisState.HEALTHY


class AdaptiveThresholdLearner:
    """
    自适应阈值学习器
    
    基于在线统计动态更新诊断阈值:
    - 使用滑动窗口计算残差的均值和标准差
    - 阈值 = μ + k*σ (k为可配置的灵敏度系数)
    - 支持不同工况的阈值分层管理
    """
    
    def __init__(self, window_size: int = 100, sensitivity: float = 3.0):
        """
        Args:
            window_size: 滑动窗口大小
            sensitivity: 灵敏度系数 (标准差倍数)
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        
        # 各指标的残差历史
        self.residual_history: Dict[str, deque] = {
            'Pmax': deque(maxlen=window_size),
            'Pcomp': deque(maxlen=window_size),
            'Texh': deque(maxlen=window_size),
        }
        
        # 学习到的阈值
        self.learned_thresholds: Dict[str, float] = {
            'Pmax': 0.03,   # 初始值 3%
            'Pcomp': 0.02,
            'Texh': 0.05,
        }
        
        # 临界阈值 (警告阈值的2倍)
        self.critical_multiplier = 2.5
        
        # 学习率 (指数移动平均)
        self.alpha = 0.1
        
        # 统计量
        self.stats: Dict[str, Dict[str, float]] = {
            key: {'mean': 0.0, 'std': 0.01, 'count': 0}
            for key in self.residual_history.keys()
        }
    
    def update(self, residuals: Dict[str, float], is_healthy: bool = True) -> None:
        """
        更新阈值学习器
        
        Args:
            residuals: 当前残差
            is_healthy: 当前是否为健康状态 (仅健康数据用于阈值学习)
        """
        for key, value in residuals.items():
            if key not in self.residual_history:
                continue
            
            # 添加到历史
            self.residual_history[key].append(abs(value))
            
            # 仅使用健康状态数据更新阈值
            if is_healthy and len(self.residual_history[key]) >= 10:
                history = list(self.residual_history[key])
                
                # 计算统计量
                new_mean = np.mean(history)
                new_std = np.std(history) + 1e-6  # 避免除零
                
                # 指数移动平均更新
                old_stats = self.stats[key]
                old_stats['mean'] = (1 - self.alpha) * old_stats['mean'] + self.alpha * new_mean
                old_stats['std'] = (1 - self.alpha) * old_stats['std'] + self.alpha * new_std
                old_stats['count'] += 1
                
                # 更新阈值: μ + k*σ
                new_threshold = old_stats['mean'] + self.sensitivity * old_stats['std']
                
                # 限制阈值范围 (防止过度敏感或过度迟钝)
                min_threshold = 0.01  # 最小1%
                max_threshold = 0.15  # 最大15%
                new_threshold = np.clip(new_threshold, min_threshold, max_threshold)
                
                # 平滑更新
                self.learned_thresholds[key] = (
                    0.9 * self.learned_thresholds[key] + 0.1 * new_threshold
                )
    
    def get_thresholds(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        获取当前阈值
        
        Returns:
            (warning_thresholds, critical_thresholds)
        """
        warning = self.learned_thresholds.copy()
        critical = {k: v * self.critical_multiplier for k, v in warning.items()}
        return warning, critical
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        return {
            'thresholds': self.learned_thresholds.copy(),
            'stats': {k: v.copy() for k, v in self.stats.items()},
            'sample_counts': {k: len(v) for k, v in self.residual_history.items()}
        }


class EnsembleFaultClassifier:
    """
    集成故障分类器
    
    融合多种分类方法:
    1. 规则基分类器 (基于故障特征矩阵)
    2. RandomForest分类器 (数据驱动)
    3. 加权投票融合
    """
    
    def __init__(self, use_ml: bool = True):
        """
        Args:
            use_ml: 是否使用机器学习分类器
        """
        self.use_ml = use_ml and SKLEARN_AVAILABLE
        
        # 故障特征矩阵 (规则基)
        self.fault_signatures = {
            FaultType.INJECTION_TIMING: {
                'early': [+1, 0, -1],    # 正时提前: Pmax↑, Pcomp不变, Texh↓
                'late': [-1, 0, +1],     # 正时滞后: Pmax↓, Pcomp不变, Texh↑
            },
            FaultType.CYLINDER_LEAK: [-1, -1, +1],   # 泄漏: Pmax↓, Pcomp↓, Texh↑
            FaultType.FUEL_DEGRADATION: [-1, 0, -1], # 燃油问题: Pmax↓, Texh↓
            FaultType.INJECTOR_DRIFT: [-1, -1, 0],   # 喷油器漂移
        }
        
        # 机器学习分类器
        if self.use_ml:
            self.rf_classifier = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42,
                class_weight='balanced'
            )
            self.scaler = StandardScaler()
            self.is_fitted = False
        
        # 训练数据缓冲
        self.training_buffer: List[Tuple[np.ndarray, int]] = []
        self.min_samples_to_fit = 50
        
        # 分类器权重
        self.rule_weight = 0.6
        self.ml_weight = 0.4
    
    def _get_sign_vector(self, residuals: Dict[str, float], 
                         threshold: float = 0.01) -> List[int]:
        """提取残差符号向量"""
        def get_sign(r):
            if r > threshold:
                return +1
            elif r < -threshold:
                return -1
            return 0
        
        return [
            get_sign(residuals.get('Pmax', 0)),
            get_sign(residuals.get('Pcomp', 0)),
            get_sign(residuals.get('Texh', 0))
        ]
    
    def _rule_based_classify(self, residuals: Dict[str, float]) -> Tuple[FaultType, float]:
        """规则基分类"""
        r_vec = self._get_sign_vector(residuals)
        
        best_match = FaultType.NONE
        best_score = 0.0
        
        for fault_type, signature in self.fault_signatures.items():
            if isinstance(signature, dict):
                for sub_type, sig in signature.items():
                    score = self._match_score(r_vec, sig)
                    if score > best_score:
                        best_score = score
                        best_match = fault_type
            else:
                score = self._match_score(r_vec, signature)
                if score > best_score:
                    best_score = score
                    best_match = fault_type
        
        return best_match, best_score
    
    def _match_score(self, r_vec: List[int], signature: List[int]) -> float:
        """计算匹配分数"""
        matches = sum(1 for r, s in zip(r_vec, signature) if r == s and s != 0)
        non_zero = sum(1 for s in signature if s != 0)
        if non_zero == 0:
            return 0.0
        return matches / non_zero
    
    def _ml_classify(self, residuals: Dict[str, float]) -> Tuple[FaultType, float]:
        """机器学习分类"""
        if not self.use_ml or not self.is_fitted:
            return FaultType.NONE, 0.0
        
        # 构建特征向量
        features = np.array([[
            residuals.get('Pmax', 0),
            residuals.get('Pcomp', 0),
            residuals.get('Texh', 0),
            abs(residuals.get('Pmax', 0)),
            abs(residuals.get('Pcomp', 0)),
            abs(residuals.get('Texh', 0)),
        ]])
        
        # 标准化
        features_scaled = self.scaler.transform(features)
        
        # 预测
        pred_class = self.rf_classifier.predict(features_scaled)[0]
        pred_proba = np.max(self.rf_classifier.predict_proba(features_scaled))
        
        # 转换为FaultType
        fault_type = self._int_to_fault_type(pred_class)
        
        return fault_type, pred_proba
    
    def classify(self, residuals: Dict[str, float]) -> Tuple[FaultType, float]:
        """
        集成分类
        
        Returns:
            (fault_type, confidence)
        """
        # 规则基分类
        rule_type, rule_conf = self._rule_based_classify(residuals)
        
        # 机器学习分类
        if self.use_ml and self.is_fitted:
            ml_type, ml_conf = self._ml_classify(residuals)
            
            # 加权投票
            if rule_type == ml_type:
                # 一致：置信度提升
                final_type = rule_type
                final_conf = min(1.0, (self.rule_weight * rule_conf + 
                                       self.ml_weight * ml_conf) * 1.2)
            else:
                # 不一致：选择置信度高的
                if rule_conf * self.rule_weight > ml_conf * self.ml_weight:
                    final_type = rule_type
                    final_conf = rule_conf * 0.8  # 降低置信度
                else:
                    final_type = ml_type
                    final_conf = ml_conf * 0.8
        else:
            final_type = rule_type
            final_conf = rule_conf
        
        return final_type, final_conf
    
    def add_training_sample(self, residuals: Dict[str, float], 
                           fault_type: FaultType) -> None:
        """添加训练样本"""
        features = np.array([
            residuals.get('Pmax', 0),
            residuals.get('Pcomp', 0),
            residuals.get('Texh', 0),
            abs(residuals.get('Pmax', 0)),
            abs(residuals.get('Pcomp', 0)),
            abs(residuals.get('Texh', 0)),
        ])
        
        label = self._fault_type_to_int(fault_type)
        self.training_buffer.append((features, label))
        
        # 自动训练
        if len(self.training_buffer) >= self.min_samples_to_fit:
            self.fit()
    
    def fit(self) -> None:
        """训练机器学习分类器"""
        if not self.use_ml or len(self.training_buffer) < self.min_samples_to_fit:
            return
        
        X = np.array([x[0] for x in self.training_buffer])
        y = np.array([x[1] for x in self.training_buffer])
        
        # 检查类别数量
        if len(np.unique(y)) < 2:
            return
        
        # 标准化
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # 训练
        self.rf_classifier.fit(X_scaled, y)
        self.is_fitted = True
    
    def _fault_type_to_int(self, fault_type: FaultType) -> int:
        """故障类型转整数"""
        mapping = {
            FaultType.NONE: 0,
            FaultType.INJECTION_TIMING: 1,
            FaultType.CYLINDER_LEAK: 2,
            FaultType.FUEL_DEGRADATION: 3,
            FaultType.INJECTOR_DRIFT: 4,
        }
        return mapping.get(fault_type, 0)
    
    def _int_to_fault_type(self, value: int) -> FaultType:
        """整数转故障类型"""
        mapping = {
            0: FaultType.NONE,
            1: FaultType.INJECTION_TIMING,
            2: FaultType.CYLINDER_LEAK,
            3: FaultType.FUEL_DEGRADATION,
            4: FaultType.INJECTOR_DRIFT,
        }
        return mapping.get(value, FaultType.NONE)


class TrendPredictor:
    """
    故障趋势预测器
    
    基于残差序列预测未来趋势:
    - 使用简单的线性回归预测
    - 可扩展为LSTM等深度学习方法
    """
    
    def __init__(self, history_length: int = 20, predict_horizon: int = 5):
        """
        Args:
            history_length: 历史窗口长度
            predict_horizon: 预测时域
        """
        self.history_length = history_length
        self.predict_horizon = predict_horizon
        
        self.residual_history: Dict[str, deque] = {
            'Pmax': deque(maxlen=history_length),
            'Pcomp': deque(maxlen=history_length),
            'Texh': deque(maxlen=history_length),
        }
    
    def update(self, residuals: Dict[str, float]) -> None:
        """更新历史"""
        for key, value in residuals.items():
            if key in self.residual_history:
                self.residual_history[key].append(value)
    
    def predict(self) -> Dict[str, float]:
        """
        预测未来残差趋势
        
        Returns:
            predicted_residuals: 预测的未来残差值
        """
        predictions = {}
        
        for key, history in self.residual_history.items():
            if len(history) < 5:
                predictions[key] = 0.0
                continue
            
            # 简单线性回归
            y = np.array(list(history))
            x = np.arange(len(y))
            
            # 最小二乘拟合
            A = np.vstack([x, np.ones(len(x))]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            
            # 预测未来值
            future_x = len(y) + self.predict_horizon
            predictions[key] = slope * future_x + intercept
        
        return predictions
    
    def get_trend_direction(self) -> Dict[str, str]:
        """获取趋势方向"""
        directions = {}
        
        for key, history in self.residual_history.items():
            if len(history) < 5:
                directions[key] = 'stable'
                continue
            
            y = np.array(list(history))
            
            # 计算斜率
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            
            if abs(slope) < 0.001:
                directions[key] = 'stable'
            elif slope > 0:
                directions[key] = 'increasing'
            else:
                directions[key] = 'decreasing'
        
        return directions


class DiagnosisAgent(Agent):
    """
    诊断智能体
    
    整合KAN+PINN混合诊断、自适应阈值学习和趋势预测
    
    诊断方法:
    - KAN (主): MIT 2024，可学习激活函数，符号规则提取
    - PINN (辅): 物理信息神经网络，热力学约束验证
    - 融合策略: 投票机制 (KAN 60% + PINN 40%)
    """
    
    def __init__(self, engine, name: str = "DiagnosisAgent", use_hybrid: bool = True):
        """
        Args:
            engine: 发动机模型
            name: 智能体名称
            use_hybrid: 是否使用KAN+PINN混合诊断器
        """
        super().__init__(name=name, engine=engine)
        
        # 尝试导入混合诊断器
        self.use_hybrid = use_hybrid
        self.hybrid_diagnoser = None
        
        if use_hybrid:
            try:
                from diagnosis import HybridDiagnoser
                self.hybrid_diagnoser = HybridDiagnoser({
                    'kan_weight': 0.6,
                    'pinn_weight': 0.4,
                    'strategy': 'weighted'
                })
                print("[DiagnosisAgent] 使用KAN+PINN混合诊断器")
            except ImportError:
                print("[DiagnosisAgent] 混合诊断器不可用，回退到规则分类器")
                self.use_hybrid = False
        
        # 子模块 (阈值学习和趋势预测仍保留)
        self.threshold_learner = AdaptiveThresholdLearner(
            window_size=100,
            sensitivity=3.0
        )
        
        # 规则分类器作为后备
        self.classifier = EnsembleFaultClassifier(use_ml=False)  # 不使用RandomForest
        
        self.trend_predictor = TrendPredictor(
            history_length=20,
            predict_horizon=5
        )
        
        # 安全限值
        self.safety_limits = {
            'Pmax': 190.0,   # bar
            'Texh': 450.0,   # °C
        }
        
        # 当前状态
        self.current_state = DiagnosisState.HEALTHY
        
        # 诊断历史
        self.diagnosis_history: List[DiagnosisResult] = []
        
        # 性能指标
        self.state.performance_metrics = {
            'total_diagnoses': 0,
            'faults_detected': 0,
            'false_alarms': 0,
            'threshold_updates': 0,
        }
    
    def perceive(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        感知：计算残差
        
        Args:
            observation: {'Pmax': ..., 'Pcomp': ..., 'Texh': ...}
        """
        # 获取模型预测值
        Pmax_model = self.engine.get_pmax()
        Pcomp_model = self.engine.get_pcomp()
        Texh_model = self.engine.get_exhaust_temp()
        
        # 计算相对残差
        residuals = {}
        
        if 'Pmax' in observation and Pmax_model > 0:
            residuals['Pmax'] = (observation['Pmax'] - Pmax_model) / Pmax_model
        else:
            residuals['Pmax'] = 0.0
        
        if 'Pcomp' in observation and Pcomp_model > 0:
            residuals['Pcomp'] = (observation['Pcomp'] - Pcomp_model) / Pcomp_model
        else:
            residuals['Pcomp'] = 0.0
        
        if 'Texh' in observation and Texh_model > 0:
            residuals['Texh'] = (observation['Texh'] - Texh_model) / (Texh_model + 273.15)
        else:
            residuals['Texh'] = 0.0
        
        # 更新趋势预测器
        self.trend_predictor.update(residuals)
        
        return {
            'residuals': residuals,
            'raw_observation': observation,
            'model_predictions': {
                'Pmax': Pmax_model,
                'Pcomp': Pcomp_model,
                'Texh': Texh_model,
            }
        }
    
    def decide(self, perception: Dict[str, Any]) -> DiagnosisResult:
        """
        决策：故障分类和状态判定
        
        使用KAN+PINN混合诊断器（如可用），否则使用规则分类器
        """
        residuals = perception['residuals']
        raw_obs = perception['raw_observation']
        
        # 获取自适应阈值
        warning_thresholds, critical_thresholds = self.threshold_learner.get_thresholds()
        
        # 检查阈值
        state = self._check_thresholds(residuals, warning_thresholds, critical_thresholds)
        
        # 检查安全限值
        safety_violation, safety_msg = self._check_safety_limits(raw_obs)
        
        # 故障分类 - 优先使用KAN+PINN混合诊断器
        if self.use_hybrid and self.hybrid_diagnoser is not None:
            try:
                # 构建特征向量
                import numpy as np
                features = np.array([
                    raw_obs.get('rpm', 80) / 100,           # 归一化转速
                    raw_obs.get('load', 0.75),              # 负荷
                    raw_obs.get('timing', 0) / 20,          # 正时
                    raw_obs.get('boost', 3.0) / 5,          # 增压
                    raw_obs.get('T_in', 320) / 400,         # 进气温度
                    raw_obs.get('P_amb', 1.0),              # 环境压力
                    1.0,                                     # 燃油品质
                    raw_obs.get('run_hours', 0) / 10000     # 运行时间
                ])
                
                hybrid_result = self.hybrid_diagnoser.diagnose(features)
                fault_type = hybrid_result.fault_type
                confidence = hybrid_result.confidence
                
                # 记录混合诊断信息
                self._last_hybrid_result = hybrid_result
                
            except Exception as e:
                # 混合诊断失败，回退到规则分类器
                fault_type, confidence = self.classifier.classify(residuals)
        else:
            # 使用规则分类器
            fault_type, confidence = self.classifier.classify(residuals)
        
        # 趋势预测
        predicted_trend = self.trend_predictor.predict()
        
        # 综合判断
        fault_detected = (state != DiagnosisState.HEALTHY) or safety_violation
        
        if safety_violation:
            state = DiagnosisState.CRITICAL
        
        # 生成建议
        recommendation = self._generate_recommendation(
            fault_type, state, safety_violation, safety_msg, predicted_trend
        )
        
        result = DiagnosisResult(
            timestamp=self.state.last_update,
            fault_detected=fault_detected,
            fault_type=fault_type if fault_detected else FaultType.NONE,
            confidence=confidence,
            residuals=residuals,
            predicted_trend=predicted_trend,
            recommendation=recommendation,
            diagnosis_state=state
        )
        
        return result
    
    def act(self, decision: DiagnosisResult) -> Dict[str, Any]:
        """
        执行：更新状态、发送消息、在线学习
        """
        # 更新状态
        self.current_state = decision.diagnosis_state
        self.diagnosis_history.append(decision)
        
        # 更新阈值学习器
        is_healthy = not decision.fault_detected
        self.threshold_learner.update(decision.residuals, is_healthy)
        
        # 更新统计
        self.state.performance_metrics['total_diagnoses'] += 1
        if decision.fault_detected:
            self.state.performance_metrics['faults_detected'] += 1
            
            # 添加训练样本
            self.classifier.add_training_sample(
                decision.residuals, 
                decision.fault_type
            )
        
        # 发送诊断结果消息
        self.send_message(
            msg_type=MessageType.DIAGNOSIS_RESULT,
            receiver="ControlAgent",
            payload={
                'result': decision,
                'thresholds': self.threshold_learner.get_thresholds()[0],
            },
            priority=8 if decision.fault_detected else 5
        )
        
        # 如果检测到故障，发送警报
        if decision.fault_detected:
            self.send_message(
                msg_type=MessageType.FAULT_ALERT,
                receiver=None,  # 广播
                payload={
                    'fault_type': decision.fault_type.name,
                    'confidence': decision.confidence,
                    'recommendation': decision.recommendation,
                },
                priority=10
            )
        
        return {
            'diagnosis': decision,
            'state': self.current_state,
            'messages_sent': len(self.outbox)
        }
    
    def _check_thresholds(self, residuals: Dict[str, float],
                          warning: Dict[str, float],
                          critical: Dict[str, float]) -> DiagnosisState:
        """检查残差是否超过阈值"""
        max_severity = 0
        
        for key, r in residuals.items():
            abs_r = abs(r)
            warn_thresh = warning.get(key, 0.05)
            crit_thresh = critical.get(key, 0.10)
            
            if abs_r >= crit_thresh:
                max_severity = max(max_severity, 2)
            elif abs_r >= warn_thresh:
                max_severity = max(max_severity, 1)
        
        if max_severity >= 2:
            return DiagnosisState.CRITICAL
        elif max_severity >= 1:
            return DiagnosisState.FAULT
        return DiagnosisState.HEALTHY
    
    def _check_safety_limits(self, Y_measured: Dict[str, float]) -> Tuple[bool, str]:
        """检查安全限值"""
        violations = []
        
        if 'Pmax' in Y_measured:
            if Y_measured['Pmax'] > self.safety_limits['Pmax']:
                violations.append(f"Pmax={Y_measured['Pmax']:.1f}bar 超过限值")
        
        if 'Texh' in Y_measured:
            if Y_measured['Texh'] > self.safety_limits['Texh']:
                violations.append(f"排温={Y_measured['Texh']:.0f}°C 超过限值")
        
        if violations:
            return True, "; ".join(violations)
        return False, ""
    
    def _generate_recommendation(self, fault_type: FaultType,
                                  state: DiagnosisState,
                                  safety_violation: bool,
                                  safety_msg: str,
                                  predicted_trend: Dict[str, float]) -> str:
        """生成处理建议"""
        if safety_violation:
            return f"⚠️ 安全警告: {safety_msg}. 建议立即启动协同控制!"
        
        # 检查趋势
        trend_warning = ""
        for key, pred_value in predicted_trend.items():
            if abs(pred_value) > 0.1:  # 预测残差超过10%
                trend_warning = f"⚡ 趋势预警: {key}残差呈恶化趋势"
                break
        
        if state == DiagnosisState.CRITICAL:
            return f"🔴 临界故障: {fault_type.name}. {trend_warning}"
        
        if state == DiagnosisState.FAULT:
            recommendations = {
                FaultType.INJECTION_TIMING: "调整VIT进行补偿",
                FaultType.CYLINDER_LEAK: "安排停机检修",
                FaultType.FUEL_DEGRADATION: "检查喷油器和燃油系统",
                FaultType.INJECTOR_DRIFT: "重新校准喷油器",
            }
            action = recommendations.get(fault_type, "进一步排查")
            return f"🟡 故障警告: {fault_type.name}. 建议: {action}. {trend_warning}"
        
        if trend_warning:
            return f"🟢 当前正常. {trend_warning}"
        
        return "✅ 系统正常运行"
    
    def diagnose(self, Y_measured: Dict[str, float], 
                 timestamp: float = 0.0) -> DiagnosisResult:
        """
        便捷接口：执行完整诊断流程
        
        Args:
            Y_measured: 测量值
            timestamp: 时间戳
            
        Returns:
            DiagnosisResult
        """
        result = self.step(Y_measured, timestamp)
        return result['diagnosis']
    
    def get_learned_thresholds(self) -> Dict[str, float]:
        """获取学习到的阈值"""
        return self.threshold_learner.learned_thresholds.copy()
    
    def get_classifier_status(self) -> Dict[str, Any]:
        """获取分类器状态"""
        return {
            'use_ml': self.classifier.use_ml,
            'is_fitted': self.classifier.is_fitted if self.classifier.use_ml else False,
            'training_samples': len(self.classifier.training_buffer),
        }
    
    def reset(self) -> None:
        """重置智能体"""
        super().reset()
        self.current_state = DiagnosisState.HEALTHY
        self.diagnosis_history.clear()
