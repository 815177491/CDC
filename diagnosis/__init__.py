# 故障诊断模块
"""
Fault Diagnosis Module
======================
基于多种方法的故障检测与识别

支持的诊断方法:
1. 规则基 + RandomForest (传统方法)
2. PINN (物理信息神经网络) - 嵌入热力学约束
3. KAN (Kolmogorov-Arnold Networks, 2024) - 可解释符号学习
4. Hybrid (混合诊断器) - KAN为主+PINN为辅，投票机制
"""

from .fault_injector import FaultInjector, FaultType, FaultProfile
from .diagnoser import FaultDiagnoser, DiagnosisResult, DiagnosisState

# 2024-2025年新诊断方法
from .pinn_diagnoser import PINNDiagnoser, PINNDiagnosisResult
from .kan_diagnoser import KANDiagnoser, KANDiagnosisResult
from .hybrid_diagnoser import HybridDiagnoser, HybridDiagnosisResult, VoteStrategy

__all__ = [
    # 传统方法
    'FaultInjector', 'FaultType', 'FaultProfile', 
    'FaultDiagnoser', 'DiagnosisResult', 'DiagnosisState',
    # PINN诊断
    'PINNDiagnoser', 'PINNDiagnosisResult',
    # KAN诊断 (2024)
    'KANDiagnoser', 'KANDiagnosisResult',
    # 混合诊断器 (KAN为主+PINN为辅)
    'HybridDiagnoser', 'HybridDiagnosisResult', 'VoteStrategy'
]
