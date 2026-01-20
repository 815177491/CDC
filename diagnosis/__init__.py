# 故障诊断模块
"""
Fault Diagnosis Module
======================
基于残差分析的故障检测与识别
"""

from .fault_injector import FaultInjector, FaultType, FaultProfile
from .diagnoser import FaultDiagnoser, DiagnosisResult, DiagnosisState

__all__ = ['FaultInjector', 'FaultType', 'FaultProfile', 
           'FaultDiagnoser', 'DiagnosisResult', 'DiagnosisState']
