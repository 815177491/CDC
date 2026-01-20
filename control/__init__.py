# 协同控制模块
"""
Control-Diagnosis Synergy Module
================================
故障诊断与容错控制的闭环交互
"""

from .synergy_controller import SynergyController, ControlAction, ControlMode

__all__ = ['SynergyController', 'ControlAction', 'ControlMode']
