"""
双智能体强化学习故障诊断与容错控制系统
==========================================
诊断智能体 + 控制智能体 并行协作

注意：使用延迟导入以避免在不需要时加载torch
"""

__all__ = [
    'EngineEnv',
    'CompositeFaultInjector', 
    'OperatingScheduler',
    'DiagnosticAgent',
    'ControlAgent'
]

def __getattr__(name):
    """延迟导入"""
    if name == 'EngineEnv':
        from .env import EngineEnv
        return EngineEnv
    elif name == 'CompositeFaultInjector':
        from .env import CompositeFaultInjector
        return CompositeFaultInjector
    elif name == 'OperatingScheduler':
        from .env import OperatingScheduler
        return OperatingScheduler
    elif name == 'DiagnosticAgent':
        from .agents import DiagnosticAgent
        return DiagnosticAgent
    elif name == 'ControlAgent':
        from .agents import ControlAgent
        return ControlAgent
    raise AttributeError(f"module 'marl' has no attribute '{name}'")
