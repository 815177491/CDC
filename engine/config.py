"""
发动机共享配置模块
==================
统一管理物理参数，确保引擎模型与PINN网络的一致性
"""

from dataclasses import dataclass


@dataclass
class EngineConfig:
    """
    发动机核心配置参数
    
    此配置被 MarineEngine0D 和 PINN 网络共享，
    确保物理约束与仿真环境的一致性。
    """
    # 几何参数
    bore: float = 0.620              # 气缸直径 [m]
    stroke: float = 2.658            # 活塞行程 [m]
    n_cylinders: int = 6             # 气缸数量
    compression_ratio: float = 13.5  # 有效压缩比
    con_rod_ratio: float = 4.0       # 连杆比
    
    # 热力学参数
    gamma: float = 1.35              # 比热比
    R: float = 287.0                 # 气体常数 [J/(kg·K)]
    
    # 基准值（健康状态）- 用于诊断
    Pmax_base: float = 150.0         # 基准最大压力 [bar]
    Pcomp_base: float = 120.0        # 基准压缩压力 [bar]
    Texh_base: float = 400.0         # 基准排温 [°C]
    
    # 物理约束权重 - 用于PINN训练
    lambda_physics: float = 0.1
    lambda_consistency: float = 0.05


# 全局默认配置实例
DEFAULT_ENGINE_CONFIG = EngineConfig()
