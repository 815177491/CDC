# 零维船用柴油机热力学模型模块
"""
Marine Diesel Engine 0D Thermodynamic Model
============================================
包含几何模块、热力学求解器、燃烧模型与传热模型
"""

from .geometry import EngineGeometry
from .thermodynamics import ThermodynamicSolver
from .combustion import DoublieWiebeCombustion
from .heat_transfer import WoschniHeatTransfer
from .engine_model import MarineEngine0D, OperatingCondition
from .config import EngineConfig, DEFAULT_ENGINE_CONFIG

__all__ = [
    'EngineGeometry',
    'ThermodynamicSolver', 
    'DoublieWiebeCombustion',
    'WoschniHeatTransfer',
    'MarineEngine0D',
    'OperatingCondition',
    'EngineConfig',
    'DEFAULT_ENGINE_CONFIG'
]
