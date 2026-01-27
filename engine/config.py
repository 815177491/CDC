"""
发动机共享配置模块
==================
统一管理物理参数，确保引擎模型与PINN网络的一致性

注意: 此模块为兼容性保留，建议使用 config.global_config 中的 ENGINE_CONFIG
"""

# 从全局配置导入
from config import ENGINE_CONFIG, EngineConfig

# 为保持向后兼容性，保留原有的DEFAULT_ENGINE_CONFIG
DEFAULT_ENGINE_CONFIG = ENGINE_CONFIG
