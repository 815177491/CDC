"""
智能体模块
==========
控制-诊断双智能体架构

包含:
- DiagnosisAgent: 诊断智能体 (自适应阈值 + 集成分类器)
- ControlAgent: 控制智能体 (DQN强化学习策略)
- CoordinatorAgent: 协调智能体 (管理双智能体通信与冲突解决)
- MultiAlgorithmController: 多算法控制器 (支持7种RL算法)

支持的RL算法 (2026年更新):
基础算法 (2015-2022):
- DQN (Nature 2015)
- Dueling DQN (ICML 2016)
- PPO (OpenAI 2017)
- SAC (ICML 2018)
- TD3 (ICML 2018)
- Decision Transformer (NeurIPS 2021)
- IQL (ICLR 2022)

2024-2025最新算法 (GPU加速):
- Diffusion Policy (RSS/CoRL 2024)
- TD-MPC2 (ICLR 2024)
- Mamba Policy (2025)
- DPMD (2025)
"""

from .base_agent import Agent, AgentMessage, MessageType
from .diagnosis_agent import DiagnosisAgent
from .control_agent import ControlAgent
from .coordinator import CoordinatorAgent

# 多算法支持
try:
    from .rl_algorithms import (
        get_algorithm, list_algorithms, ALGORITHM_INFO,
        DQN, DuelingDQN, PPO, SAC, TD3, DecisionTransformer, IQL
    )
    from .multi_algo_control import (
        MultiAlgorithmController,
        AdaptiveMultiAlgorithmControlAgent,
        create_control_agent,
        print_available_algorithms
    )
    RL_ALGORITHMS_AVAILABLE = True
except ImportError:
    RL_ALGORITHMS_AVAILABLE = False

# 2024-2025新算法支持
try:
    from .advanced_rl_algorithms import (
        get_advanced_algorithm, list_advanced_algorithms, 
        ADVANCED_ALGORITHM_INFO, print_advanced_algorithms,
        DiffusionPolicy, TDMPC2, MambaPolicy, DPMD
    )
    ADVANCED_RL_AVAILABLE = True
except ImportError:
    ADVANCED_RL_AVAILABLE = False

__all__ = [
    'Agent',
    'AgentMessage', 
    'MessageType',
    'DiagnosisAgent',
    'ControlAgent',
    'CoordinatorAgent',
    # 多算法支持
    'get_algorithm',
    'list_algorithms',
    'ALGORITHM_INFO',
    'MultiAlgorithmController',
    'AdaptiveMultiAlgorithmControlAgent',
    'create_control_agent',
    'print_available_algorithms',
    'RL_ALGORITHMS_AVAILABLE',
    # 2024-2025新算法
    'get_advanced_algorithm',
    'list_advanced_algorithms',
    'ADVANCED_ALGORITHM_INFO',
    'print_advanced_algorithms',
    'ADVANCED_RL_AVAILABLE',
]
