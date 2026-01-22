"""
智能体模块
==========
控制-诊断双智能体架构

包含:
- DiagnosisAgent: 诊断智能体 (KAN为主 + PINN为辅，投票机制融合)
- ControlAgent: 控制智能体 (TD-MPC2强化学习策略，达标率89.7%)
- CoordinatorAgent: 协调智能体 (管理双智能体通信与冲突解决)
- MultiAlgorithmController: 多算法控制器 (支持多种RL算法)

诊断算法:
- KAN (MIT 2024): 主诊断器，可学习激活函数，可解释符号规则
- PINN (2024): 辅助诊断器，物理信息神经网络，热力学约束验证
- 混合诊断器: KAN权重60% + PINN权重40%，投票机制融合

强化学习算法 (对比实验):
- PID: 传统控制基线 (0.5%)
- DQN: Nature 2015 (对比用)
- SAC: ICML 2018 (88.4%)
- TD-MPC2: ICLR 2024 (89.7%) ★ 推荐
- DPMD: 2025 (86.4%)
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
