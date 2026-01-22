# 强化学习算法对比实验

## 概述

本模块实现了多种强化学习算法的对比实验框架，最终选择**TD-MPC2**作为主控制算法。

## 论文正式对比方法

| 算法    | 名称                            | 来源   | 年份 | 达标率    | 说明              |
| ------- | ------------------------------- | ------ | ---- | --------- | ----------------- |
| PID     | 传统PID控制                     | -      | -    | 0.5%      | 传统控制基线      |
| DQN     | Deep Q-Network                  | Nature | 2015 | ~70%      | 经典RL基线        |
| SAC     | Soft Actor-Critic               | ICML   | 2018 | 88.4%     | 最大熵框架        |
| TD-MPC2 | TD Model Predictive Control     | ICLR   | 2024 | **89.7%** | **★ 推荐方法**    |
| DPMD    | Diffusion Policy Mirror Descent | -      | 2025 | 86.4%     | 扩散策略+镜像下降 |

## 主控制算法: TD-MPC2

**TD-MPC2 (ICLR 2024)** 是本项目选择的最终强化学习方法，达标率89.7%。

核心特点:

- 世界模型学习: 学习环境动态模型
- TD学习: 时序差分更新Q值
- MPC规划: CEM交叉熵方法规划最优动作序列

## 文件结构

```
agents/
├── advanced_rl_algorithms.py  # 2024-2025新算法 (TD-MPC2, DPMD等)
├── rl_algorithms.py           # 基础RL算法 (DQN, SAC等，对比用)
├── multi_algo_control.py      # 多算法控制智能体
└── __init__.py                # 模块导出

experiments/
├── five_method_comparison.py  # 五方法对比实验
└── rl_comparison.py           # RL算法对比框架

run_gpu_comparison.py          # GPU加速对比实验入口
```

## 使用方法

### 1. 运行对比实验

```bash
# 快速验证 (100 episodes)
python run_gpu_comparison.py --quick

# 完整实验 (500 episodes)
python run_gpu_comparison.py --episodes 500
```

### 2. 在代码中使用

```python
from agents import get_algorithm, list_algorithms, ALGORITHM_INFO

# 查看可用算法
print(list_algorithms())
# ['DQN', 'DuelingDQN', 'PPO', 'SAC', 'TD3', 'DecisionTransformer', 'IQL']

# 创建算法实例
agent = get_algorithm('SAC', state_dim=10, action_dim=5, config={
    'lr': 1e-3,
    'gamma': 0.99,
    'batch_size': 64
})

# 选择动作
action = agent.select_action(state, explore=True)

# 存储经验
agent.buffer.push(state, action, reward, next_state, done)

# 更新网络
batch = agent.buffer.sample(64)
metrics = agent.update(batch)

# 保存/加载模型
agent.save('model.pt')
agent.load('model.pt')
```

### 4. 切换控制智能体算法

```python
from agents import create_control_agent

# 创建使用SAC算法的控制智能体
control_agent = create_control_agent(algorithm='SAC')

# 动态切换算法
control_agent.switch_algorithm('TD3')
```

## 算法详细说明

### 1. DQN (Deep Q-Network)

**论文**: Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015

经典的深度强化学习方法，使用神经网络近似Q函数。核心创新：

- 经验回放：打破样本相关性
- 目标网络：稳定训练过程

### 2. Dueling DQN

**论文**: Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", ICML 2016

将Q值分解为状态价值V(s)和优势函数A(s,a)：

```
Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
```

对于不需要精确区分动作的状态，学习更高效。

### 3. PPO (Proximal Policy Optimization)

**论文**: Schulman et al., "Proximal Policy Optimization Algorithms", 2017

策略梯度方法，使用clip目标函数限制策略更新幅度：

```
L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
```

稳定性好，超参数不敏感，广泛用于工业控制。

### 4. SAC (Soft Actor-Critic)

**论文**: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning", ICML 2018

最大熵强化学习框架，同时最大化回报和策略熵：

```
J(π) = E[Σ γ^t (r_t + α H(π(·|s_t)))]
```

特点：

- 自动温度调节
- 探索性好
- 样本效率高

### 5. TD3 (Twin Delayed DDPG)

**论文**: Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods", ICML 2018

针对DDPG过估计问题的三大改进：

1. **双Q网络**: 取最小值减少过估计
2. **延迟策略更新**: Actor更新频率低于Critic
3. **目标策略平滑**: 给目标动作加噪声

### 6. Decision Transformer

**论文**: Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling", NeurIPS 2021

创新地将RL问题转化为序列建模问题：

- 输入: (Return-to-go, State, Action) 序列
- 使用GPT架构的Transformer
- 通过条件生成预测动作
- 适合离线RL场景

### 7. IQL (Implicit Q-Learning)

**论文**: Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning", ICLR 2022

离线RL的新方法，通过期望分位数回归避免OOD动作问题：

```
L_V(ψ) = E[(τ · |Q(s,a) - V(s)|^2) if Q > V else (1-τ) · |Q(s,a) - V(s)|^2]
```

不需要显式策略约束，实现简单且效果好。

## 超参数建议

### 通用配置

```python
config = {
    'lr': 1e-3,           # 学习率
    'gamma': 0.99,        # 折扣因子
    'batch_size': 64,     # 批大小
    'buffer_size': 100000 # 经验池大小
}
```

### 算法特定配置

**DQN/Dueling DQN**:

```python
{
    'epsilon': 1.0,           # 初始探索率
    'epsilon_min': 0.05,      # 最小探索率
    'epsilon_decay': 0.995,   # 探索率衰减
    'target_update_freq': 100 # 目标网络更新频率
}
```

**PPO**:

```python
{
    'clip_epsilon': 0.2,   # PPO clip参数
    'gae_lambda': 0.95,    # GAE参数
    'entropy_coef': 0.01,  # 熵系数
    'ppo_epochs': 10       # 每次更新迭代次数
}
```

**SAC**:

```python
{
    'tau': 0.005,  # 软更新系数
    # 自动调节温度，无需手动设置alpha
}
```

**TD3**:

```python
{
    'tau': 0.005,           # 软更新系数
    'policy_delay': 2,      # 策略延迟更新
    'policy_noise': 0.2,    # 目标策略噪声
    'noise_clip': 0.5       # 噪声裁剪
}
```

## 实验结果

实验结果保存在 `experiment_results/` 目录：

- `experiment_summary.json`: 结果摘要
- `algorithm_comparison.png`: 对比图
- `detailed_analysis.png`: 详细分析图
- `experiment_report.txt`: 文字报告
- `{算法名}_training.csv`: 各算法训练数据

## 推荐选择

根据不同场景推荐：

| 场景     | 推荐算法 | 理由               |
| -------- | -------- | ------------------ |
| 快速原型 | DQN      | 实现简单，调试方便 |
| 工业部署 | PPO/SAC  | 稳定性好，性能优秀 |
| 连续控制 | SAC/TD3  | 专为连续动作设计   |
| 离线训练 | IQL/DT   | 无需与环境交互     |
| 样本受限 | SAC      | 样本效率最高       |
| 最新研究 | DT/IQL   | 2021-2022年方法    |

## 参考文献

1. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.

2. Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning. ICML 2016.

3. Schulman, J., et al. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

4. Haarnoja, T., et al. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning. ICML 2018.

5. Fujimoto, S., et al. (2018). Addressing function approximation error in actor-critic methods. ICML 2018.

6. Chen, L., et al. (2021). Decision transformer: Reinforcement learning via sequence modeling. NeurIPS 2021.

7. Kostrikov, I., et al. (2022). Offline reinforcement learning with implicit q-learning. ICLR 2022.
