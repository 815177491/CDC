# 强化学习算法对比实验

## 概述

本模块实现了多种强化学习算法的对比实验框架，最终选择**TD-MPC2**作为主控制算法，配合**KAN+PINN混合诊断器**实现柴油机控诊协同。

## 论文正式对比方法

| 算法    | 名称                                 | 来源   | 年份 | 达标率    | 说明              |
| ------- | ------------------------------------ | ------ | ---- | --------- | ----------------- |
| PID     | 传统PID控制                          | -      | -    | 0.5%      | 传统控制基线      |
| DQN     | Deep Q-Network                       | Nature | 2015 | ~74%      | 经典RL基线        |
| SAC     | Soft Actor-Critic                    | ICML   | 2018 | 88.4%     | 最大熵框架        |
| TD-MPC2 | TD Model Predictive Control 2        | ICLR   | 2024 | **89.7%** | **★ 推荐方法**    |
| DPMD    | Diffusion Policy Mirror Descent      | arXiv  | 2025 | 86.4%     | 扩散策略+镜像下降 |

> **📊 可视化数据：** 五方法对比数据保存在 `visualization_data/five_method_accuracy.csv`

## 主控制算法: TD-MPC2 (ICLR 2024)

**TD-MPC2** 是本项目选择的最终强化学习方法，达标率89.7%，是五种方法中最优。

### 核心架构

TD-MPC2结合了三种关键技术：

1. **世界模型学习**：学习环境动态模型（潜在空间表示）
2. **时序差分学习**：高效的Q值估计
3. **CEM在线规划**：交叉熵方法优化动作序列

```
              ┌─────────────────────────────────────────────────┐
              │              TD-MPC2 控制器                      │
              ├─────────────────────────────────────────────────┤
              │                                                 │
状态 s ────►  │ ┌───────────┐    ┌────────────┐    ┌─────────┐ │
              │ │ 状态编码器 │───►│ 潜在动力学  │───►│ 奖励预测 ││ ────► 动作 a
              │ │  h=f(s)   │    │  h'=g(h,a) │    │  r=r(h) ││
              │ └───────────┘    └────────────┘    └─────────┘ │
              │       │                │                       │
              │       └───────┬────────┘                       │
              │               ▼                                │
              │         ┌───────────┐                          │
              │         │ CEM规划器  │  (多步horizon预测)        │
              │         └───────────┘                          │
              └─────────────────────────────────────────────────┘
```

### 世界模型训练损失

$$
\mathcal{L} = \mathcal{L}_{dynamics} + \mathcal{L}_{reward} + \mathcal{L}_{TD}
$$

- **动力学损失**：$\mathcal{L}_{dynamics} = \| g_\theta(h, a) - f_\theta(s') \|_2^2$
- **奖励预测损失**：$\mathcal{L}_{reward} = (r_\theta(h) - r)^2$
- **TD损失**：$\mathcal{L}_{TD} = (Q_\theta(s,a) - (r + \gamma \max_{a'} Q_{\bar\theta}(s', a')))^2$

> **📊 可视化数据：** 世界模型损失分解数据保存在 `visualization_data/training_process.csv`

### CEM规划流程

1. 初始化动作分布 $\mu^{(0)} = 0$, $\sigma^{(0)} = 2$
2. 采样N个动作序列
3. 使用世界模型rollout评估
4. 选择Top-K精英样本更新分布
5. 返回最优动作序列的第一步

> **📊 可视化数据：** Horizon效果对比数据保存在 `visualization_data/horizon_effect.csv`

## 诊断智能体: KAN+PINN混合诊断器

配合TD-MPC2控制器，诊断智能体采用**KAN (60%) + PINN (40%)** 混合架构。

### KAN诊断器

基于Kolmogorov-Arnold表示定理，使用可学习的B样条激活函数：

$$
f(x_1, \ldots, x_n) = \sum_{q=0}^{2n} \Phi_q \left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right)
$$

### PINN诊断器

将柴油机热力学方程作为物理约束嵌入神经网络：

- 压缩多变方程
- 维贝燃烧模型
- 能量守恒方程

### 混合投票

$$
\text{score}(c) = 0.6 \cdot p_{KAN}(c) + 0.4 \cdot p_{PINN}(c)
$$

> **📊 可视化数据：** 混合诊断器权重数据保存在 `visualization_data/classifier_weights.csv`

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

visualization_data/            # CSV数据 (用于Origin绑图)
├── training_process.csv       # TD-MPC2训练过程
├── five_method_learning_curves.csv  # 五方法学习曲线
├── simulation_results.csv     # 仿真结果
├── performance_metrics.csv    # 性能指标对比
└── ...                        # 其他数据文件

results/                       # 实验结果
├── five_method_summary.csv    # 五方法对比总结
└── comparison/                # 详细对比结果

run_gpu_comparison.py          # GPU加速对比实验入口
export_visualization_data.py   # CSV数据导出脚本
```

## 使用方法

### 1. 运行对比实验

```bash
# 快速验证 (100 episodes)
python run_gpu_comparison.py --quick

# 完整实验 (500 episodes)
python run_gpu_comparison.py --quick --episodes 500
```

### 2. 导出CSV数据 (用于Origin绑图)

```bash
python export_visualization_data.py
```

生成的CSV文件包括：
- `training_process.csv`: TD-MPC2训练过程（世界模型损失分解）
- `five_method_learning_curves.csv`: 五方法学习曲线对比
- `simulation_results.csv`: 仿真结果（五方法Pmax响应）
- `performance_metrics.csv`: 性能指标对比
- `step_response.csv`: 五方法阶跃响应
- `adaptive_threshold.csv`: KAN+PINN自适应阈值
- `confusion_matrix.csv`: 诊断混淆矩阵
- `roc_curve.csv`: ROC曲线数据

### 3. 在代码中使用TD-MPC2

```python
from agents import get_algorithm

# 创建TD-MPC2控制器
agent = get_algorithm('TDMPC2', state_dim=10, action_dim=45, config={
    'lr': 3e-4,
    'gamma': 0.99,
    'horizon': 3,
    'n_samples': 16,
    'n_elites': 4,
})

# 选择动作 (使用CEM规划)
action = agent.select_action(state, training=True)

# 更新世界模型
losses = agent.update(batch)
# losses包含: dynamics_loss, reward_loss, value_loss, total_loss
```

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
