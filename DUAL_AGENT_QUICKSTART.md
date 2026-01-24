# 双智能体强化学习系统 - 快速入门

## 概述

本系统采用 **SAC 诊断智能体** + **强化学习控制** 的架构，支持三种多智能体协同训练算法：

- **MAPPO** (Multi-Agent PPO): 中央化评价，分布式演员
- **QMIX** (Q-Mix): 值分解 + 超网络混合
- **独立训练** (Independent): 各智能体独立学习

## 快速开始

### 1. 训练双智能体系统

#### MAPPO 模式

```bash
python main.py --mode train-mappo --episodes 500 --save-dir models/dual_mappo
```

#### QMIX 模式

```bash
python main.py --mode train-qmix --episodes 500 --save-dir models/dual_qmix
```

#### 独立训练模式

```bash
python main.py --mode train-independent --episodes 500 --save-dir models/dual_ind
```

**参数说明**:

- `--episodes`: 训练回合数 (default: 500)
- `--eval-interval`: 评估间隔 (default: 50)
- `--save-dir`: 模型保存目录 (default: models/dual_agent)
- `--device`: 计算设备 (default: cuda)

### 2. 评估训练好的模型

```bash
python main.py --mode eval-dual --model-dir models/dual_mappo --num-episodes 100
```

**输出指标**:

```
诊断性能:
  - 准确率: XX%
  - 检测延迟: XX 步
  - 虚报率: XX%
  - 漏报率: XX%

控制性能:
  - Pmax RMSE: XX.XXXX
  - 违规次数: XX
  - 燃油经济性: XX.XX

协同性能:
  - 端到端成功率: XX%
  - 诊断正确后的控制成功率: XX%
```

### 3. 交互式演示

运行训练好的模型并生成协同响应可视化:

```bash
python main.py --mode demo-dual --model-dir models/dual_mappo
```

**输出**:

- 5 个演示回合的运行
- 协同响应时序图: `models/dual_mappo/coordination_response.png`

## 系统架构

### 诊断智能体 (RLDiagnosisAgent - SAC)

**输入状态**:

- 12D 基础状态向量 (Pmax, Pcomp, 转速等)
- 10 步历史残差序列 (3×10 张量)

**输出动作**:

- 5 故障类型 × 4 置信度 = 20 个离散动作
- 解码为: `(故障类型, 置信度)`

**奖励函数**:

```
R_diag = 准确率奖励(±1.0)
       + 置信度校准(±0.2-0.4)
       + 检测延迟惩罚(-0.05~-0.5)
       + 下游控制反馈(+0.3×改进)
```

### 控制智能体 (SAC - 连续)

**输入状态**:

- 10D 状态向量

**输出动作**:

- 2D 连续动作: [VIT, 燃油系数]

**奖励函数**:

```
R_ctrl = Pmax 目标奖励
       + 违规惩罚
       + 控制平滑性
       + 燃油效率奖励
```

### 多智能体算法

#### MAPPO

- **架构**: 中央化评价家 (共用全局状态) + 分布式演员 (独立策略)
- **优化**: PPO-Clip for actors + MSE for critic
- **适用**: 对称或近似对称的多智能体任务

#### QMIX

- **架构**: 独立 Q 网络 + 超网络混合函数
- **约束**: 单调性约束 (确保价值分解有效)
- **适用**: 完全可观察的中心化控制任务

## 核心模块

### agents/rl_diagnosis_agent.py

```python
from agents.rl_diagnosis_agent import RLDiagnosisAgent, create_rl_diagnosis_agent

# 创建诊断智能体
diag_agent = create_rl_diagnosis_agent(device='cuda')

# 执行诊断
obs = env.reset()[0]  # 获取诊断观察
action, log_prob = diag_agent.select_action(obs, training=False)

# 解码动作
fault_type, confidence = diag_agent.decode_action(action)
```

### agents/multi_agent_algorithms.py

```python
from agents.multi_agent_algorithms import get_multi_agent_algorithm

# 创建 MAPPO
mappo = get_multi_agent_algorithm('mappo', diag_agent, ctrl_agent)

# 执行步骤
diag_action, ctrl_action = mappo.select_actions(diag_obs, ctrl_obs)

# 更新
mappo.update(batch)  # 自动处理 PPO-Clip 优化
```

### environments/dual_agent_env.py

```python
from environments import create_dual_agent_env

# 创建环境
env = create_dual_agent_env()

# 重置
(diag_obs, ctrl_obs) = env.reset()

# 步进
(diag_obs, ctrl_obs), (diag_reward, ctrl_reward), done, info = env.step(
    (diag_action, ctrl_action)
)

# 获取真实标签
ground_truth_fault = info['ground_truth_fault']
```

## 评估框架

### 诊断指标

- 准确率 (Accuracy)
- 精确率/召回率 (Per-class Precision/Recall)
- 混淆矩阵 (Confusion Matrix)
- 检测延迟 (Detection Delay) - 首次正确诊断的步数
- 虚报率/漏报率 (False Positive/Negative Rate)
- 置信度校准 (Confidence Calibration)

### 控制指标

- Pmax RMSE / MAE
- 违规次数 (Violation Count)
- 超调量 (Overshoot)
- 恢复时间 (Recovery Time)
- 燃油经济性 (Fuel Economy)
- 平滑性 (Smoothness)

### 协同指标

- 端到端成功率 (E2E Success Rate)
- 诊断正确时的控制成功率
- 协同度量 (Cooperation Metrics)
- 鲁棒性测试 (噪声容差、未见过条件、多故障)

## 可视化

### 协同响应时序图

显示 4 条轨迹:

1. 故障信号 vs 诊断结果
2. Pmax 跟踪
3. VIT 动作
4. 燃油系数

### 训练曲线

- 总奖励趋势
- 诊断/控制奖励分解
- 诊断准确率
- 检测延迟
- 违规次数

### 性能雷达图

5D 对比: 准确率 | 响应速度 | 安全性 | 经济性 | 鲁棒性

### 混淆矩阵

归一化的故障分类混淆矩阵

## 训练配置示例

```python
# 在 scripts/train_dual_agents.py 中修改
config = {
    'diag_reward_weights': {
        'accuracy_weight': 1.0,
        'delay_penalty': 0.1,
        'confidence_weight': 0.2,
        'control_feedback_weight': 0.3,
    },
    'env_config': {
        'random_fault_prob': 0.1,      # 多故障概率
        'fault_onset_range': (0, 50),  # 故障发生时刻范围 (%)
        'fault_severity_range': (0.3, 1.0),
        'fault_ramp_time_range': (0, 5),
    },
    'mappo_config': {
        'clip_ratio': 0.2,
        'learning_rate': 1e-3,
        'update_epochs': 3,
    },
    'qmix_config': {
        'learning_rate': 5e-4,
        'target_update_interval': 100,
    }
}
```

## 故障类型

系统支持以下 5 种故障:

1. `FUEL_PUMP_DEGRADATION` - 燃油泵降级
2. `INJECTION_TIMING_FAULT` - 喷射正时故障
3. `AIR_SCAVENGING_ISSUE` - 扫气问题
4. `COOLING_SYSTEM_FAILURE` - 冷却系故障
5. `TURBOCHARGER_DEGRADATION` - 增压器降级

## 常见问题

**Q: MAPPO vs QMIX, 选择哪个?**  
A: MAPPO 通常收敛更快、稳定性好；QMIX 更适合有稀疏奖励的场景

**Q: 如何加快训练?**  
A: 增大 `--eval-interval`, 使用 GPU (`--device cuda`), 减少验证频率

**Q: 为什么诊断准确率不高?**  
A: 检查 `reward_weights` 中 `control_feedback_weight` 是否过大，可能导致控制反馈淹没诊断信号

**Q: 如何进行消融研究?**  
A: 修改 `agents/rl_diagnosis_agent.py` 中的 `compute_reward()` 方法，逐项移除奖励分量

## 进阶: 自定义扩展

### 添加新故障类型

编辑 `diagnosis/fault_injector.py`:

```python
class FaultType(Enum):
    # ... existing faults ...
    MY_NEW_FAULT = 6

# 在 FaultInjector._get_fault_effects() 中添加效果
```

### 修改奖励设计

编辑 `agents/rl_diagnosis_agent.py` 的 `compute_reward()`:

```python
def compute_reward(self, ...):
    # 自定义多目标奖励
    r1 = accuracy_reward
    r2 = other_metric_reward
    return weighted_sum(r1, r2, ...)
```

### 集成新智能体算法

在 `agents/multi_agent_algorithms.py` 中添加新类，继承 `BaseMultiAgentAlgorithm`

## 参考文献

- SAC: "Soft Actor-Critic Algorithms and Applications" (Haarnoja et al., 2019)
- MAPPO: "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (Huang et al., 2021)
- QMIX: "QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent RL" (Rashid et al., 2020)

---

更多细节见项目根目录的 `README.md` 和 `docs/` 文件夹。
