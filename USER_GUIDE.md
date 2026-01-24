# 🎉 双智能体强化学习系统 - 用户使用指南

## 快速开始 (5 分钟)

### 1️⃣ 第一次运行 - 训练 MAPPO 模型

```bash
cd d:\my_github\CDC
python main.py --mode train-mappo --episodes 50 --save-dir models/test_mappo
```

**预期输出**:

```
======================================================================
启动双智能体训练 - 模式: MAPPO
======================================================================

开始训练 50 个回合...
[epoch 1/50] avg_reward=15.32, diag_acc=0.45, ctrl_rmse=0.023, violations=2
[epoch 2/50] avg_reward=18.91, diag_acc=0.52, ctrl_rmse=0.021, violations=1
...
训练完成!
模型已保存至: models/test_mappo
```

### 2️⃣ 评估训练好的模型

```bash
python main.py --mode eval-dual --model-dir models/test_mappo --num-episodes 20
```

**预期输出**:

```
诊断性能:
  - 准确率: 65.0%
  - 检测延迟: 3.50 步
  - 虚报率: 5.0%
  - 漏报率: 30.0%

控制性能:
  - Pmax RMSE: 0.0023
  - 违规次数: 1
  - 燃油经济性: 170.50

协同性能:
  - 端到端成功率: 62.5%
  - 诊断正确后的控制成功率: 80.0%
```

### 3️⃣ 可视化结果

```bash
python main.py --mode demo-dual --model-dir models/test_mappo
```

**输出**:

- 5 个演示回合的执行
- `models/test_mappo/coordination_response.png` 协同时序图

---

## 详细命令参考

### 训练命令

#### MAPPO 模式（推荐）

```bash
python main.py --mode train-mappo \
    --episodes 500 \
    --eval-interval 50 \
    --save-dir models/dual_mappo \
    --device cuda
```

**参数说明**:

- `--episodes`: 训练回合数 (default: 500, 建议 300-1000)
- `--eval-interval`: 每 N 个回合评估一次 (default: 50)
- `--save-dir`: 模型保存位置 (default: models/dual_agent)
- `--device`: 计算设备 cuda/cpu (default: cuda)

#### QMIX 模式（更快收敛）

```bash
python main.py --mode train-qmix --episodes 500 --save-dir models/dual_qmix
```

QMIX 适合：

- 有稀疏奖励的场景
- 追求更快的收敛速度
- 值分解更稳定的任务

#### 独立训练模式（对比基线）

```bash
python main.py --mode train-independent --episodes 500 --save-dir models/dual_ind
```

用于消融实验和性能对比。

### 评估命令

```bash
python main.py --mode eval-dual \
    --model-dir models/dual_mappo \
    --num-episodes 100 \
    --device cuda
```

**输出指标**:

| 类别 | 指标                   | 含义                         |
| ---- | ---------------------- | ---------------------------- |
| 诊断 | 准确率                 | 故障诊断正确率 (0-100%)      |
| 诊断 | 检测延迟               | 首次正确诊断的步数           |
| 诊断 | 虚报率                 | 错误报警概率                 |
| 诊断 | 漏报率                 | 漏诊概率                     |
| 控制 | Pmax RMSE              | 最大压力控制误差             |
| 控制 | 违规次数               | 超过安全限值的次数           |
| 控制 | 燃油经济性             | 燃油效率指标                 |
| 协同 | 端到端成功率           | 完整诊-控流程成功率          |
| 协同 | 正确诊断后的控制成功率 | 给定正确诊断，控制成功的概率 |

### 演示命令

```bash
python main.py --mode demo-dual --model-dir models/dual_mappo
```

运行 5 个演示回合，生成协同响应可视化：

- **4 轨迹图**:
  1. 故障信号 vs 诊断结果
  2. Pmax 实际值与目标值
  3. VIT 控制动作
  4. 燃油系数

---

## 配置文件调整

### 环境参数 (environments/dual_agent_env.py)

在 `EnvConfig` 中修改：

```python
# 随机故障注入参数
random_fault_prob=0.1,              # 多故障同时发生概率
fault_onset_range=(0, 50),          # 故障发生时刻 (% 回合)
fault_severity_range=(0.3, 1.0),    # 故障严重程度
fault_ramp_time_range=(0, 5),       # 故障斜坡时间 (秒)

# 状态空间参数
residual_window_size=10,            # 残差历史长度
```

### 奖励权重 (agents/rl_diagnosis_agent.py)

在 `RLDiagnosisAgent.__init__()` 中修改 `reward_weights`:

```python
reward_weights = {
    'accuracy_weight': 1.0,             # 准确率权重 (主要奖励)
    'delay_penalty': 0.1,               # 检测延迟惩罚
    'confidence_weight': 0.2,           # 置信度校准
    'control_feedback_weight': 0.3,     # 下游控制反馈
}
```

调优建议：

- 增大 `accuracy_weight` 强化准确率
- 增大 `control_feedback_weight` 强化诊-控协同
- 减小 `delay_penalty` 允许更慎重的诊断

### 训练超参数 (scripts/train_dual_agents.py)

在 `DualAgentTrainer.__init__()` 中修改：

```python
# SAC 诊断智能体参数
diag_lr=1e-3,                       # 学习率
diag_batch_size=64,                 # 批大小
diag_update_interval=1,             # 更新频率

# 控制智能体参数
ctrl_lr=5e-4,                       # 学习率 (通常小于诊断)
ctrl_batch_size=64,

# MAPPO/QMIX 参数
mappo_clip_ratio=0.2,               # PPO-Clip 系数
qmix_target_update=100,             # QMIX 目标网络更新周期
```

---

## 模型保存与加载

### 自动保存

训练过程中每 100 个回合自动保存：

```
models/dual_mappo/
├── checkpoint_100.pt
├── checkpoint_200.pt
├── ...
└── final_diag.pt, final_ctrl.pt    (最终模型)
```

### 手动加载

```python
from agents.rl_diagnosis_agent import RLDiagnosisAgent
from agents.rl_algorithms import SAC

diag_agent = RLDiagnosisAgent()
diag_agent.load('models/dual_mappo/final_diag.pt')

ctrl_agent = SAC(state_dim=10, action_dim=2)
ctrl_agent.load('models/dual_mappo/final_ctrl.pt')

# 进行推理
obs = env.reset()[0]
action, _ = diag_agent.select_action(obs, training=False)
```

---

## 常见问题解答

### Q1: 训练很慢，如何加速？

**A**:

```bash
# 减少评估频率
python main.py --mode train-mappo --eval-interval 100

# 使用 GPU
--device cuda

# 增加批大小 (修改 train_dual_agents.py)
diag_batch_size=128
```

### Q2: 诊断准确率不高，怎么办？

**A**: 按以下顺序尝试：

1. 增加训练回合: `--episodes 1000`
2. 增加 accuracy_weight: 在 rl_diagnosis_agent.py 修改
3. 增加 control_feedback_weight: 加强下游反馈
4. 检查环境难度是否太高

### Q3: MAPPO vs QMIX，选哪个？

**A**:

- **MAPPO**: 收敛快，稳定性好 (推荐初学者)
- **QMIX**: 更稳定，适合复杂任务

快速对比：

```bash
# 分别训练两个模型
python main.py --mode train-mappo --episodes 300 --save-dir models/mappo_test
python main.py --mode train-qmix --episodes 300 --save-dir models/qmix_test

# 评估对比
python main.py --mode eval-dual --model-dir models/mappo_test
python main.py --mode eval-dual --model-dir models/qmix_test
```

### Q4: 如何进行消融实验？

**A**: 修改 reward_weights 中的参数：

```python
# 实验1: 移除控制反馈
reward_weights['control_feedback_weight'] = 0.0

# 实验2: 只关注准确率
reward_weights['accuracy_weight'] = 2.0
reward_weights['control_feedback_weight'] = 0.0
```

### Q5: GPU 显存不足怎么办？

**A**:

```bash
# 使用 CPU
--device cpu

# 或减少批大小 (train_dual_agents.py)
diag_batch_size=32
ctrl_batch_size=32
```

### Q6: 评估结果如何解读？

**A**:

```
诊断准确率 60%       → 性能中等，需改进
检测延迟 3.5 步     → 响应快 (越小越好)
端到端成功率 70%    → 整体协同良好

Pmax RMSE 0.005    → 控制精度好
违规次数 2         → 安全性尚可
```

---

## 工作流程示例

### 场景: 从头训练到部署

```bash
# 1. 训练基础模型 (快速测试)
python main.py --mode train-mappo --episodes 100 --save-dir models/v1

# 2. 评估基础模型
python main.py --mode eval-dual --model-dir models/v1 --num-episodes 50

# 3. 根据结果调优参数并重新训练
# 修改 scripts/train_dual_agents.py 中的 reward_weights
python main.py --mode train-mappo --episodes 500 --save-dir models/v2

# 4. 再次评估
python main.py --mode eval-dual --model-dir models/v2 --num-episodes 100

# 5. 生成演示图表
python main.py --mode demo-dual --model-dir models/v2

# 6. 查看结果 (打开 models/v2/coordination_response.png)
```

### 场景: A/B 测试对比

```bash
# 训练 MAPPO
python main.py --mode train-mappo --episodes 500 --save-dir models/a_mappo

# 训练 QMIX
python main.py --mode train-qmix --episodes 500 --save-dir models/b_qmix

# 评估 A
echo "=== MAPPO 结果 ==="
python main.py --mode eval-dual --model-dir models/a_mappo --num-episodes 100

# 评估 B
echo "=== QMIX 结果 ==="
python main.py --mode eval-dual --model-dir models/b_qmix --num-episodes 100

# 对比结果选择更好的方案
```

---

## 输出文件说明

### 训练输出

```
models/dual_mappo/
├── training_log.txt              # 训练日志
├── checkpoint_100.pt             # 检查点
├── checkpoint_200.pt
├── ...
├── final_diag.pt                 # 最终诊断模型
└── final_ctrl.pt                 # 最终控制模型
```

### 评估输出

控制台输出格式化表格，包含上述 9 个关键指标。

### 演示输出

```
models/dual_mappo/
└── coordination_response.png      # 4 轨迹时序图
```

---

## 高级用法

### 自定义环境

```python
from environments.dual_agent_env import DualAgentEngineEnv, EnvConfig

# 创建自定义配置
config = EnvConfig(
    random_fault_prob=0.05,  # 更低的故障概率
    fault_severity_range=(0.5, 0.8),  # 中等严重程度
)

# 使用自定义配置
env = DualAgentEngineEnv(config=config)
```

### 自定义奖励函数

```python
# 在 rl_diagnosis_agent.py 中修改 compute_reward()
def compute_reward(self, is_correct, detection_step, ...):
    # 自定义公式
    r = 1.0 if is_correct else -1.0
    r -= 0.05 * detection_step  # 快速响应奖励
    return r
```

### 集成到现有系统

```python
from agents.rl_diagnosis_agent import create_rl_diagnosis_agent
from scripts.train_dual_agents import DualAgentTrainer

# 创建训练器
trainer = DualAgentTrainer(training_mode='mappo')

# 训练
trainer.train()

# 获取训练好的智能体
diag_agent = trainer.diag_agent
ctrl_agent = trainer.ctrl_agent

# 在生产系统中使用
diagnosis = diag_agent.diagnose(engine_state)
control_action = ctrl_agent.select_action(control_state)
```

---

## 性能优化提示

### 加快训练

1. 减少评估频率: `--eval-interval 100`
2. 使用 GPU: `--device cuda`
3. 增加批大小 (如果显存充足)
4. 使用 QMIX (收敛更快)

### 提高准确率

1. 增加训练回合: `--episodes 1000`
2. 调整奖励权重: 增加 `accuracy_weight`
3. 增加 control_feedback_weight (加强协同)
4. 使用更长的残差窗口

### 稳定性

1. 使用 MAPPO (比 QMIX 更稳定)
2. 增加评估间隔
3. 定期保存检查点
4. 监控训练曲线

---

## 支持与反馈

### 查看完整文档

- `DUAL_AGENT_QUICKSTART.md` - API 参考
- `DUAL_AGENT_SUMMARY.md` - 系统概览
- `IMPLEMENTATION_REPORT.md` - 技术细节

### 检查示例代码

- `scripts/train_dual_agents.py` 中的 `if __name__ == "__main__"` 部分
- `experiments/dual_agent_evaluation.py` 中的使用示例

---

**快速命令速查表**:

```bash
# 训练
python main.py --mode train-mappo --episodes 500
python main.py --mode train-qmix --episodes 500

# 评估
python main.py --mode eval-dual --model-dir models/dual_mappo

# 演示
python main.py --mode demo-dual --model-dir models/dual_mappo

# 查看帮助
python main.py --help
```

祝您使用愉快！🚀
