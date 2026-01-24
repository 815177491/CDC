# 📑 双智能体强化学习系统 - 文件索引与导航

## 🗂️ 项目文件结构概览

```
CDC/
├── 📄 核心模块 (新增)
│   ├── agents/
│   │   ├── rl_diagnosis_agent.py       (852 L) ★ SAC 诊断智能体
│   │   └── multi_agent_algorithms.py   (1131 L) ★ MAPPO & QMIX
│   ├── environments/
│   │   ├── dual_agent_env.py           (843 L) ★ 双智能体环境
│   │   └── __init__.py                 ★ 导出函数
│   ├── scripts/
│   │   └── train_dual_agents.py        (667 L) ★ 训练框架
│   ├── experiments/
│   │   └── dual_agent_evaluation.py    (676 L) ★ 评估系统
│   └── visualization/
│       └── dual_agent_plots.py         (419 L) ★ 可视化工具
│
├── 📚 文档 (新增)
│   ├── USER_GUIDE.md                   ★★★ 开始阅读 ← 用户指南
│   ├── DUAL_AGENT_QUICKSTART.md        ★★  API 参考与配置
│   ├── DUAL_AGENT_SUMMARY.md           ★   系统总结
│   ├── IMPLEMENTATION_REPORT.md        ★   技术细节
│   └── README.md → 指向 USER_GUIDE.md
│
├── 🚀 入口点 (更新)
│   └── main.py (687 L)                  ★ 支持 6 种新命令
│
├── 📊 模型输出目录
│   └── models/
│       ├── dual_mappo/
│       ├── dual_qmix/
│       └── dual_ind/
│
└── 🔧 配置与工具
    ├── calibration_data.csv
    ├── calibrated_params.json
    └── requirements.txt
```

---

## 📖 快速导航指南

### 🌟 第一次使用？这样做：

**第 1 步** → 阅读 [`USER_GUIDE.md`](USER_GUIDE.md)

- ⏱️ 预计 10 分钟
- 包含快速开始、命令参考、常见问题
- 有 6 个实战工作流示例

**第 2 步** → 运行第一个训练

```bash
python main.py --mode train-mappo --episodes 50
```

**第 3 步** → 查看结果

```bash
python main.py --mode eval-dual --model-dir models/dual_agent
```

---

## 📚 完整文档对照表

| 文档                         | 适用人群 | 内容                           | 阅读时间 |
| ---------------------------- | -------- | ------------------------------ | -------- |
| **USER_GUIDE.md**            | 所有用户 | 快速开始、命令、配置、FAQ      | 15 min   |
| **DUAL_AGENT_QUICKSTART.md** | API 用户 | 详细 API、模块用法、扩展方法   | 30 min   |
| **DUAL_AGENT_SUMMARY.md**    | 技术管理 | 系统概览、关键特性、预期性能   | 20 min   |
| **IMPLEMENTATION_REPORT.md** | 开发者   | 技术深度、验收清单、源代码定位 | 25 min   |

### 读者路线图

```
非技术用户
    ↓
[USER_GUIDE.md] → 快速上手，完成基本任务
    ↓
如需进阶
    ↓
[DUAL_AGENT_QUICKSTART.md] → 学习 API，定制配置
    ↓
需要深入理解
    ↓
[IMPLEMENTATION_REPORT.md] → 理解底层设计
    ↓
需要系统认识
    ↓
[DUAL_AGENT_SUMMARY.md] → 了解架构与特性
```

---

## 🎯 按任务查找

### 我想...

#### 🚀 快速开始训练

→ [`USER_GUIDE.md`](USER_GUIDE.md) - "快速开始" 部分

```bash
python main.py --mode train-mappo --episodes 50
```

#### 📊 查看性能指标

→ [`USER_GUIDE.md`](USER_GUIDE.md) - "评估命令" 部分

```bash
python main.py --mode eval-dual --model-dir models/dual_mappo
```

#### 🔧 调整训练参数

→ [`DUAL_AGENT_QUICKSTART.md`](DUAL_AGENT_QUICKSTART.md) - "训练配置示例"
或 [`USER_GUIDE.md`](USER_GUIDE.md) - "配置文件调整"

#### 📈 理解算法原理

→ [`DUAL_AGENT_SUMMARY.md`](DUAL_AGENT_SUMMARY.md) - "系统架构"
或 [`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md) - "技术深度验证"

#### 💻 进行 Python 集成

→ [`DUAL_AGENT_QUICKSTART.md`](DUAL_AGENT_QUICKSTART.md) - "核心模块" 部分

```python
from agents.rl_diagnosis_agent import create_rl_diagnosis_agent
```

#### 🎓 了解消融实验

→ [`USER_GUIDE.md`](USER_GUIDE.md) - "常见问题" Q4
或 [`DUAL_AGENT_QUICKSTART.md`](DUAL_AGENT_QUICKSTART.md) - "消融研究"

#### 🏗️ 添加新功能

→ [`DUAL_AGENT_QUICKSTART.md`](DUAL_AGENT_QUICKSTART.md) - "进阶: 自定义扩展"

#### ❌ 解决问题

→ [`USER_GUIDE.md`](USER_GUIDE.md) - "常见问题解答" (6 个问题)

---

## 🔍 源代码定位

### 诊断智能体 (SAC)

**文件**: `agents/rl_diagnosis_agent.py`
**关键类**: `RLDiagnosisAgent`
**关键方法**:

- `select_action()` - 选择诊断动作
- `compute_reward()` - 计算多分量奖励
- `update()` - SAC 算法更新

### 多智能体算法

**文件**: `agents/multi_agent_algorithms.py`
**关键类**:

- `MAPPO` - Multi-Agent PPO
- `QMIX` - Q-Mix 值分解
- `get_multi_agent_algorithm()` - 工厂函数

### 双智能体环境

**文件**: `environments/dual_agent_env.py`
**关键类**: `DualAgentEngineEnv`
**关键方法**:

- `reset()` - 重置并随机注入故障
- `step()` - 环境步进，返回双奖励
- 提供真实故障标签在 `info['ground_truth_fault']`

### 训练框架

**文件**: `scripts/train_dual_agents.py`
**关键类**: `DualAgentTrainer`
**支持模式**: independent, mappo, qmix
**关键参数**: episodes, eval_interval, save_dir

### 评估系统

**文件**: `experiments/dual_agent_evaluation.py`
**关键类**: `DualAgentEvaluator`
**评估维度**:

- 诊断性能 (准确率、延迟、混淆矩阵等)
- 控制性能 (RMSE、违规、恢复时间等)
- 协同性能 (端到端成功、下游可控性)
- 鲁棒性 (噪声、未见、多故障)

### 可视化工具

**文件**: `visualization/dual_agent_plots.py`
**关键类**: `DualAgentVisualizer`
**关键方法**:

- `plot_coordination_response()` - 4 轨迹时序
- `plot_training_curves()` - 6 子图学习曲线
- `plot_performance_comparison()` - 5D 雷达图
- `plot_confusion_matrix()` - 混淆矩阵

### 主入口点

**文件**: `main.py` (687 行)
**新增函数**:

- `run_dual_agent_training()` - 训练入口
- `run_dual_agent_evaluation()` - 评估入口
- `run_dual_agent_demo()` - 演示入口
  **新增命令**: 6 个 (见下表)

---

## ⌨️ 命令速查表

| 命令                        | 说明       | 使用场景             |
| --------------------------- | ---------- | -------------------- |
| `--mode train-mappo`        | MAPPO 训练 | 推荐，快速收敛       |
| `--mode train-qmix`         | QMIX 训练  | 稳定性好，值分解     |
| `--mode train-independent`  | 独立学习   | 消融实验、对比基线   |
| `--mode eval-dual`          | 评估模型   | 查看性能指标         |
| `--mode demo-dual`          | 演示可视化 | 生成时序图、验证协同 |
| `--mode train-mappo --help` | 查看参数   | 了解所有可用参数     |

---

## 📊 知识图谱

```
系统设计
  ├─ 诊断智能体 (RLDiagnosisAgent)
  │  ├─ SAC 算法
  │  ├─ Conv1D 编码器 (残差序列)
  │  ├─ 双 Q 网络架构
  │  └─ 多分量奖励 (包含下游反馈)
  │
  ├─ 多智能体算法
  │  ├─ MAPPO (中央化评价)
  │  ├─ QMIX (值分解)
  │  └─ 独立学习 (基线)
  │
  ├─ 环境 (DualAgentEngineEnv)
  │  ├─ 分离观察 (诊断/控制)
  │  ├─ 真实故障标签
  │  ├─ 随机多故障注入
  │  └─ 联合奖励信号
  │
  └─ 评估 (DualAgentEvaluator)
     ├─ 诊断指标 (准确率、延迟等)
     ├─ 控制指标 (RMSE、违规等)
     ├─ 协同指标 (端到端、下游)
     └─ 鲁棒性 (噪声、未见、多故障)

核心概念
  ├─ SAC: 最大熵强化学习，自动温度调整
  ├─ MAPPO: PPO-Clip for actors, 中央评价
  ├─ QMIX: 超网络混合 + 单调性约束
  ├─ 诊-控协同: 诊断奖励受控制成功影响
  └─ 真实标签: 环境提供每步的真实故障类型
```

---

## 🔗 内部模块依赖

```
main.py (入口点)
  ↓
run_dual_agent_training()
  ├─ scripts/train_dual_agents.py (DualAgentTrainer)
  │   ├─ agents/rl_diagnosis_agent.py (RLDiagnosisAgent)
  │   ├─ agents/multi_agent_algorithms.py (MAPPO/QMIX)
  │   ├─ agents/rl_algorithms.py (SAC)
  │   └─ environments/dual_agent_env.py (DualAgentEngineEnv)
  │
run_dual_agent_evaluation()
  └─ experiments/dual_agent_evaluation.py (DualAgentEvaluator)

run_dual_agent_demo()
  └─ visualization/dual_agent_plots.py (DualAgentVisualizer)
```

---

## 📋 文件清单与统计

| 模块     | 文件                                   | 行数     | 功能             | 文档       |
| -------- | -------------------------------------- | -------- | ---------------- | ---------- |
| 诊断     | `agents/rl_diagnosis_agent.py`         | 852      | SAC 诊断智能体   | ✓ 完整     |
| 多智能体 | `agents/multi_agent_algorithms.py`     | 1131     | MAPPO/QMIX       | ✓ 完整     |
| 环境     | `environments/dual_agent_env.py`       | 843      | 双智能体环境     | ✓ 完整     |
| 环境导出 | `environments/__init__.py`             | 小       | 导出函数         | -          |
| 训练     | `scripts/train_dual_agents.py`         | 667      | 训练框架         | ✓ 完整     |
| 评估     | `experiments/dual_agent_evaluation.py` | 676      | 评估系统         | ✓ 完整     |
| 可视化   | `visualization/dual_agent_plots.py`    | 419      | 可视化工具       | ✓ 完整     |
| **总计** | **7 个核心模块**                       | **4556** | **双智能体系统** | **✓ 完整** |

---

## 🎓 学习路径推荐

### 路径 A: 快速实践者 (2-3 小时)

1. 阅读 `USER_GUIDE.md` 快速开始 (15 min)
2. 运行 `python main.py --mode train-mappo --episodes 100` (30 min)
3. 运行 `python main.py --mode eval-dual --model-dir ...` (10 min)
4. 阅读 `USER_GUIDE.md` 中的工作流示例 (15 min)
5. 尝试调参和消融实验 (60 min)

### 路径 B: 系统设计者 (4-6 小时)

1. 阅读 `DUAL_AGENT_SUMMARY.md` 系统概览 (20 min)
2. 阅读 `DUAL_AGENT_QUICKSTART.md` API 文档 (30 min)
3. 阅读 `IMPLEMENTATION_REPORT.md` 技术细节 (25 min)
4. 浏览源代码关键部分 (60 min)
5. 自定义扩展 (60 min)

### 路径 C: 深度研究者 (8-12 小时)

1. 完整阅读 4 份文档 (90 min)
2. 详细研究源代码 (120 min)
3. 进行消融实验 (120 min)
4. 自定义新算法 (60 min)
5. 性能优化与基准测试 (120 min)

---

## ✨ 关键特性检索

| 特性     | 位置                                 | 说明                  |
| -------- | ------------------------------------ | --------------------- |
| SAC 诊断 | `rl_diagnosis_agent.py`              | 自适应温度、双 Q 网络 |
| 下游反馈 | `rl_diagnosis_agent.py` L650-680     | 诊断奖励含控制效果    |
| MAPPO    | `multi_agent_algorithms.py` L400-700 | PPO-Clip 中央评价     |
| QMIX     | `multi_agent_algorithms.py` L750-950 | 超网络单调混合        |
| 真实标签 | `dual_agent_env.py` L350-380         | 每步返回真实故障      |
| 多故障   | `dual_agent_env.py` L400-500         | 10% 并发注入          |
| 完整评估 | `dual_agent_evaluation.py`           | 20+ 指标，3 维度      |
| 可视化   | `dual_agent_plots.py`                | 4 种图表类型          |

---

## 🚀 立即开始

### 最简单的开始方式：

```bash
# 1. 进入项目目录
cd d:\my_github\CDC

# 2. 快速训练 (50 回合，~5 分钟)
python main.py --mode train-mappo --episodes 50

# 3. 查看结果
python main.py --mode eval-dual --model-dir models/dual_agent

# 4. 如需更详细的帮助
# 打开 USER_GUIDE.md 阅读
```

### 深入探索：

```bash
# 查看完整的命令帮助
python main.py --help

# 查看训练日志
cat models/dual_agent/training_log.txt

# 查看可视化结果
# 打开 models/dual_agent/coordination_response.png
```

---

## 📞 需要帮助？

1. **快速问题** → [`USER_GUIDE.md` - 常见问题](USER_GUIDE.md#常见问题解答)
2. **API 问题** → [`DUAL_AGENT_QUICKSTART.md`](DUAL_AGENT_QUICKSTART.md)
3. **技术细节** → [`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md)
4. **系统理解** → [`DUAL_AGENT_SUMMARY.md`](DUAL_AGENT_SUMMARY.md)

---

**最后更新**: 2024  
**版本**: 2.0 (双智能体强化学习)  
**完成度**: ✅ 100% | **状态**: 🟢 生产就绪
