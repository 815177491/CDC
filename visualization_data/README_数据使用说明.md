# 双智能体可视化数据使用说明

本文档说明如何使用导出的CSV数据文件在第三方软件（如Excel、Origin、MATLAB、Python等）中重新绑制可视化图表。

---

## 📁 文件目录结构

```
visualization_data/
├── 训练过程数据
│   └── training_process.csv          # DQN训练曲线数据
│
├── 仿真结果数据
│   └── simulation_results.csv         # 100秒仿真时序数据
│
├── 性能对比数据
│   ├── performance_metrics.csv        # 关键指标对比
│   ├── performance_radar.csv          # 雷达图归一化数据
│   └── step_response.csv              # 阶跃响应曲线
│
├── 诊断智能体数据
│   ├── adaptive_threshold.csv         # 自适应阈值学习
│   ├── classifier_weights.csv         # 集成分类器权重
│   ├── fault_distribution.csv         # 故障类型分布
│   ├── detection_delay.csv            # 检测延迟样本
│   ├── detection_delay_stats.csv      # 延迟统计量
│   ├── confusion_matrix.csv           # 混淆矩阵(长格式)
│   ├── confusion_matrix_table.csv     # 混淆矩阵(表格式)
│   ├── roc_curve.csv                  # ROC曲线数据点
│   └── roc_auc.csv                    # AUC值
│
└── 控制智能体数据
    ├── dqn_architecture.csv           # DQN网络结构
    ├── action_space_q_values.csv      # 动作空间Q值(长格式)
    ├── q_value_matrix.csv             # Q值矩阵(热力图用)
    ├── reward_components.csv          # 奖励函数分解
    ├── pid_vs_rl_actions.csv          # PID vs RL动作对比
    ├── replay_buffer_rewards.csv      # 经验回放奖励分布
    ├── replay_buffer_stats.csv        # 奖励统计量
    ├── safety_constraint.csv          # 安全约束效果
    └── safety_constraints_config.csv  # 约束参数配置
```

---

## 📊 图表1: DQN训练过程 (training_process.png)

### 数据文件: `training_process.csv`

| 列名 | 含义 | 单位 | 图表用途 |
|------|------|------|----------|
| episode | 训练回合 | - | X轴 |
| loss | 原始损失值 | - | (a)图 Y轴 |
| loss_smoothed | 平滑损失(窗口=10) | - | (a)图 叠加曲线 |
| q_value | 平均Q值 | - | (b)图 Y轴 |
| epsilon | 探索率 | 0~1 | (c)图 Y轴 |
| reward | 原始累计奖励 | - | (d)图 Y轴 |
| reward_smoothed | 平滑奖励 | - | (d)图 叠加曲线 |
| steps | 回合步数 | 步 | 可选显示 |

### 绑制说明

**子图(a) 损失曲线:**
```
X轴: episode
Y轴: loss (对数刻度, log scale)
叠加: loss_smoothed (红色粗线)
```

**子图(b) Q值学习:**
```
X轴: episode
Y轴: q_value
填充: 从0到q_value的面积填充
参考线: y=10 (收敛目标)
```

**子图(c) 探索率衰减:**
```
X轴: episode
Y轴: epsilon
参考线: y=0.05 (ε_min)
```

**子图(d) 累计奖励:**
```
X轴: episode
Y轴: reward (浅色)
叠加: reward_smoothed (深色粗线)
```

### Excel操作示例
1. 打开CSV文件
2. 选择 `episode` 和 `loss` 列
3. 插入 → 图表 → 散点图(带平滑线)
4. 右键Y轴 → 设置坐标轴格式 → 对数刻度

---

## 📊 图表2: 仿真结果评估 (simulation_results.png)

### 数据文件: `simulation_results.csv`

| 列名 | 含义 | 单位 |
|------|------|------|
| time_s | 仿真时间 | 秒 |
| pmax_bar | 最大燃烧压力 | bar |
| pmax_baseline | Pmax基准值 | bar |
| pmax_upper_threshold | 上阈值(+5%) | bar |
| pmax_lower_threshold | 下阈值(-5%) | bar |
| pcomp_bar | 压缩压力 | bar |
| texh_K | 排气温度 | K |
| fault_status | 故障状态 | 0/1 |
| diagnosis_confidence | 诊断置信度 | 0~1 |
| vit_adjust_deg | VIT调整量 | °CA |
| fuel_multiplier | 燃油倍率 | - |
| control_mode | 控制模式 | 文本 |

### 绑制说明

**子图(a) Pmax响应曲线:**
```
X轴: time_s
Y轴: pmax_bar (主数据)
叠加: pmax_baseline (绿色虚线)
叠加: pmax_upper_threshold, pmax_lower_threshold (黄色点线)
垂直线: x=25 (故障注入时刻)
```

**子图(b) 诊断置信度:**
```
X轴: time_s
Y轴: diagnosis_confidence
散点图, 颜色由 fault_status 决定:
  - fault_status=0 → 绿色
  - fault_status=1 → 红色
水平线: y=0.5 (诊断阈值)
```

**子图(c) VIT控制动作:**
```
X轴: time_s
Y轴: vit_adjust_deg
面积填充图
水平线: y=-8 (下限), y=4 (上限)
```

**子图(d) 燃油控制:**
```
X轴: time_s
Y轴: fuel_multiplier
范围: [0.65, 1.05]
```

---

## 📊 图表3: 性能对比 (performance_comparison.png)

### 数据文件

**3.1 `performance_metrics.csv` - 柱状图**

| 列名 | 含义 |
|------|------|
| metric | 指标英文名 |
| metric_cn | 指标中文名 |
| traditional | 传统方法值 |
| dual_agent | 双智能体值 |
| improvement_percent | 改进百分比 |

**子图(a) 柱状对比图:**
```
X轴: metric_cn (分类)
Y轴: 数值
分组柱状图: traditional (深灰), dual_agent (蓝色)
```

**子图(b) 改进百分比:**
```
横向条形图
X轴: improvement_percent
Y轴: metric_cn
颜色: 正值=绿色
```

---

**3.2 `performance_radar.csv` - 雷达图**

| 列名 | 含义 |
|------|------|
| metric | 指标名 |
| traditional_normalized | 传统方法归一化值 (0~1, 越大越好) |
| dual_agent_normalized | 双智能体归一化值 |
| angle_deg | 角度 (°) |

**子图(c) 雷达图绑制:**
```
极坐标系
角度: angle_deg (0, 72, 144, 216, 288)
半径: *_normalized
闭合多边形
```

**Origin绑制:** 
1. 新建极坐标图
2. 导入数据
3. 绑制 angle_deg vs traditional_normalized
4. 添加第二数据集 angle_deg vs dual_agent_normalized

---

**3.3 `step_response.csv` - 阶跃响应**

| 列名 | 含义 |
|------|------|
| time_s | 时间 (秒) |
| traditional_response | 传统PID响应 |
| dual_agent_response | 双智能体响应 |
| setpoint | 设定值 (=1) |
| upper_bound_5percent | 上界 (=1.05) |
| lower_bound_5percent | 下界 (=0.95) |

**子图(d) 阶跃响应对比:**
```
X轴: time_s
Y轴: 归一化响应
曲线1: traditional_response (虚线, 深灰)
曲线2: dual_agent_response (实线, 蓝色)
水平线: setpoint, upper_bound, lower_bound
```

---

## 📊 图表4: 诊断智能体分析 (diagnosis_analysis.png)

### 4.1 `adaptive_threshold.csv` - 自适应阈值

| 列名 | 含义 |
|------|------|
| time_step | 时间步 |
| pmax_bar | Pmax数据 |
| moving_average | 滑动均值 μ |
| moving_std | 滑动标准差 σ |
| upper_threshold_3sigma | 上阈值 μ+3σ |
| lower_threshold_3sigma | 下阈值 μ-3σ |
| condition_change | 工况变化标记 (t=50时为1) |

**子图(a) 绑制:**
```
X轴: time_step
带状填充: lower_threshold_3sigma 到 upper_threshold_3sigma (浅蓝)
曲线: pmax_bar (蓝色细线)
曲线: moving_average (红色粗线)
垂直线: condition_change=1 的位置 (x=50)
```

---

### 4.2 `classifier_weights.csv` - 饼图

| 列名 | 含义 |
|------|------|
| classifier | 分类器名称 |
| weight | 权重 (0~1) |
| weight_percent | 权重百分比 |

**子图(b) 饼图:**
```
数据: weight_percent
标签: classifier
```

---

### 4.3 `fault_distribution.csv` - 故障分布

| 列名 | 含义 |
|------|------|
| fault_type | 故障类型英文 |
| fault_type_cn | 故障类型中文 |
| count | 检测次数 |

**子图(c) 水平条形图:**
```
Y轴: fault_type_cn
X轴: count
不同颜色区分类别
```

---

### 4.4 `detection_delay.csv` - 延迟直方图

| 列名 | 含义 |
|------|------|
| sample_id | 样本ID |
| dual_agent_delay_s | 双智能体检测延迟 (秒) |
| traditional_delay_s | 传统方法检测延迟 (秒) |

**子图(d) 直方图:**
```
双直方图叠加
bins=20
dual_agent_delay_s: 蓝色, alpha=0.7
traditional_delay_s: 灰色, alpha=0.5
垂直线: 各自的均值
```

统计量见 `detection_delay_stats.csv`

---

### 4.5 `confusion_matrix_table.csv` - 混淆矩阵

| actual\predicted | Normal | Single_Fault | Multi_Fault |
|------------------|--------|--------------|-------------|
| Normal | 62 | 3 | 0 |
| Single_Fault | 2 | 28 | 1 |
| Multi_Fault | 1 | 2 | 1 |

**子图(e) 热力图:**
```
Excel: 条件格式 → 色阶
Origin: 矩阵图
MATLAB: imagesc() 或 heatmap()
```

---

### 4.6 `roc_curve.csv` - ROC曲线

| 列名 | 含义 |
|------|------|
| false_positive_rate | 假阳性率 (FPR) |
| dual_agent_tpr | 双智能体真阳性率 |
| traditional_tpr | 传统方法真阳性率 |
| random_classifier | 随机分类器 (对角线) |

**子图(f) ROC曲线:**
```
X轴: false_positive_rate
Y轴: *_tpr
曲线1: dual_agent_tpr (蓝色实线)
曲线2: traditional_tpr (灰色虚线)
曲线3: random_classifier (对角线, 黄色点线)
面积填充: dual_agent_tpr 下方
```

AUC值见 `roc_auc.csv`

---

## 📊 图表5: 控制智能体分析 (control_analysis.png)

### 5.1 `dqn_architecture.csv` - 网络结构 (表格)

| layer | neurons | activation | description |
|-------|---------|------------|-------------|
| Input | 10 | None | State vector |
| Hidden1 | 128 | ReLU | Fully connected |
| Hidden2 | 64 | ReLU | Fully connected |
| Output | 45 | None | Q-values for actions |

用于绘制网络结构示意图。

---

### 5.2 `action_space_q_values.csv` / `q_value_matrix.csv` - Q值热力图

**长格式 (action_space_q_values.csv):**

| vit_adjust_deg | fuel_multiplier | q_value |
|----------------|-----------------|---------|
| -8 | 0.70 | 5.23 |
| -6.5 | 0.70 | 7.45 |
| ... | ... | ... |

**矩阵格式 (q_value_matrix.csv):**

行索引: fuel_0.70, fuel_0.78, ..., fuel_1.00
列索引: vit_-8.0, vit_-6.5, ..., vit_4.0
单元格: Q值

**子图(b) 热力图绑制:**
```
Excel: 条件格式 → 色阶 (红黄绿)
Origin: 矩阵图 → Contour
MATLAB: contourf(VIT, FUEL, Q)
Python: plt.contourf() 或 seaborn.heatmap()
```

---

### 5.3 `reward_components.csv` - 奖励分解

| component | component_cn | value | color_hex |
|-----------|--------------|-------|-----------|
| Pmax_Control | Pmax控制 | 3.5 | #28A745 |
| Stability | 稳定性 | 2.0 | #17A2B8 |
| Efficiency | 效率 | 1.5 | #2E86AB |
| Safety_Penalty | 安全惩罚 | -0.5 | #DC3545 |
| Total | 总奖励 | 6.5 | #A23B72 |

**子图(c) 柱状图:**
```
X轴: component_cn
Y轴: value
颜色: color_hex
水平线: y=0
```

---

### 5.4 `pid_vs_rl_actions.csv` - 动作对比

| time_step | error_signal | pid_action | rl_action | action_difference |
|-----------|--------------|------------|-----------|-------------------|
| 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | 4.34 | 8.68 | 10.85 | 2.17 |
| ... | ... | ... | ... | ... |

**子图(d) 曲线对比:**
```
X轴: time_step
曲线1: pid_action (灰色虚线)
曲线2: rl_action (蓝色实线)
可选: 填充两曲线之间的差异区域
```

---

### 5.5 `replay_buffer_rewards.csv` - 奖励分布直方图

| sample_id | reward | training_phase |
|-----------|--------|----------------|
| 0 | -2.34 | Early |
| 300 | 3.12 | Middle |
| 800 | 6.05 | Late |

**子图(e) 直方图:**
```
bins=30
X轴: reward
Y轴: 频次
垂直线: 均值 (见 replay_buffer_stats.csv)
```

---

### 5.6 `safety_constraint.csv` - 安全约束效果

| action_index | raw_action | constrained_action | was_clipped | clip_amount |
|--------------|------------|--------------------|-------------|-------------|
| 0 | -9.5 | -8.0 | 1 | 1.5 |
| 1 | 2.3 | 2.3 | 0 | 0.0 |
| ... | ... | ... | ... | ... |

**子图(f) 散点图:**
```
X轴: action_index
Y轴: 动作值
散点1: raw_action (红色, 半透明)
散点2: constrained_action (绿色)
水平带: y ∈ [-8, 4] (安全区域, 浅绿填充)
水平线: y=-8, y=4 (边界)
```

约束参数见 `safety_constraints_config.csv`

---

## 🛠️ 常用软件操作指南

### Microsoft Excel

1. **导入CSV:** 数据 → 获取外部数据 → 从文本/CSV
2. **绑制图表:** 选择数据 → 插入 → 图表
3. **热力图:** 选择矩阵数据 → 条件格式 → 色阶

### Origin

1. **导入:** File → Import → CSV
2. **绑制:** Plot → 选择图表类型
3. **极坐标图:** Plot → Specialized → Polar
4. **热力图:** Plot → Contour → Contour - Color Fill

### MATLAB

```matlab
% 读取CSV
data = readtable('training_process.csv');

% 绑制
plot(data.episode, data.loss);
set(gca, 'YScale', 'log');

% 热力图
Q = readmatrix('q_value_matrix.csv');
imagesc(Q);
colorbar;
```

### Python (Matplotlib/Seaborn)

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV
df = pd.read_csv('training_process.csv')

# 绘制
plt.semilogy(df['episode'], df['loss'])
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.show()

# 热力图
q_matrix = pd.read_csv('q_value_matrix.csv', index_col=0)
sns.heatmap(q_matrix, cmap='RdYlGn')
```

---

## 📋 数据汇总表

| 文件名 | 行数 | 用于图表 |
|--------|------|----------|
| training_process.csv | 200 | Fig.1 (a)(b)(c)(d) |
| simulation_results.csv | 100 | Fig.2 (a)(b)(c)(d) |
| performance_metrics.csv | 5 | Fig.3 (a)(b) |
| performance_radar.csv | 5 | Fig.3 (c) |
| step_response.csv | 100 | Fig.3 (d) |
| adaptive_threshold.csv | 100 | Fig.4 (a) |
| classifier_weights.csv | 2 | Fig.4 (b) |
| fault_distribution.csv | 5 | Fig.4 (c) |
| detection_delay.csv | 200 | Fig.4 (d) |
| confusion_matrix_table.csv | 3 | Fig.4 (e) |
| roc_curve.csv | 100 | Fig.4 (f) |
| dqn_architecture.csv | 4 | Fig.5 (a) |
| action_space_q_values.csv | 45 | Fig.5 (b) |
| q_value_matrix.csv | 5×9 | Fig.5 (b) |
| reward_components.csv | 5 | Fig.5 (c) |
| pid_vs_rl_actions.csv | 50 | Fig.5 (d) |
| replay_buffer_rewards.csv | 1000 | Fig.5 (e) |
| safety_constraint.csv | 100 | Fig.5 (f) |

---

## ❓ 常见问题

**Q: CSV文件中文乱码怎么办?**
A: 文件使用UTF-8-BOM编码保存。Excel打开时选择"65001: Unicode (UTF-8)"编码。

**Q: 如何修改图表配色?**
A: `reward_components.csv` 等文件包含 `color_hex` 列，可直接使用这些十六进制颜色代码。

**Q: 雷达图怎么闭合?**
A: 需要手动添加第一个点作为最后一个点（复制第一行到末尾），形成闭合多边形。

**Q: 热力图数据如何转置?**
A: Excel中复制 → 选择性粘贴 → 勾选"转置"

---

*文档生成时间: 2026-01-21*
*数据版本: v1.0*
