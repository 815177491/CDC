# 快速开始指南

## 📋 前置要求

- Python 3.8+
- Anaconda/Miniconda（推荐）或系统Python

## 🚀 5分钟快速启动

### 步骤1：克隆项目

```bash
git clone <your-repo-url>
cd CDC
```

### 步骤2：安装依赖

**方式A：完整安装（推荐）**

```bash
pip install numpy scipy pandas matplotlib seaborn torch scikit-learn
```

**方式B：最小安装**

```bash
pip install numpy scipy pandas matplotlib seaborn
# 注：双智能体功能会降级到规则/PID模式
```

### 步骤3：选择运行模式

#### 🎯 模式1：快速测试（最简单）

```bash
python test_agents.py
```

**输出示例**：

```
============================================================
双智能体系统导入测试
============================================================
✓ 智能体模块导入成功
✓ PyTorch 2.9.1+cpu 可用
✓ scikit-learn 1.8.0 可用

基准值:
  Pmax: 137.1 bar
  Pcomp: 110.9 bar
  Texh: 623.4 K

运行简短仿真 (10步)
t= 0s | Pmax= 137.8bar | VIT调整=-8.00° | 状态: 正常
t= 5s | Pmax= 150.8bar | VIT调整=-8.00° | 状态: 故障 ⚠️
...
```

⏱️ **运行时间**：约10秒

---

#### 🔬 模式2：双智能体演示（推荐）

```bash
python main.py --mode agents
```

**功能**：

- ✅ 完整的参数校准流程
- ✅ 故障注入仿真（喷油正时故障）
- ✅ 诊断智能体实时检测
- ✅ 控制智能体自动调整
- ✅ 生成性能对比图表

**输出文件**：

- `results/agents_pmax_response.png` - Pmax响应曲线
- `results/agents_control_history.png` - 控制动作历史
- `results/agents_diagnosis_timeline.png` - 诊断时间线

⏱️ **运行时间**：约2-3分钟

---

#### 🎨 模式3：传统演示

```bash
python main.py --mode demo
```

**功能**：

- 三阶段参数校准
- 传统PID控制
- 静态可视化图表

⏱️ **运行时间**：约3-5分钟

---

#### 🎓 模式4：训练DQN控制器（高级）

```bash
python scripts/train_agents.py --episodes 500
```

**说明**：

- 离线训练DQN强化学习策略
- 需要torch库
- 训练数据保存到`agents/trained_dqn.pth`

⏱️ **运行时间**：约10-20分钟（取决于CPU性能）

---

#### 📊 模式5：性能对比评估

```bash
python scripts/evaluate_agents.py --trials 5
```

**输出指标**：

```
性能对比结果
========================================
| 指标           | 无控制  | PID控制 | 双智能体 |
|---------------|--------|--------|---------|
| 最大超调 (bar)  | 28.5   | 12.3   | 8.7     |
| 稳态误差 (bar)  | 25.0   | 4.2    | 2.1     |
| 检测延迟 (s)    | -      | 3.2    | 1.8     |
```

⏱️ **运行时间**：约5-10分钟

---

## 🐛 常见问题

### Q1: ImportError: No module named 'torch'

**解决方案**：

```bash
pip install torch scikit-learn
```

或者继续使用，系统会自动降级到传统方法。

---

### Q2: 校准数据文件未找到

**解决方案**：
确保项目根目录下有 `calibration_data.csv` 文件。如果缺失，程序会使用默认参数继续运行。

---

### Q3: 内存不足

**解决方案**：

- 减少训练轮数：`--episodes 100`
- 使用快速测试模式：`python test_agents.py`

---

## 📈 理解输出

### Pmax响应曲线图

```
Pmax (bar)
  |
160|           /故障发生\
  |          /          \_____ 控制后稳定
140|_________/
  |  基准值
120|
  +-----------------------------------> Time (s)
     0        25        50       100
```

**关键指标**：

- **基准值**：正常运行时的Pmax（约137 bar）
- **故障峰值**：故障发生时的最大Pmax（无控制可达165 bar）
- **控制后稳定值**：智能体控制后稳定在安全范围内

---

### 控制动作解释

| VIT调整 | 含义            | 效果                 |
| ------- | --------------- | -------------------- |
| -8°     | 喷油正时推迟8度 | 降低Pmax，提高安全性 |
| 0°      | 保持不变        | 维持当前状态         |
| +4°     | 喷油正时提前4度 | 提高Pmax，增加功率   |

| 燃油调整 | 含义           | 效果           |
| -------- | -------------- | -------------- |
| -20%     | 减少燃油喷射量 | 降低负荷和Pmax |
| 0%       | 保持不变       | 维持当前负荷   |
| +10%     | 增加燃油喷射量 | 提高功率输出   |

---

## 🎯 推荐学习路径

### 初学者

1. ✅ 运行 `python test_agents.py`（理解系统基本工作）
2. ✅ 运行 `python main.py --mode agents`（查看完整流程）
3. ✅ 阅读输出图表（理解控制效果）

### 进阶用户

4. ✅ 修改故障参数（在`main.py`中调整故障类型和强度）
5. ✅ 运行性能对比 `python scripts/evaluate_agents.py`
6. ✅ 阅读文档 `docs/chapter_agent.md`（深入理解算法原理）

### 研究者

7. ✅ 训练自定义DQN策略 `python scripts/train_agents.py`
8. ✅ 修改奖励函数（在`agents/control_agent.py`中调整）
9. ✅ 实现新的故障类型（在`diagnosis/fault_injector.py`中扩展）

---

## 💡 下一步

- 📖 阅读详细文档：[docs/chapter_agent.md](docs/chapter_agent.md)
- 🔧 调整系统参数：编辑 `agents/` 目录下的配置
- 📊 导出实验数据：查看 `results/` 目录

---

**祝实验顺利！** 🎉
