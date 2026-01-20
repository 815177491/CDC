# 零维船用柴油机仿真与控诊协同系统

## 项目概述

本项目实现了二冲程船用低速柴油机的零维热力学仿真模型，并建立了"故障诊断-容错控制"协同机制。

### 核心功能

1. **零维热力学模型** - 基于控制体分析法的发动机仿真
2. **三阶段参数校准** - 分步解耦的模型标定流程
3. **故障注入与诊断** - 残差分析法故障检测
4. **协同容错控制** - 主动容错控制器(AFTC)

## 项目结构

```
CDC/
├── main.py                 # 主入口脚本
├── calibration_data.csv    # 监测数据
├── engine/                 # 发动机模型模块
│   ├── __init__.py
│   ├── geometry.py         # 几何运动学
│   ├── combustion.py       # 双Wiebe燃烧模型
│   ├── heat_transfer.py    # Woschni传热模型
│   ├── thermodynamics.py   # 热力学求解器
│   └── engine_model.py     # 集成模型类
├── calibration/            # 参数校准模块
│   ├── __init__.py
│   ├── data_loader.py      # 数据加载器
│   └── calibrator.py       # 三阶段校准器
├── diagnosis/              # 故障诊断模块
│   ├── __init__.py
│   ├── fault_injector.py   # 故障注入器
│   └── diagnoser.py        # 残差诊断器
├── control/                # 协同控制模块
│   ├── __init__.py
│   └── synergy_controller.py  # AFTC控制器
├── visualization/          # 可视化模块
│   ├── __init__.py
│   ├── static_plots.py     # 静态图表
│   ├── radar_chart.py      # 雷达图
│   └── dashboard.py        # 交互式仪表盘
└── results/                # 输出结果目录
```

## 发动机参数

| 参数       | 数值             | 单位 |
| ---------- | ---------------- | ---- |
| 气缸直径   | 620              | mm   |
| 活塞行程   | 2658             | mm   |
| 气缸数量   | 6                | -    |
| 冲程缸径比 | 4.29             | -    |
| 几何压缩比 | 12~15 (待校准)   | -    |
| 连杆比     | 3.5~4.5 (待校准) | -    |

## 快速开始

### 1. 安装依赖

```bash
pip install numpy scipy pandas matplotlib seaborn
```

### 2. 运行演示

```bash
# 完整演示 (校准 + 仿真 + 绑图)
python main.py --mode demo

# 仅运行校准
python main.py --mode calibrate

# 启动交互式仪表盘
python main.py --mode dashboard
```

### 3. Python API 使用

```python
from engine import MarineEngine0D, OperatingCondition

# 创建发动机模型
engine = MarineEngine0D(
    bore=0.620,
    stroke=2.658,
    n_cylinders=6
)

# 定义工况
condition = OperatingCondition(
    rpm=80.0,
    p_scav=3.5e5,
    T_scav=320.0,
    fuel_mass=0.08
)

# 运行仿真
results = engine.run_cycle(condition)

# 获取结果
print(f"Pmax: {engine.get_pmax():.1f} bar")
print(f"Pcomp: {engine.get_pcomp():.1f} bar")
```

## 三阶段校准流程

### 第一阶段: 压缩过程校准

- **目标**: 确定有效压缩比
- **对标变量**: Pcomp (Cyl 1 KPI 5)
- **方法**: 纯压缩工况下优化压缩比使误差<2%

### 第二阶段: 燃烧放热规律校准

- **目标**: 确定Wiebe函数参数
- **对标变量**: Pmax (Cyl 1 KPI 4)
- **方法**: 差分进化算法优化喷油正时、燃烧持续期、形状因子

### 第三阶段: 传热校准

- **目标**: 修正能量平衡
- **对标变量**: 排气温度 (Exhaust gas temp cyl. 1)
- **方法**: 调整Woschni传热系数

## 控诊协同控制

### 场景A: Pmax越限控制

当检测到Pmax接近或超过190 bar安全限值时:

1. 优先推迟喷油正时(VIT-)
2. 若不足则减少燃油喷射量
3. 实现安全性优先的综合优化

### 场景B: 能效衰退补偿

当检测到某缸做功不足时:

1. 识别故障气缸
2. 在健康气缸重新分配负荷
3. 维持总转速稳定

## 输出图表

1. **fig1_calibration_verification.png** - 热力参数对标图
2. **fig2_pv_diagram.png** - P-V示功图 (多工况叠加)
3. **fig3_fault_response.png** - 故障响应时序图
4. **fig4_performance_radar.png** - 性能权衡雷达图
5. **fig5_heat_release.png** - 燃烧放热率曲线

## 技术特点

- 基于物理的零维热力学模型
- 双Wiebe函数描述预混+扩散燃烧
- Woschni关联式计算缸内传热
- 分步解耦避免多参数优化陷阱
- 残差分析法实现故障分类
- PID协同控制器实现容错控制

## License

MIT License
