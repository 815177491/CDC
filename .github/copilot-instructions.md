# CDC 项目 Copilot 指令

> 本项目是二冲程低速船用柴油机零维热力学模型的智能诊断与控制系统。

## 项目结构

| 目录                    | 说明                                                               |
| ----------------------- | ------------------------------------------------------------------ |
| `engine/`               | 零维柴油机热力学模型（几何运动学、燃烧放热、缸内传热、热力学求解） |
| `calibration/`          | 三阶段分步解耦校准（压缩段→燃烧段→传热段）                         |
| `marl/`                 | 双智能体强化学习：PINN+KAN 诊断智能体 + TD-MPC2 控制智能体         |
| `multi_Agent/`          | 多智能体环境（engine_env、fault_injector、operating_scheduler）    |
| `config/`               | 全局配置模块，所有参数集中管理在 `global_config.py`                |
| `visualization/`        | 可视化绑定模块，学术期刊风格（IEEE/Elsevier）                      |
| `visualization_output/` | 可视化输出目录，按类别分子目录                                     |
| `scripts/`              | 辅助脚本（数据生成、敏感性分析等）                                 |
| `experiments/`          | 实验脚本                                                           |
| `docs/`                 | 项目文档                                                           |

## 文件头部模板

所有 Python 文件必须使用以下头部：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块名称
========
模块功能描述

Author: CDC Project
Date: YYYY-MM-DD
"""
```

## 导入规范

导入顺序：标准库 → 第三方库 → 本项目模块。

```python
# 标准库
import os
from typing import Dict, List, Optional

# 第三方库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 本项目模块
from config import PLOT_CONFIG, COLORS, PATH_CONFIG, setup_matplotlib_style, save_figure
from engine.config import ENGINE_CONFIG
```

## 参数管理

- 所有可配置参数必须放在 `config/global_config.py` 中对应的 dataclass 里
- 禁止在代码中硬编码以下内容：
  - 字号（使用 `PLOT_CONFIG.FONT_SIZE_*`）
  - 颜色值（使用 `COLORS['primary']` 等）
  - 路径（使用 `PATH_CONFIG.*`）
  - 发动机物理参数（使用 `ENGINE_CONFIG.*`）
  - 训练超参数（使用 `TRAINING_CONFIG.*`）

配置类一览：

| 配置类              | 用途           | 使用方式                 |
| ------------------- | -------------- | ------------------------ |
| `PlotConfig`        | 绑定参数       | `PLOT_CONFIG.xxx`        |
| `PathConfig`        | 路径配置       | `PATH_CONFIG.xxx`        |
| `DataConfig`        | 数据处理参数   | `DATA_CONFIG.xxx`        |
| `EngineConfig`      | 发动机物理参数 | `ENGINE_CONFIG.xxx`      |
| `CalibrationConfig` | 校准参数       | `CALIBRATION_CONFIG.xxx` |
| `TrainingConfig`    | 训练参数       | `TRAINING_CONFIG.xxx`    |

## 绘图规范（学术期刊风格）

### 输出格式

- 格式：SVG（矢量图，文字可编辑，不转换为路径）
- 中文字体：宋体 (SimSun)
- 英文字体：Times New Roman
- 保存位置：`visualization_output/<category>/`

### 字号规范

| 元素     | 字号 | 配置变量                         |
| -------- | ---- | -------------------------------- |
| 刻度标签 | 14pt | `PLOT_CONFIG.FONT_SIZE_TICK`     |
| 轴标签   | 14pt | `PLOT_CONFIG.FONT_SIZE_LABEL`    |
| 图例     | 12pt | `PLOT_CONFIG.FONT_SIZE_LEGEND`   |
| 图中文字 | 12pt | `PLOT_CONFIG.FONT_SIZE_TEXT`     |
| 子图标题 | 12pt | `PLOT_CONFIG.FONT_SIZE_TITLE`    |
| 总标题   | 14pt | `PLOT_CONFIG.FONT_SIZE_SUPTITLE` |

### 配色规范

必须使用 `COLORS` 字典中的预定义颜色：

- `'primary'` (#2E86AB) - 主色蓝色，用于主要数据线
- `'secondary'` (#A23B72) - 次色紫红色
- `'success'` (#28A745) - 成功/正常，绿色
- `'warning'` (#FFC107) - 警告，黄色
- `'danger'` (#DC3545) - 危险/异常，红色
- `'info'` (#17A2B8) - 信息，青色
- `'dark'` (#343A40) - 深灰
- `'orange'` (#FF8C00) - 橙色
- `'purple'` (#6F42C1) - 紫色
- `'teal'` (#20C997) - 青绿色

### 绘图代码必须步骤

1. 调用 `setup_matplotlib_style()` 设置全局样式（模块级别，只调用一次）
2. 使用 `PLOT_CONFIG` 中的字号变量设置字体大小
3. 使用 `COLORS` 字典中的颜色
4. 使用 `save_figure(fig, category, filename)` 保存图形

category 取值：`'preprocessing'` | `'calibration'` | `'training'` | `'experiments'` | `'modeling'`

## 命名规范

### 文件命名

| 类型        | 命名规则           | 示例                        |
| ----------- | ------------------ | --------------------------- |
| Python 模块 | 小写+下划线        | `data_loader.py`            |
| 可视化模块  | `*_plots.py`       | `preprocessing_plots.py`    |
| 实验脚本    | `*_experiments.py` | `comparison_experiments.py` |

### 函数命名

| 类型     | 前缀                            | 示例                            |
| -------- | ------------------------------- | ------------------------------- |
| 绑定函数 | `plot_*`                        | `plot_steady_state_selection()` |
| 数据处理 | `process_*`, `load_*`, `save_*` | `load_data()`                   |
| 计算函数 | `calculate_*`, `compute_*`      | `calculate_metrics()`           |
| 验证函数 | `validate_*`, `check_*`         | `validate_input()`              |

### 变量命名

- 常量和配置变量：全大写 + 下划线（`MAX_ITERATIONS`, `PLOT_CONFIG`）
- DataFrame：`df_*`（`df_clean`, `df_raw`）
- 图形对象：`fig`, `ax`, `axes`
- 字号变量：`*_size`（`tick_size`, `label_size`）

## 物理量命名约定

| 符号                 | 含义         | 单位     |
| -------------------- | ------------ | -------- |
| `Pmax` / `P_max`     | 最大爆发压力 | bar      |
| `Pcomp` / `P_comp`   | 压缩压力     | bar      |
| `Texh` / `T_exhaust` | 排气温度     | K        |
| `p_scav`             | 扫气压力     | Pa       |
| `T_scav`             | 扫气温度     | K        |
| `bore`               | 气缸直径     | m        |
| `stroke`             | 活塞行程     | m        |
| `gamma`              | 比热比       | -        |
| `R`                  | 气体常数     | J/(kg·K) |

## 代码风格

- 使用 `dataclass` 定义配置类
- 函数必须有完整的 docstring（Args、Returns、Raises）
- 推荐使用类型注解
- 使用 `matplotlib.use('Agg')` 非交互式后端

## 发动机模型关键公式

- 温度微分方程：`dT/dθ = (1/mc_v) × [dQ_comb/dθ - dQ_wall/dθ - p × dV/dθ]`
- 气体状态方程：`p = mRT/V`
- 燃烧模型：双Wiebe函数
- 传热模型：Woschni经验公式
- 压缩比范围：13.5（几何）~ 15.55（有效校准值）
