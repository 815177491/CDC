# CDC 项目代码规范

> 版本：1.0  
> 更新日期：2026-01-28  
> 适用范围：所有新增代码

---

## 目录

1. [文件结构规范](#1-文件结构规范)
2. [绑定规范](#2-绑定规范)
3. [参数管理规范](#3-参数管理规范)
4. [代码风格规范](#4-代码风格规范)
5. [命名规范](#5-命名规范)
6. [模板代码](#6-模板代码)

---

## 1. 文件结构规范

### 1.1 项目目录结构

```
CDC/
├── config/                     # 📁 全局配置模块
│   ├── __init__.py
│   └── global_config.py        # 所有参数的统一管理
│
├── data/                       # 📁 数据文件
│   └── *.csv, *.json
│
├── visualization_output/       # 📁 可视化输出（所有图片）
│   ├── preprocessing/          # 数据预处理可视化
│   ├── calibration/            # 标定过程可视化
│   ├── training/               # 训练过程可视化
│   ├── experiments/            # 实验结果可视化
│   └── modeling/               # 模型相关可视化
│
├── visualization/              # 📁 可视化绑定模块
│   └── *_plots.py
│
├── calibration/                # 📁 标定模块
├── engine/                     # 📁 发动机模型
├── marl/                       # 📁 多智能体强化学习
├── experiments/                # 📁 实验脚本
├── checkpoints/                # 📁 模型检查点
└── docs/                       # 📁 文档
```

### 1.2 新文件创建规则

| 文件类型       | 存放位置                           | 命名规则             |
| -------------- | ---------------------------------- | -------------------- |
| 可视化绑定函数 | `visualization/`                   | `*_plots.py`         |
| **生成的图片** | `visualization_output/<category>/` | `*.svg`              |
| 配置参数       | `config/global_config.py`          | 添加到对应的Config类 |
| 数据文件       | `data/`                            | 描述性名称           |
| 实验脚本       | `experiments/`                     | `*_experiments.py`   |

---

## 2. 绑定规范

### 2.1 输出格式要求

| 项目         | 规范                               |
| ------------ | ---------------------------------- |
| **格式**     | SVG（矢量图）                      |
| **文字**     | 可编辑文字，不转换为路径           |
| **中文字体** | 宋体 (SimSun)                      |
| **英文字体** | Times New Roman                    |
| **保存位置** | `visualization_output/<category>/` |

### 2.2 字号规范

| 元素     | 字号     | 配置变量                         |
| -------- | -------- | -------------------------------- |
| 刻度标签 | **14pt** | `PLOT_CONFIG.FONT_SIZE_TICK`     |
| 轴标签   | **14pt** | `PLOT_CONFIG.FONT_SIZE_LABEL`    |
| 图例     | **12pt** | `PLOT_CONFIG.FONT_SIZE_LEGEND`   |
| 图中文字 | **12pt** | `PLOT_CONFIG.FONT_SIZE_TEXT`     |
| 子图标题 | **12pt** | `PLOT_CONFIG.FONT_SIZE_TITLE`    |
| 总标题   | **14pt** | `PLOT_CONFIG.FONT_SIZE_SUPTITLE` |

### 2.3 配色规范

**必须使用** `COLORS` 字典中的预定义颜色：

```python
COLORS = {
    'primary': '#2E86AB',    # 主色：蓝色（主要数据线）
    'secondary': '#A23B72',  # 次色：紫红色
    'success': '#28A745',    # 成功/正常：绿色
    'warning': '#FFC107',    # 警告：黄色
    'danger': '#DC3545',     # 危险/异常：红色
    'info': '#17A2B8',       # 信息：青色
    'dark': '#343A40',       # 深灰（背景数据）
    'light': '#F8F9FA',      # 浅灰
    'orange': '#FF8C00',     # 橙色
    'purple': '#6F42C1',     # 紫色
    'teal': '#20C997',       # 青绿色
    'pink': '#E83E8C',       # 粉色
}
```

### 2.4 绑定代码必须步骤

```python
# 1️⃣ 导入全局配置
from config import (
    PLOT_CONFIG, COLORS, PATH_CONFIG,
    setup_matplotlib_style, save_figure
)

# 2️⃣ 应用全局样式（模块级别，只调用一次）
setup_matplotlib_style()

# 3️⃣ 在绑定函数中使用配置变量
def plot_xxx(data):
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    legend_size = PLOT_CONFIG.FONT_SIZE_LEGEND
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE

    # 绑定逻辑...
    ax.set_xlabel('X轴标签', fontsize=label_size)
    ax.set_ylabel('Y轴标签', fontsize=label_size)
    ax.legend(fontsize=legend_size)
    ax.tick_params(labelsize=tick_size)

    # 4️⃣ 使用save_figure保存到正确目录
    save_figure(fig, 'category', 'filename.svg')
    plt.close()
```

### 2.5 图形类别（category参数）

| 类别              | 说明       | 输出目录                              |
| ----------------- | ---------- | ------------------------------------- |
| `'preprocessing'` | 数据预处理 | `visualization_output/preprocessing/` |
| `'calibration'`   | 标定过程   | `visualization_output/calibration/`   |
| `'training'`      | 训练过程   | `visualization_output/training/`      |
| `'experiments'`   | 实验结果   | `visualization_output/experiments/`   |
| `'modeling'`      | 模型相关   | `visualization_output/modeling/`      |

---

## 3. 参数管理规范

### 3.1 核心原则

**所有可配置参数必须放在 `config/global_config.py` 中**，禁止在代码中硬编码。

### 3.2 配置类说明

| 配置类           | 用途           | 使用方式              |
| ---------------- | -------------- | --------------------- |
| `PlotConfig`     | 绑定参数       | `PLOT_CONFIG.xxx`     |
| `PathConfig`     | 路径配置       | `PATH_CONFIG.xxx`     |
| `DataConfig`     | 数据处理参数   | `DATA_CONFIG.xxx`     |
| `EngineConfig`   | 发动机物理参数 | `ENGINE_CONFIG.xxx`   |
| `TrainingConfig` | 训练参数       | `TRAINING_CONFIG.xxx` |

### 3.3 添加新参数

如需添加新参数，在 `config/global_config.py` 中对应的类里添加：

```python
@dataclass
class DataConfig:
    """数据处理参数配置"""
    # 现有参数...
    STEADY_STATE_WINDOW: int = 60

    # ✅ 添加新参数
    NEW_PARAMETER: float = 1.0  # 参数说明
```

### 3.4 禁止事项

❌ **禁止在代码中硬编码以下内容**：

| 禁止硬编码                | 正确做法                               |
| ------------------------- | -------------------------------------- |
| `fontsize=14`             | `fontsize=PLOT_CONFIG.FONT_SIZE_LABEL` |
| `color='#2E86AB'`         | `color=COLORS['primary']`              |
| `'visualization_output/'` | `PATH_CONFIG.VIS_PREPROCESSING_DIR`    |
| `gamma=1.35`              | `ENGINE_CONFIG.gamma`                  |
| `lr=0.001`                | `TRAINING_CONFIG.LEARNING_RATE`        |

---

## 4. 代码风格规范

### 4.1 文件头部模板

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块名称
========
模块功能描述

包括:
1. 功能1
2. 功能2
3. 功能3

Author: CDC Project
Date: YYYY-MM-DD
"""
```

### 4.2 导入顺序

```python
# 1. 标准库
import os
import sys
from typing import Dict, List, Optional

# 2. 第三方库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 3. 本项目模块
from config import PLOT_CONFIG, COLORS, setup_matplotlib_style
from engine.config import ENGINE_CONFIG
```

### 4.3 类型注解

**推荐使用** 类型注解提高代码可读性：

```python
def process_data(
    df: pd.DataFrame,
    threshold: float = 0.5,
    columns: List[str] = None
) -> pd.DataFrame:
    """处理数据"""
    pass
```

### 4.4 函数文档字符串

```python
def plot_example(
    df: pd.DataFrame,
    output_dir: str = None
) -> plt.Figure:
    """
    绑定示例函数

    Args:
        df: 输入数据DataFrame，必须包含'rpm'和'P_max'列
        output_dir: 输出目录，默认使用全局配置

    Returns:
        matplotlib Figure对象

    Raises:
        ValueError: 当输入数据为空时

    Example:
        >>> fig = plot_example(df)
    """
    pass
```

---

## 5. 命名规范

### 5.1 文件命名

| 类型       | 命名规则           | 示例                        |
| ---------- | ------------------ | --------------------------- |
| Python模块 | 小写+下划线        | `data_loader.py`            |
| 可视化模块 | `*_plots.py`       | `preprocessing_plots.py`    |
| 实验脚本   | `*_experiments.py` | `comparison_experiments.py` |
| 配置文件   | `*_config.py`      | `global_config.py`          |

### 5.2 函数命名

| 类型     | 命名规则                        | 示例                            |
| -------- | ------------------------------- | ------------------------------- |
| 绑定函数 | `plot_*`                        | `plot_steady_state_selection()` |
| 数据处理 | `process_*`, `load_*`, `save_*` | `load_data()`                   |
| 计算函数 | `calculate_*`, `compute_*`      | `calculate_metrics()`           |
| 验证函数 | `validate_*`, `check_*`         | `validate_input()`              |

### 5.3 变量命名

| 类型      | 命名规则            | 示例                       |
| --------- | ------------------- | -------------------------- |
| 常量      | 全大写+下划线       | `MAX_ITERATIONS`           |
| 配置变量  | 全大写+下划线       | `PLOT_CONFIG`              |
| DataFrame | `df_*`              | `df_clean`, `df_raw`       |
| 图形对象  | `fig`, `ax`, `axes` | `fig, ax = plt.subplots()` |
| 字号变量  | `*_size`            | `tick_size`, `label_size`  |

---

## 6. 模板代码

### 6.1 可视化模块模板

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XXX可视化模块
=============
功能描述

Author: CDC Project
Date: YYYY-MM-DD
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# 导入全局配置
from config import (
    PLOT_CONFIG,
    COLORS,
    PATH_CONFIG,
    DATA_CONFIG,
    setup_matplotlib_style,
    save_figure,
)

# 应用全局样式
setup_matplotlib_style()


def plot_xxx(
    df: pd.DataFrame,
    output_dir: str = None
) -> plt.Figure:
    """
    绑定函数说明

    Args:
        df: 输入数据
        output_dir: 输出目录（可选）

    Returns:
        Figure对象
    """
    print("\n[X/X] 生成XXX可视化...")

    # 使用全局配置
    colors = COLORS
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    legend_size = PLOT_CONFIG.FONT_SIZE_LEGEND
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE

    # 创建图形
    fig = plt.figure(figsize=PLOT_CONFIG.FIGURE_SIZE_LARGE)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # ========== (a) 子图1 ==========
    ax1 = fig.add_subplot(gs[0, 0])

    # 绑定逻辑...
    ax1.plot(df['x'], df['y'], color=colors['primary'], linewidth=1.5)

    # 设置标签（使用配置变量）
    ax1.set_xlabel('X轴标签', fontsize=label_size)
    ax1.set_ylabel('Y轴标签', fontsize=label_size)
    ax1.set_title('(a) 子图标题', fontsize=title_size, fontweight='bold')
    ax1.legend(fontsize=legend_size)
    ax1.tick_params(labelsize=tick_size)
    ax1.grid(True, alpha=0.3)

    # 总标题
    plt.suptitle('图形总标题', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE, fontweight='bold')

    # 保存图形
    save_figure(fig, 'category', 'filename.svg')
    plt.close()

    return fig


# 如果直接运行此模块
if __name__ == '__main__':
    # 测试代码
    pass
```

### 6.2 主脚本模板

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XXX主脚本
=========
功能描述

Author: CDC Project
Date: YYYY-MM-DD
"""

import os

# 导入全局配置
from config import PATH_CONFIG

# 导入可视化模块
from visualization.xxx_plots import plot_xxx


def main():
    """主函数"""
    print("=" * 70)
    print("XXX处理流程")
    print("=" * 70)

    # 使用全局配置的路径
    output_dir = PATH_CONFIG.VIS_PREPROCESSING_DIR
    print(f"输出目录: {os.path.abspath(output_dir)}")

    try:
        # 处理逻辑...
        plot_xxx()

        print("\n" + "=" * 70)
        print("✅ 处理完成!")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
```

---

## 检查清单

在提交新代码前，请确认：

- [ ] 使用了 `setup_matplotlib_style()` 设置全局样式
- [ ] 所有字号使用 `PLOT_CONFIG.FONT_SIZE_*` 配置变量
- [ ] 所有颜色使用 `COLORS` 字典
- [ ] 图片保存到 `visualization_output/<category>/` 目录
- [ ] 输出格式为 SVG
- [ ] 新参数添加到 `config/global_config.py`
- [ ] 函数有完整的文档字符串
- [ ] 文件有模块级文档字符串
- [ ] 遵循命名规范

---

## 更新日志

| 日期       | 版本 | 更新内容               |
| ---------- | ---- | ---------------------- |
| 2026-01-28 | 1.0  | 初始版本，建立代码规范 |
