# CDC 项目全局配置说明

## 概述

本项目使用统一的全局配置文件 `config/global_config.py` 管理所有参数，包括：

- 绘图参数配置
- 数据处理参数
- 发动机模型参数
- 训练参数
- 路径配置

## 绘图标准

### 字体设置

| 元素     | 字体            | 字号 |
| -------- | --------------- | ---- |
| 中文     | 宋体 (SimSun)   | -    |
| 英文     | Times New Roman | -    |
| 刻度标签 | -               | 14pt |
| 轴标签   | -               | 14pt |
| 图例     | -               | 12pt |
| 图中文字 | -               | 12pt |
| 标题     | -               | 12pt |
| 总标题   | -               | 14pt |

### 输出格式

- 格式：**SVG**
- 文字：可编辑文字（不转换为路径）
- 分辨率：150 DPI

## 使用方法

### 1. 导入配置

```python
from config import (
    PLOT_CONFIG,          # 绘图配置
    COLORS,               # 配色方案
    PATH_CONFIG,          # 路径配置
    DATA_CONFIG,          # 数据处理配置
    ENGINE_CONFIG,        # 发动机配置
    TRAINING_CONFIG,      # 训练配置
    setup_matplotlib_style,  # 设置全局matplotlib样式
    save_figure,          # 保存图形
    get_output_path,      # 获取输出路径
)
```

### 2. 设置matplotlib样式

```python
# 在绑定模块开头调用
setup_matplotlib_style()
```

### 3. 使用配色方案

```python
import matplotlib.pyplot as plt

# 使用预定义颜色
plt.plot(x, y, color=COLORS['primary'])      # 蓝色
plt.scatter(x, y, color=COLORS['success'])   # 绿色
plt.fill(x, y, color=COLORS['danger'])       # 红色
```

### 4. 保存图形

```python
# 方法1: 使用save_figure函数
save_figure(fig, 'preprocessing', 'my_plot.svg')

# 方法2: 获取路径后手动保存
output_path = get_output_path('preprocessing', 'my_plot.svg')
fig.savefig(output_path, format='svg', bbox_inches='tight')
```

## 可视化输出目录结构

所有生成的图片放在 `visualization_output/` 下对应的子文件夹中：

```
visualization_output/
├── preprocessing/      # 数据预处理可视化
├── calibration/        # 标定过程可视化
├── training/           # 训练过程可视化
├── experiments/        # 实验结果可视化
└── modeling/           # 模型相关可视化
```

## 配置类说明

### PlotConfig (绘图配置)

```python
PLOT_CONFIG.FONT_SIZE_TICK      # 刻度标签字号 (14)
PLOT_CONFIG.FONT_SIZE_LABEL     # 轴标签字号 (14)
PLOT_CONFIG.FONT_SIZE_LEGEND    # 图例字号 (12)
PLOT_CONFIG.FONT_SIZE_TEXT      # 图中文字字号 (12)
PLOT_CONFIG.FONT_SIZE_TITLE     # 标题字号 (12)
PLOT_CONFIG.FONT_SIZE_SUPTITLE  # 总标题字号 (14)
PLOT_CONFIG.OUTPUT_FORMAT       # 输出格式 ('svg')
```

### DataConfig (数据处理配置)

```python
DATA_CONFIG.STEADY_STATE_WINDOW         # 稳态检测窗口大小 (60)
DATA_CONFIG.STEADY_STATE_RPM_TOLERANCE  # RPM容差 (1.0)
DATA_CONFIG.OUTLIER_Z_THRESHOLD         # Z-score异常值阈值 (3.0)
```

### EngineConfig (发动机配置)

```python
ENGINE_CONFIG.bore               # 气缸直径 [m]
ENGINE_CONFIG.stroke             # 活塞行程 [m]
ENGINE_CONFIG.compression_ratio  # 压缩比
ENGINE_CONFIG.gamma              # 比热比
```

### TrainingConfig (训练配置)

```python
TRAINING_CONFIG.LEARNING_RATE   # 学习率
TRAINING_CONFIG.BATCH_SIZE      # 批次大小
TRAINING_CONFIG.NUM_EPOCHS      # 训练轮数
```

## 配色方案

```python
COLORS = {
    'primary': '#2E86AB',    # 主色：蓝色
    'secondary': '#A23B72',  # 次色：紫红色
    'success': '#28A745',    # 成功：绿色
    'warning': '#FFC107',    # 警告：黄色
    'danger': '#DC3545',     # 危险：红色
    'info': '#17A2B8',       # 信息：青色
    'dark': '#343A40',       # 深灰
    'light': '#F8F9FA',      # 浅灰
    'orange': '#FF8C00',     # 橙色
    'purple': '#6F42C1',     # 紫色
    'teal': '#20C997',       # 青绿色
    'pink': '#E83E8C',       # 粉色
}
```

## 新建绘图代码规范

1. **必须** 在模块开头导入并调用 `setup_matplotlib_style()`
2. **必须** 使用 `PLOT_CONFIG` 中的字号常量
3. **必须** 使用 `COLORS` 中的配色
4. **必须** 使用 `save_figure()` 或 `get_output_path()` 保存到正确目录
5. **必须** 使用 SVG 格式输出

### 模板代码

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块说明
"""

import matplotlib.pyplot as plt
from config import (
    PLOT_CONFIG, COLORS, PATH_CONFIG,
    setup_matplotlib_style, save_figure
)

# 应用全局样式
setup_matplotlib_style()


def plot_my_figure(data, output_dir=None):
    """绑定函数示例"""
    # 使用全局配置
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    legend_size = PLOT_CONFIG.FONT_SIZE_LEGEND
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE

    # 创建图形
    fig, ax = plt.subplots(figsize=PLOT_CONFIG.FIGURE_SIZE_SINGLE)

    # 绑定
    ax.plot(data['x'], data['y'], color=COLORS['primary'], linewidth=1.5)

    # 设置标签
    ax.set_xlabel('X轴标签', fontsize=label_size)
    ax.set_ylabel('Y轴标签', fontsize=label_size)
    ax.set_title('图形标题', fontsize=title_size)
    ax.legend(fontsize=legend_size)
    ax.tick_params(labelsize=tick_size)

    # 保存图形
    save_figure(fig, 'preprocessing', 'my_figure.svg')
    plt.close()

    return fig
```

## 更新日志

- **2026-01-28**: 创建全局配置模块，统一管理绘图参数和项目配置
