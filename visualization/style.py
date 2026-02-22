#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化公共样式工具
==================
提取各 *_plots.py 模块中重复使用的样式常量和辅助函数，
确保所有可视化模块的风格一致性。

Author: CDC Project
Date: 2026-02-20
"""

# 本项目模块
from config import PLOT_CONFIG


# ============================================================================
# 样式常量映射
# ============================================================================
LINE_WIDTH_MAIN = PLOT_CONFIG.LINE_WIDTH_THICK       # 主线宽 = 2.0
LINE_WIDTH_SECONDARY = PLOT_CONFIG.LINE_WIDTH         # 次线宽 = 1.5
MARKER_SIZE_LARGE = PLOT_CONFIG.MARKER_SIZE_LARGE     # 大标记 = 10
MARKER_SIZE_DEFAULT = PLOT_CONFIG.MARKER_SIZE          # 默认标记 = 6

# 学术散点图统一参数
ACADEMIC_SCATTER_PARAMS = {
    's': 100,
    'alpha': 0.8,
    'edgecolors': 'black',
    'linewidths': 1.5,
    'zorder': 5
}

# 学术参考线统一参数
ACADEMIC_REFERENCE_LINE = {
    'color': 'black',
    'linestyle': '--',
    'linewidth': 2.0,
    'zorder': 1
}

# 学术误差带统一参数
ACADEMIC_ERROR_BAND = {
    'color': 'gray',
    'alpha': 0.15
}

# 学术统计信息框统一参数
ACADEMIC_STATS_BOX = {
    'boxstyle': 'round',
    'facecolor': 'white',
    'alpha': 0.9
}


def set_tick_fontsize(ax, fontsize: int = None):
    """
    设置坐标轴刻度标签的字体大小

    Args:
        ax: matplotlib Axes 对象
        fontsize: 字号，默认使用 PLOT_CONFIG.FONT_SIZE_TICK
    """
    if fontsize is None:
        fontsize = PLOT_CONFIG.FONT_SIZE_TICK
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    for label in ax.get_xticklabels():
        label.set_fontsize(fontsize)
    for label in ax.get_yticklabels():
        label.set_fontsize(fontsize)
