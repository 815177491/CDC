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

# ── 顶刊级线条样式分级 ──
LINE_WIDTH_HERO = 2.5                                # 主角线（一张图只有1-2条）
LINE_WIDTH_MAIN = PLOT_CONFIG.LINE_WIDTH_THICK       # 主线宽 = 2.0
LINE_WIDTH_SECONDARY = 1.2                           # 辅助线（从1.5降低，拉开差距）
LINE_WIDTH_GHOST = 0.6                               # 原始数据/背景线（半透明底色）

MARKER_SIZE_LARGE = PLOT_CONFIG.MARKER_SIZE_LARGE     # 大标记 = 10
MARKER_SIZE_DEFAULT = PLOT_CONFIG.MARKER_SIZE          # 默认标记 = 6

# ── 阴影带样式（置信区间/标准差）──
ACADEMIC_CONFIDENCE_BAND = dict(alpha=0.15, linewidth=0)
ACADEMIC_CONFIDENCE_BAND_DARK = dict(alpha=0.25, linewidth=0)

# ── 高级标记样式 ──
MARKER_STYLE_HERO = dict(marker='o', markersize=8, markeredgecolor='white',
                         markeredgewidth=1.5, zorder=10)
MARKER_STYLE_SECONDARY = dict(marker='s', markersize=5, markeredgecolor='white',
                              markeredgewidth=1.0, zorder=8)

# ── 图形背景与边框 ──
SPINE_STYLE = dict(linewidth=1.2, color='#333333')
ACADEMIC_GRID = dict(alpha=0.15, linestyle='-', linewidth=0.5, color='#CCCCCC')

# ── inset 子图样式 ──
INSET_BOX_STYLE = dict(boxstyle='round,pad=0.02', facecolor='white',
                       edgecolor='#AAAAAA', linewidth=0.8, alpha=0.95)

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
