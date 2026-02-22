#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型框架流程图
==============
绘制零维模型四个子模块的层次化架构及数据流动关系

Author: CDC Project
Date: 2026-02-20
"""

import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

import matplotlib
matplotlib.use('Agg')
matplotlib.set_loglevel('error')

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

from config import PLOT_CONFIG, COLORS, save_figure
from visualization.style import set_tick_fontsize


def plot_model_framework(output_dir: str = None) -> str:
    """
    绘制零维模型框架流程图

    展示四个子模块的层次化架构及数据流动关系

    Args:
        output_dir: 输出目录

    Returns:
        输出文件路径
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 模块颜色
    colors_module = {
        'geometry': COLORS['primary'],
        'combustion': COLORS['danger'],
        'heat_transfer': COLORS['warning'],
        'solver': COLORS['success'],
        'output': COLORS.get('purple', '#6F42C1')
    }

    # 模块位置和大小
    box_width, box_height = 2.8, 1.5

    # 几何模块
    geo_box = FancyBboxPatch((0.5, 7), box_width, box_height,
                              boxstyle="round,pad=0.05,rounding_size=0.2",
                              facecolor=colors_module['geometry'], alpha=0.3,
                              edgecolor=colors_module['geometry'], linewidth=2)
    ax.add_patch(geo_box)
    ax.text(0.5 + box_width/2, 7 + box_height/2 + 0.2, '几何运动学模块',
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(0.5 + box_width/2, 7 + box_height/2 - 0.3, r'$V(\theta), dV/d\theta$',
            ha='center', va='center', fontsize=11)

    # 燃烧模块
    comb_box = FancyBboxPatch((5.1, 7), box_width, box_height,
                               boxstyle="round,pad=0.05,rounding_size=0.2",
                               facecolor=colors_module['combustion'], alpha=0.3,
                               edgecolor=colors_module['combustion'], linewidth=2)
    ax.add_patch(comb_box)
    ax.text(5.1 + box_width/2, 7 + box_height/2 + 0.2, '双Wiebe燃烧模块',
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5.1 + box_width/2, 7 + box_height/2 - 0.3, r'$dQ_{comb}/d\theta$',
            ha='center', va='center', fontsize=11)

    # 传热模块
    heat_box = FancyBboxPatch((9.7, 7), box_width, box_height,
                               boxstyle="round,pad=0.05,rounding_size=0.2",
                               facecolor=colors_module['heat_transfer'], alpha=0.3,
                               edgecolor=colors_module['heat_transfer'], linewidth=2)
    ax.add_patch(heat_box)
    ax.text(9.7 + box_width/2, 7 + box_height/2 + 0.2, 'Woschni传热模块',
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(9.7 + box_width/2, 7 + box_height/2 - 0.3, r'$dQ_{wall}/d\theta$',
            ha='center', va='center', fontsize=11)

    # 热力学求解器（中央大框）
    solver_width, solver_height = 8, 2.2
    solver_box = FancyBboxPatch((3, 3.5), solver_width, solver_height,
                                 boxstyle="round,pad=0.05,rounding_size=0.3",
                                 facecolor=colors_module['solver'], alpha=0.25,
                                 edgecolor=colors_module['solver'], linewidth=3)
    ax.add_patch(solver_box)
    ax.text(3 + solver_width/2, 3.5 + solver_height/2 + 0.4, '热力学求解器',
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(3 + solver_width/2, 3.5 + solver_height/2 - 0.3,
            r'$\frac{dT}{d\theta} = \frac{1}{mc_v}\left[\frac{dQ_{comb}}{d\theta} - \frac{dQ_{wall}}{d\theta} - p\frac{dV}{d\theta}\right]$',
            ha='center', va='center', fontsize=11)

    # 输出模块
    out_width = 6
    out_box = FancyBboxPatch((4, 0.5), out_width, 1.3,
                              boxstyle="round,pad=0.05,rounding_size=0.2",
                              facecolor=colors_module['output'], alpha=0.3,
                              edgecolor=colors_module['output'], linewidth=2)
    ax.add_patch(out_box)
    ax.text(4 + out_width/2, 1.15 + 0.2, '模型输出',
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(4 + out_width/2, 1.15 - 0.25, r'$P_{comp}$, $P_{max}$, $T_{exh}$, IMEP',
            ha='center', va='center', fontsize=11)

    # 绘制箭头连接
    # 几何 -> 求解器
    ax.annotate('', xy=(3.5, 5.7), xytext=(1.9, 7),
                arrowprops=dict(arrowstyle='->', color=colors_module['geometry'],
                               lw=2.5, connectionstyle='arc3,rad=0.2'))

    # 燃烧 -> 求解器
    ax.annotate('', xy=(7, 5.7), xytext=(6.5, 7),
                arrowprops=dict(arrowstyle='->', color=colors_module['combustion'],
                               lw=2.5))

    # 传热 -> 求解器
    ax.annotate('', xy=(10.5, 5.7), xytext=(11.1, 7),
                arrowprops=dict(arrowstyle='->', color=colors_module['heat_transfer'],
                               lw=2.5, connectionstyle='arc3,rad=-0.2'))

    # 求解器 -> 输出
    ax.annotate('', xy=(7, 1.8), xytext=(7, 3.5),
                arrowprops=dict(arrowstyle='->', color=colors_module['solver'],
                               lw=2.5))

    # 添加数据流标注
    ax.text(2.3, 6.2, r'$V, dV/d\theta$', fontsize=10, color=colors_module['geometry'],
            rotation=50)
    ax.text(6.3, 6.3, r'$dQ_{comb}$', fontsize=10, color=colors_module['combustion'])
    ax.text(10.8, 6.2, r'$dQ_{wall}$', fontsize=10, color=colors_module['heat_transfer'],
            rotation=-50)
    ax.text(7.3, 2.5, r'$T(\theta), p(\theta)$', fontsize=10, color=colors_module['solver'])

    # 添加输入参数框
    input_box = FancyBboxPatch((0.3, 3.8), 2.2, 1.6,
                                boxstyle="round,pad=0.05,rounding_size=0.15",
                                facecolor='lightgray', alpha=0.4,
                                edgecolor='gray', linewidth=1.5, linestyle='--')
    ax.add_patch(input_box)
    ax.text(1.4, 5.1, '输入工况', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(1.4, 4.6, 'RPM, $p_{scav}$', ha='center', va='center', fontsize=10)
    ax.text(1.4, 4.2, '$T_{scav}$, $m_f$', ha='center', va='center', fontsize=10)

    ax.annotate('', xy=(3, 4.6), xytext=(2.5, 4.6),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    plt.tight_layout()
    output_path = save_figure(fig, 'modeling', 'model_framework.svg')
    plt.close(fig)

    return output_path
