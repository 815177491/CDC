#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三阶段校准流程图可视化
======================
绘制参数→目标→算法的对应关系，以及各阶段之间的递进策略

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

from config import COLORS, save_figure


def plot_calibration_flowchart(output_dir: str = None) -> str:
    """
    绘制三阶段分步解耦校准流程图

    展示参数→目标→算法的对应关系，以及各阶段之间的递进策略

    Args:
        output_dir: 输出目录

    Returns:
        输出文件路径
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 阶段颜色
    stage_colors = [COLORS['primary'], COLORS['secondary'], COLORS['success']]

    # 绘制三个阶段
    stages = [
        {'name': '第一阶段\n压缩过程校准', 'param': '压缩比 ε',
         'target': '$P_{comp}$', 'algo': 'L-BFGS-B', 'error': '<2%'},
        {'name': '第二阶段\n燃烧放热校准', 'param': '$θ_{inj}$, $Δθ_d$, $m_d$',
         'target': '$P_{max}$', 'algo': '差分进化', 'error': '<3%'},
        {'name': '第三阶段\n传热系数校准', 'param': 'Woschni C',
         'target': '$T_{exh}$', 'algo': 'L-BFGS-B', 'error': '<5%'},
    ]

    y_positions = [7.5, 5.0, 2.5]
    box_width = 3.0
    box_height = 1.2

    for i, (stage, y, color) in enumerate(zip(stages, y_positions, stage_colors)):
        # 阶段名称框
        stage_box = FancyBboxPatch((0.5, y - box_height/2), box_width, box_height,
                                    boxstyle="round,pad=0.05,rounding_size=0.2",
                                    facecolor=color, alpha=0.3,
                                    edgecolor=color, linewidth=2)
        ax.add_patch(stage_box)
        ax.text(0.5 + box_width/2, y, stage['name'],
                ha='center', va='center', fontsize=11, fontweight='bold')

        # 参数框
        param_box = FancyBboxPatch((4.5, y - box_height/2), box_width, box_height,
                                    boxstyle="round,pad=0.05,rounding_size=0.2",
                                    facecolor='lightyellow', alpha=0.6,
                                    edgecolor=COLORS['warning'], linewidth=1.5)
        ax.add_patch(param_box)
        ax.text(4.5 + box_width/2, y + 0.25, '优化参数',
                ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(4.5 + box_width/2, y - 0.2, stage['param'],
                ha='center', va='center', fontsize=10)

        # 目标框
        target_box = FancyBboxPatch((8.5, y - box_height/2), 2.5, box_height,
                                     boxstyle="round,pad=0.05,rounding_size=0.2",
                                     facecolor='lightcyan', alpha=0.6,
                                     edgecolor=COLORS['info'], linewidth=1.5)
        ax.add_patch(target_box)
        ax.text(8.5 + 1.25, y + 0.25, '目标变量',
                ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(8.5 + 1.25, y - 0.2, stage['target'],
                ha='center', va='center', fontsize=10)

        # 算法框
        algo_box = FancyBboxPatch((11.5, y - box_height/2), 2.5, box_height,
                                   boxstyle="round,pad=0.05,rounding_size=0.2",
                                   facecolor='lavender', alpha=0.6,
                                   edgecolor=COLORS['purple'], linewidth=1.5)
        ax.add_patch(algo_box)
        ax.text(11.5 + 1.25, y + 0.25, '优化算法',
                ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(11.5 + 1.25, y - 0.2, stage['algo'],
                ha='center', va='center', fontsize=10)

        # 误差阈值
        ax.text(15, y, f'误差{stage["error"]}',
                ha='center', va='center', fontsize=11,
                color=COLORS['success'], fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        # 绘制箭头
        ax.annotate('', xy=(4.5, y), xytext=(3.5, y),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))
        ax.annotate('', xy=(8.5, y), xytext=(7.5, y),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))
        ax.annotate('', xy=(11.5, y), xytext=(11.0, y),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))
        ax.annotate('', xy=(14.5, y), xytext=(14.0, y),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))

        # 阶段间的"锁定参数"箭头
        if i < 2:
            next_y = y_positions[i + 1]
            ax.annotate('', xy=(2.0, next_y + box_height/2 + 0.3),
                        xytext=(2.0, y - box_height/2 - 0.3),
                        arrowprops=dict(arrowstyle='->', color=stage_colors[i],
                                       lw=2.5, connectionstyle='arc3,rad=0'))
            ax.text(2.6, (y + next_y)/2, '锁定参数', fontsize=9,
                    color=stage_colors[i], fontweight='bold', rotation=90, va='center')

    # 添加标题
    ax.text(8, 9.3, '三阶段分步解耦校准流程',
            ha='center', va='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = save_figure(fig, 'modeling', 'calibration_flowchart.svg')
    plt.close(fig)

    return output_path
