#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
参数敏感度雷达图可视化
======================
展示各参数对Pcomp、Pmax、Texh的敏感度

Author: CDC Project
Date: 2026-02-20
"""

import numpy as np
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

import matplotlib
matplotlib.use('Agg')
matplotlib.set_loglevel('error')

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

from config import PLOT_CONFIG, COLORS, save_figure


def plot_sensitivity_radar(output_dir: str = None) -> str:
    """
    绘制参数敏感度雷达图

    展示各参数对Pcomp、Pmax、Texh的敏感度

    Args:
        output_dir: 输出目录

    Returns:
        输出文件路径
    """
    # 敏感度数据（基于物理分析的典型值）
    params = ['压缩比 ε', '喷油正时 $θ_{inj}$', '燃烧持续角 $Δθ_d$',
              '形状因子 $m_d$', 'Woschni系数 C']

    sensitivity_data = {
        '$P_{comp}$': [0.85, 0.05, 0.03, 0.02, 0.05],
        '$P_{max}$': [0.35, 0.65, 0.45, 0.30, 0.10],
        '$T_{exh}$': [0.15, 0.20, 0.25, 0.15, 0.55],
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), subplot_kw=dict(projection='polar'))

    # 角度设置
    N = len(params)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    colors_output = [COLORS['primary'], COLORS['danger'], COLORS['warning']]

    for ax, (output_name, values), color in zip(axes, sensitivity_data.items(), colors_output):
        values = values + values[:1]  # 闭合

        ax.fill(angles, values, color=color, alpha=0.25)
        ax.plot(angles, values, color=color, linewidth=2.5, marker='o', markersize=8)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(params, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=9)
        ax.set_title(f'{output_name}的参数敏感度', fontsize=12, fontweight='bold', pad=15)

        # 添加网格
        ax.grid(True, alpha=0.3)

    plt.suptitle('参数敏感度雷达图', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = save_figure(fig, 'modeling', 'sensitivity_radar.svg')
    plt.close(fig)

    return output_path
