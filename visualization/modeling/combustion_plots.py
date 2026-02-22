#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双Wiebe燃烧模型可视化
=====================
绘制双Wiebe燃烧模型特性（累积燃烧曲线、燃烧率分解、形状因子影响）

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
from visualization.style import set_tick_fontsize, LINE_WIDTH_MAIN, LINE_WIDTH_SECONDARY
from engine.combustion import DoublieWiebeCombustion


def plot_dual_wiebe_combustion(output_dir: str = None) -> str:
    """
    绘制双Wiebe燃烧模型特性（4子图）

    (a) 累积燃烧曲线（预混+扩散叠加）
    (b) 瞬时燃烧率分解
    (c) 形状因子m对燃烧曲线的影响
    (d) 不同m值的燃烧率峰值对比

    Args:
        output_dir: 输出目录

    Returns:
        输出文件路径
    """
    combustion = DoublieWiebeCombustion()

    # 曲轴转角范围（燃烧阶段）
    theta = np.linspace(-10, 80, 500)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) 累积燃烧曲线
    ax = axes[0, 0]
    x_total = np.array([combustion.get_burn_fraction(th) for th in theta])

    # 分别计算预混和扩散
    theta_ign = combustion.get_ignition_angle()
    x_premix = np.array([combustion.wiebe_function(th, theta_ign,
                        combustion.premix_duration, combustion.premix_shape)
                        for th in theta])
    x_diffusion = np.array([combustion.wiebe_function(th, theta_ign,
                           combustion.diffusion_duration, combustion.diffusion_shape)
                           for th in theta])

    ax.plot(theta, x_total, color=COLORS['dark'], linewidth=LINE_WIDTH_MAIN,
            label='总燃烧分数', zorder=5)
    ax.plot(theta, combustion.premix_fraction * x_premix, color=COLORS['danger'],
            linewidth=LINE_WIDTH_SECONDARY, linestyle='--', label='预混燃烧 (15%)')
    ax.plot(theta, (1-combustion.premix_fraction) * x_diffusion, color=COLORS['primary'],
            linewidth=LINE_WIDTH_SECONDARY, linestyle='-.', label='扩散燃烧 (85%)')
    ax.axvline(x=theta_ign, color='gray', linestyle=':', linewidth=1, label='着火角')
    ax.set_xlabel('曲轴转角 θ [°ATDC]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_ylabel('累积燃烧分数 $x_b$', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_title('(a) 累积燃烧曲线', fontsize=PLOT_CONFIG.FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xlim(-10, 80)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND)
    ax.grid(True, alpha=PLOT_CONFIG.GRID_ALPHA, linestyle='--')
    set_tick_fontsize(ax)

    # (b) 瞬时燃烧率分解
    ax = axes[0, 1]
    dx_total = np.array([combustion.get_burn_rate(th) for th in theta])
    dx_premix = np.array([combustion.wiebe_derivative(th, theta_ign,
                         combustion.premix_duration, combustion.premix_shape)
                         for th in theta])
    dx_diffusion = np.array([combustion.wiebe_derivative(th, theta_ign,
                            combustion.diffusion_duration, combustion.diffusion_shape)
                            for th in theta])

    ax.fill_between(theta, 0, combustion.premix_fraction * dx_premix,
                    alpha=0.4, color=COLORS['danger'], label='预混燃烧率')
    ax.fill_between(theta, combustion.premix_fraction * dx_premix,
                    combustion.premix_fraction * dx_premix + (1-combustion.premix_fraction) * dx_diffusion,
                    alpha=0.4, color=COLORS['primary'], label='扩散燃烧率')
    ax.plot(theta, dx_total, color=COLORS['dark'], linewidth=LINE_WIDTH_MAIN,
            label='总燃烧率')
    ax.set_xlabel('曲轴转角 θ [°ATDC]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_ylabel('燃烧率 $dx_b/d\\theta$ [1/°]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_title('(b) 瞬时燃烧率分解', fontsize=PLOT_CONFIG.FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xlim(-10, 80)
    ax.legend(loc='upper right', fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND)
    ax.grid(True, alpha=PLOT_CONFIG.GRID_ALPHA, linestyle='--')
    set_tick_fontsize(ax)

    # (c) 形状因子m对累积燃烧曲线的影响
    ax = axes[1, 0]
    m_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    colors_m = [COLORS['primary'], COLORS['secondary'], COLORS['success'],
                COLORS['warning'], COLORS['danger']]

    for m, color in zip(m_values, colors_m):
        x_b = np.array([combustion.wiebe_function(th, theta_ign, 55.0, m)
                       for th in theta])
        ax.plot(theta, x_b, color=color, linewidth=LINE_WIDTH_SECONDARY,
                label=f'm = {m}')

    ax.set_xlabel('曲轴转角 θ [°ATDC]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_ylabel('累积燃烧分数 $x_b$', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_title('(c) 形状因子m对燃烧曲线的影响', fontsize=PLOT_CONFIG.FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xlim(-10, 80)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND, ncol=2)
    ax.grid(True, alpha=PLOT_CONFIG.GRID_ALPHA, linestyle='--')
    set_tick_fontsize(ax)

    # (d) 不同m值的燃烧率峰值对比
    ax = axes[1, 1]
    for m, color in zip(m_values, colors_m):
        dx_b = np.array([combustion.wiebe_derivative(th, theta_ign, 55.0, m)
                        for th in theta])
        ax.plot(theta, dx_b, color=color, linewidth=LINE_WIDTH_SECONDARY,
                label=f'm = {m}')

    ax.set_xlabel('曲轴转角 θ [°ATDC]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_ylabel('燃烧率 $dx_b/d\\theta$ [1/°]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_title('(d) 不同m值的燃烧率对比', fontsize=PLOT_CONFIG.FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xlim(-10, 80)
    ax.legend(loc='upper right', fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND, ncol=2)
    ax.grid(True, alpha=PLOT_CONFIG.GRID_ALPHA, linestyle='--')
    set_tick_fontsize(ax)

    plt.tight_layout()
    output_path = save_figure(fig, 'modeling', 'dual_wiebe_combustion.svg')
    plt.close(fig)

    return output_path
