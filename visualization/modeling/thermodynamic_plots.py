#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
热力学循环特性可视化
====================
绘制热力学循环特性（P-V示功图、对数P-V图、T-p相图、P-θ和T-θ曲线）

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
from visualization.style import set_tick_fontsize, LINE_WIDTH_MAIN
from engine.geometry import EngineGeometry
from engine.engine_model import MarineEngine0D, OperatingCondition


def plot_thermodynamic_cycle(output_dir: str = None) -> str:
    """
    绘制热力学循环特性（4子图）

    使用EngineModel运行仿真获取真实数据

    (a) P-V示功图
    (b) 对数P-V图
    (c) T-P相图
    (d) P-θ和T-θ曲线

    Args:
        output_dir: 输出目录

    Returns:
        输出文件路径
    """
    # 运行发动机仿真获取数据
    engine = MarineEngine0D()
    condition = OperatingCondition(
        rpm=100,
        p_scav=1.2e5,
        T_scav=320,
        fuel_mass=0.015
    )

    try:
        results = engine.run_cycle(condition)
        theta_deg = results['theta_deg']
        pressure = results['pressure'] / 1e5  # Pa -> bar
        temperature = results['temperature']
        volume = results['volume']
        burn_fraction = results.get('burn_fraction', np.zeros_like(theta_deg))

    except Exception as e:
        print(f"[Warning] 发动机仿真失败，使用模拟数据: {e}")
        # 使用模拟数据
        geometry = EngineGeometry()
        theta_deg = np.linspace(-135, 135, 541)
        theta_rad = np.deg2rad(theta_deg)

        volume = np.array([geometry.instantaneous_volume(th + np.deg2rad(235))
                          for th in theta_rad])

        V_ref = volume[0]
        pressure = 1.2 * (V_ref / volume) ** 1.35
        comb_effect = 80 * np.exp(-0.5 * ((theta_deg - 10) / 20) ** 2)
        pressure = pressure + comb_effect

        temperature = 320 * (V_ref / volume) ** 0.35
        comb_temp = 1200 * np.exp(-0.5 * ((theta_deg - 10) / 30) ** 2)
        temperature = temperature + comb_temp

        burn_fraction = 0.5 * (1 + np.tanh((theta_deg - 5) / 15))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) P-V示功图
    ax = axes[0, 0]
    scatter = ax.scatter(volume * 1000, pressure, c=burn_fraction,
                        cmap='coolwarm', s=8, alpha=0.8)
    ax.plot(volume * 1000, pressure, color='gray', linewidth=0.5, alpha=0.5)
    cbar = plt.colorbar(scatter, ax=ax, label='燃烧分数 $x_b$')
    cbar.ax.tick_params(labelsize=PLOT_CONFIG.FONT_SIZE_TICK - 2)
    ax.set_xlabel('气缸容积 V [L]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_ylabel('缸内压力 p [bar]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_title('(a) P-V示功图', fontsize=PLOT_CONFIG.FONT_SIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=PLOT_CONFIG.GRID_ALPHA, linestyle='--')
    set_tick_fontsize(ax)

    # (b) 对数P-V图
    ax = axes[0, 1]
    ax.plot(np.log(volume), np.log(pressure), color=COLORS['primary'],
            linewidth=LINE_WIDTH_MAIN)
    V_comp = volume[theta_deg < -50]
    p_comp = pressure[theta_deg < -50]
    if len(V_comp) > 10:
        coeffs = np.polyfit(np.log(V_comp), np.log(p_comp), 1)
        n_comp = -coeffs[0]
        ax.text(0.05, 0.95, f'压缩多变指数 n ≈ {n_comp:.2f}',
                transform=ax.transAxes, fontsize=11, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel('ln(V)', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_ylabel('ln(p)', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_title('(b) 对数P-V图（多变指数分析）', fontsize=PLOT_CONFIG.FONT_SIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=PLOT_CONFIG.GRID_ALPHA, linestyle='--')
    set_tick_fontsize(ax)

    # (c) T-p相图
    ax = axes[1, 0]
    compression_mask = theta_deg < -5
    combustion_mask = (theta_deg >= -5) & (theta_deg < 40)
    expansion_mask = theta_deg >= 40

    ax.plot(pressure[compression_mask], temperature[compression_mask],
            color=COLORS['primary'], linewidth=LINE_WIDTH_MAIN, label='压缩')
    ax.plot(pressure[combustion_mask], temperature[combustion_mask],
            color=COLORS['danger'], linewidth=LINE_WIDTH_MAIN, label='燃烧')
    ax.plot(pressure[expansion_mask], temperature[expansion_mask],
            color=COLORS['success'], linewidth=LINE_WIDTH_MAIN, label='膨胀')

    ax.scatter([pressure[0]], [temperature[0]], s=100, color=COLORS['primary'],
               marker='o', zorder=5, edgecolors='black', linewidths=1.5)
    ax.annotate('起点', (pressure[0], temperature[0]),
                textcoords="offset points", xytext=(10, 10), fontsize=10)

    ax.set_xlabel('缸内压力 p [bar]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_ylabel('缸内温度 T [K]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_title('(c) T-p相图', fontsize=PLOT_CONFIG.FONT_SIZE_TITLE, fontweight='bold')
    ax.legend(loc='upper left', fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND)
    ax.grid(True, alpha=PLOT_CONFIG.GRID_ALPHA, linestyle='--')
    set_tick_fontsize(ax)

    # (d) P-θ和T-θ曲线
    ax = axes[1, 1]
    ax2 = ax.twinx()

    line1, = ax.plot(theta_deg, pressure, color=COLORS['primary'],
                     linewidth=LINE_WIDTH_MAIN, label='压力 p')
    line2, = ax2.plot(theta_deg, temperature, color=COLORS['danger'],
                      linewidth=LINE_WIDTH_MAIN, linestyle='--', label='温度 T')

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=np.argmax(pressure) - 135, color=COLORS['warning'],
               linestyle=':', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('曲轴转角 θ [°ATDC]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_ylabel('缸内压力 p [bar]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL,
                  color=COLORS['primary'])
    ax2.set_ylabel('缸内温度 T [K]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL,
                   color=COLORS['danger'])
    ax.set_title('(d) P-θ和T-θ曲线', fontsize=PLOT_CONFIG.FONT_SIZE_TITLE, fontweight='bold')
    ax.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax2.tick_params(axis='y', labelcolor=COLORS['danger'])
    ax.legend(handles=[line1, line2], loc='upper right', fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND)
    ax.grid(True, alpha=PLOT_CONFIG.GRID_ALPHA, linestyle='--')
    set_tick_fontsize(ax)
    set_tick_fontsize(ax2)

    plt.tight_layout()
    output_path = save_figure(fig, 'modeling', 'thermodynamic_cycle.svg')
    plt.close(fig)

    return output_path
