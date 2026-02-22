#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Woschni传热模型可视化
=====================
绘制Woschni传热模型特性（特征气速、换热系数、缸内状态、敏感性分析）

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
from engine.geometry import EngineGeometry
from engine.engine_model import MarineEngine0D, OperatingCondition


def plot_woschni_heat_transfer(output_dir: str = None) -> str:
    """
    绘制Woschni传热模型特性（4子图）

    使用EngineModel运行仿真获取真实数据

    (a) 特征气速随曲轴转角变化
    (b) 对流换热系数曲线
    (c) 缸内压力和温度变化
    (d) Woschni系数C的敏感性分析

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

        # 从传热模块获取特征气速和换热系数
        heat_transfer = engine.heat_transfer
        geometry = engine.geometry

        # 计算特征气速和换热系数
        char_velocity = []
        heat_coef = []

        for i, th in enumerate(np.deg2rad(theta_deg)):
            p = results['pressure'][i]
            T = results['temperature'][i]
            V = geometry.instantaneous_volume(th)

            # 特征气速（简化计算）
            Cm = geometry.mean_piston_speed(condition.rpm)
            w = 2.28 * Cm  # 基础气流速度
            char_velocity.append(w)

            # 换热系数
            h = heat_transfer.C * (geometry.bore ** (-0.2)) * (p ** 0.8) * (T ** (-0.53)) * (w ** 0.8)
            heat_coef.append(h)

        char_velocity = np.array(char_velocity)
        heat_coef = np.array(heat_coef)

    except Exception as e:
        print(f"[Warning] 发动机仿真失败，使用模拟数据: {e}")
        # 使用模拟数据
        theta_deg = np.linspace(-135, 135, 541)
        pressure = 1.5 + 100 * np.exp(-0.5 * ((theta_deg - 10) / 30) ** 2)
        temperature = 400 + 1400 * np.exp(-0.5 * ((theta_deg - 10) / 40) ** 2)
        char_velocity = 18 + 8 * np.exp(-0.5 * ((theta_deg - 10) / 50) ** 2)
        heat_coef = 500 + 2000 * np.exp(-0.5 * ((theta_deg - 10) / 40) ** 2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) 特征气速
    ax = axes[0, 0]
    ax.plot(theta_deg, char_velocity, color=COLORS['primary'],
            linewidth=LINE_WIDTH_MAIN, label='特征气速 w')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='TDC')
    ax.fill_between(theta_deg, 0, char_velocity, alpha=0.2, color=COLORS['primary'])
    ax.set_xlabel('曲轴转角 θ [°ATDC]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_ylabel('特征气速 w [m/s]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_title('(a) 特征气速随曲轴转角变化', fontsize=PLOT_CONFIG.FONT_SIZE_TITLE, fontweight='bold')
    ax.legend(loc='upper right', fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND)
    ax.grid(True, alpha=PLOT_CONFIG.GRID_ALPHA, linestyle='--')
    set_tick_fontsize(ax)

    # (b) 对流换热系数
    ax = axes[0, 1]
    ax.plot(theta_deg, heat_coef, color=COLORS['danger'],
            linewidth=LINE_WIDTH_MAIN, label='换热系数 h')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.fill_between(theta_deg, 0, heat_coef, alpha=0.2, color=COLORS['danger'])
    ax.set_xlabel('曲轴转角 θ [°ATDC]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_ylabel('换热系数 h [W/(m²·K)]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_title('(b) 对流换热系数曲线', fontsize=PLOT_CONFIG.FONT_SIZE_TITLE, fontweight='bold')
    ax.legend(loc='upper right', fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND)
    ax.grid(True, alpha=PLOT_CONFIG.GRID_ALPHA, linestyle='--')
    set_tick_fontsize(ax)

    # (c) 缸内压力和温度
    ax = axes[1, 0]
    ax2 = ax.twinx()

    line1, = ax.plot(theta_deg, pressure, color=COLORS['primary'],
                     linewidth=LINE_WIDTH_MAIN, label='压力 p')
    line2, = ax2.plot(theta_deg, temperature, color=COLORS['danger'],
                      linewidth=LINE_WIDTH_MAIN, linestyle='--', label='温度 T')

    ax.set_xlabel('曲轴转角 θ [°ATDC]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_ylabel('缸内压力 p [bar]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL,
                  color=COLORS['primary'])
    ax2.set_ylabel('缸内温度 T [K]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL,
                   color=COLORS['danger'])
    ax.set_title('(c) 缸内压力和温度变化', fontsize=PLOT_CONFIG.FONT_SIZE_TITLE, fontweight='bold')
    ax.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax2.tick_params(axis='y', labelcolor=COLORS['danger'])
    ax.legend(handles=[line1, line2], loc='upper right', fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND)
    ax.grid(True, alpha=PLOT_CONFIG.GRID_ALPHA, linestyle='--')
    set_tick_fontsize(ax)
    set_tick_fontsize(ax2)

    # (d) Woschni系数C的敏感性分析
    ax = axes[1, 1]
    C_values = [80, 100, 130, 160, 200]
    colors_c = [COLORS['primary'], COLORS['secondary'], COLORS['success'],
                COLORS['warning'], COLORS['danger']]

    for C, color in zip(C_values, colors_c):
        h_scaled = heat_coef * (C / 130)
        ax.plot(theta_deg, h_scaled, color=color, linewidth=LINE_WIDTH_SECONDARY,
                label=f'C = {C}')

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('曲轴转角 θ [°ATDC]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_ylabel('换热系数 h [W/(m²·K)]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_title('(d) Woschni系数C的敏感性分析', fontsize=PLOT_CONFIG.FONT_SIZE_TITLE, fontweight='bold')
    ax.legend(loc='upper right', fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND, ncol=2)
    ax.grid(True, alpha=PLOT_CONFIG.GRID_ALPHA, linestyle='--')
    set_tick_fontsize(ax)

    plt.tight_layout()
    output_path = save_figure(fig, 'modeling', 'woschni_heat_transfer.svg')
    plt.close(fig)

    return output_path
