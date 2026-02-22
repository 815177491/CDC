#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配气正时可视化
==============
绘制二冲程配气正时圆形图

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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

from config import PLOT_CONFIG, COLORS, save_figure
from visualization.style import set_tick_fontsize


def plot_valve_timing_diagram(output_dir: str = None) -> str:
    """
    绘制二冲程配气正时圆形图

    展示排气阀/扫气口开闭时序，以及各工作阶段的角度范围

    Args:
        output_dir: 输出目录

    Returns:
        输出文件路径
    """
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))

    # 配气正时参数 (deg ATDC)
    evo = 130   # 排气阀开启
    evc = 230   # 排气阀关闭
    spo = 125   # 扫气口开启
    spc = 235   # 扫气口关闭

    def deg_to_rad(deg):
        return np.deg2rad(deg)

    # ========== 使用bar绘制完整的扇形区域 ==========
    n_segments = 360
    theta_segments = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
    width = 2*np.pi / n_segments

    colors_segments = []
    for i in range(n_segments):
        angle_deg = i
        if 235 <= angle_deg < 360 or angle_deg == 0:
            colors_segments.append(COLORS['primary'])
        elif 0 < angle_deg <= 125:
            colors_segments.append(COLORS['danger'])
        elif 125 < angle_deg <= 130:
            colors_segments.append(COLORS['danger'])
        elif 130 < angle_deg < 235:
            colors_segments.append(COLORS['success'])
        else:
            colors_segments.append(COLORS['primary'])

    # 绘制底层扇形区域
    bars = ax.bar(theta_segments, [0.85]*n_segments, width=width, bottom=0.15,
                  color=colors_segments, alpha=0.35, edgecolor='none')

    # ========== 绘制排气阀开启弧线 (130° -> 230°) ==========
    theta_ev = np.linspace(deg_to_rad(evo), deg_to_rad(evc), 100)
    ax.plot(theta_ev, [0.72]*len(theta_ev), color=COLORS['warning'],
            linewidth=14, solid_capstyle='butt', alpha=0.9, zorder=5)
    mid_ev = deg_to_rad((evo + evc) / 2)
    ax.text(mid_ev, 0.72, '排气阀开启', fontsize=10, ha='center', va='center',
            color='white', fontweight='bold', zorder=6)

    # ========== 绘制扫气口开启弧线 (125° -> 235°) ==========
    theta_sp = np.linspace(deg_to_rad(spo), deg_to_rad(spc), 100)
    ax.plot(theta_sp, [0.58]*len(theta_sp), color=COLORS['info'],
            linewidth=14, solid_capstyle='butt', alpha=0.9, zorder=5)
    mid_sp = deg_to_rad((spo + spc) / 2)
    ax.text(mid_sp, 0.58, '扫气口开启', fontsize=10, ha='center', va='center',
            color='white', fontweight='bold', zorder=6)

    # ========== 绘制喷油期 (约-2° 到 40°，即358°到40°) ==========
    theta_inj1 = np.linspace(deg_to_rad(358), deg_to_rad(360), 20)
    theta_inj2 = np.linspace(deg_to_rad(0), deg_to_rad(40), 40)
    ax.plot(theta_inj1, [0.45]*len(theta_inj1), color=COLORS['secondary'],
            linewidth=12, solid_capstyle='butt', alpha=0.85, zorder=5)
    ax.plot(theta_inj2, [0.45]*len(theta_inj2), color=COLORS['secondary'],
            linewidth=12, solid_capstyle='butt', alpha=0.85, zorder=5)
    ax.text(deg_to_rad(20), 0.45, '喷油', fontsize=9, ha='center', va='center',
            color='white', fontweight='bold', zorder=6)

    # ========== 绘制关键角度标注线和标签 ==========
    key_angles = [
        (0, 'TDC\n(上止点)', COLORS['dark'], 1.02, 12),
        (180, 'BDC\n(下止点)', COLORS['dark'], 1.02, 12),
        (90, '90°', COLORS['dark'], 0.98, 10),
        (270, '270°', COLORS['dark'], 0.98, 10),
        (evo, f'EVO {evo}°', COLORS['warning'], 0.95, 9),
        (evc, f'EVC {evc}°', COLORS['warning'], 0.95, 9),
        (spo, f'SPO {spo}°', COLORS['info'], 0.95, 9),
        (spc, f'SPC {spc}°', COLORS['info'], 0.95, 9),
    ]

    for angle, label, color, line_len, fontsize in key_angles:
        theta = deg_to_rad(angle)
        ax.plot([theta, theta], [0.15, line_len], color=color,
                linewidth=1.5, linestyle='--', alpha=0.6, zorder=3)
        ax.text(theta, line_len + 0.06, label, ha='center', va='center',
                fontsize=fontsize, color=color, fontweight='bold', zorder=7)

    # ========== 绘制中心圆 ==========
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax.fill_between(theta_circle, 0, 0.15, color=COLORS['dark'], alpha=0.2, zorder=4)
    ax.plot(theta_circle, [0.15]*len(theta_circle), color=COLORS['dark'],
            linewidth=2, zorder=4)

    # 中心文字
    ax.text(0, 0.02, '曲轴', ha='center', va='center', fontsize=11,
            fontweight='bold', color=COLORS['dark'], zorder=8)

    # ========== 绘制旋转方向箭头 ==========
    ax.annotate('', xy=(deg_to_rad(330), 0.97), xytext=(deg_to_rad(300), 0.97),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2.5),
                zorder=7)
    ax.text(deg_to_rad(315), 1.04, '旋转方向', ha='center', va='center',
            fontsize=9, color=COLORS['dark'])

    # ========== 在扇形区域添加阶段名称 ==========
    ax.text(deg_to_rad(297), 0.5, '压缩\n(125°)', ha='center', va='center',
            fontsize=11, color=COLORS['primary'], fontweight='bold')
    ax.text(deg_to_rad(60), 0.5, '燃烧膨胀\n(130°)', ha='center', va='center',
            fontsize=11, color=COLORS['danger'], fontweight='bold')
    ax.text(deg_to_rad(180), 0.35, '换气扫气\n(110°)', ha='center', va='center',
            fontsize=11, color=COLORS['success'], fontweight='bold')

    # ========== 设置极坐标属性 ==========
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([])

    # 设置角度刻度
    angle_ticks = np.arange(0, 360, 30)
    ax.set_xticks([deg_to_rad(a) for a in angle_ticks])
    ax.set_xticklabels([f'{a}°' for a in angle_ticks], fontsize=PLOT_CONFIG.FONT_SIZE_TICK)

    # ========== 添加图例 ==========
    legend_elements = [
        Patch(facecolor=COLORS['primary'], alpha=0.35, label='压缩 (235°→360°)'),
        Patch(facecolor=COLORS['danger'], alpha=0.35, label='燃烧膨胀 (0°→130°)'),
        Patch(facecolor=COLORS['success'], alpha=0.35, label='换气扫气 (130°→235°)'),
        Line2D([0], [0], color=COLORS['warning'], linewidth=10, label=f'排气阀开启 ({evo}°→{evc}°)'),
        Line2D([0], [0], color=COLORS['info'], linewidth=10, label=f'扫气口开启 ({spo}°→{spc}°)'),
        Line2D([0], [0], color=COLORS['secondary'], linewidth=10, label='喷油期 (358°→40°)'),
    ]

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1.0),
              fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND, framealpha=0.9)

    # ========== 添加标题 ==========
    plt.title('二冲程船用柴油机配气正时圆形图', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
              fontweight='bold', pad=25)

    plt.tight_layout()
    output_path = save_figure(fig, 'modeling', 'valve_timing_diagram.svg')
    plt.close(fig)

    return output_path
