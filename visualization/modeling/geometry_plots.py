#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
曲柄连杆几何与运动学可视化
==========================
绘制曲柄连杆机构几何示意图与运动学特性曲线

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
from matplotlib.patches import FancyBboxPatch, Circle, Arc

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

from config import PLOT_CONFIG, COLORS, ENGINE_CONFIG, save_figure
from visualization.style import set_tick_fontsize, LINE_WIDTH_MAIN, LINE_WIDTH_SECONDARY
from engine.geometry import EngineGeometry


def plot_crank_geometry_schematic(output_dir: str = None) -> str:
    """
    绘制曲柄连杆机构几何示意图

    展示不同曲轴转角下的活塞位置、连杆姿态及关键几何参数标注

    Args:
        output_dir: 输出目录

    Returns:
        输出文件路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))

    # 几何参数
    R = 1.0  # 曲柄半径（归一化）
    L = 4.0  # 连杆长度

    angles = [0, 90, 180]  # TDC, 90°, BDC
    titles = ['(a) 上止点 (TDC, θ=0°)', '(b) 曲轴转角 θ=90°', '(c) 下止点 (BDC, θ=180°)']

    for idx, (theta_deg, title) in enumerate(zip(angles, titles)):
        ax = axes[idx]
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2, 7)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=PLOT_CONFIG.FONT_SIZE_TITLE, fontweight='bold', pad=10)

        theta = np.radians(theta_deg)

        # 曲柄销位置
        crank_x = R * np.sin(theta)
        crank_y = -R * np.cos(theta)

        # 活塞位置（相对于TDC）
        lambda_cr = R / L
        x_piston = R * ((1 - np.cos(theta)) +
                        (1/lambda_cr) * (1 - np.sqrt(1 - (lambda_cr * np.sin(theta))**2)))
        piston_y = L + R - x_piston  # 活塞底部位置

        # 绘制曲轴中心
        ax.plot(0, 0, 'ko', markersize=10, zorder=10)
        ax.add_patch(Circle((0, 0), 0.15, color='black', zorder=11))

        # 绘制曲柄
        ax.plot([0, crank_x], [0, crank_y], color=COLORS['primary'],
                linewidth=8, solid_capstyle='round', zorder=5)

        # 绘制曲柄销
        ax.add_patch(Circle((crank_x, crank_y), 0.12, color=COLORS['primary'], zorder=6))

        # 绘制连杆
        ax.plot([crank_x, 0], [crank_y, piston_y], color=COLORS['secondary'],
                linewidth=6, solid_capstyle='round', zorder=4)

        # 绘制活塞
        piston_width = 1.2
        piston_height = 0.6
        piston = FancyBboxPatch((-piston_width/2, piston_y), piston_width, piston_height,
                                 boxstyle="round,pad=0.02,rounding_size=0.05",
                                 facecolor=COLORS['dark'], alpha=0.8,
                                 edgecolor='black', linewidth=2, zorder=7)
        ax.add_patch(piston)

        # 绘制活塞销
        ax.add_patch(Circle((0, piston_y), 0.1, color='white', edgecolor='black',
                            linewidth=1.5, zorder=8))

        # 绘制气缸壁
        cylinder_height = 6
        ax.plot([-piston_width/2 - 0.1, -piston_width/2 - 0.1],
                [-1, cylinder_height], 'k-', linewidth=2.5)
        ax.plot([piston_width/2 + 0.1, piston_width/2 + 0.1],
                [-1, cylinder_height], 'k-', linewidth=2.5)

        # 气缸顶部
        ax.plot([-piston_width/2 - 0.1, piston_width/2 + 0.1],
                [cylinder_height, cylinder_height], 'k-', linewidth=2.5)

        # 标注几何参数
        if idx == 0:  # 在第一个图上详细标注几何参数和部件名称
            # 曲柄半径R
            ax.annotate('', xy=(0, -R), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='<->', color=COLORS['danger'], lw=1.5))
            ax.text(0.3, -R/2, 'R', fontsize=12, color=COLORS['danger'], fontweight='bold')

            # 连杆长度L
            ax.annotate('', xy=(-0.5, piston_y), xytext=(-0.5, crank_y),
                        arrowprops=dict(arrowstyle='<->', color=COLORS['success'], lw=1.5))
            ax.text(-0.9, (piston_y + crank_y)/2, 'L', fontsize=12,
                    color=COLORS['success'], fontweight='bold')

            # 部件名称标注
            ax.text(1.5, 0, '曲轴中心', fontsize=10, ha='left', va='center', color='black')
            ax.text(1.5, crank_y, '曲柄销', fontsize=10, ha='left', va='center', color=COLORS['primary'])
            ax.text(1.5, piston_y + piston_height/2, '活塞', fontsize=10, ha='left', va='center', color=COLORS['dark'])
            ax.text(1.0, (piston_y + crank_y)/2 + 0.3, '连杆', fontsize=10, ha='left', va='center',
                    color=COLORS['secondary'], rotation=80)
            ax.text(-2.0, cylinder_height - 0.5, '气缸盖', fontsize=10, ha='center', va='center', color='black')
            ax.text(-2.0, (cylinder_height + piston_y)/2, '气缸壁', fontsize=10, ha='center', va='center',
                    color='black', rotation=90)

            # 燃烧室标注
            ax.text(0, cylinder_height - 0.4, '燃烧室', fontsize=9, ha='center', va='top',
                    color=COLORS['danger'], fontstyle='italic')

            # 连杆比公式
            ax.text(0, -1.8, r'连杆比 $\lambda = R/L$', fontsize=10, ha='center', va='center',
                    color='gray', fontstyle='italic')

        if idx == 1:  # 在第二个图上标注活塞位移
            # 活塞位移x标注
            ax.annotate('', xy=(1.5, piston_y), xytext=(1.5, L + R),
                        arrowprops=dict(arrowstyle='<->', color=COLORS['info'], lw=1.5))
            ax.text(1.7, (piston_y + L + R)/2, 'x(θ)', fontsize=11,
                    color=COLORS['info'], fontweight='bold')

            # TDC参考线
            ax.plot([-1.5, 1.5], [L + R, L + R], 'g--', linewidth=1, alpha=0.7)
            ax.text(-1.8, L + R, 'TDC', fontsize=9, ha='right', va='center', color=COLORS['success'])

            # 部件名称标注
            ax.text(-2.0, piston_y + piston_height/2, '活塞销', fontsize=10, ha='center', va='center', color='gray')

            # 活塞位移公式
            ax.text(0, -1.8, r'$x = R[(1-\cos\theta) + \frac{1}{\lambda}(1-\sqrt{1-\lambda^2\sin^2\theta})]$',
                    fontsize=9, ha='center', va='center', color='gray')

        if idx == 2:  # 在第三个图上标注行程
            # 行程S标注
            S_stroke = 2 * R
            ax.annotate('', xy=(1.8, L + R - S_stroke), xytext=(1.8, L + R),
                        arrowprops=dict(arrowstyle='<->', color=COLORS.get('purple', '#6F42C1'), lw=2))
            ax.text(2.0, L + R - R, 'S=2R', fontsize=11,
                    color=COLORS.get('purple', '#6F42C1'), fontweight='bold')

            # TDC和BDC参考线
            ax.plot([-1.5, 2.2], [L + R, L + R], 'g--', linewidth=1, alpha=0.7)
            ax.text(-1.8, L + R, 'TDC', fontsize=9, ha='right', va='center', color=COLORS['success'])
            ax.plot([-1.5, 2.2], [L - R, L - R], 'r--', linewidth=1, alpha=0.7)
            ax.text(-1.8, L - R, 'BDC', fontsize=9, ha='right', va='center', color=COLORS['danger'])

            # 余隙容积和工作容积标注
            ax.text(0, cylinder_height - 0.4, r'$V_c$ (余隙容积)', fontsize=9, ha='center', va='top',
                    color=COLORS['danger'], fontstyle='italic')
            ax.text(0, (L + R + L - R)/2, r'$V_s$ (工作容积)', fontsize=9, ha='center', va='center',
                    color=COLORS['primary'], fontstyle='italic')

            # 压缩比公式
            ax.text(0, -1.8, r'压缩比 $\varepsilon = \frac{V_c + V_s}{V_c}$',
                    fontsize=10, ha='center', va='center', color='gray')

        # 标注曲轴转角
        if theta_deg > 0:
            arc = Arc((0, 0), 1.5, 1.5, angle=90, theta1=-theta_deg, theta2=0,
                      color=COLORS['warning'], linewidth=2)
            ax.add_patch(arc)
            ax.text(0.5, 0.5, f'θ={theta_deg}°', fontsize=11, color=COLORS['warning'])

        # 添加参考线
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    output_path = save_figure(fig, 'modeling', 'crank_geometry_schematic.svg')
    plt.close(fig)

    return output_path


def plot_crank_kinematics(output_dir: str = None) -> str:
    """
    绘制曲柄连杆运动学特性（4子图）

    (a) 活塞位移曲线
    (b) 活塞速度曲线
    (c) 气缸容积曲线
    (d) 容积变化率

    Args:
        output_dir: 输出目录

    Returns:
        输出文件路径
    """
    # 使用真实发动机参数
    geometry = EngineGeometry(
        bore=ENGINE_CONFIG.bore,
        stroke=ENGINE_CONFIG.stroke,
        compression_ratio=ENGINE_CONFIG.compression_ratio,
        con_rod_ratio=ENGINE_CONFIG.con_rod_ratio
    )

    # 计算运动学数据
    theta_deg = np.linspace(0, 360, 721)
    theta_rad = np.deg2rad(theta_deg)

    # 活塞位移 [m]
    displacement = np.array([geometry.piston_position(th) for th in theta_rad])

    # 活塞速度 [m/s] (假设100 RPM)
    rpm = 100
    omega = 2 * np.pi * rpm / 60  # rad/s
    R = geometry.crank_radius
    L = geometry.con_rod_length
    lambda_cr = R / L

    velocity = omega * R * (np.sin(theta_rad) +
                            (lambda_cr * np.sin(theta_rad) * np.cos(theta_rad)) /
                            np.sqrt(1 - (lambda_cr * np.sin(theta_rad))**2))

    # 气缸容积 [m³]
    volume = np.array([geometry.instantaneous_volume(th) for th in theta_rad])

    # 容积变化率 [m³/rad]
    dV_dtheta = np.array([geometry.volume_derivative(th) for th in theta_rad])

    # 创建4子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) 活塞位移
    ax = axes[0, 0]
    ax.plot(theta_deg, displacement * 1000, color=COLORS['primary'],
            linewidth=LINE_WIDTH_MAIN, label='活塞位移')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='TDC')
    ax.axvline(x=180, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='BDC')
    ax.set_xlabel('曲轴转角 θ [°]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_ylabel('活塞位移 x [mm]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_title('(a) 活塞位移曲线', fontsize=PLOT_CONFIG.FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.legend(loc='upper right', fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND)
    ax.grid(True, alpha=PLOT_CONFIG.GRID_ALPHA, linestyle='--')
    set_tick_fontsize(ax)

    # (b) 活塞速度
    ax = axes[0, 1]
    ax.plot(theta_deg, velocity, color=COLORS['secondary'],
            linewidth=LINE_WIDTH_MAIN, label='活塞速度')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    ax.axvline(x=90, color=COLORS['warning'], linestyle='--', linewidth=1.5,
               alpha=0.7, label='最大速度附近')
    ax.set_xlabel('曲轴转角 θ [°]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_ylabel('活塞速度 v [m/s]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_title('(b) 活塞速度曲线 (100 RPM)', fontsize=PLOT_CONFIG.FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.legend(loc='upper right', fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND)
    ax.grid(True, alpha=PLOT_CONFIG.GRID_ALPHA, linestyle='--')
    set_tick_fontsize(ax)

    # (c) 气缸容积
    ax = axes[1, 0]
    ax.plot(theta_deg, volume, color=COLORS['success'],
            linewidth=LINE_WIDTH_MAIN, label='气缸容积')
    ax.axhline(y=geometry.clearance_volume, color=COLORS['danger'],
               linestyle='--', linewidth=1.5, label=f'余隙容积 Vc')
    ax.fill_between(theta_deg, geometry.clearance_volume, volume,
                    alpha=0.2, color=COLORS['success'])
    ax.set_xlabel('曲轴转角 θ [°]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_ylabel('气缸容积 V [m³]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_title('(c) 气缸容积曲线', fontsize=PLOT_CONFIG.FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.legend(loc='upper right', fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND)
    ax.grid(True, alpha=PLOT_CONFIG.GRID_ALPHA, linestyle='--')
    set_tick_fontsize(ax)

    # (d) 容积变化率
    ax = axes[1, 1]
    ax.plot(theta_deg, dV_dtheta, color=COLORS['warning'],
            linewidth=LINE_WIDTH_MAIN, label='dV/dθ')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    ax.fill_between(theta_deg, 0, dV_dtheta, where=(dV_dtheta > 0),
                    alpha=0.3, color=COLORS['danger'], label='膨胀')
    ax.fill_between(theta_deg, 0, dV_dtheta, where=(dV_dtheta < 0),
                    alpha=0.3, color=COLORS['primary'], label='压缩')
    ax.set_xlabel('曲轴转角 θ [°]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_ylabel('容积变化率 dV/dθ [m³/rad]', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_title('(d) 容积变化率', fontsize=PLOT_CONFIG.FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.legend(loc='upper right', fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND)
    ax.grid(True, alpha=PLOT_CONFIG.GRID_ALPHA, linestyle='--')
    set_tick_fontsize(ax)

    plt.tight_layout()
    output_path = save_figure(fig, 'modeling', 'crank_kinematics.svg')
    plt.close(fig)

    return output_path
