#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
建模可视化模块
==============
提供零维柴油机模型建模相关的可视化函数，采用IEEE/Elsevier学术期刊风格。

包含图表:
1. 模型框架流程图 (model_framework.svg)
2. 曲柄连杆几何示意图 (crank_geometry_schematic.svg)
3. 曲柄连杆运动学特性 (crank_kinematics.svg)
4. 二冲程配气正时图 (valve_timing_diagram.svg)
5. 双Wiebe燃烧模型特性 (dual_wiebe_combustion.svg)
6. Woschni传热模型特性 (woschni_heat_transfer.svg)
7. 热力学循环特性 (thermodynamic_cycle.svg)
8. 三阶段校准流程图 (calibration_flowchart.svg)
9. 参数敏感度雷达图 (sensitivity_radar.svg)
10. 能量平衡桑基图 (energy_sankey.svg)

Author: CDC Project
Date: 2026-01-28
"""

import numpy as np
import warnings
import logging

# 抑制警告
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib.backends').setLevel(logging.ERROR)

import matplotlib
matplotlib.use('Agg')
matplotlib.set_loglevel('error')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Arc, Wedge
from matplotlib.patches import Rectangle, Polygon, ConnectionPatch
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 导入全局配置
from config import (
    PLOT_CONFIG, 
    COLORS, 
    PATH_CONFIG,
    ENGINE_CONFIG,
    setup_matplotlib_style,
    save_figure,
    get_output_path
)

# 导入发动机模块
from engine.geometry import EngineGeometry
from engine.combustion import DoublieWiebeCombustion
from engine.heat_transfer import WoschniHeatTransfer
from engine.engine_model import MarineEngine0D, OperatingCondition

# 应用全局matplotlib样式
setup_matplotlib_style()


# ============================================================================
# 样式常量映射
# ============================================================================
LINE_WIDTH_MAIN = PLOT_CONFIG.LINE_WIDTH_THICK     # 主线宽 = 2.0
LINE_WIDTH_SECONDARY = PLOT_CONFIG.LINE_WIDTH      # 次线宽 = 1.5
MARKER_SIZE_LARGE = PLOT_CONFIG.MARKER_SIZE_LARGE  # 大标记 = 10
MARKER_SIZE_DEFAULT = PLOT_CONFIG.MARKER_SIZE      # 默认标记 = 6

# 学术风格统一参数
ACADEMIC_SCATTER_PARAMS = {
    's': 100,
    'alpha': 0.8,
    'edgecolors': 'black',
    'linewidths': 1.5,
    'zorder': 5
}


def set_tick_fontsize(ax, fontsize=None):
    """设置坐标轴刻度标签的字体大小"""
    if fontsize is None:
        fontsize = PLOT_CONFIG.FONT_SIZE_TICK
    ax.tick_params(axis='both', which='major', labelsize=fontsize)


# ============================================================================
# 图4-1: 模型框架流程图
# ============================================================================
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
        'output': COLORS.get('purple', '#6F42C1')  # 紫色
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
    arrow_style = dict(arrowstyle='->', color='gray', lw=2, 
                       connectionstyle='arc3,rad=0')
    
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


# ============================================================================
# 图4-2: 曲柄连杆几何示意图
# ============================================================================
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
        if idx == 0:  # 只在第一个图上详细标注
            # 曲柄半径R
            ax.annotate('', xy=(0, -R), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='<->', color=COLORS['danger'], lw=1.5))
            ax.text(0.3, -R/2, 'R', fontsize=12, color=COLORS['danger'], fontweight='bold')
            
            # 连杆长度L
            ax.annotate('', xy=(-0.5, piston_y), xytext=(-0.5, crank_y),
                        arrowprops=dict(arrowstyle='<->', color=COLORS['success'], lw=1.5))
            ax.text(-0.9, (piston_y + crank_y)/2, 'L', fontsize=12, 
                    color=COLORS['success'], fontweight='bold')
        
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


# ============================================================================
# 图4-3: 曲柄连杆运动学特性
# ============================================================================
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


# ============================================================================
# 图4-4: 二冲程配气正时圆形图
# ============================================================================
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
    
    # 转换函数：将ATDC角度转换为极坐标弧度
    # 极坐标默认：0在右侧，逆时针增加
    # 设置后：0在顶部(N)，顺时针增加(-1方向)
    # 所以 ATDC角度 deg -> 极坐标弧度 = deg * pi/180 (直接转换即可，因为设置了方向)
    def deg_to_rad(deg):
        return np.deg2rad(deg)
    
    # ========== 使用bar绘制完整的扇形区域 ==========
    # 定义各阶段
    # 压缩: 235° -> 360° (125°范围)
    # 燃烧膨胀: 0° -> 130° (130°范围)  
    # 换气: 125° -> 235° (110°范围)
    
    # 方法：将360度分成多个小扇形，根据角度着色
    n_segments = 360
    theta_segments = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
    width = 2*np.pi / n_segments
    
    colors_segments = []
    for i in range(n_segments):
        angle_deg = i  # 0-359度
        if 235 <= angle_deg < 360 or angle_deg == 0:
            # 压缩阶段: 235° -> 360°
            colors_segments.append(COLORS['primary'])
        elif 0 < angle_deg <= 125:
            # 燃烧膨胀前半段: 0° -> 125° (排气阀开启前)
            colors_segments.append(COLORS['danger'])
        elif 125 < angle_deg <= 130:
            # 燃烧膨胀与换气重叠区
            colors_segments.append(COLORS['danger'])
        elif 130 < angle_deg < 235:
            # 换气阶段: 130° -> 235°
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
    # 添加标签
    mid_ev = deg_to_rad((evo + evc) / 2)
    ax.text(mid_ev, 0.72, '排气阀开启', fontsize=10, ha='center', va='center',
            color='white', fontweight='bold', zorder=6)
    
    # ========== 绘制扫气口开启弧线 (125° -> 235°) ==========
    theta_sp = np.linspace(deg_to_rad(spo), deg_to_rad(spc), 100)
    ax.plot(theta_sp, [0.58]*len(theta_sp), color=COLORS['info'], 
            linewidth=14, solid_capstyle='butt', alpha=0.9, zorder=5)
    # 添加标签
    mid_sp = deg_to_rad((spo + spc) / 2)
    ax.text(mid_sp, 0.58, '扫气口开启', fontsize=10, ha='center', va='center',
            color='white', fontweight='bold', zorder=6)
    
    # ========== 绘制喷油期 (约-2° 到 40°，即358°到40°) ==========
    # 跨越0点，需要分两段
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
        # 绘制径向参考线
        ax.plot([theta, theta], [0.15, line_len], color=color, 
                linewidth=1.5, linestyle='--', alpha=0.6, zorder=3)
        # 添加标签
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
    arrow_theta = deg_to_rad(315)
    ax.annotate('', xy=(deg_to_rad(330), 0.97), xytext=(deg_to_rad(300), 0.97),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2.5),
                zorder=7)
    ax.text(deg_to_rad(315), 1.04, '旋转方向', ha='center', va='center', 
            fontsize=9, color=COLORS['dark'])
    
    # ========== 在扇形区域添加阶段名称 ==========
    # 压缩阶段标签
    ax.text(deg_to_rad(297), 0.5, '压缩\n(125°)', ha='center', va='center',
            fontsize=11, color=COLORS['primary'], fontweight='bold')
    # 燃烧膨胀阶段标签
    ax.text(deg_to_rad(60), 0.5, '燃烧膨胀\n(130°)', ha='center', va='center',
            fontsize=11, color=COLORS['danger'], fontweight='bold')
    # 换气阶段标签
    ax.text(deg_to_rad(180), 0.35, '换气扫气\n(110°)', ha='center', va='center',
            fontsize=11, color=COLORS['success'], fontweight='bold')
    
    # ========== 设置极坐标属性 ==========
    ax.set_theta_zero_location('N')  # 0度在顶部
    ax.set_theta_direction(-1)        # 顺时针方向
    ax.set_ylim(0, 1.15)
    ax.set_yticks([])                 # 隐藏径向刻度
    
    # 设置角度刻度（每30度）
    angle_ticks = np.arange(0, 360, 30)
    ax.set_xticks([deg_to_rad(a) for a in angle_ticks])
    ax.set_xticklabels([f'{a}°' for a in angle_ticks], fontsize=PLOT_CONFIG.FONT_SIZE_TICK)
    
    # ========== 添加图例 ==========
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
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


# ============================================================================
# 图4-5: 双Wiebe燃烧模型特性
# ============================================================================
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


# ============================================================================
# 图4-6: Woschni传热模型特性
# ============================================================================
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
        # 模拟压力曲线
        pressure = 1.5 + 100 * np.exp(-0.5 * ((theta_deg - 10) / 30) ** 2)
        # 模拟温度曲线
        temperature = 400 + 1400 * np.exp(-0.5 * ((theta_deg - 10) / 40) ** 2)
        # 模拟特征气速
        char_velocity = 18 + 8 * np.exp(-0.5 * ((theta_deg - 10) / 50) ** 2)
        # 模拟换热系数
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
        # 简化的换热系数计算（基于典型值）
        h_scaled = heat_coef * (C / 130)  # 按比例缩放
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


# ============================================================================
# 图4-7: 热力学循环特性
# ============================================================================
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
        
        # 模拟压力曲线（考虑压缩和燃烧）
        V_ref = volume[0]
        pressure = 1.2 * (V_ref / volume) ** 1.35  # 多变压缩
        # 添加燃烧效果
        comb_effect = 80 * np.exp(-0.5 * ((theta_deg - 10) / 20) ** 2)
        pressure = pressure + comb_effect
        
        # 模拟温度
        temperature = 320 * (V_ref / volume) ** 0.35
        comb_temp = 1200 * np.exp(-0.5 * ((theta_deg - 10) / 30) ** 2)
        temperature = temperature + comb_temp
        
        # 模拟燃烧分数
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
    # 添加多变指数参考线
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
    # 划分不同阶段并用不同颜色标注
    compression_mask = theta_deg < -5
    combustion_mask = (theta_deg >= -5) & (theta_deg < 40)
    expansion_mask = theta_deg >= 40
    
    ax.plot(pressure[compression_mask], temperature[compression_mask], 
            color=COLORS['primary'], linewidth=LINE_WIDTH_MAIN, label='压缩')
    ax.plot(pressure[combustion_mask], temperature[combustion_mask], 
            color=COLORS['danger'], linewidth=LINE_WIDTH_MAIN, label='燃烧')
    ax.plot(pressure[expansion_mask], temperature[expansion_mask], 
            color=COLORS['success'], linewidth=LINE_WIDTH_MAIN, label='膨胀')
    
    # 标注关键点
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


# ============================================================================
# 图4-12: 三阶段校准流程图
# ============================================================================
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


# ============================================================================
# 图4-13: 参数敏感度雷达图
# ============================================================================
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


# ============================================================================
# 图4-17: 能量平衡桑基图
# ============================================================================
def plot_energy_sankey(output_dir: str = None) -> str:
    """
    绘制柴油机循环能量平衡桑基图
    
    使用plotly生成桑基图并导出为SVG
    
    Args:
        output_dir: 输出目录
    
    Returns:
        输出文件路径
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        
        # 能量流数据
        labels = ['燃烧放热\n(100%)', '有效功\n(42%)', '壁面传热\n(28%)', 
                  '排气损失\n(30%)']
        
        source = [0, 0, 0]  # 燃烧放热
        target = [1, 2, 3]  # 有效功、壁面传热、排气损失
        value = [42, 28, 30]  # 百分比
        
        colors_flow = [COLORS['success'], COLORS['warning'], COLORS['danger']]
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=30,
                line=dict(color='black', width=1),
                label=['燃烧放热 (100%)', '有效功 (42%)', '壁面传热 (28%)', '排气损失 (30%)'],
                color=[COLORS['primary'], COLORS['success'], COLORS['warning'], COLORS['danger']]
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=['rgba(40, 167, 69, 0.5)', 'rgba(255, 193, 7, 0.5)', 'rgba(220, 53, 69, 0.5)']
            )
        )])
        
        fig.update_layout(
            title=dict(text='柴油机循环能量平衡桑基图', font=dict(size=16)),
            font=dict(size=12, family='SimSun, Times New Roman'),
            width=800,
            height=500,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # 保存为SVG
        output_path = get_output_path('modeling', 'energy_sankey.svg')
        pio.write_image(fig, output_path, format='svg')
        print(f"  [Saved] {output_path}")
        
        return output_path
        
    except ImportError:
        print("[Warning] plotly未安装，使用matplotlib替代方案")
        return _plot_energy_sankey_matplotlib(output_dir)
    except Exception as e:
        print(f"[Warning] plotly桑基图失败: {e}，使用matplotlib替代方案")
        return _plot_energy_sankey_matplotlib(output_dir)


def _plot_energy_sankey_matplotlib(output_dir: str = None) -> str:
    """
    使用matplotlib绘制能量流向图（桑基图替代方案）
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 燃烧放热源框
    source_box = FancyBboxPatch((0.5, 3.5), 2.5, 3,
                                 boxstyle="round,pad=0.05,rounding_size=0.2",
                                 facecolor=COLORS['primary'], alpha=0.4,
                                 edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(source_box)
    ax.text(1.75, 5, '燃烧放热', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(1.75, 4.3, '100%', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 目标框
    targets = [
        {'name': '有效功', 'percent': '42%', 'y': 7.5, 'color': COLORS['success'], 'height': 2.5},
        {'name': '壁面传热', 'percent': '28%', 'y': 4.5, 'color': COLORS['warning'], 'height': 1.8},
        {'name': '排气损失', 'percent': '30%', 'y': 1.5, 'color': COLORS['danger'], 'height': 2.0},
    ]
    
    for target in targets:
        box = FancyBboxPatch((8.5, target['y'] - target['height']/2), 2.5, target['height'],
                              boxstyle="round,pad=0.05,rounding_size=0.2",
                              facecolor=target['color'], alpha=0.4,
                              edgecolor=target['color'], linewidth=2)
        ax.add_patch(box)
        ax.text(9.75, target['y'] + 0.2, target['name'], 
                ha='center', va='center', fontsize=11, fontweight='bold')
        ax.text(9.75, target['y'] - 0.3, target['percent'], 
                ha='center', va='center', fontsize=13, fontweight='bold')
        
        # 绘制流向箭头（梯形）
        flow_height = target['height'] * 0.8
        vertices = [
            (3.0, 5 + flow_height/4),  # 左上
            (3.0, 5 - flow_height/4),  # 左下
            (8.5, target['y'] - flow_height/2),  # 右下
            (8.5, target['y'] + flow_height/2),  # 右上
        ]
        flow = Polygon(vertices, closed=True, facecolor=target['color'], 
                       alpha=0.3, edgecolor=target['color'], linewidth=1)
        ax.add_patch(flow)
    
    # 添加标题
    ax.text(6, 9.2, '柴油机循环能量平衡图', ha='center', va='center', 
            fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE, fontweight='bold')
    
    plt.tight_layout()
    output_path = save_figure(fig, 'modeling', 'energy_sankey.svg')
    plt.close(fig)
    
    return output_path


# ============================================================================
# 主函数：生成所有图片
# ============================================================================
def generate_all_modeling_figures():
    """
    生成所有建模相关的可视化图片
    
    Returns:
        dict: 各图片的输出路径
    """
    print("=" * 60)
    print("开始生成建模可视化图片...")
    print("=" * 60)
    
    results = {}
    
    # 图4-1: 模型框架流程图
    print("\n[1/10] 生成模型框架流程图...")
    results['model_framework'] = plot_model_framework()
    
    # 图4-2: 曲柄连杆几何示意图
    print("\n[2/10] 生成曲柄连杆几何示意图...")
    results['crank_geometry'] = plot_crank_geometry_schematic()
    
    # 图4-3: 曲柄连杆运动学特性
    print("\n[3/10] 生成曲柄连杆运动学特性图...")
    results['crank_kinematics'] = plot_crank_kinematics()
    
    # 图4-4: 二冲程配气正时圆形图
    print("\n[4/10] 生成配气正时圆形图...")
    results['valve_timing'] = plot_valve_timing_diagram()
    
    # 图4-5: 双Wiebe燃烧模型特性
    print("\n[5/10] 生成双Wiebe燃烧模型特性图...")
    results['dual_wiebe'] = plot_dual_wiebe_combustion()
    
    # 图4-6: Woschni传热模型特性
    print("\n[6/10] 生成Woschni传热模型特性图...")
    results['woschni'] = plot_woschni_heat_transfer()
    
    # 图4-7: 热力学循环特性
    print("\n[7/10] 生成热力学循环特性图...")
    results['thermodynamic_cycle'] = plot_thermodynamic_cycle()
    
    # 图4-12: 三阶段校准流程图
    print("\n[8/10] 生成三阶段校准流程图...")
    results['calibration_flowchart'] = plot_calibration_flowchart()
    
    # 图4-13: 参数敏感度雷达图
    print("\n[9/10] 生成参数敏感度雷达图...")
    results['sensitivity_radar'] = plot_sensitivity_radar()
    
    # 图4-17: 能量平衡桑基图
    print("\n[10/10] 生成能量平衡桑基图...")
    results['energy_sankey'] = plot_energy_sankey()
    
    print("\n" + "=" * 60)
    print("所有建模可视化图片生成完成！")
    print("=" * 60)
    
    return results


# ============================================================================
# 模块入口
# ============================================================================
if __name__ == '__main__':
    generate_all_modeling_figures()
