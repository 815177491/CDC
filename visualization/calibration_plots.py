#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
校准过程可视化绘图模块
======================
提供模型校准过程与结果的可视化绑定函数。

包括:
1. 校准收敛曲线 (单曲线展示)
2. 校准前后对比 (Pmax/Pcomp/Texh 实验vs仿真)
3. 误差分布分析 (箱线图/柱状图)
4. 校准参数汇总 (水平条形图)

Author: CDC Project
Date: 2026-01-28
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import json
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 导入全局配置
from config import (
    PLOT_CONFIG, 
    COLORS, 
    PATH_CONFIG,
    setup_matplotlib_style,
    save_figure,
    get_output_path
)

# 应用全局matplotlib样式
setup_matplotlib_style()


def set_tick_fontsize(ax, fontsize=None):
    """设置坐标轴刻度标签的字体大小"""
    if fontsize is None:
        fontsize = PLOT_CONFIG.FONT_SIZE_TICK
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    for label in ax.get_xticklabels():
        label.set_fontsize(fontsize)
    for label in ax.get_yticklabels():
        label.set_fontsize(fontsize)


def load_convergence_data(filepath: str = None) -> pd.DataFrame:
    """
    加载收敛历史数据
    
    Args:
        filepath: CSV文件路径，默认为 data/calibration_convergence.csv
        
    Returns:
        df: 收敛历史DataFrame
    """
    if filepath is None:
        filepath = os.path.join(PATH_CONFIG.DATA_DIR, 'calibration_convergence.csv')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"收敛历史文件不存在: {filepath}")
    
    return pd.read_csv(filepath)


def load_validation_data(filepath: str = None) -> pd.DataFrame:
    """
    加载验证结果数据
    
    Args:
        filepath: CSV文件路径，默认为 data/calibration_validation.csv
        
    Returns:
        df: 验证结果DataFrame
    """
    if filepath is None:
        filepath = os.path.join(PATH_CONFIG.DATA_DIR, 'calibration_validation.csv')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"验证结果文件不存在: {filepath}")
    
    return pd.read_csv(filepath)


def load_calibrated_params(filepath: str = None) -> Dict:
    """
    加载校准参数
    
    Args:
        filepath: JSON文件路径，默认为 data/calibrated_params.json
        
    Returns:
        params: 校准参数字典
    """
    if filepath is None:
        filepath = os.path.join(PATH_CONFIG.DATA_DIR, 'calibrated_params.json')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"校准参数文件不存在: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# 图1: 校准收敛与对比 (2×2布局)
# ============================================================================

def plot_calibration_convergence_and_comparison(
    convergence_df: pd.DataFrame = None,
    validation_df: pd.DataFrame = None,
    output_dir: str = None
) -> plt.Figure:
    """
    可视化1: 校准收敛曲线与实验-仿真对比
    
    2×2布局:
    (a) 收敛曲线 (单曲线)
    (b) Pmax 实验vs仿真散点图
    (c) Pcomp 实验vs仿真散点图
    (d) Texh 实验vs仿真散点图
    
    Args:
        convergence_df: 收敛历史数据，若为None则从文件加载
        validation_df: 验证结果数据，若为None则从文件加载
        output_dir: 输出目录，默认使用全局配置
        
    Returns:
        fig: matplotlib Figure对象
    """
    print("\n[1/2] 生成校准收敛与对比可视化...")
    
    # 使用全局配置
    colors = COLORS
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    legend_size = PLOT_CONFIG.FONT_SIZE_LEGEND
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE
    
    # 输出目录默认使用全局配置
    if output_dir is None:
        output_dir = PATH_CONFIG.VIS_CALIBRATION_DIR
    
    # 加载数据
    if convergence_df is None:
        convergence_df = load_convergence_data()
    if validation_df is None:
        validation_df = load_validation_data()
    
    # 创建图形 - 2×2布局
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # ========== (a) 收敛曲线 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 单曲线展示目标函数值
    ax1.plot(convergence_df['iteration'], convergence_df['objective_value'],
             color=colors['dark'], linewidth=1.5, alpha=0.5, label='目标函数值')
    ax1.plot(convergence_df['iteration'], convergence_df['best_value'],
             color=colors['primary'], linewidth=2, label='最优值')
    
    # 标记阶段分界点 (如果有多阶段数据)
    if 'stage' in convergence_df.columns:
        stages = convergence_df['stage'].unique()
        stage_colors = {'compression': colors['secondary'], 
                       'combustion': colors['orange'], 
                       'heat_transfer': colors['teal']}
        stage_names = {'compression': '压缩段', 
                      'combustion': '燃烧段', 
                      'heat_transfer': '传热段'}
        
        for stage in stages:
            stage_data = convergence_df[convergence_df['stage'] == stage]
            if len(stage_data) > 0:
                start_iter = stage_data['iteration'].iloc[0]
                ax1.axvline(start_iter, color=stage_colors.get(stage, colors['dark']),
                           linestyle='--', alpha=0.7, linewidth=1.5,
                           label=f'{stage_names.get(stage, stage)}开始')
    
    ax1.set_xlabel('迭代次数', fontsize=label_size)
    ax1.set_ylabel('目标函数值', fontsize=label_size)
    ax1.set_title('(a) 校准收敛曲线', fontsize=title_size, fontweight='bold')
    ax1.legend(fontsize=legend_size, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 对数坐标更清晰
    
    # ========== (b) Pmax 对比 ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    Pmax_exp = validation_df['Pmax_exp']
    Pmax_sim = validation_df['Pmax_sim']
    
    ax2.scatter(Pmax_exp, Pmax_sim, s=80, c=colors['primary'], 
               alpha=0.7, edgecolors='white', linewidths=1, label='工况点')
    
    # y=x参考线
    lims = [min(Pmax_exp.min(), Pmax_sim.min()) * 0.95,
            max(Pmax_exp.max(), Pmax_sim.max()) * 1.05]
    ax2.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.7, label='y=x')
    
    # ±5%误差带
    ax2.fill_between(lims, [l*0.95 for l in lims], [l*1.05 for l in lims],
                     alpha=0.15, color=colors['primary'], label='±5%误差带')
    
    # 计算R²
    ss_res = np.sum((Pmax_sim - Pmax_exp)**2)
    ss_tot = np.sum((Pmax_exp - Pmax_exp.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    ax2.text(0.05, 0.95, f'$R^2$ = {r2:.4f}', transform=ax2.transAxes,
            fontsize=legend_size, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_xlabel('$P_{max,exp}$ (bar)', fontsize=label_size)
    ax2.set_ylabel('$P_{max,sim}$ (bar)', fontsize=label_size)
    ax2.set_title('(b) 最大爆发压力对比', fontsize=title_size, fontweight='bold')
    ax2.legend(fontsize=legend_size-1, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_aspect('equal')
    
    # ========== (c) Pcomp 对比 ==========
    ax3 = fig.add_subplot(gs[1, 0])
    
    Pcomp_exp = validation_df['Pcomp_exp']
    Pcomp_sim = validation_df['Pcomp_sim']
    
    ax3.scatter(Pcomp_exp, Pcomp_sim, s=80, c=colors['secondary'], 
               alpha=0.7, edgecolors='white', linewidths=1, label='工况点')
    
    # y=x参考线
    lims = [min(Pcomp_exp.min(), Pcomp_sim.min()) * 0.95,
            max(Pcomp_exp.max(), Pcomp_sim.max()) * 1.05]
    ax3.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.7, label='y=x')
    
    # ±5%误差带
    ax3.fill_between(lims, [l*0.95 for l in lims], [l*1.05 for l in lims],
                     alpha=0.15, color=colors['secondary'], label='±5%误差带')
    
    # 计算R²
    ss_res = np.sum((Pcomp_sim - Pcomp_exp)**2)
    ss_tot = np.sum((Pcomp_exp - Pcomp_exp.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    ax3.text(0.05, 0.95, f'$R^2$ = {r2:.4f}', transform=ax3.transAxes,
            fontsize=legend_size, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax3.set_xlabel('$P_{comp,exp}$ (bar)', fontsize=label_size)
    ax3.set_ylabel('$P_{comp,sim}$ (bar)', fontsize=label_size)
    ax3.set_title('(c) 压缩压力对比', fontsize=title_size, fontweight='bold')
    ax3.legend(fontsize=legend_size-1, loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(lims)
    ax3.set_ylim(lims)
    ax3.set_aspect('equal')
    
    # ========== (d) Texh 对比 ==========
    ax4 = fig.add_subplot(gs[1, 1])
    
    Texh_exp = validation_df['Texh_exp']
    Texh_sim = validation_df['Texh_sim']
    
    # 过滤有效排温数据
    valid_mask = Texh_exp > 100
    Texh_exp_valid = Texh_exp[valid_mask]
    Texh_sim_valid = Texh_sim[valid_mask]
    
    if len(Texh_exp_valid) > 0:
        ax4.scatter(Texh_exp_valid, Texh_sim_valid, s=80, c=colors['orange'], 
                   alpha=0.7, edgecolors='white', linewidths=1, label='工况点')
        
        # y=x参考线
        lims = [min(Texh_exp_valid.min(), Texh_sim_valid.min()) * 0.95,
                max(Texh_exp_valid.max(), Texh_sim_valid.max()) * 1.05]
        ax4.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.7, label='y=x')
        
        # ±10%误差带 (排温误差允许更大)
        ax4.fill_between(lims, [l*0.90 for l in lims], [l*1.10 for l in lims],
                         alpha=0.15, color=colors['orange'], label='±10%误差带')
        
        # 计算R²
        ss_res = np.sum((Texh_sim_valid - Texh_exp_valid)**2)
        ss_tot = np.sum((Texh_exp_valid - Texh_exp_valid.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        ax4.text(0.05, 0.95, f'$R^2$ = {r2:.4f}', transform=ax4.transAxes,
                fontsize=legend_size, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax4.set_xlim(lims)
        ax4.set_ylim(lims)
        ax4.set_aspect('equal')
    else:
        ax4.text(0.5, 0.5, '无有效排温数据', transform=ax4.transAxes,
                ha='center', va='center', fontsize=14)
    
    ax4.set_xlabel('$T_{exh,exp}$ (°C)', fontsize=label_size)
    ax4.set_ylabel('$T_{exh,sim}$ (°C)', fontsize=label_size)
    ax4.set_title('(d) 排气温度对比', fontsize=title_size, fontweight='bold')
    ax4.legend(fontsize=legend_size-1, loc='lower right')
    ax4.grid(True, alpha=0.3)
    
    # 设置所有刻度标签字体大小
    for ax in [ax1, ax2, ax3, ax4]:
        set_tick_fontsize(ax, fontsize=tick_size)
    
    plt.suptitle('模型校准收敛过程与结果验证', 
                fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE, fontweight='bold', y=0.995)
    
    # 保存图形
    save_path = save_figure(fig, 'calibration', 'calibration_convergence_comparison.svg')
    plt.close()
    
    return fig


# ============================================================================
# 图2: 误差分布与参数汇总 (1×2布局)
# ============================================================================

def plot_error_distribution_and_parameters(
    validation_df: pd.DataFrame = None,
    params: Dict = None,
    output_dir: str = None
) -> plt.Figure:
    """
    可视化2: 误差分布与校准参数汇总
    
    1×2布局:
    (a) 误差分布箱线图
    (b) 校准参数汇总条形图
    
    Args:
        validation_df: 验证结果数据，若为None则从文件加载
        params: 校准参数字典，若为None则从文件加载
        output_dir: 输出目录，默认使用全局配置
        
    Returns:
        fig: matplotlib Figure对象
    """
    print("\n[2/2] 生成误差分布与参数汇总可视化...")
    
    # 使用全局配置
    colors = COLORS
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    legend_size = PLOT_CONFIG.FONT_SIZE_LEGEND
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE
    
    # 输出目录默认使用全局配置
    if output_dir is None:
        output_dir = PATH_CONFIG.VIS_CALIBRATION_DIR
    
    # 加载数据
    if validation_df is None:
        validation_df = load_validation_data()
    if params is None:
        params = load_calibrated_params()
    
    # 创建图形 - 1×2布局
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)
    
    # ========== (a) 误差分布箱线图 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 准备误差数据
    error_data = []
    error_labels = []
    error_colors = []
    
    # Pmax误差
    Pmax_errors = validation_df['Pmax_error'].dropna()
    if len(Pmax_errors) > 0:
        error_data.append(Pmax_errors)
        error_labels.append('$P_{max}$')
        error_colors.append(colors['primary'])
    
    # Pcomp误差
    Pcomp_errors = validation_df['Pcomp_error'].dropna()
    if len(Pcomp_errors) > 0:
        error_data.append(Pcomp_errors)
        error_labels.append('$P_{comp}$')
        error_colors.append(colors['secondary'])
    
    # Texh误差
    Texh_errors = validation_df['Texh_error'].dropna()
    valid_Texh = Texh_errors[validation_df['Texh_exp'] > 100]
    if len(valid_Texh) > 0:
        error_data.append(valid_Texh)
        error_labels.append('$T_{exh}$')
        error_colors.append(colors['orange'])
    
    if len(error_data) > 0:
        bp = ax1.boxplot(error_data, labels=error_labels, patch_artist=True,
                        widths=0.6, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white', 
                                      markeredgecolor='black', markersize=8))
        
        # 设置颜色
        for patch, color in zip(bp['boxes'], error_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 添加散点
        for i, (data, color) in enumerate(zip(error_data, error_colors)):
            x = np.random.normal(i+1, 0.04, size=len(data))
            ax1.scatter(x, data, alpha=0.5, s=30, c=color, edgecolors='white', zorder=10)
        
        # 添加参考线
        ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax1.axhline(5, color=colors['warning'], linestyle='--', linewidth=1.5, 
                   alpha=0.7, label='±5%阈值')
        ax1.axhline(-5, color=colors['warning'], linestyle='--', linewidth=1.5, alpha=0.7)
        
        # 添加统计信息
        for i, data in enumerate(error_data):
            mean_err = np.mean(np.abs(data))
            ax1.text(i+1, ax1.get_ylim()[1] * 0.95, f'MAE: {mean_err:.2f}%',
                    ha='center', fontsize=9, fontweight='bold')
    
    ax1.set_ylabel('相对误差 (%)', fontsize=label_size)
    ax1.set_title('(a) 校准误差分布', fontsize=title_size, fontweight='bold')
    ax1.legend(fontsize=legend_size, loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ========== (b) 校准参数汇总条形图 ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 参数信息配置
    param_config = {
        'compression_ratio': {
            'name': '有效压缩比',
            'unit': '',
            'color': colors['primary'],
            'range': (11.0, 16.0)
        },
        'injection_timing': {
            'name': '喷油正时',
            'unit': '°CA BTDC',
            'color': colors['secondary'],
            'range': (-5.0, 10.0)
        },
        'diffusion_duration': {
            'name': '燃烧持续期',
            'unit': '°CA',
            'color': colors['orange'],
            'range': (30.0, 80.0)
        },
        'diffusion_shape': {
            'name': '形状因子',
            'unit': '',
            'color': colors['teal'],
            'range': (0.5, 3.0)
        },
        'C_woschni': {
            'name': 'Woschni系数',
            'unit': '',
            'color': colors['purple'],
            'range': (80.0, 200.0)
        }
    }
    
    # 准备数据
    param_names = []
    param_values = []
    param_colors = []
    param_ranges = []
    
    for key, value in params.items():
        if key in param_config:
            cfg = param_config[key]
            param_names.append(f"{cfg['name']}\n({cfg['unit']})" if cfg['unit'] else cfg['name'])
            param_values.append(value)
            param_colors.append(cfg['color'])
            param_ranges.append(cfg['range'])
    
    # 绘制水平条形图（归一化到各自范围）
    y_pos = np.arange(len(param_names))
    normalized_values = []
    
    for val, (rmin, rmax) in zip(param_values, param_ranges):
        normalized = (val - rmin) / (rmax - rmin) * 100  # 转换为百分比
        normalized_values.append(normalized)
    
    bars = ax2.barh(y_pos, normalized_values, color=param_colors, alpha=0.8,
                   edgecolor='black', linewidth=1)
    
    # 添加数值标注
    for i, (bar, val, (rmin, rmax)) in enumerate(zip(bars, param_values, param_ranges)):
        width = bar.get_width()
        ax2.text(width + 2, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}',
                va='center', ha='left', fontsize=10, fontweight='bold')
        
        # 添加范围标注
        ax2.text(105, bar.get_y() + bar.get_height()/2,
                f'[{rmin:.1f}, {rmax:.1f}]',
                va='center', ha='left', fontsize=8, color='gray')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(param_names, fontsize=tick_size)
    ax2.set_xlabel('相对位置 (0-100%范围内)', fontsize=label_size)
    ax2.set_xlim(0, 130)
    ax2.set_title('(b) 校准参数汇总', fontsize=title_size, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 添加范围边界线
    ax2.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax2.axvline(100, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    # 设置刻度标签字体大小
    for ax in [ax1, ax2]:
        set_tick_fontsize(ax, fontsize=tick_size)
    
    plt.suptitle('校准误差分析与参数汇总', 
                fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE, fontweight='bold', y=0.98)
    
    # 保存图形
    save_path = save_figure(fig, 'calibration', 'calibration_error_parameters.svg')
    plt.close()
    
    return fig


# ============================================================================
# 单独绑定函数 (可独立调用)
# ============================================================================

def plot_calibration_convergence(
    convergence_df: pd.DataFrame = None,
    output_dir: str = None
) -> plt.Figure:
    """
    可视化: 校准收敛曲线 (单独版本)
    
    Args:
        convergence_df: 收敛历史数据
        output_dir: 输出目录
        
    Returns:
        fig: matplotlib Figure对象
    """
    print("\n[单独] 生成校准收敛曲线...")
    
    colors = COLORS
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    legend_size = PLOT_CONFIG.FONT_SIZE_LEGEND
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE
    
    if output_dir is None:
        output_dir = PATH_CONFIG.VIS_CALIBRATION_DIR
    
    if convergence_df is None:
        convergence_df = load_convergence_data()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制收敛曲线
    ax.plot(convergence_df['iteration'], convergence_df['objective_value'],
            color=colors['dark'], linewidth=1.5, alpha=0.5, label='目标函数值')
    ax.plot(convergence_df['iteration'], convergence_df['best_value'],
            color=colors['primary'], linewidth=2, label='最优值')
    
    # 标记阶段
    if 'stage' in convergence_df.columns:
        stages = convergence_df['stage'].unique()
        stage_colors = {'compression': colors['secondary'], 
                       'combustion': colors['orange'], 
                       'heat_transfer': colors['teal']}
        stage_names = {'compression': '压缩段', 
                      'combustion': '燃烧段', 
                      'heat_transfer': '传热段'}
        
        for stage in stages:
            stage_data = convergence_df[convergence_df['stage'] == stage]
            if len(stage_data) > 0:
                start_iter = stage_data['iteration'].iloc[0]
                ax.axvline(start_iter, color=stage_colors.get(stage, colors['dark']),
                          linestyle='--', alpha=0.7, linewidth=1.5,
                          label=f'{stage_names.get(stage, stage)}')
    
    ax.set_xlabel('迭代次数', fontsize=label_size)
    ax.set_ylabel('目标函数值', fontsize=label_size)
    ax.set_title('三阶段校准收敛曲线', fontsize=title_size, fontweight='bold')
    ax.legend(fontsize=legend_size)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    set_tick_fontsize(ax, fontsize=tick_size)
    
    plt.tight_layout()
    save_path = save_figure(fig, 'calibration', 'calibration_convergence.svg')
    plt.close()
    
    return fig


def plot_calibration_comparison(
    validation_df: pd.DataFrame = None,
    output_dir: str = None
) -> plt.Figure:
    """
    可视化: 实验-仿真对比散点图 (单独版本)
    
    Args:
        validation_df: 验证结果数据
        output_dir: 输出目录
        
    Returns:
        fig: matplotlib Figure对象
    """
    print("\n[单独] 生成实验-仿真对比图...")
    
    colors = COLORS
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    legend_size = PLOT_CONFIG.FONT_SIZE_LEGEND
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE
    
    if output_dir is None:
        output_dir = PATH_CONFIG.VIS_CALIBRATION_DIR
    
    if validation_df is None:
        validation_df = load_validation_data()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 指标配置
    metrics = [
        ('Pmax', '$P_{max}$ (bar)', colors['primary']),
        ('Pcomp', '$P_{comp}$ (bar)', colors['secondary']),
        ('Texh', '$T_{exh}$ (°C)', colors['orange'])
    ]
    
    for ax, (metric, label, color) in zip(axes, metrics):
        exp_col = f'{metric}_exp'
        sim_col = f'{metric}_sim'
        
        exp_data = validation_df[exp_col]
        sim_data = validation_df[sim_col]
        
        # 过滤有效数据
        if metric == 'Texh':
            valid_mask = exp_data > 100
            exp_data = exp_data[valid_mask]
            sim_data = sim_data[valid_mask]
        
        if len(exp_data) == 0:
            ax.text(0.5, 0.5, '无有效数据', transform=ax.transAxes,
                   ha='center', va='center')
            continue
        
        ax.scatter(exp_data, sim_data, s=80, c=color, alpha=0.7,
                  edgecolors='white', linewidths=1)
        
        # y=x线和误差带
        lims = [min(exp_data.min(), sim_data.min()) * 0.95,
                max(exp_data.max(), sim_data.max()) * 1.05]
        ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.7)
        
        err_band = 0.05 if metric != 'Texh' else 0.10
        ax.fill_between(lims, [l*(1-err_band) for l in lims], 
                       [l*(1+err_band) for l in lims],
                       alpha=0.15, color=color)
        
        # R²
        ss_res = np.sum((sim_data - exp_data)**2)
        ss_tot = np.sum((exp_data - exp_data.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.text(0.05, 0.95, f'$R^2$ = {r2:.4f}', transform=ax.transAxes,
               fontsize=legend_size, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel(f'{label} (实验)', fontsize=label_size)
        ax.set_ylabel(f'{label} (仿真)', fontsize=label_size)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        set_tick_fontsize(ax, fontsize=tick_size)
    
    plt.suptitle('校准结果验证: 实验值 vs 仿真值', 
                fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_figure(fig, 'calibration', 'calibration_comparison.svg')
    plt.close()
    
    return fig


def plot_error_distribution(
    validation_df: pd.DataFrame = None,
    output_dir: str = None
) -> plt.Figure:
    """
    可视化: 误差分布箱线图 (单独版本)
    
    Args:
        validation_df: 验证结果数据
        output_dir: 输出目录
        
    Returns:
        fig: matplotlib Figure对象
    """
    print("\n[单独] 生成误差分布图...")
    
    colors = COLORS
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    legend_size = PLOT_CONFIG.FONT_SIZE_LEGEND
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE
    
    if output_dir is None:
        output_dir = PATH_CONFIG.VIS_CALIBRATION_DIR
    
    if validation_df is None:
        validation_df = load_validation_data()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 准备误差数据
    error_data = []
    error_labels = []
    error_colors = [colors['primary'], colors['secondary'], colors['orange']]
    
    for metric, label in [('Pmax', '$P_{max}$'), ('Pcomp', '$P_{comp}$'), ('Texh', '$T_{exh}$')]:
        errors = validation_df[f'{metric}_error'].dropna()
        if metric == 'Texh':
            errors = errors[validation_df['Texh_exp'] > 100]
        if len(errors) > 0:
            error_data.append(errors)
            error_labels.append(label)
    
    if len(error_data) > 0:
        bp = ax.boxplot(error_data, labels=error_labels, patch_artist=True,
                       widths=0.6, showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='white',
                                     markeredgecolor='black', markersize=8))
        
        for patch, color in zip(bp['boxes'], error_colors[:len(error_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        for i, (data, color) in enumerate(zip(error_data, error_colors)):
            x = np.random.normal(i+1, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.5, s=30, c=color, edgecolors='white', zorder=10)
        
        ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.axhline(5, color=colors['warning'], linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(-5, color=colors['warning'], linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_ylabel('相对误差 (%)', fontsize=label_size)
    ax.set_title('校准误差分布', fontsize=title_size, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    set_tick_fontsize(ax, fontsize=tick_size)
    plt.tight_layout()
    
    save_path = save_figure(fig, 'calibration', 'error_distribution.svg')
    plt.close()
    
    return fig


def plot_calibrated_parameters(
    params: Dict = None,
    output_dir: str = None
) -> plt.Figure:
    """
    可视化: 校准参数汇总条形图 (单独版本)
    
    Args:
        params: 校准参数字典
        output_dir: 输出目录
        
    Returns:
        fig: matplotlib Figure对象
    """
    print("\n[单独] 生成校准参数汇总图...")
    
    colors = COLORS
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE
    
    if output_dir is None:
        output_dir = PATH_CONFIG.VIS_CALIBRATION_DIR
    
    if params is None:
        params = load_calibrated_params()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 参数配置
    param_config = {
        'compression_ratio': ('有效压缩比', '', colors['primary'], (11.0, 16.0)),
        'injection_timing': ('喷油正时', '°CA BTDC', colors['secondary'], (-5.0, 10.0)),
        'diffusion_duration': ('燃烧持续期', '°CA', colors['orange'], (30.0, 80.0)),
        'diffusion_shape': ('形状因子', '', colors['teal'], (0.5, 3.0)),
        'C_woschni': ('Woschni系数', '', colors['purple'], (80.0, 200.0))
    }
    
    param_names = []
    param_values = []
    param_colors = []
    param_ranges = []
    
    for key, value in params.items():
        if key in param_config:
            name, unit, color, rng = param_config[key]
            param_names.append(f"{name}\n({unit})" if unit else name)
            param_values.append(value)
            param_colors.append(color)
            param_ranges.append(rng)
    
    y_pos = np.arange(len(param_names))
    normalized_values = [(val - rmin) / (rmax - rmin) * 100 
                        for val, (rmin, rmax) in zip(param_values, param_ranges)]
    
    bars = ax.barh(y_pos, normalized_values, color=param_colors, alpha=0.8,
                  edgecolor='black', linewidth=1)
    
    for i, (bar, val, (rmin, rmax)) in enumerate(zip(bars, param_values, param_ranges)):
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2,
               f'{val:.2f}', va='center', ha='left', fontsize=10, fontweight='bold')
        ax.text(105, bar.get_y() + bar.get_height()/2,
               f'[{rmin:.1f}, {rmax:.1f}]', va='center', ha='left', fontsize=8, color='gray')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names, fontsize=tick_size)
    ax.set_xlabel('相对位置 (0-100%范围内)', fontsize=label_size)
    ax.set_xlim(0, 130)
    ax.set_title('校准参数汇总', fontsize=title_size, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(100, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    set_tick_fontsize(ax, fontsize=tick_size)
    plt.tight_layout()
    
    save_path = save_figure(fig, 'calibration', 'calibrated_parameters.svg')
    plt.close()
    
    return fig
