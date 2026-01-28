#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
校准结果核心可视化模块
======================
提供模型校准结果的核心可视化函数，采用IEEE/Elsevier学术期刊风格。

统一风格规范:
- 散点: s=100, alpha=0.8, edgecolors='black', linewidths=1.5, zorder=5
- 参考线: 'k--', linewidth=2.0 (45度理想线)
- 误差带: fill_between(), color='gray', alpha=0.15
- 统计框: bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), 右下角
- 网格: alpha=0.3, linestyle='--'
- 子图标题: (a), (b), (c) 格式, fontweight='bold'

核心图表:
1. 数据加载函数 (load_convergence_data, load_validation_data, load_calibrated_params)
2. 45度线散点对比图 (标准学术风格)
3. Bland-Altman一致性分析图
4. 实验-仿真点线对比图

Author: CDC Project
Date: 2026-01-28
"""

import numpy as np
import pandas as pd
import logging
import warnings
import os
import json
from typing import Dict, Optional, Tuple, List
from scipy import stats

# 抑制所有字体警告 - 在导入matplotlib之前设置
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib.backends').setLevel(logging.ERROR)

import matplotlib
matplotlib.use('Agg')
matplotlib.set_loglevel('error')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# 额外抑制字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*Glyph.*')
warnings.filterwarnings('ignore', message='.*font.*')
warnings.filterwarnings('ignore', message='.*Font.*')

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

ACADEMIC_REFERENCE_LINE = {
    'color': 'black',
    'linestyle': '--',
    'linewidth': 2.0,
    'zorder': 1
}

ACADEMIC_ERROR_BAND = {
    'color': 'gray',
    'alpha': 0.15
}

ACADEMIC_STATS_BOX = {
    'boxstyle': 'round',
    'facecolor': 'white',
    'alpha': 0.9
}


def set_tick_fontsize(ax, fontsize=None):
    """设置坐标轴刻度标签的字体大小"""
    if fontsize is None:
        fontsize = PLOT_CONFIG.FONT_SIZE_TICK
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    for label in ax.get_xticklabels():
        label.set_fontsize(fontsize)
    for label in ax.get_yticklabels():
        label.set_fontsize(fontsize)


# ============================================================================
# 数据加载函数
# ============================================================================

def load_convergence_data(filepath: str = None) -> pd.DataFrame:
    """
    加载收敛历史数据
    
    Args:
        filepath: CSV文件路径，默认为 data/calibration/calibration_convergence.csv
        
    Returns:
        df: 收敛历史DataFrame
    """
    if filepath is None:
        filepath = os.path.join(PATH_CONFIG.DATA_CALIBRATION_DIR, 'calibration_convergence.csv')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"收敛历史文件不存在: {filepath}")
    
    return pd.read_csv(filepath)


def load_validation_data(filepath: str = None) -> pd.DataFrame:
    """
    加载验证结果数据
    
    Args:
        filepath: CSV文件路径，默认为 data/calibration/calibration_validation.csv
        
    Returns:
        df: 验证结果DataFrame
    """
    if filepath is None:
        filepath = os.path.join(PATH_CONFIG.DATA_CALIBRATION_DIR, 'calibration_validation.csv')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"验证结果文件不存在: {filepath}")
    
    return pd.read_csv(filepath)


def load_calibrated_params(filepath: str = None) -> Dict:
    """
    加载校准参数
    
    Args:
        filepath: JSON文件路径，默认为 data/calibration/calibrated_params.json
        
    Returns:
        params: 校准参数字典
    """
    if filepath is None:
        filepath = os.path.join(PATH_CONFIG.DATA_CALIBRATION_DIR, 'calibrated_params.json')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"校准参数文件不存在: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# 图1: 45度线散点对比图 (标准学术风格)
# ============================================================================

def plot_45degree_scatter(validation_df: pd.DataFrame,
                          output_dir: str = None) -> str:
    """
    绘制45度线散点对比图
    
    经典的实验-仿真对比展示方式，带R²和误差带
    
    统一风格:
    - 散点: s=100, alpha=0.8, edgecolors='black', linewidths=1.5
    - 45度参考线: 'k--', linewidth=2.0
    - ±10%误差带: gray, alpha=0.15
    - 统计框: 右下角, white背景, alpha=0.9
    
    Args:
        validation_df: 验证数据DataFrame
        output_dir: 输出目录
    
    Returns:
        输出文件路径
    """
    print("\n[学术图] 生成45度线散点对比图...")
    
    if output_dir is None:
        output_dir = PATH_CONFIG.VIS_CALIBRATION_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    colors = COLORS
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    legend_size = PLOT_CONFIG.FONT_SIZE_LEGEND
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    indicators = [
        ('Pmax', 'bar', colors['primary'], '$P_{max}$'),
        ('Pcomp', 'bar', colors['success'], '$P_{comp}$'),
        ('Texh', 'K', colors['danger'], '$T_{exh}$'),
    ]
    
    for idx, (name, unit, color, label) in enumerate(indicators):
        ax = axes[idx]
        
        exp_col = f'{name}_exp'
        sim_col = f'{name}_sim'
        
        if exp_col not in validation_df.columns:
            continue
        
        exp = validation_df[exp_col].values
        sim = validation_df[sim_col].values
        
        # 绘制散点 (统一学术风格)
        ax.scatter(exp, sim, color=color, label='验证点', **ACADEMIC_SCATTER_PARAMS)
        
        # 45度理想线
        lims = [min(exp.min(), sim.min()) * 0.95, max(exp.max(), sim.max()) * 1.05]
        ax.plot(lims, lims, label='理想线 (y=x)', **ACADEMIC_REFERENCE_LINE)
        
        # 线性回归计算R²
        slope, intercept, r_value, p_value, std_err = stats.linregress(exp, sim)
        r_squared = r_value ** 2
        
        # ±10%误差带
        ax.fill_between(lims, [l*0.9 for l in lims], [l*1.1 for l in lims],
                       label='±10%误差带', **ACADEMIC_ERROR_BAND)
        
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.set_xlabel(f'实验值 [{unit}]', fontsize=label_size)
        ax.set_ylabel(f'仿真值 [{unit}]', fontsize=label_size)
        ax.set_title(f'({chr(97+idx)}) {label}', fontsize=title_size, fontweight='bold')
        ax.legend(loc='upper left', fontsize=legend_size-1, framealpha=0.9)
        ax.grid(True, alpha=PLOT_CONFIG.GRID_ALPHA, linestyle='--')
        set_tick_fontsize(ax, tick_size)
        
        # 添加统计信息框 (右下角)
        mean_err = np.mean((sim - exp) / exp * 100)
        max_err = np.max(np.abs((sim - exp) / exp * 100))
        ax.text(0.98, 0.02, f'R² = {r_squared:.4f}\n平均误差: {mean_err:.2f}%\n最大误差: {max_err:.2f}%',
               transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
               bbox=ACADEMIC_STATS_BOX)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'calibration_45degree_scatter.svg')
    fig.savefig(output_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [Saved] {output_path}")
    
    return output_path


# ============================================================================
# 图2: Bland-Altman一致性分析图
# ============================================================================

def plot_bland_altman(validation_df: pd.DataFrame,
                      output_dir: str = None) -> str:
    """
    绘制Bland-Altman一致性分析图
    
    医学/工程领域常用的实验-仿真一致性评估方法
    
    统一风格:
    - 散点: s=100, alpha=0.8, edgecolors='black', linewidths=1.5
    - 均值线: 实线, linewidth=2.0
    - LoA界限: '--', 警告色
    - 一致性区域: 浅色填充
    
    Args:
        validation_df: 验证数据DataFrame
        output_dir: 输出目录
    
    Returns:
        输出文件路径
    """
    print("\n[学术图] 生成Bland-Altman一致性分析图...")
    
    if output_dir is None:
        output_dir = PATH_CONFIG.VIS_CALIBRATION_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    colors = COLORS
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    legend_size = PLOT_CONFIG.FONT_SIZE_LEGEND
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    indicators = [
        ('Pmax', 'bar', colors['primary'], '$P_{max}$'),
        ('Pcomp', 'bar', colors['success'], '$P_{comp}$'),
        ('Texh', 'K', colors['danger'], '$T_{exh}$'),
    ]
    
    for idx, (name, unit, color, label) in enumerate(indicators):
        ax = axes[idx]
        
        exp_col = f'{name}_exp'
        sim_col = f'{name}_sim'
        
        if exp_col not in validation_df.columns:
            continue
        
        exp = validation_df[exp_col].values
        sim = validation_df[sim_col].values
        
        # 计算均值和差值
        mean_vals = (exp + sim) / 2
        diff_vals = sim - exp
        
        # 计算统计量
        mean_diff = np.mean(diff_vals)
        std_diff = np.std(diff_vals)
        upper_loa = mean_diff + 1.96 * std_diff
        lower_loa = mean_diff - 1.96 * std_diff
        
        # 绘制散点 (统一学术风格)
        ax.scatter(mean_vals, diff_vals, color=color, **ACADEMIC_SCATTER_PARAMS)
        
        # 获取x轴范围用于填充
        xlim = [mean_vals.min() * 0.95, mean_vals.max() * 1.05]
        
        # 均值线
        ax.axhline(y=mean_diff, color=colors['dark'], linestyle='-', 
                  linewidth=LINE_WIDTH_MAIN, label=f'偏差均值: {mean_diff:.2f}')
        
        # 95%一致性界限 (LoA)
        ax.axhline(y=upper_loa, color=colors['warning'], linestyle='--', 
                  linewidth=LINE_WIDTH_SECONDARY, label=f'+1.96σ: {upper_loa:.2f}')
        ax.axhline(y=lower_loa, color=colors['warning'], linestyle='--', 
                  linewidth=LINE_WIDTH_SECONDARY, label=f'-1.96σ: {lower_loa:.2f}')
        
        # 零线
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        # 填充一致性区域
        ax.fill_between(xlim, lower_loa, upper_loa, color=color, alpha=0.1)
        ax.set_xlim(xlim)
        
        ax.set_xlabel(f'均值 ({label}) [{unit}]', fontsize=label_size)
        ax.set_ylabel(f'差值 (仿真-实验) [{unit}]', fontsize=label_size)
        ax.set_title(f'({chr(97+idx)}) {label} Bland-Altman图', fontsize=title_size, fontweight='bold')
        ax.legend(loc='best', fontsize=legend_size-1, framealpha=0.9)
        ax.grid(True, alpha=PLOT_CONFIG.GRID_ALPHA, linestyle='--')
        set_tick_fontsize(ax, tick_size)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'calibration_bland_altman.svg')
    fig.savefig(output_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [Saved] {output_path}")
    
    return output_path


# ============================================================================
# 图3: 实验-仿真点线对比图
# ============================================================================

def plot_exp_sim_comparison_lines(validation_df: pd.DataFrame, 
                                   output_dir: str = None) -> str:
    """
    绘制实验-仿真点线对比图
    
    IEEE风格: 简洁、带误差棒、清晰标注
    
    Args:
        validation_df: 验证数据DataFrame
        output_dir: 输出目录
    
    Returns:
        输出文件路径
    """
    print("\n[学术图] 生成实验-仿真点线对比图...")
    
    if output_dir is None:
        output_dir = PATH_CONFIG.VIS_CALIBRATION_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    colors = COLORS
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    legend_size = PLOT_CONFIG.FONT_SIZE_LEGEND
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 工况点编号作为x轴
    x = validation_df['point_id'].values
    
    # 三个指标的配置
    indicators = [
        ('Pmax', 'bar', '最大爆发压力 $P_{max}$'),
        ('Pcomp', 'bar', '压缩压力 $P_{comp}$'),
        ('Texh', 'K', '排气温度 $T_{exh}$'),
    ]
    
    for idx, (name, unit, title) in enumerate(indicators):
        ax = axes[idx]
        
        exp_col = f'{name}_exp'
        sim_col = f'{name}_sim'
        
        if exp_col not in validation_df.columns:
            continue
        
        exp_vals = validation_df[exp_col].values
        sim_vals = validation_df[sim_col].values
        
        # 计算误差棒（假设2%测量不确定度）
        exp_err = exp_vals * 0.02
        sim_err = sim_vals * 0.015
        
        # 绘制实验值（圆形标记，实线）
        ax.errorbar(x, exp_vals, yerr=exp_err, 
                   fmt='o-', color=colors['primary'], 
                   markersize=MARKER_SIZE_LARGE, linewidth=LINE_WIDTH_MAIN,
                   capsize=4, capthick=1.5, elinewidth=1.5,
                   label='实验值')
        
        # 绘制仿真值（方形标记，虚线）
        ax.errorbar(x, sim_vals, yerr=sim_err,
                   fmt='s--', color=colors['danger'], 
                   markersize=MARKER_SIZE_LARGE-1, linewidth=LINE_WIDTH_MAIN,
                   capsize=4, capthick=1.5, elinewidth=1.5,
                   label='仿真值')
        
        ax.set_xlabel('工况点编号', fontsize=label_size)
        ax.set_ylabel(f'{title} [{unit}]', fontsize=label_size)
        ax.set_title(f'({chr(97+idx)}) {title}', fontsize=title_size, fontweight='bold')
        ax.legend(loc='best', fontsize=legend_size, framealpha=0.9)
        ax.grid(True, alpha=PLOT_CONFIG.GRID_ALPHA, linestyle='--')
        ax.set_xticks(x)
        set_tick_fontsize(ax, tick_size)
        
        # 添加统计信息框
        mean_err = validation_df[f'{name}_error'].mean()
        std_err = validation_df[f'{name}_error'].std()
        ax.text(0.98, 0.02, f'误差: {mean_err:.2f}±{std_err:.2f}%',
               transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
               bbox=ACADEMIC_STATS_BOX)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'calibration_exp_sim_lines.svg')
    fig.savefig(output_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [Saved] {output_path}")
    
    return output_path


# ============================================================================
# 生成核心学术风格图表
# ============================================================================

def generate_all_academic_plots(convergence_df: pd.DataFrame,
                                 validation_df: pd.DataFrame,
                                 params: dict = None,
                                 output_dir: str = None) -> List[str]:
    """
    生成核心学术风格校准可视化图表
    
    Args:
        convergence_df: 收敛历史DataFrame
        validation_df: 验证数据DataFrame
        params: 校准参数字典
        output_dir: 输出目录
    
    Returns:
        生成的文件路径列表
    """
    print("\n" + "=" * 60)
    print("生成核心学术风格校准可视化图表")
    print("=" * 60)
    
    if output_dir is None:
        output_dir = PATH_CONFIG.VIS_CALIBRATION_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = []
    
    # 1. 45度线散点图 (核心图表)
    try:
        path = plot_45degree_scatter(validation_df, output_dir)
        generated_files.append(path)
    except Exception as e:
        print(f"  [Error] 45度线散点图: {e}")
    
    # 2. Bland-Altman图 (核心图表)
    try:
        path = plot_bland_altman(validation_df, output_dir)
        generated_files.append(path)
    except Exception as e:
        print(f"  [Error] Bland-Altman图: {e}")
    
    # 3. 点线对比图
    try:
        path = plot_exp_sim_comparison_lines(validation_df, output_dir)
        generated_files.append(path)
    except Exception as e:
        print(f"  [Error] 点线对比图: {e}")
    
    print("\n" + "-" * 60)
    print(f"核心学术风格图表生成完成: {len(generated_files)} 个文件")
    print("-" * 60)
    
    return generated_files


if __name__ == '__main__':
    # 测试用
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 使用模拟数据
    data_dir = PATH_CONFIG.DATA_SIMULATION_DIR
    
    conv_df = load_convergence_data(os.path.join(data_dir, 'mock_calibration_convergence.csv'))
    val_df = load_validation_data(os.path.join(data_dir, 'mock_calibration_validation.csv'))
    params = load_calibrated_params(os.path.join(data_dir, 'mock_calibrated_params.json'))
    
    generate_all_academic_plots(conv_df, val_df, params)
