#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比实验可视化模块
==================
Nature / Science 级学术风格条形图，用于方法对比实验。

设计原则
--------
- 水平布局 + despine（Tufte 极简主义）
- 本方法视觉权重突出（色彩饱和度 + 粗边框 + 标签加粗）
- 低饱和度高明度一致色板
- 数值标注精准定位、无冗余装饰
- 渐变色阶传达数据趋势

图表列表 (5 张)
-  plot_overall_metrics_bars      – 综合指标水平多面板
-  plot_per_fault_f1_bars         – 分故障 F1 渐变色阶
-  plot_ablation_delta_bars       – 消融瀑布图
-  plot_severity_robustness_bars  – 鲁棒性渐变 + 趋势线
-  plot_multi_run_statistics      – 统计 CI 对比 + 显著性

Author: CDC Project
Date: 2026-02-23
"""

# 标准库
from typing import Dict, List, Optional, Tuple

# 第三方库
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import Patch

# 本项目模块
from config import PLOT_CONFIG, COLORS, setup_matplotlib_style, save_figure
from visualization.style import (
    set_tick_fontsize,
    despine,
    EXPERIMENT_PALETTE,
    EXPERIMENT_GRID,
    EXPERIMENT_BG,
    LINE_WIDTH_MAIN,
    LINE_WIDTH_HERO,
    ACADEMIC_GRID,
)

setup_matplotlib_style()


# ============================================================================
# 内部工具
# ============================================================================

# ── 本方法专用色 ──
_OURS_COLOR = '#2B5D8C'
_OURS_EDGE = '#1A3D5C'

# ── 基线方法色（低饱和度、与本方法拉开视觉差距）──
_BASELINE_COLORS = ['#B8514D', '#5E8A5E', '#B89840', '#7A6DA8', '#4D95A6']
_BASELINE_ALPHA = 0.78

_BG_STRIPE_ALPHA = 0.025
_VALUE_TEXT_PAD = 0.018


def _clean_method_name(name: str) -> str:
    """去掉换行符，清理方法名称用于显示。

    Args:
        name: 原始方法名称（可能含 ``\\n``）。

    Returns:
        清理后的字符串。
    """
    return name.replace('\n', ' ').strip()


def _annotate_bar_value(
    ax, x_val: float, y_pos: float, text: str, color: str,
    bold: bool = False, x_max: float = 1.0, fontsize: int = 11,
):
    """在水平条形右端标注数值，自动选择内侧（白字描边）或外侧。

    Args:
        ax: matplotlib Axes。
        x_val: 条形终点 x 值。
        y_pos: 条形中心 y 值。
        text: 标注文本。
        color: 文字/描边颜色。
        bold: 是否加粗。
        x_max: x 轴最大值（用于判定内/外侧阈值）。
        fontsize: 字号。
    """
    if x_val / max(x_max, 1e-9) > 0.82:
        ax.text(
            x_val - x_max * _VALUE_TEXT_PAD, y_pos, text,
            ha='right', va='center', fontsize=fontsize,
            fontweight='bold' if bold else 'normal',
            color='white',
            path_effects=[pe.withStroke(linewidth=2.8, foreground=color)],
        )
    else:
        ax.text(
            x_val + x_max * _VALUE_TEXT_PAD, y_pos, text,
            ha='left', va='center', fontsize=fontsize,
            fontweight='bold' if bold else 'normal',
            color=color,
        )


def _get_method_color(idx: int, n_methods: int):
    """返回 (fill_color, fill_alpha, edge_color, edge_width)。idx=0 为本方法。

    Args:
        idx: 方法索引。
        n_methods: 总方法数量。

    Returns:
        四元组 (fill, alpha, edge, linewidth)。
    """
    if idx == 0:
        return _OURS_COLOR, 0.92, _OURS_EDGE, 1.6
    bc = _BASELINE_COLORS[(idx - 1) % len(_BASELINE_COLORS)]
    return bc, _BASELINE_ALPHA, 'white', 0.6


def _add_row_stripes(ax, n_rows: int):
    """交替行浅色底纹，增加可读性。

    Args:
        ax: matplotlib Axes。
        n_rows: 行数。
    """
    for i in range(n_rows):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='#000000',
                       alpha=_BG_STRIPE_ALPHA, zorder=0)


def _style_ax(ax, title: str = '', xlabel: str = '', ylabel: str = ''):
    """统一 Axes 样式：despine + 极浅网格 + 标签。

    Args:
        ax: matplotlib Axes。
        title: 子图标题（左对齐）。
        xlabel: x 轴标签。
        ylabel: y 轴标签。
    """
    despine(ax)
    ax.set_facecolor('#FDFDFD')
    ax.grid(axis='x', color='#E8E8E8', linewidth=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=PLOT_CONFIG.FONT_SIZE_TITLE,
                     fontweight='bold', loc='left', pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    set_tick_fontsize(ax, PLOT_CONFIG.FONT_SIZE_TICK)


def _highlight_ours_ytick(ax, clean_names: List[str]):
    """将第 0 项（本方法）的 y-tick 加粗高亮。

    Args:
        ax: matplotlib Axes。
        clean_names: 清理后的方法名列表。
    """
    for tl in ax.get_yticklabels():
        if tl.get_text() == clean_names[0]:
            tl.set_fontweight('bold')
            tl.set_color(_OURS_COLOR)


# ============================================================================
# 图 1: 综合指标水平多面板条形图
# ============================================================================

def plot_overall_metrics_bars(
    methods: List[str],
    metrics: Dict[str, List[float]],
    metric_keys: Optional[List[str]] = None,
    normalize: bool = False,
) -> str:
    """综合指标水平条形图——每个指标占一列子图，方法纵向排列。

    本方法 (idx=0) 以深蓝色突出，其余基线为柔和灰调；
    条形右侧标注原始数值，最优值加粗 + 小圆点指示。

    Args:
        methods: 方法名称列表（第 0 项视为本方法）。
        metrics: 指标字典 {列名: [各方法值]}。
        metric_keys: 要展示的指标子集，默认自动选取。
        normalize: 是否对每个指标做 min-max 归一化。

    Returns:
        输出文件路径。
    """
    default_keys = ['诊断准确率', '$P_{max}$ 维持率', '检测延迟 (cycles)', '综合评分']
    if metric_keys is None:
        metric_keys = [k for k in default_keys if k in metrics]
    if not metric_keys:
        raise ValueError('No available metrics.')

    display = {
        '诊断准确率': 'Diagnostic\nAccuracy',
        '$P_{max}$ 维持率': '$P_{max}$\nRetention',
        '检测延迟 (cycles)': 'Detection\nDelay (cyc)',
        '综合评分': 'Overall\nScore',
    }
    lower_is_better = {'检测延迟 (cycles)'}

    n_metrics = len(metric_keys)
    n_methods = len(methods)
    clean_names = [_clean_method_name(m) for m in methods]
    bar_h = 0.58

    fig_w = max(3.6 * n_metrics, 10)
    fig_h = max(n_methods * 0.72 + 1.8, 4.0)
    fig, axes = plt.subplots(1, n_metrics, figsize=(fig_w, fig_h), sharey=True)
    if n_metrics == 1:
        axes = [axes]

    y_pos = np.arange(n_methods)[::-1]

    for ax_idx, key in enumerate(metric_keys):
        ax = axes[ax_idx]
        values = np.array(metrics[key], dtype=float)
        is_lower = key in lower_is_better
        best_idx = int(np.argmin(values)) if is_lower else int(np.argmax(values))
        x_max = values.max() * 1.28

        _add_row_stripes(ax, n_methods)

        for i in range(n_methods):
            fc, fa, ec, ew = _get_method_color(i, n_methods)
            is_best = (i == best_idx)
            bh = bar_h + 0.06 if is_best else bar_h

            ax.barh(y_pos[i], values[i], height=bh,
                    color=fc, alpha=fa, edgecolor=ec, linewidth=ew, zorder=3)

            if is_best:
                ax.plot(values[i] * 0.015, y_pos[i], 'o',
                        color=fc, markersize=5, zorder=6,
                        markeredgecolor='white', markeredgewidth=1.2)

            fmt = f'{values[i]:.1f}' if values[i] > 10 else f'{values[i]:.3f}'
            _annotate_bar_value(
                ax, values[i], y_pos[i], fmt,
                color=fc if not is_best else _OURS_EDGE,
                bold=is_best, x_max=x_max,
                fontsize=PLOT_CONFIG.FONT_SIZE_TEXT,
            )

        ax.set_xlim(0, x_max)
        ax.set_yticks(y_pos)
        if ax_idx == 0:
            ax.set_yticklabels(clean_names)
            _highlight_ours_ytick(ax, clean_names)
        else:
            ax.set_yticklabels([])

        sub = chr(97 + ax_idx)
        _style_ax(ax, title=f'({sub}) {display.get(key, key)}')

        if is_lower:
            ax.text(0.97, 0.03, '← lower is better',
                    transform=ax.transAxes, fontsize=8, color='#999999',
                    ha='right', va='bottom', style='italic')

    fig.suptitle('方法综合性能对比',
                 fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE + 1,
                 fontweight='bold', x=0.02, ha='left', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return save_figure(fig, 'experiments', 'overall_metrics_grouped_bars')


# ============================================================================
# 图 2: 分故障类型 F1 渐变色阶水平条形图
# ============================================================================

def plot_per_fault_f1_bars(
    methods: List[str],
    fault_names: List[str],
    f1_matrix: List[List[float]],
) -> str:
    """分故障类型 F1 对比——分面水平条形图（每个故障一个子图）。

    色彩同时映射数值高低（渐变色阶），本方法条形附加粗边框强调。

    Args:
        methods: 方法名称列表。
        fault_names: 故障类型名称。
        f1_matrix: shape (n_methods, n_faults)。

    Returns:
        输出文件路径。
    """
    mat = np.array(f1_matrix, dtype=float)
    n_methods, n_faults = mat.shape
    clean_names = [_clean_method_name(m) for m in methods]

    cmap_base = mcolors.LinearSegmentedColormap.from_list(
        'f1_grade', ['#E8D5D0', '#D4A59A', '#C07A6E', '#8B514A', '#5E3530'])
    cmap_ours = mcolors.LinearSegmentedColormap.from_list(
        'f1_ours', ['#C5D5E8', '#8BAFD0', '#5588B5', '#2B6A9E', '#1A4A72'])

    n_cols = min(n_faults, 5)
    n_rows_plot = (n_faults + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows_plot, n_cols,
        figsize=(3.2 * n_cols, max(n_methods * 0.55 + 1.0, 3.0) * n_rows_plot),
        sharey=True,
    )
    if n_rows_plot == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows_plot == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    y_pos = np.arange(n_methods)[::-1]
    bar_h = 0.55

    for f_idx in range(n_faults):
        r, c = divmod(f_idx, n_cols)
        ax = axes[r, c]
        col_vals = mat[:, f_idx]
        v_max = max(col_vals.max() * 1.25, 0.15)

        _add_row_stripes(ax, n_methods)

        for i in range(n_methods):
            val = col_vals[i]
            norm_v = np.clip(val, 0, 1)
            if i == 0:
                color = cmap_ours(0.25 + 0.7 * norm_v)
                edge_c, edge_w, alpha = _OURS_EDGE, 1.4, 0.92
            else:
                color = cmap_base(0.15 + 0.75 * norm_v)
                edge_c, edge_w, alpha = 'white', 0.5, _BASELINE_ALPHA

            ax.barh(y_pos[i], val, height=bar_h,
                    color=color, alpha=alpha,
                    edgecolor=edge_c, linewidth=edge_w, zorder=3)

            fmt = f'{val:.2f}'
            _annotate_bar_value(
                ax, val, y_pos[i], fmt, color='#333333',
                bold=(i == 0), x_max=v_max,
                fontsize=PLOT_CONFIG.FONT_SIZE_TEXT - 1,
            )

        ax.set_xlim(0, v_max)
        ax.set_yticks(y_pos)
        if c == 0:
            ax.set_yticklabels(clean_names)
            _highlight_ours_ytick(ax, clean_names)
        else:
            ax.set_yticklabels([])

        sub = chr(97 + f_idx)
        _style_ax(ax, title=f'({sub}) {fault_names[f_idx]}', xlabel='F1-Score')

    for f_idx in range(n_faults, n_rows_plot * n_cols):
        r, c = divmod(f_idx, n_cols)
        axes[r, c].set_visible(False)

    fig.suptitle('各故障类型 F1-Score 对比',
                 fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE + 1,
                 fontweight='bold', x=0.02, ha='left', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save_figure(fig, 'experiments', 'per_fault_f1_grouped_bars')


# ============================================================================
# 图 3: 消融实验瀑布条形图
# ============================================================================

def plot_ablation_delta_bars(
    ablation_data: Dict[str, Dict[str, float]],
    baseline_key: str = '完整协同',
    metric_name: str = '综合奖励',
) -> str:
    """消融实验瀑布图——以完整模型为基准，展示各变体性能差异。

    基准条蓝色完整绘制，其余变体只画增量差值；
    红色=性能下降，绿色=性能提升；阶梯虚线连接瀑布效果。

    Args:
        ablation_data: {变体名: {指标: 值}} 字典。
        baseline_key: 基准方法名。
        metric_name: 展示的指标名称。

    Returns:
        输出文件路径。
    """
    if baseline_key not in ablation_data:
        raise ValueError('Baseline key not found in ablation data.')

    base_val = ablation_data[baseline_key][metric_name]

    others = {k: v[metric_name] for k, v in ablation_data.items()
              if k != baseline_key}
    sorted_items = sorted(others.items(), key=lambda x: x[1], reverse=True)

    labels = [_clean_method_name(baseline_key)] + \
             [_clean_method_name(k) for k, _ in sorted_items]
    values = [base_val] + [v for _, v in sorted_items]
    deltas = [0.0] + [v - base_val for _, v in sorted_items]

    n = len(labels)
    fig, ax = plt.subplots(figsize=(9, max(n * 0.85 + 1.5, 4.0)))
    y_pos = np.arange(n)[::-1]
    bar_h = 0.52

    colors_f = []
    colors_e = []
    for i, d in enumerate(deltas):
        if i == 0:
            colors_f.append(_OURS_COLOR)
            colors_e.append(_OURS_EDGE)
        elif d >= 0:
            colors_f.append('#5E8A5E')
            colors_e.append('#3E6A3E')
        else:
            colors_f.append('#B8514D')
            colors_e.append('#8A3A37')

    for i in range(n):
        if i == 0:
            ax.barh(y_pos[i], values[i], height=bar_h,
                    color=colors_f[i], alpha=0.92,
                    edgecolor=colors_e[i], linewidth=1.4, zorder=3)
        else:
            ax.barh(y_pos[i], base_val, height=bar_h * 0.10,
                    color='#E0E0E0', edgecolor='none', zorder=1)
            left = min(base_val, values[i])
            width = abs(deltas[i])
            ax.barh(y_pos[i], width, left=left, height=bar_h,
                    color=colors_f[i], alpha=0.85,
                    edgecolor=colors_e[i], linewidth=1.0, zorder=3)
            ax.plot([base_val, base_val],
                    [y_pos[i] + bar_h * 0.6, y_pos[i] - bar_h * 0.6],
                    color='#BBBBBB', linewidth=0.8, linestyle='--', zorder=2)

    ax.axvline(base_val, color='#AAAAAA', linewidth=0.8,
               linestyle=':', zorder=1, alpha=0.6)
    ax.text(base_val, y_pos[0] + 0.55, f'基准 = {base_val:.1f}',
            ha='center', va='bottom',
            fontsize=PLOT_CONFIG.FONT_SIZE_TEXT - 1,
            color='#666666', style='italic')

    for i in range(n):
        val = values[i]
        x_max_ref = base_val * 1.3
        if i == 0:
            fmt = f'{val:.1f}'
            _annotate_bar_value(ax, val, y_pos[i], fmt,
                                color=colors_f[i], bold=True, x_max=x_max_ref,
                                fontsize=PLOT_CONFIG.FONT_SIZE_TEXT)
        else:
            sign = '+' if deltas[i] >= 0 else ''
            fmt = f'{val:.1f}  ({sign}{deltas[i]:.1f})'
            disp_x = max(values[i], base_val)
            _annotate_bar_value(ax, disp_x, y_pos[i], fmt,
                                color=colors_f[i], bold=False, x_max=x_max_ref,
                                fontsize=PLOT_CONFIG.FONT_SIZE_TEXT)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    _highlight_ours_ytick(ax, labels)

    ax.set_xlim(0, base_val * 1.38)
    _style_ax(ax, title=f'消融实验 — {metric_name}', xlabel=metric_name)

    legend_elements = [
        Patch(facecolor=_OURS_COLOR, edgecolor=_OURS_EDGE, label='完整模型（基准）'),
        Patch(facecolor='#B8514D', edgecolor='#8A3A37', label='性能下降'),
        Patch(facecolor='#5E8A5E', edgecolor='#3E6A3E', label='性能提升'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND, framealpha=0.85,
              edgecolor='#DDDDDD')

    fig.tight_layout()
    return save_figure(fig, 'experiments', 'ablation_delta_bars')


# ============================================================================
# 图 4: 鲁棒性渐变条形图 + 样条趋势线
# ============================================================================

def plot_severity_robustness_bars(
    severity_levels: List[float],
    accuracy_values: List[float],
) -> str:
    """鲁棒性渐变条形图——颜色从浅蓝到深蓝映射严重度。

    叠加样条趋势线，右上角标注总体变化 (Δ)。

    Args:
        severity_levels: 故障严重度等级列表。
        accuracy_values: 各严重度下的平均诊断准确率。

    Returns:
        输出文件路径。
    """
    n = len(severity_levels)
    x = np.arange(n)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'severity_grade',
        ['#B8D4E8', '#7EB0D0', '#4A8AB8', '#2B6A9E', '#1A4A72'],
    )
    bar_colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(n * 1.1 + 2, 8), 5.0))

    bars = ax.bar(x, accuracy_values, width=0.62,
                  color=bar_colors, edgecolor='white', linewidth=1.0, zorder=3)

    # 样条趋势线
    if n >= 3:
        try:
            from scipy.interpolate import make_interp_spline
            x_smooth = np.linspace(0, n - 1, 200)
            spl = make_interp_spline(x, accuracy_values, k=min(3, n - 1))
            y_smooth = spl(x_smooth)
            ax.plot(x_smooth, y_smooth, color=_OURS_COLOR,
                    linewidth=LINE_WIDTH_HERO, alpha=0.45, zorder=4)
        except Exception:
            ax.plot(x, accuracy_values, color=_OURS_COLOR,
                    linewidth=LINE_WIDTH_MAIN, alpha=0.45,
                    marker='o', markersize=4, zorder=4)

    for i, (bar, val) in enumerate(zip(bars, accuracy_values)):
        y_off = max(accuracy_values) * 0.018
        ax.text(bar.get_x() + bar.get_width() / 2, val + y_off,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=PLOT_CONFIG.FONT_SIZE_TEXT - 1,
                color=bar_colors[i],
                fontweight='bold' if i >= n - 2 else 'normal')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{v:.2f}' for v in severity_levels])
    ax.set_ylim(0, max(accuracy_values) * 1.14)

    despine(ax)
    ax.set_facecolor('#FDFDFD')
    ax.grid(axis='y', color='#E8E8E8', linewidth=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title('鲁棒性分析 — 故障严重度 vs 诊断准确率',
                 fontsize=PLOT_CONFIG.FONT_SIZE_TITLE,
                 fontweight='bold', loc='left', pad=8)
    ax.set_xlabel('故障严重度', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    ax.set_ylabel('平均诊断准确率', fontsize=PLOT_CONFIG.FONT_SIZE_LABEL)
    set_tick_fontsize(ax, PLOT_CONFIG.FONT_SIZE_TICK)

    trend_dir = '↑' if accuracy_values[-1] > accuracy_values[0] else '↓'
    delta = accuracy_values[-1] - accuracy_values[0]
    ax.text(0.97, 0.95, f'Δ = {delta:+.2f} {trend_dir}',
            transform=ax.transAxes, fontsize=PLOT_CONFIG.FONT_SIZE_TEXT,
            ha='right', va='top', color=_OURS_COLOR, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#DDDDDD', alpha=0.9))

    fig.tight_layout()
    return save_figure(fig, 'experiments', 'robustness_severity_bars')


# ============================================================================
# 图 5: 多次运行统计 CI 对比图（新增）
# ============================================================================

def plot_multi_run_statistics(
    methods: List[str],
    metric_name: str,
    means: List[float],
    stds: List[float],
    n_runs: int = 50,
) -> str:
    """多次运行统计对比——水平 bar + 95% CI 误差棒 + 显著性 bracket。

    按均值升序排列（最优在顶部），本方法高亮。

    Args:
        methods: 方法名称列表。
        metric_name: 指标名称。
        means: 各方法均值。
        stds: 各方法标准差。
        n_runs: 运行次数（用于计算 95% CI）。

    Returns:
        输出文件路径。
    """
    n = len(methods)
    clean_names = [_clean_method_name(m) for m in methods]
    ci_95 = [1.96 * s / np.sqrt(n_runs) for s in stds]

    order = np.argsort(means)
    names_sorted = [clean_names[i] for i in order]
    means_sorted = [means[i] for i in order]
    ci_sorted = [ci_95[i] for i in order]
    orig_sorted = list(order)

    y_pos = np.arange(n)
    bar_h = 0.52
    fig, ax = plt.subplots(figsize=(9, max(n * 0.85 + 1.5, 4.0)))

    _add_row_stripes(ax, n)

    for i in range(n):
        fc, fa, ec, ew = _get_method_color(orig_sorted[i], n)

        ax.barh(y_pos[i], means_sorted[i], height=bar_h,
                color=fc, alpha=fa, edgecolor=ec, linewidth=ew, zorder=3)

        ax.errorbar(means_sorted[i], y_pos[i],
                    xerr=ci_sorted[i], fmt='none',
                    ecolor='#333333', elinewidth=1.8,
                    capsize=4, capthick=1.5, zorder=5)

        text = f'{means_sorted[i]:.3f} ± {ci_sorted[i]:.3f}'
        x_max_val = max(means_sorted) * 1.35
        _annotate_bar_value(
            ax, means_sorted[i] + ci_sorted[i], y_pos[i], text,
            color=(fc if i < n - 1 else _OURS_EDGE),
            bold=(i == n - 1), x_max=x_max_val,
            fontsize=PLOT_CONFIG.FONT_SIZE_TEXT - 1,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted)
    _highlight_ours_ytick(ax, clean_names)

    ax.set_xlim(0, max(means_sorted) * 1.38)
    _style_ax(ax,
              title=f'{metric_name} — 统计对比 (95% CI, n={n_runs})',
              xlabel=metric_name)

    # 显著性 bracket：最优 vs 次优
    if n >= 2:
        best_m, second_m = means_sorted[-1], means_sorted[-2]
        best_c, second_c = ci_sorted[-1], ci_sorted[-2]
        sig = 'p < 0.05 *' if best_m - best_c > second_m + second_c else 'n.s.'

        x_brk = max(means_sorted) * 1.12
        y_top, y_sec = y_pos[-1], y_pos[-2]
        ax.plot([x_brk, x_brk], [y_sec, y_top],
                color='#555555', linewidth=1.0, zorder=6)
        tick_w = max(means_sorted) * 0.012
        ax.plot([x_brk - tick_w, x_brk], [y_top, y_top],
                color='#555555', linewidth=1.0, zorder=6)
        ax.plot([x_brk - tick_w, x_brk], [y_sec, y_sec],
                color='#555555', linewidth=1.0, zorder=6)
        ax.text(x_brk + tick_w, (y_top + y_sec) / 2, sig,
                fontsize=PLOT_CONFIG.FONT_SIZE_TEXT - 2,
                ha='left', va='center', color='#555555', style='italic')

    fig.tight_layout()
    fname = f'multi_run_statistics_{metric_name.replace(" ", "_")}'
    return save_figure(fig, 'experiments', fname)


# ============================================================================
# 公共辅助函数
# ============================================================================

def build_avg_severity_accuracy(
    severity_sensitivity: Dict[str, Dict[str, List[float]]],
) -> Tuple[List[float], List[float]]:
    """汇总各故障类型的严重度-准确率数据，返回平均准确率。

    Args:
        severity_sensitivity: {故障名: {severity_levels, accuracy}} 字典。

    Returns:
        (severity_levels, avg_accuracy) 元组。
    """
    severity_levels = None
    accuracy_stack = []

    for fault_data in severity_sensitivity.values():
        levels = fault_data.get('severity_levels', [])
        acc = fault_data.get('accuracy', [])
        if not levels or not acc:
            continue
        if severity_levels is None:
            severity_levels = levels
        accuracy_stack.append(acc)

    if severity_levels is None or not accuracy_stack:
        raise ValueError('No usable severity sensitivity data.')

    avg_accuracy = np.mean(np.array(accuracy_stack), axis=0).tolist()
    return severity_levels, avg_accuracy
