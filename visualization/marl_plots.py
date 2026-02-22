#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MARL 训练可视化模块
====================
双智能体强化学习训练过程与结果的学术风格可视化。

函数列表 (基础 7 张):
- plot_training_curves(history)        - 训练曲线（奖励/损失/熵/准确率）
- plot_reward_distribution(history)    - 奖励分布直方图
- plot_confusion_matrix(cm, ...)       - 故障诊断混淆矩阵
- plot_detection_delay(delays)         - 故障检测延迟箱型图
- plot_control_response(...)           - 容错控制响应分析
- plot_method_comparison(methods, ...) - 方法性能对比柱状图
- plot_dual_agent_architecture()       - 双智能体网络架构图

诊断智能体评价 (5 张):
- plot_diagnostic_roc_curves(...)            - 多故障类型 ROC 曲线
- plot_diagnostic_precision_recall(...)      - 精确率-召回率曲线
- plot_fault_severity_sensitivity(...)       - 故障严重程度敏感性分析
- plot_diagnostic_embedding_tsne(...)        - 诊断特征空间 t-SNE 可视化
- plot_diagnostic_online_accuracy(...)       - 在线诊断准确率演化

控制智能体评价 (5 张):
- plot_control_multi_setpoint_tracking(...)  - 多工况设定值跟踪
- plot_control_action_distribution(...)      - 控制动作分布
- plot_control_robustness_envelope(...)      - 控制鲁棒性包络
- plot_control_constraint_satisfaction(...)  - 约束满足率雷达图
- plot_control_energy_efficiency(...)        - 控制效率分析

控诊协同评价 (6 张):
- plot_collaborative_timeline(...)             - 控诊协同时序图
- plot_collaborative_reward_decomposition(...) - 协同奖励分解
- plot_collaborative_fault_response_matrix(...)- 故障-响应矩阵热力图
- plot_collaborative_ablation_study(...)       - 协同消融实验对比
- plot_collaborative_information_flow(...)     - 双智能体信息流示意图
- plot_collaborative_pareto_front(...)         - 诊断-控制 Pareto 前沿

Author: CDC Project
Date: 2026-02-22
"""

# 标准库
from typing import Dict, List, Optional, Tuple

# 第三方库
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# 本项目模块
from config import PLOT_CONFIG, COLORS, setup_matplotlib_style, save_figure
from visualization.style import (
    set_tick_fontsize,
    LINE_WIDTH_HERO,
    LINE_WIDTH_MAIN,
    LINE_WIDTH_SECONDARY,
    LINE_WIDTH_GHOST,
    MARKER_SIZE_DEFAULT,
    MARKER_STYLE_HERO,
    MARKER_STYLE_SECONDARY,
    ACADEMIC_STATS_BOX,
    ACADEMIC_GRID,
    ACADEMIC_CONFIDENCE_BAND,
    ACADEMIC_CONFIDENCE_BAND_DARK,
)

# 应用全局学术样式（模块级别，只调用一次）
setup_matplotlib_style()


# ============================================================================
# 辅助函数
# ============================================================================

def _smooth(data: List, window: int = 20) -> np.ndarray:
    """滑动平均平滑

    Args:
        data: 原始数据列表
        window: 平滑窗口大小

    Returns:
        平滑后的 ndarray
    """
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode='valid')


# ============================================================================
# 训练过程可视化
# ============================================================================

def plot_training_curves(
    history: Dict[str, List],
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    绘制双智能体训练曲线（顶刊级层次化布局）

    布局: 主图(奖励曲线, 占上部60%面积, 跨3列) + 收敛放大inset(右上)
          底部4个辅助子图: 策略损失 / 价值损失 / 策略熵 / 准确率&性能

    Args:
        history: 训练历史字典，键包括 'episodes', 'reward_diag', 'reward_ctrl',
                 'loss_diag_policy', 'loss_ctrl_policy', 'loss_diag_value',
                 'loss_ctrl_value', 'entropy_diag', 'entropy_ctrl',
                 'diag_accuracy', 'ctrl_performance'
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象（已保存至 visualization_output/training/）
    """
    import pandas as pd

    tick_size  = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE
    sup_size   = PLOT_CONFIG.FONT_SIZE_SUPTITLE

    # 使用顶刊级双智能体配色
    color_diag = COLORS.get('agent_diag', COLORS['primary'])
    color_ctrl = COLORS.get('agent_ctrl', COLORS['danger'])

    episodes = (history.get('episodes')
                or list(range(len(history.get('reward_diag', [])))))

    fig = plt.figure(figsize=figsize)

    # ── 层次化布局：主图大、辅助小 ──
    gs = GridSpec(3, 4, figure=fig, hspace=0.38, wspace=0.35,
                  height_ratios=[2.5, 1, 1])

    # ══ 主图：双智能体奖励曲线（跨3列）══
    ax_main = fig.add_subplot(gs[0, :3])

    if history.get('reward_diag'):
        raw = history['reward_diag']
        # 半透明原始数据作为背景纹理
        ax_main.plot(episodes[:len(raw)], raw, color=color_diag, alpha=0.08,
                     linewidth=LINE_WIDTH_GHOST)
        sm = _smooth(raw, window=30)
        ep_sm = episodes[len(episodes) - len(sm):]
        ax_main.plot(ep_sm, sm,
                     color=color_diag, linewidth=LINE_WIDTH_HERO,
                     label='诊断智能体', **MARKER_STYLE_HERO,
                     markevery=max(1, len(sm) // 8))
        # ± 1σ 置信带
        _rolling_std = pd.Series(raw).rolling(30, min_periods=1).std().values
        _rolling_mean = pd.Series(raw).rolling(30, min_periods=1).mean().values
        ax_main.fill_between(episodes[:len(raw)],
                             _rolling_mean - _rolling_std,
                             _rolling_mean + _rolling_std,
                             color=color_diag, **ACADEMIC_CONFIDENCE_BAND)

    if history.get('reward_ctrl'):
        raw = history['reward_ctrl']
        ax_main.plot(episodes[:len(raw)], raw, color=color_ctrl, alpha=0.08,
                     linewidth=LINE_WIDTH_GHOST)
        sm = _smooth(raw, window=30)
        ep_sm = episodes[len(episodes) - len(sm):]
        ax_main.plot(ep_sm, sm,
                     color=color_ctrl, linewidth=LINE_WIDTH_HERO,
                     label='控制智能体', **MARKER_STYLE_SECONDARY,
                     markevery=max(1, len(sm) // 8))
        _rolling_std = pd.Series(raw).rolling(30, min_periods=1).std().values
        _rolling_mean = pd.Series(raw).rolling(30, min_periods=1).mean().values
        ax_main.fill_between(episodes[:len(raw)],
                             _rolling_mean - _rolling_std,
                             _rolling_mean + _rolling_std,
                             color=color_ctrl, **ACADEMIC_CONFIDENCE_BAND)

    ax_main.set_xlabel('训练轮次', fontsize=label_size)
    ax_main.set_ylabel('累积奖励', fontsize=label_size)
    ax_main.set_title('(a) 双智能体奖励收敛曲线', fontsize=title_size, fontweight='bold')
    ax_main.legend(fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND, loc='lower right',
                   framealpha=0.9, edgecolor='#CCCCCC')
    ax_main.grid(**ACADEMIC_GRID)
    set_tick_fontsize(ax_main, tick_size)

    # ── 右侧 inset：收敛阶段放大 ──
    ax_inset = fig.add_subplot(gs[0, 3])
    n_last = max(1, len(episodes) // 4)
    if history.get('reward_diag') and n_last > 0:
        sm_d = _smooth(history['reward_diag'], 15)
        ax_inset.plot(episodes[-n_last:], sm_d[-n_last:],
                      color=color_diag, linewidth=LINE_WIDTH_MAIN)
    if history.get('reward_ctrl') and n_last > 0:
        sm_c = _smooth(history['reward_ctrl'], 15)
        ax_inset.plot(episodes[-n_last:], sm_c[-n_last:],
                      color=color_ctrl, linewidth=LINE_WIDTH_MAIN)
    ax_inset.set_title('收敛阶段放大', fontsize=PLOT_CONFIG.FONT_SIZE_TEXT)
    ax_inset.grid(**ACADEMIC_GRID)
    set_tick_fontsize(ax_inset, tick_size - 2)

    # ── 底部辅助子图（2行×4列 → 实际用第1-2行各4列）──
    aux_specs = [
        ('loss_diag_policy', 'loss_ctrl_policy', '(b) 策略损失'),
        ('loss_diag_value', 'loss_ctrl_value', '(c) 价值损失'),
        ('entropy_diag', 'entropy_ctrl', '(d) 策略熵'),
        (None, None, '(e) 准确率/性能'),  # 特殊处理
    ]
    for col_idx, (key_d, key_c, title_str) in enumerate(aux_specs):
        ax = fig.add_subplot(gs[1 + col_idx // 4, col_idx % 4])
        if col_idx < 3:
            if key_d and history.get(key_d):
                sm = _smooth(history[key_d])
                ax.plot(episodes[len(episodes) - len(sm):], sm,
                        color=color_diag, linewidth=LINE_WIDTH_MAIN, label='诊断')
            if key_c and history.get(key_c):
                sm = _smooth(history[key_c])
                ax.plot(episodes[len(episodes) - len(sm):], sm,
                        color=color_ctrl, linewidth=LINE_WIDTH_MAIN, label='控制')
            ax.legend(fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND - 2, loc='best')
        else:
            # 准确率 + 性能
            color_acc = COLORS.get('nature_green', COLORS['success'])
            color_perf = COLORS.get('nature_purple', COLORS['secondary'])
            if history.get('diag_accuracy'):
                sm = _smooth(history['diag_accuracy'])
                ax.plot(episodes[len(episodes) - len(sm):], sm,
                        color=color_acc, linewidth=LINE_WIDTH_MAIN, label='诊断准确率')
                ax.axhline(y=0.9, color=COLORS['dark'], linestyle='--',
                           linewidth=LINE_WIDTH_GHOST, alpha=0.5)
            if history.get('ctrl_performance'):
                sm = _smooth(history['ctrl_performance'])
                ax.plot(episodes[len(episodes) - len(sm):], sm,
                        color=color_perf, linewidth=LINE_WIDTH_MAIN, label='控制性能')
            ax.set_ylim([0, 1.05])
            ax.legend(fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND - 2, loc='best')
        ax.set_title(title_str, fontsize=PLOT_CONFIG.FONT_SIZE_TEXT, fontweight='bold')
        ax.grid(**ACADEMIC_GRID)
        set_tick_fontsize(ax, tick_size - 2)

    fig.suptitle('双智能体强化学习训练过程', fontsize=sup_size, fontweight='bold')
    save_figure(fig, 'training', 'training_curves')
    return fig


def plot_reward_distribution(
    history: Dict[str, List],
    figsize: Tuple[int, int] = (10, 7)
) -> plt.Figure:
    """
    绘制奖励分布（小提琴图 + 蜂群散点，替代传统柱状图/直方图）

    Args:
        history: 训练历史字典（包含 reward_diag / reward_ctrl / reward_total）
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象（已保存至 visualization_output/training/）
    """
    tick_size  = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    text_size  = PLOT_CONFIG.FONT_SIZE_TEXT

    color_diag  = COLORS.get('agent_diag', COLORS['primary'])
    color_ctrl  = COLORS.get('agent_ctrl', COLORS['danger'])
    color_total = COLORS.get('agent_shared', COLORS['purple'])

    fig, ax = plt.subplots(figsize=figsize)

    data_list = []
    color_list = []
    label_list = []

    for key, color, label in [
        ('reward_diag', color_diag, '诊断智能体'),
        ('reward_ctrl', color_ctrl, '控制智能体'),
        ('reward_total', color_total, '总奖励'),
    ]:
        vals = history.get(key, [])
        if vals:
            data_list.append(vals)
            color_list.append(color)
            label_list.append(label)

    if data_list:
        parts = ax.violinplot(data_list, positions=range(len(data_list)),
                              showmeans=True, showextrema=False, widths=0.7)

        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(color_list[i])
            pc.set_alpha(0.3)
            pc.set_edgecolor(color_list[i])
            pc.set_linewidth(1.5)

        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(2)

        # 叠加蜂群散点（采样避免过密）
        rng = np.random.default_rng(42)
        for i, vals in enumerate(data_list):
            n_sample = min(200, len(vals))
            sample_idx = rng.choice(len(vals), n_sample, replace=False)
            sampled = np.array(vals)[sample_idx]
            jitter = rng.normal(0, 0.06, n_sample)
            ax.scatter(np.full(n_sample, i) + jitter, sampled,
                       s=8, alpha=0.4, color=color_list[i],
                       edgecolors='white', linewidths=0.3, zorder=5)

            # 统计标注
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            ax.text(i, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else max(vals) * 1.1,
                    f'$\\mu$={mean_val:.2f}\n$\\sigma$={std_val:.2f}',
                    ha='center', va='bottom', fontsize=text_size - 1,
                    color=color_list[i], fontweight='bold')

        ax.set_xticks(range(len(label_list)))
        ax.set_xticklabels(label_list, fontsize=label_size)

    ax.set_ylabel('奖励值', fontsize=label_size)
    ax.set_title('双智能体奖励分布', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 fontweight='bold')
    ax.grid(**ACADEMIC_GRID)
    set_tick_fontsize(ax, tick_size)

    plt.tight_layout()
    save_figure(fig, 'training', 'reward_distribution')
    return fig


# ============================================================================
# 评估结果可视化
# ============================================================================

def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (9, 8)
) -> plt.Figure:
    """
    绘制故障诊断混淆矩阵（顶刊风格：自定义渐变色 + 边际精度/召回条）

    Args:
        confusion_matrix: n×n 混淆矩阵（行=真实，列=预测）
        class_names: 类别标签列表，默认 ['健康', '正时故障', '泄漏故障', '燃油故障']
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象（已保存至 visualization_output/training/）
    """
    from matplotlib.colors import LinearSegmentedColormap

    if class_names is None:
        class_names = ['健康', '正时故障', '泄漏故障', '燃油故障']

    n = len(class_names)
    tick_size  = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    text_size  = PLOT_CONFIG.FONT_SIZE_TEXT

    # 计算精确度和召回率
    col_sum = confusion_matrix.sum(axis=0)
    row_sum = confusion_matrix.sum(axis=1)
    precision = np.diag(confusion_matrix) / (col_sum + 1e-10)
    recall = np.diag(confusion_matrix) / (row_sum + 1e-10)
    overall_acc = np.diag(confusion_matrix).sum() / (confusion_matrix.sum() + 1e-10)

    # 带边际分布的布局
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[5, 1.2], height_ratios=[5, 1.2],
                  hspace=0.08, wspace=0.08)

    # 自定义渐变色板 (白 → 浅蓝 → 深蓝)
    cmap = LinearSegmentedColormap.from_list(
        'academic_cm', ['#FFFFFF', '#BEE3F8', '#63B3ED', '#3182CE', '#1A365D'])

    # ── 主热力图 ──
    ax_main = fig.add_subplot(gs[0, 0])
    im = ax_main.imshow(confusion_matrix, cmap=cmap, aspect='equal')

    threshold = confusion_matrix.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = confusion_matrix[i, j]
            ax_main.text(j, i, f'{val}', ha='center', va='center',
                         fontsize=text_size + 2, fontweight='bold',
                         color='white' if val > threshold else '#2D3748')

    ax_main.set_xticks(range(n))
    ax_main.set_xticklabels(class_names, fontsize=tick_size)
    ax_main.set_yticks(range(n))
    ax_main.set_yticklabels(class_names, fontsize=tick_size)
    ax_main.set_ylabel('真实类别', fontsize=label_size)
    ax_main.xaxis.tick_top()
    ax_main.xaxis.set_label_position('top')
    ax_main.set_xlabel('预测类别', fontsize=label_size)
    # 恢复主图 spine
    for spine in ax_main.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color('#333333')

    # ── 右侧边际：召回率 ──
    color_diag = COLORS.get('agent_diag', COLORS['primary'])
    color_ctrl = COLORS.get('agent_ctrl', COLORS['danger'])

    ax_recall = fig.add_subplot(gs[0, 1])
    bars_r = ax_recall.barh(range(n), recall * 100, color=color_diag,
                            alpha=0.75, edgecolor='white', linewidth=1.5, height=0.6)
    for i, (bar, val) in enumerate(zip(bars_r, recall)):
        ax_recall.text(bar.get_width() + 1.5, i, f'{val*100:.1f}%',
                       va='center', fontsize=text_size - 1, fontweight='bold',
                       color=color_diag)
    ax_recall.set_xlim(0, 118)
    ax_recall.set_title('召回率', fontsize=text_size, fontweight='bold')
    ax_recall.set_yticks(range(n))
    ax_recall.set_yticklabels([])
    ax_recall.tick_params(left=False)
    ax_recall.grid(**ACADEMIC_GRID)
    set_tick_fontsize(ax_recall, tick_size - 2)

    # ── 底部边际：精确度 ──
    ax_prec = fig.add_subplot(gs[1, 0])
    bars_p = ax_prec.bar(range(n), precision * 100, color=color_ctrl,
                         alpha=0.75, edgecolor='white', linewidth=1.5, width=0.6)
    for i, (bar, val) in enumerate(zip(bars_p, precision)):
        ax_prec.text(i, bar.get_height() + 1.5, f'{val*100:.1f}%',
                     ha='center', fontsize=text_size - 1, fontweight='bold',
                     color=color_ctrl)
    ax_prec.set_ylim(0, 118)
    ax_prec.set_ylabel('精确度', fontsize=text_size, fontweight='bold')
    ax_prec.set_xticks(range(n))
    ax_prec.set_xticklabels([])
    ax_prec.tick_params(bottom=False)
    ax_prec.grid(**ACADEMIC_GRID)
    set_tick_fontsize(ax_prec, tick_size - 2)

    # ── 右下角：总体准确率 ──
    ax_info = fig.add_subplot(gs[1, 1])
    ax_info.axis('off')
    ax_info.text(0.5, 0.5, f'Acc\n{overall_acc*100:.1f}%',
                 ha='center', va='center', fontsize=text_size + 2,
                 fontweight='bold', color=COLORS.get('agent_shared', COLORS['purple']),
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#F7FAFC',
                           edgecolor='#CBD5E0', linewidth=1.5))

    fig.suptitle('故障诊断混淆矩阵', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 fontweight='bold', y=1.02)

    save_figure(fig, 'training', 'confusion_matrix')
    return fig


def plot_detection_delay(
    delays: Dict[str, List[float]],
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    绘制故障检测延迟箱型图

    Args:
        delays: {故障名称: [延迟(步数), ...]} 字典
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象（已保存至 visualization_output/training/）
    """
    tick_size  = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    text_size  = PLOT_CONFIG.FONT_SIZE_TEXT

    palette = [
        COLORS.get('nature_red', COLORS['danger']),
        COLORS.get('nature_cyan', COLORS['info']),
        COLORS.get('nature_yellow', COLORS['warning']),
        COLORS.get('nature_green', COLORS['success']),
        COLORS.get('nature_purple', COLORS['purple']),
    ]

    fig, ax = plt.subplots(figsize=figsize)

    labels = list(delays.keys())
    data   = [delays[k] for k in labels]
    pos    = list(range(len(labels)))

    # 小提琴底层
    if all(len(d) > 1 for d in data):
        vp = ax.violinplot(data, positions=pos, showmeans=False,
                           showextrema=False, widths=0.7)
        for i, body in enumerate(vp['bodies']):
            body.set_facecolor(palette[i % len(palette)])
            body.set_alpha(0.2)
            body.set_edgecolor(palette[i % len(palette)])
            body.set_linewidth(1.0)

    # 箱线图叠加
    bp = ax.boxplot(data, positions=pos, patch_artist=True, widths=0.35,
                    zorder=4)

    for patch, color in zip(bp['boxes'], palette[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
    for element in ['whiskers', 'caps']:
        plt.setp(bp[element], color=COLORS['dark'], linewidth=LINE_WIDTH_SECONDARY)
    plt.setp(bp['medians'], color='white', linewidth=2.0)
    plt.setp(bp['fliers'], markerfacecolor=COLORS['dark'], marker='o',
             markersize=3, alpha=0.4)

    # 蜂群散点叠加
    rng = np.random.default_rng(42)
    for i, d in enumerate(data):
        n_pts = min(80, len(d))
        idx = rng.choice(len(d), n_pts, replace=False) if len(d) > n_pts else range(len(d))
        jitter = rng.normal(0, 0.05, len(idx))
        ax.scatter(np.full(len(idx), i) + jitter, np.array(d)[idx],
                   s=12, alpha=0.5, color=palette[i % len(palette)],
                   edgecolors='white', linewidths=0.3, zorder=5)

    ax.axhline(y=5, color=COLORS.get('agent_ctrl', COLORS['danger']),
               linestyle='--', linewidth=LINE_WIDTH_MAIN, alpha=0.7,
               label='目标: < 5 cycles')

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=tick_size)
    ax.set_ylabel('检测延迟（循环数）', fontsize=label_size)
    ax.set_title('不同故障类型的检测延迟分布', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 fontweight='bold')
    ax.legend(fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND)
    ax.grid(axis='y', **ACADEMIC_GRID)
    set_tick_fontsize(ax, tick_size)

    plt.tight_layout()
    save_figure(fig, 'training', 'detection_delay')
    return fig


def plot_control_response(
    time_steps: np.ndarray,
    fault_severity: np.ndarray,
    pmax_actual: np.ndarray,
    pmax_target: float,
    timing_offset: np.ndarray,
    fuel_adj: np.ndarray,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    绘制容错控制响应分析（4 行子图）

    Args:
        time_steps: 时间步数组
        fault_severity: 故障严重程度 [0, 1]
        pmax_actual: 实际 Pmax 数组 [bar]
        pmax_target: 目标 Pmax 标量 [bar]
        timing_offset: 正时补偿数组 [deg]
        fuel_adj: 燃油调整系数数组
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象（已保存至 visualization_output/training/）
    """
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE
    tick_size  = PLOT_CONFIG.FONT_SIZE_TICK

    color_diag = COLORS.get('agent_diag', COLORS['primary'])
    color_ctrl = COLORS.get('agent_ctrl', COLORS['danger'])

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.35)

    # (a) 故障严重程度 — 渐变填充
    axes[0].fill_between(time_steps, 0, fault_severity,
                         alpha=0.4, color=color_ctrl, label='故障严重程度')
    axes[0].plot(time_steps, fault_severity, color=color_ctrl,
                 linewidth=LINE_WIDTH_SECONDARY, alpha=0.8)
    axes[0].set_ylabel('故障强度', fontsize=label_size)
    axes[0].set_title('(a) 故障注入', fontsize=title_size, fontweight='bold')
    axes[0].legend(fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND, loc='upper right')
    axes[0].grid(**ACADEMIC_GRID)
    axes[0].set_ylim([0, 1.1])
    set_tick_fontsize(axes[0], tick_size)

    # (b) Pmax 响应 — hero线 + 置信带
    axes[1].plot(time_steps, pmax_actual,
                 color=color_diag, linewidth=LINE_WIDTH_HERO, label='实际 $P_{max}$')
    axes[1].axhline(y=pmax_target, color=COLORS.get('nature_green', COLORS['success']),
                    linestyle='--', linewidth=LINE_WIDTH_SECONDARY,
                    label=f'目标 $P_{{max}}$ = {pmax_target:.0f} bar')
    axes[1].fill_between(time_steps, pmax_target * 0.95, pmax_target * 1.05,
                         color=COLORS.get('nature_green', COLORS['success']),
                         **ACADEMIC_CONFIDENCE_BAND, label='±5% 容差带')
    axes[1].set_ylabel('$P_{max}$ [bar]', fontsize=label_size)
    axes[1].set_title('(b) 性能维持效果', fontsize=title_size, fontweight='bold')
    axes[1].legend(fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND, loc='upper right')
    axes[1].grid(**ACADEMIC_GRID)
    set_tick_fontsize(axes[1], tick_size)

    # (c) 正时补偿
    axes[2].plot(time_steps, timing_offset,
                 color=COLORS.get('nature_purple', COLORS['purple']),
                 linewidth=LINE_WIDTH_HERO)
    axes[2].axhline(y=0, color=COLORS['dark'], linestyle='-', alpha=0.3,
                    linewidth=LINE_WIDTH_GHOST)
    axes[2].set_ylabel('正时补偿 [deg]', fontsize=label_size)
    axes[2].set_title('(c) 控制智能体 — 正时调整', fontsize=title_size, fontweight='bold')
    axes[2].set_ylim([-6.5, 6.5])
    axes[2].grid(**ACADEMIC_GRID)
    set_tick_fontsize(axes[2], tick_size)

    # (d) 燃油调整
    axes[3].plot(time_steps, fuel_adj,
                 color=COLORS['orange'], linewidth=LINE_WIDTH_HERO)
    axes[3].axhline(y=1.0, color=COLORS['dark'], linestyle='-', alpha=0.3,
                    linewidth=LINE_WIDTH_GHOST)
    axes[3].set_ylabel('燃油系数', fontsize=label_size)
    axes[3].set_xlabel('时间步', fontsize=label_size)
    axes[3].set_title('(d) 控制智能体 — 燃油调整', fontsize=title_size, fontweight='bold')
    axes[3].set_ylim([0.78, 1.22])
    axes[3].grid(**ACADEMIC_GRID)
    set_tick_fontsize(axes[3], tick_size)

    fig.suptitle('容错控制响应分析', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 fontweight='bold')
    save_figure(fig, 'training', 'control_response')
    return fig


def plot_method_comparison(
    methods: List[str],
    metrics: Dict[str, List[float]],
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    绘制方法性能对比（哑铃图/棒棒糖图风格，替代传统柱状图）

    Args:
        methods: 方法名称列表
        metrics: {指标名称: [各方法数值]} 字典
        figsize: 图形尺寸，默认按指标数量自动确定

    Returns:
        fig: matplotlib Figure 对象（已保存至 visualization_output/training/）
    """
    n_metrics = len(metrics)
    n_methods = len(methods)
    if figsize is None:
        figsize = (6 * n_metrics, max(4, n_methods * 0.8 + 2))

    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE
    tick_size  = PLOT_CONFIG.FONT_SIZE_TICK
    text_size  = PLOT_CONFIG.FONT_SIZE_TEXT

    # 为每个方法分配颜色
    palette = [
        COLORS.get('agent_diag', COLORS['primary']),
        COLORS.get('agent_ctrl', COLORS['danger']),
        COLORS.get('nature_green', COLORS['success']),
        COLORS.get('nature_purple', COLORS['purple']),
        COLORS.get('nature_yellow', COLORS['warning']),
        COLORS.get('nature_cyan', COLORS['info']),
    ]

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    for ax_idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[ax_idx]
        y_pos = np.arange(n_methods)
        val_max = max(values) if values else 1.0

        # 绘制水平连接线（灰色基准线）
        for y, val in zip(y_pos, values):
            ax.plot([0, val], [y, y], color='#E2E8F0',
                    linewidth=8, solid_capstyle='round', zorder=1)

        # 绘制数值点（棒棒糖头部）
        for i, (method, val) in enumerate(zip(methods, values)):
            color = palette[i % len(palette)]
            ax.scatter(val, i, s=220, color=color, edgecolors='white',
                       linewidths=2.5, zorder=5)
            ax.text(val + val_max * 0.04, i, f'{val:.2f}',
                    va='center', fontsize=text_size, fontweight='bold',
                    color=color)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods, fontsize=tick_size)
        ax.set_xlabel(metric_name, fontsize=label_size)
        ax.set_title(f'({chr(97 + ax_idx)}) {metric_name}',
                     fontsize=title_size, fontweight='bold')
        ax.grid(axis='x', **ACADEMIC_GRID)
        ax.set_axisbelow(True)
        ax.set_xlim(left=0, right=val_max * 1.2)
        set_tick_fontsize(ax, tick_size)

    fig.suptitle('方法性能对比', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'training', 'method_comparison')
    return fig


# ============================================================================
# 网络架构可视化（category='modeling'）
# ============================================================================

def plot_dual_agent_architecture(figsize: Tuple[int, int] = (16, 8)) -> plt.Figure:
    """
    绘制双智能体网络架构示意图

    Args:
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象（已保存至 visualization_output/modeling/）
    """
    title_size = PLOT_CONFIG.FONT_SIZE_SUPTITLE
    text_size  = PLOT_CONFIG.FONT_SIZE_TEXT

    color_input   = COLORS['primary']
    color_encoder = COLORS['purple']
    color_heads = [COLORS['danger'], COLORS['success'], COLORS['warning'], COLORS['teal']]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    def _draw_agent(ax, title: str, input_label: str, head_labels: List[str]):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(title, fontsize=title_size, fontweight='bold', pad=12)

        # 输入框
        input_box = mpatches.FancyBboxPatch(
            (0.5, 4), 2, 2, boxstyle='round,pad=0.15',
            facecolor=color_input, edgecolor='black', linewidth=1.8, zorder=3
        )
        ax.add_patch(input_box)
        ax.text(1.5, 5, input_label, ha='center', va='center',
                fontsize=text_size, color='white', zorder=4)

        # 编码器框
        enc_box = mpatches.FancyBboxPatch(
            (3.5, 4), 2, 2, boxstyle='round,pad=0.15',
            facecolor=color_encoder, edgecolor='black', linewidth=1.8, zorder=3
        )
        ax.add_patch(enc_box)
        ax.text(4.5, 5, 'MLP 编码器\n64 → 64', ha='center', va='center',
                fontsize=text_size, color='white', zorder=4)

        # 输出头
        y_positions = [8.2, 6.5, 4.8, 3.1]
        for label, color, y in zip(head_labels, color_heads, y_positions):
            head_box = mpatches.FancyBboxPatch(
                (7, y - 0.55), 2.6, 1.1, boxstyle='round,pad=0.1',
                facecolor=color, edgecolor='black', linewidth=1.5, zorder=3
            )
            ax.add_patch(head_box)
            ax.text(8.3, y, label, ha='center', va='center',
                    fontsize=text_size - 1, color='white', zorder=4)

        # 连接箭头
        ax.annotate('', xy=(3.4, 5), xytext=(2.6, 5),
                    arrowprops=dict(arrowstyle='->', lw=2.0, color=COLORS['dark']),
                    zorder=5)
        for y in y_positions:
            ax.annotate(
                '', xy=(6.9, y), xytext=(5.6, 5),
                arrowprops=dict(
                    arrowstyle='->', lw=1.5, color=COLORS['dark'],
                    connectionstyle='arc3,rad=0.05'
                ),
                zorder=5
            )

    # 诊断智能体
    _draw_agent(
        axes[0],
        title='诊断智能体（PINN-KAN）',
        input_label='观测输入\n（含控制历史）',
        head_labels=[
            '故障分类\nSoftmax(4)',
            '严重程度\nBeta 分布',
            '置信度\nBeta 分布',
            'Critic  V(s)',
        ]
    )

    # 控制智能体
    _draw_agent(
        axes[1],
        title='控制智能体（TD-MPC2）',
        input_label='观测输入\n（含诊断结果）',
        head_labels=[
            '正时补偿\n高斯 [±5°]',
            '燃油调整\n高斯 [0.85, 1.15]',
            '保护级别\nSoftmax(4)',
            'Critic  V(s)',
        ]
    )

    plt.tight_layout()
    save_figure(fig, 'modeling', 'dual_agent_architecture')
    return fig


# ============================================================================
# 诊断智能体评价 (5 张)
# ============================================================================

def plot_diagnostic_roc_curves(
    roc_data: Dict[str, Dict[str, list]],
    figsize: Tuple[int, int] = (8, 7)
) -> plt.Figure:
    """
    绘制多故障类型 ROC 曲线

    Args:
        roc_data: {故障类型: {'fpr': [...], 'tpr': [...], 'auc': float}} 字典
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象
    """
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    legend_size = PLOT_CONFIG.FONT_SIZE_LEGEND
    text_size = PLOT_CONFIG.FONT_SIZE_TEXT

    palette = [
        COLORS.get('nature_blue', COLORS['primary']),
        COLORS.get('nature_red', COLORS['danger']),
        COLORS.get('nature_yellow', COLORS['warning']),
        COLORS.get('nature_green', COLORS['success']),
        COLORS.get('nature_purple', COLORS['purple']),
        COLORS.get('nature_cyan', COLORS['orange']),
    ]

    fig, ax = plt.subplots(figsize=figsize)

    for idx, (fault_name, data) in enumerate(roc_data.items()):
        fpr = data['fpr']
        tpr = data['tpr']
        auc_val = data['auc']
        color = palette[idx % len(palette)]
        # 曲线下方填充淡色
        ax.fill_between(fpr, 0, tpr, alpha=0.06, color=color)
        ax.plot(fpr, tpr, color=color, linewidth=LINE_WIDTH_HERO,
                label=f'{fault_name} (AUC = {auc_val:.3f})')

    # 对角参考线
    ax.plot([0, 1], [0, 1], color=COLORS['dark'], linestyle='--',
            linewidth=LINE_WIDTH_GHOST, alpha=0.5, label='随机基线')

    # 统计框
    auc_values = [d['auc'] for d in roc_data.values()]
    mean_auc = np.mean(auc_values)
    stats_text = f'平均 AUC = {mean_auc:.3f}'
    ax.text(0.60, 0.15, stats_text, transform=ax.transAxes,
            fontsize=text_size, verticalalignment='top',
            bbox={**ACADEMIC_STATS_BOX,
                  'edgecolor': COLORS.get('agent_diag', COLORS['primary'])})

    ax.set_xlabel('假阳性率 (FPR)', fontsize=label_size)
    ax.set_ylabel('真阳性率 (TPR)', fontsize=label_size)
    ax.set_title('多故障类型 ROC 曲线', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 fontweight='bold')
    ax.legend(fontsize=legend_size, loc='lower right')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.grid(**ACADEMIC_GRID)
    set_tick_fontsize(ax, tick_size)

    plt.tight_layout()
    save_figure(fig, 'training', 'diagnostic_roc_curves')
    return fig


def plot_diagnostic_precision_recall(
    pr_data: Dict[str, Dict[str, list]],
    figsize: Tuple[int, int] = (8, 7)
) -> plt.Figure:
    """
    绘制精确率-召回率曲线

    Args:
        pr_data: {故障类型: {'precision': [...], 'recall': [...], 'ap': float}} 字典
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象
    """
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    legend_size = PLOT_CONFIG.FONT_SIZE_LEGEND
    text_size = PLOT_CONFIG.FONT_SIZE_TEXT

    palette = [
        COLORS.get('nature_blue', COLORS['primary']),
        COLORS.get('nature_red', COLORS['danger']),
        COLORS.get('nature_yellow', COLORS['warning']),
        COLORS.get('nature_green', COLORS['success']),
        COLORS.get('nature_purple', COLORS['purple']),
        COLORS.get('nature_cyan', COLORS['orange']),
    ]

    fig, ax = plt.subplots(figsize=figsize)

    for idx, (fault_name, data) in enumerate(pr_data.items()):
        precision = data['precision']
        recall = data['recall']
        ap_val = data['ap']
        color = palette[idx % len(palette)]
        ax.plot(recall, precision, color=color, linewidth=LINE_WIDTH_MAIN,
                label=f'{fault_name} (AP = {ap_val:.3f})')

    # 统计框
    ap_values = [d['ap'] for d in pr_data.values()]
    mean_ap = np.mean(ap_values)
    stats_text = f'平均 AP = {mean_ap:.3f}'
    ax.text(0.05, 0.15, stats_text, transform=ax.transAxes,
            fontsize=text_size, verticalalignment='top',
            bbox={**ACADEMIC_STATS_BOX, 'edgecolor': COLORS['primary']})

    ax.set_xlabel('召回率 (Recall)', fontsize=label_size)
    ax.set_ylabel('精确率 (Precision)', fontsize=label_size)
    ax.set_title('精确率-召回率曲线', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 fontweight='bold')
    ax.legend(fontsize=legend_size, loc='lower left')
    ax.set_xlim([-0.02, 1.05])
    ax.set_ylim([0, 1.08])
    ax.grid(**ACADEMIC_GRID)
    set_tick_fontsize(ax, tick_size)

    plt.tight_layout()
    save_figure(fig, 'training', 'diagnostic_precision_recall')
    return fig


def plot_fault_severity_sensitivity(
    severity_data: Dict[str, Dict[str, list]],
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    绘制故障严重程度敏感性分析

    Args:
        severity_data: {故障类型: {'severity_levels': [...], 'accuracy': [...],
                        'f1_score': [...], 'accuracy_std': [...], 'f1_std': [...]}}
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象
    """
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    legend_size = PLOT_CONFIG.FONT_SIZE_LEGEND
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE

    palette = [
        COLORS.get('nature_blue', COLORS['primary']),
        COLORS.get('nature_red', COLORS['danger']),
        COLORS.get('nature_yellow', COLORS['warning']),
        COLORS.get('nature_green', COLORS['success']),
        COLORS.get('nature_purple', COLORS['purple']),
    ]
    markers = ['o', 's', '^', 'D', 'v']

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for idx, (fault_name, data) in enumerate(severity_data.items()):
        levels = data['severity_levels']
        color = palette[idx % len(palette)]
        marker = markers[idx % len(markers)]

        # (a) 准确率
        acc = data['accuracy']
        acc_std = data.get('accuracy_std', [0] * len(acc))
        axes[0].plot(levels, acc, color=color, linewidth=LINE_WIDTH_MAIN,
                     marker=marker, markersize=MARKER_SIZE_DEFAULT,
                     label=fault_name)
        axes[0].fill_between(levels,
                             np.array(acc) - np.array(acc_std),
                             np.array(acc) + np.array(acc_std),
                             alpha=0.12, color=color)

        # (b) F1-score
        f1 = data['f1_score']
        f1_std = data.get('f1_std', [0] * len(f1))
        axes[1].plot(levels, f1, color=color, linewidth=LINE_WIDTH_MAIN,
                     marker=marker, markersize=MARKER_SIZE_DEFAULT,
                     label=fault_name)
        axes[1].fill_between(levels,
                             np.array(f1) - np.array(f1_std),
                             np.array(f1) + np.array(f1_std),
                             alpha=0.12, color=color)

    for ax, ylabel, title in zip(
        axes,
        ['准确率', 'F1-Score'],
        ['(a) 诊断准确率 vs 故障强度', '(b) F1 分数 vs 故障强度']
    ):
        ax.set_xlabel('故障注入强度', fontsize=label_size)
        ax.set_ylabel(ylabel, fontsize=label_size)
        ax.set_title(title, fontsize=title_size)
        ax.legend(fontsize=legend_size)
        ax.grid(**ACADEMIC_GRID)
        ax.set_ylim([0, 1.05])
        set_tick_fontsize(ax, tick_size)

    fig.suptitle('故障严重程度敏感性分析', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'training', 'fault_severity_sensitivity')
    return fig


def plot_diagnostic_embedding_tsne(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 7)
) -> plt.Figure:
    """
    绘制诊断特征空间 t-SNE 可视化

    Args:
        embeddings_2d: (N, 2) 数组，t-SNE 降维后的 2D 坐标
        labels: (N,) 数组，类别索引
        class_names: 类别名称列表
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象
    """
    if class_names is None:
        class_names = ['健康', '正时故障', '泄漏故障', '燃油故障']

    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    legend_size = PLOT_CONFIG.FONT_SIZE_LEGEND

    palette = [COLORS['success'], COLORS['danger'], COLORS['warning'],
               COLORS['primary'], COLORS['purple'], COLORS['orange']]

    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)
    for idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = class_names[int(lbl)] if int(lbl) < len(class_names) else f'类别 {lbl}'
        color = palette[idx % len(palette)]
        ax.scatter(
            embeddings_2d[mask, 0], embeddings_2d[mask, 1],
            c=color, label=name, s=40, alpha=0.7,
            edgecolors='white', linewidths=0.3, zorder=3
        )

    ax.set_xlabel('t-SNE 维度 1', fontsize=label_size)
    ax.set_ylabel('t-SNE 维度 2', fontsize=label_size)
    ax.set_title('诊断特征空间 t-SNE 可视化', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 fontweight='bold')
    ax.legend(fontsize=legend_size, loc='best', framealpha=0.9)
    ax.grid(**ACADEMIC_GRID)
    set_tick_fontsize(ax, tick_size)

    plt.tight_layout()
    save_figure(fig, 'training', 'diagnostic_embedding_tsne')
    return fig


def plot_diagnostic_online_accuracy(
    online_acc_data: Dict[str, Dict[str, list]],
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    绘制在线诊断准确率随训练步数演化

    Args:
        online_acc_data: {'episodes': [...],
                          'overall': {'raw': [...], 'smooth': [...]},
                          '正时故障': {'raw': [...], 'smooth': [...]}, ...}
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象
    """
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    legend_size = PLOT_CONFIG.FONT_SIZE_LEGEND

    palette = {
        'overall': COLORS['dark'],
        '正时故障': COLORS['danger'],
        '泄漏故障': COLORS['warning'],
        '燃油故障': COLORS['primary'],
        '复合故障': COLORS['purple'],
    }

    episodes = online_acc_data.get('episodes', [])

    fig, ax = plt.subplots(figsize=figsize)

    for key, data in online_acc_data.items():
        if key == 'episodes':
            continue
        raw = data.get('raw', [])
        smooth = data.get('smooth', raw)
        color = palette.get(key, COLORS['info'])
        lw = LINE_WIDTH_MAIN if key == 'overall' else LINE_WIDTH_SECONDARY

        if raw:
            ax.plot(episodes[:len(raw)], raw, color=color, alpha=0.15,
                    linewidth=0.8)
        if smooth:
            ax.plot(episodes[:len(smooth)], smooth, color=color,
                    linewidth=lw, label=key)

    ax.axhline(y=0.9, color=COLORS['dark'], linestyle='--',
               linewidth=LINE_WIDTH_SECONDARY, alpha=0.5, label='目标 90%')
    ax.set_xlabel('训练轮次', fontsize=label_size)
    ax.set_ylabel('诊断准确率', fontsize=label_size)
    ax.set_title('在线诊断准确率演化', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=legend_size, loc='lower right')
    ax.grid(**ACADEMIC_GRID)
    set_tick_fontsize(ax, tick_size)

    plt.tight_layout()
    save_figure(fig, 'training', 'diagnostic_online_accuracy')
    return fig


# ============================================================================
# 控制智能体评价 (5 张)
# ============================================================================

def plot_control_multi_setpoint_tracking(
    tracking_data: Dict[str, Dict[str, list]],
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    绘制多工况设定值跟踪

    Args:
        tracking_data: {'time': [...],
                        'speed': {'actual': [...], 'target': [...]},
                        'power': {'actual': [...], 'target': [...]},
                        'p_scav': {'actual': [...], 'target': [...]},
                        'T_exhaust': {'actual': [...], 'target': [...]}}
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象
    """
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    legend_size = PLOT_CONFIG.FONT_SIZE_LEGEND

    time = tracking_data.get('time', [])

    specs = [
        ('speed', '转速 [rpm]', '(a) 转速跟踪', COLORS['primary']),
        ('power', '功率 [kW]', '(b) 功率跟踪', COLORS['danger']),
        ('p_scav', '扫气压力 [bar]', '(c) 扫气压力跟踪', COLORS['success']),
        ('T_exhaust', '排气温度 [K]', '(d) 排气温度跟踪', COLORS['orange']),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes = axes.flatten()

    for ax, (key, ylabel, title, color) in zip(axes, specs):
        data = tracking_data.get(key, {})
        actual = data.get('actual', [])
        target = data.get('target', [])
        if actual:
            ax.plot(time[:len(actual)], actual, color=color,
                    linewidth=LINE_WIDTH_MAIN, label='实际值')
        if target:
            ax.step(time[:len(target)], target, color=COLORS['dark'],
                    linewidth=LINE_WIDTH_SECONDARY, linestyle='--',
                    label='目标值', where='post')
        ax.set_ylabel(ylabel, fontsize=label_size)
        ax.set_title(title, fontsize=title_size)
        ax.legend(fontsize=legend_size, loc='best')
        ax.grid(**ACADEMIC_GRID)
        set_tick_fontsize(ax, tick_size)

    axes[2].set_xlabel('时间步', fontsize=label_size)
    axes[3].set_xlabel('时间步', fontsize=label_size)

    fig.suptitle('多工况设定值跟踪', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'training', 'control_multi_setpoint_tracking')
    return fig


def plot_control_action_distribution(
    action_data: Dict[str, list],
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    绘制控制动作分布直方图/核密度图

    Args:
        action_data: {动作名称: [值列表]} 字典
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象
    """
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE

    n_actions = len(action_data)
    palette = [COLORS['primary'], COLORS['danger'], COLORS['success'],
               COLORS['purple'], COLORS['orange']]

    fig, axes = plt.subplots(1, n_actions, figsize=figsize)
    if n_actions == 1:
        axes = [axes]

    for idx, (ax, (name, values)) in enumerate(zip(axes, action_data.items())):
        color = palette[idx % len(palette)]
        arr = np.array(values)

        # 直方图
        ax.hist(arr, bins=40, density=True, color=color, alpha=0.55,
                edgecolor=COLORS['dark'], linewidth=0.3)

        # KDE 近似（简单高斯核）
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(arr)
            x_range = np.linspace(arr.min() - 0.1 * (arr.max() - arr.min()),
                                  arr.max() + 0.1 * (arr.max() - arr.min()), 200)
            ax.plot(x_range, kde(x_range), color=color,
                    linewidth=LINE_WIDTH_MAIN, label='KDE')
        except Exception:
            pass

        # 均值和标准差
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        ax.axvline(mean_val, color=COLORS['dark'], linestyle='--',
                   linewidth=LINE_WIDTH_SECONDARY,
                   label=f'$\\mu$={mean_val:.3f}\n$\\sigma$={std_val:.3f}')

        ax.set_xlabel(name, fontsize=label_size)
        ax.set_ylabel('概率密度', fontsize=label_size)
        ax.set_title(name, fontsize=title_size)
        ax.legend(fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND - 1, loc='upper right')
        ax.grid(axis='y', **ACADEMIC_GRID)
        set_tick_fontsize(ax, tick_size)

    fig.suptitle('控制动作分布分析', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'training', 'control_action_distribution')
    return fig


def plot_control_robustness_envelope(
    robustness_data: Dict[str, Dict[str, list]],
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    绘制控制鲁棒性包络（小提琴图）

    Args:
        robustness_data: {'severity_labels': ['0.0', '0.2', ...],
                          'Pmax偏差': [[...], [...], ...],
                          'Texh偏差': [[...], [...], ...]}
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象
    """
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE

    labels = robustness_data.get('severity_labels', [])
    metrics = {k: v for k, v in robustness_data.items() if k != 'severity_labels'}
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    palette = [COLORS['primary'], COLORS['danger'], COLORS['warning']]
    subplot_labels = ['(a)', '(b)', '(c)']

    for idx, (ax, (metric_name, data_lists)) in enumerate(zip(axes, metrics.items())):
        positions = list(range(len(data_lists)))
        vp = ax.violinplot(data_lists, positions=positions, showmeans=True,
                           showmedians=True, showextrema=True)

        color = palette[idx % len(palette)]
        for body in vp['bodies']:
            body.set_facecolor(color)
            body.set_alpha(0.6)
        for key_part in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
            if key_part in vp:
                vp[key_part].set_color(COLORS['dark'])
                vp[key_part].set_linewidth(1.2)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=tick_size)
        ax.set_xlabel('故障注入强度', fontsize=label_size)
        ax.set_ylabel(metric_name, fontsize=label_size)
        lbl = subplot_labels[idx] if idx < len(subplot_labels) else ''
        ax.set_title(f'{lbl} {metric_name}', fontsize=title_size)
        ax.axhline(y=0, color=COLORS['dark'], linestyle='-', alpha=0.3, linewidth=1)
        ax.grid(axis='y', **ACADEMIC_GRID)
        set_tick_fontsize(ax, tick_size)

    fig.suptitle('控制鲁棒性包络分析', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'training', 'control_robustness_envelope')
    return fig


def plot_control_constraint_satisfaction(
    constraint_data: Dict[str, float],
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    绘制约束满足率雷达图

    Args:
        constraint_data: {约束名称: 满足率百分比 [0-100]} 字典
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象
    """
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    text_size = PLOT_CONFIG.FONT_SIZE_TEXT

    categories = list(constraint_data.keys())
    values = list(constraint_data.values())
    n = len(categories)

    # 闭合
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    ax.plot(angles_closed, values_closed, color=COLORS['primary'],
            linewidth=LINE_WIDTH_MAIN, marker='o',
            markersize=MARKER_SIZE_DEFAULT)
    ax.fill(angles_closed, values_closed, color=COLORS['primary'], alpha=0.2)

    # 安全阈值线（90%）
    threshold = [90] * (n + 1)
    ax.plot(angles_closed, threshold, color=COLORS['danger'],
            linewidth=LINE_WIDTH_SECONDARY, linestyle='--', alpha=0.7,
            label='安全阈值 90%')

    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=tick_size)
    ax.set_ylim([0, 105])
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=tick_size - 2)
    ax.set_title('多约束满足率分析', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 pad=20, fontweight='bold')
    ax.legend(fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND, loc='upper right',
              bbox_to_anchor=(1.25, 1.1))

    # 数值标注
    for angle, val in zip(angles, values):
        ax.text(angle, val + 3, f'{val:.1f}%', ha='center', va='bottom',
                fontsize=text_size - 1, color=COLORS['primary'], fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'training', 'control_constraint_satisfaction')
    return fig


def plot_control_energy_efficiency(
    efficiency_data: Dict[str, Dict[str, list]],
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    绘制控制效率分析（SFOC vs 负荷）

    Args:
        efficiency_data: {'loads': [...],
                          'RL 控制': {'sfoc': [...], 'sfoc_std': [...]},
                          'PID 控制': {'sfoc': [...], 'sfoc_std': [...]},
                          '无故障基准': {'sfoc': [...], 'sfoc_std': [...]}}
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象
    """
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    legend_size = PLOT_CONFIG.FONT_SIZE_LEGEND
    text_size = PLOT_CONFIG.FONT_SIZE_TEXT

    loads = efficiency_data.get('loads', [])
    palette = {
        'RL 控制': COLORS['primary'],
        'PID 控制': COLORS['danger'],
        '无故障基准': COLORS['success'],
    }
    markers = {'RL 控制': 'o', 'PID 控制': 's', '无故障基准': '^'}

    fig, ax = plt.subplots(figsize=figsize)

    for method_name, data in efficiency_data.items():
        if method_name == 'loads':
            continue
        sfoc = data['sfoc']
        sfoc_std = data.get('sfoc_std', [0] * len(sfoc))
        color = palette.get(method_name, COLORS['info'])
        marker = markers.get(method_name, 'o')

        ax.plot(loads[:len(sfoc)], sfoc, color=color, linewidth=LINE_WIDTH_MAIN,
                marker=marker, markersize=MARKER_SIZE_DEFAULT, label=method_name)
        ax.fill_between(loads[:len(sfoc)],
                        np.array(sfoc) - np.array(sfoc_std),
                        np.array(sfoc) + np.array(sfoc_std),
                        alpha=0.12, color=color)

    ax.set_xlabel('负荷百分比 (%)', fontsize=label_size)
    ax.set_ylabel('SFOC [g/kWh]', fontsize=label_size)
    ax.set_title('控制效率分析（比油耗对比）', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 fontweight='bold')
    ax.legend(fontsize=legend_size, loc='upper right')
    ax.grid(**ACADEMIC_GRID)
    set_tick_fontsize(ax, tick_size)

    plt.tight_layout()
    save_figure(fig, 'training', 'control_energy_efficiency')
    return fig


# ============================================================================
# 控诊协同评价 (6 张)
# ============================================================================

def plot_collaborative_timeline(
    timeline_data: Dict[str, list],
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    绘制控诊协同时序图（核心图）

    上层: 故障注入时间线；中层: 诊断结果；下层: 控制动作及参数响应
    带垂直虚线标注三个关键时刻

    Args:
        timeline_data: {'time': [...],
                        'fault_injected': [...],          # 故障注入强度
                        'fault_type_true': [...],         # 真实故障类型 (int)
                        'diag_confidence': [...],         # 诊断置信度
                        'diag_predicted': [...],          # 预测故障类型 (int)
                        'ctrl_timing': [...],             # 正时补偿
                        'ctrl_fuel': [...],               # 燃油调整
                        'pmax': [...],                    # Pmax 响应
                        'pmax_target': float,
                        'fault_onset': int,               # 故障注入时刻
                        'diag_detect': int,               # 诊断检出时刻
                        'ctrl_respond': int}              # 控制响应时刻
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象
    """
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    text_size = PLOT_CONFIG.FONT_SIZE_TEXT
    legend_size = PLOT_CONFIG.FONT_SIZE_LEGEND

    time = timeline_data['time']

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1, 1.2, 1, 1], hspace=0.35)

    # 关键时刻
    t_onset = timeline_data.get('fault_onset', None)
    t_detect = timeline_data.get('diag_detect', None)
    t_respond = timeline_data.get('ctrl_respond', None)

    def _add_event_lines(ax):
        """在各子图上添加事件标注线"""
        if t_onset is not None:
            ax.axvline(t_onset, color=COLORS['danger'], linestyle=':',
                       linewidth=1.5, alpha=0.8)
        if t_detect is not None:
            ax.axvline(t_detect, color=COLORS['warning'], linestyle=':',
                       linewidth=1.5, alpha=0.8)
        if t_respond is not None:
            ax.axvline(t_respond, color=COLORS['success'], linestyle=':',
                       linewidth=1.5, alpha=0.8)

    # ── (a) 故障注入 ──────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    fault_inj = timeline_data.get('fault_injected', [])
    ax0.fill_between(time[:len(fault_inj)], 0, fault_inj,
                     alpha=0.5, color=COLORS['danger'], label='故障注入强度')
    ax0.set_ylabel('故障强度', fontsize=label_size)
    ax0.set_title('(a) 故障注入时间线', fontsize=title_size)
    ax0.set_ylim([0, 1.15])
    ax0.legend(fontsize=legend_size, loc='upper right')
    ax0.grid(**ACADEMIC_GRID)
    _add_event_lines(ax0)
    set_tick_fontsize(ax0, tick_size)

    # ── (b) 诊断结果 ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    diag_conf = timeline_data.get('diag_confidence', [])
    diag_pred = timeline_data.get('diag_predicted', [])
    fault_true = timeline_data.get('fault_type_true', [])

    if diag_conf:
        ax1.plot(time[:len(diag_conf)], diag_conf, color=COLORS['primary'],
                 linewidth=LINE_WIDTH_MAIN, label='诊断置信度')
    # 预测正确/错误标注
    if diag_pred and fault_true:
        t_arr = np.array(time[:len(diag_pred)])
        pred_arr = np.array(diag_pred)
        true_arr = np.array(fault_true[:len(pred_arr)])
        correct = pred_arr == true_arr
        ax1.scatter(t_arr[correct], [0.1] * int(correct.sum()),
                    color=COLORS['success'], marker='|', s=15, alpha=0.5,
                    label='诊断正确', zorder=2)
        if (~correct).sum() > 0:
            ax1.scatter(t_arr[~correct], [0.05] * int((~correct).sum()),
                        color=COLORS['danger'], marker='x', s=20, alpha=0.7,
                        label='诊断错误', zorder=2)

    ax1.set_ylabel('置信度 / 诊断', fontsize=label_size)
    ax1.set_title('(b) 诊断智能体输出', fontsize=title_size)
    ax1.set_ylim([-0.05, 1.15])
    ax1.legend(fontsize=legend_size, loc='upper left', ncol=3)
    ax1.grid(**ACADEMIC_GRID)
    _add_event_lines(ax1)
    set_tick_fontsize(ax1, tick_size)

    # ── (c) 控制动作 ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ctrl_timing = timeline_data.get('ctrl_timing', [])
    ctrl_fuel = timeline_data.get('ctrl_fuel', [])

    ax2_twin = ax2.twinx()
    if ctrl_timing:
        ax2.plot(time[:len(ctrl_timing)], ctrl_timing, color=COLORS['purple'],
                 linewidth=LINE_WIDTH_MAIN, label='正时补偿 [deg]')
    if ctrl_fuel:
        ax2_twin.plot(time[:len(ctrl_fuel)], ctrl_fuel, color=COLORS['orange'],
                      linewidth=LINE_WIDTH_MAIN, label='燃油系数')

    ax2.set_ylabel('正时补偿 [deg]', fontsize=label_size, color=COLORS['purple'])
    ax2_twin.set_ylabel('燃油系数', fontsize=label_size, color=COLORS['orange'])
    ax2.set_title('(c) 控制智能体动作', fontsize=title_size)
    # 合并图例
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               fontsize=legend_size, loc='upper left')
    ax2.grid(**ACADEMIC_GRID)
    _add_event_lines(ax2)
    set_tick_fontsize(ax2, tick_size)
    ax2_twin.tick_params(axis='y', labelsize=tick_size)

    # ── (d) 性能响应 ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    pmax = timeline_data.get('pmax', [])
    pmax_target = timeline_data.get('pmax_target', 150.0)

    if pmax:
        ax3.plot(time[:len(pmax)], pmax, color=COLORS['primary'],
                 linewidth=LINE_WIDTH_MAIN, label='实际 $P_{max}$')
    ax3.axhline(pmax_target, color=COLORS['success'], linestyle='--',
                linewidth=LINE_WIDTH_SECONDARY,
                label=f'目标 $P_{{max}}$ = {pmax_target:.0f} bar')
    ax3.fill_between(time, pmax_target * 0.95, pmax_target * 1.05,
                     alpha=0.10, color=COLORS['success'])

    ax3.set_xlabel('时间步', fontsize=label_size)
    ax3.set_ylabel('$P_{max}$ [bar]', fontsize=label_size)
    ax3.set_title('(d) 性能维持效果', fontsize=title_size)
    ax3.legend(fontsize=legend_size, loc='lower right')
    ax3.grid(**ACADEMIC_GRID)
    _add_event_lines(ax3)
    set_tick_fontsize(ax3, tick_size)

    # 顶部添加事件标注
    if t_onset is not None:
        ax0.annotate('故障注入', xy=(t_onset, 1.05), fontsize=text_size - 1,
                     color=COLORS['danger'], ha='center', fontweight='bold')
    if t_detect is not None:
        ax0.annotate('诊断检出', xy=(t_detect, 1.05), fontsize=text_size - 1,
                     color=COLORS['warning'], ha='center', fontweight='bold')
    if t_respond is not None:
        ax0.annotate('控制响应', xy=(t_respond, 1.05), fontsize=text_size - 1,
                     color=COLORS['success'], ha='center', fontweight='bold')

    fig.suptitle('控诊协同时序分析', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 fontweight='bold', y=0.98)
    save_figure(fig, 'training', 'collaborative_timeline')
    return fig


def plot_collaborative_reward_decomposition(
    reward_decomp: Dict[str, list],
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    绘制协同奖励分解（堆叠面积图 + 折线图）

    Args:
        reward_decomp: {'episodes': [...],
                        '诊断奖励': [...],
                        '控制奖励': [...],
                        '协同奖励': [...]}
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象
    """
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE
    legend_size = PLOT_CONFIG.FONT_SIZE_LEGEND

    episodes = reward_decomp.get('episodes', [])
    diag_r = np.array(reward_decomp.get('诊断奖励', []))
    ctrl_r = np.array(reward_decomp.get('控制奖励', []))
    coop_r = np.array(reward_decomp.get('协同奖励', []))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # (a) 堆叠面积图
    ax = axes[0]
    ax.fill_between(episodes, 0, diag_r, alpha=0.5,
                    color=COLORS['primary'], label='诊断奖励')
    ax.fill_between(episodes, diag_r, diag_r + ctrl_r, alpha=0.5,
                    color=COLORS['danger'], label='控制奖励')
    ax.fill_between(episodes, diag_r + ctrl_r, diag_r + ctrl_r + coop_r,
                    alpha=0.5,
                    color=COLORS['success'], label='协同奖励')
    ax.set_xlabel('训练轮次', fontsize=label_size)
    ax.set_ylabel('奖励', fontsize=label_size)
    ax.set_title('(a) 奖励分量堆叠', fontsize=title_size)
    ax.legend(fontsize=legend_size, loc='upper left')
    ax.grid(**ACADEMIC_GRID)
    set_tick_fontsize(ax, tick_size)

    # (b) 各分量占比
    ax2 = axes[1]
    total = np.abs(diag_r) + np.abs(ctrl_r) + np.abs(coop_r) + 1e-8
    # 平滑处理
    window = 20
    if len(diag_r) >= window:
        d_ratio = np.convolve(np.abs(diag_r) / total, np.ones(window) / window, 'valid')
        c_ratio = np.convolve(np.abs(ctrl_r) / total, np.ones(window) / window, 'valid')
        co_ratio = np.convolve(np.abs(coop_r) / total, np.ones(window) / window, 'valid')
        ep = episodes[len(episodes) - len(d_ratio):]
    else:
        d_ratio = np.abs(diag_r) / total
        c_ratio = np.abs(ctrl_r) / total
        co_ratio = np.abs(coop_r) / total
        ep = episodes

    ax2.plot(ep, d_ratio, color=COLORS['primary'], linewidth=LINE_WIDTH_MAIN,
             label='诊断占比')
    ax2.plot(ep, c_ratio, color=COLORS['danger'], linewidth=LINE_WIDTH_MAIN,
             label='控制占比')
    ax2.plot(ep, co_ratio, color=COLORS['success'], linewidth=LINE_WIDTH_MAIN,
             label='协同占比')
    ax2.set_xlabel('训练轮次', fontsize=label_size)
    ax2.set_ylabel('奖励占比', fontsize=label_size)
    ax2.set_title('(b) 奖励分量占比演化', fontsize=title_size)
    ax2.set_ylim([0, 1.0])
    ax2.legend(fontsize=legend_size, loc='best')
    ax2.grid(**ACADEMIC_GRID)
    set_tick_fontsize(ax2, tick_size)

    fig.suptitle('协同奖励分解分析', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'training', 'collaborative_reward_decomposition')
    return fig


def plot_collaborative_fault_response_matrix(
    response_matrix: np.ndarray,
    fault_names: List[str],
    action_names: List[str],
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    绘制故障-响应矩阵热力图

    Args:
        response_matrix: (n_faults, n_actions) 矩阵
        fault_names: 故障类型名称列表
        action_names: 控制动作名称列表
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象
    """
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    text_size = PLOT_CONFIG.FONT_SIZE_TEXT

    fig, ax = plt.subplots(figsize=figsize)

    from matplotlib.colors import LinearSegmentedColormap
    cmap_fr = LinearSegmentedColormap.from_list(
        'academic_fr', ['#EBF8FF', '#90CDF4', '#4299E1', '#C53030', '#9B2C2C'])
    im = ax.imshow(response_matrix, cmap=cmap_fr, aspect='auto')
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('平均调整幅度', fontsize=label_size)
    cbar.ax.tick_params(labelsize=tick_size)

    n_faults = len(fault_names)
    n_actions = len(action_names)

    ax.set_xticks(np.arange(n_actions))
    ax.set_yticks(np.arange(n_faults))
    ax.set_xticklabels(action_names, fontsize=tick_size, rotation=30, ha='right')
    ax.set_yticklabels(fault_names, fontsize=tick_size)

    # 数值标注
    for i in range(n_faults):
        for j in range(n_actions):
            val = response_matrix[i, j]
            threshold = (response_matrix.max() + response_matrix.min()) / 2
            color = 'white' if val > threshold else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=text_size, color=color)

    ax.set_xlabel('控制动作', fontsize=label_size)
    ax.set_ylabel('故障类型', fontsize=label_size)
    ax.set_title('故障-响应矩阵', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'training', 'collaborative_fault_response_matrix')
    return fig


def plot_collaborative_ablation_study(
    ablation_data: Dict[str, Dict[str, float]],
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    绘制协同消融实验对比（分组柱状图）

    Args:
        ablation_data: {方法名: {指标名: 值}} 字典
            方法如 '完整协同', '仅诊断', '仅控制', '独立训练'
            指标如 '诊断准确率', '控制偏差', '综合奖励', '约束违反率'
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象
    """
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE
    text_size = PLOT_CONFIG.FONT_SIZE_TEXT

    methods = list(ablation_data.keys())
    metrics = list(next(iter(ablation_data.values())).keys())
    n_methods = len(methods)
    n_metrics = len(metrics)

    palette = [
        COLORS.get('nature_blue', COLORS['primary']),
        COLORS.get('nature_red', COLORS['danger']),
        COLORS.get('nature_yellow', COLORS['warning']),
        COLORS.get('nature_green', COLORS['success']),
        COLORS.get('nature_purple', COLORS['purple']),
    ]

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    for ax_idx, (ax, metric_name) in enumerate(zip(axes, metrics)):
        values = [ablation_data[m][metric_name] for m in methods]
        y_pos = np.arange(n_methods)
        val_max = max(abs(v) for v in values) if values else 1.0

        # 找最优方法（最大值）
        best_idx = int(np.argmax(values))

        # 绘制水平连接线
        for i, val in enumerate(zip(y_pos, values)):
            y, v = val
            lw = 8 if i == best_idx else 6
            ax.plot([0, v], [y, y], color='#E2E8F0',
                    linewidth=lw, solid_capstyle='round', zorder=1)

        # 绘制数值点
        for i, (method, val) in enumerate(zip(methods, values)):
            color = palette[i % len(palette)]
            size = 250 if i == best_idx else 180
            edge_w = 3.0 if i == best_idx else 2.0
            ax.scatter(val, i, s=size, color=color, edgecolors='white',
                       linewidths=edge_w, zorder=5)
            label_text = f'{val:.3f}'
            if i == best_idx:
                label_text += ' *'
            ax.text(val + val_max * 0.04, i, label_text,
                    va='center', fontsize=text_size - 1, fontweight='bold',
                    color=color)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods, fontsize=tick_size - 1)
        ax.set_xlabel(metric_name, fontsize=label_size)
        ax.set_title(f'({chr(97 + ax_idx)}) {metric_name}',
                     fontsize=title_size, fontweight='bold')
        ax.grid(axis='x', **ACADEMIC_GRID)
        ax.set_axisbelow(True)
        set_tick_fontsize(ax, tick_size)

    fig.suptitle('协同消融实验对比', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'training', 'collaborative_ablation_study')
    return fig


def plot_collaborative_information_flow(
    figsize: Tuple[int, int] = (16, 9)
) -> plt.Figure:
    """
    绘制双智能体信息流示意图

    展示诊断输出 → 控制输入、SharedCritic 的信息耦合机制

    Args:
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象
    """
    text_size = PLOT_CONFIG.FONT_SIZE_TEXT
    title_size = PLOT_CONFIG.FONT_SIZE_SUPTITLE

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def _box(x, y, w, h, color, label, fontsize=text_size):
        box = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.2',
            facecolor=color, edgecolor='black', linewidth=1.8, alpha=0.85, zorder=3
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
                fontsize=fontsize, color='white', fontweight='bold', zorder=4)

    # 环境 (顶部)
    _box(5.5, 8.5, 5, 1.2, COLORS['dark'], '发动机环境 (EngineEnv)\n状态观测: $[P_{max}, T_{exh}, n, p_{scav}, ...]$',
         fontsize=text_size - 1)

    # 诊断智能体 (左)
    _box(0.5, 4.5, 4.5, 2.8, COLORS['primary'],
         '诊断智能体\n(PINN-KAN)\n\n输出: 故障类型概率\n        严重程度\n        置信度',
         fontsize=text_size - 1)

    # 控制智能体 (右)
    _box(11, 4.5, 4.5, 2.8, COLORS['danger'],
         '控制智能体\n(TD-MPC2)\n\n输出: 正时补偿\n        燃油调整\n        保护级别',
         fontsize=text_size - 1)

    # SharedCritic (中)
    _box(5.5, 1.0, 5, 1.8, COLORS['purple'],
         'SharedCritic\n$V(s_{diag}, s_{ctrl})$',
         fontsize=text_size)

    # 信息流箭头
    # 环境 → 诊断
    ax.annotate('', xy=(2.75, 7.3), xytext=(6.5, 8.4),
                arrowprops=dict(arrowstyle='->', lw=2.0, color=COLORS['primary'],
                                connectionstyle='arc3,rad=0.2'))
    ax.text(3.0, 8.2, '观测', fontsize=text_size - 1, color=COLORS['primary'],
            fontweight='bold')

    # 环境 → 控制
    ax.annotate('', xy=(13.25, 7.3), xytext=(9.5, 8.4),
                arrowprops=dict(arrowstyle='->', lw=2.0, color=COLORS['danger'],
                                connectionstyle='arc3,rad=-0.2'))
    ax.text(12.0, 8.2, '观测', fontsize=text_size - 1, color=COLORS['danger'],
            fontweight='bold')

    # 诊断 → 控制（核心协同信息流）
    ax.annotate('', xy=(10.9, 5.9), xytext=(5.1, 5.9),
                arrowprops=dict(arrowstyle='->', lw=3.0, color=COLORS['success'],
                                connectionstyle='arc3,rad=-0.05'))
    ax.text(8.0, 6.6, '诊断结果\n[故障概率, 严重程度, 置信度]',
            fontsize=text_size, color=COLORS['success'],
            fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['success'], alpha=0.9))

    # 控制 → 环境
    ax.annotate('', xy=(9.8, 9.1), xytext=(13.25, 7.4),
                arrowprops=dict(arrowstyle='->', lw=2.0, color=COLORS['orange'],
                                connectionstyle='arc3,rad=-0.3'))
    ax.text(12.5, 7.8, '控制\n动作', fontsize=text_size - 1, color=COLORS['orange'],
            fontweight='bold')

    # 诊断 → SharedCritic
    ax.annotate('', xy=(6.5, 2.9), xytext=(2.75, 4.4),
                arrowprops=dict(arrowstyle='->', lw=1.8, color=COLORS['purple'],
                                connectionstyle='arc3,rad=0.2', linestyle='dashed'))

    # 控制 → SharedCritic
    ax.annotate('', xy=(9.5, 2.9), xytext=(13.25, 4.4),
                arrowprops=dict(arrowstyle='->', lw=1.8, color=COLORS['purple'],
                                connectionstyle='arc3,rad=-0.2', linestyle='dashed'))

    # SharedCritic → 诊断 (价值反馈)
    ax.annotate('', xy=(2.75, 4.4), xytext=(5.5, 2.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['info'],
                                connectionstyle='arc3,rad=0.2', linestyle='dotted'))
    ax.text(2.8, 3.5, '$V_{diag}$', fontsize=text_size, color=COLORS['info'])

    # SharedCritic → 控制 (价值反馈)
    ax.annotate('', xy=(13.25, 4.4), xytext=(10.5, 2.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['info'],
                                connectionstyle='arc3,rad=-0.2', linestyle='dotted'))
    ax.text(12.3, 3.5, '$V_{ctrl}$', fontsize=text_size, color=COLORS['info'])

    ax.set_title('双智能体信息流与协同机制',
                 fontsize=title_size, fontweight='bold', pad=15)

    plt.tight_layout()
    save_figure(fig, 'training', 'collaborative_information_flow')
    return fig


def plot_collaborative_pareto_front(
    pareto_data: Dict[str, list],
    figsize: Tuple[int, int] = (9, 7)
) -> plt.Figure:
    """
    绘制诊断-控制 Pareto 前沿

    Args:
        pareto_data: {'diag_f1': [...],
                      'ctrl_rmse': [...],
                      'total_reward': [...],
                      'is_pareto': [...],       # bool 列表
                      'labels': [...]}           # 超参数标签
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象
    """
    tick_size = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    text_size = PLOT_CONFIG.FONT_SIZE_TEXT
    legend_size = PLOT_CONFIG.FONT_SIZE_LEGEND

    f1 = np.array(pareto_data['diag_f1'])
    rmse = np.array(pareto_data['ctrl_rmse'])
    reward = np.array(pareto_data['total_reward'])
    is_pareto = np.array(pareto_data.get('is_pareto', [False] * len(f1)))
    labels = pareto_data.get('labels', [])

    fig, ax = plt.subplots(figsize=figsize)

    # 非 Pareto 点
    sc = ax.scatter(f1[~is_pareto], rmse[~is_pareto],
                    c=reward[~is_pareto], cmap='viridis',
                    s=80, alpha=0.6, edgecolors='gray', linewidths=0.5,
                    zorder=3, label='非最优解')

    # Pareto 点
    if is_pareto.any():
        ax.scatter(f1[is_pareto], rmse[is_pareto],
                   c=reward[is_pareto], cmap='viridis',
                   s=180, alpha=0.95, edgecolors=COLORS['danger'],
                   linewidths=2.5, marker='*', zorder=5, label='Pareto 最优')

        # 画 Pareto 前沿连线
        pareto_idx = np.where(is_pareto)[0]
        pareto_f1 = f1[pareto_idx]
        pareto_rmse = rmse[pareto_idx]
        sort_idx = np.argsort(pareto_f1)
        ax.plot(pareto_f1[sort_idx], pareto_rmse[sort_idx],
                color=COLORS['danger'], linewidth=LINE_WIDTH_SECONDARY,
                linestyle='--', alpha=0.7, zorder=4)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('综合奖励', fontsize=label_size)
    cbar.ax.tick_params(labelsize=tick_size)

    # 最优区域标注
    ax.annotate('理想区域\n(高F1, 低RMSE)',
                xy=(f1.max(), rmse.min()),
                xytext=(f1.max() - 0.08, rmse.min() + (rmse.max() - rmse.min()) * 0.2),
                fontsize=text_size, color=COLORS['success'],
                fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color=COLORS['success'],
                                lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=COLORS['success'], alpha=0.9))

    ax.set_xlabel('诊断 F1-Score', fontsize=label_size)
    ax.set_ylabel('控制 RMSE [bar]', fontsize=label_size)
    ax.set_title('诊断-控制 Pareto 前沿', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 fontweight='bold')
    ax.legend(fontsize=legend_size, loc='upper right')
    ax.grid(**ACADEMIC_GRID)
    set_tick_fontsize(ax, tick_size)

    plt.tight_layout()
    save_figure(fig, 'training', 'collaborative_pareto_front')
    return fig


# ============================================================================
# 训练过程动态相图（新增高冲击力图表）
# ============================================================================

def plot_training_phase_diagram(
    history: Dict[str, List],
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    绘制训练过程动态相图（诊断准确率 vs 控制性能 2D 轨迹）

    用颜色渐变表示训练进度（episode），揭示两个智能体的协同演化规律。
    这在 MARL 论文中具有很高辨识度。

    Args:
        history: 训练历史字典，须包含 'diag_accuracy' 和 'ctrl_performance'
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象（已保存至 visualization_output/training/）
    """
    tick_size  = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    text_size  = PLOT_CONFIG.FONT_SIZE_TEXT

    diag_acc = history.get('diag_accuracy', [])
    ctrl_perf = history.get('ctrl_performance', [])

    if not diag_acc or not ctrl_perf:
        # 无数据时返回空图
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, '数据不足', ha='center', va='center',
                fontsize=label_size, transform=ax.transAxes)
        save_figure(fig, 'training', 'training_phase_diagram')
        return fig

    n = min(len(diag_acc), len(ctrl_perf))
    x = np.array(diag_acc[:n])
    y = np.array(ctrl_perf[:n])
    episodes = np.arange(n)

    # 平滑处理
    window = min(20, max(3, n // 20))
    if n > window:
        x_sm = np.convolve(x, np.ones(window) / window, mode='valid')
        y_sm = np.convolve(y, np.ones(window) / window, mode='valid')
        ep_sm = episodes[:len(x_sm)]
    else:
        x_sm, y_sm, ep_sm = x, y, episodes

    fig, ax = plt.subplots(figsize=figsize)

    # 用颜色渐变绘制轨迹
    from matplotlib.collections import LineCollection

    points = np.array([x_sm, y_sm]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(ep_sm.min(), ep_sm.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm, linewidths=2.5,
                        alpha=0.8, zorder=3)
    lc.set_array(ep_sm[:-1])
    ax.add_collection(lc)

    # 起点和终点标记
    ax.scatter(x_sm[0], y_sm[0], s=200, color='#E53E3E', edgecolors='white',
               linewidths=2.5, marker='o', zorder=10, label='训练起点')
    ax.scatter(x_sm[-1], y_sm[-1], s=250, color='#38A169', edgecolors='white',
               linewidths=2.5, marker='*', zorder=10, label='训练终点')

    # 理想区域标注
    ax.axvline(x=0.9, color=COLORS['dark'], linestyle=':', alpha=0.3,
               linewidth=LINE_WIDTH_GHOST)
    ax.axhline(y=0.9, color=COLORS['dark'], linestyle=':', alpha=0.3,
               linewidth=LINE_WIDTH_GHOST)
    ax.fill_between([0.9, 1.05], 0.9, 1.05, alpha=0.08,
                    color=COLORS.get('nature_green', COLORS['success']),
                    zorder=1)
    ax.text(0.97, 0.97, '理想区域', ha='center', va='center',
            fontsize=text_size, color=COLORS.get('nature_green', COLORS['success']),
            fontweight='bold', alpha=0.7)

    # Colorbar
    cbar = fig.colorbar(lc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('训练轮次', fontsize=label_size)
    cbar.ax.tick_params(labelsize=tick_size - 2)

    ax.set_xlabel('诊断准确率', fontsize=label_size)
    ax.set_ylabel('控制性能（$P_{max}$ 维持率）', fontsize=label_size)
    ax.set_title('双智能体协同演化轨迹', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE,
                 fontweight='bold')
    ax.legend(fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND, loc='lower left',
              framealpha=0.9, edgecolor='#CCCCCC')
    ax.set_xlim([max(0, x_sm.min() - 0.05), min(1.05, x_sm.max() + 0.05)])
    ax.set_ylim([max(0, y_sm.min() - 0.05), min(1.05, y_sm.max() + 0.05)])
    ax.grid(**ACADEMIC_GRID)
    set_tick_fontsize(ax, tick_size)

    plt.tight_layout()
    save_figure(fig, 'training', 'training_phase_diagram')
    return fig
