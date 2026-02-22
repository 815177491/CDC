#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MARL 训练可视化模块
====================
双智能体强化学习训练过程与结果的学术风格可视化。

函数列表:
- plot_training_curves(history)       - 训练曲线（奖励/损失/熵/准确率）
- plot_reward_distribution(history)   - 奖励分布直方图
- plot_confusion_matrix(cm, ...)      - 故障诊断混淆矩阵
- plot_detection_delay(delays)        - 故障检测延迟箱型图
- plot_control_response(...)          - 容错控制响应分析
- plot_method_comparison(methods, ...) - 方法性能对比柱状图
- plot_dual_agent_architecture()      - 双智能体网络架构图

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
    LINE_WIDTH_MAIN,
    LINE_WIDTH_SECONDARY,
    MARKER_SIZE_DEFAULT,
    ACADEMIC_STATS_BOX,
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
    绘制双智能体训练曲线（学术综合视图）

    包含子图：(a) 奖励曲线、(b) 策略损失、(c) 价值损失、
             (d) 策略熵、(e) 诊断准确率、(f) 控制性能

    Args:
        history: 训练历史字典，键包括 'episodes', 'reward_diag', 'reward_ctrl',
                 'loss_diag_policy', 'loss_ctrl_policy', 'loss_diag_value',
                 'loss_ctrl_value', 'entropy_diag', 'entropy_ctrl',
                 'diag_accuracy', 'ctrl_performance'
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象（已保存至 visualization_output/training/）
    """
    tick_size  = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE
    sup_size   = PLOT_CONFIG.FONT_SIZE_SUPTITLE

    color_diag = COLORS['primary']    # 蓝色 → 诊断
    color_ctrl = COLORS['danger']     # 红色 → 控制
    color_acc  = COLORS['success']    # 绿色 → 准确率
    color_perf = COLORS['secondary']  # 紫红色 → 性能

    episodes = (history.get('episodes')
                or list(range(len(history.get('reward_diag', [])))))

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.28)

    # ── (a) 奖励曲线 ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if history.get('reward_diag'):
        ax1.plot(episodes, history['reward_diag'],
                 color=color_diag, alpha=0.25, linewidth=LINE_WIDTH_SECONDARY)
        sm = _smooth(history['reward_diag'])
        ax1.plot(episodes[len(episodes) - len(sm):], sm,
                 color=color_diag, linewidth=LINE_WIDTH_MAIN, label='诊断奖励')
    if history.get('reward_ctrl'):
        ax1.plot(episodes, history['reward_ctrl'],
                 color=color_ctrl, alpha=0.25, linewidth=LINE_WIDTH_SECONDARY)
        sm = _smooth(history['reward_ctrl'])
        ax1.plot(episodes[len(episodes) - len(sm):], sm,
                 color=color_ctrl, linewidth=LINE_WIDTH_MAIN, label='控制奖励')
    ax1.set_xlabel('训练轮次', fontsize=label_size)
    ax1.set_ylabel('奖励', fontsize=label_size)
    ax1.set_title('(a) 双智能体奖励曲线', fontsize=title_size)
    ax1.legend(fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND, loc='lower right')
    ax1.grid(True, alpha=0.3)
    set_tick_fontsize(ax1, tick_size)

    # ── (b) 策略损失 ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if history.get('loss_diag_policy'):
        ax2.plot(episodes, history['loss_diag_policy'],
                 color=color_diag, linewidth=LINE_WIDTH_MAIN, label='诊断策略损失')
    if history.get('loss_ctrl_policy'):
        ax2.plot(episodes, history['loss_ctrl_policy'],
                 color=color_ctrl, linewidth=LINE_WIDTH_MAIN, label='控制策略损失')
    ax2.set_xlabel('训练轮次', fontsize=label_size)
    ax2.set_ylabel('策略损失', fontsize=label_size)
    ax2.set_title('(b) 策略损失曲线', fontsize=title_size)
    ax2.legend(fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND)
    ax2.grid(True, alpha=0.3)
    set_tick_fontsize(ax2, tick_size)

    # ── (c) 价值损失 ──────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if history.get('loss_diag_value'):
        ax3.plot(episodes, history['loss_diag_value'],
                 color=color_diag, linewidth=LINE_WIDTH_MAIN, label='诊断价值损失')
    if history.get('loss_ctrl_value'):
        ax3.plot(episodes, history['loss_ctrl_value'],
                 color=color_ctrl, linewidth=LINE_WIDTH_MAIN, label='控制价值损失')
    ax3.set_xlabel('训练轮次', fontsize=label_size)
    ax3.set_ylabel('价值损失', fontsize=label_size)
    ax3.set_title('(c) 价值损失曲线', fontsize=title_size)
    ax3.legend(fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND)
    ax3.grid(True, alpha=0.3)
    set_tick_fontsize(ax3, tick_size)

    # ── (d) 策略熵 ────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if history.get('entropy_diag'):
        ax4.plot(episodes, history['entropy_diag'],
                 color=color_diag, linewidth=LINE_WIDTH_MAIN, label='诊断熵')
    if history.get('entropy_ctrl'):
        ax4.plot(episodes, history['entropy_ctrl'],
                 color=color_ctrl, linewidth=LINE_WIDTH_MAIN, label='控制熵')
    ax4.set_xlabel('训练轮次', fontsize=label_size)
    ax4.set_ylabel('策略熵', fontsize=label_size)
    ax4.set_title('(d) 策略熵曲线（探索程度）', fontsize=title_size)
    ax4.legend(fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND)
    ax4.grid(True, alpha=0.3)
    set_tick_fontsize(ax4, tick_size)

    # ── (e) 诊断准确率 ────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    if history.get('diag_accuracy'):
        ax5.plot(episodes, history['diag_accuracy'],
                 color=color_acc, alpha=0.3, linewidth=LINE_WIDTH_SECONDARY)
        sm = _smooth(history['diag_accuracy'])
        ax5.plot(episodes[len(episodes) - len(sm):], sm,
                 color=color_acc, linewidth=LINE_WIDTH_MAIN)
        ax5.axhline(y=0.9, color=COLORS['dark'], linestyle='--',
                    linewidth=LINE_WIDTH_SECONDARY, alpha=0.7, label='目标准确率 90%')
    ax5.set_xlabel('训练轮次', fontsize=label_size)
    ax5.set_ylabel('准确率', fontsize=label_size)
    ax5.set_title('(e) 诊断准确率', fontsize=title_size)
    ax5.set_ylim([0, 1.05])
    ax5.legend(fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND)
    ax5.grid(True, alpha=0.3)
    set_tick_fontsize(ax5, tick_size)

    # ── (f) 控制性能 ──────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    if history.get('ctrl_performance'):
        ax6.plot(episodes, history['ctrl_performance'],
                 color=color_perf, alpha=0.3, linewidth=LINE_WIDTH_SECONDARY)
        sm = _smooth(history['ctrl_performance'])
        ax6.plot(episodes[len(episodes) - len(sm):], sm,
                 color=color_perf, linewidth=LINE_WIDTH_MAIN)
    ax6.set_xlabel('训练轮次', fontsize=label_size)
    ax6.set_ylabel('性能维持率', fontsize=label_size)
    ax6.set_title('(f) 控制性能（$P_{max}$ 维持率）', fontsize=title_size)
    ax6.set_ylim([0, 1.1])
    ax6.grid(True, alpha=0.3)
    set_tick_fontsize(ax6, tick_size)

    fig.suptitle('双智能体强化学习训练过程', fontsize=sup_size, fontweight='bold')
    save_figure(fig, 'training', 'training_curves')
    return fig


def plot_reward_distribution(
    history: Dict[str, List],
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    绘制奖励分布直方图

    Args:
        history: 训练历史字典（包含 reward_diag / reward_ctrl / reward_total）
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象（已保存至 visualization_output/training/）
    """
    tick_size  = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    specs = [
        ('reward_diag',  COLORS['primary'],   COLORS['danger'],    '诊断智能体奖励分布'),
        ('reward_ctrl',  COLORS['danger'],     COLORS['primary'],   '控制智能体奖励分布'),
        ('reward_total', COLORS['success'],    COLORS['secondary'], '总奖励分布'),
    ]

    for ax, (key, bar_color, mean_color, title) in zip(axes, specs):
        data = history.get(key, [])
        if data:
            ax.hist(data, bins=30,
                    color=bar_color, alpha=0.75, edgecolor=COLORS['dark'], linewidth=0.5)
            mean_val = float(np.mean(data))
            ax.axvline(mean_val, color=mean_color, linestyle='--',
                       linewidth=LINE_WIDTH_MAIN,
                       label=f'均值: {mean_val:.2f}')
            ax.legend(fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND)
        ax.set_xlabel('奖励', fontsize=label_size)
        ax.set_ylabel('频次', fontsize=label_size)
        ax.set_title(title, fontsize=title_size)
        ax.grid(True, alpha=0.3, axis='y')
        set_tick_fontsize(ax, tick_size)

    fig.suptitle('双智能体奖励分布分析', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE)
    plt.tight_layout()
    save_figure(fig, 'training', 'reward_distribution')
    return fig


# ============================================================================
# 评估结果可视化
# ============================================================================

def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    绘制故障诊断混淆矩阵

    Args:
        confusion_matrix: n×n 混淆矩阵（行=真实，列=预测）
        class_names: 类别标签列表，默认 ['健康', '正时故障', '泄漏故障', '燃油故障']
        figsize: 图形尺寸

    Returns:
        fig: matplotlib Figure 对象（已保存至 visualization_output/training/）
    """
    if class_names is None:
        class_names = ['健康', '正时故障', '泄漏故障', '燃油故障']

    tick_size  = PLOT_CONFIG.FONT_SIZE_TICK
    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    text_size  = PLOT_CONFIG.FONT_SIZE_TEXT

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(confusion_matrix, cmap='Blues')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('样本数', rotation=-90, va='bottom', fontsize=label_size)
    cbar.ax.tick_params(labelsize=tick_size)

    n = len(class_names)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, fontsize=tick_size)
    ax.set_yticklabels(class_names, fontsize=tick_size)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    threshold = confusion_matrix.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = confusion_matrix[i, j]
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=text_size,
                    color='white' if val > threshold else 'black')

    ax.set_xlabel('预测类别', fontsize=label_size)
    ax.set_ylabel('真实类别', fontsize=label_size)
    ax.set_title('故障诊断混淆矩阵', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE)

    plt.tight_layout()
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

    palette = [
        COLORS['danger'], COLORS['info'], COLORS['warning'],
        COLORS['success'], COLORS['purple'],
    ]

    fig, ax = plt.subplots(figsize=figsize)

    labels = list(delays.keys())
    data   = [delays[k] for k in labels]
    pos    = list(range(len(labels)))

    bp = ax.boxplot(data, positions=pos, patch_artist=True, widths=0.5)

    for patch, color in zip(bp['boxes'], palette[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for element in ['whiskers', 'caps', 'fliers', 'medians']:
        plt.setp(bp[element], color=COLORS['dark'], linewidth=LINE_WIDTH_SECONDARY)

    ax.axhline(y=5, color=COLORS['danger'], linestyle='--',
               linewidth=LINE_WIDTH_MAIN, alpha=0.7, label='目标: < 5 cycles')

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=tick_size)
    ax.set_ylabel('检测延迟（循环数）', fontsize=label_size)
    ax.set_title('不同故障类型的检测延迟分布', fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE)
    ax.legend(fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND)
    ax.grid(True, alpha=0.3, axis='y')
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

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.35)

    # (a) 故障严重程度
    axes[0].fill_between(time_steps, 0, fault_severity,
                         alpha=0.5, color=COLORS['danger'], label='故障严重程度')
    axes[0].set_ylabel('故障强度', fontsize=label_size)
    axes[0].set_title('(a) 故障注入', fontsize=title_size)
    axes[0].legend(fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.1])
    set_tick_fontsize(axes[0], tick_size)

    # (b) Pmax 响应
    axes[1].plot(time_steps, pmax_actual,
                 color=COLORS['primary'], linewidth=LINE_WIDTH_MAIN, label='实际 $P_{max}$')
    axes[1].axhline(y=pmax_target, color=COLORS['success'], linestyle='--',
                    linewidth=LINE_WIDTH_SECONDARY,
                    label=f'目标 $P_{{max}}$ = {pmax_target:.0f} bar')
    axes[1].fill_between(time_steps, pmax_target * 0.95, pmax_target * 1.05,
                         alpha=0.15, color=COLORS['success'], label='±5% 容差带')
    axes[1].set_ylabel('$P_{max}$ [bar]', fontsize=label_size)
    axes[1].set_title('(b) 性能维持效果', fontsize=title_size)
    axes[1].legend(fontsize=PLOT_CONFIG.FONT_SIZE_LEGEND, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    set_tick_fontsize(axes[1], tick_size)

    # (c) 正时补偿
    axes[2].plot(time_steps, timing_offset,
                 color=COLORS['purple'], linewidth=LINE_WIDTH_MAIN)
    axes[2].axhline(y=0, color=COLORS['dark'], linestyle='-', alpha=0.4, linewidth=1)
    axes[2].set_ylabel('正时补偿 [deg]', fontsize=label_size)
    axes[2].set_title('(c) 控制智能体 — 正时调整', fontsize=title_size)
    axes[2].set_ylim([-6.5, 6.5])
    axes[2].grid(True, alpha=0.3)
    set_tick_fontsize(axes[2], tick_size)

    # (d) 燃油调整
    axes[3].plot(time_steps, fuel_adj,
                 color=COLORS['orange'], linewidth=LINE_WIDTH_MAIN)
    axes[3].axhline(y=1.0, color=COLORS['dark'], linestyle='-', alpha=0.4, linewidth=1)
    axes[3].set_ylabel('燃油系数', fontsize=label_size)
    axes[3].set_xlabel('时间步', fontsize=label_size)
    axes[3].set_title('(d) 控制智能体 — 燃油调整', fontsize=title_size)
    axes[3].set_ylim([0.78, 1.22])
    axes[3].grid(True, alpha=0.3)
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
    绘制方法性能对比柱状图（学术风格）

    Args:
        methods: 方法名称列表（对应柱宽）
        metrics: {指标名称: [各方法数值]} 字典
        figsize: 图形尺寸，默认按指标数量自动确定

    Returns:
        fig: matplotlib Figure 对象（已保存至 visualization_output/training/）
    """
    n_metrics = len(metrics)
    if figsize is None:
        figsize = (4 * n_metrics + 2, 6)

    label_size = PLOT_CONFIG.FONT_SIZE_LABEL
    title_size = PLOT_CONFIG.FONT_SIZE_TITLE
    tick_size  = PLOT_CONFIG.FONT_SIZE_TICK
    text_size  = PLOT_CONFIG.FONT_SIZE_TEXT

    palette = [
        COLORS['primary'], COLORS['danger'], COLORS['success'],
        COLORS['secondary'], COLORS['warning'], COLORS['info'],
    ]

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(len(methods))
    width = 0.6

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        bars = ax.bar(x, values, width,
                      color=palette[:len(methods)],
                      edgecolor=COLORS['dark'], linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=tick_size)
        ax.set_ylabel(metric_name, fontsize=label_size)
        ax.set_title(metric_name, fontsize=title_size)
        ax.grid(True, alpha=0.3, axis='y')
        set_tick_fontsize(ax, tick_size)

        # 数值标注
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.015,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=text_size
            )

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
