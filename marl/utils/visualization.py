#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MARL 可视化兼容层
=================
此模块保留原有类接口（TrainingVisualizer / EvaluationVisualizer /
NetworkArchitectureVisualizer），内部委托至学术风格纯函数
visualization.marl_plots。

如需新功能，请直接调用 visualization.marl_plots 中的纯函数；
此处保留只是为了不破坏 experiments/ 中的现有导入。

Author: CDC Project
Date: 2026-02-22
"""

# 标准库
import warnings
from typing import Dict, List, Optional, Tuple

# 第三方库
import numpy as np
import matplotlib.pyplot as plt

# 纯函数实现（学术风格）
from visualization.marl_plots import (
    plot_training_curves        as _plot_training_curves,
    plot_reward_distribution    as _plot_reward_distribution,
    plot_confusion_matrix       as _plot_confusion_matrix,
    plot_detection_delay        as _plot_detection_delay,
    plot_control_response       as _plot_control_response,
    plot_method_comparison      as _plot_method_comparison,
    plot_dual_agent_architecture as _plot_dual_agent_architecture,
)

__all__ = [
    "TrainingVisualizer",
    "EvaluationVisualizer",
    "NetworkArchitectureVisualizer",
]

# ============================================================================
# 模块级弃用警告
# ============================================================================
warnings.warn(
    "marl.utils.visualization 已弃用，请直接使用 visualization.marl_plots 中的纯函数。"
    "此兼容层将在未来版本中移除。",
    DeprecationWarning,
    stacklevel=2,
)


# ============================================================================
# TrainingVisualizer — 有状态包装器
# ============================================================================

class TrainingVisualizer:
    """
    训练过程可视化器（兼容层）

    保持原有接口：update(metrics) + plot_training_curves()。
    绘图实现已委托至 visualization.marl_plots.plot_training_curves。

    Examples::

        vis = TrainingVisualizer()
        for episode in range(1000):
            metrics = trainer.step()
            vis.update(metrics)
        vis.plot_training_curves()
    """

    def __init__(self, save_dir: str = "./visualization_output/training"):
        """
        Args:
            save_dir: 已废弃参数（保留以维持 API 兼容），
                      保存路径由 save_figure 统一管理。
        """
        if save_dir != "./visualization_output/training":
            warnings.warn(
                "save_dir 参数已废弃，保存路径由 save_figure() 统一管理。",
                DeprecationWarning,
                stacklevel=2,
            )

        # 训练历史（供 update() 积累数据）
        self.history: Dict[str, List] = {
            "episodes": [],
            "reward_diag": [],
            "reward_ctrl": [],
            "reward_total": [],
            "loss_diag_policy": [],
            "loss_diag_value": [],
            "loss_ctrl_policy": [],
            "loss_ctrl_value": [],
            "entropy_diag": [],
            "entropy_ctrl": [],
            "diag_accuracy": [],
            "ctrl_performance": [],
            "episode_length": [],
        }

    def update(self, metrics: Dict) -> None:
        """
        将本轮训练指标追加到历史记录。

        Args:
            metrics: 字典，键名与 self.history 中的键对应。
        """
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

    def plot_training_curves(self, figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        绘制训练曲线并保存至 visualization_output/training/。

        Args:
            figsize: 图形尺寸

        Returns:
            fig: matplotlib Figure 对象
        """
        return _plot_training_curves(self.history, figsize=figsize)

    def plot_reward_distribution(self, figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """
        绘制奖励分布直方图并保存。

        Args:
            figsize: 图形尺寸

        Returns:
            fig: matplotlib Figure 对象
        """
        return _plot_reward_distribution(self.history, figsize=figsize)

    def save_plot(self, name: str = "") -> None:
        """已废弃：保存由 save_figure() 在绘图函数内自动处理。"""
        warnings.warn(
            "save_plot() 已废弃，图形在调用 plot_*() 时自动保存至 "
            "visualization_output/training/。",
            DeprecationWarning,
            stacklevel=2,
        )


# ============================================================================
# EvaluationVisualizer — 无状态委托器
# ============================================================================

class EvaluationVisualizer:
    """评估结果可视化器（兼容层），各方法无状态，直接委托至纯函数。"""

    def __init__(self, save_dir: str = "./visualization_output/training"):
        if save_dir != "./visualization_output/training":
            warnings.warn(
                "save_dir 参数已废弃，保存路径由 save_figure() 统一管理。",
                DeprecationWarning,
                stacklevel=2,
            )

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """
        绘制故障分类混淆矩阵。

        Args:
            confusion_matrix: n×n 混淆矩阵（行=真实，列=预测）
            class_names: 类别标签；默认 ['健康','正时故障','泄漏故障','燃油故障']
            figsize: 图形尺寸

        Returns:
            fig: matplotlib Figure 对象
        """
        return _plot_confusion_matrix(confusion_matrix, class_names, figsize)

    def plot_detection_delay(
        self,
        delays: Dict[str, List[float]],
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """
        绘制故障检测延迟箱型图。

        Args:
            delays: {故障名称: [延迟(步数), ...]} 字典
            figsize: 图形尺寸

        Returns:
            fig: matplotlib Figure 对象
        """
        return _plot_detection_delay(delays, figsize)

    def plot_control_response(
        self,
        time_steps: np.ndarray,
        fault_severity: np.ndarray,
        pmax_actual: np.ndarray,
        pmax_target: float,
        timing_offset: np.ndarray,
        fuel_adj: np.ndarray,
        figsize: Tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        """
        绘制容错控制响应分析图（4 行子图）。

        Args:
            time_steps: 时间步数组
            fault_severity: 故障严重程度 [0, 1]
            pmax_actual: 实际 Pmax 数组 [bar]
            pmax_target: 目标 Pmax 标量 [bar]
            timing_offset: 正时补偿数组 [deg]
            fuel_adj: 燃油调整系数数组
            figsize: 图形尺寸

        Returns:
            fig: matplotlib Figure 对象
        """
        return _plot_control_response(
            time_steps, fault_severity, pmax_actual,
            pmax_target, timing_offset, fuel_adj, figsize,
        )

    def plot_method_comparison(
        self,
        methods: List[str],
        metrics: Dict[str, List[float]],
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """
        绘制方法性能对比柱状图。

        Args:
            methods: 方法名称列表
            metrics: {指标名称: [各方法数值]} 字典
            figsize: 图形尺寸（默认自动）

        Returns:
            fig: matplotlib Figure 对象
        """
        return _plot_method_comparison(methods, metrics, figsize)


# ============================================================================
# NetworkArchitectureVisualizer — 无状态委托器
# ============================================================================

class NetworkArchitectureVisualizer:
    """
    网络架构可视化器（兼容层）

    委托至 visualization.marl_plots.plot_dual_agent_architecture，
    保存至 visualization_output/modeling/。
    """

    def __init__(self, save_dir: str = "./visualization_output/modeling"):
        if save_dir != "./visualization_output/modeling":
            warnings.warn(
                "save_dir 参数已废弃，保存路径由 save_figure() 统一管理。",
                DeprecationWarning,
                stacklevel=2,
            )

    def plot_architecture(self, figsize: Tuple[int, int] = (16, 8)) -> plt.Figure:
        """
        绘制双智能体网络架构示意图。

        Args:
            figsize: 图形尺寸

        Returns:
            fig: matplotlib Figure 对象（保存至 modeling 类别）
        """
        return _plot_dual_agent_architecture(figsize)
