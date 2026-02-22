#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
建模可视化子包
==============
将原 modeling_plots.py 按主题拆分为独立模块，
每个模块负责一类建模可视化图表。

子模块:
- framework_plots: 模型框架流程图
- geometry_plots: 曲柄连杆几何示意图与运动学特性
- timing_plots: 二冲程配气正时圆形图
- combustion_plots: 双Wiebe燃烧模型特性
- heat_transfer_plots: Woschni传热模型特性
- thermodynamic_plots: 热力学循环特性
- calibration_flow_plots: 三阶段校准流程图
- sensitivity_plots: 参数敏感度雷达图
- energy_plots: 能量平衡桑基图

Author: CDC Project
Date: 2026-02-20
"""

from .framework_plots import plot_model_framework
from .geometry_plots import plot_crank_geometry_schematic, plot_crank_kinematics
from .timing_plots import plot_valve_timing_diagram
from .combustion_plots import plot_dual_wiebe_combustion
from .heat_transfer_plots import plot_woschni_heat_transfer
from .thermodynamic_plots import plot_thermodynamic_cycle
from .calibration_flow_plots import plot_calibration_flowchart
from .sensitivity_plots import plot_sensitivity_radar
from .energy_plots import plot_energy_sankey


def generate_all_modeling_figures():
    """
    生成所有建模相关的可视化图片

    Returns:
        dict: 各图片名称到输出路径的映射
    """
    print("=" * 60)
    print("开始生成建模可视化图片...")
    print("=" * 60)

    results = {}

    plot_functions = [
        ('model_framework', plot_model_framework),
        ('crank_geometry', plot_crank_geometry_schematic),
        ('crank_kinematics', plot_crank_kinematics),
        ('valve_timing', plot_valve_timing_diagram),
        ('dual_wiebe', plot_dual_wiebe_combustion),
        ('woschni', plot_woschni_heat_transfer),
        ('thermodynamic_cycle', plot_thermodynamic_cycle),
        ('calibration_flowchart', plot_calibration_flowchart),
        ('sensitivity_radar', plot_sensitivity_radar),
        ('energy_sankey', plot_energy_sankey),
    ]

    for i, (name, func) in enumerate(plot_functions, 1):
        print(f"\n[{i}/{len(plot_functions)}] 生成 {name}...")
        try:
            results[name] = func()
        except Exception as e:
            print(f"  [Error] {name}: {e}")

    print("\n" + "=" * 60)
    print("所有建模可视化图片生成完成！")
    print("=" * 60)

    return results


__all__ = [
    'plot_model_framework',
    'plot_crank_geometry_schematic',
    'plot_crank_kinematics',
    'plot_valve_timing_diagram',
    'plot_dual_wiebe_combustion',
    'plot_woschni_heat_transfer',
    'plot_thermodynamic_cycle',
    'plot_calibration_flowchart',
    'plot_sensitivity_radar',
    'plot_energy_sankey',
    'generate_all_modeling_figures',
]
