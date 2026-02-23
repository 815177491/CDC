#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization Module
====================
数据预处理、建模与校准可视化绑定模块

采用惰性导入，避免 import visualization 时触发 matplotlib 初始化。
具体子模块在首次访问时才加载。

Author: CDC Project
Date: 2026-02-22
"""

# ---------------------------------------------------------------------------
# __all__ 声明（仅用于 IDE 提示与 from visualization import * ）
# ---------------------------------------------------------------------------
__all__ = [
    # 公共样式
    'set_tick_fontsize',
    'LINE_WIDTH_MAIN',
    'LINE_WIDTH_SECONDARY',
    'MARKER_SIZE_LARGE',
    'MARKER_SIZE_DEFAULT',
    'ACADEMIC_SCATTER_PARAMS',
    'ACADEMIC_REFERENCE_LINE',
    'ACADEMIC_ERROR_BAND',
    'ACADEMIC_STATS_BOX',
    # 预处理可视化
    'plot_steady_state_selection',
    'plot_representative_points',
    'plot_data_cleaning',
    'plot_normalization_correlation',
    # 建模可视化
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
    # 校准可视化
    'plot_45degree_scatter',
    'plot_bland_altman',
    'plot_exp_sim_comparison_lines',
    'generate_all_academic_plots',
    # 校准数据IO
    'load_convergence_data',
    'load_validation_data',
    'load_calibrated_params',
    # MARL 双智能体可视化
    'plot_training_curves',
    'plot_reward_distribution',
    'plot_confusion_matrix',
    'plot_detection_delay',
    'plot_control_response',
    'plot_method_comparison',
    'plot_dual_agent_architecture',
    # Experiments (bar charts)
    'plot_overall_metrics_bars',
    'plot_per_fault_f1_bars',
    'plot_ablation_delta_bars',
    'plot_severity_robustness_bars',
    'plot_multi_run_statistics',
    'build_avg_severity_accuracy',
    # 向后兼容
    'COLORS',
]


# ---------------------------------------------------------------------------
# 惰性导入映射（子模块 → 导出名称列表）
# ---------------------------------------------------------------------------
_LAZY_IMPORTS = {
    '.style': [
        'set_tick_fontsize',
        'LINE_WIDTH_MAIN',
        'LINE_WIDTH_SECONDARY',
        'MARKER_SIZE_LARGE',
        'MARKER_SIZE_DEFAULT',
        'ACADEMIC_SCATTER_PARAMS',
        'ACADEMIC_REFERENCE_LINE',
        'ACADEMIC_ERROR_BAND',
        'ACADEMIC_STATS_BOX',
    ],
    '.preprocessing_plots': [
        'plot_steady_state_selection',
        'plot_representative_points',
        'plot_data_cleaning',
        'plot_normalization_correlation',
    ],
    '.modeling': [
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
    ],
    '.calibration_plots': [
        'plot_45degree_scatter',
        'plot_bland_altman',
        'plot_exp_sim_comparison_lines',
        'generate_all_academic_plots',
    ],
    '.calibration_data_io': [
        'load_convergence_data',
        'load_validation_data',
        'load_calibrated_params',
    ],
    '.marl_plots': [
        'plot_training_curves',
        'plot_reward_distribution',
        'plot_confusion_matrix',
        'plot_detection_delay',
        'plot_control_response',
        'plot_method_comparison',
        'plot_dual_agent_architecture',
    ],
    '.experiment_plots': [
        'plot_overall_metrics_bars',
        'plot_per_fault_f1_bars',
        'plot_ablation_delta_bars',
        'plot_severity_robustness_bars',
        'plot_multi_run_statistics',
        'build_avg_severity_accuracy',
    ],
}

# 反向映射：属性名 → 子模块
_ATTR_TO_MODULE = {}
for _mod, _names in _LAZY_IMPORTS.items():
    for _name in _names:
        _ATTR_TO_MODULE[_name] = _mod


def __getattr__(name: str):
    """惰性导入：首次访问属性时才加载对应子模块"""
    # 向后兼容：COLORS
    if name == 'COLORS':
        from config import COLORS
        return COLORS

    module_path = _ATTR_TO_MODULE.get(name)
    if module_path is not None:
        import importlib
        mod = importlib.import_module(module_path, __name__)
        attr = getattr(mod, name)
        # 缓存到模块命名空间，后续访问不再触发 __getattr__
        globals()[name] = attr
        return attr

    raise AttributeError(f"module 'visualization' has no attribute {name!r}")
