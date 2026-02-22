#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization Module
====================
数据预处理、建模与校准可视化绑定模块

Author: CDC Project
Date: 2025-01-01
"""

# 公共样式工具
from .style import (
    set_tick_fontsize,
    LINE_WIDTH_MAIN,
    LINE_WIDTH_SECONDARY,
    MARKER_SIZE_LARGE,
    MARKER_SIZE_DEFAULT,
    ACADEMIC_SCATTER_PARAMS,
    ACADEMIC_REFERENCE_LINE,
    ACADEMIC_ERROR_BAND,
    ACADEMIC_STATS_BOX,
)

# 预处理可视化
from .preprocessing_plots import (
    plot_steady_state_selection,
    plot_representative_points,
    plot_data_cleaning,
    plot_normalization_correlation,
)

# 建模可视化（子包）
from .modeling import (
    plot_model_framework,
    plot_crank_geometry_schematic,
    plot_crank_kinematics,
    plot_valve_timing_diagram,
    plot_dual_wiebe_combustion,
    plot_woschni_heat_transfer,
    plot_thermodynamic_cycle,
    plot_calibration_flowchart,
    plot_sensitivity_radar,
    plot_energy_sankey,
    generate_all_modeling_figures,
)

# 校准可视化
from .calibration_plots import (
    plot_45degree_scatter,
    plot_bland_altman,
    plot_exp_sim_comparison_lines,
    generate_all_academic_plots,
)

# 校准数据IO
from .calibration_data_io import (
    load_convergence_data,
    load_validation_data,
    load_calibrated_params,
)

# 向后兼容：从 config 重新导出 COLORS
from config import COLORS

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
    # 向后兼容
    'COLORS',
]
