# 可视化模块
"""
Visualization Module
====================
数据预处理与建模可视化绑定模块
"""

from .preprocessing_plots import (
    COLORS,
    set_tick_fontsize,
    plot_steady_state_selection,
    plot_representative_points,
    plot_data_cleaning,
    plot_normalization_correlation,
)

from .modeling_plots import (
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

__all__ = [
    # 预处理可视化
    'COLORS',
    'set_tick_fontsize',
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
]
