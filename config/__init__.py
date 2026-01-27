#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置模块
========
提供全局配置的便捷导入

用法示例:
    from config import PLOT_CONFIG, COLORS, setup_matplotlib_style, save_figure
    
    # 设置matplotlib样式
    setup_matplotlib_style()
    
    # 使用配色
    plt.plot(x, y, color=COLORS['primary'])
    
    # 保存图形
    save_figure(fig, 'preprocessing', 'my_plot.svg')
"""

from config.global_config import (
    # 配置类
    PathConfig,
    PlotConfig,
    DataConfig,
    EngineConfig,
    TrainingConfig,
    
    # 全局配置实例
    PATH_CONFIG,
    PLOT_CONFIG,
    DATA_CONFIG,
    ENGINE_CONFIG,
    TRAINING_CONFIG,
    
    # 配色方案
    COLORS,
    
    # 函数
    setup_matplotlib_style,
    get_output_path,
    save_figure,
)

__all__ = [
    # 配置类
    'PathConfig',
    'PlotConfig',
    'DataConfig',
    'EngineConfig',
    'TrainingConfig',
    
    # 全局配置实例
    'PATH_CONFIG',
    'PLOT_CONFIG',
    'DATA_CONFIG',
    'ENGINE_CONFIG',
    'TRAINING_CONFIG',
    
    # 配色方案
    'COLORS',
    
    # 函数
    'setup_matplotlib_style',
    'get_output_path',
    'save_figure',
]
