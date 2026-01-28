#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
校准过程可视化主脚本
====================
生成模型校准过程与结果的可视化图表。

输出图表（基础）:
1. calibration_convergence_comparison.svg - 收敛曲线与实验-仿真对比 (2×2)
2. calibration_error_parameters.svg - 误差分布与参数汇总 (1×2)

输出图表（学术风格 IEEE/Elsevier）:
3. exp_sim_comparison_lines.svg - 实验-仿真点线对比
4. convergence_log_scale.svg - 对数坐标收敛曲线
5. residual_analysis.svg - 残差分析（QQ图+直方图）
6. bland_altman.svg - Bland-Altman一致性分析
7. scatter_45degree.svg - 45度线散点图
8. operating_point_coverage.svg - 工况点覆盖分布
9. error_bar_comparison.svg - 误差条对比图
10. parameter_evolution.svg - 参数演化轨迹

配置选项:
- USE_MOCK_DATA: 是否使用模拟数据 (True/False)

使用方法:
    python visualize_calibration.py           # 使用实际数据
    python visualize_calibration.py --mock    # 使用模拟数据

Author: CDC Project
Date: 2026-01-28
"""

import os
import sys
import argparse
import warnings
import logging

# 在导入matplotlib相关模块之前抑制所有字体警告
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib.backends').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*Glyph.*')
warnings.filterwarnings('ignore', message='.*font.*')
warnings.filterwarnings('ignore', message='.*Font.*')

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PATH_CONFIG, setup_matplotlib_style

# 导入核心可视化函数
from visualization.calibration_plots import (
    # 数据加载函数
    load_convergence_data,
    load_validation_data,
    load_calibrated_params,
    # 核心学术风格绘图函数
    plot_45degree_scatter,
    plot_bland_altman,
    plot_exp_sim_comparison_lines,
    generate_all_academic_plots
)

# ============================================================================
# 配置开关
# ============================================================================
USE_MOCK_DATA = False  # 设置为True使用模拟数据，False使用实际校准数据


def main(use_mock: bool = None):
    """
    主函数
    
    Args:
        use_mock: 是否使用模拟数据，默认使用全局配置USE_MOCK_DATA
    """
    if use_mock is None:
        use_mock = USE_MOCK_DATA
    
    print("=" * 60)
    print("模型校准过程与结果可视化")
    print("=" * 60)
    
    # 根据开关选择数据目录
    if use_mock:
        print("\n[模式] 使用模拟数据")
        data_dir = PATH_CONFIG.DATA_SIMULATION_DIR
        file_prefix = 'mock_'
    else:
        print("\n[模式] 使用实际校准数据")
        data_dir = PATH_CONFIG.DATA_CALIBRATION_DIR
        file_prefix = ''
    
    # 使用全局配置的输出目录
    output_dir = PATH_CONFIG.VIS_CALIBRATION_DIR
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 构建数据文件路径
    convergence_file = os.path.join(data_dir, f'{file_prefix}calibration_convergence.csv')
    validation_file = os.path.join(data_dir, f'{file_prefix}calibration_validation.csv')
    params_file = os.path.join(data_dir, f'{file_prefix}calibrated_params.json')
    
    data_available = {
        'convergence': os.path.exists(convergence_file),
        'validation': os.path.exists(validation_file),
        'params': os.path.exists(params_file)
    }
    
    print("\n数据文件检查:")
    print(f"  - 收敛历史: {'✓' if data_available['convergence'] else '✗'} {convergence_file}")
    print(f"  - 验证结果: {'✓' if data_available['validation'] else '✗'} {validation_file}")
    print(f"  - 校准参数: {'✓' if data_available['params'] else '✗'} {params_file}")
    
    # 加载数据
    convergence_df = None
    validation_df = None
    params = None
    
    if data_available['convergence']:
        try:
            convergence_df = load_convergence_data(convergence_file)
            print(f"\n收敛历史数据: {len(convergence_df)} 条记录")
        except Exception as e:
            print(f"\n警告: 加载收敛历史失败: {e}")
    
    if data_available['validation']:
        try:
            validation_df = load_validation_data(validation_file)
            print(f"验证结果数据: {len(validation_df)} 个工况点")
        except Exception as e:
            print(f"\n警告: 加载验证结果失败: {e}")
    
    if data_available['params']:
        try:
            params = load_calibrated_params(params_file)
            print(f"校准参数: {len(params)} 个参数")
        except Exception as e:
            print(f"\n警告: 加载校准参数失败: {e}")
    
    # 生成可视化
    generated_files = []
    
    print("\n" + "-" * 60)
    print("生成学术风格校准图表 (IEEE/Elsevier)")
    print("-" * 60)
    
    # 学术风格校准图表 (统一风格)
    academic_files = []
    if convergence_df is not None and validation_df is not None:
        try:
            academic_files = generate_all_academic_plots(
                convergence_df=convergence_df,
                validation_df=validation_df,
                params=params,
                output_dir=output_dir
            )
            generated_files.extend(academic_files)
        except Exception as e:
            print(f"  [Error] 生成学术风格图表失败: {e}")
    
    # 发动机模型可视化图表已移除
    engine_model_files = []
    
    # 汇总
    print("\n" + "=" * 60)
    print("可视化完成汇总")
    print("=" * 60)
    
    # 校准相关图表
    print(f"\n[校准可视化] 输出目录: {output_dir}")
    print(f"  生成文件数: {len(generated_files)}")
    
    if generated_files:
        for f in generated_files:
            if f and os.path.exists(f):
                size_kb = os.path.getsize(f) / 1024
                print(f"  OK {os.path.basename(f)} ({size_kb:.1f} KB)")
            else:
                print(f"  FAIL {f}")
    else:
        print("  Warning: No calibration plots generated.")
        if use_mock:
            print("  Hint: Run mock data generator first:")
            print("    python scripts/generate_mock_calibration_data.py")
        else:
            print("  Hint: Run calibration first:")
            print("    python run_calibration.py")
    
    total_files = len(generated_files)
    print(f"\nTotal: {total_files} calibration plots generated.")
    
    return generated_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibration Visualization')
    parser.add_argument('--mock', action='store_true', 
                        help='使用模拟数据进行可视化')
    args = parser.parse_args()
    main(use_mock=args.mock)
