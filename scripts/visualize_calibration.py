#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
校准过程可视化主脚本
====================
生成模型校准过程与结果的可视化图表。

输出图表（学术风格 IEEE/Elsevier）:
1. exp_sim_comparison_lines.svg - 实验-仿真点线对比
2. bland_altman.svg - Bland-Altman一致性分析
3. scatter_45degree.svg - 45度线散点图

配置选项:
- RUNTIME_CONFIG.USE_MOCK_DATA: 是否使用模拟数据

使用方法:
    python scripts/visualize_calibration.py           # 使用全局配置决定数据源
    python scripts/visualize_calibration.py --mock    # 强制使用模拟数据
    python scripts/visualize_calibration.py --no-mock # 强制使用校准结果数据

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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PATH_CONFIG, RUNTIME_CONFIG, setup_matplotlib_style

# 导入可视化函数
from visualization.calibration_data_io import (
    load_convergence_data,
    load_validation_data,
    load_calibrated_params,
)
from visualization.calibration_plots import (
    plot_45degree_scatter,
    plot_bland_altman,
    plot_exp_sim_comparison_lines,
    generate_all_academic_plots,
)


def main(use_mock: bool = None):
    """
    主函数
    
    Args:
        use_mock: 是否使用模拟数据，默认使用 RUNTIME_CONFIG.USE_MOCK_DATA
    """
    if use_mock is None:
        use_mock = RUNTIME_CONFIG.USE_MOCK_DATA
    
    print("=" * 60)
    print("模型校准过程与结果可视化")
    print("=" * 60)
    
    # 根据开关选择数据目录
    if use_mock:
        print("\n[模式] 使用模拟数据")
        data_dir = PATH_CONFIG.DATA_SIMULATION_DIR
        file_prefix = 'mock_'
    else:
        print("\n[模式] 使用校准结果数据")
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
    
    # 汇总
    print("\n" + "=" * 60)
    print("可视化完成汇总")
    print("=" * 60)
    
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
            print("    python scripts/run_calibration.py")
    
    total_files = len(generated_files)
    print(f"\nTotal: {total_files} calibration plots generated.")
    
    return generated_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibration Visualization')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--mock', action='store_true', default=False,
                       help='强制使用模拟数据')
    group.add_argument('--no-mock', action='store_true', default=False,
                       help='强制使用校准结果数据')
    args = parser.parse_args()
    
    if args.mock:
        use_mock = True
    elif args.no_mock:
        use_mock = False
    else:
        use_mock = None
    
    main(use_mock=use_mock)
