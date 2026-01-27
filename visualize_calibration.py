#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
校准过程可视化主脚本
====================
生成模型校准过程与结果的可视化图表。

输出图表:
1. calibration_convergence_comparison.svg - 收敛曲线与实验-仿真对比 (2×2)
2. calibration_error_parameters.svg - 误差分布与参数汇总 (1×2)

独立图表:
3. calibration_convergence.svg - 收敛曲线 (单独)
4. calibration_comparison.svg - 实验-仿真对比 (单独)
5. error_distribution.svg - 误差分布 (单独)
6. calibrated_parameters.svg - 参数汇总 (单独)

使用方法:
    python visualize_calibration.py

Author: CDC Project
Date: 2026-01-28
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PATH_CONFIG, setup_matplotlib_style

# 导入可视化绑定函数
from visualization.calibration_plots import (
    plot_calibration_convergence_and_comparison,
    plot_error_distribution_and_parameters,
    plot_calibration_convergence,
    plot_calibration_comparison,
    plot_error_distribution,
    plot_calibrated_parameters,
    load_convergence_data,
    load_validation_data,
    load_calibrated_params
)


def main():
    """主函数"""
    print("=" * 60)
    print("模型校准过程与结果可视化")
    print("=" * 60)
    
    # 使用全局配置的输出目录
    output_dir = PATH_CONFIG.VIS_CALIBRATION_DIR
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查数据文件是否存在
    convergence_file = os.path.join(PATH_CONFIG.DATA_DIR, 'calibration_convergence.csv')
    validation_file = os.path.join(PATH_CONFIG.DATA_DIR, 'calibration_validation.csv')
    params_file = os.path.join(PATH_CONFIG.DATA_DIR, 'calibrated_params.json')
    
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
            convergence_df = load_convergence_data()
            print(f"\n收敛历史数据: {len(convergence_df)} 条记录")
        except Exception as e:
            print(f"\n警告: 加载收敛历史失败: {e}")
    
    if data_available['validation']:
        try:
            validation_df = load_validation_data()
            print(f"验证结果数据: {len(validation_df)} 个工况点")
        except Exception as e:
            print(f"\n警告: 加载验证结果失败: {e}")
    
    if data_available['params']:
        try:
            params = load_calibrated_params()
            print(f"校准参数: {len(params)} 个参数")
        except Exception as e:
            print(f"\n警告: 加载校准参数失败: {e}")
    
    # 生成可视化
    generated_files = []
    
    print("\n" + "-" * 60)
    print("生成可视化图表")
    print("-" * 60)
    
    # 综合图1: 收敛曲线与实验-仿真对比
    if convergence_df is not None and validation_df is not None:
        try:
            plot_calibration_convergence_and_comparison(
                convergence_df=convergence_df,
                validation_df=validation_df
            )
            generated_files.append('calibration_convergence_comparison.svg')
        except Exception as e:
            print(f"  [Error] 生成收敛与对比图失败: {e}")
    
    # 综合图2: 误差分布与参数汇总
    if validation_df is not None and params is not None:
        try:
            plot_error_distribution_and_parameters(
                validation_df=validation_df,
                params=params
            )
            generated_files.append('calibration_error_parameters.svg')
        except Exception as e:
            print(f"  [Error] 生成误差与参数图失败: {e}")
    
    # 单独图表 (可选)
    generate_individual = True  # 设置为False可跳过单独图表
    
    if generate_individual:
        print("\n生成单独图表...")
        
        if convergence_df is not None:
            try:
                plot_calibration_convergence(convergence_df=convergence_df)
                generated_files.append('calibration_convergence.svg')
            except Exception as e:
                print(f"  [Error] 生成收敛曲线图失败: {e}")
        
        if validation_df is not None:
            try:
                plot_calibration_comparison(validation_df=validation_df)
                generated_files.append('calibration_comparison.svg')
            except Exception as e:
                print(f"  [Error] 生成对比图失败: {e}")
            
            try:
                plot_error_distribution(validation_df=validation_df)
                generated_files.append('error_distribution.svg')
            except Exception as e:
                print(f"  [Error] 生成误差分布图失败: {e}")
        
        if params is not None:
            try:
                plot_calibrated_parameters(params=params)
                generated_files.append('calibrated_parameters.svg')
            except Exception as e:
                print(f"  [Error] 生成参数汇总图失败: {e}")
    
    # 汇总
    print("\n" + "=" * 60)
    print("可视化完成!")
    print("=" * 60)
    print(f"\n输出目录: {output_dir}")
    print(f"生成文件数: {len(generated_files)}")
    
    if generated_files:
        print("\n生成的文件:")
        for f in generated_files:
            filepath = os.path.join(output_dir, f)
            if os.path.exists(filepath):
                size_kb = os.path.getsize(filepath) / 1024
                print(f"  ✓ {f} ({size_kb:.1f} KB)")
            else:
                print(f"  ✗ {f} (文件不存在)")
    else:
        print("\n警告: 未生成任何文件，请检查数据是否存在。")
        print("提示: 请先运行校准流程生成数据文件:")
        print("  from calibration.calibrator import EngineCalibrator")
        print("  calibrator.run_full_calibration(n_points=5, export_results=True)")
    
    return generated_files


if __name__ == '__main__':
    main()
