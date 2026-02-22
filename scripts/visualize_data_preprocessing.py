#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据清洗与特征提取可视化
========================
生成数据预处理流程的可视化图表

包括:
1. 稳态工况的智能筛选
2. 代表性工况点的提取
3. 数据的清洗与异常值剔除
4. 数据的标准化处理与参数关联

使用方法:
    python scripts/visualize_data_preprocessing.py

Author: CDC Project
Date: 2026-01-24
"""

import os
import sys

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入全局配置
from config import PATH_CONFIG, RUNTIME_CONFIG, setup_matplotlib_style

# 导入数据加载模块
from calibration.data_loader import VisualizationDataLoader

# 导入可视化绑定模块
from visualization.preprocessing_plots import (
    plot_steady_state_selection,
    plot_representative_points,
    plot_data_cleaning,
    plot_normalization_correlation,
)


def main():
    """主函数 - 生成全部数据预处理可视化"""
    print("=" * 70)
    print("数据清洗与特征提取可视化")
    print("=" * 70)
    
    # 使用全局配置的输出目录
    output_dir = PATH_CONFIG.VIS_PREPROCESSING_DIR
    print(f"输出目录: {os.path.abspath(output_dir)}")
    
    # 使用新的数据加载器
    data_path = os.path.join(PATH_CONFIG.DATA_RAW_DIR, RUNTIME_CONFIG.CALIBRATION_DATA_FILE)
    loader = VisualizationDataLoader(data_path)
    
    try:
        # 读取数据段（自动选择第1段）
        df = loader.load_segment(use_full_data=False)
        
        # 按优先级顺序生成所有可视化
        plot_steady_state_selection(df)          # 1. 稳态筛选
        plot_representative_points(df)            # 2. 工况点提取
        plot_data_cleaning(df)                    # 3. 异常值剔除
        
        # 4. 标准化处理（需要清洗后的数据）
        df_clean = loader.apply_outlier_filter(df)
        plot_normalization_correlation(df, df_clean)
        
        print()
        print("=" * 70)
        print("✅ 所有数据预处理可视化已生成!")
        print("=" * 70)
        
        # 显示使用的数据段信息
        if loader.segment_stats is not None:
            print(f"\n使用的数据段统计:")
            print(loader.segment_stats.to_string(index=False))
        
        # 统计生成的文件
        svg_files = ['steady_state_selection.svg', 'representative_points.svg',
                     'data_cleaning.svg', 'normalization_correlation.svg']
        
        print(f"\n生成的SVG矢量图 ({len(svg_files)} 个):")
        for fname in svg_files:
            path = os.path.join(output_dir, fname)
            if os.path.exists(path):
                size = os.path.getsize(path) / 1024
                print(f"  {fname:<40} {size:>8.1f} KB")
        
    except Exception as e:
        print(f"\n[ERROR] 生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
