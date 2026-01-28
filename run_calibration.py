#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型校准主脚本
==============
运行三阶段分步解耦校准流程

包括:
1. 压缩段校准: 确定有效压缩比 (目标: Pcomp误差<2%)
2. 燃烧段校准: 确定Wiebe参数 (目标: Pmax误差最小)
3. 传热段校准: 确定Woschni系数 (目标: 排温匹配)

输出文件:
- data/calibrated_params.json: 校准后的参数
- data/calibration_convergence.csv: 收敛历史记录
- data/calibration_validation.csv: 验证结果

使用方法:
    python run_calibration.py

Author: CDC Project
Date: 2026-01-28
"""

import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入全局配置
from config import PATH_CONFIG, ENGINE_CONFIG

# 导入校准模块
from calibration.calibrator import EngineCalibrator
from calibration.data_loader import CalibrationDataLoader

# 导入发动机模型
from engine import MarineEngine0D


def run_calibration(n_points: int = 10, 
                    data_file: str = None,
                    export_results: bool = True,
                    verbose: bool = True) -> dict:
    """
    运行三阶段校准流程
    
    Args:
        n_points: 使用的校准工况点数量，默认10个
        data_file: 校准数据文件路径，默认为 data/raw/calibration_data.csv
        export_results: 是否导出结果文件，默认True
        verbose: 是否打印详细信息，默认True
        
    Returns:
        results: 校准结果字典，包含各阶段的参数和误差
    """
    # 确定数据文件路径
    if data_file is None:
        data_file = os.path.join(PATH_CONFIG.DATA_RAW_DIR, 'calibration_data.csv')
    
    # 检查数据文件
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"校准数据文件不存在: {data_file}")
    
    if verbose:
        print("=" * 70)
        print("三阶段分步解耦校准")
        print("=" * 70)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"数据文件: {data_file}")
        print(f"校准点数: {n_points}")
        print()
    
    # 初始化数据加载器
    data_loader = CalibrationDataLoader(data_file)
    
    # 初始化发动机模型
    engine = MarineEngine0D(config=ENGINE_CONFIG)
    
    if verbose:
        print("发动机模型参数:")
        print(f"  - 缸径: {ENGINE_CONFIG.bore*1000:.0f} mm")
        print(f"  - 行程: {ENGINE_CONFIG.stroke*1000:.0f} mm")
        print(f"  - 气缸数: {ENGINE_CONFIG.n_cylinders}")
        print(f"  - 初始压缩比: {ENGINE_CONFIG.compression_ratio:.1f}")
        print()
    
    # 创建校准器
    calibrator = EngineCalibrator(engine, data_loader)
    
    # 运行完整校准流程
    results = calibrator.run_full_calibration(
        n_points=n_points,
        export_results=export_results
    )
    
    if verbose:
        print()
        print("=" * 70)
        print(f"校准完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # 打印收敛摘要
        summary = calibrator.get_convergence_summary()
        print("\n收敛摘要:")
        for stage, stats in summary.items():
            print(f"  [{stage}]")
            print(f"    - 迭代次数: {stats['total_iterations']}")
            print(f"    - 初始目标值: {stats['initial_objective']:.6f}")
            print(f"    - 最终目标值: {stats['final_objective']:.6f}")
            print(f"    - 最优目标值: {stats['best_objective']:.6f}")
        
        # 打印输出文件
        if export_results:
            print("\n输出文件:")
            output_files = [
                ('校准参数', 'calibrated_params.json'),
                ('收敛历史', 'calibration_convergence.csv'),
                ('验证结果', 'calibration_validation.csv')
            ]
            for desc, fname in output_files:
                # 校准器导出到 data/calibration/ 目录
                fpath = os.path.join(PATH_CONFIG.DATA_CALIBRATION_DIR, fname)
                if os.path.exists(fpath):
                    size_kb = os.path.getsize(fpath) / 1024
                    print(f"  ✓ {desc}: {fpath} ({size_kb:.1f} KB)")
                else:
                    print(f"  ✗ {desc}: {fpath} (未生成)")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='运行三阶段发动机参数校准',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_calibration.py                    # 使用默认参数运行
  python run_calibration.py -n 10              # 使用10个校准点
  python run_calibration.py --data custom.csv  # 使用自定义数据文件
  python run_calibration.py --no-export        # 不导出结果文件
        """
    )
    
    parser.add_argument(
        '-n', '--n-points',
        type=int,
        default=10,
        help='校准工况点数量 (默认: 10)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='校准数据文件路径 (默认: data/calibration_data.csv)'
    )
    
    parser.add_argument(
        '--no-export',
        action='store_true',
        help='不导出结果文件'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='静默模式，减少输出'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_calibration(
            n_points=args.n_points,
            data_file=args.data,
            export_results=not args.no_export,
            verbose=not args.quiet
        )
        
        # 打印最终参数摘要
        if not args.quiet:
            print("\n" + "=" * 70)
            print("最终校准参数:")
            print("=" * 70)
            for stage, result in results.items():
                status = "✓" if result.success else "✗"
                print(f"[{status}] {stage}:")
                for param, value in result.parameters.items():
                    print(f"      {param}: {value:.4f}")
                print(f"      误差: {result.error*100:.2f}%")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请确保校准数据文件存在。")
        return 1
        
    except Exception as e:
        print(f"\n校准过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
