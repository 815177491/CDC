#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU加速RL对比实验 - 主运行脚本
===============================
功能：
1. GPU/CPU自动检测
2. 自动调整batch_size
3. 先1个种子快速验证
4. 再5个种子完整实验
5. 综合评分选择最优方法

运行方式：
    python run_gpu_comparison.py              # 完整实验
    python run_gpu_comparison.py --quick      # 仅快速验证
    python run_gpu_comparison.py --full-only  # 跳过快速验证直接完整实验

Author: CDC Project
Date: 2026-01-21
"""

import sys
import os
import argparse
import time
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def print_banner():
    """打印启动横幅"""
    print("\n" + "="*70)
    print("🚀 柴油机控制方法 GPU加速对比实验")
    print("="*70)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*70)


def check_environment():
    """检查运行环境"""
    print("\n📋 环境检查:")
    
    # Python版本
    print(f"  Python版本: {sys.version.split()[0]}")
    
    # PyTorch
    try:
        import torch
        print(f"  PyTorch版本: {torch.__version__}")
        
        # GPU检测
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  ✅ GPU可用: {gpu_name} ({gpu_mem:.1f} GB)")
            
            # 自动batch_size建议
            if gpu_mem >= 8:
                batch_size = 512
            elif gpu_mem >= 4:
                batch_size = 256
            else:
                batch_size = 128
            print(f"  📦 推荐batch_size: {batch_size}")
            
            device = 'cuda'
        else:
            print("  ⚠️  GPU不可用，使用CPU")
            batch_size = 64
            device = 'cpu'
            
    except ImportError:
        print("  ❌ PyTorch未安装")
        return None, None, None
    
    # 检查项目模块
    try:
        from agents.rl_algorithms import get_algorithm, SAC
        print("  ✅ 基础RL算法模块正常")
    except ImportError as e:
        print(f"  ❌ 基础RL算法模块导入失败: {e}")
    
    try:
        from agents.advanced_rl_algorithms import (
            get_advanced_algorithm, TDMPC2, MambaPolicy, DPMD
        )
        print("  ✅ 2024-2025新算法模块正常")
    except ImportError as e:
        print(f"  ❌ 新算法模块导入失败: {e}")
    
    try:
        from experiments.five_method_comparison import FiveMethodComparison, ExperimentConfig
        print("  ✅ 实验框架模块正常")
    except ImportError as e:
        print(f"  ❌ 实验框架模块导入失败: {e}")
        return None, None, None
    
    print("-"*70)
    return device, batch_size, torch


def run_quick_validation(config):
    """运行快速验证"""
    from experiments.five_method_comparison import FiveMethodComparison
    
    print("\n" + "="*70)
    print("🔍 快速验证 (单种子: seed=42)")
    print("="*70)
    
    experiment = FiveMethodComparison(config)
    # 论文正式对比方法：PID + DQN + SAC + TD-MPC2 + DPMD
    methods = ["PID", "DQN", "SAC", "TDMPC2", "DPMD"]
    
    results = experiment.run_quick_validation(methods)
    
    return results, experiment


def run_full_experiment(config, experiment=None):
    """运行完整实验"""
    from experiments.five_method_comparison import FiveMethodComparison
    
    if experiment is None:
        experiment = FiveMethodComparison(config)
    
    print("\n" + "="*70)
    print("🔬 完整对比实验 (5个种子: 42, 123, 456, 789, 1024)")
    print("="*70)
    
    # 论文正式对比方法：PID + DQN + SAC + TD-MPC2 + DPMD
    methods = ["PID", "DQN", "SAC", "TDMPC2", "DPMD"]
    comparison = experiment.run_full_comparison(methods)
    
    return comparison


def generate_report(comparison, save_dir: str, experiment=None):
    """生成实验报告和可视化图表"""
    import json
    import numpy as np
    
    print("\n" + "="*70)
    print("📊 生成实验报告和可视化")
    print("="*70)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成可视化图表
    if experiment is not None:
        print("\n生成可视化图表...")
        experiment.plot_comparison_results(comparison, save_dir)
    
    # 生成详细报告
    report_lines = []
    report_lines.append("# 柴油机控制方法对比实验报告")
    report_lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n## 实验配置")
    report_lines.append("- 训练Episodes: 400")
    report_lines.append("- 评估Episodes: 10")
    report_lines.append("- 随机种子: [42, 123, 456, 789, 1024]")
    
    report_lines.append("\n## 对比方法")
    report_lines.append("| 方法 | 年份 | 类型 |")
    report_lines.append("|------|------|------|")
    report_lines.append("| PID | - | 传统控制 |")
    report_lines.append("| SAC | 2018 | 最大熵RL |")
    report_lines.append("| TD-MPC2 | 2024 | 模型预测控制 |")
    report_lines.append("| Mamba Policy | 2025 | 状态空间模型 |")
    report_lines.append("| DPMD | 2025 | 扩散策略+镜像下降 |")
    
    report_lines.append("\n## 实验结果")
    report_lines.append("| 排名 | 方法 | 综合评分 | 达标率 | 平均奖励 | 收敛Episode |")
    report_lines.append("|------|------|----------|--------|----------|-------------|")
    
    sorted_methods = sorted(comparison.rankings.items(), key=lambda x: x[1])
    for method, rank in sorted_methods:
        results = comparison.all_results.get(method, [])
        if not results:
            continue
        
        mean_accuracy = np.mean([r.accuracy_rate for r in results])
        mean_reward = np.mean([r.final_reward for r in results])
        mean_convergence = np.mean([r.convergence_episode for r in results])
        score = comparison.scores.get(method, 0) if comparison.scores else 0
        
        report_lines.append(
            f"| {rank} | {method} | {score:.2f} | "
            f"{mean_accuracy*100:.1f}% | {mean_reward:.2f} | {mean_convergence:.0f} |"
        )
    
    report_lines.append(f"\n## 结论")
    report_lines.append(f"\n**推荐方法: {comparison.best_method}**")
    best_score = comparison.scores.get(comparison.best_method, 0) if comparison.scores else 0
    report_lines.append(f"\n综合评分: {best_score:.2f}")
    
    # 写入报告
    report_path = os.path.join(save_dir, "experiment_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"  📄 报告已保存: {report_path}")
    
    return report_path


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='GPU加速RL对比实验')
    parser.add_argument('--quick', action='store_true', 
                       help='仅运行快速验证')
    parser.add_argument('--full-only', action='store_true',
                       help='跳过快速验证，直接运行完整实验')
    parser.add_argument('--episodes', type=int, default=500,
                       help='训练episodes数量 (默认: 500, 最终实验)')
    parser.add_argument('--steps', type=int, default=200,
                       help='每episode步数 (默认: 200, 优化后)')
    parser.add_argument('--save-dir', type=str, default='results/comparison',
                       help='结果保存目录')
    args = parser.parse_args()
    
    # 打印横幅
    print_banner()
    
    # 环境检查
    device, batch_size, torch = check_environment()
    if device is None:
        print("\n❌ 环境检查失败，请确保PyTorch已安装")
        return
    
    # 创建配置
    from experiments.five_method_comparison import ExperimentConfig
    
    config = ExperimentConfig(
        n_episodes=args.episodes,
        max_steps_per_episode=args.steps,
        device=device,
        batch_size=batch_size,
        save_dir=args.save_dir
    )
    
    print(f"\n📋 实验配置:")
    print(f"  - Episodes: {config.n_episodes}")
    print(f"  - Batch Size: {config.batch_size}")
    print(f"  - Device: {config.device}")
    print(f"  - 保存目录: {config.save_dir}")
    
    start_time = time.time()
    
    # 运行实验
    if args.quick:
        # 仅快速验证
        results, experiment = run_quick_validation(config)
        
        # 生成快速验证的可视化图表
        if results and experiment:
            from experiments.five_method_comparison import ComparisonResult
            # 将快速验证结果转换为ComparisonResult格式
            quick_comparison = ComparisonResult(
                methods=list(results.keys()),
                all_results={m: [r] for m, r in results.items()},
                rankings={m: i+1 for i, (m, r) in enumerate(
                    sorted(results.items(), key=lambda x: -x[1].accuracy_rate)
                )},
                best_method=max(results.items(), key=lambda x: x[1].accuracy_rate)[0]
            )
            generate_report(quick_comparison, args.save_dir, experiment)
        
        # 打印推荐
        if results:
            best = max(results.items(), key=lambda x: x[1].accuracy_rate)
            print(f"\n🏆 快速验证推荐: {best[0]}")
    
    elif args.full_only:
        # 直接完整实验
        experiment = None
        comparison = run_full_experiment(config)
        generate_report(comparison, args.save_dir, experiment)
        print(f"\n🏆 最终推荐方法: {comparison.best_method}")
    
    else:
        # 默认：先快速验证，再完整实验
        results, experiment = run_quick_validation(config)
        
        print("\n" + "-"*70)
        print("快速验证完成! 继续完整实验...")
        print("-"*70)
        
        comparison = run_full_experiment(config, experiment)
        generate_report(comparison, args.save_dir, experiment)
        print(f"\n🏆 最终推荐方法: {comparison.best_method}")
    
    # 总时间
    total_time = time.time() - start_time
    print(f"\n⏱️  总运行时间: {total_time/60:.1f} 分钟")
    print("="*70)
    print("✅ 实验完成!")
    print("="*70)


if __name__ == "__main__":
    main()
