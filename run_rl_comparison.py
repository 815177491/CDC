#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速运行RL算法对比实验
======================
运行多种强化学习算法的对比实验，选出最优算法

使用方法:
    python run_rl_comparison.py

Author: CDC Project
Date: 2026-01-21
"""

import os
import sys

# 添加项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def check_dependencies():
    """检查依赖"""
    missing = []
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        missing.append("torch")
        print("✗ PyTorch (需要安装)")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        missing.append("numpy")
        print("✗ NumPy (需要安装)")
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError:
        missing.append("matplotlib")
        print("✗ Matplotlib (可选，用于绘图)")
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError:
        missing.append("pandas")
        print("✗ Pandas (可选，用于数据导出)")
    
    return missing


def main():
    """主函数"""
    print("="*70)
    print("多种强化学习算法对比实验")
    print("="*70)
    
    print("\n【依赖检查】")
    missing = check_dependencies()
    
    if 'torch' in missing:
        print("\n错误: 需要安装PyTorch才能运行实验")
        print("安装命令: pip install torch")
        return
    
    # 导入实验模块
    print("\n【导入模块】")
    try:
        from experiments.rl_comparison import ExperimentConfig, ExperimentRunner
        from agents.rl_algorithms import list_algorithms, ALGORITHM_INFO
        print("✓ 模块导入成功")
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        return
    
    # 显示可用算法
    print("\n【可用算法】")
    for name in list_algorithms():
        info = ALGORITHM_INFO.get(name, {})
        print(f"  {name:20s} | {info.get('venue', 'N/A'):10s} {info.get('year', ''):4s} | {info.get('type', 'N/A')}")
    
    # 实验配置
    print("\n【实验配置】")
    config = ExperimentConfig(
        algorithms=['DQN', 'DuelingDQN', 'PPO', 'SAC', 'TD3', 'IQL'],
        n_episodes=100,  # 快速测试用100轮
        max_steps_per_episode=200,
        eval_frequency=10,
        n_eval_episodes=3,
        seed=42,
        output_dir='experiment_results'
    )
    
    print(f"  算法: {config.algorithms}")
    print(f"  训练Episodes: {config.n_episodes}")
    print(f"  每Episode步数: {config.max_steps_per_episode}")
    print(f"  评估频率: 每{config.eval_frequency}个Episode")
    print(f"  输出目录: {config.output_dir}")
    
    # 确认运行
    print("\n是否开始实验? (输入 y 继续, 其他键退出)")
    try:
        user_input = input().strip().lower()
        if user_input != 'y':
            print("实验已取消")
            return
    except:
        pass  # 非交互模式自动继续
    
    # 运行实验
    print("\n【开始实验】")
    runner = ExperimentRunner(config)
    results = runner.run_all_experiments()
    
    # 保存和可视化
    print("\n【保存结果】")
    runner.save_results()
    runner.plot_comparison()
    runner.plot_detailed_analysis()
    
    # 生成报告
    print("\n【实验报告】")
    report = runner.generate_report()
    print(report)
    
    # 推荐最佳算法
    if results:
        best_algo = max(results.items(), 
                       key=lambda x: x[1].final_performance['eval_reward'])
        print("\n" + "="*70)
        print(f"推荐算法: {best_algo[0]}")
        info = ALGORITHM_INFO.get(best_algo[0], {})
        print(f"来源: {info.get('venue', '')} {info.get('year', '')}")
        print(f"最终评估奖励: {best_algo[1].final_performance['eval_reward']:.2f}")
        print("="*70)
    
    return results


if __name__ == '__main__':
    main()
