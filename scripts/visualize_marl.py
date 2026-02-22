#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MARL 可视化运行脚本
====================
生成 MARL 双智能体强化学习相关的所有学术风格图片

生成的图片保存在 visualization_output/training/ 和 visualization_output/modeling/ 目录下

使用方法:
    python scripts/visualize_marl.py

Author: CDC Project
Date: 2026-02-22
"""

# 标准库
import sys
import os
import json

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 第三方库
import numpy as np
import pandas as pd

# 本项目模块
from config import PATH_CONFIG
from visualization.marl_plots import (
    plot_training_curves,
    plot_reward_distribution,
    plot_confusion_matrix,
    plot_detection_delay,
    plot_control_response,
    plot_method_comparison,
    plot_dual_agent_architecture,
)


def _load_training_log() -> dict:
    """尝试从 experiments/outputs/ 加载训练日志

    Returns:
        训练历史字典，若文件不存在返回空字典
    """
    log_path = os.path.join(PATH_CONFIG.EXPERIMENT_RESULTS_DIR, 'training_log.csv')
    if not os.path.exists(log_path):
        print(f"  [Skip] 训练日志不存在: {log_path}")
        return {}
    df = pd.read_csv(log_path)
    return {col: df[col].tolist() for col in df.columns}


def _load_comparison_results() -> tuple:
    """尝试从 experiments/outputs/ 加载对比实验结果

    Returns:
        (methods, metrics) 元组，若文件不存在返回 ([], {})
    """
    comp_path = os.path.join(PATH_CONFIG.EXPERIMENT_RESULTS_DIR, 'comparison_results.csv')
    if not os.path.exists(comp_path):
        print(f"  [Skip] 对比实验结果不存在: {comp_path}")
        return [], {}
    df = pd.read_csv(comp_path)
    methods = df['method'].tolist()
    metrics = {col: df[col].tolist() for col in df.columns if col != 'method'}
    return methods, metrics


def _load_experiment_results() -> dict:
    """尝试从 experiments/outputs/results.json 加载实验结果

    Returns:
        实验结果字典，若文件不存在返回空字典
    """
    results_path = os.path.join(PATH_CONFIG.EXPERIMENT_RESULTS_DIR, 'results.json')
    if not os.path.exists(results_path):
        print(f"  [Skip] 实验结果不存在: {results_path}")
        return {}
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """主函数：生成所有 MARL 相关可视化图表"""
    print("=" * 70)
    print("MARL 双智能体可视化图片生成脚本")
    print("=" * 70)
    print(f"工作目录: {PROJECT_ROOT}")
    print(f"训练图输出目录: {PATH_CONFIG.VIS_TRAINING_DIR}")
    print(f"架构图输出目录: {PATH_CONFIG.VIS_MODELING_DIR}")
    print("=" * 70)

    results = {}

    # ── 1. 双智能体网络架构示意图（category='modeling'）────────────────
    try:
        print("\n[1/7] 生成双智能体网络架构图...")
        plot_dual_agent_architecture()
        results['dual_agent_architecture'] = 'OK'
        print("  ✓ 完成")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['dual_agent_architecture'] = f'FAILED: {e}'

    # ── 2. 训练曲线 ───────────────────────────────────────────────────
    try:
        print("\n[2/7] 生成训练曲线...")
        history = _load_training_log()
        if history:
            plot_training_curves(history)
            results['training_curves'] = 'OK'
            print("  ✓ 完成")
        else:
            results['training_curves'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['training_curves'] = f'FAILED: {e}'

    # ── 3. 奖励分布 ───────────────────────────────────────────────────
    try:
        print("\n[3/7] 生成奖励分布图...")
        history = _load_training_log()
        if history and any(k in history for k in ['reward_diag', 'reward_ctrl', 'reward_total']):
            plot_reward_distribution(history)
            results['reward_distribution'] = 'OK'
            print("  ✓ 完成")
        else:
            results['reward_distribution'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['reward_distribution'] = f'FAILED: {e}'

    # ── 4. 混淆矩阵 ───────────────────────────────────────────────────
    try:
        print("\n[4/7] 生成故障诊断混淆矩阵...")
        exp_results = _load_experiment_results()
        cm_data = exp_results.get('confusion_matrix')
        if cm_data is not None:
            cm = np.array(cm_data)
            plot_confusion_matrix(cm)
            results['confusion_matrix'] = 'OK'
            print("  ✓ 完成")
        else:
            print("  [Skip] 实验结果中无 confusion_matrix 数据")
            results['confusion_matrix'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['confusion_matrix'] = f'FAILED: {e}'

    # ── 5. 检测延迟箱型图 ─────────────────────────────────────────────
    try:
        print("\n[5/7] 生成故障检测延迟箱型图...")
        exp_results = _load_experiment_results()
        delays = exp_results.get('detection_delays')
        if delays is not None:
            plot_detection_delay(delays)
            results['detection_delay'] = 'OK'
            print("  ✓ 完成")
        else:
            print("  [Skip] 实验结果中无 detection_delays 数据")
            results['detection_delay'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['detection_delay'] = f'FAILED: {e}'

    # ── 6. 容错控制响应 ───────────────────────────────────────────────
    try:
        print("\n[6/7] 生成容错控制响应图...")
        exp_results = _load_experiment_results()
        ctrl = exp_results.get('control_response')
        if ctrl is not None:
            plot_control_response(
                time_steps=np.array(ctrl['time_steps']),
                fault_severity=np.array(ctrl['fault_severity']),
                pmax_actual=np.array(ctrl['pmax_actual']),
                pmax_target=ctrl['pmax_target'],
                timing_offset=np.array(ctrl['timing_offset']),
                fuel_adj=np.array(ctrl['fuel_adj']),
            )
            results['control_response'] = 'OK'
            print("  ✓ 完成")
        else:
            print("  [Skip] 实验结果中无 control_response 数据")
            results['control_response'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['control_response'] = f'FAILED: {e}'

    # ── 7. 方法性能对比 ───────────────────────────────────────────────
    try:
        print("\n[7/7] 生成方法性能对比图...")
        methods, metrics = _load_comparison_results()
        if methods and metrics:
            plot_method_comparison(methods, metrics)
            results['method_comparison'] = 'OK'
            print("  ✓ 完成")
        else:
            results['method_comparison'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['method_comparison'] = f'FAILED: {e}'

    # ── 结果汇总 ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("生成结果汇总:")
    print("=" * 70)
    for name, status in results.items():
        if status == 'OK':
            marker = "✓"
        elif 'SKIP' in str(status):
            marker = "○"
        else:
            marker = "✗"
        print(f"  {marker} {name}: {status}")

    ok_count = sum(1 for s in results.values() if s == 'OK')
    skip_count = sum(1 for s in results.values() if 'SKIP' in str(s))
    fail_count = len(results) - ok_count - skip_count
    print(f"\n合计: {ok_count} 成功, {skip_count} 跳过, {fail_count} 失败")
    print(f"训练图目录: {PATH_CONFIG.VIS_TRAINING_DIR}")
    print(f"架构图目录: {PATH_CONFIG.VIS_MODELING_DIR}")

    return results


if __name__ == '__main__':
    main()
