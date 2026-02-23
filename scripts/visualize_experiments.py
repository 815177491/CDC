#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment Comparison Visualization Runner
==========================================
Generate bar charts for comparison experiments.

Outputs are saved under visualization_output/experiments/.

Usage:
    python scripts/visualize_experiments.py

Author: CDC Project
Date: 2026-02-23
"""

# Standard library
import json
import os
import sys
from typing import Dict, List, Tuple

# Third-party
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Project
from config import PATH_CONFIG
from visualization.experiment_plots import (
    plot_overall_metrics_bars,
    plot_per_fault_f1_bars,
    plot_ablation_delta_bars,
    plot_severity_robustness_bars,
    plot_multi_run_statistics,
    build_avg_severity_accuracy,
)


def _load_comparison_results() -> Tuple[List[str], Dict[str, List[float]]]:
    """Load comparison_results.csv or create mock data."""
    comp_path = os.path.join(PATH_CONFIG.EXPERIMENT_RESULTS_DIR, 'comparison_results.csv')
    if os.path.exists(comp_path):
        df = pd.read_csv(comp_path)
        methods = df['method'].tolist() if 'method' in df.columns else []
        metrics = {col: df[col].tolist() for col in df.columns if col != 'method'}
        return methods, metrics

    rng = np.random.RandomState(42)
    methods = ['Ours', 'PPO', 'SAC', 'PID', 'Threshold']
    metrics = {
        '诊断准确率': (0.8 + 0.15 * rng.rand(len(methods))).tolist(),
        '$P_{max}$ 维持率': (0.85 + 0.12 * rng.rand(len(methods))).tolist(),
        '检测延迟 (cycles)': (1.0 + 10.0 * rng.rand(len(methods))).tolist(),
        '综合评分': (0.75 + 0.2 * rng.rand(len(methods))).tolist(),
    }
    return methods, metrics


def _load_results_json() -> Dict:
    """Load results.json or return empty dict."""
    results_path = os.path.join(PATH_CONFIG.EXPERIMENT_RESULTS_DIR, 'results.json')
    if not os.path.exists(results_path):
        return {}
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _mock_per_fault_f1(methods: List[str]) -> Tuple[List[str], List[List[float]]]:
    rng = np.random.RandomState(7)
    fault_names = ['Health', 'Timing', 'Leak', 'Fuel', 'Compound']
    f1_matrix = (0.55 + 0.4 * rng.rand(len(methods), len(fault_names))).tolist()
    return fault_names, f1_matrix


def _mock_ablation() -> Dict[str, Dict[str, float]]:
    return {
        '完整协同': {'综合奖励': 280.0},
        'No Control': {'综合奖励': 135.0},
        'No Diagnosis': {'综合奖励': 170.0},
        'No SharedCritic': {'综合奖励': 235.0},
    }


def _mock_severity() -> Tuple[List[float], List[float]]:
    levels = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]
    acc = [0.68, 0.70, 0.74, 0.79, 0.86, 0.91, 0.93, 0.92]
    return levels, acc


def main() -> None:
    """Generate bar charts for experiment comparison."""
    print('=' * 70)
    print('对比实验条形图生成脚本（5 张）')
    print('=' * 70)
    print(f'Output dir: {PATH_CONFIG.VIS_EXPERIMENTS_DIR}')

    results = {}

    methods, metrics = _load_comparison_results()
    exp_results = _load_results_json()

    try:
        print('[1/5] Overall metrics grouped bars...')
        plot_overall_metrics_bars(methods, metrics)
        results['overall_metrics'] = 'OK'
    except Exception as exc:
        results['overall_metrics'] = f'FAILED: {exc}'
        print(f'  FAILED: {exc}')

    try:
        print('[2/5] Per-fault F1 grouped bars...')
        per_fault = exp_results.get('per_fault_f1', {})
        if per_fault:
            pf_methods = per_fault.get('methods', methods)
            fault_names = per_fault.get('fault_names', [])
            f1_matrix = per_fault.get('f1_matrix', [])
        else:
            pf_methods = methods
            fault_names, f1_matrix = _mock_per_fault_f1(methods)

        plot_per_fault_f1_bars(pf_methods, fault_names, f1_matrix)
        results['per_fault_f1'] = 'OK'
    except Exception as exc:
        results['per_fault_f1'] = f'FAILED: {exc}'
        print(f'  FAILED: {exc}')

    try:
        print('[3/5] Ablation delta bars...')
        ablation = exp_results.get('ablation_study') or _mock_ablation()
        plot_ablation_delta_bars(ablation)
        results['ablation'] = 'OK'
    except Exception as exc:
        results['ablation'] = f'FAILED: {exc}'
        print(f'  FAILED: {exc}')

    try:
        print('[4/5] Robustness severity bars...')
        severity = exp_results.get('severity_sensitivity')
        if severity:
            levels, avg_acc = build_avg_severity_accuracy(severity)
        else:
            levels, avg_acc = _mock_severity()

        plot_severity_robustness_bars(levels, avg_acc)
        results['robustness'] = 'OK'
    except Exception as exc:
        results['robustness'] = f'FAILED: {exc}'
        print(f'  FAILED: {exc}')

    try:
        print('[5/5] Multi-run statistics CI bars...')
        multi_run = exp_results.get('multi_run_statistics')
        if multi_run:
            plot_multi_run_statistics(
                multi_run['methods'],
                multi_run['metric_name'],
                multi_run['means'],
                multi_run['stds'],
                multi_run.get('n_runs', 50),
            )
            results['multi_run'] = 'OK'
        else:
            results['multi_run'] = 'SKIPPED (no data)'
    except Exception as exc:
        results['multi_run'] = f'FAILED: {exc}'
        print(f'  FAILED: {exc}')

    ok = sum(1 for v in results.values() if v == 'OK')
    print('-' * 70)
    print(f'Completed: {ok}/{len(results)}')


if __name__ == '__main__':
    main()
