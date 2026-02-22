#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 MARL 模拟数据
==================
生成 MARL 双智能体训练过程的模拟数据，用于可视化调试与展示。

生成的文件:
- experiments/outputs/training_log.csv        训练日志（各 epoch 指标）
- experiments/outputs/results.json            实验结果（含混淆矩阵/检测延迟/控制响应）
- experiments/outputs/comparison_results.csv  方法对比结果

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


def _generate_training_log(n_episodes: int = 500, seed: int = 42) -> pd.DataFrame:
    """生成模拟训练日志

    模拟双智能体（诊断 + 控制）在 MAPPO 框架下的训练过程：
    - 奖励递增、损失递减、熵缓慢衰减、准确率提高

    Args:
        n_episodes: 训练轮次数
        seed: 随机种子

    Returns:
        包含训练指标的 DataFrame
    """
    rng = np.random.RandomState(seed)
    episodes = np.arange(1, n_episodes + 1)
    t = episodes / n_episodes  # 归一化进度 [0, 1]

    # ── 奖励曲线（先快后慢递增 + 噪声）──────────────────────────────
    reward_diag = (
        -50 + 350 * (1 - np.exp(-3.5 * t))
        + rng.normal(0, 25, n_episodes) * (1 - 0.6 * t)
    )
    reward_ctrl = (
        -80 + 400 * (1 - np.exp(-3.0 * t))
        + rng.normal(0, 30, n_episodes) * (1 - 0.5 * t)
    )
    reward_total = reward_diag + reward_ctrl + rng.normal(0, 10, n_episodes)

    # ── 策略损失（快速下降 → 缓慢收敛）─────────────────────────────
    loss_diag_policy = (
        0.8 * np.exp(-4.0 * t) + 0.05
        + rng.normal(0, 0.03, n_episodes) * (1 - 0.5 * t)
    )
    loss_ctrl_policy = (
        1.0 * np.exp(-3.5 * t) + 0.08
        + rng.normal(0, 0.04, n_episodes) * (1 - 0.5 * t)
    )

    # ── 价值损失（缓慢下降）───────────────────────────────────────
    loss_diag_value = (
        2.0 * np.exp(-2.0 * t) + 0.1
        + rng.normal(0, 0.08, n_episodes) * (1 - 0.4 * t)
    )
    loss_ctrl_value = (
        2.5 * np.exp(-2.5 * t) + 0.15
        + rng.normal(0, 0.10, n_episodes) * (1 - 0.4 * t)
    )

    # ── 策略熵（缓慢衰减 → 探索减少）─────────────────────────────
    entropy_diag = (
        1.2 * np.exp(-1.5 * t) + 0.15
        + rng.normal(0, 0.03, n_episodes)
    )
    entropy_ctrl = (
        1.0 * np.exp(-1.8 * t) + 0.12
        + rng.normal(0, 0.025, n_episodes)
    )

    # ── 诊断准确率（Sigmoid 上升 → 0.92+）────────────────────────
    diag_accuracy = (
        0.25 + 0.70 / (1 + np.exp(-10 * (t - 0.35)))
        + rng.normal(0, 0.03, n_episodes) * (1 - 0.6 * t)
    )
    diag_accuracy = np.clip(diag_accuracy, 0, 1)

    # ── 控制性能（Pmax 维持率，Sigmoid 上升 → 0.95+）──────────────
    ctrl_performance = (
        0.55 + 0.42 / (1 + np.exp(-8 * (t - 0.30)))
        + rng.normal(0, 0.025, n_episodes) * (1 - 0.5 * t)
    )
    ctrl_performance = np.clip(ctrl_performance, 0, 1)

    df = pd.DataFrame({
        'episodes': episodes,
        'reward_diag': reward_diag,
        'reward_ctrl': reward_ctrl,
        'reward_total': reward_total,
        'loss_diag_policy': np.abs(loss_diag_policy),
        'loss_ctrl_policy': np.abs(loss_ctrl_policy),
        'loss_diag_value': np.abs(loss_diag_value),
        'loss_ctrl_value': np.abs(loss_ctrl_value),
        'entropy_diag': np.abs(entropy_diag),
        'entropy_ctrl': np.abs(entropy_ctrl),
        'diag_accuracy': diag_accuracy,
        'ctrl_performance': ctrl_performance,
    })
    return df


def _generate_confusion_matrix(seed: int = 42) -> list:
    """生成 4×4 故障诊断混淆矩阵（健康/正时/泄漏/燃油）

    模拟准确率约 92% 的诊断结果。

    Args:
        seed: 随机种子

    Returns:
        4×4 嵌套列表
    """
    rng = np.random.RandomState(seed)
    # 构建对角线占优的矩阵
    cm = np.array([
        [92, 3, 3, 2],   # 健康
        [4, 88, 5, 3],   # 正时故障
        [2, 6, 85, 7],   # 泄漏故障
        [3, 2, 8, 87],   # 燃油故障
    ])
    # 加少量随机扰动使其更逼真
    noise = rng.randint(-2, 3, size=cm.shape)
    np.fill_diagonal(noise, 0)
    cm = cm + noise
    cm = np.clip(cm, 0, None)
    return cm.tolist()


def _generate_detection_delays(seed: int = 42) -> dict:
    """生成各故障类型的检测延迟数据

    Args:
        seed: 随机种子

    Returns:
        {故障名称: [延迟列表]} 字典
    """
    rng = np.random.RandomState(seed)
    delays = {
        '正时故障': (rng.exponential(2.5, 50) + 1).tolist(),
        '泄漏故障': (rng.exponential(3.0, 50) + 1.5).tolist(),
        '燃油故障': (rng.exponential(2.0, 50) + 1).tolist(),
        '复合故障': (rng.exponential(4.0, 50) + 2).tolist(),
    }
    return delays


def _generate_control_response(n_steps: int = 200, seed: int = 42) -> dict:
    """生成容错控制响应时间序列

    模拟一个渐进故障注入场景：前 50 步正常，50-120 故障加剧，120-200 控制补偿

    Args:
        n_steps: 时间步数
        seed: 随机种子

    Returns:
        包含 time_steps / fault_severity / pmax_actual / pmax_target /
        timing_offset / fuel_adj 的字典
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_steps)

    # 故障严重程度：0→0（正常期）→ 渐增 → 维持
    fault = np.zeros(n_steps)
    ramp_start, ramp_end = 50, 120
    fault[ramp_start:ramp_end] = np.linspace(0, 0.8, ramp_end - ramp_start)
    fault[ramp_end:] = 0.8 + 0.05 * np.sin(0.1 * t[ramp_end:])
    fault = np.clip(fault, 0, 1)

    pmax_target = 150.0  # bar

    # Pmax 实际值：故障导致偏离，控制智能体逐渐补偿
    pmax_actual = np.full(n_steps, pmax_target)
    # 故障引起的偏差
    fault_effect = -fault * 20  # 最大降低 20 bar
    # 控制补偿效果（延迟响应 + 逐步修正）
    compensation = np.zeros(n_steps)
    for i in range(ramp_start + 5, n_steps):
        compensation[i] = compensation[i - 1] + 0.12 * (
            -fault_effect[i] - compensation[i - 1]
        )
    pmax_actual = pmax_target + fault_effect + compensation + rng.normal(0, 1.2, n_steps)

    # 正时补偿 [deg]
    timing_offset = np.zeros(n_steps)
    for i in range(ramp_start + 3, n_steps):
        timing_offset[i] = timing_offset[i - 1] + 0.08 * (
            fault[i] * 5.0 - timing_offset[i - 1]
        )
    timing_offset += rng.normal(0, 0.15, n_steps)

    # 燃油调整系数
    fuel_adj = np.ones(n_steps)
    for i in range(ramp_start + 3, n_steps):
        fuel_adj[i] = fuel_adj[i - 1] + 0.06 * (
            1.0 + fault[i] * 0.12 - fuel_adj[i - 1]
        )
    fuel_adj += rng.normal(0, 0.008, n_steps)

    return {
        'time_steps': t.tolist(),
        'fault_severity': fault.tolist(),
        'pmax_actual': pmax_actual.tolist(),
        'pmax_target': float(pmax_target),
        'timing_offset': timing_offset.tolist(),
        'fuel_adj': fuel_adj.tolist(),
    }


def _generate_comparison_results() -> pd.DataFrame:
    """生成方法性能对比数据

    对比 MAPPO-PINN-KAN（本方法）、PPO、SAC、PID、Threshold

    Returns:
        方法对比 DataFrame
    """
    df = pd.DataFrame({
        'method': [
            'MAPPO-PINN-KAN\n(本方法)', 'PPO', 'SAC', 'PID', 'Threshold'
        ],
        '诊断准确率': [0.923, 0.856, 0.841, 0.241, 0.266],
        '$P_{max}$ 维持率': [0.961, 0.912, 0.905, 0.574, 0.531],
        '检测延迟 (cycles)': [2.1, 3.8, 4.2, 0.0, 16.5],
        '综合评分': [0.935, 0.871, 0.859, 0.412, 0.385],
    })
    return df


def main():
    """生成所有 MARL 模拟数据"""
    PATH_CONFIG.ensure_dirs()
    out_dir = PATH_CONFIG.EXPERIMENT_RESULTS_DIR
    print(f"输出目录: {out_dir}")

    # ── 1. training_log.csv ────────────────────────────────────────────
    print("[1/3] 生成训练日志 training_log.csv ...")
    df_train = _generate_training_log()
    train_path = os.path.join(out_dir, 'training_log.csv')
    df_train.to_csv(train_path, index=False)
    print(f"  ✓ {train_path}  ({len(df_train)} episodes)")

    # ── 2. results.json（合并已有内容 + 新字段）───────────────────────
    print("[2/3] 更新实验结果 results.json ...")
    results_path = os.path.join(out_dir, 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = {}

    results['confusion_matrix'] = _generate_confusion_matrix()
    results['detection_delays'] = _generate_detection_delays()
    results['control_response'] = _generate_control_response()

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {results_path}  (added confusion_matrix / detection_delays / control_response)")

    # ── 3. comparison_results.csv ─────────────────────────────────────
    print("[3/3] 生成方法对比 comparison_results.csv ...")
    df_comp = _generate_comparison_results()
    comp_path = os.path.join(out_dir, 'comparison_results.csv')
    df_comp.to_csv(comp_path, index=False)
    print(f"  ✓ {comp_path}  ({len(df_comp)} methods)")

    print("\n全部 mock 数据生成完成！")


if __name__ == '__main__':
    main()
