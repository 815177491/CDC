#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 MARL 模拟数据
==================
生成 MARL 双智能体训练过程的模拟数据，用于可视化调试与展示。

生成的文件:
- experiments/outputs/training_log.csv        训练日志（各 epoch 指标）
- experiments/outputs/results.json            实验结果（含混淆矩阵/检测延迟/控制响应等）
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


# ============================================================================
# 全新: 诊断智能体评价模拟数据
# ============================================================================

def _generate_roc_data(seed: int = 42) -> dict:
    """生成多故障类型 ROC 曲线数据

    Args:
        seed: 随机种子

    Returns:
        {故障类型: {'fpr': [...], 'tpr': [...], 'auc': float}}
    """
    rng = np.random.RandomState(seed)
    fault_types = {
        '正时故障': 0.96,
        '泄漏故障': 0.93,
        '燃油故障': 0.95,
        '复合故障': 0.89,
    }
    roc_data = {}
    for fault_name, target_auc in fault_types.items():
        # 参数化 ROC 曲线: TPR = FPR^(1/k), k 控制 AUC
        # AUC ≈ k/(k+1), 所以 k = AUC/(1-AUC)
        k = target_auc / (1 - target_auc + 1e-8)
        fpr = np.linspace(0, 1, 100)
        tpr = fpr ** (1.0 / k)
        # 加微量噪声使曲线更自然
        tpr = np.clip(tpr + rng.normal(0, 0.008, len(tpr)), 0, 1)
        tpr = np.sort(tpr)  # 保持单调
        tpr[0] = 0.0
        tpr[-1] = 1.0
        auc_val = float(np.trapezoid(tpr, fpr))
        roc_data[fault_name] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': round(auc_val, 4),
        }
    return roc_data


def _generate_pr_data(seed: int = 42) -> dict:
    """生成精确率-召回率曲线数据

    Args:
        seed: 随机种子

    Returns:
        {故障类型: {'precision': [...], 'recall': [...], 'ap': float}}
    """
    rng = np.random.RandomState(seed)
    fault_types = {
        '正时故障': 0.94,
        '泄漏故障': 0.90,
        '燃油故障': 0.92,
        '复合故障': 0.85,
    }
    pr_data = {}
    for fault_name, target_ap in fault_types.items():
        recall = np.linspace(0, 1, 100)
        # 模拟 PR 曲线: precision = a * exp(-b * recall) + c
        a = target_ap * 0.3
        b = 2.0 + rng.uniform(-0.5, 0.5)
        c = target_ap - a * 0.5
        precision = a * np.exp(-b * recall) + c + rng.normal(0, 0.01, 100)
        precision = np.clip(precision, 0.01, 1.0)
        precision = np.sort(precision)[::-1]  # 保持递减
        ap_val = float(np.trapezoid(precision, recall))
        pr_data[fault_name] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'ap': round(ap_val, 4),
        }
    return pr_data


def _generate_severity_sensitivity(seed: int = 42) -> dict:
    """生成故障严重程度敏感性分析数据

    Args:
        seed: 随机种子

    Returns:
        {故障类型: {'severity_levels': [...], 'accuracy': [...], ...}}
    """
    rng = np.random.RandomState(seed)
    levels = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]

    fault_specs = {
        '正时故障': {'base_acc': 0.55, 'max_acc': 0.96},
        '泄漏故障': {'base_acc': 0.50, 'max_acc': 0.93},
        '燃油故障': {'base_acc': 0.58, 'max_acc': 0.95},
    }
    data = {}
    for fault_name, spec in fault_specs.items():
        base = spec['base_acc']
        top = spec['max_acc']
        severity = np.array(levels)
        # Sigmoid-like accuracy growth with severity
        acc = base + (top - base) / (1 + np.exp(-8 * (severity - 0.25)))
        acc += rng.normal(0, 0.015, len(levels))
        acc = np.clip(acc, 0, 1)
        acc_std = 0.04 * (1 - severity * 0.5) + rng.uniform(0.005, 0.015, len(levels))

        f1 = acc * 0.98 + rng.normal(0, 0.01, len(levels))
        f1 = np.clip(f1, 0, 1)
        f1_std = acc_std * 0.9

        data[fault_name] = {
            'severity_levels': levels,
            'accuracy': acc.tolist(),
            'f1_score': f1.tolist(),
            'accuracy_std': acc_std.tolist(),
            'f1_std': f1_std.tolist(),
        }
    return data


def _generate_tsne_embeddings(n_samples: int = 400, seed: int = 42) -> dict:
    """生成 t-SNE 降维后的诊断特征空间数据

    Args:
        n_samples: 总样本数
        seed: 随机种子

    Returns:
        {'embeddings_2d': [[x, y], ...], 'labels': [...]}
    """
    rng = np.random.RandomState(seed)
    n_per_class = n_samples // 4
    centers = [(-5, -5), (5, -3), (-3, 5), (5, 5)]
    spreads = [1.8, 2.0, 2.2, 2.5]

    embeddings = []
    labels = []
    for cls_idx, (cx, cy) in enumerate(centers):
        x = rng.normal(cx, spreads[cls_idx], n_per_class)
        y = rng.normal(cy, spreads[cls_idx], n_per_class)
        embeddings.extend(list(zip(x.tolist(), y.tolist())))
        labels.extend([cls_idx] * n_per_class)

    return {
        'embeddings_2d': embeddings,
        'labels': labels,
    }


def _generate_online_accuracy(n_episodes: int = 500, seed: int = 42) -> dict:
    """生成在线诊断准确率演化数据

    Args:
        n_episodes: 训练轮次数
        seed: 随机种子

    Returns:
        {'episodes': [...], 'overall': {'raw': [...], 'smooth': [...]}, ...}
    """
    rng = np.random.RandomState(seed)
    episodes = list(range(1, n_episodes + 1))
    t = np.arange(1, n_episodes + 1) / n_episodes

    def _make_curve(base, top, shift, noise_scale):
        raw = base + (top - base) / (1 + np.exp(-10 * (t - shift)))
        raw += rng.normal(0, noise_scale, n_episodes) * (1 - 0.6 * t)
        raw = np.clip(raw, 0, 1)
        sm = np.convolve(raw, np.ones(20) / 20, mode='valid')
        return raw.tolist(), sm.tolist()

    data = {'episodes': episodes}
    specs = {
        'overall':  (0.30, 0.93, 0.30, 0.06),
        '正时故障': (0.25, 0.95, 0.28, 0.07),
        '泄漏故障': (0.22, 0.91, 0.35, 0.08),
        '燃油故障': (0.28, 0.94, 0.32, 0.07),
    }
    for key, (base, top, shift, noise) in specs.items():
        raw, smooth = _make_curve(base, top, shift, noise)
        data[key] = {'raw': raw, 'smooth': smooth}
    return data


# ============================================================================
# 全新: 控制智能体评价模拟数据
# ============================================================================

def _generate_tracking_data(n_steps: int = 400, seed: int = 42) -> dict:
    """生成多工况设定值跟踪数据

    模拟 4 个工况阶段（25%→50%→75%→100% 负荷）

    Args:
        n_steps: 总时间步
        seed: 随机种子

    Returns:
        {'time': [...], 'speed': {'actual': [...], 'target': [...]}, ...}
    """
    rng = np.random.RandomState(seed)
    time = list(range(n_steps))
    seg = n_steps // 4

    # 目标设定值（阶跃变化）
    speed_target = np.concatenate([
        np.full(seg, 60), np.full(seg, 80),
        np.full(seg, 95), np.full(n_steps - 3 * seg, 105)
    ])
    power_target = np.concatenate([
        np.full(seg, 5000), np.full(seg, 10000),
        np.full(seg, 16000), np.full(n_steps - 3 * seg, 22000)
    ])
    pscav_target = np.concatenate([
        np.full(seg, 1.5), np.full(seg, 2.2),
        np.full(seg, 2.8), np.full(n_steps - 3 * seg, 3.2)
    ])
    texh_target = np.concatenate([
        np.full(seg, 520), np.full(seg, 560),
        np.full(seg, 600), np.full(n_steps - 3 * seg, 630)
    ])

    def _track(target, tau=8.0, noise=0.01):
        actual = np.zeros_like(target, dtype=float)
        actual[0] = target[0]
        for i in range(1, len(target)):
            actual[i] = actual[i - 1] + (target[i] - actual[i - 1]) / tau
            actual[i] += rng.normal(0, noise * target[i])
        return actual

    return {
        'time': time,
        'speed': {
            'actual': _track(speed_target, tau=10, noise=0.008).tolist(),
            'target': speed_target.tolist(),
        },
        'power': {
            'actual': _track(power_target, tau=12, noise=0.01).tolist(),
            'target': power_target.tolist(),
        },
        'p_scav': {
            'actual': _track(pscav_target, tau=8, noise=0.015).tolist(),
            'target': pscav_target.tolist(),
        },
        'T_exhaust': {
            'actual': _track(texh_target, tau=15, noise=0.005).tolist(),
            'target': texh_target.tolist(),
        },
    }


def _generate_action_distribution(n_samples: int = 2000, seed: int = 42) -> dict:
    """生成控制动作分布数据

    Args:
        n_samples: 采样数
        seed: 随机种子

    Returns:
        {动作名称: [值列表]}
    """
    rng = np.random.RandomState(seed)
    return {
        '正时补偿 [deg]': (rng.normal(0.5, 1.8, n_samples)).tolist(),
        '燃油调整系数': (rng.normal(1.02, 0.05, n_samples)).tolist(),
        '保护级别': (rng.choice([0, 1, 2, 3], n_samples, p=[0.55, 0.25, 0.12, 0.08])).tolist(),
    }


def _generate_robustness_data(seed: int = 42) -> dict:
    """生成控制鲁棒性包络数据

    Args:
        seed: 随机种子

    Returns:
        {'severity_labels': [...], 'Pmax偏差 [bar]': [...], 'Texh偏差 [K]': [...]}
    """
    rng = np.random.RandomState(seed)
    labels = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']

    pmax_dev = []
    texh_dev = []
    for i, severity in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
        n = 60
        base_pmax = severity * 3.0
        base_texh = severity * 15.0
        pmax_dev.append((rng.normal(base_pmax, 1.0 + severity * 2, n)).tolist())
        texh_dev.append((rng.normal(base_texh, 3.0 + severity * 8, n)).tolist())

    return {
        'severity_labels': labels,
        '$P_{max}$ 偏差 [bar]': pmax_dev,
        '$T_{exh}$ 偏差 [K]': texh_dev,
    }


def _generate_constraint_satisfaction(seed: int = 42) -> dict:
    """生成约束满足率数据

    Args:
        seed: 随机种子

    Returns:
        {约束名称: 满足率百分比}
    """
    rng = np.random.RandomState(seed)
    return {
        '$P_{max}$ ≤ 安全阈值': 97.2 + rng.normal(0, 0.5),
        '$T_{exh}$ ≤ 热保护': 95.8 + rng.normal(0, 0.5),
        '转速波动 ≤ ±2%': 98.1 + rng.normal(0, 0.3),
        'SFOC ≤ 基准+5%': 93.5 + rng.normal(0, 0.8),
        '正时 ∈ 安全区间': 99.2 + rng.normal(0, 0.2),
        '燃油系数 ∈ [0.85,1.15]': 96.7 + rng.normal(0, 0.5),
    }


def _generate_energy_efficiency(seed: int = 42) -> dict:
    """生成控制效率分析数据

    Args:
        seed: 随机种子

    Returns:
        {'loads': [...], 'RL 控制': {...}, 'PID 控制': {...}, '无故障基准': {...}}
    """
    rng = np.random.RandomState(seed)
    loads = [25, 50, 75, 100]

    # 基准 SFOC (g/kWh): 低负荷高、高负荷低
    base_sfoc = np.array([195, 178, 170, 168])

    return {
        'loads': loads,
        '无故障基准': {
            'sfoc': base_sfoc.tolist(),
            'sfoc_std': rng.uniform(1, 3, 4).tolist(),
        },
        'RL 控制': {
            'sfoc': (base_sfoc + rng.uniform(2, 6, 4)).tolist(),
            'sfoc_std': rng.uniform(2, 5, 4).tolist(),
        },
        'PID 控制': {
            'sfoc': (base_sfoc + rng.uniform(8, 18, 4)).tolist(),
            'sfoc_std': rng.uniform(4, 8, 4).tolist(),
        },
    }


# ============================================================================
# 全新: 控诊协同评价模拟数据
# ============================================================================

def _generate_collaborative_timeline(n_steps: int = 300, seed: int = 42) -> dict:
    """生成控诊协同时序数据

    模拟: 正常 → 故障注入 → 诊断检出 → 控制响应 → 恢复稳定

    Args:
        n_steps: 时间步数
        seed: 随机种子

    Returns:
        时序数据字典
    """
    rng = np.random.RandomState(seed)
    time = list(range(n_steps))
    fault_onset = 60
    diag_detect = 63    # 检出延迟 3 步
    ctrl_respond = 66   # 控制延迟再 3 步

    # 故障注入
    fault_inj = np.zeros(n_steps)
    ramp_end = 100
    fault_inj[fault_onset:ramp_end] = np.linspace(0, 0.8, ramp_end - fault_onset)
    fault_inj[ramp_end:] = 0.8 + 0.03 * np.sin(0.05 * np.arange(n_steps - ramp_end))

    # 真实故障类型 (0=健康, 1=正时, 2=泄漏, 3=燃油)
    fault_true = np.zeros(n_steps, dtype=int)
    fault_true[fault_onset:] = 1  # 正时故障

    # 诊断置信度
    diag_conf = np.zeros(n_steps)
    for i in range(diag_detect, n_steps):
        target_conf = min(0.95, 0.3 + 0.65 * fault_inj[i])
        diag_conf[i] = diag_conf[i - 1] + 0.15 * (target_conf - diag_conf[i - 1])
    diag_conf += rng.normal(0, 0.02, n_steps)
    diag_conf = np.clip(diag_conf, 0, 1)

    # 诊断预测
    diag_pred = np.zeros(n_steps, dtype=int)
    for i in range(diag_detect, n_steps):
        if diag_conf[i] > 0.5:
            diag_pred[i] = 1  # 正确预测
        elif rng.random() < 0.1:
            diag_pred[i] = rng.choice([0, 2, 3])  # 偶尔误判

    # 控制动作
    ctrl_timing = np.zeros(n_steps)
    ctrl_fuel = np.ones(n_steps)
    for i in range(ctrl_respond, n_steps):
        target_timing = fault_inj[i] * 4.5
        ctrl_timing[i] = ctrl_timing[i - 1] + 0.1 * (target_timing - ctrl_timing[i - 1])
        target_fuel = 1.0 + fault_inj[i] * 0.1
        ctrl_fuel[i] = ctrl_fuel[i - 1] + 0.08 * (target_fuel - ctrl_fuel[i - 1])
    ctrl_timing += rng.normal(0, 0.12, n_steps)
    ctrl_fuel += rng.normal(0, 0.006, n_steps)

    # Pmax 响应
    pmax_target = 150.0
    pmax = np.full(n_steps, pmax_target)
    fault_effect = -fault_inj * 20
    compensation = np.zeros(n_steps)
    for i in range(ctrl_respond + 2, n_steps):
        compensation[i] = compensation[i - 1] + 0.12 * (
            -fault_effect[i] - compensation[i - 1])
    pmax = pmax_target + fault_effect + compensation + rng.normal(0, 1.0, n_steps)

    return {
        'time': time,
        'fault_injected': fault_inj.tolist(),
        'fault_type_true': fault_true.tolist(),
        'diag_confidence': diag_conf.tolist(),
        'diag_predicted': diag_pred.tolist(),
        'ctrl_timing': ctrl_timing.tolist(),
        'ctrl_fuel': ctrl_fuel.tolist(),
        'pmax': pmax.tolist(),
        'pmax_target': pmax_target,
        'fault_onset': fault_onset,
        'diag_detect': diag_detect,
        'ctrl_respond': ctrl_respond,
    }


def _generate_reward_decomposition(n_episodes: int = 500, seed: int = 42) -> dict:
    """生成协同奖励分解数据

    Args:
        n_episodes: 训练轮次
        seed: 随机种子

    Returns:
        {'episodes': [...], '诊断奖励': [...], '控制奖励': [...], '协同奖励': [...]}
    """
    rng = np.random.RandomState(seed)
    episodes = list(range(1, n_episodes + 1))
    t = np.arange(1, n_episodes + 1) / n_episodes

    diag_r = (30 + 100 * (1 - np.exp(-3.0 * t))
              + rng.normal(0, 12, n_episodes) * (1 - 0.5 * t))
    ctrl_r = (20 + 120 * (1 - np.exp(-2.5 * t))
              + rng.normal(0, 15, n_episodes) * (1 - 0.4 * t))
    # 协同奖励: 前期几乎为零，后期逐渐增大
    coop_r = (5 * t ** 2 * 80 + rng.normal(0, 5, n_episodes) * (0.2 + 0.8 * t))
    coop_r = np.clip(coop_r, 0, None)

    return {
        'episodes': episodes,
        '诊断奖励': diag_r.tolist(),
        '控制奖励': ctrl_r.tolist(),
        '协同奖励': coop_r.tolist(),
    }


def _generate_fault_response_matrix(seed: int = 42) -> dict:
    """生成故障-响应矩阵数据

    Args:
        seed: 随机种子

    Returns:
        {'matrix': [[...]], 'fault_names': [...], 'action_names': [...]}
    """
    rng = np.random.RandomState(seed)

    fault_names = ['正时故障', '泄漏故障', '燃油故障', '复合故障']
    action_names = ['正时补偿', '燃油调整', '保护级别', '功率限制']

    # 每种故障有不同的主要控制策略
    matrix = np.array([
        [0.85, 0.15, 0.30, 0.10],   # 正时故障 → 主要调正时
        [0.20, 0.35, 0.70, 0.55],   # 泄漏故障 → 保护 + 功率限制
        [0.10, 0.80, 0.25, 0.15],   # 燃油故障 → 主要调燃油
        [0.55, 0.50, 0.65, 0.45],   # 复合故障 → 综合调整
    ]) + rng.normal(0, 0.03, (4, 4))
    matrix = np.clip(matrix, 0, 1)

    return {
        'matrix': matrix.tolist(),
        'fault_names': fault_names,
        'action_names': action_names,
    }


def _generate_ablation_study(seed: int = 42) -> dict:
    """生成消融实验数据

    Args:
        seed: 随机种子

    Returns:
        {方法名: {指标名: 值}}
    """
    return {
        '完整协同': {
            '诊断准确率': 0.923,
            '控制RMSE [bar]': 2.15,
            '综合奖励': 285.0,
            '约束违反率 (%)': 2.8,
        },
        '仅诊断\n(无控制调整)': {
            '诊断准确率': 0.910,
            '控制RMSE [bar]': 8.50,
            '综合奖励': 125.0,
            '约束违反率 (%)': 12.5,
        },
        '仅控制\n(无诊断信息)': {
            '诊断准确率': 0.000,
            '控制RMSE [bar]': 4.30,
            '综合奖励': 180.0,
            '约束违反率 (%)': 6.2,
        },
        '独立训练\n(无SharedCritic)': {
            '诊断准确率': 0.885,
            '控制RMSE [bar]': 3.10,
            '综合奖励': 240.0,
            '约束违反率 (%)': 4.1,
        },
    }


def _generate_pareto_front(n_points: int = 50, seed: int = 42) -> dict:
    """生成 Pareto 前沿数据

    Args:
        n_points: 实验点数
        seed: 随机种子

    Returns:
        {'diag_f1': [...], 'ctrl_rmse': [...], 'total_reward': [...],
         'is_pareto': [...], 'labels': [...]}
    """
    rng = np.random.RandomState(seed)

    # 随机生成实验点 (F1 高 → RMSE 高 的权衡)
    diag_f1 = rng.uniform(0.70, 0.98, n_points)
    # 反向关系: 高 F1 通常对应更高的 RMSE (更多资源给诊断)
    ctrl_rmse = 8.0 - 5.0 * diag_f1 + rng.normal(0, 0.8, n_points)
    ctrl_rmse = np.clip(ctrl_rmse, 0.5, 10)

    total_reward = 100 * diag_f1 + 50 * (10 - ctrl_rmse) / 10 + rng.normal(0, 5, n_points)

    # 寻找 Pareto 最优 (最大化 F1, 最小化 RMSE)
    is_pareto = np.zeros(n_points, dtype=bool)
    for i in range(n_points):
        dominated = False
        for j in range(n_points):
            if i == j:
                continue
            if diag_f1[j] >= diag_f1[i] and ctrl_rmse[j] <= ctrl_rmse[i]:
                if diag_f1[j] > diag_f1[i] or ctrl_rmse[j] < ctrl_rmse[i]:
                    dominated = True
                    break
        if not dominated:
            is_pareto[i] = True

    labels = [f'exp_{i}' for i in range(n_points)]

    return {
        'diag_f1': diag_f1.tolist(),
        'ctrl_rmse': ctrl_rmse.tolist(),
        'total_reward': total_reward.tolist(),
        'is_pareto': is_pareto.tolist(),
        'labels': labels,
    }


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

    # ── 2. results.json（合并已有内容 + 全部新字段）────────────────────
    print("[2/3] 更新实验结果 results.json ...")
    results_path = os.path.join(out_dir, 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = {}

    # 基础数据
    results['confusion_matrix'] = _generate_confusion_matrix()
    results['detection_delays'] = _generate_detection_delays()
    results['control_response'] = _generate_control_response()

    # 诊断智能体评价
    results['roc_data'] = _generate_roc_data()
    results['pr_data'] = _generate_pr_data()
    results['severity_sensitivity'] = _generate_severity_sensitivity()
    results['tsne_embeddings'] = _generate_tsne_embeddings()
    results['online_accuracy'] = _generate_online_accuracy()

    # 控制智能体评价
    results['tracking_data'] = _generate_tracking_data()
    results['action_distribution'] = _generate_action_distribution()
    results['robustness_data'] = _generate_robustness_data()
    results['constraint_satisfaction'] = _generate_constraint_satisfaction()
    results['energy_efficiency'] = _generate_energy_efficiency()

    # 控诊协同评价
    results['collaborative_timeline'] = _generate_collaborative_timeline()
    results['reward_decomposition'] = _generate_reward_decomposition()
    results['fault_response_matrix'] = _generate_fault_response_matrix()
    results['ablation_study'] = _generate_ablation_study()
    results['pareto_front'] = _generate_pareto_front()

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {results_path}  (含全部 18 个数据字段)")

    # ── 3. comparison_results.csv ─────────────────────────────────────
    print("[3/3] 生成方法对比 comparison_results.csv ...")
    df_comp = _generate_comparison_results()
    comp_path = os.path.join(out_dir, 'comparison_results.csv')
    df_comp.to_csv(comp_path, index=False)
    print(f"  ✓ {comp_path}  ({len(df_comp)} methods)")

    print("\n全部 mock 数据生成完成！")


if __name__ == '__main__':
    main()
