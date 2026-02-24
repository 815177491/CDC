#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MARL 可视化运行脚本
====================
生成 MARL 双智能体强化学习相关的所有学术风格图片（共 30 张）

图片分类:
- 基础 (7张): 架构 / 训练曲线 / 奖励分布 / 混淆矩阵 / 检测延迟 / 控制响应 / 方法对比
- 诊断评价 (5张): ROC / PR / 严重程度敏感性 / t-SNE / 在线准确率
- 控制评价 (5张): 多工况跟踪 / 动作分布 / 鲁棒性 / 约束满足 / 效率
- 协同评价 (6张): 时序图 / 奖励分解 / 故障-响应矩阵 / 消融 / 信息流 / Pareto
- 新增 (1张): 训练过程动态相图
- 理论原理图 (6张): 奖励结构 / 故障注入 / PIKAN架构 / 控制+Critic / MAPPO流程 / 课程学习

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
import matplotlib.pyplot as plt

# 本项目模块
from config import PATH_CONFIG
from visualization.marl_plots import (
    # 基础 7 张
    plot_training_curves,
    plot_reward_distribution,
    plot_confusion_matrix,
    plot_detection_delay,
    plot_control_response,
    plot_method_comparison,
    plot_dual_agent_architecture,
    # 诊断智能体评价 5 张
    plot_diagnostic_roc_curves,
    plot_diagnostic_precision_recall,
    plot_fault_severity_sensitivity,
    plot_diagnostic_embedding_tsne,
    plot_diagnostic_online_accuracy,
    # 控制智能体评价 5 张
    plot_control_multi_setpoint_tracking,
    plot_control_action_distribution,
    plot_control_robustness_envelope,
    plot_control_constraint_satisfaction,
    plot_control_energy_efficiency,
    # 控诊协同评价 6 张
    plot_collaborative_timeline,
    plot_collaborative_reward_decomposition,
    plot_collaborative_fault_response_matrix,
    plot_collaborative_ablation_study,
    plot_collaborative_information_flow,
    plot_collaborative_pareto_front,
    # 新增高冲击力图表
    plot_training_phase_diagram,
    # 理论原理示意图 6 张
    plot_reward_structure_schematic,
    plot_fault_injection_schematic,
    plot_pikan_architecture,
    plot_control_critic_architecture,
    plot_mappo_training_flow,
    plot_curriculum_schedule,
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
    """主函数：生成所有 MARL 相关可视化图表（共 23 张）"""
    print("=" * 70)
    print("MARL 双智能体可视化图片生成脚本（24 张）")
    print("=" * 70)
    print(f"工作目录: {PROJECT_ROOT}")
    print(f"训练图输出目录: {PATH_CONFIG.VIS_TRAINING_DIR}")
    print(f"架构图输出目录: {PATH_CONFIG.VIS_MODELING_DIR}")
    print("=" * 70)

    results = {}
    total = 30
    idx = 0

    def _step(name: str):
        nonlocal idx
        idx += 1
        return f"[{idx}/{total}]"

    def _close():
        """关闭所有已打开的图形，释放内存"""
        plt.close('all')

    # ══════════════════════════════════════════════════════════════════
    # 基础 7 张
    # ══════════════════════════════════════════════════════════════════

    # ── 1. 双智能体网络架构示意图（category='modeling'）────────────────
    try:
        print(f"\n{_step('arch')} 生成双智能体网络架构图...")
        plot_dual_agent_architecture()
        results['dual_agent_architecture'] = 'OK'
        print("  ✓ 完成")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['dual_agent_architecture'] = f'FAILED: {e}'

    # ── 2. 训练曲线 ───────────────────────────────────────────────────
    try:
        print(f"\n{_step('tc')} 生成训练曲线...")
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
        print(f"\n{_step('rd')} 生成奖励分布图...")
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
        print(f"\n{_step('cm')} 生成故障诊断混淆矩阵...")
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
        print(f"\n{_step('dd')} 生成故障检测延迟箱型图...")
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
        print(f"\n{_step('cr')} 生成容错控制响应图...")
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
        print(f"\n{_step('mc')} 生成方法性能对比图...")
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

    _close()  # 释放前 7 张图的内存

    # ══════════════════════════════════════════════════════════════════
    # 诊断智能体评价 5 张
    # ══════════════════════════════════════════════════════════════════

    exp_results = _load_experiment_results()

    # ── 8. ROC 曲线 ──────────────────────────────────────────────────
    try:
        print(f"\n{_step('roc')} 生成多故障类型 ROC 曲线...")
        roc = exp_results.get('roc_data')
        if roc is not None:
            plot_diagnostic_roc_curves(roc)
            results['diagnostic_roc_curves'] = 'OK'
            print("  ✓ 完成")
        else:
            results['diagnostic_roc_curves'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['diagnostic_roc_curves'] = f'FAILED: {e}'

    # ── 9. PR 曲线 ───────────────────────────────────────────────────
    try:
        print(f"\n{_step('pr')} 生成精确率-召回率曲线...")
        pr = exp_results.get('pr_data')
        if pr is not None:
            plot_diagnostic_precision_recall(pr)
            results['diagnostic_precision_recall'] = 'OK'
            print("  ✓ 完成")
        else:
            results['diagnostic_precision_recall'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['diagnostic_precision_recall'] = f'FAILED: {e}'

    # ── 10. 故障严重程度敏感性 ────────────────────────────────────────
    try:
        print(f"\n{_step('sev')} 生成故障严重程度敏感性分析...")
        sev = exp_results.get('severity_sensitivity')
        if sev is not None:
            plot_fault_severity_sensitivity(sev)
            results['fault_severity_sensitivity'] = 'OK'
            print("  ✓ 完成")
        else:
            results['fault_severity_sensitivity'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['fault_severity_sensitivity'] = f'FAILED: {e}'

    # ── 11. t-SNE 可视化 ─────────────────────────────────────────────
    try:
        print(f"\n{_step('tsne')} 生成诊断特征空间 t-SNE 可视化...")
        tsne = exp_results.get('tsne_embeddings')
        if tsne is not None:
            embeddings_2d = np.array(tsne['embeddings_2d'])
            labels = np.array(tsne['labels'])
            plot_diagnostic_embedding_tsne(embeddings_2d, labels)
            results['diagnostic_embedding_tsne'] = 'OK'
            print("  ✓ 完成")
        else:
            results['diagnostic_embedding_tsne'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['diagnostic_embedding_tsne'] = f'FAILED: {e}'

    # ── 12. 在线诊断准确率 ───────────────────────────────────────────
    try:
        print(f"\n{_step('onl')} 生成在线诊断准确率演化图...")
        online = exp_results.get('online_accuracy')
        if online is not None:
            plot_diagnostic_online_accuracy(online)
            results['diagnostic_online_accuracy'] = 'OK'
            print("  ✓ 完成")
        else:
            results['diagnostic_online_accuracy'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['diagnostic_online_accuracy'] = f'FAILED: {e}'

    _close()  # 释放诊断评价图内存

    # ══════════════════════════════════════════════════════════════════
    # 控制智能体评价 5 张
    # ══════════════════════════════════════════════════════════════════

    # ── 13. 多工况设定值跟踪 ──────────────────────────────────────────
    try:
        print(f"\n{_step('trk')} 生成多工况设定值跟踪图...")
        trk = exp_results.get('tracking_data')
        if trk is not None:
            plot_control_multi_setpoint_tracking(trk)
            results['control_multi_setpoint_tracking'] = 'OK'
            print("  ✓ 完成")
        else:
            results['control_multi_setpoint_tracking'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['control_multi_setpoint_tracking'] = f'FAILED: {e}'

    # ── 14. 控制动作分布 ──────────────────────────────────────────────
    try:
        print(f"\n{_step('act')} 生成控制动作分布图...")
        act = exp_results.get('action_distribution')
        if act is not None:
            plot_control_action_distribution(act)
            results['control_action_distribution'] = 'OK'
            print("  ✓ 完成")
        else:
            results['control_action_distribution'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['control_action_distribution'] = f'FAILED: {e}'

    # ── 15. 控制鲁棒性包络 ────────────────────────────────────────────
    try:
        print(f"\n{_step('rob')} 生成控制鲁棒性包络图...")
        rob = exp_results.get('robustness_data')
        if rob is not None:
            plot_control_robustness_envelope(rob)
            results['control_robustness_envelope'] = 'OK'
            print("  ✓ 完成")
        else:
            results['control_robustness_envelope'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['control_robustness_envelope'] = f'FAILED: {e}'

    # ── 16. 约束满足率雷达图 ──────────────────────────────────────────
    try:
        print(f"\n{_step('cst')} 生成约束满足率雷达图...")
        cst = exp_results.get('constraint_satisfaction')
        if cst is not None:
            plot_control_constraint_satisfaction(cst)
            results['control_constraint_satisfaction'] = 'OK'
            print("  ✓ 完成")
        else:
            results['control_constraint_satisfaction'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['control_constraint_satisfaction'] = f'FAILED: {e}'

    # ── 17. 控制效率分析 ──────────────────────────────────────────────
    try:
        print(f"\n{_step('eff')} 生成控制效率分析图...")
        eff = exp_results.get('energy_efficiency')
        if eff is not None:
            plot_control_energy_efficiency(eff)
            results['control_energy_efficiency'] = 'OK'
            print("  ✓ 完成")
        else:
            results['control_energy_efficiency'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['control_energy_efficiency'] = f'FAILED: {e}'

    _close()  # 释放控制评价图内存

    # ══════════════════════════════════════════════════════════════════
    # 控诊协同评价 6 张
    # ══════════════════════════════════════════════════════════════════

    # ── 18. 控诊协同时序图 ⭐ ─────────────────────────────────────────
    try:
        print(f"\n{_step('tl')} 生成控诊协同时序图 ⭐...")
        tl = exp_results.get('collaborative_timeline')
        if tl is not None:
            plot_collaborative_timeline(tl)
            results['collaborative_timeline'] = 'OK'
            print("  ✓ 完成")
        else:
            results['collaborative_timeline'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['collaborative_timeline'] = f'FAILED: {e}'

    # ── 19. 协同奖励分解 ──────────────────────────────────────────────
    try:
        print(f"\n{_step('rd2')} 生成协同奖励分解图...")
        rd2 = exp_results.get('reward_decomposition')
        if rd2 is not None:
            plot_collaborative_reward_decomposition(rd2)
            results['collaborative_reward_decomposition'] = 'OK'
            print("  ✓ 完成")
        else:
            results['collaborative_reward_decomposition'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['collaborative_reward_decomposition'] = f'FAILED: {e}'

    # ── 20. 故障-响应矩阵 ─────────────────────────────────────────────
    try:
        print(f"\n{_step('frm')} 生成故障-响应矩阵热力图...")
        frm = exp_results.get('fault_response_matrix')
        if frm is not None:
            matrix = np.array(frm['matrix'])
            plot_collaborative_fault_response_matrix(
                matrix, frm['fault_names'], frm['action_names'])
            results['collaborative_fault_response_matrix'] = 'OK'
            print("  ✓ 完成")
        else:
            results['collaborative_fault_response_matrix'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['collaborative_fault_response_matrix'] = f'FAILED: {e}'

    # ── 21. 消融实验对比 ──────────────────────────────────────────────
    try:
        print(f"\n{_step('abl')} 生成协同消融实验对比图...")
        abl = exp_results.get('ablation_study')
        if abl is not None:
            plot_collaborative_ablation_study(abl)
            results['collaborative_ablation_study'] = 'OK'
            print("  ✓ 完成")
        else:
            results['collaborative_ablation_study'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['collaborative_ablation_study'] = f'FAILED: {e}'

    # ── 22. 信息流示意图 ──────────────────────────────────────────────
    try:
        print(f"\n{_step('inf')} 生成双智能体信息流示意图...")
        plot_collaborative_information_flow()
        results['collaborative_information_flow'] = 'OK'
        print("  ✓ 完成")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['collaborative_information_flow'] = f'FAILED: {e}'

    # ── 23. Pareto 前沿 ───────────────────────────────────────────────
    try:
        print(f"\n{_step('par')} 生成诊断-控制 Pareto 前沿图...")
        par = exp_results.get('pareto_front')
        if par is not None:
            plot_collaborative_pareto_front(par)
            results['collaborative_pareto_front'] = 'OK'
            print("  ✓ 完成")
        else:
            results['collaborative_pareto_front'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['collaborative_pareto_front'] = f'FAILED: {e}'

    # ── 24. 训练过程动态相图 ⭐ ───────────────────────────────────────
    try:
        print(f"\n{_step('phase')} 生成训练过程动态相图 ⭐...")
        history = _load_training_log()
        if history and 'diag_accuracy' in history and 'ctrl_performance' in history:
            plot_training_phase_diagram(history)
            results['training_phase_diagram'] = 'OK'
            print("  ✓ 完成")
        else:
            results['training_phase_diagram'] = 'SKIPPED (no data)'
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['training_phase_diagram'] = f'FAILED: {e}'

    _close()  # 释放协同评价图内存

    # ══════════════════════════════════════════════════════════════════
    # 理论原理示意图 6 张（category='modeling'）
    # ══════════════════════════════════════════════════════════════════

    # ── 25. 奖励函数结构示意图 ────────────────────────────────────────
    try:
        print(f"\n{_step('rew_sch')} 生成奖励函数结构示意图...")
        plot_reward_structure_schematic()
        results['reward_structure_schematic'] = 'OK'
        print("  ✓ 完成")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['reward_structure_schematic'] = f'FAILED: {e}'

    # ── 26. 故障注入与变工况调度示意图 ────────────────────────────────
    try:
        print(f"\n{_step('fi_sch')} 生成故障注入与变工况调度示意图...")
        plot_fault_injection_schematic()
        results['fault_injection_schematic'] = 'OK'
        print("  ✓ 完成")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['fault_injection_schematic'] = f'FAILED: {e}'

    # ── 27. PIKAN 网络结构示意图 ──────────────────────────────────────
    try:
        print(f"\n{_step('pikan')} 生成 PIKAN 网络结构示意图...")
        plot_pikan_architecture()
        results['pikan_architecture'] = 'OK'
        print("  ✓ 完成")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['pikan_architecture'] = f'FAILED: {e}'

    # ── 28. 控制网络与共享Critic结构 ──────────────────────────────────
    try:
        print(f"\n{_step('cc_arch')} 生成控制网络与共享 Critic 结构图...")
        plot_control_critic_architecture()
        results['control_critic_architecture'] = 'OK'
        print("  ✓ 完成")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['control_critic_architecture'] = f'FAILED: {e}'

    # ── 29. MAPPO 训练流程图 ──────────────────────────────────────────
    try:
        print(f"\n{_step('mappo')} 生成 MAPPO 训练流程图...")
        plot_mappo_training_flow()
        results['mappo_training_flow'] = 'OK'
        print("  ✓ 完成")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['mappo_training_flow'] = f'FAILED: {e}'

    # ── 30. 课程学习难度调度图 ────────────────────────────────────────
    try:
        print(f"\n{_step('curri')} 生成课程学习难度调度示意图...")
        plot_curriculum_schedule()
        results['curriculum_schedule'] = 'OK'
        print("  ✓ 完成")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results['curriculum_schedule'] = f'FAILED: {e}'

    _close()  # 释放理论图内存

    # ══════════════════════════════════════════════════════════════════
    # 结果汇总
    # ══════════════════════════════════════════════════════════════════
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
    print(f"\n合计: {ok_count} 成功, {skip_count} 跳过, {fail_count} 失败 / 共 {len(results)} 张")
    print(f"训练图目录: {PATH_CONFIG.VIS_TRAINING_DIR}")
    print(f"架构图目录: {PATH_CONFIG.VIS_MODELING_DIR}")

    return results


if __name__ == '__main__':
    main()
