#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三算法对比评估脚本
==================

对比 MAPPO、QMIX、Independent 三种训练方法的性能

Author: CDC Project
Date: 2026-01-24
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.dual_agent_env import create_dual_agent_env
from agents.multi_agent_algorithms import get_multi_agent_algorithm, MAPPO, QMIX
from agents.rl_diagnosis_agent import RLDiagnosisAgent, create_rl_diagnosis_agent


def evaluate_model(model_dir: str, method: str, n_episodes: int = 50):
    """
    评估单个模型
    
    Args:
        model_dir: 模型目录
        method: 方法名称 ('mappo', 'qmix', 'independent')
        n_episodes: 评估回合数
    
    Returns:
        dict: 评估指标
    """
    print(f"\n{'='*60}")
    print(f"评估 {method.upper()} 模型: {model_dir}")
    print(f"{'='*60}")
    
    # 检查模型文件
    final_dir = os.path.join(model_dir, 'final')
    if not os.path.exists(final_dir):
        print(f"[错误] 找不到模型目录: {final_dir}")
        return None
    
    # 创建环境
    env = create_dual_agent_env()
    
    # 加载模型
    if method in ['mappo', 'qmix']:
        # 多智能体算法
        # 参数顺序: name, diag_state_dim, diag_action_dim, ctrl_state_dim, ctrl_action_dim
        ma_agent = get_multi_agent_algorithm(
            method.upper(),
            env.diag_state_dim,
            env.diag_action_dim,
            env.ctrl_state_dim,
            env.ctrl_action_dim
        )
        ma_path = os.path.join(final_dir, 'ma_agent.pt')
        if os.path.exists(ma_path):
            ma_agent.load(ma_path)
            print(f"[加载] 多智能体模型: {ma_path}")
        else:
            print(f"[错误] 找不到模型: {ma_path}")
            return None
        
        def select_actions(diag_obs, diag_residual_seq, ctrl_obs):
            """多智能体选择动作"""
            diag_action, ctrl_action, _ = ma_agent.select_actions(
                diag_obs, diag_residual_seq, ctrl_obs, explore=False
            )
            return diag_action, ctrl_action
    else:
        # 独立学习
        diag_agent = create_rl_diagnosis_agent()
        diag_path = os.path.join(final_dir, 'diag_agent.pt')
        if os.path.exists(diag_path):
            diag_agent.load(diag_path)
            print(f"[加载] 诊断智能体: {diag_path}")
        else:
            print(f"[错误] 找不到模型: {diag_path}")
            return None
        
        def select_actions(diag_obs, diag_residual_seq, ctrl_obs):
            """独立学习选择动作"""
            diag_action = diag_agent.select_action(diag_obs, diag_residual_seq, explore=False)
            ctrl_action = np.random.randint(0, env.ctrl_action_dim)  # 简化控制
            return diag_action, ctrl_action
    
    # 评估循环
    all_rewards = []
    all_diag_acc = []
    all_detection_delays = []
    all_violations = []
    fault_type_acc = {}  # 按故障类型统计准确率
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        # 正确解包观测对象
        diag_obs = obs.diag_state
        diag_residual_seq = obs.diag_residual_seq
        ctrl_obs = obs.ctrl_state
        
        episode_reward = 0
        correct_diagnoses = 0
        total_steps = 0
        first_correct_step = -1
        violations = 0
        ground_truth = info.ground_truth_fault.name if hasattr(info, 'ground_truth_fault') else 'NONE'
        
        done = False
        while not done:
            # 选择动作 (所有方法统一使用三个参数)
            diag_action, ctrl_action = select_actions(diag_obs, diag_residual_seq, ctrl_obs)
            
            # 执行动作 (step返回: obs, diag_r, ctrl_r, done, info)
            obs, diag_r, ctrl_r, done, info = env.step(diag_action, ctrl_action)
            diag_obs = obs.diag_state
            diag_residual_seq = obs.diag_residual_seq
            ctrl_obs = obs.ctrl_state
            
            # 记录
            episode_reward += diag_r + ctrl_r
            
            # 比较预测和实际故障
            predicted_action = diag_action
            actual_fault = info.ground_truth_fault if hasattr(info, 'ground_truth_fault') else None
            
            # 诊断动作解码: action = fault_type * n_conf_levels + conf_level
            predicted_fault_idx = predicted_action // 4  # 假设4个置信度等级
            actual_fault_idx = env.FAULT_TYPES.index(actual_fault) if actual_fault in env.FAULT_TYPES else 0
            
            if predicted_fault_idx == actual_fault_idx:
                correct_diagnoses += 1
                if first_correct_step < 0:
                    first_correct_step = total_steps
            
            if hasattr(info, 'Pmax_violation') and info.Pmax_violation:
                violations += 1
            
            total_steps += 1
        
        # 记录回合统计
        diag_acc = correct_diagnoses / max(total_steps, 1)
        all_rewards.append(episode_reward)
        all_diag_acc.append(diag_acc)
        all_detection_delays.append(first_correct_step if first_correct_step >= 0 else total_steps)
        all_violations.append(violations)
        
        # 按故障类型统计
        if ground_truth not in fault_type_acc:
            fault_type_acc[ground_truth] = []
        fault_type_acc[ground_truth].append(diag_acc)
        
        if (ep + 1) % 10 == 0:
            print(f"  评估进度: {ep+1}/{n_episodes} | 当前准确率: {diag_acc:.1%}")
    
    # 汇总结果
    results = {
        'method': method,
        'n_episodes': n_episodes,
        'avg_reward': float(np.mean(all_rewards)),
        'std_reward': float(np.std(all_rewards)),
        'avg_diag_accuracy': float(np.mean(all_diag_acc)),
        'std_diag_accuracy': float(np.std(all_diag_acc)),
        'avg_detection_delay': float(np.mean(all_detection_delays)),
        'avg_violations': float(np.mean(all_violations)),
        'total_violations': int(np.sum(all_violations)),
        'fault_type_accuracy': {k: float(np.mean(v)) for k, v in fault_type_acc.items()},
    }
    
    print(f"\n结果汇总:")
    print(f"  平均奖励: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  诊断准确率: {results['avg_diag_accuracy']:.1%} ± {results['std_diag_accuracy']:.1%}")
    print(f"  平均检测延迟: {results['avg_detection_delay']:.1f} 步")
    print(f"  总违规次数: {results['total_violations']}")
    
    return results


def compare_methods(results_list: list, output_dir: str = 'results/comparison'):
    """
    生成对比报告和图表
    
    Args:
        results_list: 评估结果列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("生成对比报告")
    print(f"{'='*60}")
    
    # 提取数据
    methods = [r['method'] for r in results_list]
    rewards = [r['avg_reward'] for r in results_list]
    reward_stds = [r['std_reward'] for r in results_list]
    accuracies = [r['avg_diag_accuracy'] * 100 for r in results_list]
    acc_stds = [r['std_diag_accuracy'] * 100 for r in results_list]
    delays = [r['avg_detection_delay'] for r in results_list]
    violations = [r['total_violations'] for r in results_list]
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建对比图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 平均奖励对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(methods, rewards, yerr=reward_stds, capsize=5, 
                    color=['#3498db', '#e74c3c', '#2ecc71'])
    ax1.set_title('平均奖励对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('平均奖励')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars1, rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 2. 诊断准确率对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(methods, accuracies, yerr=acc_stds, capsize=5,
                    color=['#3498db', '#e74c3c', '#2ecc71'])
    ax2.set_title('诊断准确率对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('准确率 (%)')
    ax2.set_ylim(0, 100)
    for bar, val in zip(bars2, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 3. 检测延迟对比
    ax3 = axes[1, 0]
    bars3 = ax3.bar(methods, delays, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax3.set_title('平均检测延迟对比', fontsize=14, fontweight='bold')
    ax3.set_ylabel('延迟 (步)')
    for bar, val in zip(bars3, delays):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 4. 违规次数对比
    ax4 = axes[1, 1]
    bars4 = ax4.bar(methods, violations, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax4.set_title('总违规次数对比', fontsize=14, fontweight='bold')
    ax4.set_ylabel('违规次数')
    for bar, val in zip(bars4, violations):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'method_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"[保存] 对比图表: {fig_path}")
    plt.close()
    
    # 生成雷达图
    fig2, ax5 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # 归一化指标 (越高越好)
    max_reward = max(max(rewards), 1)
    max_acc = 100
    max_delay = max(max(delays), 1)
    max_vio = max(max(violations), 1)
    
    categories = ['奖励', '准确率', '响应速度\n(1/延迟)', '安全性\n(无违规)']
    
    for i, r in enumerate(results_list):
        values = [
            (r['avg_reward'] - min(rewards)) / (max(rewards) - min(rewards) + 1e-6) * 100,
            r['avg_diag_accuracy'] * 100,
            (1 - r['avg_detection_delay'] / max_delay) * 100 if max_delay > 0 else 0,
            (1 - r['total_violations'] / max(max_vio, 1)) * 100,
        ]
        values.append(values[0])  # 闭合
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles.append(angles[0])
        
        ax5.plot(angles, values, 'o-', linewidth=2, label=r['method'].upper())
        ax5.fill(angles, values, alpha=0.2)
    
    ax5.set_xticks(np.linspace(0, 2 * np.pi, len(categories), endpoint=False))
    ax5.set_xticklabels(categories, fontsize=11)
    ax5.set_title('综合性能雷达图', fontsize=14, fontweight='bold', pad=20)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15))
    
    radar_path = os.path.join(output_dir, 'radar_comparison.png')
    plt.savefig(radar_path, dpi=150, bbox_inches='tight')
    print(f"[保存] 雷达图: {radar_path}")
    plt.close()
    
    # 保存 JSON 结果
    json_path = os.path.join(output_dir, 'comparison_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results_list
        }, f, indent=2, ensure_ascii=False)
    print(f"[保存] JSON 结果: {json_path}")
    
    # 生成文本报告
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("双智能体强化学习系统 - 三算法对比报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        f.write("一、性能总览\n")
        f.write("-"*50 + "\n")
        f.write(f"{'方法':<15} {'平均奖励':>12} {'诊断准确率':>12} {'检测延迟':>10} {'违规次数':>10}\n")
        f.write("-"*50 + "\n")
        for r in results_list:
            f.write(f"{r['method'].upper():<15} {r['avg_reward']:>12.2f} {r['avg_diag_accuracy']*100:>11.1f}% {r['avg_detection_delay']:>10.1f} {r['total_violations']:>10}\n")
        f.write("-"*50 + "\n\n")
        
        # 找出最佳方法
        best_reward = max(results_list, key=lambda x: x['avg_reward'])
        best_acc = max(results_list, key=lambda x: x['avg_diag_accuracy'])
        best_delay = min(results_list, key=lambda x: x['avg_detection_delay'])
        best_safety = min(results_list, key=lambda x: x['total_violations'])
        
        f.write("二、最佳表现\n")
        f.write("-"*50 + "\n")
        f.write(f"最高平均奖励: {best_reward['method'].upper()} ({best_reward['avg_reward']:.2f})\n")
        f.write(f"最高诊断准确率: {best_acc['method'].upper()} ({best_acc['avg_diag_accuracy']*100:.1f}%)\n")
        f.write(f"最快检测响应: {best_delay['method'].upper()} ({best_delay['avg_detection_delay']:.1f} 步)\n")
        f.write(f"最佳安全性: {best_safety['method'].upper()} ({best_safety['total_violations']} 违规)\n")
        f.write("\n")
        
        f.write("三、结论\n")
        f.write("-"*50 + "\n")
        if best_reward['method'] == best_acc['method']:
            f.write(f"综合最佳: {best_reward['method'].upper()} 在奖励和准确率上都表现最佳\n")
        else:
            f.write(f"奖励最佳: {best_reward['method'].upper()}\n")
            f.write(f"准确率最佳: {best_acc['method'].upper()}\n")
        
        f.write("\n四、按故障类型分析\n")
        f.write("-"*50 + "\n")
        for r in results_list:
            f.write(f"\n{r['method'].upper()}:\n")
            for fault, acc in r.get('fault_type_accuracy', {}).items():
                f.write(f"  - {fault}: {acc*100:.1f}%\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("报告生成完毕\n")
    
    print(f"[保存] 文本报告: {report_path}")
    
    return {
        'best_reward': best_reward['method'],
        'best_accuracy': best_acc['method'],
        'best_speed': best_delay['method'],
        'best_safety': best_safety['method']
    }


def main():
    """主函数"""
    print("\n" + "="*70)
    print("双智能体强化学习系统 - 三算法对比评估")
    print("="*70)
    
    # 定义要评估的模型
    models = [
        ('models/mappo_200', 'mappo'),
        ('models/qmix_200', 'qmix'),
        ('models/independent_200', 'independent'),
    ]
    
    # 评估每个模型
    results = []
    for model_dir, method in models:
        if os.path.exists(model_dir):
            result = evaluate_model(model_dir, method, n_episodes=30)
            if result:
                results.append(result)
        else:
            print(f"[跳过] 模型目录不存在: {model_dir}")
    
    if len(results) >= 2:
        # 生成对比报告
        best = compare_methods(results, 'results/comparison')
        
        print("\n" + "="*70)
        print("评估完成!")
        print("="*70)
        print(f"最佳平均奖励: {best['best_reward'].upper()}")
        print(f"最佳诊断准确率: {best['best_accuracy'].upper()}")
        print(f"最快检测响应: {best['best_speed'].upper()}")
        print(f"最佳安全性: {best['best_safety'].upper()}")
        print("\n结果已保存至: results/comparison/")
    else:
        print("[错误] 需要至少 2 个模型才能进行对比")


if __name__ == "__main__":
    main()
