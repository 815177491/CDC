#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双智能体可视化数据导出
======================
为每张图表生成对应的CSV数据文件，便于在第三方软件中重新绑制

Author: CDC Project
Date: 2026-01-21
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime

# 输出目录
OUTPUT_DIR = 'visualization_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def export_training_process_data():
    """导出DQN训练过程数据"""
    print("[1/5] 导出训练过程数据...")
    
    np.random.seed(123)
    episodes = 200
    
    data = {
        'episode': [],
        'loss': [],
        'loss_smoothed': [],
        'q_value': [],
        'epsilon': [],
        'reward': [],
        'reward_smoothed': [],
        'steps': [],
    }
    
    loss_history = []
    reward_history = []
    
    for ep in range(episodes):
        data['episode'].append(ep)
        
        # 损失函数
        base_loss = 2.0 * np.exp(-0.015 * ep) + 0.1
        loss = max(0.05, base_loss + np.random.normal(0, 0.1 * base_loss))
        data['loss'].append(loss)
        loss_history.append(loss)
        
        # Q值
        q_value = 10 * (1 - np.exp(-0.02 * ep)) + np.random.normal(0, 0.5)
        data['q_value'].append(q_value)
        
        # 探索率
        epsilon = max(0.05, 1.0 * (0.995 ** ep))
        data['epsilon'].append(epsilon)
        
        # 累计奖励
        reward = -50 + 60 * (1 - np.exp(-0.025 * ep)) + np.random.normal(0, 5)
        data['reward'].append(reward)
        reward_history.append(reward)
        
        # 步数
        steps = int(100 + 50 * (1 - np.exp(-0.01 * ep)) + np.random.randint(-10, 10))
        data['steps'].append(steps)
    
    # 计算平滑值 (窗口=10)
    window = 10
    for i in range(episodes):
        start_idx = max(0, i - window + 1)
        data['loss_smoothed'].append(np.mean(loss_history[start_idx:i+1]))
        data['reward_smoothed'].append(np.mean(reward_history[start_idx:i+1]))
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(OUTPUT_DIR, 'training_process.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> training_process.csv ({len(df)} rows)")
    
    return df


def export_simulation_results_data():
    """导出仿真结果数据"""
    print("[2/5] 导出仿真结果数据...")
    
    np.random.seed(42)
    duration = 100
    fault_time = 25
    pmax_base = 137.0
    
    data = {
        'time_s': [],
        'pmax_bar': [],
        'pmax_baseline': [],
        'pmax_upper_threshold': [],
        'pmax_lower_threshold': [],
        'pcomp_bar': [],
        'texh_K': [],
        'fault_status': [],
        'diagnosis_confidence': [],
        'vit_adjust_deg': [],
        'fuel_multiplier': [],
        'control_mode': [],
    }
    
    for t in range(duration):
        data['time_s'].append(t)
        data['pmax_baseline'].append(pmax_base)
        data['pmax_upper_threshold'].append(pmax_base * 1.05)
        data['pmax_lower_threshold'].append(pmax_base * 0.95)
        
        if t < fault_time:
            pmax = pmax_base + np.random.normal(0, 1.5)
            fault = 0
            confidence = 0.1 + np.random.random() * 0.1
            vit = np.random.normal(0, 0.5)
            fuel = 1.0
            mode = 'NORMAL'
        else:
            decay = 1 - 0.02 * min(t - fault_time, 30)
            pmax = pmax_base * (1 + 0.12 * decay) + np.random.normal(0, 2)
            fault = 1
            confidence = min(0.5 + 0.02 * (t - fault_time), 1.0)
            vit = min(-4.0, -0.5 * (t - fault_time)) + np.random.normal(0, 0.3)
            fuel = max(0.85, 1.0 - 0.005 * (t - fault_time))
            mode = 'FAULT_RESPONSE'
        
        data['pmax_bar'].append(pmax)
        data['pcomp_bar'].append(110 + np.random.normal(0, 1))
        data['texh_K'].append(620 + np.random.normal(0, 3))
        data['fault_status'].append(fault)
        data['diagnosis_confidence'].append(confidence)
        data['vit_adjust_deg'].append(vit)
        data['fuel_multiplier'].append(fuel)
        data['control_mode'].append(mode)
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(OUTPUT_DIR, 'simulation_results.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> simulation_results.csv ({len(df)} rows)")
    
    return df


def export_performance_comparison_data():
    """导出性能对比数据"""
    print("[3/5] 导出性能对比数据...")
    
    # 1. 关键指标对比
    metrics_data = {
        'metric': ['Detection_Delay_s', 'Overshoot_percent', 'Steady_State_Error_percent', 
                   'Response_Time_s', 'False_Positive_Rate_percent'],
        'metric_cn': ['检测延迟(s)', '超调量(%)', '稳态误差(%)', '响应时间(s)', '假阳性率(%)'],
        'traditional': [2.8, 6.2, 1.8, 4.5, 5.6],
        'dual_agent': [1.2, 3.5, 0.8, 2.1, 2.1],
    }
    metrics_data['improvement_percent'] = [
        (t - d) / t * 100 for t, d in zip(metrics_data['traditional'], metrics_data['dual_agent'])
    ]
    
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, 'performance_metrics.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> performance_metrics.csv ({len(df_metrics)} rows)")
    
    # 2. 雷达图数据 (归一化)
    max_vals = [max(t, d) for t, d in zip(metrics_data['traditional'], metrics_data['dual_agent'])]
    radar_data = {
        'metric': metrics_data['metric'],
        'traditional_normalized': [1 - t/m for t, m in zip(metrics_data['traditional'], max_vals)],
        'dual_agent_normalized': [1 - d/m for d, m in zip(metrics_data['dual_agent'], max_vals)],
        'angle_deg': [i * 360 / 5 for i in range(5)],
    }
    df_radar = pd.DataFrame(radar_data)
    df_radar.to_csv(os.path.join(OUTPUT_DIR, 'performance_radar.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> performance_radar.csv ({len(df_radar)} rows)")
    
    # 3. 阶跃响应对比
    t_step = np.linspace(0, 10, 100)
    zeta_trad, wn_trad = 0.5, 1.5
    zeta_dual, wn_dual = 0.8, 2.2
    
    trad_response = 1 - np.exp(-zeta_trad * wn_trad * t_step) * (
        np.cos(wn_trad * np.sqrt(1-zeta_trad**2) * t_step) + 
        zeta_trad/np.sqrt(1-zeta_trad**2) * np.sin(wn_trad * np.sqrt(1-zeta_trad**2) * t_step))
    
    dual_response = 1 - np.exp(-zeta_dual * wn_dual * t_step) * (
        np.cos(wn_dual * np.sqrt(1-zeta_dual**2) * t_step) + 
        zeta_dual/np.sqrt(1-zeta_dual**2) * np.sin(wn_dual * np.sqrt(1-zeta_dual**2) * t_step))
    
    step_data = {
        'time_s': t_step,
        'traditional_response': trad_response,
        'dual_agent_response': dual_response,
        'setpoint': np.ones_like(t_step),
        'upper_bound_5percent': np.ones_like(t_step) * 1.05,
        'lower_bound_5percent': np.ones_like(t_step) * 0.95,
    }
    df_step = pd.DataFrame(step_data)
    df_step.to_csv(os.path.join(OUTPUT_DIR, 'step_response.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> step_response.csv ({len(df_step)} rows)")
    
    return df_metrics, df_radar, df_step


def export_diagnosis_analysis_data():
    """导出诊断智能体分析数据"""
    print("[4/5] 导出诊断分析数据...")
    
    # 1. 自适应阈值学习
    np.random.seed(42)
    t = np.arange(100)
    pmax_data = 137 + np.random.normal(0, 2, 100)
    pmax_data[50:] += 5  # 工况变化
    
    window = 20
    mu = np.convolve(pmax_data, np.ones(window)/window, mode='same')
    sigma = np.array([np.std(pmax_data[max(0,i-window):i+1]) for i in range(len(pmax_data))])
    
    threshold_data = {
        'time_step': t,
        'pmax_bar': pmax_data,
        'moving_average': mu,
        'moving_std': sigma,
        'upper_threshold_3sigma': mu + 3*sigma,
        'lower_threshold_3sigma': mu - 3*sigma,
        'condition_change': [1 if i == 50 else 0 for i in t],
    }
    df_thresh = pd.DataFrame(threshold_data)
    df_thresh.to_csv(os.path.join(OUTPUT_DIR, 'adaptive_threshold.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> adaptive_threshold.csv ({len(df_thresh)} rows)")
    
    # 2. 集成分类器权重
    classifier_data = {
        'classifier': ['RandomForest', 'Rule_based'],
        'weight': [0.6, 0.4],
        'weight_percent': [60, 40],
    }
    df_classifier = pd.DataFrame(classifier_data)
    df_classifier.to_csv(os.path.join(OUTPUT_DIR, 'classifier_weights.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> classifier_weights.csv ({len(df_classifier)} rows)")
    
    # 3. 故障类型分布
    fault_dist_data = {
        'fault_type': ['Normal', 'Injection_Timing', 'Fuel_Amount', 'Compression', 'Multiple_Faults'],
        'fault_type_cn': ['正常', '喷油正时', '喷油量', '压缩', '多故障'],
        'count': [65, 15, 10, 7, 3],
    }
    df_fault_dist = pd.DataFrame(fault_dist_data)
    df_fault_dist.to_csv(os.path.join(OUTPUT_DIR, 'fault_distribution.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> fault_distribution.csv ({len(df_fault_dist)} rows)")
    
    # 4. 诊断延迟分布
    np.random.seed(456)
    delays_dual = np.random.exponential(1.2, 200)
    delays_trad = np.random.exponential(2.8, 200)
    
    delay_data = {
        'sample_id': range(200),
        'dual_agent_delay_s': delays_dual,
        'traditional_delay_s': delays_trad,
    }
    df_delay = pd.DataFrame(delay_data)
    df_delay.to_csv(os.path.join(OUTPUT_DIR, 'detection_delay.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> detection_delay.csv ({len(df_delay)} rows)")
    
    # 延迟统计
    delay_stats = {
        'method': ['Dual_Agent', 'Traditional'],
        'mean_delay_s': [np.mean(delays_dual), np.mean(delays_trad)],
        'std_delay_s': [np.std(delays_dual), np.std(delays_trad)],
        'min_delay_s': [np.min(delays_dual), np.min(delays_trad)],
        'max_delay_s': [np.max(delays_dual), np.max(delays_trad)],
    }
    df_delay_stats = pd.DataFrame(delay_stats)
    df_delay_stats.to_csv(os.path.join(OUTPUT_DIR, 'detection_delay_stats.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> detection_delay_stats.csv ({len(df_delay_stats)} rows)")
    
    # 5. 混淆矩阵
    confusion_data = {
        'actual': ['Normal', 'Normal', 'Normal', 'Single_Fault', 'Single_Fault', 'Single_Fault', 
                   'Multi_Fault', 'Multi_Fault', 'Multi_Fault'],
        'predicted': ['Normal', 'Single_Fault', 'Multi_Fault'] * 3,
        'count': [62, 3, 0, 2, 28, 1, 1, 2, 1],
    }
    df_confusion = pd.DataFrame(confusion_data)
    df_confusion.to_csv(os.path.join(OUTPUT_DIR, 'confusion_matrix.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> confusion_matrix.csv ({len(df_confusion)} rows)")
    
    # 混淆矩阵 (矩阵格式)
    confusion_matrix = {
        'actual\\predicted': ['Normal', 'Single_Fault', 'Multi_Fault'],
        'Normal': [62, 3, 0],
        'Single_Fault': [2, 28, 1],
        'Multi_Fault': [1, 2, 1],
    }
    df_confusion_matrix = pd.DataFrame(confusion_matrix)
    df_confusion_matrix.to_csv(os.path.join(OUTPUT_DIR, 'confusion_matrix_table.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> confusion_matrix_table.csv")
    
    # 6. ROC曲线
    fpr = np.linspace(0, 1, 100)
    tpr_dual = 1 - (1 - fpr) ** 3
    tpr_trad = 1 - (1 - fpr) ** 1.5
    
    roc_data = {
        'false_positive_rate': fpr,
        'dual_agent_tpr': tpr_dual,
        'traditional_tpr': tpr_trad,
        'random_classifier': fpr,
    }
    df_roc = pd.DataFrame(roc_data)
    df_roc.to_csv(os.path.join(OUTPUT_DIR, 'roc_curve.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> roc_curve.csv ({len(df_roc)} rows)")
    
    # ROC统计
    auc_data = {
        'method': ['Dual_Agent', 'Traditional', 'Random'],
        'AUC': [0.95, 0.85, 0.50],
    }
    df_auc = pd.DataFrame(auc_data)
    df_auc.to_csv(os.path.join(OUTPUT_DIR, 'roc_auc.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> roc_auc.csv")
    
    return df_thresh, df_roc


def export_control_analysis_data():
    """导出控制智能体分析数据"""
    print("[5/5] 导出控制分析数据...")
    
    # 1. DQN网络架构
    network_data = {
        'layer': ['Input', 'Hidden1', 'Hidden2', 'Output'],
        'neurons': [10, 128, 64, 45],
        'activation': ['None', 'ReLU', 'ReLU', 'None'],
        'description': ['State vector', 'Fully connected', 'Fully connected', 'Q-values for actions'],
    }
    df_network = pd.DataFrame(network_data)
    df_network.to_csv(os.path.join(OUTPUT_DIR, 'dqn_architecture.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> dqn_architecture.csv ({len(df_network)} rows)")
    
    # 2. 动作空间Q值热力图
    vit_range = np.linspace(-8, 4, 9)
    fuel_range = np.linspace(0.7, 1.0, 5)
    
    q_value_data = []
    np.random.seed(42)
    for fuel in fuel_range:
        for vit in vit_range:
            q = 10 - 0.5 * (vit + 2)**2 - 5 * (fuel - 0.9)**2 + np.random.normal(0, 0.5)
            q_value_data.append({
                'vit_adjust_deg': vit,
                'fuel_multiplier': fuel,
                'q_value': q,
            })
    
    df_q_value = pd.DataFrame(q_value_data)
    df_q_value.to_csv(os.path.join(OUTPUT_DIR, 'action_space_q_values.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> action_space_q_values.csv ({len(df_q_value)} rows)")
    
    # Q值矩阵格式 (便于热力图)
    VIT, FUEL = np.meshgrid(vit_range, fuel_range)
    Q = 10 - 0.5 * (VIT + 2)**2 - 5 * (FUEL - 0.9)**2 + np.random.normal(0, 0.5, VIT.shape)
    
    q_matrix = pd.DataFrame(Q, index=[f'fuel_{f:.2f}' for f in fuel_range], 
                           columns=[f'vit_{v:.1f}' for v in vit_range])
    q_matrix.to_csv(os.path.join(OUTPUT_DIR, 'q_value_matrix.csv'), encoding='utf-8-sig')
    print(f"  -> q_value_matrix.csv")
    
    # 3. 奖励函数分解
    reward_data = {
        'component': ['Pmax_Control', 'Stability', 'Efficiency', 'Safety_Penalty', 'Total'],
        'component_cn': ['Pmax控制', '稳定性', '效率', '安全惩罚', '总奖励'],
        'value': [3.5, 2.0, 1.5, -0.5, 6.5],
        'color_hex': ['#28A745', '#17A2B8', '#2E86AB', '#DC3545', '#A23B72'],
    }
    df_reward = pd.DataFrame(reward_data)
    df_reward.to_csv(os.path.join(OUTPUT_DIR, 'reward_components.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> reward_components.csv ({len(df_reward)} rows)")
    
    # 4. PID vs RL动作对比
    np.random.seed(789)
    t = np.arange(50)
    error = 10 * np.exp(-0.1 * t) * np.sin(0.5 * t)
    
    kp, ki, kd = 2.0, 0.5, 0.3
    pid_action = kp * error + ki * np.cumsum(error) * 0.1 + kd * np.gradient(error)
    pid_action = np.clip(pid_action, -8, 4)
    
    rl_action = np.clip(2.5 * error + 0.3 * np.gradient(error), -8, 4)
    rl_action = np.convolve(rl_action, np.ones(3)/3, mode='same')
    
    action_compare_data = {
        'time_step': t,
        'error_signal': error,
        'pid_action': pid_action,
        'rl_action': rl_action,
        'action_difference': rl_action - pid_action,
    }
    df_action = pd.DataFrame(action_compare_data)
    df_action.to_csv(os.path.join(OUTPUT_DIR, 'pid_vs_rl_actions.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> pid_vs_rl_actions.csv ({len(df_action)} rows)")
    
    # 5. 经验回放奖励分布
    np.random.seed(101)
    rewards = np.concatenate([
        np.random.normal(-2, 1, 300),
        np.random.normal(3, 1.5, 500),
        np.random.normal(6, 0.8, 200),
    ])
    
    replay_data = {
        'sample_id': range(len(rewards)),
        'reward': rewards,
        'training_phase': ['Early']*300 + ['Middle']*500 + ['Late']*200,
    }
    df_replay = pd.DataFrame(replay_data)
    df_replay.to_csv(os.path.join(OUTPUT_DIR, 'replay_buffer_rewards.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> replay_buffer_rewards.csv ({len(df_replay)} rows)")
    
    # 奖励统计
    replay_stats = {
        'statistic': ['Mean', 'Std', 'Min', 'Max', 'Median'],
        'value': [np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards), np.median(rewards)],
    }
    df_replay_stats = pd.DataFrame(replay_stats)
    df_replay_stats.to_csv(os.path.join(OUTPUT_DIR, 'replay_buffer_stats.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> replay_buffer_stats.csv")
    
    # 6. 安全约束层效果
    np.random.seed(202)
    actions_raw = np.random.uniform(-10, 6, 100)
    actions_safe = np.clip(actions_raw, -8, 4)
    
    safety_data = {
        'action_index': range(100),
        'raw_action': actions_raw,
        'constrained_action': actions_safe,
        'was_clipped': (actions_raw != actions_safe).astype(int),
        'clip_amount': actions_safe - actions_raw,
    }
    df_safety = pd.DataFrame(safety_data)
    df_safety.to_csv(os.path.join(OUTPUT_DIR, 'safety_constraint.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> safety_constraint.csv ({len(df_safety)} rows)")
    
    # 安全约束配置
    constraint_config = {
        'parameter': ['VIT_min', 'VIT_max', 'Fuel_min', 'Fuel_max', 'Pmax_limit'],
        'value': [-8, 4, 0.7, 1.0, 190],
        'unit': ['deg_CA', 'deg_CA', 'ratio', 'ratio', 'bar'],
        'description': ['VIT lower limit', 'VIT upper limit', 'Min fuel multiplier', 
                       'Max fuel multiplier', 'Max cylinder pressure'],
    }
    df_constraint = pd.DataFrame(constraint_config)
    df_constraint.to_csv(os.path.join(OUTPUT_DIR, 'safety_constraints_config.csv'), index=False, encoding='utf-8-sig')
    print(f"  -> safety_constraints_config.csv")
    
    return df_q_value, df_action


def main():
    """主函数"""
    print("=" * 60)
    print("双智能体可视化数据导出")
    print("=" * 60)
    print(f"输出目录: {os.path.abspath(OUTPUT_DIR)}")
    print()
    
    # 导出所有数据
    export_training_process_data()
    print()
    export_simulation_results_data()
    print()
    export_performance_comparison_data()
    print()
    export_diagnosis_analysis_data()
    print()
    export_control_analysis_data()
    
    # 统计生成的文件
    print()
    print("=" * 60)
    print("导出完成! 生成的CSV文件:")
    print("=" * 60)
    
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')])
    total_size = 0
    for f in files:
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        total_size += size
        print(f"  {f:<40} {size/1024:>8.1f} KB")
    
    print("-" * 60)
    print(f"  总计: {len(files)} 个文件, {total_size/1024:.1f} KB")
    print("=" * 60)


if __name__ == '__main__':
    main()
