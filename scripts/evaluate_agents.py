"""
智能体评估脚本
==============
对比评估传统控制器与双智能体系统的性能

评估指标:
1. Pmax越限次数/时间
2. 控制响应速度
3. 稳态误差
4. 燃油效率损失
5. 故障检测准确率
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict, List, Tuple
import warnings

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import MarineEngine0D, OperatingCondition
from diagnosis import FaultDiagnoser, FaultInjector, FaultType
from control import SynergyController
from agents import CoordinatorAgent, DiagnosisAgent, ControlAgent


def run_evaluation_episode(engine: MarineEngine0D, 
                           use_agents: bool,
                           fault_type: str = 'timing',
                           duration: float = 100.0,
                           dt: float = 1.0) -> Dict:
    """
    运行单个评估回合
    
    Args:
        engine: 发动机模型
        use_agents: 是否使用双智能体
        fault_type: 故障类型
        duration: 仿真时长
        dt: 时间步长
        
    Returns:
        评估结果字典
    """
    # 基准工况
    base_condition = OperatingCondition(
        rpm=80.0, p_scav=3.5e5, T_scav=320.0, fuel_mass=0.08
    )
    
    # 运行基准
    engine.clear_faults()
    engine.run_cycle(base_condition)
    Pmax_baseline = engine.get_pmax()
    Pcomp_baseline = engine.get_pcomp()
    Texh_baseline = engine.get_exhaust_temp()
    
    engine.set_baseline(Pmax_baseline, Pcomp_baseline, Texh_baseline)
    
    # 创建故障注入器
    fault_injector = FaultInjector(engine)
    
    # 创建故障
    if fault_type == 'timing':
        fault = fault_injector.create_timing_fault(2.0, 25.0, 0.0)
    elif fault_type == 'leak':
        fault = fault_injector.create_leak_fault(0.1, 25.0, 5.0)
    else:
        fault = fault_injector.create_fuel_fault(0.15, 25.0, 0.0)
    
    fault_injector.inject_fault(fault)
    
    # 创建控制器
    if use_agents:
        diagnosis_agent = DiagnosisAgent(engine)
        control_agent = ControlAgent(engine, use_rl=True)
        control_agent.learning_enabled = False  # 评估时不学习
        coordinator = CoordinatorAgent(engine, diagnosis_agent, control_agent)
    else:
        diagnoser = FaultDiagnoser(engine)
        controller = SynergyController(engine, diagnoser)
    
    # 存储结果
    results = {
        'time': [],
        'Pmax': [],
        'vit': [],
        'fuel': [],
        'fault_detected': [],
        'mode': [],
    }
    
    Pmax_limit = 190.0
    
    # 仿真循环
    for t in np.arange(0, duration, dt):
        fault_injector.apply_faults(t)
        engine.run_cycle(base_condition)
        
        Y_measured = {
            'Pmax': engine.get_pmax(),
            'Pcomp': engine.get_pcomp(),
            'Texh': engine.get_exhaust_temp()
        }
        
        if use_agents:
            decision = coordinator.step(Y_measured, t)
            action = decision.control_action
            fault_detected = decision.diagnosis_result.fault_detected
        else:
            action = controller.update(Y_measured, t)
            fault_detected = diagnoser.current_state.name != 'HEALTHY'
        
        results['time'].append(t)
        results['Pmax'].append(Y_measured['Pmax'])
        results['vit'].append(action.vit_adjustment)
        results['fuel'].append(action.power_limit)
        results['fault_detected'].append(fault_detected)
        results['mode'].append(action.mode.name)
    
    # 清理
    fault_injector.clear_all_faults()
    if use_agents:
        coordinator.reset()
    else:
        controller.reset()
    
    # 转换为numpy
    for key in ['time', 'Pmax', 'vit', 'fuel']:
        results[key] = np.array(results[key])
    
    # 计算评估指标
    results['metrics'] = compute_metrics(results, Pmax_limit, fault_onset=25.0)
    
    return results


def compute_metrics(results: Dict, Pmax_limit: float, 
                    fault_onset: float) -> Dict:
    """
    计算评估指标
    
    Args:
        results: 仿真结果
        Pmax_limit: Pmax限值
        fault_onset: 故障发生时刻
        
    Returns:
        指标字典
    """
    time = results['time']
    Pmax = results['Pmax']
    vit = results['vit']
    fuel = results['fuel']
    fault_detected = results['fault_detected']
    
    # 1. Pmax越限统计
    violations = Pmax > Pmax_limit
    violation_count = np.sum(violations)
    violation_time = np.sum(violations) * (time[1] - time[0]) if len(time) > 1 else 0
    max_overshoot = max(0, np.max(Pmax) - Pmax_limit)
    
    # 2. 响应速度 (从故障发生到检测的时间)
    post_fault_detected = fault_detected[int(fault_onset):]
    if any(post_fault_detected):
        first_detect_idx = np.argmax(post_fault_detected)
        detection_delay = first_detect_idx * (time[1] - time[0]) if len(time) > 1 else 0
    else:
        detection_delay = float('inf')
    
    # 3. 稳态误差 (最后20%数据)
    n_steady = max(1, len(Pmax) // 5)
    steady_Pmax = Pmax[-n_steady:]
    steady_error = abs(np.mean(steady_Pmax) - 170)  # 相对目标170bar
    
    # 4. 燃油效率损失
    fuel_loss = 1.0 - np.mean(fuel)
    
    # 5. 控制活动量 (VIT变化的总和)
    control_effort = np.sum(np.abs(np.diff(vit)))
    
    return {
        'violation_count': int(violation_count),
        'violation_time_s': float(violation_time),
        'max_overshoot_bar': float(max_overshoot),
        'detection_delay_s': float(detection_delay),
        'steady_error_bar': float(steady_error),
        'fuel_loss_pct': float(fuel_loss * 100),
        'control_effort_deg': float(control_effort),
    }


def run_comparison(n_trials: int = 10, 
                   fault_types: List[str] = None) -> Dict:
    """
    运行对比评估
    
    Args:
        n_trials: 每种配置的试验次数
        fault_types: 要测试的故障类型列表
        
    Returns:
        对比结果
    """
    if fault_types is None:
        fault_types = ['timing', 'leak', 'fuel']
    
    print("\n" + "=" * 60)
    print("控制器对比评估")
    print("=" * 60)
    
    # 创建发动机
    engine = MarineEngine0D(
        bore=0.620, stroke=2.658, n_cylinders=6, compression_ratio=13.5
    )
    
    results = {
        'traditional': {ft: [] for ft in fault_types},
        'agents': {ft: [] for ft in fault_types},
    }
    
    for fault_type in fault_types:
        print(f"\n测试故障类型: {fault_type}")
        
        for trial in range(n_trials):
            # 传统控制器
            trad_results = run_evaluation_episode(
                engine, use_agents=False, fault_type=fault_type
            )
            results['traditional'][fault_type].append(trad_results['metrics'])
            
            # 双智能体
            agent_results = run_evaluation_episode(
                engine, use_agents=True, fault_type=fault_type
            )
            results['agents'][fault_type].append(agent_results['metrics'])
            
            print(f"  Trial {trial+1}/{n_trials}: "
                  f"Trad violations={trad_results['metrics']['violation_count']}, "
                  f"Agent violations={agent_results['metrics']['violation_count']}")
    
    return results


def summarize_results(results: Dict) -> None:
    """汇总并打印结果"""
    print("\n" + "=" * 70)
    print("评估结果汇总")
    print("=" * 70)
    
    metrics_keys = [
        'violation_count', 'max_overshoot_bar', 'detection_delay_s',
        'steady_error_bar', 'fuel_loss_pct', 'control_effort_deg'
    ]
    
    for fault_type in results['traditional'].keys():
        print(f"\n故障类型: {fault_type}")
        print("-" * 50)
        
        trad_metrics = results['traditional'][fault_type]
        agent_metrics = results['agents'][fault_type]
        
        print(f"{'指标':<25} {'传统控制器':<15} {'双智能体':<15} {'改善%':<10}")
        print("-" * 50)
        
        for key in metrics_keys:
            trad_mean = np.mean([m[key] for m in trad_metrics])
            agent_mean = np.mean([m[key] for m in agent_metrics])
            
            if trad_mean != 0:
                improvement = (trad_mean - agent_mean) / abs(trad_mean) * 100
            else:
                improvement = 0 if agent_mean == 0 else -100
            
            # 对于越限次数和超调量，改善为正表示更好
            # 对于燃油损失，改善为负可能更好（意味着更少的牺牲）
            print(f"{key:<25} {trad_mean:<15.2f} {agent_mean:<15.2f} {improvement:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description='智能体评估脚本')
    parser.add_argument('--trials', type=int, default=5, help='每种配置的试验次数')
    parser.add_argument('--fault-types', nargs='+', default=['timing', 'leak', 'fuel'],
                        help='要测试的故障类型')
    parser.add_argument('--save', type=str, default=None, help='结果保存路径')
    
    args = parser.parse_args()
    
    # 检查依赖
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
    except ImportError:
        print("警告: PyTorch未安装，双智能体将使用PID备份")
    
    # 运行对比评估
    results = run_comparison(
        n_trials=args.trials,
        fault_types=args.fault_types
    )
    
    # 汇总结果
    summarize_results(results)
    
    # 保存结果
    if args.save:
        import json
        
        # 转换为可序列化格式
        save_data = {}
        for ctrl_type, fault_data in results.items():
            save_data[ctrl_type] = {}
            for fault_type, metrics_list in fault_data.items():
                save_data[ctrl_type][fault_type] = metrics_list
        
        with open(args.save, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\n结果已保存至: {args.save}")


if __name__ == "__main__":
    main()
