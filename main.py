"""
双智能体诊-控协同强化学习系统
=============================

主入口脚本 - 集成RL诊断、RL控制和多智能体训练

新特性 (v2.0):
1. 基于SAC的RL诊断智能体 (替代规则基诊断)
2. MAPPO/QMIX多智能体协同训练
3. 完整的双智能体评估体系
4. 诊-控协同时序可视化

使用方法:
    python main.py --mode train-mappo --episodes 500  # MAPPO训练
    python main.py --mode train-qmix --episodes 500   # QMIX训练
    python main.py --mode eval --model models/dual_agent/final
    python main.py --mode demo --model models/dual_agent/final
"""

import sys
import os
import argparse
import numpy as np
import warnings

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入环境和智能体模块
from environments import create_dual_agent_env
from agents.rl_diagnosis_agent import create_rl_diagnosis_agent
from agents.multi_agent_algorithms import get_multi_agent_algorithm
from agents.rl_algorithms import SAC
from experiments.dual_agent_evaluation import DualAgentEvaluator
from visualization.dual_agent_plots import DualAgentVisualizer

# 可选导入旧模块 (用于兼容性)
try:
    from engine import MarineEngine0D
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    print("警告: engine模块不可用")


def run_calibration(csv_path: str = "calibration_data.csv", 
                    n_points: int = 5) -> MarineEngine0D:
    """
    运行三阶段参数校准
    
    Args:
        csv_path: 校准数据CSV路径
        n_points: 使用的校准点数量
        
    Returns:
        engine: 校准后的发动机模型
    """
    print("\n" + "=" * 60)
    print("零维船用柴油机模型参数校准")
    print("=" * 60)
    
    # 创建发动机模型 (使用用户提供的几何参数)
    engine = MarineEngine0D(
        bore=0.620,      # 620 mm
        stroke=2.658,    # 2658 mm
        n_cylinders=6,
        compression_ratio=13.5,  # 初始猜测值
        con_rod_ratio=4.0
    )
    
    print(f"\n发动机配置: {engine}")
    print(f"缸径: {engine.geometry.bore*1000:.0f} mm")
    print(f"行程: {engine.geometry.stroke*1000:.0f} mm")
    print(f"气缸数: {engine.geometry.n_cylinders}")
    print(f"排量/缸: {engine.geometry.displaced_volume*1000:.1f} L")
    
    # 检查CSV文件
    if not os.path.exists(csv_path):
        print(f"\n警告: 未找到校准数据文件 {csv_path}")
        print("使用默认参数运行...")
        return engine
    
    # 创建校准器
    data_loader = CalibrationDataLoader(csv_path)
    calibrator = EngineCalibrator(engine, data_loader)
    
    # 运行三阶段校准
    try:
        results = calibrator.run_full_calibration(n_points=n_points)
        
        # 导出参数
        params = calibrator.export_parameters("calibrated_params.json")
        print(f"\n校准参数已保存至: calibrated_params.json")
        
    except Exception as e:
        print(f"\n校准过程中出现错误: {e}")
        print("使用默认参数继续...")
    
    return engine


def run_fault_simulation(engine: MarineEngine0D, use_agents: bool = True) -> dict:
    """
    运行故障注入与诊断仿真
    
    Args:
        engine: 发动机模型
        use_agents: 是否使用双智能体架构 (默认True)
        
    Returns:
        response_data: 响应数据字典
    """
    print("\n" + "=" * 60)
    print("故障注入与诊断仿真")
    print("=" * 60)
    
    # 创建基准工况
    base_condition = OperatingCondition(
        rpm=80.0,
        p_scav=3.5e5,
        T_scav=320.0,
        fuel_mass=0.08
    )
    
    # 运行基准循环
    engine.clear_faults()
    results_baseline = engine.run_cycle(base_condition)
    Pmax_baseline = engine.get_pmax()
    Pcomp_baseline = engine.get_pcomp()
    
    print(f"\n基准工况: RPM={base_condition.rpm}, 扫气压力={base_condition.p_scav/1e5:.2f}bar")
    print(f"基准Pmax: {Pmax_baseline:.1f} bar")
    print(f"基准Pcomp: {Pcomp_baseline:.1f} bar")
    
    # 根据模式创建诊断器和控制器
    if use_agents and AGENTS_AVAILABLE:
        print("\n[使用双智能体架构]")
        # 创建诊断智能体和控制智能体
        diagnosis_agent = DiagnosisAgent(engine)
        control_agent = ControlAgent(engine, use_rl=True)
        
        # 创建协调器
        coordinator = CoordinatorAgent(engine, diagnosis_agent, control_agent)
        
        # 故障注入器
        fault_injector = FaultInjector(engine)
        
        using_agents = True
    else:
        print("\n[使用传统控制器]")
        # 创建传统诊断器和控制器
        diagnoser = FaultDiagnoser(engine)
        controller = SynergyController(engine, diagnoser)
        fault_injector = FaultInjector(engine)
        using_agents = False
    
    # 设置健康基准
    engine.set_baseline(Pmax_baseline, Pcomp_baseline, engine.get_exhaust_temp())
    
    # 创建喷油正时故障
    timing_fault = fault_injector.create_timing_fault(
        offset_deg=2.0,     # 正时提前2度
        onset_time=25.0,    # 25秒后发生
        ramp_time=0.0       # 阶跃故障
    )
    fault_injector.inject_fault(timing_fault)
    
    print(f"\n故障注入: {timing_fault.description}")
    print(f"发生时刻: {timing_fault.onset_time}s")
    
    # 仿真参数
    duration = 100.0
    dt = 1.0
    
    # 存储结果
    times = []
    fault_signal = []
    Pmax_baseline_list = []
    Pmax_open_loop = []
    Pmax_synergy = []
    vit_adjustments = []
    fuel_adjustments = []
    diagnosis_results = []
    action_sources = []      # 新增: 记录动作来源
    conflicts_detected = []  # 新增: 记录冲突检测
    
    print("\n开始仿真...")
    
    for t in np.arange(0, duration, dt):
        times.append(t)
        Pmax_baseline_list.append(Pmax_baseline)
        
        # 记录故障信号
        fault_mag = timing_fault.get_magnitude(t) * 5.0  # 转换为度
        fault_signal.append(fault_mag)
        
        # 应用故障
        fault_injector.apply_faults(t)
        
        # 运行仿真获取实际状态 (无控制)
        engine.run_cycle(base_condition)
        Pmax_current = engine.get_pmax()
        Pmax_open_loop.append(Pmax_current)
        
        # 诊断与协同控制
        Y_measured = {
            'Pmax': Pmax_current,
            'Pcomp': engine.get_pcomp(),
            'Texh': engine.get_exhaust_temp()
        }
        
        if using_agents:
            # 使用双智能体协调器
            decision = coordinator.step(Y_measured, t)
            action = decision.control_action
            vit_adjustments.append(action.vit_adjustment)
            fuel_adjustments.append(action.fuel_adjustment)
            action_sources.append(action.action_source)
            conflicts_detected.append(decision.conflict_resolved)
        else:
            # 使用传统控制器
            action = controller.update(Y_measured, t)
            vit_adjustments.append(action.vit_adjustment)
            fuel_adjustments.append(action.fuel_adjustment)
            action_sources.append("PID")
            conflicts_detected.append(False)
        
        # 计算协同控制后的Pmax
        # 简化模型: VIT每滞后1度, Pmax降低约2bar
        Pmax_controlled = Pmax_current + action.vit_adjustment * 2.0
        Pmax_synergy.append(Pmax_controlled)
        
        # 记录诊断结果
        if using_agents:
            if coordinator.diagnosis_agent.diagnosis_history:
                diagnosis_results.append(coordinator.diagnosis_agent.diagnosis_history[-1])
        else:
            if diagnoser.diagnosis_history:
                diagnosis_results.append(diagnoser.diagnosis_history[-1])
    
    # 清理
    fault_injector.clear_all_faults()
    
    if using_agents:
        coordinator.reset()
    else:
        controller.reset()
    
    print("仿真完成!")
    
    # 输出性能汇总
    if using_agents:
        report = coordinator.get_comprehensive_report()
        print(f"\n=== 双智能体系统性能汇总 ===")
        print(f"  系统状态: {report.get('system_state', 'UNKNOWN')}")
        if 'coordinator_metrics' in report:
            print(f"  总步数: {report['coordinator_metrics'].get('total_steps', 0)}")
            print(f"  冲突检测: {report['coordinator_metrics'].get('conflicts_detected', 0)}")
            print(f"  冲突解决: {report['coordinator_metrics'].get('conflicts_resolved', 0)}")
            print(f"  平均协调耗时: {report['coordinator_metrics'].get('avg_coordination_time_ms', 0):.2f}ms")
        if 'diagnosis_agent' in report:
            print(f"\n  诊断智能体:")
            print(f"    学习阈值: {report['diagnosis_agent'].get('learned_thresholds', {})}")
            print(f"    分类器状态: {report['diagnosis_agent'].get('classifier_status', {})}")
        if 'control_agent' in report:
            print(f"\n  控制智能体:")
            print(f"    当前VIT: {report['control_agent'].get('current_vit', 0):.2f}°")
            perf = report['control_agent'].get('performance', {})
            print(f"    RL动作占比: {perf.get('rl_actions', 0)}/{perf.get('total_actions', 0)}")
    else:
        perf = controller.get_performance_summary()
        print(f"\n控制器性能汇总:")
        print(f"  总干预次数: {perf['total_interventions']}")
        print(f"  Pmax越限次数: {perf['pmax_violations']}")
        print(f"  平均VIT调整: {perf['avg_vit_adjustment']:.2f}°")
    
    return {
        'time': np.array(times),
        'fault_signal': np.array(fault_signal),
        'Pmax_baseline': np.array(Pmax_baseline_list),
        'Pmax_open_loop': np.array(Pmax_open_loop),
        'Pmax_synergy': np.array(Pmax_synergy),
        'vit_adjustment': np.array(vit_adjustments),
        'fuel_adjustment': np.array(fuel_adjustments),
        'action_sources': action_sources,
        'conflicts': conflicts_detected,
        'using_agents': using_agents,
    }


def generate_plots(engine: MarineEngine0D, 
                   response_data: dict,
                   output_dir: str = "results"):
    """
    生成所有验证图表
    
    Args:
        engine: 发动机模型
        response_data: 故障响应数据
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("生成验证图表")
    print("=" * 60)
    
    # 初始化绑制器
    cal_plotter = CalibrationPlotter()
    syn_plotter = SynergyPlotter()
    radar = PerformanceRadar()
    process_plotter = CalibrationProcessPlotter()
    
    # ========== 图表0: 校准过程综合图 ==========
    print("\n生成图表0: 三阶段校准过程图...")
    
    fig0 = process_plotter.plot_calibration_process(
        calibration_history=None,  # 使用默认示例数据
        save_path=os.path.join(output_dir, "fig0_calibration_process.png")
    )
    print(f"  已保存: {output_dir}/fig0_calibration_process.png")
    
    # ========== 图表1: 热力参数对标图 ==========
    print("\n生成图表1: 热力参数对标图...")
    
    # 生成多个工况的仿真结果 (用于演示)
    conditions = []
    sim_results = []
    exp_results = []
    
    for rpm in [60, 70, 80, 90, 100]:
        cond = OperatingCondition(
            rpm=rpm,
            p_scav=3.0e5 + rpm * 100,
            T_scav=310 + rpm * 0.1,
            fuel_mass=0.06 + rpm * 0.0003
        )
        conditions.append({'rpm': rpm})
        
        engine.run_cycle(cond)
        sim_results.append({
            'Pmax': engine.get_pmax(),
            'Pcomp': engine.get_pcomp()
        })
        
        # 模拟实验值 (添加小量噪声)
        exp_results.append({
            'Pmax': engine.get_pmax() * (1 + np.random.uniform(-0.02, 0.02)),
            'Pcomp': engine.get_pcomp() * (1 + np.random.uniform(-0.01, 0.01))
        })
    
    fig1 = cal_plotter.plot_thermal_comparison(
        conditions, sim_results, exp_results,
        save_path=os.path.join(output_dir, "fig1_calibration_verification.png")
    )
    print(f"  已保存: {output_dir}/fig1_calibration_verification.png")
    
    # ========== 图表2: P-V示功图 ==========
    print("\n生成图表2: P-V示功图...")
    
    pv_results = []
    pv_labels = []
    for rpm, label in [(60, 'R1-低负荷'), (80, 'R2-中负荷'), 
                       (100, 'R3-高负荷'), (110, 'R4-满负荷')]:
        cond = OperatingCondition(
            rpm=rpm,
            p_scav=3.0e5 + rpm * 100,
            T_scav=320,
            fuel_mass=0.05 + rpm * 0.0005
        )
        results = engine.run_cycle(cond)
        pv_results.append(results)
        pv_labels.append(label)
    
    fig2 = cal_plotter.plot_pv_diagram(
        pv_results, pv_labels,
        save_path=os.path.join(output_dir, "fig2_pv_diagram.png")
    )
    print(f"  已保存: {output_dir}/fig2_pv_diagram.png")
    
    # ========== 图表3: 故障响应时序图 ==========
    print("\n生成图表3: 故障响应时序图...")
    
    fig3 = syn_plotter.plot_fault_response(
        response_data,
        fault_description="喷油正时提前偏差",
        save_path=os.path.join(output_dir, "fig3_fault_response.png")
    )
    print(f"  已保存: {output_dir}/fig3_fault_response.png")
    
    # ========== 图表4: 性能权衡雷达图 ==========
    print("\n生成图表4: 性能权衡雷达图...")
    
    # 定义各状态的性能指标
    baseline_state = {
        'Pmax安全裕度': 20,
        '燃油效率': 170,
        '排温裕度': 80,
        '输出功率': 100,
        '转速稳定性': 0.5
    }
    
    fault_state = {
        'Pmax安全裕度': 5,
        '燃油效率': 165,
        '排温裕度': 40,
        '输出功率': 100,
        '转速稳定性': 2.0
    }
    
    controlled_state = {
        'Pmax安全裕度': 18,
        '燃油效率': 175,
        '排温裕度': 70,
        '输出功率': 95,
        '转速稳定性': 0.8
    }
    
    fig4 = radar.plot(
        fault_state, controlled_state, baseline_state,
        save_path=os.path.join(output_dir, "fig4_performance_radar.png")
    )
    print(f"  已保存: {output_dir}/fig4_performance_radar.png")
    
    # ========== 放热率曲线 ==========
    print("\n生成附加图表: 燃烧放热率曲线...")
    
    cond = OperatingCondition(rpm=80, p_scav=3.5e5, T_scav=320, fuel_mass=0.08)
    results = engine.run_cycle(cond)
    
    theta, dQ = engine.get_heat_release_data()
    burn_fraction = results.get('burn_fraction', None)
    
    fig_hr = cal_plotter.plot_heat_release(
        theta, dQ, burn_fraction,
        save_path=os.path.join(output_dir, "fig5_heat_release.png")
    )
    print(f"  已保存: {output_dir}/fig5_heat_release.png")
    
    print("\n所有图表已生成完毕!")
    
    return [fig1, fig2, fig3, fig4, fig_hr]


def run_dashboard(engine: MarineEngine0D):
    """启动交互式仪表盘"""
    print("\n" + "=" * 60)
    print("启动交互式仪表盘")
    print("=" * 60)
    
    from visualization.dashboard import InteractiveDashboard
    
    diagnoser = FaultDiagnoser(engine)
    controller = SynergyController(engine, diagnoser)
    
    dashboard = InteractiveDashboard(engine, diagnoser, controller)
    dashboard.run_interactive()


def run_dual_agent_training(args):
    """
    运行双智能体训练 (MAPPO或QMIX)
    
    Args:
        args: 包含以下字段:
            - dual_mode: 'mappo' 或 'qmix'
            - episodes: 训练回合数
            - eval_interval: 评估间隔
            - save_dir: 模型保存目录
            - device: GPU或CPU
    """
    from scripts.train_dual_agents import DualAgentTrainer
    
    print(f"\n{'='*70}")
    print(f"启动双智能体训练 - 模式: {args.dual_mode.upper()}")
    print(f"{'='*70}\n")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建训练配置
    config = {
        'training_mode': args.dual_mode,  # 'independent', 'mappo', 'qmix'
        'n_episodes': args.episodes,
        'eval_interval': args.eval_interval,
        'save_dir': args.save_dir,
        'log_dir': os.path.join(args.save_dir, 'logs'),
    }
    
    # 创建训练器
    trainer = DualAgentTrainer(config=config)
    
    # 运行训练
    print(f"开始训练 {args.episodes} 个回合...")
    training_history = trainer.train()
    
    print("\n" + "="*70)
    print("训练完成!")
    print(f"模型已保存至: {args.save_dir}")
    print("="*70 + "\n")
    
    # 返回最终智能体用于后续操作
    return trainer.diag_agent, trainer.ctrl_agent, trainer.env


def run_dual_agent_evaluation(args):
    """
    评估训练好的双智能体模型
    
    Args:
        args: 包含以下字段:
            - model_dir: 模型所在目录
            - num_episodes: 评估回合数
            - device: GPU或CPU
    """
    print(f"\n{'='*70}")
    print("启动双智能体评估")
    print(f"{'='*70}\n")
    
    # 加载模型
    model_path = os.path.join(args.model_dir, 'final')
    if not os.path.exists(f"{model_path}_diag.pt"):
        print(f"错误: 找不到模型文件 {model_path}_diag.pt")
        return
    
    # 创建环境和智能体
    env = create_dual_agent_env()
    diag_agent = create_rl_diagnosis_agent(device=args.device)
    ctrl_agent = SAC(state_dim=10, action_dim=2, device=args.device)
    
    # 加载权重
    diag_agent.load(f"{model_path}_diag.pt")
    ctrl_agent.load(f"{model_path}_ctrl.pt")
    
    # 运行评估
    evaluator = DualAgentEvaluator(diag_agent, ctrl_agent, env)
    
    n_eval = args.num_episodes
    print(f"运行诊断评估 ({n_eval} 回合)...")
    diag_metrics = evaluator.evaluate_diagnosis(n_episodes=n_eval)
    
    print(f"运行控制评估 ({n_eval} 回合)...")
    ctrl_metrics = evaluator.evaluate_control(n_episodes=n_eval)
    
    print(f"运行协同评估 ({n_eval} 回合)...")
    coop_metrics = evaluator.evaluate_cooperation(n_episodes=n_eval)
    
    # 输出结果
    print("\n" + "="*70)
    print("评估结果")
    print("="*70)
    
    print("\n诊断性能:")
    print(f"  - 准确率: {diag_metrics['accuracy']:.1%}")
    print(f"  - 检测延迟: {diag_metrics['detection_delay']:.2f} 步")
    print(f"  - 虚报率: {diag_metrics.get('false_positive_rate', 0):.1%}")
    print(f"  - 漏报率: {diag_metrics.get('false_negative_rate', 0):.1%}")
    
    print("\n控制性能:")
    print(f"  - Pmax RMSE: {ctrl_metrics['pmax_rmse']:.4f}")
    print(f"  - 违规次数: {ctrl_metrics['violation_count']}")
    print(f"  - 燃油经济性: {ctrl_metrics.get('fuel_economy', 0):.2f}")
    
    print("\n协同性能:")
    print(f"  - 端到端成功率: {coop_metrics['end_to_end_success_rate']:.1%}")
    print(f"  - 诊断正确后的控制成功率: {coop_metrics.get('control_success_given_correct_diag', 0):.1%}")
    print("="*70 + "\n")
    
    return diag_metrics, ctrl_metrics, coop_metrics


def run_demo():
    """运行完整演示流程"""
    print("\n" + "=" * 70)
    print("  零维船用柴油机仿真与控诊协同系统 - 完整演示")
    print("=" * 70)
    
    # 步骤1: 参数校准
    engine = run_calibration()
    
    # 步骤2: 故障仿真
    response_data = run_fault_simulation(engine)
    
    # 步骤3: 生成图表
    figures = generate_plots(engine, response_data)
    
    print("\n" + "=" * 70)
    print("  演示完成!")
    print("=" * 70)
    print("\n生成的文件:")
    print("  - calibrated_params.json  (校准参数)")
    print("  - results/fig0_calibration_process.png")
    print("  - results/fig1_calibration_verification.png")
    print("  - results/fig2_pv_diagram.png")
    print("  - results/fig3_fault_response.png")
    print("  - results/fig4_performance_radar.png")
    print("  - results/fig5_heat_release.png")
    
    # 显示图表
    import matplotlib.pyplot as plt
    plt.show()


def main():
    """主函数 - 支持传统模式和双智能体强化学习模式"""
    parser = argparse.ArgumentParser(
        description='零维船用柴油机仿真与控诊协同系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  
  # ===== 传统单智能体模式 =====
  python main.py --mode demo              # 完整演示
  python main.py --mode calibrate         # 仅校准
  python main.py --mode simulate          # 仅仿真
  
  # ===== 双智能体强化学习模式 (新) =====
  python main.py --mode train-mappo --episodes 500 --save-dir models/dual_mappo
  python main.py --mode train-qmix --episodes 500 --save-dir models/dual_qmix
  python main.py --mode eval-dual --model-dir models/dual_mappo --num-episodes 100
        """
    )
    parser.add_argument(
        '--mode', type=str, default='demo',
        choices=['demo', 'calibrate', 'simulate', 'dashboard', 'plot', 'agents',
                 'train-mappo', 'train-qmix', 'train-independent',
                 'eval-dual', 'demo-dual'],
        help='运行模式'
    )
    parser.add_argument(
        '--csv', type=str, default='calibration_data.csv',
        help='校准数据CSV文件路径'
    )
    parser.add_argument(
        '--points', type=int, default=5,
        help='校准使用的工况点数量'
    )
    
    # 双智能体训练参数
    parser.add_argument(
        '--episodes', type=int, default=500,
        help='双智能体训练回合数 (default: 500)'
    )
    parser.add_argument(
        '--eval-interval', type=int, default=50,
        help='评估间隔 (default: 50)'
    )
    parser.add_argument(
        '--save-dir', type=str, default='models/dual_agent',
        help='模型保存目录 (default: models/dual_agent)'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='计算设备 (default: cuda)'
    )
    
    # 双智能体评估参数
    parser.add_argument(
        '--model-dir', type=str, default='models/dual_agent',
        help='加载模型的目录'
    )
    parser.add_argument(
        '--num-episodes', type=int, default=100,
        help='评估回合数 (default: 100)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        run_demo()
    
    elif args.mode == 'calibrate':
        engine = run_calibration(args.csv, args.points)
        print("\n校准完成。使用 --mode demo 运行完整演示。")
    
    elif args.mode == 'simulate':
        engine = run_calibration(args.csv, args.points)
        response_data = run_fault_simulation(engine)
    
    elif args.mode == 'dashboard':
        engine = run_calibration(args.csv, args.points)
        run_dashboard(engine)
    
    elif args.mode == 'plot':
        engine = run_calibration(args.csv, args.points)
        # 使用模拟响应数据
        response_data = {
            'time': np.linspace(0, 100, 100),
            'Pmax_baseline': np.ones(100) * 170,
            'Pmax_open_loop': np.concatenate([
                np.ones(25) * 170,
                np.ones(75) * 185
            ]),
            'Pmax_synergy': np.concatenate([
                np.ones(25) * 170,
                np.linspace(185, 175, 75)
            ]),
            'vit_adjustment': np.concatenate([
                np.zeros(25),
                np.linspace(0, -5, 75)
            ]),
            'fuel_adjustment': np.zeros(100),
        }
        generate_plots(engine, response_data)
    
    elif args.mode == 'agents':
        # 双智能体专用演示模式
        print("\n" + "=" * 70)
        print("  双智能体架构演示")
        print("=" * 70)
        
        if not ENGINE_AVAILABLE:
            print("错误: engine模块不可用")
            return
        
        engine = run_calibration(args.csv, args.points)
        
        # 强制使用双智能体
        response_data = run_fault_simulation(engine, use_agents=True)
        
        # 生成图表
        generate_plots(engine, response_data)
        
        print("\n双智能体演示完成!")
        import matplotlib.pyplot as plt
        plt.show()
    
    # ===== 新增: 双智能体强化学习模式 =====
    elif args.mode in ['train-mappo', 'train-qmix', 'train-independent']:
        # 提取模式名称
        dual_mode = args.mode.split('-')[1]  # 'mappo', 'qmix', 或 'independent'
        args.dual_mode = dual_mode
        run_dual_agent_training(args)
    
    elif args.mode == 'eval-dual':
        run_dual_agent_evaluation(args)
    
    elif args.mode == 'demo-dual':
        print("\n" + "="*70)
        print("启动双智能体演示 - 可视化协同过程")
        print("="*70 + "\n")
        
        import torch
        
        # 检查模型
        model_path = os.path.join(args.model_dir, 'final')
        if not os.path.exists(f"{model_path}_diag.pt"):
            print(f"错误: 找不到模型文件 {model_path}_diag.pt")
            print(f"请先运行训练: python main.py --mode train-mappo --save-dir {args.model_dir}")
            return
        
        # 创建环境和智能体
        env = create_dual_agent_env()
        diag_agent = create_rl_diagnosis_agent(device=args.device)
        ctrl_agent = SAC(state_dim=10, action_dim=2, device=args.device)
        
        # 加载权重
        diag_agent.load(f"{model_path}_diag.pt")
        ctrl_agent.load(f"{model_path}_ctrl.pt")
        
        # 运行5个演示回合
        print("正在运行演示回合...")
        os.makedirs(args.model_dir, exist_ok=True)
        
        from matplotlib import pyplot as plt
        
        trajectory_data_list = []
        for ep in range(5):
            obs = env.reset()
            ep_data = {'diag_obs': [], 'ctrl_obs': [], 'diag_actions': [], 'ctrl_actions': [],
                       'diag_rewards': [], 'ctrl_rewards': [], 'ground_truth': [],
                       'pmax': [], 'vit': [], 'fuel': []}
            done = False
            
            while not done:
                # 诊断智能体执行
                diag_action, _ = diag_agent.select_action(obs[0], training=False)
                
                # 控制智能体执行
                ctrl_action, _ = ctrl_agent.select_action(obs[1], training=False)
                
                # 环境步进
                (diag_obs, ctrl_obs), (diag_reward, ctrl_reward), done, info = env.step(
                    (diag_action, ctrl_action)
                )
                
                ep_data['diag_obs'].append(obs[0])
                ep_data['ctrl_obs'].append(obs[1])
                ep_data['diag_actions'].append(diag_action)
                ep_data['ctrl_actions'].append(ctrl_action)
                ep_data['diag_rewards'].append(diag_reward)
                ep_data['ctrl_rewards'].append(ctrl_reward)
                ep_data['ground_truth'].append(info.get('ground_truth_fault', -1))
                ep_data['pmax'].append(info.get('pmax', 0))
                ep_data['vit'].append(info.get('vit', 0))
                ep_data['fuel'].append(info.get('fuel_ratio', 0))
                
                obs = (diag_obs, ctrl_obs)
            
            trajectory_data_list.append(ep_data)
            print(f"  演示回合 {ep+1}/5 完成 (长度: {len(ep_data['ground_truth'])} 步)")
        
        # 生成可视化
        print("\n生成可视化...")
        visualizer = DualAgentVisualizer()
        
        # 绘制协同响应 (使用第一个回合)
        ep_data = trajectory_data_list[0]
        visualizer.plot_coordination_response(
            episode_data=ep_data,
            save_path=os.path.join(args.model_dir, 'coordination_response.png')
        )
        
        print(f"可视化已保存至: {args.model_dir}/coordination_response.png")
        print("演示完成!")





if __name__ == "__main__":
    main()
