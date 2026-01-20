"""
简化演示 - 跳过耗时的校准步骤
展示0D模型、故障诊断和控诊协同功能
"""

import os
import numpy as np
from engine import MarineEngine0D, OperatingCondition
from engine.combustion import DoublieWiebeCombustion
from diagnosis import FaultInjector, FaultType, FaultProfile, FaultDiagnoser
from control import SynergyController, ControlMode
from visualization import CalibrationPlotter, SynergyPlotter, PerformanceRadar

# 创建输出目录
os.makedirs('results', exist_ok=True)

print("=" * 70)
print("  零维船用柴油机仿真与控诊协同系统 - 快速演示")
print("=" * 70)

# ========== 1. 创建发动机模型 ==========
print("\n[1] 创建发动机模型...")
engine = MarineEngine0D(
    bore=0.620,
    stroke=2.658,
    n_cylinders=6,
    compression_ratio=17.0  # 预设校准值
)
print(f"    发动机配置: {engine.geometry}")

# ========== 2. 运行健康基准工况 ==========
print("\n[2] 运行健康基准工况...")
baseline_cond = OperatingCondition(
    rpm=86.0,
    p_scav=3.5e5,  # 3.5 bar
    T_scav=320.0,  # 320 K
    fuel_mass=0.08  # 80 g
)

results_healthy = engine.run_cycle(baseline_cond)
Pmax_healthy = results_healthy['pressure'].max() / 1e5
Pcomp_healthy = engine.get_pcomp()
print(f"    健康状态: Pmax={Pmax_healthy:.1f} bar, Pcomp={Pcomp_healthy:.1f} bar")

# 存储健康基准
engine.baseline_Pmax = Pmax_healthy
engine.baseline_Pcomp = Pcomp_healthy
engine.baseline_Texh = 450.0  # 假设值

# ========== 3. 故障注入与诊断 ==========
print("\n[3] 故障注入与诊断演示...")
injector = FaultInjector(engine)
diagnoser = FaultDiagnoser(engine)

# 注入喷油正时故障
print("    注入故障: 喷油正时偏移 +3°")
fault_profile = FaultProfile(
    fault_type=FaultType.INJECTION_TIMING,
    severity=3.0,  # 3度偏移
    onset_time=0.0,
    ramp_time=0.0
)
injector.inject_fault(fault_profile)

# 运行故障工况
results_faulty = engine.run_cycle(baseline_cond)
Pmax_faulty = results_faulty['pressure'].max() / 1e5
print(f"    故障状态: Pmax={Pmax_faulty:.1f} bar (变化: {(Pmax_faulty-Pmax_healthy)/Pmax_healthy*100:+.1f}%)")

# 诊断
diagnosis = diagnoser.diagnose({
    'Pmax': Pmax_faulty,
    'Pcomp': engine.get_pcomp(),
    'Texh': 460.0
})
print(f"    诊断结果: 故障类型={diagnosis.fault_type.name if diagnosis.fault_detected else '无'}")
print(f"    置信度: {diagnosis.confidence:.1%}")

# 重置故障
injector.clear_all_faults()

# ========== 4. 控诊协同演示 ==========
print("\n[4] 控诊协同控制演示...")
controller = SynergyController(engine, diagnoser)

# 模拟Pmax超限场景
print("    场景A: Pmax接近安全限值")
current_state = {
    'Pmax': 178.0,  # 接近限值
    'Pcomp': 120.0,
    'Texh': 440.0
}
control_action = controller.update(current_state, timestamp=0.0)
print(f"    控制模式: {controller.mode.name}")
print(f"    VIT调整: {control_action.vit_adjustment:.2f}°")
print(f"    燃油调整: {control_action.fuel_adjustment:.1f}%")

# ========== 5. 可视化 ==========
print("\n[5] 生成可视化图表...")
plotter = CalibrationPlotter()

# P-V图
print("    生成P-V图...")
# 运行多个工况
conditions = []
results_list = []
labels = []
for load_pct in [50, 75, 100]:
    fuel = 0.04 + load_pct * 0.0004
    cond = OperatingCondition(
        rpm=86.0,
        p_scav=3.5e5,
        T_scav=320.0,
        fuel_mass=fuel
    )
    conditions.append({'load': load_pct})
    engine_temp = MarineEngine0D(compression_ratio=17.0)
    res = engine_temp.run_cycle(cond)
    results_list.append(res)
    labels.append(f'{load_pct}% 负荷')

plotter.plot_pv_diagram(results_list, labels=labels, save_path='results/demo_pv.png')

# 燃烧放热率
print("    生成燃烧放热率...")
combustion = DoublieWiebeCombustion()
theta_range = np.linspace(-30, 90, 200)
dQ = np.array([combustion.get_heat_release_rate(th, 0.08) for th in theta_range])
x_b = np.array([combustion.get_burn_fraction(th) for th in theta_range])
plotter.plot_heat_release(theta_range, dQ, burn_fraction=x_b, save_path='results/demo_hrr.png')

# 故障响应图
print("    生成故障响应图...")
synergy_plotter = SynergyPlotter()
# 创建模拟的响应数据
time = np.linspace(0, 100, 200)
fault_signal = np.zeros_like(time)
fault_signal[50:] = 3.0  # 50秒后注入故障

response_data = {
    'time': time,
    'fault_signal': fault_signal,
    'Pmax_baseline': np.ones_like(time) * 172.0,
    'Pmax_open_loop': 172.0 + fault_signal * 2.0,  # 无控制时Pmax上升
    'Pmax_synergy': 172.0 + fault_signal * 0.3,  # 协同控制后Pmax稳定
}
synergy_plotter.plot_fault_response(
    response_data,
    fault_description='喷油正时偏移',
    save_path='results/demo_fault.png'
)

# 雷达图
print("    生成性能雷达图...")
radar = PerformanceRadar()

# 定义三种状态的性能指标
fault_state = {
    'Pmax安全裕度': 10.0,  # bar，距离190 bar限值
    '燃油效率': 180.0,     # g/kWh
    '排温裕度': 50.0,      # °C
    '输出功率': 95.0,      # %
    '转速稳定性': 2.0      # rpm波动
}
controlled_state = {
    'Pmax安全裕度': 18.0,  # 协同控制后裕度增加
    '燃油效率': 175.0,
    '排温裕度': 60.0,
    '输出功率': 92.0,
    '转速稳定性': 1.5
}
baseline_state = {
    'Pmax安全裕度': 20.0,
    '燃油效率': 170.0,
    '排温裕度': 70.0,
    '输出功率': 100.0,
    '转速稳定性': 1.0
}
radar.plot(
    fault_state=fault_state,
    controlled_state=controlled_state,
    baseline_state=baseline_state,
    save_path='results/demo_radar.png'
)

print("\n" + "=" * 70)
print("  演示完成！图表已保存至 results/ 目录")
print("=" * 70)
print("\n生成的文件:")
for f in os.listdir('results'):
    if f.startswith('demo_'):
        print(f"  - results/{f}")
