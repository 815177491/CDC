"""
双智能体系统快速测试脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# 测试导入
print("=" * 60)
print("双智能体系统导入测试")
print("=" * 60)

try:
    from agents import CoordinatorAgent, DiagnosisAgent, ControlAgent
    from agents.base_agent import Agent, AgentMessage, MessageType, ReplayBuffer
    print("✓ 智能体模块导入成功")
except ImportError as e:
    print(f"✗ 智能体模块导入失败: {e}")
    sys.exit(1)

# 测试PyTorch和sklearn
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} 可用")
except ImportError:
    print("✗ PyTorch 不可用 (将使用PID备份)")

try:
    from sklearn.ensemble import RandomForestClassifier
    import sklearn
    print(f"✓ scikit-learn {sklearn.__version__} 可用")
except ImportError:
    print("✗ scikit-learn 不可用 (将使用规则诊断)")

# 测试发动机模块
try:
    from engine import MarineEngine0D, OperatingCondition
    print("✓ 发动机模块导入成功")
except ImportError as e:
    print(f"✗ 发动机模块导入失败: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("创建发动机模型")
print("=" * 60)

engine = MarineEngine0D(
    bore=0.620,
    stroke=2.658,
    n_cylinders=6,
    compression_ratio=14.0,
    con_rod_ratio=4.0
)
print(f"发动机: {engine}")

print("\n" + "=" * 60)
print("创建双智能体系统")
print("=" * 60)

# 创建智能体
diagnosis_agent = DiagnosisAgent(engine)
print(f"✓ 诊断智能体已创建")

control_agent = ControlAgent(engine, use_rl=True)
print(f"✓ 控制智能体已创建 (使用RL: {control_agent.use_rl})")

coordinator = CoordinatorAgent(engine, diagnosis_agent, control_agent)
print(f"✓ 协调器已创建")

# 设置基准
base_condition = OperatingCondition(
    rpm=80.0,
    p_scav=3.5e5,
    T_scav=320.0,
    fuel_mass=0.08
)

# 运行基准循环获取基准值
engine.run_cycle(base_condition)
Pmax_baseline = engine.get_pmax()
Pcomp_baseline = engine.get_pcomp()
Texh_baseline = engine.get_exhaust_temp()
engine.set_baseline(Pmax_baseline, Pcomp_baseline, Texh_baseline)

print(f"\n基准值:")
print(f"  Pmax: {Pmax_baseline:.1f} bar")
print(f"  Pcomp: {Pcomp_baseline:.1f} bar")
print(f"  Texh: {Texh_baseline:.1f} K")

print("\n" + "=" * 60)
print("运行简短仿真 (10步)")
print("=" * 60)

# 模拟故障: 在第5步时Pmax突然上升
for t in range(10):
    # 模拟测量值
    if t < 5:
        Pmax = Pmax_baseline + np.random.randn() * 2
    else:
        # 故障发生
        Pmax = Pmax_baseline + 15 + np.random.randn() * 2
    
    Y_measured = {
        'Pmax': Pmax,
        'Pcomp': Pcomp_baseline + np.random.randn() * 1,
        'Texh': Texh_baseline + np.random.randn() * 5
    }
    
    # 协调器决策
    decision = coordinator.step(Y_measured, float(t))
    action = decision.control_action
    
    # 输出
    status = "正常" if t < 5 else "故障"
    print(f"t={t:2d}s | Pmax={Pmax:6.1f}bar | VIT调整={action.vit_adjustment:+5.2f}° | "
          f"燃油调整={action.fuel_adjustment:+5.1f}% | 状态: {status}")

print("\n" + "=" * 60)
print("诊断历史")
print("=" * 60)

for i, diag in enumerate(diagnosis_agent.diagnosis_history[-5:]):
    print(f"记录 {i+1}: 故障类型={diag.fault_type}, 置信度={diag.confidence:.2f}")

print("\n" + "=" * 60)
print("性能报告")
print("=" * 60)

report = coordinator.get_comprehensive_report()
print(report)

print("\n" + "=" * 60)
print("✓ 双智能体系统测试完成!")
print("=" * 60)
