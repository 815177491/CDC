"""
简单运行验证脚本
"""
print("\n" + "="*60)
print("零维船用柴油机智能控诊协同系统")
print("="*60)

print("\n[1/4] 检查依赖...")
import sys
import numpy as np
print(f"  ✓ NumPy {np.__version__}")

import scipy
print(f"  ✓ SciPy {scipy.__version__}")

import pandas as pd
print(f"  ✓ Pandas {pd.__version__}")

import matplotlib
print(f"  ✓ Matplotlib {matplotlib.__version__}")

try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    has_torch = True
except ImportError:
    print("  ⚠ PyTorch 不可用（将使用PID备份）")
    has_torch = False

try:
    import sklearn
    print(f"  ✓ scikit-learn {sklearn.__version__}")
    has_sklearn = True
except ImportError:
    print("  ⚠ scikit-learn 不可用（将使用规则诊断）")
    has_sklearn = False

print("\n[2/4] 导入模块...")
from engine import MarineEngine0D, OperatingCondition
print("  ✓ 发动机模块")

from calibration import EngineCalibrator, CalibrationDataLoader
print("  ✓ 校准模块")

from diagnosis import FaultDiagnoser, FaultInjector
print("  ✓ 诊断模块")

from control import SynergyController
print("  ✓ 控制模块")

from agents import CoordinatorAgent, DiagnosisAgent, ControlAgent
print("  ✓ 智能体模块")

print("\n[3/4] 创建发动机模型...")
engine = MarineEngine0D(
    bore=0.620,
    stroke=2.658,
    n_cylinders=6,
    compression_ratio=14.0
)
print(f"  ✓ {engine}")
print(f"    - 缸径: {engine.geometry.bore*1000:.0f} mm")
print(f"    - 行程: {engine.geometry.stroke*1000:.0f} mm")
print(f"    - 单缸排量: {engine.geometry.displaced_volume*1000:.1f} L")

print("\n[4/4] 运行简单测试...")
condition = OperatingCondition(
    rpm=80.0,
    p_scav=3.5e5,
    T_scav=320.0,
    fuel_mass=0.08
)
results = engine.run_cycle(condition)
Pmax = engine.get_pmax()
Pcomp = engine.get_pcomp()
Texh = engine.get_exhaust_temp()

print(f"  ✓ 循环仿真完成")
print(f"    - Pmax: {Pmax:.1f} bar")
print(f"    - Pcomp: {Pcomp:.1f} bar")
print(f"    - Texh: {Texh:.1f} K")

print("\n" + "="*60)
print("系统验证通过！")
print("="*60)

print("\n可用运行模式：")
print("  1. python test_agents.py           # 快速测试")
print("  2. python main.py --mode agents    # 双智能体演示")
print("  3. python main.py --mode demo      # 传统演示")

if has_torch:
    print("  4. python scripts/train_agents.py  # 训练DQN")
    print("  5. python scripts/evaluate_agents.py  # 性能评估")
else:
    print("\n提示：安装 torch 和 scikit-learn 以启用完整功能")
    print("  pip install torch scikit-learn")

print()
