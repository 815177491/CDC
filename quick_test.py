import sys
sys.path.insert(0, 'd:/my_github/CDC')

print("Step 1: Import torch")
import torch
print(f"  PyTorch version: {torch.__version__}")

print("\nStep 2: Import basic RL")
from agents.rl_algorithms import get_algorithm
print("  OK")

print("\nStep 3: Import advanced RL")  
from agents.advanced_rl_algorithms import get_advanced_algorithm, list_advanced_algorithms
print(f"  Available: {list_advanced_algorithms()}")

print("\nStep 4: Create SAC")
sac = get_algorithm("SAC", 8, 5, {'device': 'cpu'})
print(f"  SAC device: {sac.device}")

print("\nStep 5: Create TDMPC2")
tdmpc2 = get_advanced_algorithm("TDMPC2", 8, 5, {'device': 'cpu'})
print("  OK")

print("\nStep 6: Create MambaPolicy")
mamba = get_advanced_algorithm("MambaPolicy", 8, 5, {'device': 'cpu'})
print("  OK")

print("\nStep 7: Create DPMD")
dpmd = get_advanced_algorithm("DPMD", 8, 5, {'device': 'cpu'})
print("  OK")

print("\n" + "="*50)
print("All algorithms created successfully!")
print("="*50)
