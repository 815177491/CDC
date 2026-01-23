"""测试进度条格式 - 手动实现"""
import time
import sys

print("测试进度条格式...")
print("目标格式: PID      45%|████████████        | Episode 225/500 | Reward:  -850.3 | Eval:  -920.1 | Error:  1.25bar")
print()

n_eps = 20

for i in range(n_eps):
    time.sleep(0.15)
    reward = -850.3 + i * 50
    eval_r = -920.1 + i * 45
    error = 1.25 - i * 0.05
    ep = i + 1
    
    # 手动构建完整的进度条格式
    pct = int(ep / n_eps * 100)
    filled = int(20 * ep / n_eps)
    bar_str = '█' * filled + ' ' * (20 - filled)
    
    # 完整格式字符串 (严格按照目标格式)
    status = f"\rPID     {pct:3d}%|{bar_str}| Episode {ep:3d}/{n_eps} | Reward: {reward:8.1f} | Eval: {eval_r:8.1f} | Error: {error:5.2f}bar"
    sys.stdout.write(status)
    sys.stdout.flush()

print("\n测试完成!")
