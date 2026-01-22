#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""极简RL算法测试"""
import os
import sys
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.rl_algorithms import get_algorithm, list_algorithms, ALGORITHM_INFO, TORCH_AVAILABLE

print("="*60)
print("强化学习算法对比实验")
print("="*60)
print(f"PyTorch: {TORCH_AVAILABLE}")
print(f"可用算法: {list_algorithms()}")

class SimpleEnv:
    def __init__(self):
        self.state_dim = 8
        self.action_dim = 5
    def reset(self):
        return np.random.randn(self.state_dim).astype(np.float32)
    def step(self, action):
        next_state = np.random.randn(self.state_dim).astype(np.float32)
        reward = -abs(action - 2) + np.random.randn()
        return next_state, reward, False, {}

env = SimpleEnv()
config = {'lr': 1e-3, 'gamma': 0.99, 'batch_size': 16, 'buffer_size': 1000}
results = {}

# 只测试几个快的算法
test_algos = ['DQN', 'DuelingDQN', 'TD3', 'IQL']

for name in test_algos:
    print(f"\n测试 {name}...", end=" ", flush=True)
    try:
        agent = get_algorithm(name, 8, 5, config)
        
        # 快速训练10轮
        total_reward = 0
        start = time.time()
        
        for ep in range(10):
            state = env.reset()
            for _ in range(50):
                action = agent.select_action(state, explore=True)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                
                if hasattr(agent, 'buffer'):
                    agent.buffer.push(state, action, reward, next_state, done)
                    if len(agent.buffer) >= 16:
                        batch = agent.buffer.sample(16)
                        agent.update(batch)
                state = next_state
        
        elapsed = time.time() - start
        avg_reward = total_reward / 10
        results[name] = {'reward': avg_reward, 'time': elapsed}
        print(f"奖励={avg_reward:.1f}, 时间={elapsed:.2f}s")
        
    except Exception as e:
        print(f"错误: {e}")
        results[name] = {'error': str(e)}

print("\n" + "="*60)
print("结果汇总")
print("="*60)

valid = [(n, r['reward']) for n, r in results.items() if 'error' not in r]
if valid:
    valid.sort(key=lambda x: x[1], reverse=True)
    print(f"\n最佳算法: {valid[0][0]}")
    for i, (n, r) in enumerate(valid):
        print(f"  {i+1}. {n}: 平均奖励 {r:.2f}")

print("\n完成!")
