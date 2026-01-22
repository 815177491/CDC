#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速训练测试 - 验证训练循环正确性
每个算法训练20个episode
"""

import sys
import os
import numpy as np
import random
import time

sys.path.insert(0, 'd:/my_github/CDC')

print("="*60)
print("🧪 快速训练测试 (20 episodes)")
print("="*60)

# 导入
import torch
from experiments.five_method_comparison import DieselEngineEnv, PIDController, ExperimentConfig
from agents.rl_algorithms import get_algorithm
from agents.advanced_rl_algorithms import get_advanced_algorithm

# 配置
state_dim = 8
action_dim = 5
n_episodes = 20
max_steps = 100
device = 'cpu'

def train_agent(name, agent, env, is_mamba=False):
    """训练单个agent"""
    start_time = time.time()
    total_reward = 0
    
    for ep in range(n_episodes):
        state = env.reset()
        if hasattr(agent, 'reset_history'):
            agent.reset_history()
        
        ep_reward = 0
        ep_states, ep_actions, ep_rewards = [], [], []
        
        for step in range(max_steps):
            action = agent.select_action(state, explore=True)
            next_state, reward, done, _ = env.step(action)
            
            ep_reward += reward
            ep_states.append(state)
            ep_actions.append(action)
            ep_rewards.append(reward)
            
            # 存储和更新
            if hasattr(agent, 'buffer') and agent.buffer is not None:
                agent.buffer.push(state, action, reward, next_state, done)
                if len(agent.buffer) >= 32:
                    batch = agent.buffer.sample(32)
                    agent.update(batch)
            
            state = next_state
            if done:
                break
        
        # Mamba轨迹
        if is_mamba and hasattr(agent, 'store_trajectory'):
            agent.store_trajectory(ep_states, ep_actions, ep_rewards)
        
        total_reward += ep_reward
        
        if (ep + 1) % 5 == 0:
            print(f"  Episode {ep+1}: reward={ep_reward:.1f}")
    
    train_time = time.time() - start_time
    avg_reward = total_reward / n_episodes
    return avg_reward, train_time

results = {}

# PID
print("\n--- PID ---")
env = DieselEngineEnv(seed=42)
pid = PIDController()
avg, t = train_agent("PID", pid, env)
results['PID'] = (avg, t)
print(f"  平均奖励: {avg:.2f}, 时间: {t:.2f}s")

# SAC
print("\n--- SAC ---")
env = DieselEngineEnv(seed=42)
sac = get_algorithm("SAC", state_dim, action_dim, {'device': device})
avg, t = train_agent("SAC", sac, env)
results['SAC'] = (avg, t)
print(f"  平均奖励: {avg:.2f}, 时间: {t:.2f}s")

# TD-MPC2
print("\n--- TD-MPC2 ---")
env = DieselEngineEnv(seed=42)
tdmpc2 = get_advanced_algorithm("TDMPC2", state_dim, action_dim, {'device': device})
avg, t = train_agent("TDMPC2", tdmpc2, env)
results['TDMPC2'] = (avg, t)
print(f"  平均奖励: {avg:.2f}, 时间: {t:.2f}s")

# MambaPolicy
print("\n--- MambaPolicy ---")
env = DieselEngineEnv(seed=42)
mamba = get_advanced_algorithm("MambaPolicy", state_dim, action_dim, {'device': device})
avg, t = train_agent("MambaPolicy", mamba, env, is_mamba=True)
results['MambaPolicy'] = (avg, t)
print(f"  平均奖励: {avg:.2f}, 时间: {t:.2f}s")

# DPMD
print("\n--- DPMD ---")
env = DieselEngineEnv(seed=42)
dpmd = get_advanced_algorithm("DPMD", state_dim, action_dim, {'device': device})
avg, t = train_agent("DPMD", dpmd, env)
results['DPMD'] = (avg, t)
print(f"  平均奖励: {avg:.2f}, 时间: {t:.2f}s")

# 总结
print("\n" + "="*60)
print("📊 训练测试结果")
print("="*60)
print(f"{'方法':<15} {'平均奖励':<15} {'训练时间':<10}")
print("-"*40)
for name, (avg, t) in results.items():
    print(f"{name:<15} {avg:<15.2f} {t:<10.2f}s")

best = min(results.items(), key=lambda x: -x[1][0])  # 最高奖励
print(f"\n🏆 当前最佳: {best[0]} (奖励: {best[1][0]:.2f})")
print("="*60)
