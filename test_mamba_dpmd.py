#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试Mamba和DPMD"""

import sys
import os
import numpy as np
import time

sys.path.insert(0, 'd:/my_github/CDC')

print("="*50)
print("测试 MambaPolicy 和 DPMD")
print("="*50)

from experiments.five_method_comparison import DieselEngineEnv
from agents.advanced_rl_algorithms import get_advanced_algorithm

state_dim = 8
action_dim = 5
n_episodes = 10
max_steps = 50

def train_agent(name, agent, env, is_mamba=False):
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
            
            if hasattr(agent, 'buffer') and agent.buffer is not None:
                agent.buffer.push(state, action, reward, next_state, done)
                if len(agent.buffer) >= 32:
                    batch = agent.buffer.sample(32)
                    agent.update(batch)
            
            state = next_state
            if done:
                break
        
        if is_mamba and hasattr(agent, 'store_trajectory'):
            agent.store_trajectory(ep_states, ep_actions, ep_rewards)
        
        total_reward += ep_reward
        print(f"  Ep {ep+1}: reward={ep_reward:.1f}")
    
    return total_reward / n_episodes, time.time() - start_time

# MambaPolicy
print("\n--- MambaPolicy ---")
try:
    env = DieselEngineEnv(seed=42)
    mamba = get_advanced_algorithm("MambaPolicy", state_dim, action_dim, {'device': 'cpu'})
    avg, t = train_agent("MambaPolicy", mamba, env, is_mamba=True)
    print(f"✅ MambaPolicy: 平均奖励={avg:.2f}, 时间={t:.2f}s")
except Exception as e:
    print(f"❌ MambaPolicy失败: {e}")
    import traceback
    traceback.print_exc()

# DPMD
print("\n--- DPMD ---")
try:
    env = DieselEngineEnv(seed=42)
    dpmd = get_advanced_algorithm("DPMD", state_dim, action_dim, {'device': 'cpu'})
    avg, t = train_agent("DPMD", dpmd, env)
    print(f"✅ DPMD: 平均奖励={avg:.2f}, 时间={t:.2f}s")
except Exception as e:
    print(f"❌ DPMD失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("测试完成!")
print("="*50)
