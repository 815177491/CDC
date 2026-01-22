#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速RL算法对比测试
"""
import os
import sys
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.rl_algorithms import get_algorithm, list_algorithms, ALGORITHM_INFO

print("="*60)
print("多种强化学习算法快速对比测试")
print("="*60)

# 简化测试环境
class SimpleEnv:
    def __init__(self):
        self.state_dim = 8
        self.action_dim = 5
        self.target = 160.0
        
    def reset(self):
        self.state = np.random.randn(self.state_dim).astype(np.float32)
        self.state[0] = np.random.uniform(150, 170)
        self.steps = 0
        return self.state.copy()
    
    def step(self, action):
        self.steps += 1
        action_effect = (action - 2) * 0.5
        self.state[0] += action_effect + np.random.randn() * 0.5
        self.state[1:] += np.random.randn(7) * 0.1
        
        error = abs(self.state[0] - self.target)
        reward = -error * 0.5
        if error < 2:
            reward += 5.0
        
        done = self.steps >= 100
        return self.state.copy(), reward, done, {}

env = SimpleEnv()
results = {}
algorithms = ['DQN', 'DuelingDQN', 'PPO', 'SAC', 'TD3', 'IQL']

print(f"\n测试 {len(algorithms)} 种算法...")

for algo_name in algorithms:
    print(f"\n[{algo_name}] 训练中...", end=" ", flush=True)
    
    config = {
        'lr': 1e-3, 'gamma': 0.99, 'batch_size': 32,
        'buffer_size': 10000, 'epsilon': 1.0,
        'epsilon_min': 0.1, 'epsilon_decay': 0.99
    }
    
    try:
        agent = get_algorithm(algo_name, env.state_dim, env.action_dim, config)
        episode_rewards = []
        start_time = time.time()
        
        for ep in range(30):  # 快速测试30轮
            state = env.reset()
            total_reward = 0
            
            if algo_name == 'PPO':
                agent.states = []
                agent.actions = []
                agent.rewards = []
                agent.values = []
                agent.log_probs = []
                agent.dones = []
            
            for step in range(100):
                action = agent.select_action(state, explore=True)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                
                if hasattr(agent, 'buffer'):
                    agent.buffer.push(state, action, reward, next_state, done)
                    if len(agent.buffer) >= 32:
                        batch = agent.buffer.sample(32)
                        agent.update(batch)
                
                if algo_name == 'PPO':
                    agent.store_transition(state, action, reward, done)
                
                state = next_state
                if done:
                    break
            
            if algo_name == 'PPO':
                agent.update()
            
            episode_rewards.append(total_reward)
        
        train_time = time.time() - start_time
        
        # 评估
        eval_rewards = []
        for _ in range(5):
            state = env.reset()
            tr = 0
            for _ in range(100):
                action = agent.select_action(state, explore=False)
                state, reward, done, _ = env.step(action)
                tr += reward
                if done:
                    break
            eval_rewards.append(tr)
        
        results[algo_name] = {
            'train_time': train_time,
            'eval_reward': np.mean(eval_rewards),
            'final_train_reward': np.mean(episode_rewards[-10:])
        }
        print(f"评估奖励: {np.mean(eval_rewards):.1f}")
        
    except Exception as e:
        print(f"错误: {e}")
        results[algo_name] = {'error': str(e)}

# 结果
print("\n" + "="*60)
print("结果汇总")
print("="*60)
print(f"{'算法':<18} {'评估奖励':<10} {'训练时间':<10}")
print("-"*40)

rankings = []
for name, res in results.items():
    if 'error' not in res:
        print(f"{name:<18} {res['eval_reward']:<10.1f} {res['train_time']:<10.1f}s")
        rankings.append((name, res['eval_reward']))

if rankings:
    rankings.sort(key=lambda x: x[1], reverse=True)
    best = rankings[0][0]
    info = ALGORITHM_INFO[best]
    print(f"\n★ 推荐算法: {best}")
    print(f"  来源: {info['venue']} {info['year']}")
    print(f"  描述: {info['description']}")
