#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 验证所有算法实现正确性
每个算法只训练10个episode
"""

import sys
import os
import time
import numpy as np
import random

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_all_algorithms():
    """测试所有算法"""
    print("="*60)
    print("🧪 快速算法验证测试 (10 episodes)")
    print("="*60)
    
    # 检查PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")
    except ImportError:
        print("❌ PyTorch未安装")
        return
    
    # 创建简单环境
    from experiments.five_method_comparison import DieselEngineEnv, PIDController
    
    # 测试配置
    state_dim = 8
    action_dim = 5
    n_episodes = 10
    max_steps = 100
    
    results = {}
    
    # 1. PID
    print("\n--- 测试 PID ---")
    try:
        env = DieselEngineEnv(seed=42)
        agent = PIDController()
        reward = test_agent(agent, env, n_episodes, max_steps)
        results['PID'] = reward
        print(f"✅ PID: 平均奖励 = {reward:.2f}")
    except Exception as e:
        print(f"❌ PID失败: {e}")
    
    # 2. SAC
    print("\n--- 测试 SAC ---")
    try:
        from agents.rl_algorithms import get_algorithm
        env = DieselEngineEnv(seed=42)
        agent = get_algorithm("SAC", state_dim, action_dim, {'device': device})
        reward = test_agent(agent, env, n_episodes, max_steps, train=True)
        results['SAC'] = reward
        print(f"✅ SAC: 平均奖励 = {reward:.2f}")
    except Exception as e:
        print(f"❌ SAC失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. TD-MPC2
    print("\n--- 测试 TD-MPC2 ---")
    try:
        from agents.advanced_rl_algorithms import get_advanced_algorithm
        env = DieselEngineEnv(seed=42)
        agent = get_advanced_algorithm("TDMPC2", state_dim, action_dim, {'device': device})
        reward = test_agent(agent, env, n_episodes, max_steps, train=True)
        results['TDMPC2'] = reward
        print(f"✅ TD-MPC2: 平均奖励 = {reward:.2f}")
    except Exception as e:
        print(f"❌ TD-MPC2失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Mamba Policy
    print("\n--- 测试 Mamba Policy ---")
    try:
        from agents.advanced_rl_algorithms import get_advanced_algorithm
        env = DieselEngineEnv(seed=42)
        agent = get_advanced_algorithm("MambaPolicy", state_dim, action_dim, {'device': device})
        reward = test_agent(agent, env, n_episodes, max_steps, train=True, is_mamba=True)
        results['MambaPolicy'] = reward
        print(f"✅ Mamba Policy: 平均奖励 = {reward:.2f}")
    except Exception as e:
        print(f"❌ Mamba Policy失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. DPMD
    print("\n--- 测试 DPMD ---")
    try:
        from agents.advanced_rl_algorithms import get_advanced_algorithm
        env = DieselEngineEnv(seed=42)
        agent = get_advanced_algorithm("DPMD", state_dim, action_dim, {'device': device})
        reward = test_agent(agent, env, n_episodes, max_steps, train=True)
        results['DPMD'] = reward
        print(f"✅ DPMD: 平均奖励 = {reward:.2f}")
    except Exception as e:
        print(f"❌ DPMD失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 总结
    print("\n" + "="*60)
    print("📊 测试结果汇总")
    print("="*60)
    for name, reward in results.items():
        status = "✅" if reward is not None else "❌"
        print(f"  {status} {name}: {reward:.2f if reward else 'FAILED'}")
    
    print(f"\n成功: {len(results)}/5 算法")
    print("="*60)
    
    return results


def test_agent(agent, env, n_episodes, max_steps, train=False, is_mamba=False):
    """测试单个agent"""
    total_reward = 0
    
    for ep in range(n_episodes):
        state = env.reset()
        if hasattr(agent, 'reset'):
            agent.reset()
        if hasattr(agent, 'reset_history'):
            agent.reset_history()
        
        ep_reward = 0
        ep_states = []
        ep_actions = []
        ep_rewards = []
        
        for step in range(max_steps):
            action = agent.select_action(state, explore=True)
            next_state, reward, done, _ = env.step(action)
            
            ep_reward += reward
            ep_states.append(state)
            ep_actions.append(action)
            ep_rewards.append(reward)
            
            # 存储经验
            if train and hasattr(agent, 'buffer') and agent.buffer is not None:
                agent.buffer.push(state, action, reward, next_state, done)
                
                # 更新
                if len(agent.buffer) >= 32:
                    batch = agent.buffer.sample(32)
                    agent.update(batch)
            
            state = next_state
            if done:
                break
        
        # Mamba轨迹存储
        if is_mamba and hasattr(agent, 'store_trajectory'):
            agent.store_trajectory(ep_states, ep_actions, ep_rewards)
        
        total_reward += ep_reward
    
    return total_reward / n_episodes


if __name__ == "__main__":
    test_all_algorithms()
