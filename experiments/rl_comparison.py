#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多种RL算法对比实验框架
======================
对比 DQN、Dueling DQN、PPO、SAC、TD3、Decision Transformer、IQL 
在船用柴油机燃烧控制任务上的性能

Author: CDC Project
Date: 2026-01-21
"""

import numpy as np
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    import pandas as pd
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    warnings.warn("Matplotlib/Pandas not available. Plotting disabled.")

from agents.rl_algorithms import (
    get_algorithm, list_algorithms, ALGORITHM_INFO,
    ReplayBuffer, TORCH_AVAILABLE
)


# ============================================================
# 实验配置
# ============================================================

@dataclass
class ExperimentConfig:
    """实验配置"""
    # 算法列表
    algorithms: List[str] = None
    
    # 训练参数
    n_episodes: int = 200
    max_steps_per_episode: int = 500
    eval_frequency: int = 20
    n_eval_episodes: int = 5
    
    # 环境参数
    state_dim: int = 8
    action_dim: int = 5
    
    # 随机种子
    seed: int = 42
    
    # 输出
    output_dir: str = 'experiment_results'
    save_models: bool = True
    
    def __post_init__(self):
        if self.algorithms is None:
            self.algorithms = ['DQN', 'DuelingDQN', 'PPO', 'SAC', 'TD3', 'IQL']


@dataclass
class EpisodeResult:
    """单个episode的结果"""
    episode: int
    reward: float
    steps: int
    pmax_error: float
    control_accuracy: float
    safety_violations: int


@dataclass
class AlgorithmResult:
    """单个算法的完整结果"""
    name: str
    training_time: float
    episode_rewards: List[float]
    eval_rewards: List[float]
    final_performance: Dict[str, float]
    convergence_episode: int  # 收敛时的episode


# ============================================================
# 模拟环境 (基于柴油机控制场景)
# ============================================================

class DieselEngineControlEnv:
    """
    柴油机燃烧控制仿真环境
    用于RL算法对比实验
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 状态空间: [Pmax, dPmax/dt, 燃油量, 喷射时间, 缸温, 负载, 转速, 误差积分]
        self.state_dim = 8
        # 动作空间: [大减, 小减, 保持, 小增, 大增] (对喷射时间)
        self.action_dim = 5
        
        # 目标参数
        self.target_pmax = 160.0  # bar
        self.pmax_tolerance = 2.0  # ±2 bar
        
        # 安全约束
        self.pmax_max = 175.0  # 最大允许Pmax
        self.pmax_min = 140.0  # 最小Pmax
        
        # 状态变量
        self.state = None
        self.step_count = 0
        self.max_steps = 500
        
        # 控制参数
        self.injection_timing = 0.0  # 当前喷射时间偏差 (°CA)
        self.fuel_amount = 100.0  # 燃油量 (%)
        
        # 噪声参数
        self.noise_std = 0.5
    
    def reset(self, seed: int = None) -> np.ndarray:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        # 随机初始化
        self.injection_timing = np.random.uniform(-3, 3)
        self.fuel_amount = np.random.uniform(90, 110)
        
        # 计算初始Pmax (带噪声)
        pmax = self._calculate_pmax()
        dpmax_dt = np.random.uniform(-1, 1)
        
        # 其他状态量
        cyl_temp = 350 + np.random.uniform(-20, 20)
        load = np.random.uniform(0.6, 1.0)
        rpm = 100 + np.random.uniform(-5, 5)
        error_integral = 0.0
        
        self.state = np.array([
            pmax,
            dpmax_dt,
            self.fuel_amount,
            self.injection_timing,
            cyl_temp,
            load,
            rpm,
            error_integral
        ])
        
        self.step_count = 0
        return self.state.copy()
    
    def _calculate_pmax(self) -> float:
        """计算Pmax (简化模型)"""
        # Pmax与喷射时间、燃油量的关系
        base_pmax = 155.0
        timing_effect = -2.0 * self.injection_timing  # 提前喷射增加Pmax
        fuel_effect = 0.2 * (self.fuel_amount - 100)
        
        pmax = base_pmax + timing_effect + fuel_effect
        pmax += np.random.normal(0, self.noise_std)  # 测量噪声
        
        return np.clip(pmax, self.pmax_min - 5, self.pmax_max + 5)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步"""
        self.step_count += 1
        
        # 动作映射: 0-大减, 1-小减, 2-保持, 3-小增, 4-大增
        action_effects = [-1.0, -0.3, 0.0, 0.3, 1.0]
        delta_timing = action_effects[action]
        
        # 更新喷射时间
        self.injection_timing += delta_timing
        self.injection_timing = np.clip(self.injection_timing, -10, 10)
        
        # 计算新Pmax
        new_pmax = self._calculate_pmax()
        old_pmax = self.state[0]
        dpmax_dt = new_pmax - old_pmax
        
        # 更新误差积分
        error = new_pmax - self.target_pmax
        error_integral = self.state[7] + error * 0.01
        error_integral = np.clip(error_integral, -10, 10)
        
        # 模拟其他状态变化
        cyl_temp = self.state[4] + np.random.normal(0, 2)
        load = self.state[5] + np.random.normal(0, 0.01)
        load = np.clip(load, 0.3, 1.0)
        rpm = self.state[6] + np.random.normal(0, 0.5)
        
        # 负载扰动
        if np.random.random() < 0.02:
            load += np.random.choice([-0.1, 0.1])
            load = np.clip(load, 0.3, 1.0)
        
        # 更新状态
        self.state = np.array([
            new_pmax,
            dpmax_dt,
            self.fuel_amount,
            self.injection_timing,
            cyl_temp,
            load,
            rpm,
            error_integral
        ])
        
        # 计算奖励
        reward = self._calculate_reward(new_pmax, action)
        
        # 检查终止条件
        done = False
        info = {'safety_violation': False}
        
        if new_pmax > self.pmax_max or new_pmax < self.pmax_min:
            info['safety_violation'] = True
            reward -= 10.0  # 安全惩罚
        
        if self.step_count >= self.max_steps:
            done = True
        
        return self.state.copy(), reward, done, info
    
    def _calculate_reward(self, pmax: float, action: int) -> float:
        """计算奖励"""
        # 误差惩罚
        error = abs(pmax - self.target_pmax)
        reward = -error * 0.5
        
        # 达到目标奖励
        if error < self.pmax_tolerance:
            reward += 5.0
        
        # 控制平滑奖励
        if action == 2:  # 保持
            reward += 0.5
        
        # 大幅调整惩罚
        if action in [0, 4]:
            reward -= 0.3
        
        return reward
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        pmax = self.state[0]
        error = abs(pmax - self.target_pmax)
        
        return {
            'pmax_error': error,
            'pmax': pmax,
            'in_tolerance': error < self.pmax_tolerance,
            'injection_timing': self.injection_timing
        }


# ============================================================
# 实验运行器
# ============================================================

class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.env = DieselEngineControlEnv()
        self.results: Dict[str, AlgorithmResult] = {}
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 设置随机种子
        np.random.seed(config.seed)
    
    def run_single_algorithm(self, algo_name: str, verbose: bool = True) -> AlgorithmResult:
        """运行单个算法的实验"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"训练算法: {algo_name}")
            info = ALGORITHM_INFO.get(algo_name, {})
            print(f"  类型: {info.get('type', 'N/A')}")
            print(f"  来源: {info.get('venue', 'N/A')} {info.get('year', '')}")
            print(f"{'='*60}")
        
        # 创建算法实例
        algo_config = {
            'lr': 1e-3,
            'gamma': 0.99,
            'batch_size': 64,
            'buffer_size': 50000
        }
        
        agent = get_algorithm(algo_name, self.config.state_dim, self.config.action_dim, algo_config)
        
        # 训练记录
        episode_rewards = []
        eval_rewards = []
        start_time = time.time()
        
        # 收敛检测
        convergence_episode = self.config.n_episodes
        convergence_threshold = -5.0  # 平均奖励阈值
        convergence_window = 20
        
        for episode in range(self.config.n_episodes):
            state = self.env.reset(seed=self.config.seed + episode)
            total_reward = 0
            
            # PPO需要特殊处理
            if algo_name == 'PPO':
                agent.reset_history() if hasattr(agent, 'reset_history') else None
            
            # Decision Transformer需要特殊处理
            if algo_name in ['DecisionTransformer', 'DT']:
                agent.reset_history()
            
            episode_states = []
            episode_actions = []
            episode_rewards_list = []
            
            for step in range(self.config.max_steps_per_episode):
                # 选择动作
                action = agent.select_action(state, explore=True)
                
                # 环境交互
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward
                
                # 存储经验
                if hasattr(agent, 'buffer'):
                    agent.buffer.push(state, action, reward, next_state, done)
                
                if algo_name == 'PPO':
                    agent.store_transition(state, action, reward, done)
                
                if algo_name in ['DecisionTransformer', 'DT']:
                    episode_states.append(state)
                    episode_actions.append(action)
                    episode_rewards_list.append(reward)
                
                # 更新网络
                if hasattr(agent, 'buffer') and len(agent.buffer) >= algo_config['batch_size']:
                    batch = agent.buffer.sample(algo_config['batch_size'])
                    agent.update(batch)
                
                state = next_state
                
                if done:
                    break
            
            # PPO在episode结束时更新
            if algo_name == 'PPO':
                agent.update()
            
            # Decision Transformer存储轨迹
            if algo_name in ['DecisionTransformer', 'DT']:
                agent.store_trajectory(episode_states, episode_actions, episode_rewards_list)
                if len(agent.trajectories) > 10:
                    agent.update()
            
            episode_rewards.append(total_reward)
            
            # 评估
            if (episode + 1) % self.config.eval_frequency == 0:
                eval_reward = self.evaluate_agent(agent, algo_name)
                eval_rewards.append(eval_reward)
                
                if verbose:
                    avg_recent = np.mean(episode_rewards[-20:])
                    print(f"  Episode {episode+1:4d} | Reward: {total_reward:7.2f} | "
                          f"Avg(20): {avg_recent:7.2f} | Eval: {eval_reward:7.2f}")
                
                # 收敛检测
                if len(episode_rewards) >= convergence_window:
                    recent_avg = np.mean(episode_rewards[-convergence_window:])
                    if recent_avg > convergence_threshold and convergence_episode == self.config.n_episodes:
                        convergence_episode = episode + 1
        
        training_time = time.time() - start_time
        
        # 最终评估
        final_eval = self.evaluate_agent(agent, algo_name, n_episodes=10)
        final_metrics = self.env.get_performance_metrics()
        
        # 保存模型
        if self.config.save_models:
            model_path = os.path.join(self.config.output_dir, f'{algo_name}_model.pt')
            agent.save(model_path)
        
        result = AlgorithmResult(
            name=algo_name,
            training_time=training_time,
            episode_rewards=episode_rewards,
            eval_rewards=eval_rewards,
            final_performance={
                'eval_reward': final_eval,
                'pmax_error': final_metrics['pmax_error'],
                'convergence_episode': convergence_episode
            },
            convergence_episode=convergence_episode
        )
        
        self.results[algo_name] = result
        
        if verbose:
            print(f"\n  训练完成! 用时: {training_time:.1f}s")
            print(f"  最终评估奖励: {final_eval:.2f}")
            print(f"  收敛Episode: {convergence_episode}")
        
        return result
    
    def evaluate_agent(self, agent, algo_name: str, n_episodes: int = None) -> float:
        """评估智能体"""
        n_episodes = n_episodes or self.config.n_eval_episodes
        total_rewards = []
        
        for i in range(n_episodes):
            state = self.env.reset(seed=10000 + i)  # 使用固定种子
            
            if algo_name in ['DecisionTransformer', 'DT'] and hasattr(agent, 'reset_history'):
                agent.reset_history()
            
            episode_reward = 0
            
            for _ in range(self.config.max_steps_per_episode):
                action = agent.select_action(state, explore=False)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)
    
    def run_all_experiments(self) -> Dict[str, AlgorithmResult]:
        """运行所有算法的对比实验"""
        print("\n" + "="*70)
        print("开始多算法对比实验")
        print(f"算法列表: {self.config.algorithms}")
        print(f"训练Episodes: {self.config.n_episodes}")
        print("="*70)
        
        for algo_name in self.config.algorithms:
            try:
                self.run_single_algorithm(algo_name)
            except Exception as e:
                print(f"  算法 {algo_name} 训练失败: {e}")
                continue
        
        return self.results
    
    def save_results(self):
        """保存实验结果"""
        # 保存JSON摘要
        summary = {}
        for name, result in self.results.items():
            summary[name] = {
                'training_time': result.training_time,
                'final_eval_reward': result.final_performance['eval_reward'],
                'convergence_episode': result.convergence_episode,
                'mean_reward': float(np.mean(result.episode_rewards[-50:])),
                'std_reward': float(np.std(result.episode_rewards[-50:]))
            }
        
        json_path = os.path.join(self.config.output_dir, 'experiment_summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {json_path}")
        
        # 保存详细CSV
        if PLOT_AVAILABLE:
            for name, result in self.results.items():
                df = pd.DataFrame({
                    'episode': range(len(result.episode_rewards)),
                    'reward': result.episode_rewards
                })
                csv_path = os.path.join(self.config.output_dir, f'{name}_training.csv')
                df.to_csv(csv_path, index=False)
    
    def plot_comparison(self):
        """绘制对比图"""
        if not PLOT_AVAILABLE:
            print("Matplotlib未安装，跳过绘图")
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        
        # 1. 训练曲线对比
        ax1 = axes[0, 0]
        for (name, result), color in zip(self.results.items(), colors):
            # 平滑处理
            rewards = np.array(result.episode_rewards)
            window = min(20, len(rewards) // 5)
            if window > 1:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                x = np.arange(window-1, len(rewards))
            else:
                smoothed = rewards
                x = np.arange(len(rewards))
            ax1.plot(x, smoothed, label=name, color=color, linewidth=2)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('训练曲线对比')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # 2. 最终性能柱状图
        ax2 = axes[0, 1]
        names = list(self.results.keys())
        final_rewards = [r.final_performance['eval_reward'] for r in self.results.values()]
        bars = ax2.bar(names, final_rewards, color=colors)
        ax2.set_ylabel('Final Evaluation Reward')
        ax2.set_title('最终性能对比')
        ax2.tick_params(axis='x', rotation=45)
        
        # 标注数值
        for bar, val in zip(bars, final_rewards):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 3. 收敛速度对比
        ax3 = axes[1, 0]
        convergence = [r.convergence_episode for r in self.results.values()]
        bars = ax3.bar(names, convergence, color=colors)
        ax3.set_ylabel('Convergence Episode')
        ax3.set_title('收敛速度对比 (越小越好)')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, val in zip(bars, convergence):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val}', ha='center', va='bottom', fontsize=9)
        
        # 4. 训练时间对比
        ax4 = axes[1, 1]
        times = [r.training_time for r in self.results.values()]
        bars = ax4.bar(names, times, color=colors)
        ax4.set_ylabel('Training Time (s)')
        ax4.set_title('训练时间对比')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, val in zip(bars, times):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}s', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # 保存图片
        fig_path = os.path.join(self.config.output_dir, 'algorithm_comparison.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"对比图已保存: {fig_path}")
    
    def plot_detailed_analysis(self):
        """绘制详细分析图"""
        if not PLOT_AVAILABLE or len(self.results) < 2:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        
        # 1. 学习曲线带置信区间
        ax = axes[0, 0]
        for (name, result), color in zip(self.results.items(), colors):
            rewards = np.array(result.episode_rewards)
            window = 10
            
            mean_rewards = []
            std_rewards = []
            
            for i in range(len(rewards)):
                start = max(0, i - window)
                segment = rewards[start:i+1]
                mean_rewards.append(np.mean(segment))
                std_rewards.append(np.std(segment))
            
            mean_rewards = np.array(mean_rewards)
            std_rewards = np.array(std_rewards)
            x = np.arange(len(rewards))
            
            ax.plot(x, mean_rewards, label=name, color=color, linewidth=2)
            ax.fill_between(x, mean_rewards - std_rewards, mean_rewards + std_rewards,
                           alpha=0.2, color=color)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('学习曲线 (带标准差)')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 2. 雷达图 - 多维性能对比
        ax = axes[0, 1]
        
        metrics = ['最终奖励', '收敛速度', '训练效率', '稳定性']
        n_metrics = len(metrics)
        angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]
        
        for (name, result), color in zip(self.results.items(), colors):
            # 归一化各指标到0-1
            final_reward = result.final_performance['eval_reward']
            convergence = result.convergence_episode
            training_time = result.training_time
            stability = 1.0 / (np.std(result.episode_rewards[-30:]) + 1)
            
            # 归一化
            values = [
                (final_reward + 50) / 100,  # 假设范围 -50 to 50
                1 - convergence / self.config.n_episodes,  # 越小越好
                1 - training_time / max([r.training_time for r in self.results.values()]),
                stability / 2
            ]
            values = [max(0, min(1, v)) for v in values]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', label=name, color=color, linewidth=2)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('多维性能雷达图')
        ax.legend(loc='upper right', fontsize=8)
        
        # 3. 箱线图 - 奖励分布
        ax = axes[0, 2]
        
        data = []
        labels = []
        for name, result in self.results.items():
            data.append(result.episode_rewards[-50:])  # 最后50个episode
            labels.append(name)
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel('Reward')
        ax.set_title('奖励分布 (最后50个Episode)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. 评估奖励曲线
        ax = axes[1, 0]
        for (name, result), color in zip(self.results.items(), colors):
            if result.eval_rewards:
                x = np.arange(len(result.eval_rewards)) * self.config.eval_frequency
                ax.plot(x, result.eval_rewards, 'o-', label=name, color=color, 
                       linewidth=2, markersize=4)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Evaluation Reward')
        ax.set_title('评估性能曲线')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 5. 算法排名表
        ax = axes[1, 1]
        ax.axis('off')
        
        # 计算综合得分
        rankings = []
        for name, result in self.results.items():
            score = (
                result.final_performance['eval_reward'] * 0.4 +
                (1 - result.convergence_episode / self.config.n_episodes) * 30 +
                (1 - result.training_time / max([r.training_time for r in self.results.values()])) * 20 +
                np.mean(result.episode_rewards[-30:]) * 0.3
            )
            rankings.append((name, score, result.final_performance['eval_reward'],
                           result.convergence_episode, result.training_time))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        table_data = [['排名', '算法', '综合得分', '最终奖励', '收敛Episode', '训练时间']]
        for i, (name, score, reward, conv, time_) in enumerate(rankings):
            table_data.append([f'{i+1}', name, f'{score:.2f}', f'{reward:.2f}', 
                              f'{conv}', f'{time_:.1f}s'])
        
        table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                        colWidths=[0.1, 0.2, 0.15, 0.15, 0.2, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # 高亮第一名
        for j in range(6):
            table[(1, j)].set_facecolor('#90EE90')
        
        ax.set_title('算法综合排名', y=0.9)
        
        # 6. 推荐和结论
        ax = axes[1, 2]
        ax.axis('off')
        
        best_algo = rankings[0][0]
        best_info = ALGORITHM_INFO.get(best_algo, {})
        
        conclusion = f"""
实验结论
========

推荐算法: {best_algo}
来源: {best_info.get('venue', 'N/A')} {best_info.get('year', '')}
类型: {best_info.get('type', 'N/A')}

{best_info.get('description', '')}

综合得分: {rankings[0][1]:.2f}
最终奖励: {rankings[0][2]:.2f}
收敛速度: Episode {rankings[0][3]}

实验配置:
- 训练Episodes: {self.config.n_episodes}
- 评估频率: 每{self.config.eval_frequency}个Episode
- 状态维度: {self.config.state_dim}
- 动作维度: {self.config.action_dim}
"""
        
        ax.text(0.1, 0.9, conclusion, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存
        fig_path = os.path.join(self.config.output_dir, 'detailed_analysis.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"详细分析图已保存: {fig_path}")
    
    def generate_report(self) -> str:
        """生成实验报告"""
        report = []
        report.append("=" * 70)
        report.append("多种强化学习算法对比实验报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        report.append("\n【实验配置】")
        report.append(f"  训练Episodes: {self.config.n_episodes}")
        report.append(f"  每Episode最大步数: {self.config.max_steps_per_episode}")
        report.append(f"  评估频率: 每{self.config.eval_frequency}个Episode")
        report.append(f"  状态维度: {self.config.state_dim}")
        report.append(f"  动作维度: {self.config.action_dim}")
        
        report.append("\n【算法介绍】")
        for name in self.config.algorithms:
            info = ALGORITHM_INFO.get(name, {})
            report.append(f"\n  {name} ({info.get('name', '')})")
            report.append(f"    来源: {info.get('venue', 'N/A')} {info.get('year', '')}")
            report.append(f"    类型: {info.get('type', 'N/A')}")
            report.append(f"    简介: {info.get('description', 'N/A')}")
        
        report.append("\n【实验结果】")
        for name, result in self.results.items():
            report.append(f"\n  {name}:")
            report.append(f"    训练时间: {result.training_time:.1f} 秒")
            report.append(f"    最终评估奖励: {result.final_performance['eval_reward']:.2f}")
            report.append(f"    收敛Episode: {result.convergence_episode}")
            report.append(f"    最后50轮平均奖励: {np.mean(result.episode_rewards[-50:]):.2f}")
            report.append(f"    最后50轮标准差: {np.std(result.episode_rewards[-50:]):.2f}")
        
        # 排名
        rankings = []
        for name, result in self.results.items():
            score = (
                result.final_performance['eval_reward'] * 0.4 +
                (1 - result.convergence_episode / self.config.n_episodes) * 30 +
                np.mean(result.episode_rewards[-30:]) * 0.3
            )
            rankings.append((name, score))
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        report.append("\n【综合排名】")
        for i, (name, score) in enumerate(rankings):
            report.append(f"  {i+1}. {name} (综合得分: {score:.2f})")
        
        report.append("\n【推荐结论】")
        best = rankings[0][0]
        report.append(f"  推荐使用: {best}")
        info = ALGORITHM_INFO.get(best, {})
        report.append(f"  理由: {info.get('description', '综合性能最优')}")
        
        report.append("\n" + "=" * 70)
        
        report_text = "\n".join(report)
        
        # 保存报告
        report_path = os.path.join(self.config.output_dir, 'experiment_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n实验报告已保存: {report_path}")
        
        return report_text


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    print("="*70)
    print("多种强化学习算法对比实验")
    print("="*70)
    print("\n包含算法:")
    for name in list_algorithms():
        info = ALGORITHM_INFO.get(name, {})
        print(f"  - {name}: {info.get('name', '')} ({info.get('venue', '')} {info.get('year', '')})")
    
    # 实验配置
    config = ExperimentConfig(
        algorithms=['DQN', 'DuelingDQN', 'PPO', 'SAC', 'TD3', 'IQL'],
        n_episodes=150,
        max_steps_per_episode=300,
        eval_frequency=15,
        n_eval_episodes=5,
        seed=42,
        output_dir='experiment_results'
    )
    
    # 运行实验
    runner = ExperimentRunner(config)
    results = runner.run_all_experiments()
    
    # 保存结果
    runner.save_results()
    
    # 绘制对比图
    runner.plot_comparison()
    runner.plot_detailed_analysis()
    
    # 生成报告
    report = runner.generate_report()
    print(report)
    
    return results


if __name__ == '__main__':
    main()
