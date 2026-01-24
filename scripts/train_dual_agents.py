#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双智能体训练脚本
================
支持独立学习 (Independent Learning) 和 集中式训练 (MAPPO/QMIX)

训练模式:
1. independent: 诊断和控制智能体独立训练
2. mappo: 使用MAPPO进行集中式训练
3. qmix: 使用QMIX进行值分解训练

Author: CDC Project
Date: 2026-01-24
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import deque

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.dual_agent_env import create_dual_agent_env, DualAgentEngineEnv
from agents.rl_diagnosis_agent import create_rl_diagnosis_agent, RLDiagnosisAgent
from agents.multi_agent_algorithms import (
    get_multi_agent_algorithm, 
    MultiAgentExperience,
    MAPPO, QMIX
)

# 尝试导入控制智能体的SAC
try:
    from agents.rl_algorithms import SAC
    SAC_AVAILABLE = True
except ImportError:
    SAC_AVAILABLE = False
    print("[警告] SAC算法不可用，使用简化控制器")


# ============================================================
# 训练配置
# ============================================================

DEFAULT_CONFIG = {
    # 训练设置
    'n_episodes': 500,
    'max_steps_per_episode': 200,
    'eval_interval': 50,
    'save_interval': 100,
    'log_interval': 10,
    
    # 模式
    'training_mode': 'mappo',  # 'independent', 'mappo', 'qmix'
    
    # 环境设置
    'fault_probability': 0.8,
    'multi_fault_probability': 0.1,
    'sensor_noise_std': 0.0,
    
    # 诊断智能体设置 (SAC)
    'diag_lr': 3e-4,
    'diag_gamma': 0.99,
    'diag_tau': 0.005,
    'diag_batch_size': 128,
    'diag_buffer_size': 100000,
    
    # 控制智能体设置 (SAC)
    'ctrl_lr': 3e-4,
    'ctrl_gamma': 0.99,
    'ctrl_tau': 0.005,
    'ctrl_batch_size': 128,
    'ctrl_buffer_size': 100000,
    
    # MAPPO/QMIX设置
    'ma_lr': 3e-4,
    'ma_gamma': 0.99,
    'ma_gae_lambda': 0.95,
    'ma_clip_epsilon': 0.2,
    'ma_ppo_epochs': 10,
    'ma_batch_size': 64,
    
    # 奖励权重
    'diag_reward_weight': 0.4,
    'ctrl_reward_weight': 0.4,
    'coop_reward_weight': 0.2,
    
    # 保存路径
    'save_dir': 'models/dual_agent',
    'log_dir': 'logs/dual_agent'
}


# ============================================================
# 训练器
# ============================================================

class DualAgentTrainer:
    """双智能体训练器"""
    
    def __init__(self, config: Dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        # 创建目录
        os.makedirs(self.config['save_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # 创建环境
        self.env = create_dual_agent_env({
            'fault_probability': self.config['fault_probability'],
            'multi_fault_probability': self.config['multi_fault_probability'],
            'sensor_noise_std': self.config['sensor_noise_std'],
            'max_steps': self.config['max_steps_per_episode']
        })
        
        # 获取维度
        self.diag_state_dim = self.env.diag_state_dim
        self.diag_action_dim = self.env.diag_action_dim
        self.ctrl_state_dim = self.env.ctrl_state_dim
        self.ctrl_action_dim = self.env.ctrl_action_dim
        
        # 初始化智能体
        self.training_mode = self.config['training_mode']
        self._init_agents()
        
        # 训练统计
        self.episode_rewards = []
        self.diag_rewards_history = []
        self.ctrl_rewards_history = []
        self.detection_delays = []
        self.diagnosis_accuracies = []
        self.Pmax_violations = []
        self.training_losses = []
        
        # 当前episode统计
        self.current_episode = 0
        
        print(f"\n{'='*60}")
        print(f"双智能体训练器初始化完成")
        print(f"{'='*60}")
        print(f"训练模式: {self.training_mode}")
        print(f"诊断状态维度: {self.diag_state_dim}")
        print(f"诊断动作维度: {self.diag_action_dim}")
        print(f"控制状态维度: {self.ctrl_state_dim}")
        print(f"控制动作维度: {self.ctrl_action_dim}")
        print(f"{'='*60}\n")
    
    def _init_agents(self):
        """初始化智能体"""
        if self.training_mode == 'independent':
            self._init_independent_agents()
        elif self.training_mode == 'mappo':
            self._init_mappo_agent()
        elif self.training_mode == 'qmix':
            self._init_qmix_agent()
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")
    
    def _init_independent_agents(self):
        """初始化独立学习智能体"""
        # 诊断智能体 (SAC)
        self.diag_agent = create_rl_diagnosis_agent({
            'state_dim': self.diag_state_dim,
            'lr': self.config['diag_lr'],
            'gamma': self.config['diag_gamma'],
            'tau': self.config['diag_tau'],
            'batch_size': self.config['diag_batch_size'],
            'buffer_size': self.config['diag_buffer_size']
        })
        
        # 控制智能体 (SAC)
        if SAC_AVAILABLE:
            self.ctrl_agent = SAC(
                self.ctrl_state_dim, 
                self.ctrl_action_dim,
                {
                    'lr': self.config['ctrl_lr'],
                    'gamma': self.config['ctrl_gamma'],
                    'tau': self.config['ctrl_tau'],
                    'batch_size': self.config['ctrl_batch_size'],
                    'buffer_size': self.config['ctrl_buffer_size']
                }
            )
        else:
            self.ctrl_agent = None
            print("[警告] 控制智能体使用随机策略")
        
        self.ma_agent = None
    
    def _init_mappo_agent(self):
        """初始化MAPPO智能体"""
        self.ma_agent = get_multi_agent_algorithm(
            'MAPPO',
            self.diag_state_dim,
            self.diag_action_dim,
            self.ctrl_state_dim,
            self.ctrl_action_dim,
            {
                'lr_actor': self.config['ma_lr'],
                'lr_critic': self.config['ma_lr'],
                'gamma': self.config['ma_gamma'],
                'gae_lambda': self.config['ma_gae_lambda'],
                'clip_epsilon': self.config['ma_clip_epsilon'],
                'ppo_epochs': self.config['ma_ppo_epochs'],
                'batch_size': self.config['ma_batch_size'],
                'sequence_len': 10
            }
        )
        self.diag_agent = None
        self.ctrl_agent = None
    
    def _init_qmix_agent(self):
        """初始化QMIX智能体"""
        self.ma_agent = get_multi_agent_algorithm(
            'QMIX',
            self.diag_state_dim,
            self.diag_action_dim,
            self.ctrl_state_dim,
            self.ctrl_action_dim,
            {
                'lr': self.config['ma_lr'],
                'gamma': self.config['ma_gamma'],
                'tau': self.config['diag_tau'],
                'batch_size': self.config['ma_batch_size'],
                'buffer_size': self.config['diag_buffer_size'],
                'sequence_len': 10
            }
        )
        self.diag_agent = None
        self.ctrl_agent = None
    
    def train(self) -> Dict:
        """
        执行训练
        
        Returns:
            训练统计字典
        """
        start_time = time.time()
        
        for episode in range(self.config['n_episodes']):
            self.current_episode = episode
            
            # 运行一个episode
            episode_stats = self._run_episode(training=True)
            
            # 记录统计
            self.episode_rewards.append(episode_stats['total_reward'])
            self.diag_rewards_history.append(episode_stats['diag_reward'])
            self.ctrl_rewards_history.append(episode_stats['ctrl_reward'])
            self.diagnosis_accuracies.append(episode_stats['diag_accuracy'])
            self.Pmax_violations.append(episode_stats['Pmax_violations'])
            
            if episode_stats['detection_delay'] is not None:
                self.detection_delays.append(episode_stats['detection_delay'])
            
            # 日志
            if (episode + 1) % self.config['log_interval'] == 0:
                self._log_progress(episode, episode_stats)
            
            # 评估
            if (episode + 1) % self.config['eval_interval'] == 0:
                eval_stats = self.evaluate(n_episodes=10)
                self._log_evaluation(episode, eval_stats)
            
            # 保存
            if (episode + 1) % self.config['save_interval'] == 0:
                self.save(f"checkpoint_ep{episode+1}")
        
        # 训练完成
        total_time = time.time() - start_time
        
        # 最终保存
        self.save("final")
        
        # 返回训练统计
        return {
            'total_episodes': self.config['n_episodes'],
            'total_time': total_time,
            'avg_reward': np.mean(self.episode_rewards[-100:]),
            'avg_diag_accuracy': np.mean(self.diagnosis_accuracies[-100:]),
            'avg_detection_delay': np.mean(self.detection_delays[-100:]) if self.detection_delays else None,
            'avg_Pmax_violations': np.mean(self.Pmax_violations[-100:])
        }
    
    def _run_episode(self, training: bool = True) -> Dict:
        """运行一个episode"""
        # 重置环境
        obs, info = self.env.reset()
        
        # 重置智能体
        if self.diag_agent is not None:
            self.diag_agent.reset()
        
        # Episode统计
        total_diag_reward = 0
        total_ctrl_reward = 0
        total_joint_reward = 0
        n_correct = 0
        n_total = 0
        n_Pmax_violations = 0
        detection_delay = None
        
        # 存储轨迹 (用于MAPPO)
        trajectory = []
        
        # 上一步状态
        prev_obs = obs
        
        for step in range(self.config['max_steps_per_episode']):
            # 选择动作
            if self.training_mode == 'independent':
                diag_action, ctrl_action, action_info = self._select_actions_independent(
                    obs, training
                )
            else:
                diag_action, ctrl_action, action_info = self._select_actions_multi_agent(
                    obs, training
                )
            
            # 执行动作
            next_obs, diag_reward, ctrl_reward, done, info = self.env.step(
                diag_action, ctrl_action
            )
            
            # 计算联合奖励
            joint_reward = self.env.get_joint_reward(
                diag_reward, ctrl_reward,
                info.diagnosis_correct or False,
                self.config['diag_reward_weight'],
                self.config['ctrl_reward_weight'],
                self.config['coop_reward_weight']
            )
            
            # 统计
            total_diag_reward += diag_reward
            total_ctrl_reward += ctrl_reward
            total_joint_reward += joint_reward
            
            if info.diagnosis_correct is not None:
                n_total += 1
                if info.diagnosis_correct:
                    n_correct += 1
            
            if info.Pmax_violation:
                n_Pmax_violations += 1
            
            if info.detection_delay is not None and detection_delay is None:
                detection_delay = info.detection_delay
            
            # 存储经验
            if training:
                if self.training_mode == 'independent':
                    self._store_experience_independent(
                        prev_obs, diag_action, ctrl_action,
                        diag_reward, ctrl_reward,
                        next_obs, done
                    )
                else:
                    experience = MultiAgentExperience(
                        diag_state=prev_obs.diag_state,
                        diag_residual_seq=prev_obs.diag_residual_seq,
                        diag_action=diag_action,
                        diag_reward=diag_reward,
                        diag_log_prob=action_info.get('diag_log_prob', 0),
                        diag_value=action_info.get('value', 0),
                        ctrl_state=prev_obs.ctrl_state,
                        ctrl_action=ctrl_action,
                        ctrl_reward=ctrl_reward,
                        ctrl_log_prob=action_info.get('ctrl_log_prob', 0),
                        ctrl_value=action_info.get('value', 0),
                        global_state=prev_obs.global_state,
                        next_global_state=next_obs.global_state,
                        done=done,
                        joint_reward=joint_reward
                    )
                    
                    if self.training_mode == 'mappo':
                        self.ma_agent.store_transition(experience)
                    else:
                        self.ma_agent.store_experience(experience)
            
            prev_obs = next_obs
            
            if done:
                break
        
        # 训练更新
        if training:
            losses = self._update()
            self.training_losses.append(losses)
        
        # 计算诊断准确率
        diag_accuracy = n_correct / n_total if n_total > 0 else 0
        
        return {
            'total_reward': total_joint_reward,
            'diag_reward': total_diag_reward,
            'ctrl_reward': total_ctrl_reward,
            'diag_accuracy': diag_accuracy,
            'detection_delay': detection_delay,
            'Pmax_violations': n_Pmax_violations,
            'steps': step + 1,
            'ground_truth': info.ground_truth_fault.name,
            'fault_severity': info.fault_severity
        }
    
    def _select_actions_independent(self, obs, training: bool) -> Tuple[int, int, Dict]:
        """独立学习模式下选择动作"""
        # 诊断动作
        diag_action = self.diag_agent.select_action(
            obs.diag_state, obs.diag_residual_seq, explore=training
        )
        
        # 控制动作
        if self.ctrl_agent is not None:
            ctrl_action = self.ctrl_agent.select_action(
                obs.ctrl_state, explore=training
            )
        else:
            ctrl_action = np.random.randint(self.ctrl_action_dim)
        
        return diag_action, ctrl_action, {}
    
    def _select_actions_multi_agent(self, obs, training: bool) -> Tuple[int, int, Dict]:
        """多智能体模式下选择动作"""
        return self.ma_agent.select_actions(
            obs.diag_state,
            obs.diag_residual_seq,
            obs.ctrl_state,
            explore=training
        )
    
    def _store_experience_independent(self, prev_obs, diag_action, ctrl_action,
                                     diag_reward, ctrl_reward, next_obs, done):
        """独立学习模式下存储经验"""
        # 诊断智能体经验
        self.diag_agent.store_experience(
            prev_obs.diag_state,
            prev_obs.diag_residual_seq,
            diag_action,
            diag_reward,
            next_obs.diag_state,
            next_obs.diag_residual_seq,
            done
        )
        
        # 控制智能体经验
        if self.ctrl_agent is not None:
            from agents.rl_algorithms import Experience
            self.ctrl_agent.buffer.push(
                prev_obs.ctrl_state,
                ctrl_action,
                ctrl_reward,
                next_obs.ctrl_state,
                done
            )
    
    def _update(self) -> Dict:
        """执行更新"""
        losses = {}
        
        if self.training_mode == 'independent':
            # 诊断智能体更新
            diag_loss = self.diag_agent.update()
            losses['diag'] = diag_loss
            
            # 控制智能体更新
            if self.ctrl_agent is not None:
                batch = self.ctrl_agent.buffer.sample(self.config['ctrl_batch_size'])
                ctrl_loss = self.ctrl_agent.update(batch)
                losses['ctrl'] = ctrl_loss
        else:
            # 多智能体更新
            ma_loss = self.ma_agent.update()
            losses['ma'] = ma_loss
        
        return losses
    
    def evaluate(self, n_episodes: int = 20) -> Dict:
        """
        评估当前策略
        
        Args:
            n_episodes: 评估episode数
        
        Returns:
            评估统计
        """
        rewards = []
        diag_accuracies = []
        detection_delays = []
        Pmax_violations = []
        
        for _ in range(n_episodes):
            stats = self._run_episode(training=False)
            rewards.append(stats['total_reward'])
            diag_accuracies.append(stats['diag_accuracy'])
            if stats['detection_delay'] is not None:
                detection_delays.append(stats['detection_delay'])
            Pmax_violations.append(stats['Pmax_violations'])
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_diag_accuracy': np.mean(diag_accuracies),
            'avg_detection_delay': np.mean(detection_delays) if detection_delays else None,
            'avg_Pmax_violations': np.mean(Pmax_violations)
        }
    
    def _log_progress(self, episode: int, stats: Dict):
        """记录训练进度"""
        avg_reward = np.mean(self.episode_rewards[-self.config['log_interval']:])
        avg_diag_acc = np.mean(self.diagnosis_accuracies[-self.config['log_interval']:])
        
        print(f"Episode {episode+1}/{self.config['n_episodes']} | "
              f"Reward: {stats['total_reward']:.2f} (avg: {avg_reward:.2f}) | "
              f"DiagAcc: {stats['diag_accuracy']:.1%} (avg: {avg_diag_acc:.1%}) | "
              f"GT: {stats['ground_truth']} | "
              f"Violations: {stats['Pmax_violations']}")
    
    def _log_evaluation(self, episode: int, stats: Dict):
        """记录评估结果"""
        print(f"\n{'='*50}")
        print(f"评估 @ Episode {episode+1}")
        print(f"  平均奖励: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  诊断准确率: {stats['avg_diag_accuracy']:.1%}")
        if stats['avg_detection_delay'] is not None:
            print(f"  平均检测延迟: {stats['avg_detection_delay']:.1f} 步")
        print(f"  Pmax违规次数: {stats['avg_Pmax_violations']:.1f}")
        print(f"{'='*50}\n")
    
    def save(self, name: str):
        """保存模型和训练状态"""
        save_path = os.path.join(self.config['save_dir'], name)
        os.makedirs(save_path, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # 保存模型
        if self.training_mode == 'independent':
            self.diag_agent.save(os.path.join(save_path, 'diag_agent.pt'))
            if self.ctrl_agent is not None:
                self.ctrl_agent.save(os.path.join(save_path, 'ctrl_agent.pt'))
        else:
            self.ma_agent.save(os.path.join(save_path, 'ma_agent.pt'))
        
        # 保存训练统计
        stats = {
            'episode_rewards': self.episode_rewards,
            'diag_rewards': self.diag_rewards_history,
            'ctrl_rewards': self.ctrl_rewards_history,
            'diagnosis_accuracies': self.diagnosis_accuracies,
            'detection_delays': self.detection_delays,
            'Pmax_violations': self.Pmax_violations
        }
        np.save(os.path.join(save_path, 'training_stats.npy'), stats)
        
        print(f"[保存] 模型已保存至: {save_path}")
    
    def load(self, name: str):
        """加载模型"""
        load_path = os.path.join(self.config['save_dir'], name)
        
        if self.training_mode == 'independent':
            self.diag_agent.load(os.path.join(load_path, 'diag_agent.pt'))
            if self.ctrl_agent is not None:
                self.ctrl_agent.load(os.path.join(load_path, 'ctrl_agent.pt'))
        else:
            self.ma_agent.load(os.path.join(load_path, 'ma_agent.pt'))
        
        # 加载训练统计
        stats = np.load(os.path.join(load_path, 'training_stats.npy'), 
                       allow_pickle=True).item()
        self.episode_rewards = stats['episode_rewards']
        self.diag_rewards_history = stats['diag_rewards']
        self.ctrl_rewards_history = stats['ctrl_rewards']
        self.diagnosis_accuracies = stats['diagnosis_accuracies']
        self.detection_delays = stats['detection_delays']
        self.Pmax_violations = stats['Pmax_violations']
        
        print(f"[加载] 模型已加载自: {load_path}")
    
    def get_training_curves(self) -> Dict:
        """获取训练曲线数据"""
        return {
            'episode_rewards': self.episode_rewards,
            'diag_rewards': self.diag_rewards_history,
            'ctrl_rewards': self.ctrl_rewards_history,
            'diagnosis_accuracies': self.diagnosis_accuracies,
            'detection_delays': self.detection_delays,
            'Pmax_violations': self.Pmax_violations
        }


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='双智能体训练')
    parser.add_argument('--mode', type=str, default='mappo',
                       choices=['independent', 'mappo', 'qmix'],
                       help='训练模式')
    parser.add_argument('--episodes', type=int, default=500,
                       help='训练episode数')
    parser.add_argument('--eval-interval', type=int, default=50,
                       help='评估间隔')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='保存间隔')
    parser.add_argument('--save-dir', type=str, default='models/dual_agent',
                       help='保存目录')
    parser.add_argument('--load', type=str, default=None,
                       help='加载已有模型')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'training_mode': args.mode,
        'n_episodes': args.episodes,
        'eval_interval': args.eval_interval,
        'save_interval': args.save_interval,
        'save_dir': args.save_dir
    }
    
    # 创建训练器
    trainer = DualAgentTrainer(config)
    
    # 加载模型
    if args.load:
        trainer.load(args.load)
    
    # 训练
    print(f"\n开始训练 ({args.mode} 模式)...")
    print(f"计划运行 {args.episodes} 个episodes\n")
    
    try:
        stats = trainer.train()
        
        print(f"\n{'='*60}")
        print("训练完成!")
        print(f"{'='*60}")
        print(f"总episodes: {stats['total_episodes']}")
        print(f"总时间: {stats['total_time']:.1f}秒")
        print(f"平均奖励 (最后100): {stats['avg_reward']:.2f}")
        print(f"平均诊断准确率 (最后100): {stats['avg_diag_accuracy']:.1%}")
        if stats['avg_detection_delay'] is not None:
            print(f"平均检测延迟 (最后100): {stats['avg_detection_delay']:.1f}步")
        print(f"平均Pmax违规 (最后100): {stats['avg_Pmax_violations']:.1f}")
        
    except KeyboardInterrupt:
        print("\n训练被中断，保存当前进度...")
        trainer.save("interrupted")
        print("已保存!")


if __name__ == "__main__":
    main()
