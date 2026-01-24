"""
MAPPO训练器
============
双智能体并行训练
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import time

from ..env import EngineEnv, EngineEnvConfig
from ..agents import DiagnosticAgent, ControlAgent
from .buffer import RolloutBuffer


class MAPPOTrainer:
    """
    Multi-Agent PPO训练器
    
    并行训练诊断和控制智能体
    """
    
    def __init__(
        self,
        env_config: Optional[EngineEnvConfig] = None,
        diag_lr: float = 3e-4,
        ctrl_lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 2048,
        device: str = 'cpu',
        save_dir: str = './checkpoints'
    ):
        """
        初始化训练器
        """
        self.device = torch.device(device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # 创建环境
        self.env_config = env_config or EngineEnvConfig()
        self.env = EngineEnv(self.env_config)
        
        # 获取观测维度
        diag_obs_dim, ctrl_obs_dim = self.env.get_observation_dims()
        
        # 创建智能体
        self.diag_agent = DiagnosticAgent(
            obs_dim=diag_obs_dim,
            n_fault_types=self.env_config.n_fault_types,
            lr=diag_lr,
            device=device
        )
        
        self.ctrl_agent = ControlAgent(
            obs_dim=ctrl_obs_dim,
            timing_range=self.env_config.timing_offset_range,
            fuel_range=self.env_config.fuel_adj_range,
            n_protection_levels=self.env_config.n_protection_levels,
            lr=ctrl_lr,
            device=device
        )
        
        # 经验缓冲区
        self.buffer = RolloutBuffer(buffer_size, gamma, gae_lambda)
        
        # 训练统计
        self.episode_count = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
    
    def collect_rollout(self, n_steps: int) -> Dict[str, float]:
        """
        收集轨迹数据
        
        Args:
            n_steps: 收集步数
            
        Returns:
            统计信息
        """
        self.buffer.clear()
        
        episode_rewards_diag = []
        episode_rewards_ctrl = []
        episode_lengths = []
        
        obs, info = self.env.reset(options={'difficulty': self._get_difficulty()})
        ep_reward_diag = 0
        ep_reward_ctrl = 0
        ep_length = 0
        
        for step in range(n_steps):
            # 诊断智能体决策
            diag_action = self.diag_agent.act(obs['diag'])
            diag_value = self.diag_agent.get_value(obs['diag'])
            
            # 控制智能体决策
            ctrl_action = self.ctrl_agent.act(obs['ctrl'])
            ctrl_value = self.ctrl_agent.get_value(obs['ctrl'])
            
            # 执行动作
            actions = {'diag': diag_action, 'ctrl': ctrl_action}
            next_obs, rewards, terminated, truncated, info = self.env.step(actions)
            
            done = terminated or truncated
            
            # 计算log_prob（简化处理）
            log_prob_diag = 0.0  # 实际应从网络获取
            log_prob_ctrl = 0.0
            
            # 存储转移
            self.buffer.add(
                obs_diag=obs['diag'],
                obs_ctrl=obs['ctrl'],
                action_diag=diag_action,
                action_ctrl=ctrl_action,
                log_prob_diag=log_prob_diag,
                log_prob_ctrl=log_prob_ctrl,
                reward_diag=rewards['diag'],
                reward_ctrl=rewards['ctrl'],
                value_diag=diag_value,
                value_ctrl=ctrl_value,
                done=done
            )
            
            ep_reward_diag += rewards['diag']
            ep_reward_ctrl += rewards['ctrl']
            ep_length += 1
            self.total_steps += 1
            
            obs = next_obs
            
            if done:
                episode_rewards_diag.append(ep_reward_diag)
                episode_rewards_ctrl.append(ep_reward_ctrl)
                episode_lengths.append(ep_length)
                self.episode_count += 1
                
                obs, info = self.env.reset(options={'difficulty': self._get_difficulty()})
                ep_reward_diag = 0
                ep_reward_ctrl = 0
                ep_length = 0
        
        # 计算最后状态的价值
        last_value_diag = self.diag_agent.get_value(obs['diag'])
        last_value_ctrl = self.ctrl_agent.get_value(obs['ctrl'])
        
        # 计算优势和回报
        self.buffer.compute_returns_and_advantages(last_value_diag, last_value_ctrl)
        
        return {
            'mean_reward_diag': np.mean(episode_rewards_diag) if episode_rewards_diag else 0,
            'mean_reward_ctrl': np.mean(episode_rewards_ctrl) if episode_rewards_ctrl else 0,
            'mean_ep_length': np.mean(episode_lengths) if episode_lengths else 0,
            'n_episodes': len(episode_rewards_diag)
        }
    
    def update(self) -> Dict[str, float]:
        """
        更新两个智能体
        """
        diag_losses = {'policy_loss': [], 'value_loss': [], 'entropy': []}
        ctrl_losses = {'policy_loss': [], 'value_loss': [], 'entropy': []}
        
        for epoch in range(self.n_epochs):
            for diag_batch, ctrl_batch in self.buffer.get_batches(self.batch_size, self.device):
                # 更新诊断智能体
                diag_loss = self.diag_agent.update(
                    obs_batch=diag_batch['obs'],
                    action_batch=diag_batch['actions'],
                    old_log_probs=diag_batch['log_probs'],
                    advantages=diag_batch['advantages'],
                    returns=diag_batch['returns'],
                    clip_epsilon=self.clip_epsilon,
                    entropy_coef=self.entropy_coef,
                    value_coef=self.value_coef
                )
                
                for k, v in diag_loss.items():
                    diag_losses[k].append(v)
                
                # 更新控制智能体
                ctrl_loss = self.ctrl_agent.update(
                    obs_batch=ctrl_batch['obs'],
                    action_batch=ctrl_batch['actions'],
                    old_log_probs=ctrl_batch['log_probs'],
                    advantages=ctrl_batch['advantages'],
                    returns=ctrl_batch['returns'],
                    clip_epsilon=self.clip_epsilon,
                    entropy_coef=self.entropy_coef,
                    value_coef=self.value_coef
                )
                
                for k, v in ctrl_loss.items():
                    ctrl_losses[k].append(v)
        
        return {
            'diag_policy_loss': np.mean(diag_losses['policy_loss']),
            'diag_value_loss': np.mean(diag_losses['value_loss']),
            'diag_entropy': np.mean(diag_losses['entropy']),
            'ctrl_policy_loss': np.mean(ctrl_losses['policy_loss']),
            'ctrl_value_loss': np.mean(ctrl_losses['value_loss']),
            'ctrl_entropy': np.mean(ctrl_losses['entropy'])
        }
    
    def train(
        self,
        total_timesteps: int,
        rollout_steps: int = 2048,
        log_interval: int = 1,
        save_interval: int = 10
    ):
        """
        主训练循环
        """
        n_updates = total_timesteps // rollout_steps
        
        print(f"开始训练: {total_timesteps} 步, {n_updates} 次更新")
        print(f"设备: {self.device}")
        
        for update in range(1, n_updates + 1):
            start_time = time.time()
            
            # 收集轨迹
            rollout_stats = self.collect_rollout(rollout_steps)
            
            # 更新网络
            update_stats = self.update()
            
            elapsed = time.time() - start_time
            
            # 记录
            if update % log_interval == 0:
                total_reward = rollout_stats['mean_reward_diag'] + rollout_stats['mean_reward_ctrl']
                print(f"更新 {update}/{n_updates} | "
                      f"诊断奖励: {rollout_stats['mean_reward_diag']:.2f} | "
                      f"控制奖励: {rollout_stats['mean_reward_ctrl']:.2f} | "
                      f"轮数: {rollout_stats['n_episodes']} | "
                      f"时间: {elapsed:.2f}s")
                
                # 保存最佳模型
                if total_reward > self.best_reward:
                    self.best_reward = total_reward
                    self.save('best')
            
            # 定期保存
            if update % save_interval == 0:
                self.save(f'checkpoint_{update}')
        
        print("训练完成!")
        self.save('final')
    
    def _get_difficulty(self) -> float:
        """根据训练进度调整难度（课程学习）"""
        return min(1.0, self.episode_count / 500)
    
    def save(self, name: str):
        """保存模型"""
        self.diag_agent.save(self.save_dir / f'{name}_diag.pt')
        self.ctrl_agent.save(self.save_dir / f'{name}_ctrl.pt')
    
    def load(self, name: str):
        """加载模型"""
        self.diag_agent.load(self.save_dir / f'{name}_diag.pt')
        self.ctrl_agent.load(self.save_dir / f'{name}_ctrl.pt')
