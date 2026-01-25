"""
MAPPO训练器
============
真正的Multi-Agent PPO实现
- 集中式Critic（共享价值网络）
- 串行化诊断->控制架构
- 正确的log_prob处理
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import time

from ..env import EngineEnv, EngineEnvConfig
from ..agents import DiagnosticAgent, ControlAgent
from ..networks import SharedCritic, PIKANDiagnosticAgent
from .buffer import RolloutBuffer


class MAPPOTrainer:
    """
    Multi-Agent PPO训练器
    
    特性：
    1. 集中式Critic（CTDE架构）
    2. 串行化决策：诊断 -> 控制
    3. 正确的log_prob传递
    """
    
    def __init__(
        self,
        env_config: Optional[EngineEnvConfig] = None,
        use_pikan: bool = False,
        physics_weight: float = 0.1,
        diag_lr: float = 3e-4,
        ctrl_lr: float = 3e-4,
        critic_lr: float = 3e-4,
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
        """初始化训练器"""
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
        
        # 诊断结果特征维度（故障类型one-hot + 严重程度 + 置信度）
        self.diag_feature_dim = self.env_config.n_fault_types + 2
        
        # 控制智能体的扩展观测维度（包含诊断结果）
        ctrl_obs_dim_extended = ctrl_obs_dim + self.diag_feature_dim
        
        # 保存维度信息
        self.diag_obs_dim = diag_obs_dim
        self.ctrl_obs_dim = ctrl_obs_dim
        self.ctrl_obs_dim_extended = ctrl_obs_dim_extended
        self.use_pikan = use_pikan
        self.physics_weight = physics_weight
        
        # 创建诊断智能体（根据配置选择类型）
        if use_pikan:
            self.diag_agent = PIKANDiagnosticAgent(
                obs_dim=diag_obs_dim,
                n_fault_types=self.env_config.n_fault_types,
                lr=diag_lr,
                physics_weight=physics_weight,
                device=device
            )
            print("[PIKAN] 使用 Physics-Informed KAN 诊断智能体")
        else:
            self.diag_agent = DiagnosticAgent(
                obs_dim=diag_obs_dim,
                n_fault_types=self.env_config.n_fault_types,
                lr=diag_lr,
                device=device
            )
        
        self.ctrl_agent = ControlAgent(
            obs_dim=ctrl_obs_dim_extended,  # 使用扩展维度
            timing_range=self.env_config.timing_offset_range,
            fuel_range=self.env_config.fuel_adj_range,
            n_protection_levels=self.env_config.n_protection_levels,
            lr=ctrl_lr,
            device=device
        )
        
        # 共享Critic（MAPPO核心）
        self.shared_critic = SharedCritic(
            diag_obs_dim=diag_obs_dim,
            ctrl_obs_dim=ctrl_obs_dim_extended,
            hidden_dim=256
        ).to(self.device)
        
        self.critic_optimizer = torch.optim.Adam(
            self.shared_critic.parameters(), lr=critic_lr
        )
        
        # 经验缓冲区
        self.buffer = RolloutBuffer(buffer_size, gamma, gae_lambda)
        
        # 训练统计
        self.episode_count = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
    
    def _encode_diag_result(self, diag_result: Dict) -> np.ndarray:
        """
        将诊断结果编码为特征向量
        
        Args:
            diag_result: 诊断智能体输出
            
        Returns:
            特征向量 [n_fault_types + 2]
        """
        # One-hot编码故障类型
        fault_onehot = np.zeros(self.env_config.n_fault_types)
        fault_onehot[diag_result['fault_type']] = 1.0
        
        # 拼接严重程度和置信度
        features = np.concatenate([
            fault_onehot,
            [diag_result['severity']],
            [diag_result['confidence']]
        ])
        
        return features
    
    def collect_rollout(self, n_steps: int) -> Dict[str, float]:
        """
        收集轨迹数据（串行化架构）
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
            # ===== 串行化决策流程 =====
            
            # 第一步：诊断智能体决策
            diag_result = self.diag_agent.act(obs['diag'])
            log_prob_diag = diag_result['log_prob']  # 真实log_prob
            
            # 第二步：编码诊断结果并拼接到控制观测
            diag_features = self._encode_diag_result(diag_result)
            ctrl_obs_extended = np.concatenate([obs['ctrl'], diag_features])
            
            # 第三步：控制智能体基于增强观测决策
            ctrl_result = self.ctrl_agent.act(ctrl_obs_extended)
            log_prob_ctrl = ctrl_result['log_prob']  # 真实log_prob
            
            # 第四步：使用共享Critic计算价值
            with torch.no_grad():
                obs_diag_t = torch.FloatTensor(obs['diag']).unsqueeze(0).to(self.device)
                obs_ctrl_t = torch.FloatTensor(ctrl_obs_extended).unsqueeze(0).to(self.device)
                value_diag, value_ctrl = self.shared_critic(obs_diag_t, obs_ctrl_t)
                diag_value = value_diag.item()
                ctrl_value = value_ctrl.item()
            
            # 构建动作字典
            diag_action = {
                'fault_type': diag_result['fault_type'],
                'severity': diag_result['severity'],
                'confidence': diag_result['confidence']
            }
            ctrl_action = {
                'timing_offset': ctrl_result['timing_offset'],
                'fuel_adj': ctrl_result['fuel_adj'],
                'protection_level': ctrl_result['protection_level']
            }
            
            # 执行动作
            actions = {'diag': diag_action, 'ctrl': ctrl_action}
            next_obs, rewards, terminated, truncated, info = self.env.step(actions)
            
            done = terminated or truncated
            
            # 存储转移（使用真实log_prob）
            self.buffer.add(
                obs_diag=obs['diag'],
                obs_ctrl=ctrl_obs_extended,  # 存储扩展观测
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
        
        # 计算最后状态的价值（使用共享Critic）
        diag_features = self._encode_diag_result(self.diag_agent.act(obs['diag']))
        ctrl_obs_extended = np.concatenate([obs['ctrl'], diag_features])
        
        with torch.no_grad():
            obs_diag_t = torch.FloatTensor(obs['diag']).unsqueeze(0).to(self.device)
            obs_ctrl_t = torch.FloatTensor(ctrl_obs_extended).unsqueeze(0).to(self.device)
            last_value_diag, last_value_ctrl = self.shared_critic(obs_diag_t, obs_ctrl_t)
            last_value_diag = last_value_diag.item()
            last_value_ctrl = last_value_ctrl.item()
        
        # 计算优势和回报
        self.buffer.compute_returns_and_advantages(last_value_diag, last_value_ctrl)
        
        return {
            'mean_reward_diag': np.mean(episode_rewards_diag) if episode_rewards_diag else 0,
            'mean_reward_ctrl': np.mean(episode_rewards_ctrl) if episode_rewards_ctrl else 0,
            'mean_ep_length': np.mean(episode_lengths) if episode_lengths else 0,
            'n_episodes': len(episode_rewards_diag)
        }
    
    def update(self) -> Dict[str, float]:
        """更新所有网络（Actor + 共享Critic）"""
        diag_losses = {'policy_loss': [], 'value_loss': [], 'entropy': []}
        ctrl_losses = {'policy_loss': [], 'value_loss': [], 'entropy': []}
        critic_losses = []
        
        for epoch in range(self.n_epochs):
            for diag_batch, ctrl_batch in self.buffer.get_batches(self.batch_size, self.device):
                # ===== 更新共享Critic =====
                value_diag, value_ctrl = self.shared_critic(
                    diag_batch['obs'], ctrl_batch['obs']
                )
                
                critic_loss = (
                    nn.functional.mse_loss(value_diag.squeeze(), diag_batch['returns']) +
                    nn.functional.mse_loss(value_ctrl.squeeze(), ctrl_batch['returns'])
                )
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.shared_critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                critic_losses.append(critic_loss.item())
                
                # ===== 更新诊断智能体Actor =====
                diag_loss = self.diag_agent.update(
                    obs_batch=diag_batch['obs'],
                    action_batch=diag_batch['actions'],
                    old_log_probs=diag_batch['log_probs'],
                    advantages=diag_batch['advantages'],
                    returns=diag_batch['returns'],
                    clip_epsilon=self.clip_epsilon,
                    entropy_coef=self.entropy_coef,
                    value_coef=0.0  # 价值由共享Critic计算，这里不更新
                )
                
                for k, v in diag_loss.items():
                    if k in diag_losses:
                        diag_losses[k].append(v)
                
                # ===== 更新控制智能体Actor =====
                ctrl_loss = self.ctrl_agent.update(
                    obs_batch=ctrl_batch['obs'],
                    action_batch=ctrl_batch['actions'],
                    old_log_probs=ctrl_batch['log_probs'],
                    advantages=ctrl_batch['advantages'],
                    returns=ctrl_batch['returns'],
                    clip_epsilon=self.clip_epsilon,
                    entropy_coef=self.entropy_coef,
                    value_coef=0.0  # 价值由共享Critic计算
                )
                
                for k, v in ctrl_loss.items():
                    if k in ctrl_losses:
                        ctrl_losses[k].append(v)
        
        return {
            'diag_policy_loss': np.mean(diag_losses['policy_loss']),
            'diag_entropy': np.mean(diag_losses['entropy']),
            'ctrl_policy_loss': np.mean(ctrl_losses['policy_loss']),
            'ctrl_entropy': np.mean(ctrl_losses['entropy']),
            'critic_loss': np.mean(critic_losses)
        }
    
    def train(
        self,
        total_timesteps: int,
        rollout_steps: int = 2048,
        log_interval: int = 1,
        save_interval: int = 10
    ):
        """主训练循环"""
        n_updates = total_timesteps // rollout_steps
        
        print(f"开始训练: {total_timesteps} 步, {n_updates} 次更新")
        print(f"设备: {self.device}")
        print(f"诊断观测维度: {self.diag_obs_dim}")
        print(f"控制观测维度: {self.ctrl_obs_dim_extended} (原 {self.ctrl_obs_dim} + 诊断特征 {self.diag_feature_dim})")
        print(f"架构: 串行化诊断->控制 + MAPPO共享Critic")
        
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
                      f"Critic损失: {update_stats['critic_loss']:.4f} | "
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
        torch.save({
            'shared_critic': self.shared_critic.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, self.save_dir / f'{name}_critic.pt')
    
    def load(self, name: str):
        """加载模型"""
        self.diag_agent.load(self.save_dir / f'{name}_diag.pt')
        self.ctrl_agent.load(self.save_dir / f'{name}_ctrl.pt')
        checkpoint = torch.load(self.save_dir / f'{name}_critic.pt', map_location=self.device)
        self.shared_critic.load_state_dict(checkpoint['shared_critic'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
