"""
完整训练和评估脚本
==================
训练PINN+KAN诊断智能体和TD-MPC2控制智能体
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import json
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from marl.env import EngineEnv, EngineEnvConfig
from marl.networks.pinn_kan import PIKANDiagnosticAgent
from marl.agents.tdmpc2_controller import TDMPC2Controller, TDMPC2Config
from marl.training import RolloutBuffer
from marl.utils.visualization import TrainingVisualizer, EvaluationVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description='PINN+KAN & TD-MPC2 训练')
    
    parser.add_argument('--total-steps', type=int, default=100000)
    parser.add_argument('--rollout-steps', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=64)
    
    parser.add_argument('--diag-lr', type=float, default=3e-4)
    parser.add_argument('--ctrl-lr', type=float, default=3e-4)
    parser.add_argument('--physics-weight', type=float, default=0.1)
    
    parser.add_argument('--kan-hidden', type=int, nargs='+', default=[32, 32])
    parser.add_argument('--kan-grid', type=int, default=5)
    
    parser.add_argument('--mpc-horizon', type=int, default=5)
    parser.add_argument('--mpc-samples', type=int, default=256)
    
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save-dir', type=str, default='./checkpoints/pikan_tdmpc2')
    
    return parser.parse_args()


class AdvancedTrainer:
    """PINN+KAN 和 TD-MPC2 联合训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 环境
        self.env_config = EngineEnvConfig(
            max_steps=200,
            difficulty=0.5,
            enable_variable_condition=True
        )
        self.env = EngineEnv(self.env_config)
        
        # 获取维度
        diag_dim, ctrl_dim = self.env.get_observation_dims()
        
        # 诊断智能体（PINN+KAN）
        self.diag_agent = PIKANDiagnosticAgent(
            obs_dim=diag_dim,
            n_fault_types=4,
            kan_hidden_dims=args.kan_hidden,
            grid_size=args.kan_grid,
            lr=args.diag_lr,
            physics_weight=args.physics_weight,
            device=args.device
        )
        
        # 控制智能体（TD-MPC2）
        tdmpc_config = TDMPC2Config(
            obs_dim=ctrl_dim,
            action_dim=3,
            horizon=args.mpc_horizon,
            n_samples=args.mpc_samples,
            lr_world=args.ctrl_lr,
            lr_policy=args.ctrl_lr,
            lr_value=args.ctrl_lr
        )
        self.ctrl_agent = TDMPC2Controller(tdmpc_config, device=args.device)
        
        # 可视化
        self.visualizer = TrainingVisualizer(save_dir=str(self.save_dir / 'plots'))
        
        # 训练统计
        self.total_steps = 0
        self.episode_count = 0
    
    def collect_experience(self, n_steps: int):
        """收集经验数据"""
        obs, info = self.env.reset(options={'difficulty': self._get_difficulty()})
        
        experiences = {
            'obs_diag': [], 'obs_ctrl': [],
            'action_diag': [], 'action_ctrl': [],
            'reward_diag': [], 'reward_ctrl': [],
            'next_obs_diag': [], 'next_obs_ctrl': [],
            'done': []
        }
        
        episode_rewards = []
        ep_reward = 0
        
        for _ in range(n_steps):
            # 诊断
            diag_action = self.diag_agent.act(obs['diag'])
            
            # 控制（使用诊断结果）
            ctrl_action = self.ctrl_agent.act(obs['ctrl'], diag_action, use_planning=True)
            
            # 执行
            actions = {'diag': diag_action, 'ctrl': ctrl_action}
            next_obs, rewards, terminated, truncated, info = self.env.step(actions)
            done = terminated or truncated
            
            # 存储
            experiences['obs_diag'].append(obs['diag'])
            experiences['obs_ctrl'].append(obs['ctrl'])
            experiences['action_diag'].append(diag_action)
            experiences['action_ctrl'].append(ctrl_action)
            experiences['reward_diag'].append(rewards['diag'])
            experiences['reward_ctrl'].append(rewards['ctrl'])
            experiences['next_obs_diag'].append(next_obs['diag'])
            experiences['next_obs_ctrl'].append(next_obs['ctrl'])
            experiences['done'].append(done)
            
            ep_reward += rewards['diag'] + rewards['ctrl']
            self.total_steps += 1
            
            obs = next_obs
            
            if done:
                episode_rewards.append(ep_reward)
                self.episode_count += 1
                obs, info = self.env.reset(options={'difficulty': self._get_difficulty()})
                ep_reward = 0
        
        return experiences, episode_rewards
    
    def update_agents(self, experiences: dict):
        """更新两个智能体"""
        batch_size = self.args.batch_size
        n_samples = len(experiences['obs_diag'])
        
        diag_losses = []
        ctrl_losses = []
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            
            # 准备批次
            obs_diag = torch.FloatTensor(experiences['obs_diag'][start:end]).to(self.device)
            obs_ctrl = torch.FloatTensor(experiences['obs_ctrl'][start:end]).to(self.device)
            next_obs_ctrl = torch.FloatTensor(experiences['next_obs_ctrl'][start:end]).to(self.device)
            
            reward_diag = torch.FloatTensor(experiences['reward_diag'][start:end]).to(self.device)
            reward_ctrl = torch.FloatTensor(experiences['reward_ctrl'][start:end]).to(self.device)
            done = torch.FloatTensor(experiences['done'][start:end]).to(self.device)
            
            # 准备动作
            action_ctrl = torch.FloatTensor([
                [a['timing_offset'], a['fuel_adj'], a['protection_level']]
                for a in experiences['action_ctrl'][start:end]
            ]).to(self.device)
            
            # 更新诊断智能体
            action_diag_batch = {
                'fault_type': torch.LongTensor([a['fault_type'] for a in experiences['action_diag'][start:end]]).to(self.device),
                'severity': torch.FloatTensor([a['severity'] for a in experiences['action_diag'][start:end]]).to(self.device),
                'confidence': torch.FloatTensor([a['confidence'] for a in experiences['action_diag'][start:end]]).to(self.device)
            }
            
            # 简化的优势估计
            advantages = reward_diag - reward_diag.mean()
            returns = reward_diag
            old_log_probs = torch.zeros_like(reward_diag)
            
            diag_loss = self.diag_agent.update(
                obs_diag, action_diag_batch, advantages, returns, old_log_probs
            )
            diag_losses.append(diag_loss)
            
            # 更新控制智能体（TD-MPC2）
            world_loss = self.ctrl_agent.update_world_model(
                obs_ctrl, action_ctrl, reward_ctrl, next_obs_ctrl
            )
            critic_loss = self.ctrl_agent.update_critic(
                obs_ctrl, action_ctrl, reward_ctrl, next_obs_ctrl, done
            )
            policy_loss = self.ctrl_agent.update_policy(obs_ctrl)
            
            ctrl_losses.append({**world_loss, **critic_loss, **policy_loss})
        
        return diag_losses, ctrl_losses
    
    def train(self):
        """主训练循环"""
        args = self.args
        n_updates = args.total_steps // args.rollout_steps
        
        print(f"开始训练: {args.total_steps} 步, {n_updates} 次更新")
        print(f"诊断智能体: PINN+KAN (hidden={args.kan_hidden}, grid={args.kan_grid})")
        print(f"控制智能体: TD-MPC2 (horizon={args.mpc_horizon})")
        
        best_reward = -float('inf')
        
        for update in range(1, n_updates + 1):
            start_time = time.time()
            
            # 收集经验
            experiences, episode_rewards = self.collect_experience(args.rollout_steps)
            
            # 更新
            diag_losses, ctrl_losses = self.update_agents(experiences)
            
            elapsed = time.time() - start_time
            
            # 记录
            if episode_rewards:
                mean_reward = np.mean(episode_rewards)
                
                self.visualizer.update({
                    'episodes': self.episode_count,
                    'reward_diag': np.mean(experiences['reward_diag']),
                    'reward_ctrl': np.mean(experiences['reward_ctrl']),
                    'reward_total': mean_reward,
                    'loss_diag_policy': np.mean([l['policy_loss'] for l in diag_losses]),
                    'loss_ctrl_policy': np.mean([l['policy_loss'] for l in ctrl_losses]),
                    'diag_accuracy': 0.0,  # 需要单独计算
                    'ctrl_performance': 0.0
                })
                
                print(f"更新 {update}/{n_updates} | "
                      f"奖励: {mean_reward:.2f} | "
                      f"诊断损失: {np.mean([l['total_loss'] for l in diag_losses]):.4f} | "
                      f"时间: {elapsed:.2f}s")
                
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    self.save('best')
            
            if update % 10 == 0:
                self.save(f'checkpoint_{update}')
        
        # 保存最终可视化
        fig = self.visualizer.plot_training_curves()
        self.visualizer.save_plot(fig, 'training_curves')
        
        self.save('final')
        print("训练完成!")
    
    def _get_difficulty(self) -> float:
        return min(1.0, self.episode_count / 300)
    
    def save(self, name: str):
        self.diag_agent.save(self.save_dir / f'{name}_diag.pt')
        self.ctrl_agent.save(self.save_dir / f'{name}_ctrl.pt')
    
    def load(self, name: str):
        self.diag_agent.load(self.save_dir / f'{name}_diag.pt')
        self.ctrl_agent.load(self.save_dir / f'{name}_ctrl.pt')


def main():
    args = parse_args()
    
    print("=" * 60)
    print("PINN+KAN 诊断 + TD-MPC2 控制 联合训练")
    print("=" * 60)
    
    trainer = AdvancedTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
