"""
智能体训练脚本
==============
用于离线训练控制智能体的DQN策略

训练流程:
1. 创建发动机仿真环境
2. 随机注入各类故障
3. 收集经验并训练DQN
4. 保存训练后的模型
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import warnings

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import MarineEngine0D, OperatingCondition
from diagnosis import FaultInjector, FaultType
from agents import CoordinatorAgent, DiagnosisAgent, ControlAgent


class EngineEnv:
    """
    发动机仿真环境
    
    封装为类似 Gym 的接口，用于RL训练
    """
    
    def __init__(self, engine: MarineEngine0D):
        self.engine = engine
        self.fault_injector = FaultInjector(engine)
        
        # 基准工况
        self.base_condition = OperatingCondition(
            rpm=80.0,
            p_scav=3.5e5,
            T_scav=320.0,
            fuel_mass=0.08
        )
        
        # 状态空间维度
        self.state_dim = 10
        
        # 动作空间
        self.n_vit_actions = 9
        self.n_fuel_actions = 5
        self.action_dim = self.n_vit_actions * self.n_fuel_actions
        
        # 安全限值
        self.Pmax_limit = 190.0
        self.Pmax_target = 170.0
        
        # 当前步数
        self.current_step = 0
        self.max_steps = 200
        
        # 当前故障
        self.current_fault = None
        self.fault_onset = 0
        
        # 基准值
        self.Pmax_baseline = None
        self.reset()
    
    def reset(self):
        """重置环境"""
        self.engine.clear_faults()
        self.fault_injector.clear_all_faults()
        self.current_step = 0
        
        # 运行基准循环获取基准值
        self.engine.run_cycle(self.base_condition)
        self.Pmax_baseline = self.engine.get_pmax()
        
        # 随机注入故障
        self._inject_random_fault()
        
        # 获取初始状态
        state = self._get_state()
        
        return state
    
    def _inject_random_fault(self):
        """随机注入故障"""
        fault_types = [
            ('timing', 'early'),
            ('timing', 'late'),
            ('leak', None),
            ('fuel', None),
        ]
        
        choice = np.random.randint(len(fault_types))
        fault_type, sub_type = fault_types[choice]
        
        # 随机故障强度和发生时间
        severity = np.random.uniform(0.3, 1.0)
        self.fault_onset = np.random.randint(10, 50)
        
        if fault_type == 'timing':
            offset = 2.0 * severity if sub_type == 'early' else -2.0 * severity
            self.current_fault = self.fault_injector.create_timing_fault(
                offset_deg=offset,
                onset_time=float(self.fault_onset),
                ramp_time=np.random.uniform(0, 5)
            )
        elif fault_type == 'leak':
            self.current_fault = self.fault_injector.create_leak_fault(
                leak_factor=0.1 * severity,
                onset_time=float(self.fault_onset),
                ramp_time=np.random.uniform(0, 10)
            )
        else:
            self.current_fault = self.fault_injector.create_fuel_fault(
                degradation=0.15 * severity,
                onset_time=float(self.fault_onset),
                ramp_time=np.random.uniform(0, 5)
            )
        
        self.fault_injector.inject_fault(self.current_fault)
    
    def _get_state(self):
        """获取当前状态向量"""
        Pmax = self.engine.get_pmax()
        Pcomp = self.engine.get_pcomp()
        Texh = self.engine.get_exhaust_temp()
        
        # 归一化
        state = np.array([
            Pmax / 200.0,
            Pcomp / 200.0,
            Texh / 500.0,
            (Pmax - self.Pmax_baseline) / self.Pmax_baseline if self.Pmax_baseline else 0,
            (Pcomp - 150) / 150,
            (Texh - 350) / 100,
            self.current_step / self.max_steps,
            1.0 if self.current_step >= self.fault_onset else 0.0,
            0.5,  # placeholder for vit
            1.0,  # placeholder for fuel
        ], dtype=np.float32)
        
        return state
    
    def step(self, action_idx: int):
        """
        执行一步
        
        Args:
            action_idx: 动作索引
            
        Returns:
            (next_state, reward, done, info)
        """
        # 解码动作
        vit_actions = np.linspace(-8, 4, self.n_vit_actions)
        fuel_actions = np.linspace(0.7, 1.0, self.n_fuel_actions)
        
        vit_idx = action_idx // self.n_fuel_actions
        fuel_idx = action_idx % self.n_fuel_actions
        
        vit = vit_actions[vit_idx]
        fuel = fuel_actions[fuel_idx]
        
        # 应用故障
        self.fault_injector.apply_faults(float(self.current_step))
        
        # 应用控制
        base_timing = self.engine.calibrated_params.get('injection_timing', 2.0)
        self.engine.set_injection_timing(base_timing + vit)
        
        # 运行仿真
        self.engine.run_cycle(self.base_condition)
        
        # 获取新状态
        next_state = self._get_state()
        next_state[-2] = (vit + 8) / 12.0
        next_state[-1] = (fuel - 0.7) / 0.3
        
        # 计算奖励
        Pmax = self.engine.get_pmax()
        reward = self._compute_reward(Pmax, vit, fuel)
        
        # 检查终止条件
        self.current_step += 1
        done = (self.current_step >= self.max_steps) or (Pmax > self.Pmax_limit + 10)
        
        info = {
            'Pmax': Pmax,
            'vit': vit,
            'fuel': fuel,
        }
        
        return next_state, reward, done, info
    
    def _compute_reward(self, Pmax: float, vit: float, fuel: float) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 安全奖励
        if Pmax <= self.Pmax_limit:
            deviation = abs(Pmax - self.Pmax_target) / self.Pmax_target
            reward += 1.0 - deviation
        else:
            overshoot = (Pmax - self.Pmax_limit) / self.Pmax_limit
            reward -= 5.0 * overshoot
        
        # 效率惩罚
        reward -= (1.0 - fuel) * 0.5
        reward -= abs(vit) * 0.02
        
        return reward


def train_control_agent(episodes: int = 500, 
                        save_dir: str = "models",
                        verbose: bool = True):
    """
    训练控制智能体
    
    Args:
        episodes: 训练回合数
        save_dir: 模型保存目录
        verbose: 是否显示进度
    """
    print("\n" + "=" * 60)
    print("控制智能体DQN训练")
    print("=" * 60)
    
    # 创建发动机
    engine = MarineEngine0D(
        bore=0.620,
        stroke=2.658,
        n_cylinders=6,
        compression_ratio=13.5
    )
    
    # 创建环境
    env = EngineEnv(engine)
    
    # 创建控制智能体
    control_agent = ControlAgent(engine, use_rl=True)
    control_agent.learning_enabled = True
    
    # 创建诊断智能体 (用于状态编码)
    diagnosis_agent = DiagnosisAgent(engine)
    
    # 训练统计
    episode_rewards = []
    episode_lengths = []
    
    os.makedirs(save_dir, exist_ok=True)
    
    iterator = tqdm(range(episodes), desc="Training") if verbose else range(episodes)
    
    for episode in iterator:
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # 选择动作
            action_idx = control_agent.dqn.select_action(state, training=True)
            
            # 执行动作
            next_state, reward, done, info = env.step(action_idx)
            
            # 存储经验
            control_agent.dqn.replay_buffer.push(
                state, action_idx, reward, next_state, done
            )
            
            # 训练
            loss = control_agent.dqn.update()
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # 定期保存
        if (episode + 1) % 100 == 0:
            save_path = os.path.join(save_dir, f"control_agent_ep{episode+1}.pt")
            control_agent.save_model(save_path)
            
            if verbose:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                print(f"\n  Episode {episode+1}: Avg Reward={avg_reward:.2f}, "
                      f"Avg Length={avg_length:.1f}, "
                      f"Epsilon={control_agent.dqn.epsilon:.3f}")
    
    # 保存最终模型
    final_path = os.path.join(save_dir, "control_agent_final.pt")
    control_agent.save_model(final_path)
    
    print(f"\n训练完成! 模型已保存至: {final_path}")
    
    # 绘制训练曲线
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 奖励曲线
        window = 20
        rewards_smooth = np.convolve(episode_rewards, 
                                      np.ones(window)/window, 
                                      mode='valid')
        axes[0].plot(rewards_smooth)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Average Reward')
        axes[0].set_title('Training Reward')
        axes[0].grid(True)
        
        # 回合长度
        lengths_smooth = np.convolve(episode_lengths,
                                      np.ones(window)/window,
                                      mode='valid')
        axes[1].plot(lengths_smooth)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Episode Length')
        axes[1].set_title('Episode Length')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_curve.png"), dpi=150)
        print(f"训练曲线已保存: {save_dir}/training_curve.png")
        
    except ImportError:
        print("matplotlib未安装，跳过训练曲线绑制")
    
    return control_agent, episode_rewards


def main():
    parser = argparse.ArgumentParser(description='智能体训练脚本')
    parser.add_argument('--episodes', type=int, default=500, help='训练回合数')
    parser.add_argument('--save-dir', type=str, default='models', help='模型保存目录')
    parser.add_argument('--quiet', action='store_true', help='安静模式')
    
    args = parser.parse_args()
    
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
    except ImportError:
        print("错误: PyTorch未安装，无法训练DQN")
        print("请运行: pip install torch")
        return
    
    train_control_agent(
        episodes=args.episodes,
        save_dir=args.save_dir,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
