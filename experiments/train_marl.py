"""
双智能体强化学习训练脚本
========================
训练诊断智能体和控制智能体
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from marl.env import EngineEnvConfig
from marl.training import MAPPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='双智能体强化学习训练')
    
    # 训练参数
    parser.add_argument('--total-steps', type=int, default=100000,
                        help='总训练步数')
    parser.add_argument('--rollout-steps', type=int, default=2048,
                        help='每次更新的采样步数')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='训练批大小')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='每次更新的训练轮数')
    
    # 学习率
    parser.add_argument('--diag-lr', type=float, default=3e-4,
                        help='诊断智能体学习率')
    parser.add_argument('--ctrl-lr', type=float, default=3e-4,
                        help='控制智能体学习率')
    
    # PPO参数
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='折扣因子')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='GAE参数')
    parser.add_argument('--clip-epsilon', type=float, default=0.2,
                        help='PPO裁剪系数')
    
    # 环境参数
    parser.add_argument('--max-ep-steps', type=int, default=200,
                        help='单轮最大步数')
    parser.add_argument('--difficulty', type=float, default=0.5,
                        help='初始难度')
    parser.add_argument('--variable-condition', action='store_true', default=True,
                        help='是否启用变工况')
    
    # 其他
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='计算设备')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--log-interval', type=int, default=1,
                        help='日志打印间隔')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='模型保存间隔')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("双智能体强化学习 - 故障诊断与容错控制")
    print("=" * 60)
    
    # 环境配置
    env_config = EngineEnvConfig(
        max_steps=args.max_ep_steps,
        difficulty=args.difficulty,
        enable_variable_condition=args.variable_condition
    )
    
    # 创建训练器
    trainer = MAPPOTrainer(
        env_config=env_config,
        diag_lr=args.diag_lr,
        ctrl_lr=args.ctrl_lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        buffer_size=args.rollout_steps,
        device=args.device,
        save_dir=args.save_dir
    )
    
    print(f"\n配置:")
    print(f"  总步数: {args.total_steps}")
    print(f"  设备: {args.device}")
    print(f"  变工况: {args.variable_condition}")
    print(f"  难度: {args.difficulty}")
    print()
    
    # 开始训练
    trainer.train(
        total_timesteps=args.total_steps,
        rollout_steps=args.rollout_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval
    )


if __name__ == '__main__':
    main()
