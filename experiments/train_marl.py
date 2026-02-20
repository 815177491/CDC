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

from config import TRAINING_CONFIG, PATH_CONFIG
from marl.env import EngineEnvConfig
from marl.training import MAPPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='双智能体强化学习训练')
    
    # 训练参数
    parser.add_argument('--total-steps', type=int, default=100000,
                        help='总训练步数')
    parser.add_argument('--rollout-steps', type=int, default=2048,
                        help='每次更新的采样步数')
    parser.add_argument('--batch-size', type=int, default=TRAINING_CONFIG.BATCH_SIZE,
                        help=f'训练批大小 (默认: {TRAINING_CONFIG.BATCH_SIZE})')
    parser.add_argument('--n-epochs', type=int, default=TRAINING_CONFIG.NUM_EPOCHS,
                        help=f'每次更新的训练轮数 (默认: {TRAINING_CONFIG.NUM_EPOCHS})')
    
    # 学习率
    parser.add_argument('--diag-lr', type=float, default=TRAINING_CONFIG.LEARNING_RATE,
                        help=f'诊断智能体学习率 (默认: {TRAINING_CONFIG.LEARNING_RATE})')
    parser.add_argument('--ctrl-lr', type=float, default=TRAINING_CONFIG.LEARNING_RATE,
                        help=f'控制智能体学习率 (默认: {TRAINING_CONFIG.LEARNING_RATE})')
    
    # PPO参数
    parser.add_argument('--gamma', type=float, default=TRAINING_CONFIG.GAMMA,
                        help=f'折扣因子 (默认: {TRAINING_CONFIG.GAMMA})')
    parser.add_argument('--gae-lambda', type=float, default=TRAINING_CONFIG.GAE_LAMBDA,
                        help=f'GAE参数 (默认: {TRAINING_CONFIG.GAE_LAMBDA})')
    parser.add_argument('--clip-epsilon', type=float, default=TRAINING_CONFIG.CLIP_EPSILON,
                        help=f'PPO裁剪系数 (默认: {TRAINING_CONFIG.CLIP_EPSILON})')
    
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
    parser.add_argument('--save-dir', type=str, default=PATH_CONFIG.CHECKPOINTS_DIR,
                        help=f'模型保存目录 (默认: {PATH_CONFIG.CHECKPOINTS_DIR})')
    parser.add_argument('--log-interval', type=int, default=1,
                        help='日志打印间隔')
    parser.add_argument('--save-interval', type=int, default=TRAINING_CONFIG.SAVE_INTERVAL,
                        help=f'模型保存间隔 (默认: {TRAINING_CONFIG.SAVE_INTERVAL})')
    
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
