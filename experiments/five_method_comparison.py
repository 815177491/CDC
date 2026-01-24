#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强化学习控制方法对比实验框架 (GPU加速)
=====================================
论文正式对比实验，对比以下5种方法：
1. PID - 传统控制基线
2. DQN - 经典深度强化学习 (Nature 2015)
3. SAC - 最大熵强化学习 (ICML 2018)
4. TD-MPC2 - 2024年最新方法 (ICLR 2024) ★ 推荐
5. DPMD - 2025年最新方法 (扩散策略+镜像下降)

实验设计：
- 500 episodes训练（正式实验）
- 5个随机种子
- 综合评分选择最优方法
- 先1个种子快速验证，再全量运行

评估指标：
- 控制精度（Pmax误差<2bar达标率）
- 收敛速度（达到90%性能的episode）
- 推理时间（ms/step）
- 训练稳定性（奖励标准差）

快速验证结果 (100 episodes, seed=42):
- TD-MPC2: 89.7% 达标率 ★
- SAC: 88.4% 达标率
- DPMD: 86.4% 达标率
- MambaPolicy: 70.4% 达标率 (未纳入正式对比)
- PID: 0.5% 达标率

Author: CDC Project
Date: 2026-01-22
"""

import numpy as np
import random
import time
import os
import sys
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import warnings

# 进度条支持
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# 可视化支持
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available, visualization will be disabled")
    def tqdm(iterable, **kwargs):
        return iterable

# 尝试导入深度学习库
try:
    import torch
    TORCH_AVAILABLE = True
    
    # GPU检测
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEM = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[实验框架] GPU检测: {GPU_NAME} ({GPU_MEM:.1f} GB)")
    else:
        DEVICE = torch.device('cpu')
        GPU_NAME = None
        GPU_MEM = 0
        print("[实验框架] 使用CPU运行")
        
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    GPU_NAME = None
    GPU_MEM = 0


# ============================================================
# 配置类
# ============================================================

@dataclass
class ExperimentConfig:
    """实验配置"""
    # 训练参数 (优化后，加快训练速度)
    n_episodes: int = 100              # 训练episodes (原400)
    max_steps_per_episode: int = 200   # 每episode最大步数 (原500)
    
    # 评估参数
    eval_frequency: int = 50           # 评估频率 (原25)
    n_eval_episodes: int = 3           # 评估episodes数 (原10)
    
    # 环境参数
    state_dim: int = 8                 # 状态维度
    action_dim: int = 5                # 动作维度
    
    # GPU参数
    device: str = 'cuda'
    batch_size: int = 256              # GPU可用更大batch
    
    # 随机种子
    seeds: List[int] = None
    
    # 保存路径
    save_dir: str = 'results/comparison'
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 456, 789, 1024]
        
        # 根据GPU显存自动调整batch_size
        if GPU_MEM >= 8:
            self.batch_size = 512
        elif GPU_MEM >= 4:
            self.batch_size = 256
        elif GPU_MEM > 0:
            self.batch_size = 128
        else:
            self.batch_size = 64


@dataclass
class MethodResult:
    """单个方法的实验结果"""
    method_name: str
    seed: int
    
    # 性能指标
    final_reward: float = 0.0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    max_reward: float = 0.0
    
    # 控制精度
    pmax_error_mean: float = 0.0       # Pmax误差均值
    pmax_error_std: float = 0.0        # Pmax误差标准差  
    accuracy_rate: float = 0.0         # 达标率 (误差<2bar)
    
    # 效率指标
    convergence_episode: int = 0       # 收敛episode
    training_time: float = 0.0         # 训练时间(秒)
    inference_time_ms: float = 0.0     # 推理时间(毫秒/步)
    
    # 训练曲线
    reward_curve: List[float] = None
    eval_curve: List[float] = None


@dataclass
class ComparisonResult:
    """对比实验总结果"""
    methods: List[str]
    all_results: Dict[str, List[MethodResult]]
    
    # 综合评分
    scores: Dict[str, float] = None
    best_method: str = ""
    
    # 排名
    rankings: Dict[str, int] = None


# ============================================================
# 柴油机控制环境 (简化版)
# ============================================================

class DieselEngineEnv:
    """
    柴油机Pmax控制仿真环境 (含故障注入)
    
    状态: [pmax, pmax_error, rpm, fuel_rate, timing, rail_p, boost_p, temp]
    动作: 5个离散调整档位 (-2, -1, 0, +1, +2)
    故障: 30%概率随机注入各类故障
    """
    
    def __init__(self, seed: int = None, fault_probability: float = 0.3):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.state_dim = 8
        self.action_dim = 5
        self.fault_probability = fault_probability
        
        # 目标Pmax
        self.pmax_target = 180.0  # bar
        self.pmax_tolerance = 2.0  # bar
        
        # 状态范围
        self.state_ranges = {
            'pmax': (150, 210),
            'rpm': (500, 1000),
            'fuel_rate': (50, 150),
            'timing': (-5, 10),
            'rail_p': (1000, 2000),
            'boost_p': (1.5, 3.5),
            'temp': (300, 500)
        }
        
        # 故障状态初始化
        self.reset_fault_state()
        self.reset()
    
    def reset_fault_state(self):
        """重置故障状态"""
        self.fault_active = False
        self.fault_type = None
        self.fault_severity = 0.0
        self.fault_onset_step = 0
        self.fault_duration = 0
        self.steps_in_fault = 0
        
    def inject_random_fault(self):
        """随机注入故障 (30%概率)"""
        if np.random.random() < self.fault_probability:
            self.fault_active = True
            
            # 故障类型
            fault_types = [
                'injection_timing',  # 喷油正时偏移
                'fuel_system',       # 燃油系统故障  
                'compression_leak',  # 压缩泄漏
                'turbo_lag',        # 增压滞后
                'sensor_drift'      # 传感器漂移
            ]
            self.fault_type = np.random.choice(fault_types)
            
            # 故障严重程度 (0.3-1.0)
            self.fault_severity = np.random.uniform(0.3, 1.0)
            
            # 故障发生时间 (episode中的10-80步)
            self.fault_onset_step = np.random.randint(10, 80)
            
            # 故障持续时间 (20-100步)  
            self.fault_duration = np.random.randint(20, 100)
            
            print(f"[故障注入] 类型:{self.fault_type}, 严重度:{self.fault_severity:.2f}, "
                  f"开始:{self.fault_onset_step}步, 持续:{self.fault_duration}步")
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        # 随机初始化
        self.pmax = np.random.uniform(160, 200)
        self.rpm = np.random.uniform(600, 900)
        self.fuel_rate = np.random.uniform(80, 120)
        self.timing = np.random.uniform(0, 5)
        self.rail_p = np.random.uniform(1200, 1800)
        self.boost_p = np.random.uniform(2.0, 3.0)
        self.temp = np.random.uniform(350, 450)
        
        self.step_count = 0
        self.reset_fault_state()
        self.inject_random_fault()  # 决定本episode是否有故障
        
        return self._get_state()
    
    def apply_fault_effects(self):
        """应用故障对系统的影响"""
        if not self.fault_active:
            return
            
        # 检查故障是否应该开始
        if self.step_count >= self.fault_onset_step and self.steps_in_fault == 0:
            print(f"[t={self.step_count}] 故障开始: {self.fault_type}")
            
        # 故障激活期间
        if (self.step_count >= self.fault_onset_step and 
            self.steps_in_fault < self.fault_duration):
            
            self.steps_in_fault += 1
            
            # 根据故障类型应用不同影响
            if self.fault_type == 'injection_timing':
                # 喷油正时偏移 -> Pmax升高
                fault_effect = self.fault_severity * 8.0  # 最大8 bar偏差
                self.pmax += fault_effect * np.sin(0.1 * self.steps_in_fault)
                
            elif self.fault_type == 'fuel_system':
                # 燃油系统故障 -> 燃油压力波动
                self.rail_p += self.fault_severity * 200 * np.sin(0.2 * self.steps_in_fault)
                self.pmax += self.fault_severity * 5.0 * np.random.normal(0, 1)
                
            elif self.fault_type == 'compression_leak':
                # 压缩泄漏 -> Pmax下降
                self.pmax -= self.fault_severity * 6.0 * (1 - np.exp(-0.1 * self.steps_in_fault))
                
            elif self.fault_type == 'turbo_lag':
                # 增压滞后 -> 进气压力波动
                self.boost_p *= (1 - self.fault_severity * 0.3 * np.sin(0.15 * self.steps_in_fault))
                self.pmax += self.fault_severity * 3.0 * np.random.normal(0, 0.5)
                
            elif self.fault_type == 'sensor_drift':
                # 传感器漂移 -> 测量噪声增加
                drift = self.fault_severity * 4.0 * (self.steps_in_fault / self.fault_duration)
                self.pmax += drift + np.random.normal(0, self.fault_severity * 2.0)
        
        # 故障结束
        elif self.steps_in_fault >= self.fault_duration:
            if self.fault_active:
                print(f"[t={self.step_count}] 故障结束: {self.fault_type}")
                self.fault_active = False
    
    def _get_state(self) -> np.ndarray:
        """获取归一化状态"""
        pmax_norm = (self.pmax - 150) / 60
        error_norm = (self.pmax - self.pmax_target) / 30
        rpm_norm = (self.rpm - 500) / 500
        fuel_norm = (self.fuel_rate - 50) / 100
        timing_norm = (self.timing + 5) / 15
        rail_norm = (self.rail_p - 1000) / 1000
        boost_norm = (self.boost_p - 1.5) / 2.0
        temp_norm = (self.temp - 300) / 200
        
        return np.array([
            pmax_norm, error_norm, rpm_norm, fuel_norm,
            timing_norm, rail_norm, boost_norm, temp_norm
        ], dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        # 动作映射: 0=-2, 1=-1, 2=0, 3=+1, 4=+2
        adjustment = (action - 2) * 0.5  # 喷油时刻调整
        
        # 应用控制
        self.timing += adjustment
        self.timing = np.clip(self.timing, -5, 10)
        
        # 模拟发动机响应
        self._simulate_engine()
        
        # 计算奖励
        error = abs(self.pmax - self.pmax_target)
        
        if error < self.pmax_tolerance:
            reward = 10.0 - error  # 在目标范围内，高奖励
        else:
            reward = -error * 0.5  # 偏离目标，惩罚
        
        # 安全约束
        if self.pmax > 200 or self.pmax < 160:
            reward -= 10.0  # 安全惩罚
        
        self.step_count += 1
        done = self.step_count >= 500
        
        info = {
            'pmax': self.pmax,
            'error': error,
            'in_tolerance': error < self.pmax_tolerance
        }
        
        return self._get_state(), reward, done, info
    
    def _simulate_engine(self):
        """简化的发动机动力学模拟 (含故障影响)"""
        # 应用故障效果
        self.apply_fault_effects()
        
        # Pmax响应 (基于喷油时刻)
        delta_pmax = (self.timing - 2) * 1.5 + np.random.normal(0, 0.5)
        self.pmax = 0.95 * self.pmax + 0.05 * (175 + delta_pmax)
        
        # 添加基础干扰
        self.pmax += np.random.normal(0, 0.3)
        self.pmax = np.clip(self.pmax, 150, 210)
        
        # 其他状态随机变化
        self.rpm += np.random.normal(0, 5)
        self.rpm = np.clip(self.rpm, 500, 1000)
        
        self.fuel_rate += np.random.normal(0, 1)
        self.fuel_rate = np.clip(self.fuel_rate, 50, 150)
        
        # 确保boost_p和rail_p在范围内
        self.boost_p = np.clip(self.boost_p, 1.5, 3.5)
        self.rail_p = np.clip(self.rail_p, 1000, 2000)


# ============================================================
# PID控制器
# ============================================================

class PIDController:
    """PID控制器 (基线方法)"""
    
    def __init__(self, kp: float = 0.5, ki: float = 0.1, kd: float = 0.2):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.integral = 0
        self.prev_error = 0
    
    def select_action(self, state: np.ndarray, explore: bool = False) -> int:
        """根据状态选择动作"""
        # state[1]是归一化的pmax误差
        error = state[1] * 30  # 反归一化
        
        # PID计算
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        
        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # 映射到离散动作
        if control > 1:
            return 4  # +2
        elif control > 0.3:
            return 3  # +1
        elif control < -1:
            return 0  # -2
        elif control < -0.3:
            return 1  # -1
        else:
            return 2  # 0
    
    def reset(self):
        self.integral = 0
        self.prev_error = 0
    
    def update(self, batch=None):
        """PID不需要更新"""
        return {}
    
    def get_name(self):
        return "PID"


# ============================================================
# 实验运行器
# ============================================================

class FiveMethodComparison:
    """五种方法对比实验"""
    
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        
        # 确保保存目录存在
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # 导入算法
        try:
            from agents.rl_algorithms import get_algorithm, SAC
            from agents.advanced_rl_algorithms import (
                get_advanced_algorithm, TDMPC2, MambaPolicy, DPMD
            )
            self.rl_available = True
        except ImportError as e:
            print(f"[警告] 无法导入RL算法: {e}")
            self.rl_available = False
        
        self.results = {}
    
    def _create_method(self, method_name: str, seed: int):
        """创建方法实例"""
        if method_name == "PID":
            return PIDController()
        
        if not self.rl_available:
            raise RuntimeError("RL algorithms not available")
        
        from agents.rl_algorithms import get_algorithm
        from agents.advanced_rl_algorithms import get_advanced_algorithm
        
        config = {
            'device': str(DEVICE),
            'batch_size': self.config.batch_size,
            'gamma': 0.99,
        }
        
        if method_name == "SAC":
            return get_algorithm("SAC", self.config.state_dim, 
                               self.config.action_dim, config)
        elif method_name == "DQN":
            return get_algorithm("DQN", self.config.state_dim, 
                               self.config.action_dim, config)
        elif method_name in ["TDMPC2", "TD-MPC2"]:
            return get_advanced_algorithm("TDMPC2", self.config.state_dim,
                                         self.config.action_dim, config)
        elif method_name in ["MambaPolicy", "Mamba"]:
            return get_advanced_algorithm("MambaPolicy", self.config.state_dim,
                                         self.config.action_dim, config)
        elif method_name == "DPMD":
            return get_advanced_algorithm("DPMD", self.config.state_dim,
                                         self.config.action_dim, config)
        else:
            raise ValueError(f"Unknown method: {method_name}")
    
    def train_single(self, method_name: str, seed: int, 
                     verbose: bool = True) -> MethodResult:
        """训练单个方法单个种子"""
        # 设置随机种子
        np.random.seed(seed)
        random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # 创建环境和方法
        env = DieselEngineEnv(seed)
        method = self._create_method(method_name, seed)
        
        # 结果记录
        result = MethodResult(method_name=method_name, seed=seed)
        result.reward_curve = []
        result.eval_curve = []
        
        episode_rewards = []
        pmax_errors = []
        in_tolerance_count = 0
        total_steps = 0
        
        train_start = time.time()
        
        # 用于存储最新的评估结果
        last_eval_reward = 0.0
        
        # 训练配置
        n_eps = self.config.n_episodes
        show_progress = verbose  # 是否显示进度条
        
        for episode in range(self.config.n_episodes):
            state = env.reset()
            if hasattr(method, 'reset'):
                method.reset()
            if hasattr(method, 'reset_history'):
                method.reset_history()
            
            episode_reward = 0
            episode_errors = []
            episode_states = []
            episode_actions = []
            episode_rewards_list = []
            
            for step in range(self.config.max_steps_per_episode):
                # 选择动作
                step_start = time.time()
                action = method.select_action(state, explore=True)
                inference_time = (time.time() - step_start) * 1000  # ms
                
                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_errors.append(info['error'])
                if info['in_tolerance']:
                    in_tolerance_count += 1
                total_steps += 1
                
                # 存储经验
                if hasattr(method, 'buffer') and method.buffer is not None:
                    method.buffer.push(state, action, reward, next_state, done)
                
                # 记录轨迹 (用于Mamba)
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards_list.append(reward)
                
                # 更新方法
                if hasattr(method, 'buffer') and method.buffer is not None and len(method.buffer) >= 64:
                    batch = method.buffer.sample(self.config.batch_size)
                    method.update(batch)
                elif hasattr(method, 'store_transition'):
                    method.store_transition(state, action, reward, done)
                
                state = next_state
                
                if done:
                    break
            
            # 存储轨迹 (用于Mamba序列训练)
            if hasattr(method, 'store_trajectory'):
                method.store_trajectory(episode_states, episode_actions, episode_rewards_list)
            
            # PPO更新
            if hasattr(method, 'states') and len(method.states) > 0:
                method.update()
            
            episode_rewards.append(episode_reward)
            pmax_errors.extend(episode_errors)
            result.reward_curve.append(episode_reward)
            
            # 定期评估
            if (episode + 1) % self.config.eval_frequency == 0:
                eval_reward = self._evaluate(method, env, self.config.n_eval_episodes)
                result.eval_curve.append(eval_reward)
                last_eval_reward = eval_reward
            
            # 更新进度条 - 手动实现精确格式
            # 格式: PID      45%|████████████        | Episode 225/500 | Reward:  -850.3 | Eval:  -920.1 | Error:  1.25bar
            if show_progress:
                avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
                mean_error = np.mean(episode_errors)
                ep = episode + 1
                
                # 构建进度条
                pct = int(ep / n_eps * 100)
                filled = int(20 * ep / n_eps)
                bar_str = '█' * filled + ' ' * (20 - filled)
                
                # 完整格式字符串
                status = f"\r{method_name:<8s}{pct:3d}%|{bar_str}| Episode {ep:3d}/{n_eps} | Reward: {avg_reward:8.1f} | Eval: {last_eval_reward:8.1f} | Error: {mean_error:5.2f}bar"
                sys.stdout.write(status)
                sys.stdout.flush()
        
        # 进度条结束后换行
        if show_progress:
            print()
        
        # 计算结果
        result.training_time = time.time() - train_start
        result.final_reward = np.mean(episode_rewards[-20:])
        result.mean_reward = np.mean(episode_rewards)
        result.std_reward = np.std(episode_rewards)
        result.max_reward = np.max(episode_rewards)
        result.pmax_error_mean = np.mean(pmax_errors)
        result.pmax_error_std = np.std(pmax_errors)
        result.accuracy_rate = in_tolerance_count / total_steps
        result.inference_time_ms = inference_time  # 最后一步的推理时间
        
        # 估计收敛episode
        for i, r in enumerate(episode_rewards):
            if r >= 0.9 * result.max_reward:
                result.convergence_episode = i
                break
        
        if verbose:
            print(f"\n[{method_name}] 训练完成!")
            print(f"  最终奖励: {result.final_reward:.2f}")
            print(f"  Pmax误差: {result.pmax_error_mean:.2f} ± {result.pmax_error_std:.2f} bar")
            print(f"  达标率: {result.accuracy_rate*100:.1f}%")
            print(f"  训练时间: {result.training_time:.1f}s")
        
        return result
    
    def _evaluate(self, method, env: DieselEngineEnv, n_episodes: int) -> float:
        """评估方法性能"""
        total_reward = 0
        
        for _ in range(n_episodes):
            state = env.reset()
            if hasattr(method, 'reset'):
                method.reset()
            if hasattr(method, 'reset_history'):
                method.reset_history()
            
            episode_reward = 0
            for _ in range(self.config.max_steps_per_episode):
                action = method.select_action(state, explore=False)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
                if done:
                    break
            
            total_reward += episode_reward
        
        return total_reward / n_episodes
    
    def run_quick_validation(self, methods: List[str] = None) -> Dict[str, MethodResult]:
        """快速验证 - 单个种子"""
        if methods is None:
            # 论文正式对比方法：PID + DQN + SAC + TD-MPC2 + DPMD
            methods = ["PID", "DQN", "SAC", "TDMPC2", "DPMD"]
        
        print("\n" + "="*70)
        print("🚀 快速验证模式 (1个种子)")
        print(f"   Episodes: {self.config.n_episodes} | Seed: {self.config.seeds[0]}")
        print("="*70)
        
        results = {}
        seed = self.config.seeds[0]
        
        for method_idx, method_name in enumerate(methods):
            print(f"\n{'─'*70}")
            print(f"🔹 方法 [{method_idx+1}/{len(methods)}]: {method_name}")
            print(f"{'─'*70}")
            try:
                result = self.train_single(method_name, seed, verbose=True)
                results[method_name] = result
                print(f"  ✅ 完成: 奖励={result.final_reward:.1f} | "
                      f"达标率={result.accuracy_rate*100:.1f}% | "
                      f"训练时间={result.training_time:.1f}s")
            except Exception as e:
                print(f"  ❌ {method_name} 训练失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 打印快速验证结果
        self._print_validation_summary(results)
        
        return results
    
    def run_full_comparison(self, methods: List[str] = None) -> ComparisonResult:
        """完整对比实验 - 5个种子"""
        if methods is None:
            # 论文正式对比方法：PID + DQN + SAC + TD-MPC2 + DPMD
            methods = ["PID", "DQN", "SAC", "TDMPC2", "DPMD"]
        
        print("\n" + "="*70)
        print("🔬 完整对比实验 (5个种子 × 5种方法)")
        print(f"   Episodes: {self.config.n_episodes} | Seeds: {self.config.seeds}")
        print("="*70)
        
        all_results = {m: [] for m in methods}
        total_runs = len(methods) * len(self.config.seeds)
        completed_runs = 0
        
        for method_idx, method_name in enumerate(methods):
            print(f"\n{'─'*70}")
            print(f"🔹 方法 [{method_idx+1}/{len(methods)}]: {method_name}")
            print(f"{'─'*70}")
            
            # 内层进度条（种子级）- 每个种子单独训练并显示进度
            for seed_idx, seed in enumerate(self.config.seeds):
                print(f"\n  📌 种子 [{seed_idx+1}/{len(self.config.seeds)}]: {seed}")
                try:
                    result = self.train_single(method_name, seed, verbose=True)
                    all_results[method_name].append(result)
                    completed_runs += 1
                    
                    # 打印该种子的结果摘要
                    print(f"  ✅ 完成: 奖励={result.final_reward:.1f} | "
                          f"达标率={result.accuracy_rate*100:.1f}% | "
                          f"训练时间={result.training_time:.1f}s")
                        
                except Exception as e:
                    print(f"  ❌ 种子 {seed} 失败: {e}")
        
        # 计算综合评分并排名
        comparison = self._compute_comparison(methods, all_results)
        
        # 保存结果
        self._save_results(comparison)
        
        # 打印总结
        self._print_comparison_summary(comparison)
        
        return comparison
    
    def _compute_comparison(self, methods: List[str], 
                           all_results: Dict[str, List[MethodResult]]) -> ComparisonResult:
        """计算综合评分"""
        scores = {}
        
        for method in methods:
            results = all_results[method]
            if not results:
                scores[method] = 0
                continue
            
            # 计算各维度均值
            mean_accuracy = np.mean([r.accuracy_rate for r in results])
            mean_reward = np.mean([r.final_reward for r in results])
            mean_convergence = np.mean([r.convergence_episode for r in results])
            mean_inference = np.mean([r.inference_time_ms for r in results])
            
            # 综合评分 (加权)
            # 控制精度 40% + 最终奖励 30% + 收敛速度 20% + 推理速度 10%
            score = (
                0.4 * mean_accuracy * 100 +  # 达标率 (0-100)
                0.3 * (mean_reward + 100) / 10 +  # 奖励归一化
                0.2 * max(0, 100 - mean_convergence / 4) +  # 收敛速度
                0.1 * max(0, 100 - mean_inference * 10)  # 推理速度
            )
            scores[method] = score
        
        # 排名
        sorted_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rankings = {m: i+1 for i, (m, _) in enumerate(sorted_methods)}
        best_method = sorted_methods[0][0] if sorted_methods else ""
        
        return ComparisonResult(
            methods=methods,
            all_results=all_results,
            scores=scores,
            rankings=rankings,
            best_method=best_method
        )
    
    def _print_validation_summary(self, results: Dict[str, MethodResult]):
        """打印快速验证结果"""
        print("\n" + "="*70)
        print("📊 快速验证结果")
        print("="*70)
        print(f"{'方法':<15} {'奖励':<12} {'达标率':<12} {'Pmax误差':<15} {'时间':<10}")
        print("-"*70)
        
        for name, result in results.items():
            print(f"{name:<15} {result.final_reward:<12.2f} "
                  f"{result.accuracy_rate*100:<11.1f}% "
                  f"{result.pmax_error_mean:.2f}±{result.pmax_error_std:.2f}bar  "
                  f"{result.training_time:<10.1f}s")
        
        # 找出最佳方法
        best = max(results.items(), key=lambda x: x[1].accuracy_rate)
        print("\n" + "="*70)
        print(f"🏆 快速验证最佳方法: {best[0]} (达标率: {best[1].accuracy_rate*100:.1f}%)")
        print("="*70)
    
    def _print_comparison_summary(self, comparison: ComparisonResult):
        """打印完整对比结果"""
        print("\n" + "="*70)
        print("📊 完整对比实验结果")
        print("="*70)
        
        print(f"\n{'排名':<6} {'方法':<15} {'综合评分':<12} {'达标率':<12} {'奖励':<12} {'收敛':<10}")
        print("-"*70)
        
        # 按排名排序
        sorted_methods = sorted(comparison.rankings.items(), key=lambda x: x[1])
        
        for method, rank in sorted_methods:
            results = comparison.all_results.get(method, [])
            if not results:
                continue
            
            mean_accuracy = np.mean([r.accuracy_rate for r in results])
            mean_reward = np.mean([r.final_reward for r in results])
            mean_convergence = np.mean([r.convergence_episode for r in results])
            score = comparison.scores.get(method, 0)
            
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"{medal}{rank:<4} {method:<15} {score:<12.2f} "
                  f"{mean_accuracy*100:<11.1f}% {mean_reward:<12.2f} ep{mean_convergence:<8.0f}")
        
        print("\n" + "="*70)
        print(f"🏆 最优方法: {comparison.best_method}")
        print(f"   综合评分: {comparison.scores.get(comparison.best_method, 0):.2f}")
        print("="*70)
    
    def _save_results(self, comparison: ComparisonResult):
        """保存实验结果"""
        # 保存JSON摘要
        summary = {
            'methods': comparison.methods,
            'scores': comparison.scores,
            'rankings': comparison.rankings,
            'best_method': comparison.best_method,
            'config': asdict(self.config)
        }
        
        # 添加每个方法的详细结果
        for method, results in comparison.all_results.items():
            if results:
                summary[method] = {
                    'mean_accuracy': np.mean([r.accuracy_rate for r in results]),
                    'mean_reward': np.mean([r.final_reward for r in results]),
                    'std_reward': np.std([r.final_reward for r in results]),
                    'mean_convergence': np.mean([r.convergence_episode for r in results]),
                    'mean_training_time': np.mean([r.training_time for r in results]),
                }
        
        # 保存
        save_path = os.path.join(self.config.save_dir, 'comparison_results.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n结果已保存到: {save_path}")
    
    def plot_comparison_results(self, comparison: ComparisonResult, save_dir: str = None):
        """
        绘制五种方法对比图
        
        包含：
        1. 达标率对比柱状图
        2. 学习曲线对比
        3. 训练时间对比
        4. 综合雷达图
        """
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Matplotlib未安装，跳过可视化")
            return
        
        if save_dir is None:
            save_dir = self.config.save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 颜色方案
        colors = {
            'PID': '#95a5a6',      # 灰色
            'DQN': '#3498db',      # 蓝色
            'SAC': '#e74c3c',      # 红色
            'TDMPC2': '#2ecc71',   # 绿色 (最优)
            'DPMD': '#f39c12',     # 橙色
        }
        
        methods = comparison.methods
        
        # 计算平均指标
        avg_metrics = {}
        for method in methods:
            results = comparison.all_results.get(method, [])
            if results:
                avg_metrics[method] = {
                    'accuracy': np.mean([r.accuracy_rate for r in results]) * 100,
                    'reward': np.mean([r.final_reward for r in results]),
                    'convergence': np.mean([r.convergence_episode for r in results]),
                    'time': np.mean([r.training_time for r in results]),
                }
        
        # ============ 图1：达标率对比柱状图 ============
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        method_names = list(avg_metrics.keys())
        accuracies = [avg_metrics[m]['accuracy'] for m in method_names]
        bar_colors = [colors.get(m, '#34495e') for m in method_names]
        
        bars = ax1.bar(method_names, accuracies, color=bar_colors, alpha=0.8, edgecolor='black')
        
        # 标注数值
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax1.set_ylabel('Pmax控制达标率 (%)', fontsize=13, fontweight='bold')
        ax1.set_title('五种方法Pmax控制达标率对比 (PID+DQN+SAC+TD-MPC2+DPMD)', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90%目标线')
        ax1.legend(fontsize=11)
        
        plt.tight_layout()
        accuracy_path = os.path.join(save_dir, 'accuracy_comparison.png')
        plt.savefig(accuracy_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📊 达标率对比图已保存: {accuracy_path}")
        
        # ============ 图2：学习曲线对比 ============
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        
        for method in methods:
            results = comparison.all_results.get(method, [])
            if results and results[0].reward_curve:
                # 取第一个种子的学习曲线（或多个种子平均）
                curve = results[0].reward_curve
                episodes = list(range(len(curve)))
                
                # 平滑处理
                window = min(10, len(curve) // 10)
                if window > 1:
                    smoothed = np.convolve(curve, np.ones(window)/window, mode='valid')
                    episodes_smooth = episodes[window-1:]
                else:
                    smoothed = curve
                    episodes_smooth = episodes
                
                ax2.plot(episodes_smooth, smoothed, label=method, 
                        color=colors.get(method, '#34495e'), linewidth=2.5, alpha=0.9)
        
        ax2.set_xlabel('训练Episode', fontsize=13, fontweight='bold')
        ax2.set_ylabel('累计奖励', fontsize=13, fontweight='bold')
        ax2.set_title('五种方法学习曲线对比', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=12, loc='lower right')
        ax2.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        learning_curve_path = os.path.join(save_dir, 'learning_curves.png')
        plt.savefig(learning_curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 学习曲线对比图已保存: {learning_curve_path}")
        
        # ============ 图3：综合性能对比 ============
        fig3 = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig3, hspace=0.3, wspace=0.3)
        
        # 子图1：达标率
        ax31 = fig3.add_subplot(gs[0, 0])
        ax31.bar(method_names, accuracies, color=bar_colors, alpha=0.8, edgecolor='black')
        ax31.set_ylabel('达标率 (%)', fontweight='bold')
        ax31.set_title('(a) Pmax控制达标率', fontweight='bold')
        ax31.grid(axis='y', alpha=0.3)
        
        # 子图2：平均奖励
        ax32 = fig3.add_subplot(gs[0, 1])
        rewards = [avg_metrics[m]['reward'] for m in method_names]
        ax32.bar(method_names, rewards, color=bar_colors, alpha=0.8, edgecolor='black')
        ax32.set_ylabel('平均奖励', fontweight='bold')
        ax32.set_title('(b) 训练终期平均奖励', fontweight='bold')
        ax32.grid(axis='y', alpha=0.3)
        
        # 子图3：收敛速度
        ax33 = fig3.add_subplot(gs[1, 0])
        convergences = [avg_metrics[m]['convergence'] for m in method_names]
        ax33.bar(method_names, convergences, color=bar_colors, alpha=0.8, edgecolor='black')
        ax33.set_ylabel('收敛Episode', fontweight='bold')
        ax33.set_title('(c) 收敛速度 (越小越好)', fontweight='bold')
        ax33.grid(axis='y', alpha=0.3)
        
        # 子图4：训练时间
        ax34 = fig3.add_subplot(gs[1, 1])
        times = [avg_metrics[m]['time'] for m in method_names]
        ax34.bar(method_names, times, color=bar_colors, alpha=0.8, edgecolor='black')
        ax34.set_ylabel('训练时间 (秒)', fontweight='bold')
        ax34.set_title('(d) 训练耗时', fontweight='bold')
        ax34.grid(axis='y', alpha=0.3)
        
        plt.suptitle('五种控制方法综合性能对比', fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        comprehensive_path = os.path.join(save_dir, 'five_method_comparison.png')
        plt.savefig(comprehensive_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📊 综合对比图已保存: {comprehensive_path}")
        
        # 同时保存到visualization_output目录
        vis_output_dir = 'visualization_output'
        os.makedirs(vis_output_dir, exist_ok=True)
        
        import shutil
        try:
            # 复制5方法对比图到visualization_output
            shutil.copy(accuracy_path, os.path.join(vis_output_dir, 'five_method_accuracy.png'))
            shutil.copy(learning_curve_path, os.path.join(vis_output_dir, 'five_method_learning_curves.png'))
            shutil.copy(comprehensive_path, os.path.join(vis_output_dir, 'five_method_comparison.png'))
            print(f"\n✅ 可视化图表已同步到: {vis_output_dir}/")
        except Exception as e:
            print(f"⚠️ 同步到visualization_output失败: {e}")
        
        print(f"\n✅ 所有可视化图表已生成在: {save_dir}")


# ============================================================
# 可视化辅助函数
# ============================================================

def plot_training_progress(method_name: str, reward_curve: List[float], 
                           save_path: str = None):
    """
    绘制单个方法的训练进度图
    
    Args:
        method_name: 方法名称
        reward_curve: 奖励曲线
        save_path: 保存路径
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    plt.figure(figsize=(10, 6))
    
    episodes = list(range(len(reward_curve)))
    
    # 原始曲线（透明）
    plt.plot(episodes, reward_curve, alpha=0.3, color='#3498db', linewidth=1)
    
    # 平滑曲线
    window = min(20, len(reward_curve) // 10)
    if window > 1:
        smoothed = np.convolve(reward_curve, np.ones(window)/window, mode='valid')
        episodes_smooth = episodes[window-1:]
        plt.plot(episodes_smooth, smoothed, color='#e74c3c', linewidth=2.5, label='平滑曲线')
    
    plt.xlabel('训练Episode', fontsize=12, fontweight='bold')
    plt.ylabel('累计奖励', fontsize=12, fontweight='bold')
    plt.title(f'{method_name} 训练进度', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📈 训练进度图已保存: {save_path}")
    
    plt.close()


# ============================================================
# 主入口
# ============================================================

def main():
    """主函数"""
    print("="*70)
    print("🔬 柴油机控制方法对比实验")
    print("="*70)
    
    # 创建配置
    config = ExperimentConfig(
        n_episodes=400,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\n实验配置:")
    print(f"  - Episodes: {config.n_episodes}")
    print(f"  - Batch Size: {config.batch_size}")
    print(f"  - Device: {config.device}")
    print(f"  - Seeds: {config.seeds}")
    
    # 创建实验
    experiment = FiveMethodComparison(config)
    
    # 先快速验证
    print("\n" + "-"*70)
    print("步骤1: 快速验证 (单种子)")
    print("-"*70)
    
    validation_results = experiment.run_quick_validation()
    
    # 询问是否继续完整实验
    print("\n快速验证完成! 是否继续完整实验? (y/n)")
    # 在自动化场景下默认继续
    # user_input = input().strip().lower()
    user_input = 'y'  # 自动继续
    
    if user_input == 'y':
        print("\n" + "-"*70)
        print("步骤2: 完整对比实验 (5个种子)")
        print("-"*70)
        
        comparison = experiment.run_full_comparison()
        
        print(f"\n✅ 实验完成! 推荐使用: {comparison.best_method}")
    else:
        print("\n已跳过完整实验")


if __name__ == "__main__":
    main()
