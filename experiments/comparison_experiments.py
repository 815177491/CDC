"""
对比实验框架
============
包含消融实验、基线方法和评估指标
"""

import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import time
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import MarineEngine0D, OperatingCondition


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str = "experiment"
    seed: int = 42
    n_episodes: int = 100
    max_steps_per_episode: int = 200
    difficulty: float = 0.5
    enable_variable_condition: bool = True
    device: str = 'cpu'
    save_dir: str = './experiment_results'


@dataclass
class EvaluationMetrics:
    """评估指标"""
    # 诊断性能
    diag_accuracy: float = 0.0
    diag_precision: Dict[int, float] = field(default_factory=dict)
    diag_recall: Dict[int, float] = field(default_factory=dict)
    diag_f1: Dict[int, float] = field(default_factory=dict)
    detection_delay_mean: float = 0.0
    detection_delay_std: float = 0.0
    false_alarm_rate: float = 0.0
    
    # 控制性能
    pmax_maintenance_rate: float = 0.0
    pmax_exceed_rate: float = 0.0  # Pmax 超限比例（新增）
    control_smoothness: float = 0.0
    protection_accuracy: float = 0.0
    
    # 鲁棒性
    variable_cond_accuracy: float = 0.0
    composite_fault_accuracy: float = 0.0
    
    # 效率
    training_time: float = 0.0
    inference_time_mean: float = 0.0
    convergence_episodes: int = 0
    
    # 奖励
    mean_episode_reward: float = 0.0
    std_episode_reward: float = 0.0


class BaselineMethod(ABC):
    """基线方法抽象基类"""
    
    @abstractmethod
    def diagnose(self, obs: np.ndarray) -> Dict:
        """执行诊断"""
        pass
    
    @abstractmethod
    def control(self, obs: np.ndarray, diagnosis: Dict) -> Dict:
        """执行控制"""
        pass
    
    @abstractmethod
    def name(self) -> str:
        pass


class ThresholdBaseline(BaselineMethod):
    """阈值法基线"""
    
    def __init__(
        self,
        pmax_threshold: float = 0.1,
        pcomp_threshold: float = 0.1,
        texh_threshold: float = 0.15
    ):
        self.pmax_th = pmax_threshold
        self.pcomp_th = pcomp_threshold
        self.texh_th = texh_threshold
        
        # 基准值
        self.pmax_base = 150.0
        self.pcomp_base = 120.0
        self.texh_base = 400.0
    
    def diagnose(self, obs: np.ndarray) -> Dict:
        # 解析观测（假设前3维是Pmax, Pcomp, Texh归一化值）
        pmax = obs[0] * 200.0
        pcomp = obs[1] * 150.0
        texh = obs[2] * 500.0
        
        # 计算残差
        r_pmax = abs(pmax - self.pmax_base) / self.pmax_base
        r_pcomp = abs(pcomp - self.pcomp_base) / self.pcomp_base
        r_texh = abs(texh - self.texh_base) / self.texh_base
        
        # 基于阈值判断
        fault_type = 0  # 健康
        severity = 0.0
        
        if r_pcomp > self.pcomp_th and r_pmax > self.pmax_th:
            fault_type = 2  # 泄漏
            severity = (r_pcomp + r_pmax) / 2
        elif r_pmax > self.pmax_th:
            fault_type = 1  # 正时
            severity = r_pmax
        elif r_texh > self.texh_th:
            fault_type = 3  # 燃油
            severity = r_texh
        
        return {
            'fault_type': fault_type,
            'severity': min(severity * 2, 1.0),
            'confidence': 0.8 if fault_type > 0 else 0.9
        }
    
    def control(self, obs: np.ndarray, diagnosis: Dict) -> Dict:
        # 简单规则控制
        fault_type = diagnosis['fault_type']
        severity = diagnosis['severity']
        
        timing_offset = 0.0
        fuel_adj = 1.0
        protection = 0
        
        if fault_type == 1:  # 正时故障
            timing_offset = -severity * 2  # 反向补偿
        elif fault_type == 2:  # 泄漏
            fuel_adj = 1.0 + severity * 0.1  # 增加燃油
        elif fault_type == 3:  # 燃油故障
            fuel_adj = 1.0 + severity * 0.15
        
        if severity > 0.7:
            protection = 2  # 降功率
        elif severity > 0.5:
            protection = 1  # 警告
        
        return {
            'timing_offset': timing_offset,
            'fuel_adj': fuel_adj,
            'protection_level': protection
        }
    
    def name(self) -> str:
        return "Threshold"


class PIDControlBaseline(BaselineMethod):
    """
    PID控制基线
    
    参数来源说明:
    - 默认参数 (kp=0.5, ki=0.1, kd=0.05) 基于 Ziegler-Nichols 开环调参法经验值
    - 可通过 load_tuned_params() 加载网格搜索优化后的参数
    - 调参脚本: experiments/pid_tuning.py
    - 评估准则: ITAE (Integral of Time-weighted Absolute Error)
    """
    
    def __init__(self, kp: float = 0.5, ki: float = 0.1, kd: float = 0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.pmax_target = 150.0
    
    @classmethod
    def load_tuned_params(cls, params_path: str = './pid_tuning_results/best_params.json'):
        """
        从调优结果加载最优参数
        
        Args:
            params_path: 调优参数文件路径
            
        Returns:
            使用最优参数初始化的 PIDControlBaseline 实例
        """
        try:
            with open(params_path, 'r') as f:
                params = json.load(f)
            print(f"[PID] 加载调优参数: Kp={params['kp']:.3f}, Ki={params['ki']:.3f}, Kd={params['kd']:.3f}")
            return cls(kp=params['kp'], ki=params['ki'], kd=params['kd'])
        except FileNotFoundError:
            print(f"[PID] 未找到调优参数文件 {params_path}，使用默认参数")
            return cls()
    
    def reset(self):
        """重置控制器状态（每个 episode 开始时调用）"""
        self.integral = 0.0
        self.prev_error = 0.0
    
    def diagnose(self, obs: np.ndarray) -> Dict:
        # PID基线使用简单阈值诊断
        pmax = obs[0] * 200.0
        error = abs(pmax - self.pmax_target) / self.pmax_target
        
        if error > 0.1:
            return {'fault_type': 1, 'severity': min(error * 2, 1.0), 'confidence': 0.7}
        return {'fault_type': 0, 'severity': 0.0, 'confidence': 0.9}
    
    def control(self, obs: np.ndarray, diagnosis: Dict) -> Dict:
        pmax = obs[0] * 200.0
        error = (self.pmax_target - pmax) / self.pmax_target
        
        # PID计算（带积分抗饱和）
        self.integral += error
        self.integral = np.clip(self.integral, -10, 10)  # 抗积分饱和
        derivative = error - self.prev_error
        self.prev_error = error
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        timing_offset = np.clip(output * 5, -5, 5)
        fuel_adj = np.clip(1.0 + output * 0.1, 0.85, 1.15)
        
        return {
            'timing_offset': timing_offset,
            'fuel_adj': fuel_adj,
            'protection_level': 0
        }
    
    def name(self) -> str:
        return f"PID(Kp={self.kp:.2f})"


class ExperimentRunner:
    """
    实验运行器
    
    支持多种方法的对比实验
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        self.results: Dict[str, EvaluationMetrics] = {}
    
    def evaluate_method(
        self,
        method_name: str,
        diagnose_fn: Callable,
        control_fn: Callable
    ) -> EvaluationMetrics:
        """评估单个方法"""
        print(f"\n评估方法: {method_name}")
        
        # 导入环境
        from marl.env import EngineEnv, EngineEnvConfig
        
        env_config = EngineEnvConfig(
            max_steps=self.config.max_steps_per_episode,
            difficulty=self.config.difficulty,
            enable_variable_condition=self.config.enable_variable_condition
        )
        env = EngineEnv(env_config)
        
        metrics = EvaluationMetrics()
        
        # 收集评估数据
        all_diag_correct = []
        all_detection_delays = []
        all_pmax_ratios = []
        all_timing_changes = []
        episode_rewards = []
        inference_times = []
        false_alarms = 0
        total_healthy = 0
        
        confusion_matrix = np.zeros((4, 4), dtype=int)
        
        for ep in tqdm(range(self.config.n_episodes), desc=method_name):
            obs, info = env.reset()
            episode_reward = 0
            detected = False
            detection_delay = 0
            
            for step in range(self.config.max_steps_per_episode):
                # 诊断
                start_time = time.time()
                diag = diagnose_fn(obs['diag'])
                ctrl = control_fn(obs['ctrl'], diag)
                inference_times.append(time.time() - start_time)
                
                # 执行动作
                actions = {'diag': diag, 'ctrl': ctrl}
                next_obs, rewards, terminated, truncated, info = env.step(actions)
                
                # 记录指标
                true_fault = info.get('fault_label', 0)
                pred_fault = diag['fault_type']
                
                confusion_matrix[true_fault, pred_fault] += 1
                all_diag_correct.append(int(true_fault == pred_fault))
                
                # 检测延迟
                if true_fault > 0 and not detected:
                    if pred_fault == true_fault:
                        detected = True
                        all_detection_delays.append(detection_delay)
                    else:
                        detection_delay += 1
                
                # 误报
                if true_fault == 0:
                    total_healthy += 1
                    if pred_fault > 0:
                        false_alarms += 1
                
                # Pmax维持
                pmax = info.get('Pmax', 150)
                all_pmax_ratios.append(pmax / 150.0)
                
                # 控制平滑度
                all_timing_changes.append(abs(ctrl['timing_offset']))
                
                episode_reward += rewards['diag'] + rewards['ctrl']
                obs = next_obs
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
        
        # 计算指标
        metrics.diag_accuracy = np.mean(all_diag_correct)
        
        for i in range(4):
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            fn = confusion_matrix[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics.diag_precision[i] = precision
            metrics.diag_recall[i] = recall
            metrics.diag_f1[i] = f1
        
        if all_detection_delays:
            metrics.detection_delay_mean = np.mean(all_detection_delays)
            metrics.detection_delay_std = np.std(all_detection_delays)
        
        metrics.false_alarm_rate = false_alarms / total_healthy if total_healthy > 0 else 0
        # 修复：移除幸存者偏差过滤，统计所有数据
        metrics.pmax_maintenance_rate = np.mean(all_pmax_ratios) if all_pmax_ratios else 1.0
        metrics.pmax_exceed_rate = np.mean([1 if r > 1.1 else 0 for r in all_pmax_ratios]) if all_pmax_ratios else 0.0
        metrics.control_smoothness = 1.0 / (1.0 + np.mean(all_timing_changes))
        metrics.mean_episode_reward = np.mean(episode_rewards)
        metrics.std_episode_reward = np.std(episode_rewards)
        metrics.inference_time_mean = np.mean(inference_times) * 1000  # ms
        
        self.results[method_name] = metrics
        return metrics
    
    def run_baseline_experiments(self) -> Dict[str, EvaluationMetrics]:
        """运行所有基线实验"""
        
        # 阈值法
        threshold = ThresholdBaseline()
        self.evaluate_method(
            "Threshold",
            threshold.diagnose,
            threshold.control
        )
        
        # PID控制
        pid = PIDControlBaseline()
        self.evaluate_method(
            "PID",
            pid.diagnose,
            pid.control
        )
        
        return self.results
    
    def run_ablation_experiments(
        self,
        full_method_diag: Callable,
        full_method_ctrl: Callable,
        mlp_diag: Optional[Callable] = None,
        pikan_no_physics_diag: Optional[Callable] = None
    ) -> Dict[str, EvaluationMetrics]:
        """
        运行消融实验
        
        Args:
            full_method_diag: 完整方法的诊断函数 (PIKAN)
            full_method_ctrl: 完整方法的控制函数
            mlp_diag: MLP基线诊断函数（用于 MLP vs KAN 消融）
            pikan_no_physics_diag: 无物理约束的PIKAN诊断函数
        """
        
        # ===== 消融1：完整方法 vs 只有诊断 vs 只有控制 =====
        
        # 完整方法
        self.evaluate_method("PIKAN+TDMPC2", full_method_diag, full_method_ctrl)
        
        # 只有诊断（无控制）
        def no_control(obs, diag):
            return {'timing_offset': 0, 'fuel_adj': 1.0, 'protection_level': 0}
        
        self.evaluate_method("Only_Diagnosis", full_method_diag, no_control)
        
        # 只有控制（无诊断）
        def no_diagnosis(obs):
            return {'fault_type': 0, 'severity': 0.0, 'confidence': 1.0}
        
        self.evaluate_method("Only_Control", no_diagnosis, full_method_ctrl)
        
        # ===== 消融2：MLP vs KAN（如果提供了MLP基线）=====
        if mlp_diag is not None:
            self.evaluate_method("MLP_Baseline", mlp_diag, full_method_ctrl)
        
        # ===== 消融3：With/Without Physics Loss =====
        if pikan_no_physics_diag is not None:
            self.evaluate_method("PIKAN_NoPhysics", pikan_no_physics_diag, full_method_ctrl)
        
        return self.results
    
    def run_architecture_ablation(
        self,
        agents_dict: Dict[str, Tuple[Callable, Callable]]
    ) -> Dict[str, EvaluationMetrics]:
        """
        运行架构消融实验
        
        Args:
            agents_dict: 架构名称 -> (诊断函数, 控制函数) 的映射
                示例: {
                    'PINN+KAN': (pikan_diag, ctrl),
                    'PINN_Only': (pinn_diag, ctrl),
                    'KAN_Only': (kan_diag, ctrl),
                    'MLP_Baseline': (mlp_diag, ctrl)
                }
        """
        for name, (diag_fn, ctrl_fn) in agents_dict.items():
            self.evaluate_method(name, diag_fn, ctrl_fn)
        
        return self.results
    
    def save_results(self):
        """保存实验结果"""
        results_dict = {}
        for method, metrics in self.results.items():
            results_dict[method] = asdict(metrics)
        
        with open(self.save_dir / 'results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n结果已保存至: {self.save_dir / 'results.json'}")
    
    def print_comparison_table(self):
        """打印对比表格"""
        print("\n" + "=" * 100)
        print("实验结果对比")
        print("=" * 100)
        
        headers = ["方法", "诊断准确率", "检测延迟", "误报率", "性能维持", "超限率", "推理时间"]
        row_format = "{:<20}" + "{:<12}" * (len(headers) - 1)
        
        print(row_format.format(*headers))
        print("-" * 100)
        
        for method, metrics in self.results.items():
            row = [
                method,
                f"{metrics.diag_accuracy:.3f}",
                f"{metrics.detection_delay_mean:.1f}±{metrics.detection_delay_std:.1f}",
                f"{metrics.false_alarm_rate:.3f}",
                f"{metrics.pmax_maintenance_rate:.3f}",
                f"{metrics.pmax_exceed_rate:.3f}",
                f"{metrics.inference_time_mean:.2f}ms"
            ]
            print(row_format.format(*row))
        
        print("=" * 100)


def run_experiments():
    """运行完整实验套件"""
    config = ExperimentConfig(
        name="marl_comparison",
        n_episodes=50,
        difficulty=0.5,
        seed=42
    )
    
    runner = ExperimentRunner(config)
    
    # 基线实验
    print("运行基线实验...")
    runner.run_baseline_experiments()
    
    # 保存并打印结果
    runner.save_results()
    runner.print_comparison_table()


if __name__ == '__main__':
    run_experiments()
