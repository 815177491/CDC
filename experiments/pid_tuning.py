"""
PID 参数自动调优脚本
=====================
使用网格搜索为 PID 控制器寻找最优参数

方法：
1. 网格搜索 (Grid Search)
2. 基于 ITAE (Integral of Time-weighted Absolute Error) 准则

参数来源说明：
- 初始参数范围基于 Ziegler-Nichols 方法的经验值
- Kp: 0.1 ~ 2.0 (比例增益)
- Ki: 0.01 ~ 0.5 (积分增益)
- Kd: 0.01 ~ 0.2 (微分增益)
"""

import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
from tqdm import tqdm
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class PIDTuningConfig:
    """PID 调参配置"""
    # 参数搜索范围
    kp_range: Tuple[float, float, int] = (0.1, 2.0, 10)  # (min, max, steps)
    ki_range: Tuple[float, float, int] = (0.01, 0.5, 10)
    kd_range: Tuple[float, float, int] = (0.01, 0.2, 5)
    
    # 评估配置
    n_episodes: int = 20
    max_steps: int = 200
    target_pmax: float = 150.0
    
    # 保存路径
    save_dir: str = './pid_tuning_results'


class PIDController:
    """可调参的 PID 控制器"""
    
    def __init__(self, kp: float, ki: float, kd: float, target: float = 150.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.reset()
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
    
    def compute(self, current_value: float) -> float:
        """计算 PID 输出"""
        error = (self.target - current_value) / self.target
        
        self.integral += error
        self.integral = np.clip(self.integral, -10, 10)  # 防止积分饱和
        
        derivative = error - self.prev_error
        self.prev_error = error
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return output


class PIDTuner:
    """PID 参数调优器"""
    
    def __init__(self, config: PIDTuningConfig):
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict] = []
    
    def evaluate_params(self, kp: float, ki: float, kd: float) -> Dict[str, float]:
        """
        评估一组 PID 参数
        
        返回:
            - itae: 时间加权绝对误差积分 (越小越好)
            - overshoot: 最大超调量
            - settling_time: 稳定时间
            - steady_state_error: 稳态误差
        """
        from marl.env import EngineEnv, EngineEnvConfig
        
        env_config = EngineEnvConfig(
            max_steps=self.config.max_steps,
            difficulty=0.5
        )
        env = EngineEnv(env_config)
        
        pid = PIDController(kp, ki, kd, self.config.target_pmax)
        
        all_itae = []
        all_overshoot = []
        all_sse = []
        
        for ep in range(self.config.n_episodes):
            obs, _ = env.reset()
            pid.reset()
            
            itae = 0.0
            max_value = 0.0
            final_values = []
            
            for step in range(self.config.max_steps):
                # 简单阈值诊断
                pmax = obs['diag'][0] * 200.0
                error = abs(pmax - self.config.target_pmax) / self.config.target_pmax
                
                if error > 0.1:
                    diag = {'fault_type': 1, 'severity': min(error * 2, 1.0), 'confidence': 0.7}
                else:
                    diag = {'fault_type': 0, 'severity': 0.0, 'confidence': 0.9}
                
                # PID 控制
                output = pid.compute(pmax)
                timing_offset = np.clip(output * 5, -5, 5)
                fuel_adj = np.clip(1.0 + output * 0.1, 0.85, 1.15)
                
                ctrl = {
                    'timing_offset': timing_offset,
                    'fuel_adj': fuel_adj,
                    'protection_level': 0
                }
                
                actions = {'diag': diag, 'ctrl': ctrl}
                obs, _, terminated, truncated, info = env.step(actions)
                
                # 计算 ITAE
                current_pmax = info.get('Pmax', pmax)
                error_abs = abs(current_pmax - self.config.target_pmax)
                itae += (step + 1) * error_abs
                
                max_value = max(max_value, current_pmax)
                
                if step >= self.config.max_steps - 10:
                    final_values.append(current_pmax)
                
                if terminated or truncated:
                    break
            
            all_itae.append(itae)
            overshoot = (max_value - self.config.target_pmax) / self.config.target_pmax * 100
            all_overshoot.append(max(0, overshoot))
            
            if final_values:
                sse = abs(np.mean(final_values) - self.config.target_pmax) / self.config.target_pmax
                all_sse.append(sse)
        
        return {
            'itae': np.mean(all_itae),
            'overshoot': np.mean(all_overshoot),
            'steady_state_error': np.mean(all_sse) if all_sse else 0.0,
            'kp': kp,
            'ki': ki,
            'kd': kd
        }
    
    def grid_search(self) -> Dict:
        """网格搜索最优参数"""
        kp_values = np.linspace(*self.config.kp_range)
        ki_values = np.linspace(*self.config.ki_range)
        kd_values = np.linspace(*self.config.kd_range)
        
        total = len(kp_values) * len(ki_values) * len(kd_values)
        print(f"开始网格搜索: {total} 组参数组合")
        
        best_result = None
        best_itae = float('inf')
        
        with tqdm(total=total, desc="PID 调参") as pbar:
            for kp in kp_values:
                for ki in ki_values:
                    for kd in kd_values:
                        result = self.evaluate_params(kp, ki, kd)
                        self.results.append(result)
                        
                        if result['itae'] < best_itae:
                            best_itae = result['itae']
                            best_result = result
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'best_itae': f"{best_itae:.1f}",
                            'kp': f"{best_result['kp']:.2f}" if best_result else '-'
                        })
        
        return best_result
    
    def save_results(self, best_params: Dict):
        """保存调参结果"""
        # 保存所有结果
        with open(self.save_dir / 'all_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 保存最优参数
        with open(self.save_dir / 'best_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        
        # 生成报告
        report = f"""
# PID 参数调优报告

## 搜索空间
- Kp: {self.config.kp_range[0]} ~ {self.config.kp_range[1]} ({self.config.kp_range[2]} steps)
- Ki: {self.config.ki_range[0]} ~ {self.config.ki_range[1]} ({self.config.ki_range[2]} steps)
- Kd: {self.config.kd_range[0]} ~ {self.config.kd_range[1]} ({self.config.kd_range[2]} steps)

## 评估配置
- 评估回合数: {self.config.n_episodes}
- 每回合最大步数: {self.config.max_steps}
- 目标 Pmax: {self.config.target_pmax} bar

## 最优参数
- **Kp**: {best_params['kp']:.4f}
- **Ki**: {best_params['ki']:.4f}
- **Kd**: {best_params['kd']:.4f}

## 性能指标
- ITAE (时间加权误差积分): {best_params['itae']:.2f}
- 超调量: {best_params['overshoot']:.2f}%
- 稳态误差: {best_params['steady_state_error']:.4f}

## 参数来源说明
初始搜索范围基于 Ziegler-Nichols 开环调参法的经验值:
- Kp 范围覆盖从弱响应到强响应
- Ki 范围确保消除稳态误差但避免积分饱和
- Kd 范围提供适度的超前控制

最终参数通过网格搜索结合 ITAE 准则优化得出。
        """
        
        with open(self.save_dir / 'tuning_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n结果已保存至: {self.save_dir}")
        print(f"最优参数: Kp={best_params['kp']:.4f}, Ki={best_params['ki']:.4f}, Kd={best_params['kd']:.4f}")


def run_pid_tuning():
    """运行 PID 调参"""
    config = PIDTuningConfig(
        kp_range=(0.1, 2.0, 8),
        ki_range=(0.01, 0.5, 6),
        kd_range=(0.01, 0.2, 4),
        n_episodes=10
    )
    
    tuner = PIDTuner(config)
    best_params = tuner.grid_search()
    tuner.save_results(best_params)
    
    return best_params


if __name__ == '__main__':
    run_pid_tuning()
