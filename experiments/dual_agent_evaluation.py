#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双智能体评估系统
================
全面的诊-控协同系统评估

评估指标:
1. 诊断性能: 准确率、检测延迟、混淆矩阵、置信度校准
2. 控制性能: Pmax安全性、恢复时间、燃油效率
3. 协同效果: 诊断-控制的协同度、端到端性能
4. 鲁棒性: 未见工况、噪声鲁棒性、混合故障

对比: Baseline(规则诊断+RL控制) vs Proposed(RL诊断+RL控制)

Author: CDC Project
Date: 2026-01-24
"""

import os
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.dual_agent_env import create_dual_agent_env
from diagnosis.fault_injector import FaultType
from agents.rl_diagnosis_agent import create_rl_diagnosis_agent


# ============================================================
# 评估数据结构
# ============================================================

@dataclass
class DiagnosisMetrics:
    """诊断评估指标"""
    accuracy: float  # 准确率
    precision: Dict[str, float]  # 各类的精度
    recall: Dict[str, float]  # 各类的召回率
    f1_score: Dict[str, float]  # 各类的F1分数
    confusion_matrix: np.ndarray  # 混淆矩阵
    avg_detection_delay: float  # 平均检测延迟
    detection_delay_std: float  # 检测延迟标准差
    false_positive_rate: float  # 误报率
    false_negative_rate: float  # 漏报率
    mean_confidence: float  # 平均置信度
    confidence_calibration: float  # 置信度校准误差


@dataclass
class ControlMetrics:
    """控制评估指标"""
    pmax_rmse: float  # RMSE
    pmax_mae: float  # 平均绝对误差
    pmax_violations: float  # 超限次数
    pmax_overshoot: float  # 最大超限量
    recovery_time: float  # 故障恢复时间
    fuel_economy: float  # 燃油消耗相对值 (1.0为基准)
    control_smoothness: float  # 控制平稳性 (越小越好)
    setpoint_tracking_error: float  # 目标跟踪误差


@dataclass
class CooperationMetrics:
    """协同效果指标"""
    correct_diagnosis_rate: float  # 正确诊断占比
    correct_diag_control_success: float  # 正确诊断时控制成功率
    wrong_diag_control_failure: float  # 错误诊断时控制失败率
    end_to_end_success: float  # 端到端成功率 (诊+控都对)
    cooperation_score: float  # 协同得分 (0-1)


@dataclass
class RobustnessMetrics:
    """鲁棒性指标"""
    unseen_condition_accuracy: float  # 未见工况准确率
    noise_robustness: float  # 噪声鲁棒性 (相对于无噪声)
    multi_fault_accuracy: float  # 混合故障准确率
    worst_case_error: float  # 最坏情况误差


@dataclass
class ComparisonResult:
    """对比结果"""
    baseline_metrics: Dict
    proposed_metrics: Dict
    improvement: Dict  # 各指标的改进量
    statistical_significance: bool  # 是否显著改进


# ============================================================
# 评估器
# ============================================================

class DualAgentEvaluator:
    """双智能体评估器"""
    
    def __init__(self, 
                 diag_agent=None,
                 ctrl_agent=None,
                 env_config: Dict = None):
        """
        初始化评估器
        
        Args:
            diag_agent: 诊断智能体 (可选)
            ctrl_agent: 控制智能体 (可选)
            env_config: 环境配置
        """
        self.diag_agent = diag_agent
        self.ctrl_agent = ctrl_agent
        
        self.env = create_dual_agent_env(env_config or {})
        
        # 评估结果
        self.all_results = []
        
        print(f"[评估器] 初始化完成")
    
    def evaluate_diagnosis(self, n_episodes: int = 100,
                          noise_std: float = 0.0,
                          fault_types: Optional[List[FaultType]] = None) -> DiagnosisMetrics:
        """
        评估诊断性能
        
        Args:
            n_episodes: 评估episode数
            noise_std: 传感器噪声标准差
            fault_types: 指定故障类型列表 (None则使用所有)
        
        Returns:
            诊断指标
        """
        if self.diag_agent is None:
            print("[警告] 诊断智能体未指定")
            return None
        
        print(f"\n[诊断评估] 开始评估 ({n_episodes} episodes, 噪声={noise_std})")
        
        # 各类的统计
        fault_list = fault_types or FaultType
        confusion_data = defaultdict(lambda: defaultdict(int))
        detection_delays = []
        all_confidences = []
        all_predictions = []
        all_ground_truths = []
        
        for episode in range(n_episodes):
            # 重置
            self.diag_agent.reset()
            obs, info = self.env.reset(
                noise_std=noise_std if episode > n_episodes // 2 else 0.0
            )
            
            self.diag_agent.fault_onset_step = info.fault_onset_step
            self.diag_agent.first_correct_detection_step = None
            
            ground_truth = info.ground_truth_fault
            first_correct = False
            
            for step in range(self.env.config.max_steps):
                # 诊断
                diag_result = self.diag_agent.diagnose(
                    obs.measurements,
                    obs.residuals,
                    explore=False
                )
                
                # 更新环境
                ctrl_action = 0  # 只评估诊断，控制动作不重要
                obs, _, _, done, _ = self.env.step(0, ctrl_action)
                
                # 记录
                if diag_result.is_correct and not first_correct and ground_truth != FaultType.NONE:
                    first_correct = True
                    if info.fault_onset_step is not None:
                        delay = step - info.fault_onset_step
                        detection_delays.append(delay)
                
                all_predictions.append(diag_result.predicted_fault_type)
                all_confidences.append(diag_result.confidence)
                all_ground_truths.append(ground_truth)
                
                if done:
                    break
            
            # 混淆矩阵
            pred = all_predictions[-1] if all_predictions else ground_truth
            confusion_data[ground_truth.name][pred.name] += 1
        
        # 计算指标
        return self._compute_diagnosis_metrics(
            all_ground_truths, all_predictions, all_confidences,
            detection_delays, confusion_data
        )
    
    def evaluate_control(self, n_episodes: int = 100,
                        with_diagnosis: bool = True) -> ControlMetrics:
        """
        评估控制性能
        
        Args:
            n_episodes: 评估episode数
            with_diagnosis: 是否与诊断结合
        
        Returns:
            控制指标
        """
        if self.ctrl_agent is None:
            print("[警告] 控制智能体未指定")
            return None
        
        print(f"\n[控制评估] 开始评估 ({n_episodes} episodes, with_diag={with_diagnosis})")
        
        # 记录数据
        pmax_errors = []
        pmax_violations = []
        recovery_times = []
        fuel_consumptions = []
        control_diffs = []
        
        for episode in range(n_episodes):
            obs, info = self.env.reset()
            
            # 重置诊断智能体
            if with_diagnosis and self.diag_agent is not None:
                self.diag_agent.reset()
            
            episode_pmax_errors = []
            episode_fuel = []
            episode_control_actions = []
            fault_onset_step = info.fault_onset_step
            first_fault_detected_step = None
            
            for step in range(self.env.config.max_steps):
                # 诊断
                if with_diagnosis and self.diag_agent is not None:
                    diag_result = self.diag_agent.diagnose(
                        obs.measurements, obs.residuals, explore=False
                    )
                    is_diagnosis_correct = diag_result.is_correct
                    if is_diagnosis_correct and info.ground_truth_fault != FaultType.NONE:
                        if first_fault_detected_step is None:
                            first_fault_detected_step = step
                
                # 控制
                ctrl_action = self.ctrl_agent.select_action(
                    obs.ctrl_state, explore=False
                )
                
                obs, _, ctrl_r, done, info = self.env.step(0, ctrl_action)
                
                # 记录
                pmax_error = abs(info.Pmax - self.env.config.Pmax_target)
                episode_pmax_errors.append(pmax_error)
                episode_fuel.append(self.env.current_fuel)
                episode_control_actions.append(ctrl_action)
                
                if info.Pmax > self.env.config.Pmax_limit:
                    pmax_violations.append(1)
                
                if done:
                    break
            
            # 恢复时间
            if fault_onset_step is not None and first_fault_detected_step is not None:
                recovery_time = first_fault_detected_step - fault_onset_step
                recovery_times.append(recovery_time)
            
            pmax_errors.extend(episode_pmax_errors)
            fuel_consumptions.extend(episode_fuel)
            control_diffs.extend(np.diff(episode_control_actions) if len(episode_control_actions) > 1 else [0])
        
        # 计算指标
        return self._compute_control_metrics(
            pmax_errors, pmax_violations, recovery_times,
            fuel_consumptions, control_diffs
        )
    
    def evaluate_cooperation(self, n_episodes: int = 100) -> CooperationMetrics:
        """
        评估诊断-控制协同效果
        
        Args:
            n_episodes: 评估episode数
        
        Returns:
            协同指标
        """
        print(f"\n[协同评估] 开始评估 ({n_episodes} episodes)")
        
        if self.diag_agent is None or self.ctrl_agent is None:
            print("[警告] 诊断或控制智能体未指定")
            return None
        
        # 统计数据
        correct_diag_count = 0
        wrong_diag_count = 0
        correct_diag_ctrl_success = 0
        wrong_diag_ctrl_failure = 0
        both_correct = 0
        
        for episode in range(n_episodes):
            self.diag_agent.reset()
            obs, info = self.env.reset()
            
            ground_truth = info.ground_truth_fault
            
            for step in range(self.env.config.max_steps):
                # 诊断
                diag_result = self.diag_agent.diagnose(
                    obs.measurements, obs.residuals, explore=False
                )
                diagnosis_correct = (diag_result.predicted_fault_type == ground_truth)
                
                # 控制
                ctrl_action = self.ctrl_agent.select_action(
                    obs.ctrl_state, explore=False
                )
                
                obs, diag_r, ctrl_r, done, info = self.env.step(0, ctrl_action)
                
                # 记录
                if diagnosis_correct:
                    correct_diag_count += 1
                    if ctrl_r > 0:
                        correct_diag_ctrl_success += 1
                        if info.Pmax <= self.env.config.Pmax_limit:
                            both_correct += 1
                else:
                    wrong_diag_count += 1
                    if ctrl_r < 0:
                        wrong_diag_ctrl_failure += 1
                
                if done:
                    break
        
        # 计算指标
        return self._compute_cooperation_metrics(
            correct_diag_count, wrong_diag_count,
            correct_diag_ctrl_success, wrong_diag_ctrl_failure,
            both_correct, n_episodes
        )
    
    def evaluate_robustness(self, 
                           noise_stds: List[float] = None,
                           unseen_speeds: Optional[List[float]] = None,
                           unseen_loads: Optional[List[float]] = None,
                           n_episodes_per_test: int = 20) -> RobustnessMetrics:
        """
        评估鲁棒性
        
        Args:
            noise_stds: 要测试的噪声标准差列表
            unseen_speeds: 未见工况转速
            unseen_loads: 未见工况负荷
            n_episodes_per_test: 每个测试的episodes数
        
        Returns:
            鲁棒性指标
        """
        if self.diag_agent is None:
            print("[警告] 诊断智能体未指定")
            return None
        
        noise_stds = noise_stds or [0.0, 0.01, 0.05, 0.1]
        
        print(f"\n[鲁棒性评估] 开始评估")
        
        # 基准准确率 (无噪声)
        baseline_acc = self.evaluate_diagnosis(
            n_episodes=n_episodes_per_test, noise_std=0.0
        ).accuracy
        
        # 噪声影响
        noise_accuracies = []
        for noise_std in noise_stds[1:]:
            acc = self.evaluate_diagnosis(
                n_episodes=n_episodes_per_test, noise_std=noise_std
            ).accuracy
            noise_accuracies.append(acc)
        
        noise_robustness = np.mean(noise_accuracies) / baseline_acc if baseline_acc > 0 else 0
        
        # 未见工况
        unseen_acc = 0
        if unseen_speeds is not None:
            for speed in unseen_speeds:
                acc = 0
                for _ in range(n_episodes_per_test):
                    self.diag_agent.reset()
                    obs, info = self.env.reset(speed=speed)
                    
                    diag_result = self.diag_agent.diagnose(
                        obs.measurements, obs.residuals, explore=False
                    )
                    if diag_result.is_correct:
                        acc += 1
                
                unseen_acc += acc / n_episodes_per_test
        
        unseen_condition_acc = unseen_acc / len(unseen_speeds) if unseen_speeds else 0
        
        # 混合故障
        multi_fault_acc = 0
        for _ in range(n_episodes_per_test):
            self.diag_agent.reset()
            obs, info = self.env.reset()
            self.env._inject_random_fault()
            self.env._inject_random_fault()  # 第二个故障
            
            diag_result = self.diag_agent.diagnose(
                obs.measurements, obs.residuals, explore=False
            )
            if diag_result.is_correct:
                multi_fault_acc += 1
        
        multi_fault_acc /= n_episodes_per_test
        
        return RobustnessMetrics(
            unseen_condition_accuracy=unseen_condition_acc,
            noise_robustness=noise_robustness,
            multi_fault_accuracy=multi_fault_acc,
            worst_case_error=1.0 - min(noise_accuracies) if noise_accuracies else 1.0
        )
    
    def compare_baseline_vs_proposed(self,
                                     baseline_diag_fn,
                                     proposed_diag_agent,
                                     proposed_ctrl_agent,
                                     n_test_episodes: int = 50) -> ComparisonResult:
        """
        对比Baseline(规则诊断+RL控制)与Proposed(RL诊断+RL控制)
        
        Args:
            baseline_diag_fn: Baseline诊断函数 (输入observation, 输出DiagnosisResult)
            proposed_diag_agent: Proposed诊断智能体
            proposed_ctrl_agent: RL控制智能体
            n_test_episodes: 测试episode数
        
        Returns:
            对比结果
        """
        print(f"\n[对比评估] 开始A/B测试 ({n_test_episodes} episodes)")
        
        baseline_results = {'accuracy': [], 'control_reward': [], 'detection_delay': []}
        proposed_results = {'accuracy': [], 'control_reward': [], 'detection_delay': []}
        
        # 测试序列 (保证相同的故障配置)
        test_configs = []
        for i in range(n_test_episodes):
            fault_type = np.random.choice(list(FaultType))
            severity = np.random.uniform(0.3, 1.0)
            onset = np.random.uniform(0.0, 0.5)
            test_configs.append((fault_type, severity, onset))
        
        # Baseline评估
        for fault_type, severity, onset in test_configs:
            obs, info = self.env.reset(
                fault_type=fault_type,
                fault_severity=severity,
                fault_onset=onset
            )
            
            # Baseline诊断
            baseline_diag = baseline_diag_fn(obs)
            is_correct = (baseline_diag.fault_type == info.ground_truth_fault)
            baseline_results['accuracy'].append(is_correct)
            
            # RL控制
            total_ctrl_reward = 0
            for _ in range(self.env.config.max_steps):
                ctrl_action = self.ctrl_agent.select_action(
                    obs.ctrl_state, explore=False
                )
                obs, _, ctrl_r, done, _ = self.env.step(0, ctrl_action)
                total_ctrl_reward += ctrl_r
                if done:
                    break
            
            baseline_results['control_reward'].append(total_ctrl_reward)
        
        # Proposed评估
        for fault_type, severity, onset in test_configs:
            proposed_diag_agent.reset()
            obs, info = self.env.reset(
                fault_type=fault_type,
                fault_severity=severity,
                fault_onset=onset
            )
            
            # RL诊断
            diag_result = proposed_diag_agent.diagnose(
                obs.measurements, obs.residuals, explore=False
            )
            is_correct = (diag_result.predicted_fault_type == info.ground_truth_fault)
            proposed_results['accuracy'].append(is_correct)
            
            # RL控制
            total_ctrl_reward = 0
            for _ in range(self.env.config.max_steps):
                ctrl_action = proposed_ctrl_agent.select_action(
                    obs.ctrl_state, explore=False
                )
                obs, _, ctrl_r, done, _ = self.env.step(0, ctrl_action)
                total_ctrl_reward += ctrl_r
                if done:
                    break
            
            proposed_results['control_reward'].append(total_ctrl_reward)
        
        # 计算改进
        improvement = {
            'accuracy': (
                np.mean(proposed_results['accuracy']) - 
                np.mean(baseline_results['accuracy'])
            ),
            'control_reward': (
                np.mean(proposed_results['control_reward']) - 
                np.mean(baseline_results['control_reward'])
            )
        }
        
        # 显著性检验 (t-test)
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(
            np.array(proposed_results['accuracy'], dtype=float),
            np.array(baseline_results['accuracy'], dtype=float)
        )
        
        return ComparisonResult(
            baseline_metrics={
                'accuracy': np.mean(baseline_results['accuracy']),
                'control_reward': np.mean(baseline_results['control_reward'])
            },
            proposed_metrics={
                'accuracy': np.mean(proposed_results['accuracy']),
                'control_reward': np.mean(proposed_results['control_reward'])
            },
            improvement=improvement,
            statistical_significance=(p_value < 0.05)
        )
    
    # ============================================================
    # 辅助方法
    # ============================================================
    
    def _compute_diagnosis_metrics(self, ground_truths, predictions, confidences,
                                   detection_delays, confusion_data) -> DiagnosisMetrics:
        """计算诊断指标"""
        # 准确率
        accuracy = sum(gt == pred for gt, pred in zip(ground_truths, predictions)) / len(ground_truths)
        
        # 混淆矩阵和各类指标
        confusion_matrix = np.zeros((len(FaultType), len(FaultType)))
        precision = {}
        recall = {}
        f1_score = {}
        
        for i, fault in enumerate(FaultType):
            for j, pred in enumerate(FaultType):
                confusion_matrix[i, j] = confusion_data[fault.name].get(pred.name, 0)
            
            # 精度和召回
            tp = confusion_data[fault.name].get(fault.name, 0)
            fp = sum(confusion_data[other.name].get(fault.name, 0) 
                    for other in FaultType if other != fault)
            fn = sum(confusion_data[fault.name].get(other.name, 0)
                    for other in FaultType if other != fault)
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            
            precision[fault.name] = p
            recall[fault.name] = r
            f1_score[fault.name] = f
        
        # 检测延迟
        avg_delay = np.mean(detection_delays) if detection_delays else 0
        std_delay = np.std(detection_delays) if detection_delays else 0
        
        # 误报和漏报率
        false_positives = sum(gt == FaultType.NONE and pred != FaultType.NONE 
                            for gt, pred in zip(ground_truths, predictions))
        false_negatives = sum(gt != FaultType.NONE and pred == FaultType.NONE
                            for gt, pred in zip(ground_truths, predictions))
        
        fp_rate = false_positives / sum(gt == FaultType.NONE for gt in ground_truths) \
                  if any(gt == FaultType.NONE for gt in ground_truths) else 0
        fn_rate = false_negatives / sum(gt != FaultType.NONE for gt in ground_truths) \
                  if any(gt != FaultType.NONE for gt in ground_truths) else 0
        
        # 置信度校准
        mean_conf = np.mean(confidences)
        calibration_error = abs(accuracy - mean_conf)
        
        return DiagnosisMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            confusion_matrix=confusion_matrix,
            avg_detection_delay=avg_delay,
            detection_delay_std=std_delay,
            false_positive_rate=fp_rate,
            false_negative_rate=fn_rate,
            mean_confidence=mean_conf,
            confidence_calibration=calibration_error
        )
    
    def _compute_control_metrics(self, pmax_errors, pmax_violations,
                                 recovery_times, fuel_consumptions,
                                 control_diffs) -> ControlMetrics:
        """计算控制指标"""
        return ControlMetrics(
            pmax_rmse=np.sqrt(np.mean(np.array(pmax_errors)**2)),
            pmax_mae=np.mean(np.abs(pmax_errors)),
            pmax_violations=len(pmax_violations) / len(pmax_errors) if pmax_errors else 0,
            pmax_overshoot=max(pmax_errors) if pmax_errors else 0,
            recovery_time=np.mean(recovery_times) if recovery_times else 0,
            fuel_economy=np.mean(fuel_consumptions),
            control_smoothness=np.std(control_diffs),
            setpoint_tracking_error=np.mean(np.abs(pmax_errors))
        )
    
    def _compute_cooperation_metrics(self, correct_diag, wrong_diag,
                                    correct_diag_success, wrong_diag_failure,
                                    both_correct, total_steps) -> CooperationMetrics:
        """计算协同指标"""
        correct_diag_rate = correct_diag / total_steps if total_steps > 0 else 0
        correct_success_rate = correct_diag_success / correct_diag if correct_diag > 0 else 0
        wrong_failure_rate = wrong_diag_failure / wrong_diag if wrong_diag > 0 else 0
        end_to_end = both_correct / total_steps if total_steps > 0 else 0
        
        # 协同得分: 诊断正确时控制成功率高 + 诊断错误时控制失败率高
        coop_score = (correct_success_rate * correct_diag_rate +
                     wrong_failure_rate * (1 - correct_diag_rate)) / 2
        
        return CooperationMetrics(
            correct_diagnosis_rate=correct_diag_rate,
            correct_diag_control_success=correct_success_rate,
            wrong_diag_control_failure=wrong_failure_rate,
            end_to_end_success=end_to_end,
            cooperation_score=coop_score
        )
    
    def save_results(self, save_dir: str, name: str = "evaluation"):
        """保存评估结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存为JSON和CSV
        for result in self.all_results:
            filename = os.path.join(save_dir, f"{name}_{result.get('name', 'result')}.json")
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        
        print(f"[保存] 评估结果已保存至: {save_dir}")


if __name__ == "__main__":
    # 测试评估器
    print("=" * 60)
    print("评估器测试")
    print("=" * 60)
    
    evaluator = DualAgentEvaluator()
    
    # 评估诊断性能 (仅测试框架，不需要真实的智能体)
    print("\n框架测试完成！")
