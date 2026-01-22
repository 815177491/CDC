#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合诊断器 - KAN为主 + PINN为辅 (投票机制)
=========================================
结合两种2024-2025年最新方法进行故障诊断：
- KAN (主): 高效可解释，自动提取符号规则
- PINN (辅): 物理约束增强，提供机理验证

融合策略: 投票机制
1. KAN和PINN分别独立诊断
2. 如果一致 -> 直接输出
3. 如果不一致 -> 加权投票（KAN权重0.6，PINN权重0.4）
4. 同时输出物理残差和符号规则

优势:
- 结合可解释性和物理一致性
- 提高鲁棒性，减少误诊
- 综合利用数据驱动和物理驱动优势

Author: CDC Project
Date: 2026-01-22
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# 尝试导入深度学习库
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .fault_injector import FaultType
from .kan_diagnoser import KANDiagnoser, KANDiagnosisResult
from .pinn_diagnoser import PINNDiagnoser, PINNDiagnosisResult


class VoteStrategy(Enum):
    """投票策略"""
    WEIGHTED = "weighted"           # 加权投票
    CONFIDENCE = "confidence"       # 置信度优先
    PHYSICAL = "physical"          # 物理验证优先


@dataclass
class HybridDiagnosisResult:
    """混合诊断结果"""
    # 最终诊断结果
    fault_detected: bool
    fault_type: FaultType
    confidence: float
    
    # 投票详情
    vote_agreement: bool           # KAN和PINN是否一致
    kan_vote: FaultType           # KAN诊断结果
    pinn_vote: FaultType          # PINN诊断结果
    kan_confidence: float
    pinn_confidence: float
    
    # 可解释性信息
    symbolic_rules: List[str]      # KAN符号规则
    physics_residuals: Dict[str, float]  # PINN物理残差
    feature_importance: Dict[str, float] # 特征重要性
    
    # 综合解释
    explanation: str


if TORCH_AVAILABLE:
    
    class HybridDiagnoser:
        """
        混合故障诊断器
        
        结合KAN和PINN两种方法，使用投票机制融合结果
        
        使用方法:
        ```python
        diagnoser = HybridDiagnoser()
        diagnoser.train(train_data, val_data, epochs=100)
        result = diagnoser.diagnose(features)
        ```
        """
        
        def __init__(self, config: Dict = None):
            """
            初始化混合诊断器
            
            Args:
                config: 配置字典
                    - kan_weight: KAN投票权重 (默认0.6)
                    - pinn_weight: PINN投票权重 (默认0.4)
                    - strategy: 投票策略 (默认: weighted)
                    - confidence_threshold: 置信度阈值 (默认0.7)
            """
            config = config or {}
            
            # 投票权重
            self.kan_weight = config.get('kan_weight', 0.6)
            self.pinn_weight = config.get('pinn_weight', 0.4)
            assert abs(self.kan_weight + self.pinn_weight - 1.0) < 1e-6, "权重和必须为1"
            
            # 投票策略
            strategy_str = config.get('strategy', 'weighted')
            self.strategy = VoteStrategy(strategy_str)
            
            # 置信度阈值
            self.confidence_threshold = config.get('confidence_threshold', 0.7)
            
            # 创建子诊断器
            kan_config = config.get('kan_config', {})
            pinn_config = config.get('pinn_config', {})
            
            self.kan = KANDiagnoser(kan_config)
            self.pinn = PINNDiagnoser(pinn_config)
            
            # 故障类型
            self.fault_types = [
                FaultType.NONE,
                FaultType.INJECTION_TIMING,
                FaultType.CYLINDER_LEAK,
                FaultType.FUEL_DEGRADATION
            ]
            
            print(f"[混合诊断器] 初始化完成")
            print(f"  - KAN权重: {self.kan_weight:.1%}")
            print(f"  - PINN权重: {self.pinn_weight:.1%}")
            print(f"  - 投票策略: {self.strategy.value}")
            print(f"  - 置信度阈值: {self.confidence_threshold:.1%}")
        
        def train(self, train_data: Dict[str, np.ndarray], 
                  val_data: Dict[str, np.ndarray] = None,
                  epochs: int = 100, verbose: bool = True) -> Dict[str, Any]:
            """
            训练两个子模型
            
            Args:
                train_data: 训练数据
                    - features: (N, 8) 输入特征
                    - labels: (N,) 故障标签
                    - physics: (N, 3) 物理量 [Pmax, Pcomp, Texh] (PINN需要)
                val_data: 验证数据 (可选)
                epochs: 训练轮数
                verbose: 是否打印训练信息
            
            Returns:
                训练历史
            """
            history = {'kan': {}, 'pinn': {}}
            
            # 1. 训练KAN
            if verbose:
                print("\n" + "="*50)
                print("训练 KAN 诊断器")
                print("="*50)
            
            kan_train_data = {
                'features': train_data['features'],
                'labels': train_data['labels']
            }
            kan_val_data = None
            if val_data is not None:
                kan_val_data = {
                    'features': val_data['features'],
                    'labels': val_data['labels']
                }
            
            history['kan'] = self.kan.train(
                kan_train_data, kan_val_data, epochs, verbose
            )
            
            # 2. 训练PINN
            if verbose:
                print("\n" + "="*50)
                print("训练 PINN 诊断器")
                print("="*50)
            
            # PINN需要物理量标签
            if 'physics' not in train_data:
                # 如果没有提供物理量，使用简单估计
                if verbose:
                    print("[警告] 未提供物理量标签，使用估计值")
                physics = self._estimate_physics(train_data['features'])
            else:
                physics = train_data['physics']
            
            pinn_train_data = {
                'features': train_data['features'],
                'physics': physics,
                'labels': train_data['labels']
            }
            
            pinn_val_data = None
            if val_data is not None:
                if 'physics' not in val_data:
                    val_physics = self._estimate_physics(val_data['features'])
                else:
                    val_physics = val_data['physics']
                pinn_val_data = {
                    'features': val_data['features'],
                    'physics': val_physics,
                    'labels': val_data['labels']
                }
            
            history['pinn'] = self.pinn.train(
                pinn_train_data, pinn_val_data, epochs, verbose
            )
            
            if verbose:
                print("\n" + "="*50)
                print("混合诊断器训练完成")
                print("="*50)
            
            return history
        
        def _estimate_physics(self, features: np.ndarray) -> np.ndarray:
            """
            估计物理量 (当未提供时使用)
            
            基于柴油机经验模型估计 Pmax, Pcomp, Texh
            """
            n_samples = len(features)
            physics = np.zeros((n_samples, 3))
            
            # 假设特征顺序: [rpm, load, timing, boost, T_in, P_amb, fuel_q, run_h]
            load = features[:, 1] if features.shape[1] > 1 else np.ones(n_samples) * 0.7
            boost = features[:, 3] if features.shape[1] > 3 else np.ones(n_samples) * 1.5
            
            # 经验公式估计
            physics[:, 0] = 120 + 60 * load  # Pmax: 120-180 bar
            physics[:, 1] = boost * (14 ** 1.35)  # Pcomp: 压缩方程
            physics[:, 2] = 280 + 150 * load  # Texh: 280-430 °C
            
            return physics
        
        def diagnose(self, features: np.ndarray) -> HybridDiagnosisResult:
            """
            执行混合诊断
            
            Args:
                features: 输入特征 (8,) 或 (1, 8)
            
            Returns:
                混合诊断结果
            """
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # 1. KAN诊断
            kan_result = self.kan.diagnose(features)
            
            # 2. PINN诊断
            pinn_result = self.pinn.diagnose(features)
            
            # 3. 投票融合
            final_fault, final_confidence, vote_agreement = self._vote(
                kan_result, pinn_result
            )
            
            # 4. 生成综合解释
            explanation = self._generate_explanation(
                final_fault, final_confidence, vote_agreement,
                kan_result, pinn_result
            )
            
            return HybridDiagnosisResult(
                fault_detected=final_fault != FaultType.NONE,
                fault_type=final_fault,
                confidence=final_confidence,
                vote_agreement=vote_agreement,
                kan_vote=kan_result.fault_type,
                pinn_vote=pinn_result.fault_type,
                kan_confidence=kan_result.confidence,
                pinn_confidence=pinn_result.confidence,
                symbolic_rules=kan_result.symbolic_rules,
                physics_residuals=pinn_result.physics_residuals,
                feature_importance=kan_result.feature_importance,
                explanation=explanation
            )
        
        def _vote(self, kan_result: KANDiagnosisResult, 
                  pinn_result: PINNDiagnosisResult) -> Tuple[FaultType, float, bool]:
            """
            投票融合
            
            Returns:
                (最终故障类型, 最终置信度, 是否一致)
            """
            kan_fault = kan_result.fault_type
            pinn_fault = pinn_result.fault_type
            
            # 检查是否一致
            vote_agreement = (kan_fault == pinn_fault)
            
            if vote_agreement:
                # 一致情况：直接使用，置信度取加权平均
                final_fault = kan_fault
                final_confidence = (
                    self.kan_weight * kan_result.confidence +
                    self.pinn_weight * pinn_result.confidence
                )
            else:
                # 不一致情况：按策略决定
                if self.strategy == VoteStrategy.WEIGHTED:
                    # 加权投票：计算每个类别的加权得分
                    final_fault, final_confidence = self._weighted_vote(
                        kan_result, pinn_result
                    )
                    
                elif self.strategy == VoteStrategy.CONFIDENCE:
                    # 置信度优先：选择置信度高的
                    if kan_result.confidence >= pinn_result.confidence:
                        final_fault = kan_fault
                        final_confidence = kan_result.confidence * 0.9  # 降低置信度表示存在分歧
                    else:
                        final_fault = pinn_fault
                        final_confidence = pinn_result.confidence * 0.9
                        
                elif self.strategy == VoteStrategy.PHYSICAL:
                    # 物理验证优先：检查PINN物理残差
                    physics_ok = self._check_physics(pinn_result.physics_residuals)
                    if physics_ok:
                        final_fault = pinn_fault
                        final_confidence = pinn_result.confidence
                    else:
                        final_fault = kan_fault
                        final_confidence = kan_result.confidence * 0.85
                else:
                    # 默认回退到KAN
                    final_fault = kan_fault
                    final_confidence = kan_result.confidence * 0.8
            
            return final_fault, final_confidence, vote_agreement
        
        def _weighted_vote(self, kan_result: KANDiagnosisResult,
                           pinn_result: PINNDiagnosisResult) -> Tuple[FaultType, float]:
            """加权投票"""
            # 计算各类别的加权得分
            scores = {}
            
            for i, fault in enumerate(self.fault_types):
                kan_score = 1.0 if kan_result.fault_type == fault else 0.0
                pinn_score = 1.0 if pinn_result.fault_type == fault else 0.0
                
                # 加权置信度
                kan_weighted = kan_score * kan_result.confidence * self.kan_weight
                pinn_weighted = pinn_score * pinn_result.confidence * self.pinn_weight
                
                scores[fault] = kan_weighted + pinn_weighted
            
            # 选择得分最高的
            best_fault = max(scores, key=scores.get)
            best_score = scores[best_fault]
            
            # 归一化置信度
            total_score = sum(scores.values()) + 1e-8
            confidence = best_score / total_score
            
            return best_fault, confidence
        
        def _check_physics(self, residuals: Dict[str, float], threshold: float = 0.5) -> bool:
            """检查物理残差是否正常"""
            for key, value in residuals.items():
                if value > threshold:
                    return False
            return True
        
        def _generate_explanation(self, final_fault: FaultType, 
                                   final_confidence: float,
                                   vote_agreement: bool,
                                   kan_result: KANDiagnosisResult,
                                   pinn_result: PINNDiagnosisResult) -> str:
            """生成综合解释"""
            lines = []
            
            # 最终诊断结果
            if final_fault == FaultType.NONE:
                lines.append(f"✅ 诊断结论: 系统正常 (置信度: {final_confidence:.1%})")
            else:
                lines.append(f"⚠️ 诊断结论: {final_fault.value} (置信度: {final_confidence:.1%})")
            
            # 投票情况
            if vote_agreement:
                lines.append(f"📊 投票结果: KAN与PINN一致")
            else:
                lines.append(f"📊 投票结果: 存在分歧")
                lines.append(f"   - KAN诊断: {kan_result.fault_type.value} ({kan_result.confidence:.1%})")
                lines.append(f"   - PINN诊断: {pinn_result.fault_type.value} ({pinn_result.confidence:.1%})")
                lines.append(f"   采用 {self.strategy.value} 策略决策")
            
            # KAN符号规则
            if kan_result.symbolic_rules:
                lines.append(f"\n🔍 KAN发现的规则:")
                for rule in kan_result.symbolic_rules[:2]:
                    lines.append(f"   {rule}")
            
            # PINN物理残差
            lines.append(f"\n📐 物理约束检验:")
            for key, value in pinn_result.physics_residuals.items():
                status = "✓" if value < 0.5 else "✗"
                lines.append(f"   {status} {key}: {value:.4f}")
            
            # 关键特征
            top_features = sorted(
                kan_result.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            lines.append(f"\n🎯 关键特征: {', '.join([f'{k}({v:.1%})' for k, v in top_features])}")
            
            return '\n'.join(lines)
        
        def evaluate(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
            """
            评估混合诊断器
            
            Returns:
                评估指标
            """
            features = data['features']
            labels = data['labels']
            
            # 各模型单独评估
            kan_data = {'features': features, 'labels': labels}
            kan_eval = self.kan.evaluate(kan_data)
            
            # PINN评估需要物理量
            if 'physics' in data:
                physics = data['physics']
            else:
                physics = self._estimate_physics(features)
            pinn_data = {'features': features, 'physics': physics, 'labels': labels}
            pinn_eval = self.pinn.evaluate(pinn_data)
            
            # 混合诊断评估
            hybrid_correct = 0
            agreements = 0
            
            for i in range(len(features)):
                result = self.diagnose(features[i])
                true_fault = self.fault_types[labels[i]]
                
                if result.fault_type == true_fault:
                    hybrid_correct += 1
                if result.vote_agreement:
                    agreements += 1
            
            hybrid_accuracy = hybrid_correct / len(features)
            agreement_rate = agreements / len(features)
            
            return {
                'kan_accuracy': kan_eval['accuracy'],
                'pinn_accuracy': pinn_eval['accuracy'],
                'hybrid_accuracy': hybrid_accuracy,
                'agreement_rate': agreement_rate,
                'improvement': hybrid_accuracy - max(kan_eval['accuracy'], pinn_eval['accuracy'])
            }
        
        def save(self, path: str):
            """保存模型"""
            import torch
            
            torch.save({
                'kan_state': self.kan.model.state_dict(),
                'pinn_state': self.pinn.model.state_dict(),
                'config': {
                    'kan_weight': self.kan_weight,
                    'pinn_weight': self.pinn_weight,
                    'strategy': self.strategy.value,
                    'confidence_threshold': self.confidence_threshold
                }
            }, path)
            print(f"[混合诊断器] 已保存到 {path}")
        
        def load(self, path: str):
            """加载模型"""
            import torch
            
            checkpoint = torch.load(path, map_location=self.kan.device)
            self.kan.model.load_state_dict(checkpoint['kan_state'])
            self.pinn.model.load_state_dict(checkpoint['pinn_state'])
            
            config = checkpoint['config']
            self.kan_weight = config['kan_weight']
            self.pinn_weight = config['pinn_weight']
            self.strategy = VoteStrategy(config['strategy'])
            self.confidence_threshold = config['confidence_threshold']
            
            print(f"[混合诊断器] 已从 {path} 加载")


# 非PyTorch环境的占位符
if not TORCH_AVAILABLE:
    class HybridDiagnoser:
        def __init__(self, *args, **kwargs):
            raise ImportError("混合诊断器需要PyTorch支持")
