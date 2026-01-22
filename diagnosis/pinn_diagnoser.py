#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Physics-Informed Neural Network (PINN) 故障诊断器
================================================
基于物理信息神经网络的柴油机故障诊断

PINN优势:
- 嵌入柴油机热力学物理约束
- 减少数据需求（物理先验补充）
- 可解释性强（物理残差可解释故障机理）
- 外推能力强（物理约束增强泛化）

物理约束:
1. 压缩多变方程: Pcomp/P1 = (V1/Vcomp)^n
2. 燃烧方程: dP/dθ ∝ 维贝燃烧模型
3. 能量守恒: Q_fuel = W_work + Q_cool + Q_exh

References:
- Raissi et al., "Physics-informed neural networks", JCP 2019
- 2024年PINN在工业诊断中的最新应用

Author: CDC Project
Date: 2026-01-22
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings

# 尝试导入深度学习库
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. PINN diagnoser will not work.")

from .fault_injector import FaultType


@dataclass
class PINNDiagnosisResult:
    """PINN诊断结果"""
    fault_detected: bool
    fault_type: FaultType
    confidence: float
    physics_residuals: Dict[str, float]  # 物理约束残差
    data_residuals: Dict[str, float]     # 数据拟合残差
    explanation: str                      # 可解释性说明


class DiagnosisMode(Enum):
    """诊断模式"""
    CLASSIFICATION = "classification"  # 故障分类
    DETECTION = "detection"            # 故障检测
    PROGNOSIS = "prognosis"           # 故障预测


if TORCH_AVAILABLE:
    
    class ResidualBlock(nn.Module):
        """残差块 - 深层PINN的稳定训练"""
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.activation = nn.Tanh()  # PINN常用Tanh保证平滑性
            
        def forward(self, x):
            residual = x
            x = self.activation(self.fc1(x))
            x = self.fc2(x)
            return self.activation(x + residual)
    
    
    class DieselPINNNetwork(nn.Module):
        """
        柴油机物理信息神经网络
        
        输入特征 (8维):
            - 转速 (rpm)
            - 负荷 (%)
            - 喷油正时 (°CA BTDC)
            - 增压压力 (bar)
            - 进气温度 (K)
            - 环境压力 (bar)
            - 燃油品质 (相对值)
            - 运行时间 (h)
        
        输出 (7维):
            - Pmax预测 (bar)
            - Pcomp预测 (bar)
            - Texh预测 (°C)
            - 故障类型概率 (4类)
        """
        
        def __init__(self, input_dim: int = 8, hidden_dim: int = 128, n_blocks: int = 4):
            super().__init__()
            
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            
            # 输入层
            self.input_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh()
            )
            
            # 残差块
            self.res_blocks = nn.ModuleList([
                ResidualBlock(hidden_dim) for _ in range(n_blocks)
            ])
            
            # 物理量预测头 (Pmax, Pcomp, Texh)
            self.physics_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 3)
            )
            
            # 故障分类头 (4类: 正常, 正时, 泄漏, 燃油)
            self.fault_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, 4)
            )
            
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            前向传播
            
            Returns:
                physics_pred: (batch, 3) - [Pmax, Pcomp, Texh]
                fault_logits: (batch, 4) - 故障类别logits
            """
            h = self.input_layer(x)
            
            for block in self.res_blocks:
                h = block(h)
            
            physics_pred = self.physics_head(h)
            fault_logits = self.fault_head(h)
            
            return physics_pred, fault_logits
        
        def predict_physics(self, x: torch.Tensor) -> torch.Tensor:
            """仅预测物理量"""
            h = self.input_layer(x)
            for block in self.res_blocks:
                h = block(h)
            return self.physics_head(h)


    class DieselPhysicsModule:
        """
        柴油机物理模型模块
        
        嵌入热力学物理方程作为软约束
        """
        
        def __init__(self, device: torch.device = None):
            self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 物理常数
            self.gamma = 1.35  # 多变指数 (压缩过程)
            self.R = 287.0     # 气体常数 J/(kg·K)
            self.cv = 718.0    # 定容比热 J/(kg·K)
            self.compression_ratio = 14.0  # 压缩比
            
            # 归一化参数 (根据柴油机典型值)
            self.pmax_mean = 150.0  # bar
            self.pmax_std = 20.0
            self.pcomp_mean = 120.0
            self.pcomp_std = 15.0
            self.texh_mean = 350.0  # °C
            self.texh_std = 50.0
            
        def compression_law(self, P1: torch.Tensor, n: torch.Tensor = None) -> torch.Tensor:
            """
            压缩多变方程: Pcomp = P1 * (compression_ratio)^n
            
            Args:
                P1: 进气压力 (bar)
                n: 多变指数 (默认1.35)
            """
            if n is None:
                n = torch.full_like(P1, self.gamma)
            return P1 * torch.pow(torch.tensor(self.compression_ratio, device=P1.device), n)
        
        def combustion_pressure(self, Pcomp: torch.Tensor, timing: torch.Tensor, 
                                 load: torch.Tensor) -> torch.Tensor:
            """
            燃烧峰值压力模型 (简化维贝模型)
            
            Pmax ≈ Pcomp * (1 + k * load * f(timing))
            
            Args:
                Pcomp: 压缩压力 (bar)
                timing: 喷油正时 (°CA BTDC, 正值为提前)
                load: 负荷 (0-1)
            """
            # 正时影响系数 (最优正时约12-15° BTDC)
            optimal_timing = 12.0
            timing_factor = 1.0 - 0.01 * torch.abs(timing - optimal_timing)
            
            # 负荷影响
            load_factor = 0.3 + 0.5 * load
            
            # 峰压升高
            pressure_rise = 1.0 + load_factor * timing_factor
            
            return Pcomp * pressure_rise
        
        def exhaust_temperature(self, Pmax: torch.Tensor, load: torch.Tensor,
                                 efficiency: torch.Tensor = None) -> torch.Tensor:
            """
            排气温度模型 (能量平衡)
            
            Texh ∝ (1 - η) * Q_fuel / m_air
            
            简化模型: Texh = base + k1 * load - k2 * efficiency
            """
            if efficiency is None:
                efficiency = torch.full_like(load, 0.45)
            
            base_temp = 280.0  # °C
            load_effect = 150.0 * load
            efficiency_effect = 100.0 * (0.5 - efficiency)
            
            return base_temp + load_effect + efficiency_effect
        
        def physics_loss(self, features: torch.Tensor, predictions: torch.Tensor,
                         targets: torch.Tensor = None) -> Dict[str, torch.Tensor]:
            """
            计算物理约束损失
            
            Args:
                features: 输入特征 [rpm, load, timing, boost_pressure, ...]
                predictions: 网络预测 [Pmax, Pcomp, Texh]
                targets: 真实值 (可选)
            
            Returns:
                各项物理损失
            """
            # 解析特征
            load = features[:, 1]           # 负荷
            timing = features[:, 2]         # 正时
            boost_pressure = features[:, 3] # 增压压力
            
            # 解析预测
            pred_pmax = predictions[:, 0]
            pred_pcomp = predictions[:, 1]
            pred_texh = predictions[:, 2]
            
            # 1. 压缩方程约束: Pcomp ≈ f(boost_pressure)
            pcomp_theory = self.compression_law(boost_pressure)
            compression_loss = F.mse_loss(pred_pcomp, pcomp_theory)
            
            # 2. 燃烧方程约束: Pmax ≈ g(Pcomp, timing, load)
            pmax_theory = self.combustion_pressure(pred_pcomp, timing, load)
            combustion_loss = F.mse_loss(pred_pmax, pmax_theory)
            
            # 3. 能量守恒约束: Texh ≈ h(Pmax, load)
            texh_theory = self.exhaust_temperature(pred_pmax, load)
            energy_loss = F.mse_loss(pred_texh, texh_theory)
            
            # 4. 物理可行性约束 (软边界)
            feasibility_loss = torch.mean(
                F.relu(pred_pmax - 200.0) +   # Pmax < 200 bar
                F.relu(50.0 - pred_pmax) +    # Pmax > 50 bar
                F.relu(pred_texh - 500.0) +   # Texh < 500 °C
                F.relu(200.0 - pred_texh)     # Texh > 200 °C
            )
            
            return {
                'compression': compression_loss,
                'combustion': combustion_loss,
                'energy': energy_loss,
                'feasibility': feasibility_loss,
                'total': compression_loss + combustion_loss + energy_loss + 0.1 * feasibility_loss
            }


    class PINNDiagnoser:
        """
        基于PINN的柴油机故障诊断器
        
        特点:
        1. 物理约束嵌入 - 利用热力学方程作为正则化
        2. 多任务学习 - 同时预测物理量和故障类型
        3. 可解释性 - 通过物理残差解释故障原因
        """
        
        def __init__(self, config: Dict = None):
            config = config or {}
            
            self.device = torch.device(
                config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            )
            
            # 模型参数
            self.input_dim = config.get('input_dim', 8)
            self.hidden_dim = config.get('hidden_dim', 128)
            self.n_blocks = config.get('n_blocks', 4)
            
            # 训练参数
            self.lr = config.get('lr', 1e-3)
            self.batch_size = config.get('batch_size', 64)
            self.physics_weight = config.get('physics_weight', 0.1)  # 物理损失权重
            
            # 创建网络和物理模块
            self.model = DieselPINNNetwork(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                n_blocks=self.n_blocks
            ).to(self.device)
            
            self.physics_module = DieselPhysicsModule(self.device)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
            
            # 故障类型映射
            self.fault_types = [
                FaultType.NONE,
                FaultType.INJECTION_TIMING,
                FaultType.CYLINDER_LEAK,
                FaultType.FUEL_DEGRADATION
            ]
            
            # 训练统计
            self.train_losses = []
            self.val_losses = []
            
            print(f"[PINN诊断器] 初始化完成 | Device: {self.device}")
            print(f"  - 输入维度: {self.input_dim}")
            print(f"  - 隐藏维度: {self.hidden_dim}")
            print(f"  - 残差块数: {self.n_blocks}")
            print(f"  - 物理损失权重: {self.physics_weight}")
        
        def train_step(self, features: np.ndarray, physics_targets: np.ndarray,
                       fault_labels: np.ndarray) -> Dict[str, float]:
            """
            单步训练
            
            Args:
                features: 输入特征 (batch, 8)
                physics_targets: 物理量真值 [Pmax, Pcomp, Texh] (batch, 3)
                fault_labels: 故障标签 (batch,)
            
            Returns:
                损失字典
            """
            self.model.train()
            
            # 转换为tensor
            features_t = torch.FloatTensor(features).to(self.device)
            physics_t = torch.FloatTensor(physics_targets).to(self.device)
            labels_t = torch.LongTensor(fault_labels).to(self.device)
            
            # 前向传播
            physics_pred, fault_logits = self.model(features_t)
            
            # 数据损失
            physics_loss = F.mse_loss(physics_pred, physics_t)
            fault_loss = F.cross_entropy(fault_logits, labels_t)
            
            # 物理约束损失
            physics_constraints = self.physics_module.physics_loss(
                features_t, physics_pred, physics_t
            )
            
            # 总损失 = 数据损失 + 物理约束
            total_loss = (
                physics_loss + 
                fault_loss + 
                self.physics_weight * physics_constraints['total']
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            return {
                'total_loss': total_loss.item(),
                'physics_loss': physics_loss.item(),
                'fault_loss': fault_loss.item(),
                'constraint_loss': physics_constraints['total'].item(),
                'compression_loss': physics_constraints['compression'].item(),
                'combustion_loss': physics_constraints['combustion'].item(),
                'energy_loss': physics_constraints['energy'].item()
            }
        
        def train(self, train_data: Dict[str, np.ndarray], 
                  val_data: Dict[str, np.ndarray] = None,
                  epochs: int = 100, verbose: bool = True) -> Dict[str, List[float]]:
            """
            完整训练循环
            
            Args:
                train_data: {'features': (N, 8), 'physics': (N, 3), 'labels': (N,)}
                val_data: 验证数据 (可选)
                epochs: 训练轮数
            """
            features = train_data['features']
            physics = train_data['physics']
            labels = train_data['labels']
            n_samples = len(features)
            
            history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
            
            for epoch in range(epochs):
                # 打乱数据
                indices = np.random.permutation(n_samples)
                epoch_losses = []
                
                # 批次训练
                for i in range(0, n_samples, self.batch_size):
                    batch_idx = indices[i:i+self.batch_size]
                    batch_features = features[batch_idx]
                    batch_physics = physics[batch_idx]
                    batch_labels = labels[batch_idx]
                    
                    losses = self.train_step(batch_features, batch_physics, batch_labels)
                    epoch_losses.append(losses['total_loss'])
                
                avg_train_loss = np.mean(epoch_losses)
                history['train_loss'].append(avg_train_loss)
                
                # 验证
                if val_data is not None:
                    val_metrics = self.evaluate(val_data)
                    history['val_loss'].append(val_metrics['loss'])
                    history['val_acc'].append(val_metrics['accuracy'])
                
                # 训练准确率
                train_metrics = self.evaluate(train_data)
                history['train_acc'].append(train_metrics['accuracy'])
                
                if verbose and (epoch + 1) % 10 == 0:
                    msg = f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | Acc: {train_metrics['accuracy']:.2%}"
                    if val_data is not None:
                        msg += f" | Val Acc: {val_metrics['accuracy']:.2%}"
                    print(msg)
            
            self.train_losses = history['train_loss']
            return history
        
        def evaluate(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
            """评估模型"""
            self.model.eval()
            
            features = torch.FloatTensor(data['features']).to(self.device)
            physics = torch.FloatTensor(data['physics']).to(self.device)
            labels = torch.LongTensor(data['labels']).to(self.device)
            
            with torch.no_grad():
                physics_pred, fault_logits = self.model(features)
                
                # 损失
                physics_loss = F.mse_loss(physics_pred, physics)
                fault_loss = F.cross_entropy(fault_logits, labels)
                total_loss = physics_loss + fault_loss
                
                # 准确率
                predictions = torch.argmax(fault_logits, dim=1)
                accuracy = (predictions == labels).float().mean()
                
                # 物理预测误差
                pmax_error = torch.abs(physics_pred[:, 0] - physics[:, 0]).mean()
                pcomp_error = torch.abs(physics_pred[:, 1] - physics[:, 1]).mean()
                texh_error = torch.abs(physics_pred[:, 2] - physics[:, 2]).mean()
            
            return {
                'loss': total_loss.item(),
                'accuracy': accuracy.item(),
                'pmax_mae': pmax_error.item(),
                'pcomp_mae': pcomp_error.item(),
                'texh_mae': texh_error.item()
            }
        
        def diagnose(self, features: np.ndarray) -> PINNDiagnosisResult:
            """
            执行故障诊断
            
            Args:
                features: 输入特征 (8,) 或 (1, 8)
            
            Returns:
                诊断结果
            """
            self.model.eval()
            
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            features_t = torch.FloatTensor(features).to(self.device)
            
            with torch.no_grad():
                physics_pred, fault_logits = self.model(features_t)
                fault_probs = F.softmax(fault_logits, dim=1)
                
                # 获取预测结果
                fault_idx = torch.argmax(fault_probs, dim=1).item()
                confidence = fault_probs[0, fault_idx].item()
                
                # 计算物理约束残差
                physics_constraints = self.physics_module.physics_loss(
                    features_t, physics_pred
                )
            
            # 判断是否检测到故障
            fault_type = self.fault_types[fault_idx]
            fault_detected = fault_type != FaultType.NONE
            
            # 生成解释
            explanation = self._generate_explanation(
                fault_type, confidence, physics_constraints, physics_pred[0]
            )
            
            return PINNDiagnosisResult(
                fault_detected=fault_detected,
                fault_type=fault_type,
                confidence=confidence,
                physics_residuals={
                    'compression': physics_constraints['compression'].item(),
                    'combustion': physics_constraints['combustion'].item(),
                    'energy': physics_constraints['energy'].item()
                },
                data_residuals={
                    'pmax': physics_pred[0, 0].item(),
                    'pcomp': physics_pred[0, 1].item(),
                    'texh': physics_pred[0, 2].item()
                },
                explanation=explanation
            )
        
        def _generate_explanation(self, fault_type: FaultType, confidence: float,
                                   physics_constraints: Dict, predictions: torch.Tensor) -> str:
            """生成可解释性说明"""
            
            if fault_type == FaultType.NONE:
                return f"系统运行正常 (置信度: {confidence:.1%})"
            
            explanations = {
                FaultType.INJECTION_TIMING: 
                    f"检测到喷油正时异常 (置信度: {confidence:.1%})\n"
                    f"  - 燃烧压力偏差: {physics_constraints['combustion'].item():.3f}\n"
                    f"  - 可能原因: 喷油泵正时漂移或传感器故障",
                
                FaultType.CYLINDER_LEAK:
                    f"检测到气缸泄漏 (置信度: {confidence:.1%})\n"
                    f"  - 压缩压力偏差: {physics_constraints['compression'].item():.3f}\n"
                    f"  - 可能原因: 气缸环磨损、气门密封不良",
                
                FaultType.FUEL_DEGRADATION:
                    f"检测到燃油品质问题 (置信度: {confidence:.1%})\n"
                    f"  - 能量平衡偏差: {physics_constraints['energy'].item():.3f}\n"
                    f"  - 可能原因: 燃油热值异常或含水"
            }
            
            return explanations.get(fault_type, f"未知故障类型: {fault_type}")
        
        def save(self, path: str):
            """保存模型"""
            torch.save({
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'config': {
                    'input_dim': self.input_dim,
                    'hidden_dim': self.hidden_dim,
                    'n_blocks': self.n_blocks,
                    'physics_weight': self.physics_weight
                }
            }, path)
            print(f"[PINN] 模型已保存到 {path}")
        
        def load(self, path: str):
            """加载模型"""
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            print(f"[PINN] 模型已从 {path} 加载")


# 非PyTorch环境的占位符
if not TORCH_AVAILABLE:
    class PINNDiagnoser:
        def __init__(self, *args, **kwargs):
            raise ImportError("PINN诊断器需要PyTorch支持")
