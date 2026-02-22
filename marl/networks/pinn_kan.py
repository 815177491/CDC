"""
Physics-Informed Neural Network (PINN) + KAN 诊断智能体
======================================================
结合物理约束和可解释网络的故障诊断
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from .kan import KANNetwork, KANLayer

# 导入共享配置模块，确保与发动机模型一致
from config import EngineConfig, ENGINE_CONFIG

# 向后兼容别名
DEFAULT_ENGINE_CONFIG = ENGINE_CONFIG


@dataclass
class PhysicsParams:
    """
    物理参数配置
    
    注意：此类现在基于共享的 EngineConfig 初始化，
    确保与 MarineEngine0D 仿真环境的一致性。
    """
    # 发动机几何参数
    bore: float = DEFAULT_ENGINE_CONFIG.bore                    # 气缸直径 [m]
    stroke: float = DEFAULT_ENGINE_CONFIG.stroke                # 活塞行程 [m]
    compression_ratio: float = DEFAULT_ENGINE_CONFIG.compression_ratio
    
    # 热力学参数
    gamma: float = DEFAULT_ENGINE_CONFIG.gamma                  # 比热比
    R: float = DEFAULT_ENGINE_CONFIG.R                          # 气体常数
    
    # 基准值（健康状态）
    Pmax_base: float = DEFAULT_ENGINE_CONFIG.Pmax_base          # 基准最大压力 [bar]
    Pcomp_base: float = DEFAULT_ENGINE_CONFIG.Pcomp_base        # 基准压缩压力 [bar]
    Texh_base: float = DEFAULT_ENGINE_CONFIG.Texh_base          # 基准排温 [K]
    
    # 物理约束权重
    lambda_physics: float = DEFAULT_ENGINE_CONFIG.lambda_physics
    lambda_consistency: float = DEFAULT_ENGINE_CONFIG.lambda_consistency
    
    @classmethod
    def from_engine_config(cls, config: EngineConfig) -> 'PhysicsParams':
        """从共享的 EngineConfig 创建 PhysicsParams"""
        return cls(
            bore=config.bore,
            stroke=config.stroke,
            compression_ratio=config.compression_ratio,
            gamma=config.gamma,
            R=config.R,
            Pmax_base=config.Pmax_base,
            Pcomp_base=config.Pcomp_base,
            Texh_base=config.Texh_base,
            lambda_physics=config.lambda_physics,
            lambda_consistency=config.lambda_consistency
        )


class PhysicsConstraints(nn.Module):
    """
    物理约束模块
    
    基于零维热力学模型的物理先验
    """
    
    def __init__(self, physics_params: PhysicsParams):
        super().__init__()
        self.params = physics_params
        
        # 可学习的物理参数修正
        self.eta_comp = nn.Parameter(torch.tensor(1.0))  # 压缩效率修正
        self.eta_comb = nn.Parameter(torch.tensor(1.0))  # 燃烧效率修正
    
    def forward(
        self,
        obs: torch.Tensor,
        fault_probs: torch.Tensor,
        severity: torch.Tensor
    ) -> torch.Tensor:
        """
        计算物理一致性损失
        
        Args:
            obs: 观测 [Pmax, Pcomp, Texh]（归一化后）
            fault_probs: 故障概率分布 [batch, 4]
            severity: 预测的故障严重程度 [batch, 1]
            
        Returns:
            physics_loss: 物理约束损失
        """
        batch_size = obs.shape[0]
        
        # 反归一化获取实际物理量
        Pmax = obs[:, 0] * 200.0   # 假设归一化系数 [bar]
        Pcomp = obs[:, 1] * 150.0  # [bar]
        Texh = obs[:, 2] * 800.0   # [K] (范围约400-800K)
        
        # 计算残差（与健康基准的偏差）
        delta_Pmax = (Pmax - self.params.Pmax_base) / self.params.Pmax_base
        delta_Pcomp = (Pcomp - self.params.Pcomp_base) / self.params.Pcomp_base
        delta_Texh = (Texh - self.params.Texh_base) / self.params.Texh_base
        
        # ===== 物理约束1：压缩压力与压缩比的关系 =====
        # Pcomp ∝ P0 * ε^γ
        # 泄漏故障应导致Pcomp下降
        leak_prob = fault_probs[:, 2]  # 泄漏故障概率
        expected_pcomp_drop = leak_prob * severity.squeeze() * 0.3  # 泄漏应导致压缩压力下降
        pcomp_constraint = F.mse_loss(
            -delta_Pcomp.clamp(min=0),  # 实际压缩压力下降
            expected_pcomp_drop,
            reduction='mean'
        )
        
        # ===== 物理约束2：Pmax与燃烧的关系 =====
        # 正时故障影响Pmax
        timing_prob = fault_probs[:, 1]
        fuel_prob = fault_probs[:, 3]
        
        # 正时提前->Pmax上升，滞后->Pmax下降（简化处理）
        # 燃油故障->Pmax下降
        expected_pmax_change = -fuel_prob * severity.squeeze() * 0.2
        pmax_constraint = F.mse_loss(
            delta_Pmax,
            expected_pmax_change,
            reduction='mean'
        )
        
        # ===== 物理约束3：排温与燃烧效率的关系 =====
        # 燃烧不完全->排温升高
        expected_texh_rise = (timing_prob + fuel_prob) * severity.squeeze() * 0.1
        texh_constraint = F.mse_loss(
            delta_Texh.clamp(min=0),
            expected_texh_rise,
            reduction='mean'
        )
        
        # ===== 物理约束4：故障严重程度与残差幅度的一致性 =====
        residual_magnitude = torch.sqrt(delta_Pmax**2 + delta_Pcomp**2 + delta_Texh**2)
        health_prob = fault_probs[:, 0]  # 健康概率
        
        # 健康状态应残差小，故障状态应残差大
        consistency_loss = F.mse_loss(
            (1 - health_prob) * severity.squeeze(),
            residual_magnitude * 0.5,
            reduction='mean'
        )
        
        # 总物理损失
        physics_loss = (
            pcomp_constraint + 
            pmax_constraint + 
            texh_constraint + 
            self.params.lambda_consistency * consistency_loss
        )
        
        return physics_loss


class PIKANDiagnosticNetwork(nn.Module):
    """
    Physics-Informed KAN诊断网络
    
    结合物理约束和可解释KAN网络
    """
    
    def __init__(
        self,
        obs_dim: int,
        n_fault_types: int = 4,
        kan_hidden_dims: List[int] = [32, 32],
        grid_size: int = 5,
        physics_params: Optional[PhysicsParams] = None
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.n_fault_types = n_fault_types
        self.physics_params = physics_params or PhysicsParams()
        
        # KAN特征编码器
        kan_dims = [obs_dim] + kan_hidden_dims
        self.kan_encoder = KANNetwork(kan_dims, grid_size=grid_size)
        
        # 物理约束模块
        self.physics = PhysicsConstraints(self.physics_params)
        
        # 输出头
        hidden_dim = kan_hidden_dims[-1]
        
        # 故障分类头
        self.fault_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_fault_types)
        )
        
        # 严重程度头
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 置信度头
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Critic头
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        obs: torch.Tensor,
        compute_physics_loss: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Returns:
            actions, log_probs, value, physics_loss
        """
        # KAN特征提取
        features = self.kan_encoder(obs)
        
        # 故障分类
        fault_logits = self.fault_head(features)
        fault_probs = F.softmax(fault_logits, dim=-1)
        fault_type = fault_probs.argmax(dim=-1)
        fault_log_prob = F.log_softmax(fault_logits, dim=-1).gather(
            1, fault_type.unsqueeze(-1)
        )
        
        # 严重程度和置信度
        severity = self.severity_head(features)
        confidence = self.confidence_head(features)
        
        # 状态价值
        value = self.critic(features)
        
        # 物理约束损失
        physics_loss = None
        if compute_physics_loss:
            # 提取观测的前3维（Pmax, Pcomp, Texh）
            obs_physics = obs[:, :3] if obs.shape[1] >= 3 else obs
            physics_loss = self.physics(obs_physics, fault_probs, severity)
            
            # 添加KAN正则化
            physics_loss = physics_loss + 0.001 * self.kan_encoder.get_spline_regularization()
        
        actions = {
            'fault_type': fault_type,
            'fault_probs': fault_probs,
            'severity': severity,
            'confidence': confidence
        }
        
        log_probs = fault_log_prob
        
        return actions, log_probs, value, physics_loss
    
    def get_interpretable_functions(self) -> Dict[str, np.ndarray]:
        """
        获取学习到的可解释函数（用于可视化）
        
        Returns:
            每个KAN层学习到的样条函数采样
        """
        x_sample = torch.linspace(0, 1, 100).unsqueeze(1)
        
        functions = {}
        for i, layer in enumerate(self.kan_encoder.layers):
            with torch.no_grad():
                basis = layer.basis(x_sample.expand(-1, layer.in_features))
                # 对每个输入-输出对采样学习到的函数
                for j in range(min(3, layer.in_features)):
                    for k in range(min(3, layer.out_features)):
                        spline_vals = torch.einsum(
                            'n,bn->b', 
                            layer.spline_weight[k, j], 
                            basis[:, j]
                        )
                        functions[f'layer{i}_in{j}_out{k}'] = spline_vals.numpy()
        
        return functions


class PIKANDiagnosticAgent:
    """
    PINN+KAN诊断智能体封装
    """
    
    def __init__(
        self,
        obs_dim: int,
        n_fault_types: int = 4,
        kan_hidden_dims: List[int] = [32, 32],
        grid_size: int = 5,
        lr: float = 3e-4,
        physics_weight: float = 0.1,
        device: str = 'cpu'
    ):
        self.device = torch.device(device)
        self.physics_weight = physics_weight
        
        self.network = PIKANDiagnosticNetwork(
            obs_dim=obs_dim,
            n_fault_types=n_fault_types,
            kan_hidden_dims=kan_hidden_dims,
            grid_size=grid_size
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
    
    def act(self, obs: np.ndarray, deterministic: bool = False) -> Dict:
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            actions, log_probs, value, _ = self.network(obs_tensor, compute_physics_loss=False)
        
        return {
            'fault_type': actions['fault_type'].item(),
            'severity': actions['severity'].cpu().numpy().flatten()[0],
            'confidence': actions['confidence'].cpu().numpy().flatten()[0],
            'log_prob': log_probs.cpu().numpy().flatten()[0],
            'value': value.item()
        }
    
    def update(
        self,
        obs_batch: torch.Tensor,
        action_batch: Dict[str, torch.Tensor],
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5
    ) -> Dict[str, float]:
        """PPO + Physics损失更新"""
        
        actions, new_log_probs, values, physics_loss = self.network(
            obs_batch, compute_physics_loss=True
        )
        
        # PPO策略损失
        ratio = torch.exp(new_log_probs.squeeze() - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值损失（使用传入的系数，与 SharedCritic 配合时设为 0）
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # 总损失 = PPO损失 + 物理损失（value_coef 由 Trainer 控制）
        total_loss = policy_loss + value_coef * value_loss + self.physics_weight * physics_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'physics_loss': physics_loss.item(),
            'entropy': 0.0  # 与 DiagnosticAgent 接口一致
        }
    
    def save(self, path: str):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
