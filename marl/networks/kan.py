"""
Kolmogorov-Arnold Network (KAN) 层实现
=======================================
用于诊断智能体的可解释特征学习
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BSplineBasis(nn.Module):
    """B样条基函数"""
    
    def __init__(self, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.n_basis = grid_size + spline_order
        
        # 构建均匀节点向量
        h = 1.0 / grid_size
        grid = torch.linspace(-spline_order * h, 1 + spline_order * h, 
                             self.n_basis + spline_order + 1)
        self.register_buffer('grid', grid)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算B样条基函数值
        
        Args:
            x: 输入 [batch, dim], 值域 [0, 1]
            
        Returns:
            basis: [batch, dim, n_basis]
        """
        x = x.unsqueeze(-1)  # [batch, dim, 1]
        grid = self.grid[None, None, :]  # [1, 1, n_knots]
        
        # 递归计算B样条
        bases = self._compute_basis(x, grid, self.spline_order)
        return bases
    
    def _compute_basis(self, x, grid, k):
        """递归计算k阶B样条基函数"""
        if k == 0:
            # 0阶：阶梯函数
            return ((x >= grid[..., :-1]) & (x < grid[..., 1:])).float()
        else:
            # 递归公式
            b_prev = self._compute_basis(x, grid, k - 1)
            
            # 防止除零
            left_denom = grid[..., k:-1] - grid[..., :-k-1] + 1e-8
            right_denom = grid[..., k+1:] - grid[..., 1:-k] + 1e-8
            
            left = (x - grid[..., :-k-1]) / left_denom * b_prev[..., :-1]
            right = (grid[..., k+1:] - x) / right_denom * b_prev[..., 1:]
            
            return left + right


class KANLayer(nn.Module):
    """
    KAN层：用可学习的样条函数替代固定激活函数
    
    每个输入-输出连接都有独立的样条函数
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_base: float = 1.0,
        scale_spline: float = 1.0
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.n_basis = grid_size + spline_order
        
        # B样条基函数
        self.basis = BSplineBasis(grid_size, spline_order)
        
        # 基础线性变换（SiLU激活）
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * scale_base / np.sqrt(in_features)
        )
        
        # 样条系数：每个(in, out)对有n_basis个系数
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, self.n_basis) * scale_spline / np.sqrt(in_features)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch, in_features]
            
        Returns:
            y: [batch, out_features]
        """
        # 归一化到[0, 1]用于样条计算
        x_norm = torch.sigmoid(x)
        
        # 计算样条基函数值
        basis = self.basis(x_norm)  # [batch, in_features, n_basis]
        
        # 样条输出：加权求和
        # spline_weight: [out, in, n_basis]
        # basis: [batch, in, n_basis]
        spline_output = torch.einsum('oin,bin->bo', self.spline_weight, basis)
        
        # 基础输出（带SiLU激活）
        base_output = F.silu(x) @ self.base_weight.T
        
        return base_output + spline_output


class KANNetwork(nn.Module):
    """
    多层KAN网络
    """
    
    def __init__(
        self,
        layer_dims: list,
        grid_size: int = 5,
        spline_order: int = 3
    ):
        super().__init__()
        
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(KANLayer(
                layer_dims[i], 
                layer_dims[i+1],
                grid_size=grid_size,
                spline_order=spline_order
            ))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_spline_regularization(self) -> torch.Tensor:
        """计算样条平滑正则化"""
        reg = 0.0
        for layer in self.layers:
            # L2正则化样条系数
            reg = reg + torch.sum(layer.spline_weight ** 2)
        return reg
