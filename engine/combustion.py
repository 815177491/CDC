"""
燃烧模型模块
============
双Wiebe函数描述预混燃烧与扩散燃烧
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DoublieWiebeCombustion:
    """
    双Wiebe燃烧模型
    
    采用预混燃烧 + 扩散燃烧的叠加描述船机的燃烧放热规律
    
    Wiebe函数: x_b = 1 - exp(-a * ((θ - θ_start) / Δθ)^(m+1))
    
    Attributes:
        # 预混燃烧参数
        premix_fraction: 预混燃烧质量分数 (0-1)
        premix_duration: 预混燃烧持续角 [deg]
        premix_shape: 预混燃烧形状因子 m
        
        # 扩散燃烧参数
        diffusion_duration: 扩散燃烧持续角 [deg]
        diffusion_shape: 扩散燃烧形状因子 m
        
        # 共用参数
        ignition_delay: 着火延迟期 [deg] (从喷油到着火)
        injection_timing: 喷油正时 [deg BTDC]
        
        # 燃油参数
        fuel_lhv: 燃油低热值 [J/kg]
        combustion_efficiency: 燃烧效率
    """
    # 预混燃烧参数 (待校准)
    premix_fraction: float = 0.15  # 预混燃烧占比, 船机较低
    premix_duration: float = 8.0   # 预混燃烧持续角 [deg]
    premix_shape: float = 3.0      # 形状因子
    
    # 扩散燃烧参数 (待校准)
    diffusion_duration: float = 55.0  # 扩散燃烧持续角 [deg], 船机较长
    diffusion_shape: float = 1.5      # 形状因子
    
    # 着火参数
    ignition_delay: float = 2.0    # 着火延迟 [deg]
    injection_timing: float = 2.0  # 喷油正时 [deg BTDC], 即 -2 deg ATDC
    
    # 燃油参数
    fuel_lhv: float = 42.7e6       # 重油低热值 [J/kg]
    combustion_efficiency: float = 0.98
    
    # Wiebe常数
    wiebe_a: float = 6.908  # 对应99.9%燃烧完成
    
    def get_ignition_angle(self) -> float:
        """
        获取着火角 (ATDC)
        
        Returns:
            theta_ign: 着火角 [deg ATDC]
        """
        # 喷油正时为BTDC, 着火角 = -injection_timing + ignition_delay
        theta_ign = -self.injection_timing + self.ignition_delay
        return theta_ign
    
    def wiebe_function(self, theta: float, theta_start: float, 
                       duration: float, shape: float) -> float:
        """
        单Wiebe函数计算累积燃烧分数
        
        Args:
            theta: 当前曲轴转角 [deg ATDC]
            theta_start: 燃烧起始角 [deg ATDC]
            duration: 燃烧持续角 [deg]
            shape: 形状因子 m
            
        Returns:
            x_b: 累积燃烧分数 (0-1)
        """
        if theta < theta_start:
            return 0.0
        
        tau = (theta - theta_start) / duration
        if tau > 1.0:
            tau = 1.0
        
        x_b = 1.0 - np.exp(-self.wiebe_a * (tau ** (shape + 1)))
        return x_b
    
    def wiebe_derivative(self, theta: float, theta_start: float,
                         duration: float, shape: float) -> float:
        """
        Wiebe函数导数 (瞬时燃烧率)
        
        Args:
            theta: 当前曲轴转角 [deg ATDC]
            theta_start: 燃烧起始角 [deg ATDC]
            duration: 燃烧持续角 [deg]
            shape: 形状因子 m
            
        Returns:
            dx_b_dtheta: 燃烧率 [1/deg]
        """
        if theta < theta_start:
            return 0.0
        
        tau = (theta - theta_start) / duration
        if tau >= 1.0:
            return 0.0
        
        m = shape
        a = self.wiebe_a
        
        # dx_b/dθ = (a * (m+1) / Δθ) * τ^m * exp(-a * τ^(m+1))
        dx_b_dtheta = (a * (m + 1) / duration) * (tau ** m) * np.exp(-a * (tau ** (m + 1)))
        return dx_b_dtheta
    
    def get_burn_fraction(self, theta: float) -> float:
        """
        计算总累积燃烧分数 (双Wiebe叠加)
        
        Args:
            theta: 曲轴转角 [deg ATDC]
            
        Returns:
            x_b_total: 总累积燃烧分数
        """
        theta_ign = self.get_ignition_angle()
        
        # 预混燃烧
        x_premix = self.wiebe_function(theta, theta_ign, 
                                        self.premix_duration, 
                                        self.premix_shape)
        
        # 扩散燃烧 (与预混同时开始)
        x_diffusion = self.wiebe_function(theta, theta_ign,
                                           self.diffusion_duration,
                                           self.diffusion_shape)
        
        # 加权叠加
        x_b_total = (self.premix_fraction * x_premix + 
                     (1 - self.premix_fraction) * x_diffusion)
        return x_b_total
    
    def get_burn_rate(self, theta: float) -> float:
        """
        计算瞬时燃烧率 (双Wiebe叠加)
        
        Args:
            theta: 曲轴转角 [deg ATDC]
            
        Returns:
            dx_b_dtheta: 总燃烧率 [1/deg]
        """
        theta_ign = self.get_ignition_angle()
        
        # 预混燃烧率
        dxp = self.wiebe_derivative(theta, theta_ign,
                                     self.premix_duration,
                                     self.premix_shape)
        
        # 扩散燃烧率
        dxd = self.wiebe_derivative(theta, theta_ign,
                                     self.diffusion_duration,
                                     self.diffusion_shape)
        
        dx_total = self.premix_fraction * dxp + (1 - self.premix_fraction) * dxd
        return dx_total
    
    def get_heat_release_rate(self, theta: float, fuel_mass: float) -> float:
        """
        计算瞬时放热率
        
        Args:
            theta: 曲轴转角 [deg ATDC]
            fuel_mass: 循环喷油量 [kg]
            
        Returns:
            dQ_dtheta: 放热率 [J/deg]
        """
        dx_b = self.get_burn_rate(theta)
        Q_total = fuel_mass * self.fuel_lhv * self.combustion_efficiency
        dQ_dtheta = Q_total * dx_b
        return dQ_dtheta
    
    def get_cumulative_heat_release(self, theta: float, fuel_mass: float) -> float:
        """
        计算累积放热量
        
        Args:
            theta: 曲轴转角 [deg ATDC]
            fuel_mass: 循环喷油量 [kg]
            
        Returns:
            Q: 累积放热量 [J]
        """
        x_b = self.get_burn_fraction(theta)
        Q_total = fuel_mass * self.fuel_lhv * self.combustion_efficiency
        return Q_total * x_b
    
    def update_calibration_params(self, **kwargs):
        """更新校准参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def apply_injection_timing_fault(self, timing_offset: float):
        """
        注入喷油正时故障
        
        Args:
            timing_offset: 正时偏差 [deg], 正值表示提前
        """
        self.injection_timing += timing_offset
    
    def reset_injection_timing(self, original_timing: float):
        """重置喷油正时"""
        self.injection_timing = original_timing
