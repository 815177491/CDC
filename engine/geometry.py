"""
发动机几何模块
==============
计算二冲程船用柴油机的运动学参数与瞬时气缸容积
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EngineGeometry:
    """
    发动机几何参数类
    
    Attributes:
        bore: 气缸直径 [m]
        stroke: 活塞行程 [m]
        n_cylinders: 气缸数量
        compression_ratio: 有效压缩比 (待校准参数)
        con_rod_ratio: 连杆比 = 连杆长度/曲柄半径 (待校准参数)
        evc_angle: 排气阀关闭角 [deg ATDC] (扫气口关闭后开始有效压缩)
        evo_angle: 排气阀开启角 [deg ATDC]
        spc_angle: 扫气口关闭角 [deg ATDC]
        spo_angle: 扫气口开启角 [deg ATDC]
    """
    # 基本几何参数 (根据用户提供)
    bore: float = 0.620  # 620 mm -> m
    stroke: float = 2.658  # 2658 mm -> m  
    n_cylinders: int = 6
    
    # 待校准参数 (设定初始猜测值)
    compression_ratio: float = 13.5  # 二冲程船机典型值 12-15
    con_rod_ratio: float = 4.0  # 连杆长度/曲柄半径, 典型值 3.5-4.5
    
    # 气门/扫气口正时 (二冲程特有)
    evc_angle: float = 230.0  # 排气阀关闭角 [deg ATDC], 有效压缩起点
    evo_angle: float = 130.0  # 排气阀开启角 [deg ATDC]
    spc_angle: float = 235.0  # 扫气口关闭角 [deg ATDC]
    spo_angle: float = 125.0  # 扫气口开启角 [deg ATDC]
    
    # 派生量 (自动计算)
    crank_radius: float = field(init=False)
    con_rod_length: float = field(init=False)
    displaced_volume: float = field(init=False)
    clearance_volume: float = field(init=False)
    piston_area: float = field(init=False)
    
    def __post_init__(self):
        """计算派生几何量"""
        self.crank_radius = self.stroke / 2.0
        self.con_rod_length = self.con_rod_ratio * self.crank_radius
        self.piston_area = np.pi * (self.bore / 2.0) ** 2
        self.displaced_volume = self.piston_area * self.stroke
        self.clearance_volume = self.displaced_volume / (self.compression_ratio - 1.0)
    
    def update_calibration_params(self, compression_ratio: Optional[float] = None,
                                   con_rod_ratio: Optional[float] = None):
        """更新校准参数并重新计算派生量"""
        if compression_ratio is not None:
            self.compression_ratio = compression_ratio
        if con_rod_ratio is not None:
            self.con_rod_ratio = con_rod_ratio
        self.__post_init__()
    
    def piston_position(self, theta: float) -> float:
        """
        计算活塞位置 (相对于TDC的位移)
        
        Args:
            theta: 曲轴转角 [rad], 0 = TDC
            
        Returns:
            x: 活塞距TDC的位移 [m]
        """
        R = self.crank_radius
        L = self.con_rod_length
        lambda_cr = R / L  # 曲柄连杆比
        
        # 精确运动学公式
        x = R * ((1 - np.cos(theta)) + 
                 (1 / lambda_cr) * (1 - np.sqrt(1 - (lambda_cr * np.sin(theta))**2)))
        return x
    
    def instantaneous_volume(self, theta: float) -> float:
        """
        计算瞬时气缸容积
        
        Args:
            theta: 曲轴转角 [rad], 0 = TDC
            
        Returns:
            V: 瞬时容积 [m^3]
        """
        x = self.piston_position(theta)
        V = self.clearance_volume + self.piston_area * x
        return V
    
    def volume_derivative(self, theta: float) -> float:
        """
        计算容积对曲轴转角的导数 dV/dθ
        
        Args:
            theta: 曲轴转角 [rad]
            
        Returns:
            dV_dtheta: 容积变化率 [m^3/rad]
        """
        R = self.crank_radius
        L = self.con_rod_length
        lambda_cr = R / L
        
        # dV/dθ = A_p * dx/dθ
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        sqrt_term = np.sqrt(1 - (lambda_cr * sin_theta)**2)
        dx_dtheta = R * (sin_theta + 
                         (lambda_cr * sin_theta * cos_theta) / sqrt_term)
        
        dV_dtheta = self.piston_area * dx_dtheta
        return dV_dtheta
    
    def mean_piston_speed(self, rpm: float) -> float:
        """
        计算平均活塞速度
        
        Args:
            rpm: 发动机转速 [rpm]
            
        Returns:
            Cm: 平均活塞速度 [m/s]
        """
        # 二冲程: Cm = 2 * S * n (每转做功一次)
        n = rpm / 60.0  # 转速 [rps]
        Cm = 2.0 * self.stroke * n
        return Cm
    
    def get_compression_start_angle(self) -> float:
        """
        获取有效压缩起始角 (二冲程为扫气口关闭后)
        
        Returns:
            theta_comp_start: 压缩起始角 [rad]
        """
        # 取排气阀关闭和扫气口关闭中较晚的角度
        effective_close = max(self.evc_angle, self.spc_angle)
        return np.deg2rad(effective_close)
    
    def get_expansion_end_angle(self) -> float:
        """
        获取膨胀结束角 (扫气口开启)
        
        Returns:
            theta_exp_end: 膨胀结束角 [rad]
        """
        effective_open = min(self.evo_angle, self.spo_angle)
        return np.deg2rad(effective_open)
    
    def __repr__(self) -> str:
        return (f"EngineGeometry(bore={self.bore*1000:.0f}mm, "
                f"stroke={self.stroke*1000:.0f}mm, "
                f"CR={self.compression_ratio:.2f}, "
                f"λ={self.con_rod_ratio:.2f})")
