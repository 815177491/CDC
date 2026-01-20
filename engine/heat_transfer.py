"""
传热模型模块
============
Woschni传热关联式计算缸内对流换热
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class WoschniHeatTransfer:
    """
    Woschni传热模型
    
    经典Woschni公式计算对流换热系数:
    h = C * D^(-0.2) * p^0.8 * T^(-0.53) * w^0.8
    
    其中特征速度 w = C1*Cm + C2*(Vs*Tr/pr*Vr)*(p-pm)
    
    Attributes:
        C_woschni: Woschni常数 (待校准)
        C1_scav: 换气过程速度系数
        C1_comp: 压缩过程速度系数  
        C1_comb: 燃烧/膨胀过程速度系数
        C2: 燃烧附加速度系数
        T_wall: 气缸壁温 [K]
        T_piston: 活塞顶温度 [K]
        T_head: 气缸盖温度 [K]
    """
    # Woschni系数 (待校准) - 使用kPa单位时的标准值
    C_woschni: float = 3.26   # 主系数 (标准Woschni公式)
    C1_scav: float = 6.18     # 换气过程
    C1_comp: float = 2.28     # 压缩过程
    C1_comb: float = 2.28     # 燃烧/膨胀过程
    C2: float = 3.24e-3       # 燃烧附加项
    
    # 壁面温度边界条件 [K]
    T_wall: float = 450.0     # 缸套壁温
    T_piston: float = 550.0   # 活塞顶温度
    T_head: float = 500.0     # 缸盖温度
    
    # 参考状态 (IVC时刻)
    p_ref: float = 3.5e5      # 参考压力 [Pa]
    T_ref: float = 340.0      # 参考温度 [K]
    V_ref: float = 0.8        # 参考容积 [m^3]
    
    def calculate_heat_transfer_coefficient(self, 
                                             bore: float,
                                             pressure: float,
                                             temperature: float,
                                             mean_piston_speed: float,
                                             V_s: float,
                                             p_motored: float,
                                             is_combustion: bool = False) -> float:
        """
        计算Woschni对流换热系数
        
        Args:
            bore: 气缸直径 [m]
            pressure: 瞬时缸压 [Pa]
            temperature: 瞬时缸温 [K]
            mean_piston_speed: 平均活塞速度 [m/s]
            V_s: 排量 [m^3]
            p_motored: 倒拖压力 (无燃烧时的压力) [Pa]
            is_combustion: 是否处于燃烧/膨胀过程
            
        Returns:
            h: 对流换热系数 [W/(m^2·K)]
        """
        # 压力转换为 kPa (Woschni公式中压力单位为 kPa)
        p_kPa = pressure / 1000.0
        p_motored_kPa = p_motored / 1000.0
        
        # 选择速度系数
        if is_combustion:
            C1 = self.C1_comb
            # 燃烧附加速度项 (使用kPa)
            w_comb = self.C2 * (V_s * self.T_ref / (self.p_ref/1000 * self.V_ref)) * (p_kPa - p_motored_kPa)
            w_comb = max(0, w_comb)  # 确保非负
        else:
            C1 = self.C1_comp
            w_comb = 0.0
        
        # 特征速度
        w = C1 * mean_piston_speed + w_comb
        w = max(w, 0.1)  # 避免零速度
        
        # Woschni公式 (使用kPa)
        # h = C * D^(-0.2) * p^0.8 * T^(-0.53) * w^0.8
        # 对于 D[m], p[kPa], T[K], w[m/s], C=3.26 时 h 单位为 W/(m²K)
        h = self.C_woschni * (bore ** (-0.2)) * (p_kPa ** 0.8) * \
            (temperature ** (-0.53)) * (w ** 0.8)
        
        return h
    
    def calculate_heat_transfer_rate(self,
                                      h: float,
                                      surface_area: float,
                                      gas_temperature: float) -> float:
        """
        计算瞬时传热率
        
        Args:
            h: 对流换热系数 [W/(m^2·K)]
            surface_area: 换热面积 [m^2]
            gas_temperature: 缸内气体温度 [K]
            
        Returns:
            Q_dot: 传热率 [W] (正值表示热量从气体流向壁面)
        """
        # 平均壁温
        T_wall_avg = (self.T_wall + self.T_piston + self.T_head) / 3.0
        
        Q_dot = h * surface_area * (gas_temperature - T_wall_avg)
        return Q_dot
    
    def calculate_surface_area(self, bore: float, 
                                piston_position: float,
                                clearance_height: float) -> float:
        """
        计算瞬时换热面积
        
        Args:
            bore: 气缸直径 [m]
            piston_position: 活塞距TDC位移 [m]
            clearance_height: 余隙高度 [m]
            
        Returns:
            A: 总换热面积 [m^2]
        """
        r = bore / 2.0
        
        # 活塞顶面积
        A_piston = np.pi * r**2
        
        # 缸盖面积 (近似等于活塞顶)
        A_head = np.pi * r**2
        
        # 缸套面积 (侧面积)
        h_exposed = clearance_height + piston_position
        A_liner = 2 * np.pi * r * h_exposed
        
        return A_piston + A_head + A_liner
    
    def update_reference_state(self, p_ref: float, T_ref: float, V_ref: float):
        """更新参考状态"""
        self.p_ref = p_ref
        self.T_ref = T_ref
        self.V_ref = V_ref
    
    def update_wall_temperatures(self, T_wall: float = None,
                                  T_piston: float = None,
                                  T_head: float = None):
        """更新壁面温度"""
        if T_wall is not None:
            self.T_wall = T_wall
        if T_piston is not None:
            self.T_piston = T_piston
        if T_head is not None:
            self.T_head = T_head
    
    def update_calibration_params(self, **kwargs):
        """更新校准参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
