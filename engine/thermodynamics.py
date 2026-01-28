"""
热力学求解器模块
================
基于控制体分析法求解能量守恒与质量守恒方程
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List

from .geometry import EngineGeometry
from .combustion import DoublieWiebeCombustion
from .heat_transfer import WoschniHeatTransfer


@dataclass
class GasProperties:
    """工质物性参数"""
    R: float = 287.0         # 气体常数 [J/(kg·K)]
    gamma_air: float = 1.4   # 空气比热比
    gamma_burned: float = 1.3  # 燃气比热比
    cv_air: float = 717.5    # 空气定容比热 [J/(kg·K)]
    cp_air: float = 1004.5   # 空气定压比热 [J/(kg·K)]
    
    def get_gamma(self, burn_fraction: float) -> float:
        """根据燃烧分数插值比热比"""
        return self.gamma_air * (1 - burn_fraction) + self.gamma_burned * burn_fraction
    
    def get_cv(self, temperature: float, burn_fraction: float) -> float:
        """获取定容比热 (考虑温度修正)"""
        # 简化的温度修正
        cv_base = self.cv_air * (1 - burn_fraction) + 850.0 * burn_fraction
        # 高温修正 (>1000K时比热增加)
        if temperature > 1000:
            cv_base *= (1 + 0.0001 * (temperature - 1000))
        return cv_base


@dataclass
class ThermodynamicState:
    """热力学状态"""
    theta: float  # 曲轴转角 [rad]
    pressure: float  # 压力 [Pa]
    temperature: float  # 温度 [K]
    mass: float  # 缸内质量 [kg]
    volume: float  # 容积 [m^3]
    burn_fraction: float = 0.0  # 燃烧分数


class ThermodynamicSolver:
    """
    热力学求解器
    
    求解缸内热力学过程的ODE系统:
    - 能量守恒: dT/dθ
    - 质量守恒: dm/dθ (考虑泄漏)
    - 状态方程: p = mRT/V
    """
    
    def __init__(self, geometry: EngineGeometry,
                 combustion: DoublieWiebeCombustion,
                 heat_transfer: WoschniHeatTransfer):
        self.geo = geometry
        self.comb = combustion
        self.ht = heat_transfer
        self.gas = GasProperties()
        
        # 运行参数
        self.rpm: float = 80.0  # 发动机转速 [rpm]
        self.p_scav: float = 3.5e5  # 扫气压力 [Pa]
        self.T_scav: float = 320.0  # 扫气温度 [K]
        self.fuel_mass: float = 0.0  # 循环喷油量 [kg]
        
        # 故障因子
        self.leak_factor: float = 0.0  # 泄漏因子 (0-1)
        self.fuel_degradation: float = 1.0  # 燃油衰减系数
        
        # 求解结果存储
        self.results: Dict = {}
    
    def set_operating_conditions(self, rpm: float, p_scav: float, 
                                  T_scav: float, fuel_mass: float):
        """设置运行工况"""
        self.rpm = rpm
        self.p_scav = p_scav
        self.T_scav = T_scav
        self.fuel_mass = fuel_mass * self.fuel_degradation
    
    def get_initial_conditions(self) -> Tuple[float, float, float]:
        """
        获取有效压缩起点的初始条件
        
        Returns:
            (p0, T0, m0): 初始压力[Pa], 温度[K], 质量[kg]
        """
        # 压缩起始角处的容积
        theta_start = self.geo.get_compression_start_angle()
        V_start = self.geo.instantaneous_volume(theta_start)
        
        # 假设扫气过程完成后, 缸内充满新鲜空气
        p0 = self.p_scav
        T0 = self.T_scav
        m0 = p0 * V_start / (self.gas.R * T0)
        
        return p0, T0, m0
    
    def calculate_polytropic_pressure(self, theta: float, n: float = 1.35) -> float:
        """
        计算多变压缩/膨胀的倒拖压力
        
        Args:
            theta: 曲轴转角 [rad]
            n: 多变指数
            
        Returns:
            p_motored: 倒拖压力 [Pa]
        """
        theta_start = self.geo.get_compression_start_angle()
        V_start = self.geo.instantaneous_volume(theta_start)
        V_current = self.geo.instantaneous_volume(theta)
        
        p_motored = self.p_scav * (V_start / V_current) ** n
        return p_motored
    
    def ode_system(self, y: np.ndarray, theta: float) -> np.ndarray:
        """
        热力学ODE系统
        
        状态向量 y = [T, m] (温度, 质量)
        求解 dy/dθ
        
        Args:
            y: 状态向量 [T, m]
            theta: 曲轴转角 [rad]
            
        Returns:
            dydt: 状态导数 [dT/dθ, dm/dθ]
        """
        T, m = y
        # 将theta转换为ATDC度数 (360° = TDC)
        theta_deg = np.rad2deg(theta) - 360  # ATDC
        
        # 几何参数
        V = self.geo.instantaneous_volume(theta)
        dV_dtheta = self.geo.volume_derivative(theta)
        Cm = self.geo.mean_piston_speed(self.rpm)
        
        # 燃烧放热 (dQ_comb 单位是 J/deg)
        x_b = self.comb.get_burn_fraction(theta_deg)
        dQ_comb_deg = self.comb.get_heat_release_rate(theta_deg, self.fuel_mass)
        # 转换为 J/rad: 乘以 deg/rad = 180/π ≈ 57.3
        dQ_comb_rad = dQ_comb_deg * (180.0 / np.pi)
        
        # 当前压力 (状态方程)
        p = m * self.gas.R * T / V
        
        # 传热损失
        p_motored = self.calculate_polytropic_pressure(theta)
        is_combustion = (x_b > 0.01 and x_b < 0.99)
        
        piston_pos = self.geo.piston_position(theta)
        clearance_h = self.geo.clearance_volume / self.geo.piston_area
        A_surf = self.ht.calculate_surface_area(self.geo.bore, piston_pos, clearance_h)
        
        h = self.ht.calculate_heat_transfer_coefficient(
            self.geo.bore, p, T, Cm, 
            self.geo.displaced_volume, p_motored, is_combustion
        )
        
        # 转换为角度域
        omega = self.rpm * 2 * np.pi / 60  # rad/s
        Q_dot_wall = self.ht.calculate_heat_transfer_rate(h, A_surf, T)
        dQ_wall_rad = Q_dot_wall / omega  # J/rad
        
        # 泄漏质量流率
        dm_leak = 0.0
        if self.leak_factor > 0:
            # 简化泄漏模型: 泄漏流量与压差成正比
            p_crank = 1.0e5  # 曲轴箱压力
            if p > p_crank:
                dm_leak = -self.leak_factor * 0.001 * (p - p_crank) / omega
        
        # 工质物性
        gamma = self.gas.get_gamma(x_b)
        cv = self.gas.get_cv(T, x_b)
        
        # 能量守恒方程 (第一定律):
        # m*cv*dT = dQ_comb - dQ_wall - p*dV - h_out*dm_out
        # 这里忽略质量变化的焓流项 (封闭系统近似)
        
        dT_dtheta = (dQ_comb_rad - dQ_wall_rad - p * dV_dtheta) / (m * cv)
        dm_dtheta = dm_leak
        
        return np.array([dT_dtheta, dm_dtheta])
    
    def solve_cycle(self, theta_start: float = None, theta_end: float = None,
                    n_points: int = 720) -> Dict:
        """
        求解一个工作循环
        
        Args:
            theta_start: 起始角 [rad], 默认为压缩起点
            theta_end: 结束角 [rad], 默认为膨胀终点
            n_points: 输出点数
            
        Returns:
            results: 包含所有热力学变量的字典
        """
        if theta_start is None:
            theta_start = self.geo.get_compression_start_angle()
        if theta_end is None:
            # 二冲程: 从扫气口关闭到扫气口开启约240度
            theta_end = theta_start + np.deg2rad(240)
        
        # 初始条件
        p0, T0, m0 = self.get_initial_conditions()
        y0 = np.array([T0, m0])
        
        # 更新传热参考状态
        V0 = self.geo.instantaneous_volume(theta_start)
        self.ht.update_reference_state(p0, T0, V0)
        
        # 求解ODE
        theta_span = np.linspace(theta_start, theta_end, n_points)
        
        solution = odeint(self.ode_system, y0, theta_span)
        
        # 后处理
        T_sol = solution[:, 0]
        m_sol = solution[:, 1]
        
        # 计算其他变量
        V_sol = np.array([self.geo.instantaneous_volume(th) for th in theta_span])
        p_sol = m_sol * self.gas.R * T_sol / V_sol
        
        theta_deg = np.rad2deg(theta_span) - 360  # 转换为ATDC
        x_b_sol = np.array([self.comb.get_burn_fraction(th) for th in theta_deg])
        
        # 存储结果
        self.results = {
            'theta_rad': theta_span,
            'theta_deg': theta_deg,
            'pressure': p_sol,
            'temperature': T_sol,
            'mass': m_sol,
            'volume': V_sol,
            'burn_fraction': x_b_sol,
            'P_max': np.max(p_sol),
            'P_comp': None,  # 待后处理
            'theta_Pmax': theta_deg[np.argmax(p_sol)],
        }
        
        # 计算压缩终点压力 (TDC处)
        tdc_idx = np.argmin(np.abs(theta_deg))
        self.results['P_comp'] = p_sol[tdc_idx]
        
        return self.results
    
    def solve_compression_only(self, n_points: int = 360) -> Dict:
        """
        仅求解压缩过程 (燃油量=0)
        
        用于第一阶段校准
        """
        original_fuel = self.fuel_mass
        self.fuel_mass = 0.0
        
        theta_start = self.geo.get_compression_start_angle()
        theta_tdc = 2 * np.pi  # TDC = 360 deg
        
        results = self.solve_cycle(theta_start, theta_tdc + 0.1, n_points)
        
        self.fuel_mass = original_fuel
        return results
    
    def get_pmax(self) -> float:
        """获取最大爆发压力"""
        if 'P_max' in self.results:
            return self.results['P_max']
        return 0.0
    
    def get_pcomp(self) -> float:
        """获取压缩终点压力"""
        if 'P_comp' in self.results:
            return self.results['P_comp']
        return 0.0
    
    def get_exhaust_temperature(self) -> float:
        """
        估算排气温度
        
        基于膨胀终点状态和等熵膨胀过程进行物理估算
        """
        if 'temperature' not in self.results or 'pressure' not in self.results:
            return 0.0
        
        # 膨胀终点状态
        T_exp_end = self.results['temperature'][-1]  # 膨胀终点温度 [K]
        p_exp_end = self.results['pressure'][-1]     # 膨胀终点压力 [Pa]
        
        # 排气背压 (扫气压力的1.05-1.1倍)
        p_exhaust = self.p_scav * 1.05
        
        # 等熵膨胀过程估算排气温度
        # T_exhaust / T_exp_end = (p_exhaust / p_exp_end)^((gamma-1)/gamma)
        # 对于已燃气体 gamma ≈ 1.28-1.32
        gamma_exhaust = 1.30
        
        if p_exp_end > p_exhaust:
            # 正常情况：排气过程继续膨胀降温
            T_exhaust = T_exp_end * (p_exhaust / p_exp_end) ** ((gamma_exhaust - 1) / gamma_exhaust)
        else:
            # 异常情况：直接使用膨胀终点温度
            T_exhaust = T_exp_end
        
        # 考虑排气管壁面传热损失 (约5-10%)
        heat_loss_factor = 0.92
        T_exhaust = T_exhaust * heat_loss_factor
        
        return T_exhaust
    
    def inject_leak_fault(self, leak_factor: float):
        """注入气缸泄漏故障"""
        self.leak_factor = leak_factor
    
    def inject_fuel_fault(self, degradation_factor: float):
        """注入燃油系统故障"""
        self.fuel_degradation = degradation_factor
    
    def reset_faults(self):
        """重置所有故障"""
        self.leak_factor = 0.0
        self.fuel_degradation = 1.0
