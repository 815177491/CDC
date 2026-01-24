"""
零维船用柴油机模型
==================
集成几何、燃烧、传热与热力学求解器
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from copy import deepcopy
from scipy.integrate import simpson

from .geometry import EngineGeometry
from .combustion import DoublieWiebeCombustion
from .heat_transfer import WoschniHeatTransfer
from .thermodynamics import ThermodynamicSolver, ThermodynamicState
from .config import EngineConfig, DEFAULT_ENGINE_CONFIG


@dataclass
class OperatingCondition:
    """运行工况数据结构"""
    rpm: float  # 发动机转速 [rpm]
    p_scav: float  # 扫气压力 [Pa]
    T_scav: float  # 扫气温度 [K]
    fuel_mass: float  # 循环喷油量 [kg]
    p_rail: float = 0.0  # 共轨压力 [Pa]
    
    # 可选的实验对标值
    P_max_exp: Optional[float] = None
    P_comp_exp: Optional[float] = None
    T_exhaust_exp: Optional[float] = None


class MarineEngine0D:
    """
    零维二冲程船用柴油机仿真模型
    
    集成所有子模块,提供完整的循环仿真能力
    
    Attributes:
        geometry: 发动机几何模块
        combustion: 燃烧模型
        heat_transfer: 传热模型
        solver: 热力学求解器
    """
    
    def __init__(self, 
                 config: EngineConfig = None,
                 bore: float = None,
                 stroke: float = None,
                 n_cylinders: int = None,
                 compression_ratio: float = None,
                 con_rod_ratio: float = None):
        """
        初始化发动机模型
        
        Args:
            config: 共享配置对象 (推荐使用，确保与PINN一致)
            bore: 气缸直径 [m] (向后兼容)
            stroke: 活塞行程 [m] (向后兼容)
            n_cylinders: 气缸数量 (向后兼容)
            compression_ratio: 有效压缩比 (向后兼容)
            con_rod_ratio: 连杆比 (向后兼容)
        """
        # 使用配置对象或默认值
        if config is None:
            config = DEFAULT_ENGINE_CONFIG
        self.config = config
        
        # 允许单独参数覆盖配置（向后兼容）
        _bore = bore if bore is not None else config.bore
        _stroke = stroke if stroke is not None else config.stroke
        _n_cylinders = n_cylinders if n_cylinders is not None else config.n_cylinders
        _compression_ratio = compression_ratio if compression_ratio is not None else config.compression_ratio
        _con_rod_ratio = con_rod_ratio if con_rod_ratio is not None else config.con_rod_ratio
        
        # 初始化子模块
        self.geometry = EngineGeometry(
            bore=_bore,
            stroke=_stroke,
            n_cylinders=_n_cylinders,
            compression_ratio=_compression_ratio,
            con_rod_ratio=_con_rod_ratio
        )
        
        self.combustion = DoublieWiebeCombustion()
        self.heat_transfer = WoschniHeatTransfer()
        
        self.solver = ThermodynamicSolver(
            self.geometry,
            self.combustion,
            self.heat_transfer
        )
        
        # 存储校准后的参数
        self.calibrated_params: Dict = {}
        
        # 仿真结果
        self.cycle_results: Dict = {}
        
        # 多缸状态 (用于不均匀性分析)
        self.cylinder_states: List[Dict] = [{} for _ in range(_n_cylinders)]
        
        # 健康基准 (用于诊断)
        self.baseline_Pmax: float = 0.0
        self.baseline_Pcomp: float = 0.0
        self.baseline_Texh: float = 0.0
    
    def run_cycle(self, condition: OperatingCondition) -> Dict:
        """
        运行单个工作循环仿真
        
        Args:
            condition: 运行工况
            
        Returns:
            results: 循环仿真结果
        """
        # 设置工况
        self.solver.set_operating_conditions(
            rpm=condition.rpm,
            p_scav=condition.p_scav,
            T_scav=condition.T_scav,
            fuel_mass=condition.fuel_mass
        )
        
        # 求解循环
        results = self.solver.solve_cycle()
        
        # 添加派生量
        results['IMEP'] = self._calculate_imep(results)
        results['T_exhaust'] = self.solver.get_exhaust_temperature()
        
        self.cycle_results = results
        return results
    
    def run_compression_only(self, condition: OperatingCondition) -> Dict:
        """
        仅运行压缩过程 (用于第一阶段校准)
        
        Args:
            condition: 运行工况
            
        Returns:
            results: 压缩过程结果
        """
        self.solver.set_operating_conditions(
            rpm=condition.rpm,
            p_scav=condition.p_scav,
            T_scav=condition.T_scav,
            fuel_mass=0.0  # 无燃烧
        )
        
        results = self.solver.solve_compression_only()
        return results
    
    def _calculate_imep(self, results: Dict) -> float:
        """
        计算指示平均有效压力
        
        Args:
            results: 循环仿真结果
            
        Returns:
            IMEP: 指示平均有效压力 [bar]
        """
        p = results['pressure']  # Pa
        V = results['volume']    # m^3
        
        # 使用 Simpson 积分计算指示功 (更高精度)
        # 对于二冲程发动机，积分区间是有效压缩膨胀过程
        work = simpson(p, x=V)  # 单位: J
        
        # IMEP = Work / Displaced_Volume
        # 转换为 bar 以保持与 Pmax, Pcomp 单位一致
        IMEP = (work / self.geometry.displaced_volume) / 1e5
        return IMEP
    
    def get_pmax(self, cylinder_index: int = None) -> float:
        """
        获取最大爆发压力 [bar]
        
        Args:
            cylinder_index: 气缸索引 (0-based)，None表示使用当前cycle_results
        """
        if cylinder_index is not None and 0 <= cylinder_index < len(self.cylinder_states):
            result = self.cylinder_states[cylinder_index]
            if 'P_max' in result:
                return result['P_max'] / 1e5
        return self.solver.get_pmax() / 1e5
    
    def get_pcomp(self, cylinder_index: int = None) -> float:
        """
        获取压缩终点压力 [bar]
        
        Args:
            cylinder_index: 气缸索引 (0-based)，None表示使用当前cycle_results
        """
        if cylinder_index is not None and 0 <= cylinder_index < len(self.cylinder_states):
            result = self.cylinder_states[cylinder_index]
            if 'P_comp' in result:
                return result['P_comp'] / 1e5
        return self.solver.get_pcomp() / 1e5
    
    def get_exhaust_temp(self, cylinder_index: int = None) -> float:
        """
        获取排气温度 [°C]
        
        Args:
            cylinder_index: 气缸索引 (0-based)，None表示使用当前cycle_results
        """
        if cylinder_index is not None and 0 <= cylinder_index < len(self.cylinder_states):
            result = self.cylinder_states[cylinder_index]
            if 'T_exhaust' in result:
                return result['T_exhaust'] - 273.15
        return self.solver.get_exhaust_temperature() - 273.15
    
    # ==================== 校准接口 ====================
    
    def set_compression_ratio(self, cr: float):
        """设置有效压缩比"""
        self.geometry.update_calibration_params(compression_ratio=cr)
        self.calibrated_params['compression_ratio'] = cr
    
    def set_combustion_params(self, **kwargs):
        """设置燃烧模型参数"""
        self.combustion.update_calibration_params(**kwargs)
        self.calibrated_params.update(kwargs)
    
    def set_heat_transfer_params(self, **kwargs):
        """设置传热模型参数"""
        self.heat_transfer.update_calibration_params(**kwargs)
        self.calibrated_params.update(kwargs)
    
    def set_injection_timing(self, timing: float):
        """设置喷油正时 [deg BTDC]"""
        self.combustion.injection_timing = timing
        self.calibrated_params['injection_timing'] = timing
    
    # ==================== 故障注入接口 ====================
    
    def inject_timing_fault(self, offset_deg: float):
        """
        注入喷油正时故障
        
        Args:
            offset_deg: 正时偏差 [deg], 正值=提前
        """
        self.combustion.apply_injection_timing_fault(offset_deg)
    
    def inject_leak_fault(self, leak_factor: float):
        """
        注入气缸泄漏故障
        
        Args:
            leak_factor: 泄漏因子 (0-1), 0=无泄漏, 1=严重泄漏
        """
        self.solver.inject_leak_fault(leak_factor)
    
    def inject_fuel_fault(self, degradation: float):
        """
        注入燃油系统故障
        
        Args:
            degradation: 燃油衰减系数 (0-1), 1=正常, 0=完全堵塞
        """
        self.solver.inject_fuel_fault(degradation)
    
    def clear_faults(self):
        """清除所有故障"""
        self.combustion.reset_injection_timing(
            self.calibrated_params.get('injection_timing', 2.0)
        )
        self.solver.reset_faults()
    
    # ==================== 诊断接口 ====================
    
    def set_baseline(self, Pmax: float, Pcomp: float, Texh: float):
        """设置健康基准值"""
        self.baseline_Pmax = Pmax
        self.baseline_Pcomp = Pcomp
        self.baseline_Texh = Texh
    
    def compute_residuals(self, Y_real: Dict) -> Dict:
        """
        计算诊断残差
        
        Args:
            Y_real: 实际测量值 {'Pmax': ..., 'Pcomp': ..., 'Texh': ...}
            
        Returns:
            residuals: 残差字典
        """
        residuals = {
            'r_Pmax': Y_real.get('Pmax', 0) - self.get_pmax(),
            'r_Pcomp': Y_real.get('Pcomp', 0) - self.get_pcomp(),
            'r_Texh': Y_real.get('Texh', 0) - self.get_exhaust_temp()
        }
        return residuals
    
    # ==================== 多缸仿真 ====================
    
    def run_all_cylinders(self, base_condition: OperatingCondition,
                          fuel_imbalance: List[float] = None) -> List[Dict]:
        """
        运行所有气缸的仿真
        
        Args:
            base_condition: 基准工况
            fuel_imbalance: 各缸燃油不均匀系数, 如 [1.0, 1.02, 0.98, ...]
            
        Returns:
            results_list: 各缸结果列表
        """
        n_cyl = self.geometry.n_cylinders
        
        if fuel_imbalance is None:
            fuel_imbalance = [1.0] * n_cyl
        
        results_list = []
        for i in range(n_cyl):
            cond = deepcopy(base_condition)
            cond.fuel_mass *= fuel_imbalance[i]
            
            results = self.run_cycle(cond)
            results['cylinder'] = i + 1
            results_list.append(results)
            self.cylinder_states[i] = results
        
        return results_list
    
    def get_pv_diagram_data(self, cylinder_index: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取P-V示功图数据
        
        Args:
            cylinder_index: 气缸索引 (0-based)，None表示使用当前cycle_results
        
        Returns:
            (V, p): 容积和压力数组
        """
        # 选择数据源：指定气缸或当前结果
        if cylinder_index is not None and 0 <= cylinder_index < len(self.cylinder_states):
            result = self.cylinder_states[cylinder_index]
        else:
            result = self.cycle_results
        
        if 'volume' in result and 'pressure' in result:
            return result['volume'], result['pressure']
        return np.array([]), np.array([])
    
    def get_all_cylinders_summary(self) -> Dict:
        """
        获取所有气缸的汇总数据
        
        Returns:
            summary: 包含各缸Pmax, Pcomp, IMEP等的汇总字典
        """
        summary = {
            'Pmax': [],
            'Pcomp': [],
            'IMEP': [],
            'T_exhaust': []
        }
        for i, state in enumerate(self.cylinder_states):
            if state:  # 非空状态
                summary['Pmax'].append(state.get('P_max', 0) / 1e5)
                summary['Pcomp'].append(state.get('P_comp', 0) / 1e5)
                summary['IMEP'].append(state.get('IMEP', 0))
                summary['T_exhaust'].append(state.get('T_exhaust', 273.15) - 273.15)
        
        # 计算统计量
        if summary['Pmax']:
            summary['Pmax_mean'] = np.mean(summary['Pmax'])
            summary['Pmax_std'] = np.std(summary['Pmax'])
            summary['IMEP_mean'] = np.mean(summary['IMEP'])
            summary['imbalance_index'] = np.std(summary['IMEP']) / np.mean(summary['IMEP']) if np.mean(summary['IMEP']) > 0 else 0
        
        return summary
    
    def get_heat_release_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取放热率曲线数据
        
        Returns:
            (theta, dQ): 曲轴转角和放热率
        """
        if 'theta_deg' not in self.cycle_results:
            return np.array([]), np.array([])
        
        theta = self.cycle_results['theta_deg']
        dQ = np.array([
            self.combustion.get_heat_release_rate(th, self.solver.fuel_mass)
            for th in theta
        ])
        return theta, dQ
    
    def __repr__(self) -> str:
        return (f"MarineEngine0D(bore={self.geometry.bore*1000:.0f}mm, "
                f"stroke={self.geometry.stroke*1000:.0f}mm, "
                f"n_cyl={self.geometry.n_cylinders})")
