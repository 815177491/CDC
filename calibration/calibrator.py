"""
发动机参数校准器
================
分步解耦的三阶段校准流程
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
import warnings

import sys
sys.path.append('..')

from engine import MarineEngine0D, OperatingCondition
from .data_loader import CalibrationDataLoader, CalibrationPoint


@dataclass
class CalibrationResult:
    """校准结果"""
    stage: str
    parameters: Dict
    error: float
    iterations: int
    success: bool
    message: str


class EngineCalibrator:
    """
    发动机参数校准器
    
    采用分步解耦策略:
    1. 压缩段校准: 确定压缩比和多变指数
    2. 燃烧段校准: 确定Wiebe参数
    3. 传热段校准: 确定Woschni系数
    """
    
    def __init__(self, engine: MarineEngine0D, 
                 data_loader: CalibrationDataLoader):
        """
        初始化校准器
        
        Args:
            engine: 发动机模型实例
            data_loader: 校准数据加载器
        """
        self.engine = engine
        self.data_loader = data_loader
        
        # 校准点
        self.calibration_points: List[CalibrationPoint] = []
        
        # 校准结果
        self.results: Dict[str, CalibrationResult] = {}
        
        # 校准权重
        self.weights = {
            'P_comp': 1.0,
            'P_max': 1.0,
            'T_exhaust': 0.5
        }
    
    def load_calibration_data(self, n_points: int = 5):
        """加载校准数据点"""
        self.data_loader.load_data(nrows=100000)  # 限制读取量
        self.data_loader.preprocess()
        self.calibration_points = self.data_loader.extract_steady_state_points(
            n_points=n_points
        )
        print(f"已加载 {len(self.calibration_points)} 个校准工况点")
    
    def _point_to_condition(self, point: CalibrationPoint) -> OperatingCondition:
        """将校准点转换为运行工况"""
        fuel_mass = self.data_loader.convert_fuel_command_to_mass(
            point.fuel_command, point.rpm
        )
        
        return OperatingCondition(
            rpm=point.rpm,
            p_scav=point.p_scav,
            T_scav=point.T_scav,
            fuel_mass=fuel_mass,
            p_rail=point.p_rail,
            P_max_exp=point.P_max,
            P_comp_exp=point.P_comp,
            T_exhaust_exp=point.T_exhaust
        )
    
    # ==================== 第一阶段: 压缩段校准 ====================
    
    def calibrate_compression(self, 
                               cr_bounds: Tuple[float, float] = (11.0, 16.0),
                               tol: float = 0.02) -> CalibrationResult:
        """
        第一阶段: 压缩过程校准
        
        目标: 调整有效压缩比使仿真Pcomp与实验值误差<2%
        
        Args:
            cr_bounds: 压缩比搜索范围
            tol: 目标误差阈值
            
        Returns:
            result: 校准结果
        """
        print("=" * 50)
        print("第一阶段: 压缩过程校准")
        print("=" * 50)
        
        if not self.calibration_points:
            raise ValueError("请先加载校准数据")
        
        # 使用所有点的平均Pcomp作为目标
        target_Pcomp = np.mean([p.P_comp for p in self.calibration_points])
        
        # 选取一个代表性工况点
        ref_point = self.calibration_points[len(self.calibration_points) // 2]
        condition = self._point_to_condition(ref_point)
        
        def objective(x):
            cr = x[0]
            self.engine.set_compression_ratio(cr)
            
            # 仅运行压缩过程
            try:
                self.engine.run_compression_only(condition)
                P_comp_sim = self.engine.get_pcomp()
                error = ((P_comp_sim - ref_point.P_comp) / ref_point.P_comp) ** 2
            except Exception as e:
                error = 1e6
            
            return error
        
        # 优化求解
        result = minimize(
            objective,
            x0=[13.5],
            bounds=[cr_bounds],
            method='L-BFGS-B',
            options={'maxiter': 100}
        )
        
        # 应用最优参数
        optimal_cr = result.x[0]
        self.engine.set_compression_ratio(optimal_cr)
        
        # 验证
        self.engine.run_compression_only(condition)
        P_comp_sim = self.engine.get_pcomp()
        final_error = abs(P_comp_sim - ref_point.P_comp) / ref_point.P_comp
        
        cal_result = CalibrationResult(
            stage='compression',
            parameters={'compression_ratio': optimal_cr},
            error=final_error,
            iterations=result.nit,
            success=final_error < tol,
            message=f"Pcomp: {P_comp_sim:.2f} bar (目标: {ref_point.P_comp:.2f} bar)"
        )
        
        self.results['compression'] = cal_result
        print(f"最优压缩比: {optimal_cr:.3f}")
        print(f"压缩压力误差: {final_error*100:.2f}%")
        
        return cal_result
    
    # ==================== 第二阶段: 燃烧段校准 ====================
    
    def calibrate_combustion(self,
                              timing_bounds: Tuple[float, float] = (-5.0, 10.0),
                              duration_bounds: Tuple[float, float] = (30.0, 80.0),
                              shape_bounds: Tuple[float, float] = (0.5, 3.0),
                              tol: float = 0.03) -> CalibrationResult:
        """
        第二阶段: 燃烧放热规律校准
        
        目标: 调整Wiebe参数使仿真Pmax与实验值误差最小
        
        Args:
            timing_bounds: 喷油正时范围 [deg BTDC]
            duration_bounds: 燃烧持续期范围 [deg]
            shape_bounds: 形状因子范围
            tol: 目标误差阈值
            
        Returns:
            result: 校准结果
        """
        print("=" * 50)
        print("第二阶段: 燃烧放热规律校准")
        print("=" * 50)
        
        if 'compression' not in self.results or not self.results['compression'].success:
            warnings.warn("压缩段校准未完成或失败, 继续燃烧段校准")
        
        # 使用多个工况点进行校准
        conditions = [self._point_to_condition(p) for p in self.calibration_points]
        targets = [(p.P_max, p.P_comp) for p in self.calibration_points]
        
        def objective(x):
            inj_timing, diff_duration, diff_shape = x
            
            self.engine.set_combustion_params(
                injection_timing=inj_timing,
                diffusion_duration=diff_duration,
                diffusion_shape=diff_shape
            )
            
            total_error = 0.0
            valid_count = 0
            
            for cond, (Pmax_exp, Pcomp_exp) in zip(conditions, targets):
                try:
                    self.engine.run_cycle(cond)
                    Pmax_sim = self.engine.get_pmax()
                    
                    # Pmax相对误差
                    error_pmax = ((Pmax_sim - Pmax_exp) / Pmax_exp) ** 2
                    total_error += self.weights['P_max'] * error_pmax
                    valid_count += 1
                except Exception:
                    total_error += 1.0
            
            return total_error / max(valid_count, 1)
        
        # 使用差分进化算法进行全局优化
        bounds = [timing_bounds, duration_bounds, shape_bounds]
        
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=15,  # 减少迭代次数以加快演示
            popsize=5,   # 减少种群大小
            seed=42,
            polish=False,  # 跳过抛光步骤
            disp=True,
            workers=1
        )
        
        # 应用最优参数
        optimal_timing, optimal_duration, optimal_shape = result.x
        self.engine.set_combustion_params(
            injection_timing=optimal_timing,
            diffusion_duration=optimal_duration,
            diffusion_shape=optimal_shape
        )
        
        # 验证
        errors = []
        for cond, (Pmax_exp, _) in zip(conditions, targets):
            self.engine.run_cycle(cond)
            Pmax_sim = self.engine.get_pmax()
            errors.append(abs(Pmax_sim - Pmax_exp) / Pmax_exp)
        
        mean_error = np.mean(errors)
        
        cal_result = CalibrationResult(
            stage='combustion',
            parameters={
                'injection_timing': optimal_timing,
                'diffusion_duration': optimal_duration,
                'diffusion_shape': optimal_shape
            },
            error=mean_error,
            iterations=result.nit,
            success=mean_error < tol,
            message=f"Pmax平均误差: {mean_error*100:.2f}%"
        )
        
        self.results['combustion'] = cal_result
        print(f"最优喷油正时: {optimal_timing:.2f} deg BTDC")
        print(f"最优燃烧持续期: {optimal_duration:.1f} deg")
        print(f"最优形状因子: {optimal_shape:.2f}")
        print(f"Pmax平均误差: {mean_error*100:.2f}%")
        
        return cal_result
    
    # ==================== 第三阶段: 传热段校准 ====================
    
    def calibrate_heat_transfer(self,
                                 C_bounds: Tuple[float, float] = (80.0, 200.0),
                                 tol: float = 0.05) -> CalibrationResult:
        """
        第三阶段: 传热校准
        
        目标: 调整Woschni系数使仿真排温与实验值匹配
        
        Args:
            C_bounds: Woschni系数范围
            tol: 目标误差阈值
            
        Returns:
            result: 校准结果
        """
        print("=" * 50)
        print("第三阶段: 传热校准")
        print("=" * 50)
        
        # 筛选有排温数据的点
        valid_points = [p for p in self.calibration_points if p.T_exhaust > 100]
        
        if not valid_points:
            print("警告: 无有效排温数据, 跳过传热校准")
            return CalibrationResult(
                stage='heat_transfer',
                parameters={'C_woschni': 130.0},
                error=0.0,
                iterations=0,
                success=True,
                message="使用默认传热系数"
            )
        
        conditions = [self._point_to_condition(p) for p in valid_points]
        T_exp_list = [p.T_exhaust for p in valid_points]
        
        def objective(x):
            C_woschni = x[0]
            self.engine.set_heat_transfer_params(C_woschni=C_woschni)
            
            total_error = 0.0
            valid_count = 0
            
            for cond, T_exp in zip(conditions, T_exp_list):
                try:
                    self.engine.run_cycle(cond)
                    T_sim = self.engine.get_exhaust_temp()
                    
                    # 排温绝对误差
                    error_T = ((T_sim - T_exp) / (T_exp + 273.15)) ** 2
                    total_error += error_T
                    valid_count += 1
                except Exception:
                    total_error += 1.0
            
            return total_error / max(valid_count, 1)
        
        result = minimize(
            objective,
            x0=[130.0],
            bounds=[C_bounds],
            method='L-BFGS-B',
            options={'maxiter': 50}
        )
        
        optimal_C = result.x[0]
        self.engine.set_heat_transfer_params(C_woschni=optimal_C)
        
        # 验证
        errors = []
        for cond, T_exp in zip(conditions, T_exp_list):
            self.engine.run_cycle(cond)
            T_sim = self.engine.get_exhaust_temp()
            errors.append(abs(T_sim - T_exp) / (T_exp + 273.15))
        
        mean_error = np.mean(errors)
        
        cal_result = CalibrationResult(
            stage='heat_transfer',
            parameters={'C_woschni': optimal_C},
            error=mean_error,
            iterations=result.nit,
            success=mean_error < tol,
            message=f"排温平均误差: {mean_error*100:.2f}%"
        )
        
        self.results['heat_transfer'] = cal_result
        print(f"最优Woschni系数: {optimal_C:.1f}")
        print(f"排温平均误差: {mean_error*100:.2f}%")
        
        return cal_result
    
    # ==================== 完整校准流程 ====================
    
    def run_full_calibration(self, n_points: int = 5) -> Dict[str, CalibrationResult]:
        """
        运行完整的三阶段校准流程
        
        Args:
            n_points: 使用的校准点数量
            
        Returns:
            results: 各阶段校准结果
        """
        print("开始三阶段分步解耦校准...")
        print("=" * 60)
        
        # 加载数据
        self.load_calibration_data(n_points)
        
        # 第一阶段
        self.calibrate_compression()
        
        # 第二阶段
        self.calibrate_combustion()
        
        # 第三阶段
        self.calibrate_heat_transfer()
        
        # 汇总
        print("\n" + "=" * 60)
        print("校准完成! 最终参数:")
        print("=" * 60)
        for stage, result in self.results.items():
            status = "✓" if result.success else "✗"
            print(f"[{status}] {stage}: {result.parameters}")
            print(f"    误差: {result.error*100:.2f}%, {result.message}")
        
        return self.results
    
    def get_calibrated_engine(self) -> MarineEngine0D:
        """返回校准后的发动机模型"""
        return self.engine
    
    def export_parameters(self, filepath: str = None) -> Dict:
        """导出校准参数"""
        params = {}
        for stage, result in self.results.items():
            params.update(result.parameters)
        
        if filepath:
            import json
            with open(filepath, 'w') as f:
                json.dump(params, f, indent=2)
        
        return params
