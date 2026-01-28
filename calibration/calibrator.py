"""
发动机参数校准器
================
分步解耦的三阶段校准流程

包括:
1. 压缩段校准: 确定有效压缩比
2. 燃烧段校准: 确定Wiebe参数 (喷油正时、燃烧持续期、形状因子)
3. 传热段校准: 确定Woschni系数

支持功能:
- 收敛历史记录 (包含参数值)
- 验证结果导出
- 可视化数据生成

Author: CDC Project
Date: 2026-01-28
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass, field
import warnings
from datetime import datetime

import sys
sys.path.append('..')

from engine import MarineEngine0D, OperatingCondition
from .data_loader import CalibrationDataLoader, CalibrationPoint

# 尝试导入全局配置
try:
    from config import PATH_CONFIG
except ImportError:
    PATH_CONFIG = None


@dataclass
class CalibrationResult:
    """校准结果"""
    stage: str
    parameters: Dict
    error: float
    iterations: int
    success: bool
    message: str


@dataclass
class ConvergenceRecord:
    """收敛记录（单次迭代）"""
    iteration: int
    objective_value: float
    best_value: float
    parameters: Dict
    stage: str


class EngineCalibrator:
    """
    发动机参数校准器
    
    采用分步解耦策略:
    1. 压缩段校准: 确定压缩比和多变指数
    2. 燃烧段校准: 确定Wiebe参数
    3. 传热段校准: 确定Woschni系数
    
    Attributes:
        engine: 发动机模型实例
        data_loader: 校准数据加载器
        calibration_points: 校准工况点列表
        results: 各阶段校准结果
        convergence_history: 收敛历史记录
        validation_results: 验证结果数据
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
        
        # 收敛历史记录
        self.convergence_history: List[ConvergenceRecord] = []
        
        # 验证结果
        self.validation_results: List[Dict] = []
        
        # 校准权重
        self.weights = {
            'P_comp': 1.0,
            'P_max': 1.0,
            'T_exhaust': 0.5
        }
        
        # 当前阶段的最优值跟踪
        self._current_best_value = float('inf')
        self._iteration_counter = 0
    
    def load_calibration_data(self, n_points: int = 5):
        """加载校准数据点"""
        self.data_loader.load_data(nrows=100000)  # 限制读取量
        self.data_loader.preprocess()
        self.calibration_points = self.data_loader.extract_steady_state_points(
            n_points=n_points
        )
        print(f"已加载 {len(self.calibration_points)} 个校准工况点")
    
    def _reset_convergence_tracking(self, stage: str):
        """重置收敛跟踪状态"""
        self._current_best_value = float('inf')
        self._iteration_counter = 0
        self._current_stage = stage
    
    def _record_convergence(self, objective_value: float, parameters: Dict, stage: str):
        """
        记录收敛历史
        
        Args:
            objective_value: 当前目标函数值
            parameters: 当前参数值
            stage: 校准阶段名称
        """
        self._iteration_counter += 1
        
        # 更新最优值
        if objective_value < self._current_best_value:
            self._current_best_value = objective_value
        
        # 创建记录
        record = ConvergenceRecord(
            iteration=self._iteration_counter,
            objective_value=objective_value,
            best_value=self._current_best_value,
            parameters=parameters.copy(),
            stage=stage
        )
        self.convergence_history.append(record)
    
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
        
        # 重置收敛跟踪
        self._reset_convergence_tracking('compression')
        
        # 使用多个代表性工况点进行加权校准（选取高、中、低负荷各一个）
        n_pts = len(self.calibration_points)
        if n_pts >= 3:
            # 选取低、中、高负荷的代表性工况点
            ref_indices = [0, n_pts // 2, n_pts - 1]
            ref_points = [self.calibration_points[i] for i in ref_indices]
        else:
            ref_points = self.calibration_points
        
        conditions = [self._point_to_condition(p) for p in ref_points]
        
        def objective(x):
            cr = x[0]
            self.engine.set_compression_ratio(cr)
            
            total_error = 0.0
            valid_count = 0
            
            # 在多个工况点上计算加权误差
            for point, cond in zip(ref_points, conditions):
                try:
                    self.engine.run_compression_only(cond)
                    P_comp_sim = self.engine.get_pcomp()
                    error = ((P_comp_sim - point.P_comp) / point.P_comp) ** 2
                    total_error += error
                    valid_count += 1
                except Exception as e:
                    total_error += 1e6
            
            avg_error = total_error / max(valid_count, 1)
            
            # 记录收敛历史
            self._record_convergence(
                objective_value=avg_error,
                parameters={'compression_ratio': cr},
                stage='compression'
            )
            
            return avg_error
        
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
        
        # 验证 - 在所有参考工况点上计算平均误差
        errors = []
        for point, cond in zip(ref_points, conditions):
            self.engine.run_compression_only(cond)
            P_comp_sim = self.engine.get_pcomp()
            errors.append(abs(P_comp_sim - point.P_comp) / point.P_comp)
        
        final_error = np.mean(errors)
        
        cal_result = CalibrationResult(
            stage='compression',
            parameters={'compression_ratio': optimal_cr},
            error=final_error,
            iterations=result.nit,
            success=final_error < tol,
            message=f"Pcomp平均误差: {final_error*100:.2f}% (使用{len(ref_points)}个工况点)"
        )
        
        self.results['compression'] = cal_result
        print(f"最优压缩比: {optimal_cr:.3f}")
        print(f"压缩压力误差: {final_error*100:.2f}%")
        
        return cal_result
    
    # ==================== 第二阶段: 燃烧段校准 ====================
    
    def calibrate_combustion(self,
                              timing_bounds: Tuple[float, float] = (-5.0, 10.0),
                              duration_bounds: Tuple[float, float] = (20.0, 100.0),
                              shape_bounds: Tuple[float, float] = (0.3, 4.0),
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
        
        # 重置收敛跟踪
        self._reset_convergence_tracking('combustion')
        
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
            
            avg_error = total_error / max(valid_count, 1)
            
            # 记录收敛历史
            self._record_convergence(
                objective_value=avg_error,
                parameters={
                    'injection_timing': inj_timing,
                    'diffusion_duration': diff_duration,
                    'diffusion_shape': diff_shape
                },
                stage='combustion'
            )
            
            return avg_error
        
        # 使用差分进化算法进行全局优化
        bounds = [timing_bounds, duration_bounds, shape_bounds]
        
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=50,  # 增加迭代次数以充分探索参数空间
            popsize=15,  # 增加种群大小以提高全局搜索能力
            seed=42,
            polish=True,  # 启用局部精化以提高精度
            disp=True,
            workers=1,
            tol=0.001    # 收敛容差
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
        
        # 重置收敛跟踪
        self._reset_convergence_tracking('heat_transfer')
        
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
            
            avg_error = total_error / max(valid_count, 1)
            
            # 记录收敛历史
            self._record_convergence(
                objective_value=avg_error,
                parameters={'C_woschni': C_woschni},
                stage='heat_transfer'
            )
            
            return avg_error
        
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
    
    def run_full_calibration(self, n_points: int = 5, 
                              export_results: bool = True) -> Dict[str, CalibrationResult]:
        """
        运行完整的三阶段校准流程
        
        Args:
            n_points: 使用的校准点数量
            export_results: 是否自动导出收敛历史和验证结果
            
        Returns:
            results: 各阶段校准结果
        """
        print("开始三阶段分步解耦校准...")
        print("=" * 60)
        
        # 清空历史记录
        self.convergence_history = []
        self.validation_results = []
        
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
        
        # 生成验证结果
        self._generate_validation_results()
        
        # 导出结果
        if export_results:
            self.export_convergence_history()
            self.export_validation_results()
            self.export_parameters()
        
        return self.results
    
    def _generate_validation_results(self):
        """
        生成验证结果
        
        使用校准后的模型对所有校准工况点进行仿真，
        计算各指标的误差
        """
        self.validation_results = []
        
        for idx, point in enumerate(self.calibration_points):
            condition = self._point_to_condition(point)
            
            try:
                self.engine.run_cycle(condition)
                Pmax_sim = self.engine.get_pmax()
                Pcomp_sim = self.engine.get_pcomp()
                Texh_sim = self.engine.get_exhaust_temp()
                
                result = {
                    'point_id': idx + 1,
                    'rpm': point.rpm,
                    'fuel_command': point.fuel_command,
                    'p_scav': point.p_scav,
                    'T_scav': point.T_scav,
                    'Pmax_exp': point.P_max,
                    'Pmax_sim': Pmax_sim,
                    'Pmax_error': (Pmax_sim - point.P_max) / point.P_max * 100,
                    'Pcomp_exp': point.P_comp,
                    'Pcomp_sim': Pcomp_sim,
                    'Pcomp_error': (Pcomp_sim - point.P_comp) / point.P_comp * 100,
                    'Texh_exp': point.T_exhaust,
                    'Texh_sim': Texh_sim,
                    'Texh_error': (Texh_sim - point.T_exhaust) / (point.T_exhaust + 273.15) * 100 if point.T_exhaust > 100 else 0.0
                }
                self.validation_results.append(result)
                
            except Exception as e:
                print(f"警告: 工况点 {idx+1} 仿真失败: {e}")
                continue
    
    def export_convergence_history(self, filepath: str = None) -> str:
        """
        导出收敛历史到CSV
        
        Args:
            filepath: 输出文件路径，默认为 data/calibration/calibration_convergence.csv
            
        Returns:
            filepath: 保存的文件路径
        """
        if filepath is None:
            if PATH_CONFIG is not None:
                filepath = os.path.join(PATH_CONFIG.DATA_CALIBRATION_DIR, 'calibration_convergence.csv')
            else:
                filepath = 'data/calibration/calibration_convergence.csv'
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 转换为DataFrame
        records = []
        for rec in self.convergence_history:
            row = {
                'iteration': rec.iteration,
                'objective_value': rec.objective_value,
                'best_value': rec.best_value,
                'stage': rec.stage
            }
            # 添加参数列
            for k, v in rec.parameters.items():
                row[f'param_{k}'] = v
            records.append(row)
        
        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False)
        
        print(f"收敛历史已导出: {filepath}")
        return filepath
    
    def export_validation_results(self, filepath: str = None) -> str:
        """
        导出验证结果到CSV
        
        Args:
            filepath: 输出文件路径，默认为 data/calibration/calibration_validation.csv
            
        Returns:
            filepath: 保存的文件路径
        """
        if filepath is None:
            if PATH_CONFIG is not None:
                filepath = os.path.join(PATH_CONFIG.DATA_CALIBRATION_DIR, 'calibration_validation.csv')
            else:
                filepath = 'data/calibration/calibration_validation.csv'
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        df = pd.DataFrame(self.validation_results)
        df.to_csv(filepath, index=False)
        
        print(f"验证结果已导出: {filepath}")
        return filepath
    
    def get_calibrated_engine(self) -> MarineEngine0D:
        """返回校准后的发动机模型"""
        return self.engine
    
    def export_parameters(self, filepath: str = None) -> Dict:
        """
        导出校准参数到JSON
        
        Args:
            filepath: 输出文件路径，默认为 data/calibration/calibrated_params.json
            
        Returns:
            params: 校准参数字典
        """
        params = {}
        for stage, result in self.results.items():
            params.update(result.parameters)
        
        if filepath is None:
            if PATH_CONFIG is not None:
                filepath = os.path.join(PATH_CONFIG.DATA_CALIBRATION_DIR, 'calibrated_params.json')
            else:
                filepath = 'data/calibration/calibrated_params.json'
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        
        print(f"校准参数已导出: {filepath}")
        return params
    
    def get_convergence_summary(self) -> Dict:
        """
        获取收敛历史摘要
        
        Returns:
            summary: 包含各阶段收敛统计的字典
        """
        summary = {}
        
        for stage in ['compression', 'combustion', 'heat_transfer']:
            stage_records = [r for r in self.convergence_history if r.stage == stage]
            
            if stage_records:
                summary[stage] = {
                    'total_iterations': len(stage_records),
                    'initial_objective': stage_records[0].objective_value,
                    'final_objective': stage_records[-1].objective_value,
                    'best_objective': min(r.objective_value for r in stage_records),
                    'convergence_ratio': stage_records[-1].best_value / stage_records[0].objective_value if stage_records[0].objective_value > 0 else 0
                }
        
        return summary
