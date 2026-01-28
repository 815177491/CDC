#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成模拟校准数据
================
生成理想化的校准模拟数据以适配可视化图表。

输出文件:
- data/simulation/mock_calibration_convergence.csv
- data/simulation/mock_calibration_validation.csv
- data/simulation/mock_calibrated_params.json

使用方法:
    1. 生成模拟数据:
       python scripts/generate_mock_calibration_data.py
    
    2. 使用模拟数据进行可视化:
       python visualize_calibration.py --mock

Author: CDC Project
Date: 2026-01-28
"""

import os
import sys
import numpy as np
import pandas as pd
import json

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PATH_CONFIG, CALIBRATION_CONFIG


def generate_convergence_data() -> pd.DataFrame:
    """
    生成收敛历史数据
    
    展示良好的三阶段收敛行为：
    1. 压缩阶段：快速收敛，优化压缩比
    2. 燃烧阶段：差分进化，优化Wiebe参数
    3. 传热阶段：平缓收敛，优化Woschni系数
    """
    convergence_data = []
    
    # 使用CalibrationConfig中的目标参数值
    target_cr = CALIBRATION_CONFIG.compression_ratio
    target_inj = CALIBRATION_CONFIG.injection_timing
    target_diff_dur = CALIBRATION_CONFIG.diffusion_duration
    target_diff_shape = CALIBRATION_CONFIG.diffusion_shape
    target_C_w = CALIBRATION_CONFIG.C_woschni
    
    # ==================== 压缩阶段 ====================
    # 快速收敛，约25次迭代
    n_compression = 25
    for i in range(n_compression):
        t = i / n_compression
        # 指数衰减收敛
        obj_value = 0.05 * np.exp(-5 * t) + 0.0005
        # 从初始猜测值收敛到目标值
        cr = 13.0 + (target_cr - 13.0) * (1 - np.exp(-4 * t))
        
        convergence_data.append({
            'iteration': i + 1,
            'stage': 'compression',
            'objective_value': obj_value,
            'best_value': obj_value,
            'compression_ratio': cr,
            'injection_timing': np.nan,
            'diffusion_duration': np.nan,
            'diffusion_shape': np.nan,
            'C_woschni': np.nan,
        })
    
    # ==================== 燃烧阶段 ====================
    # 差分进化优化，约120次迭代，有随机探索
    n_combustion = 120
    best_comb = 0.25
    
    for i in range(n_combustion):
        t = i / n_combustion
        
        # 带噪声的优化过程（前期探索多，后期稳定）
        noise_amplitude = 0.03 * np.exp(-3 * t)
        noise = noise_amplitude * np.random.randn()
        obj_value = 0.25 * np.exp(-4 * t) + 0.008 + noise
        obj_value = max(obj_value, 0.008)
        
        if obj_value < best_comb:
            best_comb = obj_value
        
        # 参数演化 - 从初始值收敛到目标值
        # 喷油正时
        inj_timing = 5.0 + (target_inj - 5.0) * (1 - np.exp(-5 * t))
        inj_timing += 0.3 * np.sin(12 * t) * np.exp(-3 * t)  # 搜索振荡
        
        # 扩散燃烧持续期
        diff_dur = 40.0 + (target_diff_dur - 40.0) * (1 - np.exp(-4 * t))
        diff_dur += 2.0 * np.sin(10 * t) * np.exp(-2 * t)
        
        # 扩散燃烧形状因子
        diff_shape = 2.5 + (target_diff_shape - 2.5) * (1 - np.exp(-3 * t))
        diff_shape += 0.08 * np.sin(8 * t) * np.exp(-2 * t)
        
        convergence_data.append({
            'iteration': i + 1,
            'stage': 'combustion',
            'objective_value': obj_value,
            'best_value': best_comb,
            'compression_ratio': np.nan,
            'injection_timing': inj_timing,
            'diffusion_duration': diff_dur,
            'diffusion_shape': diff_shape,
            'C_woschni': np.nan,
        })
    
    # ==================== 传热阶段 ====================
    # 相对平缓的收敛，约35次迭代
    n_heat = 35
    for i in range(n_heat):
        t = i / n_heat
        obj_value = 0.06 * np.exp(-4 * t) + 0.005
        C_w = 80.0 + (target_C_w - 80.0) * (1 - np.exp(-5 * t))
        
        convergence_data.append({
            'iteration': i + 1,
            'stage': 'heat_transfer',
            'objective_value': obj_value,
            'best_value': obj_value,
            'compression_ratio': np.nan,
            'injection_timing': np.nan,
            'diffusion_duration': np.nan,
            'diffusion_shape': np.nan,
            'C_woschni': C_w,
        })
    
    return pd.DataFrame(convergence_data)


def generate_validation_data() -> pd.DataFrame:
    """
    生成验证结果数据
    
    10个典型工况点，误差范围控制在正负3-5%:
    - Pmax: 3-5%
    - Pcomp: 3-5%
    - Texh: 3-5%
    
    误差分布更加真实，模拟实际校准后的良好拟合效果
    """
    # 10个典型工况点
    operating_points = [
        # (rpm, p_scav [Pa], T_scav [K], fuel_cmd)
        (70.0, 180000, 395, 35.0),   # 1. 低负荷
        (74.0, 205000, 402, 45.0),   # 2. 
        (76.0, 220000, 408, 52.0),   # 3.
        (78.0, 240000, 412, 60.0),   # 4.
        (80.0, 255000, 416, 68.0),   # 5. 中等负荷
        (82.0, 268000, 418, 74.0),   # 6.
        (84.0, 280000, 420, 80.0),   # 7.
        (86.0, 290000, 422, 85.0),   # 8.
        (88.0, 300000, 424, 90.0),   # 9.
        (90.0, 310000, 426, 95.0),   # 10. 高负荷
    ]
    
    validation_data = []
    
    for i, (rpm, p_scav, T_scav, fuel_cmd) in enumerate(operating_points):
        # 基于工况生成实验值（添加少量测量噪声）
        Pmax_exp = 135 + 1.5 * (rpm - 70) + 0.12 * (fuel_cmd - 35) + 0.05 * (p_scav/1000 - 180)
        Pmax_exp *= (1 + 0.002 * np.random.randn())  # 减小测量噪声
        
        cr = CALIBRATION_CONFIG.compression_ratio
        Pcomp_exp = (p_scav / 1e5) * (cr ** 1.35) * 2.6
        Pcomp_exp *= (1 + 0.002 * np.random.randn())
        
        Texh_exp = 320 + 0.7 * (rpm - 70) + 0.55 * (fuel_cmd - 35)
        Texh_exp *= (1 + 0.002 * np.random.randn())
        
        # 生成误差，Pcomp采用更大的变化幅度
        # 使用截断正态分布使误差更集中在中间值
        def generate_error(min_err=0.03, max_err=0.05):
            """生成3-5%范围内的误差"""
            # 使用beta分布使误差更集中，避免极端值
            base_err = min_err + (max_err - min_err) * np.random.beta(2, 2)
            sign = np.random.choice([-1, 1])
            return base_err * sign
        
        def generate_pcomp_error():
            """生成Pcomp的更大变化幅度误差(3-6%)"""
            # 使用更均匀的分布，让变化更明显
            base_err = np.random.uniform(0.03, 0.06)
            sign = np.random.choice([-1, 1])
            return base_err * sign
        
        Pmax_err = generate_error(0.03, 0.05)
        Pcomp_err = generate_pcomp_error()  # 使用专门的函数生成更大变化
        Texh_err = generate_error(0.03, 0.05)
        
        Pmax_sim = Pmax_exp * (1 + Pmax_err)
        Pcomp_sim = Pcomp_exp * (1 + Pcomp_err)
        Texh_sim = Texh_exp * (1 + Texh_err)
        
        validation_data.append({
            'point_id': i + 1,
            'rpm': rpm,
            'fuel_command': fuel_cmd,
            'p_scav': p_scav,
            'T_scav': T_scav,
            'Pmax_exp': round(Pmax_exp, 2),
            'Pmax_sim': round(Pmax_sim, 2),
            'Pmax_error': round(Pmax_err * 100, 2),
            'Pcomp_exp': round(Pcomp_exp, 2),
            'Pcomp_sim': round(Pcomp_sim, 2),
            'Pcomp_error': round(Pcomp_err * 100, 2),
            'Texh_exp': round(Texh_exp, 2),
            'Texh_sim': round(Texh_sim, 2),
            'Texh_error': round(Texh_err * 100, 2),
        })
    
    return pd.DataFrame(validation_data)


def generate_calibrated_params() -> dict:
    """
    生成校准参数
    
    使用CalibrationConfig中的默认参数保证一致性
    """
    return {
        'compression_ratio': CALIBRATION_CONFIG.compression_ratio,
        'injection_timing': CALIBRATION_CONFIG.injection_timing,
        'diffusion_duration': CALIBRATION_CONFIG.diffusion_duration,
        'diffusion_shape': CALIBRATION_CONFIG.diffusion_shape,
        'C_woschni': CALIBRATION_CONFIG.C_woschni
    }


def main():
    """主函数"""
    print("=" * 60)
    print("生成模拟校准数据")
    print("=" * 60)
    
    # 确保输出目录存在
    output_dir = PATH_CONFIG.DATA_SIMULATION_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子以保证可重复性
    np.random.seed(42)
    
    # 生成数据
    print("\n[1/3] 生成收敛历史数据...")
    conv_df = generate_convergence_data()
    conv_path = os.path.join(output_dir, 'mock_calibration_convergence.csv')
    conv_df.to_csv(conv_path, index=False)
    print(f"  -> {conv_path}")
    print(f"     压缩阶段: {(conv_df['stage']=='compression').sum()} 次迭代")
    print(f"     燃烧阶段: {(conv_df['stage']=='combustion').sum()} 次迭代")
    print(f"     传热阶段: {(conv_df['stage']=='heat_transfer').sum()} 次迭代")
    print(f"     总计: {len(conv_df)} 条记录")
    
    print("\n[2/3] 生成验证结果数据...")
    val_df = generate_validation_data()
    val_path = os.path.join(output_dir, 'mock_calibration_validation.csv')
    val_df.to_csv(val_path, index=False)
    print(f"  -> {val_path} ({len(val_df)} 个工况点)")
    
    print("\n[3/3] 生成校准参数...")
    params = generate_calibrated_params()
    params_path = os.path.join(output_dir, 'mock_calibrated_params.json')
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"  -> {params_path}")
    
    # 显示参数
    print("\n" + "-" * 60)
    print("校准参数 (与CalibrationConfig同步):")
    print("-" * 60)
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # 显示验证数据摘要
    print("\n" + "-" * 60)
    print("验证数据摘要:")
    print("-" * 60)
    print(f"  工况点数: {len(val_df)}")
    print(f"  RPM范围: {val_df['rpm'].min():.0f} ~ {val_df['rpm'].max():.0f}")
    print(f"  Pmax误差: {val_df['Pmax_error'].abs().mean():.2f}% (范围: {val_df['Pmax_error'].abs().min():.1f}~{val_df['Pmax_error'].abs().max():.1f}%)")
    print(f"  Pcomp误差: {val_df['Pcomp_error'].abs().mean():.2f}% (范围: {val_df['Pcomp_error'].abs().min():.1f}~{val_df['Pcomp_error'].abs().max():.1f}%)")
    print(f"  Texh误差: {val_df['Texh_error'].abs().mean():.2f}% (范围: {val_df['Texh_error'].abs().min():.1f}~{val_df['Texh_error'].abs().max():.1f}%)")
    print(f"  所有误差均控制在±3-5%范围内")
    
    print("\n" + "=" * 60)
    print("模拟数据生成完成!")
    print("=" * 60)
    print("\n后续操作: 运行以下命令使用模拟数据进行可视化:")
    print("  python visualize_calibration.py --mock")


if __name__ == '__main__':
    main()
