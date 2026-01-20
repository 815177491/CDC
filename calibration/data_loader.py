"""
校准数据加载器
==============
读取和预处理CSV监测数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CalibrationPoint:
    """单个校准工况点"""
    time: float
    rpm: float              # 发动机转速 [rpm]
    fuel_command: float     # 燃油指令 [%或kg/s]
    p_scav: float           # 扫气压力 [Pa]
    T_scav: float           # 扫气温度 [K]
    p_rail: float           # 共轨压力 [Pa]
    P_comp: float           # 压缩压力 [bar]
    P_max: float            # 最大爆发压力 [bar]
    P_ign: float            # 点火压力 [bar]
    T_exhaust: float        # 排气温度 [°C]
    T_exhaust_tc: float     # 涡轮前排温 [°C]


class CalibrationDataLoader:
    """
    校准数据加载器
    
    从CSV文件读取监测数据并转换为校准工况点
    """
    
    # CSV列名映射
    COLUMN_MAPPING = {
        'Time': 'time',
        'Main Engine RPM': 'rpm',
        'Fuel Command': 'fuel_command',
        'Scav. Air Press. Mean Value': 'p_scav',
        'Scavenge Air Temperature After Cooler 1': 'T_scav',
        'Main Fuel Rail Pressure': 'p_rail',
        'Cyl 1 KPI 5 - Pcomp': 'P_comp',
        'Cyl 1 KPI 4 - PMAx': 'P_max',
        'Cyl 1 KPI 14 - Pign': 'P_ign',
        'Exhaust gas temp cyl. 1': 'T_exhaust',
        'Exhaust Gas Temp. before TC #1': 'T_exhaust_tc'
    }
    
    def __init__(self, csv_path: str):
        """
        初始化数据加载器
        
        Args:
            csv_path: CSV文件路径
        """
        self.csv_path = Path(csv_path)
        self.raw_data: pd.DataFrame = None
        self.processed_data: pd.DataFrame = None
        self.calibration_points: List[CalibrationPoint] = []
    
    def load_data(self, nrows: int = None) -> pd.DataFrame:
        """
        加载CSV数据
        
        Args:
            nrows: 读取的行数 (用于大文件调试)
            
        Returns:
            raw_data: 原始数据DataFrame
        """
        self.raw_data = pd.read_csv(
            self.csv_path,
            nrows=nrows,
            low_memory=False
        )
        return self.raw_data
    
    def preprocess(self) -> pd.DataFrame:
        """
        数据预处理
        
        - 重命名列
        - 单位转换
        - 异常值处理
        - 提取稳态工况
        
        Returns:
            processed_data: 处理后的数据
        """
        if self.raw_data is None:
            self.load_data()
        
        df = self.raw_data.copy()
        
        # 重命名列 (仅保留需要的列)
        available_cols = {k: v for k, v in self.COLUMN_MAPPING.items() 
                          if k in df.columns}
        df = df[list(available_cols.keys())].rename(columns=available_cols)
        
        # 转换为数值类型 (跳过time列)
        for col in df.columns:
            if col != 'time':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 单位转换
        # 扫气压力: 原始单位为bar, 转换为Pa
        if 'p_scav' in df.columns:
            df['p_scav'] = df['p_scav'] * 1e5
        
        # 扫气温度: 检查是否已经是K (>200则假设是K)
        if 'T_scav' in df.columns:
            if df['T_scav'].mean() < 200:
                # 原始是°C, 转换为K
                df['T_scav'] = df['T_scav'] + 273.15
            # 否则已经是K, 保持不变
        
        # 共轨压力: 原始单位为bar, 转换为Pa
        if 'p_rail' in df.columns:
            df['p_rail'] = df['p_rail'] * 1e5
        
        # 估算压缩终点压力 Pcomp_est (如果原始Pcomp数据不符合物理意义)
        # 对于二冲程柴油机: Pcomp ≈ Pscav × (压缩比)^n, n≈1.35
        # 使用 Pmax / Pscav 比值来反推合理的 Pcomp
        if 'p_scav' in df.columns and 'P_max' in df.columns:
            # 典型压缩比13-15，多变指数1.3-1.4
            # Pcomp 通常是 Pmax 的 0.6-0.8 倍 (取决于燃烧)
            # 使用经验公式估算
            p_scav_bar = df['p_scav'] / 1e5
            compression_ratio_est = 14.0  # 初始估算
            n_poly = 1.35  # 多变指数
            df['P_comp_est'] = p_scav_bar * (compression_ratio_est ** n_poly)
            
            # 如果原始 P_comp 数据明显偏离物理意义，使用估算值
            if df['P_comp'].mean() < df['P_comp_est'].mean() * 0.3:
                print(f"Warning: Original P_comp ({df['P_comp'].mean():.1f} bar) seems to be a KPI, not actual pressure.")
                print(f"Using estimated P_comp ({df['P_comp_est'].mean():.1f} bar) based on polytropic compression.")
                df['P_comp_original'] = df['P_comp']
                df['P_comp'] = df['P_comp_est']
        
        # 删除关键列的NaN行 (不包括time)
        key_cols = ['rpm', 'P_max', 'P_comp', 'p_scav', 'T_scav']
        existing_key_cols = [c for c in key_cols if c in df.columns]
        df = df.dropna(subset=existing_key_cols)
        
        # 异常值过滤
        df = self._filter_outliers(df)
        
        self.processed_data = df
        return df
    
    def _filter_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤异常值
        
        使用IQR方法或物理边界过滤
        """
        # 物理边界过滤 - 调整为实际船机数据范围
        filters = [
            (df['rpm'] > 30) & (df['rpm'] < 120),  # 船机转速范围
            (df['P_max'] > 50) & (df['P_max'] < 250),  # Pmax范围 [bar]
            (df['P_comp'] > 1) & (df['P_comp'] < 200),  # Pcomp范围 [bar] - 放宽下限
        ]
        
        mask = np.ones(len(df), dtype=bool)
        for f in filters:
            mask &= f
        
        return df[mask]
    
    def extract_steady_state_points(self, 
                                     window_size: int = 60,
                                     rpm_tolerance: float = 1.0,
                                     n_points: int = 10) -> List[CalibrationPoint]:
        """
        提取稳态工况点
        
        通过检测RPM变化率来识别稳态区间
        
        Args:
            window_size: 滑动窗口大小 (秒)
            rpm_tolerance: RPM稳定性阈值 [rpm]
            n_points: 提取的工况点数量
            
        Returns:
            calibration_points: 校准工况点列表
        """
        if self.processed_data is None:
            self.preprocess()
        
        df = self.processed_data
        
        # 计算RPM滚动标准差
        df['rpm_std'] = df['rpm'].rolling(window=window_size, center=True).std()
        
        # 识别稳态区间 (RPM变化小于阈值)
        steady_mask = df['rpm_std'] < rpm_tolerance
        steady_df = df[steady_mask]
        
        if len(steady_df) == 0:
            # 如果没有严格稳态,放宽条件
            steady_df = df.nsmallest(n_points * 10, 'rpm_std')
        
        # 按RPM分层抽样, 覆盖不同负荷
        rpm_range = steady_df['rpm'].max() - steady_df['rpm'].min()
        if rpm_range > 10:
            # 分层抽样
            steady_df['rpm_bin'] = pd.cut(steady_df['rpm'], bins=n_points)
            sampled = steady_df.groupby('rpm_bin', observed=True).apply(
                lambda x: x.sample(n=min(1, len(x))) if len(x) > 0 else x
            ).reset_index(drop=True)
        else:
            # 均匀抽样
            sampled = steady_df.sample(n=min(n_points, len(steady_df)))
        
        # 转换为CalibrationPoint对象
        self.calibration_points = []
        for _, row in sampled.iterrows():
            point = CalibrationPoint(
                time=row.get('time', 0),
                rpm=row['rpm'],
                fuel_command=row.get('fuel_command', 0),
                p_scav=row['p_scav'],
                T_scav=row['T_scav'],
                p_rail=row.get('p_rail', 0),
                P_comp=row['P_comp'],
                P_max=row['P_max'],
                P_ign=row.get('P_ign', 0),
                T_exhaust=row.get('T_exhaust', 0),
                T_exhaust_tc=row.get('T_exhaust_tc', 0)
            )
            self.calibration_points.append(point)
        
        return self.calibration_points
    
    def get_operating_range(self) -> Dict:
        """
        获取数据覆盖的运行范围
        
        Returns:
            ranges: 各变量的范围统计
        """
        if self.processed_data is None:
            self.preprocess()
        
        df = self.processed_data
        
        ranges = {
            'rpm': (df['rpm'].min(), df['rpm'].max(), df['rpm'].mean()),
            'P_max': (df['P_max'].min(), df['P_max'].max(), df['P_max'].mean()),
            'P_comp': (df['P_comp'].min(), df['P_comp'].max(), df['P_comp'].mean()),
        }
        
        if 'T_exhaust' in df.columns:
            ranges['T_exhaust'] = (df['T_exhaust'].min(), df['T_exhaust'].max(), 
                                   df['T_exhaust'].mean())
        
        return ranges
    
    def convert_fuel_command_to_mass(self, 
                                      fuel_command: float,
                                      rpm: float,
                                      max_fuel_rate: float = 0.15) -> float:
        """
        将燃油指令转换为循环喷油质量
        
        Args:
            fuel_command: 燃油指令 [%]
            rpm: 发动机转速 [rpm]
            max_fuel_rate: 最大燃油流率 [kg/s] per cylinder
            
        Returns:
            fuel_mass: 循环喷油量 [kg/cycle]
        """
        # 假设Fuel Command为0-100%
        fuel_rate = (fuel_command / 100.0) * max_fuel_rate
        
        # 二冲程每转做功一次
        cycles_per_second = rpm / 60.0
        
        if cycles_per_second > 0:
            fuel_mass = fuel_rate / cycles_per_second
        else:
            fuel_mass = 0.0
        
        return fuel_mass
