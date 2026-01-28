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
    T_exhaust: float        # 排气温度 [K]
    T_exhaust_tc: float     # 涡轮前排温 [K]


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
        
        # 排气温度: 检查是否已经是K (>200则假设是K)
        if 'T_exhaust' in df.columns:
            if df['T_exhaust'].mean() < 400:
                # 原始是°C, 转换为K (排温通常在200-500°C)
                df['T_exhaust'] = df['T_exhaust'] + 273.15
            # 否则已经是K, 保持不变
        
        if 'T_exhaust_tc' in df.columns:
            if df['T_exhaust_tc'].mean() < 400:
                df['T_exhaust_tc'] = df['T_exhaust_tc'] + 273.15
        
        # 共轨压力: 原始单位为bar, 转换为Pa
        if 'p_rail' in df.columns:
            df['p_rail'] = df['p_rail'] * 1e5
        
        # P_comp数据校正：检测并修正可能的比例因子
        # 对于二冲程柴油机: Pcomp ≈ Pscav × (压缩比)^n, n≈1.35
        if 'p_scav' in df.columns and 'P_max' in df.columns and 'P_comp' in df.columns:
            # 典型压缩比13-15，多变指数1.3-1.4
            p_scav_bar = df['p_scav'] / 1e5
            compression_ratio_est = 14.0  # 初始估算
            n_poly = 1.35  # 多变指数
            df['P_comp_est'] = p_scav_bar * (compression_ratio_est ** n_poly)
            
            # 计算原始P_comp与理论估计值的比值
            ratio = df['P_comp_est'].mean() / df['P_comp'].mean()
            
            # 如果比值接近10，说明P_comp数据可能被除以了10（或类似比例因子）
            if 8.0 < ratio < 12.0:
                scale_factor = round(ratio)  # 取整得到比例因子
                print(f"检测到P_comp数据存在比例因子问题:")
                print(f"  原始P_comp均值: {df['P_comp'].mean():.2f} bar")
                print(f"  理论估计P_comp均值: {df['P_comp_est'].mean():.2f} bar")
                print(f"  比值: {ratio:.2f}, 应用比例因子: {scale_factor}")
                df['P_comp_original'] = df['P_comp']
                df['P_comp'] = df['P_comp'] * scale_factor
                print(f"  修正后P_comp均值: {df['P_comp'].mean():.2f} bar")
            elif df['P_comp'].mean() < df['P_comp_est'].mean() * 0.3:
                # 如果比例差异太大且不是整数倍，使用估算值
                print(f"Warning: Original P_comp ({df['P_comp'].mean():.1f} bar) seems abnormal.")
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
        original_len = len(df)
        
        # 物理边界过滤 - 调整为实际船机数据范围
        filters = [
            (df['rpm'] > 30) & (df['rpm'] < 120),  # 船机转速范围
            (df['P_max'] > 50) & (df['P_max'] < 250),  # Pmax范围 [bar]
            (df['P_comp'] > 1) & (df['P_comp'] < 200),  # Pcomp范围 [bar]
        ]
        
        # 添加扫气压力过滤 - 剔除异常低负荷工况
        if 'p_scav' in df.columns:
            p_scav_bar = df['p_scav'] / 1e5 if df['p_scav'].mean() > 1000 else df['p_scav']
            filters.append(p_scav_bar > 1.0)  # 扫气压力 > 1 bar
            
        mask = np.ones(len(df), dtype=bool)
        for f in filters:
            mask &= f
        
        filtered_df = df[mask]
        n_filtered = original_len - len(filtered_df)
        if n_filtered > 0:
            print(f"数据质量过滤: 剔除 {n_filtered} 个异常工况点 (包括p_scav < 1 bar的低负荷点)")
        
        return filtered_df
    
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
            # 分层抽样 - 使用 .copy() 避免 SettingWithCopyWarning
            steady_df = steady_df.copy()
            steady_df['rpm_bin'] = pd.cut(steady_df['rpm'], bins=n_points)
            # 使用 include_groups=False 避免 FutureWarning
            sampled = steady_df.groupby('rpm_bin', observed=True).apply(
                lambda x: x.sample(n=min(1, len(x))) if len(x) > 0 else x,
                include_groups=False
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


class VisualizationDataLoader(CalibrationDataLoader):
    """
    可视化数据加载器
    
    继承自 CalibrationDataLoader，增加数据段分析和模拟数据生成功能，
    专门用于数据预处理可视化。
    """
    
    # 可视化所需的列映射（与基类略有不同）
    VIZ_COLUMN_MAPPING = {
        'Time': 'time',
        'Main Engine RPM': 'rpm',
        'Fuel Command': 'fuel_command',
        'Scav. Air Press. Mean Value': 'p_scav',
        'Scavenge Air Temperature After Cooler 1': 'T_scav',
        'Main Fuel Rail Pressure': 'p_rail',
        'Cyl 1 KPI 5 - Pcomp': 'P_comp',
        'Cyl 1 KPI 4 - PMAx': 'P_max',
        'Cyl 1 KPI 14 - Pign': 'P_ign',
        'Exhaust gas temp cyl. 1': 'T_exh',
        'Exhaust Gas Temp. before TC #1': 'T_exh_turbo',
    }
    
    def __init__(self, csv_path: str = 'data/calibration_data.csv'):
        super().__init__(csv_path)
        self.segment_stats = None
    
    def analyze_segments(self, n_segments: int = 10) -> Optional[int]:
        """
        分析数据集的不同时间段特征，选择最具代表性的段
        
        Args:
            n_segments: 将数据分为多少段分析
            
        Returns:
            best_segment: 最佳数据段索引 (0-based)，如果失败返回 None
        """
        print(f"\n正在分析数据集的 {n_segments} 个时间段...")
        
        try:
            column_mapping = {
                'Time': 'time',
                'Main Engine RPM': 'rpm',
                'Cyl 1 KPI 4 - PMAx': 'P_max',
                'Cyl 1 KPI 5 - Pcomp': 'P_comp',
            }
            
            # 快速读取小样本估算总行数
            first_chunk = pd.read_csv(self.csv_path, nrows=1000)
            
            # 逐段分析
            segment_stats = []
            chunk_size = 10000
            
            for seg_idx in range(n_segments):
                skip_rows = seg_idx * chunk_size + 1  # +1 跳过表头
                
                try:
                    df_seg = pd.read_csv(self.csv_path, 
                                        skiprows=range(1, skip_rows),
                                        nrows=chunk_size)
                    
                    # 重命名列
                    available_cols = {k: v for k, v in column_mapping.items() if k in df_seg.columns}
                    if len(available_cols) < 3:
                        continue
                    
                    df_seg = df_seg[list(available_cols.keys())].rename(columns=available_cols)
                    
                    # 转换数值
                    for col in df_seg.columns:
                        if col != 'time':
                            df_seg[col] = pd.to_numeric(df_seg[col], errors='coerce')
                    
                    # 删除缺失值
                    df_seg = df_seg.dropna()
                    
                    if len(df_seg) < 100:
                        continue
                    
                    # 计算统计特征
                    stats = {
                        'segment': seg_idx,
                        'start_row': skip_rows - 1,
                        'end_row': skip_rows + chunk_size - 1,
                        'valid_samples': len(df_seg),
                        
                        # RPM变化特征
                        'rpm_mean': df_seg['rpm'].mean(),
                        'rpm_std': df_seg['rpm'].std(),
                        'rpm_range': df_seg['rpm'].max() - df_seg['rpm'].min(),
                        
                        # 异常值特征
                        'outlier_ratio': 0,
                        'rpm_outliers': 0,
                        'pmax_outliers': 0,
                        
                        # 稳态特征
                        'rpm_stability': 0,
                        'n_steady_regions': 0,
                        
                        # 综合评分
                        'diversity_score': 0,
                    }
                    
                    # 检测异常值
                    rpm_outliers = ((df_seg['rpm'] < 30) | (df_seg['rpm'] > 120)).sum()
                    pmax_outliers = ((df_seg['P_max'] < 50) | (df_seg['P_max'] > 250)).sum()
                    stats['rpm_outliers'] = rpm_outliers
                    stats['pmax_outliers'] = pmax_outliers
                    stats['outlier_ratio'] = (rpm_outliers + pmax_outliers) / (2 * len(df_seg))
                    
                    # 计算稳态段数量
                    df_seg['rpm_rolling_std'] = df_seg['rpm'].rolling(window=30, center=True).std()
                    steady_mask = df_seg['rpm_rolling_std'] < 1.0
                    
                    # 统计连续稳态段
                    steady_changes = steady_mask.astype(int).diff().fillna(0)
                    n_steady_regions = (steady_changes == 1).sum()
                    stats['n_steady_regions'] = n_steady_regions
                    stats['rpm_stability'] = steady_mask.sum() / len(df_seg)
                    
                    # 计算多样性评分（综合考虑异常值、稳态段、RPM变化）
                    # 异常值多 = 好展示清洗效果
                    # 稳态段多 = 好展示稳态筛选
                    # RPM变化大 = 工况多样
                    diversity_score = (
                        stats['outlier_ratio'] * 100 +      # 异常值权重
                        stats['n_steady_regions'] * 10 +     # 稳态段权重
                        stats['rpm_range'] / 10              # 变化范围权重
                    )
                    stats['diversity_score'] = diversity_score
                    
                    segment_stats.append(stats)
                    print(f"  段 {seg_idx} [{skip_rows:6d}-{skip_rows+chunk_size:6d}]: "
                          f"异常{stats['outlier_ratio']*100:5.1f}%, "
                          f"稳态段{n_steady_regions:2d}个, "
                          f"RPM范围{stats['rpm_range']:5.1f}, "
                          f"评分{diversity_score:6.1f}")
                    
                except Exception as e:
                    print(f"  段 {seg_idx}: 读取失败")
                    continue
            
            if len(segment_stats) == 0:
                print("  [WARNING] 未能分析任何数据段，使用默认前10%")
                self.segment_stats = None
                return 0
            
            # 选择第1段数据（用户指定）
            segment_stats_df = pd.DataFrame(segment_stats)
            best_idx = 0  # 强制选择第1段
            best_seg = segment_stats_df.iloc[best_idx]
            
            print(f"\n✅ 选择第1段数据 (用户指定):")
            print(f"   段索引: {best_seg['segment']}")
            print(f"   行范围: {best_seg['start_row']:,} - {best_seg['end_row']:,}")
            print(f"   异常值比例: {best_seg['outlier_ratio']*100:.1f}%")
            print(f"   稳态段数量: {best_seg['n_steady_regions']:.0f}")
            print(f"   RPM变化范围: {best_seg['rpm_range']:.1f}")
            print(f"   多样性评分: {best_seg['diversity_score']:.1f}")
            
            self.segment_stats = segment_stats_df
            return int(best_seg['segment'])
            
        except FileNotFoundError:
            print("  [WARNING] 未找到calibration_data.csv，将生成模拟数据")
            return None
        except Exception as e:
            print(f"  [WARNING] 分析出错: {e}，使用默认前10%")
            return 0
    
    def load_segment(self, segment_index: Optional[int] = None, use_full_data: bool = False) -> pd.DataFrame:
        """
        读取指定数据段，用于可视化
        
        Args:
            segment_index: 数据段索引 (None=自动选择, 0=前10%, 1=10%-20%, ...)
            use_full_data: 是否使用全部数据 (True=读取所有行, False=读取10,000行子集)
            
        Returns:
            df: 预处理后的DataFrame
        """
        # 如果使用全部数据
        if use_full_data:
            print(f"\n正在读取全部数据...")
            chunk_size = None
            skip_rows = None
        else:
            # 如果未指定段索引，自动分析选择
            if segment_index is None:
                segment_index = self.analyze_segments()
                if segment_index is None:
                    return self.get_synthetic_data()
            
            chunk_size = 10000
            skip_rows = segment_index * chunk_size + 1  # +1跳过表头
            
            print(f"\n正在读取数据段 {segment_index} (行 {skip_rows-1:,} - {skip_rows+chunk_size-1:,})...")
        
        try:
            # 读取数据
            if use_full_data:
                df = pd.read_csv(self.csv_path)
                print(f"  已读取 {len(df):,} 行数据 (全部数据,约 {df.memory_usage(deep=True).sum() / 1024**2:.0f} MB)")
            else:
                df = pd.read_csv(self.csv_path,
                               skiprows=range(1, skip_rows),
                               nrows=chunk_size)
                print(f"  已读取 {len(df):,} 行数据")
            
            # 重命名列
            available_cols = {k: v for k, v in self.VIZ_COLUMN_MAPPING.items() if k in df.columns}
            df = df[list(available_cols.keys())].rename(columns=available_cols)
            
            # 转换为数值类型
            for col in df.columns:
                if col != 'time':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 单位转换 - 参考data_loader.py
            if 'p_scav' in df.columns:
                df['p_scav'] = df['p_scav'] * 1e5  # bar → Pa
            
            if 'T_scav' in df.columns:
                if df['T_scav'].mean() < 200:
                    df['T_scav'] = df['T_scav'] + 273.15  # °C → K
            
            if 'p_rail' in df.columns:
                df['p_rail'] = df['p_rail'] * 1e5  # bar → Pa
            
            # 估算P_comp (如果需要) - 参考data_loader.py
            if 'P_comp' in df.columns and 'p_scav' in df.columns:
                p_scav_bar = df['p_scav'] / 1e5
                compression_ratio_est = 14.0
                n_poly = 1.35
                df['P_comp_est'] = p_scav_bar * (compression_ratio_est ** n_poly)
                
                # 检查P_comp是否合理
                if df['P_comp'].mean() < df['P_comp_est'].mean() * 0.3:
                    print("  检测到P_comp可能为KPI值，使用估算值替换...")
                    df['P_comp'] = df['P_comp_est']
            
            # 删除缺失值
            key_cols = ['rpm', 'P_max', 'P_comp']
            existing_key_cols = [c for c in key_cols if c in df.columns]
            df = df.dropna(subset=existing_key_cols)
            
            print(f"  预处理后剩余 {len(df):,} 行数据")
            
            self.processed_data = df
            return df
            
        except FileNotFoundError:
            print("  [WARNING] 未找到calibration_data.csv，将生成模拟数据...")
            return self.get_synthetic_data()
        except Exception as e:
            print(f"  [WARNING] 读取数据出错: {e}，将生成模拟数据...")
            return self.get_synthetic_data()
    
    def get_synthetic_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """生成模拟数据作为备选"""
        print(f"  生成 {n_samples} 个模拟数据点...")
        
        time = np.arange(n_samples)
        
        # 模拟3个稳态段 + 2个瞬态段
        rpm = np.zeros(n_samples)
        load = np.zeros(n_samples)
        P_max = np.zeros(n_samples)
        P_comp = np.zeros(n_samples)
        T_exh = np.zeros(n_samples)
        fuel_command = np.zeros(n_samples)
        
        # 稳态1: 75rpm, 50%负荷
        rpm[0:1500] = 75 + np.random.normal(0, 0.3, 1500)
        load[0:1500] = 0.50 + np.random.normal(0, 0.01, 1500)
        P_max[0:1500] = 120 + np.random.normal(0, 1.5, 1500)
        P_comp[0:1500] = 95 + np.random.normal(0, 1.2, 1500)
        T_exh[0:1500] = 300 + np.random.normal(0, 5, 1500)
        fuel_command[0:1500] = 50 + np.random.normal(0, 2, 1500)
        
        # 瞬态1: 加速
        t_trans1 = np.linspace(0, 1, 500)
        rpm[1500:2000] = 75 + (100-75) * t_trans1 + np.random.normal(0, 2.0, 500)
        load[1500:2000] = 0.50 + (0.75-0.50) * t_trans1 + np.random.normal(0, 0.03, 500)
        P_max[1500:2000] = 120 + (140-120) * t_trans1 + np.random.normal(0, 3.0, 500)
        P_comp[1500:2000] = 95 + (115-95) * t_trans1 + np.random.normal(0, 2.5, 500)
        T_exh[1500:2000] = 300 + (340-300) * t_trans1 + np.random.normal(0, 8, 500)
        fuel_command[1500:2000] = 50 + (75-50) * t_trans1 + np.random.normal(0, 3, 500)
        
        # 稳态2: 100rpm, 75%负荷
        rpm[2000:3500] = 100 + np.random.normal(0, 0.4, 1500)
        load[2000:3500] = 0.75 + np.random.normal(0, 0.015, 1500)
        P_max[2000:3500] = 140 + np.random.normal(0, 2.0, 1500)
        P_comp[2000:3500] = 115 + np.random.normal(0, 1.5, 1500)
        T_exh[2000:3500] = 340 + np.random.normal(0, 6, 1500)
        fuel_command[2000:3500] = 75 + np.random.normal(0, 2.5, 1500)
        
        # 瞬态2: 减速
        t_trans2 = np.linspace(0, 1, 500)
        rpm[3500:4000] = 100 - (100-60) * t_trans2 + np.random.normal(0, 2.5, 500)
        load[3500:4000] = 0.75 - (0.75-0.30) * t_trans2 + np.random.normal(0, 0.04, 500)
        P_max[3500:4000] = 140 - (140-100) * t_trans2 + np.random.normal(0, 3.5, 500)
        P_comp[3500:4000] = 115 - (115-80) * t_trans2 + np.random.normal(0, 3.0, 500)
        T_exh[3500:4000] = 340 - (340-280) * t_trans2 + np.random.normal(0, 10, 500)
        fuel_command[3500:4000] = 75 - (75-30) * t_trans2 + np.random.normal(0, 4, 500)
        
        # 稳态3: 60rpm, 30%负荷
        rpm[4000:5000] = 60 + np.random.normal(0, 0.25, 1000)
        load[4000:5000] = 0.30 + np.random.normal(0, 0.008, 1000)
        P_max[4000:5000] = 100 + np.random.normal(0, 1.2, 1000)
        P_comp[4000:5000] = 80 + np.random.normal(0, 1.0, 1000)
        T_exh[4000:5000] = 280 + np.random.normal(0, 4, 1000)
        fuel_command[4000:5000] = 30 + np.random.normal(0, 1.5, 1000)
        
        # 注入异常值 (5%)
        n_outliers = int(n_samples * 0.05)
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        
        for i in outlier_indices[:n_outliers//2]:
            P_max[i] = np.random.choice([200, 40])  # 极端值
            P_comp[i] = np.random.choice([160, 10])
        
        for i in outlier_indices[n_outliers//2:]:
            rpm[i] = rpm[i] + np.random.choice([-20, 20])  # 尖峰
        
        df = pd.DataFrame({
            'time': time,
            'rpm': rpm,
            'load': load,
            'P_max': P_max,
            'P_comp': P_comp,
            'T_exh': T_exh,
            'fuel_command': fuel_command,
            'p_scav': 2.0e5 + np.random.normal(0, 1e4, n_samples),  # Pa
            'T_scav': 310 + np.random.normal(0, 3, n_samples),  # K
        })
        
        self.processed_data = df
        return df
    
    def apply_outlier_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用异常值过滤 - 与 _filter_outliers 保持一致"""
        # 物理边界过滤
        filters = [
            (df['rpm'] > 30) & (df['rpm'] < 120),
            (df['P_max'] > 50) & (df['P_max'] < 250),
            (df['P_comp'] > 1) & (df['P_comp'] < 200),
        ]
        
        mask = np.ones(len(df), dtype=bool)
        for f in filters:
            mask &= f
        
        df_clean = df[mask].reset_index(drop=True)
        print(f"  异常值过滤: {len(df)} → {len(df_clean)} 行 (剔除 {len(df)-len(df_clean)} 行, {(len(df)-len(df_clean))/len(df)*100:.1f}%)")
        
        return df_clean
