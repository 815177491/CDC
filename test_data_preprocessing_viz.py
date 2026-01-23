#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据清洗与特征提取可视化 - 测试版本
=====================================
生成数据预处理流程的可视化图表

包括:
1. 稳态工况的智能筛选 (优先级1)
2. 代表性工况点的提取 (优先级2)
3. 数据的清洗与异常值剔除 (优先级3)
4. 数据的标准化处理与参数关联 (优先级4)

Author: CDC Project
Date: 2026-01-23
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 中文字体设置 - 与visualize_agents.py保持一致
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150

# 输出目录
OUTPUT_DIR = 'visualization_output'
DATA_DIR = 'visualization_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 配色方案 - 与visualize_agents.py保持一致
COLORS = {
    'primary': '#2E86AB',      # 蓝色
    'secondary': '#A23B72',    # 紫红色
    'success': '#28A745',      # 绿色
    'warning': '#FFC107',      # 黄色
    'danger': '#DC3545',       # 红色
    'info': '#17A2B8',         # 青色
    'dark': '#343A40',         # 深灰
    'light': '#F8F9FA',        # 浅灰
}


class DataPreprocessingVisualizer:
    """数据预处理可视化器"""
    
    def __init__(self):
        self.colors = COLORS
        self.data = None
        self.data_clean = None
        
    def load_real_data(self, sample_rate=0.1):
        """
        读取真实校准数据的前10%
        
        Args:
            sample_rate: 采样率 (0.1 = 10%)
        """
        print(f"正在读取校准数据 (采样率: {sample_rate*100:.0f}%)...")
        
        try:
            # 列名映射 - 参考data_loader.py
            column_mapping = {
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
            
            # 分块读取
            chunks = []
            total_rows = 0
            target_rows = None
            
            for chunk in pd.read_csv('calibration_data.csv', chunksize=10000):
                if target_rows is None:
                    # 第一块：估算总行数
                    print(f"  读取数据块 {len(chunks)+1}...")
                chunks.append(chunk)
                total_rows += len(chunk)
                
                # 读取到目标行数后停止
                if target_rows is None and len(chunks) == 1:
                    # 读取前10%
                    target_rows = int(len(chunk) / sample_rate)
                    print(f"  预计总行数: ~{target_rows:,}, 目标行数: {int(target_rows * sample_rate):,}")
                
                if total_rows >= target_rows * sample_rate:
                    break
            
            # 合并数据块
            df = pd.concat(chunks, ignore_index=True)
            print(f"  已读取 {len(df):,} 行数据")
            
            # 重命名列
            available_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
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
            
            self.data = df
            return df
            
        except FileNotFoundError:
            print("  ⚠ 未找到calibration_data.csv，将生成模拟数据...")
            return self._generate_synthetic_data()
        except Exception as e:
            print(f"  ⚠ 读取数据出错: {e}，将生成模拟数据...")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, n_samples=5000):
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
        
        self.data = df
        return df
    
    def apply_outlier_filter(self):
        """应用异常值过滤 - 参考data_loader.py的_filter_outliers"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        df = self.data.copy()
        
        # 物理边界过滤 - 与data_loader.py保持一致
        filters = [
            (df['rpm'] > 30) & (df['rpm'] < 120),
            (df['P_max'] > 50) & (df['P_max'] < 250),
            (df['P_comp'] > 1) & (df['P_comp'] < 200),
        ]
        
        mask = np.ones(len(df), dtype=bool)
        for f in filters:
            mask &= f
        
        self.data_clean = df[mask].reset_index(drop=True)
        print(f"  异常值过滤: {len(df)} → {len(self.data_clean)} 行 (剔除 {len(df)-len(self.data_clean)} 行, {(len(df)-len(self.data_clean))/len(df)*100:.1f}%)")
        
        return self.data_clean
    
    def plot_steady_state_selection(self):
        """可视化1: 稳态工况的智能筛选"""
        print("\n[1/4] 生成稳态工况智能筛选可视化...")
        
        if self.data is None:
            self.load_real_data()
        
        df = self.data.copy()
        
        # 参考data_loader.py的extract_steady_state_points
        window_size = 60
        rpm_tolerance = 1.0
        
        # 计算RPM滚动标准差
        df['rpm_std'] = df['rpm'].rolling(window=window_size, center=True).std()
        steady_mask = df['rpm_std'] < rpm_tolerance
        
        # 统计稳态段
        steady_regions = []
        in_steady = False
        start_idx = 0
        
        for i in range(len(steady_mask)):
            if pd.notna(steady_mask.iloc[i]):
                if steady_mask.iloc[i] and not in_steady:
                    start_idx = i
                    in_steady = True
                elif not steady_mask.iloc[i] and in_steady:
                    if i - start_idx > 50:  # 至少50个点
                        steady_regions.append((start_idx, i))
                    in_steady = False
        
        if in_steady and len(df) - start_idx > 50:
            steady_regions.append((start_idx, len(df)))
        
        print(f"  检测到 {len(steady_regions)} 个稳态段")
        
        # 创建图形
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        # ========== (a) RPM变化率检测 ==========
        ax1 = fig.add_subplot(gs[0, :])
        
        ax1_twin = ax1.twinx()
        
        # 使用索引作为时间轴（避免time列可能的字符串问题）
        time_index = np.arange(len(df))
        
        # RPM曲线
        ax1.plot(time_index, df['rpm'], color=self.colors['primary'], 
                linewidth=1.2, alpha=0.7, label='RPM原始数据')
        
        # RPM标准差曲线
        ax1_twin.plot(time_index, df['rpm_std'], color=self.colors['danger'],
                     linewidth=2, linestyle='--', label='RPM标准差 (60s窗口)')
        ax1_twin.axhline(rpm_tolerance, color=self.colors['warning'],
                        linestyle=':', linewidth=2, label=f'稳态阈值 ({rpm_tolerance} rpm)')
        
        # 标记稳态段
        for idx, (start, end) in enumerate(steady_regions):
            ax1.axvspan(start, end,
                       alpha=0.2, color=self.colors['success'],
                       label='稳态段' if idx == 0 else '')
        
        ax1.set_xlabel('时间 (采样点)', fontsize=11)
        ax1.set_ylabel('RPM', fontsize=11, color=self.colors['primary'])
        ax1_twin.set_ylabel('RPM标准差', fontsize=11, color=self.colors['danger'])
        ax1.set_title('(a) 基于RPM变化率的稳态段识别', fontsize=12, fontweight='bold')
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # ========== (b) 稳态段统计 ==========
        ax2 = fig.add_subplot(gs[1, 0])
        
        total_steady_time = sum([end - start for start, end in steady_regions])
        total_time = len(df)
        transient_time = total_time - total_steady_time
        
        categories = ['总时长', '稳态时长', '瞬态时长']
        durations = [total_time, total_steady_time, transient_time]
        colors_bar = [self.colors['dark'], self.colors['success'], self.colors['warning']]
        
        bars = ax2.barh(categories, durations, color=colors_bar, alpha=0.8)
        for bar, dur in zip(bars, durations):
            ax2.text(dur + total_time*0.02, bar.get_y() + bar.get_height()/2,
                    f'{dur}s ({dur/total_time*100:.1f}%)',
                    va='center', fontsize=10, fontweight='bold')
        
        ax2.set_xlabel('时长 (s)', fontsize=10)
        ax2.set_title(f'(b) 稳态段统计 (共{len(steady_regions)}段)',
                     fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_xlim(0, total_time * 1.25)
        
        # ========== (c) 稳态质量评分 ==========
        ax3 = fig.add_subplot(gs[1, 1])
        
        quality_scores = []
        for start, end in steady_regions:
            segment_rpm_std = df['rpm_std'].iloc[start:end].mean()
            duration = end - start
            
            # 评分: 标准差越小越好, 持续时间越长越好
            score_std = 100 * (1 - min(1, segment_rpm_std / 2.0))
            score_duration = min(100, duration / 3)
            overall_score = score_std * 0.6 + score_duration * 0.4
            
            quality_scores.append(overall_score)
        
        if len(quality_scores) > 0:
            colors_quality = [self.colors['success'] if s >= 70 else 
                             self.colors['warning'] if s >= 50 else
                             self.colors['danger'] for s in quality_scores]
            
            bars = ax3.bar(range(1, len(quality_scores)+1), quality_scores,
                          color=colors_quality, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            for i, (bar, score) in enumerate(zip(bars, quality_scores)):
                ax3.text(i+1, score + 3, f'{score:.0f}',
                        ha='center', fontsize=9, fontweight='bold')
            
            ax3.axhline(70, color=self.colors['success'], linestyle='--',
                       linewidth=1.5, alpha=0.7, label='优秀 (≥70分)')
            ax3.axhline(50, color=self.colors['warning'], linestyle='--',
                       linewidth=1.5, alpha=0.7, label='良好 (≥50分)')
        
        ax3.set_xlabel('稳态段编号', fontsize=10)
        ax3.set_ylabel('质量评分', fontsize=10)
        ax3.set_title('(c) 稳态段质量评估', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim([0, 110])
        
        # ========== (d) 稳态工况点聚类 ==========
        ax4 = fig.add_subplot(gs[2, 0])
        
        steady_df = df[steady_mask & pd.notna(steady_mask)]
        
        if len(steady_df) > 0:
            ax4.scatter(steady_df['rpm'], steady_df['P_max'], 
                       s=15, alpha=0.4, color=self.colors['success'], label='稳态点')
        
        transient_df = df[~steady_mask & pd.notna(steady_mask)]
        if len(transient_df) > 0:
            ax4.scatter(transient_df['rpm'], transient_df['P_max'],
                       s=10, alpha=0.2, color=self.colors['dark'], label='瞬态点')
        
        ax4.set_xlabel('RPM', fontsize=10)
        ax4.set_ylabel('P_max (bar)', fontsize=10)
        ax4.set_title('(d) 稳态/瞬态工况点分布', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # ========== (e) 持续时间分布 ==========
        ax5 = fig.add_subplot(gs[2, 1])
        
        durations_list = [end - start for start, end in steady_regions]
        
        if len(durations_list) > 0:
            ax5.hist(durations_list, bins=min(15, len(durations_list)),
                    alpha=0.7, color=self.colors['info'], edgecolor='black', linewidth=1)
            
            avg_duration = np.mean(durations_list)
            ax5.axvline(avg_duration, color=self.colors['success'],
                       linestyle='-', linewidth=2, label=f'平均: {avg_duration:.0f}s')
            ax5.axvline(50, color=self.colors['warning'], linestyle='--',
                       linewidth=2, label='最小阈值: 50s')
        
        ax5.set_xlabel('持续时间 (s)', fontsize=10)
        ax5.set_ylabel('稳态段数量', fontsize=10)
        ax5.set_title('(e) 稳态持续时间分布', fontsize=11, fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        plt.suptitle('稳态工况的智能筛选流程', fontsize=14, fontweight='bold', y=0.995)
        
        save_path = os.path.join(OUTPUT_DIR, 'steady_state_selection.png')
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=150)
        plt.close()
        print(f"  ✓ 已保存: {save_path}")
        
        # 保存CSV
        self._save_steady_state_csv(df, steady_mask, quality_scores, steady_regions)
        
        return fig
    
    def _save_steady_state_csv(self, df, steady_mask, quality_scores, steady_regions):
        """保存稳态筛选CSV数据"""
        # 时序数据
        steady_data = df[['time', 'rpm', 'P_max', 'rpm_std']].copy()
        steady_data['is_steady'] = steady_mask.astype(int)
        steady_data.to_csv(os.path.join(DATA_DIR, 'steady_state_timeseries.csv'),
                          index=False, encoding='utf-8-sig')
        
        # 质量评分
        if len(quality_scores) > 0:
            # 使用索引作为时间中心（避免time列类型问题）
            time_centers = [(s + e) / 2 for s, e in steady_regions]
            quality_data = pd.DataFrame({
                'segment_id': range(1, len(quality_scores)+1),
                'time_center': time_centers,
                'quality_score': quality_scores
            })
            quality_data.to_csv(os.path.join(DATA_DIR, 'steady_state_quality.csv'),
                               index=False, encoding='utf-8-sig')
        
        print(f"  ✓ CSV: steady_state_timeseries.csv, steady_state_quality.csv")
    
    def plot_representative_points(self):
        """可视化2: 代表性工况点的提取"""
        print("\n[2/4] 生成代表性工况点提取可视化...")
        
        if self.data is None:
            self.load_real_data()
        
        df = self.data.copy()
        
        # 提取稳态点 - 参考extract_steady_state_points
        window_size = 60
        rpm_tolerance = 1.0
        n_points = 10
        
        df['rpm_std'] = df['rpm'].rolling(window=window_size, center=True).std()
        steady_mask = df['rpm_std'] < rpm_tolerance
        steady_df = df[steady_mask & pd.notna(steady_mask)]
        
        if len(steady_df) == 0:
            steady_df = df.nsmallest(n_points * 10, 'rpm_std')
        
        # 分层抽样
        rpm_range = steady_df['rpm'].max() - steady_df['rpm'].min()
        
        if rpm_range > 10:
            steady_df = steady_df.copy()
            steady_df['rpm_bin'] = pd.cut(steady_df['rpm'], bins=n_points)
            sampled = steady_df.groupby('rpm_bin', observed=True).apply(
                lambda x: x.sample(n=min(1, len(x))) if len(x) > 0 else x
            ).reset_index(drop=True)
        else:
            sampled = steady_df.sample(n=min(n_points, len(steady_df)))
        
        print(f"  提取 {len(sampled)} 个代表性工况点")
        
        # 创建图形
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.35)
        
        # ========== (a) 工况图谱覆盖 ==========
        ax1 = fig.add_subplot(gs[0, 0])
        
        scatter = ax1.scatter(df['rpm'], df['P_max'], c=df['T_exh'], s=20,
                             cmap='coolwarm', alpha=0.4, edgecolors='none')
        
        # 代表点
        ax1.scatter(sampled['rpm'], sampled['P_max'], s=300, marker='*',
                   color='yellow', edgecolors='red', linewidths=2,
                   label='代表工况点', zorder=10)
        
        for i, row in sampled.iterrows():
            ax1.annotate(f'P{i+1}', xy=(row['rpm'], row['P_max']),
                        xytext=(3, 3), textcoords='offset points',
                        fontsize=8, fontweight='bold', color='darkred')
        
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('T_exh (K)', fontsize=9)
        
        ax1.set_xlabel('RPM', fontsize=10)
        ax1.set_ylabel('P_max (bar)', fontsize=10)
        ax1.set_title('(a) 全工况图谱与代表点', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # ========== (b) 负荷区间分箱 ==========
        ax2 = fig.add_subplot(gs[0, 1])
        
        # 为所有稳态点分配bin
        if 'rpm_bin' in steady_df.columns:
            # 为可视化创建数值bin_id
            bin_mapping = {bin_val: i for i, bin_val in enumerate(steady_df['rpm_bin'].unique())}
            bin_ids = steady_df['rpm_bin'].map(bin_mapping)
            
            scatter2 = ax2.scatter(steady_df['rpm'], steady_df['P_max'],
                                  c=bin_ids, s=25, cmap='tab10', alpha=0.5)
            
            # 代表点
            ax2.scatter(sampled['rpm'], sampled['P_max'], s=200, marker='D',
                       c='red', edgecolors='black', linewidths=2,
                       label='抽样点', zorder=10)
            
            plt.colorbar(scatter2, ax=ax2, label='RPM分箱')
        
        ax2.set_xlabel('RPM', fontsize=10)
        ax2.set_ylabel('P_max (bar)', fontsize=10)
        ax2.set_title(f'(b) 分层抽样 ({n_points}个分箱)', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # ========== (c) 代表性评分 ==========
        ax3 = fig.add_subplot(gs[0, 2])
        
        # 计算每个代表点的评分
        rep_scores = []
        for _, point in sampled.iterrows():
            # 计算该点周围的样本密度
            distance = np.sqrt((steady_df['rpm'] - point['rpm'])**2 + 
                              (steady_df['P_max'] - point['P_max'])**2)
            nearby_count = (distance < 5).sum()  # 5 rpm/bar半径内
            
            # 评分: 样本密度 + 稳定性
            score_density = min(100, nearby_count / 5)
            score_stability = 100 * (1 - min(1, point['rpm_std'] / 2.0))
            overall_score = score_density * 0.5 + score_stability * 0.5
            
            rep_scores.append(overall_score)
        
        colors_rep = [self.colors['success'] if s >= 60 else
                     self.colors['warning'] if s >= 40 else
                     self.colors['danger'] for s in rep_scores]
        
        bars = ax3.barh([f'P{i+1}' for i in range(len(rep_scores))],
                       rep_scores, color=colors_rep, alpha=0.8)
        
        for bar, score in zip(bars, rep_scores):
            ax3.text(score + 2, bar.get_y() + bar.get_height()/2,
                    f'{score:.0f}',
                    va='center', fontsize=9, fontweight='bold')
        
        ax3.set_xlabel('代表性评分', fontsize=10)
        ax3.set_title('(c) 代表点质量评估', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.set_xlim([0, 110])
        
        # ========== (d) 参数对比雷达图 ==========
        ax4 = fig.add_subplot(gs[1, 0], projection='polar')
        
        # 选择4-5个参数
        params = ['rpm', 'P_max', 'P_comp', 'T_exh']
        available_params = [p for p in params if p in sampled.columns]
        
        if len(available_params) >= 3:
            # 归一化
            sampled_norm = sampled[available_params].copy()
            for col in available_params:
                min_val = sampled_norm[col].min()
                max_val = sampled_norm[col].max()
                if max_val > min_val:
                    sampled_norm[col] = (sampled_norm[col] - min_val) / (max_val - min_val)
            
            # 绘制前3个代表点
            angles = np.linspace(0, 2*np.pi, len(available_params), endpoint=False).tolist()
            angles += angles[:1]
            
            for i in range(min(3, len(sampled_norm))):
                values = sampled_norm.iloc[i].tolist()
                values += values[:1]
                ax4.plot(angles, values, 'o-', linewidth=2, label=f'P{i+1}')
                ax4.fill(angles, values, alpha=0.15)
            
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(available_params, fontsize=9)
            ax4.set_ylim([0, 1])
            ax4.set_title('(d) 代表点参数对比', fontsize=11, fontweight='bold', pad=20)
            ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
            ax4.grid(True)
        
        # ========== (e) 工况覆盖均匀性 ==========
        ax5 = fig.add_subplot(gs[1, 1])
        
        # 绘制Voronoi图的简化版本 - 显示覆盖范围
        ax5.scatter(steady_df['rpm'], steady_df['P_max'], s=10, alpha=0.2, color='gray')
        
        # 绘制代表点及其影响范围
        for i, point in sampled.iterrows():
            circle = plt.Circle((point['rpm'], point['P_max']), radius=5,
                               color=self.colors['info'], alpha=0.2, linewidth=2,
                               edgecolor=self.colors['primary'], linestyle='--')
            ax5.add_patch(circle)
        
        ax5.scatter(sampled['rpm'], sampled['P_max'], s=200, marker='D',
                   c='red', edgecolors='black', linewidths=2, zorder=10)
        
        for i, row in sampled.iterrows():
            ax5.annotate(f'P{i+1}', xy=(row['rpm'], row['P_max']),
                        xytext=(0, -15), textcoords='offset points',
                        fontsize=8, ha='center', fontweight='bold')
        
        ax5.set_xlabel('RPM', fontsize=10)
        ax5.set_ylabel('P_max (bar)', fontsize=10)
        ax5.set_title('(e) 代表点覆盖区域', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # ========== (f) 参数统计表 ==========
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        # 创建表格
        table_data = []
        display_params = ['rpm', 'P_max', 'P_comp', 'T_exh']
        for i, row in sampled.head(8).iterrows():  # 最多显示8个点
            row_data = [f'P{i+1}']
            for param in display_params:
                if param in row:
                    row_data.append(f'{row[param]:.1f}')
                else:
                    row_data.append('N/A')
            table_data.append(row_data)
        
        table = ax6.table(cellText=table_data,
                         colLabels=['点', 'RPM', 'Pmax', 'Pcomp', 'Texh'],
                         cellLoc='center', loc='center',
                         colWidths=[0.12, 0.22, 0.22, 0.22, 0.22])
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2.2)
        
        # 表头样式
        for i in range(5):
            table[(0, i)].set_facecolor(self.colors['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 交替行颜色
        for i in range(1, len(table_data) + 1):
            if i % 2 == 0:
                for j in range(5):
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax6.set_title('(f) 代表工况点参数表', fontsize=11, fontweight='bold', pad=20)
        
        plt.suptitle('代表性工况点的提取与验证', fontsize=14, fontweight='bold', y=0.995)
        
        save_path = os.path.join(OUTPUT_DIR, 'representative_points.png')
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=150)
        plt.close()
        print(f"  ✓ 已保存: {save_path}")
        
        # 保存CSV
        self._save_representative_csv(df, steady_df, sampled)
        
        return fig
    
    def _save_representative_csv(self, df_all, steady_df, sampled):
        """保存代表点CSV"""
        # 所有工况点
        all_points = df_all[['rpm', 'P_max', 'P_comp', 'T_exh']].copy()
        all_points.to_csv(os.path.join(DATA_DIR, 'operating_points_all.csv'),
                         index=False, encoding='utf-8-sig')
        
        # 代表点
        rep_points = sampled[['rpm', 'P_max', 'P_comp', 'T_exh']].copy()
        rep_points.insert(0, 'point_id', [f'P{i+1}' for i in range(len(rep_points))])
        rep_points.to_csv(os.path.join(DATA_DIR, 'representative_points.csv'),
                         index=False, encoding='utf-8-sig')
        
        print(f"  ✓ CSV: operating_points_all.csv, representative_points.csv")
    
    def plot_data_cleaning(self):
        """可视化3: 数据的清洗与异常值剔除"""
        print("\n[3/4] 生成数据清洗与异常值剔除可视化...")
        
        if self.data is None:
            self.load_real_data()
        
        df = self.data.copy()
        
        # 应用物理边界过滤
        filters = [
            (df['rpm'] > 30) & (df['rpm'] < 120),
            (df['P_max'] > 50) & (df['P_max'] < 250),
            (df['P_comp'] > 1) & (df['P_comp'] < 200),
        ]
        
        mask_combined = np.ones(len(df), dtype=bool)
        outlier_masks = {}
        
        for i, f in enumerate(filters):
            outlier_masks[i] = ~f
            mask_combined &= f
        
        df_clean = df[mask_combined]
        n_outliers = (~mask_combined).sum()
        
        print(f"  剔除异常值: {n_outliers} 个 ({n_outliers/len(df)*100:.1f}%)")
        
        # 创建图形
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # ========== (a) 原始数据 vs 清洗后数据 ==========
        ax1 = fig.add_subplot(gs[0, :])
        
        # 正常点和异常点
        normal_mask = mask_combined
        outlier_mask = ~mask_combined
        
        ax1.scatter(np.arange(len(df))[normal_mask], df['P_max'][normal_mask],
                   s=10, alpha=0.5, color=self.colors['success'], label='正常数据')
        ax1.scatter(np.arange(len(df))[outlier_mask], df['P_max'][outlier_mask],
                   s=30, alpha=0.8, color=self.colors['danger'], marker='x',
                   linewidths=1.5, label='异常值')
        
        # 物理边界线
        ax1.axhline(250, color=self.colors['danger'], linestyle='--',
                   linewidth=2, alpha=0.7, label='Pmax上界 (250 bar)')
        ax1.axhline(50, color=self.colors['danger'], linestyle='--',
                   linewidth=2, alpha=0.7, label='Pmax下界 (50 bar)')
        
        mean_pmax = df_clean['P_max'].mean()
        ax1.axhline(mean_pmax, color=self.colors['primary'],
                   linestyle=':', linewidth=1.5, alpha=0.5, label=f'清洗后均值 ({mean_pmax:.1f})')
        
        ax1.set_xlabel('采样点', fontsize=11)
        ax1.set_ylabel('P_max (bar)', fontsize=11)
        ax1.set_title('(a) 物理边界异常值检测 - P_max', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9, ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # 统计信息
        ax1.text(0.02, 0.98, 
                f'总样本: {len(df)}\n异常值: {n_outliers} ({n_outliers/len(df)*100:.1f}%)\n保留率: {len(df_clean)/len(df)*100:.1f}%',
                transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
        
        # ========== (b) RPM异常检测 ==========
        ax2 = fig.add_subplot(gs[1, 0])
        
        rpm_outliers = outlier_masks[0]
        ax2.hist([df[~rpm_outliers]['rpm'], df[rpm_outliers]['rpm']],
                bins=30, label=['正常', '异常'], color=[self.colors['success'], self.colors['danger']],
                alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax2.axvline(30, color=self.colors['warning'], linestyle='--',
                   linewidth=2, label='下界 (30 rpm)')
        ax2.axvline(120, color=self.colors['warning'], linestyle='--',
                   linewidth=2, label='上界 (120 rpm)')
        
        ax2.set_xlabel('RPM', fontsize=10)
        ax2.set_ylabel('频次', fontsize=10)
        ax2.set_title(f'(b) RPM异常检测 ({rpm_outliers.sum()}个异常)', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # ========== (c) Pmax异常检测 ==========
        ax3 = fig.add_subplot(gs[1, 1])
        
        pmax_outliers = outlier_masks[1]
        
        # 箱线图
        bp = ax3.boxplot([df_clean['P_max']], labels=['清洗后'],
                         patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(self.colors['success'])
        bp['boxes'][0].set_alpha(0.7)
        
        # 叠加原始数据点
        ax3.scatter(np.ones(len(df))*1.2, df['P_max'], s=10, alpha=0.3,
                   color=self.colors['dark'], label='原始数据')
        
        # 边界线
        ax3.axhline(250, color=self.colors['danger'], linestyle='--', linewidth=2)
        ax3.axhline(50, color=self.colors['danger'], linestyle='--', linewidth=2)
        
        ax3.set_ylabel('P_max (bar)', fontsize=10)
        ax3.set_title(f'(c) Pmax箱线图 ({pmax_outliers.sum()}个异常)', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_xlim([0.5, 1.5])
        
        # ========== (d) Pcomp异常检测 ==========
        ax4 = fig.add_subplot(gs[1, 2])
        
        pcomp_outliers = outlier_masks[2]
        
        # Z-score分布
        z_scores = np.abs(stats.zscore(df_clean['P_comp'].dropna()))
        ax4.hist(z_scores, bins=30, alpha=0.7, color=self.colors['info'],
                edgecolor='black', linewidth=0.5)
        ax4.axvline(3, color=self.colors['danger'], linestyle='--',
                   linewidth=2, label='Z=3阈值')
        ax4.axvline(2, color=self.colors['warning'], linestyle=':',
                   linewidth=1.5, label='Z=2参考线')
        
        outlier_zscore = (z_scores > 3).sum()
        ax4.text(0.98, 0.98, f'Z>3: {outlier_zscore}个',
                transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
        
        ax4.set_xlabel('Z-score', fontsize=10)
        ax4.set_ylabel('频次', fontsize=10)
        ax4.set_title('(d) Pcomp Z-score分布', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # ========== (e) 清洗效果对比 ==========
        ax5 = fig.add_subplot(gs[2, 0])
        
        categories = ['原始数据', '清洗后']
        counts = [len(df), len(df_clean)]
        colors_bar = [self.colors['dark'], self.colors['success']]
        
        bars = ax5.barh(categories, counts, color=colors_bar, alpha=0.8,
                       edgecolor='black', linewidth=1.5)
        
        for bar, count in zip(bars, counts):
            ax5.text(count + len(df)*0.02, bar.get_y() + bar.get_height()/2,
                    f'{count} ({count/len(df)*100:.1f}%)',
                    va='center', fontsize=10, fontweight='bold')
        
        ax5.set_xlabel('样本数量', fontsize=10)
        ax5.set_title('(e) 数据清洗流程效果', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        ax5.set_xlim(0, len(df) * 1.2)
        
        # ========== (f) 物理约束检查 ==========
        ax6 = fig.add_subplot(gs[2, 1])
        
        # 物理约束: Pmax > Pcomp
        valid_physics = df['P_max'] > df['P_comp']
        invalid_physics = ~valid_physics
        
        ratio = df['P_max'] / df['P_comp']
        
        ax6.scatter(np.arange(len(df))[valid_physics], ratio[valid_physics],
                   s=10, alpha=0.5, color=self.colors['success'], label='物理合理')
        ax6.scatter(np.arange(len(df))[invalid_physics], ratio[invalid_physics],
                   s=30, alpha=0.8, color=self.colors['danger'], marker='x',
                   linewidths=1.5, label='物理不合理')
        
        ax6.axhline(1.0, color=self.colors['warning'], linestyle='--',
                   linewidth=2, label='物理边界 (Pmax/Pcomp=1)')
        ax6.axhline(1.3, color=self.colors['info'], linestyle=':',
                   linewidth=1, alpha=0.7, label='典型范围 (1.2-1.5)')
        ax6.axhline(1.2, color=self.colors['info'], linestyle=':',
                   linewidth=1, alpha=0.7)
        
        ax6.set_xlabel('采样点', fontsize=10)
        ax6.set_ylabel('P_max / P_comp', fontsize=10)
        ax6.set_title('(f) 物理约束一致性检查', fontsize=11, fontweight='bold')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([0.5, 2.5])
        
        n_invalid = invalid_physics.sum()
        if n_invalid > 0:
            ax6.text(0.98, 0.02, f'违反物理约束: {n_invalid} ({n_invalid/len(df)*100:.1f}%)',
                    transform=ax6.transAxes, fontsize=9,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.6))
        
        # ========== (g) 清洗前后分布对比 ==========
        ax7 = fig.add_subplot(gs[2, 2])
        
        ax7.hist(df['P_max'].dropna(), bins=40, alpha=0.5,
                color=self.colors['dark'], label='清洗前', density=True)
        ax7.hist(df_clean['P_max'], bins=40, alpha=0.7,
                color=self.colors['success'], label='清洗后', density=True)
        
        # 正态分布拟合
        mu, sigma = df_clean['P_max'].mean(), df_clean['P_max'].std()
        x = np.linspace(df_clean['P_max'].min(), df_clean['P_max'].max(), 100)
        ax7.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                label=f'正态拟合 μ={mu:.1f}, σ={sigma:.1f}')
        
        ax7.set_xlabel('P_max (bar)', fontsize=10)
        ax7.set_ylabel('概率密度', fontsize=10)
        ax7.set_title('(g) 清洗前后分布对比', fontsize=11, fontweight='bold')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
        
        plt.suptitle('数据清洗与异常值剔除流程', fontsize=14, fontweight='bold', y=0.995)
        
        save_path = os.path.join(OUTPUT_DIR, 'data_cleaning.png')
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=150)
        plt.close()
        print(f"  ✓ 已保存: {save_path}")
        
        # 保存CSV
        self._save_cleaning_csv(df, df_clean, mask_combined)
        
        return fig
    
    def _save_cleaning_csv(self, df_raw, df_clean, clean_mask):
        """保存清洗CSV数据"""
        # 原始数据标记
        cleaning_data = df_raw[['rpm', 'P_max', 'P_comp', 'T_exh']].copy()
        cleaning_data['is_outlier'] = (~clean_mask).astype(int)
        cleaning_data.to_csv(os.path.join(DATA_DIR, 'data_cleaning_raw.csv'),
                            index=False, encoding='utf-8-sig')
        
        # 清洗后数据
        clean_data = df_clean[['rpm', 'P_max', 'P_comp', 'T_exh']].copy()
        clean_data.to_csv(os.path.join(DATA_DIR, 'data_cleaning_final.csv'),
                         index=False, encoding='utf-8-sig')
        
        print(f"  ✓ CSV: data_cleaning_raw.csv, data_cleaning_final.csv")
    
    def plot_normalization_correlation(self):
        """可视化4: 数据的标准化处理与参数关联"""
        print("\n[4/4] 生成标准化处理与参数关联可视化...")
        
        if self.data_clean is None:
            if self.data is None:
                self.load_real_data()
            self.apply_outlier_filter()
        
        df = self.data_clean.copy()
        
        # 选择关键参数并删除缺失值
        params = ['rpm', 'P_max', 'P_comp', 'T_exh']
        available_params = [p for p in params if p in df.columns]
        
        data_raw = df[available_params].dropna().copy()
        
        if len(data_raw) == 0:
            print("  ⚠ 警告: 清洗后数据为空，跳过标准化可视化")
            return None
        
        # Min-Max标准化
        scaler_minmax = MinMaxScaler()
        data_minmax = pd.DataFrame(
            scaler_minmax.fit_transform(data_raw),
            columns=data_raw.columns
        )
        
        # Z-score标准化
        scaler_zscore = StandardScaler()
        data_zscore = pd.DataFrame(
            scaler_zscore.fit_transform(data_raw),
            columns=data_raw.columns
        )
        
        print(f"  标准化处理: {len(available_params)} 个参数")
        
        # 创建图形
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
        
        # ========== (a) 原始数据分布 ==========
        ax1 = fig.add_subplot(gs[0, 0])
        
        bp1 = data_raw.boxplot(ax=ax1, patch_artist=True, return_type='dict')
        for patch in bp1['boxes']:
            patch.set_facecolor(self.colors['dark'])
            patch.set_alpha(0.6)
        
        ax1.set_ylabel('原始值 (不同单位)', fontsize=10)
        ax1.set_title('(a) 原始数据 - 量纲不一致', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=45)
        
        # 标注单位
        units = {'rpm': 'rpm', 'P_max': 'bar', 'P_comp': 'bar', 'T_exh': 'K'}
        for i, param in enumerate(available_params):
            if param in units:
                ax1.text(i+1, ax1.get_ylim()[1]*0.9, units[param],
                        ha='center', fontsize=8, style='italic', color='red')
        
        # ========== (b) Min-Max标准化 ==========
        ax2 = fig.add_subplot(gs[0, 1])
        
        bp2 = data_minmax.boxplot(ax=ax2, patch_artist=True, return_type='dict')
        for patch in bp2['boxes']:
            patch.set_facecolor(self.colors['info'])
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('归一化值 [0, 1]', fontsize=10)
        ax2.set_title('(b) Min-Max标准化', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim([-0.1, 1.1])
        
        # ========== (c) Z-score标准化 ==========
        ax3 = fig.add_subplot(gs[0, 2])
        
        bp3 = data_zscore.boxplot(ax=ax3, patch_artist=True, return_type='dict')
        for patch in bp3['boxes']:
            patch.set_facecolor(self.colors['success'])
            patch.set_alpha(0.7)
        
        ax3.axhline(0, color=self.colors['primary'], linestyle='--',
                   linewidth=2, label='μ=0')
        ax3.axhline(-3, color=self.colors['danger'], linestyle=':',
                   linewidth=1, alpha=0.7)
        ax3.axhline(3, color=self.colors['danger'], linestyle=':',
                   linewidth=1, alpha=0.7)
        
        ax3.set_ylabel('标准化值 (μ=0, σ=1)', fontsize=10)
        ax3.set_title('(c) Z-score标准化', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim([-4, 4])
        
        # ========== (d) 相关性热力图 - 原始数据 ==========
        ax4 = fig.add_subplot(gs[1, 0])
        
        corr_raw = data_raw.corr()
        im = ax4.imshow(corr_raw, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        ax4.set_xticks(range(len(corr_raw.columns)))
        ax4.set_yticks(range(len(corr_raw.columns)))
        ax4.set_xticklabels(corr_raw.columns, rotation=45, ha='right', fontsize=9)
        ax4.set_yticklabels(corr_raw.columns, fontsize=9)
        
        # 添加相关系数数值
        for i in range(len(corr_raw)):
            for j in range(len(corr_raw)):
                text_color = "white" if abs(corr_raw.iloc[i, j]) > 0.5 else "black"
                ax4.text(j, i, f'{corr_raw.iloc[i, j]:.2f}',
                        ha="center", va="center", fontsize=8, color=text_color)
        
        ax4.set_title('(d) 原始数据相关性矩阵', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        
        # ========== (e) Pcomp估算 vs 实测 ==========
        ax5 = fig.add_subplot(gs[1, 1])
        
        if 'P_comp_est' in df.columns:
            ax5.scatter(df['P_comp'], df['P_comp_est'], s=20, alpha=0.5,
                       color=self.colors['primary'], edgecolors='black', linewidths=0.3)
            
            # 对角线
            min_val = min(df['P_comp'].min(), df['P_comp_est'].min())
            max_val = max(df['P_comp'].max(), df['P_comp_est'].max())
            ax5.plot([min_val, max_val], [min_val, max_val],
                    'r--', linewidth=2, label='y=x')
            
            # 计算R²
            from sklearn.metrics import r2_score
            r2 = r2_score(df['P_comp'], df['P_comp_est'])
            
            ax5.text(0.05, 0.95, f'R² = {r2:.3f}',
                    transform=ax5.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
        
        ax5.set_xlabel('P_comp 实测 (bar)', fontsize=10)
        ax5.set_ylabel('P_comp 估算 (bar)', fontsize=10)
        ax5.set_title('(e) Pcomp物理模型估算', fontsize=11, fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # ========== (f) 参数散点矩阵 ==========
        ax6 = fig.add_subplot(gs[1, 2])
        
        # 最强相关参数对
        corr_abs = corr_raw.abs()
        np.fill_diagonal(corr_abs.values, 0)
        max_corr_idx = np.unravel_index(corr_abs.values.argmax(), corr_abs.shape)
        param1 = corr_raw.columns[max_corr_idx[0]]
        param2 = corr_raw.columns[max_corr_idx[1]]
        
        ax6.scatter(data_raw[param1], data_raw[param2], s=20, alpha=0.5,
                   color=self.colors['primary'], edgecolors='black', linewidths=0.3)
        
        # 线性拟合
        z = np.polyfit(data_raw[param1], data_raw[param2], 1)
        p = np.poly1d(z)
        x_fit = np.linspace(data_raw[param1].min(), data_raw[param1].max(), 100)
        ax6.plot(x_fit, p(x_fit), color=self.colors['danger'], linewidth=2,
                label=f'y={z[0]:.2f}x+{z[1]:.1f}')
        
        r = corr_raw.loc[param1, param2]
        ax6.text(0.05, 0.95, f'相关系数: r={r:.3f}',
                transform=ax6.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
        
        ax6.set_xlabel(f'{param1}', fontsize=10)
        ax6.set_ylabel(f'{param2}', fontsize=10)
        ax6.set_title(f'(f) 最强相关: {param1} vs {param2}', fontsize=11, fontweight='bold')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        # ========== (g) PCA降维可视化 ==========
        ax7 = fig.add_subplot(gs[2, 0])
        
        if len(available_params) >= 3:
            pca = PCA(n_components=2)
            data_pca = pca.fit_transform(data_zscore)
            
            scatter = ax7.scatter(data_pca[:, 0], data_pca[:, 1],
                                 c=data_raw['rpm'], cmap='viridis',
                                 s=30, alpha=0.6, edgecolors='black', linewidths=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax7)
            cbar.set_label('RPM', fontsize=9)
            
            ax7.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
            ax7.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
            ax7.set_title('(g) PCA主成分分析', fontsize=11, fontweight='bold')
            ax7.grid(True, alpha=0.3)
            ax7.axhline(0, color='k', linewidth=0.5)
            ax7.axvline(0, color='k', linewidth=0.5)
        
        # ========== (h) 特征方差重要性 ==========
        ax8 = fig.add_subplot(gs[2, 1])
        
        feature_variance = data_zscore.var().sort_values(ascending=True)
        
        bars = ax8.barh(feature_variance.index, feature_variance.values,
                       color=self.colors['info'], alpha=0.8,
                       edgecolor='black', linewidth=1)
        
        for bar, val in zip(bars, feature_variance.values):
            ax8.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', fontsize=9, fontweight='bold')
        
        ax8.set_xlabel('标准化方差', fontsize=10)
        ax8.set_title('(h) 特征方差重要性', fontsize=11, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='x')
        
        # ========== (i) 标准化方法对比表 ==========
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        comparison_data = [
            ['Min-Max', '[0, 1]', '保留分布', '神经网络'],
            ['Z-score', 'μ=0,σ=1', '改变分布', '线性模型'],
            ['Robust', '中位数', '最鲁棒', '异常数据'],
        ]
        
        table = ax9.table(cellText=comparison_data,
                         colLabels=['方法', '范围', '特点', '适用场景'],
                         cellLoc='center', loc='center',
                         colWidths=[0.2, 0.2, 0.3, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 3.0)
        
        # 表头样式
        for i in range(4):
            table[(0, i)].set_facecolor(self.colors['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 高亮Z-score (本项目使用)
        for j in range(4):
            table[(2, j)].set_facecolor(self.colors['success'])
            table[(2, j)].set_alpha(0.3)
        
        ax9.set_title('(i) 标准化方法对比', fontsize=11, fontweight='bold', pad=20)
        
        plt.suptitle('数据标准化处理与参数关联分析', fontsize=14, fontweight='bold', y=0.995)
        
        save_path = os.path.join(OUTPUT_DIR, 'normalization_correlation.png')
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=150)
        plt.close()
        print(f"  ✓ 已保存: {save_path}")
        
        # 保存CSV
        self._save_normalization_csv(data_raw, data_minmax, data_zscore, corr_raw)
        
        return fig
    
    def _save_normalization_csv(self, data_raw, data_minmax, data_zscore, corr_raw):
        """保存标准化CSV数据"""
        data_raw.to_csv(os.path.join(DATA_DIR, 'data_raw.csv'),
                       index=False, encoding='utf-8-sig')
        data_minmax.to_csv(os.path.join(DATA_DIR, 'data_minmax_normalized.csv'),
                          index=False, encoding='utf-8-sig')
        data_zscore.to_csv(os.path.join(DATA_DIR, 'data_zscore_normalized.csv'),
                          index=False, encoding='utf-8-sig')
        corr_raw.to_csv(os.path.join(DATA_DIR, 'correlation_matrix.csv'),
                       encoding='utf-8-sig')
        
        print(f"  ✓ CSV: data_raw.csv, data_minmax_normalized.csv, data_zscore_normalized.csv, correlation_matrix.csv")


def main():
    """主函数 - 按顺序测试可视化"""
    print("=" * 70)
    print("数据清洗与特征提取可视化 - 测试运行")
    print("=" * 70)
    print(f"输出目录: {os.path.abspath(OUTPUT_DIR)}")
    print(f"数据目录: {os.path.abspath(DATA_DIR)}")
    print()
    
    visualizer = DataPreprocessingVisualizer()
    
    # 顺序测试
    try:
        # 1. 稳态筛选
        visualizer.plot_steady_state_selection()
        
        # 2. 工况点提取
        visualizer.plot_representative_points()
        
        # 3. 异常值剔除
        visualizer.plot_data_cleaning()
        
        # 4. 标准化处理
        visualizer.plot_normalization_correlation()
        
        print()
        print("=" * 70)
        print("✅ 测试完成！已生成:")
        print("=" * 70)
        
        # 统计生成的文件
        png_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        
        print(f"\nPNG图片 ({len([f for f in png_files if any(kw in f for kw in ['steady', 'representative', 'cleaning', 'normalization'])])} 个新增):")
        for fname in sorted(png_files):
            if any(kw in fname for kw in ['steady', 'representative', 'cleaning', 'normalization']):
                path = os.path.join(OUTPUT_DIR, fname)
                size = os.path.getsize(path) / 1024
                print(f"  {fname:<40} {size:>8.1f} KB")
        
        print(f"\nCSV数据 ({len([f for f in csv_files if any(kw in f for kw in ['steady', 'operating', 'representative', 'cleaning', 'data_'])])} 个新增):")
        for fname in sorted(csv_files):
            if any(kw in fname for kw in ['steady', 'operating', 'representative', 'cleaning', 'data_', 'correlation']):
                path = os.path.join(DATA_DIR, fname)
                size = os.path.getsize(path) / 1024
                print(f"  {fname:<40} {size:>8.1f} KB")
        
        print("\n✅ 所有4个可视化测试通过！")
        print("下一步: 实装到主代码 visualize_data_preprocessing.py")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    """主函数 - 按顺序测试可视化"""
    print("=" * 70)
    print("数据清洗与特征提取可视化 - 测试运行")
    print("=" * 70)
    print(f"输出目录: {os.path.abspath(OUTPUT_DIR)}")
    print(f"数据目录: {os.path.abspath(DATA_DIR)}")
    print()
    
    visualizer = DataPreprocessingVisualizer()
    
    # 顺序测试
    try:
        # 1. 稳态筛选
        visualizer.plot_steady_state_selection()
        
        # 2. 工况点提取
        visualizer.plot_representative_points()
        
        print()
        print("=" * 70)
        print("✅ 测试完成！已生成:")
        print("=" * 70)
        
        # 统计生成的文件
        png_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        
        print(f"\nPNG图片 ({len(png_files)} 个):")
        for fname in sorted(png_files):
            if 'steady' in fname or 'representative' in fname:
                path = os.path.join(OUTPUT_DIR, fname)
                size = os.path.getsize(path) / 1024
                print(f"  {fname:<40} {size:>8.1f} KB")
        
        print(f"\nCSV数据 ({len(csv_files)} 个):")
        for fname in sorted(csv_files):
            if 'steady' in fname or 'operating' in fname or 'representative' in fname:
                path = os.path.join(DATA_DIR, fname)
                size = os.path.getsize(path) / 1024
                print(f"  {fname:<40} {size:>8.1f} KB")
        
        print("\n下一步: 继续实现可视化3(异常值剔除)和可视化4(标准化处理)")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
