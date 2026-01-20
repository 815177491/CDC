"""
静态图表绑定
============
校验图谱、协同效果时序图、P-V示功图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from typing import Dict, List, Tuple, Optional
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


class CalibrationPlotter:
    """
    校准验证绑定
    
    生成模型精度验证图谱
    """
    
    def __init__(self, figsize: Tuple[float, float] = (12, 8)):
        self.figsize = figsize
        self.colors = {
            'sim': '#2E86AB',     # 仿真值 - 蓝色
            'exp': '#E94F37',     # 实验值 - 红色
            'error_band': '#A1C181',  # 误差带 - 绿色
        }
    
    def plot_thermal_comparison(self,
                                  conditions: List[Dict],
                                  sim_results: List[Dict],
                                  exp_results: List[Dict],
                                  save_path: str = None) -> plt.Figure:
        """
        图表1: 热力参数对标图
        
        双Y轴折线图展示Pmax和Pcomp的仿真vs实验对比
        
        Args:
            conditions: 工况列表 [{'rpm': ..., 'load': ...}, ...]
            sim_results: 仿真结果 [{'Pmax': ..., 'Pcomp': ...}, ...]
            exp_results: 实验结果 [{'Pmax': ..., 'Pcomp': ...}, ...]
            save_path: 保存路径
            
        Returns:
            fig: 图形对象
        """
        fig, ax1 = plt.subplots(figsize=self.figsize)
        
        n_points = len(conditions)
        x = np.arange(n_points)
        
        # 提取数据
        Pmax_sim = [r.get('Pmax', 0) for r in sim_results]
        Pmax_exp = [r.get('Pmax', 0) for r in exp_results]
        Pcomp_sim = [r.get('Pcomp', 0) for r in sim_results]
        Pcomp_exp = [r.get('Pcomp', 0) for r in exp_results]
        
        # 左Y轴: Pmax
        color1 = self.colors['sim']
        ax1.set_xlabel('工况编号', fontsize=12)
        ax1.set_ylabel('最大爆发压力 Pmax [bar]', color=color1, fontsize=12)
        
        line1, = ax1.plot(x, Pmax_sim, 'o-', color=color1, linewidth=2,
                          markersize=8, label='Pmax 仿真')
        line2, = ax1.plot(x, Pmax_exp, 's--', color=self.colors['exp'], linewidth=2,
                          markersize=8, label='Pmax 实验')
        
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.fill_between(x, 
                          np.array(Pmax_exp) * 0.97,
                          np.array(Pmax_exp) * 1.03,
                          color=self.colors['error_band'], alpha=0.3,
                          label='±3% 误差带')
        
        # 右Y轴: Pcomp
        ax2 = ax1.twinx()
        color2 = '#6B4E71'
        ax2.set_ylabel('压缩终点压力 Pcomp [bar]', color=color2, fontsize=12)
        
        line3, = ax2.plot(x, Pcomp_sim, '^-', color=color2, linewidth=2,
                          markersize=8, label='Pcomp 仿真')
        line4, = ax2.plot(x, Pcomp_exp, 'v--', color='#9B59B6', linewidth=2,
                          markersize=8, label='Pcomp 实验')
        
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # 图例
        lines = [line1, line2, line3, line4]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=10)
        
        # 标题
        ax1.set_title('零维模型校准验证: 热力参数对标', fontsize=14, fontweight='bold')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.grid(True, alpha=0.3)
        
        # 计算并标注误差
        Pmax_errors = [abs(s - e) / e * 100 for s, e in zip(Pmax_sim, Pmax_exp)]
        Pcomp_errors = [abs(s - e) / e * 100 for s, e in zip(Pcomp_sim, Pcomp_exp)]
        
        textstr = f'Pmax 平均误差: {np.mean(Pmax_errors):.2f}%\n' \
                  f'Pcomp 平均误差: {np.mean(Pcomp_errors):.2f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.98, 0.02, textstr, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    def plot_pv_diagram(self,
                         results_list: List[Dict],
                         labels: List[str] = None,
                         save_path: str = None) -> plt.Figure:
        """
        图表2: P-V示功图
        
        叠加多个工况的P-V环
        
        Args:
            results_list: 各工况的仿真结果
            labels: 工况标签
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(results_list)))
        
        for i, results in enumerate(results_list):
            V = results.get('volume', []) * 1000  # m^3 -> L
            p = results.get('pressure', []) / 1e5  # Pa -> bar
            
            label = labels[i] if labels else f'工况 {i+1}'
            ax.plot(V, p, linewidth=2, color=colors[i], label=label)
        
        ax.set_xlabel('气缸容积 [L]', fontsize=12)
        ax.set_ylabel('缸内压力 [bar]', fontsize=12)
        ax.set_title('P-V 示功图 (多工况叠加)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 标注压缩和膨胀方向
        ax.annotate('压缩', xy=(0.3, 0.5), xycoords='axes fraction',
                    fontsize=12, ha='center')
        ax.annotate('膨胀', xy=(0.7, 0.5), xycoords='axes fraction',
                    fontsize=12, ha='center')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    def plot_heat_release(self,
                           theta: np.ndarray,
                           dQ: np.ndarray,
                           burn_fraction: np.ndarray = None,
                           save_path: str = None) -> plt.Figure:
        """
        燃烧放热率曲线
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 放热率
        ax1.plot(theta, dQ / 1000, 'b-', linewidth=2, label='放热率 dQ/dθ')
        ax1.set_xlabel('曲轴转角 [°CA ATDC]', fontsize=12)
        ax1.set_ylabel('放热率 [kJ/°CA]', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax1.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # 累积燃烧分数
        if burn_fraction is not None:
            ax2 = ax1.twinx()
            ax2.plot(theta, burn_fraction * 100, 'r--', linewidth=2, 
                     label='累积燃烧分数')
            ax2.set_ylabel('累积燃烧分数 [%]', color='r', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.set_ylim(0, 110)
        
        ax1.set_title('燃烧放热特性曲线', fontsize=14, fontweight='bold')
        ax1.set_xlim(-30, 90)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        
        return fig


class SynergyPlotter:
    """
    协同控制效果绑定
    
    生成故障响应时序图与控制效果展示
    """
    
    def __init__(self, figsize: Tuple[float, float] = (12, 10)):
        self.figsize = figsize
    
    def plot_fault_response(self,
                             response_data: Dict,
                             fault_description: str = "喷油正时偏差",
                             save_path: str = None) -> plt.Figure:
        """
        图表3: 故障响应时序图
        
        三子图垂直排列:
        A) 故障注入信号
        B) Pmax响应对比
        C) 控制动作
        
        Args:
            response_data: 响应数据字典
            fault_description: 故障描述
            save_path: 保存路径
        """
        fig, axes = plt.subplots(3, 1, figsize=self.figsize, sharex=True)
        
        time = response_data.get('time', np.array([]))
        
        # 子图A: 故障注入
        ax_fault = axes[0]
        fault_signal = response_data.get('fault_signal', np.zeros_like(time))
        if len(fault_signal) == 0:
            # 生成示意性故障信号
            fault_signal = np.zeros_like(time)
            onset_idx = len(time) // 4
            fault_signal[onset_idx:] = 2.0  # 2度正时偏差
        
        ax_fault.plot(time, fault_signal, 'k-', linewidth=2)
        ax_fault.fill_between(time, 0, fault_signal, color='gray', alpha=0.3)
        ax_fault.set_ylabel(f'{fault_description}\n[°CA]', fontsize=11)
        ax_fault.set_title('(A) 故障注入信号', fontsize=12, fontweight='bold', loc='left')
        ax_fault.grid(True, alpha=0.3)
        ax_fault.set_ylim(-0.5, 3.5)
        
        # 子图B: Pmax响应
        ax_pmax = axes[1]
        Pmax_baseline = response_data.get('Pmax_baseline', np.array([]))
        Pmax_open = response_data.get('Pmax_open_loop', np.array([]))
        Pmax_synergy = response_data.get('Pmax_synergy', np.array([]))
        
        if len(Pmax_baseline) > 0:
            ax_pmax.plot(time, Pmax_baseline, 'g-', linewidth=2, 
                         label='正常基准 (Baseline)')
        if len(Pmax_open) > 0:
            ax_pmax.plot(time, Pmax_open, 'r-', linewidth=2, 
                         label='无控制 (Open-loop)')
        if len(Pmax_synergy) > 0:
            ax_pmax.plot(time, Pmax_synergy, 'b-', linewidth=2, 
                         label='协同控制 (Synergy)')
        
        # 安全限值线
        ax_pmax.axhline(y=190, color='r', linestyle='--', linewidth=1.5, 
                        label='安全限值 190 bar')
        ax_pmax.axhline(y=180, color='orange', linestyle=':', linewidth=1.5,
                        label='预警值 180 bar')
        
        ax_pmax.set_ylabel('Pmax [bar]', fontsize=11)
        ax_pmax.set_title('(B) 最大爆发压力响应', fontsize=12, fontweight='bold', loc='left')
        ax_pmax.legend(loc='upper right', fontsize=9)
        ax_pmax.grid(True, alpha=0.3)
        
        # 标注关键区域
        if len(Pmax_open) > 0 and np.any(Pmax_open > 190):
            over_idx = np.where(Pmax_open > 190)[0]
            if len(over_idx) > 0:
                ax_pmax.fill_between(time[over_idx], 190, Pmax_open[over_idx],
                                     color='red', alpha=0.2, label='_越限区域')
        
        # 子图C: 控制动作
        ax_ctrl = axes[2]
        vit_adj = response_data.get('vit_adjustment', np.array([]))
        fuel_adj = response_data.get('fuel_adjustment', np.array([]))
        
        if len(vit_adj) > 0:
            ax_ctrl.plot(time, vit_adj, 'b-', linewidth=2, label='VIT 调整')
        
        ax_ctrl.set_xlabel('时间 [s]', fontsize=12)
        ax_ctrl.set_ylabel('VIT 调整量 [°CA]', color='b', fontsize=11)
        ax_ctrl.tick_params(axis='y', labelcolor='b')
        ax_ctrl.set_title('(C) 协同控制动作', fontsize=12, fontweight='bold', loc='left')
        ax_ctrl.grid(True, alpha=0.3)
        
        if len(fuel_adj) > 0 and np.any(fuel_adj != 0):
            ax_ctrl2 = ax_ctrl.twinx()
            ax_ctrl2.plot(time, fuel_adj, 'r--', linewidth=2, label='燃油调整')
            ax_ctrl2.set_ylabel('燃油调整 [%]', color='r', fontsize=11)
            ax_ctrl2.tick_params(axis='y', labelcolor='r')
        
        ax_ctrl.legend(loc='lower right', fontsize=9)
        
        # 添加标注
        fig.text(0.02, 0.5, '协同控制介入', va='center', rotation='vertical',
                 fontsize=12, color='blue', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    def plot_comparison_bars(self,
                              before: Dict[str, float],
                              after: Dict[str, float],
                              metrics: List[str] = None,
                              save_path: str = None) -> plt.Figure:
        """
        协同控制前后对比柱状图
        """
        if metrics is None:
            metrics = list(before.keys())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        before_vals = [before.get(m, 0) for m in metrics]
        after_vals = [after.get(m, 0) for m in metrics]
        
        bars1 = ax.bar(x - width/2, before_vals, width, label='故障发生时',
                       color='#E94F37', edgecolor='black')
        bars2 = ax.bar(x + width/2, after_vals, width, label='协同控制后',
                       color='#2E86AB', edgecolor='black')
        
        ax.set_ylabel('数值', fontsize=12)
        ax.set_title('协同控制效果对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars1, before_vals):
            ax.annotate(f'{val:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        for bar, val in zip(bars2, after_vals):
            ax.annotate(f'{val:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        
        return fig
