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


class CalibrationProcessPlotter:
    """
    校准过程可视化
    
    展示三阶段分步解耦校准的过程和结果
    """
    
    def __init__(self, figsize: Tuple[float, float] = (14, 10)):
        self.figsize = figsize
        self.colors = {
            'stage1': '#3498DB',   # 压缩段 - 蓝色
            'stage2': '#E74C3C',   # 燃烧段 - 红色
            'stage3': '#2ECC71',   # 传热段 - 绿色
            'target': '#F39C12',   # 目标值 - 橙色
            'sim': '#9B59B6',      # 仿真值 - 紫色
        }
    
    def plot_calibration_process(self,
                                  calibration_history: Dict = None,
                                  save_path: str = None) -> plt.Figure:
        """
        图表: 三阶段校准过程综合图
        
        展示分步解耦校准策略的完整流程
        
        Args:
            calibration_history: 校准历史数据 (可选)
            save_path: 保存路径
            
        Returns:
            fig: 图形对象
        """
        fig = plt.figure(figsize=self.figsize)
        
        # 创建2x2网格布局
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3,
                              left=0.08, right=0.95, top=0.92, bottom=0.08)
        
        # 子图1: 校准流程示意图
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_flowchart(ax1)
        
        # 子图2: 参数敏感性分析
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_sensitivity(ax2, calibration_history)
        
        # 子图3: 校准收敛曲线
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_convergence(ax3, calibration_history)
        
        # 子图4: 校准结果对比
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_results_comparison(ax4, calibration_history)
        
        fig.suptitle('Three-Stage Decoupled Calibration Process', 
                     fontsize=16, fontweight='bold')
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def _plot_flowchart(self, ax):
        """绘制校准流程示意图"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('(A) Calibration Workflow', fontsize=12, fontweight='bold', loc='left')
        
        # 三个阶段的方框
        stages = [
            {'name': 'Stage 1\nCompression', 'target': 'Pcomp', 'param': 'CR', 
             'y': 8, 'color': self.colors['stage1']},
            {'name': 'Stage 2\nCombustion', 'target': 'Pmax', 'param': 'Wiebe', 
             'y': 5, 'color': self.colors['stage2']},
            {'name': 'Stage 3\nHeat Transfer', 'target': 'Texh', 'param': 'Woschni', 
             'y': 2, 'color': self.colors['stage3']},
        ]
        
        for stage in stages:
            # 主方框
            rect = mpatches.FancyBboxPatch(
                (0.5, stage['y'] - 0.8), 3, 1.6,
                boxstyle="round,pad=0.05,rounding_size=0.2",
                facecolor=stage['color'], edgecolor='black', linewidth=2, alpha=0.8
            )
            ax.add_patch(rect)
            ax.text(2, stage['y'], stage['name'], ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
            
            # 目标参数
            ax.annotate(f"Target: {stage['target']}", xy=(3.7, stage['y']),
                       xytext=(5.5, stage['y'] + 0.3),
                       fontsize=9, ha='left',
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
            
            # 调整参数
            ax.annotate(f"Tune: {stage['param']}", xy=(3.7, stage['y']),
                       xytext=(5.5, stage['y'] - 0.3),
                       fontsize=9, ha='left', color=stage['color'],
                       arrowprops=dict(arrowstyle='->', color=stage['color'], lw=1.5))
        
        # 连接箭头
        for i in range(len(stages) - 1):
            ax.annotate('', xy=(2, stages[i+1]['y'] + 1.0), 
                       xytext=(2, stages[i]['y'] - 1.0),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))
        
        # 添加"Fix & Proceed"标注
        ax.text(2.8, 6.5, 'Fix CR', fontsize=8, style='italic', color='gray')
        ax.text(2.8, 3.5, 'Fix Wiebe', fontsize=8, style='italic', color='gray')
    
    def _plot_sensitivity(self, ax, history: Dict = None):
        """绘制参数敏感性分析图"""
        ax.set_title('(B) Parameter Sensitivity', fontsize=12, fontweight='bold', loc='left')
        
        # 示例敏感性数据
        params = ['CR', 'Inj.Timing', 'Duration', 'Shape', 'C_woschni']
        targets = ['Pcomp', 'Pmax', 'Texh']
        
        # 敏感性矩阵 (示意数据)
        sensitivity = np.array([
            [0.95, 0.15, 0.05],   # CR
            [0.10, 0.85, 0.20],   # Inj.Timing
            [0.05, 0.70, 0.15],   # Duration
            [0.03, 0.45, 0.10],   # Shape
            [0.08, 0.12, 0.90],   # C_woschni
        ])
        
        im = ax.imshow(sensitivity, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(targets)))
        ax.set_yticks(np.arange(len(params)))
        ax.set_xticklabels(targets, fontsize=10)
        ax.set_yticklabels(params, fontsize=10)
        
        # 添加数值标注
        for i in range(len(params)):
            for j in range(len(targets)):
                val = sensitivity[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       fontsize=9, color=color, fontweight='bold')
        
        # 颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Sensitivity', fontsize=10)
        
        ax.set_xlabel('Target Variables', fontsize=10)
        ax.set_ylabel('Parameters', fontsize=10)
    
    def _plot_convergence(self, ax, history: Dict = None):
        """绘制校准收敛曲线"""
        ax.set_title('(C) Convergence History', fontsize=12, fontweight='bold', loc='left')
        
        # 生成示例收敛数据
        if history and 'convergence' in history:
            conv_data = history['convergence']
        else:
            # 模拟收敛曲线
            iters_s1 = np.arange(1, 11)
            iters_s2 = np.arange(1, 16)
            iters_s3 = np.arange(1, 9)
            
            error_s1 = 15 * np.exp(-0.5 * iters_s1) + 1.5
            error_s2 = 12 * np.exp(-0.25 * iters_s2) + 2.0
            error_s3 = 8 * np.exp(-0.4 * iters_s3) + 3.0
            
            conv_data = {
                'stage1': {'iter': iters_s1, 'error': error_s1},
                'stage2': {'iter': iters_s2, 'error': error_s2},
                'stage3': {'iter': iters_s3, 'error': error_s3},
            }
        
        # 绘制各阶段收敛曲线
        offset = 0
        stage_labels = ['Stage 1 (CR)', 'Stage 2 (Wiebe)', 'Stage 3 (Woschni)']
        colors = [self.colors['stage1'], self.colors['stage2'], self.colors['stage3']]
        
        for i, (stage, data) in enumerate(conv_data.items()):
            iters = data['iter'] + offset
            errors = data['error']
            ax.plot(iters, errors, 'o-', color=colors[i], linewidth=2, 
                   markersize=5, label=stage_labels[i])
            
            # 添加阶段分隔线
            if i < 2:
                ax.axvline(x=iters[-1] + 0.5, color='gray', linestyle='--', 
                          alpha=0.5, linewidth=1)
            
            offset = iters[-1]
        
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Relative Error [%]', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)
        
        # 添加目标线
        ax.axhline(y=2, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.text(offset + 1, 2.5, 'Target < 2%', fontsize=8, color='green')
    
    def _plot_results_comparison(self, ax, history: Dict = None):
        """绘制校准结果对比图"""
        ax.set_title('(D) Calibration Results', fontsize=12, fontweight='bold', loc='left')
        
        # 校准前后对比数据
        if history and 'results' in history:
            results = history['results']
        else:
            results = {
                'Pcomp': {'exp': 145.0, 'before': 132.5, 'after': 144.2},
                'Pmax': {'exp': 180.0, 'before': 165.0, 'after': 178.5},
                'Texh': {'exp': 380.0, 'before': 420.0, 'after': 388.0},
            }
        
        params = list(results.keys())
        x = np.arange(len(params))
        width = 0.25
        
        exp_vals = [results[p]['exp'] for p in params]
        before_vals = [results[p]['before'] for p in params]
        after_vals = [results[p]['after'] for p in params]
        
        # 归一化显示
        exp_norm = np.array(exp_vals)
        before_norm = np.array(before_vals) / exp_norm * 100
        after_norm = np.array(after_vals) / exp_norm * 100
        exp_norm = np.ones(len(params)) * 100
        
        bars1 = ax.bar(x - width, before_norm, width, label='Before Cal.',
                       color='#95A5A6', edgecolor='black')
        bars2 = ax.bar(x, exp_norm, width, label='Experiment',
                       color=self.colors['target'], edgecolor='black')
        bars3 = ax.bar(x + width, after_norm, width, label='After Cal.',
                       color=self.colors['sim'], edgecolor='black')
        
        ax.set_ylabel('Normalized Value [%]', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(params, fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加误差带
        ax.axhline(y=98, color='green', linestyle=':', alpha=0.5)
        ax.axhline(y=102, color='green', linestyle=':', alpha=0.5)
        ax.fill_between([-0.5, len(params) - 0.5], 98, 102, 
                        color='green', alpha=0.1, label='_2% band')
        
        ax.set_ylim(85, 115)
        ax.set_xlim(-0.5, len(params) - 0.5)
        
        # 添加误差百分比标注
        for i, p in enumerate(params):
            error_before = abs(before_norm[i] - 100)
            error_after = abs(after_norm[i] - 100)
            ax.text(i - width, before_norm[i] + 2, f'{error_before:.1f}%', 
                   ha='center', fontsize=8, color='gray')
            ax.text(i + width, after_norm[i] + 2, f'{error_after:.1f}%', 
                   ha='center', fontsize=8, color=self.colors['sim'])
    
    def plot_stage_detail(self, 
                          stage: int,
                          param_range: np.ndarray,
                          error_curve: np.ndarray,
                          optimal_value: float,
                          target_name: str,
                          param_name: str,
                          save_path: str = None) -> plt.Figure:
        """
        绘制单阶段校准详细图
        
        Args:
            stage: 阶段编号 (1, 2, 3)
            param_range: 参数扫描范围
            error_curve: 对应的误差曲线
            optimal_value: 最优参数值
            target_name: 目标变量名
            param_name: 参数名
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {1: self.colors['stage1'], 
                  2: self.colors['stage2'], 
                  3: self.colors['stage3']}
        
        ax.plot(param_range, error_curve, '-', color=colors[stage], 
                linewidth=2.5, label='Error Curve')
        ax.axvline(x=optimal_value, color='red', linestyle='--', 
                   linewidth=2, label=f'Optimal: {optimal_value:.3f}')
        ax.axhline(y=2, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        
        ax.fill_between(param_range, 0, 2, color='green', alpha=0.1)
        
        ax.set_xlabel(f'{param_name}', fontsize=12)
        ax.set_ylabel(f'{target_name} Error [%]', fontsize=12)
        ax.set_title(f'Stage {stage}: {param_name} Optimization', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)
        
        # 标注最优点
        opt_idx = np.argmin(np.abs(param_range - optimal_value))
        opt_error = error_curve[opt_idx]
        ax.plot(optimal_value, opt_error, 'ro', markersize=12, zorder=5)
        ax.annotate(f'Min Error: {opt_error:.2f}%', 
                   xy=(optimal_value, opt_error),
                   xytext=(optimal_value + 0.1 * (param_range[-1] - param_range[0]), 
                          opt_error + 2),
                   fontsize=10,
                   arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
