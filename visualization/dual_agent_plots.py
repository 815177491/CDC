"""
双智能体可视化扩展
==================
添加诊-控协同时序图、训练曲线、雷达图对比

Author: CDC Project
Date: 2026-01-24
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple, Optional
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


class DualAgentVisualizer:
    """双智能体可视化工具"""
    
    def __init__(self):
        self.colors = {
            'pmax': '#E74C3C',     # Pmax - 红色
            'vit': '#3498DB',      # VIT - 蓝色
            'fuel': '#F39C12',     # Fuel - 橙色
            'diagnosis': '#27AE60', # 诊断 - 绿色
            'baseline': '#95A5A6',  # Baseline - 灰色
            'proposed': '#9B59B6'   # Proposed - 紫色
        }
    
    def plot_coordination_response(self,
                                   time_steps: List[float],
                                   ground_truth_fault: List[int],
                                   diagnosis_output: List[int],
                                   pmax_values: List[float],
                                   vit_actions: List[float],
                                   fuel_actions: List[float],
                                   pmax_limit: float = 190.0,
                                   pmax_target: float = 170.0,
                                   save_path: str = None) -> plt.Figure:
        """
        诊-控协同时序响应图
        
        将真实故障、诊断输出、控制动作、系统状态绘制在同一时间轴上
        
        Args:
            time_steps: 时间步
            ground_truth_fault: 真实故障类型 (0=正常, 1=喷油正时, 2=气缸泄漏, 3=燃油, 4=喷油器)
            diagnosis_output: 诊断输出 (0-4对应故障类型)
            pmax_values: Pmax曲线
            vit_actions: VIT调整量
            fuel_actions: 燃油系数
            pmax_limit: Pmax限值
            pmax_target: Pmax目标值
            save_path: 保存路径
        
        Returns:
            fig: 图形对象
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(4, 1, hspace=0.35)
        
        # 轨道1: 真实故障 vs 诊断输出
        ax1 = fig.add_subplot(gs[0])
        
        fault_types = ['正常', '喷油正时', '气缸泄漏', '燃油系统', '喷油器']
        
        # 真实故障背景
        for i in range(len(time_steps)-1):
            if ground_truth_fault[i] != 0:
                ax1.axvspan(time_steps[i], time_steps[i+1], 
                           color=self.colors['diagnosis'], alpha=0.1)
        
        ax1.plot(time_steps, ground_truth_fault, 'o-', linewidth=2, 
                markersize=4, label='真实故障', color='darkred')
        ax1.plot(time_steps, diagnosis_output, 's--', linewidth=2,
                markersize=4, label='诊断输出', color=self.colors['diagnosis'])
        
        ax1.set_ylabel('故障类型', fontsize=11, fontweight='bold')
        ax1.set_yticks(range(len(fault_types)))
        ax1.set_yticklabels(fault_types, fontsize=9)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_xlim(time_steps[0], time_steps[-1])
        
        # 轨道2: Pmax与安全限值
        ax2 = fig.add_subplot(gs[1])
        
        # 安全限值和目标值带状区域
        ax2.axhline(pmax_limit, color='red', linestyle='--', linewidth=2, 
                   label=f'限值 ({pmax_limit:.0f}bar)')
        ax2.axhline(pmax_target, color='green', linestyle='--', linewidth=2,
                   label=f'目标 ({pmax_target:.0f}bar)')
        ax2.fill_between(time_steps, 0, pmax_limit, color=self.colors['pmax'],
                        alpha=0.1, label='安全区域')
        ax2.fill_between(time_steps, pmax_limit, max(pmax_values) * 1.1,
                        color='red', alpha=0.1, label='危险区域')
        
        # Pmax曲线
        ax2.plot(time_steps, pmax_values, 'o-', linewidth=2.5, markersize=4,
                color=self.colors['pmax'], label='实际Pmax')
        
        ax2.set_ylabel('Pmax [bar]', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(time_steps[0], time_steps[-1])
        
        # 轨道3: 控制动作 - VIT
        ax3 = fig.add_subplot(gs[2])
        
        ax3.bar(time_steps, vit_actions, width=0.8, color=self.colors['vit'],
               alpha=0.7, label='VIT调整')
        ax3.axhline(0, color='black', linewidth=0.8)
        
        ax3.set_ylabel('VIT调整量 [°CA]', fontsize=11, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_xlim(time_steps[0], time_steps[-1])
        
        # 轨道4: 控制动作 - 燃油系数
        ax4 = fig.add_subplot(gs[3])
        
        ax4.plot(time_steps, fuel_actions, 'o-', linewidth=2.5, markersize=4,
                color=self.colors['fuel'], label='燃油系数')
        ax4.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        ax4.set_ylabel('燃油系数', fontsize=11, fontweight='bold')
        ax4.set_xlabel('时间步', fontsize=11, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(time_steps[0], time_steps[-1])
        ax4.set_ylim(0.65, 1.05)
        
        # 总标题
        fig.suptitle('双智能体诊-控协同响应时序图', fontsize=14, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[保存] 协同响应图已保存: {save_path}")
        
        return fig
    
    def plot_training_curves(self,
                            episode_rewards: List[float],
                            diag_rewards: List[float],
                            ctrl_rewards: List[float],
                            diagnosis_accuracies: List[float],
                            detection_delays: List[float],
                            pmax_violations: List[float],
                            save_path: str = None) -> plt.Figure:
        """
        训练曲线图
        
        绘制双智能体训练过程的各项指标
        
        Args:
            episode_rewards: 每个episode的总奖励
            diag_rewards: 诊断奖励历史
            ctrl_rewards: 控制奖励历史
            diagnosis_accuracies: 诊断准确率历史
            detection_delays: 检测延迟历史
            pmax_violations: Pmax违规次数历史
            save_path: 保存路径
        
        Returns:
            fig: 图形对象
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        episodes = np.arange(len(episode_rewards))
        window = min(50, len(episode_rewards) // 10)  # 平滑窗口
        
        # 1. 总奖励
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(episodes, episode_rewards, 'o-', linewidth=1, markersize=2,
                alpha=0.5, color='lightblue', label='原始')
        if window > 1:
            smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(episodes[window-1:], smoothed, '-', linewidth=2.5,
                    color='darkblue', label=f'平滑({window})')
        ax1.set_xlabel('Episode', fontsize=10)
        ax1.set_ylabel('总奖励', fontsize=10)
        ax1.set_title('总奖励曲线', fontsize=11, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 诊断 vs 控制奖励
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(episodes, diag_rewards, 'o-', linewidth=1.5, markersize=3,
                color=self.colors['diagnosis'], label='诊断奖励', alpha=0.7)
        ax2.plot(episodes, ctrl_rewards, 's-', linewidth=1.5, markersize=3,
                color=self.colors['pmax'], label='控制奖励', alpha=0.7)
        ax2.set_xlabel('Episode', fontsize=10)
        ax2.set_ylabel('奖励', fontsize=10)
        ax2.set_title('诊断 vs 控制奖励', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 诊断准确率
        ax3 = fig.add_subplot(gs[1, 0])
        if diagnosis_accuracies:
            ax3.plot(episodes, diagnosis_accuracies, 'o-', linewidth=1.5, markersize=3,
                    color=self.colors['diagnosis'])
            if window > 1:
                smoothed_acc = np.convolve(diagnosis_accuracies, np.ones(window)/window, mode='valid')
                ax3.plot(episodes[window-1:], smoothed_acc, '-', linewidth=2.5,
                        color='darkgreen')
            ax3.axhline(0.8, color='red', linestyle='--', alpha=0.5, label='目标(80%)')
            ax3.set_ylabel('准确率', fontsize=10)
            ax3.set_ylim(0, 1.05)
        ax3.set_xlabel('Episode', fontsize=10)
        ax3.set_title('诊断准确率', fontsize=11, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 检测延迟
        ax4 = fig.add_subplot(gs[1, 1])
        if detection_delays:
            ax4.plot(range(len(detection_delays)), detection_delays, 'o-',
                    linewidth=1.5, markersize=3, color='orange')
            ax4.axhline(np.mean(detection_delays), color='red', linestyle='--',
                       label=f'平均: {np.mean(detection_delays):.1f}步')
            ax4.set_ylabel('延迟 [步]', fontsize=10)
            ax4.set_xlabel('故障次数', fontsize=10)
            ax4.set_title('故障检测延迟', fontsize=11, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Pmax违规次数
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.bar(episodes, pmax_violations, color=self.colors['pmax'], alpha=0.6)
        ax5.set_xlabel('Episode', fontsize=10)
        ax5.set_ylabel('违规次数', fontsize=10)
        ax5.set_title('Pmax超限违规', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. 学习进度指标
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        
        # 统计信息
        stats_text = f"""
        训练统计信息:
        
        总Episodes: {len(episode_rewards)}
        最终平均奖励: {np.mean(episode_rewards[-20:]):.2f}
        最高奖励: {np.max(episode_rewards):.2f}
        
        诊断性能:
        • 最终准确率: {diagnosis_accuracies[-1]:.1%} (如果有)
        • 平均检测延迟: {np.mean(detection_delays):.1f}步 (如果有)
        
        控制性能:
        • 最终平均违规: {np.mean(pmax_violations[-20:]):.1f}次
        • 总违规次数: {sum(pmax_violations):.0f}次
        """
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                family='monospace')
        
        fig.suptitle('双智能体训练曲线', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[保存] 训练曲线已保存: {save_path}")
        
        return fig
    
    def plot_performance_comparison(self,
                                   baseline_metrics: Dict,
                                   proposed_metrics: Dict,
                                   metric_names: List[str] = None,
                                   save_path: str = None) -> plt.Figure:
        """
        Baseline vs Proposed性能对比雷达图
        
        Args:
            baseline_metrics: Baseline指标字典
            proposed_metrics: Proposed指标字典
            metric_names: 指标名称列表
            save_path: 保存路径
        
        Returns:
            fig: 图形对象
        """
        # 默认指标
        if metric_names is None:
            metric_names = ['准确率', '响应速度', '安全性', '经济性', '鲁棒性']
        
        # 提取值并归一化到0-1
        baseline_vals = [baseline_metrics.get(name, 0.5) for name in metric_names]
        proposed_vals = [proposed_metrics.get(name, 0.5) for name in metric_names]
        
        # 确保所有值在0-1范围内
        baseline_vals = [np.clip(v, 0, 1) for v in baseline_vals]
        proposed_vals = [np.clip(v, 0, 1) for v in proposed_vals]
        
        # 绘制雷达图
        angles = np.linspace(0, 2*np.pi, len(metric_names), endpoint=False).tolist()
        baseline_vals += baseline_vals[:1]
        proposed_vals += proposed_vals[:1]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, baseline_vals, 'o-', linewidth=2.5, markersize=8,
               color=self.colors['baseline'], label='Baseline')
        ax.fill(angles, baseline_vals, color=self.colors['baseline'], alpha=0.15)
        
        ax.plot(angles, proposed_vals, 's-', linewidth=2.5, markersize=8,
               color=self.colors['proposed'], label='Proposed (RL诊断)')
        ax.fill(angles, proposed_vals, color=self.colors['proposed'], alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, alpha=0.3)
        
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        
        fig.suptitle('Baseline vs Proposed性能对比\n(规则诊断+RL控制 vs RL诊断+RL控制)',
                    fontsize=13, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[保存] 性能对比雷达图已保存: {save_path}")
        
        return fig
    
    def plot_confusion_matrix(self,
                             confusion_matrix: np.ndarray,
                             class_names: List[str] = None,
                             save_path: str = None) -> plt.Figure:
        """
        诊断混淆矩阵
        
        Args:
            confusion_matrix: 混淆矩阵 (n_classes x n_classes)
            class_names: 类别名称
            save_path: 保存路径
        
        Returns:
            fig: 图形对象
        """
        if class_names is None:
            class_names = ['正常', '喷油正时', '气缸泄漏', '燃油系统', '喷油器']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 归一化混淆矩阵用于颜色编码
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        
        # 添加数字和百分比
        n_classes = len(class_names)
        for i in range(n_classes):
            for j in range(n_classes):
                count = confusion_matrix[i, j]
                percentage = cm_normalized[i, j]
                text = ax.text(j, i, f'{count}\n({percentage:.1%})',
                             ha="center", va="center",
                             color="white" if percentage > 0.5 else "black",
                             fontsize=10, fontweight='bold')
        
        # 设置标签
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(class_names, fontsize=10)
        ax.set_yticklabels(class_names, fontsize=10)
        
        ax.set_xlabel('预测标签', fontsize=11, fontweight='bold')
        ax.set_ylabel('真实标签', fontsize=11, fontweight='bold')
        ax.set_title('故障诊断混淆矩阵', fontsize=12, fontweight='bold')
        
        # 颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('正规化比例', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[保存] 混淆矩阵已保存: {save_path}")
        
        return fig


if __name__ == "__main__":
    print("双智能体可视化模块")
    
    # 测试数据
    time_steps = np.arange(100)
    ground_truth = np.concatenate([np.zeros(20), np.ones(30), np.zeros(50)])
    diagnosis = np.concatenate([np.zeros(25), np.ones(25), np.zeros(50)])
    pmax = 170 + 5*np.sin(time_steps/10) + np.concatenate([np.zeros(20), 15*np.ones(30), np.zeros(50)])
    vit = -2 + np.random.randn(100) * 0.5
    fuel = 1.0 - 0.2*np.concatenate([np.zeros(20), np.ones(30), np.zeros(50)])
    
    visualizer = DualAgentVisualizer()
    
    # 测试协同响应图
    visualizer.plot_coordination_response(
        time_steps, ground_truth.astype(int), diagnosis.astype(int),
        pmax, vit, fuel
    )
    
    plt.show()
