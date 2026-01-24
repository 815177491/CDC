"""
MARL训练可视化模块
==================
包含训练过程、结果和评估的可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TrainingVisualizer:
    """
    训练过程可视化器
    
    功能：
    1. 实时训练曲线
    2. 奖励分布
    3. 损失曲线
    4. 智能体行为分析
    """
    
    def __init__(self, save_dir: str = './visualization_output/marl'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史
        self.history = {
            'episodes': [],
            'reward_diag': [],
            'reward_ctrl': [],
            'reward_total': [],
            'loss_diag_policy': [],
            'loss_diag_value': [],
            'loss_ctrl_policy': [],
            'loss_ctrl_value': [],
            'entropy_diag': [],
            'entropy_ctrl': [],
            'diag_accuracy': [],
            'ctrl_performance': [],
            'episode_length': []
        }
    
    def update(self, metrics: Dict):
        """更新训练指标"""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def plot_training_curves(self, figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        绘制训练曲线（综合视图）
        
        包含：奖励曲线、损失曲线、熵曲线、性能指标
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
        
        episodes = self.history['episodes'] if self.history['episodes'] else list(range(len(self.history['reward_diag'])))
        
        # (a) 奖励曲线
        ax1 = fig.add_subplot(gs[0, 0])
        if self.history['reward_diag']:
            ax1.plot(episodes, self.history['reward_diag'], 'b-', alpha=0.3, label='诊断奖励(原始)')
            smooth_data = self._smooth(self.history['reward_diag'])
            smooth_x = episodes[len(episodes)-len(smooth_data):]
            ax1.plot(smooth_x, smooth_data, 'b-', linewidth=2, label='诊断奖励(平滑)')
        if self.history['reward_ctrl']:
            ax1.plot(episodes, self.history['reward_ctrl'], 'r-', alpha=0.3, label='控制奖励(原始)')
            smooth_data = self._smooth(self.history['reward_ctrl'])
            smooth_x = episodes[len(episodes)-len(smooth_data):]
            ax1.plot(smooth_x, smooth_data, 'r-', linewidth=2, label='控制奖励(平滑)')
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('奖励')
        ax1.set_title('(a) 双智能体奖励曲线')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # (b) 策略损失
        ax2 = fig.add_subplot(gs[0, 1])
        if self.history['loss_diag_policy']:
            ax2.plot(episodes, self.history['loss_diag_policy'], 'b-', label='诊断策略损失')
        if self.history['loss_ctrl_policy']:
            ax2.plot(episodes, self.history['loss_ctrl_policy'], 'r-', label='控制策略损失')
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('策略损失')
        ax2.set_title('(b) 策略损失曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # (c) 价值损失
        ax3 = fig.add_subplot(gs[1, 0])
        if self.history['loss_diag_value']:
            ax3.plot(episodes, self.history['loss_diag_value'], 'b-', label='诊断价值损失')
        if self.history['loss_ctrl_value']:
            ax3.plot(episodes, self.history['loss_ctrl_value'], 'r-', label='控制价值损失')
        ax3.set_xlabel('训练轮次')
        ax3.set_ylabel('价值损失')
        ax3.set_title('(c) 价值损失曲线')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # (d) 熵曲线
        ax4 = fig.add_subplot(gs[1, 1])
        if self.history['entropy_diag']:
            ax4.plot(episodes, self.history['entropy_diag'], 'b-', label='诊断熵')
        if self.history['entropy_ctrl']:
            ax4.plot(episodes, self.history['entropy_ctrl'], 'r-', label='控制熵')
        ax4.set_xlabel('训练轮次')
        ax4.set_ylabel('策略熵')
        ax4.set_title('(d) 策略熵曲线（探索程度）')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # (e) 诊断准确率
        ax5 = fig.add_subplot(gs[2, 0])
        if self.history['diag_accuracy']:
            ax5.plot(episodes, self.history['diag_accuracy'], 'g-', alpha=0.3)
            smooth_data = self._smooth(self.history['diag_accuracy'])
            smooth_x = episodes[len(episodes)-len(smooth_data):]
            ax5.plot(smooth_x, smooth_data, 'g-', linewidth=2)
            ax5.axhline(y=0.9, color='k', linestyle='--', alpha=0.5, label='目标准确率90%')
        ax5.set_xlabel('训练轮次')
        ax5.set_ylabel('准确率')
        ax5.set_title('(e) 诊断准确率')
        ax5.set_ylim([0, 1.05])
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # (f) 控制性能
        ax6 = fig.add_subplot(gs[2, 1])
        if self.history['ctrl_performance']:
            ax6.plot(episodes, self.history['ctrl_performance'], 'm-', alpha=0.3)
            smooth_data = self._smooth(self.history['ctrl_performance'])
            smooth_x = episodes[len(episodes)-len(smooth_data):]
            ax6.plot(smooth_x, smooth_data, 'm-', linewidth=2)
        ax6.set_xlabel('训练轮次')
        ax6.set_ylabel('性能维持率')
        ax6.set_title('(f) 控制性能（Pmax维持率）')
        ax6.set_ylim([0, 1.1])
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('双智能体强化学习训练过程', fontsize=14, fontweight='bold')
        
        return fig
    
    def plot_reward_distribution(self, figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """绘制奖励分布直方图"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 诊断奖励分布
        if self.history['reward_diag']:
            axes[0].hist(self.history['reward_diag'], bins=30, color='blue', alpha=0.7, edgecolor='black')
            axes[0].axvline(np.mean(self.history['reward_diag']), color='red', linestyle='--', label=f'均值: {np.mean(self.history["reward_diag"]):.2f}')
            axes[0].set_xlabel('奖励')
            axes[0].set_ylabel('频次')
            axes[0].set_title('诊断智能体奖励分布')
            axes[0].legend()
        
        # 控制奖励分布
        if self.history['reward_ctrl']:
            axes[1].hist(self.history['reward_ctrl'], bins=30, color='red', alpha=0.7, edgecolor='black')
            axes[1].axvline(np.mean(self.history['reward_ctrl']), color='blue', linestyle='--', label=f'均值: {np.mean(self.history["reward_ctrl"]):.2f}')
            axes[1].set_xlabel('奖励')
            axes[1].set_ylabel('频次')
            axes[1].set_title('控制智能体奖励分布')
            axes[1].legend()
        
        # 总奖励分布
        if self.history['reward_total']:
            axes[2].hist(self.history['reward_total'], bins=30, color='green', alpha=0.7, edgecolor='black')
            axes[2].axvline(np.mean(self.history['reward_total']), color='purple', linestyle='--', label=f'均值: {np.mean(self.history["reward_total"]):.2f}')
            axes[2].set_xlabel('奖励')
            axes[2].set_ylabel('频次')
            axes[2].set_title('总奖励分布')
            axes[2].legend()
        
        plt.tight_layout()
        return fig
    
    def _smooth(self, data: List, window: int = 20) -> np.ndarray:
        """滑动平均平滑"""
        if len(data) < window:
            return np.array(data)
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    def save_plot(self, fig: plt.Figure, name: str):
        """保存图像"""
        fig.savefig(self.save_dir / f'{name}.png', dpi=150, bbox_inches='tight')
        fig.savefig(self.save_dir / f'{name}.pdf', bbox_inches='tight')
        print(f"已保存: {self.save_dir / name}")


class EvaluationVisualizer:
    """
    训练效果评估可视化器
    
    功能：
    1. 混淆矩阵
    2. 故障检测延迟分析
    3. 控制响应分析
    4. 对比实验结果
    """
    
    def __init__(self, save_dir: str = './visualization_output/marl'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_confusion_matrix(
        self, 
        confusion_matrix: np.ndarray,
        class_names: List[str] = ['健康', '正时故障', '泄漏故障', '燃油故障'],
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """绘制诊断混淆矩阵"""
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(confusion_matrix, cmap='Blues')
        
        # 添加colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('样本数', rotation=-90, va="bottom")
        
        # 设置刻度
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加数值标注
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                value = confusion_matrix[i, j]
                text = ax.text(j, i, f'{value}',
                              ha="center", va="center",
                              color="white" if value > confusion_matrix.max()/2 else "black")
        
        ax.set_xlabel('预测类别')
        ax.set_ylabel('真实类别')
        ax.set_title('故障诊断混淆矩阵')
        
        plt.tight_layout()
        return fig
    
    def plot_detection_delay(
        self,
        delays: Dict[str, List[float]],
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """绘制故障检测延迟分析"""
        fig, ax = plt.subplots(figsize=figsize)
        
        positions = list(range(len(delays)))
        labels = list(delays.keys())
        data = [delays[k] for k in labels]
        
        bp = ax.boxplot(data, positions=positions, patch_artist=True)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel('检测延迟（循环数）')
        ax.set_title('不同故障类型的检测延迟分布')
        ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='目标: <5 cycles')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_control_response(
        self,
        time_steps: np.ndarray,
        fault_severity: np.ndarray,
        pmax_actual: np.ndarray,
        pmax_target: float,
        timing_offset: np.ndarray,
        fuel_adj: np.ndarray,
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """绘制控制响应分析"""
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # (a) 故障严重程度
        axes[0].fill_between(time_steps, 0, fault_severity, alpha=0.5, color='red', label='故障严重程度')
        axes[0].set_ylabel('故障强度')
        axes[0].set_title('(a) 故障注入')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])
        
        # (b) Pmax响应
        axes[1].plot(time_steps, pmax_actual, 'b-', linewidth=2, label='实际Pmax')
        axes[1].axhline(y=pmax_target, color='green', linestyle='--', label=f'目标Pmax={pmax_target}')
        axes[1].fill_between(time_steps, pmax_target*0.95, pmax_target*1.05, alpha=0.2, color='green', label='±5%容差')
        axes[1].set_ylabel('Pmax [bar]')
        axes[1].set_title('(b) 性能维持效果')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        # (c) 正时补偿
        axes[2].plot(time_steps, timing_offset, 'purple', linewidth=2)
        axes[2].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        axes[2].set_ylabel('正时补偿 [deg]')
        axes[2].set_title('(c) 控制智能体-正时调整')
        axes[2].set_ylim([-6, 6])
        axes[2].grid(True, alpha=0.3)
        
        # (d) 燃油调整
        axes[3].plot(time_steps, fuel_adj, 'orange', linewidth=2)
        axes[3].axhline(y=1.0, color='gray', linestyle='-', alpha=0.5)
        axes[3].set_ylabel('燃油系数')
        axes[3].set_xlabel('时间步')
        axes[3].set_title('(d) 控制智能体-燃油调整')
        axes[3].set_ylim([0.8, 1.2])
        axes[3].grid(True, alpha=0.3)
        
        plt.suptitle('容错控制响应分析', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_comparison(
        self,
        methods: List[str],
        metrics: Dict[str, List[float]],
        figsize: Tuple[int, int] = (12, 5)
    ) -> plt.Figure:
        """绘制方法对比"""
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        
        if len(metrics) == 1:
            axes = [axes]
        
        x = np.arange(len(methods))
        width = 0.6
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f1c40f']
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            bars = axes[idx].bar(x, values, width, color=colors[:len(methods)], edgecolor='black')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(methods, rotation=45, ha='right')
            axes[idx].set_ylabel(metric_name)
            axes[idx].set_title(metric_name)
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # 添加数值标注
            for bar, val in zip(bars, values):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                              f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('方法性能对比', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig


class NetworkArchitectureVisualizer:
    """
    网络架构可视化
    """
    
    @staticmethod
    def plot_architecture(figsize: Tuple[int, int] = (16, 8)) -> plt.Figure:
        """绘制双智能体网络架构图"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # ===== 诊断智能体 =====
        ax = axes[0]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('诊断智能体网络架构', fontsize=14, fontweight='bold')
        
        # 输入层
        input_box = mpatches.FancyBboxPatch((0.5, 4), 2, 2, boxstyle="round,pad=0.1",
                                            facecolor='#3498db', edgecolor='black', linewidth=2)
        ax.add_patch(input_box)
        ax.text(1.5, 5, '观测输入\n(含控制历史)', ha='center', va='center', fontsize=10, color='white')
        
        # 编码器
        encoder_box = mpatches.FancyBboxPatch((3.5, 4), 2, 2, boxstyle="round,pad=0.1",
                                              facecolor='#9b59b6', edgecolor='black', linewidth=2)
        ax.add_patch(encoder_box)
        ax.text(4.5, 5, 'MLP编码器\n64-64', ha='center', va='center', fontsize=10, color='white')
        
        # 输出头
        heads = [
            ('故障分类\nSoftmax(4)', '#e74c3c', 8),
            ('严重程度\nBeta分布', '#2ecc71', 6),
            ('置信度\nBeta分布', '#f1c40f', 4),
            ('Critic\nV(s)', '#1abc9c', 2)
        ]
        
        for name, color, y in heads:
            head_box = mpatches.FancyBboxPatch((7, y-0.5), 2.5, 1.2, boxstyle="round,pad=0.05",
                                               facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(head_box)
            ax.text(8.25, y+0.1, name, ha='center', va='center', fontsize=9, color='white')
        
        # 连接箭头
        ax.annotate('', xy=(3.4, 5), xytext=(2.6, 5),
                   arrowprops=dict(arrowstyle='->', lw=2))
        for y in [8, 6, 4, 2]:
            ax.annotate('', xy=(6.9, y+0.1), xytext=(5.6, 5),
                       arrowprops=dict(arrowstyle='->', lw=1.5, connectionstyle="arc3,rad=0.1"))
        
        # ===== 控制智能体 =====
        ax = axes[1]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('控制智能体网络架构', fontsize=14, fontweight='bold')
        
        # 输入层
        input_box = mpatches.FancyBboxPatch((0.5, 4), 2, 2, boxstyle="round,pad=0.1",
                                            facecolor='#3498db', edgecolor='black', linewidth=2)
        ax.add_patch(input_box)
        ax.text(1.5, 5, '观测输入\n(含诊断结果)', ha='center', va='center', fontsize=10, color='white')
        
        # 编码器
        encoder_box = mpatches.FancyBboxPatch((3.5, 4), 2, 2, boxstyle="round,pad=0.1",
                                              facecolor='#9b59b6', edgecolor='black', linewidth=2)
        ax.add_patch(encoder_box)
        ax.text(4.5, 5, 'MLP编码器\n64-64', ha='center', va='center', fontsize=10, color='white')
        
        # 输出头
        heads = [
            ('正时补偿\n高斯[-5,5]°', '#e74c3c', 8),
            ('燃油调整\n高斯[0.85,1.15]', '#2ecc71', 6),
            ('保护级别\nSoftmax(4)', '#f1c40f', 4),
            ('Critic\nV(s)', '#1abc9c', 2)
        ]
        
        for name, color, y in heads:
            head_box = mpatches.FancyBboxPatch((7, y-0.5), 2.5, 1.2, boxstyle="round,pad=0.05",
                                               facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(head_box)
            ax.text(8.25, y+0.1, name, ha='center', va='center', fontsize=9, color='white')
        
        # 连接箭头
        ax.annotate('', xy=(3.4, 5), xytext=(2.6, 5),
                   arrowprops=dict(arrowstyle='->', lw=2))
        for y in [8, 6, 4, 2]:
            ax.annotate('', xy=(6.9, y+0.1), xytext=(5.6, 5),
                       arrowprops=dict(arrowstyle='->', lw=1.5, connectionstyle="arc3,rad=0.1"))
        
        plt.tight_layout()
        return fig


def demo_visualizations():
    """
    演示可视化功能（使用模拟数据）
    """
    print("生成演示可视化...")
    
    # 创建模拟训练数据
    n_episodes = 200
    np.random.seed(42)
    
    # 训练可视化
    trainer_viz = TrainingVisualizer()
    
    for ep in range(n_episodes):
        progress = ep / n_episodes
        trainer_viz.update({
            'episodes': ep,
            'reward_diag': -5 + 15 * progress + np.random.randn() * 2,
            'reward_ctrl': -3 + 10 * progress + np.random.randn() * 1.5,
            'reward_total': -8 + 25 * progress + np.random.randn() * 3,
            'loss_diag_policy': 0.5 - 0.3 * progress + np.random.rand() * 0.1,
            'loss_diag_value': 1.0 - 0.5 * progress + np.random.rand() * 0.2,
            'loss_ctrl_policy': 0.4 - 0.25 * progress + np.random.rand() * 0.1,
            'loss_ctrl_value': 0.8 - 0.4 * progress + np.random.rand() * 0.15,
            'entropy_diag': 1.5 - 0.8 * progress + np.random.rand() * 0.1,
            'entropy_ctrl': 1.2 - 0.6 * progress + np.random.rand() * 0.1,
            'diag_accuracy': 0.5 + 0.4 * progress + np.random.rand() * 0.05,
            'ctrl_performance': 0.6 + 0.35 * progress + np.random.rand() * 0.05,
            'episode_length': 150 + int(50 * progress) + np.random.randint(-20, 20)
        })
    
    # 绘制训练曲线
    fig1 = trainer_viz.plot_training_curves()
    trainer_viz.save_plot(fig1, 'training_curves')
    
    fig2 = trainer_viz.plot_reward_distribution()
    trainer_viz.save_plot(fig2, 'reward_distribution')
    
    # 评估可视化
    eval_viz = EvaluationVisualizer()
    
    # 混淆矩阵
    cm = np.array([
        [95, 2, 2, 1],
        [3, 88, 5, 4],
        [2, 4, 90, 4],
        [1, 3, 4, 92]
    ])
    fig3 = eval_viz.plot_confusion_matrix(cm)
    eval_viz.save_dir = trainer_viz.save_dir
    fig3.savefig(trainer_viz.save_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    
    # 检测延迟
    delays = {
        '正时故障': np.random.exponential(2, 50) + 1,
        '泄漏故障': np.random.exponential(3, 50) + 1,
        '燃油故障': np.random.exponential(2.5, 50) + 1
    }
    fig4 = eval_viz.plot_detection_delay(delays)
    fig4.savefig(trainer_viz.save_dir / 'detection_delay.png', dpi=150, bbox_inches='tight')
    
    # 控制响应
    t = np.arange(100)
    fault = np.zeros(100)
    fault[30:] = np.minimum((t[30:] - 30) * 0.02, 0.5)
    pmax = 150 - fault * 20 + np.random.randn(100) * 2
    timing = np.zeros(100)
    timing[35:] = fault[35:] * 3 + np.random.randn(65) * 0.3
    fuel = np.ones(100)
    fuel[35:] = 1 + fault[35:] * 0.1 + np.random.randn(65) * 0.02
    
    fig5 = eval_viz.plot_control_response(t, fault, pmax, 150, timing, fuel)
    fig5.savefig(trainer_viz.save_dir / 'control_response.png', dpi=150, bbox_inches='tight')
    
    # 方法对比
    methods = ['无控制', 'PID', '单智能体RL', '双智能体MARL']
    metrics = {
        '诊断准确率': [0.65, 0.72, 0.85, 0.93],
        '性能维持率': [0.70, 0.82, 0.88, 0.95],
        '检测延迟': [8.5, 6.2, 4.1, 2.3]
    }
    fig6 = eval_viz.plot_comparison(methods, metrics)
    fig6.savefig(trainer_viz.save_dir / 'method_comparison.png', dpi=150, bbox_inches='tight')
    
    # 网络架构
    fig7 = NetworkArchitectureVisualizer.plot_architecture()
    fig7.savefig(trainer_viz.save_dir / 'network_architecture.png', dpi=150, bbox_inches='tight')
    
    print(f"所有可视化已保存到: {trainer_viz.save_dir}")
    plt.show()


if __name__ == '__main__':
    demo_visualizations()
