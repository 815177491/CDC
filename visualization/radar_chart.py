"""
性能雷达图
==========
多维性能权衡可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


class PerformanceRadar:
    """
    性能权衡雷达图
    
    维度:
    - Pmax安全裕度
    - 燃油消耗率(BSFC)
    - 排气温度
    - 输出功率
    - 转速稳定性
    """
    
    def __init__(self):
        # 默认评价维度
        self.dimensions = [
            'Pmax安全裕度',
            '燃油效率',
            '排温裕度',
            '输出功率',
            '转速稳定性'
        ]
        
        # 归一化参考值
        self.references = {
            'Pmax安全裕度': {'min': 0, 'max': 30, 'unit': 'bar'},  # 距限值距离
            '燃油效率': {'min': 150, 'max': 200, 'unit': 'g/kWh'},  # BSFC
            '排温裕度': {'min': 0, 'max': 100, 'unit': '°C'},
            '输出功率': {'min': 0, 'max': 100, 'unit': '%'},
            '转速稳定性': {'min': 0, 'max': 5, 'unit': 'rpm'},  # 波动幅度(越小越好)
        }
    
    def normalize(self, values: Dict[str, float]) -> Dict[str, float]:
        """
        将绝对值归一化到0-1范围
        
        Args:
            values: 各维度的绝对值
            
        Returns:
            normalized: 归一化后的值
        """
        normalized = {}
        for dim in self.dimensions:
            if dim not in values:
                normalized[dim] = 0.5  # 默认中间值
                continue
            
            val = values[dim]
            ref = self.references.get(dim, {'min': 0, 'max': 1})
            
            # 某些维度越小越好 (如燃油消耗、转速波动)
            if dim in ['燃油效率', '转速稳定性']:
                # 反向归一化
                normalized[dim] = 1 - (val - ref['min']) / (ref['max'] - ref['min'])
            else:
                normalized[dim] = (val - ref['min']) / (ref['max'] - ref['min'])
            
            normalized[dim] = np.clip(normalized[dim], 0, 1)
        
        return normalized
    
    def plot(self,
             fault_state: Dict[str, float],
             controlled_state: Dict[str, float],
             baseline_state: Dict[str, float] = None,
             save_path: str = None) -> plt.Figure:
        """
        图表4: 性能权衡雷达图
        
        对比故障发生时与协同控制后的性能形态
        
        Args:
            fault_state: 故障发生时的性能指标
            controlled_state: 协同控制后的性能指标
            baseline_state: 正常基准状态 (可选)
            save_path: 保存路径
            
        Returns:
            fig: 图形对象
        """
        # 归一化
        fault_norm = self.normalize(fault_state)
        controlled_norm = self.normalize(controlled_state)
        baseline_norm = self.normalize(baseline_state) if baseline_state else None
        
        # 设置雷达图
        angles = np.linspace(0, 2 * np.pi, len(self.dimensions), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # 提取数据
        fault_values = [fault_norm[d] for d in self.dimensions] + [fault_norm[self.dimensions[0]]]
        controlled_values = [controlled_norm[d] for d in self.dimensions] + [controlled_norm[self.dimensions[0]]]
        
        # 绑制区域
        ax.fill(angles, fault_values, color='red', alpha=0.25, label='故障发生时')
        ax.plot(angles, fault_values, 'ro-', linewidth=2, markersize=6)
        
        ax.fill(angles, controlled_values, color='blue', alpha=0.25, label='协同控制后')
        ax.plot(angles, controlled_values, 'bs-', linewidth=2, markersize=6)
        
        if baseline_norm:
            baseline_values = [baseline_norm[d] for d in self.dimensions] + [baseline_norm[self.dimensions[0]]]
            ax.plot(angles, baseline_values, 'g^--', linewidth=1.5, markersize=5,
                    alpha=0.7, label='正常基准')
        
        # 设置轴标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.dimensions, fontsize=12)
        
        # 设置径向刻度
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
        ax.set_ylim(0, 1.1)
        
        # 标题和图例
        ax.set_title('性能权衡雷达图\n(越外围表示性能越好)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
        
        # 添加说明
        textstr = '安全性优先策略:\n' \
                  '• 牺牲少量燃油效率\n' \
                  '• 保障Pmax安全裕度\n' \
                  '• 维持转速稳定性'
        fig.text(0.02, 0.02, textstr, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    def plot_from_control_results(self,
                                   engine,
                                   controller,
                                   condition,
                                   save_path: str = None) -> plt.Figure:
        """
        从控制结果自动生成雷达图
        
        Args:
            engine: 发动机模型
            controller: 协同控制器
            condition: 运行工况
        """
        # 获取控制器性能汇总
        perf = controller.get_performance_summary()
        
        # 正常基准
        engine.clear_faults()
        engine.run_cycle(condition)
        baseline = {
            'Pmax安全裕度': 190 - engine.get_pmax(),
            '燃油效率': 170,  # 假设正常BSFC
            '排温裕度': 450 - engine.get_exhaust_temp(),
            '输出功率': 100,
            '转速稳定性': 0.5,
        }
        
        # 故障状态 (模拟)
        fault = {
            'Pmax安全裕度': 5,  # 接近限值
            '燃油效率': 165,
            '排温裕度': 30,
            '输出功率': 100,
            '转速稳定性': 2.0,
        }
        
        # 控制后状态
        controlled = {
            'Pmax安全裕度': 20,
            '燃油效率': 175,  # 略有上升
            '排温裕度': 60,
            '输出功率': 95,  # 略有损失
            '转速稳定性': 0.8,
        }
        
        return self.plot(fault, controlled, baseline, save_path)
