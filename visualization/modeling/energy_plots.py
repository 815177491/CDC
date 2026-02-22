#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
能量平衡桑基图可视化
====================
绘制柴油机循环能量平衡桑基图（plotly优先，matplotlib备选）

Author: CDC Project
Date: 2026-02-20
"""

import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

import matplotlib
matplotlib.use('Agg')
matplotlib.set_loglevel('error')

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

from config import PLOT_CONFIG, COLORS, save_figure, get_output_path


def plot_energy_sankey(output_dir: str = None) -> str:
    """
    绘制柴油机循环能量平衡桑基图

    优先使用plotly生成并导出为SVG，若plotly不可用则使用matplotlib替代方案

    Args:
        output_dir: 输出目录

    Returns:
        输出文件路径
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        # 能量流数据
        source = [0, 0, 0]  # 燃烧放热
        target = [1, 2, 3]  # 有效功、壁面传热、排气损失
        value = [42, 28, 30]  # 百分比

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=30,
                line=dict(color='black', width=1),
                label=['燃烧放热 (100%)', '有效功 (42%)', '壁面传热 (28%)', '排气损失 (30%)'],
                color=[COLORS['primary'], COLORS['success'], COLORS['warning'], COLORS['danger']]
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=['rgba(40, 167, 69, 0.5)', 'rgba(255, 193, 7, 0.5)', 'rgba(220, 53, 69, 0.5)']
            )
        )])

        fig.update_layout(
            title=dict(text='柴油机循环能量平衡桑基图', font=dict(size=16)),
            font=dict(size=12, family='SimSun, Times New Roman'),
            width=800,
            height=500,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        # 保存为SVG
        output_path = get_output_path('modeling', 'energy_sankey.svg')
        pio.write_image(fig, output_path, format='svg')
        print(f"  [Saved] {output_path}")

        return output_path

    except ImportError:
        print("[Warning] plotly未安装，使用matplotlib替代方案")
        return _plot_energy_sankey_matplotlib(output_dir)
    except Exception as e:
        print(f"[Warning] plotly桑基图失败: {e}，使用matplotlib替代方案")
        return _plot_energy_sankey_matplotlib(output_dir)


def _plot_energy_sankey_matplotlib(output_dir: str = None) -> str:
    """
    使用matplotlib绘制能量流向图（桑基图替代方案）

    Args:
        output_dir: 输出目录

    Returns:
        输出文件路径
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 燃烧放热源框
    source_box = FancyBboxPatch((0.5, 3.5), 2.5, 3,
                                 boxstyle="round,pad=0.05,rounding_size=0.2",
                                 facecolor=COLORS['primary'], alpha=0.4,
                                 edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(source_box)
    ax.text(1.75, 5, '燃烧放热', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(1.75, 4.3, '100%', ha='center', va='center', fontsize=14, fontweight='bold')

    # 目标框
    targets = [
        {'name': '有效功', 'percent': '42%', 'y': 7.5, 'color': COLORS['success'], 'height': 2.5},
        {'name': '壁面传热', 'percent': '28%', 'y': 4.5, 'color': COLORS['warning'], 'height': 1.8},
        {'name': '排气损失', 'percent': '30%', 'y': 1.5, 'color': COLORS['danger'], 'height': 2.0},
    ]

    for target in targets:
        box = FancyBboxPatch((8.5, target['y'] - target['height']/2), 2.5, target['height'],
                              boxstyle="round,pad=0.05,rounding_size=0.2",
                              facecolor=target['color'], alpha=0.4,
                              edgecolor=target['color'], linewidth=2)
        ax.add_patch(box)
        ax.text(9.75, target['y'] + 0.2, target['name'],
                ha='center', va='center', fontsize=11, fontweight='bold')
        ax.text(9.75, target['y'] - 0.3, target['percent'],
                ha='center', va='center', fontsize=13, fontweight='bold')

        # 绘制流向箭头（梯形）
        flow_height = target['height'] * 0.8
        vertices = [
            (3.0, 5 + flow_height/4),
            (3.0, 5 - flow_height/4),
            (8.5, target['y'] - flow_height/2),
            (8.5, target['y'] + flow_height/2),
        ]
        flow = Polygon(vertices, closed=True, facecolor=target['color'],
                       alpha=0.3, edgecolor=target['color'], linewidth=1)
        ax.add_patch(flow)

    # 添加标题
    ax.text(6, 9.2, '柴油机循环能量平衡图', ha='center', va='center',
            fontsize=PLOT_CONFIG.FONT_SIZE_SUPTITLE, fontweight='bold')

    plt.tight_layout()
    output_path = save_figure(fig, 'modeling', 'energy_sankey.svg')
    plt.close(fig)

    return output_path
