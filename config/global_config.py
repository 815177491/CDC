#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全局配置模块
============
统一管理项目中所有参数，包括：
1. 绘图参数配置
2. 数据处理参数
3. 模型参数
4. 路径配置

Author: CDC Project
Date: 2026-01-28
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ============================================================================
# 路径配置
# ============================================================================
@dataclass
class PathConfig:
    """路径配置"""
    # 项目根目录
    ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 数据目录
    DATA_DIR: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
    
    # 数据子目录 - 分类存放
    DATA_RAW_DIR: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw'))
    DATA_CALIBRATION_DIR: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'calibration'))
    DATA_TRAINING_DIR: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'training'))
    DATA_SIMULATION_DIR: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'simulation'))
    
    # 可视化输出目录
    VISUALIZATION_OUTPUT_DIR: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualization_output'))
    
    # 可视化子目录
    VIS_PREPROCESSING_DIR: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualization_output', 'preprocessing'))
    VIS_CALIBRATION_DIR: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualization_output', 'calibration'))
    VIS_TRAINING_DIR: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualization_output', 'training'))
    VIS_EXPERIMENTS_DIR: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualization_output', 'experiments'))
    VIS_MODELING_DIR: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualization_output', 'modeling'))
    
    # 模型检查点目录
    CHECKPOINTS_DIR: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints'))
    
    # 实验结果目录
    EXPERIMENT_RESULTS_DIR: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiment_results'))
    
    def ensure_dirs(self):
        """确保所有目录存在"""
        dirs = [
            self.DATA_DIR,
            self.DATA_RAW_DIR,
            self.DATA_CALIBRATION_DIR,
            self.DATA_TRAINING_DIR,
            self.DATA_SIMULATION_DIR,
            self.VISUALIZATION_OUTPUT_DIR,
            self.VIS_PREPROCESSING_DIR,
            self.VIS_CALIBRATION_DIR,
            self.VIS_TRAINING_DIR,
            self.VIS_EXPERIMENTS_DIR,
            self.VIS_MODELING_DIR,
            self.CHECKPOINTS_DIR,
            self.EXPERIMENT_RESULTS_DIR,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)


# ============================================================================
# 绘图配置
# ============================================================================
@dataclass
class PlotConfig:
    """
    绘图配置参数
    
    标准要求：
    - 输出格式：SVG（文字为可编辑文字，不转换成路径）
    - 中文字体：宋体 (SimSun)
    - 英文字体：Times New Roman
    - 刻度标签与轴标签：14pt
    - 图例及图中文字：12pt
    """
    # 字体配置
    FONT_CHINESE: str = 'SimSun'                    # 中文字体：宋体
    FONT_ENGLISH: str = 'Times New Roman'           # 英文字体：新罗马
    FONT_MONOSPACE: str = 'Courier New'             # 等宽字体
    
    # 字号配置
    FONT_SIZE_TICK: int = 14                        # 刻度标签字号
    FONT_SIZE_LABEL: int = 14                       # 轴标签字号
    FONT_SIZE_LEGEND: int = 12                      # 图例字号
    FONT_SIZE_TEXT: int = 12                        # 图中文字字号
    FONT_SIZE_TITLE: int = 12                       # 标题字号（子图标题）
    FONT_SIZE_SUPTITLE: int = 14                    # 总标题字号
    
    # 输出格式
    OUTPUT_FORMAT: str = 'svg'                      # 输出格式
    SVG_FONTTYPE: str = 'none'                      # SVG文字保持可编辑
    
    # DPI设置
    FIGURE_DPI: int = 100                           # 图形DPI
    SAVEFIG_DPI: int = 150                          # 保存DPI
    
    # 默认图形大小
    FIGURE_SIZE_SINGLE: tuple = (10, 8)             # 单图大小
    FIGURE_SIZE_DOUBLE: tuple = (16, 8)             # 双列图大小
    FIGURE_SIZE_LARGE: tuple = (16, 12)             # 大图大小
    
    # 配色方案
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        'primary': '#2E86AB',      # 主色：蓝色
        'secondary': '#A23B72',    # 次色：紫红色
        'success': '#28A745',      # 成功：绿色
        'warning': '#FFC107',      # 警告：黄色
        'danger': '#DC3545',       # 危险：红色
        'info': '#17A2B8',         # 信息：青色
        'dark': '#343A40',         # 深灰
        'light': '#F8F9FA',        # 浅灰
        'orange': '#FF8C00',       # 橙色
        'purple': '#6F42C1',       # 紫色
        'teal': '#20C997',         # 青绿色
        'pink': '#E83E8C',         # 粉色
    })
    
    # 线条样式
    LINE_WIDTH: float = 1.5                         # 默认线宽
    LINE_WIDTH_THICK: float = 2.0                   # 粗线宽
    LINE_WIDTH_THIN: float = 1.0                    # 细线宽
    
    # 标记样式
    MARKER_SIZE: int = 6                            # 标记大小
    MARKER_SIZE_LARGE: int = 10                     # 大标记
    MARKER_SIZE_SMALL: int = 4                      # 小标记
    
    # 网格样式
    GRID_ALPHA: float = 0.3                         # 网格透明度
    GRID_LINESTYLE: str = '-'                       # 网格线型
    
    # 图例配置
    LEGEND_LOC: str = 'best'                        # 图例位置
    LEGEND_FRAMEALPHA: float = 0.9                  # 图例框透明度
    
    # 坐标轴配置
    AXIS_LABEL_PAD: int = 10                        # 轴标签间距
    TICK_DIRECTION: str = 'in'                      # 刻度方向


# ============================================================================
# 数据处理配置
# ============================================================================
@dataclass
class DataConfig:
    """数据处理参数配置"""
    # 稳态检测参数
    STEADY_STATE_WINDOW: int = 60                   # 滑动窗口大小
    STEADY_STATE_RPM_TOLERANCE: float = 1.0         # RPM容差
    STEADY_STATE_MIN_DURATION: int = 50             # 最小稳态持续点数
    
    # 异常值检测参数
    OUTLIER_Z_THRESHOLD: float = 3.0                # Z-score阈值
    OUTLIER_IQR_FACTOR: float = 1.5                 # IQR因子
    
    # 数据采样参数
    DEFAULT_SAMPLE_SIZE: int = 10000                # 默认采样大小
    
    # 标准化参数
    NORMALIZE_METHOD: str = 'standard'              # 标准化方法: 'standard', 'minmax'


# ============================================================================
# 发动机配置
# ============================================================================
@dataclass
class EngineConfig:
    """
    发动机核心配置参数
    
    此配置被 MarineEngine0D 和 PINN 网络共享，
    确保物理约束与仿真环境的一致性。
    """
    # 几何参数
    bore: float = 0.620              # 气缸直径 [m]
    stroke: float = 2.658            # 活塞行程 [m]
    n_cylinders: int = 6             # 气缸数量
    compression_ratio: float = 13.5  # 有效压缩比
    con_rod_ratio: float = 4.0       # 连杆比
    
    # 热力学参数
    gamma: float = 1.35              # 比热比
    R: float = 287.0                 # 气体常数 [J/(kg·K)]
    
    # 基准值（健康状态）- 用于诊断
    Pmax_base: float = 150.0         # 基准最大压力 [bar]
    Pcomp_base: float = 120.0        # 基准压缩压力 [bar]
    Texh_base: float = 673.15        # 基准排温 [K] (400°C = 673.15K)
    
    # 物理约束权重 - 用于PINN训练
    lambda_physics: float = 0.1
    lambda_consistency: float = 0.05


# ============================================================================
# 校准参数配置
# ============================================================================
@dataclass
class CalibrationConfig:
    """
    校准参数配置
    
    存储校准后的模型参数，由run_calibration.py更新，
    供其他模块统一读取。
    """
    # 压缩段参数
    compression_ratio: float = 15.55       # 有效压缩比
    
    # 燃烧段参数 (Wiebe参数)
    injection_timing: float = 10.0         # 喷油正时 [deg BTDC]
    diffusion_duration: float = 57.9       # 扩散燃烧持续期 [deg]
    diffusion_shape: float = 0.30          # 扩散燃烧形状因子
    
    # 传热段参数
    C_woschni: float = 130.0               # Woschni传热系数
    
    # 校准误差记录
    error_Pcomp: float = 0.14              # Pcomp误差 [%]
    error_Pmax: float = 10.97              # Pmax误差 [%]
    error_Texh: float = 31.23              # 排温误差 [%]
    
    # 校准时间戳
    calibration_timestamp: str = "2026-01-28"
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'compression_ratio': self.compression_ratio,
            'injection_timing': self.injection_timing,
            'diffusion_duration': self.diffusion_duration,
            'diffusion_shape': self.diffusion_shape,
            'C_woschni': self.C_woschni,
            'error_Pcomp': self.error_Pcomp,
            'error_Pmax': self.error_Pmax,
            'error_Texh': self.error_Texh,
            'calibration_timestamp': self.calibration_timestamp,
        }
    
    def update_from_dict(self, params: Dict):
        """从字典更新参数"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)


# ============================================================================
# 运行时配置
# ============================================================================
@dataclass
class RuntimeConfig:
    """
    运行时行为配置
    
    集中管理各脚本的运行时开关和默认参数，
    避免在各个脚本中分散硬编码。
    """
    # 数据源开关
    USE_MOCK_DATA: bool = True                     # True=模拟数据, False=校准结果数据
    
    # 校准相关
    CALIBRATION_N_POINTS: int = 10                  # 校准使用的工况点数量
    CALIBRATION_DATA_FILE: str = 'calibration_data.csv'  # 校准数据文件名


# ============================================================================
# 训练配置
# ============================================================================
@dataclass
class TrainingConfig:
    """训练参数配置"""
    # 学习率
    LEARNING_RATE: float = 3e-4
    LEARNING_RATE_CRITIC: float = 1e-3
    
    # 批次大小
    BATCH_SIZE: int = 256
    
    # 训练轮数
    NUM_EPOCHS: int = 100
    
    # 折扣因子
    GAMMA: float = 0.99
    
    # GAE参数
    GAE_LAMBDA: float = 0.95
    
    # PPO裁剪参数
    CLIP_EPSILON: float = 0.2
    
    # 熵系数
    ENTROPY_COEF: float = 0.01
    
    # 值函数系数
    VALUE_COEF: float = 0.5
    
    # 最大梯度范数
    MAX_GRAD_NORM: float = 0.5
    
    # 保存间隔
    SAVE_INTERVAL: int = 10


# ============================================================================
# 全局配置实例
# ============================================================================
# 创建全局配置实例
PATH_CONFIG = PathConfig()
PLOT_CONFIG = PlotConfig()
DATA_CONFIG = DataConfig()
ENGINE_CONFIG = EngineConfig()
CALIBRATION_CONFIG = CalibrationConfig()
TRAINING_CONFIG = TrainingConfig()
RUNTIME_CONFIG = RuntimeConfig()


# ============================================================================
# Matplotlib全局设置函数
# ============================================================================
def setup_matplotlib_style():
    """
    设置matplotlib全局样式
    
    调用此函数以应用全局绘图标准：
    - SVG格式输出（文字可编辑）
    - 中文宋体，英文Times New Roman
    - 刻度标签与轴标签14pt
    - 图例及图中文字12pt
    """
    import warnings
    # 抑制matplotlib字体相关警告
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    warnings.filterwarnings('ignore', message='.*Glyph.*')
    warnings.filterwarnings('ignore', message='.*font.*')
    
    config = PLOT_CONFIG
    
    # 使用非交互式后端
    matplotlib.use('Agg')
    
    # 字体设置
    plt.rcParams['font.sans-serif'] = [config.FONT_CHINESE, 'DejaVu Sans']
    plt.rcParams['font.serif'] = [config.FONT_ENGLISH, config.FONT_CHINESE]
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.monospace'] = [config.FONT_MONOSPACE]
    # 使用ASCII减号而非Unicode减号，避免字体缺失警告
    plt.rcParams['axes.unicode_minus'] = False
    
    # 字号设置
    plt.rcParams['font.size'] = config.FONT_SIZE_TEXT
    plt.rcParams['axes.titlesize'] = config.FONT_SIZE_TITLE
    plt.rcParams['axes.labelsize'] = config.FONT_SIZE_LABEL
    plt.rcParams['xtick.labelsize'] = config.FONT_SIZE_TICK
    plt.rcParams['ytick.labelsize'] = config.FONT_SIZE_TICK
    plt.rcParams['legend.fontsize'] = config.FONT_SIZE_LEGEND
    plt.rcParams['figure.titlesize'] = config.FONT_SIZE_SUPTITLE
    
    # DPI设置
    plt.rcParams['figure.dpi'] = config.FIGURE_DPI
    plt.rcParams['savefig.dpi'] = config.SAVEFIG_DPI
    
    # SVG设置 - 保持文字可编辑
    plt.rcParams['svg.fonttype'] = config.SVG_FONTTYPE
    
    # 坐标轴设置
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.axisbelow'] = True
    
    # 刻度设置
    plt.rcParams['xtick.direction'] = config.TICK_DIRECTION
    plt.rcParams['ytick.direction'] = config.TICK_DIRECTION
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['ytick.major.size'] = 5
    
    # 网格设置
    plt.rcParams['grid.alpha'] = config.GRID_ALPHA
    plt.rcParams['grid.linestyle'] = config.GRID_LINESTYLE
    
    # 图例设置
    plt.rcParams['legend.framealpha'] = config.LEGEND_FRAMEALPHA
    plt.rcParams['legend.edgecolor'] = 'gray'
    
    # 尝试加载系统字体
    _load_system_fonts()
    
    print("[PlotConfig] Matplotlib全局样式已设置")
    print(f"  - 中文字体: {config.FONT_CHINESE}")
    print(f"  - 英文字体: {config.FONT_ENGLISH}")
    print(f"  - 刻度/轴标签: {config.FONT_SIZE_TICK}pt")
    print(f"  - 图例/文字: {config.FONT_SIZE_LEGEND}pt")
    print(f"  - 输出格式: {config.OUTPUT_FORMAT.upper()}")


def _load_system_fonts():
    """加载系统字体"""
    try:
        # Windows字体路径
        font_paths = [
            'C:\\Windows\\Fonts\\simsun.ttc',    # 宋体
            'C:\\Windows\\Fonts\\times.ttf',     # Times New Roman
            'C:\\Windows\\Fonts\\timesbd.ttf',   # Times New Roman Bold
            'C:\\Windows\\Fonts\\timesi.ttf',    # Times New Roman Italic
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                fm.fontManager.addfont(font_path)
    except Exception as e:
        print(f"[Warning] 字体加载失败: {e}")


def get_output_path(category: str, filename: str) -> str:
    """
    获取可视化输出路径
    
    Args:
        category: 类别 ('preprocessing', 'calibration', 'training', 'experiments', 'modeling')
        filename: 文件名（不含路径）
        
    Returns:
        完整的输出文件路径
    """
    category_dirs = {
        'preprocessing': PATH_CONFIG.VIS_PREPROCESSING_DIR,
        'calibration': PATH_CONFIG.VIS_CALIBRATION_DIR,
        'training': PATH_CONFIG.VIS_TRAINING_DIR,
        'experiments': PATH_CONFIG.VIS_EXPERIMENTS_DIR,
        'modeling': PATH_CONFIG.VIS_MODELING_DIR,
    }
    
    output_dir = category_dirs.get(category, PATH_CONFIG.VISUALIZATION_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保文件扩展名为svg
    if not filename.lower().endswith('.svg'):
        filename = os.path.splitext(filename)[0] + '.svg'
    
    return os.path.join(output_dir, filename)


def save_figure(fig, category: str, filename: str, **kwargs):
    """
    保存图形到指定类别目录
    
    Args:
        fig: matplotlib Figure对象
        category: 类别 ('preprocessing', 'calibration', 'training', 'experiments', 'modeling')
        filename: 文件名
        **kwargs: 传递给savefig的其他参数
    """
    output_path = get_output_path(category, filename)
    
    # 默认保存参数
    save_kwargs = {
        'format': PLOT_CONFIG.OUTPUT_FORMAT,
        'bbox_inches': 'tight',
        'facecolor': 'white',
        'edgecolor': 'none',
    }
    save_kwargs.update(kwargs)
    
    fig.savefig(output_path, **save_kwargs)
    print(f"  [Saved] {output_path}")
    
    return output_path


# ============================================================================
# 便捷导入
# ============================================================================
# 配色方案快捷访问
COLORS = PLOT_CONFIG.COLORS


# ============================================================================
# 初始化
# ============================================================================
# 确保输出目录存在
PATH_CONFIG.ensure_dirs()
