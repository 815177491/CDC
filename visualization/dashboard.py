"""
交互式仪表盘
============
基于Matplotlib的简易交互式界面
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.patches import Circle
from typing import Dict, Callable, Optional
import warnings

# 设置中文字体，解决乱码问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100


class InteractiveDashboard:
    """
    交互式仪表盘
    
    布局:
    - 左侧面板: 输入滑块 (RPM, 故障程度)
    - 中间面板: P-V示功图 & 放热率曲线
    - 右侧面板: 诊断状态指示灯 & 协同修正量
    """
    
    def __init__(self, engine, diagnoser=None, controller=None):
        """
        初始化仪表盘
        
        Args:
            engine: MarineEngine0D实例
            diagnoser: FaultDiagnoser实例 (可选)
            controller: SynergyController实例 (可选)
        """
        self.engine = engine
        self.diagnoser = diagnoser
        self.controller = controller
        
        # 当前工况
        self.current_rpm = 80.0
        self.current_fault = 0.0  # 故障程度 0-100%
        
        # 创建图形
        self.fig = None
        self.axes = {}
        self.artists = {}
        self.sliders = {}
        self.buttons = {}
    
    def create_layout(self):
        """创建仪表盘布局"""
        self.fig = plt.figure(figsize=(18, 11))
        self.fig.suptitle('船用柴油机零维仿真与控诊协同仪表盘', 
                          fontsize=18, fontweight='bold', y=0.98)
        
        # 网格布局: 3列，增加间距避免重叠
        gs = self.fig.add_gridspec(3, 4, width_ratios=[1, 2, 2, 1.5],
                                    height_ratios=[1, 1, 0.25],
                                    hspace=0.4, wspace=0.4,
                                    left=0.06, right=0.96, top=0.92, bottom=0.08)
        
        # 左侧面板: 控制输入
        self.axes['controls'] = self.fig.add_subplot(gs[0:2, 0])
        self.axes['controls'].set_title('控制输入', fontsize=12, fontweight='bold', pad=10)
        self.axes['controls'].axis('off')
        
        # 中间面板: P-V图
        self.axes['pv'] = self.fig.add_subplot(gs[0, 1:3])
        self.axes['pv'].set_title('P-V 示功图', fontsize=12, fontweight='bold', pad=10)
        self.axes['pv'].set_xlabel('容积 [L]', fontsize=10)
        self.axes['pv'].set_ylabel('压力 [bar]', fontsize=10)
        self.axes['pv'].tick_params(labelsize=9)
        self.axes['pv'].grid(True, alpha=0.3)
        
        # 中间面板: 放热率
        self.axes['hr'] = self.fig.add_subplot(gs[1, 1:3])
        self.axes['hr'].set_title('燃烧放热率曲线', fontsize=12, fontweight='bold', pad=10)
        self.axes['hr'].set_xlabel('曲轴转角 [deg CA ATDC]', fontsize=10)
        self.axes['hr'].set_ylabel('放热率 [kJ/deg CA]', fontsize=10)
        self.axes['hr'].tick_params(labelsize=9)
        self.axes['hr'].grid(True, alpha=0.3)
        
        # 右侧面板: 诊断状态
        self.axes['status'] = self.fig.add_subplot(gs[0, 3])
        self.axes['status'].set_title('诊断状态', fontsize=12, fontweight='bold', pad=10)
        self.axes['status'].axis('off')
        
        # 右侧面板: 数值显示
        self.axes['values'] = self.fig.add_subplot(gs[1, 3])
        self.axes['values'].set_title('关键参数', fontsize=12, fontweight='bold', pad=10)
        self.axes['values'].axis('off')
        
        # 底部: 滑块区域
        ax_rpm = self.fig.add_subplot(gs[2, 1])
        ax_fault = self.fig.add_subplot(gs[2, 2])
        
        # 创建滑块
        self.sliders['rpm'] = Slider(
            ax_rpm, 'RPM', 40, 120, valinit=80,
            valstep=5, color='steelblue'
        )
        self.sliders['rpm'].label.set_fontsize(10)
        self.sliders['rpm'].valtext.set_fontsize(10)
        
        self.sliders['fault'] = Slider(
            ax_fault, '故障程度[%]', 0, 100, valinit=0,
            valstep=5, color='coral'
        )
        self.sliders['fault'].label.set_fontsize(10)
        self.sliders['fault'].valtext.set_fontsize(10)
        
        # 绑定事件
        self.sliders['rpm'].on_changed(self._on_rpm_change)
        self.sliders['fault'].on_changed(self._on_fault_change)
        
        # 初始化绑制
        self._init_plots()
    
    def _init_plots(self):
        """初始化图形元素"""
        # P-V曲线 (空白)
        self.artists['pv_line'], = self.axes['pv'].plot(
            [], [], 'b-', linewidth=2, label='当前工况')
        self.axes['pv'].legend()
        
        # 放热率曲线
        self.artists['hr_line'], = self.axes['hr'].plot(
            [], [], 'r-', linewidth=2)
        self.artists['burn_line'], = self.axes['hr'].plot(
            [], [], 'g--', linewidth=1.5)
        
        # 状态指示灯
        self.artists['status_light'] = Circle(
            (0.5, 0.55), 0.25, color='green',
            transform=self.axes['status'].transAxes
        )
        self.axes['status'].add_patch(self.artists['status_light'])
        self.artists['status_text'] = self.axes['status'].text(
            0.5, 0.15, '正常', ha='center', va='center', fontsize=13,
            fontweight='bold', transform=self.axes['status'].transAxes
        )
        
        # 数值显示
        self.artists['values_text'] = self.axes['values'].text(
            0.08, 0.85, '', fontsize=10, family='monospace',
            transform=self.axes['values'].transAxes,
            verticalalignment='top', linespacing=1.4
        )
    
    def _on_rpm_change(self, val):
        """RPM滑块变化回调"""
        self.current_rpm = val
        self._update_simulation()
    
    def _on_fault_change(self, val):
        """故障程度滑块变化回调"""
        self.current_fault = val
        self._update_simulation()
    
    def _update_simulation(self):
        """更新仿真并刷新显示"""
        from engine import OperatingCondition
        
        # 创建工况
        condition = OperatingCondition(
            rpm=self.current_rpm,
            p_scav=3.5e5,
            T_scav=320,
            fuel_mass=0.08  # 固定燃油量
        )
        
        # 注入故障
        if self.current_fault > 0:
            timing_offset = self.current_fault / 100 * 5  # 最大5度偏差
            self.engine.inject_timing_fault(timing_offset)
        else:
            self.engine.clear_faults()
        
        # 运行仿真
        try:
            results = self.engine.run_cycle(condition)
            self._update_pv_plot(results)
            self._update_hr_plot(results)
            self._update_status()
            self._update_values()
        except Exception as e:
            warnings.warn(f"仿真失败: {e}")
        
        self.fig.canvas.draw_idle()
    
    def _update_pv_plot(self, results: Dict):
        """更新P-V图"""
        V = results.get('volume', np.array([])) * 1000  # m³ -> L
        p = results.get('pressure', np.array([])) / 1e5  # Pa -> bar
        
        if len(V) > 0 and len(p) > 0:
            self.artists['pv_line'].set_data(V, p)
            self.axes['pv'].relim()
            self.axes['pv'].autoscale_view()
    
    def _update_hr_plot(self, results: Dict):
        """更新放热率图"""
        theta = results.get('theta_deg', np.array([]))
        x_b = results.get('burn_fraction', np.array([]))
        
        # 计算放热率
        dQ = np.zeros_like(theta)
        if len(theta) > 0:
            dQ = np.array([
                self.engine.combustion.get_heat_release_rate(th, self.engine.solver.fuel_mass)
                for th in theta
            ]) / 1000  # J -> kJ
        
        if len(theta) > 0:
            self.artists['hr_line'].set_data(theta, dQ)
            self.axes['hr'].relim()
            self.axes['hr'].autoscale_view()
            self.axes['hr'].set_xlim(-30, 90)
    
    def _update_status(self):
        """更新诊断状态"""
        Pmax = self.engine.get_pmax()
        
        if Pmax > 190:
            color = 'red'
            text = '临界!'
        elif Pmax > 180:
            color = 'orange'
            text = '警告'
        elif self.current_fault > 0:
            color = 'yellow'
            text = '故障'
        else:
            color = 'green'
            text = '正常'
        
        self.artists['status_light'].set_color(color)
        self.artists['status_text'].set_text(text)
    
    def _update_values(self):
        """更新数值显示"""
        Pmax = self.engine.get_pmax()
        Pcomp = self.engine.get_pcomp()
        Texh = self.engine.get_exhaust_temp()
        
        vit_adj = 0.0
        if self.controller and self.controller.control_history:
            vit_adj = self.controller.control_history[-1].vit_adjustment
        
        text = f"Pmax:  {Pmax:6.1f} bar\n"
        text += f"Pcomp: {Pcomp:6.1f} bar\n"
        text += f"Texh:  {Texh:6.0f} C\n"
        text += "-" * 15 + "\n"
        text += f"VIT:   {vit_adj:+5.1f} deg\n"
        text += f"RPM:   {self.current_rpm:6.1f}\n"
        text += f"Fault: {self.current_fault:5.0f} %"
        
        self.artists['values_text'].set_text(text)
    
    def show(self):
        """显示仪表盘"""
        if self.fig is None:
            self.create_layout()
        
        self._update_simulation()
        plt.show()
    
    def run_interactive(self):
        """运行交互式仪表盘"""
        plt.ion()  # 开启交互模式
        self.show()
        
        print("交互式仪表盘已启动。")
        print("使用滑块调整RPM和故障程度，观察系统响应。")
        print("关闭窗口或按Ctrl+C退出。")
        
        try:
            while plt.fignum_exists(self.fig.number):
                plt.pause(0.1)
        except KeyboardInterrupt:
            print("\n仪表盘已关闭。")
        finally:
            plt.ioff()
