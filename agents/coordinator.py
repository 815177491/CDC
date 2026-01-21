"""
协调智能体
==========
管理诊断智能体和控制智能体的协同工作

职责:
1. 消息路由与分发
2. 冲突检测与解决
3. 系统状态管理
4. 性能监控与优化
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum, auto
from collections import deque
import time

from .base_agent import Agent, AgentMessage, MessageType
from .diagnosis_agent import DiagnosisAgent, DiagnosisResult, DiagnosisState
from .control_agent import ControlAgent, ControlAction, ControlMode

import sys
sys.path.append('..')
from diagnosis.fault_injector import FaultType


class SystemState(Enum):
    """系统整体状态"""
    INITIALIZING = auto()
    RUNNING = auto()
    FAULT_HANDLING = auto()
    DEGRADED = auto()
    EMERGENCY = auto()
    SHUTDOWN = auto()


class ConflictType(Enum):
    """冲突类型"""
    NONE = auto()
    DIAGNOSIS_CONTROL_MISMATCH = auto()  # 诊断与控制建议不一致
    SAFETY_PERFORMANCE_TRADEOFF = auto()  # 安全与性能的权衡
    RESOURCE_CONTENTION = auto()          # 资源竞争
    TIMING_CONFLICT = auto()              # 时序冲突


@dataclass
class ConflictReport:
    """冲突报告"""
    conflict_type: ConflictType
    description: str
    diagnosis_recommendation: str
    control_intention: str
    resolution: str
    timestamp: float


@dataclass
class SystemDecision:
    """系统级决策"""
    timestamp: float
    system_state: SystemState
    diagnosis_result: DiagnosisResult
    control_action: ControlAction
    conflict_resolved: bool = False
    conflict_report: Optional[ConflictReport] = None
    coordination_overhead_ms: float = 0.0


class ConflictResolver:
    """
    冲突解决器
    
    解决诊断智能体和控制智能体之间的冲突
    """
    
    def __init__(self):
        # 冲突解决策略权重
        self.safety_weight = 0.6
        self.performance_weight = 0.3
        self.stability_weight = 0.1
        
        # 冲突历史
        self.conflict_history: List[ConflictReport] = []
    
    def detect_conflict(self, diagnosis: DiagnosisResult, 
                        control_action: ControlAction) -> Tuple[ConflictType, str]:
        """
        检测冲突
        
        Args:
            diagnosis: 诊断结果
            control_action: 控制动作
            
        Returns:
            (conflict_type, description)
        """
        # 情况1: 诊断建议紧急处理，但控制器处于正常模式
        if (diagnosis.diagnosis_state == DiagnosisState.CRITICAL and 
            control_action.mode == ControlMode.NORMAL):
            return (ConflictType.DIAGNOSIS_CONTROL_MISMATCH,
                    "诊断器报告临界故障，但控制器处于正常模式")
        
        # 情况2: 控制器要降功，但诊断器认为系统健康
        if (control_action.power_limit < 0.9 and 
            diagnosis.diagnosis_state == DiagnosisState.HEALTHY):
            return (ConflictType.SAFETY_PERFORMANCE_TRADEOFF,
                    "控制器请求降功，但诊断器认为系统健康")
        
        # 情况3: VIT调整方向与故障类型不匹配
        if diagnosis.fault_type == FaultType.INJECTION_TIMING:
            # 正时提前应该减小VIT，正时滞后应该增大VIT
            r_Pmax = diagnosis.residuals.get('Pmax', 0)
            if r_Pmax > 0 and control_action.vit_adjustment > 0:
                return (ConflictType.DIAGNOSIS_CONTROL_MISMATCH,
                        "Pmax过高但VIT继续提前，可能加剧问题")
        
        return (ConflictType.NONE, "")
    
    def resolve(self, conflict_type: ConflictType,
                diagnosis: DiagnosisResult,
                control_action: ControlAction,
                observation: Dict[str, float]) -> Tuple[ControlAction, ConflictReport]:
        """
        解决冲突
        
        Args:
            conflict_type: 冲突类型
            diagnosis: 诊断结果
            control_action: 原控制动作
            observation: 当前观测
            
        Returns:
            (resolved_action, conflict_report)
        """
        report = ConflictReport(
            conflict_type=conflict_type,
            description="",
            diagnosis_recommendation=diagnosis.recommendation,
            control_intention=control_action.message,
            resolution="",
            timestamp=time.time()
        )
        
        resolved_action = control_action
        
        if conflict_type == ConflictType.DIAGNOSIS_CONTROL_MISMATCH:
            # 安全优先: 以诊断结果为准
            if diagnosis.diagnosis_state == DiagnosisState.CRITICAL:
                resolved_action = ControlAction(
                    timestamp=control_action.timestamp,
                    mode=ControlMode.EMERGENCY,
                    vit_adjustment=-8.0,  # 最大滞后
                    fuel_adjustment=-20.0,
                    power_limit=0.8,
                    cylinder_mask=control_action.cylinder_mask,
                    message="冲突解决: 升级为紧急模式",
                    action_source="CONFLICT_RESOLVER"
                )
                report.resolution = "升级控制模式以匹配诊断严重程度"
            else:
                # 调整VIT方向
                r_Pmax = diagnosis.residuals.get('Pmax', 0)
                new_vit = -abs(control_action.vit_adjustment) if r_Pmax > 0 else abs(control_action.vit_adjustment)
                resolved_action = ControlAction(
                    timestamp=control_action.timestamp,
                    mode=control_action.mode,
                    vit_adjustment=new_vit,
                    fuel_adjustment=control_action.fuel_adjustment,
                    power_limit=control_action.power_limit,
                    cylinder_mask=control_action.cylinder_mask,
                    message=f"冲突解决: VIT调整为{new_vit:+.1f}°",
                    action_source="CONFLICT_RESOLVER"
                )
                report.resolution = "调整VIT方向以匹配故障类型"
        
        elif conflict_type == ConflictType.SAFETY_PERFORMANCE_TRADEOFF:
            # 逐步恢复性能
            Pmax = observation.get('Pmax', 170)
            if Pmax < 175:  # 安全余量充足
                new_power_limit = min(control_action.power_limit + 0.05, 1.0)
                resolved_action = ControlAction(
                    timestamp=control_action.timestamp,
                    mode=ControlMode.NORMAL,
                    vit_adjustment=control_action.vit_adjustment * 0.5,
                    fuel_adjustment=control_action.fuel_adjustment * 0.5,
                    power_limit=new_power_limit,
                    cylinder_mask=control_action.cylinder_mask,
                    message="冲突解决: 逐步恢复性能",
                    action_source="CONFLICT_RESOLVER"
                )
                report.resolution = "诊断健康，逐步恢复功率限制"
            else:
                report.resolution = "保持当前控制策略，等待进一步观察"
        
        report.description = f"检测到{conflict_type.name}冲突"
        self.conflict_history.append(report)
        
        return resolved_action, report


class MessageBroker:
    """
    消息代理
    
    负责智能体间的消息路由和分发
    """
    
    def __init__(self):
        self.message_queue: List[AgentMessage] = []
        self.subscribers: Dict[MessageType, List[str]] = {}
        self.message_history: deque = deque(maxlen=1000)
    
    def subscribe(self, agent_name: str, msg_types: List[MessageType]) -> None:
        """订阅消息类型"""
        for msg_type in msg_types:
            if msg_type not in self.subscribers:
                self.subscribers[msg_type] = []
            if agent_name not in self.subscribers[msg_type]:
                self.subscribers[msg_type].append(agent_name)
    
    def publish(self, msg: AgentMessage) -> None:
        """发布消息"""
        self.message_queue.append(msg)
        self.message_history.append(msg)
    
    def route(self, agents: Dict[str, Agent]) -> int:
        """
        路由消息到目标智能体
        
        Returns:
            routed_count: 路由的消息数量
        """
        routed = 0
        
        while self.message_queue:
            msg = self.message_queue.pop(0)
            
            if msg.receiver:
                # 点对点消息
                if msg.receiver in agents:
                    agents[msg.receiver].receive_message(msg)
                    routed += 1
            else:
                # 广播消息
                subscribers = self.subscribers.get(msg.msg_type, [])
                for agent_name in subscribers:
                    if agent_name in agents and agent_name != msg.sender:
                        agents[agent_name].receive_message(msg)
                        routed += 1
        
        return routed
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取消息统计"""
        type_counts = {}
        for msg in self.message_history:
            type_name = msg.msg_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return {
            'total_messages': len(self.message_history),
            'pending_messages': len(self.message_queue),
            'type_distribution': type_counts,
        }


class CoordinatorAgent(Agent):
    """
    协调智能体
    
    系统级协调器，管理诊断和控制智能体的协同工作
    """
    
    def __init__(self, engine, diagnosis_agent: DiagnosisAgent = None,
                 control_agent: ControlAgent = None, name: str = "Coordinator"):
        """
        Args:
            engine: 发动机模型
            diagnosis_agent: 诊断智能体 (如果为None则自动创建)
            control_agent: 控制智能体 (如果为None则自动创建)
            name: 协调器名称
        """
        super().__init__(name=name, engine=engine)
        
        # 创建或使用传入的智能体
        self.diagnosis_agent = diagnosis_agent or DiagnosisAgent(engine)
        self.control_agent = control_agent or ControlAgent(engine)
        
        # 智能体注册表
        self.agents: Dict[str, Agent] = {
            self.diagnosis_agent.name: self.diagnosis_agent,
            self.control_agent.name: self.control_agent,
            self.name: self,
        }
        
        # 消息代理
        self.message_broker = MessageBroker()
        
        # 设置消息订阅
        self.message_broker.subscribe(
            self.control_agent.name,
            [MessageType.DIAGNOSIS_RESULT, MessageType.FAULT_ALERT]
        )
        self.message_broker.subscribe(
            self.diagnosis_agent.name,
            [MessageType.CONTROL_FEEDBACK]
        )
        self.message_broker.subscribe(
            self.name,
            [MessageType.FAULT_ALERT, MessageType.MODE_CHANGE]
        )
        
        # 冲突解决器
        self.conflict_resolver = ConflictResolver()
        
        # 系统状态
        self.system_state = SystemState.INITIALIZING
        
        # 决策历史
        self.decision_history: List[SystemDecision] = []
        
        # 性能指标
        self.state.performance_metrics = {
            'total_steps': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'avg_coordination_time_ms': 0.0,
        }
    
    def perceive(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """感知环境状态"""
        return {
            'observation': observation,
            'system_state': self.system_state,
            'diagnosis_state': self.diagnosis_agent.current_state,
            'control_mode': self.control_agent.mode,
        }
    
    def decide(self, perception: Dict[str, Any]) -> SystemDecision:
        """
        系统级决策
        
        协调诊断和控制智能体，解决潜在冲突
        """
        start_time = time.time()
        observation = perception['observation']
        
        # Step 1: 诊断智能体执行诊断
        diagnosis_result = self.diagnosis_agent.diagnose(
            observation, 
            self.state.last_update
        )
        
        # Step 2: 收集诊断智能体的输出消息
        for msg in self.diagnosis_agent.outbox:
            self.message_broker.publish(msg)
        self.diagnosis_agent.outbox.clear()
        
        # Step 3: 路由消息
        self.message_broker.route(self.agents)
        
        # Step 4: 控制智能体执行控制
        control_action = self.control_agent.update(
            observation,
            self.state.last_update
        )
        
        # Step 5: 收集控制智能体的输出消息
        for msg in self.control_agent.outbox:
            self.message_broker.publish(msg)
        self.control_agent.outbox.clear()
        
        # Step 6: 冲突检测
        conflict_type, conflict_desc = self.conflict_resolver.detect_conflict(
            diagnosis_result,
            control_action
        )
        
        # Step 7: 冲突解决
        conflict_report = None
        conflict_resolved = False
        
        if conflict_type != ConflictType.NONE:
            self.state.performance_metrics['conflicts_detected'] += 1
            
            control_action, conflict_report = self.conflict_resolver.resolve(
                conflict_type,
                diagnosis_result,
                control_action,
                observation
            )
            conflict_resolved = True
            self.state.performance_metrics['conflicts_resolved'] += 1
        
        # Step 8: 更新系统状态
        self._update_system_state(diagnosis_result, control_action)
        
        # 计算协调开销
        coordination_time = (time.time() - start_time) * 1000
        
        decision = SystemDecision(
            timestamp=self.state.last_update,
            system_state=self.system_state,
            diagnosis_result=diagnosis_result,
            control_action=control_action,
            conflict_resolved=conflict_resolved,
            conflict_report=conflict_report,
            coordination_overhead_ms=coordination_time
        )
        
        return decision
    
    def act(self, decision: SystemDecision) -> Dict[str, Any]:
        """执行系统决策"""
        # 记录决策
        self.decision_history.append(decision)
        
        # 更新统计
        self.state.performance_metrics['total_steps'] += 1
        n = self.state.performance_metrics['total_steps']
        old_avg = self.state.performance_metrics['avg_coordination_time_ms']
        self.state.performance_metrics['avg_coordination_time_ms'] = (
            (old_avg * (n - 1) + decision.coordination_overhead_ms) / n
        )
        
        # 如果发生冲突，广播冲突解决消息
        if decision.conflict_resolved:
            self.send_message(
                msg_type=MessageType.CONFLICT_RESOLUTION,
                receiver=None,
                payload={
                    'conflict_report': decision.conflict_report,
                    'resolved_action': decision.control_action,
                },
                priority=9
            )
        
        return {
            'decision': decision,
            'system_state': self.system_state,
        }
    
    def step(self, observation: Dict[str, Any], 
             timestamp: float = 0.0) -> SystemDecision:
        """
        执行完整的协调步骤
        
        Args:
            observation: 测量值
            timestamp: 时间戳
            
        Returns:
            SystemDecision
        """
        self.state.last_update = timestamp
        
        # Perceive
        perception = self.perceive(observation)
        
        # Decide
        decision = self.decide(perception)
        
        # Act
        self.act(decision)
        
        # 控制智能体在线学习
        if self.control_agent.learning_enabled:
            self.control_agent.learn_from_experience(observation)
        
        return decision
    
    def _update_system_state(self, diagnosis: DiagnosisResult, 
                              control: ControlAction) -> None:
        """更新系统整体状态"""
        if control.mode == ControlMode.EMERGENCY:
            self.system_state = SystemState.EMERGENCY
        elif control.mode == ControlMode.DEGRADED:
            self.system_state = SystemState.DEGRADED
        elif diagnosis.fault_detected:
            self.system_state = SystemState.FAULT_HANDLING
        else:
            self.system_state = SystemState.RUNNING
    
    def simulate(self, fault_injector, base_condition, 
                 duration: float = 100.0, dt: float = 1.0) -> Dict[str, Any]:
        """
        运行仿真
        
        Args:
            fault_injector: 故障注入器
            base_condition: 基准工况
            duration: 仿真时长 [s]
            dt: 时间步长 [s]
            
        Returns:
            simulation_results
        """
        # 重置
        self.reset()
        
        # 结果存储
        results = {
            'time': [],
            'Pmax': [],
            'Pcomp': [],
            'Texh': [],
            'vit_adjustment': [],
            'fuel_adjustment': [],
            'control_mode': [],
            'fault_detected': [],
            'fault_type': [],
            'conflicts': [],
            'action_source': [],
        }
        
        # 仿真循环
        for t in np.arange(0, duration, dt):
            # 应用故障
            fault_injector.apply_faults(t)
            
            # 运行发动机模型
            self.engine.run_cycle(base_condition)
            
            # 获取测量值
            Y_measured = {
                'Pmax': self.engine.get_pmax(),
                'Pcomp': self.engine.get_pcomp(),
                'Texh': self.engine.get_exhaust_temp()
            }
            
            # 执行协调步骤
            decision = self.step(Y_measured, t)
            
            # 记录结果
            results['time'].append(t)
            results['Pmax'].append(Y_measured['Pmax'])
            results['Pcomp'].append(Y_measured['Pcomp'])
            results['Texh'].append(Y_measured['Texh'])
            results['vit_adjustment'].append(decision.control_action.vit_adjustment)
            results['fuel_adjustment'].append(decision.control_action.fuel_adjustment)
            results['control_mode'].append(decision.control_action.mode.name)
            results['fault_detected'].append(decision.diagnosis_result.fault_detected)
            results['fault_type'].append(decision.diagnosis_result.fault_type.name)
            results['conflicts'].append(decision.conflict_resolved)
            results['action_source'].append(decision.control_action.action_source)
        
        # 转换为numpy数组
        for key in ['time', 'Pmax', 'Pcomp', 'Texh', 'vit_adjustment', 'fuel_adjustment']:
            results[key] = np.array(results[key])
        
        return results
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """获取综合报告"""
        return {
            'system_state': self.system_state.name,
            'coordinator_metrics': self.state.performance_metrics.copy(),
            'diagnosis_agent': {
                'state': self.diagnosis_agent.current_state.name,
                'learned_thresholds': self.diagnosis_agent.get_learned_thresholds(),
                'classifier_status': self.diagnosis_agent.get_classifier_status(),
                'metrics': self.diagnosis_agent.state.performance_metrics.copy(),
            },
            'control_agent': {
                'mode': self.control_agent.mode.name,
                'current_vit': self.control_agent.vit_current,
                'current_fuel': self.control_agent.fuel_current,
                'performance': self.control_agent.get_performance_summary(),
            },
            'message_broker': self.message_broker.get_statistics(),
            'conflict_resolver': {
                'history_length': len(self.conflict_resolver.conflict_history),
                'recent_conflicts': [
                    {
                        'type': c.conflict_type.name,
                        'resolution': c.resolution
                    }
                    for c in self.conflict_resolver.conflict_history[-5:]
                ]
            }
        }
    
    def reset(self) -> None:
        """重置协调器和所有子智能体"""
        super().reset()
        self.diagnosis_agent.reset()
        self.control_agent.reset()
        self.system_state = SystemState.INITIALIZING
        self.decision_history.clear()
        self.conflict_resolver.conflict_history.clear()
    
    def save_models(self, directory: str) -> None:
        """保存所有模型"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        # 保存控制智能体的RL模型
        self.control_agent.save_model(os.path.join(directory, 'control_agent_dqn.pt'))
        
        # 保存诊断智能体的阈值和分类器 (如果需要)
        import json
        thresholds = self.diagnosis_agent.get_learned_thresholds()
        with open(os.path.join(directory, 'diagnosis_thresholds.json'), 'w') as f:
            json.dump(thresholds, f)
    
    def load_models(self, directory: str) -> None:
        """加载所有模型"""
        import os
        
        # 加载控制智能体的RL模型
        dqn_path = os.path.join(directory, 'control_agent_dqn.pt')
        if os.path.exists(dqn_path):
            self.control_agent.load_model(dqn_path)
