"""
智能体基础框架
==============
定义智能体抽象基类和通信协议
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum, auto
import numpy as np
from datetime import datetime


class MessageType(Enum):
    """智能体消息类型"""
    # 诊断相关
    DIAGNOSIS_REQUEST = auto()      # 请求诊断
    DIAGNOSIS_RESULT = auto()       # 诊断结果
    FAULT_ALERT = auto()            # 故障警报
    
    # 控制相关
    CONTROL_REQUEST = auto()        # 请求控制动作
    CONTROL_ACTION = auto()         # 控制动作输出
    CONTROL_FEEDBACK = auto()       # 控制效果反馈
    
    # 协调相关
    STATE_UPDATE = auto()           # 状态更新
    MODE_CHANGE = auto()            # 模式切换
    CONFLICT_RESOLUTION = auto()    # 冲突解决
    
    # 学习相关
    EXPERIENCE = auto()             # 经验数据 (用于在线学习)
    MODEL_UPDATE = auto()           # 模型更新通知


@dataclass
class AgentMessage:
    """
    智能体间通信消息
    
    Attributes:
        msg_type: 消息类型
        sender: 发送者标识
        receiver: 接收者标识 (None表示广播)
        payload: 消息内容
        timestamp: 时间戳
        priority: 优先级 (0-10, 10最高)
    """
    msg_type: MessageType
    sender: str
    receiver: Optional[str]
    payload: Dict[str, Any]
    timestamp: float = 0.0
    priority: int = 5
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = datetime.now().timestamp()


@dataclass
class AgentState:
    """智能体状态"""
    name: str
    is_active: bool = True
    last_update: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class Agent(ABC):
    """
    智能体抽象基类
    
    遵循 Perceive -> Decide -> Act -> Learn 循环
    """
    
    def __init__(self, name: str, engine=None):
        """
        初始化智能体
        
        Args:
            name: 智能体名称
            engine: 发动机模型引用 (可选)
        """
        self.name = name
        self.engine = engine
        self.state = AgentState(name=name)
        
        # 消息队列
        self.inbox: List[AgentMessage] = []
        self.outbox: List[AgentMessage] = []
        
        # 经验缓冲区 (用于学习)
        self.experience_buffer: List[Dict] = []
        self.buffer_size = 10000
        
        # 学习参数
        self.learning_enabled = True
        self.learning_rate = 0.001
    
    @abstractmethod
    def perceive(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        感知环境
        
        Args:
            observation: 原始观测数据
            
        Returns:
            processed: 处理后的感知结果
        """
        pass
    
    @abstractmethod
    def decide(self, perception: Dict[str, Any]) -> Any:
        """
        决策
        
        Args:
            perception: 感知结果
            
        Returns:
            decision: 决策结果
        """
        pass
    
    @abstractmethod
    def act(self, decision: Any) -> Dict[str, Any]:
        """
        执行动作
        
        Args:
            decision: 决策结果
            
        Returns:
            action_result: 执行结果
        """
        pass
    
    def learn(self, experience: Dict[str, Any]) -> None:
        """
        从经验中学习
        
        Args:
            experience: 经验数据 (state, action, reward, next_state)
        """
        if not self.learning_enabled:
            return
        
        # 添加到经验缓冲区
        self.experience_buffer.append(experience)
        
        # 限制缓冲区大小
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)
    
    def step(self, observation: Dict[str, Any], timestamp: float = 0.0) -> Any:
        """
        执行一个完整的 Perceive-Decide-Act 循环
        
        Args:
            observation: 观测数据
            timestamp: 时间戳
            
        Returns:
            result: 执行结果
        """
        # 更新状态
        self.state.last_update = timestamp
        
        # 处理收到的消息
        self._process_inbox()
        
        # Perceive
        perception = self.perceive(observation)
        
        # Decide
        decision = self.decide(perception)
        
        # Act
        result = self.act(decision)
        
        return result
    
    def send_message(self, msg_type: MessageType, receiver: Optional[str], 
                     payload: Dict[str, Any], priority: int = 5) -> AgentMessage:
        """发送消息"""
        msg = AgentMessage(
            msg_type=msg_type,
            sender=self.name,
            receiver=receiver,
            payload=payload,
            timestamp=self.state.last_update,
            priority=priority
        )
        self.outbox.append(msg)
        return msg
    
    def receive_message(self, msg: AgentMessage) -> None:
        """接收消息"""
        self.inbox.append(msg)
        # 按优先级排序
        self.inbox.sort(key=lambda m: -m.priority)
    
    def _process_inbox(self) -> None:
        """处理收件箱中的消息"""
        while self.inbox:
            msg = self.inbox.pop(0)
            self._handle_message(msg)
    
    def _handle_message(self, msg: AgentMessage) -> None:
        """
        处理单条消息 (子类可重写)
        
        Args:
            msg: 消息对象
        """
        pass
    
    def reset(self) -> None:
        """重置智能体状态"""
        self.inbox.clear()
        self.outbox.clear()
        self.state.is_active = True
        self.state.last_update = 0.0
        self.state.performance_metrics.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取智能体统计信息"""
        return {
            'name': self.name,
            'is_active': self.state.is_active,
            'last_update': self.state.last_update,
            'experience_count': len(self.experience_buffer),
            'metrics': self.state.performance_metrics.copy()
        }


class ReplayBuffer:
    """
    经验回放缓冲区
    用于强化学习的离线训练
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Dict] = []
        self.position = 0
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """添加一条经验"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Dict]:
        """随机采样一个批次"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.buffer)
