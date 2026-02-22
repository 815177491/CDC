#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境模块桥接
============
从 multi_agent.env 重新导出环境类，
使 ``from marl.env import EngineEnv`` 等导入方式正常工作。

Author: CDC Project
Date: 2025-01-01
"""

from multi_agent.env import (
    EngineEnv,
    EngineEnvConfig,
    CompositeFaultInjector,
    OperatingScheduler,
)

__all__ = [
    'EngineEnv',
    'EngineEnvConfig',
    'CompositeFaultInjector',
    'OperatingScheduler',
]
