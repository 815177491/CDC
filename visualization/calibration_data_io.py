#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
校准数据IO模块
==============
从 calibration_plots.py 中拆分出的数据加载函数，
负责收敛历史、验证结果、校准参数等数据文件的读写。

Author: CDC Project
Date: 2026-02-20
"""

# 标准库
import os
import json
from typing import Dict, Optional

# 第三方库
import pandas as pd

# 本项目模块
from config import PATH_CONFIG


def load_convergence_data(filepath: str = None) -> pd.DataFrame:
    """
    加载收敛历史数据

    Args:
        filepath: CSV文件路径，默认为 data/calibration/calibration_convergence.csv

    Returns:
        收敛历史DataFrame

    Raises:
        FileNotFoundError: 文件不存在时抛出
    """
    if filepath is None:
        filepath = os.path.join(PATH_CONFIG.DATA_CALIBRATION_DIR, 'calibration_convergence.csv')

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"收敛历史文件不存在: {filepath}")

    return pd.read_csv(filepath)


def load_validation_data(filepath: str = None) -> pd.DataFrame:
    """
    加载验证结果数据

    Args:
        filepath: CSV文件路径，默认为 data/calibration/calibration_validation.csv

    Returns:
        验证结果DataFrame

    Raises:
        FileNotFoundError: 文件不存在时抛出
    """
    if filepath is None:
        filepath = os.path.join(PATH_CONFIG.DATA_CALIBRATION_DIR, 'calibration_validation.csv')

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"验证结果文件不存在: {filepath}")

    return pd.read_csv(filepath)


def load_calibrated_params(filepath: str = None) -> Dict:
    """
    加载校准参数

    Args:
        filepath: JSON文件路径，默认为 data/calibration/calibrated_params.json

    Returns:
        校准参数字典

    Raises:
        FileNotFoundError: 文件不存在时抛出
    """
    if filepath is None:
        filepath = os.path.join(PATH_CONFIG.DATA_CALIBRATION_DIR, 'calibrated_params.json')

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"校准参数文件不存在: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
