#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断模型对比实验
================
对比三种故障诊断方法:
1. RandomForest (传统方法)
2. PINN (物理信息神经网络)
3. KAN (Kolmogorov-Arnold Networks, 2024)

评估指标:
- 准确率 (Accuracy)
- 各类别F1分数
- 训练时间
- 推理时间
- 可解释性

Author: CDC Project
Date: 2026-01-22
"""

import numpy as np
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnosis import PINNDiagnoser, KANDiagnoser, FaultType


@dataclass
class DiagnosisExperimentResult:
    """诊断实验结果"""
    method: str
    accuracy: float
    f1_macro: float
    f1_per_class: Dict[str, float]
    train_time: float
    inference_time: float  # ms per sample
    n_parameters: int
    interpretability: str


def generate_synthetic_data(n_samples: int = 2000, noise: float = 0.1) -> Dict[str, np.ndarray]:
    """
    生成合成柴油机故障诊断数据
    
    特征 (8维):
        0: 转速 (rpm) - 归一化到[0,1]
        1: 负荷 (%) - [0,1]
        2: 喷油正时 (°CA BTDC) - 归一化
        3: 增压压力 (bar) - 归一化
        4: 进气温度 (K) - 归一化
        5: 环境压力 (bar) - 归一化
        6: 燃油品质 - [0,1]
        7: 运行时间 (h) - 归一化
    
    物理量 (3维):
        0: Pmax (bar)
        1: Pcomp (bar)
        2: Texh (°C)
    
    故障类型 (4类):
        0: 正常
        1: 喷油正时异常
        2: 气缸泄漏
        3: 燃油品质问题
    """
    np.random.seed(42)
    
    n_per_class = n_samples // 4
    features_list = []
    physics_list = []
    labels_list = []
    
    for fault_type in range(4):
        # 基础特征 (正常工况)
        rpm = np.random.uniform(0.3, 0.9, n_per_class)
        load = np.random.uniform(0.2, 1.0, n_per_class)
        timing = np.random.uniform(0.4, 0.6, n_per_class)  # 正常正时
        boost = np.random.uniform(0.3, 0.7, n_per_class)
        t_in = np.random.uniform(0.4, 0.6, n_per_class)
        p_amb = np.random.uniform(0.45, 0.55, n_per_class)
        fuel_q = np.random.uniform(0.8, 1.0, n_per_class)  # 正常燃油
        run_h = np.random.uniform(0, 1, n_per_class)
        
        # 基础物理量
        pmax_base = 120 + 40 * load + 10 * boost
        pcomp_base = 100 + 20 * boost
        texh_base = 280 + 100 * load
        
        if fault_type == 0:  # 正常
            pmax = pmax_base
            pcomp = pcomp_base
            texh = texh_base
            
        elif fault_type == 1:  # 喷油正时异常
            timing = timing + np.random.choice([-0.2, 0.2], n_per_class)  # 正时偏移
            timing = np.clip(timing, 0, 1)
            # 正时提前: Pmax升高, Texh降低
            # 正时滞后: Pmax降低, Texh升高
            timing_shift = (timing - 0.5) * 2
            pmax = pmax_base + 15 * timing_shift
            pcomp = pcomp_base
            texh = texh_base - 20 * timing_shift
            
        elif fault_type == 2:  # 气缸泄漏
            leak_severity = np.random.uniform(0.1, 0.3, n_per_class)
            pmax = pmax_base * (1 - leak_severity)
            pcomp = pcomp_base * (1 - leak_severity * 1.2)
            texh = texh_base + 30 * leak_severity
            
        elif fault_type == 3:  # 燃油品质问题
            fuel_q = np.random.uniform(0.5, 0.8, n_per_class)  # 燃油品质下降
            fuel_effect = 1 - fuel_q
            pmax = pmax_base * (1 - fuel_effect * 0.15)
            pcomp = pcomp_base
            texh = texh_base + 40 * fuel_effect - 20 * fuel_effect
        
        # 添加噪声
        pmax += np.random.randn(n_per_class) * noise * 10
        pcomp += np.random.randn(n_per_class) * noise * 5
        texh += np.random.randn(n_per_class) * noise * 15
        
        # 合并特征
        features = np.stack([rpm, load, timing, boost, t_in, p_amb, fuel_q, run_h], axis=1)
        physics = np.stack([pmax, pcomp, texh], axis=1)
        labels = np.full(n_per_class, fault_type)
        
        features_list.append(features)
        physics_list.append(physics)
        labels_list.append(labels)
    
    # 合并并打乱
    all_features = np.vstack(features_list)
    all_physics = np.vstack(physics_list)
    all_labels = np.concatenate(labels_list)
    
    indices = np.random.permutation(len(all_labels))
    
    return {
        'features': all_features[indices].astype(np.float32),
        'physics': all_physics[indices].astype(np.float32),
        'labels': all_labels[indices].astype(np.int64)
    }


def train_random_forest(train_data: Dict, test_data: Dict) -> DiagnosisExperimentResult:
    """训练并评估RandomForest"""
    print("\n" + "="*60)
    print("🌲 训练 RandomForest (传统基线)")
    print("="*60)
    
    # 合并特征和物理量作为输入
    X_train = np.hstack([train_data['features'], train_data['physics']])
    y_train = train_data['labels']
    X_test = np.hstack([test_data['features'], test_data['physics']])
    y_test = test_data['labels']
    
    # 训练
    start_time = time.time()
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # 预测
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    fault_names = ['Normal', 'Timing', 'Leak', 'Fuel']
    f1_per_class = {}
    for i, name in enumerate(fault_names):
        mask = y_test == i
        if mask.sum() > 0:
            f1_per_class[name] = f1_score(y_test == i, y_pred == i)
    
    # 特征重要性
    feature_names = ['rpm', 'load', 'timing', 'boost', 't_in', 'p_amb', 'fuel_q', 'run_h',
                     'Pmax', 'Pcomp', 'Texh']
    importances = dict(zip(feature_names, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print(f"  准确率: {accuracy:.2%}")
    print(f"  F1-macro: {f1_macro:.4f}")
    print(f"  训练时间: {train_time:.2f}s")
    print(f"  推理时间: {inference_time:.4f}ms/sample")
    print(f"  Top特征: {top_features[:3]}")
    
    return DiagnosisExperimentResult(
        method="RandomForest",
        accuracy=accuracy,
        f1_macro=f1_macro,
        f1_per_class=f1_per_class,
        train_time=train_time,
        inference_time=inference_time,
        n_parameters=sum(tree.tree_.node_count for tree in model.estimators_),
        interpretability="低: 仅提供特征重要性"
    )


def train_pinn(train_data: Dict, test_data: Dict, epochs: int = 50) -> DiagnosisExperimentResult:
    """训练并评估PINN"""
    print("\n" + "="*60)
    print("🧠 训练 PINN (物理信息神经网络)")
    print("="*60)
    
    # 创建PINN
    config = {
        'input_dim': 8,
        'hidden_dim': 128,
        'n_blocks': 4,
        'physics_weight': 0.1,
        'lr': 1e-3
    }
    
    pinn = PINNDiagnoser(config)
    
    # 训练
    start_time = time.time()
    history = pinn.train(train_data, test_data, epochs=epochs, verbose=True)
    train_time = time.time() - start_time
    
    # 评估
    metrics = pinn.evaluate(test_data)
    
    # 推理时间
    start_time = time.time()
    for i in range(min(100, len(test_data['features']))):
        _ = pinn.diagnose(test_data['features'][i])
    inference_time = (time.time() - start_time) / min(100, len(test_data['features'])) * 1000
    
    # 计算F1
    y_test = test_data['labels']
    y_pred = []
    for feat in test_data['features']:
        result = pinn.diagnose(feat)
        y_pred.append(pinn.fault_types.index(result.fault_type))
    y_pred = np.array(y_pred)
    
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    fault_names = ['Normal', 'Timing', 'Leak', 'Fuel']
    f1_per_class = {}
    for i, name in enumerate(fault_names):
        mask = y_test == i
        if mask.sum() > 0:
            f1_per_class[name] = f1_score(y_test == i, y_pred == i)
    
    print(f"  准确率: {metrics['accuracy']:.2%}")
    print(f"  F1-macro: {f1_macro:.4f}")
    print(f"  Pmax MAE: {metrics['pmax_mae']:.2f} bar")
    print(f"  训练时间: {train_time:.2f}s")
    print(f"  推理时间: {inference_time:.4f}ms/sample")
    
    return DiagnosisExperimentResult(
        method="PINN",
        accuracy=metrics['accuracy'],
        f1_macro=f1_macro,
        f1_per_class=f1_per_class,
        train_time=train_time,
        inference_time=inference_time,
        n_parameters=sum(p.numel() for p in pinn.model.parameters()),
        interpretability="高: 物理残差解释故障机理"
    )


def train_kan(train_data: Dict, test_data: Dict, epochs: int = 50) -> DiagnosisExperimentResult:
    """训练并评估KAN"""
    print("\n" + "="*60)
    print("🔮 训练 KAN (Kolmogorov-Arnold Networks, 2024)")
    print("="*60)
    
    # 创建KAN
    config = {
        'input_dim': 8,
        'hidden_dims': [16, 8],
        'output_dim': 4,
        'grid_size': 5,
        'lr': 1e-3
    }
    
    kan = KANDiagnoser(config)
    
    # 训练
    start_time = time.time()
    history = kan.train(train_data, test_data, epochs=epochs, verbose=True)
    train_time = time.time() - start_time
    
    # 评估
    metrics = kan.evaluate(test_data)
    
    # 推理时间
    start_time = time.time()
    for i in range(min(100, len(test_data['features']))):
        _ = kan.diagnose(test_data['features'][i])
    inference_time = (time.time() - start_time) / min(100, len(test_data['features'])) * 1000
    
    # 计算F1
    y_test = test_data['labels']
    y_pred = []
    for feat in test_data['features']:
        result = kan.diagnose(feat)
        y_pred.append(kan.fault_types.index(result.fault_type))
    y_pred = np.array(y_pred)
    
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    fault_names = ['Normal', 'Timing', 'Leak', 'Fuel']
    f1_per_class = {}
    for i, name in enumerate(fault_names):
        mask = y_test == i
        if mask.sum() > 0:
            f1_per_class[name] = f1_score(y_test == i, y_pred == i)
    
    # 提取符号规则
    rules = kan.get_symbolic_rules()
    
    print(f"  准确率: {metrics['accuracy']:.2%}")
    print(f"  F1-macro: {f1_macro:.4f}")
    print(f"  训练时间: {train_time:.2f}s")
    print(f"  推理时间: {inference_time:.4f}ms/sample")
    print(f"  特征重要性: {rules['feature_importance']}")
    
    return DiagnosisExperimentResult(
        method="KAN (2024)",
        accuracy=metrics['accuracy'],
        f1_macro=f1_macro,
        f1_per_class=f1_per_class,
        train_time=train_time,
        inference_time=inference_time,
        n_parameters=kan.model.count_parameters(),
        interpretability="极高: 可提取符号化诊断规则"
    )


def print_comparison_table(results: List[DiagnosisExperimentResult]):
    """打印对比表格"""
    print("\n" + "="*80)
    print("📊 诊断模型对比结果")
    print("="*80)
    
    print(f"\n{'方法':<20} {'准确率':<10} {'F1-macro':<10} {'参数量':<12} {'训练时间':<10} {'推理时间':<12}")
    print("-"*80)
    
    for r in results:
        print(f"{r.method:<20} {r.accuracy:.2%}     {r.f1_macro:.4f}     {r.n_parameters:<12,} {r.train_time:.2f}s      {r.inference_time:.4f}ms")
    
    print("-"*80)
    
    # 找最佳
    best_acc = max(results, key=lambda x: x.accuracy)
    best_f1 = max(results, key=lambda x: x.f1_macro)
    fastest = min(results, key=lambda x: x.train_time)
    smallest = min(results, key=lambda x: x.n_parameters)
    
    print(f"\n🏆 最佳结果:")
    print(f"  - 最高准确率: {best_acc.method} ({best_acc.accuracy:.2%})")
    print(f"  - 最高F1: {best_f1.method} ({best_f1.f1_macro:.4f})")
    print(f"  - 最快训练: {fastest.method} ({fastest.train_time:.2f}s)")
    print(f"  - 最少参数: {smallest.method} ({smallest.n_parameters:,})")
    
    print(f"\n📝 可解释性对比:")
    for r in results:
        print(f"  - {r.method}: {r.interpretability}")
    
    # 推荐
    print("\n" + "="*80)
    print("💡 推荐选择")
    print("="*80)
    
    # 综合评分
    scores = {}
    for r in results:
        score = (
            r.accuracy * 0.3 +
            r.f1_macro * 0.3 +
            (1 - r.train_time / max(x.train_time for x in results)) * 0.1 +
            (1 - r.n_parameters / max(x.n_parameters for x in results)) * 0.1 +
            (0.8 if 'KAN' in r.method else 0.5 if 'PINN' in r.method else 0.2) * 0.2  # 可解释性加分
        )
        scores[r.method] = score
    
    best_overall = max(scores.items(), key=lambda x: x[1])
    print(f"\n综合推荐: {best_overall[0]} (综合得分: {best_overall[1]:.3f})")
    
    if 'PINN' in best_overall[0]:
        print("  理由: 物理约束提升泛化能力，残差可解释故障机理")
    elif 'KAN' in best_overall[0]:
        print("  理由: 2024年最新方法，可提取符号规则，参数效率高")
    else:
        print("  理由: 传统方法，简单可靠，适合快速部署")


def main():
    """主函数"""
    print("="*80)
    print("🔬 柴油机故障诊断模型对比实验")
    print("="*80)
    print("对比方法: RandomForest (传统) vs PINN (物理信息) vs KAN (2024)")
    print("-"*80)
    
    # 生成数据
    print("\n📦 生成合成诊断数据...")
    all_data = generate_synthetic_data(n_samples=2000, noise=0.1)
    
    # 划分训练/测试集 (80/20)
    n_train = int(len(all_data['labels']) * 0.8)
    indices = np.random.permutation(len(all_data['labels']))
    
    train_data = {
        'features': all_data['features'][indices[:n_train]],
        'physics': all_data['physics'][indices[:n_train]],
        'labels': all_data['labels'][indices[:n_train]]
    }
    test_data = {
        'features': all_data['features'][indices[n_train:]],
        'physics': all_data['physics'][indices[n_train:]],
        'labels': all_data['labels'][indices[n_train:]]
    }
    
    print(f"  训练集: {len(train_data['labels'])} 样本")
    print(f"  测试集: {len(test_data['labels'])} 样本")
    print(f"  类别分布: {np.bincount(train_data['labels'])}")
    
    results = []
    
    # 1. RandomForest
    rf_result = train_random_forest(train_data, test_data)
    results.append(rf_result)
    
    # 2. PINN
    pinn_result = train_pinn(train_data, test_data, epochs=50)
    results.append(pinn_result)
    
    # 3. KAN
    kan_result = train_kan(train_data, test_data, epochs=50)
    results.append(kan_result)
    
    # 对比结果
    print_comparison_table(results)
    
    print("\n" + "="*80)
    print("✅ 诊断模型对比实验完成!")
    print("="*80)


if __name__ == "__main__":
    main()
