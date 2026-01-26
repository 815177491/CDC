# CDC 模型训练指南

本文档介绍如何配置环境并运行 CDC（控诊协同）系统的模型训练脚本。

## 目录

1. [环境配置](#环境配置)
2. [训练脚本概述](#训练脚本概述)
3. [快速开始](#快速开始)
4. [详细训练命令](#详细训练命令)
5. [对比实验](#对比实验)
6. [结果查看](#结果查看)

---

## 环境配置

### 1. 激活 CDC 虚拟环境

使用 Anaconda 激活 CDC 环境：

```powershell
# Windows PowerShell
conda activate CDC
```

```bash
# Linux/Mac
conda activate CDC
```

### 2. 切换到项目目录

```powershell
cd D:\my_github\CDC
```

### 3. 验证环境配置

运行以下命令检查所有依赖是否正确安装：

```powershell
python -c "import torch; import numpy; import scipy; print('环境检查通过!'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```

预期输出：

```
环境检查通过!
PyTorch: 2.5.1+cu121
CUDA可用: True
```

### 4. 安装依赖（如果缺失）

```powershell
pip install -r requirements.txt
```

---

## 训练脚本概述

| 脚本                                    | 描述                        | 智能体类型              |
| --------------------------------------- | --------------------------- | ----------------------- |
| `experiments/train_marl.py`             | 双智能体 MAPPO 训练         | 诊断智能体 + 控制智能体 |
| `experiments/train_advanced.py`         | PINN+KAN & TD-MPC2 联合训练 | 高级诊断 + 模型预测控制 |
| `experiments/comparison_experiments.py` | 基线对比实验                | 阈值法、PID 等基线方法  |

---

## 快速开始

### 一键运行（完整流程）

```powershell
# 激活环境并进入项目目录
conda activate CDC
cd D:\my_github\CDC

# 1. 运行 MARL 训练（GPU加速）
python experiments/train_marl.py --device cuda

# 2. 运行高级训练
python experiments/train_advanced.py --device cuda

# 3. 运行对比实验
python experiments/comparison_experiments.py
```

---

## 详细训练命令

### 1. MARL 双智能体训练 (`train_marl.py`)

基于 MAPPO 算法的双智能体强化学习训练。

**基本用法：**

```powershell
python experiments/train_marl.py --device cuda
```

**完整参数：**

```powershell
python experiments/train_marl.py \
    --total-steps 100000 \      # 总训练步数
    --rollout-steps 2048 \      # 每次更新的采样步数
    --batch-size 64 \           # 训练批大小
    --n-epochs 10 \             # 每次更新的训练轮数
    --diag-lr 3e-4 \            # 诊断智能体学习率
    --ctrl-lr 3e-4 \            # 控制智能体学习率
    --gamma 0.99 \              # 折扣因子
    --gae-lambda 0.95 \         # GAE 参数
    --clip-epsilon 0.2 \        # PPO 裁剪系数
    --max-ep-steps 200 \        # 单轮最大步数
    --difficulty 0.5 \          # 初始难度 (0-1)
    --variable-condition \      # 启用变工况
    --device cuda \             # 计算设备 (cuda/cpu)
    --save-dir ./checkpoints    # 模型保存目录
```

**短时间测试（5000步）：**

```powershell
python experiments/train_marl.py --total-steps 5000 --device cuda
```

### 2. PINN+KAN & TD-MPC2 高级训练 (`train_advanced.py`)

结合物理信息神经网络(PINN)、Kolmogorov-Arnold Network(KAN) 和 TD-MPC2 控制器的高级训练。

**基本用法：**

```powershell
python experiments/train_advanced.py --device cuda
```

**完整参数：**

```powershell
python experiments/train_advanced.py \
    --total-steps 100000 \      # 总训练步数
    --rollout-steps 1024 \      # 采样步数
    --batch-size 64 \           # 批大小
    --diag-lr 3e-4 \            # 诊断学习率
    --ctrl-lr 3e-4 \            # 控制学习率
    --physics-weight 0.1 \      # 物理损失权重
    --kan-hidden 32 32 \        # KAN 隐藏层维度
    --kan-grid 5 \              # KAN 网格大小
    --mpc-horizon 5 \           # MPC 预测步长
    --mpc-samples 256 \         # MPC 采样数
    --device cuda \             # 计算设备
    --save-dir ./checkpoints/pikan_tdmpc2  # 保存目录
```

**短时间测试：**

```powershell
python experiments/train_advanced.py --total-steps 5000 --device cuda
```

---

## 对比实验

运行基线方法对比实验：

```powershell
python experiments/comparison_experiments.py
```

### 包含的基线方法

| 方法          | 描述                       |
| ------------- | -------------------------- |
| **Threshold** | 基于阈值的传统故障诊断方法 |
| **PID**       | 经典 PID 控制器            |

### 实验输出示例

```
================================================================================
实验结果对比
================================================================================
方法                  诊断准确率          检测延迟           误报率            性能维持
--------------------------------------------------------------------------------
Threshold           0.312          17.53±30.49    1.000          0.473
PID                 0.315          0.86±2.80      0.899          0.544
================================================================================
```

---

## 结果查看

### 1. 模型检查点

训练好的模型保存在 `checkpoints/` 目录：

```
checkpoints/
├── best_ctrl.pt           # 最佳控制智能体
├── best_diag.pt           # 最佳诊断智能体
├── best_critic.pt         # 最佳共享 Critic
├── checkpoint_10_*.pt     # 第10次更新的检查点
├── checkpoint_20_*.pt     # 第20次更新的检查点
└── pikan_tdmpc2/          # PINN+KAN 训练结果
    ├── best_ctrl.pt
    ├── best_diag.pt
    └── plots/             # 训练曲线图
```

### 2. 实验结果

对比实验结果保存在 `experiment_results/results.json`。

### 3. 训练曲线

高级训练的可视化结果保存在 `checkpoints/pikan_tdmpc2/plots/` 目录。

---

## 常用命令速查

```powershell
# === 环境设置 ===
conda activate CDC
cd D:\my_github\CDC

# === MARL 训练 ===
# 快速测试
python experiments/train_marl.py --total-steps 5000 --device cuda

# 完整训练
python experiments/train_marl.py --total-steps 100000 --device cuda

# CPU 训练（无 GPU 时）
python experiments/train_marl.py --total-steps 50000 --device cpu

# === 高级训练 ===
# 快速测试
python experiments/train_advanced.py --total-steps 5000 --device cuda

# 完整训练
python experiments/train_advanced.py --total-steps 100000 --device cuda

# === 对比实验 ===
python experiments/comparison_experiments.py
```

---

## 故障排除

### 问题 1: CUDA 不可用

```powershell
# 检查 CUDA 状态
python -c "import torch; print(torch.cuda.is_available())"

# 如果返回 False，使用 CPU 训练
python experiments/train_marl.py --device cpu
```

### 问题 2: 内存不足

减小批大小和采样步数：

```powershell
python experiments/train_marl.py --batch-size 32 --rollout-steps 1024 --device cuda
```

### 问题 3: 模块导入错误

确保在项目根目录运行：

```powershell
cd D:\my_github\CDC
python experiments/train_marl.py
```

---

## 联系方式

如有问题，请查阅 `docs/modeling.md` 了解模型架构详情。
