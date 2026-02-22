#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
建模可视化运行脚本
==================
生成 modeling.md 文档所需的所有建模相关图片

生成的图片保存在 visualization_output/modeling/ 目录下

使用方法:
    python scripts/visualize_modeling.py

Author: CDC Project
Date: 2026-01-28
"""

import sys
import os

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from visualization.modeling import generate_all_modeling_figures


def main():
    """主函数"""
    print("=" * 70)
    print("建模可视化图片生成脚本")
    print("=" * 70)
    print(f"工作目录: {PROJECT_ROOT}")
    print(f"输出目录: visualization_output/modeling/")
    print("=" * 70)
    
    # 生成所有图片
    results = generate_all_modeling_figures()
    
    # 输出结果汇总
    print("\n" + "=" * 70)
    print("生成结果汇总:")
    print("=" * 70)
    for name, path in results.items():
        print(f"  ✓ {name}: {path}")
    
    print("\n所有图片已生成完成！")
    print("请检查 visualization_output/modeling/ 目录")
    
    return results


if __name__ == '__main__':
    main()
