#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
建模可视化运行脚本
==================
生成 modeling.md 文档所需的所有建模相关图片

生成的图片保存在 visualization_output/modeling/ 目录下

Author: CDC Project
Date: 2026-01-28
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from visualization.modeling_plots import generate_all_modeling_figures


def main():
    """主函数"""
    print("=" * 70)
    print("建模可视化图片生成脚本")
    print("=" * 70)
    print(f"工作目录: {project_root}")
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
