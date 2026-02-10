#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试数据加载pipeline

运行方法:
    python tools/test_data_loading.py configs/bevdet/bevdet-r50-depth-modulation.py
"""

import sys
import os
import argparse
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset


def test_data_loading(config_file):
    """测试数据加载"""

    print("=" * 80)
    print("测试数据加载Pipeline")
    print("=" * 80)

    # 加载配置
    print(f"\n[1/4] 加载配置文件: {config_file}")
    cfg = Config.fromfile(config_file)

    # 构建数据集
    print("\n[2/4] 构建训练数据集...")
    try:
        dataset = build_dataset(cfg.data.train)
        print(f"  ✅ 数据集构建成功，共 {len(dataset)} 个样本")
    except Exception as e:
        print(f"  ❌ 数据集构建失败: {e}")
        return False

    # 测试加载第一个样本
    print("\n[3/4] 测试加载第一个样本...")
    try:
        data = dataset[0]
        print(f"  ✅ 样本加载成功")
        print(f"  数据键: {list(data.keys())}")

        # 检查关键字段
        if 'img_inputs' in data:
            img_inputs = data['img_inputs']
            print(f"  - img_inputs: {len(img_inputs)} 个元素")
            if len(img_inputs) > 0:
                imgs = img_inputs[0]
                print(f"    - 图像张量形状: {imgs.shape}")

        if 'pretrained_depths' in data:
            depths = data['pretrained_depths']
            print(f"  - pretrained_depths 形状: {depths.shape}")
            print(f"    - 深度范围: [{depths.min():.2f}, {depths.max():.2f}]")
        else:
            print(f"  ⚠️  警告: 未找到 pretrained_depths 字段")

        if 'gt_depth' in data:
            gt_depth = data['gt_depth']
            print(f"  - gt_depth 形状: {gt_depth.shape}")
            print(f"    - GT深度范围: [{gt_depth.min():.2f}, {gt_depth.max():.2f}]")

        if 'gt_bboxes_3d' in data:
            print(f"  - gt_bboxes_3d: {len(data['gt_bboxes_3d'])} 个框")

    except Exception as e:
        print(f"  ❌ 样本加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试加载多个样本
    print("\n[4/4] 测试加载多个样本...")
    try:
        for i in range(min(3, len(dataset))):
            data = dataset[i]
            if 'pretrained_depths' in data:
                depths = data['pretrained_depths']
                print(f"  样本 {i}: pretrained_depths 形状 {depths.shape}, "
                      f"范围 [{depths.min():.2f}, {depths.max():.2f}]")
            else:
                print(f"  样本 {i}: ⚠️  缺少 pretrained_depths")
        print(f"  ✅ 多样本加载测试通过")
    except Exception as e:
        print(f"  ❌ 多样本加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✅ 所有测试通过!")
    print("=" * 80)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='测试数据加载')
    parser.add_argument('config', help='配置文件路径')
    args = parser.parse_args()

    success = test_data_loading(args.config)
    sys.exit(0 if success else 1)
