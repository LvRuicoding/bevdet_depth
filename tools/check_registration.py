#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证所有模块注册是否正确

运行方法:
    python tools/check_registration.py
"""

import sys
import importlib


def check_module_registration():
    """检查所有模块注册是否正确"""

    print("=" * 80)
    print("开始检查模块注册...")
    print("=" * 80)

    errors = []
    warnings = []

    # 1. 检查 Detectors
    print("\n[1/4] 检查 Detectors...")
    try:
        from mmdet3d.models.detectors import (
            BEVDet, BEVDetTRT, BEVDet4D, BEVDepth4D, BEVStereo4D,
            BEVStereo4DOCC, BEVDetDepthModulation, DAL,
            Base3DDetector, VoxelNet, DynamicVoxelNet, MVXTwoStageDetector,
            DynamicMVXFasterRCNN, MVXFasterRCNN, PartA2, VoteNet, H3DNet,
            CenterPoint, SSD3DNet, ImVoteNet, SingleStageMono3DDetector,
            FCOSMono3D, ImVoxelNet, GroupFree3DNet, PointRCNN, SMOKEMono3D,
            MinkSingleStage3DDetector, SASSD
        )
        print("  ✅ 所有 Detector 类导入成功")
    except ImportError as e:
        errors.append(f"Detector 导入失败: {e}")
        print(f"  ❌ Detector 导入失败: {e}")

    # 2. 检查 Necks
    print("\n[2/4] 检查 Necks...")
    try:
        from mmdet3d.models.necks import (
            FPN, SECONDFPN, OutdoorImVoxelNeck, PointNetFPNeck, DLANeck,
            LSSViewTransformer, CustomFPN, FPN_LSS, LSSFPN3D,
            LSSViewTransformerBEVDepth, LSSViewTransformerBEVStereo,
            DepthModulationNetwork, LSSViewTransformerDepthModulation
        )
        print("  ✅ 所有 Neck 类导入成功")
    except ImportError as e:
        errors.append(f"Neck 导入失败: {e}")
        print(f"  ❌ Neck 导入失败: {e}")

    # 3. 检查 Backbones
    print("\n[3/4] 检查 Backbones...")
    try:
        from mmdet3d.models.backbones import (
            ResNet, ResNetV1d, ResNeXt, ResNetRGBD, SSDVGG, HRNet,
            NoStemRegNet, SECOND, DGCNNBackbone, PointNet2SASSG,
            PointNet2SAMSG, MultiBackbone, DLANet, MinkResNet,
            CustomResNet, CustomResNet3D, SwinTransformer
        )
        print("  ✅ 所有 Backbone 类导入成功")
    except ImportError as e:
        errors.append(f"Backbone 导入失败: {e}")
        print(f"  ❌ Backbone 导入失败: {e}")

    # 4. 检查 Pipelines
    print("\n[4/4] 检查 Pipelines...")
    try:
        from mmdet3d.datasets.pipelines import (
            LoadPretrainedDepth, PrepareImageInputs, PointToMultiViewDepth,
            LoadAnnotations, LoadAnnotations3D, BEVAug, Collect3D,
            DefaultFormatBundle3D
        )
        print("  ✅ 所有 Pipeline 类导入成功")
    except ImportError as e:
        errors.append(f"Pipeline 导入失败: {e}")
        print(f"  ❌ Pipeline 导入失败: {e}")

    # 5. 检查新增的深度调制模块
    print("\n[5/5] 检查深度调制相关模块...")
    try:
        from mmdet3d.models.necks import DepthModulationNetwork
        from mmdet3d.models.necks import LSSViewTransformerDepthModulation
        from mmdet3d.models.detectors import BEVDetDepthModulation
        from mmdet3d.datasets.pipelines import LoadPretrainedDepth
        print("  ✅ 所有深度调制模块导入成功")
        print("    - DepthModulationNetwork")
        print("    - LSSViewTransformerDepthModulation")
        print("    - BEVDetDepthModulation")
        print("    - LoadPretrainedDepth")
    except ImportError as e:
        errors.append(f"深度调制模块导入失败: {e}")
        print(f"  ❌ 深度调制模块导入失败: {e}")

    # 打印总结
    print("\n" + "=" * 80)
    print("检查完成!")
    print("=" * 80)

    if errors:
        print(f"\n❌ 发现 {len(errors)} 个错误:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        return False
    else:
        print("\n✅ 所有模块注册正确，没有发现错误!")
        return True


if __name__ == '__main__':
    success = check_module_registration()
    sys.exit(0 if success else 1)
