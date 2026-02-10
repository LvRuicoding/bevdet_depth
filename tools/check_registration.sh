#!/bin/bash
# 静态检查模块注册是否正确

echo "================================================================================"
echo "模块注册静态检查"
echo "================================================================================"

errors=0

echo ""
echo "[1/4] 检查 Detectors 注册..."
echo "----------------------------------------"

# 检查__init__.py中注册的类
registered_detectors=$(grep -oP "'\K[^']+" mmdet3d/models/detectors/__init__.py | grep -v "^#" | sort | uniq)

# 检查实际存在的类
echo "  检查已注册的类是否存在..."
for class_name in $registered_detectors; do
    # 跳过一些特殊的类名
    if [[ "$class_name" == "Base3DDetector" ]] || [[ "$class_name" == "VoxelNet" ]] || \
       [[ "$class_name" == "DynamicVoxelNet" ]] || [[ "$class_name" == "MVXTwoStageDetector" ]] || \
       [[ "$class_name" == "DynamicMVXFasterRCNN" ]] || [[ "$class_name" == "MVXFasterRCNN" ]] || \
       [[ "$class_name" == "PartA2" ]] || [[ "$class_name" == "VoteNet" ]] || \
       [[ "$class_name" == "H3DNet" ]] || [[ "$class_name" == "CenterPoint" ]] || \
       [[ "$class_name" == "SSD3DNet" ]] || [[ "$class_name" == "ImVoteNet" ]] || \
       [[ "$class_name" == "SingleStageMono3DDetector" ]] || [[ "$class_name" == "FCOSMono3D" ]] || \
       [[ "$class_name" == "ImVoxelNet" ]] || [[ "$class_name" == "GroupFree3DNet" ]] || \
       [[ "$class_name" == "PointRCNN" ]] || [[ "$class_name" == "SMOKEMono3D" ]] || \
       [[ "$class_name" == "MinkSingleStage3DDetector" ]] || [[ "$class_name" == "SASSD" ]]; then
        continue
    fi

    found=$(grep -r "class $class_name" mmdet3d/models/detectors/*.py | grep -v "__init__" | wc -l)
    if [ "$found" -eq 0 ]; then
        echo "  ❌ 错误: $class_name 已注册但未找到实现"
        errors=$((errors + 1))
    else
        echo "  ✅ $class_name"
    fi
done

echo ""
echo "[2/4] 检查 Necks 注册..."
echo "----------------------------------------"

# 检查关键的neck类
necks=("DepthModulationNetwork" "LSSViewTransformerDepthModulation" "LSSFPN3D" "LSSViewTransformer" "LSSViewTransformerBEVDepth" "LSSViewTransformerBEVStereo")

for neck in "${necks[@]}"; do
    # 检查是否在__init__.py中注册
    registered=$(grep "'$neck'" mmdet3d/models/necks/__init__.py | wc -l)
    # 检查是否有实现
    implemented=$(grep -r "class $neck" mmdet3d/models/necks/*.py | grep -v "__init__" | wc -l)

    if [ "$registered" -gt 0 ] && [ "$implemented" -gt 0 ]; then
        echo "  ✅ $neck (已注册且已实现)"
    elif [ "$registered" -gt 0 ] && [ "$implemented" -eq 0 ]; then
        echo "  ❌ 错误: $neck (已注册但未实现)"
        errors=$((errors + 1))
    elif [ "$registered" -eq 0 ] && [ "$implemented" -gt 0 ]; then
        echo "  ⚠️  警告: $neck (已实现但未注册)"
        errors=$((errors + 1))
    fi
done

echo ""
echo "[3/4] 检查 Backbones 注册..."
echo "----------------------------------------"

# 检查ResNetRGBD
if grep -q "ResNetRGBD" mmdet3d/models/backbones/__init__.py; then
    if [ -f "mmdet3d/models/backbones/resnet_rgbd.py" ]; then
        echo "  ✅ ResNetRGBD (已注册且已实现)"
    else
        echo "  ❌ 错误: ResNetRGBD (已注册但文件不存在)"
        errors=$((errors + 1))
    fi
else
    if [ -f "mmdet3d/models/backbones/resnet_rgbd.py" ]; then
        echo "  ⚠️  警告: ResNetRGBD (已实现但未注册)"
        errors=$((errors + 1))
    fi
fi

echo ""
echo "[4/4] 检查 Pipelines 注册..."
echo "----------------------------------------"

# 检查LoadPretrainedDepth
if grep -q "LoadPretrainedDepth" mmdet3d/datasets/pipelines/__init__.py; then
    if [ -f "mmdet3d/datasets/pipelines/loaddepth.py" ]; then
        echo "  ✅ LoadPretrainedDepth (已注册且已实现)"
    else
        echo "  ❌ 错误: LoadPretrainedDepth (已注册但文件不存在)"
        errors=$((errors + 1))
    fi
else
    if [ -f "mmdet3d/datasets/pipelines/loaddepth.py" ]; then
        echo "  ⚠️  警告: LoadPretrainedDepth (已实现但未注册)"
        errors=$((errors + 1))
    fi
fi

echo ""
echo "================================================================================"
echo "检查完成!"
echo "================================================================================"

if [ "$errors" -eq 0 ]; then
    echo ""
    echo "✅ 所有检查通过，没有发现错误!"
    exit 0
else
    echo ""
    echo "❌ 发现 $errors 个问题，请检查上述输出"
    exit 1
fi
