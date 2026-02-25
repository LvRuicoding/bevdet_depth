# BEVDet-R50-MUSt3R-Decoder-MultiView 模型结构文档

## 概述

在标准 BEVDet-R50 的基础上，将 neck 从 CustomFPN 替换为 MUSt3R 的 decoder blocks，
作为特征增强层插入 ResNet-50 backbone 和 LSS View Transformer 之间。
**关键特性：6 张图在 decoder 中进行跨视图融合**，然后输出每张图的增强特征供 LSS 处理。

```
输入图像 (B, 6, 3, 256, 704)
        │
        ▼
┌─────────────────────────┐
│  ResNet-50 (stage 0-2)  │  pretrained: torchvision://resnet50
│  out_indices=(2,)       │  输出: (B*6, 1024, 16, 44)
│  23.51M params          │
└─────────┬───────────────┘
          │ (B*6, 1024, 16, 44)
          │ reshape → (B, 6, 1024, 16, 44)
          ▼
┌──────────────────────────────────────────────┐
│  Must3rDecoderNeckMultiView                  │  pretrained: MUSt3R_224_cvpr.pth (decoder部分)
│                                              │
│  1. flatten → tokens  (B, 6*h*w, 1024)       │
│     6张图的tokens拼接在一起                    │
│  2. feat_embed_enc_to_dec                    │  Linear(1024→768)     0.79M
│     tokens → (B, 6*h*w, 768)                 │
│  3. 12× CachedDecoderBlock (跨视图融合)       │  每个block有：
│     - Self-Attention (768-dim, 12 heads)     │    - Self-Attn on all 6*h*w tokens
│     - Cross-Attention (attend自身tokens)     │    - Cross-Attn (memory from self)
│     - MLP (768→3072→768)                     │    - MLP
│     - RoPE positional embedding              │    - 113.44M
│  4. LayerNorm                                │
│  5. reshape → (B, 6, h*w, 768)               │
│  6. reshape → (B*6, 768, 16, 44)             │
│  7. output_proj Conv1x1(768→256)+BN+ReLU     │  0.20M (随机初始化)
│                                              │
│  114.43M params total                        │
└─────────┬──────────────────────────────────┘
          │ (B*6, 256, 16, 44)
          │ reshape → (B, 6, 256, 16, 44)
          ▼
┌─────────────────────────┐
│  LSSViewTransformer     │  depth prediction + Lift-Splat
│  downsample=16          │  输出: (B, 64, 128, 128)
│  0.03M params           │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  CustomResNet (BEV enc) │  numC: 64→128→256→512
│  12.39M params          │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  FPN_LSS (BEV neck)     │  输出: (B, 256, 128, 128)
│  6.56M params           │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  CenterHead             │  10-class 3D detection
│  0.38M params           │
└─────────────────────────┘
```

## 参数统计

| 组件 | 参数量 | 权重来源 |
|------|--------|----------|
| ResNet-50 backbone | 23.51M | torchvision pretrained |
| Must3rDecoderNeckMultiView (decoder部分) | 114.23M | MUSt3R_224_cvpr.pth |
| Must3rDecoderNeckMultiView (output_proj) | 0.20M | 随机初始化 |
| LSSViewTransformer | 0.03M | 随机初始化 |
| BEV encoder backbone | 12.39M | 随机初始化 |
| BEV encoder neck | 6.56M | 随机初始化 |
| CenterHead | 0.38M | 随机初始化 |
| **总计** | **157.30M** | **全部可训练** |

## 训练设置

| 设置 | 值 |
|------|-----|
| Optimizer | AdamW, lr=2.5e-5, weight_decay=1e-7 |
| Grad clip | max_norm=5, norm_type=2 |
| LR schedule | step decay at epoch 24, linear warmup 200 iters |
| Epochs | 24 |
| Batch size | 2 per GPU |
| Workers | 16 per GPU |
| EMA | MEGVIIEMAHook, init_updates=10560 |
| Input size | (256, 704) |
| BDA augmentation | rot=±22.5°, scale=0.95~1.05, flip=0.5 |
| Gradient checkpointing | 关闭 (with_cp=False) |
| DDP find_unused_parameters | True |

**学习率调整说明：**
- 原始 batch_size=8 时 lr=1e-4
- 现在 batch_size=2，按比例缩放：lr = 1e-4 × (2/8) = 2.5e-5

## Decoder Neck 工作方式

### 跨视图融合流程

```
(B*6, 1024, 16, 44)  [ResNet stage2 输出]
        ↓
reshape to (B, 6, 1024, 16, 44)
        ↓
flatten to tokens
        ↓
(B, 6*704, 1024)  [6张图的704个patch拼接]
        ↓
feat_embed_enc_to_dec: 1024→768
        ↓
(B, 4224, 768)  [6*704=4224]
        ↓
12× CachedDecoderBlock (跨视图 attention)
  - Self-Attention: 所有 4224 个 tokens 之间的自注意力
  - Cross-Attention: memory 来自自身 tokens
  - MLP
  - RoPE positional embedding
        ↓
(B, 4224, 768)  [融合后的特征]
        ↓
LayerNorm + reshape
        ↓
(B, 6, 768, 16, 44)
        ↓
reshape to (B*6, 768, 16, 44)
        ↓
output_proj: Conv1x1(768→256) + BN + ReLU
        ↓
(B*6, 256, 16, 44)  [LSS View Transformer 输入]
```

### 关键特性

1. **跨视图融合** - 6 张图的 tokens 在 decoder 中进行全局 attention，实现跨视图特征交互
2. **每张图独立输出** - 融合后仍然输出每张图的特征，保持与 LSS 的兼容性
3. **预训练权重** - decoder 的 12 个 block 从 MUSt3R 加载预训练权重
4. **显存占用** - 比单视图版本增加约 6 倍（attention 矩阵从 704×704 变成 4224×4224）

## 与单视图版本的对比

| 特性 | 单视图 (Must3rDecoderNeck) | 多视图 (Must3rDecoderNeckMultiView) |
|------|---------------------------|-----------------------------------|
| 输入处理 | 每张图独立处理 | 6张图一起处理 |
| Attention 矩阵 | 6 个 (704, 704) | 1 个 (4224, 4224) |
| 跨视图融合 | 无 | 在 decoder 中进行 |
| 显存占用 | 基准 | 约 6 倍 |
| 学习率 | 1e-4 (batch_size=8) | 2.5e-5 (batch_size=2) |
| 预期性能 | 基准 | 可能更好（利用多视图信息） |

## 文件清单

| 文件 | 说明 |
|------|------|
| `configs/bevdet/bevdet-r50-must3r-decoder-multiview.py` | 配置文件 |
| `mmdet3d/models/necks/must3r_decoder_neck_multiview.py` | Must3rDecoderNeckMultiView 模块 |
| `mmdet3d/models/necks/__init__.py` | 注册入口 |

## 关键实现细节

### Must3rDecoderNeckMultiView 特性

1. **权重加载**
   - 从 MUSt3R 预训练检查点加载 decoder 权重
   - 缺失权重会报错，不会随机初始化
   - 支持权重重命名（DUSt3R/CroCo → MUSt3R 命名规范）

2. **梯度检查点**
   - 已关闭 (`with_cp=False`) 以避免 DDP 兼容性问题
   - 如需节省显存，可重新启用但需处理 DDP 参数标记问题

3. **冻结策略**
   - `frozen=False`: 所有参数可训练
   - `frozen=True`: 冻结除 output_proj 外的所有参数
   - `frozen_blocks=N`: 冻结前 N+1 个 decoder blocks

4. **DDP 配置**
   - `find_unused_parameters=True`: ResNet layer4 未使用但保留以保持与 baseline 对齐

## 训练命令

```bash
# 单 GPU
python tools/train.py configs/bevdet/bevdet-r50-must3r-decoder-multiview.py

# 多 GPU (DDP)
./tools/dist_train.sh configs/bevdet/bevdet-r50-must3r-decoder-multiview.py <NUM_GPUS>
```

## 已知问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| DDP "parameter marked ready twice" | 梯度检查点与 DDP reducer 交互 | 关闭 `with_cp=True` |
| ResNet layer4 未使用 | `out_indices=(2,)` 只输出 stage2 | 设置 `find_unused_parameters=True` |
| 权重加载失败 | 检查点路径错误或格式不匹配 | 确保路径正确，检查点包含 'decoder' 键 |
| 显存不足 | 跨视图 attention 矩阵过大 | 减小 batch_size 或关闭梯度检查点 |
| cuDNN 算法查找失败 | 多 GPU 验证时数据分割导致特殊形状 | 禁用 cuDNN 自动算法查找 |

## 性能预期

相比单视图版本：
- **优势**：利用多视图信息进行特征融合，可能提升检测性能
- **劣势**：显存占用增加 6 倍，训练速度变慢，需要更小的 batch size

## 扩展方向

1. **显存优化**
   - 使用 Flash Attention 加速 attention 计算
   - 实现局部 attention（只融合相邻视图）
   - 使用低秩分解减少 attention 矩阵大小

2. **性能优化**
   - 调整 decoder 的冻结策略
   - 尝试不同的学习率调度
   - 使用更大的 batch size（如果显存允许）

3. **架构改进**
   - 在 decoder 之前添加视图对齐模块
   - 使用相机参数指导跨视图融合
   - 结合 LSS 的几何约束
