# BEVDet-R50-MUSt3R-Decoder 模型结构文档

## 概述

在标准 BEVDet-R50 的基础上，将 neck 从 CustomFPN 替换为 MUSt3R 的 decoder blocks，
作为特征增强层插入 ResNet-50 backbone 和 LSS View Transformer 之间。

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
          ▼
┌─────────────────────────────────────────┐
│  Must3rDecoderNeck                      │  pretrained: MUSt3R_224_cvpr.pth (decoder部分)
│                                         │
│  1. flatten → tokens  (B*6, 704, 1024)  │
│  2. feat_embed_enc_to_dec               │  Linear(1024→768)     0.79M
│     tokens → (B*6, 704, 768)            │
│  3. 12× CachedDecoderBlock              │  每张图独立处理       113.44M
│     - Self-Attention (768-dim, 12 heads) │
│     - Cross-Attention (attend自身tokens) │
│     - MLP (768→3072→768)                │
│     - RoPE positional embedding         │
│  4. LayerNorm                           │
│  5. reshape → (B*6, 768, 16, 44)        │
│  6. output_proj Conv1x1(768→256)+BN+ReLU│  0.20M (随机初始化)
│                                         │
│  114.43M params total                   │
└─────────┬───────────────────────────────┘
          │ (B*6, 256, 16, 44)
          ▼
     reshape → (B, 6, 256, 16, 44)
          │
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
| Must3rDecoderNeck (decoder部分) | 114.23M | MUSt3R_224_cvpr.pth |
| Must3rDecoderNeck (output_proj) | 0.20M | 随机初始化 |
| LSSViewTransformer | 0.03M | 随机初始化 |
| BEV encoder backbone | 12.39M | 随机初始化 |
| BEV encoder neck | 6.56M | 随机初始化 |
| CenterHead | 0.38M | 随机初始化 |
| **总计** | **157.30M** | **全部可训练** |

## 训练设置 (与 bevdet-r50.py 完全对齐)

| 设置 | 值 |
|------|-----|
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-7 |
| Grad clip | max_norm=5, norm_type=2 |
| LR schedule | step decay at epoch 24, linear warmup 200 iters |
| Epochs | 24 |
| Batch size | 8 per GPU |
| Workers | 16 per GPU |
| EMA | MEGVIIEMAHook, init_updates=10560 |
| Input size | (256, 704) |
| BDA augmentation | rot=±22.5°, scale=0.95~1.05, flip=0.5 |
| Gradient checkpointing | 关闭 (with_cp=False) |
| DDP find_unused_parameters | True |

## Decoder Neck 工作方式

每张图独立通过 decoder（nimgs=1），不做跨视图融合：
- Self-Attention: 图像 tokens 之间的自注意力
- Cross-Attention: memory 来自自身 tokens（等效于另一种形式的自注意力）
- 跨视图融合由下游的 LSS View Transformer 负责

## 文件清单

| 文件 | 说明 |
|------|------|
| `configs/bevdet/bevdet-r50-must3r-decoder.py` | 配置文件 |
| `mmdet3d/models/necks/must3r_decoder_neck.py` | Must3rDecoderNeck 模块 |
| `mmdet3d/models/necks/__init__.py` | 注册入口 |

## 关键实现细节

### Must3rDecoderNeck 特性

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

### 数据流

```
(B*6, 1024, 16, 44)  [ResNet stage2 输出]
        ↓
    flatten to tokens
        ↓
(B*6, 704, 1024)  [704 = 16*44 patches]
        ↓
feat_embed_enc_to_dec: 1024→768
        ↓
(B*6, 704, 768)
        ↓
12× CachedDecoderBlock (每张图独立)
  - Self-Attention on tokens
  - Cross-Attention (memory from self)
  - MLP
  - RoPE positional embedding
        ↓
(B*6, 704, 768)
        ↓
LayerNorm + reshape
        ↓
(B*6, 768, 16, 44)
        ↓
output_proj: Conv1x1(768→256) + BN + ReLU
        ↓
(B*6, 256, 16, 44)  [LSS View Transformer 输入]
```

## 训练命令

```bash
# 单 GPU
python tools/train.py configs/bevdet/bevdet-r50-must3r-decoder.py

# 多 GPU (DDP)
./tools/dist_train.sh configs/bevdet/bevdet-r50-must3r-decoder.py <NUM_GPUS>
```

## 已知问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| DDP "parameter marked ready twice" | 梯度检查点与 DDP reducer 交互 | 关闭 `with_cp=True` |
| ResNet layer4 未使用 | `out_indices=(2,)` 只输出 stage2 | 设置 `find_unused_parameters=True` |
| 权重加载失败 | 检查点路径错误或格式不匹配 | 确保路径正确，检查点包含 'decoder' 键 |

