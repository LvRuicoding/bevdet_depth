# BEVDet 模型结构对比：ResNet50 Baseline vs MUSt3R Encoder Backbone

## 整体 Pipeline

两个模型共享相同的 BEVDet pipeline，仅 img_backbone 和 img_neck 不同：

```
img_backbone → img_neck (FPN) → LSSViewTransformer → BEV Encoder Backbone → BEV Encoder Neck → CenterHead
```

---

## 1. Image Backbone

### Baseline: ResNet50

```
输入: (B×6, 3, 256, 704)

ResNet50 (pretrained=torchvision, frozen_stages=-1, 全部可训练)
├─ stem + stage1:  → (B×6,  256, 64, 176)   不输出
├─ stage2:         → (B×6, 1024, 16,  44)   ← out_indices=(2,)
└─ stage3:         → (B×6, 2048,  8,  22)   ← out_indices=(3,)

输出: 2 个尺度 [(B×6, 1024, 16, 44), (B×6, 2048, 8, 22)]
```

### MUSt3R: Dust3rEncoder (ViT-Large)

```
输入: (B×6, 3, 256, 704)

Dust3rEncoder (pretrained=MUSt3R_224_cvpr.pth, frozen=False, lr_mult=0.1)
├─ PatchEmbedDust3R:  Conv2d(3, 1024, k=16, s=16) → (B×6, 704, 1024)  [704 = 16×44 patches]
│                      + RoPE 位置编码
├─ blocks_enc × 24:   Transformer Block (dim=1024, heads=16, mlp_ratio=4)
│                      每个 block: LayerNorm → Attention(+RoPE) → LayerNorm → MLP
├─ norm_enc:           LayerNorm(1024)
└─ reshape:            (B×6, 704, 1024) → (B×6, 1024, 16, 44)

输出: 1 个尺度 [(B×6, 1024, 16, 44)]
```

### 对比

| | ResNet50 | MUSt3R Encoder |
|---|---|---|
| 架构 | CNN (卷积) | ViT (自注意力) |
| 输出尺度数 | 2 (1/16 + 1/32) | 1 (1/16) |
| 输出通道 | [1024, 2048] | [1024] |
| 输出空间尺寸 | [(16,44), (8,22)] | [(16,44)] |
| 感受野 | 局部→逐层扩大 | 全局 (self-attention) |
| 位置编码 | 无 (隐式由卷积提供) | RoPE (旋转位置编码) |
| 归一化 | BatchNorm | LayerNorm |
| 预训练来源 | ImageNet 分类 | MUSt3R 双目3D重建 |
| 参数量 | ~23M | ~300M |
| 训练时 lr | 1e-4 | 1e-5 (lr_mult=0.1) |

---

## 2. Image Neck (CustomFPN)

### Baseline

```
输入: [(B×6, 1024, 16, 44), (B×6, 2048, 8, 22)]

CustomFPN(in_channels=[1024, 2048], out_channels=256, out_ids=[0])
├─ lateral_conv[0]: Conv2d(1024, 256, 1×1) on stage2 → (B×6, 256, 16, 44)
├─ lateral_conv[1]: Conv2d(2048, 256, 1×1) on stage3 → (B×6, 256,  8, 22)
├─ top-down:        upsample lateral[1] → (B×6, 256, 16, 44)
│                   lateral[0] += upsampled lateral[1]   ← 多尺度融合
└─ fpn_conv[0]:     Conv2d(256, 256, 3×3) → (B×6, 256, 16, 44)

输出: (B×6, 256, 16, 44)
```

### MUSt3R

```
输入: [(B×6, 1024, 16, 44)]

CustomFPN(in_channels=[1024], out_channels=256, out_ids=[0])
├─ lateral_conv[0]: Conv2d(1024, 256, 1×1) → (B×6, 256, 16, 44)
├─ top-down:        无 (只有一个尺度，循环不执行)
└─ fpn_conv[0]:     Conv2d(256, 256, 3×3) → (B×6, 256, 16, 44)

输出: (B×6, 256, 16, 44)
```

FPN 之后两者输出形状完全一致: `(B×6, 256, 16, 44)`。

---

## 3. 共享的下游模块 (两者完全相同)

### LSSViewTransformer

```
输入: (B, 6, 256, 16, 44) + 相机内外参

reshape:    (B×6, 256, 16, 44)
depth_net:  Conv2d(256, 60+64, 1×1) → (B×6, 124, 16, 44)
            ├─ depth:     (B×6, 60, 16, 44)  softmax → 深度概率分布 (1~60m, 1m间隔)
            └─ tran_feat: (B×6, 64, 16, 44)  上下文特征
Lift-Splat: 根据深度分布 + 相机参数，将2D特征投射到3D BEV网格
BEV Pool:   聚合到 (B, 64, 128, 128) BEV特征图

输出: (B, 64, 128, 128)
```

### BEV Encoder Backbone (CustomResNet)

```
输入: (B, 64, 128, 128)

CustomResNet(numC_input=64, num_channels=[128, 256, 512])
├─ stage0: BasicBlock × 2, stride=2 → (B, 128,  64,  64)  ← backbone_output_ids[0]
├─ stage1: BasicBlock × 2, stride=2 → (B, 256,  32,  32)
└─ stage2: BasicBlock × 2, stride=2 → (B, 512,  16,  16)  ← backbone_output_ids[2]

输出: [(B, 128, 64, 64), (B, 512, 16, 16)]
```

### BEV Encoder Neck (FPN_LSS)

```
输入: [(B, 128, 64, 64), (B, 512, 16, 16)]

FPN_LSS(in_channels=512+128=640, out_channels=256)
├─ upsample stage2:  (B, 512, 16, 16) → (B, 512, 64, 64)
├─ concat with stage0: → (B, 640, 64, 64)
├─ input_conv:        Conv-BN-ReLU → (B, 128, 64, 64)
└─ extra_upsample:    Upsample(×2) + Conv-BN-ReLU → (B, 256, 128, 128)

输出: (B, 256, 128, 128)
```

### CenterHead

```
输入: (B, 256, 128, 128)

CenterHead(in_channels=256, 10 classes, single task)
├─ shared_conv: Conv2d(256, 64, 3×3)
├─ heatmap head: Conv(64,64,3) → Conv(64,10,1)     → (B, 10, 128, 128)
├─ reg head:     Conv(64,64,3) → Conv(64, 2,1)     → (B,  2, 128, 128)
├─ height head:  Conv(64,64,3) → Conv(64, 1,1)     → (B,  1, 128, 128)
├─ dim head:     Conv(64,64,3) → Conv(64, 3,1)     → (B,  3, 128, 128)
├─ rot head:     Conv(64,64,3) → Conv(64, 2,1)     → (B,  2, 128, 128)
└─ vel head:     Conv(64,64,3) → Conv(64, 2,1)     → (B,  2, 128, 128)

输出: 10类目标的 heatmap + bbox 回归
```

---

## 4. 完整数据流对比

```
                    Baseline (ResNet50)                    MUSt3R Encoder
                    ──────────────────                     ──────────────

输入图像             (B×6, 3, 256, 704)                    (B×6, 3, 256, 704)
                          │                                      │
Backbone            ResNet50 (CNN, ~23M)                  Dust3rEncoder (ViT-L, ~300M)
                    ├─ (B×6, 1024, 16, 44)                └─ (B×6, 1024, 16, 44)
                    └─ (B×6, 2048,  8, 22)
                          │                                      │
FPN                 多尺度 top-down 融合                    单尺度 1×1+3×3 conv
                    → (B×6, 256, 16, 44)                  → (B×6, 256, 16, 44)
                          │                                      │
                          └──────────────┬───────────────────────┘
                                         │
                                         ▼
LSSViewTransformer          depth_net → Lift-Splat → BEV Pool
                                → (B, 64, 128, 128)
                                         │
BEV Encoder                 CustomResNet → FPN_LSS
                                → (B, 256, 128, 128)
                                         │
CenterHead                  heatmap + bbox regression
                                → 10类 3D 检测结果
```

---

## 5. 训练配置

| 配置项 | Baseline | MUSt3R |
|--------|----------|--------|
| optimizer | AdamW | AdamW |
| backbone lr | 1e-4 | 1e-5 (lr_mult=0.1) |
| 其余模块 lr | 1e-4 | 1e-4 |
| weight_decay | 1e-7 | 1e-7 |
| grad_clip | max_norm=5 | max_norm=5 |
| lr_schedule | step=[24], warmup=200 | step=[24], warmup=200 |
| epochs | 24 | 24 |
| batch_size | 8/gpu | 8/gpu |
| EMA | init_updates=10560 | init_updates=10560 |
| 数据增强 | 相同 | 相同 |
