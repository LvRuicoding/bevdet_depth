#!/usr/bin/env python
"""Test script for temporal model with FPN."""
import torch
import sys
sys.path.insert(0, '/home/batchcom/lr/BEVDet')

from mmcv import Config
from mmdet3d.models import build_model

def test_model():
    cfg = Config.fromfile('configs/bevdet/bevdet-r50-must3r-decoder-temporal.py')

    # Build model
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.eval()

    # Move to CUDA if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print("Model built successfully!")
    print(f"  - Device: {device}")
    print(f"  - Backbone out_indices: {model.img_backbone.out_indices}")
    print(f"  - Has img_neck (FPN): {model.with_img_neck}")
    print(f"  - Has img_neck_3d (MUSt3R): {hasattr(model, 'img_neck_3d') and model.img_neck_3d is not None}")

    if model.with_img_neck:
        print(f"  - FPN type: {type(model.img_neck).__name__}")

    if hasattr(model, 'img_neck_3d') and model.img_neck_3d is not None:
        print(f"  - MUSt3R neck type: {type(model.img_neck_3d).__name__}")
        print(f"  - MUSt3R in_channels: {model.img_neck_3d.in_channels}")
        print(f"  - MUSt3R enc_embed_dim: {model.img_neck_3d.enc_embed_dim}")
        print(f"  - MUSt3R embed_dim: {model.img_neck_3d.embed_dim}")
        print(f"  - MUSt3R depth: {len(model.img_neck_3d.blocks_dec)}")
        print(f"  - MUSt3R num_views: {model.img_neck_3d.num_views}")
        print(f"  - MUSt3R view_embed shape: {model.img_neck_3d.view_embed.shape}")
        print(f"  - MUSt3R hist_frame_embed shape: {model.img_neck_3d.hist_frame_embed.shape}")

    # Test forward pass
    B, N, C, H, W = 1, 6, 3, 256, 704
    num_adj = 2
    num_frame = num_adj + 1

    imgs = torch.randn(B, N * num_frame, C, H, W).to(device)
    sensor2egos = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, num_frame * N, 1, 1).to(device)
    ego2globals = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, num_frame * N, 1, 1).to(device)
    intrins = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, num_frame * N, 1, 1).to(device)
    post_rots = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, num_frame * N, 1, 1).to(device)
    post_trans = torch.zeros(B, num_frame * N, 3).to(device)
    bda = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(device)

    img_inputs = [imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda]

    print("\nTesting forward pass...")
    with torch.no_grad():
        try:
            feats, depth = model.extract_img_feat(img_inputs, img_metas=[{}])
            print(f"✓ Forward pass successful!")
            print(f"  - Output shape: {feats[0].shape}")
            print(f"  - Depth shape: {depth.shape}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    test_model()
