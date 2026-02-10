# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from torch.cuda.amp.autocast_mode import autocast

from .view_transformer import LSSViewTransformer
from .depth_modulation import DepthModulationNetwork
from ..builder import NECKS


@NECKS.register_module()
class LSSViewTransformerDepthModulation(LSSViewTransformer):
    """LSS View Transformer with Depth Modulation Network.

    This extends the standard LSSViewTransformer to refine pre-extracted depth maps
    using a learned per-pixel multiplicative modulation (delta).

    Args:
        depth_modulation (dict): Config for DepthModulationNetwork.
        use_depth_loss (bool): Whether to compute depth supervision loss. Default: False.
        depth_loss_weight (float): Weight for depth supervision loss. Default: 1.0.
        **kwargs: Arguments for LSSViewTransformer.
    """

    def __init__(self,
                 depth_modulation=dict(
                     in_channels=256,
                     mid_channels=128,
                     delta_activation='sigmoid_scale'),
                 use_depth_loss=False,
                 depth_loss_weight=1.0,
                 **kwargs):
        super(LSSViewTransformerDepthModulation, self).__init__(**kwargs)

        # Build depth modulation network
        self.depth_modulation_net = DepthModulationNetwork(**depth_modulation)

        # Depth supervision settings
        self.use_depth_loss = use_depth_loss
        self.depth_loss_weight = depth_loss_weight

        # Remove the original depth_net since we use pretrained depth
        del self.depth_net

        # Create a simple conv for context features (tran_feat)
        self.context_net = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=1, padding=0)

    def depth_to_distribution(self, depth):
        """Convert continuous depth values to discrete depth distribution.

        Args:
            depth (torch.Tensor): Continuous depth values of shape (B*N, H, W).

        Returns:
            torch.Tensor: Depth distribution of shape (B*N, D, H, W).
        """
        B_N, H, W = depth.shape

        # Get depth bin configuration
        depth_min = self.grid_config['depth'][0]
        depth_max = self.grid_config['depth'][1]
        depth_interval = self.grid_config['depth'][2]

        # Discretize depth into bins (hard assignment)
        if not self.sid:
            # Linear discretization
            depth_idx = ((depth - depth_min) / depth_interval).long()
        else:
            # Spacing Increasing Discretization (logarithmic)
            depth_idx = torch.log(depth.clamp(min=depth_min)) - torch.log(
                torch.tensor(depth_min).float().to(depth))
            depth_idx = depth_idx * (self.D - 1) / torch.log(
                torch.tensor(depth_max - 1.).float().to(depth) / depth_min)
            depth_idx = (depth_idx + 1.).long()

        # Clamp to valid range
        depth_idx = depth_idx.clamp(0, self.D - 1)

        # Convert to one-hot distribution
        depth_dist = F.one_hot(depth_idx, num_classes=self.D).float()
        depth_dist = depth_dist.permute(0, 3, 1, 2).contiguous()  # (B*N, D, H, W)

        return depth_dist

    def forward(self, input, pretrained_depth=None):
        """Transform image-view feature into bird-eye-view feature with depth modulation.

        Args:
            input (list(torch.tensor)): List of (image-view feature, rots, trans,
                intrins, post_rots, post_trans, bda).
            pretrained_depth (torch.Tensor or list, optional): Pre-extracted depth maps of shape
                (B, N, H_depth, W_depth) or list of tensors. If None, will use standard depth prediction.

        Returns:
            tuple: (bev_feat, refined_depth)
                - bev_feat (torch.Tensor): Bird-eye-view feature in shape (B, C, H_BEV, W_BEV)
                - refined_depth (torch.Tensor): Refined depth maps in shape (B, N, H, W)
        """
        x = input[0]  # Image features
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)

        # Generate context features for view transformation
        tran_feat = self.context_net(x)

        if pretrained_depth is not None:
            # Handle list input (from test dataloader)
            if isinstance(pretrained_depth, list):
                pretrained_depth = pretrained_depth[0]

            # Ensure pretrained_depth is a tensor
            if not isinstance(pretrained_depth, torch.Tensor):
                raise TypeError(f"pretrained_depth must be a tensor or list of tensors, got {type(pretrained_depth)}")

            # Add batch dimension if needed (N, H, W) -> (1, N, H, W)
            if pretrained_depth.dim() == 3:
                pretrained_depth = pretrained_depth.unsqueeze(0)

            # Use depth modulation network to refine pretrained depth
            # Predict per-pixel delta
            delta = self.depth_modulation_net(x)  # (B*N, 1, H, W)

            # Resize pretrained depth to match feature resolution if needed
            _, _, H_depth, W_depth = pretrained_depth.shape
            if H_depth != H or W_depth != W:
                pretrained_depth = F.interpolate(
                    pretrained_depth.view(B * N, 1, H_depth, W_depth),
                    size=(H, W),
                    mode='nearest')
            else:
                pretrained_depth = pretrained_depth.view(B * N, 1, H, W)

            # Apply depth modulation: refined_depth = pretrained_depth * delta
            refined_depth = pretrained_depth * delta  # (B*N, 1, H, W)
            refined_depth = refined_depth.squeeze(1)  # (B*N, H, W)

            # Convert refined depth to depth distribution
            depth = self.depth_to_distribution(refined_depth)  # (B*N, D, H, W)

            # Reshape refined_depth for output
            refined_depth = refined_depth.view(B, N, H, W)
        else:
            # Fallback: use standard depth prediction (should not happen in normal use)
            raise NotImplementedError(
                "LSSViewTransformerDepthModulation requires pretrained_depth input")

        # Apply softmax to depth distribution
        depth = depth.softmax(dim=1)

        # Perform view transformation
        bev_feat, depth = self.view_transform(input, depth, tran_feat)

        return bev_feat, refined_depth

    @force_fp32()
    def get_depth_loss(self, refined_depth, gt_depth):
        """Compute depth supervision loss.

        Args:
            refined_depth (torch.Tensor): Refined depth predictions of shape (B, N, H, W).
            gt_depth (torch.Tensor): Ground truth depth from LiDAR of shape (B, N, H_gt, W_gt).

        Returns:
            torch.Tensor: Depth loss scalar.
        """
        B, N, H, W = refined_depth.shape
        _, _, H_gt, W_gt = gt_depth.shape

        # Downsample GT depth to match refined depth resolution if needed
        if H_gt != H or W_gt != W:
            gt_depth = F.interpolate(
                gt_depth.view(B * N, 1, H_gt, W_gt),
                size=(H, W),
                mode='nearest').view(B, N, H, W)

        # Flatten for loss computation
        refined_depth = refined_depth.reshape(-1)
        gt_depth = gt_depth.reshape(-1)

        # Mask: only compute loss where GT depth > 0 (valid LiDAR points)
        mask = gt_depth > 0

        if mask.sum() == 0:
            return refined_depth.new_tensor(0.0)

        # Compute SmoothL1 loss
        with autocast(enabled=False):
            loss = F.smooth_l1_loss(
                refined_depth[mask],
                gt_depth[mask],
                reduction='mean')

        return self.depth_loss_weight * loss
