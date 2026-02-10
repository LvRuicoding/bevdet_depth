# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from ..builder import NECKS


@NECKS.register_module()
class DepthModulationNetwork(BaseModule):
    """Lightweight CNN that predicts per-pixel delta for depth refinement.

    This network takes image features from the backbone and predicts a per-pixel
    multiplicative modulation coefficient (delta) to refine pre-extracted depth maps.

    Args:
        in_channels (int): Number of input feature channels. Default: 256.
        mid_channels (int): Number of hidden layer channels. Default: 128.
        delta_activation (str): Activation function for delta output.
            Options: 'sigmoid_scale' (sigmoid * 2.0) or 'tanh_shift' (1.0 + tanh).
            Default: 'sigmoid_scale'.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels=256,
                 mid_channels=128,
                 delta_activation='sigmoid_scale',
                 init_cfg=None):
        super(DepthModulationNetwork, self).__init__(init_cfg)

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.delta_activation = delta_activation

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.relu3 = nn.ReLU(inplace=True)

        # Final 1x1 convolution to predict delta
        self.conv_delta = nn.Conv2d(mid_channels, 1, kernel_size=1,
                                     stride=1, padding=0)

        # Initialize final layer bias to 0 for initial delta ≈ 1.0
        nn.init.constant_(self.conv_delta.bias, 0.0)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input image features of shape (B*N, C, H, W).

        Returns:
            torch.Tensor: Per-pixel delta of shape (B*N, 1, H, W) in range (0, 2).
        """
        # Feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Predict delta
        delta = self.conv_delta(x)

        # Apply activation to map to (0, 2) range
        if self.delta_activation == 'sigmoid_scale':
            # sigmoid(0) = 0.5, so sigmoid(0) * 2.0 = 1.0
            delta = torch.sigmoid(delta) * 2.0
        elif self.delta_activation == 'tanh_shift':
            # tanh(0) = 0, so 1.0 + tanh(0) = 1.0
            delta = 1.0 + torch.tanh(delta)
        else:
            raise ValueError(f"Unknown delta_activation: {self.delta_activation}")

        return delta
