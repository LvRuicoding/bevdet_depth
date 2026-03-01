"""Adapter neck to bridge MapAnything features to BEVDet view transformer."""
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule

from ..builder import NECKS


@NECKS.register_module()
class MapAnythingAdapterNeck(BaseModule):
    """Adapts MapAnything features to BEVDet view transformer input.

    Handles:
    1. Spatial interpolation from (H/14, W/14) to (H/16, W/16)
    2. Channel projection from 1536 to out_channels (e.g. 256)

    Args:
        in_channels (int): Input channels from MapAnything. Default 1536.
        out_channels (int): Output channels for view transformer. Default 256.
        target_h (int): Target feature height (input_h // downsample). Default 16.
        target_w (int): Target feature width (input_w // downsample). Default 44.
        num_convs (int): Number of conv layers. Default 2.
    """

    def __init__(self,
                 in_channels=1536,
                 out_channels=256,
                 target_h=16,
                 target_w=44,
                 num_convs=2):
        super().__init__()
        self.target_h = target_h
        self.target_w = target_w

        layers = [
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_convs - 1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B*N, 1536, H_p, W_p) e.g. (B*6, 1536, 19, 51)
        Returns:
            (B*N, out_channels, target_h, target_w) e.g. (B*6, 256, 16, 44)
        """
        if x.shape[2] != self.target_h or x.shape[3] != self.target_w:
            x = F.interpolate(x, size=(self.target_h, self.target_w),
                              mode='bilinear', align_corners=False)
        x = self.proj(x)
        return x
