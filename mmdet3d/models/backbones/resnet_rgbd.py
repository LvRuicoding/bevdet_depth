# ====================================================================
# 新增: 适配 RGBD (4通道) 的 ResNet
# ====================================================================

from mmdet.models.backbones import ResNet
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES
import torch
import torch.nn as nn

@BACKBONES.register_module()
class ResNetRGBD(ResNet):
    """
    专门为 RGBD (4通道) 设计的 ResNet。
    继承自 MMDetection 的 ResNet，只修改第一层卷积结构，
    并自动处理权重加载时的维度匹配问题。
    """
    def __init__(self, in_channels=4, **kwargs):
        # 1. 调用父类初始化
        # 父类会默认创建 in_channels=3 的 conv1，我们稍后替换
        super(ResNetRGBD, self).__init__(**kwargs)
        
        self.in_channels = in_channels
        
        # 2. 如果输入不是3通道，我们需要替换掉父类创建的 conv1
        if self.in_channels != 3:
            # 获取原 conv1 的配置 (kernel_size, stride, padding 等)
            cfg = dict(
                in_channels=self.in_channels,
                out_channels=self.conv1.out_channels,
                kernel_size=self.conv1.kernel_size,
                stride=self.conv1.stride,
                padding=self.conv1.padding,
                bias=self.conv1.bias is not None
            )
            # 替换为新的 Conv2d (参数继承原 ResNet 设置)
            self.conv1 = nn.Conv2d(**cfg)
            # BN 层不需要改，因为输出通道数依然是 64

    def init_weights(self):
        """
        重写权重初始化逻辑：
        1. 加载官方 3通道 预训练权重
        2. 自动把 conv1 的权重从 3通道 扩展到 4通道 (第4通道初始化为0)
        """
        if isinstance(self.init_cfg, dict) and self.init_cfg.get('type') == 'Pretrained':
            # 获取 checkpoint 路径
            checkpoint_path = self.init_cfg['checkpoint']
            logger = get_root_logger()
            logger.info(f'ResNetRGBD: 正在加载并适配权重: {checkpoint_path}')
            
            # 1. 加载 checkpoint 到 CPU
            checkpoint = load_checkpoint(self, checkpoint_path, map_location='cpu', strict=False, logger=logger)
            
            # 2. 手动处理 conv1 权重的形状不匹配问题
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 寻找 conv1 键名 (兼容不同来源的权重)
            target_key = None
            for key in state_dict.keys():
                # 常见名称: 'conv1.weight' 或 'backbone.conv1.weight'
                if 'conv1.weight' in key and 'layer' not in key: 
                    target_key = key
                    break
            
            if target_key:
                w_rgb = state_dict[target_key] # [64, 3, 7, 7]
                
                # 检查是否需要转换 (当模型是4通道，权重是3通道时)
                if w_rgb.shape[1] == 3 and self.in_channels == 4:
                    # 创建 4通道 权重容器
                    w_rgbd = torch.zeros_like(self.conv1.weight) # [64, 4, 7, 7]
                    
                    # 复制前3个通道 (RGB)
                    w_rgbd[:, :3, :, :] = w_rgb
                    # 第4个通道 (Depth) 初始化微小高斯噪声
                    nn.init.normal_(w_rgbd[:, 3, :, :], mean=0, std=0.01)
                    # 强制覆盖当前模型的 conv1 权重
                    self.conv1.weight.data = w_rgbd.to(self.conv1.weight.device)
                    logger.info(f'ResNetRGBD: 成功将 {target_key} 从 {w_rgb.shape} 扩展为 {w_rgbd.shape} (Depth initialized to 0)')
                elif w_rgb.shape == self.conv1.weight.shape:
                    logger.info(f'ResNetRGBD: 权重形状已匹配 {w_rgb.shape}，无需转换。')
            else:
                logger.warning('ResNetRGBD: 在预训练权重中没找到 conv1.weight，将使用随机初始化。')
                
        else:
            # 如果没有指定 Pretrained，使用默认初始化
            super(ResNetRGBD, self).init_weights()