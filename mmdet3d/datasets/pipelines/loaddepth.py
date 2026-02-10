import numpy as np
import os
import torch
import torch.nn.functional as F
from PIL import Image
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadPretrainedDepth:
    """加载预提取的深度图

    这个pipeline应该在PrepareImageInputs之后调用，以便获取相机信息。
    深度图将被调整到与图像特征相同的分辨率。

    Args:
        depth_root (str): 深度图根目录路径
        downsample (int): 下采样倍数，用于将深度图调整到特征分辨率
    """
    def __init__(self,
                 depth_root='data/nuscenes-depth/samples/',
                 downsample=16):
        self.depth_root = depth_root
        self.downsample = downsample

    def __call__(self, results):
        """加载深度图并调整到特征分辨率

        Args:
            results (dict): 包含图像信息的字典

        Returns:
            dict: 添加了pretrained_depths字段的results
        """
        depths = []

        # 获取目标尺寸（特征分辨率）
        # img_inputs中的图像尺寸是 (N, C, H, W)
        if 'img_inputs' in results:
            imgs = results['img_inputs'][0]  # (N, C, H, W)
            _, _, H_img, W_img = imgs.shape
            H_feat = H_img // self.downsample
            W_feat = W_img // self.downsample
        else:
            # 如果还没有img_inputs，使用默认尺寸
            H_feat, W_feat = 16, 44  # 默认: 256//16, 704//16

        # 获取相机名称列表
        if 'cam_names' in results:
            # PrepareImageInputs已经处理过，使用cam_names
            cam_names = results['cam_names']
            # 从curr中获取相机数据路径
            if 'curr' in results:
                for cam_name in cam_names:
                    cam_data = results['curr']['cams'][cam_name]
                    img_path = cam_data['data_path']

                    # 提取文件名信息
                    parts = img_path.split('/')
                    cam_folder = parts[-2]  # CAM_FRONT等
                    basename = os.path.splitext(parts[-1])[0]  # 去掉扩展名

                    depth_path = os.path.join(
                        self.depth_root, cam_folder, basename + '.npy')

                    # 加载深度图
                    if os.path.exists(depth_path):
                        depth = np.load(depth_path).astype(np.float32)  # (H, W)
                    else:
                        # 如果深度图不存在，创建全零深度图
                        print(f"Warning: Depth file not found: {depth_path}, using zeros")
                        depth = np.zeros((H_feat, W_feat), dtype=np.float32)
                        depths.append(depth)
                        continue

                    # 调整到特征分辨率（使用最近邻插值）
                    depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                    depth_tensor = F.interpolate(
                        depth_tensor,
                        size=(H_feat, W_feat),
                        mode='nearest'
                    )
                    depth = depth_tensor.squeeze().numpy()  # (H_feat, W_feat)

                    depths.append(depth)
            else:
                raise KeyError("'curr' not found in results. LoadPretrainedDepth should be called after PrepareImageInputs.")
        elif 'img_filename' in results:
            # 兼容旧版本的数据格式
            for img_path in results['img_filename']:
                parts = img_path.split('/')
                cam_folder = parts[-2]
                basename = os.path.splitext(parts[-1])[0]

                depth_path = os.path.join(
                    self.depth_root, cam_folder, basename + '.npy')

                if os.path.exists(depth_path):
                    depth = np.load(depth_path).astype(np.float32)
                else:
                    print(f"Warning: Depth file not found: {depth_path}, using zeros")
                    depth = np.zeros((H_feat, W_feat), dtype=np.float32)
                    depths.append(depth)
                    continue

                depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
                depth_tensor = F.interpolate(
                    depth_tensor,
                    size=(H_feat, W_feat),
                    mode='nearest'
                )
                depth = depth_tensor.squeeze().numpy()
                depths.append(depth)
        else:
            raise KeyError("Neither 'cam_names' nor 'img_filename' found in results. "
                         "LoadPretrainedDepth should be called after PrepareImageInputs.")

        # 转换为张量 (N, H, W)
        depths_tensor = torch.stack([
            torch.from_numpy(d) for d in depths
        ], dim=0)

        results['pretrained_depths'] = depths_tensor
        return results