# Copyright (c) Phigent Robotics. All rights reserved.
# BEVDet with Depth Modulation Network (带深度监督版本)
# 使用预提取的深度图，通过深度调制网络进行精化，并使用LiDAR深度监督

_base_ = ['./bevdet-r50-depth-modulation.py']

# 启用深度监督
model = dict(
    use_depth_loss=True,  # 启用深度监督损失
    img_view_transformer=dict(
        use_depth_loss=True,
        depth_loss_weight=1.0,
    ),
)

# 训练数据pipeline（需要加载点云用于生成GT深度）
train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=_base_.data_config),
    dict(type='LoadAnnotations'),
    dict(
        type='LoadPretrainedDepth',
        depth_root=_base_.depth_root,
        downsample=16),
    dict(
        type='LoadPointsFromFile',  # 加载点云
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='PointToMultiViewDepth',  # 生成GT深度用于监督
        downsample=1,
        grid_config=_base_.grid_config),
    dict(
        type='BEVAug',
        bda_aug_conf=_base_.bda_aug_conf,
        classes=_base_.class_names),
    dict(type='ObjectRangeFilter', point_cloud_range=_base_.point_cloud_range),
    dict(type='ObjectNameFilter', classes=_base_.class_names),
    dict(type='DefaultFormatBundle3D', class_names=_base_.class_names),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'pretrained_depths', 'gt_depth',
              'gt_bboxes_3d', 'gt_labels_3d'])
]

data = dict(
    train=dict(pipeline=train_pipeline)
)
