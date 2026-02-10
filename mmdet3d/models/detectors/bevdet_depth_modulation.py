# Copyright (c) Phigent Robotics. All rights reserved.
import torch
from mmdet.models import DETECTORS
from .bevdet import BEVDet


@DETECTORS.register_module()
class BEVDetDepthModulation(BEVDet):
    """BEVDet with Depth Modulation Network.

    This detector extends BEVDet to use pre-extracted depth maps that are refined
    by a learned depth modulation network.

    Args:
        use_depth_loss (bool): Whether to compute depth supervision loss. Default: False.
        **kwargs: Arguments for BEVDet.
    """

    def __init__(self, use_depth_loss=False, **kwargs):
        super(BEVDetDepthModulation, self).__init__(**kwargs)
        self.use_depth_loss = use_depth_loss

    def extract_img_feat(self, img, img_metas, pretrained_depths=None, **kwargs):
        """Extract features of images with depth modulation.

        Args:
            img: Image inputs (7-tuple).
            img_metas: Image meta information.
            pretrained_depths (torch.Tensor, optional): Pre-extracted depth maps.
            **kwargs: Additional arguments.

        Returns:
            tuple: (img_feats, refined_depth)
        """
        img = self.prepare_inputs(img)
        x, _ = self.image_encoder(img[0])

        # Pass pretrained depths to view transformer
        x, refined_depth = self.img_view_transformer(
            [x] + img[1:7], pretrained_depth=pretrained_depths)

        x = self.bev_encoder(x)
        return [x], refined_depth

    def extract_feat(self, points, img, img_metas, pretrained_depths=None, **kwargs):
        """Extract features from images and points.

        Args:
            points: Point cloud data (not used in BEVDet).
            img: Image inputs.
            img_metas: Image meta information.
            pretrained_depths (torch.Tensor, optional): Pre-extracted depth maps.
            **kwargs: Additional arguments.

        Returns:
            tuple: (img_feats, pts_feats, refined_depth)
        """
        img_feats, refined_depth = self.extract_img_feat(
            img, img_metas, pretrained_depths=pretrained_depths, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, refined_depth)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      pretrained_depths=None,
                      gt_depth=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
            img_metas (list[dict], optional): Meta information of each sample.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images.
            img_inputs: Images of each sample.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored.
            pretrained_depths (torch.Tensor, optional): Pre-extracted depth maps.
            gt_depth (torch.Tensor, optional): Ground truth depth from LiDAR.
            **kwargs: Additional arguments.

        Returns:
            dict: Losses of different branches.
        """
        # Extract features with depth modulation
        img_feats, pts_feats, refined_depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas,
            pretrained_depths=pretrained_depths, **kwargs)

        # Compute detection losses
        losses = dict()
        losses_pts = self.forward_pts_train(
            img_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore)
        losses.update(losses_pts)

        # Compute depth supervision loss if enabled
        if self.use_depth_loss and gt_depth is not None:
            loss_depth = self.img_view_transformer.get_depth_loss(
                refined_depth, gt_depth)
            losses['loss_depth'] = loss_depth

        return losses

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    pretrained_depths=None,
                    **kwargs):
        """Test function without augmentation.

        Args:
            points: Point cloud data.
            img_metas: Image meta information.
            img: Image inputs.
            rescale (bool): Whether to rescale results.
            pretrained_depths (torch.Tensor, optional): Pre-extracted depth maps.
            **kwargs: Additional arguments.

        Returns:
            list[dict]: Detection results.
        """
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas,
            pretrained_depths=pretrained_depths, **kwargs)

        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      pretrained_depths=None,
                      **kwargs):
        """Dummy forward for model analysis.

        Args:
            points: Point cloud data.
            img_metas: Image meta information.
            img_inputs: Image inputs.
            pretrained_depths (torch.Tensor, optional): Pre-extracted depth maps.
            **kwargs: Additional arguments.

        Returns:
            Detection head outputs.
        """
        img_feats, _, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas,
            pretrained_depths=pretrained_depths, **kwargs)

        assert self.with_pts_bbox
        outs = self.pts_bbox_head(img_feats)
        return outs
