"""BEVDet with MUSt3R temporal cross-attention decoder.

Loads current + historical frames, runs backbone+FPN on both, then the
Must3rDecoderNeckTemporalCrossView does SA on current views and CA
with historical views.
"""
import torch
from mmdet.models import DETECTORS
from .bevdet import BEVDet
from .. import builder


@DETECTORS.register_module()
class BEVDetMust3rTemporal(BEVDet):
    """BEVDet variant that feeds historical-frame backbone features
    into the MUSt3R decoder as cross-attention key/value.

    Args:
        num_adj (int): Number of adjacent (historical) frames. Default 1.
        with_prev (bool): If False, zero-out historical features during
            the first few iterations when no real history is available.
        img_neck_3d (dict): Config for the 3D temporal fusion neck (MUSt3R decoder).
    """

    def __init__(self, num_adj=1, with_prev=True, img_neck_3d=None, **kwargs):
        super().__init__(**kwargs)
        self.num_adj = num_adj
        self.num_frame = num_adj + 1  # current + adjacent
        self.with_prev = with_prev

        # Build the 3D temporal fusion neck (MUSt3R decoder)
        if img_neck_3d is not None:
            self.img_neck_3d = builder.build_neck(img_neck_3d)
        else:
            self.img_neck_3d = None

    # ------------------------------------------------------------------
    def _backbone_forward(self, img):
        """Run backbone only (no neck). Returns tuple of feature maps."""
        B, N, C, imH, imW = img.shape
        imgs = img.view(B * N, C, imH, imW)
        if self.grid_mask is not None:
            imgs = self.grid_mask(imgs)
        x = self.img_backbone(imgs)
        return x  # tuple, e.g. ((B*N, 1024, h, w),)

    # ------------------------------------------------------------------
    def prepare_inputs(self, inputs):
        """Split stacked frames into current / historical and compute
        sensor2keyego transforms.

        The data pipeline (PrepareImageInputs with sequential=True) packs
        images as (B, N*num_frame, C, H, W) interleaved per camera:
            [cam0_f0, cam0_f1, cam1_f0, cam1_f1, ...]
        """
        assert len(inputs) == 7
        B, N_total, C, H, W = inputs[0].shape
        N = N_total // self.num_frame  # cameras per frame
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs

        # --- split images by frame ---
        # interleaved: (B, N*num_frame, C, H, W) -> (B, N, num_frame, C, H, W)
        imgs = imgs.view(B, N, self.num_frame, C, H, W)
        imgs_curr = imgs[:, :, 0]   # (B, N, C, H, W)
        imgs_hist_list = [imgs[:, :, t] for t in range(1, self.num_frame)]
        # stack all historical frames: (B, N*num_adj, C, H, W)
        imgs_hist = torch.cat(imgs_hist_list, dim=1)

        # --- sensor transforms (grouped by frame) ---
        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)
        intrins = intrins.view(B, self.num_frame, N, 3, 3)
        post_rots = post_rots.view(B, self.num_frame, N, 3, 3)
        post_trans = post_trans.view(B, self.num_frame, N, 3)

        # key ego = current frame, first camera
        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        # current frame transforms only (B, N, ...)
        sensor2keyegos_curr = sensor2keyegos[:, 0]
        ego2globals_curr = ego2globals[:, 0]
        intrins_curr = intrins[:, 0]
        post_rots_curr = post_rots[:, 0]
        post_trans_curr = post_trans[:, 0]

        return dict(
            imgs_curr=imgs_curr,
            imgs_hist=imgs_hist,
            sensor2keyegos=sensor2keyegos_curr,
            ego2globals=ego2globals_curr,
            intrins=intrins_curr,
            post_rots=post_rots_curr,
            post_trans=post_trans_curr,
            bda=bda,
        )

    # ------------------------------------------------------------------
    def extract_img_feat(self, img, img_metas, **kwargs):
        """Extract features with temporal cross-attention."""
        data = self.prepare_inputs(img)

        imgs_curr = data['imgs_curr']   # (B, N, C, H, W)
        imgs_hist = data['imgs_hist']   # (B, N*num_adj, C, H, W)

        # --- backbone for current frame ---
        curr_backbone_feats = self._backbone_forward(imgs_curr)

        # --- backbone for historical frame(s), no grad ---
        with torch.no_grad():
            hist_backbone_feats = self._backbone_forward(imgs_hist)

        # --- FPN neck (if exists) ---
        if self.with_img_neck:
            curr_fpn_feats = self.img_neck(curr_backbone_feats)
            with torch.no_grad():
                hist_fpn_feats = self.img_neck(hist_backbone_feats)
        else:
            curr_fpn_feats = curr_backbone_feats
            hist_fpn_feats = hist_backbone_feats

        # --- temporal decoder neck (MUSt3R) ---
        if self.img_neck_3d is not None:
            x = self.img_neck_3d(curr_fpn_feats, hist_fpn_feats)
        else:
            # Fallback: just use current FPN features
            x = curr_fpn_feats
            if isinstance(x, (list, tuple)):
                x = x[0]

        if isinstance(x, (list, tuple)):
            x = x[0]

        B = imgs_curr.shape[0]
        N = imgs_curr.shape[1]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)

        # --- view transformer (uses current-frame transforms) ---
        vt_inputs = [
            x,
            data['sensor2keyegos'],
            data['ego2globals'],
            data['intrins'],
            data['post_rots'],
            data['post_trans'],
            data['bda'],
        ]
        x, depth = self.img_view_transformer(vt_inputs)
        x = self.bev_encoder(x)
        return [x], depth
