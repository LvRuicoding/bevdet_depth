"""BEVDet variant that consumes pre-extracted multi-view image features."""

from mmdet.models import DETECTORS

from .bevdet import BEVDet


@DETECTORS.register_module()
class BEVDetFeatureCache(BEVDet):
    """Use cached image features as BEVDet image encoder output.

    The input pipeline should provide ``img_inputs[0]`` with shape
    (B, N, C, H, W), where C/H/W match ``img_view_transformer.in_channels``
    and its expected downsampled spatial size.
    """

    def image_encoder(self, img, stereo=False):
        """Return cached features directly.

        Args:
            img (Tensor): Cached multi-view features in shape (B, N, C, H, W).
            stereo (bool): Kept for interface compatibility.

        Returns:
            tuple: (features, None)
        """
        return img, None

    @property
    def with_img_neck(self):
        """No image neck is used in cached-feature mode."""
        return False
