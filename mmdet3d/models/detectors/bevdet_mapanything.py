"""BEVDet with MapAnything (DINOv2 + AAT-IFR) as image backbone.

Replaces ResNet50+FPN with MapAnything's DINOv2 encoder + multi-view
alternating attention transformer. DINOv2 encoder is frozen; AAT-IFR
is fine-tunable.
"""
import sys
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DETECTORS
from .. import builder
from .bevdet import BEVDet

# Add mapanything and uniception to sys.path
MAPANYTHING_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '../../../../map-anything'))
if MAPANYTHING_ROOT not in sys.path:
    sys.path.insert(0, MAPANYTHING_ROOT)


@DETECTORS.register_module()
class BEVDetMapAnything(BEVDet):
    """BEVDet variant using MapAnything as image backbone.

    Args:
        mapanything_ckpt (str): Path to MapAnything checkpoint directory
            containing config.json and model.safetensors.
        mapanything_adapter (dict): Config for MapAnythingAdapterNeck.
        freeze_encoder (bool): Freeze DINOv2 encoder. Default True.
        freeze_info_sharing (bool): Freeze AAT-IFR transformer. Default False.
        gradient_checkpointing (bool): Enable gradient checkpointing for
            AAT-IFR to save memory. Default True.
    """
    def __init__(self,
                 mapanything_ckpt,
                 mapanything_adapter,
                 freeze_encoder=True,
                 freeze_info_sharing=False,
                 gradient_checkpointing=True,
                 **kwargs):
        super().__init__(**kwargs)

        # Remove unused ResNet backbone and FPN neck
        if hasattr(self, 'img_backbone'):
            del self.img_backbone
        if hasattr(self, 'img_neck'):
            del self.img_neck

        # Build MapAnything encoder + info_sharing
        self._build_mapanything(mapanything_ckpt, gradient_checkpointing)

        # Build adapter neck (1536 -> 256, spatial resize)
        self.mapanything_adapter = builder.build_neck(mapanything_adapter)

        # Freeze strategy
        self.freeze_encoder = freeze_encoder
        self.freeze_info_sharing = freeze_info_sharing
        if freeze_encoder:
            self._freeze_module(self.ma_encoder)
        if freeze_info_sharing:
            self._freeze_module(self.ma_info_sharing)

    # ------------------------------------------------------------------
    # Build & freeze helpers
    # ------------------------------------------------------------------
    def _build_mapanything(self, ckpt_dir, gradient_checkpointing):
        """Build MapAnything model, load weights, keep only encoder + info_sharing."""
        try:
            from mapanything.models.mapanything.model import MapAnything
        except Exception as exc:
            py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
            raise ImportError(
                "Failed to import MapAnything dependencies in the current "
                f"environment (python {py_ver}). "
                "Do not add site-packages from a different Python version "
                "(e.g. python3.12 into python3.8). "
                "Please install MapAnything/UniCeption dependencies directly "
                "into this environment."
            ) from exc

        config_path = os.path.join(ckpt_dir, 'config.json')
        weights_path = os.path.join(ckpt_dir, 'model.safetensors')

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Build full model (on CPU to save GPU memory during init)
        ma_model = MapAnything(**config)

        # Load pretrained weights
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
        ma_model.load_state_dict(state_dict, strict=False)
        print(f"[BEVDetMapAnything] Loaded MapAnything weights from {weights_path}")

        # Extract only the modules we need
        self.ma_encoder = ma_model.encoder
        self.ma_info_sharing = ma_model.info_sharing
        self.ma_fusion_norm = ma_model.fusion_norm_layer
        self.ma_scale_token = ma_model.scale_token

        # Enable gradient checkpointing on info_sharing if requested
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()

        # Delete the rest to free memory
        del ma_model

    def _freeze_module(self, module):
        """Freeze all parameters in a module."""
        for param in module.parameters():
            param.requires_grad = False

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on AAT-IFR transformer blocks."""
        if hasattr(self.ma_info_sharing, 'blocks'):
            for block in self.ma_info_sharing.blocks:
                self.ma_info_sharing.wrap_module_with_gradient_checkpointing(block)
            print("[BEVDetMapAnything] Enabled gradient checkpointing on info_sharing")

    # ------------------------------------------------------------------
    # Forward methods
    # ------------------------------------------------------------------
    def image_encoder(self, img, stereo=False):
        """Override: use MapAnything instead of ResNet+FPN.

        Args:
            img: (B, N, 3, H, W) normalized images from data pipeline.
        Returns:
            x: (B, N, out_channels, H_feat, W_feat) features for view transformer.
            stereo_feat: None (not supported).
        """
        from uniception.models.encoders.base import ViTEncoderInput
        from uniception.models.info_sharing.base import MultiViewTransformerInput

        B, N, C, imH, imW = img.shape
        patch_size = 14

        # --- Step 1: Pad to multiple of patch_size ---
        pad_h = (patch_size - imH % patch_size) % patch_size
        pad_w = (patch_size - imW % patch_size) % patch_size
        # Reorder to view-major layout before encoder:
        # [v0_b0, v0_b1, ..., v1_b0, v1_b1, ...]
        # so chunk(N) later yields one tensor per camera view.
        imgs = img.permute(1, 0, 2, 3, 4).contiguous().view(N * B, C, imH, imW)
        if pad_h > 0 or pad_w > 0:
            imgs = F.pad(imgs, (0, pad_w, 0, pad_h), mode='reflect')

        # --- Step 2: Run DINOv2 encoder (frozen) ---
        if self.freeze_encoder:
            with torch.no_grad():
                encoder_features, encoder_registers = self._run_encoder(
                    imgs, B, N)
        else:
            encoder_features, encoder_registers = self._run_encoder(
                imgs, B, N)

        # --- Step 3: Run AAT-IFR multi-view transformer ---
        final_features = self._run_info_sharing(
            encoder_features, encoder_registers, B, N)
        # final_features: List[N x (B, 1536, H_p, W_p)]

        # --- Step 4: Restore (B, N, C, H, W), then run adapter ---
        x = torch.stack(final_features, dim=1)  # (B, N, 1536, H_p, W_p)
        x = x.contiguous().view(B * N, x.shape[2], x.shape[3], x.shape[4])
        x = self.mapanything_adapter(x)  # (B*N, out_ch, target_h, target_w)

        _, out_C, out_H, out_W = x.shape
        x = x.view(B, N, out_C, out_H, out_W)
        return x, None

    def _run_encoder(self, imgs, B, N):
        """Run DINOv2 on flattened images.

        Args:
            imgs: (B*N, 3, H_padded, W_padded)
        Returns:
            features: List[N x (B, 1536, H_p, W_p)]
            registers: List[N x (B, 1536, num_reg)] or None
        """
        from uniception.models.encoders.base import ViTEncoderInput

        encoder_input = ViTEncoderInput(
            image=imgs, data_norm_type="dinov2")
        encoder_output = self.ma_encoder(encoder_input)

        # Split (B*N, C, H_p, W_p) into N chunks of (B, C, H_p, W_p)
        features = list(encoder_output.features.chunk(N, dim=0))

        registers = None
        if encoder_output.registers is not None:
            registers = list(encoder_output.registers.chunk(N, dim=0))

        return features, registers

    def _run_info_sharing(self, encoder_features, encoder_registers, B, N):
        """Run AAT-IFR multi-view transformer.

        Args:
            encoder_features: List[N x (B, 1536, H_p, W_p)]
            encoder_registers: List[N x (B, 1536, num_reg)] or None
        Returns:
            List[N x (B, 1536, H_p, W_p)] final features
        """
        from uniception.models.info_sharing.base import MultiViewTransformerInput

        # Prepare scale token: (B, C, 1)
        scale_token = (self.ma_scale_token.unsqueeze(0)
                       .unsqueeze(-1)
                       .repeat(B, 1, 1))

        info_input = MultiViewTransformerInput(
            features=encoder_features,
            additional_input_tokens=scale_token,
            additional_input_tokens_per_view=encoder_registers,
        )

        # The info_sharing returns (final, intermediates) for IFR type
        result = self.ma_info_sharing(info_input)
        if isinstance(result, tuple):
            final_output = result[0]
        else:
            final_output = result

        return final_output.features

    # ------------------------------------------------------------------
    # Overrides for freeze / train behavior
    # ------------------------------------------------------------------
    def train(self, mode=True):
        """Override to keep frozen modules in eval mode."""
        super().train(mode)
        if self.freeze_encoder:
            self.ma_encoder.eval()
        if self.freeze_info_sharing:
            self.ma_info_sharing.eval()
        return self

    @property
    def with_img_neck(self):
        """Override: we don't have img_neck."""
        return False

    def extract_img_feat(self, img, img_metas, **kwargs):
        """Extract features: MapAnything backbone → view transformer → BEV encoder."""
        img = self.prepare_inputs(img)
        x, _ = self.image_encoder(img[0])
        x, depth = self.img_view_transformer([x] + img[1:7])
        x = self.bev_encoder(x)
        return [x], depth
