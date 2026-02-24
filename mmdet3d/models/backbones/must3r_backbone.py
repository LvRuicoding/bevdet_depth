import sys
import os.path as osp
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as cp
from mmcv.runner import BaseModule

from ..builder import BACKBONES

# Add must3r and its dependencies to sys.path
MUST3R_ROOT = osp.normpath(osp.join(osp.dirname(__file__), '../../../../must3r'))
DUST3R_ROOT = osp.join(MUST3R_ROOT, 'dust3r')
CROCO_ROOT = osp.join(DUST3R_ROOT, 'croco')
for p in [MUST3R_ROOT, DUST3R_ROOT, CROCO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from must3r.model.encoder import Dust3rEncoder


@BACKBONES.register_module()
class Must3rBackbone(BaseModule):
    """Wrapper around MUSt3R's Dust3rEncoder for use as BEVDet image backbone.

    Converts the ViT token sequence output (B, N_patches, embed_dim) into
    spatial feature maps (B, embed_dim, H/patch_size, W/patch_size) that
    are compatible with BEVDet's FPN neck.
    """

    def __init__(self,
                 img_size=(256, 704),
                 patch_size=16,
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 mlp_ratio=4,
                 pretrained=None,
                 with_cp=False,
                 frozen=True):
        super().__init__()
        self.encoder = Dust3rEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self._img_size = img_size
        self.with_cp = with_cp

        if pretrained is not None:
            self._load_pretrained(pretrained)
        if frozen:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # Keep encoder in eval mode when frozen
        if not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()
        return self

    def _load_pretrained(self, checkpoint_path):
        """Load pretrained weights from a MUSt3R / DUSt3R / CroCo checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        # Handle different checkpoint formats
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        # Filter encoder-only keys and strip common prefixes
        enc_state = {}
        for k, v in state_dict.items():
            for prefix in ['module.', 'encoder.', 'backbone.']:
                if k.startswith(prefix):
                    k = k[len(prefix):]
            # Keep only keys that belong to the encoder
            if k.startswith('patch_embed') or k.startswith('blocks_enc') \
                    or k.startswith('norm_enc') or k.startswith('rope'):
                enc_state[k] = v
            # DUSt3R format uses enc_blocks / enc_norm
            elif k.startswith('enc_blocks') or k.startswith('enc_norm'):
                k = k.replace('enc_blocks', 'blocks_enc').replace(
                    'enc_norm', 'norm_enc')
                enc_state[k] = v

        if enc_state:
            info = self.encoder.load_state_dict(enc_state, strict=False)
            print(f'[Must3rBackbone] Loaded pretrained encoder: {info}')
        else:
            print('[Must3rBackbone] Warning: no matching encoder keys found '
                  'in checkpoint, trying to load full state_dict')
            info = self.encoder.load_state_dict(state_dict, strict=False)
            print(f'[Must3rBackbone] load result: {info}')

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) image tensor (B = batch * num_cams in BEVDet)
        Returns:
            tuple of feature maps: ((B, embed_dim, H/ps, W/ps),)
        """
        B, C, H, W = x.shape
        true_shape = torch.tensor(
            [[H, W]], dtype=torch.long, device=x.device).expand(B, -1)

        # Patch embedding (lightweight, no need to checkpoint)
        tokens, pos = self.encoder.patch_embed(x, true_shape=true_shape)

        # Transformer blocks with optional gradient checkpointing
        for blk in self.encoder.blocks_enc:
            if self.with_cp and self.training:
                tokens = cp(blk, tokens, pos, use_reentrant=False)
            else:
                tokens = blk(tokens, pos)
        tokens = self.encoder.norm_enc(tokens)

        h = H // self.patch_size
        w = W // self.patch_size
        feat = tokens.transpose(1, 2).reshape(B, self.embed_dim, h, w)
        feat = feat.contiguous()
        return (feat,)
