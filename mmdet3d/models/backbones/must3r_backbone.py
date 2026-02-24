import sys
import os
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
                 frozen=True,
                 frozen_stages=-1,
                 adapter_channels=None):
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
        self.frozen = frozen
        self.frozen_stages = frozen_stages

        # Adapter: lightweight conv layers to bridge ViT → CNN feature distribution
        # Always trainable, even when encoder is frozen.
        if adapter_channels is not None:
            self.adapter = nn.Sequential(
                nn.Conv2d(embed_dim, adapter_channels, 1, bias=False),
                nn.BatchNorm2d(adapter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(adapter_channels, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
            )
            self.out_channels = embed_dim
        else:
            self.adapter = None

        if pretrained is not None:
            self._load_pretrained(pretrained)
        else:
            raise ValueError(
                'Must3rBackbone requires a pretrained checkpoint. '
                'Set pretrained=/path/to/MUSt3R_checkpoint.pth')
        self._freeze()

    def _freeze(self):
        """Freeze parameters based on frozen / frozen_stages settings.

        frozen=True:  freeze everything (patch_embed + all blocks + norm).
        frozen=False, frozen_stages=N (0-indexed):
            freeze patch_embed + blocks_enc[0..N].
            N=-1 means freeze nothing.
        """
        if self.frozen:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
            return

        if self.frozen_stages < 0:
            return

        # Always freeze patch_embed when frozen_stages >= 0
        self.encoder.patch_embed.eval()
        for param in self.encoder.patch_embed.parameters():
            param.requires_grad = False

        # Freeze blocks_enc[0..frozen_stages]
        for i in range(self.frozen_stages + 1):
            blk = self.encoder.blocks_enc[i]
            blk.eval()
            for param in blk.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
            return self
        # Keep frozen stages in eval mode
        if self.frozen_stages >= 0:
            self.encoder.patch_embed.eval()
            for i in range(self.frozen_stages + 1):
                self.encoder.blocks_enc[i].eval()
        return self

    def _load_pretrained(self, checkpoint_path):
        """Load pretrained weights from a MUSt3R / DUSt3R / CroCo checkpoint."""
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f'Pretrained checkpoint not found: {checkpoint_path}')

        ckpt = torch.load(checkpoint_path, map_location='cpu')

        # MUSt3R format: encoder/decoder stored as separate keys
        if 'encoder' in ckpt and isinstance(ckpt['encoder'], dict):
            state_dict = ckpt['encoder']
        # DUSt3R format: full model under 'model' key (need to filter)
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            raise KeyError(
                f'Unrecognised checkpoint format. '
                f'Available top-level keys: {list(ckpt.keys())}. '
                f'Expected "encoder", "model", or "state_dict".')

        # Rename DUSt3R/CroCo key conventions → MUSt3R conventions
        enc_state = {}
        for k, v in state_dict.items():
            k = k.replace('enc_blocks', 'blocks_enc').replace(
                'enc_norm', 'norm_enc')
            enc_state[k] = v

        info = self.encoder.load_state_dict(enc_state, strict=False)

        # Fail loudly if any encoder parameter was NOT loaded
        if info.missing_keys:
            raise RuntimeError(
                f'Missing keys when loading pretrained encoder '
                f'(these parameters would be randomly initialised): '
                f'{info.missing_keys}')

        if info.unexpected_keys:
            print(f'[Must3rBackbone] Unexpected keys in checkpoint (ignored): '
                  f'{info.unexpected_keys}')
        print('[Must3rBackbone] All encoder weights loaded successfully.')

    @torch.autocast("cuda", enabled=False)
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

        # Patch embedding
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

        if self.adapter is not None:
            feat = self.adapter(feat) + feat  # residual connection

        return (feat,)
