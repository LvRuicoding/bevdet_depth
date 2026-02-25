import sys
import os
import os.path as osp
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as cp
from functools import partial
from mmcv.runner import BaseModule

from ..builder import NECKS

# Add must3r and its dependencies to sys.path
MUST3R_ROOT = osp.normpath(osp.join(osp.dirname(__file__), '../../../../must3r'))
DUST3R_ROOT = osp.join(MUST3R_ROOT, 'dust3r')
CROCO_ROOT = osp.join(DUST3R_ROOT, 'croco')
for p in [MUST3R_ROOT, DUST3R_ROOT, CROCO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from must3r.model.blocks.layers import CachedDecoderBlock
from must3r.model.blocks.pos_embed import get_pos_embed


@NECKS.register_module()
class Must3rDecoderNeckMultiView(BaseModule):
    """MUSt3R decoder as a neck with cross-view fusion.

    Processes 6 images together with cross-view attention, then outputs
    per-image features for LSS View Transformer.

    Flow: (B*6, C_in, h, w) -> (B, 6, h*w, C) -> decoder blocks (cross-view)
          -> (B, 6, h*w, C) -> (B*6, out_channels, h, w)
    """
    def __init__(self,
                 in_channels=1024,
                 out_channels=256,
                 enc_embed_dim=1024,
                 embed_dim=768,
                 depth=4,
                 num_heads=12,
                 mlp_ratio=4,
                 memory_mode='norm_y',
                 pos_embed='RoPE100',
                 frozen=False,
                 frozen_blocks=-1,
                 with_cp=False,
                 num_views=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.enc_embed_dim = enc_embed_dim
        self.with_cp = with_cp
        self.frozen = frozen
        self.frozen_blocks = frozen_blocks
        self.memory_mode = memory_mode
        self.num_views = num_views

        # --- projection: backbone channels -> enc_embed_dim (if needed) ---
        if in_channels != enc_embed_dim:
            self.input_proj = nn.Linear(in_channels, enc_embed_dim)
        else:
            self.input_proj = None

        # --- decoder components ---
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.rope = get_pos_embed(pos_embed)

        self.feat_embed_enc_to_dec = nn.Linear(enc_embed_dim, embed_dim, bias=True)

        self.blocks_dec = nn.ModuleList([
            CachedDecoderBlock(
                embed_dim, num_heads, pos_embed=self.rope,
                mlp_ratio=mlp_ratio, qkv_bias=True,
                norm_layer=norm_layer, act_layer=nn.GELU,
                memory_mode=memory_mode)
            for _ in range(depth)])

        self.norm_dec = norm_layer(embed_dim)

        # --- output projection: embed_dim -> out_channels ---
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()
        self._freeze()

    def _init_weights(self):
        """Xavier uniform for Linear layers, Kaiming normal for Conv2d."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _freeze(self):
        """Freeze decoder parameters based on settings."""
        if self.frozen:
            for name, param in self.named_parameters():
                if not name.startswith('output_proj'):
                    param.requires_grad = False
            return

        if self.frozen_blocks < 0:
            return

        # Freeze feat_embed_enc_to_dec + blocks[0..frozen_blocks]
        for param in self.feat_embed_enc_to_dec.parameters():
            param.requires_grad = False
        for i in range(self.frozen_blocks + 1):
            for param in self.blocks_dec[i].parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            for name, module in self.named_modules():
                if not name.startswith('output_proj') and name != '':
                    module.eval()
        elif self.frozen_blocks >= 0:
            self.feat_embed_enc_to_dec.eval()
            for i in range(self.frozen_blocks + 1):
                self.blocks_dec[i].eval()
        return self

    @staticmethod
    def _block_forward(blk, tokens, pos):
        """Wrapper that puts prepare_y + blk.forward inside one checkpoint unit."""
        mem_i = blk.prepare_y(tokens)
        return blk(tokens, mem_i, pos, None)

    def _make_pos(self, h, w, device):
        """Create 2D integer grid positions for RoPE.

        Returns: (1, h*w, 2) with (y, x) integer coordinates.
        """
        y = torch.arange(h, device=device)
        x = torch.arange(w, device=device)
        pos = torch.cartesian_prod(y, x)  # (h*w, 2)
        return pos.unsqueeze(0)  # (1, h*w, 2)

    @torch.autocast("cuda", enabled=False)
    def forward(self, inputs):
        """
        Args:
            inputs: tuple of feature maps from backbone.
                    We use inputs[0] with shape (B*N, C_in, h, w) where N=6.
        Returns:
            (B*N, out_channels, h, w) feature map with cross-view fusion.
        """
        x = inputs[0]  # (B*6, C_in, h, w)
        BN, C, h, w = x.shape
        B = BN // self.num_views
        N = self.num_views

        # --- reshape to separate batch and views ---
        x = x.reshape(B, N, C, h, w)  # (B, 6, C_in, h, w)

        # --- flatten to tokens ---
        tokens = x.flatten(3).transpose(2, 3)  # (B, 6, h*w, C_in)
        tokens = tokens.reshape(B, N * h * w, C)  # (B, 6*h*w, C_in)

        if self.input_proj is not None:
            tokens = self.input_proj(tokens)  # (B, 6*h*w, enc_embed_dim)

        # --- project enc -> dec ---
        tokens = self.feat_embed_enc_to_dec(tokens)  # (B, 6*h*w, embed_dim)

        D = self.embed_dim

        # Create position embeddings for all views concatenated
        pos = self._make_pos(h, w, tokens.device)  # (1, h*w, 2)
        pos = pos.expand(B, -1, -1)  # (B, h*w, 2)
        # Repeat for each view
        pos = pos.repeat(1, N, 1)  # (B, 6*h*w, 2)

        # --- run decoder blocks (cross-view fusion) ---
        for blk in self.blocks_dec:
            if self.with_cp and self.training:
                tokens = cp(self._block_forward, blk, tokens, pos,
                            use_reentrant=False)
            else:
                mem_i = blk.prepare_y(tokens)
                tokens = blk(tokens, mem_i, pos, None)

        tokens = self.norm_dec(tokens)  # (B, 6*h*w, embed_dim)

        # --- reshape back to per-view features ---
        tokens = tokens.reshape(B, N, h * w, D)  # (B, 6, h*w, embed_dim)
        feat = tokens.transpose(2, 3).reshape(B * N, D, h, w).contiguous()  # (B*6, embed_dim, h, w)

        feat = self.output_proj(feat)  # (B*6, out_channels, h, w)
        return feat
