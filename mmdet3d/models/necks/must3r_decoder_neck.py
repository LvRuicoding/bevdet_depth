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
class Must3rDecoderNeck(BaseModule):
    """MUSt3R decoder as a neck between ResNet backbone and view transformer.

    Each image is processed independently (nimgs=1, no cross-view fusion).
    The decoder blocks provide self-attention + cross-attention-with-self
    feature enhancement using pretrained MUSt3R weights.

    Flow: (B*N, C_in, h, w) -> tokens -> decoder blocks -> (B*N, out_channels, h, w)
    """
    def __init__(self,
                 in_channels=1024,
                 out_channels=256,
                 # decoder architecture (must match pretrained checkpoint)
                 enc_embed_dim=1024,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 memory_mode='norm_y',
                 pos_embed='RoPE100',
                 # loading & freezing
                 pretrained=None,
                 frozen=False,
                 frozen_blocks=-1,
                 with_cp=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.enc_embed_dim = enc_embed_dim
        self.with_cp = with_cp
        self.frozen = frozen
        self.frozen_blocks = frozen_blocks
        self.memory_mode = memory_mode

        # --- projection: backbone channels -> enc_embed_dim (if needed) ---
        if in_channels != enc_embed_dim:
            self.input_proj = nn.Linear(in_channels, enc_embed_dim)
        else:
            self.input_proj = None

        # --- decoder components (matching MUSt3R decoder structure) ---
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

        # --- load pretrained & freeze ---
        if pretrained is not None:
            self._load_pretrained(pretrained)
        else:
            raise ValueError(
                'Must3rDecoderNeck requires a pretrained checkpoint. '
                'Set pretrained=/path/to/MUSt3R_checkpoint.pth')
        self._freeze()

    def _load_pretrained(self, checkpoint_path):
        """Load decoder weights from MUSt3R checkpoint."""
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f'Pretrained checkpoint not found: {checkpoint_path}')

        ckpt = torch.load(checkpoint_path, map_location='cpu')

        if 'decoder' in ckpt and isinstance(ckpt['decoder'], dict):
            state_dict = ckpt['decoder']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            raise KeyError(
                f'Unrecognised checkpoint format. '
                f'Available top-level keys: {list(ckpt.keys())}. '
                f'Expected "decoder", "model", or "state_dict".')

        # Rename DUSt3R/CroCo conventions -> MUSt3R conventions
        renamed = {}
        for k, v in state_dict.items():
            k = k.replace('dec_blocks.', 'blocks_dec.').replace(
                'decoder_embed.', 'feat_embed_enc_to_dec.').replace(
                'dec_norm.', 'norm_dec.')
            renamed[k] = v

        # Only load decoder block weights (skip head_dec, feedback, etc.)
        my_keys = set(self.state_dict().keys())
        load_dict = {}
        for k, v in renamed.items():
            if k in my_keys:
                load_dict[k] = v

        info = self.load_state_dict(load_dict, strict=False)

        # Check: decoder core weights must all be loaded
        missing_decoder_keys = [
            k for k in info.missing_keys
            if k.startswith(('feat_embed_enc_to_dec', 'blocks_dec', 'norm_dec'))]
        if missing_decoder_keys:
            raise RuntimeError(
                f'Missing decoder keys (would be randomly initialised): '
                f'{missing_decoder_keys}')

        # output_proj and input_proj are new layers, expected to be missing
        expected_missing = {k for k in info.missing_keys
                           if k.startswith(('output_proj', 'input_proj'))}
        unexpected_missing = set(info.missing_keys) - expected_missing
        if unexpected_missing:
            raise RuntimeError(
                f'Unexpected missing keys: {unexpected_missing}')

        print(f'[Must3rDecoderNeck] Decoder weights loaded from {checkpoint_path}')
        if info.unexpected_keys:
            print(f'[Must3rDecoderNeck] Ignored keys from ckpt: '
                  f'{len(info.unexpected_keys)} (head_dec, feedback, etc.)')

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
        """Create 2D integer grid positions for RoPE, matching patch_embed format.

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
                    We use inputs[0] with shape (B*N, C_in, h, w).
        Returns:
            (B*N, out_channels, h, w) feature map.
        """
        x = inputs[0]  # (BN, C_in, h, w)
        BN, C, h, w = x.shape

        # --- flatten to tokens ---
        tokens = x.flatten(2).transpose(1, 2)  # (BN, h*w, C_in)

        if self.input_proj is not None:
            tokens = self.input_proj(tokens)  # (BN, h*w, enc_embed_dim)

        # --- project enc -> dec ---
        tokens = self.feat_embed_enc_to_dec(tokens)  # (BN, h*w, embed_dim)
        # nimgs=1 per image, so image2_embed is NOT added (matches decoder init path)

        N_tok = tokens.shape[1]
        D = self.embed_dim
        mem_D = 2 * D if self.memory_mode == 'kv' else D

        pos = self._make_pos(h, w, tokens.device).expand(BN, -1, -1)  # (BN, h*w, 2)

        # --- run decoder blocks (each image independently) ---
        for blk in self.blocks_dec:
            if self.with_cp and self.training:
                tokens = cp(self._block_forward, blk, tokens, pos,
                            use_reentrant=False)
            else:
                mem_i = blk.prepare_y(tokens)
                tokens = blk(tokens, mem_i, pos, None)

        tokens = self.norm_dec(tokens)  # (BN, h*w, embed_dim)

        # --- reshape to spatial + output projection ---
        feat = tokens.transpose(1, 2).reshape(BN, D, h, w).contiguous()
        feat = self.output_proj(feat)  # (BN, out_channels, h, w)
        return feat
