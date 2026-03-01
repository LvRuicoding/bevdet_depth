import sys
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
from must3r.model.blocks.attention import toggle_memory_efficient_attention

# Enable xformers memory-efficient attention if available
toggle_memory_efficient_attention(True)


@NECKS.register_module()
class Must3rDecoderNeckTemporalCrossView(BaseModule):
    """MUSt3R-faithful decoder neck with temporal + spatial fusion.

    Follows MUSt3R's decoder design:
      - All frames (current + historical) are processed together.
      - SA (per-frame): each frame's 6 views attend to each other.
      - CA (cross-frame): each frame attends to OTHER frames' tokens
        (own tokens masked out, like MUSt3R's mem_mask).
      - All tokens update at every layer (not just current frame).
      - Only current-frame tokens are taken as output.

    Forward signature:
        forward(curr_feats, hist_feats)
    Returns:
        (B*N, out_channels, h, w) enhanced current-frame features.
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
                 num_views=6,
                 num_hist_frames=1,
                 pretrained=None):
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
        self.num_hist_frames = num_hist_frames
        self.num_frames = 1 + num_hist_frames  # current + historical
        self.pretrained = pretrained

        # --- projection: backbone channels -> enc_embed_dim ---
        if in_channels != enc_embed_dim:
            self.input_proj = nn.Linear(in_channels, enc_embed_dim)
        else:
            self.input_proj = None

        # --- decoder components ---
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.rope = get_pos_embed(pos_embed)

        # Single projection for all tokens (like MUSt3R)
        self.feat_embed_enc_to_dec = nn.Linear(enc_embed_dim, embed_dim, bias=True)

        # Learnable embedding added to historical-frame tokens
        # (analogous to MUSt3R's image2_embed)
        self.hist_frame_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.hist_frame_embed, std=0.02)

        # Per-view position embedding (NEW)
        # Each of the 6 views gets a learnable embedding
        self.view_embed = nn.Parameter(torch.zeros(num_views, 1, embed_dim))
        nn.init.normal_(self.view_embed, std=0.02)

        self.blocks_dec = nn.ModuleList([
            CachedDecoderBlock(
                embed_dim, num_heads, pos_embed=self.rope,
                mlp_ratio=mlp_ratio, qkv_bias=True,
                norm_layer=norm_layer, act_layer=nn.GELU,
                memory_mode=memory_mode)
            for _ in range(depth)])

        self.norm_dec = norm_layer(embed_dim)

        # --- output projection ---
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()
        if pretrained is not None:
            self._load_pretrained(pretrained)
        self._freeze()

    # ------------------------------------------------------------------
    # Init / freeze / train
    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _freeze(self):
        if self.frozen:
            for name, param in self.named_parameters():
                if not name.startswith('output_proj'):
                    param.requires_grad = False
            return
        if self.frozen_blocks < 0:
            return
        for param in self.feat_embed_enc_to_dec.parameters():
            param.requires_grad = False
        self.hist_frame_embed.requires_grad = False
        for i in range(self.frozen_blocks + 1):
            for param in self.blocks_dec[i].parameters():
                param.requires_grad = False

    def _load_pretrained(self, pretrained_path):
        """Load MUSt3R pretrained decoder weights.

        Loads feat_embed_enc_to_dec, blocks_dec (first depth layers),
        norm_dec, and hist_frame_embed (from image2_embed).
        """
        import os
        if not os.path.exists(pretrained_path):
            print(f"Warning: Pretrained checkpoint not found at {pretrained_path}")
            return

        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'decoder' not in checkpoint:
            print(f"Warning: No 'decoder' key in checkpoint {pretrained_path}")
            return

        pretrained_dict = checkpoint['decoder']
        model_dict = self.state_dict()

        # Map pretrained keys to model keys
        matched_keys = []

        # 1. feat_embed_enc_to_dec (if enc_embed_dim matches)
        if self.enc_embed_dim == 1024:  # MUSt3R's encoder output dim
            for key in ['feat_embed_enc_to_dec.weight', 'feat_embed_enc_to_dec.bias']:
                if key in pretrained_dict and key in model_dict:
                    model_dict[key] = pretrained_dict[key]
                    matched_keys.append(key)

        # 2. hist_frame_embed from image2_embed
        if 'image2_embed' in pretrained_dict:
            model_dict['hist_frame_embed'] = pretrained_dict['image2_embed']
            matched_keys.append('hist_frame_embed <- image2_embed')

        # 3. blocks_dec (first depth layers)
        for i in range(len(self.blocks_dec)):
            prefix = f'blocks_dec.{i}.'
            for key in pretrained_dict.keys():
                if key.startswith(prefix) and key in model_dict:
                    model_dict[key] = pretrained_dict[key]
                    matched_keys.append(key)

        # 4. norm_dec
        for key in ['norm_dec.weight', 'norm_dec.bias']:
            if key in pretrained_dict and key in model_dict:
                model_dict[key] = pretrained_dict[key]
                matched_keys.append(key)

        # Load the matched weights
        self.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(matched_keys)} pretrained parameters from {pretrained_path}")
        print(f"  - feat_embed_enc_to_dec: {'✓' if any('feat_embed' in k for k in matched_keys) else '✗'}")
        print(f"  - hist_frame_embed: {'✓' if any('hist_frame' in k for k in matched_keys) else '✗'}")
        print(f"  - blocks_dec: {sum(1 for k in matched_keys if 'blocks_dec' in k)} params")
        print(f"  - norm_dec: {'✓' if any('norm_dec' in k for k in matched_keys) else '✗'}")

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_pos(self, h, w, device):
        """(1, h*w, 2) integer grid for RoPE."""
        y = torch.arange(h, device=device)
        x = torch.arange(w, device=device)
        return torch.cartesian_prod(y, x).unsqueeze(0)

    def _build_cross_frame_memory(self, blk, x, B, nF, Npf):
        """Build per-frame CA memory that excludes each frame's own tokens.

        Args:
            blk: current CachedDecoderBlock
            x: (B*nF, Npf, D) all tokens, grouped per frame
            B: batch size
            nF: number of frames
            Npf: tokens per frame (num_views * h * w)
        Returns:
            mem: (B*nF, (nF-1)*Npf, D_mem) memory for cross-attention
        """
        D = x.shape[-1]
        # Gather all tokens and prepare memory
        all_tokens = x.view(B, nF * Npf, D)
        mem = blk.prepare_y(all_tokens)          # (B, nF*Npf, D_mem)
        mem = mem.view(B, nF, Npf, mem.shape[-1])  # (B, nF, Npf, D_mem)

        # For each frame i, select tokens from all other frames
        # Pre-compute index lists: [[1,2], [0,2], [0,1]] for nF=3
        mem_per_frame = []
        for i in range(nF):
            idx = [j for j in range(nF) if j != i]
            # (B, (nF-1), Npf, D_mem) -> (B, (nF-1)*Npf, D_mem)
            mem_per_frame.append(mem[:, idx].reshape(B, -1, mem.shape[-1]))

        # (B, nF, (nF-1)*Npf, D_mem) -> (B*nF, (nF-1)*Npf, D_mem)
        return torch.stack(mem_per_frame, dim=1).reshape(B * nF, -1, mem.shape[-1])

    @staticmethod
    def _block_forward(blk, x, mem, pos):
        """Checkpoint-friendly wrapper for one decoder block."""
        with torch.cuda.amp.autocast(enabled=True):
            return blk(x, mem, pos, None)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, curr_feats, hist_feats):
        """
        Args:
            curr_feats: FPN output, can be tensor or tuple
                        - tensor: (B*N, C_in, h, w)
                        - tuple: [0] -> (B*N, C_in, h, w)
            hist_feats: FPN output, same format as curr_feats
        Returns:
            (B*N, out_channels, h, w)
        """
        # Handle both tensor and tuple inputs
        if isinstance(curr_feats, (list, tuple)):
            curr_x = curr_feats[0]
        else:
            curr_x = curr_feats

        if isinstance(hist_feats, (list, tuple)):
            hist_x = hist_feats[0]
        else:
            hist_x = hist_feats

        BN, C_in, h, w = curr_x.shape
        B = BN // self.num_views
        N = self.num_views
        nF = self.num_frames                       # 1 + num_hist_frames
        Npf = N * h * w                            # tokens per frame

        # ---- tokenize all views (reshape per-batch first to avoid mixing) ----
        curr_tokens = curr_x.flatten(2).transpose(1, 2).contiguous()  # (B*N, hw, C_in)
        curr_tokens = curr_tokens.view(B, Npf, C_in)                  # (B, Npf, C_in)

        hist_tokens = hist_x.flatten(2).transpose(1, 2).contiguous()  # (B*N*num_hist, hw, C_in)
        hist_tokens = hist_tokens.view(B, self.num_hist_frames * Npf, C_in)

        tokens = torch.cat([curr_tokens, hist_tokens], dim=1)         # (B, nF*Npf, C_in)

        if self.input_proj is not None:
            tokens = self.input_proj(tokens)             # -> enc_embed_dim

        tokens = self.feat_embed_enc_to_dec(tokens)      # -> embed_dim
        D = self.embed_dim

        # ---- add hist_frame_embed to historical tokens ----
        tokens = tokens.view(B, nF, Npf, D)
        tokens[:, 1:] = tokens[:, 1:] + self.hist_frame_embed

        # ---- add per-view embedding ----
        # tokens: (B, nF, Npf, D) where Npf = N * h * w
        # Reshape to (B, nF, N, hw, D) to add view_embed per view
        tokens = tokens.view(B, nF, N, h * w, D)
        # view_embed: (N, 1, D) -> (1, 1, N, 1, D) for broadcasting
        view_embed_broadcast = self.view_embed.view(1, 1, N, 1, D)
        tokens = tokens + view_embed_broadcast
        # Reshape back to (B, nF, Npf, D)
        tokens = tokens.view(B, nF, Npf, D)

        # Reshape to per-frame batching: (B*nF, Npf, D)
        x = tokens.reshape(B * nF, Npf, D)

        # ---- position embeddings (same grid per frame, repeated for 6 views) ----
        single_pos = self._make_pos(h, w, x.device)     # (1, hw, 2)
        pos = single_pos.expand(B * nF, -1, -1)         # (B*nF, hw, 2)
        pos = pos.repeat(1, N, 1)                        # (B*nF, Npf, 2)

        # ---- decoder blocks ----
        for blk in self.blocks_dec:
            mem = self._build_cross_frame_memory(blk, x, B, nF, Npf)
            if self.with_cp and self.training:
                x = cp(self._block_forward, blk, x, mem, pos,
                       use_reentrant=False)
            else:
                with torch.cuda.amp.autocast(enabled=True):
                    x = blk(x, mem, pos, None)
            x = x.float()

        # ---- extract current-frame tokens (frame 0) ----
        x = x.view(B, nF, Npf, D)
        curr_tokens = x[:, 0]                            # (B, Npf, D)
        curr_tokens = self.norm_dec(curr_tokens)

        # ---- reshape to feature map ----
        curr_tokens = curr_tokens.view(B, N, h * w, D)
        feat = curr_tokens.transpose(2, 3).reshape(B * N, D, h, w).contiguous()
        feat = self.output_proj(feat)                    # (B*N, out_channels, h, w)
        return feat