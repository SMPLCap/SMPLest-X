

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFrameWeighting(nn.Module):

    def __init__(self, embed_dim=1280, num_frames=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.frame_quality_estimator = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x, temporal_valid):
        # x: [B, T, C, H, W];  temporal_valid: [B, T]
        B, T, C, H, W = x.shape
        scores = self.frame_quality_estimator(x.reshape(B * T, C, H, W)).view(B, T)
        scores = scores.masked_fill(temporal_valid == 0, -1e9)
        weights = F.softmax(scores, dim=1)              # [B, T]
        weighted_x = x * weights.view(B, T, 1, 1, 1)
        return weighted_x, weights


class BottleneckCrossAttention(nn.Module):

    def __init__(self, in_dim=1280, bottleneck_dim=256, num_heads=4):
        super().__init__()
        self.down = nn.Linear(in_dim, bottleneck_dim)
        self.attn = nn.MultiheadAttention(bottleneck_dim, num_heads, batch_first=True)
        self.up = nn.Linear(bottleneck_dim, in_dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, center, context, key_padding_mask=None, delta_valid=None):
        q = self.down(center)                              # [B, S, d]
        kv = self.down(context)                            # [B, S*(T-1), d]
        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        delta = self.up(attn_out)                          # [B, S, in_dim]
        if delta_valid is not None:
            delta = delta * delta_valid.view(-1, 1, 1).to(delta.dtype)
        return center + self.gate * delta


class TemporalAdapter(nn.Module):
    def __init__(self, embed_dim=1280, num_patches=192, num_tokens=80,
                 bottleneck_dim=256, num_heads=4, num_frames=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.num_tokens = num_tokens
        self.num_frames = num_frames
        self.center_idx = num_frames // 2

        self.afw = AdaptiveFrameWeighting(embed_dim, num_frames)
        self.spatial_xattn = BottleneckCrossAttention(embed_dim, bottleneck_dim, num_heads)
        self.token_xattn = BottleneckCrossAttention(embed_dim, bottleneck_dim, num_heads)

    @staticmethod
    def _context_mask(temporal_valid, center_idx, seq_len):
        B, T = temporal_valid.shape
        ctx_idx = [t for t in range(T) if t != center_idx]
        ctx_valid = temporal_valid[:, ctx_idx]                       # [B, T-1]
        delta_valid = ctx_valid.sum(dim=1) > 0                       # [B]
        # True where a key should be ignored (frame invalid)
        mask = (ctx_valid == 0)                                      # [B, T-1]
        mask[~delta_valid] = False
        key_padding_mask = mask.unsqueeze(-1).expand(B, len(ctx_idx), seq_len).reshape(B, -1)
        return key_padding_mask, delta_valid

    def forward(self, img_feat, task_tokens, temporal_valid):
        B, T, C, Hp, Wp = img_feat.shape
        N = task_tokens.shape[2]
        ci = self.center_idx
        ctx_idx = [t for t in range(T) if t != ci]

        weighted_img, weights = self.afw(img_feat, temporal_valid)

        # ---- Spatial path ----
        center_patch = img_feat[:, ci].flatten(2).transpose(1, 2)            # [B, Hp*Wp, C]
        ctx_patch = weighted_img[:, ctx_idx]                                 # [B, T-1, C, Hp, Wp]
        ctx_patch = ctx_patch.flatten(3).permute(0, 1, 3, 2).reshape(B, -1, C)  # [B, (T-1)*Hp*Wp, C]
        spk_mask, delta_valid = self._context_mask(temporal_valid, ci, Hp * Wp)
        fused_patch = self.spatial_xattn(center_patch, ctx_patch, spk_mask, delta_valid)
        img_feat_center = fused_patch.transpose(1, 2).reshape(B, C, Hp, Wp).contiguous()

        # ---- Token path ----
        center_tok = task_tokens[:, ci]                                      # [B, N, C]
        weighted_tok = task_tokens * weights.view(B, T, 1, 1)
        ctx_tok = weighted_tok[:, ctx_idx].reshape(B, -1, C)                 # [B, (T-1)*N, C]
        tok_mask, _ = self._context_mask(temporal_valid, ci, N)
        task_tokens_center = self.token_xattn(center_tok, ctx_tok, tok_mask, delta_valid)

        return img_feat_center, task_tokens_center
