"""Self-contained ViT backbone for CanViT."""

import logging
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from canvit_pytorch.rope import RoPE, rope_apply_with_prefix

log = logging.getLogger(__name__)


class NormFeatures(NamedTuple):
    patches: Tensor  # [B, H*W, D]
    cls: Tensor  # [B, D]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int, stride: int | None = None) -> None:
        super().__init__()
        # stride defaults to patch_size (standard non-overlapping patchify). A
        # smaller stride yields overlapping patches: a grid of g = (in-patch)/stride+1
        # patches per side, each still a patch_size x patch_size conv (weights are
        # kernel-shaped, so they're identical/loadable regardless of stride).
        self.stride = stride if stride is not None else patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=self.stride)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        x = self.proj(x)  # [B, D, H, W]
        H, W = x.shape[2], x.shape[3]
        return x.flatten(2).transpose(1, 2), H, W  # [B, H*W, D]


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_values))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        mask = torch.ones(dim * 3)
        mask[dim : 2 * dim] = 0
        self.register_buffer("_bias_mask", mask, persistent=False)
        self.proj = nn.Linear(dim, dim)

    _bias_mask: Tensor

    def forward(self, x: Tensor, rope: RoPE) -> Tensor:
        B, N, D = x.shape
        assert self.qkv.bias is not None
        qkv = F.linear(x, self.qkv.weight, self.qkv.bias * self._bias_mask).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = [qkv[:, :, i].transpose(1, 2) for i in range(3)]  # [B, H, N, D_h]
        q = rope_apply_with_prefix(x=q, rope=rope)
        k = rope_apply_with_prefix(x=k, rope=rope)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class ViTBlock(nn.Module):
    """Pre-norm transformer block.

    Standard (``modulated=False``): LayerNorm-affine + LayerScale residual --
    CanViT's current block, unchanged.

    Modulated (``modulated=True``): no LayerNorm affine and no LayerScale;
    instead ``forward`` takes raw per-token modulation ``mod`` (``[n_tokens,
    6*D]``) and applies adaLN-zero -- ``gamma = 1 + raw``, ``beta = raw``, and a
    residual gate ``alpha = layerscale_init + raw`` -- to the attn and mlp
    branches. With ``mod=None`` (or all-zero raw) this reproduces the standard
    block's init behavior exactly (``x + layerscale_init * sublayer(LN(x))``).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float,
        layerscale_init: float,
        *,
        modulated: bool = False,
    ) -> None:
        super().__init__()
        self.modulated = modulated
        self._alpha0 = layerscale_init  # residual-gate identity offset
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=not modulated)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=not modulated)
        self.mlp = MLP(dim, int(dim * ffn_ratio))
        if not modulated:
            self.ls1 = LayerScale(dim, layerscale_init)
            self.ls2 = LayerScale(dim, layerscale_init)

    def forward(self, x: Tensor, rope: RoPE, mod: Tensor | None = None) -> Tensor:
        if not self.modulated:
            x = x + self.ls1(self.attn(self.norm1(x), rope))
            return x + self.ls2(self.mlp(self.norm2(x)))
        if mod is None:
            # Identity-at-init regime (raw == 0): gamma=1, beta=0, alpha=alpha0.
            x = x + self._alpha0 * self.attn(self.norm1(x), rope)
            return x + self._alpha0 * self.mlp(self.norm2(x))
        # mod: [n_tokens, 6*D] raw -> per-token (gamma, beta, alpha) per branch.
        g1, b1, a1, g2, b2, a2 = mod.chunk(6, dim=-1)  # each [n_tokens, D]
        h = self.norm1(x)
        x = x + (self._alpha0 + a1) * self.attn((1.0 + g1) * h + b1, rope)
        h = self.norm2(x)
        return x + (self._alpha0 + a2) * self.mlp((1.0 + g2) * h + b2)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class ViTBackbone(nn.Module):
    """ViT backbone: patch embedding + transformer blocks."""

    embed_dim: int
    num_heads: int
    n_blocks: int
    patch_size_px: int
    ffn_ratio: float
    rope_base: float
    layerscale_init: float
    modulated: bool

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        n_blocks: int,
        patch_size: int,
        ffn_ratio: float,
        rope_base: float,
        layerscale_init: float,
        modulated: bool = False,
        patch_stride: int | None = None,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_blocks = n_blocks
        self.patch_size_px = patch_size
        # Patch-embed conv stride; defaults to patch_size (non-overlapping). A
        # smaller value makes the uniform patcher's patches overlap. Only the
        # uniform path uses patch_embed, so this is inert for foveated/square.
        self.patch_stride_px = patch_stride if patch_stride is not None else patch_size
        self.ffn_ratio = ffn_ratio
        self.rope_base = rope_base
        self.layerscale_init = layerscale_init
        self.modulated = modulated

        self.patch_embed = PatchEmbed(patch_size, embed_dim, stride=self.patch_stride_px)
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, ffn_ratio, layerscale_init, modulated=modulated)
            for _ in range(n_blocks)
        ])
        self._init_weights()

    def _init_weights(self) -> None:
        """Match DINOv3 init: trunc_normal_(std=0.02) for Linear weights, zeros for biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, LayerScale):
                nn.init.constant_(module.gamma, self.layerscale_init)
            elif isinstance(module, PatchEmbed):
                proj = module.proj
                k = 1.0 / (proj.in_channels * proj.kernel_size[0] * proj.kernel_size[1])
                nn.init.uniform_(proj.weight, -math.sqrt(k), math.sqrt(k))
                if proj.bias is not None:
                    nn.init.uniform_(proj.bias, -math.sqrt(k), math.sqrt(k))

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads
