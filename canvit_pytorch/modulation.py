"""Per-token adaLN-style modulation generator for the CanViT trunk / cross-attn.

DiT-style factorization, adapted to *per-patch* (spatially-varying) conditioning:

  1. Fixed encoding of each patch's fovea-centric ``(x, y, r)`` (Fourier or
     sinusoidal; from :mod:`canvit_pytorch.encoding`).
  2. A shared base MLP -> per-patch base vector ``c``. Non-image tokens
     (cls / registers / vpe) get their own *learnable* base vectors instead
     (they have no position).
  3. Per-block (and optionally per cross-attn op) ``SiLU -> Linear`` heads map
     ``c`` to raw modulation values.

The heads' output Linears are **zero-initialized**, so every raw value is 0 at
init. The *appliers* (the adaLN block, the read/write cross-attn) turn raw
values into ``gamma = 1 + raw``, ``beta = raw``, ``alpha = alpha0 + raw`` using
their own identity offset ``alpha0`` (``layerscale_init`` in the trunk, ``1`` in
read cross-attn) — so at init the modulated model is bit-identical to the
unmodulated baseline. This module stays a dumb "raw producer".

The conditioning signal is constant across batch and glimpses, so callers
compute the bundle once per step (hoisted out of the viewpoint loop) and reuse
it for every glimpse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn

from canvit_pytorch.encoding import (
    FourierConfig,
    FourierEncoder,
    SinusoidalConfig,
    SinusoidalEncoder,
    build_coord_features,
)


@dataclass
class ViTModulationConfig:
    """Config for per-token trunk (and optional cross-attn) modulation."""

    enabled: bool = False
    """Master switch. False (default) -> no modulation built; the model is
    bit-identical to the unmodulated baseline. True -> requires a ``*_modulate``
    backbone (enforced at construction)."""
    encoding: Literal["fourier", "sinusoidal"] = "fourier"
    """Position encoder: random Gaussian Fourier (mixes axes) or deterministic
    axis-wise NeRF sinusoidal."""
    fourier: FourierConfig = field(default_factory=FourierConfig)
    """Encoder settings used when ``encoding='fourier'``."""
    sinusoidal: SinusoidalConfig = field(default_factory=SinusoidalConfig)
    """Encoder settings used when ``encoding='sinusoidal'``."""
    base_dim: int | None = None
    """Width of the shared per-token base vector ``c``. ``None`` -> ``embed_dim``
    (DiT-like). Smaller values make the per-block heads cheaper."""
    modulate_cross_attn: bool = False
    """Also modulate the local (glimpse) side of the read/write cross-attn."""


@dataclass
class Modulation:
    """Raw (zero-at-init) per-token modulation, one entry per consuming site.

    Each tensor is full-length over the local token stream ``[n_tokens, k*D]``
    (``n_tokens = n_prefix + n_patches``, pack order ``[vpe?, cls, registers,
    patches]``); the applier chunks it into ``k`` per-channel vectors.
    """

    block: list[Tensor]   # n_blocks x [n_tokens, 6*D]  (g1,b1,a1, g2,b2,a2)
    read: list[Tensor]    # n_read   x [n_tokens, 3*D]  (g,b for q ; a for output)
    write: list[Tensor]   # n_write  x [n_tokens, 2*D]  (g,b for kv)


class TokenModulation(nn.Module):
    """Generates the :class:`Modulation` bundle from per-patch positions."""

    def __init__(
        self,
        cfg: ViTModulationConfig,
        *,
        embed_dim: int,
        n_blocks: int,
        n_prefix: int,
        n_read: int,
        n_write: int,
    ) -> None:
        super().__init__()
        self.modulate_cross_attn = cfg.modulate_cross_attn
        base_dim = cfg.base_dim if cfg.base_dim is not None else embed_dim
        in_dim = 3  # (x, y, r)

        if cfg.encoding == "sinusoidal":
            self.encoder: nn.Module = SinusoidalEncoder(in_dim, cfg.sinusoidal.num_freqs)
        else:
            self.encoder = FourierEncoder(
                in_dim, cfg.fourier.num_features, cfg.fourier.sigma, cfg.fourier.seed
            )
        enc_dim = self.encoder.out_dim

        # Shared base MLP (one hidden SiLU layer) -> per-patch base vector c.
        self.base_mlp = nn.Sequential(
            nn.Linear(enc_dim, base_dim), nn.SiLU(), nn.Linear(base_dim, base_dim)
        )
        # Learnable base vectors for the (positionless) prefix tokens, in pack
        # order [vpe?, cls, registers]. Small random init so they can differ.
        self.prefix_codes = nn.Parameter(torch.randn(n_prefix, base_dim) * 0.02)

        # Per-block heads -> 6*D raw (gamma/beta/alpha for attn and mlp branches).
        self.block_heads = nn.ModuleList(
            self._head(base_dim, 6 * embed_dim) for _ in range(n_blocks)
        )
        # Cross-attn heads: read -> 3*D (q gamma,beta + output alpha); write ->
        # 2*D (kv gamma,beta). Empty unless cross-attn modulation is enabled.
        if cfg.modulate_cross_attn:
            self.read_heads = nn.ModuleList(self._head(base_dim, 3 * embed_dim) for _ in range(n_read))
            self.write_heads = nn.ModuleList(self._head(base_dim, 2 * embed_dim) for _ in range(n_write))
        else:
            self.read_heads = nn.ModuleList()
            self.write_heads = nn.ModuleList()

    @staticmethod
    def _head(base_dim: int, out_dim: int) -> nn.Module:
        """``SiLU -> Linear`` with the Linear zero-initialized (raw = 0 at init)."""
        lin = nn.Linear(base_dim, out_dim)
        nn.init.zeros_(lin.weight)
        nn.init.zeros_(lin.bias)
        return nn.Sequential(nn.SiLU(), lin)

    def forward(self, positions: Tensor) -> Modulation:
        """``positions``: ``[n_patches, 2]`` fovea-centric ``(x, y)`` (constant)."""
        ref = self.base_mlp[0].weight
        pos = positions.to(device=ref.device, dtype=ref.dtype)
        xyr = build_coord_features(pos, ["x", "y", "r"])      # [n_patches, 3]
        c_patch = self.base_mlp(self.encoder(xyr))            # [n_patches, base_dim]
        c = torch.cat([self.prefix_codes.to(c_patch.dtype), c_patch], dim=0)  # [n_tokens, base_dim]
        return Modulation(
            block=[h(c) for h in self.block_heads],
            read=[h(c) for h in self.read_heads],
            write=[h(c) for h in self.write_heads],
        )
