"""Backbone factory: create ViT backbones by name."""

import logging
from dataclasses import dataclass, replace
from typing import Literal

from canvit_pytorch.backbone.vit import ViTBackbone

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackboneConfig:
    embed_dim: int
    num_heads: int
    n_blocks: int
    patch_size: int
    ffn_ratio: float = 4.0
    rope_base: float = 100.0
    layerscale_init: float = 1e-5
    modulated: bool = False
    """If True, build adaLN-style blocks (no LayerNorm affine / LayerScale) that
    accept per-token modulation. Used by the ``*_modulate`` variants; requires a
    modulation generator at the model level to feed them."""


BackboneName = Literal[
    "vits16",
    "vitb16",
    "vitl16",
    "vitb8",
    "vitb7",
    "vitb6",
    "vits16_modulate",
    "vitb16_modulate",
    "vitl16_modulate",
    "vitb8_modulate",
    "vitb7_modulate",
    "vitb6_modulate",
]

_BASE: dict[str, BackboneConfig] = {
    "vits16": BackboneConfig(embed_dim=384, num_heads=6, n_blocks=12, patch_size=16),
    "vitb16": BackboneConfig(embed_dim=768, num_heads=12, n_blocks=12, patch_size=16),
    "vitl16": BackboneConfig(embed_dim=1024, num_heads=16, n_blocks=24, patch_size=16),
    # Sub-16px patch variants of vitb16 (random-init student only): same width/depth,
    # but a smaller patch_embed kernel so a glimpse covers the same crop at a lower
    # per-patch pixel resolution. Each keeps the canonical 8x8=64-token glimpse
    # (glimpse_size_px = 8 x patch_size): 8px->64px, 7px->56px, 6px->48px glimpses.
    "vitb8": BackboneConfig(embed_dim=768, num_heads=12, n_blocks=12, patch_size=8),
    "vitb7": BackboneConfig(embed_dim=768, num_heads=12, n_blocks=12, patch_size=7),
    "vitb6": BackboneConfig(embed_dim=768, num_heads=12, n_blocks=12, patch_size=6),
}

# Each base backbone also gets a ``<name>_modulate`` variant with adaLN blocks.
REGISTRY: dict[str, BackboneConfig] = {
    **_BASE,
    **{f"{name}_modulate": replace(cfg, modulated=True) for name, cfg in _BASE.items()},
}


def create_backbone(name: BackboneName) -> ViTBackbone:
    """Create a ViT backbone by name (random weights)."""
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown backbone: {name!r}. Available: {available}")
    cfg = REGISTRY[name]
    backbone = ViTBackbone(
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        n_blocks=cfg.n_blocks,
        patch_size=cfg.patch_size,
        ffn_ratio=cfg.ffn_ratio,
        rope_base=cfg.rope_base,
        layerscale_init=cfg.layerscale_init,
        modulated=cfg.modulated,
    )
    log.info(
        "Created %s: %d blocks, embed_dim=%d, modulated=%s",
        name, cfg.n_blocks, cfg.embed_dim, cfg.modulated,
    )
    return backbone
