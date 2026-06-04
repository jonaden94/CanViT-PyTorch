"""Patcher factory: create patchers by name."""

import logging
from typing import Literal

import torch

from canvit_pytorch.backbone.vit import ViTBackbone
from canvit_pytorch.patcher.base import Patcher
from canvit_pytorch.patcher.foveated import FoveatedPatcherConfig
from canvit_pytorch.patcher.square import SquarePatcherConfig
from canvit_pytorch.patcher.uniform import UniformPatcher

log = logging.getLogger(__name__)


PatcherName = Literal["uniform", "foveated", "square"]


def create_patcher(
    name: PatcherName,
    *,
    backbone: ViTBackbone,
    glimpse_size_px: int | None = None,
    foveated_config: FoveatedPatcherConfig | None = None,
    square_config: SquarePatcherConfig | None = None,
    device: torch.device | str = "cpu",
) -> Patcher:
    """Create a Patcher by name.

    Args:
        name: ``"uniform"`` for the default uniform-grid patcher (delegates to
            ``backbone.patch_embed``), ``"foveated"`` for the fovi-based
            foveated patcher, or ``"square"`` for the axis-aligned square-patch
            patcher (fovi-derived or standalone strided-square).
        backbone: the constructed ViT backbone. The uniform patcher references
            ``backbone.patch_embed``; the foveated / square patchers only read
            ``backbone.embed_dim``.
        glimpse_size_px: side length (in pixels) of the crop the uniform
            patcher takes from the full image. Ignored by the foveated / square
            patchers (which operate on the full image).
        foveated_config: only consulted for ``name="foveated"``. Defaults to
            ``FoveatedPatcherConfig()`` if ``None``.
        square_config: only consulted for ``name="square"``. Defaults to
            ``SquarePatcherConfig()`` if ``None``.
        device: target device for fovi's sampling state. Only consulted for
            ``name in {"foveated", "square"}``.
    """
    if name == "uniform":
        return UniformPatcher(backbone=backbone, glimpse_size_px=glimpse_size_px)
    if name == "foveated":
        # Lazy import — fovi is an optional dependency.
        from canvit_pytorch.patcher.foveated import FoveatedPatcher

        cfg = foveated_config if foveated_config is not None else FoveatedPatcherConfig()
        patcher = FoveatedPatcher(cfg, embed_dim=backbone.embed_dim, device=device)
        st = patcher.pattern_stats()
        log.info(
            "Created foveated patcher: fov=%.1f cmf_a=%.4f resolution=%d "
            "fixation_size=%d cart_patch_size=%d sampler=%s n_patches=%d "
            "samples_per_patch=%d n_padded=%d unique_pixels=%d",
            cfg.fov, cfg.cmf_a, cfg.resolution, cfg.fixation_size,
            cfg.cart_patch_size, cfg.sampler, st["n_patches"],
            st["samples_per_patch"], st["n_padded"], st["unique_pixels"],
        )
        return patcher
    if name == "square":
        # Lazy import — fovi is an optional dependency.
        from canvit_pytorch.patcher.square import SquarePatcher

        cfg = square_config if square_config is not None else SquarePatcherConfig()
        patcher = SquarePatcher(cfg, embed_dim=backbone.embed_dim, device=device)
        st = patcher.pattern_stats()
        log.info(
            "Created square patcher: method=%s fixation_size=%d side=%d "
            "n_patches=%d padding=%s samples_per_patch=%d n_padded=%d unique_pixels=%d",
            cfg.method, cfg.fixation_size, patcher._side,
            patcher.n_patches, cfg.padding,
            st["samples_per_patch"], st["n_padded"], st["unique_pixels"],
        )
        return patcher
    raise ValueError(
        f"Unknown patcher: {name!r}. Available: 'uniform', 'foveated', 'square'"
    )
