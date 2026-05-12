"""Patcher factory: create patchers by name."""

import logging
from typing import Literal

import torch

from canvit_pytorch.backbone.vit import ViTBackbone
from canvit_pytorch.patcher.base import Patcher
from canvit_pytorch.patcher.foveated import FoveatedPatcherConfig
from canvit_pytorch.patcher.uniform import UniformPatcher

log = logging.getLogger(__name__)


PatcherName = Literal["uniform", "foveated"]


def create_patcher(
    name: PatcherName,
    *,
    backbone: ViTBackbone,
    foveated_config: FoveatedPatcherConfig | None = None,
    device: torch.device | str = "cpu",
) -> Patcher:
    """Create a Patcher by name.

    Args:
        name: ``"uniform"`` for the default uniform-grid patcher (delegates to
            ``backbone.patch_embed``), or ``"foveated"`` for the fovi-based
            foveated patcher.
        backbone: the constructed ViT backbone. The uniform patcher references
            ``backbone.patch_embed``; the foveated patcher only reads
            ``backbone.embed_dim``.
        foveated_config: only consulted for ``name="foveated"``. Defaults to
            ``FoveatedPatcherConfig()`` if ``None``.
        device: target device for fovi's sampling state. Only consulted for
            ``name="foveated"``.
    """
    if name == "uniform":
        return UniformPatcher(backbone=backbone)
    if name == "foveated":
        # Lazy import — fovi is an optional dependency.
        from canvit_pytorch.patcher.foveated import FoveatedPatcher

        cfg = foveated_config if foveated_config is not None else FoveatedPatcherConfig()
        patcher = FoveatedPatcher(cfg, embed_dim=backbone.embed_dim, device=device)
        log.info(
            "Created foveated patcher: fov=%.1f cmf_a=%.4f resolution=%d "
            "fixation_size=%d cart_patch_size=%d n_patches=%d",
            cfg.fov, cfg.cmf_a, cfg.resolution, cfg.fixation_size,
            cfg.cart_patch_size, patcher.n_patches,
        )
        return patcher
    raise ValueError(f"Unknown patcher: {name!r}. Available: 'uniform', 'foveated'")
