"""Patcher: pluggable glimpse-to-patches abstraction for CanViT.

Two implementations:
    - :class:`UniformPatcher` (default): the existing uniform-grid behavior;
      delegates to ``backbone.patch_embed`` and uses ``canvas_coords_for_glimpse``
      for positions. No extra dependencies.
    - :class:`FoveatedPatcher` (opt-in): foveated sampling via fovi's
      ``RetinalTransform`` + ``KNNPartitioningPatchEmbedding``. Requires the
      ``[fovi]`` optional extra.

Selected by ``CanViTConfig.patcher_name``. The foveated patcher is configured
by :class:`FoveatedPatcherConfig`.
"""

from canvit_pytorch.patcher.base import Patcher
from canvit_pytorch.patcher.conditioning import (
    CoordConvConfig,
    FiLMConfig,
    FourierConfig,
    PatchConditioningConfig,
)
from canvit_pytorch.patcher.foveated import FoveatedPatcherConfig
from canvit_pytorch.patcher.registry import PatcherName, create_patcher
from canvit_pytorch.patcher.uniform import UniformPatcher

__all__ = [
    "CoordConvConfig",
    "FiLMConfig",
    "FourierConfig",
    "FoveatedPatcherConfig",
    "PatchConditioningConfig",
    "Patcher",
    "PatcherName",
    "UniformPatcher",
    "create_patcher",
]
