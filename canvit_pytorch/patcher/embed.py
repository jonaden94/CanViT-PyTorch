"""Shared patch-embedding building blocks for the foveated / square patchers.

Both :class:`~canvit_pytorch.patcher.foveated.FoveatedPatcher` and
:class:`~canvit_pytorch.patcher.square.SquarePatcher` follow the same
``core-embed -> embed_head -> conditioner`` pipeline (see
``canvit_pytorch/patcher/conditioning.py``). The only difference is the core
embed: fovi's KNN-conv (foveated) vs. a plain linear projection over a regular
square grid (square). This module factors out the parts that are identical so
the two patchers stay aligned.
"""

from __future__ import annotations

import torch
from torch import nn


def build_embed_head(
    hidden_dims: list[int], embed_dim: int, device: torch.device | str
) -> nn.Sequential:
    """Build the MLP head mapping the core embed's output to ``embed_dim``.

    Empty ``hidden_dims`` -> identity ``nn.Sequential`` (the core embed already
    outputs ``embed_dim``). Otherwise the core embed outputs ``hidden_dims[0]``
    and this head maps it to ``embed_dim`` with a ReLU between every pair of
    linear layers and no trailing activation, e.g. ``[1000]`` ->
    ``ReLU, Linear(1000->D)``; ``[1000, 1000]`` -> ``ReLU, Linear(1000->1000),
    ReLU, Linear(1000->D)``.
    """
    head_dims = list(hidden_dims) + [embed_dim]
    layers: list[nn.Module] = []
    for i in range(len(hidden_dims)):
        layers += [nn.ReLU(), nn.Linear(head_dims[i], head_dims[i + 1])]
    return nn.Sequential(*layers).to(device)
