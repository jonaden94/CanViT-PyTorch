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


def count_unique_pixels(positions_xy: torch.Tensor, reference_size: int) -> int:
    """Distinct pixel cells the visual-field sample positions map to.

    ``positions_xy`` are fixation-window-normalized ``(x, y)`` in ``[-1, 1]^2``
    (``[-1, 1]`` = the foveation window). Each is floored onto a
    ``reference_size x reference_size`` grid (centered fixation,
    ``image_size == fixation_size == reference_size``) and the number of distinct
    cells is returned. This mirrors the reference notebook's
    ``_vf_unique_pixels`` exactly: **no clipping** and **no out-of-field
    exclusion** — out-of-field samples (the pattern can slightly overshoot the
    window) land outside ``[0, reference_size)`` but are still counted as their
    own cells, i.e. "assuming the image is large enough to cover all samples".
    The caller is expected to drop structurally-padded slots beforehand (the
    square patcher passes ``positions[~pad_mask]``), matching the notebook's
    ``pad_mask=outermost_pad_mask`` argument. A high-level characterization of
    the pattern's effective resolution; the value is fixed by the reference-scale
    convention (it would differ at a smaller/larger fixation window).
    """
    r = int(reference_size)
    px = torch.floor((positions_xy.reshape(-1, 2).to(torch.float64) + 1.0) * 0.5 * r).long()
    return int(torch.unique(px, dim=0).shape[0])
