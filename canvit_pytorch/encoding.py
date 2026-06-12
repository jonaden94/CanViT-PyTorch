"""Shared positional / coordinate encoders.

Patcher-agnostic building blocks used by both the patch-embedding conditioners
(:mod:`canvit_pytorch.patcher.conditioning`) and the transformer-trunk
modulation: random Gaussian Fourier features and deterministic axis-wise
(NeRF) sinusoidal features, plus the fovea-centric ``(x, y) -> (x, y, r)``
coordinate-feature helper. Kept here (a leaf module) so both the patcher and the
model can import them without a model->patcher dependency.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #


@dataclass
class FourierConfig:
    """Random Gaussian Fourier features (Tancik et al. 2020).

    The random projection ``B`` (entries ~ ``N(0, sigma^2)``) is drawn once from a
    dedicated, seeded generator and persisted in the checkpoint, so it is
    identical across resumes (and, given the same ``seed``, across fresh runs).
    """

    num_features: int = 64
    """Number of random frequencies. Encoded output width is ``2 * num_features``."""
    sigma: float = 10.0
    """Frequency bandwidth (std of ``B``). Larger -> higher frequencies."""
    seed: int = 0
    """Seed for the dedicated generator used to draw ``B`` (does not touch global RNG)."""


@dataclass
class SinusoidalConfig:
    """Deterministic axis-wise (NeRF-style) sinusoidal features.

    Each of ``(x, y, r)`` is encoded independently with fixed octave frequencies
    ``2^k * pi`` (``k = 0 .. num_freqs - 1``) -- no randomness, no checkpoint
    state. Encoded output width is ``in_dim * 2 * num_freqs`` (``= 6 * num_freqs``
    for ``(x, y, r)``). The ``pi`` scaling suits fovea-centric coords normalized
    to ~``[-1, 1]`` (the base wave spans the full range). To match a Fourier
    encoder's width, set its ``num_features = 3 * num_freqs``.
    """

    num_freqs: int = 6
    """Number of octave frequencies per input axis."""


# --------------------------------------------------------------------------- #
# Coordinate-feature vocabulary (fovea-centric (x, y) -> selected channels)
# --------------------------------------------------------------------------- #

_COORD_FNS: dict[str, Callable[[Tensor], Tensor]] = {
    "x": lambda xy: xy[:, 0:1],
    "y": lambda xy: xy[:, 1:2],
    "r": lambda xy: torch.linalg.vector_norm(xy, dim=1, keepdim=True),
}


def build_coord_features(cartesian: Tensor, channels: list[str]) -> Tensor:
    """Map fovea-centric ``[N, 2]`` ``(x, y)`` to ``[N, len(channels)]``."""
    missing = [c for c in channels if c not in _COORD_FNS]
    assert not missing, f"Unknown coord channels {missing}; available: {sorted(_COORD_FNS)}"
    return torch.cat([_COORD_FNS[c](cartesian) for c in channels], dim=1)


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #


class FourierEncoder(nn.Module):
    """Random Gaussian Fourier features: ``x -> [sin(2*pi*xB), cos(2*pi*xB)]``."""

    B: Tensor

    def __init__(self, in_dim: int, num_features: int, sigma: float, seed: int) -> None:
        super().__init__()
        # Dedicated generator: reproducible B without perturbing the global RNG.
        gen = torch.Generator().manual_seed(seed)
        b = torch.randn(in_dim, num_features, generator=gen) * sigma
        # Persistent: saved in the checkpoint so resumes use the exact same B.
        self.register_buffer("B", b, persistent=True)
        self.out_dim = 2 * num_features

    def forward(self, x: Tensor) -> Tensor:
        proj = 2.0 * math.pi * (x @ self.B)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class SinusoidalEncoder(nn.Module):
    """Deterministic axis-wise (NeRF) sinusoidal features: each component of
    ``x`` is encoded independently with fixed octave frequencies ``2^k * pi``."""

    freqs: Tensor

    def __init__(self, in_dim: int, num_freqs: int) -> None:
        super().__init__()
        freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32) * math.pi  # [L]
        # Deterministic -> non-persistent (regenerated identically on construction).
        self.register_buffer("freqs", freqs, persistent=False)
        self.out_dim = in_dim * 2 * num_freqs

    def forward(self, x: Tensor) -> Tensor:
        # x: [N, in_dim] -> per-axis sin/cos at each frequency, concatenated.
        proj = (x[..., None] * self.freqs).flatten(-2)  # [N, in_dim * L]
        return torch.cat([proj.sin(), proj.cos()], dim=-1)
