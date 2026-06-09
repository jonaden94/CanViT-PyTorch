"""Position-conditioning for the foveated patch embedding.

Under foveation every patch samples a different region of the retinal manifold
(eccentricity-dependent receptive-field size/shape), yet ``self.kpe`` applies
one shared, position-blind projection to all of them (its ``local_rf`` step
normalizes each neighborhood's coordinates, discarding absolute eccentricity).
This module makes the embedding position-aware via small, swappable, mutually
exclusive conditioners.

All positions are fovea-centric ``cartesian`` ``(x, y)`` in ``[-1, 1]^2`` (the
``(0, 0)`` origin is the fovea), constant across batch/fixation:
    - per-sample positions  -> ``kpe.in_coords.cartesian``  ([N_samples, 2])
    - per-patch  positions  -> ``kpe.out_coords.cartesian`` ([N_patches, 2])

Conditioning hooks (each identity by default; a mode overrides only the ones it
needs), called by ``FoveatedPatcher.forward`` at three points of the pipeline
``sensor -> kpe -> embed_head -> tokens``:
    - ``transform_sensor``     : before ``kpe`` (CoordConv extra channels)
    - ``modulate_kpe_output``  : after ``kpe``, before ``embed_head`` (FiLM)
    - ``add_to_output``        : after ``embed_head`` (learned per-patch bias)
plus ``after_kpe_built`` for one-time weight surgery once ``kpe`` exists.

Every mode is a no-op at initialization, so a freshly-built conditioned model is
bit-identical to the unconditioned one at step 0. ``mode="none"`` (the default)
adds no parameters/buffers, preserving backward compatibility with existing
checkpoints.

Adding a new mode: subclass :class:`PatchConditioner`, override the relevant
hook(s), register it in :func:`create_conditioner` (and, if it widens the sensor,
:func:`conditioner_extra_in_channels`), and add its name to
``PatchConditioningConfig.mode``. ``FoveatedPatcher`` needs no changes.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

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
class FiLMConfig:
    """FiLM modulation of ``self.kpe``'s output, conditioned per patch."""

    input: Literal["position", "learned"] = "position"
    """``position``: Fourier-expanded ``(x, y, r)`` per patch. ``learned``: a
    trainable per-patch code fed to the MLP directly (no Fourier)."""
    learned_dim: int = 8
    """Dimension of the learned per-patch code (your choice). Only used when
    ``input='learned'``."""
    mlp_hidden: list[int] = field(default_factory=lambda: [128])
    """Hidden widths of the MLP mapping the (encoded) input to ``(gamma, beta)``."""
    modulate: Literal["both", "scale", "shift"] = "both"
    """``both`` -> learn ``gamma`` and ``beta``; ``scale`` -> ``gamma`` only;
    ``shift`` -> ``beta`` only."""
    fourier: FourierConfig = field(default_factory=FourierConfig)
    """Fourier settings; only consulted when ``input='position'``."""


@dataclass
class CoordConvConfig:
    """Extra fovea-centric coordinate channels appended to the sensor (CoordConv)."""

    channels: list[str] = field(default_factory=lambda: ["x", "y", "r"])
    """Channel vocabulary appended per retinal sample. Available: ``x``, ``y``,
    ``r`` (= ``sqrt(x^2 + y^2)``)."""


@dataclass
class PatchConditioningConfig:
    """Selector + per-mode settings for patch-embedding conditioning."""

    mode: Literal["none", "bias", "film", "coordconv", "coordconv_film"] = "none"
    """``none`` (default) = unconditioned; ``bias`` = learned per-patch output bias;
    ``film`` = FiLM on ``kpe`` output; ``coordconv`` = extra input channels;
    ``coordconv_film`` = both at once (CoordConv input channels + FiLM output
    modulation -- disjoint hooks, so they compose; uses the ``film`` and
    ``coordconv`` sub-configs below)."""
    film: FiLMConfig = field(default_factory=FiLMConfig)
    coordconv: CoordConvConfig = field(default_factory=CoordConvConfig)


# --------------------------------------------------------------------------- #
# Coordinate-feature vocabulary (shared by FiLM 'position' and CoordConv)
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
# Fourier encoder
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


# --------------------------------------------------------------------------- #
# Conditioners
# --------------------------------------------------------------------------- #


class PatchConditioner(nn.Module):
    """Base conditioner: identity at every hook. Subclasses override what they use."""

    extra_in_channels: int = 0  # channels appended to the sensor before kpe

    def transform_sensor(self, sensor: Tensor) -> Tensor:
        return sensor

    def modulate_kpe_output(self, h: Tensor) -> Tensor:
        return h

    def add_to_output(self, tokens: Tensor) -> Tensor:
        return tokens

    def after_kpe_built(self, kpe: nn.Module) -> None:
        """One-time hook after ``kpe`` is constructed (e.g. zero-init new weights)."""
        return None


class NoConditioning(PatchConditioner):
    """Unconditioned: no parameters, no buffers (backward-compatible state_dict)."""


class BiasConditioner(PatchConditioner):
    """Learned per-patch additive bias on the final embedding output."""

    def __init__(self, *, n_patches: int, embed_dim: int) -> None:
        super().__init__()
        # Zero-init -> no-op at step 0.
        self.bias = nn.Parameter(torch.zeros(n_patches, embed_dim))

    def add_to_output(self, tokens: Tensor) -> Tensor:
        return tokens + self.bias  # [N, D] broadcast over [B, N, D]


class FiLMConditioner(PatchConditioner):
    """Per-patch FiLM (``h <- gamma * h + beta``) on ``kpe``'s output."""

    cond_input: Tensor

    def __init__(self, cfg: FiLMConfig, *, n_patches: int, kpe_out: int, patch_xyr: Tensor) -> None:
        super().__init__()
        self.modulate = cfg.modulate
        self.kpe_out = kpe_out
        self.encoder: FourierEncoder | None
        self.learned: nn.Parameter | None

        if cfg.input == "position":
            # Constant per-patch (x, y, r); not trained, not saved (regenerated).
            self.register_buffer("cond_input", patch_xyr.detach().clone().float(), persistent=False)
            self.encoder = FourierEncoder(
                patch_xyr.shape[1], cfg.fourier.num_features, cfg.fourier.sigma, cfg.fourier.seed
            )
            self.learned = None
            enc_dim = self.encoder.out_dim
        else:  # learned
            self.encoder = None
            self.learned = nn.Parameter(torch.randn(n_patches, cfg.learned_dim) * 0.02)
            enc_dim = cfg.learned_dim

        out_dim = (2 if cfg.modulate == "both" else 1) * kpe_out
        dims = [enc_dim, *cfg.mlp_hidden, out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        # Zero the final layer -> MLP outputs 0 -> gamma=1, beta=0 -> no-op at init.
        out_layer = layers[-1]
        assert isinstance(out_layer, nn.Linear)
        nn.init.zeros_(out_layer.weight)
        nn.init.zeros_(out_layer.bias)

    def _gamma_beta(self) -> tuple[Tensor, Tensor]:
        if self.encoder is not None:
            feat = self.encoder(self.cond_input)
        else:
            assert self.learned is not None
            feat = self.learned
        out = self.mlp(feat)  # [N, out_dim]
        if self.modulate == "both":
            d_gamma, beta = out[:, : self.kpe_out], out[:, self.kpe_out :]
            gamma = 1.0 + d_gamma
        elif self.modulate == "scale":
            gamma = 1.0 + out
            beta = torch.zeros_like(out)
        else:  # shift
            gamma = torch.ones_like(out)
            beta = out
        return gamma, beta

    def modulate_kpe_output(self, h: Tensor) -> Tensor:
        gamma, beta = self._gamma_beta()  # [N, kpe_out]
        return h * gamma + beta  # broadcast over [B, N, kpe_out]


class CoordConvConditioner(PatchConditioner):
    """Append fovea-centric coordinate channels to the sensor before ``kpe``."""

    coords: Tensor

    def __init__(self, cfg: CoordConvConfig, *, sample_xy: Tensor) -> None:
        super().__init__()
        feats = build_coord_features(sample_xy.detach().clone().float(), cfg.channels)  # [N_samples, c]
        self.extra_in_channels = int(feats.shape[1])
        # Constant per-sample coordinate channels; regenerated on construction.
        self.register_buffer("coords", feats.t().contiguous(), persistent=False)  # [c, N_samples]

    def transform_sensor(self, sensor: Tensor) -> Tensor:
        b, _, n = sensor.shape
        assert self.coords.shape[1] == n, (
            f"CoordConv sample count {self.coords.shape[1]} != sensor samples {n}"
        )
        extra = self.coords.unsqueeze(0).expand(b, -1, -1).to(sensor.dtype)  # [B, c, N]
        return torch.cat([sensor, extra], dim=1)  # [B, 3 + c, N]

    def after_kpe_built(self, kpe: nn.Module) -> None:
        # No-op at init: zero the projection weights reading the appended channels.
        # kpe.weight is [out, in_channels * n_ref], channel-major (channel d ->
        # columns [d*n_ref : (d+1)*n_ref]); the original RGB are channels 0..2.
        n_ref = int(kpe.ref_coords.shape[0])  # type: ignore[attr-defined]
        with torch.no_grad():
            kpe.weight[:, 3 * n_ref :].zero_()  # type: ignore[attr-defined,index]


class CompositeConditioner(PatchConditioner):
    """Apply several conditioners in sequence.

    Used to combine conditioning that acts on *disjoint* hooks -- specifically
    CoordConv (``transform_sensor``, input side) + FiLM (``modulate_kpe_output``,
    output side). Each child is identity on the hooks it does not own, so chaining
    never double-applies a hook. The no-op-at-init property holds as long as every
    child is a no-op at init (CoordConv zeros the appended-channel kpe weights;
    FiLM zero-inits gamma/beta), so the composite is itself bit-identical to the
    unconditioned model at step 0.
    """

    def __init__(self, conditioners: list[PatchConditioner]) -> None:
        super().__init__()
        self.conditioners = nn.ModuleList(conditioners)
        self.extra_in_channels = sum(int(c.extra_in_channels) for c in conditioners)

    def transform_sensor(self, sensor: Tensor) -> Tensor:
        for c in self.conditioners:
            sensor = c.transform_sensor(sensor)
        return sensor

    def modulate_kpe_output(self, h: Tensor) -> Tensor:
        for c in self.conditioners:
            h = c.modulate_kpe_output(h)
        return h

    def add_to_output(self, tokens: Tensor) -> Tensor:
        for c in self.conditioners:
            tokens = c.add_to_output(tokens)
        return tokens

    def after_kpe_built(self, kpe: nn.Module) -> None:
        for c in self.conditioners:
            c.after_kpe_built(kpe)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def conditioner_extra_in_channels(cfg: PatchConditioningConfig) -> int:
    """Channels the conditioning appends to the sensor (needed before kpe is built)."""
    if cfg.mode in ("coordconv", "coordconv_film"):
        return len(cfg.coordconv.channels)
    return 0


def create_conditioner(
    cfg: PatchConditioningConfig,
    *,
    n_patches: int,
    kpe_out: int,
    embed_dim: int,
    sample_xy: Tensor,
    patch_xy: Tensor,
) -> PatchConditioner:
    """Build the conditioner for ``cfg.mode``.

    ``sample_xy``/``patch_xy`` are fovea-centric ``(x, y)`` for retinal samples /
    patch centers; ``kpe_out`` is ``self.kpe``'s output width (= ``embed_dim`` or
    the first MLP hidden width).
    """
    if cfg.mode == "none":
        return NoConditioning()
    if cfg.mode == "bias":
        return BiasConditioner(n_patches=n_patches, embed_dim=embed_dim)
    if cfg.mode == "film":
        patch_xyr = build_coord_features(patch_xy, ["x", "y", "r"])
        return FiLMConditioner(cfg.film, n_patches=n_patches, kpe_out=kpe_out, patch_xyr=patch_xyr)
    if cfg.mode == "coordconv":
        return CoordConvConditioner(cfg.coordconv, sample_xy=sample_xy)
    if cfg.mode == "coordconv_film":
        # CoordConv (input channels) + FiLM (output modulation): disjoint hooks.
        # CoordConv must come first so the appended channels are present when kpe
        # runs; both are no-ops at init, keeping the composite step-0 identity.
        patch_xyr = build_coord_features(patch_xy, ["x", "y", "r"])
        return CompositeConditioner([
            CoordConvConditioner(cfg.coordconv, sample_xy=sample_xy),
            FiLMConditioner(cfg.film, n_patches=n_patches, kpe_out=kpe_out, patch_xyr=patch_xyr),
        ])
    raise ValueError(f"Unknown conditioning mode: {cfg.mode!r}")
