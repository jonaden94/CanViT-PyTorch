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
(The one exception is FiLM with ``encode_scale=True``, which additionally feeds
the per-forward, per-sample view ``scale`` into its encoder — so that conditioner
alone is batch/glimpse-dependent; the patch geometry above stays constant.)

Conditioning hooks (each identity by default; a mode overrides only the ones it
needs), called by ``FoveatedPatcher.forward`` at two points of the pipeline
``sensor -> kpe -> embed_head -> tokens``:
    - ``transform_sensor``     : before ``kpe`` (CoordConv extra channels)
    - ``modulate_kpe_output``  : after ``kpe``, before ``embed_head`` (FiLM)
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

# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #


@dataclass
class FiLMConfig:
    """FiLM modulation of ``self.kpe``'s output, conditioned per patch on an
    encoding of the fovea-centric ``(x, y, r)``; learns both ``gamma`` and
    ``beta``."""

    encoding: Literal["fourier", "sinusoidal"] = "fourier"
    """``fourier`` = random Gaussian Fourier features (mixes axes; uses the
    ``fourier`` sub-config); ``sinusoidal`` = deterministic axis-wise NeRF
    features (uses the ``sinusoidal`` sub-config)."""
    mlp_hidden: list[int] = field(default_factory=lambda: [128])
    """Hidden widths of the MLP mapping the encoded features to ``(gamma, beta)``."""
    fourier: FourierConfig = field(default_factory=FourierConfig)
    """Encoder settings used when ``encoding='fourier'``."""
    sinusoidal: SinusoidalConfig = field(default_factory=SinusoidalConfig)
    """Encoder settings used when ``encoding='sinusoidal'``."""
    encode_scale: bool = False
    """If True, append the per-forward view ``scale`` (= ``fix_size / H``; ``1`` =
    full image, ``<1`` zooms in) to the encoder input, so the conditioning becomes
    ``(x, y, r, scale)`` instead of ``(x, y, r)``. The patch geometry ``(x, y, r)``
    is constant, but ``scale`` is per-sample (``[B]``) and may vary per glimpse, so
    with this on the FiLM ``(gamma, beta)`` are recomputed per forward and are
    batch-dependent (``[B, N, kpe_out]`` instead of ``[N, kpe_out]``). Lets the
    patch embedding adapt to the current zoom (e.g. for ``per_glimpse`` /
    ``per_rollout`` scale sampling); for a single fixed training scale it carries
    no information. ``False`` (default) is bit-identical to the original
    scale-blind FiLM (encoder input stays 3-dim; ``gamma/beta`` stay ``[N, kpe_out]``)."""


@dataclass
class CoordConvConfig:
    """Extra fovea-centric coordinate channels appended to the sensor (CoordConv)."""

    channels: list[str] = field(default_factory=lambda: ["x", "y", "r"])
    """Channel vocabulary appended per retinal sample. Available: ``x``, ``y``,
    ``r`` (= ``sqrt(x^2 + y^2)``)."""


@dataclass
class PatchConditioningConfig:
    """Selector + per-mode settings for patch-embedding conditioning."""

    mode: Literal["none", "film", "coordconv", "coordconv_film"] = "none"
    """``none`` (default) = unconditioned; ``film`` = FiLM on ``kpe`` output;
    ``coordconv`` = extra input channels; ``coordconv_film`` = both at once
    (CoordConv input channels + FiLM output modulation -- disjoint hooks, so they
    compose; uses the ``film`` and ``coordconv`` sub-configs below)."""
    film: FiLMConfig = field(default_factory=FiLMConfig)
    coordconv: CoordConvConfig = field(default_factory=CoordConvConfig)


# --------------------------------------------------------------------------- #
# Conditioners
# --------------------------------------------------------------------------- #


class PatchConditioner(nn.Module):
    """Base conditioner: identity at every hook. Subclasses override what they use."""

    extra_in_channels: int = 0  # channels appended to the sensor before kpe

    def transform_sensor(self, sensor: Tensor) -> Tensor:
        return sensor

    def modulate_kpe_output(self, h: Tensor, scale: Tensor | None = None) -> Tensor:
        # ``scale`` ([B] per-sample view scale) is consumed only by scale-aware
        # FiLM; every other conditioner ignores it (identity hook here).
        return h

    def after_kpe_built(self, kpe: nn.Module) -> None:
        """One-time hook after ``kpe`` is constructed (e.g. zero-init new weights)."""
        return None


class NoConditioning(PatchConditioner):
    """Unconditioned: no parameters, no buffers (backward-compatible state_dict)."""


class FiLMConditioner(PatchConditioner):
    """Per-patch FiLM (``h <- gamma * h + beta``) on ``kpe``'s output,
    conditioned on an encoding (Fourier or sinusoidal) of the fovea-centric
    ``(x, y, r)``."""

    cond_input: Tensor

    def __init__(self, cfg: FiLMConfig, *, kpe_out: int, patch_xyr: Tensor) -> None:
        super().__init__()
        self.kpe_out = kpe_out
        self.encode_scale = cfg.encode_scale
        # Constant per-patch (x, y, r); not trained, not saved (regenerated).
        self.register_buffer("cond_input", patch_xyr.detach().clone().float(), persistent=False)
        # +1 input dim for the appended per-forward scale when encode_scale is on.
        # NB: for the Fourier encoder the output width is 2*num_features regardless
        # of in_dim (only its fixed projection buffer B gains a row), so the MLP is
        # shape-identical with/without scale; for sinusoidal the encoder output
        # (hence the MLP's first Linear) grows by 2*num_freqs. Either way no FiLM
        # hyperparameter changes between scale-on and scale-off runs.
        in_dim = patch_xyr.shape[1] + (1 if cfg.encode_scale else 0)
        if cfg.encoding == "sinusoidal":
            self.encoder = SinusoidalEncoder(in_dim, cfg.sinusoidal.num_freqs)
        else:
            self.encoder = FourierEncoder(
                in_dim, cfg.fourier.num_features, cfg.fourier.sigma, cfg.fourier.seed
            )

        out_dim = 2 * kpe_out  # gamma and beta
        dims = [self.encoder.out_dim, *cfg.mlp_hidden, out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        # Zero the final layer -> MLP outputs 0 -> gamma=1, beta=0 -> no-op at init
        # (holds with or without scale, so a scale-aware FiLM still starts
        # bit-identical to the unconditioned model).
        out_layer = layers[-1]
        assert isinstance(out_layer, nn.Linear)
        nn.init.zeros_(out_layer.weight)
        nn.init.zeros_(out_layer.bias)

    def _gamma_beta(self, scale: Tensor | None) -> tuple[Tensor, Tensor]:
        if self.encode_scale:
            assert scale is not None, "encode_scale=True FiLM requires a per-sample view scale"
            b = scale.shape[0]
            n = self.cond_input.shape[0]
            xyr = self.cond_input.unsqueeze(0).expand(b, -1, -1)               # [B, N, 3]
            s = scale.to(self.cond_input.dtype).view(b, 1, 1).expand(b, n, 1)  # [B, N, 1]
            feats = torch.cat([xyr, s], dim=-1)                               # [B, N, 4]
        else:
            feats = self.cond_input  # [N, 3]
        out = self.mlp(self.encoder(feats))  # [N, 2*kpe_out] or [B, N, 2*kpe_out]
        d_gamma, beta = out[..., : self.kpe_out], out[..., self.kpe_out :]
        gamma = 1.0 + d_gamma
        return gamma, beta

    def modulate_kpe_output(self, h: Tensor, scale: Tensor | None = None) -> Tensor:
        gamma, beta = self._gamma_beta(scale)  # [N, kpe_out] or [B, N, kpe_out]
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

    def modulate_kpe_output(self, h: Tensor, scale: Tensor | None = None) -> Tensor:
        for c in self.conditioners:
            h = c.modulate_kpe_output(h, scale=scale)
        return h

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
    kpe_out: int,
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
    if cfg.mode == "film":
        patch_xyr = build_coord_features(patch_xy, ["x", "y", "r"])
        return FiLMConditioner(cfg.film, kpe_out=kpe_out, patch_xyr=patch_xyr)
    if cfg.mode == "coordconv":
        return CoordConvConditioner(cfg.coordconv, sample_xy=sample_xy)
    if cfg.mode == "coordconv_film":
        # CoordConv (input channels) + FiLM (output modulation): disjoint hooks.
        # CoordConv must come first so the appended channels are present when kpe
        # runs; both are no-ops at init, keeping the composite step-0 identity.
        patch_xyr = build_coord_features(patch_xy, ["x", "y", "r"])
        return CompositeConditioner([
            CoordConvConditioner(cfg.coordconv, sample_xy=sample_xy),
            FiLMConditioner(cfg.film, kpe_out=kpe_out, patch_xyr=patch_xyr),
        ])
    raise ValueError(f"Unknown conditioning mode: {cfg.mode!r}")
