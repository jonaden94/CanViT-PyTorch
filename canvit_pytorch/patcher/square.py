"""Square patcher: axis-aligned, uniform-resolution foveated patches.

A counterpart to :class:`~canvit_pytorch.patcher.foveated.FoveatedPatcher` that
samples the full image on a frozen set of ``P`` axis-aligned ``side x side``
square patches (``K = side**2`` samples each) and embeds each patch with a plain
linear projection. The sampling *pattern* is derived once (at fixation
``(0.5, 0.5)``) by one of three builders in :mod:`fovi.sensing.square`:

  - ``method="fovi"``            — squarified approximation of a fovi
    ``KNNPartitioningPatchEmbedding`` partition;
  - ``method="fovi_regularized"`` — the same, with outer rings snapped to an
    integer image-pixel lattice;
  - ``method="strided"``          — standalone strided-square foveation (no fovi
    sampling geometry; only :func:`fovi.sensing.square.build_strided_square`).

The pattern is frozen in fixation-window-normalized visual-field coords and
re-placed at the model's fixation at forward time via fovi's
``transform_sampling_grid`` — exactly mirroring ``FoveatedPatcher``'s
``scene_pos = centers + (fix_size / image_size) * vf_rowcol`` contract, so the
two patchers are interchangeable from CanViT's point of view.

Out-of-field samples (``pad_mask``) read zero (``padding="zero"``) or a learned
per-channel value (``padding="learned"``) — the square analog of fovi's
out-of-FOV neighbor padding. There is no token-level padding mask: fully
out-of-image patches become normal tokens placed by RoPE at ``|scene_pos| > 1``,
identical to ``FoveatedPatcher``.

This patcher reuses the embed-head MLP (:func:`canvit_pytorch.patcher.embed
.build_embed_head`) and the FiLM conditioner
(:mod:`canvit_pytorch.patcher.conditioning`). CoordConv conditioning is not
supported (its kernel-weight surgery is specific to the KNN-conv embed) and is
rejected at construction.

``fovi`` is required only for ``method in {"fovi", "fovi_regularized"}`` (to
build the geometry source) and for the shared coordinate transforms; it is
declared as the ``[fovi]`` extra of ``canvit-pytorch``.
"""

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from canvit_pytorch.patcher.base import Patcher
from canvit_pytorch.patcher.conditioning import PatchConditioningConfig, create_conditioner
from canvit_pytorch.patcher.embed import build_embed_head, count_unique_pixels
from canvit_pytorch.viewpoint import Viewpoint


@dataclass
class SquarePatcherConfig:
    """Configuration for the square patcher.

    A single ``method`` selector chooses the builder; the relevant parameter
    group is consulted and the rest ignored.
    """

    method: Literal["fovi", "fovi_regularized", "strided"] = "fovi"
    """Which sampling-pattern builder to use (see module docstring)."""

    # --- fovi geometry source (method = "fovi" / "fovi_regularized") --------
    fov: float = 180.0
    cmf_a: float = 0.5
    resolution: int = 36
    style: str = "isotropic"
    cart_patch_size: int = 6
    sample_cortex: Literal["geodesic"] | bool = True
    auto_match_cart_resources: bool = True
    force_patches_less_than_matched: bool = True
    max_coord_val: float | Literal["auto"] = "auto"
    ref_frame_side_length: int | None = None
    m_override: int | None = None
    """Override the square side length (default ``round(sqrt(k))`` from fovi)."""
    rel_tol: float = 0.05
    """Within-ring edge-length spread tolerance before warning."""

    # --- regularized-only (method = "fovi_regularized") ---------------------
    max_edge_change: float = 0.25
    strict_nest_when_possible: bool = False

    # --- strided (method = "strided") ---------------------------------------
    grid_size_fovea: int = 2
    patch_size: int = 6
    edge_length_multipliers: list[int] = field(default_factory=lambda: [2, 6])
    drop_corners: bool = False
    """Strided-only: drop the 4 corner patches (top-left/right, bottom-left/right)
    of the outermost ring, leaving the rest of the pattern bit-identical
    (``n_patches`` decreases by exactly 4). Only valid for ``method="strided"``,
    where the rings are squares with well-defined corners."""
    add_to_patch_size: int = 0
    """Strided-only: change the per-patch sampling density WITHOUT changing patch
    geometry. Each patch is sampled with ``(patch_size + add_to_patch_size)`` points
    per axis (``K = (patch_size + add_to_patch_size)**2``), spaced evenly and centered
    within its (unchanged) cell. Patch centers, extents and ``n_patches`` are invariant
    to this; only the sample count/spacing changes. ``0`` (default) = native resolution;
    ``>0`` oversamples (sub-pixel, bilinearly interpolated); ``<0`` undersamples. Must
    satisfy ``patch_size + add_to_patch_size >= 2``. Only valid for ``method="strided"``."""

    # --- shared deploy / embed ----------------------------------------------
    pattern_reference_size: int = 512
    """Reference window size (px) the frozen sampling pattern is *built* against,
    decoupled from the deploy window (which is per-forward ``scale * H``). Used by
    ``fovi_regularized`` (the integer-pixel snapping grid) and ``strided`` (the
    visual-field scale and the out-of-window pad threshold); **ignored for
    geometry by** ``fovi`` (its geometry and padding are reference-independent).
    Also the scale at which ``min_ring_new_pixels`` pixel-coverage pruning is
    reckoned (all methods). Pinning it to a fixed value makes the frozen pattern
    — and its padded-FOV bound and pruned token set — invariant to the deploy
    zoom. For ``fovi_regularized``, integer-pixel alignment is exact only when the
    deploy window (``scale * H``) equals this reference; at other sizes the frozen
    pattern is rescaled (sample coincidences preserved)."""
    min_ring_new_pixels: int = 0
    """Prune every patch whose concentric ring contributes fewer than this many
    *new* image pixels (not already covered by an outer ring) at the
    ``pattern_reference_size`` scale — the ``ring_pixel_stats`` "new_pixel" metric
    from ``fovi/notebooks/fovi_square_patches/fovi_plus_square_patches.ipynb``.
    ``0`` (default) disables pruning and is bit-identical to the unpruned pattern.
    Drops whole rings; mirrors ``FoveatedPatcherConfig.min_ring_new_pixels``."""
    hidden_dims_patch_embed: list[int] = field(default_factory=list)
    """Hidden widths for an MLP patch embedding (see ``build_embed_head``).
    Empty (default) -> the linear embed maps straight to ``embed_dim``."""
    conditioning: PatchConditioningConfig = field(default_factory=PatchConditioningConfig)
    """Optional position-conditioning. ``coordconv`` is not supported here."""
    padding: Literal["zero", "learned"] = "zero"
    """Fill for the ``pad_mask`` sample slots — **always applied** (it bounds the
    sampled visual field to the pattern's design, matching fovi's FOV). ``zero``
    (default): the masked slots contribute 0, exactly like fovi's default
    zero-padding of out-of-FOV neighbors. ``learned``: a learned per-channel
    value (zero-init, so identical to ``zero`` at step 0). Independent of
    ``grid_sample``'s zero-padding, which only blanks out-of-*image* samples."""


def _require_fovi() -> None:
    try:
        import fovi  # noqa: F401
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "SquarePatcher requires the optional `fovi` dependency. "
            "Install via `pip install 'canvit-pytorch[fovi]'` or "
            "`uv add 'canvit-pytorch[fovi]'`."
        ) from e


def _build_fovi_pe(cfg: SquarePatcherConfig, device: torch.device | str):
    """Construct a fovi ``KNNPartitioningPatchEmbedding`` as the geometry source.

    Only its sampling coordinates / KNN partition are used (to derive the square
    pattern); the embedding weights are discarded. Mirrors the kpe construction
    in ``FoveatedPatcher`` so the squarified pattern approximates the same
    foveation the foveated patcher would produce under matching config.
    """
    from fovi.arch.knnvit import KNNPartitioningPatchEmbedding

    return KNNPartitioningPatchEmbedding(
        in_channels=3,
        embed_dim=1,  # unused (geometry only)
        in_res=cfg.resolution,
        in_cart_res=cfg.resolution,
        fov=cfg.fov,
        cmf_a=cfg.cmf_a,
        style=cfg.style,
        auto_match_cart_resources=cfg.auto_match_cart_resources,
        cart_patch_size=cfg.cart_patch_size,
        force_patches_less_than_matched=cfg.force_patches_less_than_matched,
        transposed=False,
        max_coord_val=cfg.max_coord_val,
        sample_cortex=cfg.sample_cortex,
        ref_frame_side_length=cfg.ref_frame_side_length,
        device=str(device),
    )


def _build_pattern(cfg: SquarePatcherConfig, device: torch.device | str):
    """Dispatch to the selected :mod:`fovi.sensing.square` builder."""
    from fovi.sensing.square import (
        build_fovi_square,
        build_fovi_square_regularized,
        build_strided_square,
    )

    # The pattern is built against a fixed reference window, decoupled from the
    # per-forward deploy window (scale * H).
    ref_size = cfg.pattern_reference_size
    if cfg.method == "strided":
        return build_strided_square(
            grid_size_fovea=cfg.grid_size_fovea,
            patch_size=cfg.patch_size,
            edge_length_multipliers=list(cfg.edge_length_multipliers),
            fixation_size=ref_size,
            add_to_patch_size=cfg.add_to_patch_size,
        )
    pe = _build_fovi_pe(cfg, device)
    if cfg.method == "fovi":
        return build_fovi_square(pe, m_override=cfg.m_override, rel_tol=cfg.rel_tol)
    if cfg.method == "fovi_regularized":
        return build_fovi_square_regularized(
            pe,
            fix_size=ref_size,
            max_edge_change=cfg.max_edge_change,
            rel_tol=cfg.rel_tol,
            m_override=cfg.m_override,
            strict_nest_when_possible=cfg.strict_nest_when_possible,
        )
    raise ValueError(f"Unknown square method: {cfg.method!r}")


class SquarePatcher(Patcher):
    """Square-patch patcher backed by a frozen :class:`fovi.sensing.square.SquarePattern`.

    The sampling grid (``_sample_colrow``), per-patch scene rowcol
    (``_patch_rowcol``) and out-of-field mask (``_pad_mask``) are cached as
    non-persistent buffers so ``patcher.to(device)`` migrates them with the
    model. Unlike ``FoveatedPatcher`` there is no fovi runtime state to migrate
    (the geometry source is discarded after construction).
    """

    _sample_colrow: Tensor  # [1, 1, P*K, 2] grid_sample-frame (x, y) at fixation (0.5, 0.5)
    _patch_rowcol: Tensor   # [P, 2] visual-field (row, col)
    _pad_mask: Tensor       # [P, K] bool, True = out-of-field sample
    _patch_xy: Tensor       # [P, 2] fovea-centric (x, y); trunk-modulation signal

    def __init__(
        self,
        cfg: SquarePatcherConfig,
        *,
        embed_dim: int,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        _require_fovi()
        from fovi.sensing.coords import xy_to_colrow, xy_to_rowcol

        assert cfg.conditioning.mode not in ("coordconv", "coordconv_film"), (
            "SquarePatcher does not support coordconv conditioning "
            f"(got mode={cfg.conditioning.mode!r}); its weight surgery is "
            "specific to the KNN-conv embed."
        )
        if cfg.add_to_patch_size != 0 and cfg.method != "strided":
            raise ValueError(
                "add_to_patch_size is only supported for method='strided' (it "
                "resamples the strided cell grid); got "
                f"method={cfg.method!r} with add_to_patch_size={cfg.add_to_patch_size}."
            )

        self.cfg = cfg
        self.embed_dim = embed_dim
        dev = torch.device(device) if isinstance(device, str) else device

        pattern = _build_pattern(cfg, dev)

        # Optional ring pruning: drop patches whose ring adds fewer than
        # `min_ring_new_pixels` new pixels at the `pattern_reference_size` scale
        # (the notebook's ring_pixel_stats metric). Done on the frozen pattern
        # before any buffer is built, so every downstream array (sample grid,
        # positions, pad mask, FiLM, trunk modulation) is the pruned set. `0`
        # disables it (bit-identical to the unpruned pattern).
        if cfg.min_ring_new_pixels > 0:
            from fovi.sensing.square import square_ring_keep_mask

            keep = square_ring_keep_mask(
                pattern,
                reference_size=cfg.pattern_reference_size,
                min_new_pixels=cfg.min_ring_new_pixels,
            )
            if not bool(keep.all()):
                pattern = pattern.subset(keep)

        # Optional corner drop (strided only): remove the 4 corner patches of the
        # outermost ring — top-left/right, bottom-left/right, i.e. the 4 diagonal
        # extremes (argmax of ±x±y). The rest of the pattern is bit-identical;
        # n_patches decreases by exactly 4. Done on the frozen pattern before any
        # buffer is built, so every downstream array is the reduced set.
        if cfg.drop_corners:
            if cfg.method != "strided":
                raise ValueError(
                    "drop_corners is only supported for method='strided' (its square "
                    f"rings have well-defined corners); got method={cfg.method!r}."
                )
            c = pattern.centers_xy
            outer = (
                pattern.ring_idx == pattern.ring_idx.max()
                if pattern.ring_idx is not None
                else torch.ones(c.shape[0], dtype=torch.bool)
            )
            outer_pos = torch.nonzero(outer, as_tuple=False).flatten()
            co = c[outer]
            corners = {
                int(outer_pos[int((sx * co[:, 0] + sy * co[:, 1]).argmax())])
                for sx in (1, -1)
                for sy in (1, -1)
            }
            assert len(corners) == 4, (
                f"drop_corners expected 4 distinct corner patches, found {len(corners)}"
            )
            keep = torch.ones(c.shape[0], dtype=torch.bool)
            keep[list(corners)] = False
            pattern = pattern.subset(keep)

        self._n_patches = pattern.n_patches
        self._k = pattern.k
        self._side = pattern.side
        in_channels = 3

        # Frozen sampling grid in grid_sample frame (mirrors fovi's
        # _relative_to_gridsample: xy -> colrow, then transform_sampling_grid at
        # forward time). Shape [1, 1, P*K, 2] so transform_sampling_grid (which
        # indexes a 4-D grid) can broadcast over the batch.
        colrow = xy_to_colrow(
            pattern.positions_xy.reshape(-1, 2).to(torch.float32), do_norm=False, format="-11"
        )
        self.register_buffer("_sample_colrow", colrow.reshape(1, 1, -1, 2).contiguous(), persistent=False)

        # Per-patch scene rowcol (same convention as SamplingCoords.cartesian_rowcol).
        patch_rowcol = xy_to_rowcol(
            pattern.centers_xy.to(torch.float32), do_norm=False, format="-11"
        )
        self.register_buffer("_patch_rowcol", patch_rowcol.contiguous(), persistent=False)
        self.register_buffer("_pad_mask", pattern.pad_mask.contiguous(), persistent=False)

        # Per-patch concentric-ring index (0 = innermost), exact for all three
        # methods (eccentricity rank for fovi/regularized; fovea+layer id for
        # strided). Carried purely for downstream inspection / visualization
        # (e.g. coloring patches by ring); not used in forward. Falls back to a
        # single ring if a builder ever omits it.
        ring_idx = pattern.ring_idx
        if ring_idx is None:
            ring_idx = torch.zeros(self._n_patches, dtype=torch.long)
        self.register_buffer("_ring_idx", ring_idx.to(torch.long).contiguous(), persistent=False)

        # Core linear embed over each patch's flattened K*C samples. Mirrors
        # FoveatedPatcher: `embed` outputs `kpe_embed_dim` (= embed_dim, or the
        # first MLP hidden width), then `embed_head` maps to `embed_dim`.
        hidden_dims = list(cfg.hidden_dims_patch_embed)
        kpe_embed_dim = hidden_dims[0] if hidden_dims else embed_dim
        self.embed = nn.Linear(in_channels * self._k, kpe_embed_dim).to(dev)
        self.embed_head = build_embed_head(hidden_dims, embed_dim, dev)

        # Optional learned per-channel padding value (zero-init -> no-op at init).
        if cfg.padding == "learned":
            self.pad_value = nn.Parameter(torch.zeros(in_channels, device=dev))
        else:
            self.pad_value = None

        # Position-conditioning (FiLM / none). Fovea-centric (x, y) for samples
        # and patch centers, constant across batch/fixation.
        sample_xy = pattern.positions_xy.reshape(-1, 2).detach().clone().to(torch.float32)
        patch_xy = pattern.centers_xy.detach().clone().to(torch.float32)
        # Cache fovea-centric (x, y) for trunk/cross-attn modulation (constant).
        self.register_buffer("_patch_xy", patch_xy, persistent=False)
        self.conditioner = create_conditioner(
            cfg.conditioning,
            kpe_out=kpe_embed_dim,
            sample_xy=sample_xy,
            patch_xy=patch_xy,
        ).to(dev)

    @property
    def n_patches(self) -> int:
        return self._n_patches

    def patch_positions(self) -> Tensor:
        return self._patch_xy

    @property
    def ring_idx(self) -> Tensor:
        """Per-patch concentric-ring index ``[P]`` (0 = innermost). For
        inspection / visualization (e.g. coloring patches by ring)."""
        return self._ring_idx

    @property
    def pad_mask(self) -> Tensor:
        """Per-sample out-of-field mask ``[P, K]`` (True = padding)."""
        return self._pad_mask

    def sample_positions_xy(self) -> Tensor:
        """Recover the frozen visual-field sample positions ``[P, K, 2]`` (x, y).

        Inverse of the ``xy -> colrow`` used to build ``_sample_colrow``
        (``col = x``, ``row = -y``). These are the fixation-window-normalized
        coordinates the patcher samples at ``fix_loc = (0.5, 0.5)``; useful for
        reproducing the notebook sampling-pattern metrics from the live patcher.
        """
        colrow = self._sample_colrow.reshape(self._n_patches, self._k, 2)
        x = colrow[..., 0]
        y = -colrow[..., 1]
        return torch.stack([x, y], dim=-1).contiguous()

    def pattern_stats(self) -> dict[str, int]:
        """High-level characterization of the frozen sampling pattern, for logging.

        ``n_patches`` / ``samples_per_patch`` (= ``K``); ``n_padded`` = the
        structurally-padded sample slots (the fixation-invariant FOV mask, *not*
        out-of-image padding); ``unique_pixels`` = distinct in-image pixels the
        *non-padded* samples resolve at the fixed ``pattern_reference_size``
        scale, centered (the deploy window is per-forward ``scale * H``).
        See :func:`canvit_pytorch.patcher.embed.count_unique_pixels`.
        """
        non_padded = self.sample_positions_xy()[~self._pad_mask]
        return {
            "n_patches": int(self._n_patches),
            "samples_per_patch": int(self._k),
            "n_padded": int(self._pad_mask.sum()),
            "unique_pixels": count_unique_pixels(non_padded, self.cfg.pattern_reference_size),
        }

    def forward(self, image: Tensor, viewpoint: Viewpoint) -> tuple[Tensor, Tensor]:
        from fovi.sensing.coords import transform_sampling_grid

        B, C, H, W = image.shape
        assert H == W, f"SquarePatcher expects a square image; got H={H}, W={W}"
        assert C == 3, f"SquarePatcher expects 3-channel images; got C={C}"
        P, K = self._n_patches, self._k

        # fovi's fix_loc is (row, col) in [0, 1]; viewpoint.centers is (row, col)
        # in [-1, 1]. Per-sample foveation window: fix_size = scale * H.
        fix_loc = (viewpoint.centers.to(torch.float32) + 1.0) * 0.5  # [B, 2]
        scales = viewpoint.scales.to(torch.float32)  # [B]
        fix_px = scales * float(H)  # [B]
        fix_size_t = torch.stack([fix_px, fix_px], dim=-1)  # [B, 2]

        abs_grid = transform_sampling_grid(
            self._sample_colrow, fix_loc, fix_size_t, (H, W)
        )  # [B, 1, P*K, 2]
        samp = F.grid_sample(
            image.to(torch.float32), abs_grid,
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )  # [B, C, 1, P*K]
        samp = samp[:, :, 0, :].reshape(B, C, P, K)  # [B, C, P, K]

        # Always blank the structurally-padded sample slots so each patch's
        # effective visual field is bounded by its `pad_mask` (matching fovi's
        # FOV) rather than reading real image content there. `padding="zero"`
        # (default) fills 0 — identical to fovi's default zero-padding of
        # out-of-FOV neighbors; `padding="learned"` fills the learned per-channel
        # value (zero-init, so identical at step 0). This is independent of
        # `grid_sample`'s zero-padding, which only blanks out-of-*image* samples.
        pad = self._pad_mask.view(1, 1, P, K)
        if self.pad_value is not None:
            samp = torch.where(pad, self.pad_value.view(1, C, 1, 1), samp)
        else:
            samp = samp.masked_fill(pad, 0.0)

        # Flatten each patch's samples (channel-major, then sample) and embed.
        x = samp.permute(0, 2, 1, 3).reshape(B, P, C * K)  # [B, P, C*K]
        x = self.embed(x)                                   # [B, P, kpe_embed_dim]
        # `scale` ([B]) is used only by scale-aware FiLM (encode_scale=True);
        # other conditioners ignore it, so this stays identity when unused.
        x = self.conditioner.modulate_kpe_output(x, scale=scales)  # FiLM (identity if unused)
        x = self.embed_head(x)                              # [B, P, embed_dim]

        # Scene positions: same window-to-image mapping as FoveatedPatcher
        # (fix_size / H == scale, per-sample).
        scene_pos = (
            viewpoint.centers.view(B, 1, 2).to(torch.float32)
            + scales.view(B, 1, 1) * self._patch_rowcol.view(1, -1, 2)
        )  # [B, P, 2]
        return x, scene_pos
