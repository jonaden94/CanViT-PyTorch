"""Foveated patcher: ``fovi`` RetinalTransform + KNNPartitioningPatchEmbedding.

Operates on the **full image** (not a pre-cropped glimpse). Foveation is
anchored at the model's current fixation point, which lives in
``viewpoint.centers`` ((row, col) in ``[-1, 1]^2``, image-coord frame). The
``viewpoint.scales`` field is ignored — the fixation always covers the entire
image at scale 1.

Per-patch positions in the visual-field frame are exposed by fovi at
``KNNPartitioningPatchEmbedding.out_coords.cartesian_rowcol`` in ``[-1, 1]^2``
(row, col); they map to image-frame scene positions as

    scene_pos = viewpoint.centers + (fix_size / image_size) * vf_rowcol

With ``fix_size == image_size`` (full-image foveation) this reduces to
``scene_pos = viewpoint.centers + vf_rowcol``. Fixations near the edge produce
patches with ``|scene_pos| > 1`` — these encode out-of-image patches, which
RoPE handles gracefully (the model learns to ignore them implicitly).

fovi's modules keep their sampling state as plain attributes rather than
``nn.Module`` buffers. We override ``_apply`` so that the usual
``model.to(device)`` walks those tensors too — without this, the sampling
grid stays on CPU and ``grid_sample`` errors out at the first forward.

``fovi`` is an optional dependency declared as the ``[fovi]`` extra of
``canvit-pytorch``. Importing this module without fovi installed raises a
clear ``ImportError``.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch
from torch import Tensor, nn

from canvit_pytorch.patcher.base import Patcher
from canvit_pytorch.patcher.conditioning import (
    PatchConditioningConfig,
    conditioner_extra_in_channels,
    create_conditioner,
)
from canvit_pytorch.viewpoint import Viewpoint


@dataclass
class FoveatedPatcherConfig:
    """Configuration for the foveated patcher.

    Defaults track ``fovi/notebooks/explore_foveated_config.ipynb`` (the
    "real peripheral retina" regime: wide fov, mild cmf, ``pooling`` sampler).
    ``fixation_size`` matches the image side length used in training
    (``scene_resolution=512`` in the pretrain config). At forward time the
    fixation always covers the full image, so the relationship between
    ``cfg.fixation_size`` (fovi's reference) and the actual image size used
    determines whether the foveation pattern is correctly calibrated in
    pixels — keep them in sync.
    """

    fov: float = 180.0
    cmf_a: float = 0.5
    resolution: int = 36
    fixation_size: int = 512
    style: str = "isotropic"
    sampler: str = "pooling"
    cart_patch_size: int = 6
    sample_cortex: Literal["geodesic"] | bool = True
    arch_flag: str = ""
    ref_frame_side_length: int | None = None
    max_coord_val: float | Literal["auto"] = "auto"
    auto_match_cart_resources: bool = True
    force_patches_less_than_matched: bool = True
    hidden_dims_patch_embed: list[int] = field(default_factory=list)
    """Hidden layer widths for an MLP patch embedding. Empty (default) keeps the
    original pure-linear embedding (``kpe`` projects straight to ``embed_dim``).
    When non-empty, ``kpe`` outputs ``hidden_dims_patch_embed[0]`` and an MLP maps
    it to ``embed_dim`` with a ReLU between every pair of linear layers and no
    trailing activation. E.g. ``[1000]`` -> ``kpe``->1000, ReLU, Linear(1000->D);
    ``[1000, 1000]`` -> ``kpe``->1000, ReLU, Linear(1000->1000), ReLU,
    Linear(1000->D)."""
    conditioning: PatchConditioningConfig = field(default_factory=PatchConditioningConfig)
    """Optional position-conditioning of the patch embedding (see
    :class:`PatchConditioningConfig`). Default ``mode='none'`` reproduces the
    original unconditioned behavior exactly."""


def _require_fovi() -> None:
    try:
        import fovi  # noqa: F401
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "FoveatedPatcher requires the optional `fovi` dependency. "
            "Install via `pip install 'canvit-pytorch[fovi]'` or "
            "`uv add 'canvit-pytorch[fovi]'`."
        ) from e


class FoveatedPatcher(Patcher):
    """Foveated patcher backed by fovi.

    Patch positions are cached as a (non-persistent) buffer so that
    ``patcher.to(device)`` migrates them along with the model.
    """

    _patch_rowcol: Tensor  # [N, 2] in [-1, 1]^2, (row, col)

    def __init__(
        self,
        cfg: FoveatedPatcherConfig,
        *,
        embed_dim: int,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        _require_fovi()
        # Imported lazily so the uniform path has no fovi dependency.
        from fovi.arch.knnvit import KNNPartitioningPatchEmbedding
        from fovi.sensing.retina import RetinalTransform

        self.cfg = cfg
        self.embed_dim = embed_dim
        dev = torch.device(device) if isinstance(device, str) else device

        # Optional MLP patch embedding. With an empty `hidden_dims_patch_embed`
        # the KNN-conv projection (`self.kpe`) maps straight to `embed_dim`
        # (original pure-linear behavior). Otherwise `self.kpe` outputs the first
        # hidden width and `self.embed_head` (built below) maps it to `embed_dim`.
        # `self.embed_dim` always stays `embed_dim` — the patcher's output width
        # is fixed by the backbone, only the kpe's output width changes.
        hidden_dims = list(cfg.hidden_dims_patch_embed)
        kpe_embed_dim = hidden_dims[0] if hidden_dims else embed_dim

        # Extra sensor channels appended by the conditioner (CoordConv); known
        # from config alone so kpe's in_channels can be set before kpe is built.
        extra_in_channels = conditioner_extra_in_channels(cfg.conditioning)

        # Retinal sampling: ``start_res`` / ``fixation_size`` set fovi's
        # reference at construction time; at forward time we pass
        # ``fixation_size=image_H`` so the fixation always covers the full
        # image (the caller is expected to keep ``cfg.fixation_size`` in sync
        # with the image side length used at runtime).
        self.retina = RetinalTransform(
            resolution=cfg.resolution,
            start_res=cfg.fixation_size,
            fov=cfg.fov,
            cmf_a=cfg.cmf_a,
            style=cfg.style,
            sampler=cfg.sampler,
            fixation_size=cfg.fixation_size,
            auto_match_cart_resources=cfg.auto_match_cart_resources,
            device=str(dev),
        )

        # KNN patch embedding on the foveated samples. in_res / in_cart_res
        # use the *configured* resolution so the auto-match step inside the
        # embedding matches the one inside RetinalTransform (notebook checks
        # `in_coords match RT coords: True` under this convention).
        self.kpe = KNNPartitioningPatchEmbedding(
            in_channels=3 + extra_in_channels,
            embed_dim=kpe_embed_dim,
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
            arch_flag=cfg.arch_flag,
            ref_frame_side_length=cfg.ref_frame_side_length,
            device=str(dev),
        )

        # Cache patch positions (visual-field frame, [-1, 1]^2, (row, col)).
        # Registered as a buffer so .to(device) keeps them in lockstep with
        # the model. Non-persistent: regenerated on construction, not saved
        # in state_dict.
        rowcol = self.kpe.out_coords.cartesian_rowcol.detach().clone().to(torch.float32)
        self.register_buffer("_patch_rowcol", rowcol, persistent=False)

        # MLP head over the per-patch tokens produced by `self.kpe`. Empty when
        # `hidden_dims_patch_embed` is empty, in which case `self.embed_head` is
        # an identity `nn.Sequential` and `self.kpe` already outputs `embed_dim`.
        head_dims = hidden_dims + [embed_dim]
        head_layers: list[nn.Module] = []
        for i in range(len(hidden_dims)):
            head_layers += [nn.ReLU(), nn.Linear(head_dims[i], head_dims[i + 1])]
        self.embed_head = nn.Sequential(*head_layers).to(dev)

        # Position-conditioning. Fovea-centric (x, y) for retinal samples and
        # patch centers; conditioner built after kpe (needs kpe_out and coords)
        # and given a chance to touch kpe weights (CoordConv no-op-at-init).
        sample_xy = self.kpe.in_coords.cartesian.detach().clone().to(torch.float32)
        patch_xy = self.kpe.out_coords.cartesian.detach().clone().to(torch.float32)
        self.conditioner = create_conditioner(
            cfg.conditioning,
            n_patches=self.n_patches,
            kpe_out=kpe_embed_dim,
            embed_dim=embed_dim,
            sample_xy=sample_xy,
            patch_xy=patch_xy,
        ).to(dev)
        self.conditioner.after_kpe_built(self.kpe)

    @property
    def n_patches(self) -> int:
        return int(self._patch_rowcol.shape[0])

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> "FoveatedPatcher":
        out = super()._apply(fn, recurse=recurse)
        out._migrate_fovi_state(fn)
        return out

    def _migrate_fovi_state(self, fn: Callable[[Tensor], Tensor]) -> None:
        """Apply ``fn`` to fovi tensors held as plain Python attributes.

        ``RetinalTransform`` / ``GridSampler`` / ``KNNGridSampler`` /
        ``SamplingCoords`` / KNN layers stash sampling grids, KNN indices,
        reference coords, etc. as ordinary attributes (not
        ``register_buffer``), so ``nn.Module._apply`` does not migrate them
        when ``.to(device)`` is called. Walk those objects' ``__dict__`` and
        re-bind any Tensor attribute through ``fn``.

        Also patches each object's stored ``device`` attribute (used by some
        fovi forwards, e.g. ``KNNGridSampler.forward``'s ``img.to(self.device)``)
        so they don't drag tensors back to the original construction device
        after we have already moved everything.
        """

        def migrate(obj: Any) -> None:
            if obj is None:
                return
            d = getattr(obj, "__dict__", None)
            if not d:
                return
            buffers = getattr(obj, "_buffers", {})
            parameters = getattr(obj, "_parameters", {})
            for key, val in list(d.items()):
                # nn.Module routes registered buffers/params via _buffers /
                # _parameters dicts; skip those (already handled by super).
                if key in buffers or key in parameters:
                    continue
                if isinstance(val, Tensor) and not isinstance(val, nn.Parameter):
                    try:
                        new_val = fn(val)
                    except Exception:
                        continue
                    try:
                        setattr(obj, key, new_val)
                    except Exception:
                        pass

        sampler = self.retina.sampler
        # Order matters less than coverage; SamplingCoords lives both on the
        # sampler and on the KNN patcher's in_coords / out_coords. With
        # sampler="pooling" the sampler is a KNNGridSampler which carries
        # additional `highres_coords` and a `pooler` submodule worth walking.
        fovi_objs = (
            self.retina,
            sampler,
            getattr(sampler, "coords", None),
            getattr(sampler, "highres_coords", None),
            getattr(sampler, "pooler", None),
            self.kpe,
            getattr(self.kpe, "in_coords", None),
            getattr(self.kpe, "out_coords", None),
        )
        for obj in fovi_objs:
            migrate(obj)

        # Infer the target device from a known buffer that just got migrated,
        # and patch each fovi object's stored ``device`` so its forward
        # doesn't move incoming tensors back to the construction device.
        target_device = self._patch_rowcol.device
        for obj in fovi_objs:
            if obj is None:
                continue
            if hasattr(obj, "device"):
                try:
                    setattr(obj, "device", target_device)
                except Exception:
                    pass

    @property
    def fixation_size(self) -> int:
        return self.cfg.fixation_size

    def forward(self, image: Tensor, viewpoint: Viewpoint) -> tuple[Tensor, Tensor]:
        B, _, H, W = image.shape
        assert H == W, f"FoveatedPatcher expects a square image; got H={H}, W={W}"

        # fovi's ``fix_loc`` is (row, col) in [0, 1] normalized image coords.
        # ``viewpoint.centers`` is (row, col) in [-1, 1]; rescale.
        fix_loc = (viewpoint.centers.to(torch.float32) + 1.0) * 0.5  # [B, 2]
        # Full-image foveation: the fixation window equals the image.
        sensor = self.retina(image, fix_loc=fix_loc, fixation_size=H)  # [B, 3, N_samples]
        sensor = self.conditioner.transform_sensor(sensor)  # +coord channels (CoordConv)
        patches = self.kpe(sensor)  # [B, N_patches, kpe_embed_dim]
        patches = self.conditioner.modulate_kpe_output(patches)  # FiLM
        patches = self.embed_head(patches)  # [B, N_patches, embed_dim] (identity if no MLP)
        patches = self.conditioner.add_to_output(patches)  # learned per-patch bias

        # Scene positions for each patch, image-coord frame [-1, 1]^2.
        # When fix_size == image_size the conversion factor between
        # visual-field rowcol and image rowcol is 1.0; patches near the edge
        # may land at |scene_pos| > 1 (out-of-image), which is intentional —
        # RoPE handles it and the model learns to ignore those patches.
        rowcol = self._patch_rowcol.to(torch.float32)  # [N, 2]
        fix_size_norm = float(H) / float(H)  # = 1.0; explicit for clarity
        scene_pos = (
            viewpoint.centers.view(B, 1, 2).to(torch.float32)
            + fix_size_norm * rowcol.view(1, -1, 2)
        )  # [B, N, 2]
        return patches, scene_pos
