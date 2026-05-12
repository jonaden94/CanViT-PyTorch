"""Foveated patcher: ``fovi`` RetinalTransform + KNNPartitioningPatchEmbedding.

Treats the pre-cropped square glimpse as a fixation window centered on
``fix_loc=(0.5, 0.5)`` and reads it through fovi's foveated sensor + KNN-based
patch embedding. Per-patch positions in the visual-field frame are exposed by
fovi at ``KNNPartitioningPatchEmbedding.out_coords.cartesian_rowcol`` in
``[-1, 1]^2`` ``(row, col)``; they map to scene-relative coords via the
same formula the uniform patcher uses for its grid:

    scene_pos = viewpoint.center + viewpoint.scale * visual_field_rowcol

fovi's modules keep their sampling state as plain attributes rather than
``nn.Module`` buffers. We override ``_apply`` so that the usual
``model.to(device)`` walks those tensors too — without this, the sampling
grid stays on CPU and ``grid_sample`` errors out at the first forward.

``fovi`` is an optional dependency declared as the ``[fovi]`` extra of
``canvit-pytorch``. Importing this module without fovi installed raises a
clear ``ImportError``.
"""

from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
from torch import Tensor, nn

from canvit_pytorch.patcher.base import Patcher
from canvit_pytorch.viewpoint import Viewpoint


@dataclass
class FoveatedPatcherConfig:
    """Configuration for the foveated patcher.

    Defaults match fovi's production ``config/fovi-dinov3.yaml`` (fov=16°,
    cmf_a≈2.79, isotropic / grid_nn / geodesic) at 256-px fixation and 64-px
    foveated grid.
    """

    fov: float = 16.0
    cmf_a: float = 2.785765
    resolution: int = 64
    fixation_size: int = 256
    style: str = "isotropic"
    sampler: str = "grid_nn"
    cart_patch_size: int = 8
    sample_cortex: Literal["geodesic"] | bool = "geodesic"
    arch_flag: str = ""
    ref_frame_side_length: int | None = None
    max_coord_val: float | Literal["auto"] = "auto"
    auto_match_cart_resources: bool = True
    force_patches_less_than_matched: bool = True


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

        # Retinal sampling: the entire glimpse is the fixation window, so
        # start_res / fixation_size both equal the foveated glimpse size.
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
            in_channels=3,
            embed_dim=embed_dim,
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

    @property
    def n_patches(self) -> int:
        return int(self._patch_rowcol.shape[0])

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> "FoveatedPatcher":
        out = super()._apply(fn, recurse=recurse)
        out._migrate_fovi_state(fn)
        return out

    def _migrate_fovi_state(self, fn: Callable[[Tensor], Tensor]) -> None:
        """Apply ``fn`` to fovi tensors held as plain Python attributes.

        ``RetinalTransform`` / ``GridSampler`` / ``SamplingCoords`` / KNN
        layers stash sampling grids, KNN indices, reference coords, etc. as
        ordinary attributes (not ``register_buffer``), so ``nn.Module._apply``
        does not migrate them when ``.to(device)`` is called. Walk those
        objects' ``__dict__`` and re-bind any Tensor attribute through ``fn``.
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
        # sampler and on the KNN patcher's in_coords / out_coords.
        for obj in (
            self.retina,
            sampler,
            getattr(sampler, "coords", None),
            self.kpe,
            getattr(self.kpe, "in_coords", None),
            getattr(self.kpe, "out_coords", None),
        ):
            migrate(obj)

    @property
    def fixation_size(self) -> int:
        return self.cfg.fixation_size

    def forward(self, glimpse: Tensor, viewpoint: Viewpoint) -> tuple[Tensor, Tensor]:
        B = glimpse.shape[0]
        gpx = glimpse.shape[-1]
        # RetinalTransform expects fix_loc as (row, col) in [0, 1]; the whole
        # crop is the fixation window, so fix_loc=(0.5, 0.5) and
        # fixation_size=gpx. We pass a per-batch tensor to avoid the
        # constructor's default scalar path.
        fix_loc = glimpse.new_full((B, 2), 0.5, dtype=torch.float32)
        # `fixation_size` accepts an int or per-batch array
        sensor = self.retina(glimpse, fix_loc=fix_loc, fixation_size=gpx)  # [B, 3, N_samples]
        patches = self.kpe(sensor)  # [B, N_patches, embed_dim]

        # Map visual-field positions to scene-relative coords.
        rowcol = self._patch_rowcol.to(dtype=torch.float32)  # [N, 2]
        scene_pos = (
            viewpoint.centers.view(B, 1, 2).to(torch.float32)
            + viewpoint.scales.view(B, 1, 1).to(torch.float32) * rowcol.view(1, -1, 2)
        )  # [B, N, 2]
        return patches, scene_pos
