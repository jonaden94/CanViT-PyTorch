"""Uniform-grid patcher: wraps the backbone's existing PatchEmbed.

Owns the viewpoint crop step (``sample_at_viewpoint``) that used to live in
``CanViT.forward_reduce`` / the trainer. Lifting it inside the patcher keeps
the two patcher paths symmetric: both consume an image + viewpoint and
return patch tokens + scene positions.

When constructed with ``glimpse_size_px=None`` (the downstream-app path —
classification, segmentation, etc.), the patcher skips the crop and treats
the input tensor as the already-cropped glimpse. This keeps the existing
``clf(glimpse=...)`` / ``seg(glimpse=...)`` call sites working unchanged.
"""

from torch import Tensor

from canvit_pytorch.backbone.vit import ViTBackbone
from canvit_pytorch.coords import canvas_coords_for_glimpse
from canvit_pytorch.patcher.base import Patcher
from canvit_pytorch.viewpoint import Viewpoint, sample_at_viewpoint


class UniformPatcher(Patcher):
    """Uniform-grid patch extraction (CanViT's default behavior).

    When ``glimpse_size_px`` is set: crops a ``glimpse_size_px``-sized window
    at ``viewpoint`` via ``sample_at_viewpoint``, then delegates the embedding
    to ``backbone.patch_embed``. When ``glimpse_size_px`` is ``None``: skips
    the crop and patch-embeds the input directly. Per-patch scene-relative
    positions come from ``canvas_coords_for_glimpse``.

    The backbone is held by reference (not registered as a submodule) so the
    ``patch_embed.*`` keys stay under their original ``backbone.patch_embed.*``
    path in ``state_dict`` — existing checkpoints continue to load unchanged.
    """

    def __init__(self, backbone: ViTBackbone, *, glimpse_size_px: int | None = None) -> None:
        super().__init__()
        # Hold via tuple so nn.Module's __setattr__ does not register the
        # backbone as a submodule of the patcher.
        self._backbone_ref: tuple[ViTBackbone, ...] = (backbone,)
        self.embed_dim = backbone.embed_dim
        self.glimpse_size_px = glimpse_size_px

    @property
    def backbone(self) -> ViTBackbone:
        return self._backbone_ref[0]

    def forward(self, image: Tensor, viewpoint: Viewpoint) -> tuple[Tensor, Tensor]:
        if self.glimpse_size_px is not None:
            glimpse = sample_at_viewpoint(
                spatial=image, viewpoint=viewpoint, glimpse_size_px=self.glimpse_size_px,
            )
        else:
            glimpse = image
        patches, H, W = self.backbone.patch_embed(glimpse)
        assert H == W, f"UniformPatcher expects a square glimpse grid; got H={H}, W={W}"
        scene_pos = canvas_coords_for_glimpse(
            center=viewpoint.centers,
            scale=viewpoint.scales,
            H=H,
            W=W,
        ).flatten(1, 2)  # [B, H*W, 2]
        return patches, scene_pos
