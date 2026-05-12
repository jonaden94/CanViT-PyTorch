"""Uniform-grid patcher: wraps the backbone's existing PatchEmbed."""

from torch import Tensor

from canvit_pytorch.backbone.vit import ViTBackbone
from canvit_pytorch.coords import canvas_coords_for_glimpse
from canvit_pytorch.patcher.base import Patcher
from canvit_pytorch.viewpoint import Viewpoint


class UniformPatcher(Patcher):
    """Uniform-grid patch extraction (CanViT's default behavior).

    Delegates to ``backbone.patch_embed`` for the embedding and computes per-patch
    scene-relative positions via ``canvas_coords_for_glimpse``.

    The backbone is held by reference (not registered as a submodule) so the
    ``patch_embed.*`` keys stay under their original ``backbone.patch_embed.*``
    path in ``state_dict`` — existing checkpoints continue to load unchanged.
    """

    def __init__(self, backbone: ViTBackbone) -> None:
        super().__init__()
        # Hold via tuple so nn.Module's __setattr__ does not register the
        # backbone as a submodule of the patcher.
        self._backbone_ref: tuple[ViTBackbone, ...] = (backbone,)
        self.embed_dim = backbone.embed_dim

    @property
    def backbone(self) -> ViTBackbone:
        return self._backbone_ref[0]

    def forward(self, glimpse: Tensor, viewpoint: Viewpoint) -> tuple[Tensor, Tensor]:
        patches, H, W = self.backbone.patch_embed(glimpse)
        assert H == W, f"UniformPatcher expects a square glimpse grid; got H={H}, W={W}"
        scene_pos = canvas_coords_for_glimpse(
            center=viewpoint.centers,
            scale=viewpoint.scales,
            H=H,
            W=W,
        ).flatten(1, 2)  # [B, H*W, 2]
        return patches, scene_pos
