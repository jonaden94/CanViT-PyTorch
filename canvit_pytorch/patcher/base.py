"""Patcher base class.

A Patcher maps a pixel glimpse + viewpoint to (patch tokens, scene positions).
Two implementations:
    - UniformPatcher: existing uniform-grid behavior (delegates to backbone.patch_embed)
    - FoveatedPatcher: foveated sampling via fovi.RetinalTransform + KNNPartitioningPatchEmbedding
"""

from torch import Tensor, nn

from canvit_pytorch.viewpoint import Viewpoint


class Patcher(nn.Module):
    """Maps a pixel glimpse and viewpoint to patch tokens and scene-relative positions.

    Contract:
        forward(glimpse, viewpoint) -> (patches, scene_positions)
            glimpse: [B, 3, gpx, gpx] pixel image around the viewpoint
            viewpoint: Viewpoint(centers=[B, 2], scales=[B]) in scene-relative coords
            patches: [B, N, embed_dim]
            scene_positions: [B, N, 2] in scene-relative [-1, 1]^2, (row, col)

    N is fixed per patcher instance but may differ between implementations.
    """

    embed_dim: int

    def forward(self, glimpse: Tensor, viewpoint: Viewpoint) -> tuple[Tensor, Tensor]:
        raise NotImplementedError
