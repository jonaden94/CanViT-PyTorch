"""Patcher base class.

A Patcher maps a (full) image + viewpoint to (patch tokens, scene positions).
Two implementations:
    - :class:`UniformPatcher`: existing uniform-grid behavior — internally crops
      a glimpse via ``sample_at_viewpoint`` then patchifies with the backbone.
    - :class:`FoveatedPatcher`: foveated sampling via fovi's RetinalTransform
      + KNNPartitioningPatchEmbedding; operates on the full image, anchored at
      ``viewpoint.centers`` (``scale`` is ignored).
"""

from torch import Tensor, nn

from canvit_pytorch.viewpoint import Viewpoint


class Patcher(nn.Module):
    """Maps a full image and viewpoint to patch tokens and scene-relative positions.

    Contract:
        forward(image, viewpoint) -> (patches, scene_positions)
            image: [B, 3, H, W] pixel image (full scene)
            viewpoint: Viewpoint(centers=[B, 2], scales=[B]) in scene-relative
                       coords. Uniform uses both; foveated uses only ``centers``.
            patches: [B, N, embed_dim]
            scene_positions: [B, N, 2] in scene-relative [-1, 1]^2, (row, col).
                             For foveated, may extend outside [-1, 1] when the
                             fixation is near an image edge.

    N is fixed per patcher instance but may differ between implementations.
    """

    embed_dim: int

    def forward(self, image: Tensor, viewpoint: Viewpoint) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def patch_positions(self) -> Tensor:
        """Constant fovea-centric patch-center positions ``[N, 2]`` as ``(x, y)``
        in ~``[-1, 1]`` (origin = fixation).

        Used as the conditioning signal for transformer-trunk / cross-attn
        modulation; fixed across batch and viewpoint. Patchers without a fixed
        per-patch layout (e.g. uniform with a variable glimpse grid) raise.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide patch_positions()"
        )
