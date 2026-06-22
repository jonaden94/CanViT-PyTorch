import torch
from torch import Tensor
from canvit_pytorch.correctness import assert_shape


def grid_coords(
    *,
    H: int,
    W: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Normalized coordinates for an H×W grid in [-1, 1]².

    Used directly for canvas or as base for retinotopic_to_canvas.
    Convention: coord = (idx + 0.5) / size * 2 - 1 (cell centers, not edges).

    Returns:
        [H, W, 2] with [..., 0] = y, [..., 1] = x
    """
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    y = (y + 0.5) / H * 2 - 1
    x = (x + 0.5) / W * 2 - 1
    out = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1)
    assert_shape(out, (H, W, 2))
    return out


def uniform_grid_coords(
    *,
    g: int,
    patch_size: int,
    stride: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Normalized [-1, 1]² centers of an overlapping uniform patch grid.

    A g×g grid of ``patch_size``-px patches placed every ``stride`` px spans a
    glimpse of ``(g-1)*stride + patch_size`` px; patch i's true center sits at
    ``i*stride + patch_size/2`` px. These are the coordinates fed to RoPE and the
    canvas read/write, so overlapping patches get their real centers (matching the
    foveated/square paths). With ``stride == patch_size`` this is algebraically the
    non-overlapping cell-center grid (``grid_coords``); the uniform patcher still
    routes the ``stride == patch_size`` case through ``grid_coords`` for bit-exact
    reproduction.

    Returns:
        [g, g, 2] with [..., 0] = y, [..., 1] = x
    """
    glimpse = (g - 1) * stride + patch_size
    c = (torch.arange(g, device=device, dtype=dtype) * stride + patch_size / 2) / glimpse * 2 - 1
    out = torch.stack(torch.meshgrid(c, c, indexing="ij"), dim=-1)
    assert_shape(out, (g, g, 2))
    return out


def canvas_coords_for_glimpse(
    *,
    center: Tensor,
    scale: Tensor,
    H: int,
    W: int,
    retinotopic: Tensor | None = None,
) -> Tensor:
    """Compute canvas coordinates for each cell of a glimpse grid.

    Formula: canvas_pos = center + scale * retinotopic_pos

    Args:
        center: [B, 2] where the glimpse is centered (in canvas coords, y/x), must be float32
        scale: [B] glimpse size relative to canvas (1.0 = full canvas), must be float32
        H, W: glimpse grid dimensions
        retinotopic: optional [H, W, 2] normalized cell positions to use instead of
            the default ``grid_coords`` (e.g. true overlapping-patch centers from
            ``uniform_grid_coords``). Defaults to ``grid_coords(H, W)``.

    Returns:
        [B, H, W, 2] canvas coordinates, float32
    """
    B = center.shape[0]
    assert_shape(center, (B, 2))
    assert_shape(scale, (B,))
    assert center.dtype == torch.float32, f"center must be float32, got {center.dtype}"
    assert scale.dtype == torch.float32, f"scale must be float32, got {scale.dtype}"

    if retinotopic is None:
        retinotopic = grid_coords(H=H, W=W, device=center.device)
    else:
        assert_shape(retinotopic, (H, W, 2))
    canvas = center.view(B, 1, 1, 2) + scale.view(B, 1, 1, 1) * retinotopic
    assert_shape(canvas, (B, H, W, 2))
    return canvas
