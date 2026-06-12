"""Sanity tests for the foveated patcher.

Covers:
  - Output shapes (patches, scene_positions).
  - Position-frame correctness: under viewpoint=(center=0, scale=1) the foveal
    patch sits at scene origin, and rowcol sign convention matches CanViT's
    (row=-1 → top, row=+1 → bottom).
  - Viewpoint translation/scaling: scene_pos = center + scale * rowcol.
  - Drop-in integration with CanViT (uniform-mode regression unaffected).

Skipped when the optional ``fovi`` dependency is not installed.
"""

import importlib.util

import pytest
import torch

from canvit_pytorch import (
    CanViT,
    CanViTConfig,
    FoveatedPatcherConfig,
    Viewpoint,
    create_backbone,
    create_patcher,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fovi") is None,
    reason="fovi optional dependency not installed",
)

# Compact foveated config that produces ~13 patches quickly. Keeps cmf_a /
# fixation_size / fov in a regime where the patch nearest the foveal center
# is genuinely near (0, 0) — the sign-convention test depends on this.
SMALL_FOVEATED_CFG = FoveatedPatcherConfig(
    fov=16.0,
    cmf_a=2.785765,
    resolution=32,
    fixation_size=128,
    style="isotropic",
    sampler="grid_nn",
    cart_patch_size=8,
    sample_cortex=True,
)


@pytest.fixture(scope="module")
def backbone():
    return create_backbone("vits16")


@pytest.fixture(scope="module")
def foveated_patcher(backbone):
    return create_patcher(
        "foveated", backbone=backbone, foveated_config=SMALL_FOVEATED_CFG
    )


def test_n_patches_and_buffer_shape(foveated_patcher):
    assert foveated_patcher.n_patches > 0
    assert foveated_patcher._patch_rowcol.shape == (foveated_patcher.n_patches, 2)
    assert foveated_patcher._patch_rowcol.dtype == torch.float32


def test_forward_shapes(foveated_patcher):
    B = 2
    gpx = SMALL_FOVEATED_CFG.fixation_size
    embed_dim = foveated_patcher.embed_dim
    image = torch.randn(B, 3, gpx, gpx)
    vp = Viewpoint.full_scene(batch_size=B, device=image.device)
    with torch.inference_mode():
        patches, scene_pos = foveated_patcher(image, vp)
    N = foveated_patcher.n_patches
    assert patches.shape == (B, N, embed_dim)
    assert scene_pos.shape == (B, N, 2)
    assert scene_pos.dtype == torch.float32


def test_foveal_patch_at_origin(foveated_patcher):
    """The patch nearest the fixation center should sit at ~origin in rowcol."""
    rowcol = foveated_patcher._patch_rowcol
    norms = (rowcol ** 2).sum(-1)
    foveal = rowcol[int(norms.argmin())]
    assert foveal.abs().max() < 1e-3, f"Expected foveal patch near origin; got {foveal}"


def test_scene_positions_full_scene(foveated_patcher):
    """Full-scene viewpoint (centers=0) -> scene_pos == rowcol (foveation
    covers full image, viewpoint.scales is ignored)."""
    B = 3
    gpx = SMALL_FOVEATED_CFG.fixation_size
    image = torch.randn(B, 3, gpx, gpx)
    vp = Viewpoint.full_scene(batch_size=B, device=image.device)
    with torch.inference_mode():
        _, scene_pos = foveated_patcher(image, vp)
    rowcol = foveated_patcher._patch_rowcol
    for b in range(B):
        assert torch.allclose(scene_pos[b], rowcol, atol=1e-6)


def test_scene_positions_translated(foveated_patcher):
    """Off-center viewpoint shifts scene_pos by viewpoint.centers
    (scale is ignored in the new full-image foveation contract).
    Edge fixations may produce |scene_pos| > 1 — out-of-image patches."""
    B = 1
    gpx = SMALL_FOVEATED_CFG.fixation_size
    image = torch.randn(B, 3, gpx, gpx)
    centers = torch.tensor([[0.3, -0.2]], dtype=torch.float32)
    scales = torch.tensor([0.4], dtype=torch.float32)  # ignored
    vp = Viewpoint(centers=centers, scales=scales)
    with torch.inference_mode():
        _, scene_pos = foveated_patcher(image, vp)
    rowcol = foveated_patcher._patch_rowcol
    expected = centers.view(1, 1, 2) + rowcol.view(1, -1, 2)
    assert torch.allclose(scene_pos, expected, atol=1e-6)


def test_canvit_end_to_end_foveated(backbone):
    """Full CanViT forward in foveated mode produces canvas-shape outputs."""
    cfg = CanViTConfig(
        patcher_name="foveated",
        foveated_patcher=SMALL_FOVEATED_CFG,
    )
    model = CanViT(backbone=backbone, cfg=cfg).eval()
    B = 2
    gpx = SMALL_FOVEATED_CFG.fixation_size
    canvas_grid = 16
    image = torch.randn(B, 3, gpx, gpx)
    vp = Viewpoint.full_scene(batch_size=B, device=image.device)
    state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)
    with torch.inference_mode():
        out = model(image=image, state=state, viewpoint=vp)
    assert out.local_patches.shape[0] == B
    assert out.local_patches.shape[1] == model.patcher.n_patches
    assert out.local_patches.shape[2] == backbone.embed_dim
    n_canvas = cfg.n_canvas_registers + canvas_grid ** 2
    assert out.state.canvas.shape == (B, n_canvas, cfg.canvas_dim)


def test_uniform_state_dict_no_patcher_keys(backbone):
    """Uniform-mode model must not introduce new state_dict keys."""
    model = CanViT(backbone=backbone, cfg=CanViTConfig()).eval()
    patcher_keys = [k for k in model.state_dict() if k.startswith("patcher")]
    assert patcher_keys == [], (
        f"Uniform mode should have no patcher.* keys; got {patcher_keys}"
    )


# --------------------------------------------------------------------------- #
# Ring pruning (min_ring_new_pixels)
# --------------------------------------------------------------------------- #

# A strongly-foveated config whose innermost rings oversample below the 512-px
# reference grid, so they get pruned at min_ring_new_pixels=40 (26 -> 21 here).
PRUNING_FOVEATED_CFG = FoveatedPatcherConfig(
    fov=180.0,
    cmf_a=0.5,
    resolution=36,
    fixation_size=128,
    style="isotropic",
    sampler="pooling",
    cart_patch_size=6,
    sample_cortex=True,
)


def _fov_cfg(**over):
    from dataclasses import replace
    return replace(PRUNING_FOVEATED_CFG, **over)


def test_prune_disabled_is_noop():
    """min_ring_new_pixels=0 keeps all patches; pattern_reference_size is inert
    when pruning is off (different ref -> identical patcher)."""
    p_ref512 = create_patcher("foveated", backbone=create_backbone("vits16"),
                              foveated_config=_fov_cfg(min_ring_new_pixels=0, pattern_reference_size=512))
    p_ref999 = create_patcher("foveated", backbone=create_backbone("vits16"),
                              foveated_config=_fov_cfg(min_ring_new_pixels=0, pattern_reference_size=999))
    assert p_ref512.n_patches == p_ref999.n_patches
    assert torch.equal(p_ref512._patch_xy, p_ref999._patch_xy)


def test_prune_reduces_patches_and_forward_consistent():
    base = create_patcher("foveated", backbone=create_backbone("vits16"),
                          foveated_config=_fov_cfg(min_ring_new_pixels=0))
    pruned = create_patcher("foveated", backbone=create_backbone("vits16"),
                            foveated_config=_fov_cfg(min_ring_new_pixels=40, pattern_reference_size=512))
    assert 0 < pruned.n_patches < base.n_patches
    # buffers + forward stay mutually consistent at the pruned count.
    assert pruned._patch_rowcol.shape == (pruned.n_patches, 2)
    assert pruned._patch_xy.shape == (pruned.n_patches, 2)
    gpx = PRUNING_FOVEATED_CFG.fixation_size
    image = torch.randn(2, 3, gpx, gpx)
    vp = Viewpoint.full_scene(batch_size=2, device=image.device)
    with torch.inference_mode():
        patches, scene_pos = pruned(image, vp)
    assert patches.shape == (2, pruned.n_patches, pruned.embed_dim)
    assert scene_pos.shape == (2, pruned.n_patches, 2)


def test_prune_decision_uses_reference_not_fixation_size():
    """The kept set is fixed by pattern_reference_size, independent of the
    (possibly changing) deploy fixation_size."""
    a = create_patcher("foveated", backbone=create_backbone("vits16"),
                       foveated_config=_fov_cfg(min_ring_new_pixels=40, pattern_reference_size=512,
                                                fixation_size=128))
    b = create_patcher("foveated", backbone=create_backbone("vits16"),
                       foveated_config=_fov_cfg(min_ring_new_pixels=40, pattern_reference_size=512,
                                                fixation_size=256))
    assert a.n_patches == b.n_patches


def test_prune_with_per_ring_kernel():
    """Pruning composes with per-ring kernels (rings remap, no crash)."""
    pruned = create_patcher("foveated", backbone=create_backbone("vits16"),
                            foveated_config=_fov_cfg(min_ring_new_pixels=40, per_ring_kernel=True))
    gpx = PRUNING_FOVEATED_CFG.fixation_size
    image = torch.randn(1, 3, gpx, gpx)
    vp = Viewpoint.full_scene(batch_size=1, device=image.device)
    with torch.inference_mode():
        patches, _ = pruned(image, vp)
    assert patches.shape[1] == pruned.n_patches
    # one weight slab per surviving ring
    assert pruned.kpe.weight.shape[0] == pruned.kpe.n_rings


def test_to_migrates_fovi_state(foveated_patcher):
    """``.to(device)`` must migrate fovi's plain-attribute tensors.

    fovi stores sampling grids and KNN coords as ordinary Python attributes,
    not ``register_buffer``. Without our ``_apply`` override, those tensors
    stay on the original device and ``grid_sample`` then errors with a
    'tensors on different devices' RuntimeError at the first forward pass.

    We don't need a CUDA device to exercise the migration logic — moving to
    ``torch.float64`` triggers the same ``_apply`` walk and is portable. The
    assertion is that every Tensor attribute on fovi's internal objects ends
    up with the requested dtype.
    """
    import copy
    patcher = copy.deepcopy(foveated_patcher)
    patcher = patcher.to(torch.float64)

    sampler = patcher.retina.sampler

    def _all_tensor_dtypes(obj):
        d = getattr(obj, "__dict__", None) or {}
        buffers = getattr(obj, "_buffers", {})
        parameters = getattr(obj, "_parameters", {})
        for key, val in d.items():
            if key in buffers or key in parameters:
                continue
            if isinstance(val, torch.Tensor):
                yield key, val.dtype

    objs = {
        "retina": patcher.retina,
        "sampler": sampler,
        "sampler.coords": getattr(sampler, "coords", None),
        "kpe": patcher.kpe,
        "kpe.in_coords": getattr(patcher.kpe, "in_coords", None),
        "kpe.out_coords": getattr(patcher.kpe, "out_coords", None),
    }
    for name, obj in objs.items():
        if obj is None:
            continue
        for key, dtype in _all_tensor_dtypes(obj):
            # Integer tensors (e.g. knn_indices) won't migrate to float64 —
            # fn returns the same tensor in that case. Only float/complex
            # tensors should follow the migration.
            orig = dict(_all_tensor_dtypes(obj))  # not strictly needed, just
            # informational; we assert on float dtypes only.
            if dtype.is_floating_point:
                assert dtype == torch.float64, (
                    f"{name}.{key} did not migrate: dtype={dtype}"
                )
