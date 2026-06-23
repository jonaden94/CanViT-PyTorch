"""Sanity tests for the square patcher.

Covers, across the three sampling-pattern methods (``fovi`` /
``fovi_regularized`` / ``strided``):
  - Output shapes (patches, scene_positions) and ``n_patches``.
  - Position-frame correctness: full-scene viewpoint -> scene_pos == patch
    rowcol; off-center viewpoint shifts scene_pos by viewpoint.centers (same
    contract as the foveated patcher).
  - Optional FiLM conditioning and learned padding are no-ops at
    init (output identical to the unconditioned / zero-padding patcher).
  - CoordConv conditioning is rejected.
  - ``.to(dtype)`` migrates the cached buffers.
  - Drop-in integration with CanViT (uniform-mode regression unaffected).

Skipped when the optional ``fovi`` dependency is not installed.
"""

import copy
import importlib.util

import pytest
import torch

from canvit_pytorch import (
    CanViT,
    CanViTConfig,
    SquarePatcherConfig,
    Viewpoint,
    create_backbone,
    create_patcher,
)
from canvit_pytorch.patcher.conditioning import FiLMConfig, PatchConditioningConfig
from canvit_pytorch.patcher.square import SquarePatcher

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fovi") is None,
    reason="fovi optional dependency not installed",
)

# Compact configs (small fixation window so peripheral square samples spill out
# of field and exercise the padding path). The fovi-derived params match the
# foveated test's SMALL config so the geometry source is cheap to build.
FOVI_KW = dict(
    fov=16.0, cmf_a=2.785765, resolution=32, style="isotropic",
    cart_patch_size=8, sample_cortex=True,
    # Image size fed in these tests == pattern_reference_size, so a scale=1
    # viewpoint deploys the pattern at its build window.
    pattern_reference_size=128,
)
CFGS = {
    "fovi": SquarePatcherConfig(method="fovi", **FOVI_KW),
    "fovi_regularized": SquarePatcherConfig(method="fovi_regularized", **FOVI_KW),
    # Pattern built at a 64-px reference < the strided receptive field (~103 px),
    # so peripheral samples fall outside the window and exercise the padding path.
    "strided": SquarePatcherConfig(
        method="strided", grid_size_fovea=2, patch_size=6,
        edge_length_multipliers=[2, 6], pattern_reference_size=64,
    ),
}
EMBED_DIM = 32


def _patcher(cfg: SquarePatcherConfig) -> SquarePatcher:
    return SquarePatcher(cfg, embed_dim=EMBED_DIM, device="cpu")


@pytest.mark.parametrize("method", list(CFGS))
def test_buffers_and_npatches(method):
    p = _patcher(CFGS[method])
    assert p.n_patches > 0
    assert p._patch_rowcol.shape == (p.n_patches, 2)
    assert p._pad_mask.shape == (p.n_patches, p._k)
    assert p._sample_colrow.shape == (1, 1, p.n_patches * p._k, 2)
    assert p._patch_rowcol.dtype == torch.float32


@pytest.mark.parametrize("method", list(CFGS))
def test_forward_shapes(method):
    cfg = CFGS[method]
    p = _patcher(cfg)
    B, gpx = 2, cfg.pattern_reference_size
    image = torch.randn(B, 3, gpx, gpx)
    vp = Viewpoint.full_scene(batch_size=B, device=image.device)
    with torch.inference_mode():
        patches, scene_pos = p(image, vp)
    assert patches.shape == (B, p.n_patches, EMBED_DIM)
    assert scene_pos.shape == (B, p.n_patches, 2)
    assert scene_pos.dtype == torch.float32


@pytest.mark.parametrize("method", list(CFGS))
def test_scene_positions_full_scene(method):
    """Full-scene viewpoint (centers=0) -> scene_pos == patch rowcol."""
    cfg = CFGS[method]
    p = _patcher(cfg)
    B, gpx = 3, cfg.pattern_reference_size
    image = torch.randn(B, 3, gpx, gpx)
    vp = Viewpoint.full_scene(batch_size=B, device=image.device)
    with torch.inference_mode():
        _, scene_pos = p(image, vp)
    for b in range(B):
        assert torch.allclose(scene_pos[b], p._patch_rowcol, atol=1e-5)


@pytest.mark.parametrize("method", list(CFGS))
def test_scene_positions_scaled_and_translated(method):
    """scene_pos = centers + scale * rowcol (scale is now honored)."""
    cfg = CFGS[method]
    p = _patcher(cfg)
    gpx = cfg.pattern_reference_size
    image = torch.randn(1, 3, gpx, gpx)
    centers = torch.tensor([[0.3, -0.2]], dtype=torch.float32)
    for s in (0.4, 1.0, 1.5):
        vp = Viewpoint(centers=centers, scales=torch.tensor([s]))
        with torch.inference_mode():
            _, scene_pos = p(image, vp)
        expected = centers.view(1, 1, 2) + s * p._patch_rowcol.view(1, -1, 2)
        assert torch.allclose(scene_pos, expected, atol=1e-5), f"{method} scale={s}"


def test_foveal_patch_near_origin():
    """The fovi-square patch nearest the fixation sits at ~origin in rowcol."""
    p = _patcher(CFGS["fovi"])
    rowcol = p._patch_rowcol
    foveal = rowcol[int((rowcol ** 2).sum(-1).argmin())]
    assert foveal.abs().max() < 1e-2, f"Expected foveal patch near origin; got {foveal}"


def test_strided_has_padding():
    """A small fixation window makes peripheral strided samples out-of-field."""
    p = _patcher(CFGS["strided"])
    assert int(p._pad_mask.sum()) > 0


@pytest.mark.parametrize("encoding", ["fourier", "sinusoidal"])
def test_conditioning_film_noop_at_init(encoding):
    """FiLM conditioning is a no-op at init (zero-init final layer), for both
    the Fourier and sinusoidal position encoders."""
    base = _patcher(CFGS["fovi"])
    cfg = copy.deepcopy(CFGS["fovi"])
    cfg.conditioning = PatchConditioningConfig(mode="film", film=FiLMConfig(encoding=encoding))
    cond = SquarePatcher(cfg, embed_dim=EMBED_DIM, device="cpu")
    # Share the (randomly-initialized) embed + head so only conditioning differs.
    cond.embed.load_state_dict(base.embed.state_dict())
    cond.embed_head.load_state_dict(base.embed_head.state_dict())
    gpx = cfg.pattern_reference_size
    image = torch.randn(1, 3, gpx, gpx)
    vp = Viewpoint.full_scene(batch_size=1, device=image.device)
    with torch.inference_mode():
        a, _ = base(image, vp)
        b, _ = cond(image, vp)
    assert torch.allclose(a, b, atol=1e-5)


def test_learned_padding_noop_at_init():
    """Learned padding (zero-init) == zero padding at init."""
    base = _patcher(CFGS["strided"])
    cfg = copy.deepcopy(CFGS["strided"])
    cfg.padding = "learned"
    learned = SquarePatcher(cfg, embed_dim=EMBED_DIM, device="cpu")
    learned.embed.load_state_dict(base.embed.state_dict())
    learned.embed_head.load_state_dict(base.embed_head.state_dict())
    assert learned.pad_value is not None and float(learned.pad_value.detach().abs().sum()) == 0.0
    gpx = cfg.pattern_reference_size
    image = torch.randn(1, 3, gpx, gpx)
    vp = Viewpoint.full_scene(batch_size=1, device=image.device)
    with torch.inference_mode():
        a, _ = base(image, vp)
        b, _ = learned(image, vp)
    assert torch.allclose(a, b, atol=1e-5)


def test_pad_mask_applied_under_zero_padding():
    """The pad mask must blank masked slots even with padding='zero' (default),
    independent of grid_sample's out-of-image zeroing. Uses a window (64) smaller
    than the image (128) so masked samples land *inside* the image — there,
    zeroing genuinely changes the result."""
    import torch.nn.functional as F
    from fovi.sensing.coords import transform_sampling_grid

    p = _patcher(CFGS["strided"])  # pattern built at pattern_reference_size=64
    assert int(p.pad_mask.sum()) > 0
    H, s = 128, 0.5  # scale=0.5 -> 64-px deploy window < image, so masked
    img = torch.randn(1, 3, H, H)   # samples land in-image; mask blanks them
    vp = Viewpoint(centers=torch.zeros(1, 2), scales=torch.full((1,), s))
    B, C, P, K = 1, 3, p.n_patches, p._k

    def _embed(samp):
        x = samp.permute(0, 2, 1, 3).reshape(B, P, C * K)
        x = p.embed(x)
        x = p.conditioner.modulate_kpe_output(x)
        return p.embed_head(x)

    with torch.inference_mode():
        out, _ = p(img, vp)
        fix_loc = (vp.centers.float() + 1) * 0.5
        fst = torch.tensor([[s * H, s * H]], dtype=torch.float32)  # fix_size = scale * H
        grid = transform_sampling_grid(p._sample_colrow, fix_loc, fst, (H, H))
        raw = F.grid_sample(img.float(), grid, mode="bilinear", padding_mode="zeros",
                            align_corners=False)[:, :, 0, :].reshape(B, C, P, K)
        zeroed = raw.masked_fill(p.pad_mask.view(1, 1, P, K), 0.0)

    # forward == sample-then-zero-pad-then-embed
    assert torch.allclose(out, _embed(zeroed), atol=1e-5)
    # and NOT zeroing the mask gives a different result -> the mask is active & matters
    assert not torch.allclose(out, _embed(raw), atol=1e-5)


def test_coordconv_rejected():
    cfg = copy.deepcopy(CFGS["fovi"])
    cfg.conditioning = PatchConditioningConfig(mode="coordconv")
    with pytest.raises(AssertionError, match="coordconv"):
        SquarePatcher(cfg, embed_dim=EMBED_DIM, device="cpu")


def test_to_migrates_buffers():
    p = _patcher(CFGS["fovi"]).to(torch.float64)
    assert p._sample_colrow.dtype == torch.float64
    assert p._patch_rowcol.dtype == torch.float64
    assert p._pad_mask.dtype == torch.bool  # non-float unchanged


@pytest.fixture(scope="module")
def backbone():
    return create_backbone("vits16")


def test_create_patcher_square(backbone):
    p = create_patcher("square", backbone=backbone, square_config=CFGS["strided"])
    assert isinstance(p, SquarePatcher)
    assert p.embed_dim == backbone.embed_dim


def test_canvit_end_to_end_square(backbone):
    cfg = CanViTConfig(patcher_name="square", square_patcher=CFGS["strided"])
    model = CanViT(backbone=backbone, cfg=cfg).eval()
    B, gpx, canvas_grid = 2, CFGS["strided"].pattern_reference_size, 16
    image = torch.randn(B, 3, gpx, gpx)
    vp = Viewpoint.full_scene(batch_size=B, device=image.device)
    state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)
    with torch.inference_mode():
        out = model(image=image, state=state, viewpoint=vp)
    assert out.local_patches.shape[0] == B
    assert out.local_patches.shape[1] == model.patcher.n_patches
    assert out.local_patches.shape[2] == backbone.embed_dim


def test_uniform_unaffected(backbone):
    """Uniform mode still introduces no patcher.* state_dict keys."""
    model = CanViT(backbone=backbone, cfg=CanViTConfig()).eval()
    assert [k for k in model.state_dict() if k.startswith("patcher")] == []


# --------------------------------------------------------------------------- #
# Ring pruning (min_ring_new_pixels)
# --------------------------------------------------------------------------- #

# Strongly-foveated geometry source so the inner rings oversample below the
# 512-px reference grid and get pruned (mirrors the foveated patcher test).
PRUNING_KW = dict(
    method="fovi", fov=180.0, cmf_a=0.5, resolution=36, style="isotropic",
    cart_patch_size=6, sample_cortex=True,
    force_patches_less_than_matched=False,
)


def test_prune_disabled_is_noop_square():
    """min_ring_new_pixels=0 keeps all patches regardless of pattern_reference_size."""
    p_a = _patcher(SquarePatcherConfig(min_ring_new_pixels=0, pattern_reference_size=512, **PRUNING_KW))
    p_b = _patcher(SquarePatcherConfig(min_ring_new_pixels=0, pattern_reference_size=999, **PRUNING_KW))
    assert p_a.n_patches == p_b.n_patches
    assert torch.equal(p_a._patch_xy, p_b._patch_xy)


def test_prune_reduces_patches_square():
    base = _patcher(SquarePatcherConfig(min_ring_new_pixels=0, **PRUNING_KW))
    pruned = _patcher(SquarePatcherConfig(min_ring_new_pixels=40, pattern_reference_size=512, **PRUNING_KW))
    assert 0 < pruned.n_patches < base.n_patches
    # Every per-patch buffer is subset consistently to the pruned count.
    assert pruned._patch_rowcol.shape == (pruned.n_patches, 2)
    assert pruned._patch_xy.shape == (pruned.n_patches, 2)
    assert pruned._pad_mask.shape == (pruned.n_patches, pruned._k)
    assert pruned._ring_idx.shape == (pruned.n_patches,)
    assert pruned._sample_colrow.shape == (1, 1, pruned.n_patches * pruned._k, 2)
    gpx = 128  # any image size; scale=1 -> full image
    image = torch.randn(2, 3, gpx, gpx)
    vp = Viewpoint.full_scene(batch_size=2, device=image.device)
    with torch.inference_mode():
        patches, scene_pos = pruned(image, vp)
    assert patches.shape == (2, pruned.n_patches, EMBED_DIM)
    assert scene_pos.shape == (2, pruned.n_patches, 2)


# --------------------------------------------------------------------------- #
# Strided add_to_patch_size (decouple sampling density from patch geometry)
# --------------------------------------------------------------------------- #

def _strided_cfg(add):
    return SquarePatcherConfig(
        method="strided", grid_size_fovea=2, patch_size=6,
        edge_length_multipliers=[2, 6], pattern_reference_size=64,
        add_to_patch_size=add,
    )


@pytest.mark.parametrize("add", [-2, 0, 4])
def test_strided_add_to_patch_size_geometry_invariant(add):
    """add_to_patch_size changes K but NOT patch geometry: same n_patches, same
    per-patch centers, K = (patch_size + add)**2, side = patch_size + add."""
    base = _patcher(_strided_cfg(0))
    p = _patcher(_strided_cfg(add))
    m = 6 + add
    assert p.n_patches == base.n_patches
    assert p._k == m * m
    assert p._side == m
    # Patch centers (-> scene_pos / RoPE) are independent of the sample count.
    assert torch.allclose(p._patch_rowcol, base._patch_rowcol, atol=1e-6)
    assert torch.allclose(p._patch_xy, base._patch_xy, atol=1e-6)


@pytest.mark.parametrize("add", [-2, 0, 4])
def test_strided_add_scene_pos_invariant_and_shapes(add):
    """End-to-end: forward shapes track K, and scene_pos is invariant to add
    (geometry unchanged) for both full-scene and off-center viewpoints."""
    base = _patcher(_strided_cfg(0))
    p = _patcher(_strided_cfg(add))
    gpx = 64
    image = torch.randn(2, 3, gpx, gpx)
    for vp in (
        Viewpoint.full_scene(batch_size=2, device=image.device),
        Viewpoint(centers=torch.tensor([[0.3, -0.2], [0.0, 0.0]]),
                  scales=torch.tensor([0.5, 1.0])),
    ):
        with torch.inference_mode():
            patches_b, sp_b = base(image, vp)
            patches_p, sp_p = p(image, vp)
        assert patches_p.shape == (2, p.n_patches, EMBED_DIM)
        assert torch.allclose(sp_p, sp_b, atol=1e-5)


def test_strided_samples_centered_in_patch():
    """Option B: samples are perfectly centered in each patch (the offset multiset
    is symmetric about the center) — for every ring, incl. even strides whose native
    integer layout sat half a pixel off-center."""
    pat = _patcher(_strided_cfg(0))
    pos = pat.sample_positions_xy()                       # [P, K, 2]
    off = pos - pat._patch_xy[:, None, :]                 # center-relative offsets
    for pidx in range(off.shape[0]):
        o = off[pidx].reshape(-1)
        a, _ = torch.sort(o)
        b, _ = torch.sort(-o)
        assert torch.allclose(a, b, atol=1e-5), f"patch {pidx} not centered"


def test_add_to_patch_size_min_two():
    """patch_size + add_to_patch_size must be >= 2."""
    with pytest.raises(ValueError, match=">= 2"):
        _patcher(_strided_cfg(-5))   # 6 + (-5) = 1


@pytest.mark.parametrize("method", ["fovi", "fovi_regularized"])
def test_add_to_patch_size_rejected_non_strided(method):
    cfg = copy.deepcopy(CFGS[method])
    cfg.add_to_patch_size = 3
    with pytest.raises(ValueError, match="only supported for method='strided'"):
        SquarePatcher(cfg, embed_dim=EMBED_DIM, device="cpu")
