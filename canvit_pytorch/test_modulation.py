"""Tests for per-token adaLN-style trunk / cross-attn modulation.

Uses the uniform patcher so these need no fovi (never skipped). Covers the
generator in isolation and the end-to-end no-op-at-init property of the
``*_modulate`` backbone.
"""

import importlib.util

import pytest
import torch

from canvit_pytorch import CanViT, CanViTConfig, FoveatedPatcherConfig, Viewpoint, create_backbone
from canvit_pytorch.modulation import Modulation, TokenModulation, ViTModulationConfig

EMBED_DIM = 384  # vits16
GLIMPSE = 64     # 64 / patch16 = 4x4 = 16 patches
CANVAS_GRID = 8


# --------------------------------------------------------------------------- #
# Generator in isolation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("encoding", ["fourier", "sinusoidal"])
def test_generator_shapes_and_zero_at_init(encoding):
    n_blocks, n_prefix, n_read, n_write, n_patches = 12, 7, 3, 3, 16
    tm = TokenModulation(
        ViTModulationConfig(encoding=encoding, modulate_cross_attn=True),
        embed_dim=EMBED_DIM, n_blocks=n_blocks, n_prefix=n_prefix, n_read=n_read, n_write=n_write,
    )
    m = tm(torch.randn(n_patches, 2))
    n_tokens = n_prefix + n_patches
    assert isinstance(m, Modulation)
    assert len(m.block) == n_blocks and all(t.shape == (n_tokens, 6 * EMBED_DIM) for t in m.block)
    assert len(m.read) == n_read and all(t.shape == (n_tokens, 3 * EMBED_DIM) for t in m.read)
    assert len(m.write) == n_write and all(t.shape == (n_tokens, 2 * EMBED_DIM) for t in m.write)
    # zero-init heads -> raw modulation is exactly 0 at init.
    assert all((t == 0).all() for t in (*m.block, *m.read, *m.write))


def test_generator_cross_attn_toggle_and_base_dim():
    tm_off = TokenModulation(ViTModulationConfig(modulate_cross_attn=False),
                             embed_dim=EMBED_DIM, n_blocks=4, n_prefix=3, n_read=2, n_write=2)
    m = tm_off(torch.randn(8, 2))
    assert len(m.block) == 4 and m.read == [] and m.write == []
    # base_dim None -> embed_dim; explicit value honored.
    assert tm_off.base_mlp[-1].out_features == EMBED_DIM
    tm_small = TokenModulation(ViTModulationConfig(base_dim=64),
                               embed_dim=EMBED_DIM, n_blocks=1, n_prefix=1, n_read=0, n_write=0)
    assert tm_small.base_mlp[-1].out_features == 64
    assert tm_small.block_heads[0][-1].out_features == 6 * EMBED_DIM


# --------------------------------------------------------------------------- #
# End-to-end: no-op at init vs the standard backbone
# --------------------------------------------------------------------------- #


def _img_vp(batch=2):
    img = torch.randn(batch, 3, GLIMPSE, GLIMPSE)
    return img, Viewpoint.full_scene(batch_size=batch, device=img.device)


def _run(model, img, vp):
    state = model.init_state(batch_size=img.shape[0], canvas_grid_size=CANVAS_GRID)
    with torch.inference_mode():
        return model(image=img, state=state, viewpoint=vp, canvas_grid_size=CANVAS_GRID)


def _make_pair(*, encoding="fourier", modulate_cross_attn=False):
    std = CanViT(backbone=create_backbone("vits16"), cfg=CanViTConfig(), glimpse_size_px=GLIMPSE).eval()
    mcfg = CanViTConfig(
        vit_modulation=ViTModulationConfig(
            enabled=True, encoding=encoding, modulate_cross_attn=modulate_cross_attn
        )
    )
    mod = CanViT(backbone=create_backbone("vits16_modulate"), cfg=mcfg, glimpse_size_px=GLIMPSE).eval()
    # Share the weights present in both; the modulate model's generator stays
    # zero-init and the standard model's norm-affine / LayerScale are dropped.
    res = mod.load_state_dict(std.state_dict(), strict=False)
    assert all("token_modulation" in k for k in res.missing_keys)
    assert all(any(s in k for s in ("norm1", "norm2", ".ls1.", ".ls2.")) for k in res.unexpected_keys)
    return std, mod


@pytest.mark.parametrize("encoding", ["fourier", "sinusoidal"])
@pytest.mark.parametrize("cross_attn", [False, True])
def test_modulation_noop_at_init(encoding, cross_attn):
    """Modulate model with shared weights == standard model at init (bit-identical)."""
    std, mod = _make_pair(encoding=encoding, modulate_cross_attn=cross_attn)
    img, vp = _img_vp()
    o_std, o_mod = _run(std, img, vp), _run(mod, img, vp)
    assert torch.allclose(o_std.local_patches, o_mod.local_patches, atol=1e-5)
    assert torch.allclose(o_std.state.canvas, o_mod.state.canvas, atol=1e-5)


def test_modulation_changes_output_when_trained():
    std, mod = _make_pair(modulate_cross_attn=True)
    img, vp = _img_vp()
    base = _run(mod, img, vp)
    with torch.no_grad():
        mod.token_modulation.block_heads[0][-1].weight.normal_(0, 0.1)
    after = _run(mod, img, vp)
    assert not torch.allclose(base.local_patches, after.local_patches, atol=1e-4)
    assert torch.isfinite(after.local_patches).all()


def test_forward_reduce_hoist_matches_forward():
    """The hoisted (forward_reduce) path must agree with per-call forward."""
    _, mod = _make_pair(modulate_cross_attn=True)
    img, vp = _img_vp()
    # one-step forward_reduce
    acc, _ = mod.forward_reduce(
        image=img, viewpoints=[vp], canvas_grid_size=CANVAS_GRID,
        init_fn=lambda s: None, step_fn=lambda acc, out, vp: out,
    )
    direct = _run(mod, img, vp)
    assert torch.allclose(acc.local_patches, direct.local_patches, atol=1e-6)


# --------------------------------------------------------------------------- #
# Consistency between backbone variant and modulation config
# --------------------------------------------------------------------------- #


def test_modulate_backbone_requires_config():
    with pytest.raises(AssertionError):
        CanViT(backbone=create_backbone("vits16_modulate"), cfg=CanViTConfig(), glimpse_size_px=GLIMPSE)


def test_config_requires_modulate_backbone():
    cfg = CanViTConfig(vit_modulation=ViTModulationConfig(enabled=True))
    with pytest.raises(AssertionError):
        CanViT(backbone=create_backbone("vits16"), cfg=cfg, glimpse_size_px=GLIMPSE)


def test_standard_model_unchanged_no_modulation_keys():
    """Standard (non-modulate) model has no token_modulation params."""
    model = CanViT(backbone=create_backbone("vits16"), cfg=CanViTConfig(), glimpse_size_px=GLIMPSE)
    assert model.token_modulation is None
    assert not any("token_modulation" in k for k in model.state_dict())


@pytest.mark.skipif(importlib.util.find_spec("fovi") is None, reason="fovi not installed")
def test_modulation_noop_at_init_foveated():
    """Same no-op-at-init guarantee with the foveated patcher (the real use case)."""
    fov = FoveatedPatcherConfig(
        fov=16.0, cmf_a=2.785765, resolution=32, fixation_size=128,
        style="isotropic", sampler="grid_nn", cart_patch_size=8, sample_cortex=True,
    )
    std = CanViT(backbone=create_backbone("vits16"),
                 cfg=CanViTConfig(patcher_name="foveated", foveated_patcher=fov)).eval()
    mcfg = CanViTConfig(patcher_name="foveated", foveated_patcher=fov,
                        vit_modulation=ViTModulationConfig(enabled=True, modulate_cross_attn=True))
    mod = CanViT(backbone=create_backbone("vits16_modulate"), cfg=mcfg).eval()
    mod.load_state_dict(std.state_dict(), strict=False)

    gpx = fov.fixation_size
    img = torch.randn(2, 3, gpx, gpx)
    vp = Viewpoint.full_scene(batch_size=2, device=img.device)

    def run(m):
        st = m.init_state(batch_size=2, canvas_grid_size=CANVAS_GRID)
        with torch.inference_mode():
            return m(image=img, state=st, viewpoint=vp, canvas_grid_size=CANVAS_GRID)

    o_std, o_mod = run(std), run(mod)
    assert torch.allclose(o_std.local_patches, o_mod.local_patches, atol=1e-5)
    assert torch.allclose(o_std.state.canvas, o_mod.state.canvas, atol=1e-5)
