"""CPU tests for the core policy package (unification P3).

Covers: scorer output shapes for BOTH action spaces (safebox = historical,
fixation = the new foveated one), dueling argmax-invariance, local
save_pretrained/from_pretrained round-trip incl. the action_space config key and
a legacy config WITHOUT it, and StateEncoder featurization on a tiny wrapper
(full groups with probe entropy + intrinsic-only for probe-free tasks)."""

import pytest
import torch

from canvit_pytorch import CanViTForSemanticSegmentation
from canvit_pytorch.policy import (
    FEATURE_GROUPS,
    StateEncoder,
    ViewpointScorer,
    candidate_viewpoints,
    feature_channels,
    fixation_candidates,
)
from canvit_pytorch.policy.features import INTRINSIC_GROUPS

_B, _G, _CD = 2, 8, 768  # canvas_dim for vits16 defaults


def _tiny_scorer(**kw) -> ViewpointScorer:
    torch.manual_seed(0)
    defaults = dict(
        canvas_dim=16, width=16, n_scale=2, scales=(0.5, 0.25),
        centers_per_axis=4, block_layers=1,
    )
    defaults.update(kw)
    return ViewpointScorer(**defaults)


def test_candidate_grids() -> None:
    cv = candidate_viewpoints((0.5, 0.25), 16)
    assert cv.shape == (2, 16, 16, 3)
    # safe box: |center| <= 1 - s per scale
    for k, s in enumerate((0.5, 0.25)):
        assert cv[k, ..., :2].abs().max() <= (1 - s) + 1e-6
        assert (cv[k, ..., 2] == s).all()
    fx = fixation_candidates(32)
    assert fx.shape == (1, 32, 32, 3)
    assert fx[..., :2].abs().max() < 1.0  # cell centers, strictly inside the field
    step = 1 / 32
    assert torch.isclose(fx[0, 0, 0, 0], torch.tensor(-1 + step))


def test_scorer_shapes_safebox_and_fixation() -> None:
    feats = torch.randn(_B, feature_channels(16), 32, 32)
    q = _tiny_scorer()(feats)
    assert q.shape == (_B, 2, 4, 4)
    qf = _tiny_scorer(action_space="fixation", n_scale=1, scales=(1.0,), centers_per_axis=32)(feats)
    assert qf.shape == (_B, 1, 32, 32)
    with pytest.raises(AssertionError):
        _tiny_scorer(action_space="fixation", n_scale=2, scales=(0.5, 0.25))


def test_dueling_preserves_argmax() -> None:
    feats = torch.randn(_B, feature_channels(16), 32, 32)
    net = _tiny_scorer(dueling=True)
    net.eval()
    q = net(feats).reshape(_B, -1)
    # V(s) is a per-image constant added to a mean-zero advantage: subtracting the
    # mean recovers the advantage; argmax must equal the advantage argmax.
    adv = q - q.mean(dim=1, keepdim=True)
    assert torch.equal(q.argmax(1), adv.argmax(1))


def test_hub_roundtrip_and_legacy_config(tmp_path) -> None:
    net = _tiny_scorer(dueling=True)
    net.save_pretrained(tmp_path / "ckpt")
    re = ViewpointScorer.from_pretrained(tmp_path / "ckpt")
    assert re.action_space == "safebox"
    feats = torch.randn(_B, feature_channels(16), 32, 32)
    net.eval(), re.eval()
    with torch.no_grad():
        assert torch.allclose(net(feats), re(feats))
    # legacy checkpoints predate action_space: strip the key, must still load
    import json

    cfg_file = tmp_path / "ckpt" / "config.json"
    cfg = json.loads(cfg_file.read_text())
    cfg.pop("action_space")
    cfg_file.write_text(json.dumps(cfg))
    legacy = ViewpointScorer.from_pretrained(tmp_path / "ckpt")
    assert legacy.action_space == "safebox"


def test_state_encoder_full_and_intrinsic() -> None:
    torch.manual_seed(0)
    seg = CanViTForSemanticSegmentation(backbone_name="vits16", model_config={}, num_classes=150)
    seg.eval().requires_grad_(False)
    state = seg.init_state(batch_size=_B, canvas_grid_size=_G)

    enc = StateEncoder(seg, canvas_grid=_G, feature_groups=FEATURE_GROUPS)
    f = enc(state)
    assert f.shape == (_B, feature_channels(seg.canvas_dim), 32, 32)
    assert torch.isfinite(f).all()

    enc_i = StateEncoder(seg, canvas_grid=_G, feature_groups=INTRINSIC_GROUPS)
    assert not enc_i.needs_entropy  # probe-free tasks: no head_logits call
    fi = enc_i(state)
    assert fi.shape == (_B, feature_channels(seg.canvas_dim, INTRINSIC_GROUPS), 32, 32)
    assert torch.isfinite(fi).all()


# --------------------------------------------------------------------------- #
# readout='local': the autoreg_tryout policy head (per-cell 1x1, no U-Net).
# --------------------------------------------------------------------------- #

def test_local_readout_shapes_and_has_no_unet() -> None:
    feats = torch.randn(_B, feature_channels(16), 32, 32)
    net = _tiny_scorer(readout="local")
    assert net(feats).shape == (_B, 2, 4, 4)  # same contract as 'unet'
    assert net.enc is None and net.dec is None
    # The U-Net must not be INSTANTIATED either, or its dead params would ride in the
    # optimizer and the published state_dict.
    assert not any(k.startswith(("enc.", "dec.")) for k in net.state_dict())
    assert sum(p.numel() for p in net.parameters()) < sum(
        p.numel() for p in _tiny_scorer(readout="unet").parameters())


def test_local_readout_score_is_spatially_LOCAL() -> None:
    """The whole point: with 'local' a candidate's score depends only on its own canvas
    cell, where 'unet' pools to a 1x1 bottleneck so every score sees the whole scene.

    Uses centers_per_axis == 32 (the POLICY_GRID the features arrive on) so candidate
    centres land exactly on map pixels and no grid_sample interpolation blurs the test.
    """
    torch.manual_seed(0)
    kw = dict(action_space="fixation", n_scale=1, scales=(1.0,), centers_per_axis=32)
    feats = torch.randn(_B, feature_channels(16), 32, 32)

    local = _tiny_scorer(readout="local", **kw).eval()
    unet = _tiny_scorer(readout="unet", **kw).eval()
    with torch.no_grad():
        q_local, q_unet = local(feats), unet(feats)
        # Perturb ONE canvas cell and see which scores move.
        bumped = feats.clone()
        bumped[:, :, 0, 0] += 25.0
        d_local = (local(bumped) - q_local).abs()[0, 0]
        d_unet = (unet(bumped) - q_unet).abs()[0, 0]

    # 'local': only the co-located candidate moves; everything else is untouched.
    assert d_local[0, 0] > 1e-4
    assert d_local[1:, 1:].max() < 1e-5, d_local[1:, 1:].max()
    # 'unet': the perturbation reaches distant candidates through the bottleneck.
    assert d_unet[-1, -1] > 1e-5, d_unet[-1, -1]


def test_local_readout_round_trips_and_defaults_to_unet(tmp_path) -> None:
    """`readout` must behave like `action_space` did: persisted in config.json, and a
    legacy config WITHOUT the key loads as the historical 'unet'."""
    import json

    net = _tiny_scorer(readout="local")
    net.save_pretrained(tmp_path)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["readout"] == "local"
    back = ViewpointScorer.from_pretrained(tmp_path)
    assert back.readout == "local" and back.enc is None
    feats = torch.randn(_B, feature_channels(16), 32, 32)
    net.eval(), back.eval()
    with torch.no_grad():
        assert torch.allclose(net(feats), back(feats))

    del cfg["readout"]  # a pre-existing published policy
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    legacy = ViewpointScorer.from_pretrained(tmp_path, strict=False)
    assert legacy.readout == "unet"


def test_local_readout_keeps_the_dueling_value_head() -> None:
    """VPG reads its REINFORCE baseline off `vhead`, so it must exist (and stay
    argmax-neutral) with the local readout too."""
    feats = torch.randn(_B, feature_channels(16), 32, 32)
    net = _tiny_scorer(readout="local", dueling=True).eval()
    assert net.vhead is not None
    plain = _tiny_scorer(readout="local", dueling=False).eval()
    with torch.no_grad():
        assert net(feats).shape == plain(feats).shape
