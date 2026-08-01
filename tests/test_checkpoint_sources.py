"""A model loaded from a training ``.pt`` must equal the same model loaded from that
checkpoint's HF export.

That equality is the whole contract of ``from_checkpoint`` / ``canvit_pytorch.model_source``:
the two sources are interchangeable, so development can stay on local ``.pt`` files and
publishing to the HF layout becomes a separate, deliberate step. Every test here builds its
own model and writes both formats to a tmp dir — no network, no Hub, no dependence on
anyone's local runs.

The failure modes pinned down here are all SILENT ones (wrong numbers, not exceptions):
  * nested patcher configs arriving as plain dicts after ``asdict`` flattened them,
  * ``glimpse_grid_size`` (tokens) confused with ``glimpse_size_px`` (pixels),
  * ``pretrain_view_scale`` lost, which makes a foveated model evaluate out of distribution.

Writing a model costs ~90 MB per copy and pytest's tmp dirs often live on tmpfs, so the two
model fixtures are module-scoped (written once) and the unit-level checks below poke the
pure functions directly instead of round-tripping another file.
"""

import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch

from canvit_pytorch.backbone import create_backbone
from canvit_pytorch.model.pretraining.hub import CanViTForPretrainingHFHub
from canvit_pytorch.model.pretraining.impl import (
    CanViTForPretraining,
    CanViTForPretrainingConfig,
    glimpse_grid_size_of,
)
from canvit_pytorch.model_source import (
    is_checkpoint,
    load_pretraining,
    load_segmentation_probe,
    read_backbone_name,
    read_pretrain_metadata,
)
from canvit_pytorch.patcher import FoveatedPatcherConfig
from canvit_pytorch.probes import SegmentationProbe

BACKBONE = "vits16"      # smallest registered backbone; these tests are about I/O, not size
GRID = 8
TEACHER_DIM = 384
VIEW_SCALE = {"mode": "fixed", "fixed_scale": 2.0, "distribution": "uniform",
              "min_scale": 0.5, "max_scale": 1.0}


def _build(cfg: CanViTForPretrainingConfig) -> CanViTForPretraining:
    torch.manual_seed(0)
    return CanViTForPretraining(
        backbone=create_backbone(BACKBONE), cfg=cfg,
        backbone_name=BACKBONE, canvas_patch_grid_sizes=[GRID],
    )


def _write_both(dir_, model, *, view_scale=None):
    """Write the SAME weights in both schemas: the unified trainer's ``.pt``
    (model_state + nested metadata) and the HF layout ``to_hf`` produces."""
    from safetensors.torch import save_file

    md = {"task": "distill", "backbone_name": model.backbone_name,
          "canvas_patch_grid_sizes": model.canvas_patch_grid_sizes,
          "glimpse_grid_size": 8, "patch_stride": None,
          "teacher_name": "dinov3_vitb16", "dataset": "in21k"}
    if view_scale is not None:
        md["training_config_history"] = {"2026-01-01T00:00:00+00:00": {"foveated_scale": view_scale}}
    pt = dir_ / "step-42.pt"
    torch.save({"model_state": model.state_dict(), "step": 42, "metadata": md,
                "model_config": {"task": "distill", "teacher_dim": TEACHER_DIM,
                                 "canvas_grid": GRID, "backbone_name": model.backbone_name,
                                 "canvit": dataclasses.asdict(model.cfg)}}, pt)

    hf = dir_ / "hf"
    hf.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              str(hf / "model.safetensors"))
    (hf / "config.json").write_text(json.dumps({
        "backbone_name": model.backbone_name,
        "model_config": dataclasses.asdict(model.cfg),
        "canvas_patch_grid_sizes": model.canvas_patch_grid_sizes,
        "glimpse_grid_size": 8,
        "metadata": {"teacher_name": "dinov3_vitb16", "dataset": "in21k", "step": 42,
                     "pretrain_view_scale": (None if view_scale is None
                                             else {"patcher_name": model.cfg.patcher_name,
                                                   **view_scale})},
    }))
    return pt, hf


@pytest.fixture(scope="module")
def uniform_pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("uniform")
    return _write_both(d, _build(CanViTForPretrainingConfig(teacher_dim=TEACHER_DIM)))


@pytest.fixture(scope="module")
def foveated_pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("foveated")
    cfg = CanViTForPretrainingConfig(
        teacher_dim=TEACHER_DIM, patcher_name="foveated",
        foveated_patcher=FoveatedPatcherConfig(fov=35.0, cmf_a=0.5))
    return _write_both(d, _build(cfg), view_scale=VIEW_SCALE)


def _assert_same_model(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb), set(sa) ^ set(sb)
    bad = [k for k in sa if not torch.equal(sa[k], sb[k])]
    assert not bad, f"tensors differ: {bad[:5]}"
    assert a.cfg == b.cfg
    assert a.backbone_name == b.backbone_name
    assert a.canvas_patch_grid_sizes == b.canvas_patch_grid_sizes
    # The attribute contract downstream consumers read via getattr. glimpse_size_px must
    # stay None on BOTH: consumers crop the glimpse themselves from glimpse_grid_size, so a
    # model that also cropped internally would crop twice.
    assert a.glimpse_grid_size == b.glimpse_grid_size
    assert a.glimpse_size_px is None and b.glimpse_size_px is None


def test_uniform_pt_equals_hf(uniform_pair):
    pt, hf = uniform_pair
    _assert_same_model(load_pretraining(pt), load_pretraining(hf))


def test_foveated_nested_config_survives_both_paths(foveated_pair):
    """The nesting bug: ``asdict`` flattens ``foveated_patcher`` to a dict, and a shallow
    rebuild then fails deep in the patcher (``'dict' object has no attribute
    'hidden_dims_patch_embed'``). Non-default values, so a silent fallback to defaults fails."""
    pt, hf = foveated_pair
    a, b = load_pretraining(pt), load_pretraining(hf)
    _assert_same_model(a, b)
    assert isinstance(a.cfg.foveated_patcher, FoveatedPatcherConfig)
    assert (a.cfg.foveated_patcher.fov, a.cfg.foveated_patcher.cmf_a) == (35.0, 0.5)


def test_hf_gives_the_subclass_and_pt_the_base(uniform_pair):
    """Interchangeable for consumers, but not the same class — pin it so a future change
    that starts returning something else is deliberate."""
    pt, hf = uniform_pair
    assert isinstance(load_pretraining(hf), CanViTForPretrainingHFHub)
    assert isinstance(load_pretraining(pt), CanViTForPretraining)


def test_backbone_name_readable_from_both(uniform_pair):
    """in1k finetune picks its DINOv3 probe from this WITHOUT loading the model, so it has
    to work for a .pt or passing one would fail before any weights are touched."""
    pt, hf = uniform_pair
    assert read_backbone_name(pt) == read_backbone_name(hf) == BACKBONE


def test_view_scale_metadata_matches_across_sources(foveated_pair):
    """The silent one: lose ``pretrain_view_scale`` and a foveated model is evaluated at a
    scale it never saw — no error, just a metric that decays with every glimpse."""
    pt, hf = foveated_pair
    a = read_pretrain_metadata(pt)["pretrain_view_scale"]
    b = read_pretrain_metadata(hf)["pretrain_view_scale"]
    assert a == b
    assert a["fixed_scale"] == 2.0 and a["patcher_name"] == "foveated"


def test_uniform_model_reports_no_view_scale(uniform_pair):
    """View scale is not a uniform model's out-of-distribution axis, so it must stay None —
    callers must not read that as "scale 1.0"."""
    pt, _ = uniform_pair
    assert read_pretrain_metadata(pt)["pretrain_view_scale"] is None


# --- unit-level: the units trap, poked directly so no 90 MB file is written ------------
@pytest.mark.parametrize("grid", [6, 8, 12])
def test_glimpse_grid_size_is_tokens(grid):
    """``glimpse_grid_size`` is TOKENS. Grid 8 with a 16px patch makes several wrong
    formulas agree, so other grids are tested too."""
    bb = SimpleNamespace(patch_size_px=16, patch_stride_px=16)
    assert glimpse_grid_size_of({"glimpse_grid_size": grid}, bb) == grid


@pytest.mark.parametrize(("stride", "tokens"), [(16, 6), (8, 6), (16, 12)])
def test_legacy_pixel_field_inverts_through_the_stride(stride, tokens):
    """A legacy checkpoint records PIXELS. Inverting needs the STRIDE, not the patch size:
    the window is ``(tokens-1)*stride + patch``, equal to ``tokens*patch`` only when they
    match. The stride=8 case is the one a patch-size-based inverse gets wrong."""
    bb = SimpleNamespace(patch_size_px=16, patch_stride_px=stride)
    px = (tokens - 1) * stride + 16
    assert glimpse_grid_size_of({"glimpse_size_px": px}, bb) == tokens


def test_glimpse_defaults_to_eight_when_unrecorded():
    bb = SimpleNamespace(patch_size_px=16, patch_stride_px=16)
    assert glimpse_grid_size_of({}, bb) == 8


# --- probe + error paths (cheap: no CanViT built) --------------------------------------
def test_probe_pt_equals_hf(tmp_path):
    torch.manual_seed(0)
    probe = SegmentationProbe(embed_dim=64, num_classes=150, dropout=0.1, use_ln=True)
    torch.save({"metadata": {"task": "ade20k"},
                "model_state": {f"head.{k}": v for k, v in probe.state_dict().items()}
                | {"canvit.dummy": torch.zeros(1)}},
               tmp_path / "best.pt")
    probe.save_pretrained(tmp_path / "probe-hf")
    a = load_segmentation_probe(tmp_path / "best.pt")
    b = load_segmentation_probe(tmp_path / "probe-hf")
    assert (a.embed_dim, a.num_classes, a.use_ln) == (b.embed_dim, b.num_classes, b.use_ln)
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_downstream_checkpoint_cannot_pose_as_pretraining(tmp_path):
    """An ade20k/in1k checkpoint records a ``model_repo`` POINTER, not its architecture, so
    it must fail with a message that says so rather than half-loading."""
    torch.save({"model_state": {}, "metadata": {"task": "ade20k"},
                "model_config": {"task": "ade20k", "model_repo": "somewhere"}},
               tmp_path / "ade.pt")
    with pytest.raises(KeyError, match="only a pretraining"):
        load_pretraining(tmp_path / "ade.pt")


def test_missing_pt_fails_loudly_instead_of_becoming_a_hub_id(tmp_path):
    """A typo'd ``.pt`` path must not fall through to a Hub lookup — that turns a local
    mistake into a confusing network error about a repo nobody meant to name."""
    with pytest.raises(FileNotFoundError):
        is_checkpoint(tmp_path / "nope.pt")
    assert is_checkpoint("canvit/some-hub-repo") is False
