"""HuggingFace Hub integration for CanViTForPretraining."""

import dataclasses
import json
import logging
import tempfile
import typing
from dataclasses import asdict
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, PyTorchModelHubMixin
from safetensors.torch import save_file

from canvit_pytorch.backbone import create_backbone
from canvit_pytorch.model.hub_mixin import SafeHubMixin
from canvit_pytorch.modulation import ViTModulationConfig
from canvit_pytorch.patcher import FoveatedPatcherConfig, SquarePatcherConfig

from ..impl import CanViTForPretraining, CanViTForPretrainingConfig


def _coerce(tp, value):
    """Recursively rebuild a dataclass of type ``tp`` from ``value`` when
    ``value`` is a dict (as produced by ``asdict`` at save time), coercing any
    nested dataclass-typed fields at any depth. Non-dict values and
    non-dataclass targets pass through unchanged.

    Faithful-or-loud-fail by design — it never fabricates values:
      * EVERY key in ``value`` is passed to the dataclass constructor, so a key
        this code does not know (e.g. a config field added to the model *after*
        this loader was written, loaded without updating the loader) raises a
        loud ``TypeError`` rather than being silently dropped. This is what
        prevents a future, un-updated eval from silently evaluating a model that
        diverges from pretraining.
      * A field present in the dataclass but ABSENT from ``value`` takes the
        dataclass default (a checkpoint predating that field). This is the only
        place defaults enter, so new fields MUST default to backward-compatible
        behavior — the one invariant no loader can enforce for you (the
        strict state_dict load is the backstop for anything affecting weights)."""
    if not isinstance(value, dict):
        return value
    if typing.get_origin(tp) is not None:  # Optional[X] / Union[...] -> the dataclass member
        tp = next((a for a in typing.get_args(tp) if dataclasses.is_dataclass(a)), None)
    if tp is None or not dataclasses.is_dataclass(tp):
        return value
    try:
        hints = typing.get_type_hints(tp)
    except Exception:  # noqa: BLE001 — unresolved annotations: fall back to raw field types
        hints = {f.name: f.type for f in dataclasses.fields(tp)}
    # Pass ALL keys (recursing into known dataclass-typed fields); an unknown key
    # reaches tp(**...) and raises TypeError — never silently dropped.
    return tp(**{k: (_coerce(hints[k], v) if k in hints else v) for k, v in value.items()})

log = logging.getLogger(__name__)

# Teacher full name → hub shortname
TEACHER_SHORT = {
    "dinov3_vits16": "dv3s16",
    "dinov3_vitb16": "dv3b16",
    "dinov3_vitl16": "dv3l16",
}


def teacher_shortname(teacher_name: str) -> str:
    assert teacher_name in TEACHER_SHORT, f"Unknown teacher {teacher_name!r}, known: {sorted(TEACHER_SHORT)}"
    return TEACHER_SHORT[teacher_name]


def make_repo_id(
    *, owner: str, backbone_name: str, glimpse_size_px: int, scene_size_px: int,
    dataset: str, teacher_name: str, enable_vpe: bool = True,
    canvas_update_mode: str = "additive",
) -> str:
    """Compute HF Hub repo ID from checkpoint metadata. Single source of truth."""
    assert backbone_name.startswith("vit"), f"Expected vit* backbone, got {backbone_name}"
    short = backbone_name.removeprefix("vit")
    update_tag = {"additive": "-add", "convex": "-cvx"}[canvas_update_mode]
    variant = update_tag
    if enable_vpe:
        variant += "-vpe"
    return f"{owner}/canvit{short}{variant}-pretrain-g{glimpse_size_px}px-s{scene_size_px}px-{dataset}-{teacher_shortname(teacher_name)}"


def upload_to_hf(
    model: CanViTForPretraining,
    repo_id: str,
    *,
    private: bool = True,
    extra_metadata: dict | None = None,
) -> str:
    """Upload model to HuggingFace Hub under the given repo_id. Returns repo_id.

    extra_metadata is merged into config.json alongside model_config,
    backbone_name, and canvas_patch_grid_sizes.
    """
    assert model.backbone_name is not None, "backbone_name not set - load via from_checkpoint"

    cfg = model.cfg
    assert isinstance(cfg, CanViTForPretrainingConfig)
    config: dict = {
        "backbone_name": model.backbone_name,
        "model_config": asdict(cfg),
        "canvas_patch_grid_sizes": model.canvas_patch_grid_sizes,
    }
    # Persist patch_stride ONLY for overlapping-patch models (stride < patch);
    # it lives outside model_config. Omitted otherwise so non-overlap configs
    # stay byte-for-byte identical and load with create_backbone's default.
    if model.backbone.patch_stride_px != model.backbone.patch_size_px:
        config["patch_stride"] = model.backbone.patch_stride_px
    if extra_metadata is not None:
        config["metadata"] = extra_metadata

    log.info("Pushing to %s (private=%s)", repo_id, private)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "config.json").write_text(json.dumps(config, indent=2, default=str))
        save_file(model.state_dict(), tmppath / "model.safetensors")

        api = HfApi()
        api.create_repo(repo_id, private=private, exist_ok=True)
        api.upload_folder(folder_path=tmpdir, repo_id=repo_id)

    log.info("Pushed to https://huggingface.co/%s", repo_id)
    return repo_id


def push_to_hf_hub(
    model: CanViTForPretraining,
    *,
    owner: str,
    dataset: str,
    teacher_name: str,
    glimpse_size_px: int,
    private: bool = True,
) -> str:
    """Push model with auto-generated repo_id from metadata. Returns repo_id."""
    assert len(model.canvas_patch_grid_sizes) == 1, f"Expected single grid size, got {model.canvas_patch_grid_sizes}"

    grid_size = model.canvas_patch_grid_sizes[0]
    scene_size_px = grid_size * model.backbone.patch_size_px
    repo_id = make_repo_id(
        owner=owner,
        backbone_name=model.backbone_name,
        glimpse_size_px=glimpse_size_px,
        scene_size_px=scene_size_px,
        dataset=dataset,
        teacher_name=teacher_name,
        enable_vpe=model.cfg.enable_vpe,
        canvas_update_mode=model.cfg.canvas_update_mode,
    )
    return upload_to_hf(model, repo_id, private=private)


class CanViTForPretrainingHFHub(
    CanViTForPretraining,
    SafeHubMixin,
    PyTorchModelHubMixin,
    library_name="canvit-pytorch",
    repo_url="https://github.com/m2b3/CanViT-PyTorch",
):
    """CanViTForPretraining with HuggingFace Hub integration.

    Usage:
        model = CanViTForPretrainingHFHub.from_pretrained("<org>/canvitb16-add-vpe-pretrain-...")
    """

    def __init__(
        self,
        backbone_name: str,
        model_config: dict[str, Any],
        canvas_patch_grid_sizes: list[int],
        glimpse_grid_size: int | None = None,
        patch_stride: int | None = None,
    ):
        # asdict() flattens nested dataclasses to dicts on save, but
        # CanViTForPretrainingConfig(**model_config) only builds shallowly — so
        # nested dataclass fields (patcher.conditioning.{film.{fourier,
        # sinusoidal},coordconv}, vit_modulation.{fourier,sinusoidal}, …) arrive
        # as dicts and break on first attribute access. Rebuild them with the
        # generic recursive _coerce, which handles ANY depth (no per-field
        # special-casing — that approach repeatedly missed fields). Gated exactly
        # as before — only the ACTIVE patcher + vit_modulation — so every
        # existing checkpoint instantiates a byte-for-byte identical config.
        if (model_config.get("patcher_name") == "foveated"
                and isinstance(model_config.get("foveated_patcher"), dict)):
            model_config = {**model_config,
                            "foveated_patcher": _coerce(FoveatedPatcherConfig, model_config["foveated_patcher"])}
        if (model_config.get("patcher_name") == "square"
                and isinstance(model_config.get("square_patcher"), dict)):
            model_config = {**model_config,
                            "square_patcher": _coerce(SquarePatcherConfig, model_config["square_patcher"])}
        if isinstance(model_config.get("vit_modulation"), dict):
            model_config = {**model_config,
                            "vit_modulation": _coerce(ViTModulationConfig, model_config["vit_modulation"])}
        # ``patch_stride`` (overlapping patches: stride < patch_size) must be
        # rebuilt here — it is NOT in model_config (it's a top-level training
        # field). ``None`` -> create_backbone defaults to patch_size, so every
        # non-overlapping checkpoint is byte-for-byte unaffected.
        super().__init__(
            backbone=create_backbone(backbone_name, patch_stride=patch_stride),
            cfg=CanViTForPretrainingConfig(**model_config),
            backbone_name=backbone_name,
            canvas_patch_grid_sizes=canvas_patch_grid_sizes,
        )
        # Glimpse token-grid side (tokens per glimpse edge) the model was trained
        # with. The trained pixel glimpse size is ``glimpse_grid_size *
        # backbone.patch_size_px`` (see CanViT-pretrain train/model.py). Persisted
        # so downstream eval can crop glimpses at the SAME pixel size the model
        # saw in training, for ANY patch size -- not a hardcoded constant. ``None``
        # (older checkpoints that predate this field) -> callers fall back to the
        # canonical default of 8.
        self.glimpse_grid_size = glimpse_grid_size
