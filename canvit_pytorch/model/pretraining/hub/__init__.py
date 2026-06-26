"""HuggingFace Hub integration for CanViTForPretraining."""

import json
import logging
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, PyTorchModelHubMixin
from safetensors.torch import save_file

from canvit_pytorch.backbone import create_backbone
from canvit_pytorch.model.hub_mixin import SafeHubMixin
from canvit_pytorch.encoding import SinusoidalConfig
from canvit_pytorch.modulation import ViTModulationConfig
from canvit_pytorch.patcher import FoveatedPatcherConfig, SquarePatcherConfig
from canvit_pytorch.patcher.conditioning import (
    CoordConvConfig,
    FiLMConfig,
    FourierConfig,
    PatchConditioningConfig,
)

from ..impl import CanViTForPretraining, CanViTForPretrainingConfig

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
        # Coerce nested patcher dict → dataclass for foveated / square
        # checkpoints. upload_to_hf serializes the config via asdict (flattens
        # nested dataclasses to dicts), but CanViTForPretrainingConfig(
        # **model_config) does no recursive coercion, so the patcher would
        # receive a dict. Strictly gated on patcher_name so the uniform path is
        # byte-for-byte unaffected. The conditioning subtree has its own nested
        # dataclasses (FiLMConfig / FourierConfig / CoordConvConfig) that must
        # also be coerced recursively.
        def _coerce_conditioning(fp: dict) -> dict:
            fp = dict(fp)
            cond = fp.get("conditioning")
            if isinstance(cond, dict):
                cond = dict(cond)
                if isinstance(cond.get("film"), dict):
                    film = dict(cond["film"])
                    if isinstance(film.get("fourier"), dict):
                        film["fourier"] = FourierConfig(**film["fourier"])
                    cond["film"] = FiLMConfig(**film)
                if isinstance(cond.get("coordconv"), dict):
                    cond["coordconv"] = CoordConvConfig(**cond["coordconv"])
                fp["conditioning"] = PatchConditioningConfig(**cond)
            return fp

        if (model_config.get("patcher_name") == "foveated"
                and isinstance(model_config.get("foveated_patcher"), dict)):
            model_config = {
                **model_config,
                "foveated_patcher": FoveatedPatcherConfig(
                    **_coerce_conditioning(model_config["foveated_patcher"])
                ),
            }
        if (model_config.get("patcher_name") == "square"
                and isinstance(model_config.get("square_patcher"), dict)):
            model_config = {
                **model_config,
                "square_patcher": SquarePatcherConfig(
                    **_coerce_conditioning(model_config["square_patcher"])
                ),
            }
        # vit_modulation is the third nested-dataclass field of the config (its
        # own subtree: fourier / sinusoidal, both flat). asdict flattened it to a
        # dict on save; coerce it back so cfg.vit_modulation is a dataclass. Gated
        # on the dict being present, so checkpoints predating the field (which
        # omit it and fall back to the default dataclass) are unaffected.
        if isinstance(model_config.get("vit_modulation"), dict):
            vm = dict(model_config["vit_modulation"])
            if isinstance(vm.get("fourier"), dict):
                vm["fourier"] = FourierConfig(**vm["fourier"])
            if isinstance(vm.get("sinusoidal"), dict):
                vm["sinusoidal"] = SinusoidalConfig(**vm["sinusoidal"])
            model_config = {**model_config, "vit_modulation": ViTModulationConfig(**vm)}
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
