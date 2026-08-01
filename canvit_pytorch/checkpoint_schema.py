"""Read a CanViT training ``.pt`` checkpoint written by *either* trainer schema.

Two writers exist and they nest things differently:

* **legacy / flat** (CanViT-train's original ``train/loop.py``): ``state_dict`` plus
  top-level ``backbone_name`` / ``canvas_patch_grid_sizes`` / ``training_config_history``.
* **unified trainer** (CanViT-train's ``harness/infra/checkpoint.py``): ``model_state``
  plus a nested ``metadata`` dict built from the task's ``checkpoint_metadata``.

Anything that loads a ``.pt`` must normalize first, or it KeyErrors on ``state_dict`` --
or worse, silently reads ``pretrain_view_scale=None`` because ``training_config_history``
is not at the top level, which is the foveated out-of-distribution footgun these helpers
exist to prevent (a model evaluated at a view scale it never saw does not crash; its mIoU
just decays as glimpses accumulate).

This lives in ``canvit_pytorch`` rather than in the trainer because it is needed by every
consumer of a checkpoint -- the model constructors here, the trainer's HF exporter, and
CanViT-eval. It is pure dict manipulation: no torch, no I/O.
"""

import logging
from typing import Any

log = logging.getLogger(__name__)

# Patchers whose in-distribution behavior depends on the pretraining view scale.
SCALE_SENSITIVE_PATCHERS = ("foveated", "square")

__all__ = [
    "SCALE_SENSITIVE_PATCHERS",
    "extract_pretrain_view_scale",
    "migrate_standardizers_in_place",
    "normalize_schema",
]


def migrate_standardizers_in_place(raw: dict) -> None:
    """Fold legacy top-level standardizer state into the ``state_dict``.

    Old checkpoints stored the teacher-target standardizers beside the weights
    (``scene_norm_state`` / ``cls_norm_state``) instead of as submodules. Call after
    :func:`normalize_schema`, which is what guarantees ``state_dict`` exists.
    """
    if (scene_legacy := raw.get("scene_norm_state")) is None:
        return
    cls_legacy = raw["cls_norm_state"]
    grids = raw["canvas_patch_grid_sizes"]
    assert len(grids) == 1, f"Expected single grid size, got {grids}"
    G = str(grids[0])
    sd = raw["state_dict"]
    for prefix, legacy in [("scene_standardizers", scene_legacy), ("cls_standardizers", cls_legacy)]:
        for stat in ("mean", "var", "_initialized"):
            sd[f"{prefix}.{G}.{stat}"] = legacy[stat]
    del raw["scene_norm_state"], raw["cls_norm_state"]
    log.info("Migrated legacy standardizers (grid=%s)", G)


def normalize_schema(raw: dict) -> dict:
    """Return the checkpoint in the FLAT schema, whichever writer produced it.

    Legacy payloads pass through untouched.

    Raises:
        KeyError: if a unified-trainer payload has no backbone architecture in it. That
            is the case for the downstream tasks (ade20k / in1k), whose ``model_config``
            records only a ``model_repo`` *path* pointing at the backbone they were built
            on -- so such a checkpoint genuinely cannot rebuild its own model, and no
            amount of dispatching fixes that. Load those via their ``model_repo``.
    """
    if "model_state" not in raw:
        return raw
    md = raw.get("metadata") or {}
    missing = [k for k in ("backbone_name", "canvas_patch_grid_sizes") if k not in md]
    if missing:
        raise KeyError(
            f"checkpoint metadata is missing {missing}; only a pretraining (distill) "
            f"checkpoint carries its own backbone architecture (this one is "
            f"task={md.get('task')!r})")
    # The unified trainer nests the real architecture under model_config["canvit"]; the
    # rest of that dict is resume bookkeeping (task/teacher_dim/canvas_grid/
    # backbone_name). The flat schema -- and `patcher_name`, which drives the whole
    # view-scale check -- wants the FLAT CanViTForPretrainingConfig, so unwrap it.
    mc = raw.get("model_config") or {}
    mc = mc.get("canvit", mc)
    return {
        **raw,
        "state_dict": raw["model_state"],
        "model_config": mc,
        "backbone_name": md["backbone_name"],
        "canvas_patch_grid_sizes": md["canvas_patch_grid_sizes"],
        "glimpse_grid_size": md.get("glimpse_grid_size"),
        "patch_stride": md.get("patch_stride"),
        "teacher_name": md.get("teacher_name"),
        "dataset": md.get("dataset"),
        "training_config_history": md.get("training_config_history"),
    }


def _scale_fields(entry: dict) -> dict[str, Any]:
    """The ``foveated_scale`` fields of one ``training_config_history`` entry.

    Legacy entries are FLAT (``foveated_scale.mode``, … -- ``train/loop.py:flatten_dict``);
    unified-trainer entries carry the config as a NESTED ``foveated_scale`` dict.
    """
    if isinstance(nested := entry.get("foveated_scale"), dict):
        return nested
    return {k[len("foveated_scale."):]: v for k, v in entry.items()
            if k.startswith("foveated_scale.")}


def extract_pretrain_view_scale(raw: dict) -> dict[str, Any] | None:
    """Recover the pretraining foveated/square view scale from a NORMALIZED checkpoint.

    The scale is a training-time *viewpoint* parameter (``FoveatedScaleConfig``), not part
    of ``model_config``, so it has to be dug out of ``training_config_history``. Returns
    ``None`` for uniform models (view scale is not their out-of-distribution axis --
    glimpse crop pixels are) and for checkpoints with no history recorded. Callers must
    treat ``None`` as "unknown", never as "scale 1.0".
    """
    patcher = (raw.get("model_config") or {}).get("patcher_name")
    if patcher not in SCALE_SENSITIVE_PATCHERS:
        return None
    history = raw.get("training_config_history") or {}
    if not history:
        return None
    # Entries are keyed by ISO-8601 timestamp; the most recent is the config the run
    # finished with. The view scale is a model-defining choice and is expected constant
    # across a run, but taking the latest is the safe pick.
    fields = _scale_fields(history[max(history)])
    mode = fields.get("mode")
    if mode is None:
        return None
    return {
        "patcher_name": patcher,
        "mode": mode,
        "distribution": fields.get("distribution"),
        "fixed_scale": fields.get("fixed_scale"),
        "min_scale": fields.get("min_scale"),
        "max_scale": fields.get("max_scale"),
    }
