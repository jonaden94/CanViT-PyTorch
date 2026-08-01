"""Load a trained CanViT from whatever you have: a training ``.pt``, a local HF-layout
directory, or a Hub repo id.

One dispatch point, so every consumer accepts all three without branching. The rule is the
``.pt`` suffix; anything else is handed to ``from_pretrained``, which already treats a
local directory and a Hub id interchangeably.

Why the metadata half exists too. A model is not only weights: a foveated/square model
pretrained at a FIXED view scale must be *evaluated* at that same scale or every glimpse is
out of distribution — which does not crash, it just decays the metric as glimpses
accumulate. CanViT-eval reads that scale out of ``config.json``'s ``metadata``. A loader
that handled ``.pt`` weights but left metadata to the config.json path would therefore load
a checkpoint successfully and silently evaluate it wrong. So :func:`read_pretrain_metadata`
returns the same dict shape from either source.

A ``.pt`` is in fact the richer source: ``pretrain_view_scale`` is derived from the
training config recorded inside it, and is written into ``config.json`` only because
``to_hf`` puts it there.
"""

import json
import logging
from pathlib import Path
from typing import Any

from canvit_pytorch.checkpoint_schema import extract_pretrain_view_scale, normalize_schema

log = logging.getLogger(__name__)

__all__ = [
    "is_checkpoint",
    "load_classifier",
    "load_pretraining",
    "load_segmentation",
    "load_segmentation_probe",
    "read_pretrain_metadata",
]


def is_checkpoint(source: str | Path) -> bool:
    """True if ``source`` names a training ``.pt`` rather than an HF dir / Hub id.

    Raises:
        FileNotFoundError: if it looks like a ``.pt`` but is not there. Falling through to
            a Hub lookup would turn a typo into a confusing network error about a repo id
            that was never meant to be one.
    """
    s = str(source)
    if not s.endswith(".pt"):
        return False
    if not Path(s).is_file():
        raise FileNotFoundError(f"checkpoint not found: {s}")
    return True


def load_pretraining(source: str | Path, *, map_location: str = "cpu"):
    """Load a pretrained CanViT from a ``.pt``, a local HF directory, or a Hub id.

    Both paths return an equivalent model — same weights, same config, and the same
    ``glimpse_grid_size`` / ``glimpse_size_px`` attribute contract (see
    ``CanViTForPretraining.from_checkpoint``).
    """
    from canvit_pytorch.model.pretraining.hub import CanViTForPretrainingHFHub
    from canvit_pytorch.model.pretraining.impl import CanViTForPretraining

    if is_checkpoint(source):
        return CanViTForPretraining.from_checkpoint(source, map_location=map_location)
    return CanViTForPretrainingHFHub.from_pretrained(str(source))


def load_segmentation_probe(source: str | Path, *, dropout: float = 0.1):
    """Load a segmentation probe from a downstream training ``.pt``, a local HF directory,
    or a Hub id. ``dropout`` applies to the ``.pt`` path only (checkpoints do not record
    it); an HF probe carries its own.
    """
    from canvit_pytorch.probes import SegmentationProbe

    if is_checkpoint(source):
        return SegmentationProbe.from_checkpoint(source, dropout=dropout)
    return SegmentationProbe.from_pretrained(str(source))


def load_segmentation(source: str | Path, *, map_location: str = "cpu"):
    """Load a whole segmentation model (backbone + head) from an ``ade20k`` training
    ``.pt``, a local HF directory, or a Hub id.

    For a FINETUNE the ``.pt`` is the only place the backbone weights exist, which is why
    this is not the same as assembling ``model_repo`` + probe.
    """
    from canvit_pytorch.model.segmentation import CanViTForSemanticSegmentation

    if is_checkpoint(source):
        return CanViTForSemanticSegmentation.from_checkpoint(source, map_location=map_location)
    return CanViTForSemanticSegmentation.from_pretrained(str(source))


def load_classifier(source: str | Path, *, map_location: str = "cpu"):
    """Load a whole classifier from an ``in1k`` training ``.pt``, a local HF directory, or
    a Hub id. As with segmentation, an in1k finetune's ``.pt`` is authoritative."""
    from canvit_pytorch.model.classification import CanViTForImageClassification

    if is_checkpoint(source):
        return CanViTForImageClassification.from_checkpoint(source, map_location=map_location)
    return CanViTForImageClassification.from_pretrained(str(source))


def read_backbone_name(source: str | Path) -> str | None:
    """The backbone name of a pretrained CanViT, without loading the model.

    Callers use this to pick a companion artifact (e.g. which DINOv3 in1k probe to fuse),
    so it must work for a ``.pt`` as well — otherwise passing a checkpoint would fail at
    the point of choosing the probe rather than at the point of loading weights.
    """
    if is_checkpoint(source):
        import torch

        raw = normalize_schema(torch.load(source, map_location="cpu", weights_only=False))
        return raw.get("backbone_name")
    cfg = Path(str(source)) / "config.json"
    if not cfg.is_file():
        from huggingface_hub import hf_hub_download

        cfg = Path(hf_hub_download(str(source), "config.json"))
    return json.loads(cfg.read_text()).get("backbone_name")


def read_pretrain_metadata(source: str | Path) -> dict[str, Any]:
    """The pretraining ``metadata`` dict for ``source``, whatever kind of source it is.

    Same shape either way — notably ``pretrain_view_scale`` and ``teacher_name``. Returns
    ``{}`` when it cannot be read, matching the best-effort contract callers already rely
    on (an unreadable source must degrade to "unknown", never to a wrong default).
    """
    try:
        if is_checkpoint(source):
            import torch

            raw = normalize_schema(torch.load(source, map_location="cpu", weights_only=False))
            return {
                "source_pt": str(source),
                "step": raw.get("step"),
                "teacher_name": raw.get("teacher_name"),
                "dataset": raw.get("dataset"),
                "git_commit": raw.get("git_commit"),
                "timestamp": raw.get("timestamp"),
                "pretrain_view_scale": extract_pretrain_view_scale(raw),
            }
        cfg = Path(str(source)) / "config.json"
        if not cfg.is_file():
            from huggingface_hub import hf_hub_download

            cfg = Path(hf_hub_download(str(source), "config.json"))
        return json.loads(cfg.read_text()).get("metadata") or {}
    except FileNotFoundError:
        raise
    except Exception as e:  # noqa: BLE001 — best effort; callers treat {} as "unknown"
        log.warning("Could not read metadata from %s: %s", source, e)
        return {}
