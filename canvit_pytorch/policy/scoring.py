"""Probe scoring primitives: logits from a canvas, per-position entropy, per-image CE.

Ported from canvit_pytorch_rl.scoring, minus the eval-only I/U helpers (those
stay with evaluation code). ADE20K constants are parameters here (ignore_label
defaults to 255) so core carries no dataset dependency."""

import torch
import torch.nn.functional as F
from torch import Tensor

from canvit_pytorch.model.segmentation import CanViTForSemanticSegmentation

IGNORE_LABEL_DEFAULT = 255


def head_logits(seg: CanViTForSemanticSegmentation, canvas: Tensor, *, canvas_grid: int) -> Tensor:
    """Probe logits [B, C, G, G] from a recurrent-state canvas, always in fp32."""
    spatial = seg.canvit.get_spatial(canvas)
    B, n_spatial, D = spatial.shape
    assert n_spatial == canvas_grid * canvas_grid, f"{n_spatial} spatial tokens != {canvas_grid}^2"
    with torch.autocast(device_type=spatial.device.type, enabled=False):
        return seg.head(spatial.view(B, canvas_grid, canvas_grid, D).float())


def entropy_from_logits(logits: Tensor) -> Tensor:
    """Per-position probe entropy [B, G, G] from [B, C, G, G] logits — reuse logits you already have."""
    log_probs = F.log_softmax(logits, dim=1)
    return -(log_probs.exp() * log_probs).sum(dim=1)


def probe_entropy(seg: CanViTForSemanticSegmentation, canvas: Tensor, *, canvas_grid: int) -> Tensor:
    """Per-position entropy of the probe, [B, G, G]."""
    return entropy_from_logits(head_logits(seg, canvas, canvas_grid=canvas_grid))


def per_image_ce(logits: Tensor, masks: Tensor, *, ignore_label: int = IGNORE_LABEL_DEFAULT) -> Tensor:
    """Mean probe CE over valid pixels per image — the policy reward's raw material.
    logits [B, C, S, S] fp32, masks [B, S, S]; ignore_label pixels excluded."""
    ce = F.cross_entropy(logits, masks, ignore_index=ignore_label, reduction="none")
    valid = (masks != ignore_label).sum(dim=(1, 2)).clamp(min=1)
    return ce.sum(dim=(1, 2)) / valid
