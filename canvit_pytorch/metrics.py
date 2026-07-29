"""Shared evaluation metrics used across the CanViT stack (training + eval)."""

import torch
from torch import Tensor


class mIoUAccumulator:
    """Global mIoU: sum intersection/union across all images, then average over
    classes. No GPU sync until compute()."""

    def __init__(self, num_classes: int, ignore_index: int, device: torch.device) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # int64, not float32: these are PIXEL COUNTS, and `bincount` already returns
        # int64. ADE20K val is 2000 x 512^2 = 524M pixels, well past float32's 2^24
        # exact-integer limit, so a float32 accumulator rounds the running totals
        # order-dependently. CanViT-PyTorch-RL — whose published numbers the stack is
        # measured against — keeps per-image counts int64 and sums them in float64
        # (`canvit_pytorch_rl/metrics.py::miou` casts `.double()`); float32 was this
        # repo's own deviation. Measured impact is small (0.0002 mIoU on ADE20K val),
        # so this aligns the arithmetic with the reference rather than fixing a
        # visible defect — do not cite it as explaining any larger discrepancy.
        self.intersection = torch.zeros(num_classes, device=device, dtype=torch.int64)
        self.union = torch.zeros(num_classes, device=device, dtype=torch.int64)

    def update(self, preds: Tensor, targets: Tensor) -> None:
        assert preds.ndim == 3, f"Expected [B, H, W], got shape {preds.shape}"
        assert preds.shape == targets.shape, f"Shape mismatch: {preds.shape} vs {targets.shape}"
        n = self.num_classes
        p = preds.flatten().long()
        t = targets.flatten().long()
        valid = t != self.ignore_index
        p, t = p[valid], t[valid]
        cm = torch.bincount(t * n + p, minlength=n * n).view(n, n)
        diag = cm.diag()
        self.intersection += diag
        self.union += cm.sum(dim=1) + cm.sum(dim=0) - diag

    def reset(self) -> None:
        self.intersection.zero_()
        self.union.zero_()

    def compute(self) -> float:
        # Mask FIRST, then divide in float64 — `canvit_pytorch_rl/metrics.py::miou`'s
        # exact expression. Dividing int64 by a Python float would promote to float32
        # and throw away the precision the int64 accumulators just preserved, and the
        # old `+ 1e-8` epsilon only guarded classes that the union>0 mask discards
        # anyway, so masking first is both exact and simpler.
        valid = self.union > 0
        if not bool(valid.any()):
            return 0.0
        return (self.intersection[valid].double() / self.union[valid].double()).mean().item()
