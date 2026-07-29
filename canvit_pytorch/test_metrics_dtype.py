"""mIoU counts are integers and must be accumulated exactly.

The accumulators were float32, which cannot represent consecutive integers above 2^24
(ADE20K val is 2000 x 512^2 = 524M pixels). CanViT-PyTorch-RL — whose published numbers
the stack is measured against — keeps per-image counts int64 and sums in float64
(`canvit_pytorch_rl/metrics.py::miou` casts `.double()`); float32 was this repo's own
deviation. Measured impact on ADE20K val is small (0.0002 mIoU).
"""
import torch

from canvit_pytorch.metrics import mIoUAccumulator


def _acc(n=4):
    return mIoUAccumulator(n, 255, torch.device("cpu"))


def test_counts_are_accumulated_as_int64():
    a = _acc()
    assert a.intersection.dtype == torch.int64
    assert a.union.dtype == torch.int64


def test_large_counts_stay_exact_past_the_float32_integer_limit():
    """The property float32 could not provide. 2^24 = 16,777,216."""
    a = _acc(2)
    big = 2**24 + 1
    preds = torch.zeros(1, big, 1, dtype=torch.long)
    targets = torch.zeros(1, big, 1, dtype=torch.long)
    a.update(preds, targets)
    assert a.intersection[0].item() == big, "exact integer count required"
    assert a.union[0].item() == big
    assert abs(a.compute() - 1.0) < 1e-12


def test_matches_the_rl_repos_expression():
    """`miou` masks on union>0 then divides in float64 — no epsilon."""
    a = _acc(3)
    a.intersection = torch.tensor([3, 0, 7], dtype=torch.int64)
    a.union = torch.tensor([6, 0, 7], dtype=torch.int64)  # class 1 absent -> excluded
    inter, union = a.intersection.double(), a.union.double()
    valid = union > 0
    expected = (inter[valid] / union[valid]).mean().item()
    assert a.compute() == expected
    assert abs(a.compute() - 0.75) < 1e-12  # mean(3/6, 7/7)


def test_all_classes_absent_returns_zero():
    a = _acc()
    assert a.compute() == 0.0
