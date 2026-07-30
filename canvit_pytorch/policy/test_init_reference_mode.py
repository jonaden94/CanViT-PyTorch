"""The StateEncoder's init template must not depend on module MODE.

`init_reference` runs the segmentation probe on a blank canvas to build an
image-independent template for the ent_delta / cos_init features. The probe head carries
a BatchNorm, so under train-mode BN it normalized a batch of ONE synthetic canvas by its
own statistics -- a different template.

Measured 2026-07-30: train vs eval construction moved the entropy template by 1.621288,
which propagated verbatim into the features and shifted ~14/32 of a trained policy's
chosen glimpses (~0.1 mIoU per policy timestep). CanViT-pretrain's harness hit this by
building the policy before freezing the model; ade20k/rl_train.py did not.
"""
import torch
import torch.nn as nn

from canvit_pytorch.policy.features import init_reference


class _Head(nn.Module):
    """Stands in for the segmentation probe: BN makes it mode-sensitive, as the real one is."""

    def __init__(self, dim=8, classes=5):
        super().__init__()
        self.bn = nn.BatchNorm2d(dim)
        self.proj = nn.Conv2d(dim, classes, 1)
        self.bn.running_mean.fill_(3.0)  # far from the blank canvas's own statistics
        self.bn.running_var.fill_(0.5)

    def forward(self, x):                      # [B, G, G, D] -> [B, C, G, G]
        return self.proj(self.bn(x.permute(0, 3, 1, 2)))


class _Canvit(nn.Module):
    def __init__(self, dim=8, grid=4):
        super().__init__()
        self.dim, self.grid = dim, grid

    def init_state(self, *, batch_size, canvas_grid_size):
        n = canvas_grid_size * canvas_grid_size
        return type("S", (), {"canvas": torch.full((batch_size, n, self.dim), 0.7)})()

    def get_spatial(self, canvas):
        return canvas


class _Seg(nn.Module):
    def __init__(self):
        super().__init__()
        self.canvit = _Canvit()
        self.head = _Head()


GRID = 4


def test_template_is_identical_in_train_and_eval_mode():
    seg = _Seg()
    seg.train()
    sp_t, ent_t = init_reference(seg, canvas_grid=GRID, with_entropy=True)
    seg.eval()
    sp_e, ent_e = init_reference(seg, canvas_grid=GRID, with_entropy=True)
    assert torch.equal(sp_t, sp_e)
    assert ent_t is not None and ent_e is not None
    assert torch.equal(ent_t, ent_e), (
        "the entropy template changed with module mode — the bug that shifted 14/32 "
        "glimpses of a trained policy")


def test_it_matches_the_eval_mode_value_specifically():
    """Eval mode (running stats) is the correct reference, not merely a consistent one."""
    seg = _Seg()
    seg.eval()
    _, ent_eval = init_reference(seg, canvas_grid=GRID, with_entropy=True)
    seg.train()
    _, ent_from_train = init_reference(seg, canvas_grid=GRID, with_entropy=True)
    assert torch.equal(ent_from_train, ent_eval)


def test_callers_module_mode_is_restored():
    """It must not leave the head in eval mode — that would silently freeze BN for training."""
    seg = _Seg()
    seg.train()
    assert seg.head.training
    init_reference(seg, canvas_grid=GRID, with_entropy=True)
    assert seg.head.training, "train mode must be restored"
    seg.eval()
    init_reference(seg, canvas_grid=GRID, with_entropy=True)
    assert not seg.head.training, "eval mode must not be flipped to train"


def test_no_probe_forward_when_entropy_is_not_needed():
    """Probe-free tasks (intrinsic groups) must not touch the head at all."""
    seg = _Seg()
    seg.train()
    sp, ent = init_reference(seg, canvas_grid=GRID, with_entropy=False)
    assert ent is None
    assert sp is not None
