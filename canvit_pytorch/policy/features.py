"""Canvas state -> scorer input: the derived, scale-equalized feature groups and the
encoder that builds them. Ported from canvit_pytorch_rl.policy.features, decoupled
from the RL TrainConfig (explicit canvas_grid / feature_groups arguments).

FEATURE_GROUPS names the groups in channel order; ent/ent_delta/cos_* are 1 channel,
ln_feat/feat_delta are canvas_dim wide. The probe-entropy groups (ent, ent_delta)
require a task probe — tasks without one (e.g. distillation pretraining) select the
four intrinsic groups (master plan §3: feature groups are task-configurable)."""

import torch
import torch.nn.functional as F
from torch import Tensor

from canvit_pytorch.model.segmentation import CanViTForSemanticSegmentation

from .scoring import entropy_from_logits, head_logits, probe_entropy

FEATURE_GROUPS = ("ent", "ent_delta", "cos_prev", "cos_init", "ln_feat", "feat_delta")
INTRINSIC_GROUPS = ("cos_prev", "cos_init", "ln_feat", "feat_delta")  # no probe needed

POLICY_GRID = 32  # the scorer's working resolution; larger canvases are pooled to it

_SCALAR_GROUPS = {"ent", "ent_delta", "cos_prev", "cos_init"}


def group_sizes(canvas_dim: int, groups: tuple[str, ...]) -> list[int]:
    return [1 if g in _SCALAR_GROUPS else canvas_dim for g in groups]


def feature_channels(canvas_dim: int, groups: tuple[str, ...] = FEATURE_GROUPS) -> int:
    return sum(group_sizes(canvas_dim, groups))


def _lnc(x: Tensor) -> Tensor:  # channel-wise LayerNorm per token
    return F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],)).permute(0, 3, 1, 2)


def _canvas_spatial(seg: CanViTForSemanticSegmentation, canvas: Tensor, canvas_grid: int) -> Tensor:
    s = seg.canvit.get_spatial(canvas).float()  # [N,T,D] tokens -> [N,D,g,g] map
    b, _, d = s.shape
    return s.permute(0, 2, 1).reshape(b, d, canvas_grid, canvas_grid)


def init_reference(
    seg: CanViTForSemanticSegmentation, *, canvas_grid: int, with_entropy: bool
) -> tuple[Tensor, Tensor | None]:
    """The init (blank) canvas's (spatial feats, probe entropy): an image-independent
    template, so the t0 delta/cos features carry deviation-from-template instead of
    dead zeros. with_entropy=False for probe-free tasks (intrinsic groups only)."""
    canvas = seg.canvit.init_state(batch_size=1, canvas_grid_size=canvas_grid).canvas
    ent = probe_entropy(seg, canvas, canvas_grid=canvas_grid).float() if with_entropy else None
    return _canvas_spatial(seg, canvas, canvas_grid), ent


def assemble_features(
    cur: Tensor,
    prev: Tensor,
    cur_ent: Tensor | None,
    prev_ent: Tensor | None,
    init_ln: Tensor,
    groups: tuple[str, ...],
) -> Tensor:
    """The selected feature groups -> [B, sum(sizes), POLICY_GRID, POLICY_GRID].
    `prev` is the previous state's spatial feats (the init template at t0), so
    deltas read deviation-from-prev. Entropy tensors may be None iff no entropy
    group is selected."""
    ln, ln_prev = _lnc(cur), _lnc(prev)
    avail: dict[str, Tensor] = {
        "cos_prev": (1 - F.cosine_similarity(ln, ln_prev, dim=1)).unsqueeze(1),
        "cos_init": (1 - F.cosine_similarity(ln, init_ln, dim=1)).unsqueeze(1),
        "ln_feat": ln,
        "feat_delta": ln - ln_prev,
    }
    if cur_ent is not None:
        assert prev_ent is not None
        avail["ent"] = cur_ent.unsqueeze(1)
        avail["ent_delta"] = (cur_ent - prev_ent).unsqueeze(1)
    assert all(g in avail for g in groups), (
        f"groups {groups} include a probe-entropy group but no entropy was provided "
        f"(probe-free task? use INTRINSIC_GROUPS)"
    )
    out = torch.cat([avail[g] for g in groups], dim=1)
    return F.adaptive_avg_pool2d(out, POLICY_GRID) if out.shape[-1] != POLICY_GRID else out


class StateEncoder:
    """Canvas state -> scorer input, holding the rolling previous-state reference the
    delta features need. Built once from (seg, canvas_grid, feature_groups);
    `reset()` at t0, then call per step. The ONE place featurization is done —
    trainer, rollout, policy and eval all share it."""

    def __init__(
        self,
        seg: CanViTForSemanticSegmentation,
        *,
        canvas_grid: int,
        feature_groups: tuple[str, ...] = FEATURE_GROUPS,
    ):
        self.seg = seg
        self.canvas_grid = canvas_grid
        self.feature_groups = feature_groups
        self.needs_entropy = any(g in ("ent", "ent_delta") for g in feature_groups)
        self.init = init_reference(seg, canvas_grid=canvas_grid, with_entropy=self.needs_entropy)
        self.init_ln = _lnc(self.init[0])
        self.prev: tuple[Tensor, Tensor | None] = self.init  # rolling (spatial, entropy)

    def reset(self) -> None:
        self.prev = self.init

    def __call__(self, state, logits: Tensor | None = None) -> Tensor:
        """`logits` = probe logits of state.canvas when the caller already has them
        (a rollout computing them for the CE reward shares that one head_logits
        call). None -> computed here when an entropy group needs them."""
        cur = _canvas_spatial(self.seg, state.canvas, self.canvas_grid)
        cur_ent: Tensor | None = None
        if self.needs_entropy:
            if logits is None:
                logits = head_logits(self.seg, state.canvas, canvas_grid=self.canvas_grid)
            cur_ent = entropy_from_logits(logits).float()
        feats = assemble_features(
            cur, self.prev[0], cur_ent, self.prev[1], self.init_ln, self.feature_groups
        )
        self.prev = (cur, cur_ent)
        return feats
