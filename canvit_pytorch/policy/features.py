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
    ent = None
    if with_entropy:
        # Force EVAL mode for the probe forward. The template must be a property of the
        # WEIGHTS, not of whatever mode the caller happened to be in when it constructed
        # the StateEncoder: the segmentation head carries a BatchNorm, so under train-mode
        # BN this normalizes a batch of ONE synthetic blank canvas by its own statistics
        # and returns a different template entirely.
        #
        # Measured 2026-07-30: train- vs eval-mode construction moved the entropy template
        # by 1.621288, which propagated verbatim into every `ent_delta`/`cos_init` feature
        # and shifted ~14/32 of a trained policy's chosen glimpses, costing ~0.1 mIoU at
        # each policy timestep. It bit CanViT-pretrain's harness, which builds the policy
        # BEFORE freezing the model (`harness/run.py` build_policy at 277, freeze at 280),
        # while `ade20k/rl_train.py` happens to call seg.eval() first and was unaffected.
        # Fixing it here rather than at one call site: no caller should have to know that
        # constructing a feature encoder depends on module mode.
        head = getattr(seg, "head", None)
        was_training = head is not None and head.training
        if head is not None:
            head.eval()
        try:
            ent = probe_entropy(seg, canvas, canvas_grid=canvas_grid).float()
        finally:
            if was_training:
                head.train()
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
        # prev/init are REFERENCES the delta features read, never backprop targets, so
        # keep them detached: otherwise (joint task+policy training, CanViT-pretrain
        # P4b) the retained graph is re-entered on the next TBPTT chunk / step and
        # autograd raises "backward through the graph a second time". Under the frozen
        # no_grad callers (rl_train, eval) this detach is a harmless no-op.
        init_sp, init_ent = init_reference(seg, canvas_grid=canvas_grid, with_entropy=self.needs_entropy)
        self.init = (init_sp.detach(), init_ent.detach() if init_ent is not None else None)
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
        # Store the reference DETACHED (see __init__): the current feats keep grad
        # through `cur`, but next step's delta must not backprop into this step's state.
        self.prev = (cur.detach(), cur_ent.detach() if cur_ent is not None else None)
        return feats
