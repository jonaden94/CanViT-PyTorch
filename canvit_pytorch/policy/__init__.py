"""Learned viewing policies for CanViT (unification master plan P3, decision D1).

The policy MODEL lives in core so every consumer (training in CanViT-pretrain,
benchmarking in CanViT-eval) can load published policies without depending on a
training repo. Training losses/objectives (QReg/PG) live in CanViT-pretrain.

Ported from CanViT-PyTorch-RL (canvit_pytorch_rl.policy.{net,features} +
scoring primitives); published ViewpointScorer checkpoints keep loading — the
config.json schema is unchanged (the new action_space key defaults to the
historical "safebox").
"""

from .features import (
    FEATURE_GROUPS,
    StateEncoder,
    assemble_features,
    feature_channels,
    group_sizes,
    init_reference,
)
from .net import (
    DEFAULT_POLICY_REPO,
    ViewpointScorer,
    candidate_viewpoints,
    fixation_candidates,
)
from .scoring import entropy_from_logits, head_logits, per_image_ce, probe_entropy

__all__ = [
    "DEFAULT_POLICY_REPO",
    "FEATURE_GROUPS",
    "StateEncoder",
    "ViewpointScorer",
    "assemble_features",
    "candidate_viewpoints",
    "entropy_from_logits",
    "feature_channels",
    "fixation_candidates",
    "group_sizes",
    "head_logits",
    "init_reference",
    "per_image_ce",
    "probe_entropy",
]
