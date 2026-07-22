"""The candidate-scorer network over viewpoints: action geometry and the net itself
(state featurization lives in policy.features). One architecture, two readings of its
output map (chosen by the TRAINING objective, which lives in CanViT-pretrain):
Q-values regressed on rewards (QReg) or policy logits (PG).

Ported from canvit_pytorch_rl.policy.net unchanged except: (1) decoupled from the RL
TrainConfig (explicit kwargs only — build_scorer/objective_from_ckpt stay with the
trainer); (2) the NEW "fixation" action space for foveated/square models (user
decision 2026-07-22: actions = fixation centers only, one per grid cell over the
full field, no scale dimension). config.json schema is backward-compatible — old
checkpoints lack action_space and get the historical "safebox".

Arch [2026-06-14: import a battle-tested block, don't hand-roll init]: timm
ConvNeXt-V2 blocks. LayerNorm2d in the body (BatchNorm's running stats are corrupted
by the raw std-186 backbone features -> eval instability; the Frontend's input
BatchNorm is safe because its inputs are pre-normalized). Readout: encoder pools the
32x32 canvas to a 1x1 global bottleneck (so 'where to look' sees the whole scene);
the decoder rebuilds the full 32x32 map; scores are read at the candidate centres
through a shared head, so output cell (i,j) reads the features AT viewpoint (i,j)'s
centre — aligned by construction.
"""

from typing import Literal

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from timm.layers.norm import LayerNorm2d
from timm.models.convnext import ConvNeXtBlock
from torch import Tensor, nn

from canvit_pytorch.checkpoints import resolve_canvit_repo
from canvit_pytorch.model.hub_mixin import SafeHubMixin

from .features import FEATURE_GROUPS, feature_channels, group_sizes

# The flagship published policy — ViewpointScorer.from_pretrained(DEFAULT_POLICY_REPO)
# just works. = the 8-seed qband's best seed by mean(t1-t4) val CE; all 8 seeds are
# published as ...-qband-2026-07-04-s{0..7} for reproduction.
DEFAULT_POLICY_REPO = resolve_canvit_repo("qpolicy-ade20k-c64-t5-qband-2026-07-04-s2")

ActionSpace = Literal["safebox", "fixation"]


def candidate_viewpoints(scales: tuple[float, ...], centers_per_axis: int) -> Tensor:
    """[n_scale, centers_per_axis, centers_per_axis, 3] (cy, cx, scale): a grid of
    centres spanning the safe box [-(1-s), 1-s] at each scale s (overlapping
    glimpses; placement resolution decoupled from scale, so the glimpse box never
    leaves the image). The UNIFORM-patcher action space."""
    vps = []
    for s in scales:
        c = torch.linspace(-(1 - s), 1 - s, centers_per_axis)
        cy, cx = torch.meshgrid(c, c, indexing="ij")
        vps.append(torch.stack([cy, cx, torch.full_like(cy, s)], dim=-1))
    return torch.stack(vps)


def fixation_candidates(centers_per_axis: int) -> Tensor:
    """[1, centers_per_axis, centers_per_axis, 3] (cy, cx, 1.0): fixation centres at
    the CELL CENTRES of a grid over the full field [-1, 1]² — the FOVEATED/SQUARE
    action space (no scale dimension; foveation always sees the whole image, the
    view scale comes from FoveatedScaleConfig at glimpse time, not from the policy).
    Cell centres align with score-map pixels under grid_sample align_corners=False,
    so the readout degenerates to reading the map directly."""
    step = 1.0 / centers_per_axis
    c = torch.linspace(-1 + step, 1 - step, centers_per_axis)
    cy, cx = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([cy, cx, torch.ones_like(cy)], dim=-1).unsqueeze(0)


def _stage(ci: int, co: int, n: int) -> nn.Sequential:
    """A 1x1 channel projection (only if ci != co) then n ConvNeXt-V2 blocks at co
    channels. ConvNeXtBlock keeps channels (depthwise conv needs out%groups==0), so
    cross-channel changes (decoder 2w->w) are an explicit 1x1; the blocks do spatial
    mixing at fixed width."""
    layers: list[nn.Module] = [] if ci == co else [nn.Conv2d(ci, co, 1)]
    layers += [ConvNeXtBlock(co, co, use_grn=True) for _ in range(n)]
    return nn.Sequential(*layers)


class _TokenMLP(nn.Module):
    """Per-token (1x1) pre-norm residual MLP: x + mlp(LayerNorm(x))."""

    def __init__(self, ch: int, *, hidden_mult: int = 2):
        super().__init__()
        self.norm = LayerNorm2d(ch)
        self.net = nn.Sequential(nn.Conv2d(ch, hidden_mult * ch, 1), nn.GELU(), nn.Conv2d(hidden_mult * ch, ch, 1))

    def forward(self, x: Tensor) -> Tensor:  # [B, C, H, W]
        return x + self.net(self.norm(x))


class Frontend(nn.Module):
    """Fuse the derived feature groups into [B, width, 32, 32]: per-channel BatchNorm
    (scale-equalize the heterogeneous inputs) -> each group its OWN 1x1 conv to width
    -> per-group LayerNorm2d (affine, so it reweights per-channel importance) -> x
    per-group gate -> sum -> pre-norm token-MLP -> 1x1. (A plain per-group-proj sum
    WITHOUT the LN equals one shared proj; the LN is what makes it non-trivial.)"""

    sizes: list[int]
    groups: tuple[str, ...]

    def __init__(self, canvas_dim: int, width: int, *, groups: tuple[str, ...] = FEATURE_GROUPS):
        super().__init__()
        self.groups = groups
        self.sizes = group_sizes(canvas_dim, groups)
        self.bn = nn.BatchNorm2d(sum(self.sizes))
        self.proj = nn.ModuleList([nn.Conv2d(sz, width, 1) for sz in self.sizes])
        self.norm = nn.ModuleList([LayerNorm2d(width) for _ in self.sizes])
        self.gate = nn.Parameter(torch.ones(len(self.sizes)))
        self.mlp = _TokenMLP(width)
        self.out = nn.Conv2d(width, width, 1)

    def forward(self, feats: Tensor) -> Tensor:
        b = self.bn(feats)
        x, off = None, 0
        for i, (proj, norm, sz) in enumerate(zip(self.proj, self.norm, self.sizes)):
            xg = norm(proj(b[:, off : off + sz])) * self.gate[i]
            x, off = xg if x is None else x + xg, off + sz
        return self.out(self.mlp(x))

    @torch.no_grad()
    def log(self) -> dict[str, float]:  # per-group gate (favor/suppress importance readout)
        return {f"gate_{n}": float(self.gate[i]) for i, n in enumerate(self.groups)}


class ViewpointScorer(  # pyright: ignore[reportIncompatibleMethodOverride]  # SafeHubMixin types
    # _load_as_safetensor as nn.Module where the hub mixin uses a TypeVar — upstream clash, not ours
    nn.Module,
    SafeHubMixin,
    PyTorchModelHubMixin,
    library_name="canvit-pytorch",
    repo_url="https://github.com/m2b3/CanViT-PyTorch",
):
    """Frontend -> a ConvNeXt-V2 U-Net (encoder pools 32->1, decoder with skips back
    to 32, full-image-registered canvas map) -> readout at the candidate centres +
    a shared head -> one score per candidate [B, n_scale, cpa, cpa]
    (cpa=centers_per_axis).

    Hub I/O follows the release-stack pattern (PyTorchModelHubMixin + the loud
    strict-load SafeHubMixin): the __init__ kwargs ARE config.json, so
    `ViewpointScorer.from_pretrained(repo_or_dir)` reconstructs the exact
    architecture, and `save_pretrained`/`push_to_hub` publish it."""

    sb_grids: Tensor  # [n_scale, cpa, cpa, 2] per-scale candidate grid_sample grids (registered buffer)

    def __init__(
        self,
        *,
        canvas_dim: int,
        width: int,
        n_scale: int,
        scales: tuple[float, ...],
        centers_per_axis: int,
        block_layers: int,
        groups: tuple[str, ...] = FEATURE_GROUPS,
        dueling: bool = False,
        action_space: ActionSpace = "safebox",
    ):
        super().__init__()
        scales, groups = tuple(scales), tuple(groups)  # config.json round-trips tuples as lists
        assert len(scales) == n_scale, "need one entry in `scales` per scale channel"
        self.action_space: ActionSpace = action_space
        self.frontend = Frontend(canvas_dim, width, groups=groups)
        self.enc = nn.ModuleList([_stage(width, width, block_layers) for _ in range(6)])  # 32,16,8,4,2,1
        self.dec = nn.ModuleList([_stage(2 * width, width, block_layers) for _ in range(5)])  # ->2,4,8,16,32
        if action_space == "fixation":
            assert n_scale == 1, "fixation action space has no scale dimension (n_scale must be 1)"
            cand = fixation_candidates(centers_per_axis)
        else:
            cand = candidate_viewpoints(scales, centers_per_axis)
        sb = cand[..., :2].flip(-1)  # (x,y) for grid_sample
        self.register_buffer("sb_grids", sb)
        self.scale_emb = nn.Parameter(torch.zeros(n_scale, width, 1, 1))  # per-scale conditioning of the shared head
        self.head = nn.Sequential(_stage(width, width, block_layers), nn.Conv2d(width, 1, 1))
        # dueling: V(s) from the mean-pooled INPUT features (a scalar is not location-resolved, so pooled
        # conditioning is fine); the map becomes a mean-zero advantage. Argmax — hence deploy — is unchanged.
        ch = feature_channels(canvas_dim, groups)
        self.vhead = nn.Sequential(nn.Linear(ch, 64), nn.GELU(), nn.Linear(64, 1)) if dueling else None

    def forward(self, feats: Tensor) -> Tensor:
        h = self.frontend(feats)
        skips = []
        for i, enc in enumerate(self.enc):
            h = enc(h)
            skips.append(h)
            if i < 5:
                h = F.avg_pool2d(h, 2)
        d = skips[5]  # 1x1 global bottleneck
        for j in range(len(self.dec)):
            d = self.dec[j](torch.cat([F.interpolate(d, scale_factor=2, mode="nearest"), skips[4 - j]], dim=1))
        b = d.shape[0]
        vals = [
            self.head(F.grid_sample(d, self.sb_grids[k].expand(b, -1, -1, -1), align_corners=False) + self.scale_emb[k])
            for k in range(self.sb_grids.shape[0])
        ]
        q = torch.cat(vals, dim=1)  # [B, n_scale, cpa, cpa]
        if self.vhead is not None:  # Q = V(s) + mean-zero A(s,a)
            flat = q.reshape(b, -1)
            q = (flat - flat.mean(dim=1, keepdim=True) + self.vhead(feats.mean(dim=(2, 3)))).reshape(q.shape)
        return q

    def log(self) -> dict[str, float]:
        return self.frontend.log()
