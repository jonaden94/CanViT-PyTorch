"""CanViT for pretraining implementation."""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch
from torch import Tensor, nn

from canvit_pytorch.backbone import ViTBackbone, create_backbone
from canvit_pytorch.model.base import CanViT, CanViTOutput, RecurrentState
from canvit_pytorch.model.base.config import CanViTConfig
from canvit_pytorch.rope import RoPE
from canvit_pytorch.standardizers import CLSStandardizer, PatchStandardizer
from canvit_pytorch.viewpoint import Viewpoint

log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class CanViTForPretrainingConfig(CanViTConfig):
    """Model configuration for CanViTForPretraining."""

    teacher_dim: int


@dataclass
class CanViTForPretrainingOutput:
    """Output of ``CanViTForPretraining.forward``.

    Bundles the base CanViT output with the prediction-head outputs so that
    *all* gradient-relevant computation happens inside one ``forward`` call.
    Critical under DDP: gradient sync only fires reliably for parameters whose
    autograd path runs through the DDP-wrapped forward; calling the heads via
    ``model.module.predict_*`` outside ``forward`` produces per-rank-only
    gradients on the head parameters (CanViT issue: heads weren't averaged
    across ranks, manifest as √N grad-norm scaling on head params only).
    """

    state: RecurrentState
    local_patches: Tensor
    vpe: Tensor | None
    scene_pred: Tensor
    cls_pred: Tensor


def _init_ln_weight(ln: nn.LayerNorm, dim: int) -> None:
    ln.weight.data.fill_(1.0 / math.sqrt(dim))


class CanViTForPretraining(CanViT):
    """CanViT with prediction heads for pretraining.

    Predicts full-image teacher features from canvas (shared targets across timesteps).
    Loss computation is handled by the trainer, not this class.
    """

    def __init__(
        self,
        *,
        backbone: ViTBackbone,
        cfg: CanViTForPretrainingConfig,
        backbone_name: str,
        canvas_patch_grid_sizes: list[int],
    ) -> None:
        super().__init__(backbone=backbone, cfg=cfg)

        canvas_dim = cfg.canvas_dim
        local_dim = backbone.embed_dim
        teacher_dim = cfg.teacher_dim

        self.scene_patches_head = nn.ModuleDict({
            "norm": nn.LayerNorm(canvas_dim),
            "proj": nn.Linear(canvas_dim, teacher_dim),
        })
        self.scene_cls_head = nn.ModuleDict({
            "norm": nn.LayerNorm(local_dim),
            "proj": nn.Linear(local_dim, teacher_dim),
        })

        patches_norm = self.scene_patches_head["norm"]
        cls_norm = self.scene_cls_head["norm"]
        assert isinstance(patches_norm, nn.LayerNorm) and isinstance(cls_norm, nn.LayerNorm)
        _init_ln_weight(patches_norm, canvas_dim)
        _init_ln_weight(cls_norm, local_dim)

        # Standardizers keyed by canvas grid size (spatial tokens only, excludes registers)
        self.cls_standardizers: nn.ModuleDict = nn.ModuleDict()
        self.scene_standardizers: nn.ModuleDict = nn.ModuleDict()

        self.backbone_name = backbone_name
        for g in canvas_patch_grid_sizes:
            self.standardizers(g)  # get-or-create, ensures they exist in state_dict

    @property
    def canvas_patch_grid_sizes(self) -> list[int]:
        """Canvas grid sizes (spatial side lengths in tokens) for which standardizers exist."""
        return [int(k) for k in self.cls_standardizers.keys()]

    def standardizers(self, grid_size: int) -> tuple[CLSStandardizer, PatchStandardizer]:
        """Get standardizers for a grid size, creating if needed."""
        key = str(grid_size)
        if key not in self.cls_standardizers:
            cfg = self.cfg
            assert isinstance(cfg, CanViTForPretrainingConfig)
            self.cls_standardizers[key] = CLSStandardizer(embed_dim=cfg.teacher_dim)
            self.scene_standardizers[key] = PatchStandardizer(grid_size=grid_size, embed_dim=cfg.teacher_dim)
        return self.cls_standardizers[key], self.scene_standardizers[key]  # type: ignore[return-value]

    @classmethod
    def from_checkpoint(cls, path: Path | str, *, map_location: str | torch.device = "cpu") -> Self:
        """Load from local .pt checkpoint file."""
        log.info("Loading checkpoint from %s (map_location=%s)", path, map_location)
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        log.info("backbone_name=%s, canvas_patch_grid_sizes=%s", ckpt["backbone_name"], ckpt["canvas_patch_grid_sizes"])
        model = cls(
            backbone=create_backbone(ckpt["backbone_name"]),
            cfg=CanViTForPretrainingConfig(**ckpt["model_config"]),
            backbone_name=ckpt["backbone_name"],
            canvas_patch_grid_sizes=ckpt["canvas_patch_grid_sizes"],
        )
        model.load_state_dict(ckpt["state_dict"])
        log.info("Loaded %d parameters", sum(p.numel() for p in model.parameters()))
        return model

    def predict_teacher_scene(self, canvas: Tensor) -> Tensor:
        """Predict full-image teacher patch features from canvas spatial tokens.

        Kept for callers that want the prediction without running a glimpse
        forward (e.g. visualization of the initial-state scene, validation).
        For training, prefer ``forward(...)`` which produces ``scene_pred``
        inside the DDP-wrapped forward so the head's gradient is AllReduced.
        """
        x = self.get_spatial(canvas)
        return self.scene_patches_head["proj"](self.scene_patches_head["norm"](x)).contiguous()

    def predict_scene_teacher_cls(self, global_cls: Tensor) -> Tensor:
        """Predict full-image teacher CLS from recurrent global CLS.

        See ``predict_teacher_scene`` for the DDP caveat.

        Args:
            global_cls: [B, 1, local_dim] recurrent global CLS token
        """
        B, one, D = global_cls.shape
        assert one == 1, f"Expected global_cls shape [B, 1, D], got {global_cls.shape}"
        x = global_cls[:, 0]
        return self.scene_cls_head["proj"](self.scene_cls_head["norm"](x)).contiguous()

    def forward(  # type: ignore[override]
        self,
        *,
        glimpse: Tensor,
        state: RecurrentState,
        viewpoint: Viewpoint,
        canvas_grid_size: int | None = None,
        canvas_rope: RoPE | None = None,
    ) -> CanViTForPretrainingOutput:
        """Glimpse forward + prediction heads in one call.

        Running the heads INSIDE forward (rather than as separate
        ``self.predict_*`` calls from the trainer) is what gets their
        parameters into the autograd graph that DDP's Reducer instruments
        each iteration. Without this, DDP's per-iteration bucket setup
        misses the head params and their gradients are not AllReduced
        across ranks — every rank's heads then drift independently and
        rank 0's heads see effectively per-rank (un-averaged, ~√N larger)
        gradient noise.
        """
        base: CanViTOutput = super().forward(
            glimpse=glimpse,
            state=state,
            viewpoint=viewpoint,
            canvas_grid_size=canvas_grid_size,
            canvas_rope=canvas_rope,
        )
        scene_pred = self.predict_teacher_scene(base.state.canvas)
        cls_pred = self.predict_scene_teacher_cls(base.state.recurrent_cls)
        return CanViTForPretrainingOutput(
            state=base.state,
            local_patches=base.local_patches,
            vpe=base.vpe,
            scene_pred=scene_pred,
            cls_pred=cls_pred,
        )
