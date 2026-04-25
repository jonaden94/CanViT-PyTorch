"""CanViT: Dual-stream vision transformer with canvas cross-attention."""

from canvit_pytorch.backbone import BackboneName, ViTBackbone, create_backbone
from canvit_pytorch.checkpoints import CANVIT_REPO_ROOT, resolve_canvit_repo
from canvit_pytorch.model import (
    CanViT,
    CanViTConfig,
    CanViTForImageClassification,
    CanViTForPretraining,
    CanViTForPretrainingConfig,
    CanViTForPretrainingHFHub,
    CanViTForSemanticSegmentation,
    CanViTOutput,
    RecurrentState,
    fuse_probe,
)
from canvit_pytorch.probes import SegmentationProbe
from canvit_pytorch.standardizers import CLSStandardizer, PatchStandardizer, PositionAwareStandardizer
from canvit_pytorch.viewpoint import Viewpoint, sample_at_viewpoint
from canvit_pytorch.vpe import VPEEncoder

__all__ = [
    "BackboneName",
    "CLSStandardizer",
    "CanViT",
    "CanViTConfig",
    "CanViTForImageClassification",
    "CanViTForPretraining",
    "CanViTForPretrainingConfig",
    "CanViTForPretrainingHFHub",
    "CanViTForSemanticSegmentation",
    "CANVIT_REPO_ROOT",
    "CanViTOutput",
    "PatchStandardizer",
    "PositionAwareStandardizer",
    "RecurrentState",
    "SegmentationProbe",
    "VPEEncoder",
    "Viewpoint",
    "ViTBackbone",
    "create_backbone",
    "fuse_probe",
    "resolve_canvit_repo",
    "sample_at_viewpoint",
]
