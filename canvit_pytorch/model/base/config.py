"""CanViT configuration."""

from dataclasses import dataclass, field
from typing import Literal

from canvit_pytorch.modulation import ViTModulationConfig
from canvit_pytorch.patcher import FoveatedPatcherConfig, PatcherName, SquarePatcherConfig


@dataclass
class CanViTConfig:
    """CanViT configuration."""

    rw_stride: int = 2
    enable_reads: bool = True
    n_backbone_registers: int = 5
    n_canvas_registers: int = 16
    canvas_num_heads: int = 8
    canvas_head_dim: int = 128
    enable_vpe: bool = True
    canvas_update_mode: Literal["additive", "convex"] = "additive"
    canvas_proj_mode: Literal["asymmetric", "full"] = "asymmetric"
    gate_bias_init: float | None = None
    # Patcher: "uniform" (default, current behavior), "foveated" (fovi-based) or
    # "square" (axis-aligned square patches, fovi-derived or strided; both
    # require the canvit-pytorch[fovi] extra). Per-patcher geometry params live
    # in `foveated_patcher` / `square_patcher` and are ignored unless the
    # matching `patcher_name` is selected.
    patcher_name: PatcherName = "uniform"
    foveated_patcher: FoveatedPatcherConfig = field(default_factory=FoveatedPatcherConfig)
    square_patcher: SquarePatcherConfig = field(default_factory=SquarePatcherConfig)
    # Per-token adaLN-style modulation of the transformer trunk (and optionally
    # the read/write cross-attn). Disabled by default (current behavior). When
    # `vit_modulation.enabled`, the backbone must be a "*_modulate" variant
    # (enforced at construction); the two settings go together.
    vit_modulation: ViTModulationConfig = field(default_factory=ViTModulationConfig)

    def __post_init__(self) -> None:
        is_convex = self.canvas_update_mode == "convex"
        has_gate = self.gate_bias_init is not None
        assert is_convex == has_gate, (
            f"Inconsistent config: canvas_update_mode={self.canvas_update_mode!r}, "
            f"gate_bias_init={self.gate_bias_init!r}"
        )

    @property
    def canvas_dim(self) -> int:
        return self.canvas_num_heads * self.canvas_head_dim
