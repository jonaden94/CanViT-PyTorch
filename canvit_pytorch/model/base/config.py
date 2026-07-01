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
    # Self-attention over the canvas (memory) tokens, applied once per glimpse
    # after that glimpse's writes. `n_canvas_self_attn_blocks` stacked blocks run
    # over the full canvas [registers | spatial]; RoPE rotates only the spatial
    # tokens (registers are positionless, same convention as read/write).
    # Disabled by default (0 blocks -> current behavior, no new params).
    # `canvas_self_attn_mlp_ratios` gives the per-block MLP hidden ratio (× canvas_dim);
    # it MUST have length == n_canvas_self_attn_blocks. A ratio of 0 -> attention-only
    # (no MLP) for that block, e.g. [2, 0] = a 2×-MLP block then an attention-only block.
    n_canvas_self_attn_blocks: int = 0
    canvas_self_attn_mlp_ratios: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        is_convex = self.canvas_update_mode == "convex"
        has_gate = self.gate_bias_init is not None
        assert is_convex == has_gate, (
            f"Inconsistent config: canvas_update_mode={self.canvas_update_mode!r}, "
            f"gate_bias_init={self.gate_bias_init!r}"
        )
        assert len(self.canvas_self_attn_mlp_ratios) == self.n_canvas_self_attn_blocks, (
            f"canvas_self_attn_mlp_ratios must have length n_canvas_self_attn_blocks="
            f"{self.n_canvas_self_attn_blocks}, got {self.canvas_self_attn_mlp_ratios}"
        )

    @property
    def canvas_dim(self) -> int:
        return self.canvas_num_heads * self.canvas_head_dim
