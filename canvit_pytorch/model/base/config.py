"""CanViT configuration."""

import dataclasses
import typing
from dataclasses import dataclass, field
from typing import Any, Literal

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


def _coerce(tp, value):
    """Recursively rebuild a dataclass of type ``tp`` from ``value`` when ``value`` is a
    dict (as produced by ``asdict`` at save time), coercing any nested dataclass-typed
    fields at any depth. Non-dict values and non-dataclass targets pass through unchanged.

    Faithful-or-loud-fail by design — it never fabricates values:
      * EVERY key in ``value`` is passed to the dataclass constructor, so a key this code
        does not know (e.g. a config field added to the model *after* this loader was
        written, loaded without updating the loader) raises a loud ``TypeError`` rather
        than being silently dropped. This is what prevents a future, un-updated eval from
        silently evaluating a model that diverges from pretraining.
      * A field present in the dataclass but ABSENT from ``value`` takes the dataclass
        default (a checkpoint predating that field). This is the only place defaults enter,
        so new fields MUST default to backward-compatible behavior — the one invariant no
        loader can enforce for you (the strict state_dict load is the backstop for anything
        affecting weights).
    """
    if not isinstance(value, dict):
        return value
    if typing.get_origin(tp) is not None:  # Optional[X] / Union[...] -> the dataclass member
        tp = next((a for a in typing.get_args(tp) if dataclasses.is_dataclass(a)), None)
    if tp is None or not dataclasses.is_dataclass(tp):
        return value
    try:
        hints = typing.get_type_hints(tp)
    except Exception:  # noqa: BLE001 — unresolved annotations: fall back to raw field types
        hints = {f.name: f.type for f in dataclasses.fields(tp)}
    # Pass ALL keys (recursing into known dataclass-typed fields); an unknown key
    # reaches tp(**...) and raises TypeError — never silently dropped.
    return tp(**{k: (_coerce(hints[k], v) if k in hints else v) for k, v in value.items()})


def coerce_nested_configs(model_config: dict) -> dict[str, Any]:
    """Turn the dataclass-valued entries of a serialized config back into dataclasses.

    ``asdict()`` flattens nested dataclasses to dicts on save, but ``SomeConfig(**d)`` only
    builds SHALLOWLY — so ``foveated_patcher``, ``square_patcher`` and ``vit_modulation``
    would arrive as dicts and blow up on first attribute access, deep inside the patcher
    (``'dict' object has no attribute 'hidden_dims_patch_embed'``). Every loader that
    rebuilds a config from a dict must go through here.

    Gated to only the ACTIVE patcher + vit_modulation, so an existing checkpoint
    instantiates a byte-for-byte identical config.
    """
    if (model_config.get("patcher_name") == "foveated"
            and isinstance(model_config.get("foveated_patcher"), dict)):
        model_config = {**model_config,
                        "foveated_patcher": _coerce(FoveatedPatcherConfig, model_config["foveated_patcher"])}
    if (model_config.get("patcher_name") == "square"
            and isinstance(model_config.get("square_patcher"), dict)):
        model_config = {**model_config,
                        "square_patcher": _coerce(SquarePatcherConfig, model_config["square_patcher"])}
    if isinstance(model_config.get("vit_modulation"), dict):
        model_config = {**model_config,
                        "vit_modulation": _coerce(ViTModulationConfig, model_config["vit_modulation"])}
    return model_config


def serialize_canvit_config(cfg: "CanViTConfig") -> dict[str, Any]:
    """The inverse of :func:`rebuild_canvit_config`: a JSON-encodable dict of ``cfg``.

    Two callers need this and both are easy to get wrong by reaching for ``vars(cfg)``:
    writing a config.json, and passing ``model_config=`` to a downstream wrapper that will
    later be published with ``PyTorchModelHubMixin.save_pretrained``. The mixin records only
    the ``__init__`` kwargs it can JSON-encode, so live nested dataclasses are **silently
    dropped** — the published dir then has no ``model_config`` and ``from_pretrained`` fails
    with "missing 1 required keyword-only argument". ``asdict`` flattens the nesting, which
    is the form ``rebuild_canvit_config`` expects anyway.

    Accepts a subclass (``CanViTForPretrainingConfig``) and drops its extra fields, since a
    downstream wrapper's ``CanViTConfig`` does not accept ``teacher_dim``.
    """
    known = CanViTConfig.__dataclass_fields__
    return {k: v for k, v in dataclasses.asdict(cfg).items() if k in known}


def rebuild_canvit_config(model_config: dict) -> "CanViTConfig":
    """Rebuild a :class:`CanViTConfig` from its serialized dict form.

    Filters to known ``CanViTConfig`` fields first, because a config.json written for a
    PRETRAINING model carries extras this class does not accept (``teacher_dim``), then
    restores the nested dataclasses. Used by the downstream wrappers
    (``CanViTForSemanticSegmentation`` / ``CanViTForImageClassification``), which are handed
    a plain dict by ``from_pretrained`` and by their ``from_checkpoint``.
    """
    known = CanViTConfig.__dataclass_fields__
    return CanViTConfig(**coerce_nested_configs(
        {k: v for k, v in model_config.items() if k in known}))
