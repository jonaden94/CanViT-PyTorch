"""Initialize a student ``ViTBackbone`` from DINOv3 (HuggingFace) teacher weights.

The student backbone mirrors DINOv3 ViT-B/16 (12 layers, 12 heads, head_dim 64,
RoPE theta 100, LayerScale, 4x MLP), so the whole transformer trunk transfers 1:1.
The only mechanical differences vs the HF DINOv3 layout:

* HF keeps separate ``q_proj / k_proj / v_proj``; the student fuses them into one
  ``attn.qkv`` -> we ``cat`` along dim 0.
* HF has no K bias (``key_bias=False``) -> the student's fused qkv bias gets a
  zero block in the K slice (which the student's ``_bias_mask`` masks anyway).
* ``patch_embed`` transfers only when patch sizes match (student vitb16 == teacher
  patch 16). For vitb8 the conv kernels differ -> patch_embed is left as-is.
* A ``*_modulate`` backbone has no LayerNorm affine / no LayerScale; those keys are
  simply absent from the target and skipped.

Only the backbone is touched; the surrounding CanViT modules are unrelated.
HF parameter names below match ``facebook/dinov3-vitb16-pretrain-lvd1689m``.
"""

import logging

import torch
from torch import nn

log = logging.getLogger(__name__)


def load_dinov3_weights_into_backbone(backbone: nn.Module, hf_teacher_model: nn.Module) -> dict:
    """Copy DINOv3 (HF) weights into ``backbone`` in place.

    Returns a summary dict: ``{"copied": int, "skipped": [(key, reason), ...],
    "n_blocks": int}``. Raises if the teacher/backbone dims are incompatible.
    """
    t = dict(hf_teacher_model.named_parameters())
    sd = backbone.state_dict()

    n_blocks = len(backbone.blocks)  # type: ignore[arg-type]
    # Sanity: teacher must have at least as many layers, and matching width.
    if f"model.layer.{n_blocks - 1}.norm1.weight" not in t:
        raise ValueError(
            f"Teacher has fewer than {n_blocks} transformer layers; cannot init "
            f"a {n_blocks}-block backbone from it (student/teacher size mismatch)."
        )
    dim = sd["blocks.0.attn.qkv.weight"].shape[1]
    t_dim = t["model.layer.0.norm1.weight"].shape[0]
    if dim != t_dim:
        raise ValueError(f"Width mismatch: backbone dim={dim} vs teacher dim={t_dim}.")

    new: dict[str, torch.Tensor] = {}
    skipped: list[tuple[str, str]] = []

    # --- patch embed (only if conv shapes match: student vitb16 <-> teacher patch16) ---
    pe_t = t.get("embeddings.patch_embeddings.weight")
    pe_s_shape = sd["patch_embed.proj.weight"].shape
    if pe_t is not None and tuple(pe_t.shape) == tuple(pe_s_shape):
        new["patch_embed.proj.weight"] = pe_t
        new["patch_embed.proj.bias"] = t["embeddings.patch_embeddings.bias"]
    else:
        skipped.append((
            "patch_embed.proj.*",
            f"conv shape {tuple(pe_s_shape)} != teacher "
            f"{tuple(pe_t.shape) if pe_t is not None else None} (patch size differs) — left as-is",
        ))

    # --- transformer blocks ---
    for i in range(n_blocks):
        s = f"blocks.{i}."
        tp = f"model.layer.{i}."

        new[s + "norm1.weight"] = t[tp + "norm1.weight"]
        new[s + "norm1.bias"] = t[tp + "norm1.bias"]
        new[s + "norm2.weight"] = t[tp + "norm2.weight"]
        new[s + "norm2.bias"] = t[tp + "norm2.bias"]

        # fuse q/k/v -> qkv (dim 0); K has no bias in DINOv3 -> zero block
        qw, kw, vw = (t[tp + f"attention.{n}_proj.weight"] for n in ("q", "k", "v"))
        new[s + "attn.qkv.weight"] = torch.cat([qw, kw, vw], dim=0)
        qb = t[tp + "attention.q_proj.bias"]
        vb = t[tp + "attention.v_proj.bias"]
        kb = t.get(tp + "attention.k_proj.bias", torch.zeros_like(qb))
        new[s + "attn.qkv.bias"] = torch.cat([qb, kb, vb], dim=0)

        new[s + "attn.proj.weight"] = t[tp + "attention.o_proj.weight"]
        new[s + "attn.proj.bias"] = t[tp + "attention.o_proj.bias"]

        new[s + "mlp.fc1.weight"] = t[tp + "mlp.up_proj.weight"]
        new[s + "mlp.fc1.bias"] = t[tp + "mlp.up_proj.bias"]
        new[s + "mlp.fc2.weight"] = t[tp + "mlp.down_proj.weight"]
        new[s + "mlp.fc2.bias"] = t[tp + "mlp.down_proj.bias"]

        # LayerScale — present only on non-modulated backbones
        if s + "ls1.gamma" in sd:
            new[s + "ls1.gamma"] = t[tp + "layer_scale1.lambda1"]
            new[s + "ls2.gamma"] = t[tp + "layer_scale2.lambda1"]

    # Drop any keys the target doesn't have (e.g. norm affine on a *_modulate backbone).
    dropped = [k for k in new if k not in sd]
    for k in dropped:
        skipped.append((k, "absent from backbone state_dict (modulated backbone?) — skipped"))
        del new[k]

    # Shape check before loading (fail loud rather than silently mis-init).
    bad = [(k, tuple(v.shape), tuple(sd[k].shape)) for k, v in new.items() if v.shape != sd[k].shape]
    if bad:
        raise ValueError(f"Shape mismatches while init-ing backbone from teacher: {bad}")

    missing, unexpected = backbone.load_state_dict(new, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"

    log.info(
        "Initialized backbone from DINOv3 teacher: copied %d/%d params over %d blocks%s",
        len(new), len(sd), n_blocks,
        (f"; skipped: {[k for k, _ in skipped]}" if skipped else " (full trunk + patch_embed)"),
    )
    return {"copied": len(new), "skipped": skipped, "n_blocks": n_blocks}
