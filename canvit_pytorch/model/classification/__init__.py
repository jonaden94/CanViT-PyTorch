"""CanViT for image classification."""

import logging
from pathlib import Path
from typing import cast, get_args

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from safetensors.torch import load_file
from torch import Tensor, nn

from canvit_pytorch.backbone import BackboneName, create_backbone
from canvit_pytorch.model.hub_mixin import SafeHubMixin
from canvit_pytorch.model.base.config import (
    rebuild_canvit_config,
    serialize_canvit_config,
)
from canvit_pytorch.model.base.impl import CanViT, RecurrentState
from canvit_pytorch.model_source import load_pretraining
from canvit_pytorch.viewpoint import Viewpoint

log = logging.getLogger(__name__)


def fuse_probe(
    *,
    W_proj: Tensor,
    b_proj: Tensor,
    mu: Tensor,
    sigma: Tensor,
    W_probe: Tensor,
    b_probe: Tensor,
) -> tuple[Tensor, Tensor]:
    """Fuse proj → destandardize → probe into a single linear transform.

    The pretrained eval chain after LayerNorm is three affine transforms::

        s = W_proj @ z + b_proj
        d = σ ⊙ s + μ
        logits = W_probe @ d + b_probe

    Since affine ∘ affine = affine, these collapse into::

        W_fused = W_probe @ diag(σ) @ W_proj
        b_fused = W_probe @ (σ ⊙ b_proj + μ) + b_probe

    Returns (W_fused [n_classes, D], b_fused [n_classes]).
    """
    teacher_dim, D = W_proj.shape
    n_classes = W_probe.shape[0]
    assert b_proj.shape == (teacher_dim,)
    assert mu.shape == (teacher_dim,) and sigma.shape == (teacher_dim,)
    assert W_probe.shape == (n_classes, teacher_dim) and b_probe.shape == (n_classes,)

    B_mat = sigma.unsqueeze(1) * W_proj
    assert B_mat.shape == (teacher_dim, D)
    b_mid = sigma * b_proj + mu
    assert b_mid.shape == (teacher_dim,)
    W_fused = W_probe @ B_mat
    assert W_fused.shape == (n_classes, D)
    b_fused = W_probe @ b_mid + b_probe
    assert b_fused.shape == (n_classes,)
    return W_fused, b_fused


class CanViTForImageClassification(
    nn.Module,
    SafeHubMixin,
    PyTorchModelHubMixin,
    library_name="canvit-pytorch",
    repo_url="https://github.com/m2b3/CanViT-PyTorch",
):
    """:class:`CanViT` + LN → Linear classification head.

    Wraps a bare :class:`CanViT` (no pretraining heads, no standardizers).
    The classification head is always LN(D) → Linear(D, n_classes).

    Example::

        # From a fused checkpoint:
        clf = CanViTForImageClassification.from_pretrained("<org>/<repo>").eval()

        # From a pretrained CanViT + probe (fuses at construction time):
        clf = CanViTForImageClassification.from_pretrained_with_probe(
            pretrained_repo="<org>/canvitb16-add-vpe-pretrain-...",
            probe_repo="<org>/dinov3-vitb16-...-linear-clf-probe",
        ).eval()

        # Both have the same forward:
        state = clf.init_state(batch_size=B, canvas_grid_size=32)
        logits, state = clf(glimpse=glimpse, state=state, viewpoint=vp)
    """

    def __init__(
        self,
        *,
        backbone_name: BackboneName,
        model_config: dict,
        n_classes: int,
        glimpse_grid_size: int | None = None,
    ):
        super().__init__()
        # Filters pretraining-only extras (teacher_dim) AND restores the nested
        # dataclasses that asdict() flattened — a foveated config arrives with
        # foveated_patcher as a plain dict and fails deep in the patcher otherwise.
        cfg = rebuild_canvit_config(model_config)
        self.canvit = CanViT(backbone=create_backbone(backbone_name), cfg=cfg)
        # Kept so a training checkpoint of this wrapper can record enough to rebuild
        # itself (the trainers' `model_config`); __init__ takes it but nothing else
        # retained it, leaving the arch recoverable only from the source HF repo.
        self.backbone_name = backbone_name
        # Glimpse token-grid side the model was trained with (see
        # CanViTForPretrainingHFHub). Stored on the INNER canvit because episode
        # runners receive `self.canvit`, not this wrapper, and read the value via
        # getattr to derive the training-matched glimpse crop size. ``None``
        # (checkpoints predating the field) -> runners fall back to the canonical
        # default of 8.
        self.glimpse_grid_size = glimpse_grid_size
        self.canvit.glimpse_grid_size = glimpse_grid_size
        D = self.canvit.local_dim
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, n_classes)

    @property
    def local_dim(self) -> int:
        """Embedding dimension of the CanViT local stream and head input."""
        return self.canvit.local_dim

    @property
    def n_classes(self) -> int:
        return self.head.out_features

    def init_state(self, *, batch_size: int, canvas_grid_size: int) -> RecurrentState:
        return self.canvit.init_state(batch_size=batch_size, canvas_grid_size=canvas_grid_size)

    def forward(
        self, *, glimpse: Tensor, state: RecurrentState, viewpoint: Viewpoint,
    ) -> tuple[Tensor, RecurrentState]:
        """Returns (logits [B, n_classes], new_state).

        For CanViT-only execution without the classification head, call ``self.canvit(...)`` directly.
        For head-only on a cached CLS token, call ``self.head(self.norm(cls))``.
        """
        out = self.canvit(image=glimpse, state=state, viewpoint=viewpoint)
        cls = out.state.recurrent_cls[:, 0].float()
        return self.head(self.norm(cls)), out.state

    @classmethod
    def from_pretrained_with_probe(
        cls,
        *,
        pretrained_repo: str,
        probe_repo: str,
        canvas_grid: int = 32,
    ) -> "CanViTForImageClassification":
        """Load a pretrained CanViT, fuse proj → destandardize → probe into LN → Linear.

        Loads the full pretraining model temporarily to extract fusion ingredients
        (scene_cls_head, standardizers, probe), then copies only the base CanViT
        weights into the classifier. Pretraining heads are discarded.

        See :func:`fuse_probe` for the algebra.
        """
        log.info("Loading pretrained model from %s", pretrained_repo)
        pretrained = load_pretraining(pretrained_repo)
        D = pretrained.local_dim

        log.info("Loading probe from %s", probe_repo)
        probe_path = Path(probe_repo)
        probe_st = probe_path / "model.safetensors" if probe_path.is_dir() else hf_hub_download(probe_repo, "model.safetensors")
        probe_sd = load_file(probe_st)

        # Validate probe/projection compatibility — the probe consumes the
        # output of scene_cls_head["proj"] (a Linear layer in the pretrained
        # CanViT's CLS head, NOT the inner ViT backbone).
        proj = pretrained.scene_cls_head["proj"]
        assert isinstance(proj, nn.Linear)
        probe_in_dim = probe_sd["weight"].shape[1]
        assert probe_in_dim == proj.out_features, (
            f"Probe/projection dim mismatch: probe expects {probe_in_dim}, "
            f"scene_cls_head proj produces {proj.out_features}"
        )

        cls_std, _ = pretrained.standardizers(canvas_grid)
        assert cls_std.initialized, "CLS standardizer not initialized — wrong canvas_grid?"

        W_fused, b_fused = fuse_probe(
            W_proj=proj.weight.data,
            b_proj=proj.bias.data,
            mu=cls_std.mean.squeeze(0),
            sigma=(cls_std.var.squeeze(0) + cls_std.eps).sqrt(),
            W_probe=probe_sd["weight"],
            b_probe=probe_sd["bias"],
        )

        # Build classifier wrapping a bare CanViT (no pretraining heads)
        n_classes = W_fused.shape[0]
        cfg = pretrained.cfg
        assert pretrained.backbone_name in get_args(BackboneName), f"Unknown backbone: {pretrained.backbone_name!r}"
        model = cls(
            backbone_name=cast(BackboneName, pretrained.backbone_name),
            model_config=serialize_canvit_config(cfg),
            n_classes=n_classes,
            glimpse_grid_size=pretrained.glimpse_grid_size,
        )

        # Copy base CanViT weights from pretrained (excluding pretraining heads)
        base_sd = {k: v for k, v in pretrained.state_dict().items()
                   if not any(k.startswith(pfx) for pfx in
                              ("scene_cls_head.", "scene_patches_head.",
                               "cls_standardizers.", "scene_standardizers."))}
        missing, unexpected = model.canvit.load_state_dict(base_sd, strict=False)
        assert not missing, f"Missing CanViT keys: {missing}"
        assert not unexpected, f"Unexpected CanViT keys: {unexpected}"

        # Set fused head weights
        model.head.weight.data.copy_(W_fused)
        model.head.bias.data.copy_(b_fused)
        pretrained_norm = pretrained.scene_cls_head["norm"]
        assert isinstance(pretrained_norm, nn.LayerNorm)
        model.norm.weight.data.copy_(pretrained_norm.weight.data)
        model.norm.bias.data.copy_(pretrained_norm.bias.data)

        log.info("Fused classifier: LN(%d) → Linear(%d, %d), pretraining heads discarded", D, D, n_classes)
        return model

    @classmethod
    def from_checkpoint(cls, path, *, map_location: str = "cpu") -> "CanViTForImageClassification":
        """Load a whole classifier out of an ``in1k`` training ``.pt``.

        An in1k FINETUNE updates the backbone, so its checkpoint is the only place those
        weights exist -- the source ``model_repo`` describes the model it *started* from,
        not the one that was trained. This is the peer of ``from_pretrained``; the same
        checkpoint can also be exported with ``canvit_train.checkpoint.to_hf``.

        Requires a checkpoint whose ``model_config`` carries the architecture. Older in1k
        checkpoints recorded only a ``model_repo`` path; export those with ``to_hf`` while
        the pointer still resolves.
        """
        import torch

        raw = torch.load(path, map_location=map_location, weights_only=False)
        mc = raw.get("model_config") or {}
        if "canvit" not in mc or "backbone_name" not in mc:
            raise KeyError(
                f"{path} does not record its architecture (model_config keys: "
                f"{sorted(mc)}). It predates self-describing downstream checkpoints -- "
                f"convert it with canvit_train.checkpoint.to_hf instead.")
        model = cls(
            backbone_name=cast(BackboneName, mc["backbone_name"]),
            model_config=mc["canvit"],
            n_classes=mc["n_classes"],
            glimpse_grid_size=mc.get("glimpse_grid_size"),
        )
        model.load_state_dict(raw["model_state"], strict=True)
        return model

    @classmethod
    def from_pretrained_with_new_head(
        cls,
        *,
        pretrained_repo: str,
        n_classes: int,
    ) -> "CanViTForImageClassification":
        """Load a pretrained CanViT and attach a FRESH (randomly initialized) LN → Linear
        classification head — the construction path for classifier *training*
        (CanViT-pretrain's in1k task, P5). Same bare-backbone copy as
        :meth:`from_pretrained_with_probe`, but the head/norm start untrained instead of
        fused from a published probe checkpoint (mirrors the segmentation model's
        ``from_pretrained_with_new_probe``)."""
        log.info("Loading pretrained CanViT from %s (fresh %d-class head)", pretrained_repo, n_classes)
        pretrained = load_pretraining(pretrained_repo)
        cfg = pretrained.cfg
        assert pretrained.backbone_name in get_args(BackboneName), f"Unknown backbone: {pretrained.backbone_name!r}"
        model = cls(
            backbone_name=cast(BackboneName, pretrained.backbone_name),
            model_config=serialize_canvit_config(cfg),
            n_classes=n_classes,
            glimpse_grid_size=pretrained.glimpse_grid_size,
        )
        base_sd = {k: v for k, v in pretrained.state_dict().items()
                   if not any(k.startswith(pfx) for pfx in
                              ("scene_cls_head.", "scene_patches_head.",
                               "cls_standardizers.", "scene_standardizers."))}
        missing, unexpected = model.canvit.load_state_dict(base_sd, strict=False)
        assert not missing, f"Missing CanViT keys: {missing}"
        assert not unexpected, f"Unexpected CanViT keys: {unexpected}"
        log.info("Fresh classifier: LN(%d) → Linear(%d, %d) over pretrained CanViT",
                 model.local_dim, model.local_dim, n_classes)
        return model
