"""CanViT + SegmentationProbe head, in one ``nn.Module``.

Construction goes through :meth:`CanViTForSemanticSegmentation.from_pretrained_with_probe`,
which loads a pretrained CanViT and a separately trained ``SegmentationProbe``
from repo IDs or local checkpoint directories so callers do not manage the two halves
separately. Mirrors :class:`CanViTForImageClassification`'s API shape.
"""

import logging
from typing import cast, get_args

from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from torch.nn import functional as F

from canvit_pytorch.backbone import BackboneName, create_backbone
from canvit_pytorch.model.base.config import CanViTConfig, rebuild_canvit_config
from canvit_pytorch.model.base.impl import CanViT, RecurrentState
from canvit_pytorch.model.hub_mixin import SafeHubMixin
from canvit_pytorch.model.pretraining.impl import CanViTForPretraining
from canvit_pytorch.model_source import load_pretraining, load_segmentation_probe
from canvit_pytorch.probes import SegmentationProbe
from canvit_pytorch.viewpoint import Viewpoint

log = logging.getLogger(__name__)


class CanViTForSemanticSegmentation(
    nn.Module,
    SafeHubMixin,
    PyTorchModelHubMixin,
    library_name="canvit-pytorch",
    repo_url="https://github.com/m2b3/CanViT-PyTorch",
):
    """:class:`CanViT` (``self.canvit``) + :class:`SegmentationProbe` (``self.head``).

    The wrapped CanViT is bare — no pretraining heads or standardizers,
    since those are unused for downstream segmentation.

    Example::

        seg = CanViTForSemanticSegmentation.from_pretrained_with_probe(
            pretrained_repo="<org>/canvitb16-add-vpe-pretrain-...",
            probe_repo="<org>/probe-ade20k-40k-s512-c32-in21k",
        ).eval()

        state = seg.init_state(batch_size=B, canvas_grid_size=32)
        logits, state = seg(glimpse=glimpse, state=state, viewpoint=vp)
        # logits: [B, num_classes, 32, 32]
    """

    def __init__(
        self,
        *,
        backbone_name: BackboneName,
        model_config: dict,
        num_classes: int,
        dropout: float = 0.1,
        use_ln: bool = True,
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
        # the pretraining wrapper). Stored on the INNER canvit because episode
        # runners receive `self.canvit`, not this wrapper, and read the value via
        # getattr to derive the training-matched glimpse crop size. ``None``
        # (checkpoints predating the field) -> runners fall back to the canonical
        # default of 8.
        self.glimpse_grid_size = glimpse_grid_size
        self.canvit.glimpse_grid_size = glimpse_grid_size
        # Head consumes canvas spatial tokens, so its dim is canvas_dim (not local_dim).
        D = self.canvit.canvas_dim
        self.head = SegmentationProbe(
            embed_dim=D,
            num_classes=num_classes,
            dropout=dropout,
            use_ln=use_ln,
        )

    @property
    def canvas_dim(self) -> int:
        return self.canvit.canvas_dim

    @property
    def num_classes(self) -> int:
        return self.head.num_classes

    def init_state(self, *, batch_size: int, canvas_grid_size: int) -> RecurrentState:
        return self.canvit.init_state(batch_size=batch_size, canvas_grid_size=canvas_grid_size)

    def forward(
        self, *, glimpse: Tensor, state: RecurrentState, viewpoint: Viewpoint,
    ) -> tuple[Tensor, RecurrentState]:
        """One CanViT step + head application.

        Returns ``(logits [B, num_classes, G, G], new_state)`` where ``G`` is the
        canvas grid size of ``state``. Use :meth:`predict` to also bilinearly
        upsample logits to a target spatial resolution.

        For CanViT-only execution without the segmentation head, call ``self.canvit(...)`` directly.
        For head-only on a cached state, call ``self.head(spatial_hwd)`` directly.
        """
        out = self.canvit(image=glimpse, state=state, viewpoint=viewpoint)
        spatial = self.canvit.get_spatial(out.state.canvas)  # [B, G*G, D]
        B, n_spatial, D = spatial.shape
        canvas_grid = int(n_spatial ** 0.5)
        assert canvas_grid * canvas_grid == n_spatial, (
            f"Canvas has {n_spatial} spatial tokens, not a perfect square — "
            f"init_state must be called with a valid canvas_grid_size."
        )
        return self.head(spatial.view(B, canvas_grid, canvas_grid, D)), out.state

    def predict(
        self,
        *,
        glimpse: Tensor,
        state: RecurrentState,
        viewpoint: Viewpoint,
        target_size: tuple[int, int],
    ) -> tuple[Tensor, RecurrentState]:
        """:meth:`forward` + bilinear upsample of logits to ``target_size``.

        Returns ``(logits [B, num_classes, *target_size], new_state)``.
        """
        logits, new_state = self(glimpse=glimpse, state=state, viewpoint=viewpoint)
        return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False), new_state

    @classmethod
    def _from_pretrained_backbone(
        cls,
        pretrained: CanViTForPretraining,   # HFHub subclasses this; a .pt gives the base
        *,
        num_classes: int,
        dropout: float,
        use_ln: bool,
    ) -> "CanViTForSemanticSegmentation":
        """Build the wrapper around a loaded pretrained CanViT (bare weights copied,
        pretraining-only modules dropped); the head is left as constructed."""
        cfg = pretrained.cfg
        assert pretrained.backbone_name in get_args(BackboneName), (
            f"Unknown ViT backbone: {pretrained.backbone_name!r}"
        )
        model = cls(
            backbone_name=cast(BackboneName, pretrained.backbone_name),
            model_config={k: v for k, v in vars(cfg).items() if k in CanViTConfig.__dataclass_fields__},
            num_classes=num_classes,
            dropout=dropout,
            use_ln=use_ln,
            glimpse_grid_size=pretrained.glimpse_grid_size,
        )

        # Copy bare-CanViT weights (drop pretraining-only modules)
        pretraining_only_prefixes = (
            "scene_cls_head.", "scene_patches_head.",
            "cls_standardizers.", "scene_standardizers.",
        )
        base_sd = {
            k: v for k, v in pretrained.state_dict().items()
            if not any(k.startswith(p) for p in pretraining_only_prefixes)
        }
        missing, unexpected = model.canvit.load_state_dict(base_sd, strict=False)
        assert not missing, f"Missing CanViT keys: {missing}"
        assert not unexpected, f"Unexpected CanViT keys: {unexpected}"
        return model

    @classmethod
    def from_pretrained_with_probe(
        cls,
        *,
        pretrained_repo: str,
        probe_repo: str,
    ) -> "CanViTForSemanticSegmentation":
        """Load a pretrained CanViT + a published seg probe; bundle them into one model.

        Pretraining-only modules (``scene_cls_head``, ``scene_patches_head``,
        ``cls_standardizers``, ``scene_standardizers``) on the loaded model
        are discarded.
        """
        log.info("Loading pretrained CanViT from %s", pretrained_repo)
        pretrained = load_pretraining(pretrained_repo)
        D = pretrained.canvas_dim

        log.info("Loading probe from %s", probe_repo)
        probe = load_segmentation_probe(probe_repo)
        assert probe.embed_dim == D, (
            f"Probe expects embed_dim={probe.embed_dim} but the CanViT produces "
            f"canvas_dim={D}. Probe was trained for a different model variant."
        )

        model = cls._from_pretrained_backbone(
            pretrained, num_classes=probe.num_classes, dropout=probe.dropout_p, use_ln=probe.use_ln
        )

        # Copy seg head weights
        missing, unexpected = model.head.load_state_dict(probe.state_dict(), strict=True)
        assert not missing and not unexpected, (
            f"Probe state_dict mismatch: missing={missing}, unexpected={unexpected}"
        )

        log.info(
            "Constructed CanViTForSemanticSegmentation: %d classes, canvas_dim=%d, dropout=%s, use_ln=%s",
            probe.num_classes, D, probe.dropout_p, probe.use_ln,
        )
        return model

    @classmethod
    def from_checkpoint(cls, path, *, map_location: str = "cpu") -> "CanViTForSemanticSegmentation":
        """Load a whole segmentation model out of an ``ade20k`` training ``.pt``.

        Needed because such a checkpoint may be the ONLY place its weights exist: a
        *finetune* run updates the backbone, so the source ``model_repo`` no longer
        describes the model that was trained. (For a *probe* run the backbone is frozen, so
        ``from_pretrained_with_probe(model_repo, <that .pt>)`` is equivalent and needs no
        arch in the checkpoint.)

        Requires a checkpoint whose ``model_config`` carries the architecture -- written by
        the trainer from this class's own constructor arguments. Older ade20k checkpoints
        recorded only a ``model_repo`` path and cannot be loaded this way; use that repo
        plus the probe instead.
        """
        import torch

        raw = torch.load(path, map_location=map_location, weights_only=False)
        mc = raw.get("model_config") or {}
        if "canvit" not in mc or "backbone_name" not in mc:
            raise KeyError(
                f"{path} does not record its architecture (model_config keys: "
                f"{sorted(mc)}). It predates self-describing downstream checkpoints -- "
                f"load it as model_repo={mc.get('model_repo')!r} + this file as the probe.")
        model = cls(
            backbone_name=cast(BackboneName, mc["backbone_name"]),
            model_config=mc["canvit"],
            num_classes=mc["num_classes"],
            dropout=mc.get("dropout", 0.1),
            use_ln=mc.get("use_ln", True),
            glimpse_grid_size=mc.get("glimpse_grid_size"),
        )
        model.load_state_dict(raw["model_state"], strict=True)
        return model

    @classmethod
    def from_pretrained_with_new_probe(
        cls,
        *,
        pretrained_repo: str,
        num_classes: int,
        dropout: float = 0.1,
        use_ln: bool = True,
    ) -> "CanViTForSemanticSegmentation":
        """Load a pretrained CanViT and attach a FRESH (randomly initialized) probe.

        The construction path for probe *training* (CanViT-pretrain's ade20k task):
        same bare-backbone loading as :meth:`from_pretrained_with_probe`, but the
        head starts untrained instead of from a published probe checkpoint.
        """
        log.info("Loading pretrained CanViT from %s (fresh %d-class probe)", pretrained_repo, num_classes)
        pretrained = load_pretraining(pretrained_repo)
        return cls._from_pretrained_backbone(
            pretrained, num_classes=num_classes, dropout=dropout, use_ln=use_ln
        )
