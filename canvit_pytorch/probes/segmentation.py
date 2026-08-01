"""Segmentation probe: LN -> Dropout -> BN -> Conv1x1.

Architecture follows DINOv3. Loadable through ``PyTorchModelHubMixin`` from a
repo ID or local checkpoint directory.

Example::

    probe = SegmentationProbe.from_pretrained("<org>/probe-ade20k-...")
    logits = probe(features)  # [B, H, W, D] -> [B, num_classes, H, W]
"""

from typing import TYPE_CHECKING

from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from pathlib import Path

    import torch


class SegmentationProbe(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="canvit-pytorch",
    repo_url="https://github.com/m2b3/CanViT-PyTorch",
):
    """Linear segmentation head on spatial features.

    Input: [B, H, W, D] spatial features (canvas tokens or DINOv3 patches).
    Output: [B, num_classes, H, W] logits at the input spatial resolution.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_ln: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.dropout_p = dropout
        self.use_ln = use_ln
        self.ln: nn.Module = nn.LayerNorm(embed_dim) if use_ln else nn.Identity()
        self.bn = nn.BatchNorm2d(embed_dim)
        self.dropout = nn.Dropout2d(dropout)
        self.conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        assert self.conv.bias is not None
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """[B, H, W, D] -> [B, num_classes, H, W]."""
        B, H, W, D = x.shape
        assert D == self.embed_dim, f"Expected embed_dim={self.embed_dim}, got {D}"
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.dropout(x)
        x = self.bn(x)
        return self.conv(x)

    def predict(self, x: Tensor, target_size: tuple[int, int]) -> Tensor:
        """Forward + bilinear upsample to target resolution."""
        return F.interpolate(self(x), size=target_size, mode="bilinear", align_corners=False)

    @classmethod
    def from_checkpoint(
        cls, path: "str | Path", *, dropout: float = 0.1,
        map_location: "str | torch.device" = "cpu",
    ) -> "SegmentationProbe":
        """Load the probe out of a downstream training ``.pt`` (the ``head.*`` weights).

        The peer of ``from_pretrained``: same probe, different source. A segmentation
        training checkpoint holds the whole model — frozen backbone under ``canvit.*``
        plus this head under ``head.*`` — and the head is the only part that run produced.

        Shape is authoritative over flags: ``num_classes`` and ``embed_dim`` come from
        ``conv.weight`` and ``use_ln`` from whether ``ln.*`` was saved, so a mislabelled
        config cannot produce a probe that mismatches its own weights.

        ``dropout`` is the one thing a checkpoint does not record. It affects training
        only — a probe used as a frozen reward model or eval head runs in eval mode where
        Dropout2d is the identity — but it is stored in the probe config, so pass the
        source run's ``--cfg.dropout`` if it was not the default.
        """
        import torch

        raw = torch.load(path, map_location=map_location, weights_only=False)
        state = raw.get("model_state", raw.get("state_dict", raw))
        head = {k.removeprefix("head."): v for k, v in state.items() if k.startswith("head.")}
        if not head:
            raise KeyError(
                f"{path} has no 'head.*' weights — it is not a segmentation training "
                f"checkpoint (task={((raw.get('metadata') or {}).get('task'))!r})")
        num_classes, embed_dim = head["conv.weight"].shape[:2]
        probe = cls(embed_dim=int(embed_dim), num_classes=int(num_classes),
                    dropout=dropout, use_ln="ln.weight" in head)
        probe.load_state_dict(head, strict=True)
        return probe
