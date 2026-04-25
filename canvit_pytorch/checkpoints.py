"""Checkpoint repo-id resolution.

Single source of truth for the HF org and the local-checkpoint env var.
If ``$CANVIT_CHECKPOINTS`` is set and contains a directory matching the
sanitized form of the canonical repo-id (``<root>/<HF_ORG>--<name>``),
:func:`resolve_repo` returns that local path; otherwise it returns the
canonical Hub repo-id. Both shapes are accepted by ``from_pretrained``.
"""

import os
from pathlib import Path

HF_ORG = "canvit"
_CHECKPOINT_ROOT = os.environ.get("CANVIT_CHECKPOINTS")


def resolve_repo(name: str) -> str:
    canonical = f"{HF_ORG}/{name}"
    if not _CHECKPOINT_ROOT:
        return canonical
    local = Path(_CHECKPOINT_ROOT) / canonical.replace("/", "--")
    return str(local) if local.is_dir() else canonical
