# CanViT-PyTorch is ARCHIVED (2026-09-03) — read-only reference

The model lives in the **`canvit`** repo (formerly `CanViT-train`) as the **`canvit.core`**
subpackage. There is no live `canvit_pytorch` package anywhere. Do not edit this clone, and do
not run it in preference to the merged version. Full record:
`../canvit/unification_docs/21-core-merge.md`.

**The merge changed no number.** All 82 evaluation scalars across four configs were bit-identical
before and after, on the same GPU (§9–§10 of that doc). It was a packaging change and nothing else.

## Why this clone still exists, and must keep existing

**116 launchers under `../canvit/slurm/` `git archive` a core commit out of this repo's `.git`.**
Deleting this directory makes 116 historical runs unreproducible. It is retained deliberately and
indefinitely, exactly like `CanViT-specialize` and `CanViT-PyTorch-RL`.

The live `slurm/runs/` launchers pin four distinct commits here: `d616b7b`, `1f5121b`, `017ce9b`,
`3277048`.

## Where things went

The rule is uniform — **`canvit_pytorch.X` → `canvit.core.X`**:

```python
from canvit_pytorch import CanViTForPretrainingHFHub      # old
from canvit.core import CanViTForPretrainingHFHub          # new

from canvit_pytorch.patcher import FoveatedPatcher         # old
from canvit.core.patcher import FoveatedPatcher            # new
```

Three things did not follow that rule:

| was here | is now |
|---|---|
| `bench/pt/` | `../canvit/bench/pt/` (repo root, not inside the package) |
| `tests/` | `../canvit/canvit/core/tests/` |
| `test_data/`, `demos/`, `assets/` | `../canvit/` repo root |

`.github/workflows/release.yml` was **not** carried over: it ran `uv publish` on any `v*` tag, and
the merged package is not published to PyPI. `LICENSE.md` was dropped as byte-identical to the
merged repo's `LICENSE`.

## `PYTORCH_COMMIT` — still load-bearing here, forbidden in new launchers

For the 116 existing launchers it is **required**: their pinned `TRAIN_COMMIT` predates the merge,
so that snapshot imports the top-level `canvit_pytorch`, which only a snapshot of *this* repo
supplies.

**New launchers must not set it.** `TRAIN_COMMIT` now pins the model and the trainer together, and
`PYTORCH_COMMIT` would have no effect — the archived `canvit_pytorch/` lands on `PYTHONPATH` and
nothing imports it. `harness_train.sbatch` detects which side of the merge a pinned snapshot is on
and warns in both directions.

Worth knowing: an old snapshot of this repo **cannot** shadow a post-merge `canvit.core`. The
top-level names differ, so `PYTHONPATH` order is irrelevant. The core-merge plan predicted the
opposite and was wrong; see §10.2 of the doc above.

## ⚠️ Two defects in this repo's final state, both fixed in `canvit`

**The README Quickstart raises `TypeError`.** `CanViTForPretrainingHFHub.forward` takes keyword-only
`image=`, but `README.md` calls `model(glimpse=glimpse, …)` at both call sites. Verified by running
it: `TypeError: CanViTForPretraining.forward() got an unexpected keyword argument 'glimpse'`. The
classification and segmentation examples are correct — those wrappers really do take `glimpse=`.
That asymmetry is the trap; the merged README states it explicitly. Third instance of this same rot,
after `bench/pt/run.py` (fixed in `3a0dcc2`) and CanViT-eval's episode runner.

**Every classifier and probe this stack published was unloadable** until `2679d6b`. The root cause
was here, in three sites that serialized `vars(cfg)` rather than the dataclass fields; the
segmentation wrapper was affected too. Fixed before the merge — but any checkpoint published by an
*older* commit of this repo carries the broken `config.json`.

## Published checkpoints are unaffected by the rename

`config.json` records architecture only, no module paths. `library_name="canvit-pytorch"` and
`repo_url=".../m2b3/CanViT-PyTorch"` are **kept verbatim** in `canvit.core`'s HF mixins: they are
upstream attribution that lands in published model-card metadata, and renaming them would make new
publications declare a different library from every existing one.
