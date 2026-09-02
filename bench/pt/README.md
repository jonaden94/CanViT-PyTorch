# `bench/pt` — inference latency

Per-forward-pass latency at batch 1 with an explicit device sync each iteration.

    python bench/pt/run.py --model canvit --device cuda --scene-px 512 --dtype amp-bf16
    python bench/pt/analyze.py --pattern 'bench/pt/results/*.jsonl'
    python bench/pt/matrix.py --help     # sweep driver: one subprocess per cell

`run.py` measures one cell and streams JSONL; `analyze.py` reports median/p95/p99 with
bootstrap CIs, compares thread counts pairwise, and flags per-run time drift; `matrix.py`
drives a sweep with GPU/CPU idle gating and CPU-topology-aware thread pinning.

Moved here from `CanViT-eval/bench/` on 2026-09-02. It benchmarks `canvit_pytorch` and its
only non-core import was two DINOv3 repo ids, so this is the layer it belongs at — hosting it
in the training repo would have inverted the dependency.

## `baselines/`

A stored reference so `analyze.py` answers "did this regress?" rather than only "what is it
now": pass a baseline alongside a fresh run and the CI/pairwise machinery compares them.

**A latency baseline is only valid on the hardware that produced it**, which is why the
filenames and the `meta` record in each JSONL carry the device. The committed ones are:

| cell | device | median |
|---|---|---|
| `canvit`, 512 px, cg 32, amp-bf16, eager | **A100-SXM4-80GB MIG 3g.40gb** | 11.08 ms (CI 11.08–11.09, p99 11.52) |
| `dinov3-vitb16`, 512 px, amp-bf16, eager | **A100-SXM4-80GB MIG 3g.40gb** | 8.98 ms (CI 8.98–8.98, p99 9.89) |

300 measured iterations each. CanViT at canvas-grid 32 costs ~1.23x the teacher's single
forward at the same scene resolution on this slice.

Comparing against these from a full A100, an H100 or CPU tells you nothing. Re-measure a
baseline on the machine you care about and commit it beside these.

Fixed on the way in: `run.py` called `CanViT.forward(glimpse=...)`, but bare `CanViT` takes
`image=` — only the downstream wrappers use `glimpse=`. Every CanViT cell had been raising
`TypeError`, so the benchmark had not run against current core for some time.
