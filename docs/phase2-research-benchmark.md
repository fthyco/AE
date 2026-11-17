# Phase 2 — research generalization and visualizer comparison (v0.25)

## Scope

This independent gate checks AE against *original, pinned, unmodified* research
code, without importing or executing that research code during extraction:

| Case | GitHub repository | Commit | Entrypoint |
| --- | --- | --- | --- |
| AWN / BT | fthyco/Adaptive-Width-Neural-Networks-WITH-ORDERING-GUIDENCE | `e75ff0959a3b21ff997c393a26711e23745e96c7` | `model:AWN` |
| Chessformer CF-ACT | fthyco/maia3 | `853785c4a8a512b6b2181520bfef21ff1a74566f` | `maia3.models_cf_act:MAIA3ActModel` |
| Chessformer CF-CAH | fthyco/maia3 | `f9f66ed3feb7ab5a4498472d708c426ca8914436` | `maia3.models_cf_cah:MAIA3CahModel` |

For each: independently parse original AST to verify the requested target class
and `forward` method; enforce the exact repository commit, source-only import
boundary, valid IR, resolved entrypoint, two byte-identical extractions,
absence of fabricated runtime-flow edges, and evidence-labeled
overview/block/operation views.

**A valid source index is not a validated full computational graph.**
Conditional widths, input-dependent paths, tensor shapes, optimizer behavior,
and runtime operation/edge parity remain **not verified** for these models.
Never count a source-only success as FX node/edge/shape equivalence. Do not
silently invent a full forward graph when static evidence is insufficient.

## Independent visualizer comparison

`scripts/compare_visualizers_v025.py` uses the fixed TorchVision 0.25.0
ResNet18 model at input NCHW (1,3,64,64). It records:

- `torch.fx` + `ShapeProp`: independent forward reference.
- `torchview` 0.2.7: executed model flow, DOT file.
- `torchviz` 0.0.3: autograd computation graph, DOT file.
- `visualtorch` 1.4.1: model-layer graph, PNG file.

These graph semantics differ. Their node counts are **not** interchangeable,
and diagram appearance is not a substitute for operation/edge matching. Each
tool's success or error is recorded separately. The previously verified
source-only AE ResNet18-vs-FX gate remains the reference for exact forward
matching.

## Evidence and license handling

AWN upstream carries noncommercial / research-use conditions. The workflow
checks out code only to the isolated CI workspace and never commits or
redistributes its source in AE. The uploaded research artifact is only the
sanitized JSON summary; AE source snapshots and copied upstream files are
excluded from uploads. Tool output artifacts on original TorchVision models
are uploaded separately.

Run the independent gate using GitHub Actions
`research-generalization-v025.yml`. All three pinned source tests must
succeed, and the FX verifier must pass before Phase 2 can be called complete.
Comparison tool failures must be reported, not hidden. Every benchmark
statement must cite the run ID and corresponding uploaded summary artifact.
