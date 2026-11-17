# AE v0.22 — MobileNetV3 Small (Stage 6)

## Audited original reference

- Unmodified `torchvision==0.25.0+cpu` and `torch==2.10.0+cpu`.
- Original `mobilenet_v3_small(weights=None)` in `eval()`.
- Only default width multiplier 1.0, no reduced tail/dilation, 1,000 classes.
- Supported input geometries: `1×3×64×64`, `1×3×128×128`, `1×3×224×224`, `2×3×224×224`.
- Source SHA-256 pins: `torchvision/models/mobilenetv3.py`, `torchvision/ops/misc.py`, `torchvision/models/_utils.py`.

## Independent original-model evidence

| Property | Audited value |
|---|---:|
| FX operations | 159 |
| FX ordered operand dependencies | 173 |
| Tensor shape facts | 159 |
| FX leaf calls | 141 |
| Runtime leaf calls | 141 |
| InvertedResidual blocks | 11 |
| Squeeze-and-Excitation gates / multiplication nodes | 9 |
| Residual additions | 6 |
| Parameters (default factory) | 2,542,856 |

Independent reference includes `torch.fx`, `ShapeProp`, real forward hooks, and exact upstream source fingerprints. The AE verifier checks **every** operation, ordered operand slot, shape, module identity, output, and 1:1 SVG/PDF/TikZ element/edge manifest against that separate reference at each audited input geometry.

## Source-only CLI

Original TorchVision source files must be placed in a source snapshot at the canonical package paths named above.

```bash
ae extract source-snapshot --entrypoint torchvision.models.mobilenetv3:mobilenet_v3_small --input-shape 1,3,224,224 --output ae-bundle
ae validate ae-bundle
ae present ae-bundle --view instance --level operation --format all --output figures/operation
ae present ae-bundle --view instance --level overview --format all --output figures/overview
ae present ae-bundle --view instance --level block --format all --output figures/blocks
ae present ae-bundle --view instance --level operation --focus features.4 --format all --output figures/focus-4
```

The extractor **never imports or runs** the original model. Source-based AST checks establish the original fixed 11-block blueprint; stage shapes, SE scaling, depthwise/pointwise branches, skip-additions, classifier, and invocation order are expanded conservatively. Changed/missing source, unsupported geometry, and unresolved construction fail closed.

## Run the independent acceptance

```bash
python scripts/verify_full_mobilenetv3_small_v022.py --output benchmark-output/mobilenetv3-small-v022
```

The gate additionally checks test-suite regression, reference package and source SHA-256, rendering evidence conservation, deterministic canonical IR across hash seeds, zero target imports during AE execution, and isolated wall-time/peak-RSS profiling.

An `hierarchy-gallery.html` and per-case original SVG/PDF/TikZ diagrams are emitted from the verified Presenter output.

## Boundaries

This is a **source/version/factory/input-scoped specialization**, not arbitrary MobileNet support. No claims are made about altered constructor kwargs, nondefault weights, reduced tail, dilated paths, training-only behavior, quantized variants, arbitrary Tensor shapes, device-specific kernels, or original source edits. Publisher health may remain Partial because runtime verification evidence lives in an external CI verifier, not inside the static bundle.

Release gate: no merge before complete PR workflows succeed; verify main after merge.
