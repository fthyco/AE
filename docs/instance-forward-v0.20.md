# AE v0.20 — timm ViT-Tiny full instance-forward (Stage 5)

## Exact scope

Unmodified `timm==1.0.22`, `vit_tiny_patch16_224(pretrained=False)`, `eval()`.
Original package source files are individually checked by SHA-256. Independent
runtime verifier pins torch 2.10.0+cpu and torchvision 0.25.0+cpu.
Factory: patch 16, embed dimension 192, 12 transformer blocks, 3 heads,
class token, 1,000-class output. Tested inputs: `(1,3,224,224)` and `(2,3,224,224)`.

Fused and unfused attention are separate forward graphs. The caller must select
the path explicitly; AE never infers this choice from runtime environment.

## CLI

```bash
ae extract source-snapshot --entrypoint timm.models.vision_transformer:vit_tiny_patch16_224 --input-shape 1,3,224,224 --attention-mode fused --output ae-bundle
ae validate ae-bundle
ae present ae-bundle --view instance --level operation --format all --output figures
```

Use `--attention-mode unfused` for the other graph. The input source snapshot
must contain the five original, unchanged timm files at their original paths.
Extraction parses AST/source hashes; it does not import or execute timm or torch.
Unsupported sources, missing files, implicit attention mode or invalid geometry
fail closed without speculative instance nodes or edges.

## Independent gate

```bash
python scripts/verify_full_vit_tiny_v020.py --output benchmark-output/vit-tiny-v020
```

The external verifier instantiates the original model, traces FX, applies
ShapeProp and runtime leaf hooks, and independently checks every source-derived
AE node/target, ordered operand, Tensor shape and output. It also checks
source provenance and 1:1 SVG/PDF/TikZ mapping.

| Attention | FX nodes | Edges | Tensor shapes | FX module calls | Runtime leaf hooks |
|---|---:|---:|---:|---:|---:|
| Fused | 477 | 596 | 405 | 200 | 225 |
| Unfused | 549 | 668 | 477 | 212 | 237 |

Counts are per tested batch size. The 25 timm LayerNorm module invocations
(24 block norms and final norm) are represented as functional layer_norm nodes
in FX, and are cross-checked rather than silently removed.

CI additionally tests source mutation/missing-file rejection, unsupported
input geometry, deterministic hash seeds, forbidden target imports, isolated
extraction time/RSS, and all historical regression tests.

## Limits and publication

Exact audited source/factory/input-specific specialization only. Arbitrary
ViTs, nondefault kwargs, non-224 geometry, modified files, training-only paths,
dynamic branches and arbitrary runtime CUDA kernels are NOT verified.
Static publisher health remains `Partial`: runtime proof stays in the
independent verifier rather than being embedded in static IR.

Merge only after green PR CI; repeat full gate on main after merge.
