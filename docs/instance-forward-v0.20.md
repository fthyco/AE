# AE v0.20 — ViT Tiny (Stage 5, IN PROGRESS)

## Original reference

Use unmodified `timm==1.0.22` factory `vit_tiny_patch16_224` with `pretrained=False`,
`torch==2.10.0+cpu`, `torchvision==0.25.0+cpu`, and `eval()`.
The known default constructor parameters are patch size 16, embedding dimension
192, depth 12, three attention heads, one class token, and a 1,000-class output.
For now the explicit test geometries are `1×3×224×224` and `2×3×224×224`.

## Completed in the preflight

```bash
python scripts/preflight_vit_tiny_v020.py --output benchmark-output/vit-tiny-v020
```

This independent verifier instantiates original timm outside SAE, records the
original source-file SHA-256s, traverses actual FX operations, edges and ShapeProp
Tensor shapes, and checks the leaf-module invocation sequence and output shapes
against real forward hooks. It explicitly tests fused and unfused attention
**as separate implementation configurations**, preserving their separate FX
graphs. The resulting `fx.json`, `hooks.json`, `result.json`, and `summary.json`
are reference artifacts only.

## Independently measured reference (2026-11-17)

Both batch-one and batch-two input tests passed in the pinned GitHub Action
using real timm, FX, ShapeProp and runtime hooks. The source SHA-256 values
of the five audited original timm files are now hard-coded and checked.

| Reference path | FX nodes | FX edges | Tensor shapes | FX call_module | Runtime leaf hooks | Inlined timm LayerNorm calls |
|---|---:|---:|---:|---:|---:|---:|
| Fused attention | 477 | 596 | 405 | 200 | 225 | 25 |
| Unfused attention | 549 | 668 | 477 | 212 | 237 | 25 |

Each metric is per input shape. The 25 inlined LayerNorm invocations consist
of `blocks.0..11.norm1`, `blocks.0..11.norm2`, and final `norm`.
FX represents these as underlying functional operations whereas hooks observe
the original modules. Both views are preserved; neither is dropped silently.

This remains **reference-only**: the AE-vs-reference equality gate is not
implemented or passed yet.

## Required before claiming AE v0.20 complete

1. Implement a source-only, fail-closed, SHA-256-pinned instance expansion for
   the exact original timm source and selected input geometry. SAE must never
   import `timm`, `torch`, or run the analyzed model.
2. Establish source AST sites, ordered QKV and attention dependencies,
   token/position embedding, MLPs, 12 block invocations, residuals, normalization,
   pooling/head, shape metadata and unique reused-module call identities.
3. Independently compare every AE operation/target, ordered dependency, Tensor
   shape, leaf invocation, source evidence and output against each supported FX
   configuration; unsupported paths must emit no speculative graph.
4. Compare SVG/PDF/TikZ element/edge IDs 1:1 against the verified AE instance IR.
5. Add source mutation, unsupported input/factory, deterministic hash-seed,
   zero-target-import, isolated performance, and full test-suite gates.
6. After a green PR check, merge only after the same gates pass on `main`.

The preflight **does not** prove that AE extracts ViT Tiny. `main` remains v0.19
until all acceptance gates pass.
