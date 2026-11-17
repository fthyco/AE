# AE Presenter v0.21 — hierarchical instance-forward (draft)

## Scope

This version introduces a provenance-preserving **presentation-only** hierarchy
for the exact pinned `timm==1.0.22` ViT-Tiny static instance graph. It does not
extend the static extractor's supported model families or alter its results.

Three levels:

- `operation`: unchanged complete source-proven operation/dependency graph.
- `block`: input, embedding, each of the twelve Transformer blocks, classifier,
  output (16 groups).
- `overview`: input, embedding, 12-block Transformer encoder, classifier,
  output (five groups).

Focused operation view:

```bash
ae present ae-bundle --view instance --level operation \
  --focus blocks.0 --format all --output figures/block-zero
```

The focused graph contains the selected stage and immediate incoming/outgoing
neighbor operations. Anything outside focus is explicitly listed as omitted
original node/edge IDs in the presentation manifest. Unknown focus fails closed.

Overview/block visuals do **not** assert a new operation or a new tensor
connection. Each summary node holds all original node and source evidence IDs
it represents. Each visible summary edge holds the exact original IR edge IDs;
within-group edges are recorded in `collapsed_instance_edge_ids`. All original
nodes and edges must be accounted for by that conservation rule.

## Acceptance requirements

- The existing v0.20 operation-level 477/596 fused and 549/668 unfused matching
  with independent original timm FX/ShapeProp/runtime hooks must remain green.
- 5 visual nodes in overview and 16 at block level (pinned ViT-Tiny only).
- Full original instance edge coverage must equal grouped visible edges union
  explicitly collapsed edges, with no overlap or missing evidence.
- Focusing on `blocks.0` records every hidden original ID.
- Unsupported `blocks.99` is blocked; no stale diagrams.
- SVG/PDF/TikZ output remains deterministic and source-grounded.
- Generic model families must not silently receive invented semantic groupings;
  new hierarchy is currently restricted to pinned ViT-Tiny.

## Boundaries

This initial hierarchy is specific to the audited ViT-Tiny source and
supported input branches. It is not a generic CNN/U-Net/DenseNet block extractor.
The word "Transformer Encoder" names a presentation group of operations already
present in the verified instance graph; it is not a new computational node.

Status: **draft branch; CI and visual acceptance required before merging**.
