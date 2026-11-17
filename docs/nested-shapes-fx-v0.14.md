# AE v0.14 — Nested Shapes and FX Source-Site Audit

## Extraction

The shape pass propagates conditional tensor-shape facts through verified `STATIC_FORWARD_RETURN`, `STATIC_CALL_RETURN`, `STATIC_CALL_ARGUMENT_BINDING`, and indexed collection-element bindings. Composite operations and nested returns are iterated to a bounded fixed point. Shapes are not transferred across unresolved, repeated, recursive, or ambiguous calls.

Supported examples: `Child.forward() → Parent.forward()`, nested `MatMul → Reshape`, and nested `MatMul → Concat` with concrete source-derived dimensions. Unknown tensor axes remain unknown. `tuple`/`list` aggregates are not tensors.

## Independent FX audit

```bash
python scripts/verify_fx_correspondence.py fixture
python scripts/verify_fx_correspondence.py nested_composite
python scripts/verify_fx_correspondence.py resnet18
python scripts/verify_fx_correspondence.py resnet50
python scripts/verify_fx_correspondence.py densenet121
```

The verifier imports and runs the model independently of AE extraction. It records FX operations with source line and column, matches source-backed AE sites, tracks repeated FX occurrences per site, and compares directly connected edges only when both endpoints have unique matches. It compares concrete tensor shapes and supported symbolic suffixes against FX ShapeProp.

**Coverage denominator:** supported FX operations with a traceable original source location. `observed_site_recall` is **not** total architectural operation recall. `edge_precision` and `edge_recall` apply only to uniquely matched single-use operation sites. Repeated blocks, dynamic paths, module identity, and whole-model tensor-flow equivalence remain unverified.

A source-level operation site can be executed multiple times. Its FX event count is recorded, not treated as multiple uniquely identified AE instances. FX may constant-fold source expressions; those operations are absent from the FX graph and excluded from the site denominator. Final-output shapes can still be checked independently.

## Outputs

`correspondence.json`, `fx_events.json`, `sae/` with source snapshots and IR. The source snapshot retains the original file's SHA-256.
