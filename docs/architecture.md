# Architecture Extractor (AE) — Architecture

`Python/PyTorch sources → SAE → validated IR → Presenter → SVG/PDF/TikZ`

**SAE:** source snapshots, AST/tokens, configuration trees, module construction, imports, local calls, value dependencies, branch alternatives, tuple/list positions, conditional symbolic shapes, and unresolved records.

**Entrypoint:** `--entrypoint module:ClassName` selects a model in `model-index.json`. The full graph remains in `architecture.ir.json`.

**Presenter:** source-backed module structure (`structure`) or static value relations (`flow`). Rendered nodes and edges retain IR IDs and source references.

**Health:** `complete` for the declared static view; `partial` for relevant gaps; `blocked` for invalid or insufficient data. `--mode publication` rejects partial output unless `--allow-partial` is set.

**Bundle:** `architecture.ir.json`, `model-index.json`, `source-index.json`, `source-manifest.json`, `coverage.json`, `diagnostics.json`, `extraction-meta.json`, and `sources/`.

**Rendered outputs:** `architecture.svg`, `architecture.pdf`, `architecture.tex`, `presentation-health.json`, `presentation-manifest.json`.

**Schemas:** `schemas/architecture-ir.v1.schema.json` and `schemas/model-index.v1.schema.json`.
