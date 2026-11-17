# Architecture executer (AE) — Architecture and evidence contract

**Pipeline:** Python/PyTorch source → SAE → validated JSON IR → Presenter → SVG/PDF/TikZ.

SAE inventories supported source and configuration files, preserving original bytes and SHA-256 hashes. It extracts AST/tokens, PyTorch construction sites, local links, supported static value relationships, branch alternatives, indexed fixed-arity tuple/list returns, and conditional symbolic shape facts. Every inferred fact has a source reference; ambiguity remains unresolved. Source code is never executed.

The optional `--entrypoint module:ClassName` creates `model-index.json`, an index into the complete, unfiltered `architecture.ir.json`. The Presenter consumes only validated SAE artifacts: it may group or omit details, but every visual node and connector must retain evidence IDs and source locations. `structure` is module/call composition; `flow` is the supported **static** value-dependency graph, not an observed execution trace.

**Health:** `complete` refers to the selected supported static view, not runtime correctness. `partial` records unknown view-relevant relations. `blocked` rejects invalid or insufficient evidence. Publication mode withholds partial diagrams unless explicitly allowed. Styling does not alter source evidence.

**Bundle:** `architecture.ir.json`, `model-index.json`, `source-index.json`, `source-manifest.json`, `sources/`, `coverage.json`, `diagnostics.json`, and `extraction-meta.json`. Presenter outputs include `presentation-health.json` and `presentation-manifest.json` with visual-to-IR mappings.

**Schemas:** `schemas/architecture-ir.v1.schema.json`, `schemas/model-index.v1.schema.json`. See `docs/limitations.md` for supported scope and unknowns.
