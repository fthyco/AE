# AE intermediate representation

`architecture.ir.json` is the full source-backed graph (see `schemas/architecture-ir.v1.schema.json`). `model-index.json` references an entrypoint's supported modules, calls, operations and IR relations without filtering the complete graph (`schemas/model-index.v1.schema.json`).

Every node and edge has a deterministic ID, kind/relation, evidence/resolution status and source references. Derived facts follow named static rules; uncertain facts remain conditional or unresolved. Byte offsets refer to exact original bytes stored in `sources/`. Stable IDs do not imply stability across arbitrary refactors.

`source-index.json` records lexical tokens; `source-manifest.json`, `coverage.json`, `diagnostics.json`, and `extraction-meta.json` preserve scope and integrity. `ae validate` checks graph references, hashes and source snapshots.

Presenter elements must map back to these IR IDs and source spans. Syntactic source order and expression dependencies are not verified runtime tensor flow.
