# Architecture Extractor — IR

`architecture.ir.json`: source-backed nodes, edges, evidence classes, resolution states, and source spans. Schema: `schemas/architecture-ir.v1.schema.json`.

`model-index.json`: entrypoint-specific modules, constructors, calls, operations, and IR references. Schema: `schemas/model-index.v1.schema.json`.

Node and edge IDs are deterministic for fixed inputs. Derived facts carry analysis rules. Unresolved references remain explicit. Byte offsets point to original files in `sources/`.

`source-index.json` indexes tokens. `source-manifest.json`, `coverage.json`, `diagnostics.json`, and `extraction-meta.json` record scope and integrity.

`ae validate` checks references, artifact hashes, and source snapshots. Presenter mappings retain contributing IR IDs and source spans.

Static source dependencies do not establish runtime tensor flow.
