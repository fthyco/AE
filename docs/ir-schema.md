# Full-fidelity SAE IR v1.0

The authoritative artifact is `architecture.ir.json`; it is a typed, source-backed multigraph independent of layout. `schemas/architecture-ir.v1.schema.json` defines the machine-readable shape. The schema version is **1.0** (the extractor package is 0.1.0).

## Objects and invariants

- Every node has a stable ID, `kind`, attributes, evidence class, resolution status and at least one precise source reference.
- Every edge has its own stable ID, source and target node IDs, relation, evidence status, and a source reference. Derived edges name a versioned rule.
- Every Python AST element is indexed (including expression, context and operator objects). Its `ast_path` gives a deterministic position in the original AST; `CONTAINS` edges retain parent/child relationships.
- `source-index.json` includes every lexical token and comment from Python's tokenizer. `sources/` contains byte-exact copies of every included file.
- `source-manifest.json` reports included and excluded content. Excluded content is not part of the extraction claim.
- Derived `module_construction_site` nodes are distinct from syntactic `call_site` nodes. Multiple `self.layer(...)` calls can point to the same construction site.
- IDs are SHA-256-derived from normalized source paths, AST traversal paths and entity kinds. They are stable within the same source and extractor version, **not** universal across code refactors.
- `unresolved` entries are first-class diagnostic records; they are not invented graph edges.

## Evidence

`explicit` records are sourced directly from syntax. `derived` records result from an explicit, named static rule. `conditional` resolution is used for PyTorch name recognition because Python reassignment/dynamic dispatch cannot be universally ruled out in source-only mode.

Source offsets are zero-based byte offsets into the original file. For AST nodes that have no direct position (such as `ast.Load` and `ast.Add`), `precision: enclosing` identifies the enclosing expression rather than inventing a token position. This is important for honest provenance.

## Consumer contract

A presenter can read `architecture.ir.json`, select subgraphs, and maintain a list of contributing IR IDs for each visual block. It must never assume node traversal order is execution order or infer runtime paths from static links.

## SAE v0.2.0 additions

- Optional `entrypoint_target_id` points to a unique source-resolved top-level definition or `null`.
- `lexical_definition` nodes and `READS_LEXICAL_DEFINITION` edges record intraprocedural source-level reaching definitions, never runtime tensor flow.
- `pytorch_layer_spec` and `pytorch_module_definition` nodes distinguish versioned type-level signatures from constructed instances.
- `configuration_entry` trees preserve parsed JSON/TOML value relationships; provenance is file-level for parsed config entries.
- Cross-file call/import, inheritance and custom module construction relations are explicitly named and source-backed.
- `coverage.json` includes relation counts and unresolved types, but they are not semantic recall/precision scores.


## Model-centric MVP index (v0.3)

model-index.json is an additive index referencing the full IR without filtering it. Its schema is schemas/model-index.v1.schema.json. Unknown component assignments are explicit. Source order is lexical, not a runtime trace.
