# SAE v0.3.0 — verified capabilities and remaining limitations

## Implemented

- Non-executing, deterministic file inventory and byte-exact source snapshots.
- Every Python AST node and lexical token in included, parsable `.py` files.
- Direct syntax facts for declarations, imports, assignments, calls, operators, returns and control constructs.
- Local project import/call and class-method resolution where a unique static declaration exists.
- Direct and transitive local `nn.Module` inheritance hints, custom module constructor sites and reuse links.
- Intraprocedural source-level reaching-definition edges with conditional/loop ambiguity preserved.
- Symbolic PyTorch shape templates and nested constructor containment; static JSON/TOML data trees.
- Source evidence, stable identifiers, integrity validation, explicit unresolved evidence and checksums.
- Entrypoint resolution to a top-level module symbol when unique.
- Entrypoint model-centric index with source-backed component and forward references.
- Safe typed YAML configuration trees with duplicate key, unsafe tag and multiple document diagnostics.

## Not yet complete

- General Python imports and re-exports, closures and name lookup with arbitrary dynamic scope behavior.
- Full alias and parameter-sharing analysis, descriptor and property resolution, dynamic `setattr`, metaclasses.
- True static SSA for all control constructs, interprocedural reaching definitions, pointer/dataflow or runtime tensor graph.
- Nontrivial symbolic shape propagation through all PyTorch/functional operations.
- Hydra/OmegaConf interpolation, runtime config evaluation and user-defined factories.
- Systematic measurement of recall/completeness on real-world models such as Chessformer and AWN.
- Presenter, paper-quality architecture figures or graphical export.

All shape templates are conditional type-level formulas, not observed tensor shapes.
Source-dependency edges are **not** runtime tensor-flow proof. Python runtime
semantics are not generally decidable from static source. SAE records unknowns
rather than claiming false certainty.

## Security

Analyzed Python is not imported or executed; no `eval`, `exec`, `torch.load`
or pickle is used. Symlinks are rejected and file-count/byte limits are enforced.

## v0.5 forward-flow boundary

`--view flow` is now available for straight-line `forward` source expressions and module-local value dependencies. Control-flow branches, loop-carried values, in-place mutations (`+=` etc.), dynamic Python semantics and unknown destructuring remain partial, not false tensor connections. The older structural view stays available. This source-backed projection is not a verified runtime computational graph.
