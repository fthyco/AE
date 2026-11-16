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
- Entrypoint-centric index with linked construction sites, symbolic layer specs,
  source-order forward calls/operations and IR dependency references.
- Safe, typed YAML configuration trees; duplicate keys, unsafe tags and
  multi-document inputs are diagnosed rather than evaluated.

## Not yet complete

- General Python imports and re-exports, closures and name lookup with arbitrary dynamic scope behavior.
- Full alias and parameter-sharing analysis, descriptor and property resolution, dynamic `setattr`, metaclasses.
- True static SSA for all control constructs, interprocedural reaching definitions, pointer/dataflow or runtime tensor graph.
- Nontrivial symbolic shape propagation through all PyTorch/functional operations.
- Hydra/OmegaConf interpolation semantics, dynamic YAML evaluation and user-defined factories.
- Systematic measurement of recall/completeness on real-world models such as Chessformer and AWN.
- Presenter, paper-quality architecture figures or graphical export.

All shape templates are conditional type-level formulas, not observed tensor shapes.
Source-dependency edges are **not** runtime tensor-flow proof. Python runtime
semantics are not generally decidable from static source. SAE records unknowns
rather than claiming false certainty.

## Security

Analyzed Python is not imported or executed; no `eval`, `exec`, `torch.load`
or pickle is used. Symlinks are rejected and file-count/byte limits are enforced.

## v0.6 static flow extensions

Supported `if/else` branch environments yield conditional join facts; `+=` and
other augmented assignments yield operand dependencies with mutation caveats.
Guarded calls remain conditional, and shape/alias/runtime behavior is not
claimed. `--focus` can scope the diagram to one model definition.
