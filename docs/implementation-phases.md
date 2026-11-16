# SAE implementation phases — v0.2.0 evidence report

This report describes implemented behavior, not a claim of complete semantic recovery.
No code from analyzed projects is imported or executed.

| Phase | Implemented now | Known boundary |
| --- | --- | --- |
| 1. Semantic resolution | Local project module index, relative/from imports, direct local function/class calls, class method references, static `nn.Module` subclasses and local custom module construction | Dynamic imports, re-exports, runtime attribute assignment, bound aliases, monkey-patching, implicit dispatch, nested class definitions and non-unique candidates |
| 2. Source-level dataflow | Intraprocedural parameter/assignment definitions and read links, conservative branch/loop merge, explicit ambiguity/unbound candidates | Not true SSA, runtime tensor graph, precise exception CFG, full alias analysis, nested closure/comprehension semantics or side effects |
| 3. PyTorch & config semantics | Versioned layer recognition, symbolic signature templates (e.g. Linear, Embedding, Conv channels, MHA default), nested constructors; JSON/TOML typed trees | Unknown/custom operator signatures; limited shape propagation; no dynamic config evaluation, YAML semantic parser, tensor metadata from runtime, specialized MHA options |
| 4. Verification | Stable graph IDs, source refs, checksums, referential integrity, unresolved candidate validation, explicit entrypoint resolution, deterministic JSON, baseline and regression tests | No proof of total Python semantics; CI confirms tests only |
| 5. Integration smoke | Self-extraction of SAE source tree and example MLP using the public API | Validation against full third-party training repositories remains open |

## Evidence classification

- `explicit`: Source syntax and direct syntactic relationships.
- `derived`: A named, versioned static-analysis rule; a derived relation may be conditional.
- `unresolved`: Source syntax does not uniquely determine the relationship.
- `READS_LEXICAL_DEFINITION` relates source-level definitions and uses. It must **not** be represented as verified runtime tensor flow.
- A symbolic shape template describes a module contract conditional on its constructor and applicable input; it is **not** an observed tensor dimension.

## Guarantees under pinned inputs

1. Full bytes of every included source file are preserved.
2. AST and token indexing is deterministic under unchanged bytes and extractor version.
3. Valid in-scope source references and edge endpoints are checked.
4. Unsupported syntax and semantic uncertainty are not silently converted into concrete architecture edges.
5. Presenter is out of scope; the IR is the only output contract.

## Remaining work before research-paper figure generation

Precise closure/scope resolution, deeper alias and parameter-sharing analysis,
interprocedural data-flow, full conditional control-flow graph, symbolic shape
propagation through tensor operations, YAML and custom configuration resolution,
and empirical extraction quality measurements across real model repositories.

This is an **increment**, not an assertion that all five phases are complete.
