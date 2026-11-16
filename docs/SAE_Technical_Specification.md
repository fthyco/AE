# MAGG — Static Architecture Extractor (SAE)
## Full-Fidelity Static Extraction: Technical Specification

**Document version:** 0.1 (design baseline)  
**Status:** Design baseline; working v0.1.0 prototype implements a subset (see `limitations.md`)  
**Primary language:** Python / PyTorch  
**Current scope:** Extraction layer only

---

## 1. Purpose and design position

MAGG (*Model Architecture Graph Generator*) has two independent layers:

```text
                 MAGG
                  |
       +----------+-----------+
       |                      |
   1. SAE Extractor      2. Presenter (later)
       |                      ^
       +---- Architecture IR--+
```

**SAE's responsibility:** read the selected source repository *without executing the model* and produce the deepest source-grounded, internally linked, machine-readable description of its architecture and associated computations that static analysis can substantiate.

**Presenter's responsibility (future work):** select, group, simplify, lay out, and render information from the IR as a consistent architecture figure suitable for research papers (SVG, PDF, TikZ). It must not invent architectural facts.

**Core rule: Extract at maximum fidelity; abstract only downstream.** No loss of detail is permitted merely to make a figure simpler.

### 1.1 Non-negotiable principles

1. **Source as the ground truth:** every extracted claim must be attributable to source code, a declared configuration, or a documented inference rule.
2. **Static-only extraction:** no model instantiation, imports of target modules, forward passes, tracing, training, or checkpoint execution.
3. **Source preservation:** the output bundle includes a byte-preserving snapshot (or content-addressed copy) of every in-scope source file plus precise source references.
4. **Maximum representational depth:** preserve syntax, definitions, expressions, operations, dependencies, control structure, configurations, and all statically supportable cross-references.
5. **Conservative semantics:** represent dynamic or ambiguous information explicitly; never fabricate a unique answer when source analysis cannot prove one.
6. **Reproducibility:** identical source bytes, included dependency specifications, extraction configuration, tool version, and rules produce byte-identical canonical output.
7. **Renderer independence:** the IR is not a layout file; it contains no coordinates, colors, paper-figure block choices, or aesthetic labels.

### 1.2 Guarantee boundary

A *lossless source bundle* means that every included input file can be reconstructed byte-for-byte. It does **not** mean that arbitrary Python program behavior can always be resolved from static analysis. In particular, runtime-generated modules, reflection, data-dependent branches, dynamic dispatch, and external library internals may remain unresolved. Those cases remain visible and linked to their exact source expressions.

---

## 2. Inputs and extraction scope

### Required inputs

- **Source root:** local project/repository directory.
- **Extraction scope:** include/exclude rules for files and packages. All excluded files must be listed in the manifest with a reason.
- **Target entry points:** model class(es), factory function(s), or module(s) of interest; defaults may be discovered but must be labeled as candidates.

### Optional inputs

- Explicit model constructor arguments and named configuration profiles; values are interpreted statically, not executed.
- Configuration files: JSON, TOML, YAML, and safe literal Python assignments where supported.
- Framework/library version declarations such as `pyproject.toml` or requirements files.
- Optional versioned symbol/operator-signature catalog for known PyTorch APIs. Catalog facts must declare their provenance and applicable versions.

### Files in the initial implementation

- Parse all in-scope `.py` files and index their original bytes, lexical tokens, AST, comments, and source positions.
- Read supported configuration/dependency files as data; index unrecognized or invalid content and emit diagnostics.
- Do not traverse or execute arbitrary third-party package source by default. External symbols are recorded as external references; deeper analysis requires their source to be included explicitly in the scope.

### Exclusions in v1

- Executing repository code or resolving output by actual tensor runs.
- Inferring checkpoint-dependent or learned tensor values.
- Guaranteeing all runtime paths or all runtime tensor shapes.
- Generating drawings or deciding the visual abstraction level.
- Cross-framework generalization beyond the Python/PyTorch adapter.

---

## 3. Extraction depth and expected knowledge

| Depth | Domain | Information to capture | Static certainty |
|---|---|---|---|
| L0 | Source | Files, bytes/hash, spans, tokens, comments, docstrings, AST and expressions | Exact for included, readable source |
| L1 | Symbols | Imports, scopes, names, declarations, bindings, shadowing, aliases, classes, methods | Exact syntax; resolution when provable |
| L2 | Structure | Inheritance, object attributes, declared module construction sites, module hierarchy and ownership expressions | Explicit or conditionally derived |
| L3 | Configuration | Literal values, constructor keywords, nested configs, symbolic constants, constraints and provenance | Concrete or symbolic |
| L4 | Computation | Calls, operators, arguments, returns, indexing, transforms, functional API expressions | Explicit syntax; callee resolution may vary |
| L5 | Control | Branches, guards, loops, early returns, exception paths, nested call structure | Explicit branches, not asserted runtime coverage |
| L6 | Data flow | Definitions/uses, value and tensor dependencies, assignments, phi/merge points, possible aliases | Conservative when paths are ambiguous |
| L7 | Architecture | Linear/Conv/Embedding/Attention components where statically identifiable, parameter-use sites, shared references, residual/concat patterns supported by evidence | Derived from facts, never guessed |
| L8 | Tensor metadata | Known/symbolic shapes, dimensions, axes, dtypes and constraints from explicit arguments and valid rules | Exact/derived/unresolved |
| L9 | Provenance | Exact file/span and evidence chain for each claim, unresolved reason and parser/analyzer diagnostics | Required for every semantic fact |

**Critical distinction:** a module *definition*, a module *construction site*, and a *call site* are different entities. The same module/parameter can appear in multiple calls without duplication of its identity. If identity cannot be established statically, record candidate bindings or an unresolved reference rather than an invented match.

---

## 4. SAE internal pipeline

```text
Repository / Source Scope / Config
             |
             v
[01] Inventory & Source Snapshot
             |
             v
[02] Lexical + AST Parser
             |
             v
[03] Scope, Symbol & Import Index
             |
             v
[04] Static Analyzers
     | Definitions / Classes / Inheritance
     | Module constructor sites / Attributes
     | Calls / Expressions / Operators
     | CFG / Def-Use / Aliasing / Data flow
     | Config evaluation / Symbolic dimensions
             |
             v
[05] Cross-File Linking & Evidence Resolution
             |
             v
[06] Full-Fidelity Architecture IR Builder
             |
             v
[07] Validator, Coverage & Diagnostics
             |
             v
[08] Canonical Serializer + Source Bundle
             |
             v
  Architecture IR + Reports + Source Snapshot
             |
             v
       Presenter (not built in phase 1)
```

### Stage contracts

**01 — Inventory.** Deterministically traverse paths; preserve byte content and content hashes; record included, excluded, inaccessible, or unsupported files. Follow symlinks only under an explicit policy; prevent duplicate traversal and path escape.

**02 — Parsing.** Use source-preserving token/span indexing alongside Python AST. AST alone is not lossless for whitespace/comments; retain the original file and token metadata. On syntax failure, retain bytes and record the issue rather than silently skipping the file.

**03 — Symbols.** Build lexical scopes, symbol tables, import references, class relationships, and binding occurrences across files. Distinguish resolved, external, ambiguous, and unresolved names.

**04 — Analysis.** Build conservative control-flow and data-flow facts per function; link assignments and uses; preserve all branch bodies and loop syntax. Extract statically recognizable PyTorch constructor sites and functional operations. Symbolically propagate only mathematically and programmatically justified dimensions.

**05 — Linking.** Connect definitions, uses, attributes, imports, call sites, possible callees, control guards, values, and module/parameter ownership. Every resolved link retains its evidence trail. Cross-file unknowns are explicit.

**06 — IR building.** Materialize distinct source, semantic, and architecture entities, typed relations, symbolic expressions, facts, and unresolved references without information-reducing grouping.

**07 — Verification.** Check schema validity, identity uniqueness, reference integrity, source-span fidelity, inference rule preconditions, source inventory completeness, and diagnostic/coverage accounting. A passing report verifies these invariants, not arbitrary program equivalence.

**08 — Serialization.** Sort and normalize all collections with stable ordering; write canonical UTF-8 JSON with a documented schema version and stable content hashes. Do not inject timestamps or machine-specific absolute paths into canonical fields.

---

## 5. Full-Fidelity Architecture IR: logical schema

The IR is a **typed, attributed multigraph**, accompanied by a preserved source snapshot, symbolic expressions, diagnostics, and a manifest. Multiple edges between the same entities are allowed because they may represent distinct call sites and facts.

### 5.1 Entity families

| Family | Representative types |
|---|---|
| Source | `source_file`, `token`, `syntax_node`, `source_span`, `comment` |
| Definitions | `module_scope`, `class_definition`, `function_definition`, `method_definition`, `parameter_definition`, `symbol` |
| Bindings | `binding`, `assignment`, `attribute_reference`, `import_reference`, `alias_candidate` |
| Architecture | `module_construction_site`, `module_reference`, `parameter_reference`, `operation_site`, `tensor_value` |
| Computation | `call_site`, `operator_site`, `literal`, `expression`, `return_site`, `index_site` |
| Control | `basic_block`, `condition`, `loop`, `merge`, `exception_handler` |
| Symbolic | `dimension_symbol`, `shape_expression`, `type_expression`, `constraint`, `configuration_value` |
| Evidence | `inference_record`, `unresolved_reference`, `diagnostic` |

**All syntax entities need not be materialized as verbose objects if they can be represented without loss via a versioned source index plus stable references.** They must nevertheless remain addressable from semantic entities and retrievable from the bundle.

### 5.2 Core relationship types

- Structural: `DEFINES`, `CONTAINS`, `INHERITS_FROM`, `DECLARES_ATTRIBUTE`, `CONSTRUCTS_AT`.
- Symbolic: `IMPORTS`, `BINDS`, `REFERENCES`, `ALIASES_OR_MAY_ALIAS`, `RESOLVES_TO`.
- Computational: `CALLS`, `HAS_ARGUMENT`, `PRODUCES`, `CONSUMES`, `RETURNS`, `DEPENDS_ON`.
- Control: `GUARDED_BY`, `BRANCHES_TO`, `MERGES_AT`, `LOOP_BODY_OF`.
- Architecture: `HAS_PARAMETER_REFERENCE`, `USES_MODULE_REFERENCE`, `MAY_SHARE_REFERENCE`.
- Provenance: `APPEARS_AT`, `DERIVED_FROM`, `SUPPORTED_BY_RULE`.

The graph must keep **possible** relationships distinct from **proven** relationships. A syntactic arrow does not automatically establish a unique runtime dependency.

### 5.3 Required fields for a semantic fact

Every semantic entity or edge must expose at least:

- Stable `id` and `kind`/`relation`.
- `source_refs`: one or more byte/line/column source spans, or an explicitly cited external catalog item.
- `evidence_class`: `explicit` or `derived`; unresolved claims are separate records.
- `analysis_rule`: versioned rule identifier for derived facts.
- `resolution`: `resolved`, `conditional`, `ambiguous`, or `unresolved`, as applicable.
- Typed attributes (including symbolic expressions when concrete values are not proved).

Avoid ungrounded confidence percentages. Use proof status and evidence instead.

### 5.4 Example (illustrative, not the complete schema)

```python
class Net(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(width, 32)

    def forward(self, x):
        return torch.relu(self.proj(x))
```

```json
{
  "schema_version": "0.1",
  "model_entrypoint": "Net",
  "entities": [
    {"id": "module:Net.proj", "kind": "module_construction_site",
     "type_ref": "torch.nn.Linear",
     "arguments": {"in_features": {"symbol": "width"}, "out_features": 32},
     "evidence_class": "explicit", "source_refs": ["model.py:4"]},
    {"id": "call:Net.forward:proj", "kind": "call_site",
     "callee_ref": "module:Net.proj", "source_refs": ["model.py:7"],
     "evidence_class": "derived", "analysis_rule": "resolve-attribute-call/v1"},
    {"id": "call:Net.forward:relu", "kind": "call_site",
     "callee_ref": "torch.relu", "source_refs": ["model.py:7"],
     "evidence_class": "explicit"}
  ],
  "relations": [
    {"from": "call:Net.forward:proj", "to": "call:Net.forward:relu",
     "relation": "DEPENDS_ON", "analysis_rule": "nested-call-dataflow/v1"}
  ],
  "unresolved": [
    {"symbol": "width", "reason": "Constructor value not supplied"}
  ]
}
```

The real schema uses **unique source-derived IDs**, **exact source spans**, **explicit call inputs/outputs**, **referentially valid symbols**, and **fully qualified types**; the sample shortens those fields for readability.

---

## 6. Determinism, completeness, and trust model

### Determinism

Define a canonical extraction function:

`IR = SAE(source_bytes, scope, static_config, rule_catalog_version, sae_version)`

For identical arguments, the canonical `architecture.ir.json` and associated reports must have identical bytes. File traversal, sets, dictionary keys, ID construction, symbol ordering, and serialization must be normalized. Use cryptographic hashing for source/canonical artifacts, not runtime-dependent language hash values.

### Completeness is reported, not assumed

- **Source coverage:** every in-scope path is accounted for (parsed, unparseable, unreadable, or unsupported).
- **Syntax coverage:** all supported Python constructs have a recorded representation or explicit unsupported diagnostic.
- **Semantic resolution:** report counts/categories of resolved, ambiguous, and unresolved references; do not treat unresolved as missing.
- **No runtime coverage claim:** static enumeration of branch syntax is not proof that all dynamic execution cases have been determined.

### Safety

The extractor reads target files as *data*. It never uses `eval`, `exec`, target imports, arbitrary setup hooks, pickle deserialization, model construction, or runtime tracing. Config inputs are parsed using safe data parsers, not arbitrary Python execution. Enforce file-count/size limits and symlink/path policies.

---

## 7. Project hierarchy (proposed repository)

```text
magg/
├── README.md                           # Project purpose and quick start
├── pyproject.toml                      # Python package, CLI, dependencies
├── LICENSE
│
├── docs/
│   ├── SAE_Technical_Specification.md  # This design contract
│   ├── ir-schema.md                    # Entities, edges, statuses, invariants
│   ├── extraction-rules.md             # Static inference rules and limits
│   ├── determinism.md                  # Canonicalization and hash policies
│   └── presenter-contract.md           # Future consumer-facing interface
│
├── schemas/
│   ├── architecture-ir.v1.schema.json  # Machine-validated IR schema
│   ├── source-manifest.v1.schema.json
│   ├── coverage.v1.schema.json
│   └── diagnostics.v1.schema.json
│
├── src/
│   └── magg/
│       ├── __init__.py
│       ├── cli.py                       # magg extract / validate / inspect
│       │
│       ├── sae/                         # Phase 1: implemented subsystem
│       │   ├── __init__.py
│       │   ├── api.py                   # Public Python extraction API
│       │   ├── settings.py              # Scope, entrypoints, rule versions
│       │   ├── pipeline.py              # Ordered stages and stage outputs
│       │   │
│       │   ├── inventory/
│       │   │   ├── walker.py            # Deterministic, safe path discovery
│       │   │   ├── manifest.py          # File status, hashes, exclusions
│       │   │   └── snapshot.py          # Byte-exact source preservation
│       │   │
│       │   ├── parsing/
│       │   │   ├── python_parser.py     # AST parsing (no target execution)
│       │   │   ├── tokens.py            # Comments, tokens, original spans
│       │   │   ├── source_spans.py       # Byte/line/column resolution
│       │   │   └── config_parser.py     # JSON/TOML/YAML and safe literals
│       │   │
│       │   ├── indexing/
│       │   │   ├── syntax_index.py      # All addressable syntax elements
│       │   │   ├── scope_index.py       # Lexical scopes and name bindings
│       │   │   ├── symbol_index.py      # Definitions and references
│       │   │   └── import_index.py      # Internal/external imports
│       │   │
│       │   ├── analysis/
│       │   │   ├── definitions.py       # Functions, classes, methods
│       │   │   ├── inheritance.py       # Base classes and class relations
│       │   │   ├── attributes.py        # self.*, property/binding sites
│       │   │   ├── modules.py           # nn.Module construction patterns
│       │   │   ├── calls.py             # Call sites and candidate callees
│       │   │   ├── expressions.py       # Operators, literals, indexing
│       │   │   ├── control_flow.py      # Guards, branches, loops, exits
│       │   │   ├── data_flow.py         # Def-use and value dependencies
│       │   │   ├── aliasing.py          # Shared/possible references
│       │   │   ├── configurations.py    # Static values and symbols
│       │   │   └── tensor_shapes.py     # Symbolic shape rules, if provable
│       │   │
│       │   ├── linking/
│       │   │   ├── resolver.py          # Cross-file/cross-scope resolution
│       │   │   ├── evidence.py          # Source + inference-rule provenance
│       │   │   └── unresolved.py        # Explicit ambiguity and unknowns
│       │   │
│       │   ├── ir/
│       │   │   ├── models.py            # Typed entities, edges, artifacts
│       │   │   ├── builder.py           # Assemble full-fidelity IR
│       │   │   ├── ids.py               # Deterministic identifiers
│       │   │   ├── canonical.py         # Canonical sorting/serialization
│       │   │   └── validator.py         # Schema and graph integrity
│       │   │
│       │   ├── verification/
│       │   │   ├── invariants.py        # Grounding and reference checks
│       │   │   ├── coverage.py          # Source/syntax/resolution reports
│       │   │   └── diagnostics.py       # Errors, warnings, limitations
│       │   │
│       │   └── output/
│       │       ├── bundle.py            # Package IR and source snapshot
│       │       └── writer.py            # Canonical JSON and reports
│       │
│       └── presenter/                   # Phase 2: NOT implemented now
│           └── README.md               # Consumes IR, never source code
│
├── examples/
│   ├── simple_mlp/
│   ├── residual_block/
│   ├── shared_module/
│   ├── conditional_forward/
│   └── symbolic_dimensions/
│
└── tests/
    ├── unit/
    │   ├── test_inventory.py
    │   ├── test_parser_and_spans.py
    │   ├── test_symbols_and_imports.py
    │   ├── test_control_and_dataflow.py
    │   ├── test_modules_and_aliases.py
    │   ├── test_symbolic_shapes.py
    │   └── test_ir_integrity.py
    ├── integration/
    │   ├── test_end_to_end_extraction.py
    │   ├── test_cross_file_linking.py
    │   └── test_unknowns_are_preserved.py
    └── reproducibility/
        ├── test_byte_identical_outputs.py
        └── test_source_snapshot_roundtrip.py
```

### 7.1 Module ownership boundaries

- `inventory/` knows **files** but not model semantics.
- `parsing/` knows **syntax** but does not guess architectural intent.
- `indexing/` creates **addressable scopes and symbols**.
- `analysis/` creates **statically justified semantic facts**.
- `linking/` joins facts **with explicit evidence and uncertainty**.
- `ir/` defines the **data contract**, never drawing instructions.
- `verification/` checks **claims and completeness accounting**.
- `output/` packages artifacts **deterministically**.
- `presenter/` is a future, separate consumer. It must not modify SAE's extraction truth.

---

## 8. Public API and resulting artifacts

### Proposed CLI

```bash
magg extract ./repository \
  --entrypoint package.models:Model \
  --config config.json \
  --output ./magg-output

magg validate ./magg-output/architecture.ir.json
magg inspect ./magg-output/architecture.ir.json --summary
```

CLI commands are **design targets**, not implemented features.

### Output bundle

```text
magg-output/
├── architecture.ir.json     # Full semantic and source-linked graph
├── source-index.json        # Addressable syntax and byte-span references
├── source-manifest.json     # Included/excluded files and hashes
├── coverage.json            # Parse/analysis/resolution coverage
├── diagnostics.json         # Errors, warnings, ambiguous/unresolved facts
├── extraction-meta.json     # Versions, scope, rule catalog, IR hash
└── sources/                 # Exact included source bytes, mirrored paths
```

Artifacts reference each other with portable, normalized relative paths and content hashes. `source-index.json` can be separate for size/performance, but it is **part of the IR bundle contract**, not optional lost detail.

### Presenter handoff contract

The future presenter accepts a **validated IR bundle**, can query any graph entity or source reference, and may create reversible groupings (for example, a paper-style Attention block covering 20 underlying operation sites). It must retain the mapping from each presented element back to the exact contributing IR entity IDs. Presentation concerns cannot silently alter evidence or resolve unknowns.

---

## 9. Implementation milestones and acceptance criteria

### M0 — Contract before implementation

- [ ] Freeze entity/relation vocabulary, evidence statuses, required fields, and schema-version policy.
- [ ] Define canonical ID and sorting rules; write schema fixtures.
- [ ] Define the exact meaning of `resolved`, `conditional`, `ambiguous`, and `unresolved`.

### M1 — Lossless inventory and syntax indexing

- [ ] Enumerate the selected source tree deterministically.
- [ ] Preserve original bytes and accurate byte/line/column spans.
- [ ] Index AST and lexical tokens; preserve comments/docstrings.
- [ ] Prove source snapshot round-trip equality for fixtures.

### M2 — Symbol and structural extraction

- [ ] Resolve local definitions, imports, inheritance, attributes and binding sites.
- [ ] Detect supported `nn.Module` constructor patterns, argument expressions, and ownership sites.
- [ ] Distinguish declaration/construction/call sites and repeated/shared references.

### M3 — Computation, flow and symbolic reasoning

- [ ] Index all supported expressions, calls, returns, conditions and loops.
- [ ] Compute conservative intra-/interprocedural links where provable.
- [ ] Support explicit/symbolic tensor dimensions; record unmet preconditions.

### M4 — Complete IR, verification, reproducibility

- [ ] Emit typed graph, evidence chains, source index, unknowns and diagnostics.
- [ ] Validate all referenced IDs and source spans.
- [ ] Produce byte-identical canonical output on two identical runs.
- [ ] Never silently drop unsupported or unparseable in-scope content.

### Minimal acceptance fixtures

1. **Simple MLP:** exact constructor sites and nested `forward` calls; literal/symbolic dimensions preserved.
2. **Residual block:** branch and addition dependence accurately linked without flattening syntax.
3. **Shared module:** one construction site, multiple call sites; no false duplicated module instance.
4. **Conditional forward:** preserve both branch bodies and guard; never assert which branch always runs.
5. **Unresolved dynamic construction:** source preserved; uncertainty and reason explicitly emitted.
6. **Two-run reproducibility:** byte-identical outputs for the same source, scope, configuration, and SAE version.

---

## 10. Decisions fixed vs open

**Fixed for this design:** two-layer MAGG architecture; SAE first; static-only PyTorch/Python scope; maximal source-grounded detail; byte-preserving source bundle; evidence-linked graph; explicit unknowns; no visualization decisions in SAE; research-paper-style presenter later.

**Open for schema-design phase:** exact entity ID grammar; authoritative JSON Schema definitions; symbolic expression grammar; minimum supported Python/PyTorch patterns; dependency catalog versioning; scaling strategy for very large source graphs; IR schema migration policy.

**Next design task:** formalize `architecture-ir.v1.schema.json` and the rule/evidence catalog from this specification **before** implementing extraction algorithms.
