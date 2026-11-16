# MAGG Presenter — Implementation Plan and Extraction Health Contract

**Status:** Design specification; not yet implemented.  
**Repository:** `fthyco/AE` · **SAE baseline:** v0.3.0 · **IR/model-index schema:** v1  
**Purpose:** Produce consistent research-paper architecture diagrams from source-grounded SAE outputs.

> **Extract at maximum fidelity. Abstract only when presenting.**

## 1. Scope and principles

The Presenter is **only a consumer of existing SAE data**. It never imports or executes target models, reparses Python, changes the canonical IR, or guesses missing nodes and connections. Its job is to select, group, lay out and render supported architectural facts, with every visible block and connector traceable to one or more valid IR IDs. The final output is a monochrome scientific diagram with an explicit statement of its static-evidence scope.

A diagram is not evidence that the model is executable or numerically correct. SAE's lexical call order is **not** runtime execution order; syntactic expression dependencies are **not** verified tensor flow.

**MVP:** From one validated SAE bundle and selected entrypoint, produce a deterministic, legible SVG and a health report. PDF and TikZ follow as exporters of the **same** scene, never separate analysis pipelines.

## 2. Existing SAE → Presenter data contract

Input is a directory containing:

```text
sae-output/
├── architecture.ir.json       # Source-backed nodes/edges and provenance
├── model-index.json           # Entrypoint-scoped components, calls, operations
├── source-index.json          # Original lexical tokens and offsets
├── source-manifest.json       # Included/excluded inventory and hashes
├── coverage.json              # Parsing and uncertainty coverage
├── diagnostics.json           # Parse/analysis issues
├── extraction-meta.json       # Checksums and verification
└── sources/                   # Byte-identical input source snapshots
```

Preserve the current handoff contract in `docs/presenter-contract.md`. Before projecting:
1. Call the existing bundle integrity validator. Any failure blocks rendering.
2. Require a uniquely resolved model `root_definition_id` from `model-index.json` for model-specific drawings.
3. Confirm that all projected IR node/edge references exist and have source provenance.
4. Respect `evidence_class`, resolution status, rule identifiers and `unresolved` records.
5. Never promote `EXPRESSION_DEPENDS_ON`, `READS_LEXICAL_DEFINITION`, or lexical call order to proof of tensor flow.
6. Keep `architecture.ir.json` and `model-index.json` immutable; derive a separate presenter projection.

The new health contract is **Presenter-owned** and does not require a redesign of SAE or its schemas.

## 3. One simple pipeline

```text
SAE bundle
  → Reader + integrity check
  → Architecture projection + grouping
  → Health evaluation (complete / partial / blocked)
  → Stable layout
  → Unified scene + SVG/PDF/TikZ exporters
  → Diagram + health + mapping manifest
```

**Reader** indexes existing IR and model references. **Projection** selects modules/operations and groups them into blocks while preserving all contributing IDs. **Health** evaluates issues relevant to this selected model and level. **Layout** chooses stable positions and connector routing. **Renderer** produces publication-style vector output. There is **no second extractor**, no AI-based architectural completion, and no different parsing engine for Publication mode.

## 4. Presenter projection

Each visible **node** has a stable visual ID, label, kind, optional known/symbolic dimensions, contributing `ir_node_ids`/`ir_edge_ids`, evidence status and source references. Each visible **connection** has endpoints, relation kind, contributing IR edge IDs, and evidence status. A collapsed group retains mappings to *all* source elements it represents.

Supported levels:
- `overview`: selected model and supported major submodules.
- `block` (default): hierarchy, subcomponents, supported calls and operations.
- `operation`: supported operations and source-level dependency facts.

**Truthful relationship rendering**

| Evidence in IR | Visual meaning |
| --- | --- |
| Containment/ownership | Nested module box |
| Linked call/module use | Labeled `calls` or `uses` relationship |
| Syntactic expression dependency | Explicitly labeled source-expression dependency |
| Proven semantic data-flow dependency, if provided | Directed tensor/data-flow arrow |
| Conditional or unresolved association | Marked unknown in Exploration; never confirmed |

Do **not** connect two boxes because they happen to appear consecutively in source code. Do **not** fabricate residual, attention or sequential flow arrows. If evidence only supports module structure, render a **module architecture view**, not a purported verified runtime forward graph.

## 5. Extraction Health Contract

Health describes the **requested diagram's evidence coverage**, not a universal correctness score. Derive it from validated IR, model index, coverage, diagnostics, manifest, and chosen view. Emit `presentation-health.json` separately from SAE artifacts.

| Status | Definition | Result |
| --- | --- | --- |
| `complete` | The facts/links needed for this particular static view are sufficiently supported within its declared scope. | Render with scope note. No claim about runtime correctness. |
| `partial` | Useful supported architecture exists, but a **view-relevant** definition, dependency, condition or required dimension is missing/uncertain. | Render supported parts with conspicuous uncertainty and report. |
| `blocked` | Bundle integrity failed, entrypoint cannot be selected, or no defensible content exists for this requested view. | Do not render architecture. Emit diagnostic report. |

Precedence: `blocked > partial > complete`. A problem in an **unrelated** source file does not automatically downgrade the selected view. An unknown dimension degrades only views requiring that dimension; otherwise keep the symbolic name or show `?`. `complete` means complete **for the constrained static view**, not complete recovery of all Python behavior.

Recommended issue codes:

| Code | When | Default effect |
| --- | --- | --- |
| `INVALID_BUNDLE` | Tamper, missing artifact, broken reference | Block |
| `ENTRYPOINT_UNRESOLVED` | Missing/ambiguous selected model | Block |
| `NO_SUPPORTED_VIEW` | Not enough evidence to draw selected view | Block |
| `SOURCE_PARSE_ERROR` | Relevant file did not parse | Partial, block if root unusable |
| `MISSING_DEFINITION` | Relevant import/custom block unavailable | Partial |
| `UNRESOLVED_CONNECTION` | Necessary link cannot be proven | Partial |
| `UNKNOWN_DIMENSION` | Required dimension unavailable | Partial only if dimension required |
| `UNSUPPORTED_OPERATION` | Operation cannot be semantically expanded | Show generic call with provenance |
| `EXCLUDED_SOURCE` | Relevant file excluded by SAE | Partial or block if root affected |

Every issue should record `code`, `severity`, `message`, `related_ir_ids`, any real `source_refs`, and `affects_view`. Never fabricate IDs or source spans.

**Exploration mode (default):** render partial results, mark unknowns, include clear legend and health report.  
**Publication mode:** normally render only `complete`. A `partial` view is rejected unless `--allow-partial` is explicit; then the figure must permanently state **Partial / not fully verified** and include the issue/scope summary. A `blocked` view never renders. Modes control publication policy, not the extracted facts.

Example health payload (IDs omitted rather than fabricated):

```json
{
  "schema_version": "1.0",
  "status": "partial",
  "mode": "exploration",
  "level": "block",
  "entrypoint": "model:MyModel",
  "scope": "Static source-backed architecture of the selected entrypoint",
  "issues": [{
    "code": "MISSING_DEFINITION",
    "severity": "warning",
    "message": "Encoder is referenced but its implementation is unavailable",
    "related_ir_ids": [],
    "source_refs": [],
    "affects_view": true
  }],
  "render_allowed": true
}
```

## 6. Broken/incomplete input behavior

| Input problem | Required result |
| --- | --- |
| Syntax error | Preserve other supported parts; report file; block if model root unusable. |
| Missing import or missing custom block | Show known reference as unresolved; never invent implementation. |
| Undefined/ambiguous variable | No guessed dependency or arrow. |
| Unknown tensor shape | Keep actual symbol or `?`, not estimated values. |
| Unsupported function or operator | Show generic source-backed call only. |
| Unknown runtime branch | Show branch fact or omit unproved execution path; no assumed outcome. |
| Invalid IR or altered artifact hashes | Block in both modes. |
| Error outside selected model | Report if useful, but do not falsely mark selected view as incomplete. |

A correct extraction does not guarantee runnable model code.

## 7. Research-paper visual style

Default **Monochrome Scientific**: white canvas, black/gray outlines, consistent type, restrained spacing, hierarchical boxes and clean labeled connections. Use line patterns and text labels for uncertainty, never color alone. Include dimensions only when source-backed. No invented encoder/decoder/head labels if they cannot be justified from definitions or a declared mapping.

One deterministic **scene graph** is shared by:
- SVG — primary MVP vector output;
- PDF — vector export from the scene/SVG using pinned tooling;
- LaTeX/TikZ — same scene's boxes, paths and labels, not recomputed architecture.

Fixed ordering, stable IDs, layout rules, font assets and dependency versions enable byte-identical output in a pinned environment. Cross-platform text metrics require pinned fonts for exact reproducibility.

## 8. Minimal proposed project hierarchy

```text
AE/
├── docs/
│   ├── presenter-contract.md      # Existing, preserved
│   └── presenter-plan.md          # This design + health contract
├── src/magg/
│   ├── cli.py                     # Add `present` subcommand
│   ├── sae/                       # Existing, unchanged
│   └── presenter/
│       ├── __init__.py
│       ├── api.py                 # Single public entrypoint
│       ├── reader.py              # Validate & index SAE bundle
│       ├── projection.py          # Selection/grouping + provenance
│       ├── health.py              # Status/issue evaluation
│       ├── layout.py              # Stable placement and routing
│       └── render.py              # Scene model and SVG/PDF/TikZ
├── tests/
│   └── test_presenter.py
└── examples/
    ├── mlp_mvp/
    ├── cnn_residual_mvp/
    └── transformer_mvp/
```

Do not add plugin frameworks, a second generic IR, duplicate parsers, or empty placeholder modules. Presenter-specific health and manifest reports may start with small validated Python dictionaries; new JSON Schemas can wait until external consumers need them.

## 9. Planned CLI and generated files

```bash
magg present sae-output --level block --mode exploration --format svg --output diagram-output
magg present sae-output --level overview --mode publication --format pdf --output paper-output
```

In Publication mode the second command blocks when health is `partial` unless `--allow-partial` is supplied.

```text
diagram-output/
├── architecture.svg             # Only when rendering allowed
├── presentation-health.json     # Always, including blocked
└── presentation-manifest.json   # IR→visual mapping, omissions, hashes, scope
```

For other formats output `architecture.pdf` or `architecture.tex`. The manifest records input artifact hashes, style/version, mode/level, visual-to-IR mappings and omissions with reasons. These are **derived artifacts** and never overwrite SAE data.

## 10. Implementation order and acceptance

| Stage | Work | Exit condition |
| --- | --- | --- |
| 1 | Reader and Health Contract | Reject corrupt/ambiguous input; correct view-scoped status and reasons |
| 2 | Architecture Projection | Every block/link backed by real IR IDs, grouping preserves provenance |
| 3 | Layout and SVG | Readable deterministic `overview/block/operation` output |
| 4 | PDF/TikZ and Publication policy | Same architecture across exports; enforced partial/blocked rules |

**Test matrix:** MLP, CNN with residual structure, Transformer with nested attention, missing custom module, missing import, invalid syntax, unknown dimension, unsupported call, unresolved entrypoint, corrupt bundle, irrelevant-file errors, repeated runs and hash seeds, IR mapping integrity, no invented tensor-flow arrows. Manually inspect overlaps, arrow crossings, label truncation and legibility.

**MVP is done when** a supported model's SAE bundle produces a readable, source-traceable, deterministic scientific figure and precise health/omission report, with honest partial or blocked behavior. Do not require advanced aliasing, interprocedural Python analysis, model execution, interactive editing or architecture diffing.

## 11. Deferred

Interactive editing, automatic cross-version diffs, runtime tracing and complete Python semantic recovery. If a real model requires missing source evidence, extend SAE and version its extraction rule; **never patch the gap by speculative Presenter logic**.
