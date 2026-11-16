# MAGG — Model Architecture Graph Generator

MAGG is designed as **two independent components**:

1. **SAE (Static Architecture Extractor):** reads source code *without running it*, captures maximal source-grounded architecture facts into a deterministic intermediate representation (IR), and preserves exact source bytes.
2. **Presenter:** converts the IR into a unified research-paper-style model architecture figure. It may group and simplify, but must retain links to the original facts.

> **Extract at maximum fidelity. Abstract only when presenting.**

## SAE v0.3.0 — static architecture MVP

Supports Python 3.11+. PyTorch is **not** required: models are never imported, initialized, trained or traced.

```bash
python -m pip install -e '.[dev]'
magg extract examples/transformer_mvp --entrypoint model:Transformer --output magg-output
magg inspect magg-output
magg validate magg-output
pytest -q
```

Or run without installing using `PYTHONPATH=src python -m magg.cli ...`.

`magg extract` writes:

```text
magg-output/
├── architecture.ir.json    # Full linked AST + supported semantic facts
├── model-index.json       # Entrypoint-scoped cross-references for Presenter
├── source-index.json       # Tokens, comments and original byte offsets
├── source-manifest.json    # Included/excluded file inventory and hashes
├── coverage.json           # Source/parse and unresolved counts
├── diagnostics.json        # Parse and analysis diagnostics
├── extraction-meta.json   # Artifact checksums and verification summary
└── sources/                # Byte-exact copy of in-scope files
```

### Scope and accuracy

SAE captures every AST node and lexical token of every parsable included `.py` file; it also records supported definitions, `torch.nn` construction sites, argument expressions, operations, conditional syntax and module-use links. Every extracted semantic fact has a source reference; configuration trees currently have file-level provenance. Unresolved symbolic constructor arguments and ambiguous module-use links remain explicit. The v0.2 pass links local imports/functions/custom subclasses of `nn.Module`, resolves conservative intraprocedural lexical reaching definitions, emits symbolic templates for recognized PyTorch layers, and parses JSON/TOML configuration without evaluation. Branch/loop ambiguity is preserved explicitly.

The analyzer makes **no claim** of complete Python semantic inference or runtime behavior. Only `.py` and common text configurations are included by default; everything else is recorded as excluded. See [implemented features and limitations](docs/limitations.md).

The `--entrypoint` selects a unique top-level module symbol where statically resolvable; `entrypoint_target_id` references its IR node. **v0.3** adds `model-index.json`: a deterministic, source-grounded index of known construction sites, linked local custom module definitions, forward calls, operations and dependency-edge IDs. Unknown component assignments remain explicit. The original full IR is never filtered. JSON, TOML and safely parsed YAML are indexed as typed configuration trees, without executing interpolation or Python code.

### Project hierarchy

```text
AE/
├── README.md, LICENSE, pyproject.toml
├── docs/
│   ├── SAE_Technical_Specification.md
│   ├── ir-schema.md
│   ├── limitations.md
│   └── mvp-acceptance.md
├── schemas/
│   └── architecture-ir.v1.schema.json
├── src/magg/
│   ├── cli.py
│   ├── presenter/               # Reader, projection, health, scene, renderers
│   └── sae/
│       ├── api.py              # Extract, validate, inspect
│       ├── settings.py
│       ├── inventory/walker.py # Source inventory, exclusions
│       ├── parsing/            # Python AST/tokens and source spans
│       ├── analysis/analyzer.py# Semantic facts and links
│       ├── ir/                 # Graph, stable IDs, validation
│       └── output/bundle.py   # Canonical bundle and source snapshot
├── examples/{simple_mlp,mlp_mvp,cnn_residual_mvp,transformer_mvp}/
└── tests/
```

## Presenter v0.4.0 — source-backed research architecture diagrams

Presenter now reads only validated SAE bundles and renders a **source-grounded static module/call diagram**. It does not invent runtime tensor-flow links.

```bash
magg present magg-output --level block --mode exploration --format svg --output paper-output
magg present magg-output --level operation --mode exploration --format all --output detailed-output
magg present magg-output --level overview --mode publication --format pdf --output publication-output
```

Install the optional renderer for PDF export: `pip install -e '.[render]'`. Publication mode blocks partial output unless `--allow-partial` is explicitly set; blocked input never renders. Outputs include `presentation-health.json` and `presentation-manifest.json` with per-element source mappings. `complete` describes the supported selected view, **not** the model's runtime correctness.

See [Presenter implementation and limits](docs/presenter-implementation.md) and [Presenter contract](docs/presenter-plan.md).

### Design documentation

- [Full SAE specification](docs/SAE_Technical_Specification.md)
- [IR data contract](docs/ir-schema.md)
- [Implemented coverage and limitations](docs/limitations.md)
- [MVP acceptance and Presenter handoff](docs/mvp-acceptance.md)

See the technical specification for the intended expanded package hierarchy, rules, milestones and acceptance criteria. Modules that are still planned have not been represented by fake empty implementations.

- [Implementation phases and evidence](docs/implementation-phases.md)

## Presenter v0.6 — conditional static forward diagrams

The static forward view now preserves conditional alternatives (`if/else`),
augmented assignments (with explicit unverified alias effects), guarded calls
and conservative source-level value joins. These are **not runtime tensor flows**.
Use `--focus` to draw one selected module at a time from the complete IR:

```bash
magg extract examples/cnn_residual_mvp --entrypoint model:CNN --output sae-output
magg present sae-output --view flow --focus ResidualBlock --format all --output figure
```

`--focus` accepts a unique module basename or qualified name. Missing or
ambiguous names block rendering. Conditional connections are dashed; guarded
calls are dotted. Export manifests retain all source IR mappings. See
[static conditional flow acceptance](docs/forward-flow-v0.6-acceptance.md).

## Presenter v0.7 — `scientific-latex` paper theme

The default Presenter style follows the attached Chessformer architecture-report visual language: Latin Modern serif, white/gray fills, thin near-black borders and **orthogonal** arrows. No source evidence is added or changed by styling. SVG, PDF and TikZ share the same scene; PDF embeds Latin Modern Type1 locally when available and falls back to standard PDF serif fonts otherwise.

```bash
magg present sae-output --view flow --focus ResidualBlock \
  --theme scientific-latex --format all --output paper-figure
# Optional prior style for A/B comparison:
magg present sae-output --view flow --theme legacy --format svg --output prior-figure
```

For narrow paper figures, use `--focus` to export individual modules; long pipelines may still produce wide vector canvases. Details and acceptance checks: [scientific-latex theme](docs/scientific-latex-theme.md).
