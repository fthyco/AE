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
├── model-index.json       # Entrypoint-scoped references for Presenter
├── source-index.json       # Tokens, comments and original byte offsets
├── source-manifest.json    # Included/excluded file inventory and hashes
├── coverage.json           # Source/parse and unresolved counts
├── diagnostics.json        # Parse and analysis diagnostics
├── extraction-meta.json   # Artifact checksums and verification summary
└── sources/                # Byte-exact copy of in-scope files
```

### Scope and accuracy

SAE captures every AST node and lexical token of every parsable included `.py` file; it also records supported definitions, `torch.nn` construction sites, argument expressions, operations, conditional syntax and module-use links. Every extracted semantic fact has a source reference; configuration trees currently have file-level provenance. Unresolved symbolic constructor arguments and ambiguous module-use links remain explicit. The v0.2 pass additionally links local imports/functions/custom subclasses of `nn.Module`, resolves conservative intraprocedural lexical reaching definitions, emits symbolic templates for recognized PyTorch layers, and parses JSON/TOML configuration without evaluation. Branch/loop ambiguity is preserved explicitly.

The analyzer makes **no claim** of complete Python semantic inference or runtime behavior. Only `.py` and common text configurations are included by default; everything else is recorded as excluded. See [implemented features and limitations](docs/limitations.md).

Version 0.3 adds a deterministic model-index.json with composition, forward-call, operation and dependency references back into the unfiltered IR. YAML is parsed safely alongside JSON/TOML.

The `--entrypoint` selects a unique top-level module symbol where statically resolvable; `entrypoint_target_id` references its IR node. An unresolved entrypoint is recorded rather than guessed. It does not filter the complete syntax graph.

### Project hierarchy

```text
AE/
├── README.md, LICENSE, pyproject.toml
├── docs/
│   ├── SAE_Technical_Specification.md
│   ├── ir-schema.md
│   └── limitations.md
├── schemas/
│   └── architecture-ir.v1.schema.json
├── src/magg/
│   ├── cli.py
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

Presenter now reads only validated SAE bundles and renders a source-grounded static module/call diagram. It does not invent runtime tensor-flow links.

```bash
magg present magg-output --level block --mode exploration --format svg --output paper-output
magg present magg-output --level operation --mode exploration --format all --output detailed-output
magg present magg-output --level overview --mode publication --format pdf --output publication-output
```

Install the optional renderer for PDF export: `pip install -e '.[render]'`. Publication mode blocks partial output unless `--allow-partial` is explicitly set; blocked input never renders. Outputs include `presentation-health.json` and `presentation-manifest.json` with per-element source mappings. `complete` describes the supported selected view, not model runtime correctness.

See [Presenter implementation](docs/presenter-implementation.md) and [Presenter design contract](docs/presenter-plan.md).

### Design documentation

- [Full SAE specification](docs/SAE_Technical_Specification.md)
- [IR data contract](docs/ir-schema.md)
- [Implemented coverage and limitations](docs/limitations.md)
- [MVP acceptance and Presenter handoff](docs/mvp-acceptance.md)
- [Presenter implementation plan and Extraction Health Contract](docs/presenter-plan.md) *(design only)*

See the technical specification for the intended expanded package hierarchy, rules, milestones and acceptance criteria. Modules that are still planned have not been represented by fake empty implementations.

- [Implementation phases and evidence](docs/implementation-phases.md)
