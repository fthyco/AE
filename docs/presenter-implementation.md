# MAGG Presenter v0.4.0 — implementation and acceptance

Presenter is a **read-only consumer** of an existing SAE v0.3.0 extraction bundle. It does not parse, import, execute, or reinterpret the target model. Source-backed identities and provenance remain anchored in architecture.ir.json and model-index.json.

## Completed staged implementation

| Stage | Code | Acceptance gate |
| --- | --- | --- |
| 1. Contract reader | reader.py | validate_bundle and validate_model_index verify source bytes, hashes and references; corrupt bundles never render |
| 2. Conservative projection | projection.py | overview, block and operation levels retain module composition, forward calls and supported operators; every rendered fact retains IR IDs |
| 3. Health classification | health.py | complete, partial and blocked reflect the selected view; unknown components remain explicit; unrelated parse errors do not downgrade the selected model |
| 4. Layout and renderers | layout.py, render.py | One deterministic geometric scene produces black-and-white SVG, PDF and TikZ. Nested blocks do not imply runtime tensor flow |
| 5. Integration | api.py, cli.py, tests/test_presenter.py | Publication safety policy, stable output files, and MLP/CNN/Transformer fixtures |

## Installation and usage

```bash
python -m pip install -e '.[dev]'
magg extract examples/transformer_mvp --entrypoint model:Transformer --output sae-output
magg present sae-output --level block --mode exploration --format all --output paper-output
```

--level: overview, block, operation. --mode: exploration, publication.
--format: svg, pdf, tikz, all. PDF requires optional CairoSVG: pip install -e '.[render]'.
TikZ outputs standalone architecture.tex; compile with pdflatex.

Publication withholds partial diagrams by default (CLI exits 2). Explicit --allow-partial permits export with a permanent partial disclaimer. Blocked always refuses to render.

## Outputs

paper-output/
- architecture.svg (if requested and allowed)
- architecture.pdf (if requested and allowed)
- architecture.tex (if requested and allowed)
- presentation-health.json (always)
- presentation-manifest.json (always)

The manifest maps each visual ID and connection to precise source-backed IR IDs and source references; it contains input/output hashes, omitted view references and renderer style version.

## Source-evidence requirements

- Nesting means declared structure/construction only, not a proven runtime call sequence.
- Dashed arrows mean USES_MODULE_REFERENCE only.
- Dotted arrows mean EXPRESSION_DEPENDS_ON only; **not verified tensor flow**.
- All forward calls are syntactic call facts; no runtime execution order is implied.
- Unknown modules and unsupported assignments remain unresolved, not guessed.
- complete means the selected static view is adequately supported, not that code runs or that the entire Python semantics have been recovered.

## Verification

Run: python -m pytest -q

Presenter's integration tests cover three real fixtures, source provenance for blocks and links, missing imports/components, unresolved entrypoints, unrelated syntax errors, corrupt IR, publication policies, byte reproducibility and stable outputs across PYTHONHASHSEED. PDF and TikZ are separately rendered/compiled and visually reviewed.

## Deliberate MVP boundary

This is a truthful static module-and-call architecture figure, not yet an experimentally verified model forward-dataflow graph. The SAE has not proven all tensor dependencies. Future tensor-flow diagrams require new SAE evidence rules, rather than speculative rendering logic. Interactive editing and architecture diffs remain deferred.

## v0.5 — Evidence-based forward value flow

A separate `--view flow` presenter projection consumes new SAE relations `STATIC_VALUE_DEPENDENCY` and `STATIC_FORWARD_RETURN` plus `forward_input/output/unpack` facts. It never uses lexical call order to invent an arrow. Supported scope: straight-line AST operands, local assignments, nested calls, residual Add and source-backed return. Unsupported control flow, in-place mutation and ambiguous unpacking remain explicit in health diagnostics, and publication mode withholds partial diagrams by default. The flow renderer uses a deterministic left-to-right dependency DAG; SVG, vector PDF (ReportLab invariant mode) and TikZ share the scene. This does not prove tensors' runtime shape, numerical behavior or executed paths.
