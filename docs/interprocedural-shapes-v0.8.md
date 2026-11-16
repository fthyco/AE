# MAGG v0.8 — Minimal Interprocedural and Shape Evidence

**Scope:** Extract static, reproducible value and shape evidence without
importing/executing target models, modifying the source, or guessing paths.

## Evidence extensions

| Relation/node | Proven source-level meaning | Critical limitation |
| --- | --- | --- |
| `STATIC_CALL_ARGUMENT_BINDING` | A unique, unambiguous actual expression's source value is bound to a formal parameter | Not a runtime activation edge; repeated callees and dynamic binding rejected |
| `STATIC_CALL_RETURN` | A uniquely analyzable returned value relates to a specific unique call site | Recursion, ambiguous/multi-context calls rejected |
| `symbolic_shape_fact` | A type-level output signature or conservatively propagated template for a source value | Conditional, not an observed runtime tensor |
| `HAS_SYMBOLIC_SHAPE` | Maps the value node to that documented conditional template | No executable validity claim |
| `symbolic_shape_conflict` | Source-derived, definite numeric mismatch candidate across supported Linear signatures | Call might never execute; not a runtime error guarantee |

## Step-by-step implementation

1. SAE `forward_flow.py` creates summaries for unambiguously called local helpers
   in addition to model forwards. Existing value edges remain unchanged.
2. SAE `interprocedural.py` binds only uniquely resolved calls, with exact
   argument positions/names, available argument provenance, analyzable return,
   no recursion, and a unique call-site context per callee.
3. SAE `symbolic_shapes.py` seeds `Linear`, `Embedding` and supported `Conv*`
   output templates; it propagates through proven call-argument/return links,
   shape-preserving modules and equal-shape merges only. Unknowns are not filled.
4. Presenter `flow.py` displays facts and attaches their source IR IDs to each
   visual element. `flow_layout.py` labels shapes **symbolic**.

## Explicitly deferred

Full SSA, interprocedural alias analysis, general function/context sensitivity,
full spatial propagation, dynamic collection unpacking, F.relu call identity,
shape solving for MatMul/Concat/Reshape, true runtime execution order.

## Run

```bash
python -m pip install -e '.[dev]'
pytest -q
magg extract examples/transformer_mvp --entrypoint model:Transformer --output /tmp/sae
magg present /tmp/sae --view flow --level block --format svg --output /tmp/fig
```

## Acceptance

- Relevant value sources, formal parameter and returns retain stable IR IDs.
- Unknown or multiple/reentrant contexts do not create fake edges.
- A nested `Block(Linear(128,64))` propagates a conditional `[...,64]`
  template to its unique caller, with correct source references.
- A mismatch of known numeric Linear interface requirements produces a
  diagnostic without executing the project.
- Presenter annotates only backed shape facts; structural view remains stable.
- Source bundles validate and render deterministically across hash seeds.
- Real-model tests may produce `partial`, not falsely claim total recovery.

The analysis is intentionally finite and incomplete. **Static provenance is
not runtime verification.**
