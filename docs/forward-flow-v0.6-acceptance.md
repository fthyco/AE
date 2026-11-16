# MAGG v0.6 — conditional forward-flow increment

## Implementation

SAE remains non-executing and source-grounded. It retains all source bytes, AST and prior graph facts.

- `if`/`else`: records guarded call sites and joins two supported branch value environments using conditional dependency edges, without selecting a branch. When a value is undefined in one branch, no unconditional return relation is created.
- `AugAssign`: records previous-target and RHS dependencies into an explicitly conditional operation. Because Python can mutate aliased objects, side effects remain unresolved.
- Tuple unpacking from calls remains **conditional**, preserving element position metadata while acknowledging unverified arity and tuple-element identity.
- Unsupported loops, returns inside branches, and dynamic expressions continue producing explicit incomplete-flow diagnostics. No speculative edges are emitted.
- New rule `conditional-forward-values/v2` covers conditional facts; legacy `straight-line-forward-value/v1` remains on previously supported edges.

## Presenter

`--view flow --focus BasicBlock` selects exactly one uniquely matching indexed module, avoiding unrelated lanes and preserving the full SAE IR and mapping manifest. Unknown or ambiguous focus blocks output. The diagram distinguishes solid value dependencies, dashed alternative value dependencies, and dotted guarded-call/control relationships. Both PDF and TikZ are generated from the same deterministic scene.

```bash
magg extract examples/cnn_residual_mvp --entrypoint model:CNN --output output/sae
magg present output/sae --view flow --focus ResidualBlock --format all --output output/figure
```

## Acceptance and residual limits

- Source-derived conditional merge and augmented assignment tested on an AWN-like `BasicBlock` fixture.
- No runtime Tensor Flow, alias identity, mutation side effects, branch predicate truth, or dynamic tuple arity are asserted.
- Publication mode still withholds partial views by default; the sourced AWN-style `BasicBlock` appropriately remains **partial**.
- Per-module figures improve readability but do not yet automate publication-quality layout for all large models.
- Remaining priorities: more precise call/attribute semantics, collection provenance and truly scalable layout; measure original-repository coverage instead of declaring full-model fidelity.
