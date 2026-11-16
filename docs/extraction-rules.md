# SAE inference rules — v0.1

The output records every derived fact with a rule identifier. Rules operate only on source syntax and static import hints.

| Rule ID | Evidence and scope |
|---|---|
| `pytorch-constructor-import-resolution/v1` | `self.attr = call(...)`, where a top-level unshadowed import alias resolves the callee syntax to `torch.nn.<name>`; stores literal/symbolic arguments. This is a *conditional* syntactic constructor identification, not proof of instantiated objects. |
| `constructor-call-link/v1` | A construction site's corresponding syntactic call site from the same AST expression. |
| `attribute-constructor-binding/v1` | An attribute assignment with direct constructor-call RHS binds to that construction site. |
| `unique-static-attribute-construction/v1` | One recognized class-level `self.attr` constructor site corresponding to a syntactic `self.attr(...)` call. If multiple/none exist, an unresolved record is created. |
| `syntactic-operand-dependency/v1` | One call or operation occurs as a nested operand/argument expression of another. Not runtime data-flow inference. |

Source-fact edges (`CONTAINS`, `HAS_ARGUMENT`, `HAS_BRANCH_STATEMENT` etc.) directly describe AST composition. They do not require a derived inference rule. `resolution: conditional` is deliberately used for derived semantic identification; dynamic imports, decorators, monkey-patching, assignment order, custom dispatch, and closure shadowing are outside the proven domain.

Never add an inferred edge by name similarity alone. Never equate repeated call sites with repeated module definitions.

## Added in SAE v0.2.0

- `local-project-import-resolution/v1`: unique local import symbol/module resolution.
- `unique-local-call-resolution/v1`: local callable declarations with conservative shadow checking.
- `static-nn-module-inheritance/v1`: source-visible direct and transitive `nn.Module` subclass hints.
- `local-module-construction/v1`, `unique-local-module-reference/v1`: local class construction assigned to `self.<attr>` and linked reuse.
- `lexical-definition/v1`, `unique-reaching-definition/v1`: intraprocedural source-level def-use; ambiguous candidates remain unresolved.
- `pytorch-constructor-syntax/v1`, `nested-pytorch-constructor/v1`: recognized `torch.nn` layer signatures and constructor nesting.
- `linear-last-dimension/v1`, `embedding-append-dimension/v1`, `conv-channel-signature/v1`, `mha-default-shape-template/v1`, `shape-preserving-module/v1`: conditional type-level shape templates; not executed tensor dimensions.
- `json-toml-parse/v1`: static configuration tree; file-level provenance only.

These rules are conservative but **not** an exhaustive semantic interpretation of Python.

## Conditional forward rules (v0.6)

`conditional-forward-values/v2` records condition guards, branch-local calls,
conditional value joins, and uncertain augmented assignments without executing
target code. `STATIC_CONDITIONAL_DEPENDENCY` edges connect each alternative to
a static merge; `STATIC_GUARDED_CALL` names control dependence, not value flow.
`STATIC_GUARD_DEPENDENCY` connects a known guard input to its guard/join.
Unverified alias effects and unpacking arity are explicit unresolved records.
