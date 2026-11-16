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
