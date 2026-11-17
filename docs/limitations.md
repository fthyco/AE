# AE — Static analysis boundaries

AE never imports, initializes, runs or trains the target model. A successful extraction or diagram does not certify model correctness.

Supported: AST/tokens, source snapshots, recognized PyTorch construction sites, local imports/definitions, conservative intra-function dependencies, uniquely resolved limited interprocedural bindings, branch alternatives, augmented assignments, fixed-position tuple/list provenance, safe configuration parsing and conditional symbolic shape facts.

Not generally supported: dynamic imports/dispatch, aliasing or parameter sharing, recursion or multi-use call sensitivity, starred/dynamic unpacking, arbitrary runtime paths, verified tensor-flow graphs, general broadcasting, MatMul/Concat/Reshape shape rules, complete spatial Conv shape propagation or Hydra/OmegaConf runtime semantics. `torch.chunk` does not guarantee a requested number of return elements. Unresolved cases remain visible.

`complete` means complete within a declared supported static view, never full runtime semantics; `partial` reports relevant unknowns; `blocked` rejects invalid evidence. Resource limits and exclusion rules apply; analyzed source is not executed.
