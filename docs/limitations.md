# SAE v0.3.0 — verified capabilities and remaining limitations

## Implemented

- Non-executing, deterministic file inventory and byte-exact source snapshots.
- Every Python AST node and lexical token in included, parsable `.py` files.
- Direct syntax facts for declarations, imports, assignments, calls, operators, returns and control constructs.
- Local project import/call and class-method resolution where a unique static declaration exists.
- Direct and transitive local `nn.Module` inheritance hints, custom module constructor sites and reuse links.
- Intraprocedural source-level reaching-definition edges with conditional/loop ambiguity preserved.
- Symbolic PyTorch shape templates and nested constructor containment; static JSON/TOML data trees.
- Source evidence, stable identifiers, integrity validation, explicit unresolved evidence and checksums.
- Entrypoint resolution to a top-level module symbol when unique.
- Entrypoint-centric index with linked construction sites, symbolic layer specs,
  source-order forward calls/operations and IR dependency references.
- Safe, typed YAML configuration trees; duplicate keys, unsafe tags and
  multi-document inputs are diagnosed rather than evaluated.

## Not yet complete

- General Python imports and re-exports, closures and name lookup with arbitrary dynamic scope behavior.
- Full alias and parameter-sharing analysis, descriptor and property resolution, dynamic `setattr`, metaclasses.
- True static SSA for all control constructs, interprocedural reaching definitions, pointer/dataflow or runtime tensor graph.
- Nontrivial symbolic shape propagation through all PyTorch/functional operations.
- Hydra/OmegaConf interpolation semantics, dynamic YAML evaluation and user-defined factories.
- Systematic measurement of recall/completeness on real-world models such as Chessformer and AWN.
- Presenter, paper-quality architecture figures or graphical export.

All shape templates are conditional type-level formulas, not observed tensor shapes.
Source-dependency edges are **not** runtime tensor-flow proof. Python runtime
semantics are not generally decidable from static source. SAE records unknowns
rather than claiming false certainty.

## Security

Analyzed Python is not imported or executed; no `eval`, `exec`, `torch.load`
or pickle is used. Symlinks are rejected and file-count/byte limits are enforced.

## v0.6 static flow extensions

Supported `if/else` branch environments yield conditional join facts; `+=` and
other augmented assignments yield operand dependencies with mutation caveats.
Guarded calls remain conditional, and shape/alias/runtime behavior is not
claimed. `--focus` can scope the diagram to one model definition.


## v0.8 static-summary additions and boundaries

- Unique single-use local calls and custom component forwards may have
  source-backed formal-parameter / caller-argument and return / call-site edges.
- Only the supported straight-line summaries are used. General polymorphism,
  aliased module instances, multi-site context sensitivity, recursion, star
  unpacking, default arguments and runtime dispatch remain unresolved.
- Symbolic shape facts are **conditional templates**. A `Linear` output
  template `[..., d]` is not an observation of a tensor. Known numeric
  last-feature conflicts are surfaced as diagnostics; arbitrary broadcasting,
  reshaping, convolution spatial formulas and context-sensitive batch axes
  remain unproved. Presenter shows only existing IR facts.

## v0.9 positional tuple/list evidence

Literal fixed-size tuple/list source expressions and supported single-use,
nonrecursive cross-function returns provide indexed collection provenance.
Each element can retain its own conditional symbolic shape. Dynamic or starred
arity, `torch.chunk`, nested unpacking and general concat output shape remain
unproven; no tuple aggregate is treated as a tensor. Presenter draws only
source-supported cross-module positional links.

Bounded static instance-forward expansion is currently verified only for the default ResNet18 and ResNet50 factories and fixed, explicitly supplied input sizes. The separate `instance` view is not a general runtime graph; independent FX/hooks verification is mandatory for a model-level claim.

Full DenseNet121 `instance` forward expansion is available only for an audited original TorchVision 0.25.0 source fingerprint, default factory arguments and explicit concrete input shapes. Modified or unsupported sources remain unresolved. Independent FX/hooks runtime verification is separate from AE extraction.

U-Net ResNet34 instance-forward expansion is limited to audited, unmodified SMP 0.6.0 and TorchVision 0.25.0 source files, the default topology, and explicit concrete input sizes with H/W divisible by 32. Other configurations remain unsupported and fail closed. Full FX/hooks evidence is produced only by a separate verifier.

ViT-Tiny expansion: only original SHA-256-pinned timm 1.0.22 `vit_tiny_patch16_224`, explicit fused/unfused attention branch and concrete RGB 224×224 inputs. Dynamic/nondefault/modified implementations are unsupported.
