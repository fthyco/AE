# v0.27 — Generic source hierarchy, source taxonomy, and runtime reference

## Separation of evidence and execution

AE provides TWO separate and explicitly labeled modes.

1. Safe default: source-only AE extract and offline AE viewer, with NO target
   module import and NO PyTorch dependency.
2. Deliberate opt-in: AE runtime-audit imports and executes TRUSTED user Python
   with synthetic inputs, providing an independent FX/hook reference.
   This mode is NOT sandboxed, and is never invoked by extraction or presentation.

A successful FX trace does not prove equivalence with AE's static IR, or prove
the graph is complete for other inputs. FX failures are reported as unsupported
without invented runtime tensor shapes or edges.

## Generic source architecture: Architecture / Block / Operation

The source-backed structure viewer uses a uniquely indexed root, constructor
sites, linked local module definitions, nested construction templates, forward
calls, operation sites, and existing IR edges.

- Architecture: root and source-declared components.
- Block: selected source-defined submodule decomposition and lexical call sites.
- Operation: original operation syntax sites and backed expression relations.

These are SOURCE declarations, not dynamic execution counts. A constructor
loop does not prove a specific number of runtime layers.

The default viewer mode AUTO selects a verified source-pinned instance graph
where its three hierarchy levels are supported, otherwise structure.
Source-group membership IDs allow clicking a parent and highlighting only
the original operation nodes supported by that source group, even for unknown
new repositories. This is containment identity, not fabricated tensor flow.

## Semantic classification with unknown fallback

Static component colors are assigned only for explicitly recognized source
constructor names (Linear, Conv2d, LayerNorm, MultiheadAttention, etc).
Unrecognized custom modules stay neutral. Generic matrix multiplication is
not mislabeled as attention or elementwise addition. Semantic colors and labels
are presentation hints, not execution evidence.

## Opt-in PyTorch reference command

Install PyTorch independently in a trusted virtual environment. Prepare a
zero-argument callable returning a fully configured torch.nn.Module:

~~~python
def build():
    return MyModel(config)
~~~

Then explicitly allow execution of the trusted source:

~~~sh
ae runtime-audit ./trusted-source --factory model:build \
  --input-shape 1,3,64,64 --allow-execute --output ./runtime-reference
~~~

Multiple tensor inputs are supported by repeated --input-shape flags.
Repeated --input-dtype flags support float32, int64, etc. Output stores FX
node targets, ordered operand edges, inferred tensor shapes and observed leaf
module runtime calls; tensors and weights are never serialized.

If factory invocation, tracing, shape propagation or runtime execution fails,
the result is UNSUPPORTED, not a fake graph. Data-dependent conditionals may
not be FX-traceable. Factory code executes with the current user's privileges:
NEVER use this command on unknown/untrusted projects.

## Original external research code audit

Tests pin the original AWN and Chessformer CF-ACT/CF-CAH commits and compare
source-only nested views with genuine three-level structure views. They enforce
original AST class identity, repeatable IR, source/span and edge provenance,
and label ALL unsupported runtime facts.

Do not redistribute the original AWN sources or create synthetic "AWN result"
graphs from a static source index. Research outputs include ONLY sanitized JSON
summaries (no third-party source or bundles).

Full runtime equivalence on AWN/Chessformer is still not validated. It requires
exact original dependencies, authorized executable factories, fixed config/
input cases and independent runtime operation, edge, and shape references.
Dynamic paths which cannot be traced must be marked unsupported.

## Acceptance

- Source-only tests run without torch and without target project imports.
- Opt-in trusted runtime tests with CPU torch verify traced and unsupported
  paths.
- Chromium runs against offline HTML verify search, pan/zoom, provenance
  selection and drill-down without any external network requests.
- Existing eight original model acceptance gates must remain green.
- A GitHub Release and PyPI publication are different distribution actions.

## PyPI Trusted Publisher (separate authorization)

A dispatch-only workflow is included at .github/workflows/pypi-publish.yml.
It can publish ONLY a version matching an existing, non-draft GitHub Release
tag, with a clean build and metadata check. It uses PyPI OIDC instead of
storing API tokens in GitHub.

First register the project name architecture-extractor in an authorized
PyPI account (or choose another available distribution name). Configure
the PyPI project's Trusted Publisher as GitHub owner fthyco, repository AE,
workflow pypi-publish.yml, environment pypi. Configure GitHub environment
approval as appropriate. Then manually dispatch this workflow from main
for the already-released tag. Without this external PyPI ownership and
publisher authorization, the workflow cannot publish: passing GitHub CI
alone must never be described as a PyPI upload.
