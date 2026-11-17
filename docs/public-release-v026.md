# AE v0.26 — Offline Viewer and Public Release Gate

Architecture Extractor remains a **source-only** Python/PyTorch architecture
inspection tool. An analyzed repository is read from disk; it is **not imported
or executed**. Only bounded audited model/input variants may provide an exact
source-proven instance graph. All other projections remain clearly labeled
`partial` or `blocked`.

## Clean installation

Python 3.11+ required. No PyTorch is required to install or run AE:

```sh
python -m pip install architecture_extractor-0.26.0-py3-none-any.whl
ae extract ./source --entrypoint model:Model --output ./bundle
ae validate ./bundle
ae viewer ./bundle --view structure --output ./viewer
```

Open `viewer/index.html` locally in a modern browser. No HTTP server, remote
assets, external scripts, accounts, or network permissions are needed. The
viewer has Architecture, Block and Operation levels; zoom and pan; source
identity inspection; a drill-to-operation highlight; and the original SVG
files and evidence manifests.

For a verified original model and input shape, choose `--view instance`:

```sh
ae extract ./torchvision-source --entrypoint torchvision.models.vgg:vgg16 \
  --input-shape 1,3,224,224 --output ./bundle
ae viewer ./bundle --view instance --output ./viewer
```

Only the exact pinned original versions and audited inputs described in the
v0.16–v0.25 documents are supported for proven full-instance views. For source
trees outside that scope (AWN, Chessformer, custom projects), start with
`--view nested` or `--view structure` and respect partial labels.

## Evidence contract

- The viewer uses `ae present` source-grounded manifests for all levels.
- Every visible block maps to original IR IDs and source spans.
- Clicking Architecture/Block elements highlights source-mapped Operation nodes.
  It does not create new edges, infer missing tensor shapes, or imply runtime
  execution.
- Unknown or unsupported levels remain disabled; a corrupt bundle cannot
  leave stale SVG artifacts in an output folder.
- Semantic operation colors are presentation hints; canonical shapes,
  original connections, and source-proven identity are authoritative.
- Manifest hashes and original SVGs are emitted alongside the offline page.

## Release acceptance

The `public-release-v026.yml` workflow independently builds a wheel and
source distribution, performs metadata checks, installs the **wheel into an
isolated environment without dev extras or torch**, and exercises the shipped
`ae extract`, `ae validate`, and `ae viewer` commands. It also tests
valid, partial, blocked, deterministic and corruption behavior.

Prior to declaring a public release, require current-head green on the full
suite (including previous models), a completed clean-wheel acceptance, review
of the generated offline HTML, and artifacts retained in GitHub Actions.
CI artifacts are build evidence, **not a published PyPI release**. No PyPI
publish should be claimed unless the package is actually uploaded there.
