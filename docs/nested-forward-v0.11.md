# Nested Forward — v0.11

`ae extract <source> --entrypoint module:Class --output bundle`

`ae present bundle --view nested --level block --format all --output figures`

## Extraction

- `forward_origin`, `forward_owner_definition_id`: unique declared or inherited `forward()` method.
- `nested_forward_calls`: source-backed call-site to known component `forward()` associations.
- `nested_module_templates`: module constructors appended to known containers; dynamic repetition counts remain unknown.
- TorchScript `_overload_method` declaration stubs are excluded from the selected implementation.
- Every relation retains source spans and IR IDs.

## Limits

- Single-base static inheritance with directly verified imports or package reexports.
- No runtime module invocation, general MRO, parameter aliasing, tensor-flow equivalence, or automatic dynamic-loop unrolling.
- `nested` shows invocation references separately from the `flow` value-dependency view.
- Models with dynamic templates or unresolved calls remain `partial`.

## Acceptance

`python -m pytest -q`

`python scripts/verify_nested_real_models.py`

Reference models: TorchVision DenseNet121 and segmentation-models-pytorch U-Net with ResNet34 encoder. Model source bytes are copied without editing and checked against installed package source hashes. FX/ShapeProp/runtime are independent verification signals, not a declaration of complete equivalence.
