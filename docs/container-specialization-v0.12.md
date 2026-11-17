# Static Container Specialization — v0.12

Resolve a uniquely linked factory call chain and bounded `enumerate(config)` / `range(count)` nested module registration.

The extractor emits `container_configuration`, `container_stage`, `container_block_instance`, `container_member_instance`, and `container_transition_instance` nodes, with source-backed registration and class-reference edges. `model-index.json` includes `container_specialization` for fully verified construction patterns.

Scope: bounded integer counts, unique local classes, fixed `add_module` registration, known last-stage transition guard, and unique member constructor. Unknown counts, dynamic rebindings, altered guards, and unsupported construction patterns remain unresolved. The records describe conditional constructor registration, not runtime tensor dependencies.

Independent checks: TorchVision 0.25.0 DenseNet121/161/169/201; stage counts `[6,12,24,16]`, `[6,12,36,24]`, `[6,12,32,32]`, `[6,12,48,32]` match instantiated module inventories. FX `cat` calls are measured independently; they are not asserted as extracted AE flow edges. Byte-exact upstream source SHA-256 and deterministic IR hashes are included in the benchmark output.

```bash
ae extract <source-root> --entrypoint models.densenet:densenet121 --output sae
ae present sae --view structure --level operation --format svg --output figure
```
