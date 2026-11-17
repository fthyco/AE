# Static factory specialization

`ae extract <source> --entrypoint models.resnet:resnet18 --output ae-bundle`

Supported: uniquely resolved local factory calls, fixed class selection, static integer stage counts, and `Sequential` construction with a verified first append and fixed `range(1, blocks)` repetitions.

`model-index.json` contains `factory_specialization`: selected class, stage counts, block identity, instance IDs, and source references. `architecture.ir.json` contains `factory_configuration`, `factory_stage`, and `factory_block_instance` nodes with construction and ordering edges.

Structure views display stages and instances. Flow views remain partial until nested `forward` execution paths are expanded.

Dynamic factories, unknown counts, alternate loop patterns, and ambiguous return paths remain unresolved. No target model is imported or executed by SAE.
