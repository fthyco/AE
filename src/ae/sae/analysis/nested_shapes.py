"""Propagate source-backed shape facts across validated call bindings."""
from __future__ import annotations

from collections import defaultdict, deque

from ..ir.canonical import stable_id
from ..ir.models import Graph

RULE = 'nested-static-shape-binding/v1'
BINDINGS = frozenset({
    'STATIC_FORWARD_RETURN',
    'STATIC_CALL_RETURN',
    'STATIC_CALL_ARGUMENT_BINDING',
    'STATIC_COLLECTION_ELEMENT',
    'STATIC_COLLECTION_ELEMENT_BINDING',
})


def propagate_nested_shapes(graph: Graph) -> int:
    """Add conditional shape facts along supported value bindings."""
    nodes = {n['id']: n for n in graph.nodes}
    incoming: dict[str, set[str]] = defaultdict(set)
    outgoing: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        if edge['relation'] in BINDINGS:
            incoming[edge['target']].add(edge['source'])
            outgoing[edge['source']].add(edge['target'])

    seeds: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for node in graph.nodes:
        if node['kind'] == 'symbolic_shape_fact':
            seeds[node['attributes']['value_id']].add(tuple(node['attributes']['dimensions']))
    conflicting = {name for name, alternatives in seeds.items() if len(alternatives) > 1}
    shapes = {name: next(iter(alternatives)) for name, alternatives in seeds.items()
              if len(alternatives) == 1}
    added: dict[str, tuple[str, ...]] = {}
    # Re-evaluate descendants when any source obtains a shape fact.
    queue = deque(sorted(shapes))
    queued = set(queue)
    while queue:
        current = queue.popleft()
        queued.discard(current)
        if current in conflicting:
            continue
        for target in sorted(outgoing.get(current, ())):
            sources = incoming[target]
            if (target in conflicting or not sources or
                    any(s not in shapes or s in conflicting for s in sources)):
                continue
            alternatives = {shapes[s] for s in sources}
            if len(alternatives) != 1:
                conflicting.add(target)
                added.pop(target, None)
                continue
            dims = next(iter(alternatives))
            if target in shapes:
                if shapes[target] != dims:
                    conflicting.add(target)
                    added.pop(target, None)
                continue
            shapes[target] = dims
            added[target] = dims
            if target not in queued:
                queue.append(target)
                queued.add(target)

    for target in sorted(added):
        if target in conflicting or target not in nodes:
            continue
        dimensions = added[target]
        source_refs = nodes[target]['source_refs']
        ident = stable_id('ae-nested-shape/v1', target, dimensions)
        graph.add_node({
            'id': ident, 'kind': 'symbolic_shape_fact',
            'attributes': {'value_id': target, 'dimensions': list(dimensions),
                           'scope': 'conditional_type_level',
                           'reason': 'source-backed-nested-binding'},
            'source_refs': source_refs, 'evidence_class': 'derived',
            'analysis_rule': RULE, 'resolution': 'conditional',
        })
        graph.add_edge({
            'id': stable_id('ae-nested-shape-edge/v1', target, ident),
            'source': target, 'target': ident, 'relation': 'HAS_SYMBOLIC_SHAPE',
            'source_refs': source_refs, 'evidence_class': 'derived',
            'analysis_rule': RULE,
        })
    return sum(target not in conflicting for target in added)
