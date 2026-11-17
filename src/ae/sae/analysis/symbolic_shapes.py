"""Small, conservative symbolic last-dimension propagation.

A *template* describes a conditional type-level signature. It is not a
runtime-observed tensor, rank, or proof of layer-input compatibility. Only
supported, source-backed calls can seed template facts; only semantics known
to preserve shape may transfer them to another value.
"""
from __future__ import annotations

from collections import defaultdict, deque

from ..ir.canonical import stable_id
from ..ir.models import Graph
from .pytorch_static import SHAPE_PRESERVING

RULE = 'conditional-symbolic-shape-propagation/v1'
PROPAGATES = {'STATIC_FORWARD_RETURN', 'STATIC_CALL_RETURN', 'STATIC_CALL_ARGUMENT_BINDING',
              'STATIC_COLLECTION_ELEMENT', 'STATIC_COLLECTION_ELEMENT_BINDING'}


def analyze_symbolic_shapes(graph: Graph, model_index: dict) -> None:
    nodes = {n['id']: n for n in graph.nodes}
    by_forward: dict[str, list[dict]] = defaultdict(list)
    for node in graph.nodes:
        if node['kind'] in ('forward_input', 'forward_output', 'forward_choice', 'forward_augassign'):
            by_forward[node['attributes']['forward_id']].append(node)
    incoming: dict[str, list[dict]] = defaultdict(list)
    successors: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        incoming[edge['target']].append(edge)
        if edge['relation'] in PROPAGATES:
            successors[edge['source']].add(edge['target'])

    shapes: dict[str, tuple[str, ...]] = {}
    causes: dict[str, str] = {}
    shape_preserving: set[str] = set()
    linear_expected: dict[str, tuple[str, str]] = {}
    queue = deque()
    reported: set[tuple[str, str]] = set()

    def add(value: str, dims: tuple[str, ...], reason: str):
        if not dims or any(not d or d == '?' for d in dims) or value not in nodes:
            return
        if value not in shapes:
            shapes[value] = dims
            causes[value] = reason
            queue.append(value)
        elif shapes[value] != dims:
            key = (value, 'conflicting-shape')
            if key not in reported:
                graph.unresolved.append({'kind': 'symbolic_shape_incomplete', 'path': nodes[value]['source_refs'][0]['path'],
                                         'reason': f'Inconsistent conditional shape templates at value {value}',
                                         'site_id': value, 'source_refs': nodes[value]['source_refs']})
                reported.add(key)
            # Do not expose a contradictory unique shape.
            shapes.pop(value, None)
            causes.pop(value, None)

    for module in model_index.get('modules', []):
        fid = module.get('forward_method_id')
        if not fid:
            continue
        components: dict[str, list[dict]] = defaultdict(list)
        for c in module['components']:
            components[c['owner_expression']].append(c)
        for call in module['forward_calls']:
            matches = components.get(call['callee_expression'], [])
            if len(matches) != 1 or len(matches[0].get('layer_spec_ids', [])) != 1:
                continue
            spec = nodes.get(matches[0]['layer_spec_ids'][0])
            if not spec:
                continue
            declared = spec['attributes'].get('declared_type', '').split('.')[-1]
            template = spec['attributes'].get('shape_template', {})
            site = call['call_site_id']
            if declared == 'Linear':
                expected = template.get('input', [])
                result = template.get('output', [])
                if (len(result) == 2 and len(expected) == 2 and result[0] == '...' and
                        result[-1] != '?' and expected[-1] != '?'):
                    add(site, tuple(result), 'linear-output-template')
                    linear_expected[site] = (str(expected[-1]), fid)
            elif declared == 'Embedding':
                result = template.get('output', [])
                if len(result) == 2 and result[-1] != '?':
                    add(site, tuple(result), 'embedding-output-template')
            elif declared.startswith('Conv') and not declared.startswith('ConvTranspose'):
                result = template.get('output', [])
                if result and all(str(x) != '?' for x in result):
                    # Channel fact only; spatial axes are formal placeholders.
                    add(site, tuple(map(str, result)), 'conv-output-channel-template')
            elif declared in SHAPE_PRESERVING:
                shape_preserving.add(site)

    for site in shape_preserving:
        for edge in incoming.get(site, []):
            if edge['relation'] == 'STATIC_VALUE_DEPENDENCY':
                successors[edge['source']].add(site)

    # A branch join / augmented assignment may preserve shape only when both
    # explicitly participating values have exactly the same known template.
    merge_nodes = {n['id'] for group in by_forward.values() for n in group
                   if n['kind'] in ('forward_choice', 'forward_augassign')}
    for site in merge_nodes:
        for edge in incoming.get(site, []):
            if edge['relation'] in ('STATIC_VALUE_DEPENDENCY','STATIC_CONDITIONAL_DEPENDENCY'):
                successors[edge['source']].add(site)

    while queue:
        src = queue.popleft()
        if src not in shapes:
            continue
        for dst in sorted(successors.get(src, ())):
            if dst in linear_expected:  # Linear is not shape-preserving.
                continue
            if dst in merge_nodes:
                operands = [e['source'] for e in incoming.get(dst, [])
                            if e['relation'] in ('STATIC_VALUE_DEPENDENCY','STATIC_CONDITIONAL_DEPENDENCY')]
                if len(operands) < 2 or any(v not in shapes for v in operands):
                    continue
                if len({shapes[v] for v in operands}) != 1:
                    continue
                add(dst, shapes[src], 'equal-shape-branch-or-augassign')
            elif dst in shape_preserving:
                operands = [e['source'] for e in incoming.get(dst, [])
                            if e['relation'] == 'STATIC_VALUE_DEPENDENCY']
                if len(operands) == 1 and operands[0] in shapes:
                    add(dst, shapes[operands[0]], 'shape-preserving-layer')
            else:
                # Shape propagates through supported identity/argument/return
                # and *indexed element* bindings, never into the collection
                # aggregate (tuple/list itself is not a tensor).
                if any(e['source'] == src and e['relation'] in PROPAGATES
                       for e in incoming.get(dst, [])):
                    add(dst, shapes[src], 'supported-parameter-or-return-binding')

    for site, (expected, fid) in sorted(linear_expected.items()):
        operands = [e['source'] for e in incoming.get(site, [])
                    if e['relation'] == 'STATIC_VALUE_DEPENDENCY']
        if len(operands) != 1 or operands[0] not in shapes:
            continue
        prior = shapes[operands[0]][-1]
        # A definite numeric conflict is a model error *candidate*, not proof
        # of a runtime exception (the call may not execute).
        if prior.isdecimal() and expected.isdecimal() and int(prior) != int(expected):
            graph.unresolved.append({'kind': 'symbolic_shape_conflict', 'path': nodes[site]['source_refs'][0]['path'],
                                     'function_id': fid, 'site_id': site,
                                     'reason': f'Linear input last dimension requires {expected}, preceding template supplies {prior}',
                                     'source_refs': nodes[site]['source_refs']})

    for value, dimensions in sorted(shapes.items()):
        origin = nodes[value]
        fact_id = stable_id('ae-symbolic-shape/v1', value, dimensions)
        graph.add_node({'id':fact_id, 'kind':'symbolic_shape_fact',
                        'attributes': {'value_id':value, 'dimensions':list(dimensions),
                                       'scope':'conditional_type_level', 'reason':causes[value]},
                        'source_refs':origin['source_refs'], 'evidence_class':'derived',
                        'analysis_rule':RULE, 'resolution':'conditional'})
        graph.add_edge({'id':stable_id('ae-symbolic-shape-edge/v1', value, fact_id),
                        'source':value, 'target':fact_id, 'relation':'HAS_SYMBOLIC_SHAPE',
                        'source_refs':origin['source_refs'], 'evidence_class':'derived',
                        'analysis_rule':RULE})
