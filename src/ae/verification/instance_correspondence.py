"""Match factory block identities to FX events with independent module scopes."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .fx_correspondence import compare_source_sites, fx_operation


def _block_catalog(index: dict, ir: dict) -> tuple[dict, list[dict], list[str]]:
    factory = index.get('factory_specialization')
    if not factory or factory.get('resolution') != 'conditional':
        return {}, [], ['NO_FACTORY_BLUEPRINT']
    nodes = {node['id']: node for node in ir['nodes']}
    catalog: dict[str, dict] = {}
    order = []
    errors = []
    for stage in factory.get('stages', []):
        name, block_type = stage['name'], stage['block_type']
        for offset, ident in enumerate(stage['block_instance_ids']):
            path = f'{name}.{offset}'
            if path in catalog or ident not in nodes or nodes[ident]['kind'] != 'factory_block_instance':
                errors.append('INVALID_BLOCK_BLUEPRINT')
                continue
            attrs = nodes[ident]['attributes']
            if attrs.get('stage') != name or attrs.get('position') != offset or attrs.get('block_type') != block_type:
                errors.append('INVALID_BLOCK_BLUEPRINT')
                continue
            catalog[path] = {'instance_id': ident, 'module_path': path, 'block_type': block_type,
                             'stage': name, 'position': offset, 'source_refs': nodes[ident]['source_refs']}
            order.append(path)
    return catalog, order, sorted(set(errors))


def _block_scope(event: dict, catalog: dict) -> tuple[str | None, str | None]:
    stack = event.get('module_stack') or []
    if not isinstance(stack, list):
        return None, 'MISSING_FX_MODULE_STACK'
    matching = [item for item in stack if isinstance(item, dict) and item.get('path') in catalog]
    if len(matching) != 1:
        return None, 'AMBIGUOUS_BLOCK_SCOPE' if matching else None
    scope = matching[0]
    expected = catalog[scope['path']]['block_type']
    if scope.get('type_name') != expected:
        return None, 'BLOCK_TYPE_MISMATCH'
    return scope['path'], None


def _site_matches_fx(ir_nodes: dict[str, dict], ident: str, event: dict) -> bool:
    ref = ir_nodes.get(ident, {}).get('source_refs', [{}])[0]
    source = event.get('source') or {}
    if ref.get('path') != source.get('path') or ref.get('start_line') != source.get('line'):
        return False
    if (source.get('column') is not None and ref.get('start_column') is not None
            and source['column'] != ref['start_column']):
        return False
    if (source.get('end_column') is not None and ref.get('end_column') is not None
            and source['end_column'] != ref['end_column']):
        return False
    return True


def _shape_checks(nodes: dict[str, dict], ir: dict, matched: dict, events: dict) -> list[dict]:
    shapes: dict[str, set[tuple]] = defaultdict(set)
    for node in ir['nodes']:
        if node['kind'] == 'symbolic_shape_fact':
            shapes[node['attributes']['value_id']].add(tuple(node['attributes']['dimensions']))
    results = []
    for fx_name, (block_path, ident) in sorted(matched.items()):
        if len(shapes[ident]) != 1:
            continue
        event = events[fx_name]
        shape = event.get('runtime_shape')
        if shape is None:
            continue
        template = next(iter(shapes[ident]))
        if template and template[0] == '...':
            suffix = template[1:]
            if not all(str(d).isdigit() for d in suffix):
                continue
            correct = (len(shape) >= len(suffix) and [int(d) for d in suffix] == shape[-len(suffix):])
        elif all(str(d).isdigit() for d in template):
            correct = [int(d) for d in template] == shape
        else:
            continue
        results.append({'fx_event': fx_name, 'block_path': block_path, 'source_site_id': ident,
                        'static_dimensions': list(template), 'fx_dimensions': shape, 'equal': correct})
    return results


def compare_factory_instances(ir: dict, index: dict, events: list[dict], *,
                              runtime_modules: dict[str, str] | None = None) -> dict[str, Any]:
    """Verify source-site matches per factory block without asserting full tensor equivalence."""
    catalog, expected_paths, errors = _block_catalog(index, ir)
    if not catalog:
        return {'scope': 'factory_blueprint_to_fx_block_events', 'status': 'unsupported',
                'issues': errors or ['NO_VALID_BLOCK_INSTANCES'], 'full_equivalence': 'not_verified'}
    nodes = {n['id']: n for n in ir['nodes']}
    events_by_name = {e['name']: e for e in events}
    if len(events_by_name) != len(events):
        raise ValueError('Duplicate FX event names')

    source_sites = compare_source_sites(ir, events)
    candidate_sites: dict[str, set[str]] = defaultdict(set)
    for site in source_sites['matched_site_details']:
        if len(site['ae_sites']) != 1:
            continue
        for name in site['fx_names']:
            candidate_sites[name].add(site['ae_sites'][0])

    in_scope: list[dict] = []
    unmatched, ambiguous = [], []
    matched: dict[str, tuple[str, str]] = {}
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    event_order = []
    for event in events:
        if fx_operation(event) is None:
            continue
        block, reason = _block_scope(event, catalog)
        if reason:
            errors.append(reason)
            continue
        if block is None:
            continue
        event_order.append(block)
        in_scope.append(event)
        matches = [ident for ident in candidate_sites.get(event['name'], set())
                   if _site_matches_fx(nodes, ident, event)]
        if len(matches) == 1:
            grouped[(block, matches[0])].append(event['name'])
        elif len(matches) > 1:
            ambiguous.append({'fx_event': event['name'], 'block_path': block,
                              'candidate_ids': sorted(matches)})
        else:
            unmatched.append({'fx_event': event['name'], 'block_path': block,
                              'source': event.get('source')})
    for (block, ident), names in sorted(grouped.items()):
        if len(names) != 1:
            for name in names:
                ambiguous.append({'fx_event': name, 'block_path': block,
                                  'reason': 'DUPLICATE_OPERATION_INSTANCE_SITE', 'candidate_ids': [ident]})
        else:
            matched[names[0]] = (block, ident)
    seen = []
    for block in event_order:
        if not seen or seen[-1] != block:
            seen.append(block)
    missing_blocks = sorted(set(expected_paths) - set(seen))
    unexpected_blocks = sorted(set(seen) - set(expected_paths))
    if seen != expected_paths:
        errors.append('BLOCK_EXECUTION_ORDER_OR_COVERAGE_MISMATCH')

    runtime_checks = []
    if runtime_modules is not None:
        runtime_checks = [
            {'module_path': path, 'expected_type': info['block_type'],
             'runtime_type': runtime_modules.get(path),
             'matched': runtime_modules.get(path) == info['block_type']}
            for path, info in sorted(catalog.items())]
        if not all(x['matched'] for x in runtime_checks):
            errors.append('MODULE_IDENTITY_MISMATCH')

    # Direct FX dependencies are independently observed; SAE edges remain source-level.
    fx_edges = []
    for event in in_scope:
        if event['name'] not in matched:
            continue
        for predecessor in event.get('input_nodes', []):
            if predecessor in matched:
                previous_block, source_id = matched[predecessor]
                target_block, target_id = matched[event['name']]
                fx_edges.append({'source_fx': predecessor, 'target_fx': event['name'],
                                 'source_instance_id': catalog[previous_block]['instance_id'],
                                 'target_instance_id': catalog[target_block]['instance_id'],
                                 'source_site_id': source_id, 'target_site_id': target_id})
    # Compare direct IR dependencies only between uniquely matched operations in one block.
    fx_intra = {(item['source_fx'], item['target_fx']) for item in fx_edges
                if item['source_instance_id'] == item['target_instance_id']}
    by_instance_site = {(block, ident): name for name, (block, ident) in matched.items()}
    ae_intra = set()
    for edge in ir['edges']:
        if edge['relation'] != 'STATIC_VALUE_DEPENDENCY':
            continue
        for block in catalog:
            source_name = by_instance_site.get((block, edge['source']))
            target_name = by_instance_site.get((block, edge['target']))
            if source_name and target_name:
                ae_intra.add((source_name, target_name))
    correct_edges = fx_intra & ae_intra
    direct_edges = {
        'scope': 'direct_intrablock_dependencies_between_unique_matched_events',
        'ae_edges': len(ae_intra), 'fx_edges': len(fx_intra),
        'true_positive': len(correct_edges),
        'false_positive': len(ae_intra - fx_intra),
        'false_negative': len(fx_intra - ae_intra),
        'precision': len(correct_edges) / len(ae_intra) if ae_intra else None,
        'recall': len(correct_edges) / len(fx_intra) if fx_intra else None,
        'cross_block_fx_edges_excluded': len(fx_edges) - len(fx_intra),
    }
    if direct_edges['false_positive'] or direct_edges['false_negative']:
        errors.append('DIRECT_INTRA_BLOCK_EDGE_MISMATCH')
    shape_checks = _shape_checks(nodes, ir, matched, events_by_name)
    results = [
        {'fx_event': name, 'instance_path': path, 'instance_id': catalog[path]['instance_id'],
         'source_site_id': ident, 'source_refs': nodes[ident]['source_refs']}
        for name, (path, ident) in sorted(matched.items())]
    issues = sorted(set(errors))
    if ambiguous:
        issues.append('AMBIGUOUS_INSTANCE_EVENT')
    if unmatched:
        issues.append('UNMATCHED_FX_BLOCK_OPERATION')
    if missing_blocks:
        issues.append('MISSING_BLOCK_OCCURRENCES')
    status = 'verified_in_scope' if not issues else 'partial'
    return {
        'scope': 'factory_blueprint_to_fx_block_events', 'status': status,
        'factory': index['factory_specialization']['factory_definition_id'],
        'block_type': index['factory_specialization']['block_type'],
        'expected_block_instances': len(expected_paths), 'fx_observed_block_instances': len(seen),
        'block_sequence_matches': seen == expected_paths,
        'expected_instance_paths': expected_paths, 'observed_instance_paths': seen,
        'missing_instance_paths': missing_blocks, 'unexpected_instance_paths': unexpected_blocks,
        'runtime_module_type_checks': runtime_checks,
        'observed_in_scope_fx_operations': len(in_scope), 'matched_instance_operations': len(results),
        'instance_operation_recall': len(results) / len(in_scope) if in_scope else None,
        'unmatched_operations': unmatched, 'ambiguous_operations': ambiguous,
        'matched_events': results, 'fx_edges_with_both_endpoints_matched': fx_edges,
        'fx_comparable_direct_edges': len(fx_edges),
        'direct_edge_correspondence': direct_edges,
        'ae_to_fx_edge_equivalence': 'not_verified',
        'shape_checks': shape_checks,
        'shape_mismatches': sum(not row['equal'] for row in shape_checks),
        'issues': sorted(set(issues)),
        'full_equivalence': 'not_verified',
    }
