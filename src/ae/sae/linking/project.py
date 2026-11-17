"""Conservative, non-executing cross-file symbol and call linking.

The linker only creates edges for locally visible unique declarations. It never
imports analyzed modules or guesses dynamic Python dispatch semantics.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable

from ..analysis.scopes import parent_links, alias_is_shadowed

from ..ir.canonical import stable_id
from ..ir.models import Graph
from ..parsing.source_spans import SourceMap


@dataclass(frozen=True)
class Unit:
    source: SourceMap
    tree: ast.Module
    source_id: str

    @property
    def path(self) -> str:
        return self.source.path

    @property
    def module(self) -> str:
        path = PurePosixPath(self.path)
        pieces = list(path.with_suffix('').parts)
        if pieces[-1] == '__init__':
            pieces.pop()
        return '.'.join(pieces)


def syntax_ids(unit: Unit) -> dict[ast.AST, str]:
    result: dict[ast.AST, str] = {}
    def visit(node: ast.AST, key: str) -> None:
        result[node] = stable_id(unit.path, 'syntax', key, type(node).__name__)
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                visit(value, f'{key}/{field}')
            elif isinstance(value, list):
                for i, child in enumerate(value):
                    if isinstance(child, ast.AST):
                        visit(child, f'{key}/{field}[{i}]')
    visit(unit.tree, 'root')
    return result


def span(unit: Unit, node: ast.AST) -> dict:
    if hasattr(node, 'lineno'):
        return {**unit.source.ast_span(node), 'precision': 'exact'}
    return {'path': unit.path, 'start_byte': 0, 'end_byte': len(unit.source.raw),
            'start_line': 1, 'start_column': 0,
            'end_line': len(unit.source.lines) or 1, 'end_column': 0, 'precision': 'file'}


def derived_edge(graph: Graph, unit: Unit, node: ast.AST, source: str,
                 target: str, relation: str, rule: str, *, label: str = '') -> None:
    # Source-scoped IDs do not depend on whole-graph insertion order.
    ident = stable_id('project-edge', unit.path, source, target, relation, rule, label,
                      getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
    if ident in graph._edge_ids:
        return
    graph.add_edge({'id': ident, 'source': source, 'target': target, 'relation': relation,
                    'evidence_class': 'derived', 'analysis_rule': rule,
                    'source_refs': [span(unit, node)]})


def unresolved(graph: Graph, unit: Unit, node: ast.AST, kind: str, reason: str,
               *, candidates: Iterable[str] = ()) -> None:
    graph.unresolved.append({'kind': kind, 'path': unit.path, 'reason': reason,
                             'candidate_ids': sorted(set(candidates)), 'source_refs': [span(unit, node)]})


@dataclass(frozen=True)
class Definition:
    ident: str
    node: ast.AST
    kind: str
    unit: Unit


class ProjectLinker:
    def __init__(self, graph: Graph, units: list[Unit]):
        self.graph = graph
        self.units = sorted(units, key=lambda u: u.path)
        self.ids = {u.path: syntax_ids(u) for u in units}
        self.modules: dict[str, list[Unit]] = {}
        self.definitions: dict[str, dict[str, list[Definition]]] = {}
        self.imports: dict[str, dict[str, list[tuple[str, str | None, ast.AST]]]] = {}

    def _definition(self, unit: Unit, node: ast.AST, kind: str) -> Definition:
        ident = stable_id(unit.path, 'semantic', kind, self.ids[unit.path][node], '')
        return Definition(ident, node, kind, unit)

    def _absolute_import(self, unit: Unit, node: ast.ImportFrom) -> str:
        if not node.level:
            return node.module or ''
        parts = unit.module.split('.') if unit.module else []
        if not unit.path.endswith('/__init__.py') and unit.path != '__init__.py':
            parts = parts[:-1]
        if node.level > len(parts) + 1:
            return ''
        prefix = parts[:len(parts) - (node.level - 1)]
        return '.'.join([*prefix, *([node.module] if node.module else [])])

    def collect(self) -> None:
        for unit in self.units:
            self.modules.setdefault(unit.module, []).append(unit)
            defs: dict[str, list[Definition]] = {}
            imps: dict[str, list[tuple[str, str | None, ast.AST]]] = {}
            for stmt in unit.tree.body:
                if isinstance(stmt, ast.ClassDef):
                    defs.setdefault(stmt.name, []).append(self._definition(unit, stmt, 'class_definition'))
                elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defs.setdefault(stmt.name, []).append(self._definition(unit, stmt, 'function_definition'))
                elif isinstance(stmt, ast.ImportFrom):
                    module = self._absolute_import(unit, stmt)
                    for alias in stmt.names:
                        if alias.name != '*':
                            imps.setdefault(alias.asname or alias.name, []).append((module, alias.name, stmt))
                        else:
                            unresolved(self.graph, unit, stmt, 'star_import', 'Wildcard imports are not statically expanded')
                elif isinstance(stmt, ast.Import):
                    for alias in stmt.names:
                        # 'import x.y' binds x; 'import x.y as z' binds z.
                        imps.setdefault(alias.asname or alias.name.split('.')[0], []).append((alias.name, None, stmt))
            self.definitions[unit.path] = defs
            self.imports[unit.path] = imps

    def _export(self, module: str, name: str) -> list[Definition]:
        result: list[Definition] = []
        for unit in self.modules.get(module, []):
            result.extend(self.definitions[unit.path].get(name, []))
        return result

    def _shadows(self, function: ast.FunctionDef | ast.AsyncFunctionDef, name: str,
                 *, allow_parameter: bool = False) -> bool:
        # Python scopes are determined by bindings anywhere in the function,
        # not merely statements earlier than the call.
        params = {arg.arg for arg in [*function.args.posonlyargs, *function.args.args,
                                      *function.args.kwonlyargs, *([function.args.vararg] if function.args.vararg else []),
                                      *([function.args.kwarg] if function.args.kwarg else [])]}
        if name in params and not allow_parameter:
            return True
        # Do not traverse nested functions or classes when identifying bindings.
        def within_scope(root: ast.AST):
            for child in ast.iter_child_nodes(root):
                yield child
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    yield from within_scope(child)
        for node in within_scope(function):
            if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, (ast.Store, ast.Del)):
                return True
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                aliases = node.names
                if any((a.asname or (a.name.split('.')[0] if isinstance(node, ast.Import) else a.name)) == name for a in aliases):
                    return True
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not function and node.name == name:
                return True
        return False

    def _top_level_shadow(self, unit: Unit, name: str) -> bool:
        for stmt in unit.tree.body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target] if hasattr(stmt, 'target') else stmt.targets
                if any(isinstance(n, ast.Name) and n.id == name for t in targets for n in ast.walk(t)):
                    return True
        return False

    def _resolve_name(self, unit: Unit, name: str) -> list[Definition]:
        if self._top_level_shadow(unit, name):
            return []
        direct = self.definitions[unit.path].get(name, [])
        imports = self.imports[unit.path].get(name, [])
        if direct and imports:
            return []  # assignment order/import rebinding not proven
        if direct:
            return direct
        result = []
        for module, symbol, _ in imports:
            if symbol is not None:
                result.extend(self._export(module, symbol))
        return result

    def _visit_unit(self, unit: Unit) -> None:
        ids = self.ids[unit.path]
        # Explicit import-to-definition references are resolved only within the inventory.
        for local, bindings in self.imports[unit.path].items():
            if self._top_level_shadow(unit, local) or len(bindings) != 1 or local in self.definitions[unit.path]:
                continue
            module, symbol, import_node = bindings[0]
            target = self.modules.get(module, []) if symbol is None else self._export(module, symbol)
            if len(target) == 1:
                dst = target[0].source_id if symbol is None else target[0].ident
                derived_edge(self.graph, unit, import_node, ids[import_node], dst,
                             'IMPORTS_LOCAL_DEFINITION' if symbol is not None else 'IMPORTS_LOCAL_MODULE',
                             'local-project-import-resolution/v1', label=local)
            elif len(target) > 1:
                unresolved(self.graph, unit, import_node, 'ambiguous_local_import',
                           f'Multiple declarations for imported name {local}',
                           candidates=(t.source_id if symbol is None else t.ident for t in target))
            elif module in self.modules:
                unresolved(self.graph, unit, import_node, 'unresolved_local_import',
                           f'Name {symbol} not defined unambiguously in local module {module}')

        visitor = self
        class Walker(ast.NodeVisitor):
            def __init__(self) -> None:
                self.class_stack: list[ast.ClassDef] = []
                self.function_stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self.class_stack.append(node)
                self.generic_visit(node)
                self.class_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.function_stack.append(node)
                self.generic_visit(node)
                self.function_stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node: ast.Call) -> None:
                target: list[Definition] = []
                name = None
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                    if not any(visitor._shadows(f, name) for f in self.function_stack):
                        target = visitor._resolve_name(unit, name)
                elif (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                      and node.func.value.id == 'self' and self.class_stack and self.function_stack):
                    name = node.func.attr
                    klass = self.class_stack[-1]
                    assigned_attrs = {n.targets[0].attr for n in ast.walk(klass)
                            if isinstance(n, ast.Assign) and len(n.targets) == 1
                            and isinstance(n.targets[0], ast.Attribute)
                            and isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == 'self'}
                    if (name not in assigned_attrs
                        and self.function_stack[-1].args.args and self.function_stack[-1].args.args[0].arg == 'self'
                        and not visitor._shadows(self.function_stack[-1], 'self', allow_parameter=True)):
                        methods = [n for n in klass.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
                        target = [visitor._definition(unit, m, 'method_definition') for m in methods]
                elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    module_alias = node.func.value.id
                    if not any(visitor._shadows(f, module_alias) for f in self.function_stack):
                        imports = visitor.imports[unit.path].get(module_alias, [])
                        if len(imports) == 1 and imports[0][1] is None:
                            target = visitor._export(imports[0][0], node.func.attr)
                if len(target) == 1:
                    callee = target[0]
                    site = stable_id(unit.path, 'semantic', 'call_site', ids[node], '')
                    derived_edge(visitor.graph, unit, node, site, callee.ident,
                                 'CALLS_LOCAL_DEFINITION', 'unique-local-call-resolution/v1')
                elif len(target) > 1:
                    unresolved(visitor.graph, unit, node, 'ambiguous_local_call',
                               f'Multiple candidates for {name or ast.unparse(node.func)}',
                               candidates=(d.ident for d in target))
                self.generic_visit(node)

        Walker().visit(unit.tree)

    def _custom_module_sites(self) -> None:
        """Prove direct/transitive local nn.Module subclasses and construction sites."""
        classes = [d for unit in self.units for values in self.definitions[unit.path].values()
                   for d in values if d.kind == 'class_definition']
        proven: set[str] = set()
        def direct_nn_module(definition: Definition) -> bool:
            imports = self.imports[definition.unit.path]
            for base in definition.node.bases:
                if not isinstance(base, ast.Attribute) and not isinstance(base, ast.Name):
                    continue
                expr = ast.unparse(base)
                if expr == 'torch.nn.Module' and any(a[0] == 'torch' and a[1] is None for a in imports.get('torch', [])):
                    return True
                if expr == 'nn.Module' and any(a[0] == 'torch' and a[1] == 'nn' for a in imports.get('nn', [])):
                    return True
                if isinstance(base, ast.Name):
                    if any(a[0] == 'torch.nn' and a[1] == 'Module' for a in imports.get(base.id, [])):
                        return True
                if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                    if base.attr == 'Module' and any(a[0] == 'torch.nn' and a[1] is None
                            for a in imports.get(base.value.id, [])):
                        return True
            return False
        for d in classes:
            if direct_nn_module(d):
                proven.add(d.ident)
        changed = True
        while changed:
            changed = False
            for d in classes:
                if d.ident in proven:
                    continue
                for base in d.node.bases:
                    if isinstance(base, ast.Name):
                        parents = self._resolve_name(d.unit, base.id)
                        if len(parents) == 1 and parents[0].ident in proven:
                            derived_edge(self.graph, d.unit, base, d.ident, parents[0].ident,
                                         'INHERITS_LOCAL_DEFINITION', 'unique-local-inheritance/v1')
                            proven.add(d.ident)
                            changed = True
        for d in classes:
            if d.ident not in proven:
                continue
            graph_id = stable_id(d.unit.path, 'pytorch-module-definition', d.ident)
            self.graph.add_node({'id': graph_id, 'kind': 'pytorch_module_definition',
                                 'attributes': {'name': d.node.name, 'module': d.unit.module},
                                 'source_refs': [span(d.unit, d.node)], 'evidence_class': 'derived',
                                 'resolution': 'conditional', 'analysis_rule': 'static-nn-module-inheritance/v1'})
            derived_edge(self.graph, d.unit, d.node, graph_id, d.ident,
                         'DESCRIBES_CUSTOM_MODULE', 'static-nn-module-inheritance/v1')

        for unit in self.units:
            ids = self.ids[unit.path]
            parents = parent_links(unit.tree)
            sites: dict[tuple[str, str], list[str]] = {}
            for klass in (stmt for stmt in unit.tree.body if isinstance(stmt, ast.ClassDef)):
                for node in ast.walk(klass):
                    if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                            and isinstance(node.targets[0], ast.Attribute)
                            and isinstance(node.targets[0].value, ast.Name)
                            and node.targets[0].value.id == 'self' and isinstance(node.value, ast.Call)
                            and isinstance(node.value.func, ast.Name)):
                        continue
                    if alias_is_shadowed(node.value, node.value.func.id, parents):
                        continue
                    target = self._resolve_name(unit, node.value.func.id)
                    if len(target) != 1 or target[0].ident not in proven:
                        continue
                    definition = target[0]
                    constructed = stable_id(unit.path, 'custom-module-construction', ids[node.value])
                    self.graph.add_node({'id': constructed, 'kind': 'module_construction_site',
                                         'attributes': {'declared_type': f'{definition.unit.module}.{definition.node.name}',
                                                        'owner_expression': ast.unparse(node.targets[0]),
                                                        'constructor_arguments': [ast.unparse(a) for a in node.value.args]},
                                         'source_refs': [span(unit, node.value)],
                                         'evidence_class': 'derived', 'resolution': 'conditional',
                                         'analysis_rule': 'local-module-construction/v1'})
                    derived_edge(self.graph, unit, node.value, constructed, definition.ident,
                                 'CONSTRUCTS_LOCAL_MODULE', 'local-module-construction/v1')
                    sites.setdefault((klass.name, node.targets[0].attr), []).append(constructed)
                for node in ast.walk(klass):
                    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                        continue
                    if not isinstance(node.func.value, ast.Name) or node.func.value.id != 'self':
                        continue
                    options = sites.get((klass.name, node.func.attr), [])
                    container = parents.get(node)
                    while container is not None and not isinstance(container, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        container = parents.get(container)
                    if not isinstance(container, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if not container.args.args or container.args.args[0].arg != 'self':
                        continue
                    if self._shadows(container, 'self', allow_parameter=True):
                        continue
                    if len(options) == 1:
                        site = stable_id(unit.path, 'semantic', 'call_site', ids[node], '')
                        derived_edge(self.graph, unit, node, site, options[0],
                                     'USES_MODULE_REFERENCE', 'unique-local-module-reference/v1')
                        self.graph.unresolved[:] = [record for record in self.graph.unresolved
                            if not (record['kind'] == 'module_attribute_resolution'
                                    and record.get('site_id') == site)]

    def run(self) -> None:
        self.collect()
        for unit in self.units:
            self._visit_unit(unit)
        self._custom_module_sites()


def link_project(graph: Graph, units: list[Unit]) -> None:
    ProjectLinker(graph, units).run()
