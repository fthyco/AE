"""Conservative non-executing, source-grounded forward value dependencies.

This records syntactic data dependencies. Alternatives from supported if/else
branches are represented by conditional joins, never a selected runtime path.
Augmented assignment records result dependencies but does not claim alias or
mutation side-effect completeness. Unhandled syntax stays explicitly unresolved.
"""
from __future__ import annotations

import ast

from ..ir.canonical import stable_id
from ..ir.models import Graph
from ..linking.project import Unit, derived_edge, span, syntax_ids

RULE = 'straight-line-forward-value/v1'
EXTENDED_RULE = 'conditional-forward-values/v2'
COLLECTION_RULE = 'static-collection-position/v1'


class ForwardValues:
    def __init__(self, graph: Graph, unit: Unit, fn: ast.FunctionDef | ast.AsyncFunctionDef,
                 *, definition_kind: str = 'method_definition'):
        self.graph, self.unit, self.fn = graph, unit, fn
        self.ids = syntax_ids(unit)
        self.forward_id = stable_id(unit.path, 'semantic', definition_kind, self.ids[fn], '')
        self.env: dict[str, str] = {}
        self.collections: dict[str, tuple[str, ...]] = {}
        self.collection_ast_ids: dict[ast.AST, str] = {}
        self.terminated = False

    def node(self, kind: str, astnode: ast.AST, label: str, *, conditional: bool = False) -> str:
        ident = stable_id(self.unit.path, 'forward-value', kind, self.ids[astnode], label)
        self.graph.add_node({'id': ident, 'kind': kind,
                             'attributes': {'name': label, 'forward_id': self.forward_id},
                             'source_refs': [span(self.unit, astnode)],
                             'evidence_class': 'derived', 'analysis_rule': EXTENDED_RULE if conditional else RULE,
                             'resolution': 'conditional' if conditional else 'resolved'})
        derived_edge(self.graph, self.unit, astnode, self.forward_id, ident,
                     'HAS_FORWARD_VALUE', EXTENDED_RULE if conditional else RULE)
        return ident

    def link(self, source: str | None, target: str, at: ast.AST,
             relation: str = 'STATIC_VALUE_DEPENDENCY', *, label: str = '') -> None:
        if source is not None:
            derived_edge(self.graph, self.unit, at, source, target, relation,
                         EXTENDED_RULE if relation in ('STATIC_CONDITIONAL_DEPENDENCY','STATIC_GUARD_DEPENDENCY','STATIC_GUARDED_CALL') else RULE, label=label)

    def problem(self, astnode: ast.AST, reason: str, *, site_id: str | None = None) -> None:
        entry = {'kind': 'forward_flow_incomplete', 'path': self.unit.path,
                 'function_id': self.forward_id, 'reason': reason,
                 'source_refs': [span(self.unit, astnode)]}
        if site_id:
            entry['site_id'] = site_id
        self.graph.unresolved.append(entry)

    def collection(self, node: ast.Tuple | ast.List) -> str:
        """Materialize explicit, fixed-position syntax without evaluating elements."""
        if node in self.collection_ast_ids:
            return self.collection_ast_ids[node]
        kind = type(node).__name__.lower()
        if any(isinstance(elt, ast.Starred) for elt in node.elts):
            self.problem(node, 'Starred collection length/element positions unknown')
            # No artificial fixed-arity collection may be produced.
            for elt in node.elts:
                self.expression(elt.value if isinstance(elt, ast.Starred) else elt)
            result = self.node('forward_collection', node, f'{kind}[unknown]', conditional=True)
            self.collection_ast_ids[node] = result
            return result
        aggregate = self.node('forward_collection', node, f'{kind}[{len(node.elts)}]')
        self.graph.nodes[-1]['attributes'].update({'collection_kind':kind, 'element_count':len(node.elts)})
        members = []
        for index, elem in enumerate(node.elts):
            source = self.expression(elem)
            member = self.node('forward_collection_element', elem, f'{kind}[{index}]')
            self.graph.nodes[-1]['attributes'].update({'collection_id':aggregate, 'element_index':index})
            members.append(member)
            self.link(source, member, elem, 'STATIC_COLLECTION_ELEMENT')
            self.link(member, aggregate, elem, 'STATIC_COLLECTION_MEMBER')
        self.collections[aggregate] = tuple(members)
        self.collection_ast_ids[node] = aggregate
        return aggregate

    def expression(self, node: ast.AST | None) -> str | None:
        if node is None or isinstance(node, ast.Constant):
            return None
        if isinstance(node, ast.Name):
            if node.id not in self.env:
                self.problem(node, f'Unresolved forward value: {node.id}')
            return self.env.get(node.id)
        if isinstance(node, ast.Call):
            value = stable_id(self.unit.path, 'semantic', 'call_site', self.ids[node], '')
            if isinstance(node.func, ast.Attribute) and (
                not isinstance(node.func.value, ast.Name) or node.func.value.id in self.env):
                self.link(self.expression(node.func.value), value, node.func.value)
            elif not isinstance(node.func, (ast.Attribute, ast.Name)):
                self.problem(node, 'Dynamically computed callee')
            for arg in node.args:
                self.link(self.expression(arg), value, arg)
            for kw in node.keywords:
                self.link(self.expression(kw.value), value, kw.value)
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float, complex)):
            return None
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Subscript)):
            value = stable_id(self.unit.path, 'semantic', 'operation_site', self.ids[node], '')
            operands = ([node.left, node.right] if isinstance(node, ast.BinOp) else
                        [node.operand] if isinstance(node, ast.UnaryOp) else [node.value])
            for arg in operands:
                self.link(self.expression(arg), value, arg)
            return value
        if isinstance(node, (ast.Tuple, ast.List)):
            return self.collection(node)
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == 'self':
                self.problem(node, f'Parameter or attribute reference not resolved: {ast.unparse(node)}')
                return None
            return self.expression(node.value)
        self.problem(node, f'Unsupported forward expression: {type(node).__name__}')
        return None

    def bind(self, target: ast.AST, expr: ast.AST) -> None:
        if isinstance(target, ast.Name):
            value = self.expression(expr)
            if value is None:
                self.env.pop(target.id, None)
            else:
                self.env[target.id] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            if isinstance(expr, (ast.Tuple, ast.List)) and len(expr.elts) == len(target.elts):
                for left, right in zip(target.elts, expr.elts):
                    self.bind(left, right)
                return
            origin = self.expression(expr)
            known = self.collections.get(origin, ()) if origin else ()
            exact = bool(known) and len(known) == len(target.elts) and all(
                isinstance(item, ast.Name) for item in target.elts)
            for index, item in enumerate(target.elts):
                if isinstance(item, ast.Name):
                    binding = self.node('forward_unpack', item, item.id, conditional=not exact)
                    self.graph.nodes[-1]['attributes'].update({
                        'element_index':index, 'source_collection_id': origin})
                    if exact:
                        self.link(known[index], binding, item, 'STATIC_COLLECTION_ELEMENT_BINDING')
                    else:
                        self.link(origin, binding, item)  # aggregate provenance; not an element proof
                    self.env[item.id] = binding
                else:
                    self.problem(item, 'Nested/dynamic destructuring not resolved')
            if not exact:
                self.problem(target, 'Unpacking arity/element identity unverified', site_id=origin)
            return
        self.problem(target, f'Mutating or computed assignment target: {ast.unparse(target)}')
        self.expression(expr)

    def augment(self, stmt: ast.AugAssign) -> None:
        if not isinstance(stmt.target, ast.Name):
            self.problem(stmt, 'Augmented mutation target is not a local name')
            self.expression(stmt.value)
            return
        old = self.env.get(stmt.target.id)
        if old is None:
            self.problem(stmt.target, f'Unknown previous value for {stmt.target.id}')
        value = self.expression(stmt.value)
        node_id = self.node('forward_augassign', stmt,
                            ast.unparse(stmt), conditional=True)
        self.link(old, node_id, stmt.target)
        self.link(value, node_id, stmt.value)
        self.env[stmt.target.id] = node_id
        self.problem(stmt, 'AugAssign value operands known; in-place mutation/aliases not proven')

    def branch(self, stmt: ast.If) -> None:
        guard = self.node('forward_guard', stmt, f'if {ast.unparse(stmt.test)}', conditional=True)
        if isinstance(stmt.test, ast.Name):
            self.link(self.expression(stmt.test), guard, stmt.test, 'STATIC_GUARD_DEPENDENCY')
        self.problem(stmt, 'Conditional alternatives represented, branch predicate not evaluated')
        before = dict(self.env)
        states: list[dict[str, str]] = []
        valid = True
        for side, body in (('true', stmt.body), ('false', stmt.orelse)):
            self.env = dict(before)
            for item in body:
                # Calls in either body are conditional execution sites, even
                # when a downstream merge can statically represent both paths.
                for call in (n for n in ast.walk(item) if isinstance(n, ast.Call)):
                    call_id = stable_id(self.unit.path, 'semantic', 'call_site', self.ids[call], '')
                    self.link(guard, call_id, call, 'STATIC_GUARDED_CALL', label=side)
            if not self.statements(body, inside_branch=True):
                valid = False
            states.append(dict(self.env))
        if not valid:
            self.env = {}
            self.problem(stmt, 'Unsupported control inside conditional; forward values invalidated')
            return
        merged: dict[str, str] = {}
        for name in sorted(set(before) | set(states[0]) | set(states[1])):
            left, right = states[0].get(name), states[1].get(name)
            if left is None or right is None:
                if left != right:
                    self.problem(stmt, f'Conditional value {name} unbound on one path')
                continue
            if left == right:
                merged[name] = left
            else:
                join = self.node('forward_choice', stmt, f'{name} | branches', conditional=True)
                self.link(left, join, stmt, 'STATIC_CONDITIONAL_DEPENDENCY', label=f'true:{name}')
                self.link(right, join, stmt, 'STATIC_CONDITIONAL_DEPENDENCY', label=f'false:{name}')
                self.link(guard, join, stmt, 'STATIC_GUARD_DEPENDENCY', label=name)
                merged[name] = join
        self.env = merged

    def statements(self, stmts: list[ast.stmt], *, inside_branch: bool = False) -> bool:
        for stmt in stmts:
            if self.terminated:
                break
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    self.bind(target, stmt.value)
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                self.bind(stmt.target, stmt.value)
            elif isinstance(stmt, ast.AugAssign):
                self.augment(stmt)
            elif isinstance(stmt, ast.If):
                self.branch(stmt)
            elif isinstance(stmt, ast.Return) and not inside_branch:
                ret = self.node('forward_output', stmt, 'Output')
                self.link(self.expression(stmt.value), ret, stmt, 'STATIC_FORWARD_RETURN')
                self.terminated = True
            elif isinstance(stmt, ast.Expr):
                self.expression(stmt.value)
            elif isinstance(stmt, ast.Pass):
                continue
            else:
                self.problem(stmt, f'Unsupported forward statement: {type(stmt).__name__}')
                self.env.clear()
                return False
        return True

    def run(self) -> None:
        args = [*self.fn.args.posonlyargs, *self.fn.args.args, *self.fn.args.kwonlyargs]
        for arg in args:
            if arg.arg not in ('self','cls'):
                self.env[arg.arg] = self.node('forward_input', arg, arg.arg)
        if self.fn.args.vararg or self.fn.args.kwarg:
            self.problem(self.fn, 'Variadic forward arguments require conservative treatment')
        self.statements(self.fn.body)
        if not self.terminated:
            self.problem(self.fn, 'No straight-line return proved')


def analyze_forward_values(graph: Graph, units: list[Unit]) -> None:
    # Functions/methods referenced by an unambiguous local call are eligible
    # for a conservative summary. Unreachable arbitrary source is not executed.
    local_targets = {e['target'] for e in graph.edges
                     if e['relation'] == 'CALLS_LOCAL_DEFINITION'}
    for unit in sorted(units, key=lambda u: u.path):
        ids = syntax_ids(unit)
        for statement in unit.tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ident = stable_id(unit.path, 'semantic', 'function_definition', ids[statement], '')
                if ident in local_targets:
                    ForwardValues(graph, unit, statement, definition_kind='function_definition').run()
            if isinstance(statement, ast.ClassDef):
                for method in statement.body:
                    if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    ident = stable_id(unit.path, 'semantic', 'method_definition', ids[method], '')
                    if method.name == 'forward' or ident in local_targets:
                        ForwardValues(graph, unit, method).run()

