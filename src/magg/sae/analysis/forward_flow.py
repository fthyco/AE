"""Conservative source-only forward value dependencies.

This pass represents operand/return dependencies in straight-line forward code.
It does not assert tensor types, runtime paths, shape correctness, or effects
inside called functions. Uncertain control flow and mutations are reported.
"""
from __future__ import annotations

import ast

from ..ir.canonical import stable_id
from ..ir.models import Graph
from ..linking.project import Unit, derived_edge, span, syntax_ids

RULE = 'straight-line-forward-value/v1'


class ForwardValues:
    def __init__(self, graph: Graph, unit: Unit, fn: ast.FunctionDef | ast.AsyncFunctionDef):
        self.graph, self.unit, self.fn = graph, unit, fn
        self.ids = syntax_ids(unit)
        self.forward_id = stable_id(unit.path, 'semantic', 'method_definition', self.ids[fn], '')
        self.env: dict[str, str] = {}
        self.terminated = False

    def node(self, kind: str, astnode: ast.AST, label: str, *, conditional: bool = False) -> str:
        ident = stable_id(self.unit.path, 'forward-value', kind, self.ids[astnode])
        self.graph.add_node({'id': ident, 'kind': kind,
                             'attributes': {'name': label, 'forward_id': self.forward_id},
                             'source_refs': [span(self.unit, astnode)],
                             'evidence_class': 'derived',
                             'analysis_rule': RULE,
                             'resolution': 'conditional' if conditional else 'resolved'})
        derived_edge(self.graph, self.unit, astnode, self.forward_id, ident,
                     'HAS_FORWARD_VALUE', RULE)
        return ident

    def link(self, source: str | None, target: str, at: ast.AST,
             relation: str = 'STATIC_VALUE_DEPENDENCY') -> None:
        if source is not None:
            derived_edge(self.graph, self.unit, at, source, target, relation, RULE)

    def problem(self, astnode: ast.AST, reason: str) -> None:
        self.graph.unresolved.append({'kind': 'forward_flow_incomplete', 'path': self.unit.path,
                                      'function_id': self.forward_id, 'reason': reason,
                                      'source_refs': [span(self.unit, astnode)]})

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
            if isinstance(node, ast.BinOp):
                operands = [node.left, node.right]
            elif isinstance(node, ast.UnaryOp):
                operands = [node.operand]
            else:
                operands = [node.value]
            for arg in operands:
                self.link(self.expression(arg), value, arg)
            return value
        if isinstance(node, (ast.Tuple, ast.List)):
            if all(isinstance(element, ast.Constant) or (
                    isinstance(element, ast.UnaryOp) and isinstance(element.operand, ast.Constant))
                   for element in node.elts):
                return None
            self.problem(node, 'Tuple/list expression requires element-wise handling')
            return None
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
            for item in target.elts:
                if isinstance(item, ast.Name):
                    binding = self.node('forward_unpack', item, item.id, conditional=True)
                    self.link(origin, binding, item)
                    self.env[item.id] = binding
                else:
                    self.problem(item, 'Nested/dynamic destructuring not resolved')
            self.problem(target, 'Unpacking arity/element identity unverified')
            return
        self.problem(target, f'Mutating or computed assignment target: {ast.unparse(target)}')
        self.expression(expr)

    def run(self) -> None:
        args = [*self.fn.args.posonlyargs, *self.fn.args.args, *self.fn.args.kwonlyargs]
        for arg in args:
            if arg.arg in ('self', 'cls'):
                continue
            self.env[arg.arg] = self.node('forward_input', arg, arg.arg)
        if self.fn.args.vararg or self.fn.args.kwarg:
            self.problem(self.fn, 'Variadic forward arguments require conservative treatment')
        for stmt in self.fn.body:
            if self.terminated:
                break
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    self.bind(target, stmt.value)
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                self.bind(stmt.target, stmt.value)
            elif isinstance(stmt, ast.Return):
                ret = self.node('forward_output', stmt, 'Output')
                self.link(self.expression(stmt.value), ret, stmt, 'STATIC_FORWARD_RETURN')
                self.terminated = True
            elif isinstance(stmt, ast.Expr):
                self.expression(stmt.value)
            elif isinstance(stmt, ast.Pass):
                continue
            else:
                self.problem(stmt, f'Control flow or mutation not proven: {type(stmt).__name__}')
                self.env.clear()
        if not self.terminated:
            self.problem(self.fn, 'No straight-line return proved')


def analyze_forward_values(graph: Graph, units: list[Unit]) -> None:
    for unit in sorted(units, key=lambda u: u.path):
        for klass in unit.tree.body:
            if not isinstance(klass, ast.ClassDef):
                continue
            for statement in klass.body:
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name == 'forward':
                    ForwardValues(graph, unit, statement).run()
