"""Conservative intraprocedural lexical def/use for statically visible Python.

Edges indicate lexical reaching-definition evidence, *not* runtime tensor flow.
Conditional assignments and loop-carried values remain ambiguous, with their
candidate definitions preserved for later reasoning.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from typing import TypeAlias

from ..ir.canonical import stable_id
from ..ir.models import Graph
from ..linking.project import Unit, derived_edge, span, syntax_ids, unresolved

Env: TypeAlias = dict[str, set[str | None]]


def _merge(*envs: Env) -> Env:
    names = set().union(*(env.keys() for env in envs))
    return {name: set().union(*(env.get(name, {None}) for env in envs)) for name in names}


class FunctionFlow:
    def __init__(self, graph: Graph, unit: Unit, function: ast.FunctionDef | ast.AsyncFunctionDef,
                 ids: dict[ast.AST, str]):
        self.graph, self.unit, self.function, self.ids = graph, unit, function, ids
        self.scope = stable_id(unit.path, 'lexical-function', ids[function])
        self.local_names = self._locals(function)
        self.module_names = set()
        for stmt in unit.tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.module_names.add(stmt.name)
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                for item in stmt.names:
                    if item.name != '*':
                        self.module_names.add(item.asname or (item.name.split('.')[0] if isinstance(stmt, ast.Import) else item.name))
        self.globals: set[str] = set()
        self.nonlocals: set[str] = set()
        self._find_global_declarations(function.body)

    def _find_global_declarations(self, stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.Global):
                self.globals.update(stmt.names)
            elif isinstance(stmt, ast.Nonlocal):
                self.nonlocals.update(stmt.names)
            elif isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            else:
                for _, value in ast.iter_fields(stmt):
                    if isinstance(value, ast.stmt):
                        self._find_global_declarations([value])
                    elif isinstance(value, list):
                        self._find_global_declarations([v for v in value if isinstance(v, ast.stmt)])

    def _locals(self, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        names = {n.arg for n in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs,
                 *([fn.args.vararg] if fn.args.vararg else []), *([fn.args.kwarg] if fn.args.kwarg else [])]}
        def descend(node: ast.AST):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(child.name)
                    continue
                if isinstance(child, ast.Lambda):
                    continue
                if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                    names.add(child.id)
                elif isinstance(child, (ast.Import, ast.ImportFrom)):
                    for alias in child.names:
                        names.add(alias.asname or (alias.name.split('.')[0] if isinstance(child, ast.Import) else alias.name))
                descend(child)
        for statement in fn.body:
            descend(statement)
        return names

    def _definition(self, node: ast.AST, name: str, kind: str) -> str:
        ident = stable_id(self.unit.path, 'lexical-definition', self.scope, self.ids[node], name, kind)
        self.graph.add_node({'id': ident, 'kind': 'lexical_definition',
                             'attributes': {'name': name, 'binding_kind': kind, 'scope_id': self.scope},
                             'source_refs': [span(self.unit, node)], 'evidence_class': 'derived',
                             'resolution': 'resolved', 'analysis_rule': 'lexical-definition/v1'})
        derived_edge(self.graph, self.unit, node, ident, self.ids[node],
                     'DEFINES_AT_SYNTAX', 'lexical-definition/v1', label=name)
        return ident

    def _binding_target(self, target: ast.AST, env: Env, kind: str) -> None:
        if isinstance(target, ast.Name):
            if target.id in self.globals or target.id in self.nonlocals:
                unresolved(self.graph, self.unit, target, 'nonlocal_write',
                           f'{target.id} requires interprocedural global/nonlocal analysis')
                return
            env[target.id] = {self._definition(target, target.id, kind)}
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._binding_target(element, env, kind)
        elif isinstance(target, (ast.Attribute, ast.Subscript)):
            # Mutation of referenced storage is not a rebinding of the base name.
            self._expression(target, env)

    def _read(self, node: ast.Name, env: Env) -> None:
        name = node.id
        if name in self.globals or name in self.nonlocals:
            unresolved(self.graph, self.unit, node, 'global_or_nonlocal_read',
                       f'{name} is declared global/nonlocal in this scope')
            return
        reaching = env.get(name, {None})
        valid = {ident for ident in reaching if ident is not None}
        if len(valid) == 1 and None not in reaching:
            derived_edge(self.graph, self.unit, node, self.ids[node], next(iter(valid)),
                         'READS_LEXICAL_DEFINITION', 'unique-reaching-definition/v1')
        elif name in self.local_names or reaching != {None}:
            unresolved(self.graph, self.unit, node, 'ambiguous_reaching_definition',
                       f'{name} has {len(valid)} candidate definitions; may be unbound={None in reaching}',
                       candidates=valid)
        elif name in self.module_names:
            return  # module binding is represented by import/definition facts
        else:
            # A module/global/builtin/import or free-variable reference cannot be
            # equated with a local definition from source syntax alone.
            unresolved(self.graph, self.unit, node, 'external_or_free_variable',
                       f'No intraprocedural lexical definition for {name}')

    def _expression(self, node: ast.AST | None, env: Env) -> None:
        if node is None:
            return
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                self._read(node, env)
        elif isinstance(node, ast.NamedExpr):
            self._expression(node.value, env)
            self._binding_target(node.target, env, 'walrus')
        elif isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            # Independent nested scopes, not yet part of this intraprocedural pass.
            unresolved(self.graph, self.unit, node, 'nested_expression_scope',
                       'Comprehension or lambda has a distinct lexical scope')
        else:
            for child in ast.iter_child_nodes(node):
                self._expression(child, env)

    def _block(self, statements: list[ast.stmt], incoming: Env) -> Env | None:
        env = {k: set(v) for k, v in incoming.items()}
        for stmt in statements:
            if env is None:
                break  # lexical block terminated by unconditional return/raise
            env = self._statement(stmt, env)
        return env

    def _statement(self, stmt: ast.stmt, env: Env) -> Env | None:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            self._expression(stmt.value, env)
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                self._binding_target(target, env, 'assignment')
        elif isinstance(stmt, ast.AugAssign):
            self._expression(stmt.value, env)
            if isinstance(stmt.target, ast.Name):
                self._read(stmt.target, env)
            else:
                self._expression(stmt.target, env)
            self._binding_target(stmt.target, env, 'augmented_assignment')
        elif isinstance(stmt, (ast.Return, ast.Raise)):
            self._expression(stmt.value if isinstance(stmt, ast.Return) else stmt.exc, env)
            if isinstance(stmt, ast.Raise):
                self._expression(stmt.cause, env)
            return None
        elif isinstance(stmt, ast.If):
            self._expression(stmt.test, env)
            first = self._block(stmt.body, env)
            second = self._block(stmt.orelse, env) if stmt.orelse else env
            endings = [part for part in (first, second) if part is not None]
            return _merge(*endings) if endings else None
        elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            if isinstance(stmt, (ast.For, ast.AsyncFor)):
                self._expression(stmt.iter, env)
            else:
                self._expression(stmt.test, env)
            body_env = {k: set(v) for k, v in env.items()}
            if isinstance(stmt, (ast.For, ast.AsyncFor)):
                self._binding_target(stmt.target, body_env, 'loop_target')
            after_body = self._block(stmt.body, body_env)
            # The loop may execute zero, one, or many times. Mark all modified
            # bindings as conservative alternatives, including pre-loop values.
            possible = _merge(env, *([after_body] if after_body is not None else []))
            if stmt.orelse:
                after_else = self._block(stmt.orelse, possible)
                return _merge(possible, after_else) if after_else is not None else possible
            return possible
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                self._expression(item.context_expr, env)
                if item.optional_vars:
                    self._binding_target(item.optional_vars, env, 'with_binding')
            return self._block(stmt.body, env)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            env[stmt.name] = {self._definition(stmt, stmt.name, 'nested_definition')}
            # Do not enter nested method/class body in outer function's flow.
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                name = alias.asname or (alias.name.split('.')[0] if isinstance(stmt, ast.Import) else alias.name)
                if name != '*':
                    env[name] = {self._definition(stmt, name, 'import')}
        elif isinstance(stmt, ast.Delete):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    env[target.id] = {None}
                else:
                    self._expression(target, env)
        elif isinstance(stmt, ast.Try):
            normal = self._block(stmt.body, env)
            outcomes = []
            if normal is not None:
                outcomes.append(self._block(stmt.orelse, normal) if stmt.orelse else normal)
            for handler in stmt.handlers:
                self._expression(handler.type, env)
                handle_env = {k: set(v) for k, v in env.items()}
                if handler.name:
                    handle_env[handler.name] = {self._definition(handler, handler.name, 'exception')}
                outcomes.append(self._block(handler.body, handle_env))
            outcomes = [x for x in outcomes if x is not None]
            joint = _merge(env, *outcomes)
            return self._block(stmt.finalbody, joint) if stmt.finalbody else joint
        elif isinstance(stmt, ast.Expr):
            self._expression(stmt.value, env)
        elif isinstance(stmt, ast.Assert):
            self._expression(stmt.test, env)
            self._expression(stmt.msg, env)
        elif isinstance(stmt, (ast.Break, ast.Continue)):
            # Early loop exits need a dedicated CFG to recover exact reachability.
            unresolved(self.graph, self.unit, stmt, 'nonlinear_control_flow',
                       'Break/continue requires dedicated control-flow modeling')
            return None
        else:
            unresolved(self.graph, self.unit, stmt, 'unsupported_statement_flow',
                       f'No source-level flow rule for {type(stmt).__name__}')
        return env

    def run(self) -> None:
        env: Env = {}
        args = [*self.function.args.posonlyargs, *self.function.args.args, *self.function.args.kwonlyargs]
        for optional in (self.function.args.vararg, self.function.args.kwarg):
            if optional is not None:
                args.append(optional)
        for arg in args:
            env[arg.arg] = {self._definition(arg, arg.arg, 'parameter')}
        self._block(self.function.body, env)


def analyze_dataflow(graph: Graph, units: list[Unit]) -> None:
    for unit in sorted(units, key=lambda u: u.path):
        ids = syntax_ids(unit)
        def visit(node: ast.AST) -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                FunctionFlow(graph, unit, node, ids).run()
            for child in ast.iter_child_nodes(node):
                visit(child)
        visit(unit.tree)
