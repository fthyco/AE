"""Static forward inheritance through locally proved class imports."""
from __future__ import annotations

import ast
from collections import defaultdict
from typing import Any

from ..linking.project import Unit


def select_forward_method(klass: ast.ClassDef) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Select a unique forward, including explicit TorchScript overload stubs."""
    methods = [method for method in klass.body
               if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
               and method.name == 'forward']
    if len(methods) == 1:
        return methods[0]
    if len(methods) < 2:
        return None
    candidate = methods[-1]
    if any('overload' in ast.unparse(decorator) for decorator in candidate.decorator_list):
        return None
    for method in methods[:-1]:
        if not any(ast.unparse(decorator) == 'torch.jit._overload_method'
                   for decorator in method.decorator_list):
            return None
        if not method.body or any(not isinstance(node, (ast.Pass, ast.Expr))
                                  for node in method.body):
            return None
        if any(isinstance(node, ast.Expr)
               and not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
               for node in method.body):
            return None
    return candidate


def resolve_forward_owner(definitions: dict[str, tuple[Unit, ast.ClassDef]],
                          units: list[Unit], definition_id: str
                          ) -> tuple[str | None, str, list[str]]:
    """Return a unique method owner, resolution state and base-chain IDs."""
    by_name: dict[str, list[str]] = defaultdict(list)
    for ident, (_, klass) in definitions.items():
        by_name[klass.name].append(ident)
    for values in by_name.values():
        values.sort()
    packages = {u.module: u for u in units if u.path.endswith('/__init__.py')}

    def reexport(package: str, symbol: str, target: str) -> bool:
        pkg = packages.get(package)
        if pkg is None:
            return False
        for stmt in pkg.tree.body:
            if not isinstance(stmt, ast.ImportFrom) or stmt.level != 1:
                continue
            module = (package + '.' + (stmt.module or '')).rstrip('.')
            if module != target:
                continue
            if any(alias.name == symbol and alias.asname is None for alias in stmt.names):
                return True
        return False

    def imported_class(owner: Unit, symbol: str, candidate: Unit) -> bool:
        owner_classes = [stmt for stmt in owner.tree.body
                         if isinstance(stmt, ast.ClassDef) and stmt.name != symbol]
        if not owner_classes:
            return False
        # Resolve in the module's static top-level namespace only.
        # A conflicting assignment or declaration blocks the candidate.
        conflicting = [stmt for stmt in owner.tree.body
                       if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                       and any(isinstance(target, ast.Name) and target.id == symbol
                               for target in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]))]
        if conflicting:
            return False
        if candidate.path == owner.path:
            return (len([stmt for stmt in owner.tree.body
                         if isinstance(stmt, ast.ClassDef) and stmt.name == symbol]) == 1
                    and not any(isinstance(stmt, (ast.Import, ast.ImportFrom))
                                and any((alias.asname or alias.name) == symbol
                                        for alias in stmt.names)
                                for stmt in owner.tree.body))
        for stmt in owner.tree.body:
            if not isinstance(stmt, ast.ImportFrom):
                continue
            for alias in stmt.names:
                if (alias.asname or alias.name) != symbol or alias.name != symbol:
                    continue
                module = stmt.module or ''
                if stmt.level:
                    bits = owner.module.split('.')[:-stmt.level]
                    module = '.'.join([*bits, module]) if module else '.'.join(bits)
                if module == candidate.module or module.endswith('.' + candidate.module):
                    return True
                pkg = candidate.module.rsplit('.', 1)[0] if '.' in candidate.module else ''
                if pkg and (module == pkg or module.endswith('.' + pkg)):
                    if reexport(pkg, alias.name, candidate.module):
                        return True
        return False

    def visit(ident: str, active: frozenset[str]
              ) -> tuple[str | None, str, list[str]]:
        if ident in active:
            return None, 'recursive-base-class', []
        unit, klass = definitions[ident]
        selected = select_forward_method(klass)
        if selected is not None:
            return ident, 'declared', []
        own = [x for x in klass.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
               and x.name == 'forward']
        if len(own) > 1:
            return None, 'ambiguous-forward-override', []
        if not klass.bases:
            return None, 'no-forward-declaration-or-base', []
        # Conservative: only a single statically identified base is supported.
        # Known torch.nn.Module adds no useful forward implementation here.
        if len(klass.bases) != 1 or not isinstance(klass.bases[0], ast.Name):
            return None, 'unsupported-or-multiple-bases', []
        symbol = klass.bases[0].id
        candidates = [value for value in by_name[symbol]
                      if imported_class(unit, symbol, definitions[value][0])]
        if len(candidates) != 1:
            return None, 'unresolved-or-ambiguous-base', []
        base_id = candidates[0]
        found, status, tail = visit(base_id, active | {ident})
        return found, ('inherited' if found else status), ([base_id, *tail] if found else [])

    return visit(definition_id, frozenset())
