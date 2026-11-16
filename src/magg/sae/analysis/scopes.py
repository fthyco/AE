"""Conservative function-local alias shadow detection, no code execution."""
from __future__ import annotations
import ast


def binding_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names = {a.arg for a in [*function.args.posonlyargs, *function.args.args,
             *function.args.kwonlyargs, *([function.args.vararg] if function.args.vararg else []),
             *([function.args.kwarg] if function.args.kwarg else [])]}
    def recur(node: ast.AST) -> None:
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
            recur(child)
    for stmt in function.body:
        recur(stmt)
    # A global declaration means module namespace is used, but assignments to
    # that name at runtime still prohibit proving a fixed binding from syntax.
    return names


def alias_is_shadowed(node: ast.AST, name: str, parents: dict[ast.AST, ast.AST | None]) -> bool:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if name in binding_names(current):
                return True
        elif isinstance(current, ast.Lambda):
            if name in {a.arg for a in [*current.args.posonlyargs, *current.args.args,
                         *current.args.kwonlyargs]}:
                return True
        current = parents.get(current)
    return False


def parent_links(tree: ast.AST) -> dict[ast.AST, ast.AST | None]:
    parents: dict[ast.AST, ast.AST | None] = {tree: None}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents
