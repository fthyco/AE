"""Source-grounded AST facts. No model imports, eval, tracing or execution.

This is intentionally a conservative first implementation: syntax is indexed
exhaustively; supported PyTorch constructor/call patterns are linked without
claiming that Python's dynamic semantics have been exhaustively resolved.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from typing import Any

from ..ir.canonical import stable_id
from ..ir.models import Graph
from ..parsing.source_spans import SourceMap


def expression(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except (ValueError, TypeError, RecursionError):
        return f"<{type(node).__name__}>"


def value_descriptor(node: ast.AST) -> dict[str, Any]:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, MemoryError, RecursionError):
        if isinstance(node, ast.Name):
            return {"kind": "symbol", "name": node.id}
        return {"kind": "expression", "text": expression(node)}
    if isinstance(value, (type(None), bool, int, float, str)):
        return {"kind": "literal", "value": value}
    if isinstance(value, (tuple, list)):
        return {"kind": "literal", "value": repr(value)}
    return {"kind": "literal_repr", "value": repr(value)}


def scalar_fields(node: ast.AST) -> dict[str, Any]:
    result = {}
    for key, val in ast.iter_fields(node):
        if isinstance(val, ast.AST) or (isinstance(val, list) and any(isinstance(x, ast.AST) for x in val)):
            continue
        if isinstance(val, (type(None), bool, str, int, float)):
            result[key] = val
        elif isinstance(val, list):
            result[key] = [v if isinstance(v, (type(None), bool, str, int, float)) else repr(v) for v in val]
        else:
            result[key] = repr(val)
    return result


class Analyzer(ast.NodeVisitor):
    def __init__(self, graph: Graph, source: SourceMap, tree: ast.Module, source_id: str):
        self.graph = graph
        self.source = source
        self.tree = tree
        self.source_id = source_id
        self.syntax: dict[ast.AST, str] = {}
        self.parents: dict[ast.AST, ast.AST | None] = {}
        self.scopes: list[str] = []
        self.class_nodes: list[str] = []
        self.class_names: list[str] = []
        self.imports: dict[str, str] = {}
        self.module_sites: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.module_calls: list[tuple[str, str, str, dict]] = []
        self.semantics: dict[ast.AST, str] = {}
        self._index_syntax()
        self._collect_imports()

    def span(self, node: ast.AST) -> dict:
        if isinstance(node, ast.Module) or not hasattr(node, "lineno"):
            current = node
            while current not in self.parents or not hasattr(current, "lineno"):
                parent = self.parents.get(current)
                if parent is None:
                    return {"path": self.source.path, "start_byte": 0,
                            "end_byte": len(self.source.raw), "start_line": 1,
                            "start_column": 0, "end_line": len(self.source.lines) or 1,
                            "end_column": 0, "precision": "file"}
                current = parent
            span = self.source.ast_span(current)
            span["precision"] = "enclosing"
            return span
        span = self.source.ast_span(node)
        span["precision"] = "exact"
        return span

    def add_node(self, kind: str, node: ast.AST, *, attrs: dict | None = None,
                 evidence: str = "explicit", rule: str | None = None, discriminator: str = "") -> str:
        id_ = stable_id(self.source.path, "semantic", kind, self.syntax[node], discriminator)
        entry: dict = {"id": id_, "kind": kind, "attributes": attrs or {},
                       "source_refs": [self.span(node)], "evidence_class": evidence,
                       "resolution": "resolved" if evidence == "explicit" else "conditional"}
        if rule:
            entry["analysis_rule"] = rule
        self.graph.add_node(entry)
        self.edge(id_, self.syntax[node], "APPEARS_AT", node)
        self.semantics[node] = id_
        return id_

    def edge(self, src: str, dst: str, relation: str, at: ast.AST, *,
             rule: str | None = None, attrs: dict | None = None) -> None:
        number = len(self.graph.edges)
        edge = {"id": stable_id(self.source.path, "edge", src, dst, relation, number),
                "source": src, "target": dst, "relation": relation,
                "evidence_class": "derived" if rule else "explicit",
                "source_refs": [self.span(at)]}
        if rule:
            edge["analysis_rule"] = rule
        if attrs:
            edge["attributes"] = attrs
        self.graph.add_edge(edge)

    def _index_syntax(self) -> None:
        def recurse(node: ast.AST, parent: ast.AST | None, key: str) -> None:
            node_id = stable_id(self.source.path, "syntax", key, type(node).__name__)
            self.parents[node] = parent
            self.syntax[node] = node_id
            self.graph.add_node({"id": node_id, "kind": "syntax_node",
                                 "attributes": {"ast_type": type(node).__name__, "fields": scalar_fields(node), "ast_path": key},
                                 "source_refs": [self.span(node)], "evidence_class": "explicit", "resolution": "resolved"})
            self.edge(self.syntax[parent] if parent is not None else self.source_id,
                      node_id, "CONTAINS", node, attrs={"field_path": key})
            for field, val in ast.iter_fields(node):
                if isinstance(val, ast.AST):
                    recurse(val, node, f"{key}/{field}")
                elif isinstance(val, list):
                    for i, child in enumerate(val):
                        if isinstance(child, ast.AST):
                            recurse(child, node, f"{key}/{field}[{i}]")
        recurse(self.tree, None, "root")

    def _collect_imports(self) -> None:
        # Only unshadowed top-level aliases are used as recognition hints.
        aliases: dict[str, str] = {}
        shadowed: set[str] = set()
        for statement in self.tree.body:
            if isinstance(statement, ast.Import):
                for item in statement.names:
                    aliases[item.asname or item.name.split(".")[0]] = item.name if item.asname else item.name.split(".")[0]
            elif isinstance(statement, ast.ImportFrom):
                if statement.level != 0:
                    continue
                for item in statement.names:
                    if item.name != "*":
                        aliases[item.asname or item.name] = f"{statement.module}.{item.name}" if statement.module else item.name
            elif isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                for target in targets:
                    shadowed |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        self.imports = {k: v for k, v in aliases.items() if k not in shadowed}

    def qualified(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return self.imports.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{self.qualified(node.value)}.{node.attr}"
        return expression(node)

    def pytorch_constructor(self, call: ast.Call) -> str | None:
        name = self.qualified(call.func)
        if name.startswith("torch.nn.") and len(name.split(".")) == 3:
            return name
        # Avoid misclassifying custom objects simply named Linear.
        return None

    def run(self) -> None:
        self.visit(self.tree)
        self._link_module_calls()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        ident = ".".join([*self.scopes, node.name])
        class_id = self.add_node("class_definition", node, attrs={"name": node.name, "qualname": ident,
                       "bases": [self.qualified(base) for base in node.bases]})
        if self.class_nodes:
            self.edge(self.class_nodes[-1], class_id, "DEFINES", node)
        for base in node.bases:
            self.edge(class_id, self.syntax[base], "INHERITS_EXPRESSION", base)
        self.scopes.append(node.name)
        self.class_nodes.append(class_id)
        self.class_names.append(ident)
        self.generic_visit(node)
        self.class_names.pop()
        self.class_nodes.pop()
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = ".".join([*self.scopes, node.name])
        kind = "method_definition" if self.class_nodes else "function_definition"
        func = self.add_node(kind, node, attrs={"name": node.name, "qualname": qualname,
                   "async": isinstance(node, ast.AsyncFunctionDef)})
        if self.class_nodes:
            self.edge(self.class_nodes[-1], func, "DEFINES", node)
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_Import(self, node: ast.Import) -> None:
        self.add_node("import_reference", node, attrs={"imports": [{"name": n.name, "asname": n.asname} for n in node.names]})

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.add_node("import_reference", node, attrs={"module": node.module, "level": node.level,
                    "names": [{"name": n.name, "asname": n.asname} for n in node.names]})

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        for target in node.targets:
            self._binding(node, target, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.generic_visit(node)
        self._binding(node, node.target, node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.generic_visit(node)
        self._binding(node, node.target, node.value)

    def _binding(self, node: ast.AST, target: ast.AST, value: ast.AST | None) -> None:
        target_name = expression(target)
        is_attribute = isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"
        attrs = {"target": target_name, "scope": ".".join(self.scopes)}
        kind = "attribute_binding" if is_attribute else "assignment_site"
        binding = self.add_node(kind, target, attrs=attrs, discriminator=self.syntax[node])
        if value is not None:
            self.edge(binding, self.syntax[value], "VALUE_EXPRESSION", value)
        if is_attribute and self.class_nodes:
            self.edge(self.class_nodes[-1], binding, "DECLARES_ATTRIBUTE", target)
            # Constructors can be assigned from a call directly. This does not
            # claim a unique runtime module instance in dynamic code.
            if isinstance(value, ast.Call):
                site = self.semantics.get(value)
                constructor = self.pytorch_constructor(value)
                if constructor and site:
                    construction = self.add_node("module_construction_site", value,
                            attrs={"declared_type": constructor, "owner_expression": target_name,
                                   "constructor_arguments": [value_descriptor(a) for a in value.args],
                                   "constructor_keywords": {k.arg or "**": value_descriptor(k.value) for k in value.keywords}},
                            evidence="derived", rule="pytorch-constructor-import-resolution/v1")
                    self.edge(construction, site, "CONSTRUCTS_AT", value, rule="constructor-call-link/v1")
                    self.edge(binding, construction, "BINDS_CONSTRUCTION", target, rule="attribute-constructor-binding/v1")
                    self.module_sites[(self.class_names[-1], target.attr)].append(construction)
                    for argument in [*value.args, *(k.value for k in value.keywords)]:
                        desc = value_descriptor(argument)
                        if desc["kind"] != "literal":
                            self.graph.unresolved.append({"kind": "symbolic_constructor_argument", "path": self.source.path,
                                 "site_id": construction, "expression": expression(argument),
                                 "reason": "Value not proven as literal by static analysis", "source_refs": [self.span(argument)]})

    def visit_Call(self, node: ast.Call) -> None:
        # Semantic call is distinct from the module construction site.
        callee = expression(node.func)
        call = self.add_node("call_site", node, attrs={"callee_expression": callee,
                      "qualified_callee_hint": self.qualified(node.func),
                      "positional_arguments": [value_descriptor(a) for a in node.args],
                      "keyword_arguments": [{"name": k.arg, "value": value_descriptor(k.value)} for k in node.keywords]})
        self.edge(call, self.syntax[node.func], "CALLEE_EXPRESSION", node.func)
        for i, arg in enumerate(node.args):
            self.edge(call, self.syntax[arg], "HAS_ARGUMENT", arg, attrs={"index": i})
        for kw in node.keywords:
            self.edge(call, self.syntax[kw.value], "HAS_KEYWORD", kw.value, attrs={"name": kw.arg})
        if self.class_names and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
            self.module_calls.append((call, self.class_names[-1], node.func.attr, node))
        self.generic_visit(node)
        self._nested_dependencies(node, call)

    def _nested_dependencies(self, node: ast.AST, site: str) -> None:
        for arg in ast.iter_child_nodes(node):
            # An operation syntactically contains calls/ops in its operands.
            if isinstance(arg, (ast.Call, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Subscript)):
                child = self.semantics.get(arg)
                if child and child != site:
                    self.edge(site, child, "EXPRESSION_DEPENDS_ON", arg,
                              rule="syntactic-operand-dependency/v1")
            elif not isinstance(arg, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                # recurse through arguments/keyword wrappers, not callable body
                for sub in ast.walk(arg):
                    if isinstance(sub, (ast.Call, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Subscript)):
                        child = self.semantics.get(sub)
                        if child and child != site:
                            self.edge(site, child, "EXPRESSION_DEPENDS_ON", sub,
                                      rule="syntactic-operand-dependency/v1")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self._operation(node, "binary_operator", type(node.op).__name__)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        self._operation(node, "unary_operator", type(node.op).__name__)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self._operation(node, "boolean_operator", type(node.op).__name__)

    def visit_Compare(self, node: ast.Compare) -> None:
        self._operation(node, "comparison", ",".join(type(op).__name__ for op in node.ops))

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self._operation(node, "indexing_operation", "Subscript")

    def _operation(self, node: ast.AST, kind: str, name: str) -> None:
        ident = self.add_node("operation_site", node, attrs={"operation_kind": kind, "operator": name,
                                                              "expression": expression(node)})
        self.generic_visit(node)
        self._nested_dependencies(node, ident)

    def visit_Return(self, node: ast.Return) -> None:
        ident = self.add_node("return_site", node, attrs={"scope": ".".join(self.scopes)})
        if node.value is not None:
            self.edge(ident, self.syntax[node.value], "RETURNS_EXPRESSION", node.value)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self._control(node, "if", node.test, node.body, node.orelse)

    def visit_For(self, node: ast.For) -> None:
        self._control(node, "for", node.iter, node.body, node.orelse)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._control(node, "async_for", node.iter, node.body, node.orelse)

    def visit_While(self, node: ast.While) -> None:
        self._control(node, "while", node.test, node.body, node.orelse)

    def _control(self, node: ast.AST, control_kind: str, condition: ast.AST,
                 body: list[ast.stmt], other: list[ast.stmt]) -> None:
        ident = self.add_node("control_site", node, attrs={"control_kind": control_kind,
                    "condition_expression": expression(condition)})
        self.edge(ident, self.syntax[condition], "CONDITION_EXPRESSION", condition)
        for branch, statements in (("body", body), ("else", other)):
            for statement in statements:
                self.edge(ident, self.syntax[statement], "HAS_BRANCH_STATEMENT", statement,
                          attrs={"branch": branch})
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._control(node, "conditional_expression", node.test, [], [])

    def visit_Try(self, node: ast.Try) -> None:
        ident = self.add_node("control_site", node, attrs={"control_kind": "try"})
        for part, statements in (("body", node.body), ("handler", node.handlers), ("else", node.orelse), ("finally", node.finalbody)):
            for statement in statements:
                self.edge(ident, self.syntax[statement], "HAS_BRANCH_STATEMENT", statement, attrs={"branch": part})
        self.generic_visit(node)

    def _link_module_calls(self) -> None:
        for call_id, scope, attr, call_node in self.module_calls:
            candidates = self.module_sites.get((scope, attr), [])
            if len(candidates) == 1:
                self.edge(call_id, candidates[0], "USES_MODULE_REFERENCE", call_node,
                          rule="unique-static-attribute-construction/v1", attrs={"attribute": attr})
            else:
                self.graph.unresolved.append({"kind": "module_attribute_resolution", "path": self.source.path,
                        "site_id": call_id, "expression": f"self.{attr}",
                        "reason": "No unique statically recognized module construction" if not candidates else "Multiple possible module construction sites",
                        "candidate_ids": candidates, "source_refs": [self.span(call_node)]})
