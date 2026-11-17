"""Non-executing Python parser with exact-source token indexing."""
from __future__ import annotations

import ast
import io
import tokenize

from .source_spans import SourceMap
from ..ir.canonical import stable_id


def parse_python(path: str, raw: bytes) -> tuple[SourceMap, ast.Module | None, list[dict], dict | None]:
    try:
        source = SourceMap.from_bytes(path, raw)
    except (UnicodeError, SyntaxError, LookupError) as exc:
        # No AST, but caller preserves the original bytes in the snapshot.
        return None, None, [], {"kind": "decode_error", "path": path, "message": str(exc)}  # type: ignore[return-value]
    tokens: list[dict] = []
    try:
        for i, token in enumerate(tokenize.tokenize(io.BytesIO(raw).readline)):
            if token.type in (tokenize.ENCODING, tokenize.ENDMARKER):
                continue
            start = source.token_offset(*token.start)
            end = source.token_offset(*token.end)
            tokens.append({"id": stable_id(path, "token", i, start, end), "kind": tokenize.tok_name[token.type],
                           "value": token.string, "start_byte": start, "end_byte": end,
                           "start_line": token.start[0], "start_column": token.start[1],
                           "end_line": token.end[0], "end_column": token.end[1]})
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        return source, None, tokens, {"kind": "tokenize_error", "path": path, "message": str(exc)}
    try:
        tree = ast.parse(raw, filename=path, type_comments=True)
    except (SyntaxError, ValueError, UnicodeError) as exc:
        return source, None, tokens, {"kind": "syntax_error", "path": path, "message": str(exc)}
    return source, tree, tokens, None
