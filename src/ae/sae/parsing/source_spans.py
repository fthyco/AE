"""Source position conversion. AST offsets are UTF-8 byte columns, not chars."""
from __future__ import annotations

from dataclasses import dataclass
import ast
import io
import tokenize


@dataclass
class SourceMap:
    path: str
    raw: bytes
    encoding: str
    text: str
    lines: list[str]
    line_offsets: list[int]

    @classmethod
    def from_bytes(cls, path: str, raw: bytes) -> "SourceMap":
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        text = raw.decode(encoding)
        # splitlines(True) preserves original line endings; a final empty line is not fabricated.
        lines = text.splitlines(keepends=True)
        # A UTF-8 BOM occurs only at the very start of the original byte stream.
        base_encoding = "utf-8" if encoding == "utf-8-sig" else encoding
        offsets = [3 if raw.startswith(b"\xef\xbb\xbf") and encoding == "utf-8-sig" else 0]
        for line in lines:
            offsets.append(offsets[-1] + len(line.encode(base_encoding)))
        return cls(path, raw, base_encoding, text, lines, offsets)

    def offset(self, lineno: int, utf8_col: int) -> int:
        # AST columns are measured in UTF-8 bytes (even for an encoding cookie).
        if lineno < 1 or lineno > len(self.lines):
            raise ValueError("Invalid source line")
        segment = self.lines[lineno - 1].encode("utf-8")[:utf8_col].decode("utf-8")
        return self.line_offsets[lineno - 1] + len(segment.encode(self.encoding))

    def token_offset(self, lineno: int, char_col: int) -> int:
        if lineno < 1:
            return 0
        if lineno > len(self.lines):
            return len(self.raw)
        return self.line_offsets[lineno - 1] + len(self.lines[lineno - 1][:char_col].encode(self.encoding))

    def ast_span(self, node: ast.AST) -> dict:
        start_line = getattr(node, "lineno", 1)
        start_col = getattr(node, "col_offset", 0)
        end_line = getattr(node, "end_lineno", start_line)
        end_col = getattr(node, "end_col_offset", start_col)
        return {"path": self.path, "start_byte": self.offset(start_line, start_col) if self.lines else 0,
                "end_byte": self.offset(end_line, end_col) if self.lines else 0,
                "start_line": start_line, "start_column": start_col,
                "end_line": end_line, "end_column": end_col}
