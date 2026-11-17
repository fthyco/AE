"""End-to-end and invariant checks for source-only SAE extraction."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ae.sae import extract, inspect_bundle, validate_bundle
from ae.sae.ir.validator import ValidationError, validate_ir
from ae.sae.parsing.source_spans import SourceMap


def setup(tmp_path: Path, code: bytes, filename: str = "model.py") -> Path:
    root = tmp_path / "repository"
    root.mkdir(exist_ok=True)
    (root / filename).write_bytes(code)
    return root


def run(tmp_path: Path, code: str, filename: str = "model.py") -> tuple[dict, Path, Path]:
    root = setup(tmp_path, code.encode("utf-8"), filename)
    out = tmp_path / "bundle"
    extract(root, out)
    ir = json.loads((out / "architecture.ir.json").read_bytes())
    assert validate_bundle(out)["valid"]
    return ir, root, out


def select(ir: dict, kind: str) -> list[dict]:
    return [n for n in ir["nodes"] if n["kind"] == kind]


def test_mlp_constructs_modules_and_reuses_calls(tmp_path: Path) -> None:
    ir, _, _ = run(tmp_path, '''from torch import nn
class M(nn.Module):
    def __init__(self, n):
        self.proj = nn.Linear(n, 10)
    def forward(self, x):
        a = self.proj(x)
        return self.proj(a)
''')
    sites = select(ir, "module_construction_site")
    assert len(sites) == 1
    assert sites[0]["attributes"]["declared_type"] == "torch.nn.Linear"
    assert sites[0]["attributes"]["constructor_arguments"][0] == {"kind": "symbol", "name": "n"}
    uses = [r for r in ir["edges"] if r["relation"] == "USES_MODULE_REFERENCE"]
    assert len(uses) == 2
    assert {r["target"] for r in uses} == {sites[0]["id"]}
    assert len(ir["unresolved"]) == 1  # n is not given a static value


def test_residual_addition_is_not_flattened(tmp_path: Path) -> None:
    ir, _, _ = run(tmp_path, "def forward(x, f):\n    return f(x) + x\n")
    ops = select(ir, "operation_site")
    assert any(n["attributes"]["operator"] == "Add" for n in ops)
    assert any(e["relation"] == "EXPRESSION_DEPENDS_ON" for e in ir["edges"])


def test_conditional_branch_preserved_without_claiming_execution(tmp_path: Path) -> None:
    ir, _, _ = run(tmp_path, "def f(x, flag):\n    if flag:\n        return x + 1\n    return x - 1\n")
    controls = select(ir, "control_site")
    assert len(controls) == 1
    assert controls[0]["attributes"]["control_kind"] == "if"
    assert len([n for n in select(ir, "operation_site") if n["attributes"]["operator"] in ("Add", "Sub")]) == 2


def test_unrecognized_constructor_stays_unresolved(tmp_path: Path) -> None:
    ir, _, _ = run(tmp_path, "class M:\n    def forward(self, x):\n        return self.generated(x)\n")
    assert not select(ir, "module_construction_site")
    assert any(u["kind"] == "module_attribute_resolution" for u in ir["unresolved"])


def test_unrelated_linear_class_not_mistaken_for_torch(tmp_path: Path) -> None:
    ir, _, _ = run(tmp_path, "class Linear: pass\nclass M:\n    def f(self):\n        self.x = Linear(2, 3)\n")
    assert not select(ir, "module_construction_site")


def test_imported_linear_alias_recognized(tmp_path: Path) -> None:
    ir, _, _ = run(tmp_path, "from torch.nn import Linear as Dense\nclass M:\n    def f(self):\n        self.x = Dense(2, 3)\n")
    assert len(select(ir, "module_construction_site")) == 1


def test_import_shadowing_not_recognized(tmp_path: Path) -> None:
    ir, _, _ = run(tmp_path, "from torch import nn\nnn = other_lib\nclass M:\n    def f(self):\n        self.x = nn.Linear(2, 3)\n")
    assert not select(ir, "module_construction_site")


def test_source_snapshot_byte_exact_and_comment_token(tmp_path: Path) -> None:
    code = b"# \xc2\xb5 comment\r\ndef fn(x):\r\n    return x\r\n"
    root = setup(tmp_path, code)
    out = tmp_path / "bundle"
    extract(root, out)
    assert (out / "sources/model.py").read_bytes() == code
    index = json.loads((out / "source-index.json").read_bytes())
    assert any(t["kind"] == "COMMENT" for t in index["files"][0]["tokens"])
    assert validate_bundle(out)["valid"]


def test_source_offsets_multibyte_unicode(tmp_path: Path) -> None:
    code = 'label = "λ"\n'.encode()
    source = SourceMap.from_bytes("a.py", code)
    assert source.token_offset(1, 9) == len('label = "'.encode())
    assert source.offset(1, len('label = "λ"'.encode())) == len('label = "λ"'.encode())


def test_source_offsets_utf8_bom(tmp_path: Path) -> None:
    data = b"\xef\xbb\xbf" + "x = 3\n".encode()
    source = SourceMap.from_bytes("a.py", data)
    assert source.offset(1, 0) == 3
    assert source.offset(1, 1) == 4


def test_syntax_failure_preserves_source_and_diagnostic(tmp_path: Path) -> None:
    root = setup(tmp_path, b"def invalid(:\n")
    out = tmp_path / "bundle"
    extract(root, out)
    assert (out / "sources/model.py").read_bytes() == b"def invalid(:\n"
    errors = json.loads((out / "diagnostics.json").read_bytes())["diagnostics"]
    assert errors and errors[0]["kind"] in ("syntax_error", "tokenize_error")
    assert validate_bundle(out)["valid"]


def test_every_ast_is_indexed(tmp_path: Path) -> None:
    code = "def f(x):\n    return (x + 1) * 2\n"
    ir, _, _ = run(tmp_path, code)
    assert any(n["attributes"]["ast_type"] == "Mult" for n in select(ir, "syntax_node"))
    assert any(n["attributes"]["ast_type"] == "Load" for n in select(ir, "syntax_node"))
    assert all(n["source_refs"] for n in ir["nodes"])


def test_deterministic_bytes_same_input(tmp_path: Path) -> None:
    root = setup(tmp_path, b"from torch import nn\nclass M:\n    def f(self): self.x = nn.Linear(4, 8)\n")
    a, b = tmp_path / "a", tmp_path / "b"
    extract(root, a)
    extract(root, b)
    for file in a.glob("*.json"):
        assert file.read_bytes() == (b / file.name).read_bytes()


def test_reproducible_across_different_root_paths(tmp_path: Path) -> None:
    code = b"class M:\n    def f(self, x): return x\n"
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    (one / "source.py").write_bytes(code)
    (two / "source.py").write_bytes(code)
    extract(one, tmp_path / "out1")
    extract(two, tmp_path / "out2")
    assert (tmp_path / "out1/architecture.ir.json").read_bytes() == (tmp_path / "out2/architecture.ir.json").read_bytes()


def test_invalid_edge_detected(tmp_path: Path) -> None:
    ir, _, _ = run(tmp_path, "x = 1\n")
    ir["edges"][0]["target"] = "does-not-exist"
    with pytest.raises(ValidationError, match="Dangling"):
        validate_ir(ir)


def test_snapshot_tampering_detected(tmp_path: Path) -> None:
    _, _, bundle = run(tmp_path, "x = 1\n")
    (bundle / "sources/model.py").write_text("x = 2\n")
    with pytest.raises(ValueError, match="checksum"):
        validate_bundle(bundle)


def test_scope_exclusions_are_explicit(tmp_path: Path) -> None:
    root = setup(tmp_path, b"x = 2\n")
    (root / "image.png").write_bytes(b"fake")
    (root / "ignored.py").symlink_to(root / "model.py")
    extract(root, tmp_path / "bundle")
    manifest = json.loads((tmp_path / "bundle/source-manifest.json").read_bytes())["files"]
    reasons = {f["path"]: f.get("reason") for f in manifest}
    assert reasons["image.png"] == "unsupported_extension"
    assert reasons["ignored.py"] == "symlink_file"


def test_json_schema(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    ir, _, _ = run(tmp_path, "def x(): return 1\n")
    schema = json.loads((Path(__file__).parents[1] / "schemas/architecture-ir.v1.schema.json").read_text())
    jsonschema.validate(ir, schema)


def test_cli_end_to_end(tmp_path: Path) -> None:
    root = setup(tmp_path, b"x = 1\n")
    out = tmp_path / "bundle"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    cmd = [sys.executable, "-m", "ae.cli", "extract", str(root), "--output", str(out)]
    assert subprocess.run(cmd, env=env, capture_output=True).returncode == 0
    cmd = [sys.executable, "-m", "ae.cli", "validate", str(out)]
    assert subprocess.run(cmd, env=env, capture_output=True).returncode == 0
    assert inspect_bundle(out)["coverage"]["parsed_python_files"] == 1


def test_output_dir_inside_repo_is_not_ingested(tmp_path: Path) -> None:
    root = setup(tmp_path, b"x = 1\n")
    output = root / "ae-output"
    extract(root, output)
    first = {p.name: p.read_bytes() for p in output.glob("*.json")}
    extract(root, output)
    assert first == {p.name: p.read_bytes() for p in output.glob("*.json")}
    manifest = json.loads((output / "source-manifest.json").read_bytes())
    assert not any(e.get("status") == "included" and "ae-output" in e["path"] for e in manifest["files"])
