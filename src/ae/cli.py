"""Command-line entry point for the static extractor."""
from __future__ import annotations

import argparse
import json
import sys

from .sae.api import extract, inspect_bundle, validate_bundle
from .presenter import present
from .presenter.web_viewer import build_viewer
from .verification.runtime_reference import runtime_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ae", description="AE: static source-grounded model architecture extraction")
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser("extract", help="Extract from source without importing or executing it")
    extraction.add_argument("source", help="Source repository directory")
    extraction.add_argument("--output", "-o", required=True, help="Output IR bundle directory")
    extraction.add_argument("--entrypoint", help="Optional class/factory identifier (metadata hint, not executed)")
    extraction.add_argument("--input-shape", help="Concrete input dimensions, e.g. 1,3,64,64")
    extraction.add_argument("--attention-mode", choices=["fused", "unfused"],
                            help="Explicit ViT-Tiny attention implementation branch")
    extraction.add_argument("--max-files", type=int, default=2000)
    extraction.add_argument("--max-file-bytes", type=int, default=2_000_000)
    for command in ("validate", "inspect"):
        p = commands.add_parser(command)
        p.add_argument("bundle", help="IR bundle directory or architecture.ir.json")
    presenter = commands.add_parser("present", help="Draw a static architecture from a validated SAE bundle")
    presenter.add_argument("bundle", help="Existing SAE extraction bundle directory")
    presenter.add_argument("--output", "-o", required=True, help="Target diagram directory")
    presenter.add_argument("--view", choices=["structure", "flow", "nested", "instance"], default="structure")
    presenter.add_argument("--level", choices=["overview", "block", "operation"], default="block")
    presenter.add_argument("--mode", choices=["exploration", "publication"], default="exploration")
    presenter.add_argument("--format", choices=["svg", "pdf", "tikz", "all"], default="svg")
    presenter.add_argument("--focus", help="Focus a module in flow/nested or a proven instance stage such as blocks.0")
    presenter.add_argument("--theme", choices=["scientific-latex", "legacy"], default="scientific-latex")
    presenter.add_argument("--allow-partial", action="store_true", help="Allow clearly labeled partial publication export")
    viewer = commands.add_parser('viewer', help='Build a self-contained offline HTML viewer from a validated bundle')
    viewer.add_argument('bundle', help='Existing SAE extraction bundle directory')
    viewer.add_argument('--output', '-o', required=True, help='Output directory containing index.html')
    viewer.add_argument('--view', choices=['auto','instance','nested','flow','structure'], default='auto')
    viewer.add_argument('--title', default='Architecture Extractor')
    runtime = commands.add_parser('runtime-audit', help='OPT-IN: execute trusted PyTorch model as independent FX/runtime reference')
    runtime.add_argument('source', help='Trusted Python source directory. Executed: review first.')
    runtime.add_argument('--factory', required=True, help='module:zero_arg_factory returning torch.nn.Module')
    runtime.add_argument('--input-shape', required=True, action='append', help='Repeated positional synthetic input shape, e.g. 1,3,64,64')
    runtime.add_argument('--input-dtype', action='append', help='Repeated input dtype: float32,int64,etc; defaults float32')
    runtime.add_argument('--output', '-o', required=True, help='Reference JSON output directory')
    runtime.add_argument('--allow-execute', action='store_true', help='Explicit approval to import/execute target model; NOT a sandbox')
    runtime.add_argument('--max-nodes', type=int, default=20000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "extract":
            result = extract(args.source, args.output, entrypoint=args.entrypoint,
                             max_files=args.max_files, max_file_bytes=args.max_file_bytes,
                             input_shape=tuple(map(int, args.input_shape.split(","))) if args.input_shape else None,
                             attention_mode=args.attention_mode)
        elif args.command == "validate":
            result = validate_bundle(args.bundle)
        elif args.command == 'runtime-audit':
            shapes = [tuple(int(x) for x in item.split(',')) for item in args.input_shape]
            result = runtime_audit(source_root=args.source, factory=args.factory,
                input_shapes=shapes, input_dtypes=args.input_dtype,
                output=args.output, allow_execute=args.allow_execute,
                max_nodes=args.max_nodes)
        elif args.command == 'viewer':
            result = build_viewer(args.bundle, args.output, view=args.view, title=args.title)
        elif args.command == "present":
            result = present(args.bundle, args.output, level=args.level, mode=args.mode,
                             format=args.format, allow_partial=args.allow_partial, view=args.view, focus=args.focus, theme=args.theme)
        else:
            result = inspect_bundle(args.bundle)
        print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
        if args.command == 'viewer':
            return 0 if result['rendered_levels'] else 2
        if args.command == 'runtime-audit':
            return 0 if result['status'] == 'traced' else 2
        return 0 if args.command != "present" or result["render_allowed"] else 2
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ae: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
