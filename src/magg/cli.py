"""Command-line entry point for the static extractor."""
from __future__ import annotations

import argparse
import json
import sys

from .sae.api import extract, inspect_bundle, validate_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="magg", description="MAGG: static source-grounded model architecture extraction")
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser("extract", help="Extract from source without importing or executing it")
    extraction.add_argument("source", help="Source repository directory")
    extraction.add_argument("--output", "-o", required=True, help="Output IR bundle directory")
    extraction.add_argument("--entrypoint", help="Optional class/factory identifier (metadata hint, not executed)")
    extraction.add_argument("--max-files", type=int, default=2000)
    extraction.add_argument("--max-file-bytes", type=int, default=2_000_000)
    for command in ("validate", "inspect"):
        p = commands.add_parser(command)
        p.add_argument("bundle", help="IR bundle directory or architecture.ir.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "extract":
            result = extract(args.source, args.output, entrypoint=args.entrypoint,
                             max_files=args.max_files, max_file_bytes=args.max_file_bytes)
        elif args.command == "validate":
            result = validate_bundle(args.bundle)
        else:
            result = inspect_bundle(args.bundle)
        print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"magg: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
