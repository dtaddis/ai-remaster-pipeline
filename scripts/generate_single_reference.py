from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import qwen_colorize_references as qwen
from common import ROOT, resolve_path, root_relative


def write_temp_manifest(source: Path, output: Path) -> Path:
    temp_dir = ROOT / 'manifests' / '_single_reference_tmp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='', suffix='.csv', prefix='single_reference_', dir=temp_dir, delete=False)
    with handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['enabled', 'end', 'source_reference', 'color_reference', 'prompt'])
        writer.writerow(['true', '0:00:01', root_relative(source), root_relative(output), ''])
    return Path(handle.name)


def build_parser() -> argparse.ArgumentParser:
    parser = qwen.build_parser()
    parser.description = 'Colorize one reference image with Qwen Image Edit.'
    parser.add_argument('--source-image', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    for action in parser._actions:
        if action.dest == 'manifest':
            action.required = False
        if action.dest == 'limit':
            action.default = 1
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    source = resolve_path(args.source_image)
    output = resolve_path(args.output)
    manifest = write_temp_manifest(source, output)
    args.manifest = manifest
    args.limit = 1
    return qwen.main_with_args(args) if hasattr(qwen, 'main_with_args') else _run_with_args(args)


def _run_with_args(args: argparse.Namespace) -> int:
    # Keep this wrapper resilient even if qwen_colorize_references.py is run as a standalone script.
    import sys
    old_argv = sys.argv[:]
    argv = ['qwen_colorize_references.py']
    for key, value in vars(args).items():
        if key in {'source_image', 'output'} or value in (None, False):
            continue
        option = '--' + key.replace('_', '-')
        if value is True:
            argv.append(option)
        elif isinstance(value, list):
            for item in value:
                argv.extend([option, str(item)])
        else:
            argv.extend([option, str(value)])
    try:
        sys.argv = argv
        return qwen.main()
    finally:
        sys.argv = old_argv


if __name__ == '__main__':
    raise SystemExit(main())
