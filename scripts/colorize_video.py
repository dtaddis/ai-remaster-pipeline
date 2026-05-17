from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import ROOT, resolve_path


def build_parser():
    parser = argparse.ArgumentParser(description='Compatibility wrapper for a ComfyUI Deep Exemplar/ColorMNet manifest runner.')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--comfy-runner', default=str(ROOT / 'tools' / 'comfyui' / 'scripts' / 'colorize_manifest_runner.py'))
    parser.add_argument('extra', nargs=argparse.REMAINDER)
    return parser


def main():
    args = build_parser().parse_args()
    runner = resolve_path(args.comfy_runner)
    if not runner.exists():
        raise FileNotFoundError(
            'Deep Exemplar rendering depends on your ComfyUI workflow/runner. '
            f'Expected {runner}. Copy or point --comfy-runner at the working runner from your ComfyUI setup.'
        )
    cmd = [sys.executable, str(runner), '--manifest', str(resolve_path(args.manifest))]
    cmd += args.extra
    return subprocess.run(cmd).returncode


if __name__ == '__main__':
    raise SystemExit(main())

