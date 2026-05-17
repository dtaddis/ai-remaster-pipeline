from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from comfy_api import extract_output_files, node_by_id, queue_prompt, set_widget, wait_for_prompt, workflow_to_prompt
from common import ROOT, file_fingerprint, resolve_path, root_relative, signature_matches, write_signature

DEFAULT_PROMPT = 'Colorize this image.'
DEFAULT_MANIFEST_ROOT = ROOT / 'manifests' / 'references'
DEFAULT_OUTPUT_ROOT = ROOT / 'intermediate' / 'outpainted_references_color'


def read_manifest(path: Path):
    source_video = None
    rows = []
    with path.open('r', encoding='utf-8', newline='') as handle:
        comments = []
        while True:
            pos = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.startswith('#'):
                comments.append(line.strip())
                if line.startswith('# source_video='):
                    source_video = line.split('=', 1)[1].strip()
                continue
            handle.seek(pos)
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get('enabled', 'true').strip().lower() in {'false', '0', 'no', 'off'}:
                    continue
                rows.append(row)
            break
    return source_video, rows


def load_workflow(path: Path):
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def row_source(row):
    return row.get('source_reference') or row.get('reference')


def row_target(row):
    return row.get('color_reference') or row.get('reference')


def output_for_row(args, row):
    target = row_target(row)
    if not target:
        raise ValueError('Manifest row has no color_reference/reference column.')
    target_path = resolve_path(target)
    if args.output_root:
        source_path = Path(row_source(row))
        target_path = resolve_path(args.output_root) / source_path.name
    return target_path


def signature(args, workflow_path, source_path, prompt):
    return {
        'version': 1,
        'tool': 'qwen_colorize_references.py',
        'source': root_relative(source_path),
        'source_fingerprint': file_fingerprint(source_path),
        'workflow': root_relative(workflow_path),
        'workflow_fingerprint': file_fingerprint(workflow_path),
        'prompt': prompt,
        'load_image_node_id': str(args.load_image_node_id),
        'prompt_node_id': str(args.prompt_node_id) if args.prompt_node_id else None,
        'save_node_id': str(args.save_node_id),
    }


def patch_workflow(args, workflow, source_path, output_path, prompt):
    load_node = node_by_id(workflow, str(args.load_image_node_id))
    set_widget(load_node, args.load_image_widget, root_relative(source_path))
    if args.prompt_node_id:
        prompt_node = node_by_id(workflow, str(args.prompt_node_id))
        set_widget(prompt_node, args.prompt_widget, prompt)
    save_node = node_by_id(workflow, str(args.save_node_id))
    prefix = str(Path('ai_remaster_qwen') / output_path.stem).replace('\\', '/')
    set_widget(save_node, args.save_prefix_widget, prefix)
    return workflow_to_prompt(workflow, str(args.save_node_id))


def newest_output(files):
    if not files:
        raise RuntimeError('Comfy completed but no image output was found.')
    paths = [p for p in files if p.exists()]
    if not paths:
        raise RuntimeError(f'Comfy reported outputs, but none exist on disk: {files}')
    return max(paths, key=lambda p: p.stat().st_mtime_ns)


def build_parser():
    parser = argparse.ArgumentParser(description='Colorize extracted reference stills with a Qwen Image Edit ComfyUI workflow.')
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--workflow', required=True, type=Path, help='ComfyUI Qwen Image Edit workflow JSON/API file.')
    parser.add_argument('--comfy-url', default='http://127.0.0.1:8188')
    parser.add_argument('--comfy-output-root', type=Path, default=ROOT / 'tools' / 'comfyui' / 'output', help='ComfyUI output directory used to locate saved images.')
    parser.add_argument('--prompt', default=DEFAULT_PROMPT)
    parser.add_argument('--prompt-suffix', default='')
    parser.add_argument('--output-root', type=Path, help='Override manifest color_reference destinations.')
    parser.add_argument('--load-image-node-id', default='1')
    parser.add_argument('--load-image-widget', default='0', help='Widget name for API-format workflows, or widget index for normal exported workflows.')
    parser.add_argument('--prompt-node-id')
    parser.add_argument('--prompt-widget', default='0', help='Widget name for API-format workflows, or widget index for normal exported workflows.')
    parser.add_argument('--save-node-id', required=True)
    parser.add_argument('--save-prefix-widget', default='0', help='Widget name for API-format workflows, or widget index for normal exported workflows.')
    parser.add_argument('--poll-seconds', type=float, default=2.0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    return parser


def main():
    args = build_parser().parse_args()
    manifest = resolve_path(args.manifest)
    workflow_path = resolve_path(args.workflow)
    _, rows = read_manifest(manifest)
    if args.limit is not None:
        rows = rows[:args.limit]
    prompt = args.prompt if not args.prompt_suffix else f'{args.prompt} {args.prompt_suffix}'.strip()
    print(f'Manifest: {manifest}')
    print(f'Rows: {len(rows)}')
    for index, row in enumerate(rows):
        src = resolve_path(row_source(row))
        dst = output_for_row(args, row)
        sig = signature(args, workflow_path, src, prompt)
        if not args.force and signature_matches(dst, sig):
            print(f'Reuse {index:04d}: {dst}')
            continue
        print(f'Colorize {index:04d}: {src} -> {dst}')
        if args.dry_run:
            continue
        if not src.exists():
            raise FileNotFoundError(f'Reference source not found: {src}')
        workflow = load_workflow(workflow_path)
        prompt_payload = patch_workflow(args, workflow, src, dst, prompt)
        prompt_id = queue_prompt(args.comfy_url, prompt_payload)
        history = wait_for_prompt(args.comfy_url, prompt_id, args.poll_seconds)
        produced = newest_output(extract_output_files(history, resolve_path(args.comfy_output_root)))
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + '.partial')
        shutil.copy2(produced, tmp)
        tmp.replace(dst)
        write_signature(dst, sig)
        print(f'Wrote {dst}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())



