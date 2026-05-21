from __future__ import annotations

import argparse
import base64
import copy
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

from comfy_api import extract_output_files, node_by_id, queue_prompt, set_widget, wait_for_comfy, wait_for_prompt, workflow_to_prompt, http_json
from common import ROOT, file_fingerprint, resolve_path, root_relative, resumable_output, write_signature
from dependency_manager import ensure_qwen_image_edit_models

DEFAULT_PROMPT = (
    'Colorize this image. Preserve the drawing and composition. '
    'Use clean modern cartoon colours, not sepia. Do not add text or new objects.'
)
DEFAULT_PROMPT_SUFFIX = (
    'Keep black ink deep, whites clean, and props/backgrounds naturally coloured.'
)
DEFAULT_OUTPUT_ROOT = ROOT / 'intermediate' / 'outpainted_references_color'
DEFAULT_OLLAMA_URL = 'http://127.0.0.1:11434'
DEFAULT_OLLAMA_VISION_MODEL = 'qwen2.5vl:7b'
REFERENCE_DESCRIPTION_PROMPT = (
    'In one short sentence, describe only the reusable colour palette of this already-colourised film frame. '
    'Mention skin, clothing, walls/wood/fabric/metal, and lighting only if clearly visible. '
    'No markdown, no headings, no bullets, no film-history commentary, no composition summary. Maximum 35 words.'
)


def load_local_config() -> dict[str, str]:
    path = ROOT / '.ai_remaster_config.json'
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
    except json.JSONDecodeError:
        return {}
    return {str(key): str(value) for key, value in data.items() if value is not None}


def read_manifest(path: Path):
    source_video = None
    rows = []
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        while True:
            pos = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.startswith('#'):
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
    with path.open('r', encoding='utf-8-sig') as handle:
        return json.load(handle)


def row_source(row: dict[str, str]) -> str:
    return row.get('source_reference') or row.get('reference') or ''


def row_target(row: dict[str, str]) -> str:
    return row.get('color_reference') or row.get('reference') or ''


def output_for_row(args: argparse.Namespace, row: dict[str, str]) -> Path:
    target = row_target(row)
    if not target:
        raise ValueError('Manifest row has no color_reference/reference column.')
    target_path = resolve_path(target)
    if args.output_root:
        source_path = Path(row_source(row))
        target_path = resolve_path(args.output_root) / source_path.name
    return target_path


def copy_to_comfy_input(path: Path, comfy_dir: Path, subfolder: str = 'arp_qwen_refs') -> str:
    target_dir = comfy_dir / 'input' / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / path.name
    if not target.exists() or target.stat().st_size != path.stat().st_size:
        shutil.copy2(path, target)
    return str(Path(subfolder) / target.name).replace('\\', '/')


def read_text_file(path: Path | None) -> str:
    if not path:
        return ''
    return resolve_path(path).read_text(encoding='utf-8-sig').strip()


def describe_with_ollama(args: argparse.Namespace, image_path: Path) -> str:
    image_bytes = image_path.read_bytes()
    payload = {
        'model': args.ollama_vision_model,
        'prompt': args.reference_description_prompt,
        'images': [base64.b64encode(image_bytes).decode('ascii')],
        'stream': False,
    }
    response = http_json('POST', f"{args.ollama_url.rstrip('/')}/api/generate", payload, timeout=180)
    return str(response.get('response') or '').strip()


def description_cache_path(args: argparse.Namespace, image_path: Path) -> Path:
    cache_dir = image_path.parent / '_reference_descriptions'
    cache_dir.mkdir(parents=True, exist_ok=True)
    provider = args.reference_description_provider.replace('-', '_')
    model = args.ollama_vision_model.replace(':', '_').replace('/', '_')
    return cache_dir / f'{image_path.stem}.{provider}.{model}.txt'


def describe_reference(args: argparse.Namespace, image_path: Path) -> str:
    cache_path = description_cache_path(args, image_path)
    if not args.force_reference_descriptions and cache_path.exists():
        return cache_path.read_text(encoding='utf-8-sig').strip()
    print(f'Describe colour reference with {args.ollama_vision_model}: {image_path}')
    if args.reference_description_provider == 'ollama':
        text = describe_with_ollama(args, image_path)
    else:
        raise RuntimeError(f'Unknown reference description provider: {args.reference_description_provider}')
    cache_path.write_text(text + '\n', encoding='utf-8')
    return text


def continuity_reference_paths(args: argparse.Namespace, rows: list[dict[str, str]], row_index: int) -> list[Path]:
    paths: list[Path] = []
    for manual in args.reference:
        path = resolve_path(manual)
        if path.exists():
            paths.append(path)
    if args.continuity_reference_count <= 0:
        return paths[: args.continuity_reference_count or len(paths)]
    for previous in reversed(rows[:row_index]):
        if len(paths) >= args.continuity_reference_count:
            break
        candidate = output_for_row(args, previous)
        if candidate.exists() and candidate not in paths:
            paths.append(candidate)
    return paths[: args.continuity_reference_count]


def compact_reference_description(text: str, max_chars: int) -> str:
    text = re.sub(r'(?m)^#+\s*', '', text)
    text = re.sub(r'(?m)^\s*[-*]\s*', '', text)
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(?i)\b(colou?r palette and restoration style|colou?r palette|restoration style)\s*:?', '', text).strip()
    text = re.sub(
        r'(?i)\b(skin tones?|clothing colou?rs?|wall/fabric/metal colou?rs?|lighting temperature|shadows?|overall grade)\s*:\s*',
        '',
        text,
    )
    text = re.sub(r'\bThis description should help.*$', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\bThis palette and style should be applied.*$', '', text, flags=re.IGNORECASE).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(' ', 1)[0].rstrip(' ,;:')
    return cut + '.'


def continuity_description(args: argparse.Namespace, refs: list[Path]) -> str:
    if args.reference_description_provider == 'none' or not refs:
        return ''
    lines = ['Local colour descriptions from already-colourised nearby reference frames:']
    for index, ref in enumerate(refs, start=1):
        description = compact_reference_description(describe_reference(args, ref), args.reference_description_max_chars)
        if description:
            lines.append(f'{index}. {description}')
    return '\n'.join(lines) if len(lines) > 1 else ''


def build_prompt(args: argparse.Namespace, extra_description: str = '', row_prompt: str = '') -> str:
    parts = [args.prompt.strip()]
    suffix = (args.prompt_suffix or '').strip()
    static_description = read_text_file(args.reference_description_file)
    if args.reference_description:
        static_description = (static_description + '\n' + args.reference_description).strip() if static_description else args.reference_description.strip()
    if suffix:
        parts.append(suffix)
    if static_description:
        parts.append(static_description)
    if extra_description:
        parts.append(extra_description.strip())
    if row_prompt:
        parts.append(row_prompt.strip())
    # The user's one-off direction belongs last so it is not buried under generated palette text.
    if args.add_prompt:
        parts.append(args.add_prompt.strip())
    return ' '.join(part for part in parts if part).strip()


def signature(args: argparse.Namespace, workflow_path: Path, source_path: Path, prompt: str) -> dict[str, Any]:
    return {
        'version': 2,
        'tool': 'qwen_colorize_references.py',
        'source': root_relative(source_path),
        'source_fingerprint': file_fingerprint(source_path),
        'workflow': root_relative(workflow_path),
        'workflow_fingerprint': file_fingerprint(workflow_path),
        'prompt': prompt,
        'load_image_node_id': str(args.load_image_node_id),
        'prompt_node_id': str(args.prompt_node_id) if args.prompt_node_id else None,
        'save_node_id': str(args.save_node_id),
        'reference_description_provider': args.reference_description_provider,
        'continuity_reference_count': args.continuity_reference_count,
        'reference_description_max_chars': args.reference_description_max_chars,
    }


def patch_workflow(args: argparse.Namespace, workflow: dict[str, Any], source_path: Path, output_path: Path, prompt: str) -> dict[str, Any]:
    if has_frontend_subgraphs(workflow):
        return subgraph_workflow_to_prompt(args, workflow, source_path, output_path, prompt)
    comfy_image = copy_to_comfy_input(source_path, resolve_path(args.comfy_dir))
    load_id = resolve_node_id(workflow, args.load_image_node_id, {'LoadImage'})
    save_id = resolve_node_id(workflow, args.save_node_id, {'SaveImage'})
    prompt_id = resolve_node_id(workflow, args.prompt_node_id, {'TextEncodeQwenImageEditPlus', 'CLIPTextEncode'}, prefer_title='positive') if args.prompt_node_id else ''
    load_node = node_by_id(workflow, load_id)
    set_widget(load_node, args.load_image_widget, comfy_image)
    if prompt_id:
        prompt_node = node_by_id(workflow, prompt_id)
        set_widget(prompt_node, args.prompt_widget, prompt)
    save_node = node_by_id(workflow, save_id)
    prefix = str(Path('ai_remaster_qwen') / output_path.stem).replace('\\', '/')
    set_widget(save_node, args.save_prefix_widget, prefix)
    return workflow_to_prompt(workflow, save_id)


def has_frontend_subgraphs(workflow: dict[str, Any]) -> bool:
    definitions = workflow.get('definitions')
    if not isinstance(definitions, dict):
        return False
    subgraphs = definitions.get('subgraphs')
    return isinstance(subgraphs, list) and bool(subgraphs)


def subgraph_workflow_to_prompt(args: argparse.Namespace, workflow: dict[str, Any], source_path: Path, output_path: Path, prompt: str) -> dict[str, Any]:
    subgraphs = workflow.get('definitions', {}).get('subgraphs') or []
    if not subgraphs:
        raise RuntimeError('Workflow does not contain a frontend subgraph definition.')
    subgraph = copy.deepcopy(subgraphs[0])
    nodes = subgraph.get('nodes') or []
    links = subgraph.get('links') or []
    load_id = 90001
    save_id = 90002
    comfy_image = copy_to_comfy_input(source_path, resolve_path(args.comfy_dir))
    link_lookup = {int(link.get('id')): link for link in links if isinstance(link, dict) and 'id' in link}

    for node in nodes:
        if not isinstance(node, dict):
            continue
        for item in node.get('inputs') or []:
            link_id = item.get('link')
            if link_id is None:
                continue
            link = link_lookup.get(int(link_id))
            if not link or int(link.get('origin_id', 0)) != -10:
                continue
            slot = int(link.get('origin_slot', -1))
            if slot == 0:
                item['link'] = 900001
            elif slot in {1, 2}:
                item['link'] = None
            elif slot == 3:
                item['link'] = None
                item['widget'] = {'name': item.get('name', 'prompt')}
                set_workflow_widget(node, item.get('name', 'prompt'), prompt)
            elif slot == 4:
                item['link'] = None
                item['widget'] = {'name': item.get('name', 'value')}
                set_workflow_widget(node, item.get('name', 'value'), True)
            elif slot == 5:
                item['link'] = None
                item['widget'] = {'name': item.get('name', 'unet_name')}
                set_workflow_widget(node, item.get('name', 'unet_name'), args.gguf_model if args.model_backend == 'gguf' else 'qwen_image_edit_2511_bf16.safetensors')
            elif slot == 6:
                item['link'] = None
                item['widget'] = {'name': item.get('name', 'clip_name')}
                set_workflow_widget(node, item.get('name', 'clip_name'), 'qwen_2.5_vl_7b_fp8_scaled.safetensors')
            elif slot == 7:
                item['link'] = None
                item['widget'] = {'name': item.get('name', 'vae_name')}
                set_workflow_widget(node, item.get('name', 'vae_name'), 'qwen_image_vae.safetensors')
            elif slot == 8:
                item['link'] = None
                item['widget'] = {'name': item.get('name', 'lora_name')}
                set_workflow_widget(node, item.get('name', 'lora_name'), 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors')

    nodes.append(
        {
            'id': load_id,
            'type': 'LoadImage',
            'inputs': [{'name': 'image', 'type': 'COMBO', 'widget': {'name': 'image'}}],
            'outputs': [{'name': 'IMAGE', 'type': 'IMAGE', 'links': [900001]}],
            'widgets_values': [comfy_image],
        }
    )
    links.append({'id': 900001, 'origin_id': load_id, 'origin_slot': 0, 'target_id': 160, 'target_slot': 0, 'type': 'IMAGE'})

    output_link = next((link for link in links if isinstance(link, dict) and int(link.get('target_id', 0)) == -20), None)
    if not output_link:
        raise RuntimeError('Could not find subgraph output image link.')
    output_link['target_id'] = save_id
    output_link['target_slot'] = 0
    prefix = str(Path('ai_remaster_qwen') / output_path.stem).replace('\\', '/')
    nodes.append(
        {
            'id': save_id,
            'type': 'SaveImage',
            'inputs': [{'name': 'images', 'type': 'IMAGE', 'link': int(output_link['id'])}, {'name': 'filename_prefix', 'type': 'STRING', 'widget': {'name': 'filename_prefix'}}],
            'widgets_values': [prefix],
        }
    )
    flat = {'nodes': nodes, 'links': [link_to_list(link) for link in links]}
    return workflow_to_prompt(flat, save_id)


def link_to_list(link: Any) -> Any:
    if not isinstance(link, dict):
        return link
    return [
        link.get('id'),
        link.get('origin_id'),
        link.get('origin_slot', 0),
        link.get('target_id'),
        link.get('target_slot', 0),
        link.get('type', '*'),
    ]


def set_workflow_widget(node: dict[str, Any], input_name: str, value: Any) -> None:
    widgets = node.setdefault('widgets_values', [])
    if isinstance(widgets, dict):
        widgets[input_name] = value
        return
    if not isinstance(widgets, list):
        widgets = [widgets]
        node['widgets_values'] = widgets
    widget_index = 0
    for item in node.get('inputs') or []:
        if 'widget' not in item:
            continue
        if item.get('name') == input_name or item.get('widget', {}).get('name') == input_name:
            while len(widgets) <= widget_index:
                widgets.append(None)
            widgets[widget_index] = value
            return
        widget_index += 1
    widgets.append(value)


def resolve_node_id(workflow: dict[str, Any], value: str | None, class_types: set[str], prefer_title: str = '') -> str:
    if value and str(value).lower() != 'auto':
        return str(value)
    matches = [node for node in iter_workflow_nodes(workflow) if (node.get('class_type') or node.get('type')) in class_types]
    if prefer_title:
        titled = [node for node in matches if prefer_title.lower() in str(node.get('title') or '').lower()]
        if titled:
            matches = titled
    if not matches:
        raise RuntimeError(f"Could not auto-detect workflow node for: {', '.join(sorted(class_types))}")
    return str(matches[0].get('id'))


def newest_output(files: list[Path]) -> Path:
    if not files:
        raise RuntimeError('Comfy completed but no image output was found.')
    paths = [p for p in files if p.exists()]
    if not paths:
        raise RuntimeError(f'Comfy reported outputs, but none exist on disk: {files}')
    return max(paths, key=lambda p: p.stat().st_mtime_ns)


def iter_workflow_nodes(workflow: dict[str, Any]):
    seen: set[int] = set()

    def visit(value: Any):
        if isinstance(value, dict):
            ident = id(value)
            if ident in seen:
                return
            seen.add(ident)
            if 'class_type' in value or 'type' in value:
                yield value
            for child in value.values():
                yield from visit(child)
        elif isinstance(value, list):
            for child in value:
                yield from visit(child)

    yield from visit(workflow)


def node_widget_values(node: dict[str, Any]) -> list[Any]:
    inputs = node.get('inputs', {})
    if isinstance(inputs, dict):
        input_values = list(inputs.values())
    else:
        input_values = []
    values = node.get('widgets_values', [])
    if isinstance(values, list):
        return input_values + values
    if isinstance(values, dict):
        return input_values + list(values.values())
    return input_values + [values]


def patch_qwen_model_backend(args, workflow: dict[str, Any]) -> None:
    if args.model_backend != 'gguf':
        return
    patched = 0
    for node in iter_workflow_nodes(workflow):
        class_type = node.get('class_type') or node.get('type')
        if class_type not in {'UNETLoader', 'UnetLoader'}:
            continue
        values = ' '.join(str(value).lower() for value in node_widget_values(node))
        title = str(node.get('title') or '').lower()
        if 'qwen' not in values and 'qwen' not in title:
            continue
        if 'class_type' in node:
            node['class_type'] = 'UnetLoaderGGUF'
            node.setdefault('inputs', {})['unet_name'] = args.gguf_model
        else:
            node['type'] = 'UnetLoaderGGUF'
            node['title'] = 'Unet Loader (GGUF)'
            node['inputs'] = [{'name': 'unet_name', 'type': 'COMBO', 'widget': {'name': 'unet_name'}}]
            node['widgets_values'] = [args.gguf_model]
        patched += 1
    if not patched:
        raise RuntimeError('Could not find a Qwen UNETLoader node to patch to UnetLoaderGGUF. Use a Qwen Image Edit workflow with a visible UNETLoader node, or run with --model-backend safetensors.')


def build_parser() -> argparse.ArgumentParser:
    config = load_local_config()
    parser = argparse.ArgumentParser(description='Colorize extracted reference stills with a single-image Qwen Image Edit ComfyUI workflow.')
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--workflow', required=True, type=Path, help='ComfyUI Qwen Image Edit workflow JSON/API file.')
    parser.add_argument('--comfy-url', default='http://127.0.0.1:8188')
    parser.add_argument('--comfy-dir', type=Path, default=Path(config.get('comfy_dir', ROOT / 'tools' / 'comfyui')), help='ComfyUI directory used for on-demand model downloads.')
    parser.add_argument('--comfy-output-root', type=Path, default=ROOT / 'tools' / 'comfyui' / 'output', help='ComfyUI output directory used to locate saved images.')
    parser.add_argument('--model-backend', choices=['gguf', 'safetensors'], default='gguf')
    parser.add_argument('--gguf-model', default='qwen-image-edit-2511-Q4_K_M.gguf')
    parser.add_argument('--prompt', default=DEFAULT_PROMPT)
    parser.add_argument('--prompt-suffix', default=DEFAULT_PROMPT_SUFFIX)
    parser.add_argument('--add-prompt', default='', help='Extra one-off guidance appended last, after generated reference descriptions.')
    parser.add_argument('--reference-description', default='', help='Static palette/continuity guidance appended to the image edit prompt.')
    parser.add_argument('--reference-description-file', type=Path, help='UTF-8 text file with static palette/continuity guidance.')
    parser.add_argument('--reference', action='append', default=[], help='Manual colour reference image to describe as text; can be repeated.')
    parser.add_argument('--reference-description-provider', choices=['none', 'ollama'], default='none', help='Describe continuity/reference images as text instead of passing multiple images to Qwen.')
    parser.add_argument('--reference-description-prompt', default=REFERENCE_DESCRIPTION_PROMPT)
    parser.add_argument('--ollama-url', default=DEFAULT_OLLAMA_URL)
    parser.add_argument('--ollama-vision-model', default=DEFAULT_OLLAMA_VISION_MODEL)
    parser.add_argument('--force-reference-descriptions', action='store_true')
    parser.add_argument('--reference-description-max-chars', type=int, default=220, help='Clamp each local continuity description before appending it to the Qwen prompt.')
    parser.add_argument('--continuity-reference-count', type=int, default=0, help='Describe this many previous colour references and append them to the prompt. 0 disables automatic continuity descriptions.')
    parser.add_argument('--output-root', type=Path, help='Override manifest color_reference destinations.')
    parser.add_argument('--load-image-node-id', default='auto')
    parser.add_argument('--load-image-widget', default='0', help='Widget name for API-format workflows, or widget index for normal exported workflows.')
    parser.add_argument('--prompt-node-id')
    parser.add_argument('--prompt-widget', default='0', help='Widget name for API-format workflows, or widget index for normal exported workflows.')
    parser.add_argument('--save-node-id', default='auto')
    parser.add_argument('--save-prefix-widget', default='0', help='Widget name for API-format workflows, or widget index for normal exported workflows.')
    parser.add_argument('--poll-seconds', type=float, default=2.0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--print-final-prompt', action='store_true')
    parser.add_argument('--print-api-prompt', action='store_true')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.continuity_reference_count < 0:
        raise RuntimeError('--continuity-reference-count must be zero or greater')
    manifest = resolve_path(args.manifest)
    workflow_path = resolve_path(args.workflow)
    comfy_dir = resolve_path(args.comfy_dir)
    _, rows = read_manifest(manifest)
    if args.limit is not None:
        rows = rows[:args.limit]
    print(f'Manifest: {manifest}')
    print(f'Rows: {len(rows)}')
    print(f'Qwen mode: {args.model_backend}; one source image only; extra references are converted to text guidance when enabled.')
    if not args.dry_run and rows:
        if args.model_backend == 'gguf' and not (comfy_dir / 'custom_nodes' / 'ComfyUI-GGUF').exists():
            raise FileNotFoundError(f'ComfyUI-GGUF is required for Qwen GGUF. Re-run install_windows.bat, then restart ComfyUI: {comfy_dir / "custom_nodes" / "ComfyUI-GGUF"}')
        ensure_qwen_image_edit_models(comfy_dir)
        print(f'Waiting for ComfyUI at {args.comfy_url}...')
        wait_for_comfy(args.comfy_url, timeout_seconds=180, poll_seconds=args.poll_seconds)
    for index, row in enumerate(rows):
        src = resolve_path(row_source(row))
        dst = output_for_row(args, row)
        refs = continuity_reference_paths(args, rows, index)
        extra_description = continuity_description(args, refs)
        prompt = build_prompt(args, extra_description, row.get('prompt') or row.get('custom_prompt') or '')
        sig = signature(args, workflow_path, src, prompt)
        if args.print_final_prompt:
            print(f'Final prompt {index:04d}: {prompt}')
        if not args.force and resumable_output(dst, sig, image_like=src):
            print(f'Reuse {index:04d}: {dst}')
            continue
        ref_note = f', described refs={len(refs)}' if refs else ''
        print(f'Colorize {index:04d}: {src} -> {dst}{ref_note}')
        print(f'Qwen prompt {index:04d}: {prompt}')
        workflow = load_workflow(workflow_path)
        patch_qwen_model_backend(args, workflow)
        if args.dry_run:
            prompt_payload = patch_workflow(args, workflow, src, dst, prompt)
            if args.print_api_prompt:
                print(json.dumps(prompt_payload, indent=2))
            continue
        if not src.exists():
            raise FileNotFoundError(f'Reference source not found: {src}')
        prompt_payload = patch_workflow(args, workflow, src, dst, prompt)
        if args.print_api_prompt:
            print(json.dumps(prompt_payload, indent=2))
        prompt_id = queue_prompt(args.comfy_url, prompt_payload)
        print(f'Queued ComfyUI prompt: {prompt_id}', flush=True)
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


