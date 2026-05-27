from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(url, data=data, method=method, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'HTTP {exc.code} from {url}: {body}') from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f'Could not connect to ComfyUI at {url}: {exc.reason}') from exc


def queue_prompt(comfy_url: str, prompt: dict[str, Any], client_id: str | None = None) -> str:
    response = http_json('POST', f"{comfy_url.rstrip('/')}/prompt", {'prompt': prompt, 'client_id': client_id or str(uuid.uuid4())})
    prompt_id = response.get('prompt_id')
    if not prompt_id:
        raise RuntimeError(f'ComfyUI did not return prompt_id: {response}')
    return str(prompt_id)


def wait_for_comfy(comfy_url: str, timeout_seconds: float = 180.0, poll_seconds: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            http_json('GET', f"{comfy_url.rstrip('/')}/queue", timeout=5)
            return
        except RuntimeError as exc:
            last_error = str(exc)
            time.sleep(poll_seconds)
    raise RuntimeError(f"ComfyUI did not become ready at {comfy_url} within {timeout_seconds:.0f}s. Last error: {last_error}")


def object_info(comfy_url: str) -> dict[str, Any]:
    return http_json('GET', f"{comfy_url.rstrip('/')}/object_info", timeout=30)


def ensure_node_types(comfy_url: str, required: dict[str, str], context: str = "workflow") -> None:
    available = object_info(comfy_url)
    missing = [node_type for node_type in required if node_type not in available]
    if not missing:
        return

    details = "; ".join(f"{node_type} ({required[node_type]})" for node_type in missing)
    packages = ", ".join(sorted(set(required[node_type] for node_type in missing)))
    raise RuntimeError(
        f"ComfyUI is running at {comfy_url}, but the {context} cannot start because required node types are missing: {details}. "
        f"Re-run install_windows.bat, choose the same ComfyUI directory, then fully close and restart ComfyUI. "
        f"If you use your own ComfyUI checkout, install or update: {packages}."
    )


def wait_for_prompt(comfy_url: str, prompt_id: str, poll_seconds: float) -> dict[str, Any]:
    while True:
        history = http_json('GET', f"{comfy_url.rstrip('/')}/history/{prompt_id}", timeout=30)
        entry = history.get(prompt_id)
        if entry:
            status = entry.get('status', {})
            if status.get('completed'):
                return entry
            if status.get('status_str') == 'error':
                messages = status.get('messages') or []
                raise RuntimeError(json.dumps(messages[-1] if messages else status, ensure_ascii=False))
            for message in status.get('messages') or []:
                if isinstance(message, list) and message and message[0] == 'execution_error':
                    raise RuntimeError(json.dumps(message[1], ensure_ascii=False))
        time.sleep(poll_seconds)


def extract_output_files(history_entry: dict[str, Any], output_root: Path) -> list[Path]:
    outputs = history_entry.get('outputs', {})
    files: list[Path] = []
    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        for key in ('images', 'videos', 'gifs'):
            for item in output.get(key, []):
                filename = item.get('filename')
                if not filename:
                    continue
                subfolder = item.get('subfolder') or ''
                files.append(output_root / subfolder / filename)
    return [path for path in files if path.exists()]


def node_by_id(workflow: dict[str, Any], node_id: str) -> dict[str, Any]:
    if 'nodes' in workflow:
        for node in workflow.get('nodes', []):
            if str(node.get('id')) == str(node_id):
                return node
    if str(node_id) in workflow and isinstance(workflow[str(node_id)], dict):
        return workflow[str(node_id)]
    for value in workflow.values():
        found = node_by_id_nested(value, node_id)
        if found is not None:
            return found
    raise KeyError(f'Workflow node not found: {node_id}')


def node_by_id_nested(value: Any, node_id: str) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if ('type' in value or 'class_type' in value) and str(value.get('id')) == str(node_id):
            return value
        for child in value.values():
            found = node_by_id_nested(child, node_id)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = node_by_id_nested(child, node_id)
            if found is not None:
                return found
    return None


def set_widget(node: dict[str, Any], key: str | int, value: Any) -> None:
    if 'class_type' in node and 'inputs' in node and 'widgets_values' not in node:
        node.setdefault('inputs', {})[str(key)] = value
        return
    widgets = node.setdefault('widgets_values', {})
    if isinstance(widgets, dict):
        widgets[str(key)] = value
        return
    if not isinstance(widgets, list):
        widgets = [widgets]
        node['widgets_values'] = widgets
    index = int(key)
    while len(widgets) <= index:
        widgets.append(None)
    widgets[index] = value


def workflow_to_prompt(workflow: dict[str, Any], output_node_id: str) -> dict[str, Any]:
    if 'nodes' not in workflow:
        return workflow
    nodes = {str(node['id']): node for node in workflow['nodes'] if int(node.get('mode', 0)) != 4}
    links = {int(link[0]): link for link in workflow.get('links', [])}
    needed: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in needed:
            return
        if node_id not in nodes:
            raise ValueError(f'Output path references disabled/missing node {node_id}')
        needed.add(node_id)
        for item in nodes[node_id].get('inputs', []):
            link_id = item.get('link')
            if link_id is not None:
                visit(str(links[int(link_id)][1]))

    visit(str(output_node_id))
    prompt: dict[str, Any] = {}
    for node_id in sorted(needed, key=lambda value: int(value)):
        node = nodes[node_id]
        inputs: dict[str, Any] = {}
        widget_values = node.get('widgets_values', [])
        widget_index = 0
        for item in node.get('inputs', []):
            name = item['name']
            link_id = item.get('link')
            has_widget = 'widget' in item
            if link_id is not None:
                link = links[int(link_id)]
                inputs[name] = [str(link[1]), int(link[2])]
            elif has_widget:
                if isinstance(widget_values, dict):
                    if name in widget_values:
                        inputs[name] = widget_values[name]
                else:
                    values = widget_values if isinstance(widget_values, list) else [widget_values]
                    if widget_index < len(values):
                        inputs[name] = values[widget_index]
            if has_widget:
                widget_index += 1
        if isinstance(widget_values, dict):
            for key, value in widget_values.items():
                if key not in inputs and not isinstance(value, dict):
                    inputs[key] = value
        elif not any('widget' in item for item in node.get('inputs', [])):
            values = widget_values if isinstance(widget_values, list) else [widget_values]
            fallback_names = {
                'CheckpointLoaderSimple': ('ckpt_name',),
                'LoadImage': ('image',),
                'ManualSigmas': ('sigmas',),
                'PrimitiveBoolean': ('value',),
                'PrimitiveInt': ('value',),
                'PrimitiveFloat': ('value',),
                'PrimitiveString': ('value',),
                'KSamplerSelect': ('sampler_name',),
            }.get(node.get('type'), ())
            for name, value in zip(fallback_names, values):
                if name not in inputs:
                    inputs[name] = value
        if not isinstance(widget_values, dict):
            for name, value in widget_fallback_inputs(node.get('type'), widget_values).items():
                if name not in inputs:
                    inputs[name] = value
        prompt[node_id] = {'class_type': node['type'], 'inputs': inputs}
        if node.get('title'):
            prompt[node_id]['_meta'] = {'title': node['title']}
    return prompt


def widget_fallback_inputs(class_type: str | None, widget_values: Any) -> dict[str, Any]:
    values = widget_values if isinstance(widget_values, list) else [widget_values]
    if not class_type or not values:
        return {}
    if class_type == 'ImagePadKJ':
        return dict(zip(('left', 'right', 'top', 'bottom', 'extra_padding', 'pad_mode', 'color'), values))
    if class_type == 'ResizeImageMaskNode':
        resize_type = str(values[0]) if values else 'scale by multiplier'
        out: dict[str, Any] = {'resize_type': resize_type}
        if resize_type == 'scale by multiplier' and len(values) > 1:
            out['resize_type.multiplier'] = values[1]
        elif resize_type == 'scale to multiple' and len(values) > 1:
            out['resize_type.multiple'] = values[1]
        elif resize_type == 'match size' and len(values) > 1:
            out['resize_type.crop'] = values[1]
        elif resize_type == 'scale dimensions':
            if len(values) > 1:
                out['resize_type.width'] = values[1]
            if len(values) > 2:
                out['resize_type.height'] = values[2]
            if len(values) > 3:
                out['resize_type.crop'] = values[3]
        if values:
            out['scale_method'] = values[-1]
        return out
    simple_maps = {
        'LTXVPreprocess': ('img_compression',),
        'EmptyLTXVLatentVideo': ('width', 'height', 'length', 'batch_size'),
        'LTXVImgToVideoConditionOnly': ('strength', 'bypass'),
        'CLIPTextEncode': ('text',),
        'LTXAddVideoICLoRAGuide': ('frame_idx', 'strength', 'latent_downscale_factor', 'crop', 'use_tiled_encode', 'tile_size', 'tile_overlap'),
        'LTXVEmptyLatentAudio': ('frames_number', 'frame_rate', 'batch_size'),
        'RandomNoise': ('noise_seed', 'control_after_generate'),
        'CFGGuider': ('cfg',),
        'VAEDecodeTiled': ('tile_size', 'overlap', 'temporal_size', 'temporal_overlap'),
        'ModelSamplingAuraFlow': ('shift',),
        'CFGNorm': ('strength',),
        'FluxKontextMultiReferenceLatentMethod': ('reference_latents_method',),
        'LoraLoaderModelOnly': ('lora_name', 'strength_model'),
        'CLIPLoader': ('clip_name', 'type', 'device'),
        'KSampler': ('seed', 'control_after_generate', 'steps', 'cfg', 'sampler_name', 'scheduler', 'denoise'),
        'TextEncodeQwenImageEditPlus': ('prompt',),
    }
    names = simple_maps.get(class_type)
    return dict(zip(names, values)) if names else {}

