from __future__ import annotations

import json
import socket
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
        print(f'ComfyUI HTTP {exc.code} error body: {body}', flush=True)
        raise RuntimeError(f'HTTP {exc.code} from {url}: {body}') from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f'Could not connect to ComfyUI at {url}: {exc.reason}') from exc
    except TimeoutError as exc:
        raise RuntimeError(f'Timed out waiting for ComfyUI at {url}') from exc
    except socket.timeout as exc:
        raise RuntimeError(f'Timed out waiting for ComfyUI at {url}') from exc


def queue_prompt(comfy_url: str, prompt: dict[str, Any], client_id: str | None = None) -> str:
    print('Sending prompt nodes:', {k: v['class_type'] for k, v in prompt.items()}, flush=True)
    for node_id, node in sorted(prompt.items(), key=lambda x: int(x[0])):
        if node['class_type'] in ('KSampler', 'UnetLoaderGGUF', 'CLIPLoader', 'VAELoader'):
            print(f'  Node {node_id} ({node["class_type"]}): {node["inputs"]}', flush=True)
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


def package_defines_node(comfy_dir: Path | None, package: str, node_type: str) -> bool:
    if comfy_dir is None:
        return False
    package_dir = comfy_dir / "custom_nodes" / package
    if not package_dir.is_dir():
        return False
    needle = f'"{node_type}"'
    alt_needle = f"name={needle}"
    try:
        for path in package_dir.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if needle in text or alt_needle in text:
                return True
    except OSError:
        return False
    return False


def ensure_node_types(comfy_url: str, required: dict[str, str], context: str = "workflow", comfy_dir: Path | None = None) -> None:
    available = object_info(comfy_url)
    missing = [node_type for node_type in required if node_type not in available]
    if not missing:
        return

    details = "; ".join(f"{node_type} ({required[node_type]})" for node_type in missing)
    packages = ", ".join(sorted(set(required[node_type] for node_type in missing)))
    install_hints = {
        "ComfyUI-LTXVideo": "https://github.com/Lightricks/ComfyUI-LTXVideo -> ComfyUI/custom_nodes/ComfyUI-LTXVideo",
        "ComfyUI-GGUF": "https://github.com/city96/ComfyUI-GGUF -> ComfyUI/custom_nodes/ComfyUI-GGUF",
        "ComfyUI-VideoHelperSuite": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite -> ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
        "ComfyUI_ProPainter_Nodes": "https://github.com/daniabib/ComfyUI_ProPainter_Nodes -> ComfyUI/custom_nodes/ComfyUI_ProPainter_Nodes",
        "ComfyUI-FlashVSR_Ultra_Fast": "https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast -> ComfyUI/custom_nodes/ComfyUI-FlashVSR_Ultra_Fast",
        "ComfyUI-Reference-Based-Video-Colorization": "https://github.com/jonstreeter/ComfyUI-Reference-Based-Video-Colorization -> ComfyUI/custom_nodes/reference-video-colorization",
        "ComfyUI-MMAudio": "https://github.com/kijai/ComfyUI-MMAudio -> ComfyUI/custom_nodes/ComfyUI-MMAudio",
    }
    hints = "; ".join(install_hints.get(package, package) for package in sorted(set(required[node_type] for node_type in missing)))
    stale_running_hints = [
        node_type
        for node_type in missing
        if package_defines_node(comfy_dir, required[node_type], node_type)
    ]
    if stale_running_hints:
        package_paths = "; ".join(
            str(comfy_dir / "custom_nodes" / package)
            for package in sorted({required[node_type] for node_type in stale_running_hints})
            if comfy_dir is not None
        )
        raise RuntimeError(
            f"ComfyUI is running at {comfy_url}, but the {context} cannot start because required node types are missing from the live server: {details}. "
            f"The configured ComfyUI folder already contains those node definitions ({package_paths}), so that custom-node package either failed to import or the server at {comfy_url} is an older/stale ComfyUI process that was not restarted after install. "
            f"Fully close every ComfyUI window/process using port 8188, then start ComfyUI from ARP again so it loads: {comfy_dir}. "
            f"If it still fails after a restart, the ComfyUI console import error for the package(s) is the root cause to fix: {packages}."
        )
    raise RuntimeError(
        f"ComfyUI is running at {comfy_url}, but the {context} cannot start because required node types are missing: {details}. "
        f"To fix: re-run install_windows.bat and choose the same ComfyUI directory when prompted. "
        f"If the folder was installed by extracting a zip download (rather than via install_windows.bat), "
        f"the install script cannot update it automatically — delete the folder(s) under ComfyUI/custom_nodes "
        f"for the missing package(s) ({packages}) and re-run install_windows.bat so it can clone the latest version. "
        f"After install, fully close and restart ComfyUI before retrying. "
        f"Repo(s): {hints}."
    )


def wait_for_prompt(comfy_url: str, prompt_id: str, poll_seconds: float, transient_timeout_seconds: float = 900.0) -> dict[str, Any]:
    transient_deadline = time.monotonic() + transient_timeout_seconds
    last_transient_error = ""
    while True:
        try:
            history = http_json('GET', f"{comfy_url.rstrip('/')}/history/{prompt_id}", timeout=30)
            transient_deadline = time.monotonic() + transient_timeout_seconds
            last_transient_error = ""
        except RuntimeError as exc:
            last_transient_error = str(exc)
            if time.monotonic() >= transient_deadline:
                raise RuntimeError(
                    f"Timed out polling ComfyUI prompt {prompt_id} after transient connection errors. "
                    f"Last error: {last_transient_error}"
                ) from exc
            print(f"Waiting for ComfyUI prompt {prompt_id}; polling temporarily failed: {last_transient_error}", flush=True)
            time.sleep(max(poll_seconds, 5.0))
            continue
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
    def file_items(value: Any):
        if isinstance(value, dict):
            if value.get("filename"):
                yield value
            for child in value.values():
                yield from file_items(child)
        elif isinstance(value, list):
            for child in value:
                yield from file_items(child)

    outputs = history_entry.get("outputs", {})
    files: list[Path] = []
    seen: set[Path] = set()
    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        for item in file_items(output):
            filename = item.get("filename")
            if not filename:
                continue
            subfolder = item.get("subfolder") or ""
            kind = str(item.get("type") or "output").lower()
            base = {"input": output_root.parent / "input", "temp": output_root.parent / "temp"}.get(kind, output_root)
            path = base / subfolder / filename
            if path not in seen:
                files.append(path)
                seen.add(path)
    return files


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

    def resolve_origin(link_id: int, seen: set[str] | None = None) -> tuple[str, int]:
        """Follow frontend-only Reroute nodes to the executable source socket."""
        link_key = int(link_id)
        if link_key not in links:
            raise ValueError(f'Workflow references missing link {link_id}')
        link = links[link_key]
        origin = str(link[1])
        origin_slot = int(link[2])
        node = nodes.get(origin)
        if not node or node.get('type') != 'Reroute':
            return origin, origin_slot
        visited = set() if seen is None else set(seen)
        if origin in visited:
            raise ValueError(f'Reroute cycle detected at node {origin}')
        visited.add(origin)
        reroute_link = next(
            (item.get('link') for item in node.get('inputs', []) if isinstance(item, dict) and item.get('link') is not None),
            None,
        )
        if reroute_link is None:
            raise ValueError(f'Reroute node {origin} has no connected input')
        return resolve_origin(int(reroute_link), visited)

    def visit(node_id: str) -> None:
        if node_id in needed:
            return
        if node_id not in nodes:
            raise ValueError(f'Output path references disabled/missing node {node_id}')
        needed.add(node_id)
        for item in nodes[node_id].get('inputs', []):
            if not isinstance(item, dict):
                continue
            link_id = item.get('link')
            if link_id is None:
                continue
            link_key = int(link_id)
            if link_key not in links:
                raise ValueError(f'Node {node_id} input "{item.get("name")}" references missing link {link_id}')
            origin, _ = resolve_origin(link_key)
            visit(origin)

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
                origin, origin_slot = resolve_origin(int(link_id))
                inputs[name] = [origin, origin_slot]
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
        elif resize_type == 'scale longer dimension' and len(values) > 1:
            out['resize_type.longer_size'] = values[1]
        elif resize_type == 'scale shorter dimension' and len(values) > 1:
            out['resize_type.shorter_size'] = values[1]
        elif resize_type == 'scale width' and len(values) > 1:
            out['resize_type.width'] = values[1]
        elif resize_type == 'scale height' and len(values) > 1:
            out['resize_type.height'] = values[1]
        elif resize_type == 'scale total pixels' and len(values) > 1:
            out['resize_type.megapixels'] = values[1]
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
        'LTXAddVideoICLoRAGuideAdvanced': ('frame_idx', 'strength', 'latent_downscale_factor', 'crop', 'use_tiled_encode', 'tile_size', 'tile_overlap', 'attention_strength'),
        'LTXICLoRALoaderModelOnly': ('lora_name', 'strength_model', 'video_only'),
        'LTXAVTextEncoderLoader': ('text_encoder', 'ckpt_name', 'device'),
        'LTXVAudioVAELoader': ('ckpt_name',),
        'LTXVLaplacianPyramidBlend': ('trim_to_shortest', 'mask_low_res_dilation'),
        'LTXVTiledVAEDecode': ('horizontal_tiles', 'vertical_tiles', 'overlap', 'last_frame_fix', 'working_device', 'working_dtype'),
        'ImagePadForOutpaintTargetSize': ('target_width', 'target_height', 'feathering', 'mask_interpolation'),
        'LoadVideo': ('file',),
        'SaveVideo': ('filename_prefix', 'format', 'codec'),
        'CreateVideo': ('fps',),
        'VAEEncodeTiled': ('tile_size', 'overlap', 'temporal_size', 'temporal_overlap'),
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

