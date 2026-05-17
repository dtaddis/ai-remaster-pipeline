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


def wait_for_prompt(comfy_url: str, prompt_id: str, poll_seconds: float) -> dict[str, Any]:
    while True:
        history = http_json('GET', f"{comfy_url.rstrip('/')}/history/{prompt_id}", timeout=30)
        entry = history.get(prompt_id)
        if entry:
            status = entry.get('status', {})
            if status.get('completed'):
                return entry
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
    raise KeyError(f'Workflow node not found: {node_id}')


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
        prompt[node_id] = {'class_type': node['type'], 'inputs': inputs}
        if node.get('title'):
            prompt[node_id]['_meta'] = {'title': node['title']}
    return prompt

