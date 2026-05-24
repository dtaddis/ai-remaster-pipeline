from __future__ import annotations

import json
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen


def comfy_is_running(url: str) -> bool:
    try:
        with urlopen(url.rstrip("/") + "/queue", timeout=2) as response:
            return 200 <= response.status < 300
    except (URLError, OSError, TimeoutError):
        return False


def comfy_queue(url: str, timeout: float = 2.0) -> dict | None:
    try:
        with urlopen(url.rstrip("/") + "/queue", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (URLError, OSError, TimeoutError, json.JSONDecodeError):
        return None


def queue_count(queue: dict | None, key: str) -> int:
    value = queue.get(key) if isinstance(queue, dict) else None
    return len(value) if isinstance(value, list) else 0


def comfy_busy_message(url: str, queue: dict | None) -> str:
    running = queue_count(queue, "queue_running")
    pending = queue_count(queue, "queue_pending")
    if running or pending:
        return f"ComfyUI at {url} is busy ({running} running, {pending} pending). Wait for it to finish or clear the ComfyUI queue."
    return ""


def discover_comfy_instances(configured_url: str) -> list[str]:
    parsed = urlparse(configured_url)
    host = parsed.hostname or "127.0.0.1"
    configured_port = parsed.port or 8188
    urls = [configured_url.rstrip("/")]
    if host in {"127.0.0.1", "localhost"}:
        for port in range(8188, 8199):
            candidate = f"{parsed.scheme or 'http'}://{host}:{port}"
            if port != configured_port:
                urls.append(candidate)
    found: list[str] = []
    for url in dict.fromkeys(urls):
        if comfy_queue(url, timeout=0.35) is not None:
            found.append(url)
    return found
