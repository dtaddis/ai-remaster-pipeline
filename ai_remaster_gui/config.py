from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SETTINGS_FILE = ROOT / ".ai_remaster_gui.json"
CONFIG_FILE = ROOT / ".ai_remaster_config.json"
PREVIEW_DIR = ROOT / ".cache" / "previews"
FILE_PREVIEW_DIR = ROOT / ".cache" / "file_previews"
ASPECT_PREVIEW_DIR = ROOT / ".cache" / "aspect_previews"
MEDIA_CLIP_DIR = ROOT / ".cache" / "media_clips"
STATIC_DIR = Path(__file__).resolve().parent / "static"

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
TEXT_EXTS = {".csv", ".json", ".txt", ".log", ".md"}

REFERENCE_PROMPT = "Colorize this image."
REFERENCE_PROMPT_SUFFIX = "Preserve composition, lighting, identity, and detail. Do not add text or new objects."


def load_config() -> dict[str, str]:
    config = {
        "comfy_dir": str(ROOT / "tools" / "comfyui"),
        "comfy_url": "http://127.0.0.1:8188",
        "comfy_host": "127.0.0.1",
        "comfy_port": "8188",
    }
    if CONFIG_FILE.exists():
        try:
            stored = json.loads(CONFIG_FILE.read_text(encoding="utf-8-sig"))
            if isinstance(stored, dict):
                config.update({key: str(value) for key, value in stored.items() if value is not None})
        except json.JSONDecodeError:
            pass
    return config


def current_config() -> dict[str, str]:
    return load_config()
