from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
# Settings normally live next to the project, but ARP_SETTINGS_FILE can redirect them (the test
# suite points it at a throwaway file so tests neither read nor clobber the developer's real UI
# state). No effect on normal runs where the variable is unset.
SETTINGS_FILE = Path(os.environ["ARP_SETTINGS_FILE"]) if os.environ.get("ARP_SETTINGS_FILE") else ROOT / ".ai_remaster_gui.json"
CONFIG_FILE = ROOT / ".ai_remaster_config.json"
PREVIEW_DIR = ROOT / ".cache" / "previews"
FILE_PREVIEW_DIR = ROOT / ".cache" / "file_previews"
ASPECT_PREVIEW_DIR = ROOT / ".cache" / "aspect_previews"
MEDIA_CLIP_DIR = ROOT / ".cache" / "media_clips"
STATIC_DIR = Path(__file__).resolve().parent / "static"

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
IMAGE_EXTS = {".png", ".bmp", ".jpg", ".jpeg", ".dpx", ".webp", ".tif", ".tiff"}
TEXT_EXTS = {".csv", ".json", ".txt", ".log", ".md"}

REFERENCE_PROMPT = "Colorize this image."
REFERENCE_PROMPT_SUFFIX = "Preserve composition, lighting, identity, and detail. Do not add text or new objects."
OUTPAINT_PROMPT = "outpaint"
DEFAULT_OUTPAINT_LORA = "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"

# The single Qwen Image Edit model used everywhere we run Qwen (colour references, outpaint
# guide frames, and shot-change seed guides). Keep this as the one source of truth.
QWEN_IMAGE_EDIT_MODEL = "qwen-image-edit-2511-Q4_K_M.gguf"


def load_config() -> dict[str, str]:
    config = {
        "comfy_dir": str(ROOT / "tools" / "comfyui"),
        "comfy_url": "http://127.0.0.1:8188",
        "comfy_host": "127.0.0.1",
        "comfy_port": "8188",
        "comfy_managed_by_arp": "true",
    }
    if CONFIG_FILE.exists():
        try:
            stored = json.loads(CONFIG_FILE.read_text(encoding="utf-8-sig"))
            if isinstance(stored, dict):
                config.update({key: str(value) for key, value in stored.items() if value is not None})
        except json.JSONDecodeError:
            pass
    comfy_dir = resolve_comfy_dir(config["comfy_dir"])
    config["comfy_dir"] = str(comfy_dir)
    if same_path(comfy_dir, ROOT / "tools" / "comfyui"):
        config["comfy_managed_by_arp"] = "true"
    return config


def resolve_comfy_dir(path_text: str) -> Path:
    path = Path(path_text)
    if (path / "main.py").exists():
        return path
    for child in ("ComfyUI", "comfyui"):
        nested = path / child
        if (nested / "main.py").exists():
            return nested
    return path


def same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except OSError:
        return left.absolute() == right.absolute()


def comfy_dir_for(config: dict[str, str] | None = None) -> str:
    """ComfyUI install directory, falling back to the bundled tools/comfyui."""
    active = load_config() if config is None else config
    return active.get("comfy_dir", str(ROOT / "tools" / "comfyui"))


def comfy_url_for(config: dict[str, str] | None = None) -> str:
    """ComfyUI server URL, falling back to the local default."""
    active = load_config() if config is None else config
    return active.get("comfy_url", "http://127.0.0.1:8188")


def comfy_output_root_for(config: dict[str, str] | None = None) -> str:
    return str(Path(comfy_dir_for(config)) / "output")


def current_config() -> dict[str, str]:
    return load_config()
