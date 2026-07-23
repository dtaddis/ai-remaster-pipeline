from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


MODEL_PATH_ALIASES = {
    "diffusion_models": ("diffusion_models", "unet"),
    "unet": ("unet", "diffusion_models"),
    "text_encoders": ("text_encoders", "clip"),
}


def huggingface_cache_dir(default: Path) -> Path:
    """Return the Hugging Face Hub cache, honoring standard and ARP overrides."""
    explicit = os.environ.get("ARP_HF_CACHE_DIR") or os.environ.get("HF_HUB_CACHE")
    if explicit:
        return _expanded_path(explicit)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return _expanded_path(hf_home) / "hub"
    return default


def resolve_comfy_model_path(comfy_dir: Path, destination: str | Path) -> Path:
    """Resolve an ARP model destination through ComfyUI's extra model paths.

    Existing files in any configured search path are reused. New downloads prefer
    ``download_model_base`` or a category path from the section marked
    ``is_default``. Without a usable config, the traditional ComfyUI-local path is
    returned.
    """
    comfy_dir = Path(comfy_dir)
    destination = Path(destination)
    if destination.is_absolute():
        return destination

    parts = destination.parts
    if not parts or parts[0].lower() != "models" or len(parts) < 2:
        return comfy_dir / destination

    model_relative = Path(*parts[1:])
    category = model_relative.parts[0]
    remainder = Path(*model_relative.parts[1:]) if len(model_relative.parts) > 1 else Path()
    sections = _load_extra_model_sections(comfy_dir / "extra_model_paths.yaml")

    configured_paths: list[Path] = []
    for section in _ordered_sections(sections):
        for key in MODEL_PATH_ALIASES.get(category, (category,)):
            configured_paths.extend(section["paths"].get(key, []))

    for folder in configured_paths:
        candidate = folder / remainder
        if candidate.exists():
            return candidate

    for section in sections:
        if not section["is_default"]:
            continue
        download_root = section.get("download_model_base")
        if download_root is not None:
            return download_root / model_relative
        for key in MODEL_PATH_ALIASES.get(category, (category,)):
            paths = section["paths"].get(key, [])
            if paths:
                return paths[0] / remainder

    return comfy_dir / destination


def _expanded_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value)))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_config_path(value: str, base_path: Path | None, yaml_dir: Path) -> Path:
    path = _expanded_path(value)
    if path.is_absolute():
        return Path(os.path.abspath(path))
    return Path(os.path.abspath((base_path or yaml_dir) / path))


def _load_extra_model_sections(config_path: Path) -> list[dict[str, Any]]:
    if not config_path.is_file():
        return []
    try:
        import yaml

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"Warning: could not read ComfyUI model paths from {config_path}: {exc}", file=sys.stderr)
        return []
    if not isinstance(raw, dict):
        return []

    sections: list[dict[str, Any]] = []
    for value in raw.values():
        if not isinstance(value, dict):
            continue
        base_path = None
        base_value = value.get("base_path")
        if isinstance(base_value, str) and base_value.strip():
            base_path = _resolve_config_path(base_value.strip(), None, config_path.parent)

        paths: dict[str, list[Path]] = {}
        for key, path_value in value.items():
            if key in {"base_path", "is_default", "download_model_base"} or not isinstance(path_value, str):
                continue
            resolved = [
                _resolve_config_path(line.strip(), base_path, config_path.parent)
                for line in path_value.splitlines()
                if line.strip()
            ]
            if resolved:
                paths[str(key)] = resolved

        download_model_base = None
        download_value = value.get("download_model_base")
        if isinstance(download_value, str) and download_value.strip():
            download_model_base = _resolve_config_path(download_value.strip(), base_path, config_path.parent)

        sections.append(
            {
                "is_default": _as_bool(value.get("is_default", False)),
                "download_model_base": download_model_base,
                "paths": paths,
            }
        )
    return sections


def _ordered_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [section for section in sections if section["is_default"]] + [
        section for section in sections if not section["is_default"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve ARP model paths using ComfyUI configuration.")
    parser.add_argument("--comfy-dir", type=Path, required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    print(resolve_comfy_model_path(args.comfy_dir, args.destination))


if __name__ == "__main__":
    main()
