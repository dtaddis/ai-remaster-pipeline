from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .config import (
    OUTPAINT_PROMPT,
    QWEN_IMAGE_EDIT_MODEL,
    REFERENCE_PROMPT,
    REFERENCE_PROMPT_SUFFIX,
    ROOT,
    SETTINGS_FILE,
    VIDEO_EXTS,
    comfy_output_root_for,
    current_config,
)
from .models import CLEANUP_NEGATIVE_PROMPT, CLEANUP_PROMPT, STAGES
from .paths import newest, rel, resolve


def app_version() -> str:
    version_file = ROOT / "VERSION"
    base = version_file.read_text(encoding="utf-8").strip() if version_file.exists() else "0.0.0"
    try:
        commit = subprocess.run(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        commit = ""
    return "v" + base.lstrip("v") + (f"-{commit}" if commit else "")

def default_qwen_workflow(config: dict[str, str]) -> str:
    comfy_dir = Path(config.get("comfy_dir", ROOT / "tools" / "comfyui"))
    search_dirs = [
        ROOT / "workflows" / "qwen_image_edit",
        ROOT / "blueprints",
        comfy_dir / "blueprints",
        comfy_dir / "user" / "default" / "workflows",
    ]
    all_matches: list[Path] = []

    def workflow_rank(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        if "2511" in name and ("image edit" in name or "image_edit" in name):
            return (0, name)
        if "image edit" in name or "image_edit" in name:
            return (1, name)
        if "edit" in name:
            return (2, name)
        return (3, name)

    for directory in search_dirs:
        if not directory.exists():
            continue
        matches = sorted(
            path
            for path in directory.glob("*.json")
            if "qwen" in path.name.lower() and path.is_file()
        )
        preferred = [path for path in matches if workflow_rank(path)[0] < 3]
        if preferred:
            return rel(sorted(preferred, key=workflow_rank)[0])
        all_matches.extend(matches)
    if all_matches:
        return rel(sorted(all_matches)[0])
    return ""

def default_qwen_masked_workflow(_config: dict[str, str]) -> str:
    bundled = ROOT / "workflows" / "qwen_image_edit" / "Image Edit Inpaint (Qwen 2511).json"
    return rel(bundled) if bundled.is_file() else ""

def should_migrate_qwen_workflow(workflow: str) -> bool:
    if not workflow:
        return True
    resolved = resolve(workflow)
    try:
        resolved.relative_to(ROOT / "workflows" / "qwen_image_edit")
        return False
    except ValueError:
        pass
    workflow_lower = workflow.lower()
    name = resolved.name.lower()
    return (
        "qwen" in workflow_lower
        and ("2511" in workflow_lower or "image edit" in workflow_lower or "image_edit" in workflow_lower)
        and (
            "blueprints" in workflow_lower
            or "workflow_templates" in workflow_lower
            or "workflows" in workflow_lower
            or "templates" in workflow_lower
            or "image edit" in name
            or "image_edit" in name
        )
    )

def qwen_workflow_for(values: dict[str, str], config: dict[str, str]) -> str:
    configured = values.get("workflow", "")
    default_workflow = default_qwen_workflow(config)
    if default_workflow and should_migrate_qwen_workflow(configured):
        return default_workflow
    if configured and resolve(configured).exists():
        return configured
    return default_workflow

def qwen_masked_workflow_for(values: dict[str, str], config: dict[str, str]) -> str:
    configured = values.get("masked_workflow", "")
    if configured and resolve(configured).exists():
        return configured
    return default_qwen_masked_workflow(config)

def base_settings() -> dict[str, dict[str, str]]:
    defaults = {stage.key: {key: default for key, _label, _kind, default in stage.fields} for stage in STAGES}
    defaults["global"] = {"source": "", "cleanup": "false", "expand_outpaint": "true", "colorize": "true", "upscale": "false", "add_soundtrack": "false", "section_start": "0", "section_end": "", "last_browse_dir": ""}
    return defaults


def normalize_settings(defaults: dict[str, dict[str, str]], include_newest_source: bool = True) -> dict[str, dict[str, str]]:
    app_module = sys.modules.get("ai_remaster_gui.app")
    newest_fn = getattr(app_module, "newest", newest) if app_module else newest
    source = newest_fn(ROOT / "input", VIDEO_EXTS) if include_newest_source else None
    if source and not defaults["global"].get("source"):
        defaults["global"]["source"] = rel(source)
    if not defaults["global"].get("source"):
        defaults["global"]["expand_outpaint"] = "true"
    # 4.0 was the old implicit Clean Up default. It requested 96 frames at 24 fps even though
    # Dearchive's native window is 97; migrate that untouched default to the precise duration.
    if defaults["cleanup"].get("chunk_seconds", "") in {"4", "4.0"}:
        defaults["cleanup"]["chunk_seconds"] = "4.04"
    # The former deterministic DeScratch filter was removed in favour of masked AI DeScratch.
    defaults["cleanup"].pop("descratch", None)
    old_cleanup_prompts = {
        "",
        "Clean, restored archive footage with sharp detail, clean tonality, and natural cinematography.",
    }
    old_cleanup_negative_prompts = {
        "",
        "cartoon, game, 3d render, oversaturated color, color bleeding, flicker, warped geometry, smeared details",
    }
    if defaults["cleanup"].get("prompt", "").strip() in old_cleanup_prompts:
        defaults["cleanup"]["prompt"] = CLEANUP_PROMPT
    if defaults["cleanup"].get("negative_prompt", "").strip() in old_cleanup_negative_prompts:
        defaults["cleanup"]["negative_prompt"] = CLEANUP_NEGATIVE_PROMPT
    if "colormnet" in defaults["recomp"].get("colorized_video", "").lower():
        defaults["recomp"]["colorized_video"] = ""
    old_outpaint_prompts = {
        "Outpaint the black margins with a natural continuation of the black-and-white film frame. Replace all black padding/bars with coherent background, clothing, bodies, props, and set detail that matches the original centre footage. Preserve camera motion, composition, lighting, film grain, and monochrome style. Do not colorize.",
    }
    if defaults["outpaint"].get("prompt", "").strip() != OUTPAINT_PROMPT or defaults["outpaint"].get("prompt", "") in old_outpaint_prompts:
        defaults["outpaint"]["prompt"] = OUTPAINT_PROMPT
    # The alternative pure-black LoRA path was removed. Discard the old
    # selector so saved settings cannot silently reactivate it.
    defaults["outpaint"].pop("outpaint_lora", None)
    defaults["outpaint"]["seed_qwen_guides"] = "false"
    defaults["colour"].setdefault("method", "deepexemplar")
    defaults["colour"].setdefault("processing_height", "source")
    defaults["recomp"].setdefault("colorization_method", "deepexemplar")
    # Older builds stored recomposition color controls as small 0-1-ish floats.
    # The UI now exposes Kelvin and percentages, so migrate old values once on load.
    try:
        sat = float(defaults["recomp"].get("saturation", "82"))
        if sat <= 4:
            defaults["recomp"]["saturation"] = str(int(round(sat * 100)))
    except ValueError:
        defaults["recomp"]["saturation"] = "82"
    try:
        opacity = float(defaults["recomp"].get("color_opacity", "100"))
        if opacity <= 1:
            defaults["recomp"]["color_opacity"] = str(int(round(opacity * 100)))
    except ValueError:
        defaults["recomp"]["color_opacity"] = "100"
    try:
        temp = float(defaults["recomp"].get("temperature", "6500"))
        if abs(temp) <= 20:
            defaults["recomp"]["temperature"] = str(int(round(6500 + (-temp * 40000))))
    except ValueError:
        defaults["recomp"]["temperature"] = "6500"
    config = current_config()
    defaults["references"]["comfy_output_root"] = rel(Path(comfy_output_root_for(config)))
    if not defaults["references"].get("comfy_url"):
        defaults["references"]["comfy_url"] = config["comfy_url"]
    if should_migrate_qwen_workflow(defaults["references"].get("workflow", "")):
        migrated_workflow = default_qwen_workflow(config)
        if migrated_workflow:
            defaults["references"]["workflow"] = migrated_workflow
    if not defaults["references"].get("load_image_node_id") or defaults["references"].get("load_image_node_id") == "1":
        defaults["references"]["load_image_node_id"] = "auto"
    defaults["references"].setdefault("prompt_node_id", "")
    if not defaults["references"].get("save_node_id") or defaults["references"].get("save_node_id") == "9":
        defaults["references"]["save_node_id"] = "auto"
    defaults["references"].setdefault("model_backend", "gguf")
    defaults["references"].setdefault("gguf_model", QWEN_IMAGE_EDIT_MODEL)
    if not defaults["references"].get("masked_workflow"):
        defaults["references"]["masked_workflow"] = default_qwen_masked_workflow(config)
    defaults["references"].setdefault("method", "qwen")
    defaults["references"].setdefault("openai_api_key", "")
    defaults["references"].setdefault("openai_image_model", "gpt-image-2")
    defaults["references"].setdefault("openai_image_size", "auto")
    defaults["references"].setdefault("openai_image_quality", "auto")
    defaults["references"].setdefault("openai_send_references", "false")
    old_reference_prompts = {
        "",
        "Colorize this image.",
        "Colorize this image. Preserve the drawing and composition. Use clean modern cartoon colours, not sepia. Do not add text or new objects.",
        "Colorize this image. Preserve the original image. Do not add text, captions, logos, labels, signs, subtitles, or new objects.",
        "Colorize this image as a clean modern full-colour cartoon production still. Preserve the exact drawing, characters, line art, camera angle, and composition. Use natural vivid colours, not sepia or a single tint. Do not add text or new objects.",
        "Transform this black-and-white frame into a clean modern full-colour animation production still. Keep the exact drawing, characters, camera angle, line art, shapes, and composition. Use vivid but tasteful contemporary cartoon colours as if the same scene had been made today with modern colour cameras and animation paint. Do not use sepia, monochrome tinting, hand-tinted antique colours, washed-out beige, or archival restoration grading. Do not add text, captions, logos, labels, signs, subtitles, or new objects.",
    }
    old_reference_suffixes = {
        "",
        "Preserve composition, lighting, identity, and detail. Do not add text or new objects.",
        "Natural period color, preserve lighting and composition.",
        "Modern clean restoration, natural period color, preserve composition and text.",
        "Keep black ink deep and whites clean. Give sky, water, wood, metal, fabric, and props distinct believable colours.",
        "White gloves and faces should stay clean and bright, black ink areas should stay deep black, wood, metal, sky, water, fabric, and background props should receive distinct natural colours. Preserve original lighting, shadows, outlines, and film grain while making the colour read as genuine full colour, not a tint.",
    }
    if defaults["references"].get("prompt", "") in old_reference_prompts or "cartoon" in defaults["references"].get("prompt", "").lower():
        defaults["references"]["prompt"] = REFERENCE_PROMPT
    if defaults["references"].get("prompt_suffix", "") in old_reference_suffixes or "props" in defaults["references"].get("prompt_suffix", "").lower():
        defaults["references"]["prompt_suffix"] = REFERENCE_PROMPT_SUFFIX
    return defaults


def default_settings(include_newest_source: bool = False) -> dict[str, dict[str, str]]:
    return normalize_settings(base_settings(), include_newest_source=include_newest_source)


def load_settings() -> dict[str, dict[str, str]]:
    defaults = base_settings()
    app_module = sys.modules.get("ai_remaster_gui.app")
    settings_file = getattr(app_module, "SETTINGS_FILE", SETTINGS_FILE)
    if settings_file.exists():
        try:
            stored = json.loads(settings_file.read_text(encoding="utf-8"))
            for key, values in stored.items():
                if key in defaults and isinstance(values, dict):
                    defaults[key].update({k: str(v) for k, v in values.items()})
        except json.JSONDecodeError:
            pass
    return normalize_settings(defaults, include_newest_source=True)


APP_VERSION = app_version()
