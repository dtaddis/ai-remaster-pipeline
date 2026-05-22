from __future__ import annotations

import csv
import hashlib
import html
import json
import mimetypes
import os
import atexit
import signal
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SETTINGS_FILE = ROOT / ".ai_remaster_gui.json"
CONFIG_FILE = ROOT / ".ai_remaster_config.json"
PREVIEW_DIR = ROOT / ".cache" / "previews"
FILE_PREVIEW_DIR = ROOT / ".cache" / "file_previews"
ASPECT_PREVIEW_DIR = ROOT / ".cache" / "aspect_previews"
MEDIA_CLIP_DIR = ROOT / ".cache" / "media_clips"
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
TEXT_EXTS = {".csv", ".json", ".txt", ".log", ".md"}
COLORIZE_STAGE_KEYS = {"shots", "references", "colour"}
REFERENCE_PROMPT = (
    "Colorize this image."
)
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


CONFIG = load_config()
STARTED_COMFY_PROCESS: subprocess.Popen | None = None


def current_config() -> dict[str, str]:
    return load_config()


def default_qwen_workflow(config: dict[str, str] | None = None) -> str:
    config = config or current_config()
    candidates = [
        ROOT / "workflows" / "qwen_image_edit" / "Qwen Image Edit Reference Colorize.json",
        Path(config.get("comfy_dir", "")) / "venv" / "Lib" / "site-packages" / "comfyui_workflow_templates_media_image" / "templates" / "image_qwen_image_edit_2511.json",
        Path(config.get("comfy_dir", "")) / "blueprints" / "Image Edit (Qwen 2511).json",
        Path(config.get("comfy_dir", "")) / "blueprints" / "Image Edit (Qwen 2509).json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return rel(candidate)
    return ""


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    description: str
    folders: tuple[str, ...]
    fields: tuple[tuple[str, str, str, str], ...]
    required: tuple[str, ...]


STAGES = (
    Stage(
        "outpaint",
        "Outpainting",
        "Prepare the source clip chosen on the Global tab for LTX outpainting.",
        ("input", "intermediate/outpaint_prepared", "intermediate/outpainted"),
        (
            ("target_aspect", "Target aspect ratio", "select:16:9|9:16|4:3|3:4|1:1|21:9|2.39:1|2.35:1|1.85:1|3:2|2:3|5:4|4:5", "16:9"),
            ("target_height", "Output height", "select:544|576|720|768|1080", "720"),
            ("chunk_seconds", "Chunk seconds", "number", "4"),
            ("overlap_frames", "Overlap frames", "range:0|48|1", "8"),
            ("crop_left", "Crop left", "range:0|240|1", "0"),
            ("crop_right", "Crop right", "range:0|240|1", "0"),
            ("crop_top", "Crop top", "range:0|240|1", "0"),
            ("crop_bottom", "Crop bottom", "range:0|240|1", "0"),
        ),
        (),
    ),
    Stage(
        "output",
        "Output",
        "Preview the final remastered movie once recomposition has finished.",
        ("output/reassembled",),
        (
            ("output", "Final output", "file", ""),
        ),
        (),
    ),
    Stage(
        "shots",
        "Shot Detection",
        "Detect cuts and extract one useful reference frame per shot.",
        ("intermediate/outpainted", "intermediate/outpainted_references", "manifests/references"),
        (
            ("outpainted_video", "Outpainted video", "file", ""),
            ("sample_seconds", "Sample seconds", "number", "0"),
            ("shot_threshold", "Shot threshold", "number", "0.075"),
            ("min_shot_seconds", "Minimum shot seconds", "number", "1.0"),
            ("limit", "Limit rows", "number", ""),
        ),
        ("outpainted_video",),
    ),
    Stage(
        "references",
        "Reference Generation",
        "Colorize extracted stills through a Qwen Image Edit ComfyUI workflow.",
        ("intermediate/outpainted_references", "intermediate/outpainted_references_color", "manifests/references"),
        (
            ("manifest", "Manifest", "file", ""),
            ("prompt", "Prompt", "text", REFERENCE_PROMPT),
            ("prompt_suffix", "Prompt suffix", "text", REFERENCE_PROMPT_SUFFIX),
            ("limit", "Limit rows", "number", ""),
        ),
        ("manifest",),
    ),
    Stage(
        "colour",
        "Colorization",
        "Run Deep Exemplar in ComfyUI over the outpainted video, using the generated color references.",
        ("intermediate/outpainted_references_color", "intermediate/outpainted_colorized", "manifests/references"),
        (
            ("manifest", "Manifest", "file", ""),
            ("frame_propagate", "Frame propagation", "select:true|false", "true"),
            ("use_half_resolution", "Half-resolution processing", "select:true|false", "true"),
            ("use_torch_compile", "Torch compile", "select:false|true", "false"),
            ("use_sage_attention", "SageAttention", "select:false|true", "false"),
            ("crf", "CRF", "number", "18"),
        ),
        ("manifest",),
    ),
    Stage(
        "recomp",
        "Recomposition",
        "Composite outpainted video, original centre footage, and optional colorized video.",
        ("input", "intermediate/outpainted", "intermediate/outpainted_colorized", "output/reassembled"),
        (
            ("outpainted_video", "Outpainted video", "file", ""),
            ("source", "Original source", "file", ""),
            ("colorized_video", "Colorized video", "file", ""),
            ("feather_pixels", "Feather pixels", "number", "80"),
            ("saturation", "Saturation", "number", "0.82"),
            ("temperature", "Temperature", "number", "-0.015"),
            ("color_opacity", "Color opacity", "number", "1.0"),
            ("encoder", "Encoder", "select:h264|prores", "h264"),
        ),
        ("outpainted_video", "source"),
    ),
)


def output_stage() -> Stage:
    return next(stage for stage in STAGES if stage.key == "output")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve(text: str) -> Path:
    path = Path(text).expanduser()
    return path if path.is_absolute() else ROOT / path


def newest(folder: Path, exts: set[str]) -> Path | None:
    if not folder.exists():
        return None
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return max(files, key=lambda p: p.stat().st_mtime_ns) if files else None


def load_settings() -> dict[str, dict[str, str]]:
    defaults = {stage.key: {key: default for key, _label, _kind, default in stage.fields} for stage in STAGES}
    defaults["global"] = {"source": "", "colorize": "true"}
    if SETTINGS_FILE.exists():
        try:
            stored = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            for key, values in stored.items():
                if key in defaults and isinstance(values, dict):
                    defaults[key].update({k: str(v) for k, v in values.items()})
        except json.JSONDecodeError:
            pass
    source = newest(ROOT / "input", VIDEO_EXTS)
    outpainted = newest(ROOT / "intermediate" / "outpainted", VIDEO_EXTS)
    manifest = newest(ROOT / "manifests" / "references", {".csv"})
    colorized = newest(ROOT / "intermediate" / "outpainted_colorized", VIDEO_EXTS)
    if source and not defaults["global"].get("source"):
        defaults["global"]["source"] = rel(source)
    for stage_key, values in defaults.items():
        if stage_key == "global":
            continue
        if outpainted and not values.get("outpainted_video"):
            values["outpainted_video"] = rel(outpainted)
        if manifest and not values.get("manifest"):
            values["manifest"] = rel(manifest)
        if colorized and not values.get("colorized_video"):
            values["colorized_video"] = rel(colorized)
    if outpainted and not defaults["recomp"].get("output"):
        defaults["recomp"]["output"] = rel(ROOT / "output" / "reassembled" / f"{outpainted.stem}_final.mp4")
    if "colormnet" in defaults["recomp"].get("colorized_video", "").lower():
        defaults["recomp"]["colorized_video"] = ""
    bundled_output = rel(ROOT / "tools" / "comfyui" / "output")
    config = current_config()
    if not defaults["references"].get("comfy_output_root") or (CONFIG_FILE.exists() and defaults["references"].get("comfy_output_root") == bundled_output):
        defaults["references"]["comfy_output_root"] = rel(Path(config["comfy_dir"]) / "output")
    if not defaults["references"].get("comfy_url"):
        defaults["references"]["comfy_url"] = config["comfy_url"]
    if not defaults["references"].get("workflow"):
        defaults["references"]["workflow"] = default_qwen_workflow(config)
    elif "blueprints" in defaults["references"].get("workflow", "").lower() and "qwen 2511" in defaults["references"].get("workflow", "").lower():
        migrated_workflow = default_qwen_workflow(config)
        if migrated_workflow:
            defaults["references"]["workflow"] = migrated_workflow
    if not defaults["references"].get("load_image_node_id") or defaults["references"].get("load_image_node_id") == "1":
        defaults["references"]["load_image_node_id"] = "auto"
    defaults["references"].setdefault("prompt_node_id", "")
    if not defaults["references"].get("save_node_id") or defaults["references"].get("save_node_id") == "9":
        defaults["references"]["save_node_id"] = "auto"
    defaults["references"].setdefault("model_backend", "gguf")
    defaults["references"].setdefault("gguf_model", "qwen-image-edit-2511-Q4_K_M.gguf")
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


class PipelineApp:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.log: list[str] = []
        self.process: subprocess.Popen[str] | None = None
        self.running_stage = ""
        self.running_stage_key = ""
        self.running_reference_manifest = ""
        self.running_reference_index: int | None = None
        self.run_started_at = 0.0
        self.lock = threading.Lock()

    def colorize_enabled(self) -> bool:
        return self.settings.get("global", {}).get("colorize", "true") == "true"

    def active_stages(self) -> tuple[Stage, ...]:
        stages = tuple(stage for stage in STAGES if stage.key != "output")
        if self.colorize_enabled():
            return stages
        return tuple(stage for stage in stages if stage.key not in COLORIZE_STAGE_KEYS)

    def save(self) -> None:
        SETTINGS_FILE.write_text(json.dumps(self.settings, indent=2) + "\n", encoding="utf-8")

    def files_for(self, stage: Stage) -> list[dict[str, str | int]]:
        exts = VIDEO_EXTS | IMAGE_EXTS | TEXT_EXTS
        scoped_prefixes = self.stage_file_prefixes(stage.key)
        out = []
        for folder_text in stage.folders:
            folder = ROOT / folder_text
            if not folder.exists():
                continue
            for path in folder.rglob("*"):
                if path.is_file() and path.suffix.lower() in exts and self.stage_file_matches(stage.key, path, scoped_prefixes):
                    stat = path.stat()
                    out.append({"path": rel(path), "size": stat.st_size, "mtime": int(stat.st_mtime), "preview": file_preview(path)})
        return sorted(out, key=lambda item: str(item["path"]).lower())

    def stage_file_prefixes(self, stage_key: str) -> tuple[str, ...]:
        source = self.settings.get("global", {}).get("source", "")
        if stage_key == "outpaint" and source:
            stem = safe_stem(resolve(source).name)
            values = self.settings.get("outpaint", {})
            aspect = aspect_slug(values.get("target_aspect", "16:9"))
            try:
                height = int(float(values.get("target_height", "720") or "720"))
            except ValueError:
                height = 720
            size = f"{even_int(height * parse_aspect(values.get('target_aspect', '16:9')))}x{even_int(height)}"
            return (stem,)
        return ()

    def stage_file_matches(self, stage_key: str, path: Path, prefixes: tuple[str, ...]) -> bool:
        if stage_key != "outpaint" or not prefixes:
            return True
        name = path.stem
        return any(name == prefix or name.startswith(prefix + "_") for prefix in prefixes)

    def progress(self) -> list[dict[str, str]]:
        checks = {
            "Outpainting": newest(ROOT / "intermediate" / "outpainted", VIDEO_EXTS),
            "Shot Detection": newest(ROOT / "manifests" / "references", {".csv"}),
            "Reference Generation": newest(ROOT / "intermediate" / "outpainted_references_color", IMAGE_EXTS),
            "Colorization": newest(ROOT / "intermediate" / "outpainted_colorized", VIDEO_EXTS),
            "Recomposition": newest(ROOT / "output" / "reassembled", VIDEO_EXTS),
        }
        active_titles = {stage.title for stage in self.active_stages()}
        return [{"stage": key, "status": "Ready" if value else "Waiting", "latest": rel(value) if value else ""} for key, value in checks.items() if key in active_titles]

    def phase_progress(self) -> dict:
        current = self.estimate_running_progress()
        stages = []
        completed = 0.0
        active = self.active_stages()
        for stage in active:
            title = stage.title
            latest = next((item["latest"] for item in self.progress() if item["stage"] == title), "")
            if self.running_stage_key == stage.key and current:
                percent = current["percent"]
                label = current["label"]
            elif latest:
                percent = 100
                label = "Ready"
            else:
                percent = 0
                label = "Waiting"
            completed += percent / 100
            stages.append({"key": stage.key, "stage": title, "percent": percent, "label": label})
        global_percent = int(round((completed / max(1, len(active))) * 100))
        return {"global": {"percent": global_percent, "label": f"{global_percent}% complete"}, "stages": stages}

    def estimate_running_progress(self) -> dict:
        if not self.running_stage_key:
            return {}
        elapsed = max(0.0, time.time() - self.run_started_at)
        log_text = "\n".join(self.log[-300:])
        lower = log_text.lower()
        percent = min(90, 5 + int(elapsed / 60 * 20))
        label = "Running"
        if self.running_stage_key == "outpaint":
            chunk = outpaint_chunk_progress(log_text)
            milestones = [
                ("checking model", 8, "Checking models"),
                ("downloading model", 10, "Downloading models"),
                ("downloaded:", 11, "Model download complete"),
                ("preparing expanded outpaint canvas", 12, "Preparing expanded canvas"),
                ("reuse prepared outpaint input", 20, "Prepared input reused"),
                ("wrote prepared outpaint input", 20, "Prepared input written"),
                ("prepared expanded canvas for comfyui", 25, "Prepared for ComfyUI"),
                ("splitting prepared canvas", 28, "Splitting into chunks"),
                ("waiting for comfyui", 30, "Waiting for ComfyUI"),
                ("queued comfyui prompt", 40, "Queued in ComfyUI"),
                ("outpaint chunk", 42, "Outpainting chunks"),
                ("wrote raw comfy render", 82, "Raw outpaint render written"),
                ("reuse raw comfy render", 82, "Raw outpaint render reused"),
                ("wrote outpainted video", 100, "Outpainted video written"),
            ]
            for token, value, text in milestones:
                if token in lower and value >= percent:
                    percent, label = value, text
            if chunk["total"] and percent < 100:
                percent = max(percent, min(95, 35 + int((chunk["done"] / chunk["total"]) * 55)))
                eta = outpaint_eta_label(elapsed, chunk["done"], chunk["current"], chunk["total"])
                if chunk["done"] >= chunk["total"]:
                    label = f"Chunks complete, finalizing{eta}"
                else:
                    label = f"Chunk {chunk['current']}/{chunk['total']} ({chunk['done']} done){eta}"
        elif self.running_stage_key == "shots":
            if "detected " in lower:
                percent, label = max(percent, 75), "Shots detected"
            if "wrote manifest" in lower:
                percent, label = 100, "Manifest written"
        elif self.running_stage_key == "references":
            if self.running_reference_index is not None:
                label = f"Regenerating shot {self.running_reference_index + 1}"
                if "queued comfyui prompt" in lower or "waiting for comfyui" in lower:
                    percent = max(percent, 35)
                    label = f"Shot {self.running_reference_index + 1}: waiting for ComfyUI"
                if "copied comfyui output" in lower or "wrote " in lower:
                    percent = max(percent, 85)
                    label = f"Shot {self.running_reference_index + 1}: saving reference"
                if "regenerated colour reference" in lower or "finished with exit code 0" in lower:
                    percent = 100
                    label = f"Shot {self.running_reference_index + 1}: complete"
            else:
                rows = first_int_after(log_text, "Rows:")
                done = count_lines_matching(log_text, ("Reuse ", "Wrote "))
                if rows:
                    percent = min(99, int((done / rows) * 100))
                    label = f"{done}/{rows} references"
        elif self.running_stage_key == "colour":
            if "reuse" in lower:
                percent, label = max(percent, 75), "Existing colorized video reused"
            if "wrote" in lower or "finished with exit code 0" in lower:
                percent, label = 100, "Colorization complete"
        elif self.running_stage_key == "recomp":
            if "wrote composite" in lower:
                percent, label = 100, "Composite written"
            else:
                label = "Compositing"
        return {"key": self.running_stage_key, "stage": self.running_stage, "percent": percent, "label": label}

    def state(self) -> dict:
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            source_text = self.settings.get("global", {}).get("source", "")
            previews = source_previews(source_text)
            info = source_info(source_text)
            return {
                "root": str(ROOT),
                "stages": [stage.__dict__ | {"files": self.files_for(stage)} for stage in (*self.active_stages(), output_stage())],
                "settings": self.settings,
                "progress": self.progress(),
                "phase_progress": self.phase_progress(),
                "expected_outputs": {stage.key: self.expected_outputs(stage.key) for stage in (*self.active_stages(), output_stage())},
                "source_previews": previews,
                "source_info": info,
                "source_monochrome": source_monochrome(source_text),
                "aspect_preview": aspect_preview(source_text, self.settings.get("outpaint", {}).get("target_aspect", "16:9")),
                "shot_views": shot_views(self.settings),
                "running": running,
                "running_stage": self.running_stage,
                "running_reference": {
                    "manifest": self.running_reference_manifest,
                    "index": self.running_reference_index,
                } if self.running_reference_index is not None else None,
                "log": "\n".join(self.log[-800:]),
            }

    def update_settings(self, stage: str, values: dict[str, str]) -> None:
        self.settings.setdefault(stage, {}).update({key: str(value) for key, value in values.items()})
        if stage == "global" and "source" in values:
            self.log.append(f"Loading source material: {values.get('source')}")
            self.settings.setdefault("global", {})["colorize"] = "true" if source_monochrome(str(values.get("source", ""))) else "false"
            self.hydrate_stage_inputs("global")
        elif stage == "global" and "colorize" in values:
            self.hydrate_stage_inputs("global")
        if stage == "shots" and "outpainted_video" in values:
            manifest = manifest_for_outpainted(values.get("outpainted_video", ""))
            self.settings.setdefault("references", {}).setdefault("manifest", manifest)
            self.settings.setdefault("colour", {}).setdefault("manifest", manifest)
        self.save()

    def clear_overview(self) -> None:
        self.settings.setdefault("global", {}).update({"source": "", "colorize": "true"})
        for stage_key, keys in {
            "outpaint": ("source", "output"),
            "shots": ("outpainted_video", "manifest"),
            "references": ("manifest",),
            "colour": ("manifest", "outpainted_video", "colorized_video"),
            "recomp": ("outpainted_video", "source", "colorized_video", "output"),
            "output": ("output",),
        }.items():
            stage_settings = self.settings.setdefault(stage_key, {})
            for key in keys:
                stage_settings[key] = ""
        self.log.append("Cleared source material from the Overview.")
        self.save()

    def hydrate_stage_inputs(self, completed_stage: str = "") -> None:
        expected_outpainted = resolve(self.expected_outputs("outpaint")[0]) if self.expected_outputs("outpaint") else None
        if completed_stage == "global" and not (expected_outpainted and expected_outpainted.exists()):
            outpainted = None
        else:
            outpainted = expected_outpainted if expected_outpainted and expected_outpainted.exists() else newest(ROOT / "intermediate" / "outpainted", VIDEO_EXTS)
        if outpainted:
            outpainted_text = rel(outpainted)
            self.settings.setdefault("shots", {})["outpainted_video"] = outpainted_text
            self.settings.setdefault("recomp", {})["outpainted_video"] = outpainted_text
            manifest = manifest_for_outpainted(outpainted_text)
            self.settings.setdefault("references", {})["manifest"] = manifest
            self.settings.setdefault("colour", {})["manifest"] = manifest
            self.log.append(f"Updated Shot Detection input: {outpainted_text}")
        elif completed_stage == "global":
            for stage_key in ("shots", "recomp"):
                self.settings.setdefault(stage_key, {})["outpainted_video"] = ""
            for stage_key in ("references", "colour"):
                self.settings.setdefault(stage_key, {})["manifest"] = ""
        expected_manifest = resolve(self.expected_outputs("shots")[0]) if self.expected_outputs("shots") else None
        manifest = expected_manifest if expected_manifest and expected_manifest.exists() else None
        if manifest:
            manifest_text = rel(manifest)
            self.settings.setdefault("references", {})["manifest"] = manifest_text
            self.settings.setdefault("colour", {})["manifest"] = manifest_text
            self.log.append(f"Updated manifest inputs: {manifest_text}")
        expected_colorized_text = self.expected_outputs("colour")[0] if self.expected_outputs("colour") else ""
        expected_colorized = resolve(expected_colorized_text) if expected_colorized_text else None
        colorized = expected_colorized if expected_colorized and expected_colorized.exists() else None
        if self.colorize_enabled() and colorized:
            self.settings.setdefault("recomp", {})["colorized_video"] = rel(colorized)
        elif not self.colorize_enabled():
            self.settings.setdefault("recomp", {})["colorized_video"] = ""
        source = self.settings.get("global", {}).get("source")
        if source:
            self.settings.setdefault("recomp", {})["source"] = source
        output = recomposition_output_for(self.settings.get("recomp", {}).get("outpainted_video", ""))
        if output:
            self.settings.setdefault("recomp", {})["output"] = output
            self.settings.setdefault("output", {})["output"] = output
        self.save()

    def expected_outputs(self, stage_key: str) -> list[str]:
        values = self.settings.get(stage_key, {})
        if stage_key == "outpaint":
            source = self.settings.get("global", {}).get("source", "")
            return [outpaint_output_for(source, values.get("target_aspect", "16:9"), values.get("target_height", "720"))] if source else []
        if stage_key == "shots":
            return [manifest_for_outpainted(values.get("outpainted_video", ""))]
        if stage_key == "references":
            return color_reference_outputs(values.get("manifest", ""))
        if stage_key == "colour":
            output = colorized_output_for_manifest(values.get("manifest", ""))
            return [output] if output else []
        if stage_key == "recomp":
            return [values.get("output") or recomposition_output_for(values.get("outpainted_video", ""))]
        if stage_key == "output":
            output = self.settings.get("recomp", {}).get("output") or recomposition_output_for(self.settings.get("recomp", {}).get("outpainted_video", ""))
            return [output] if output else []
        return []

    def existing_outputs(self, stage_key: str) -> list[str]:
        return [path for path in self.expected_outputs(stage_key) if path and resolve(path).exists()]

    def command_for(self, stage_key: str) -> list[str]:
        values = self.settings[stage_key]
        config = current_config()
        py = sys.executable
        cmd = [py, "-u"]
        add = cmd.extend
        if stage_key == "outpaint":
            cmd.append(str(SCRIPTS / "outpaint_video.py"))
            add(["--source", self.settings.get("global", {}).get("source", "")])
            add(["--target-aspect", values.get("target_aspect", "16:9")])
            add(["--target-height", values.get("target_height", "720")])
            add(["--chunk-seconds", values.get("chunk_seconds", "20")])
            add(["--overlap-frames", values.get("overlap_frames", "8")])
            for key in ("crop_left", "crop_right", "crop_top", "crop_bottom"):
                add([f"--{key.replace('_', '-')}", values.get(key, "0")])
            add(["--comfy-dir", config.get("comfy_dir", str(ROOT / "tools" / "comfyui"))])
            add(["--comfy-url", config.get("comfy_url", "http://127.0.0.1:8188")])
        elif stage_key == "shots":
            cmd.append(str(SCRIPTS / "generate_references.py"))
            add(["--source-video", values.get("outpainted_video", "")])
            manifest = manifest_for_outpainted(values.get("outpainted_video", ""))
            if manifest:
                add(["--output-manifest", manifest])
            for key in ("sample_seconds", "shot_threshold", "min_shot_seconds"):
                add([f"--{key.replace('_', '-')}", values.get(key, "")])
            if values.get("limit"):
                add(["--limit", values["limit"]])
        elif stage_key == "references":
            cmd.append(str(SCRIPTS / "qwen_colorize_references.py"))
            workflow = values.get("workflow") or default_qwen_workflow(config)
            comfy_url = values.get("comfy_url") or config.get("comfy_url", "http://127.0.0.1:8188")
            comfy_dir = config.get("comfy_dir", str(ROOT / "tools" / "comfyui"))
            comfy_output = values.get("comfy_output_root") or str(Path(comfy_dir) / "output")
            add(["--manifest", values.get("manifest", ""), "--workflow", workflow, "--comfy-url", comfy_url])
            add(["--comfy-dir", comfy_dir, "--comfy-output-root", comfy_output])
            add(["--model-backend", values.get("model_backend", "gguf"), "--gguf-model", values.get("gguf_model", "qwen-image-edit-2511-Q4_K_M.gguf")])
            add(["--prompt", values.get("prompt", ""), "--prompt-suffix", values.get("prompt_suffix", ""), "--load-image-node-id", values.get("load_image_node_id", "auto"), "--save-node-id", values.get("save_node_id", "auto")])
            if values.get("prompt_node_id"):
                add(["--prompt-node-id", values["prompt_node_id"]])
            if values.get("limit"):
                add(["--limit", values["limit"]])
        elif stage_key == "colour":
            cmd.append(str(SCRIPTS / "colorize_video.py"))
            add(["--manifest", values.get("manifest", "")])
            output = colorized_output_for_manifest(values.get("manifest", ""))
            if output:
                add(["--output", output])
            add(["--comfy-dir", config.get("comfy_dir", str(ROOT / "tools" / "comfyui"))])
            add(["--comfy-url", config.get("comfy_url", "http://127.0.0.1:8188")])
            add(["--comfy-output-root", str(Path(config.get("comfy_dir", str(ROOT / "tools" / "comfyui"))) / "output")])
            add(["--crf", values.get("crf", "18")])
            for key in ("frame_propagate", "use_half_resolution", "use_torch_compile", "use_sage_attention"):
                flag = "--" + key.replace("_", "-")
                add([flag if values.get(key, "false") == "true" else "--no-" + flag[2:]])
        elif stage_key == "recomp":
            cmd.append(str(SCRIPTS / "final_composite.py"))
            output = values.get("output") or recomposition_output_for(values.get("outpainted_video", ""))
            add(["--outpainted", values.get("outpainted_video", ""), "--source", values.get("source", ""), "--output", output])
            if self.colorize_enabled() and values.get("colorized_video"):
                add(["--colorized", values["colorized_video"]])
            add(["--feather-pixels", values.get("feather_pixels", "80"), "--saturation", values.get("saturation", "0.82"), "--temperature", values.get("temperature", "-0.015"), "--color-opacity", values.get("color_opacity", "1.0"), "--encoder", values.get("encoder", "h264")])
            outpaint_values = self.settings.get("outpaint", {})
            for key in ("crop_left", "crop_right", "crop_top", "crop_bottom"):
                add([f"--{key.replace('_', '-')}", outpaint_values.get(key, "0")])
        if values.get("force") == "true":
            cmd.append("--force")
        if values.get("dry_run") == "true":
            cmd.append("--dry-run")
        return [part for part in cmd if part != ""]

    def run_stage(self, stage_key: str) -> tuple[bool, str]:
        if stage_key in COLORIZE_STAGE_KEYS and not self.colorize_enabled():
            return False, "Colorize is disabled on the Global tab."
        stage = next(item for item in STAGES if item.key == stage_key)
        values = self.settings[stage_key]
        missing = [key for key in stage.required if not values.get(key)]
        if stage_key == "outpaint" and not self.settings.get("global", {}).get("source"):
            missing = ["source material on the Global tab"]
        if missing:
            return False, "Missing settings: " + ", ".join(missing)
        if stage_key in {"outpaint", "references", "colour"}:
            ok, message = ensure_comfy_available_for_stage(stage.title)
            if not ok:
                return False, message
        with self.lock:
            if self.process and self.process.poll() is None:
                return False, "A command is already running."
            self.running_stage = stage.title
            self.running_stage_key = stage.key
            self.run_started_at = time.time()
            cmd = self.command_for(stage_key)
            self.log.append("> " + " ".join(cmd))
            kwargs: dict = {"cwd": ROOT, "text": True, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            self.process = subprocess.Popen(cmd, **kwargs)
            threading.Thread(target=self._collect_output, args=(stage_key,), daemon=True).start()
        return True, "Started " + stage.title

    def run_reference_regeneration(self, manifest_text: str, index: int) -> tuple[bool, str]:
        ok, message = ensure_comfy_available_for_stage("Reference Generation")
        if not ok:
            return False, message
        try:
            cmd, output = reference_regeneration_command(manifest_text, index)
        except Exception as exc:
            return False, str(exc)
        with self.lock:
            if self.process and self.process.poll() is None:
                return False, "A command is already running."
            self.running_stage = "Reference Generation"
            self.running_stage_key = "references"
            self.running_reference_manifest = manifest_text
            self.running_reference_index = index
            self.run_started_at = time.time()
            self.log.append(f"Regenerating colour reference for shot {index + 1}: {output}")
            self.log.append("> " + " ".join(cmd))
            kwargs: dict = {"cwd": ROOT, "text": True, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            try:
                self.process = subprocess.Popen(cmd, **kwargs)
            except Exception as exc:
                self.running_stage = ""
                self.running_stage_key = ""
                self.running_reference_manifest = ""
                self.running_reference_index = None
                self.run_started_at = 0.0
                self.log.append(f"Could not start reference regeneration: {exc}")
                return False, f"Could not start reference regeneration: {exc}"
            threading.Thread(target=self._collect_output, args=("references",), daemon=True).start()
        return True, f"Started reference regeneration for shot {index + 1}."

    def run_all(self) -> tuple[bool, str]:
        threading.Thread(target=self._run_all_worker, daemon=True).start()
        return True, "Started whole remaster queue."

    def _run_all_worker(self) -> None:
        for stage in self.active_stages():
            ok, message = self.run_stage(stage.key)
            if not ok:
                with self.lock:
                    self.log.append(f"Skipping {stage.title}: {message}")
                continue
            while self.process and self.process.poll() is None:
                time.sleep(0.5)
            if self.process and self.process.returncode:
                break

    def _collect_output(self, stage_key: str) -> None:
        assert self.process and self.process.stdout
        for line in self.process.stdout:
            with self.lock:
                self.log.append(line.rstrip())
        code = self.process.wait()
        with self.lock:
            self.log.append(f"Process finished with exit code {code}.")
            self.running_stage = ""
            self.running_stage_key = ""
            self.running_reference_manifest = ""
            self.running_reference_index = None
            self.run_started_at = 0.0
            if code == 0:
                self.hydrate_stage_inputs(stage_key)

    def stop(self) -> None:
        with self.lock:
            if self.process and self.process.poll() is None:
                terminate_process_tree(self.process)
                self.log.append("Stop requested.")


APP = PipelineApp()


def first_int_after(text: str, marker: str) -> int:
    for line in text.splitlines():
        if marker in line:
            tail = line.split(marker, 1)[1].strip().split()
            if tail:
                try:
                    return int(tail[0].strip(":,"))
                except ValueError:
                    return 0
    return 0
def outpaint_chunk_progress(text: str) -> dict[str, int]:
    done = count_lines_matching(text, ("Wrote raw Comfy chunk", "Reuse raw Comfy chunk"))
    total = 0
    current = 0
    for line in text.splitlines():
        marker = "Outpaint chunk "
        if marker not in line:
            continue
        tail = line.split(marker, 1)[1].split(":", 1)[0]
        if "/" not in tail:
            continue
        try:
            left, right = tail.split("/", 1)
            current = max(current, int(left.strip()))
            total = max(total, int(right.strip()))
        except ValueError:
            pass
    if total:
        current = max(1, min(total, current or min(done + 1, total)))
    return {"done": done, "current": current, "total": total}


def outpaint_eta_label(elapsed: float, done: int, current: int, total: int) -> str:
    if total <= 0 or done >= total:
        return ""
    if done <= 0:
        return ", ETA calculating"
    average_seconds = elapsed / done
    remaining_seconds = max(0.0, average_seconds * (total - done))
    return f", ETA {format_duration(remaining_seconds)}"


def count_lines_matching(text: str, prefixes: tuple[str, ...]) -> int:
    return sum(1 for line in text.splitlines() if line.startswith(prefixes))


def terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except Exception:
        process.terminate()


def read_manifest(path: Path) -> list[dict[str, str]]:
    _source, _fields, rows = read_manifest_details(path)
    return rows


def read_manifest_details(path: Path) -> tuple[str, list[str], list[dict[str, str]]]:
    if not path.exists() or not path.is_file():
        return "", [], []
    source_video = ""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if line.startswith("#"):
                if line.startswith("# source_video="):
                    source_video = line.split("=", 1)[1].strip()
                continue
            reader = csv.DictReader([line, *handle.readlines()])
            return source_video, list(reader.fieldnames or []), list(reader)
    return source_video, [], []


def manifest_source_video(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if line.startswith("# source_video="):
                return line.split("=", 1)[1].strip()
            if line and not line.startswith("#"):
                return ""
    return ""


def update_manifest_row(path: Path, index: int, values: dict[str, str]) -> None:
    source_video, fieldnames, rows = read_manifest_details(path)
    if index < 0 or index >= len(rows):
        raise IndexError(f"Manifest row {index} is out of range.")
    for key in values:
        if key not in fieldnames:
            fieldnames.append(key)
    rows[index].update(values)
    write_manifest_details(path, source_video, fieldnames, rows)


def write_manifest_details(path: Path, source_video: str, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if source_video:
            handle.write(f"# source_video={source_video}\n")
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def merge_manifest_shots(manifest_text: str, index: int) -> dict[str, str]:
    manifest = resolve(manifest_text)
    source_video, fieldnames, rows = read_manifest_details(manifest)
    if index < 0 or index >= len(rows) - 1:
        raise IndexError(f"Shot {index + 1} cannot be merged because there is no following shot.")
    rows[index]["end"] = rows[index + 1].get("end", rows[index].get("end", ""))
    removed = rows.pop(index + 1)
    write_manifest_details(manifest, source_video, fieldnames, rows)
    APP.log.append(f"Merged shot {index + 1} with shot {index + 2}; shared reference: {rows[index].get('source_reference', '')}")
    return {"manifest": rel(manifest), "removed_reference": removed.get("source_reference", ""), "new_end": rows[index].get("end", "")}


def update_shot_boundary(manifest_text: str, index: int, edge: str, seconds: float) -> dict[str, str]:
    manifest = resolve(manifest_text)
    source_video, fieldnames, rows = read_manifest_details(manifest)
    if index < 0 or index >= len(rows):
        raise IndexError(f"Manifest row {index} is out of range.")
    if edge == "start":
        if index == 0:
            raise RuntimeError("The first shot must start at 00:00:00.")
        previous_start = parse_time_seconds(rows[index - 2].get("end", "")) if index > 1 else 0.0
        current_end = parse_time_seconds(rows[index].get("end", ""))
        seconds = max(previous_start, min(current_end - 0.001, seconds))
        rows[index - 1]["end"] = format_timecode(seconds)
    elif edge == "end":
        start = parse_time_seconds(rows[index - 1].get("end", "")) if index > 0 else 0.0
        next_end = parse_time_seconds(rows[index + 1].get("end", "")) if index + 1 < len(rows) else seconds
        upper = max(start + 0.001, next_end)
        seconds = max(start + 0.001, min(upper, seconds))
        rows[index]["end"] = format_timecode(seconds)
    else:
        raise RuntimeError("Boundary edge must be start or end.")
    write_manifest_details(manifest, source_video, fieldnames, rows)
    APP.log.append(f"Updated shot {index + 1} {edge} boundary to {format_timecode(seconds)}")
    return {"manifest": rel(manifest), "time": format_timecode(seconds)}


def source_signature(source_text: str) -> tuple[str, int, int] | None:
    if not source_text:
        return None
    source = resolve(source_text)
    if not source.exists() or source.suffix.lower() not in VIDEO_EXTS:
        return None
    stat = source.stat()
    return str(source), stat.st_size, stat.st_mtime_ns


def manifest_for_outpainted(outpainted_text: str) -> str:
    if not outpainted_text:
        return ""
    source = resolve(outpainted_text)
    stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in source.name.replace(" ", "_"))
    return rel(ROOT / "manifests" / "references" / f"colorize_manifest_{Path(stem).stem}_shots_auto.csv")


def aspect_slug(value: str) -> str:
    return value.replace(":", "x").replace(".", "_")


def safe_stem(path_text: str) -> str:
    stem = Path(path_text).stem.replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)


def outpaint_output_for(source_text: str, aspect: str, target_height_text: str = "720") -> str:
    if not source_text:
        return ""
    source = resolve(source_text)
    try:
        target_height = int(float(target_height_text or "720"))
    except ValueError:
        target_height = 720
    width = even_int(target_height * parse_aspect(aspect))
    height = even_int(target_height)
    values = APP.settings.get("outpaint", {}) if "APP" in globals() else {}
    crops = [int(float(values.get(key, "0") or 0)) for key in ("crop_left", "crop_right", "crop_top", "crop_bottom")]
    crop = "" if not any(crops) else f"_crop{crops[0]}-{crops[1]}-{crops[2]}-{crops[3]}"
    return rel(ROOT / "intermediate" / "outpainted" / f"{safe_stem(source.name)}_{aspect_slug(aspect)}_{width}x{height}{crop}_outpainted.mp4")


def recomposition_output_for(outpainted_text: str) -> str:
    if not outpainted_text:
        return ""
    outpainted = resolve(outpainted_text)
    return rel(ROOT / "output" / "reassembled" / f"{safe_stem(outpainted.name)}_final.mp4")


def colorized_output_for_manifest(manifest_text: str) -> str:
    if not manifest_text:
        return ""
    manifest = resolve(manifest_text)
    source_video = manifest_source_video(manifest)
    if source_video:
        source = resolve(source_video)
        return rel(ROOT / "intermediate" / "outpainted_colorized" / f"{safe_stem(source.name)}_deepexemplar_colorized.mp4")
    if manifest_text:
        stem = safe_stem(Path(manifest_text).stem.replace("colorize_manifest_", "").replace("_shots_auto", ""))
        return rel(ROOT / "intermediate" / "outpainted_colorized" / f"{stem}_deepexemplar_colorized.mp4")
    return ""


def color_reference_outputs(manifest_text: str) -> list[str]:
    if not manifest_text:
        return []
    manifest = resolve(manifest_text)
    if not manifest.is_file():
        return []
    rows = read_manifest(manifest)
    return [row.get("color_reference", "") for row in rows if row.get("color_reference")]


def shot_views(settings: dict[str, dict[str, str]]) -> dict[str, object]:
    shots_manifest = manifest_for_outpainted(settings.get("shots", {}).get("outpainted_video", ""))
    references_manifest = settings.get("references", {}).get("manifest", "")
    colour_manifest = settings.get("colour", {}).get("manifest", "") or references_manifest
    return {
        "shots_manifest": shots_manifest,
        "shots": shot_rows(shots_manifest, include_previews=True),
        "references_manifest": references_manifest,
        "references": shot_rows(references_manifest),
        "colour_manifest": colour_manifest,
        "colour": shot_rows(colour_manifest),
    }


def shot_rows(manifest_text: str, include_previews: bool = False) -> list[dict[str, object]]:
    if not manifest_text:
        return []
    path = resolve(manifest_text)
    rows = read_manifest(path)
    out: list[dict[str, object]] = []
    start = 0.0
    for index, row in enumerate(rows):
        end = parse_time_seconds(row.get("end", "")) or start
        selected = selected_seconds_from_reference(row.get("source_reference", "")) or ((start + end) / 2 if end > start else start)
        selected = max(start, min(end, selected))
        item = {
                "index": index,
                "enabled": row.get("enabled", "true"),
                "start": round(start, 3),
                "end": round(end, 3),
                "start_frame": int(round(start * manifest_fps(path))),
                "end_frame": max(0, int(round(end * manifest_fps(path))) - 1),
                "duration": round(max(0.0, end - start), 3),
                "selected_time": round(selected, 3),
                "start_label": format_timecode(start),
                "end_label": format_timecode(end),
                "selected_label": format_timecode(selected),
                "source_reference": row.get("source_reference", ""),
                "color_reference": row.get("color_reference", ""),
                "source_reference_mtime": file_mtime(row.get("source_reference", "")),
                "color_reference_mtime": file_mtime(row.get("color_reference", "")),
                "can_merge_next": index < len(rows) - 1,
                "prompt": row.get("prompt", ""),
            }
        if include_previews:
            mid = (start + end) / 2 if end > start else start
            for key, value in (("start_preview", start), ("middle_preview", mid), ("end_preview", max(start, end - (1 / max(1.0, manifest_fps(path)))))):
                try:
                    item[key] = preview_reference_frame(manifest_text, index, value)
                except Exception:
                    item[key] = ""
        out.append(item)
        start = end
    return out


@lru_cache(maxsize=16)
def manifest_fps(path: Path) -> float:
    source = resolve(manifest_source_video(path))
    try:
        rate = ffprobe_info(source).get("frame_rate", "")
        if rate.endswith(" fps"):
            return float(rate[:-4])
    except Exception:
        pass
    return 24.0


def file_mtime(path_text: str) -> int:
    if not path_text:
        return 0
    path = resolve(path_text)
    try:
        return int(path.stat().st_mtime)
    except OSError:
        return 0


def parse_time_seconds(value: str) -> float:
    value = str(value or "").strip()
    if not value:
        return 0.0
    parts = value.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(value)
    except ValueError:
        return 0.0


def selected_seconds_from_reference(path_text: str) -> float:
    stem = Path(path_text).stem
    parts = stem.split("_")
    if len(parts) < 3 or parts[0] != "cut":
        return 0.0
    time_parts = parts[-1].split(".")
    try:
        if len(time_parts) >= 3:
            seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
            if len(time_parts) > 3:
                seconds += float("0." + "".join(time_parts[3:]))
            return seconds
    except ValueError:
        return 0.0
    return 0.0


def format_timecode(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def reference_name_for_time(index: int, seconds: float) -> str:
    return f"cut_{index:04d}_{format_timecode(seconds).replace(':', '.')}.png"


def color_reference_for_source(source_reference: str) -> str:
    source = resolve(source_reference)
    try:
        relative = source.relative_to(ROOT / "intermediate" / "outpainted_references")
        return rel(ROOT / "intermediate" / "outpainted_references_color" / relative)
    except ValueError:
        return rel(source.with_name(source.stem + "_color" + source.suffix))


def delete_color_reference(manifest_text: str, index: int) -> dict[str, str]:
    manifest = resolve(manifest_text)
    _source_video, _fields, rows = read_manifest_details(manifest)
    if index < 0 or index >= len(rows):
        raise IndexError(f"Manifest row {index} is out of range.")
    target = rows[index].get("color_reference", "")
    if not target:
        raise RuntimeError("Manifest row does not have a color_reference path.")
    path = resolve(target)
    sig = path.with_suffix(path.suffix + ".sig.json")
    deleted = []
    for item in (path, sig):
        if item.exists() and item.is_file():
            item.unlink()
            deleted.append(rel(item))
    APP.log.append(f"Deleted colour reference for shot {index + 1}: {target}")
    return {"deleted": ", ".join(deleted), "color_reference": target}


def extract_reference_frame(manifest_text: str, index: int, seconds: float) -> dict[str, str]:
    manifest = resolve(manifest_text)
    source_video, _fields, rows = read_manifest_details(manifest)
    if not source_video:
        raise RuntimeError("Manifest does not record a source_video, so ARP cannot rescrub this shot.")
    if index < 0 or index >= len(rows):
        raise IndexError(f"Manifest row {index} is out of range.")
    old_reference = rows[index].get("source_reference", "")
    if old_reference:
        folder = resolve(old_reference).parent
    else:
        source = resolve(source_video)
        folder = ROOT / "intermediate" / "outpainted_references" / safe_stem(source.name)
    new_source = folder / reference_name_for_time(index, seconds)
    ffmpeg = local_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Run install_windows.bat to install local FFmpeg for shot scrubbing.")
    new_source.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-y", "-ss", f"{seconds:.3f}", "-i", str(resolve(source_video)), "-frames:v", "1", "-q:v", "2", str(new_source)]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg failed").strip())
    new_color = color_reference_for_source(rel(new_source))
    update_manifest_row(manifest, index, {"source_reference": rel(new_source), "color_reference": new_color})
    APP.log.append(f"Updated shot {index + 1} reference frame to {format_timecode(seconds)}: {rel(new_source)}")
    return {"source_reference": rel(new_source), "color_reference": new_color}


def preview_reference_frame(manifest_text: str, index: int, seconds: float) -> str:
    manifest = resolve(manifest_text)
    source_video, _fields, rows = read_manifest_details(manifest)
    if not source_video:
        raise RuntimeError("Manifest does not record a source_video, so ARP cannot preview this shot.")
    if index < 0 or index >= len(rows):
        raise IndexError(f"Manifest row {index} is out of range.")
    source = resolve(source_video)
    target_dir = PREVIEW_DIR / "shot_scrub" / safe_preview_name(manifest)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"shot_{index:04d}_{int(seconds * 1000):010d}.jpg"
    if target.exists():
        return rel(target)
    ffmpeg = local_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Run install_windows.bat to install local FFmpeg for shot previews.")
    command = [ffmpeg, "-y", "-ss", f"{seconds:.3f}", "-i", str(source), "-frames:v", "1", "-q:v", "4", str(target)]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg failed").strip())
    return rel(target)


def reference_regeneration_command(manifest_text: str, index: int) -> tuple[list[str], str]:
    manifest = resolve(manifest_text)
    _source_video, _fields, rows = read_manifest_details(manifest)
    if index < 0 or index >= len(rows):
        raise IndexError(f"Manifest row {index} is out of range.")
    row = rows[index]
    source = row.get("source_reference", "")
    output = row.get("color_reference", "")
    if not source or not output:
        raise RuntimeError("Manifest row must have source_reference and color_reference.")
    values = APP.settings.get("references", {})
    config = current_config()
    workflow = values.get("workflow") or default_qwen_workflow(config)
    if not workflow:
        raise RuntimeError("No Qwen Image Edit workflow found. Install/configure ComfyUI first.")
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPTS / "generate_single_reference.py"),
        "--source-image",
        source,
        "--output",
        output,
        "--workflow",
        workflow,
        "--comfy-url",
        values.get("comfy_url") or config.get("comfy_url", "http://127.0.0.1:8188"),
        "--comfy-dir",
        config.get("comfy_dir", str(ROOT / "tools" / "comfyui")),
        "--comfy-output-root",
        values.get("comfy_output_root") or str(Path(config.get("comfy_dir", str(ROOT / "tools" / "comfyui"))) / "output"),
        "--model-backend",
        values.get("model_backend", "gguf"),
        "--gguf-model",
        values.get("gguf_model", "qwen-image-edit-2511-Q4_K_M.gguf"),
        "--prompt",
        values.get("prompt", REFERENCE_PROMPT),
        "--prompt-suffix",
        values.get("prompt_suffix", REFERENCE_PROMPT_SUFFIX),
        "--load-image-node-id",
        values.get("load_image_node_id", "auto"),
        "--save-node-id",
        values.get("save_node_id", "auto"),
        "--force",
    ]
    if values.get("prompt_node_id"):
        cmd.extend(["--prompt-node-id", values["prompt_node_id"]])
    if row.get("prompt"):
        cmd.extend(["--add-prompt", row["prompt"]])
    return cmd, output


def regenerate_reference_image(manifest_text: str, index: int) -> dict[str, str]:
    cmd, output = reference_regeneration_command(manifest_text, index)
    APP.log.append("> " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in result.stdout.splitlines():
        APP.log.append(line)
    if result.returncode != 0:
        raise RuntimeError(f"Reference regeneration failed with exit code {result.returncode}.")
    APP.log.append(f"Regenerated colour reference for shot {index + 1}: {output}")
    return {"color_reference": output}


def source_previews(source_text: str) -> list[str]:
    signature = source_signature(source_text)
    if signature is None:
        if source_text:
            source = resolve(source_text)
            APP.log.append(f"Source preview skipped; file was not found or is not a supported video: {source}")
        return []
    return list(source_previews_cached(*signature))


@lru_cache(maxsize=16)
def source_previews_cached(source_path: str, _size: int, mtime_ns: int) -> tuple[str, ...]:
    source = Path(source_path)
    safe = safe_preview_name(source)
    target_dir = PREVIEW_DIR / safe
    frames = [target_dir / f"preview_{index}.jpg" for index in range(5)]
    try:
        if all(frame.exists() and frame.stat().st_mtime_ns >= mtime_ns for frame in frames):
            return tuple(rel(frame) for frame in frames)
        APP.log.append(f"Generating 5 source previews from: {source}")
        generate_video_previews(source, target_dir)
        APP.log.append(f"Generated source previews in: {target_dir}")
        return tuple(rel(frame) for frame in frames if frame.exists())
    except Exception as exc:
        APP.log.append(f"Could not generate source previews: {exc}")
        return ()


def source_info(source_text: str) -> dict[str, str]:
    signature = source_signature(source_text)
    if signature is None:
        if source_text:
            source = resolve(source_text)
            APP.log.append(f"Source info skipped; file was not found or is not a supported video: {source}")
        return {}
    return dict(source_info_cached(*signature))


def source_monochrome(source_text: str) -> bool:
    signature = source_signature(source_text)
    if signature is None:
        return True
    return source_monochrome_cached(*signature)


@lru_cache(maxsize=16)
def source_monochrome_cached(source_path: str, size: int, mtime_ns: int) -> bool:
    try:
        from PIL import Image, ImageChops, ImageStat
    except ModuleNotFoundError:
        return True
    previews = source_previews_cached(source_path, size, mtime_ns)
    if not previews:
        return True
    scores = []
    for preview_path in previews:
        try:
            image = Image.open(resolve(preview_path)).convert("RGB").resize((160, 90))
            r, g, b = image.split()
            rg = ImageStat.Stat(ImageChops.difference(r, g)).mean[0]
            rb = ImageStat.Stat(ImageChops.difference(r, b)).mean[0]
            gb = ImageStat.Stat(ImageChops.difference(g, b)).mean[0]
            scores.append((rg + rb + gb) / 3)
        except Exception:
            continue
    return (sum(scores) / max(1, len(scores))) < 2.5


@lru_cache(maxsize=16)
def source_info_cached(source_path: str, size: int, _mtime_ns: int) -> tuple[tuple[str, str], ...]:
    source = Path(source_path)
    APP.log.append(f"Probing source file info: {source}")
    info: dict[str, str] = {"file": rel(source), "size": human_size(size)}
    info.update(ffprobe_info(source))
    return tuple(info.items())


def current_crop_values() -> tuple[int, int, int, int]:
    values = APP.settings.get("outpaint", {}) if "APP" in globals() else {}
    return tuple(max(0, int(float(values.get(key, "0") or 0))) for key in ("crop_left", "crop_right", "crop_top", "crop_bottom"))  # type: ignore[return-value]


def aspect_preview(source_text: str, aspect: str) -> str:
    signature = source_signature(source_text)
    if signature is None:
        return ""
    return aspect_preview_cached(signature[0], signature[1], signature[2], aspect, current_crop_values(), 10.0)


def aspect_preview_at(source_text: str, aspect: str, seconds: float) -> str:
    signature = source_signature(source_text)
    if signature is None:
        return ""
    return aspect_preview_cached(signature[0], signature[1], signature[2], aspect, current_crop_values(), round(max(0.0, seconds), 3))


@lru_cache(maxsize=96)
def aspect_preview_cached(source_path: str, _size: int, mtime_ns: int, aspect: str, crops: tuple[int, int, int, int], seconds: float) -> str:
    source = Path(source_path)
    source_frame = extract_video_frame_at(source, ASPECT_PREVIEW_DIR / "frames", f"aspect_{int(seconds * 1000):010d}", seconds)
    if not source_frame:
        return ""
    crop_slug = "" if not any(crops) else "_crop" + "-".join(str(v) for v in crops)
    target = ASPECT_PREVIEW_DIR / f"{safe_preview_name(source)}_{aspect_slug(aspect)}{crop_slug}_{int(seconds * 1000):010d}.jpg"
    if target.exists() and target.stat().st_mtime_ns >= mtime_ns:
        return rel(target)
    try:
        from PIL import Image, ImageOps
    except ModuleNotFoundError:
        APP.log.append("Pillow is not available; using FFmpeg for the aspect preview.")
        return ffmpeg_aspect_preview(source, target, aspect, mtime_ns) or source_frame
    ratio = parse_aspect(aspect)
    image = Image.open(resolve(source_frame)).convert("RGB")
    width, height = image.size
    left, right, top, bottom = crops
    crop_box = (min(left, width - 2), min(top, height - 2), max(min(width - right, width), left + 2), max(min(height - bottom, height), top + 2))
    image = image.crop(crop_box)
    width, height = image.size
    if width / height < ratio:
        target_h = height
        target_w = int(round(height * ratio))
    else:
        target_w = width
        target_h = int(round(width / ratio))
    canvas = patterned_canvas(target_w, target_h)
    canvas.paste(image, ((target_w - width) // 2, (target_h - height) // 2))
    preview = ImageOps.contain(canvas, (960, 540), Image.Resampling.LANCZOS)
    target.parent.mkdir(parents=True, exist_ok=True)
    preview.save(target, quality=90)
    return rel(target)


def patterned_canvas(width: int, height: int):
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (width, height), (21, 39, 43))
    draw = ImageDraw.Draw(canvas)
    spacing = max(14, min(width, height) // 28)
    line_color = (58, 139, 128)
    accent = (211, 164, 58)
    for offset in range(-height, width, spacing):
        draw.line((offset, 0, offset + height, height), fill=line_color, width=max(1, spacing // 9))
    for offset in range(0, width + height, spacing * 4):
        draw.line((offset, 0, offset - height, height), fill=accent, width=max(1, spacing // 12))
    return canvas


def ffmpeg_aspect_preview(source: Path, target: Path, aspect: str, mtime_ns: int) -> str:
    ffmpeg = local_tool("ffmpeg")
    dims = video_dimensions(source)
    if not ffmpeg or not dims:
        return ""
    source_w, source_h = dims
    ratio = parse_aspect(aspect)
    if source_w / source_h < ratio:
        canvas_h = source_h
        canvas_w = int(round(source_h * ratio))
    else:
        canvas_w = source_w
        canvas_h = int(round(source_w / ratio))
    scale = min(960 / canvas_w, 540 / canvas_h, 1.0)
    out_w = max(2, even_int(canvas_w * scale))
    out_h = max(2, even_int(canvas_h * scale))
    scaled_w = max(2, even_int(source_w * scale))
    scaled_h = max(2, even_int(source_h * scale))
    target.parent.mkdir(parents=True, exist_ok=True)
    filter_text = (
        f"scale={scaled_w}:{scaled_h}[src];"
        f"color=c=0x15272b:s={out_w}x{out_h}[bg];"
        f"[bg]geq=r='34+34*mod(floor((X+Y)/18)\\,2)':g='62+48*mod(floor((X+Y)/18)\\,2)':b='67+40*mod(floor((X+Y)/18)\\,2)'[pat];"
        f"[pat][src]overlay=(W-w)/2:(H-h)/2"
    )
    command = [ffmpeg, "-y", "-ss", "10", "-i", str(source), "-frames:v", "1", "-vf", filter_text, "-q:v", "3", str(target)]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        command[3] = "0"
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        APP.log.append(f"Could not generate aspect preview: {(result.stderr or result.stdout).strip()}")
    return rel(target) if result.returncode == 0 and target.exists() and target.stat().st_mtime_ns >= mtime_ns else ""


def video_dimensions(source: Path) -> tuple[int, int] | None:
    resolution = ffprobe_info(source).get("resolution", "")
    if "x" not in resolution:
        return None
    left, right = resolution.split("x", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None


def even_int(value: float) -> int:
    return max(2, int(round(value / 2)) * 2)


def parse_aspect(value: str) -> float:
    if ":" in value:
        left, right = value.split(":", 1)
        return float(left) / float(right)
    return float(value)


def file_preview(path: Path) -> str:
    if path.suffix.lower() in IMAGE_EXTS:
        return rel(path)
    if path.suffix.lower() in VIDEO_EXTS:
        signature = source_signature(str(path))
        if signature is None:
            return ""
        return file_preview_cached(*signature)
    return ""


@lru_cache(maxsize=128)
def file_preview_cached(source_path: str, _size: int, _mtime_ns: int) -> str:
    return extract_video_frame(Path(source_path), FILE_PREVIEW_DIR, "thumb")


def extract_video_frame(source: Path, target_dir: Path, suffix: str) -> str:
    return extract_video_frame_at(source, target_dir, suffix, 10.0)


def extract_video_frame_at(source: Path, target_dir: Path, suffix: str, seconds: float) -> str:
    ffmpeg = local_tool("ffmpeg")
    if not ffmpeg:
        return ""
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{safe_preview_name(source)}_{suffix}.jpg"
    command = [ffmpeg, "-y", "-ss", f"{max(0.0, seconds):.3f}", "-i", str(source), "-frames:v", "1", "-q:v", "4", str(target)]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        command = [ffmpeg, "-y", "-ss", "0", "-i", str(source), "-frames:v", "1", "-q:v", "4", str(target)]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    return rel(target) if result.returncode == 0 and target.exists() else ""


def ffprobe_info(source: Path) -> dict[str, str]:
    found = local_tool("ffprobe")
    if not found:
        return {"codec_note": "Run install_windows.bat to install local FFmpeg/ffprobe for codec and colour metadata."}
    command = [
        str(found),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(source),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return {"codec_note": (result.stderr or "ffprobe failed").strip()}
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"codec_note": "ffprobe returned invalid JSON."}
    out: dict[str, str] = {}
    streams = data.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if video:
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
        if width and height:
            out["resolution"] = f"{width}x{height}"
            out["aspect"] = f"{width / height:.3f}:1"
        fps = parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate"))
        if fps:
            out["frame_rate"] = f"{fps:.3f} fps"
        if video.get("nb_frames"):
            out["frames"] = f"{int(video['nb_frames']):,}"
        if video.get("codec_name"):
            out["video_codec"] = str(video["codec_name"])
        if video.get("pix_fmt"):
            out["pixel_format"] = str(video["pix_fmt"])
        color_parts = [video.get("color_space"), video.get("color_transfer"), video.get("color_primaries"), video.get("color_range")]
        color = " / ".join(str(part) for part in color_parts if part)
        if color:
            out["colour"] = color
        if video.get("bit_rate"):
            out["video_bitrate"] = human_bitrate(video["bit_rate"])
    if audio:
        audio_parts = [audio.get("codec_name"), audio.get("sample_rate"), audio.get("channels")]
        values = [str(part) for part in audio_parts if part]
        if values:
            out["audio"] = ", ".join(values)
    fmt = data.get("format") or {}
    if fmt.get("duration"):
        try:
            out["duration"] = format_duration(float(fmt["duration"]))
        except ValueError:
            pass
    if fmt.get("format_name"):
        out["container"] = str(fmt["format_name"])
    if fmt.get("bit_rate"):
        out["overall_bitrate"] = human_bitrate(fmt["bit_rate"])
    return out


def local_tool(name: str) -> str | None:
    exe = f"{name}.exe" if os.name == "nt" else name
    local = ROOT / ".cache" / "tools" / "ffmpeg" / exe
    if local.exists():
        return str(local)
    return shutil.which(name)


def parse_rate(value: str | None) -> float | None:
    if not value:
        return None
    if "/" in value:
        left, right = value.split("/", 1)
        try:
            denominator = float(right)
            return float(left) / denominator if denominator else None
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def human_bitrate(value: str | int | float) -> str:
    try:
        bits = float(value)
    except (TypeError, ValueError):
        return str(value)
    if bits >= 1_000_000:
        return f"{bits / 1_000_000:.2f} Mbps"
    if bits >= 1_000:
        return f"{bits / 1_000:.1f} Kbps"
    return f"{bits:.0f} bps"


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def safe_preview_name(path: Path) -> str:
    text = str(path.resolve()).replace(":", "").replace("\\", "_").replace("/", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:180]


def media_clip_path(source: Path, start: float, end: float, key: str = "") -> Path:
    ffmpeg = local_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Run install_windows.bat to install local FFmpeg for shot video previews.")
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(source)
    duration = max(0.041, end - start)
    stat = source.stat()
    digest = hashlib.sha1(
        f"{source.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{start:.3f}|{end:.3f}|{key}".encode("utf-8", errors="ignore")
    ).hexdigest()[:20]
    target = MEDIA_CLIP_DIR / f"{safe_preview_name(source)[:80]}_{digest}.mp4"
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.mp4")
    if partial.exists():
        partial.unlink()
    command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{max(0.0, start):.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-an",
        "-vf",
        "setpts=PTS-STARTPTS,setsar=1",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(partial),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        if partial.exists():
            partial.unlink()
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg clip extraction failed").strip())
    partial.replace(target)
    return target


def generate_video_previews(source: Path, target_dir: Path) -> None:
    ffmpeg = local_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Run install_windows.bat to install local FFmpeg for source previews.")
    info = ffprobe_info(source)
    duration = parse_duration(info.get("duration"))
    target_dir.mkdir(parents=True, exist_ok=True)
    if duration:
        positions = [duration * fraction for fraction in (0.08, 0.26, 0.44, 0.62, 0.80)]
    else:
        positions = [0, 10, 20, 30, 40]
    for index, seconds in enumerate(positions):
        out = target_dir / f"preview_{index}.jpg"
        command = [
            ffmpeg,
            "-y",
            "-ss",
            f"{seconds:.3f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(out),
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            APP.log.append(f"Preview frame {index + 1} failed: {(result.stderr or result.stdout).strip()}")


def parse_duration(value: str | None) -> float | None:
    if not value:
        return None
    parts = value.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(value)
    except ValueError:
        return None


def browse_path(kind: str, current: str = "") -> str:
    initial = ROOT
    if current:
        current_path = resolve(current)
        initial = current_path if current_path.is_dir() else current_path.parent
        if not initial.exists():
            initial = ROOT
    if os.name == "nt":
        return browse_path_windows(kind, initial)
    if sys.platform == "darwin":
        return browse_path_macos(kind, initial)
    return browse_path_linux(kind, initial)


def browse_path_windows(kind: str, initial: Path) -> str:
    initial_text = str(initial).replace("'", "''")
    if kind == "folder":
        script = f"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.SelectedPath = '{initial_text}'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{ [Console]::Out.Write($dialog.SelectedPath) }}
"""
    elif kind == "save":
        script = f"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.SaveFileDialog
$dialog.InitialDirectory = '{initial_text}'
$dialog.Filter = 'Video files (*.mp4;*.mov;*.mkv;*.webm)|*.mp4;*.mov;*.mkv;*.webm|All files (*.*)|*.*'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{ [Console]::Out.Write($dialog.FileName) }}
"""
    else:
        script = f"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.InitialDirectory = '{initial_text}'
$dialog.Filter = 'Media/workflow files (*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v;*.png;*.jpg;*.jpeg;*.json;*.csv)|*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v;*.png;*.jpg;*.jpeg;*.json;*.csv|All files (*.*)|*.*'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{ [Console]::Out.Write($dialog.FileName) }}
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "Windows file dialog failed.").strip())
    selected = result.stdout.strip()
    if selected:
        APP.log.append(f"Browse selected: {selected}")
    else:
        APP.log.append("Browse cancelled.")
    return rel(Path(selected)) if selected else ""


def browse_path_macos(kind: str, initial: Path) -> str:
    initial_script = applescript_quote(str(initial))
    if kind == "folder":
        script = f'set chosen to choose folder with prompt "Choose folder" default location POSIX file {initial_script}\nPOSIX path of chosen'
    elif kind == "save":
        script = f'set chosen to choose file name with prompt "Choose output path" default location POSIX file {initial_script}\nPOSIX path of chosen'
    else:
        script = f'set chosen to choose file with prompt "Choose file" default location POSIX file {initial_script}\nPOSIX path of chosen'
    result = subprocess.run(["osascript", "-e", script], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "User canceled" in stderr or "(-128)" in stderr:
            APP.log.append("Browse cancelled.")
            return ""
        raise RuntimeError(stderr or "macOS file dialog failed.")
    selected = result.stdout.strip()
    if selected:
        APP.log.append(f"Browse selected: {selected}")
    else:
        APP.log.append("Browse cancelled.")
    return rel(Path(selected)) if selected else ""


def applescript_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def browse_path_linux(kind: str, initial: Path) -> str:
    if shutil.which("zenity"):
        return browse_path_zenity(kind, initial)
    if shutil.which("kdialog"):
        return browse_path_kdialog(kind, initial)
    raise RuntimeError("No native file picker found. Install zenity or kdialog, or paste the path into the field.")


def browse_path_zenity(kind: str, initial: Path) -> str:
    command = ["zenity", "--file-selection", f"--filename={initial}/"]
    if kind == "folder":
        command.append("--directory")
    elif kind == "save":
        command.append("--save")
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        APP.log.append("Browse cancelled.")
        return ""
    selected = result.stdout.strip()
    if selected:
        APP.log.append(f"Browse selected: {selected}")
    return rel(Path(selected)) if selected else ""


def browse_path_kdialog(kind: str, initial: Path) -> str:
    if kind == "folder":
        command = ["kdialog", "--getexistingdirectory", str(initial)]
    elif kind == "save":
        command = ["kdialog", "--getsavefilename", str(initial)]
    else:
        command = ["kdialog", "--getopenfilename", str(initial)]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        APP.log.append("Browse cancelled.")
        return ""
    selected = result.stdout.strip()
    if selected:
        APP.log.append(f"Browse selected: {selected}")
    return rel(Path(selected)) if selected else ""


class Handler(BaseHTTPRequestHandler):
    server_version = "AIRemasterGUI/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(INDEX_HTML, "text/html; charset=utf-8")
        elif parsed.path == "/site.webmanifest":
            self.send_json(
                {
                    "name": "ARP - AI Remaster Pipeline",
                    "short_name": "ARP",
                    "start_url": "/",
                    "display": "standalone",
                    "background_color": "#101316",
                    "theme_color": "#2d8f7d",
                    "icons": [
                        {"src": "/media?path=assets/branding/arp-app-icon-192.png", "sizes": "192x192", "type": "image/png"},
                        {"src": "/media?path=assets/branding/arp-app-icon-512.png", "sizes": "512x512", "type": "image/png"},
                    ],
                }
            )
        elif parsed.path == "/api/state":
            self.send_json(APP.state())
        elif parsed.path == "/api/command":
            stage = parse_qs(parsed.query).get("stage", [""])[0]
            self.send_json({"command": APP.command_for(stage) if stage else []})
        elif parsed.path == "/api/existing-outputs":
            stage = parse_qs(parsed.query).get("stage", [""])[0]
            self.send_json({"paths": APP.existing_outputs(stage) if stage else []})
        elif parsed.path == "/api/comfy":
            url = parse_qs(parsed.query).get("url", ["http://127.0.0.1:8188"])[0].rstrip("/")
            try:
                with urlopen(url + "/queue", timeout=3) as response:
                    self.send_json({"ok": True, "queue": json.loads(response.read().decode("utf-8"))})
            except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/logfile":
            path = resolve(parse_qs(parsed.query).get("path", [""])[0])
            text = path.read_text(encoding="utf-8", errors="replace")[-12000:] if path.exists() else ""
            self.send_json({"text": text})
        elif parsed.path == "/api/shot-preview":
            query = parse_qs(parsed.query)
            try:
                path = preview_reference_frame(query.get("manifest", [""])[0], int(query.get("index", ["0"])[0]), float(query.get("time", ["0"])[0]))
                self.send_json({"ok": True, "path": path})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/aspect-preview":
            query = parse_qs(parsed.query)
            try:
                path = aspect_preview_at(APP.settings.get("global", {}).get("source", ""), APP.settings.get("outpaint", {}).get("target_aspect", "16:9"), float(query.get("time", ["0"])[0]))
                self.send_json({"ok": True, "path": path})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/media":
            query = parse_qs(parsed.query)
            path = resolve(unquote(query.get("path", [""])[0]))
            if "clip_start" in query or "clip_end" in query:
                try:
                    start = float(query.get("clip_start", ["0"])[0])
                    end = float(query.get("clip_end", [str(start + 0.041)])[0])
                    path = media_clip_path(path, start, end, query.get("clip_key", [""])[0])
                except Exception as exc:
                    APP.log.append(f"Shot video preview failed: {exc}")
                    self.send_error(404)
                    return
            self.send_media(path)
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        data = self.read_json()
        if parsed.path == "/api/settings":
            APP.update_settings(str(data.get("stage", "")), data.get("values", {}))
            self.send_json({"ok": True})
        elif parsed.path == "/api/run":
            if data.get("all"):
                ok, message = APP.run_all()
            else:
                ok, message = APP.run_stage(str(data.get("stage", "")))
            self.send_json({"ok": ok, "message": message})
        elif parsed.path == "/api/stop":
            APP.stop()
            self.send_json({"ok": True})
        elif parsed.path == "/api/shot-scrub":
            try:
                result = extract_reference_frame(str(data.get("manifest", "")), int(data.get("index", 0)), float(data.get("time", 0)))
                self.send_json({"ok": True, **result})
            except Exception as exc:
                APP.log.append(f"Shot scrub failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/shot-prompt":
            try:
                update_manifest_row(resolve(str(data.get("manifest", ""))), int(data.get("index", 0)), {"prompt": str(data.get("prompt", ""))})
                self.send_json({"ok": True})
            except Exception as exc:
                APP.log.append(f"Shot prompt save failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/shot-enabled":
            try:
                enabled = "true" if data.get("enabled") else "false"
                update_manifest_row(resolve(str(data.get("manifest", ""))), int(data.get("index", 0)), {"enabled": enabled})
                self.send_json({"ok": True})
            except Exception as exc:
                APP.log.append(f"Shot enabled save failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/shot-merge":
            try:
                result = merge_manifest_shots(str(data.get("manifest", "")), int(data.get("index", 0)))
                self.send_json({"ok": True, **result, "state": APP.state()})
            except Exception as exc:
                APP.log.append(f"Shot merge failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/shot-boundary":
            try:
                result = update_shot_boundary(str(data.get("manifest", "")), int(data.get("index", 0)), str(data.get("edge", "")), float(data.get("time", 0)))
                self.send_json({"ok": True, **result, "state": APP.state()})
            except Exception as exc:
                APP.log.append(f"Shot boundary update failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/reference-regenerate":
            try:
                ok, message = APP.run_reference_regeneration(str(data.get("manifest", "")), int(data.get("index", 0)))
                self.send_json({"ok": ok, "message": message, "state": APP.state() if ok else None, "error": "" if ok else message})
            except Exception as exc:
                APP.log.append(f"Reference regeneration failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/reference-delete":
            try:
                result = delete_color_reference(str(data.get("manifest", "")), int(data.get("index", 0)))
                self.send_json({"ok": True, **result, "state": APP.state()})
            except Exception as exc:
                APP.log.append(f"Reference delete failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/browse":
            try:
                selected = browse_path(str(data.get("kind", "file")), str(data.get("current", "")))
                self.send_json({"ok": True, "path": selected})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/browse-global-source":
            try:
                selected = browse_path("file", str(data.get("current", "")))
                if selected:
                    APP.update_settings("global", {"source": selected})
                self.send_json({"ok": True, "path": selected, "state": APP.state()})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/overview-clear":
            APP.clear_overview()
            self.send_json({"ok": True, "state": APP.state()})
        else:
            self.send_error(404)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_media(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        file_size = path.stat().st_size
        range_header = self.headers.get("Range", "")
        start = 0
        end = file_size - 1
        status = 200
        if range_header.startswith("bytes="):
            raw_range = range_header.removeprefix("bytes=").split(",", 1)[0].strip()
            left, _, right = raw_range.partition("-")
            try:
                if left:
                    start = int(left)
                    if right:
                        end = int(right)
                elif right:
                    suffix = int(right)
                    start = max(0, file_size - suffix)
                if start < 0 or end < start or start >= file_size:
                    raise ValueError
                end = min(end, file_size - 1)
                status = 206
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        try:
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARP - AI Remaster Pipeline</title>
<link rel="icon" type="image/png" href="/media?path=assets/branding/favicon.png">
<link rel="apple-touch-icon" href="/media?path=assets/branding/arp-app-icon-192.png">
<link rel="manifest" href="/site.webmanifest">
<style>
:root{color-scheme:dark;--bg:#101316;--panel:#171d22;--line:#2d3941;--text:#edf4f6;--muted:#9db0b8;--accent:#2d8f7d;--warn:#d3a43a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,Segoe UI,Roboto,Arial,sans-serif}
header{height:68px;display:flex;align-items:center;justify-content:space-between;padding:0 22px;border-bottom:1px solid var(--line);background:#12181d;position:sticky;top:0;z-index:2}
.brand{display:flex;gap:12px;align-items:center}.brand img{width:46px;height:46px;border-radius:6px}.brand-title{font-size:24px;font-weight:800;letter-spacing:.02em}.brand-subtitle{color:var(--muted);font-size:12px;margin-top:-3px}
.root{color:var(--muted);font-size:12px}.tabs{display:flex;gap:6px;padding:12px 18px 0;border-bottom:1px solid var(--line);background:#11171b;position:sticky;top:68px;z-index:2}
.tab{border:1px solid var(--line);border-bottom:0;border-radius:8px 8px 0 0;padding:9px 13px;background:#172027;color:var(--text);cursor:pointer}.tab.active{background:#24323a}
main{padding:18px}.grid{display:grid;grid-template-columns:360px minmax(260px,1fr) minmax(420px,1.25fr);gap:16px;align-items:start}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}.hero{font-size:32px;font-weight:800;margin:0 0 5px}.hero-logo{width:min(920px,100%);max-height:170px;object-fit:contain;display:block;margin:0 auto 14px}
label{display:block;color:var(--muted);font-size:12px;margin:10px 0 5px}input,select,textarea{width:100%;background:#202930;color:var(--text);border:1px solid #3b4b55;border-radius:6px;padding:8px}
button{background:#26333b;color:var(--text);border:1px solid #43545f;border-radius:6px;padding:8px 12px;cursor:pointer}button:hover{background:#30404a}.primary{background:var(--accent);border-color:#44a995;font-weight:700}.warn{background:#5b4322;border-color:#8a6830}
.row{display:flex;gap:8px;align-items:center}.row>*{flex:1}.field-row{display:flex;gap:8px;align-items:center}.field-row input,.field-row select{flex:1}.field-row button{flex:0 0 auto}.checks label{display:inline-flex;gap:6px;align-items:center;margin-right:12px}.checks input{width:auto}
.files{max-height:62vh;overflow:auto}.file{display:grid;grid-template-columns:74px 1fr;gap:10px;align-items:center;padding:8px;border-bottom:1px solid #27333a;cursor:pointer;color:#cfe0e5}.file:hover{background:#202a31}.file.no-thumb{display:block}.file-thumb{width:74px;aspect-ratio:16/9;object-fit:cover;background:#050607;border:1px solid var(--line);border-radius:5px}.file-path{word-break:break-word}.output-list{margin:10px 0 0;padding-left:18px;color:#c5d5da;font-size:12px;word-break:break-word}
.preview img,.preview video{width:100%;max-height:62vh;object-fit:contain;background:#050607;border-radius:8px}.preview pre,pre.log{white-space:pre-wrap;background:#0b0e10;border:1px solid var(--line);border-radius:8px;padding:10px;max-height:230px;overflow:auto}.log-heading{display:flex;align-items:center;justify-content:space-between;gap:12px}.log-heading h3{margin:0}.log-error{color:#ff8f8f}.log-warn{color:#ffd27d}.log-ok{color:#75d6b9}.actions{display:flex;gap:8px;align-items:center;margin:14px 0}.actions button{flex:0 0 auto}.phase-progress{margin:12px 0}.phase-progress progress{width:100%;height:18px}.phase-progress div{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-bottom:4px}
.filmstrip{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:14px 0}.filmstrip img{width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:6px;background:#050607;border:1px solid var(--line)}
.global-top{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}.support-link{color:#9db0b8;text-decoration:none;font-size:12px;border:1px solid var(--line);border-radius:6px;padding:6px 8px;white-space:nowrap}.support-link:hover{color:var(--text);background:#202930}.coffee-icon{font-size:15px;margin-right:5px}.source-info{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:12px 0}.source-info div{background:#11181d;border:1px solid var(--line);border-radius:6px;padding:8px}.source-info span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}.source-info strong{display:block;margin-top:2px;font-size:13px;word-break:break-word}
table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid var(--line);padding:8px;text-align:left}th{color:#b8cbd1;background:#202a31}.status-ready{color:#75d6b9}.status-waiting{color:var(--warn)}
.shot-page{display:grid;grid-template-columns:minmax(280px,360px) 1fr;gap:16px;align-items:start}.shot-list{display:grid;gap:12px}.shot-card{display:grid;grid-template-columns:120px minmax(220px,.8fr) minmax(220px,.8fr) 1fr;gap:12px;align-items:start;background:#11181d;border:1px solid var(--line);border-radius:8px;padding:10px}.shot-number{font-size:18px;font-weight:800}.shot-time{color:var(--muted);font-size:12px}.shot-card img,.shot-card video{width:100%;aspect-ratio:16/9;object-fit:cover;background:#050607;border:1px solid var(--line);border-radius:6px}.shot-card textarea{min-height:88px;resize:vertical}.shot-card input[type=range]{padding:0}.shot-card label input[type=checkbox]{width:auto;margin-right:6px}.shot-tools{display:flex;gap:8px;margin-top:8px;align-items:center;flex-wrap:wrap}.shot-tools button{flex:0 0 auto}.shot-empty{color:var(--muted)}.inline-warning{border:1px solid #8a6a28;background:#211a0b;color:#ffd27d;border-radius:6px;padding:9px 10px;margin:10px 0;font-size:13px}.missing-image{display:grid;place-items:center;gap:6px;width:100%;aspect-ratio:16/9;background:#0c1114;border:1px dashed #45555e;border-radius:6px;color:#8fa3ab;text-align:center;font-size:12px}.missing-image .missing-icon{font-size:28px;line-height:1}.thumb-wrap{position:relative}.icon-button{position:absolute;right:6px;top:6px;width:28px;height:28px;padding:0;border-radius:999px;background:rgba(16,19,22,.82);border-color:#5b6b74;font-size:15px;line-height:1}.mini-progress{margin-top:8px;color:var(--muted);font-size:12px}.mini-progress progress{width:100%;height:12px}.spinner{width:14px;height:14px;border:2px solid #43545f;border-top-color:var(--accent);border-radius:50%;display:inline-block;animation:spin .8s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}
.editor-page{display:grid;grid-template-columns:minmax(300px,380px) 1fr;gap:16px;align-items:start}.editor-viewer video{width:100%;max-height:64vh;background:#050607;border:1px solid var(--line);border-radius:8px}.live-composite{position:relative;width:100%;aspect-ratio:16/9;background:#050607;border:1px solid var(--line);border-radius:8px;overflow:hidden}.live-composite video{position:absolute;inset:0;width:100%;height:100%;max-height:none;border:0;border-radius:0;object-fit:contain;background:transparent}.live-composite .live-outpaint{object-fit:cover}.live-composite .live-original{-webkit-mask-image:linear-gradient(90deg,transparent 0,#000 10%,#000 90%,transparent 100%);mask-image:linear-gradient(90deg,transparent 0,#000 10%,#000 90%,transparent 100%)}.live-composite .live-color{mix-blend-mode:color;opacity:.88}.layer-preview-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:12px}.layer-preview-grid video{width:100%;aspect-ratio:16/9;object-fit:cover;max-height:none}.layer-preview-grid .layer-original{box-shadow:0 0 0 1px rgba(255,255,255,.12), inset 0 0 28px rgba(255,255,255,.18)}.timeline{margin-top:12px;background:#0b0e10;border:1px solid var(--line);border-radius:8px;padding:10px}.timeline input[type=range]{padding:0}.track{display:grid;grid-template-columns:92px 1fr;gap:10px;align-items:center;margin:8px 0}.track-name{color:#b8cbd1;font-size:12px}.track-bar{height:18px;border:1px solid #3b4b55;border-radius:4px;background:#223038;position:relative;overflow:hidden}.track-bar::after{content:"";display:block;height:100%;width:100%;opacity:.85}.track-outpaint::after{background:#52636d}.track-original::after{background:#d6d6d6}.track-colour::after{background:#2d8f7d}.layer-grid{display:grid;gap:8px}.layer-item{background:#11181d;border:1px solid var(--line);border-radius:6px;padding:8px}.layer-item span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}.layer-item strong{display:block;word-break:break-word}.editor-controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}.editor-controls label{margin-top:0}
.hidden{display:none}.command{font-size:12px;color:#c5d5da;word-break:break-all}
</style>
</head>
<body>
<header><div class="brand"><img src="/media?path=assets/branding/arp-app-icon-192.png" alt=""><div><div class="brand-title">ARP</div><div class="brand-subtitle">AI Remaster Pipeline</div><div id="root" class="root"></div></div></div><div class="row" style="max-width:480px"><a class="support-link" href="https://buymeacoffee.com/davidaddis" target="_blank" rel="noreferrer"><span class="coffee-icon">☕</span>Buy me a coffee</a><button onclick="refresh(true)">Refresh</button><button class="warn" onclick="stopRun()">Stop</button></div></header>
<nav id="tabs" class="tabs"></nav>
<main id="app"></main>
<script>
let state=null, active='global', selected={}, lastRenderSignature='';
const media=p=>'/media?path='+encodeURIComponent(p);
const mediaClip=(p,start,end,key)=>'/media?path='+encodeURIComponent(p)+'&clip_start='+encodeURIComponent(start)+'&clip_end='+encodeURIComponent(end)+'&clip_key='+encodeURIComponent(key||'');
async function api(path, opts={}){const r=await fetch(path,{headers:{'Content-Type':'application/json'},...opts});return await r.json();}
async function refresh(force=false){const snap=captureScrollState(),editing=isEditingField(),mediaActive=hasMediaOnPage();state=await api('/api/state');pruneSelected();if(!availableTabs().includes(active))active='global';document.getElementById('root').textContent=state.root+(state.running?'  |  Running: '+state.running_stage:'');const sig=renderSignature();if(!force&&(editing||mediaActive||sig===lastRenderSignature)){updateRunLogs();return}drawTabs();draw(false);wireColourShotVideos();lastRenderSignature=sig;restoreScrollState(snap);}
function renderSignature(){if(!state)return '';return JSON.stringify({active,stages:state.stages,settings:state.settings,expected_outputs:state.expected_outputs,source_previews:state.source_previews,source_info:state.source_info,source_monochrome:state.source_monochrome,aspect_preview:state.aspect_preview,shot_views:state.shot_views,progress:state.progress,phase_progress:state.phase_progress,running:state.running,running_stage:state.running_stage,running_reference:state.running_reference})}
function hasMediaOnPage(){return active==='colour'||active==='recomp'?document.querySelectorAll('video').length>0:false}
function updateRunLogs(){document.querySelectorAll('[data-run-log]').forEach(el=>{const html=logHtml(state.log);if(el.innerHTML!==html)el.innerHTML=html})}
function availableTabs(){return ['global',...state.stages.map(s=>s.key),'settings']}
function drawTabs(){const tabs=availableTabs();const names={global:'Overview',settings:'Settings'};document.getElementById('tabs').innerHTML=tabs.map(t=>`<button class="tab ${active===t?'active':''}" onclick="active='${t}';draw();wireColourShotVideos();lastRenderSignature=renderSignature()">${names[t]||stage(t).title}</button>`).join('');}
function stage(k){return state.stages.find(s=>s.key===k)}
function settings(k){return state.settings[k]||{}}
function pruneSelected(){if(!state||!state.stages)return;for(const st of state.stages){if(selected[st.key]&&!st.files.some(f=>f.path===selected[st.key]))delete selected[st.key]}}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function draw(followLogs=false){if(active==='global')return drawGlobal(followLogs); if(active==='settings')return drawSettings(); if(active==='output')return drawOutput(); if(active==='shots')return drawShots(followLogs); if(active==='references')return drawReferences(followLogs); if(active==='colour')return drawColour(followLogs); if(active==='recomp')return drawRecomp(followLogs); return drawStage(stage(active),followLogs);}
function wireColourShotVideos(){if(active!=='colour')return;document.querySelectorAll('.shot-card video').forEach((video,index)=>{try{const url=new URL(video.getAttribute('src'),window.location.href),hash=url.hash||'';if(!hash.startsWith('#t='))return;const parts=hash.slice(3).split(',');if(parts.length<2)return;video.src=mediaClip(url.searchParams.get('path')||'',parts[0],parts[1],'colour_'+index);video.removeAttribute('data-cued')}catch{}})}
function drawGlobal(followLogs=false){const src=(state.settings.global&&state.settings.global.source)||'',colorize=(state.settings.global&&state.settings.global.colorize)!=='false';const thumbs=(state.source_previews||[]).map(p=>`<img src="${media(p)}" alt="">`).join('');const info=sourceInfoHtml(state.source_info||{});const gp=(state.phase_progress&&state.phase_progress.global)||{percent:0,label:'Waiting'};const mono=state.source_monochrome?'Source looks black and white':'Source appears to contain colour';document.getElementById('app').innerHTML=`<section class="card"><div class="global-top"><div><p class="hero">AI Remaster Pipeline</p><p>Choose the source material, then run or inspect each stage.</p></div><button type="button" onclick="clearOverview()">Clear</button></div><img class="hero-logo" src="/media?path=assets/branding/arp-logo-wide.png" alt="ARP - AI Remaster Pipeline"><label>Source material</label><div class="field-row"><input id="globalSource" value="${esc(src)}"><button type="button" onclick="browseGlobalSource()">Browse</button></div><div class="checks"><label><input id="globalColorize" type="checkbox" ${colorize?'checked':''}>Colorize</label><span class="shot-time">${esc(mono)}</span></div>${thumbs?`<div class="filmstrip">${thumbs}</div>`:''}${info}${progressHtml(gp.percent,gp.label)}<div class="actions"><button class="primary" onclick="runAll()">Run Whole Remaster</button><button class="warn" onclick="stopRun()" ${state.running?'':'disabled'}>Stop</button></div><table><tr><th>Stage</th><th>Status</th><th>Progress</th><th>Latest output</th></tr>${state.progress.map(p=>{const sp=stageProgressByTitle(p.stage);return `<tr><td>${p.stage}</td><td class="status-${p.status.toLowerCase()}">${p.status}</td><td>${progressHtml(sp.percent,sp.label)}</td><td>${esc(p.latest)}</td></tr>`}).join('')}</table>${runLogHtml()}</section>`;document.getElementById('globalSource').addEventListener('change',saveGlobal);document.getElementById('globalColorize').addEventListener('change',saveGlobalColorize);}
function sourceInfoHtml(info){const labels={resolution:'Resolution',aspect:'Aspect',duration:'Duration',frame_rate:'Frame rate',frames:'Frames',video_codec:'Video codec',pixel_format:'Pixel format',colour:'Color',audio:'Audio',container:'Container',overall_bitrate:'Overall bitrate',video_bitrate:'Video bitrate',size:'File size',codec_note:'Note'};const keys=['resolution','aspect','duration','frame_rate','frames','video_codec','pixel_format','colour','audio','container','overall_bitrate','video_bitrate','size','codec_note'];const items=keys.filter(k=>info[k]).map(k=>`<div><span>${labels[k]||k}</span><strong>${esc(info[k])}</strong></div>`).join('');return items?`<div class="source-info">${items}</div>`:''}
function fieldHtml(st,[key,label,kind,def]){const v=settings(st.key)[key]??def??'';if(kind.startsWith('select:')){return `<label>${label}</label><select data-field="${key}">${kind.slice(7).split('|').map(o=>`<option ${v===o?'selected':''}>${o}</option>`).join('')}</select>`}if(kind.startsWith('range:')){const [min,max,step]=kind.slice(6).split('|');return `<label>${label}: <span id="${key}Value">${esc(v)}</span></label><input data-field="${key}" data-kind="${kind}" type="range" min="${esc(min)}" max="${esc(max)}" step="${esc(step||'1')}" value="${esc(v)}" oninput="document.getElementById('${key}Value').textContent=this.value">`}const input=`<input data-field="${key}" data-kind="${kind}" type="${kind==='number'?'number':'text'}" step="any" value="${esc(v)}">`;if(['file','folder','save'].includes(kind)){return `<label>${label}</label><div class="field-row">${input}<button type="button" onclick="browseField('${st.key}','${key}','${kind}')">Browse</button></div>`}return `<label>${label}</label>${input}`}
function aspectPreviewHtml(st){if(st.key!=='outpaint')return '';const img=state.aspect_preview;const outputs=(state.expected_outputs&&state.expected_outputs.outpaint)||[],duration=parseDuration((state.source_info&&state.source_info.duration)||'0');return `<h3>Target Preview</h3>${img?`<img id="aspectPreviewImg" src="${media(img)}" alt="Target aspect preview">`:'<p>Choose source material on the Overview tab to preview the target frame.</p>'}${duration?`<label>Preview time: <span id="aspectPreviewLabel">${formatSeconds(10)}</span></label><input type="range" min="0" max="${duration}" step="0.041" value="${Math.min(10,duration)}" oninput="updateAspectPreview(this.value)">`:''}${outputs.length?`<h3>Output Path</h3><ul class="output-list">${outputs.map(p=>`<li>${esc(p)}</li>`).join('')}</ul>`:''}`}
function outpaintOverlapWarning(s){const value=Number(s.overlap_frames??8);return Number.isFinite(value)&&value<8?`<div class="inline-warning">Overlap below 8 frames can cause held-frame seams if LTX returns short chunks. 8 or 9 frames is recommended.</div>`:''}
function fileRow(st,f){const thumb=f.preview?`<img class="file-thumb" src="${media(f.preview)}" alt="">`:'';return `<div class="file ${thumb?'':'no-thumb'}" onclick="selected['${st.key}']='${esc(f.path)}';draw()">${thumb}<div class="file-path">${esc(f.path)}</div></div>`}
function drawStage(st,followLogs=false){const s=settings(st.key);const file=selected[st.key];const expected=(state.expected_outputs&&state.expected_outputs[st.key])||[];const sp=stageProgress(st.key);if(st.key==='outpaint')return drawOutpaint(st,s,expected,sp);document.getElementById('app').innerHTML=`<div class="grid"><section class="card"><h2>${st.title}</h2><p>${st.description}</p>${progressHtml(sp.percent,sp.label)}${st.fields.map(f=>fieldHtml(st,f)).join('')}${expected.length&&st.key!=='outpaint'?`<h3>Output Path</h3><ul class="output-list">${expected.map(p=>`<li>${esc(p)}</li>`).join('')}</ul>`:''}<div class="checks"><label><input data-field="force" type="checkbox" ${s.force==='true'?'checked':''}>Regenerate</label><label><input data-field="dry_run" type="checkbox" ${s.dry_run==='true'?'checked':''}>Dry run</label></div><div class="actions"><button class="primary" onclick="runStage('${st.key}')" ${state.running?'disabled':''}>Run ${st.title}</button><button class="warn" onclick="stopRun()" ${state.running?'':'disabled'}>Stop</button></div><div class="command" id="cmd"></div></section><section class="card files"><h3>Intermediate Files</h3>${st.files.map(f=>fileRow(st,f)).join('')||'<p>No files yet.</p>'}</section><section class="card preview">${aspectPreviewHtml(st)}<h3>${file?esc(file):'Preview'}</h3>${preview(file)}</section></div><section class="card" style="margin-top:16px">${runLogHtml()}</section>`;document.querySelectorAll('[data-field]').forEach(el=>el.addEventListener('change',()=>saveStage(st.key,true)));showCommand(st.key)}
function drawOutpaint(st,s,expected,sp){const mainFields=st.fields.filter(f=>!f[0].startsWith('crop_')),cropFields=st.fields.filter(f=>f[0].startsWith('crop_'));document.getElementById('app').innerHTML=`<div class="editor-page"><section class="card"><h2>${st.title}</h2><p>${st.description}</p>${progressHtml(sp.percent,sp.label)}${mainFields.map(f=>fieldHtml(st,f)).join('')}${outpaintOverlapWarning(s)}<h3>Source Crop</h3><p class="shot-empty">Crop away black borders before ARP expands the frame.</p><div class="editor-controls">${cropFields.map(f=>`<div>${fieldHtml(st,f)}</div>`).join('')}</div><div class="checks"><label><input data-field="force" type="checkbox" ${s.force==='true'?'checked':''}>Regenerate</label><label><input data-field="dry_run" type="checkbox" ${s.dry_run==='true'?'checked':''}>Dry run</label></div><div class="actions"><button class="primary" onclick="runStage('outpaint')" ${state.running?'disabled':''}>Run Outpainting</button><button class="warn" onclick="stopRun()" ${state.running?'':'disabled'}>Stop</button></div><div class="command" id="cmd"></div></section><section class="card preview">${aspectPreviewHtml(st)}</section></div><section class="card" style="margin-top:16px">${runLogHtml()}</section>`;document.querySelectorAll('[data-field]').forEach(el=>el.addEventListener('change',()=>saveStage('outpaint',true)));showCommand('outpaint')}
function drawShots(followLogs=false){const st=stage('shots'),s=settings('shots'),expected=(state.expected_outputs&&state.expected_outputs.shots)||[],sp=stageProgress('shots');document.getElementById('app').innerHTML=`<div class="shot-page"><section class="card"><h2>${st.title}</h2><p>${st.description}</p>${progressHtml(sp.percent,sp.label)}${st.fields.map(f=>fieldHtml(st,f)).join('')}${expected.length?`<h3>Output Path</h3><ul class="output-list">${expected.map(p=>`<li>${esc(p)}</li>`).join('')}</ul>`:''}<div class="checks"><label><input data-field="force" type="checkbox" ${s.force==='true'?'checked':''}>Regenerate</label><label><input data-field="dry_run" type="checkbox" ${s.dry_run==='true'?'checked':''}>Dry run</label></div><div class="actions"><button class="primary" onclick="runStage('shots')" ${state.running?'disabled':''}>Run Shot Detection</button><button class="warn" onclick="stopRun()" ${state.running?'':'disabled'}>Stop</button></div><div class="command" id="cmd"></div></section><section class="card"><h2>Shots</h2>${shotCards('shots')}</section></div><section class="card" style="margin-top:16px">${runLogHtml()}</section>`;document.querySelectorAll('[data-field]').forEach(el=>el.addEventListener('change',()=>saveStage('shots',true)));showCommand('shots')}
function drawReferences(followLogs=false){const st=stage('references'),s=settings('references'),expected=(state.expected_outputs&&state.expected_outputs.references)||[],sp=stageProgress('references');document.getElementById('app').innerHTML=`<div class="shot-page"><section class="card"><h2>${st.title}</h2><p>${st.description}</p>${progressHtml(sp.percent,sp.label)}${st.fields.map(f=>fieldHtml(st,f)).join('')}${expected.length?`<h3>Output Path</h3><ul class="output-list">${expected.slice(0,8).map(p=>`<li>${esc(p)}</li>`).join('')}${expected.length>8?`<li>${expected.length-8} more...</li>`:''}</ul>`:''}<div class="checks"><label><input data-field="force" type="checkbox" ${s.force==='true'?'checked':''}>Regenerate</label><label><input data-field="dry_run" type="checkbox" ${s.dry_run==='true'?'checked':''}>Dry run</label></div><div class="actions"><button class="primary" onclick="runStage('references')" ${state.running?'disabled':''}>Run Reference Generation</button><button class="warn" onclick="stopRun()" ${state.running?'':'disabled'}>Stop</button></div><div class="command" id="cmd"></div></section><section class="card"><h2>References</h2>${shotCards('references')}</section></div><section class="card" style="margin-top:16px">${runLogHtml()}</section>`;document.querySelectorAll('[data-field]').forEach(el=>el.addEventListener('change',()=>saveStage('references',true)));wireReferenceTimeControls();showCommand('references')}
function drawColour(followLogs=false){const st=stage('colour'),s=settings('colour'),expected=(state.expected_outputs&&state.expected_outputs.colour)||[],sp=stageProgress('colour');document.getElementById('app').innerHTML=`<div class="shot-page"><section class="card"><h2>${st.title}</h2><p>${st.description}</p>${progressHtml(sp.percent,sp.label)}${st.fields.map(f=>fieldHtml(st,f)).join('')}${expected.length?`<h3>Output Path</h3><ul class="output-list">${expected.map(p=>`<li>${esc(p)}</li>`).join('')}</ul>`:''}<div class="checks"><label><input data-field="force" type="checkbox" ${s.force==='true'?'checked':''}>Regenerate</label><label><input data-field="dry_run" type="checkbox" ${s.dry_run==='true'?'checked':''}>Dry run</label></div><div class="actions"><button class="primary" onclick="runStage('colour')" ${state.running?'disabled':''}>Run Colorization</button><button class="warn" onclick="stopRun()" ${state.running?'':'disabled'}>Stop</button></div><div class="command" id="cmd"></div></section><section class="card"><h2>Shot Segments</h2>${shotCards('colour')}</section></div><section class="card" style="margin-top:16px">${runLogHtml()}</section>`;document.querySelectorAll('[data-field]').forEach(el=>el.addEventListener('change',()=>saveStage('colour',true)));showCommand('colour')}
function drawRecomp(followLogs=false){const st=stage('recomp'),s=settings('recomp'),expected=(state.expected_outputs&&state.expected_outputs.recomp)||[],sp=stageProgress('recomp');const pathFields=['outpainted_video','source','colorized_video'];const controlFields=['feather_pixels','saturation','temperature','color_opacity','encoder'];document.getElementById('app').innerHTML=`<div class="editor-page"><section class="card"><h2>${st.title}</h2><p>${st.description}</p>${progressHtml(sp.percent,sp.label)}<div class="layer-grid"><div class="layer-item"><span>Top layer - Color blend</span><strong>${esc(s.colorized_video||'Colorized video not set')}</strong></div><div class="layer-item"><span>Middle layer</span><strong>${esc(s.source||'Original source not set')}</strong></div><div class="layer-item"><span>Bottom layer</span><strong>${esc(s.outpainted_video||'Outpainted video not set')}</strong></div></div>${pathFields.map(k=>fieldHtml(st,st.fields.find(f=>f[0]===k))).join('')}<h3>Blend Parameters</h3><div class="editor-controls">${controlFields.map(k=>`<div>${fieldHtml(st,st.fields.find(f=>f[0]===k))}</div>`).join('')}</div>${expected.length?`<h3>Output Path</h3><ul class="output-list">${expected.map(p=>`<li>${esc(p)}</li>`).join('')}</ul>`:''}<div class="checks"><label><input data-field="force" type="checkbox" ${s.force==='true'?'checked':''}>Regenerate</label><label><input data-field="dry_run" type="checkbox" ${s.dry_run==='true'?'checked':''}>Dry run</label></div><div class="actions"><button class="primary" onclick="runStage('recomp')" ${state.running?'disabled':''}>Run Recomposition</button><button class="warn" onclick="stopRun()" ${state.running?'':'disabled'}>Stop</button></div><div class="command" id="cmd"></div></section><section class="card editor-viewer"><h2>Live Composite Preview</h2>${liveCompositeHtml(s)}${layerPreviewHtml(s)}<div class="timeline"><input id="recompScrub" type="range" min="0" max="1000" value="0" oninput="scrubEditorVideo(this.value)"><div class="track"><div class="track-name">Color</div><div class="track-bar track-colour"></div></div><div class="track"><div class="track-name">Original</div><div class="track-bar track-original"></div></div><div class="track"><div class="track-name">Outpainted</div><div class="track-bar track-outpaint"></div></div></div></section></div><section class="card" style="margin-top:16px">${runLogHtml()}</section>`;document.querySelectorAll('[data-field]').forEach(el=>el.addEventListener('change',()=>saveStage('recomp',true)));wireEditorVideo();showCommand('recomp')}
function drawOutput(){const expected=(state.expected_outputs&&state.expected_outputs.output)||[],path=expected[0]||settings('recomp').output||'';document.getElementById('app').innerHTML=`<section class="card editor-viewer"><h2>Output</h2>${path?`<video src="${media(path)}" controls preload="metadata"></video><h3>Final output</h3><ul class="output-list"><li>${esc(path)}</li></ul>`:'<p class="shot-empty">Run Recomposition to create the final movie.</p>'}</section>`}
function videoEditorHtml(path){return path?`<video id="recompVideo" src="${media(path)}" controls preload="metadata"></video>`:'<p class="shot-empty">Run recomposition to preview the final movie, or set one of the layer videos.</p>'}
function liveCompositeHtml(s){if(!s.outpainted_video&&!s.source&&!s.colorized_video)return '<p class="shot-empty">Run the earlier phases to preview the live composite.</p>';return `<div class="live-composite">${s.outpainted_video?`<video id="recompVideo" class="sync-layer-video live-outpaint" src="${media(s.outpainted_video)}" controls preload="metadata"></video>`:''}${s.source?`<video class="sync-layer-video live-original" src="${media(s.source)}" muted preload="metadata"></video>`:''}${s.colorized_video?`<video class="sync-layer-video live-color" src="${media(s.colorized_video)}" muted preload="metadata"></video>`:''}</div>`}
function layerPreviewHtml(s){return `<div class="layer-preview-grid"><div><label>Outpainted</label>${layerVideo(s.outpainted_video,'layer-outpaint')}</div><div><label>Original, feathered</label>${layerVideo(s.source,'layer-original')}</div><div><label>Color</label>${layerVideo(s.colorized_video,'layer-colour')}</div></div>`}
function layerVideo(path,cls){return path?`<video class="sync-layer-video ${cls}" src="${media(path)}" muted preload="metadata"></video>`:missingImage('Video not present')}
function wireEditorVideo(){const v=document.getElementById('recompVideo'),s=document.getElementById('recompScrub'),layers=[...document.querySelectorAll('.sync-layer-video')].filter(item=>item!==v);if(!v||!s)return;const sync=(force=false)=>{for(const item of layers){if(item.readyState&&Math.abs((item.currentTime||0)-(v.currentTime||0))>(force?0.02:0.18)){try{item.currentTime=v.currentTime||0}catch{}}}};v.addEventListener('loadedmetadata',()=>{s.value=0;sync(true)});v.addEventListener('play',()=>{sync(true);layers.forEach(item=>item.play().catch(()=>{}))});v.addEventListener('pause',()=>layers.forEach(item=>item.pause()));v.addEventListener('seeking',()=>sync(true));v.addEventListener('timeupdate',()=>{if(v.duration&&!s.matches(':active'))s.value=Math.round((v.currentTime/v.duration)*1000);sync(false)});v.addEventListener('ratechange',()=>layers.forEach(item=>item.playbackRate=v.playbackRate))}
function scrubEditorVideo(value){const v=document.getElementById('recompVideo');if(v&&v.duration){v.currentTime=(Number(value)||0)/1000*v.duration;document.querySelectorAll('.sync-layer-video').forEach(item=>{try{item.currentTime=v.currentTime}catch{}})}}
function shotCards(mode){const view=state.shot_views||{},rows=view[mode]||[],manifest=view[mode+'_manifest']||'';if(!rows.length)return `<p class="shot-empty">No shot manifest yet. Run Shot Detection first.</p>`;return `<div class="shot-list">${rows.map(row=>shotCard(mode,manifest,row)).join('')}</div>`}
function shotCard(mode,manifest,row){const src=row.source_reference||'',col=row.color_reference||'',idx=row.index,slider=`shotSlider_${mode}_${idx}`,label=`shotLabel_${mode}_${idx}`,img=`shotImg_${mode}_${idx}`;const canScrub=mode==='shots';const enabled=String(row.enabled||'true').toLowerCase()!=='false';const regen=mode==='references'&&state.running_reference&&state.running_reference.index===idx&&state.running_reference.manifest===manifest;const rp=stageProgress('references');const srcReady=src&&row.source_reference_mtime,colReady=col&&row.color_reference_mtime;const srcUrl=srcReady?media(src)+'&t='+(row.source_reference_mtime||0):'';const colUrl=colReady?media(col)+'&t='+(row.color_reference_mtime||0):'';const colourVideo=(state.expected_outputs&&state.expected_outputs.colour&&state.expected_outputs.colour[0])||settings('recomp').colorized_video||'';if(mode==='shots'){return `<article class="shot-card"><div><div class="shot-number">Shot ${idx+1}</div><div class="shot-time">${esc(row.start_label)} to ${esc(row.end_label)}</div><label><input type="checkbox" ${enabled?'checked':''} onchange="saveShotEnabled('${esc(manifest)}',${idx},this.checked)"> Use shot</label><div class="shot-tools">${row.can_merge_next?`<button type="button" onclick="mergeShot('${esc(manifest)}',${idx})">Merge Next</button>`:''}</div></div><div><label>Start frame ${row.start_frame??''}</label>${row.start_preview?`<img src="${media(row.start_preview)}" alt="">`:missingImage('Image not present')}<input type="range" min="${Math.max(0,Number(row.start)-1)}" max="${row.end}" step="0.041" value="${row.start}" ${idx===0?'disabled':''} onchange="setShotBoundary('${esc(manifest)}',${idx},'start',this.value)"><div class="shot-tools"><button type="button" ${idx===0?'disabled':''} onclick="nudgeShotBoundary('${esc(manifest)}',${idx},'start',-1)">-1 frame</button><button type="button" ${idx===0?'disabled':''} onclick="nudgeShotBoundary('${esc(manifest)}',${idx},'start',1)">+1 frame</button></div></div><div><label>Middle</label>${row.middle_preview?`<img src="${media(row.middle_preview)}" alt="">`:missingImage('Image not present')}</div><div><label>End frame ${row.end_frame??''}</label>${row.end_preview?`<img src="${media(row.end_preview)}" alt="">`:missingImage('Image not present')}<input type="range" min="${row.start}" max="${Number(row.end)+1}" step="0.041" value="${row.end}" onchange="setShotBoundary('${esc(manifest)}',${idx},'end',this.value)"><div class="shot-tools"><button type="button" onclick="nudgeShotBoundary('${esc(manifest)}',${idx},'end',-1)">-1 frame</button><button type="button" onclick="nudgeShotBoundary('${esc(manifest)}',${idx},'end',1)">+1 frame</button></div></div></article>`}if(mode==='colour'){const start=Math.max(0,Number(row.start)||0).toFixed(3),end=Math.max(0,Number(row.end)||0).toFixed(3);return `<article class="shot-card"><div><div class="shot-number">Shot ${idx+1}</div><div class="shot-time">${esc(row.start_label)} to ${esc(row.end_label)}</div><label><input type="checkbox" ${enabled?'checked':''} onchange="saveShotEnabled('${esc(manifest)}',${idx},this.checked)"> Use shot</label><p class="shot-empty">${enabled?(colReady?'Ready for Deep Exemplar':'Missing color reference'):'Disabled in manifest'}</p></div><div><label>Color reference</label>${colReady?`<img src="${colUrl}" alt="">`:missingImage('Image not present')}</div><div><label>Colorized shot video</label>${colourVideo?`<video src="${media(colourVideo)}#t=${start},${end}" controls preload="metadata"></video>`:missingImage('Video not present')}</div><div><label>Segment</label><p class="shot-time">Deep Exemplar uses this reference for the selected shot range.</p></div></article>`}const regenProgress=regen?`<div class="mini-progress"><div>${esc(rp.label||'Regenerating reference')}</div><progress value="${Math.max(5,Math.min(100,Number(rp.percent)||5))}" max="100"></progress></div>`:'';const prompt=mode==='references'?`<label>Shot prompt</label><textarea data-shot-prompt="${idx}" onblur="saveShotPrompt('${esc(manifest)}',${idx},this.value)" placeholder="Optional extra direction for this shot">${esc(row.prompt||'')}</textarea><div class="shot-tools"><button type="button" onclick="regenerateReference('${esc(manifest)}',${idx})" ${state.running?'disabled':''}>${regen?'Regenerating...':'Regenerate Reference'}</button>${regen?'<span class="spinner" aria-label="In progress"></span>':''}</div>${regenProgress}`:'';const color=mode==='references'?`<div><label>Qwen color reference</label>${colReady?`<div class="thumb-wrap"><img src="${colUrl}" alt=""><button class="icon-button" type="button" title="Delete color reference" onclick="deleteReference('${esc(manifest)}',${idx})">&#128465;</button></div>`:missingImage('Image not present')}</div>`:'';return `<article class="shot-card"><div><div class="shot-number">Shot ${idx+1}</div><div class="shot-time">${esc(row.start_label)} to ${esc(row.end_label)}</div><label><input type="checkbox" ${enabled?'checked':''} onchange="saveShotEnabled('${esc(manifest)}',${idx},this.checked)"> Use shot</label><label>Reference time</label><input id="${slider}" type="range" min="${row.start}" max="${row.end}" step="0.041" value="${row.selected_time}" disabled oninput="updateShotPreview('${esc(manifest)}',${idx},this.value,'${img}','${label}')"><div class="shot-time" id="${label}">${esc(row.selected_label)}</div></div><div><label>B&W screenshot</label>${srcReady?`<img id="${img}" src="${srcUrl}" alt="">`:missingImage('Image not present')}</div>${color}<div>${prompt}</div></article>`}
function missingImage(text){return `<div class="missing-image" role="img" aria-label="${esc(text)}"><div class="missing-icon">□</div><div>${esc(text)}</div></div>`}
function wireReferenceTimeControls(){const manifest=(state.shot_views&&state.shot_views.references_manifest)||'';for(const row of ((state.shot_views&&state.shot_views.references)||[])){const slider=document.getElementById(`shotSlider_references_${row.index}`);if(!slider)continue;slider.disabled=false;slider.min=row.start;slider.max=row.end;let tools=slider.parentElement.querySelector('.reference-time-tools');if(!tools){tools=document.createElement('div');tools.className='shot-tools reference-time-tools';slider.parentElement.appendChild(tools)}tools.innerHTML=`<button type="button" onclick="scrubShot('${esc(manifest)}',${row.index},document.getElementById('shotSlider_references_${row.index}').value)">Use Frame</button>`}}
function formatSeconds(value){const total=Math.max(0,Number(value)||0),h=Math.floor(total/3600),m=Math.floor((total%3600)/60),s=(total%60).toFixed(3).padStart(6,'0');return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${s}`}
function parseDuration(value){const text=String(value||'').trim();if(!text)return 0;const parts=text.split(':').map(Number);if(parts.length===3)return parts[0]*3600+parts[1]*60+parts[2];if(parts.length===2)return parts[0]*60+parts[1];const n=Number(text);return Number.isFinite(n)?n:0}
let aspectPreviewTimer=null;
function updateAspectPreview(time){const label=document.getElementById('aspectPreviewLabel');if(label)label.textContent=formatSeconds(time);clearTimeout(aspectPreviewTimer);aspectPreviewTimer=setTimeout(async()=>{const r=await api('/api/aspect-preview?time='+encodeURIComponent(time));const img=document.getElementById('aspectPreviewImg');if(r.ok&&r.path&&img)img.src=media(r.path)+'&t='+Date.now()},160)}
async function scrubShot(manifest,index,time){const snap=captureScrollState();const r=await api('/api/shot-scrub',{method:'POST',body:JSON.stringify({manifest,index,time})});if(!r.ok){alert(r.error||'Could not update shot frame');return}state=await api('/api/state');draw(false);restoreScrollState(snap)}
async function saveShotPrompt(manifest,index,prompt){const r=await api('/api/shot-prompt',{method:'POST',body:JSON.stringify({manifest,index,prompt})});if(!r.ok){alert(r.error||'Could not save prompt');return}state=await api('/api/state')}
async function saveShotEnabled(manifest,index,enabled){const snap=captureScrollState();const r=await api('/api/shot-enabled',{method:'POST',body:JSON.stringify({manifest,index,enabled})});if(!r.ok){alert(r.error||'Could not save shot setting');return}state=await api('/api/state');draw(false);restoreScrollState(snap)}
async function mergeShot(manifest,index){if(!confirm('Merge this shot with the next one and use the same reference?'))return;const snap=captureScrollState();const r=await api('/api/shot-merge',{method:'POST',body:JSON.stringify({manifest,index})});if(!r.ok){alert(r.error||'Could not merge shots');return}state=r.state||await api('/api/state');draw(false);lastRenderSignature=renderSignature();restoreScrollState(snap)}
async function setShotBoundary(manifest,index,edge,time){const snap=captureScrollState();const r=await api('/api/shot-boundary',{method:'POST',body:JSON.stringify({manifest,index,edge,time})});if(!r.ok){alert(r.error||'Could not update shot boundary');return}state=r.state||await api('/api/state');draw(false);lastRenderSignature=renderSignature();restoreScrollState(snap)}
function nudgeShotBoundary(manifest,index,edge,frames){const rows=(state.shot_views&&state.shot_views.shots)||[],row=rows[index];if(!row)return;const fps=Math.max(1,(Number(row.end_frame)-Number(row.start_frame)+1)/Math.max(0.001,Number(row.end)-Number(row.start)));const base=edge==='start'?Number(row.start):Number(row.end);setShotBoundary(manifest,index,edge,base+(Number(frames)||0)/fps)}
const previewTimers={};
function updateShotPreview(manifest,index,time,imgId,labelId){document.getElementById(labelId).textContent=formatSeconds(time);clearTimeout(previewTimers[imgId]);previewTimers[imgId]=setTimeout(async()=>{const r=await api('/api/shot-preview?manifest='+encodeURIComponent(manifest)+'&index='+index+'&time='+encodeURIComponent(time));if(r.ok&&r.path){const img=document.getElementById(imgId);if(img)img.src=media(r.path)+'&t='+Date.now()}},180)}
async function regenerateReference(manifest,index){const snap=captureScrollState();const r=await api('/api/reference-regenerate',{method:'POST',body:JSON.stringify({manifest,index})});if(!r.ok){alert(r.error||'Could not regenerate reference');return}state=r.state||await api('/api/state');draw(false);restoreScrollState(snap);setTimeout(refresh,1000)}
async function deleteReference(manifest,index){if(!confirm('Delete this color reference? It will be regenerated next time you run Reference Generation.'))return;const snap=captureScrollState();const r=await api('/api/reference-delete',{method:'POST',body:JSON.stringify({manifest,index})});if(!r.ok){alert(r.error||'Could not delete reference');return}state=r.state||await api('/api/state');draw(false);restoreScrollState(snap)}
function stageProgress(key){return ((state.phase_progress&&state.phase_progress.stages)||[]).find(p=>p.key===key)||{percent:0,label:'Waiting'}}
function stageProgressByTitle(title){return ((state.phase_progress&&state.phase_progress.stages)||[]).find(p=>p.stage===title)||{percent:0,label:'Waiting'}}
function progressHtml(percent,label){const p=Math.max(0,Math.min(100,Number(percent)||0));return `<div class="phase-progress"><div><span>${esc(label||'Waiting')}</span><span>${p}%</span></div><progress value="${p}" max="100"></progress></div>`}
function scrollableElements(){return [...document.querySelectorAll('.files, pre.log')]}
function scrollElementKey(el,index){if(el.id)return '#'+el.id;if(el.classList.contains('files'))return 'files:'+index;if(el.classList.contains('log'))return 'log:'+index;return 'scroll:'+index}
function captureScrollState(){const entries=scrollableElements().map((el,index)=>({key:scrollElementKey(el,index),top:el.scrollTop,left:el.scrollLeft}));return {windowX:window.scrollX,windowY:window.scrollY,entries}}
function restoreScrollState(snap){if(!snap)return;const apply=()=>{const byKey=new Map(snap.entries.map(item=>[item.key,item]));scrollableElements().forEach((el,index)=>{const saved=byKey.get(scrollElementKey(el,index));if(saved){el.scrollTop=saved.top;el.scrollLeft=saved.left}});window.scrollTo(snap.windowX||0,snap.windowY||0)};apply();setTimeout(apply,80)}
function isEditingField(){const el=document.activeElement;return !!(el&&['INPUT','TEXTAREA','SELECT'].includes(el.tagName))}
function runLogHtml(){return `<div class="log-heading"><h3>Run Log</h3><button type="button" onclick="copyRunLog()">Copy Log</button></div><pre class="log" data-run-log="true">${logHtml(state.log)}</pre>`}
function logHtml(text){return String(text||'').split('\n').map(line=>`<span class="${logClass(line)}">${esc(line)}</span>`).join('\n')}
function logClass(line){const lower=String(line).toLowerCase();if(/traceback|runtimeerror|exception|error|failed|refused|exit code [1-9]|filenotfound/.test(lower))return 'log-error';if(/warning|skipping|timed out/.test(lower))return 'log-warn';if(/ready|reuse|wrote|finished with exit code 0|started/.test(lower))return 'log-ok';return ''}
async function copyRunLog(){const text=state.log||'';try{await navigator.clipboard.writeText(text)}catch{const area=document.createElement('textarea');area.value=text;document.body.appendChild(area);area.select();document.execCommand('copy');area.remove()}}
function preview(p){if(!p)return '<p>Select an image, video, manifest, workflow, or log file.</p>';const ext=p.split('.').pop().toLowerCase();if(['png','jpg','jpeg','webp','tif','tiff'].includes(ext))return `<img src="${media(p)}">`;if(['mp4','mov','mkv','avi','webm','m4v'].includes(ext))return `<video src="${media(p)}" controls></video>`;return `<pre id="textPreview">Text preview opens via the browser media endpoint.</pre><p><a href="${media(p)}" target="_blank">Open file</a></p>`}
async function saveStage(k,redraw=false){const snap=captureScrollState();const values={};document.querySelectorAll('[data-field]').forEach(el=>{values[el.dataset.field]=el.type==='checkbox'?String(el.checked):el.value});await api('/api/settings',{method:'POST',body:JSON.stringify({stage:k,values})});state=await api('/api/state');pruneSelected();if(redraw){draw(false);restoreScrollState(snap)}showCommand(k)}
async function saveGlobal(){await api('/api/settings',{method:'POST',body:JSON.stringify({stage:'global',values:{source:document.getElementById('globalSource').value}})});selected={};state=await api('/api/state');pruneSelected();if(!availableTabs().includes(active))active='global';drawTabs();draw();lastRenderSignature=renderSignature()}
async function saveGlobalColorize(){const snap=captureScrollState();await api('/api/settings',{method:'POST',body:JSON.stringify({stage:'global',values:{colorize:String(document.getElementById('globalColorize').checked)}})});state=await api('/api/state');pruneSelected();if(!availableTabs().includes(active))active='global';drawTabs();draw(false);restoreScrollState(snap)}
async function browseGlobalSource(){const el=document.getElementById('globalSource');const r=await api('/api/browse-global-source',{method:'POST',body:JSON.stringify({current:el.value})});if(!r.ok){alert(r.error||'Browse failed');return}if(r.path){selected={};state=r.state;pruneSelected();draw();lastRenderSignature=renderSignature()}else{await refresh(true)}}
async function clearOverview(){if(!confirm('Clear the selected source material from the UI? Generated files are left on disk.'))return;selected={};const r=await api('/api/overview-clear',{method:'POST',body:'{}'});if(!r.ok){alert(r.error||'Could not clear overview');return}state=r.state;pruneSelected();active='global';drawTabs();draw();lastRenderSignature=renderSignature()}
async function browseField(stageKey,fieldKey,kind){const el=document.querySelector(`[data-field="${fieldKey}"]`);const r=await api('/api/browse',{method:'POST',body:JSON.stringify({kind,current:el.value})});if(!r.ok){alert(r.error||'Browse failed');return}if(r.path){el.value=r.path;await saveStage(stageKey)}}
async function showCommand(k){const r=await api('/api/command?stage='+encodeURIComponent(k));const el=document.getElementById('cmd');if(el)el.textContent=r.command.join(' ')}
async function confirmOverwrite(k){const force=(settings(k).force==='true');if(!force&&k!=='shots')return true;const r=await api('/api/existing-outputs?stage='+encodeURIComponent(k));if(!r.paths||!r.paths.length)return true;const reason=force?'Regenerate is enabled':'Shot Detection rewrites its manifest';return confirm(reason+' and these output paths already exist:\n\n'+r.paths.join('\n')+'\n\nOverwrite them?')}
async function runStage(k){await saveStage(k);if(!(await confirmOverwrite(k)))return;const r=await api('/api/run',{method:'POST',body:JSON.stringify({stage:k})});if(!r.ok)alert(r.message);setTimeout(()=>refresh(true),500)}
async function runAll(){for(const st of state.stages){if(st.key==='output')continue;if(!(await confirmOverwrite(st.key)))return}const r=await api('/api/run',{method:'POST',body:JSON.stringify({all:true})});if(!r.ok)alert(r.message);setTimeout(()=>refresh(true),500)}
async function stopRun(){await api('/api/stop',{method:'POST',body:'{}'});refresh(true)}
function drawSettings(){const refs=settings('references'),out=settings('outpaint'),colour=settings('colour'),recomp=settings('recomp');document.getElementById('app').innerHTML=`<section class="card"><h2>Settings</h2><h3>ComfyUI</h3><div class="row"><input id="comfyUrl" value="http://127.0.0.1:8188"><button onclick="loadComfy()">Refresh Queue</button></div><pre class="log" id="queue"></pre><h3>Qwen Reference Generation</h3><label>Workflow</label><input value="${esc(refs.workflow||'')}" readonly><label>Model backend</label><input value="${esc(refs.model_backend||'gguf')}" readonly><label>GGUF model</label><input value="${esc(refs.gguf_model||'qwen-image-edit-2511-Q4_K_M.gguf')}" readonly><label>Prompt</label><textarea readonly>${esc(refs.prompt||'')}</textarea><label>Prompt suffix</label><textarea readonly>${esc(refs.prompt_suffix||'')}</textarea><h3>Pipeline Defaults</h3><div class="source-info"><div><span>Outpaint aspect</span><strong>${esc(out.target_aspect||'16:9')}</strong></div><div><span>Outpaint height</span><strong>${esc(out.target_height||'720')}</strong></div><div><span>Color CRF</span><strong>${esc(colour.crf||'18')}</strong></div><div><span>Feather pixels</span><strong>${esc(recomp.feather_pixels||'80')}</strong></div></div><h3>Log file</h3><div class="row"><input id="comfyLog" placeholder="path/to/comfy.log"><button onclick="loadLogFile()">Load</button></div><pre class="log" id="comfyLogText"></pre></section>`}
async function loadComfy(){const r=await api('/api/comfy?url='+encodeURIComponent(document.getElementById('comfyUrl').value));document.getElementById('queue').textContent=r.ok?JSON.stringify(r.queue,null,2):r.error}
async function loadLogFile(){const r=await api('/api/logfile?path='+encodeURIComponent(document.getElementById('comfyLog').value));document.getElementById('comfyLogText').textContent=r.text}
setInterval(refresh,4000);refresh();
</script>
</body>
</html>"""


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


def ensure_comfy_available_for_stage(stage_title: str) -> tuple[bool, str]:
    if os.environ.get("AI_REMASTER_NO_COMFY_AUTOSTART") == "1":
        return True, ""
    config = current_config()
    url = config.get("comfy_url", "http://127.0.0.1:8188")
    instances = discover_comfy_instances(url)
    if len(instances) > 1:
        message = "Multiple ComfyUI instances appear to be running: " + ", ".join(instances) + ". Close extras or update .ai_remaster_config.json to the one ARP should use."
        startup_log(message)
        return False, message
    if instances:
        queue = comfy_queue(url)
        message = comfy_busy_message(url, queue)
        if message:
            startup_log(message)
            return False, message
        startup_log(f"Found ComfyUI already running at {url}; queue is idle.")
        return True, ""
    if STARTED_COMFY_PROCESS and STARTED_COMFY_PROCESS.poll() is None:
        startup_log(f"ComfyUI launch is already in progress at {url}.")
        return True, ""
    startup_log(f"ComfyUI is not running at {url}; launching it now.")
    start_comfy_if_needed()
    return True, ""


def startup_log(message: str) -> None:
    print(message)
    app = globals().get("APP")
    if app is not None:
        app.log.append(message)


def wait_for_comfy_ready(url: str, process: subprocess.Popen | None, timeout_seconds: float = 180.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if comfy_is_running(url):
            startup_log(f"ComfyUI is ready at {url}")
            return True
        if process and process.poll() is not None:
            startup_log(f"ComfyUI exited before it became ready. Exit code: {process.returncode}")
            return False
        time.sleep(2)
    startup_log(f"Timed out waiting for ComfyUI to become ready at {url}")
    return False


def monitor_comfy_startup(url: str, process: subprocess.Popen | None) -> None:
    wait_for_comfy_ready(url, process, float(os.environ.get("AI_REMASTER_COMFY_START_TIMEOUT", "180")))


def start_comfy_if_needed() -> None:
    global STARTED_COMFY_PROCESS
    config = current_config()
    url = config.get("comfy_url", "http://127.0.0.1:8188")
    if STARTED_COMFY_PROCESS and STARTED_COMFY_PROCESS.poll() is None:
        startup_log(f"ComfyUI launch is already in progress at {url}.")
        return
    instances = discover_comfy_instances(url)
    if len(instances) > 1:
        startup_log("Multiple ComfyUI instances appear to be running: " + ", ".join(instances) + ". Close extras or update .ai_remaster_config.json to the one ARP should use.")
        return
    if instances:
        if instances[0].rstrip("/") == url.rstrip("/"):
            startup_log(f"ComfyUI already running at {url}")
        else:
            startup_log(f"ComfyUI appears to be running at {instances[0]}, but ARP is configured for {url}. Close it or update .ai_remaster_config.json.")
        return
    comfy_dir = Path(config.get("comfy_dir", str(ROOT / "tools" / "comfyui")))
    main_py = comfy_dir / "main.py"
    if not main_py.exists():
        if CONFIG_FILE.exists():
            startup_log(f"ComfyUI is configured but main.py was not found: {main_py}")
            startup_log("Run install_windows.bat again and choose your ComfyUI directory.")
        else:
            startup_log("ComfyUI is not configured yet.")
            startup_log("Run install_windows.bat again and choose whether to clone ComfyUI or use an existing ComfyUI directory.")
        return
    host = config.get("comfy_host", "127.0.0.1")
    port = str(config.get("comfy_port", "8188"))
    command = [sys.executable, "main.py", "--listen", host, "--port", port]
    kwargs: dict = {"cwd": str(comfy_dir)}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    STARTED_COMFY_PROCESS = subprocess.Popen(command, **kwargs)
    startup_log(f"Started ComfyUI in a new process at {url}")
    startup_log("ComfyUI is still starting; the pipeline will wait before queueing prompts.")
    threading.Thread(target=monitor_comfy_startup, args=(url, STARTED_COMFY_PROCESS), daemon=True).start()


def stop_started_comfy() -> None:
    global STARTED_COMFY_PROCESS
    process = STARTED_COMFY_PROCESS
    if not process or process.poll() is not None:
        STARTED_COMFY_PROCESS = None
        return
    print("Stopping ComfyUI started by ARP...")
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
    except Exception as exc:
        print(f"Could not stop ComfyUI cleanly: {exc}")
    finally:
        STARTED_COMFY_PROCESS = None


def install_shutdown_handlers() -> None:
    atexit.register(stop_started_comfy)

    def handle_signal(signum, _frame) -> None:
        stop_started_comfy()
        raise SystemExit(0 if signum in (signal.SIGINT, signal.SIGTERM) else 1)

    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), handle_signal)


def create_server(host: str, requested_port: int) -> ThreadingHTTPServer:
    ports = [requested_port, 0] if requested_port != 0 else [0]
    last_error: OSError | None = None
    for port in ports:
        try:
            return ThreadingHTTPServer((host, port), Handler)
        except OSError as exc:
            last_error = exc
            if port != 0:
                print(f"GUI port {port} was unavailable ({exc}); trying a free port.")
    assert last_error is not None
    raise last_error


def main() -> int:
    os.chdir(ROOT)
    install_shutdown_handlers()
    if os.environ.get("AI_REMASTER_NO_COMFY_AUTOSTART") != "1":
        start_comfy_if_needed()
    host = "127.0.0.1"
    requested_port = int(os.environ.get("AI_REMASTER_GUI_PORT", "8765"))
    server = create_server(host, requested_port)
    url = f"http://{host}:{server.server_port}/"
    print(f"AI Remaster GUI running at {url}")
    if os.environ.get("AI_REMASTER_NO_BROWSER") != "1":
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        stop_started_comfy()
    return 0
