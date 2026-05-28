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
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import urlopen

from .comfy import comfy_busy_message, comfy_is_running, comfy_queue, discover_comfy_instances
from .config import (
    ASPECT_PREVIEW_DIR,
    CONFIG_FILE,
    FILE_PREVIEW_DIR,
    IMAGE_EXTS,
    MEDIA_CLIP_DIR,
    PREVIEW_DIR,
    REFERENCE_PROMPT,
    REFERENCE_PROMPT_SUFFIX,
    ROOT,
    SCRIPTS,
    SETTINGS_FILE,
    STATIC_DIR,
    TEXT_EXTS,
    VIDEO_EXTS,
    current_config,
)
from .models import COLORIZE_STAGE_KEYS, STAGES, Stage, output_stage
from .paths import aspect_slug, even_int, newest, parse_aspect, rel, resolve, resolve_video_source, safe_stem

STARTED_COMFY_PROCESS: subprocess.Popen | None = None
PROJECT_SCHEMA_VERSION = 1


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


APP_VERSION = app_version()
DEFAULT_ANCHOR_PROMPT = (
    "Fill the black outpaint margins with a natural continuation of this black-and-white film frame. "
    "Preserve the centre/original frame area, composition, lighting, paper, clothing, and background. "
    "If hands or fingers extend into the new margins, make them anatomically natural with five fingers and normal joints. "
    "Do not colorize. Do not add text, captions, logos, or unrelated new objects."
)


def default_qwen_workflow(config: dict[str, str]) -> str:
    comfy_dir = Path(config.get("comfy_dir", ROOT / "tools" / "comfyui"))
    search_dirs = [
        ROOT / "workflows" / "qwen_image_edit",
        ROOT / "blueprints",
        comfy_dir / "user" / "default" / "workflows",
    ]
    for directory in search_dirs:
        if not directory.exists():
            continue
        matches = sorted(
            path
            for path in directory.glob("*.json")
            if "qwen" in path.name.lower() and path.is_file()
        )
        if matches:
            return rel(matches[0])
    return ""


def load_settings() -> dict[str, dict[str, str]]:
    defaults = {stage.key: {key: default for key, _label, _kind, default in stage.fields} for stage in STAGES}
    defaults["global"] = {"source": "", "expand_outpaint": "true", "colorize": "true", "section_start": "0", "section_end": "", "last_browse_dir": ""}
    if SETTINGS_FILE.exists():
        try:
            stored = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            for key, values in stored.items():
                if key in defaults and isinstance(values, dict):
                    defaults[key].update({k: str(v) for k, v in values.items()})
        except json.JSONDecodeError:
            pass
    source = newest(ROOT / "input", VIDEO_EXTS)
    if source and not defaults["global"].get("source"):
        defaults["global"]["source"] = rel(source)
    if "colormnet" in defaults["recomp"].get("colorized_video", "").lower():
        defaults["recomp"]["colorized_video"] = ""
    defaults["colour"].setdefault("method", "deepexemplar")
    defaults["recomp"].setdefault("colorization_method", "deepexemplar")
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
        self.project_path: Path | None = None
        self.log: list[str] = []
        self.process: subprocess.Popen[str] | None = None
        self.running_stage = ""
        self.running_stage_key = ""
        self.running_reference_manifest = ""
        self.running_reference_index: int | None = None
        self.run_started_at = 0.0
        self.lock = threading.Lock()

    def normalize_loaded_source_state(self) -> None:
        source_text = self.settings.get("global", {}).get("source", "")
        if source_text:
            source = resolve_video_source(source_text)
            if source.exists() and str(source) != source_text:
                self.settings.setdefault("global", {})["source"] = str(source)
                self.log.append(f"Resolved source material path to: {source}")
            self.clear_derived_stage_inputs()
            self.hydrate_stage_inputs("global")

    def colorize_enabled(self) -> bool:
        return self.settings.get("global", {}).get("colorize", "true") == "true"

    def outpaint_enabled(self) -> bool:
        return self.settings.get("global", {}).get("expand_outpaint", "true") == "true"

    def active_stages(self) -> tuple[Stage, ...]:
        stages = tuple(stage for stage in STAGES if stage.key != "output")
        if not self.outpaint_enabled():
            stages = tuple(stage for stage in stages if stage.key != "outpaint")
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
                    try:
                        stat = path.stat()
                        preview = file_preview(path)
                    except FileNotFoundError:
                        continue
                    except OSError as exc:
                        self.log.append(f"Skipped unreadable file while refreshing {stage.title}: {rel(path)} ({exc})")
                        continue
                    out.append({"path": rel(path), "size": stat.st_size, "mtime": int(stat.st_mtime), "preview": preview})
        return sorted(out, key=lambda item: str(item["path"]).lower())

    def stage_file_prefixes(self, stage_key: str) -> tuple[str, ...]:
        source = self.settings.get("global", {}).get("source", "")
        if stage_key == "outpaint" and source:
            stem = safe_stem(resolve(source).name)
            values = self.settings.get("outpaint", {})
            return (stem,)
        return ()

    def stage_file_matches(self, stage_key: str, path: Path, prefixes: tuple[str, ...]) -> bool:
        if stage_key != "outpaint" or not prefixes:
            return True
        name = path.stem
        return any(name == prefix or name.startswith(prefix + "_") for prefix in prefixes)

    def progress(self) -> list[dict[str, str]]:
        rows = []
        for stage in self.active_stages():
            expected = [resolve(path) for path in self.expected_outputs(stage.key) if path]
            existing = [path for path in expected if path.exists()]
            ready = bool(expected) and len(existing) == len(expected)
            latest = max(existing, key=lambda path: path.stat().st_mtime_ns) if existing else None
            rows.append({"stage": stage.title, "status": "Ready" if ready else "Waiting", "latest": rel(latest) if latest else ""})
        return rows

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
            section = source_section_state(self.settings)
            return {
                "root": str(ROOT),
                "version": APP_VERSION,
                "stages": [stage.__dict__ | {"files": self.files_for(stage)} for stage in (*self.active_stages(), output_stage())],
                "settings": self.settings,
                "progress": self.progress(),
                "phase_progress": self.phase_progress(),
                "expected_outputs": {stage.key: self.expected_outputs(stage.key) for stage in (*self.active_stages(), output_stage())},
                "existing_outputs": {stage.key: self.existing_outputs(stage.key) for stage in (*self.active_stages(), output_stage())},
                "source_previews": previews,
                "source_info": info,
                "source_section": section,
                "project_path": str(self.project_path) if self.project_path else "",
                "source_monochrome": source_monochrome(source_text),
                "aspect_preview": aspect_preview_for_settings(self.settings),
                "outpaint_chunks": outpaint_chunks_state(self.settings),
                "shot_views": shot_views(self.settings),
                "cache": cache_state(),
                "running": running,
                "running_stage": self.running_stage,
                "running_reference": {
                    "manifest": self.running_reference_manifest,
                    "index": self.running_reference_index,
                } if self.running_reference_index is not None else None,
                "log": "\n".join(self.log[-800:]),
                "log_count": len(self.log),
            }

    def update_settings(self, stage: str, values: dict[str, str]) -> None:
        if stage == "global" and "source" in values:
            source = resolve_video_source(str(values.get("source", "")))
            if source.exists() and str(source) != str(values.get("source", "")):
                values = dict(values)
                values["source"] = str(source)
                self.log.append(f"Resolved source material path to: {source}")
        self.settings.setdefault(stage, {}).update({key: str(value) for key, value in values.items()})
        if stage == "global" and {"source", "section_start", "section_end"} & set(values):
            self.log.append(f"Loading source material: {values.get('source')}")
            if "source" in values:
                self.settings.setdefault("global", {})["colorize"] = "true" if source_monochrome(str(values.get("source", ""))) else "false"
            self.clear_derived_stage_inputs()
            self.hydrate_stage_inputs("global")
        elif stage == "global" and ({"colorize", "expand_outpaint"} & set(values)):
            self.hydrate_stage_inputs("global")
        elif stage == "colour" and "method" in values:
            if values.get("method") in {"deepexemplar", "colormnet"}:
                self.settings.setdefault("recomp", {})["colorization_method"] = str(values["method"])
            self.hydrate_stage_inputs("colour")
        elif stage == "recomp" and "colorization_method" in values:
            preferred = colorized_output_for_manifest(self.settings.get("colour", {}).get("manifest", ""), str(values.get("colorization_method", "deepexemplar")))
            if preferred:
                self.settings.setdefault("recomp", {})["colorized_video"] = preferred
        if stage == "shots" and "outpainted_video" in values:
            manifest = manifest_for_outpainted(values.get("outpainted_video", ""))
            self.settings.setdefault("references", {}).setdefault("manifest", manifest)
            self.settings.setdefault("colour", {}).setdefault("manifest", manifest)
        self.save()

    def clear_overview(self) -> None:
        self.settings.setdefault("global", {}).update({"source": "", "expand_outpaint": "true", "colorize": "true", "section_start": "0", "section_end": ""})
        self.clear_derived_stage_inputs()
        self.log.append("Cleared source material from the Overview.")
        self.save()

    def save_project(self, save_as: bool = False) -> dict[str, str]:
        if save_as or not self.project_path:
            suggested = project_save_suggestion(self.settings, self.project_path)
            selected = browse_path("project_save", str(suggested))
            if not selected:
                return {"path": ""}
            path = resolve(selected)
            if path.suffix.lower() != ".arpp":
                path = path.with_suffix(".arpp")
            self.project_path = path
        else:
            path = self.project_path
        payload = project_payload(self.settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.log.append(f"Saved ARP project: {path}")
        return {"path": str(path)}

    def load_project(self) -> dict[str, str]:
        selected = browse_path("project_open", "")
        if not selected:
            return {"path": ""}
        path = resolve(selected)
        loaded = read_project_file(path)
        self.settings = loaded
        self.project_path = path
        self.hydrate_stage_inputs("")
        self.save()
        self.log.append(f"Loaded ARP project: {path}")
        return {"path": str(path)}

    def clear_derived_stage_inputs(self) -> None:
        for stage_key, keys in {
            "outpaint": ("source", "output", "outpainted_video", "manifest", "colorized_video"),
            "shots": ("outpainted_video", "manifest", "colorized_video"),
            "references": ("manifest", "outpainted_video", "colorized_video"),
            "colour": ("manifest", "outpainted_video", "colorized_video"),
            "recomp": ("outpainted_video", "source", "colorized_video", "output"),
            "output": ("output", "outpainted_video", "manifest", "colorized_video"),
        }.items():
            stage_settings = self.settings.setdefault(stage_key, {})
            for key in keys:
                stage_settings[key] = ""

    def hydrate_stage_inputs(self, completed_stage: str = "") -> None:
        if not self.outpaint_enabled():
            outpainted_text = pipeline_source_text(self.settings)
            if outpainted_text:
                self.settings.setdefault("shots", {})["outpainted_video"] = outpainted_text
                self.settings.setdefault("recomp", {})["outpainted_video"] = outpainted_text
                manifest = manifest_for_outpainted(outpainted_text)
                self.settings.setdefault("references", {})["manifest"] = manifest
                self.settings.setdefault("colour", {})["manifest"] = manifest
                self.log.append(f"Updated Shot Detection input: {outpainted_text}")
            outpainted = None
        else:
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
        if self.outpaint_enabled() and not outpainted and completed_stage == "global":
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
        preferred_colorized_text = colorized_output_for_manifest(
            self.settings.get("colour", {}).get("manifest", ""),
            self.settings.get("recomp", {}).get("colorization_method", self.settings.get("colour", {}).get("method", "deepexemplar")),
        )
        if preferred_colorized_text and resolve(preferred_colorized_text).exists():
            expected_colorized_text = preferred_colorized_text
        expected_colorized = resolve(expected_colorized_text) if expected_colorized_text else None
        colorized = expected_colorized if expected_colorized and expected_colorized.exists() else None
        if self.colorize_enabled() and colorized:
            self.settings.setdefault("recomp", {})["colorized_video"] = rel(colorized)
        elif not self.colorize_enabled():
            self.settings.setdefault("recomp", {})["colorized_video"] = ""
        source = self.settings.get("global", {}).get("source")
        if source:
            self.settings.setdefault("recomp", {})["source"] = pipeline_source_text(self.settings)
        output = recomposition_output_for(self.settings.get("recomp", {}).get("outpainted_video", ""))
        if output:
            self.settings.setdefault("recomp", {})["output"] = output
            self.settings.setdefault("output", {})["output"] = output
        self.save()

    def expected_outputs(self, stage_key: str) -> list[str]:
        values = self.settings.get(stage_key, {})
        if stage_key == "outpaint":
            if not self.outpaint_enabled():
                return []
            source = pipeline_source_text(self.settings)
            return [outpaint_output_for(source, values.get("target_aspect", "16:9"), values.get("target_height", "720"))] if source else []
        if stage_key == "shots":
            return [manifest_for_outpainted(values.get("outpainted_video", ""))]
        if stage_key == "references":
            return color_reference_outputs(values.get("manifest", ""))
        if stage_key == "colour":
            return colorized_outputs_for_manifest(values.get("manifest", ""), values.get("method", "deepexemplar"))
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
            add(["--source", pipeline_source_text(self.settings)])
            add(["--target-aspect", values.get("target_aspect", "16:9")])
            add(["--target-height", str(resolved_outpaint_height(pipeline_source_text(self.settings), values.get("target_height", "720")))])
            add(["--chunk-seconds", values.get("chunk_seconds", "20")])
            add(["--overlap-frames", values.get("overlap_frames", "8")])
            if values.get("negative_prompt"):
                add(["--negative-prompt", values.get("negative_prompt", "")])
            manifest = outpaint_chunk_manifest_for(pipeline_source_text(self.settings), values)
            if manifest:
                add(["--chunk-manifest", manifest])
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
            method = values.get("method", "deepexemplar")
            add(["--method", method])
            output = colorized_output_for_manifest(values.get("manifest", ""), method)
            if output:
                add(["--output", output])
            add(["--comfy-dir", config.get("comfy_dir", str(ROOT / "tools" / "comfyui"))])
            add(["--comfy-url", config.get("comfy_url", "http://127.0.0.1:8188")])
            add(["--comfy-output-root", str(Path(config.get("comfy_dir", str(ROOT / "tools" / "comfyui"))) / "output")])
            add(["--crf", values.get("crf", "18")])
            add(["--colormnet-memory-mode", values.get("colormnet_memory_mode", "balanced")])
            add(["--colormnet-feature-encoder", values.get("colormnet_feature_encoder", "resnet50")])
            if values.get("colormnet_text_guidance"):
                add(["--colormnet-text-guidance", values["colormnet_text_guidance"]])
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
        if stage_key == "outpaint" and not self.outpaint_enabled():
            return False, "Expand using Outpainting is disabled on the Overview tab."
        if stage_key in COLORIZE_STAGE_KEYS and not self.colorize_enabled():
            return False, "Colorize is disabled on the Global tab."
        stage = next(item for item in STAGES if item.key == stage_key)
        values = self.settings[stage_key]
        missing = [key for key in stage.required if not values.get(key)]
        if stage_key == "outpaint" and not self.settings.get("global", {}).get("source"):
            missing = ["source material on the Global tab"]
        if missing:
            return False, "Missing settings: " + ", ".join(missing)
        try:
            self.ensure_pipeline_source()
        except Exception as exc:
            return False, f"Could not prepare selected source section: {exc}"
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

    def run_outpaint_chunk(self, index: int) -> tuple[bool, str]:
        if not self.settings.get("global", {}).get("source"):
            return False, "Choose source material on the Overview tab first."
        try:
            self.ensure_pipeline_source()
        except Exception as exc:
            return False, f"Could not prepare selected source section: {exc}"
        ok, message = ensure_comfy_available_for_stage("Outpainting")
        if not ok:
            return False, message
        with self.lock:
            if self.process and self.process.poll() is None:
                return False, "A command is already running."
            self.running_stage = f"Outpainting chunk {index + 1}"
            self.running_stage_key = "outpaint"
            self.run_started_at = time.time()
            cmd = self.command_for("outpaint")
            cmd.extend(["--only-chunk", str(index), "--force"])
            self.log.append("> " + " ".join(cmd))
            kwargs: dict = {"cwd": ROOT, "text": True, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            self.process = subprocess.Popen(cmd, **kwargs)
            threading.Thread(target=self._collect_output, args=("outpaint",), daemon=True).start()
        return True, f"Started outpaint chunk {index + 1}"

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

    def run_outpaint_anchor_generation(self, index: int, seconds: str, prompt: str) -> tuple[bool, str]:
        ok, message = ensure_comfy_available_for_stage("Guide Frame Generation")
        if not ok:
            return False, message
        try:
            cmd, output = outpaint_anchor_generation_command(index, seconds, prompt)
        except Exception as exc:
            return False, str(exc)
        with self.lock:
            if self.process and self.process.poll() is None:
                return False, "A command is already running."
            self.running_stage = f"Generating guide frame for chunk {index + 1}"
            self.running_stage_key = "outpaint"
            self.run_started_at = time.time()
            self.log.append(f"Generating Qwen guide frame for chunk {index + 1}: {output}")
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
                self.run_started_at = 0.0
                self.log.append(f"Could not start guide frame generation: {exc}")
                return False, f"Could not start guide frame generation: {exc}"
            threading.Thread(target=self._collect_output, args=("outpaint",), daemon=True).start()
        return True, f"Started Qwen guide frame generation for chunk {index + 1}."

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

    def ensure_pipeline_source(self) -> None:
        ensure_source_section_clip(self.settings)

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


def split_manifest_shot(manifest_text: str, index: int, seconds: float | None = None) -> dict[str, str]:
    manifest = resolve(manifest_text)
    source_video, fieldnames, rows = read_manifest_details(manifest)
    if index < 0 or index >= len(rows):
        raise IndexError(f"Manifest row {index} is out of range.")

    for key in ("enabled", "end", "source_reference", "color_reference", "prompt"):
        if key not in fieldnames:
            fieldnames.append(key)

    start = parse_time_seconds(rows[index - 1].get("end", "")) if index > 0 else 0.0
    end = parse_time_seconds(rows[index].get("end", ""))
    if end <= start:
        raise RuntimeError(f"Shot {index + 1} cannot be split because its duration is not valid.")

    split_at = (start + end) / 2 if seconds is None else float(seconds)
    split_at = max(start + 0.001, min(end - 0.001, split_at))
    if end - start < 0.1:
        raise RuntimeError(f"Shot {index + 1} is too short to split.")

    first = dict(rows[index])
    second = dict(rows[index])
    first["end"] = format_timecode(split_at)
    first["source_reference"] = ""
    first["color_reference"] = ""
    second["end"] = rows[index].get("end", "")
    second["source_reference"] = ""
    second["color_reference"] = ""
    rows[index] = first
    rows.insert(index + 1, second)
    write_manifest_details(manifest, source_video, fieldnames, rows)
    APP.log.append(f"Split shot {index + 1} at {format_timecode(split_at)}")
    return {"manifest": rel(manifest), "split": format_timecode(split_at)}


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
    source = resolve_video_source(source_text)
    if not source.exists() or source.suffix.lower() not in VIDEO_EXTS:
        return None
    stat = source.stat()
    return str(source), stat.st_size, stat.st_mtime_ns


def project_payload(settings: dict[str, dict[str, str]]) -> dict:
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "app": "AI Remaster Pipeline",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "settings": settings,
    }


def read_project_file(path: Path) -> dict[str, dict[str, str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Project file is not valid JSON: {exc}") from exc
    version = int(data.get("schema_version", 0) or 0)
    if version < 1:
        raise RuntimeError("Project file does not include a supported schema_version.")
    if version > PROJECT_SCHEMA_VERSION:
        raise RuntimeError(f"Project schema {version} is newer than this ARP build supports.")
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise RuntimeError("Project file does not contain settings.")
    loaded = load_settings()
    for stage, values in settings.items():
        if stage in loaded and isinstance(values, dict):
            loaded[stage].update({str(key): str(value) for key, value in values.items()})
    return loaded


def project_default_path(settings: dict[str, dict[str, str]]) -> Path:
    source = resolve_video_source(settings.get("global", {}).get("source", ""))
    stem = safe_stem(source.name if source.name else "arp_project")
    return ROOT / "projects" / f"{stem}.arpp"


def project_save_suggestion(settings: dict[str, dict[str, str]], project_path: Path | None = None) -> Path:
    if project_path:
        return project_path
    default_path = project_default_path(settings)
    last_dir = last_browse_dir(settings)
    return (last_dir / default_path.name) if last_dir else default_path


def last_browse_dir(settings: dict[str, dict[str, str]] | None = None) -> Path | None:
    values = settings or (APP.settings if "APP" in globals() else {})
    text = values.get("global", {}).get("last_browse_dir", "") if isinstance(values, dict) else ""
    if not text:
        return None
    path = resolve(str(text))
    return path if path.exists() and path.is_dir() else None


def pipeline_source_text(settings: dict) -> str:
    global_settings = settings.get("global", {})
    source_text = global_settings.get("source", "")
    if not source_text or not source_section_is_active(settings):
        return source_text
    return rel(source_section_output_for(settings))


def source_section_state(settings: dict) -> dict:
    global_settings = settings.get("global", {})
    source_text = global_settings.get("source", "")
    start = section_float(global_settings.get("section_start", "0"), 0.0)
    end = section_float(global_settings.get("section_end", ""), 0.0)
    enabled = source_section_is_active(settings)
    output = source_section_output_for(settings) if source_text and enabled else None
    return {
        "enabled": enabled,
        "start": start,
        "end": end,
        "start_label": format_timecode(start),
        "end_label": format_timecode(end) if end > 0 else "",
        "output": rel(output) if output else "",
        "output_exists": bool(output and output.exists()),
    }


def source_section_output_for(settings: dict) -> Path:
    global_settings = settings.get("global", {})
    source = resolve_video_source(global_settings.get("source", ""))
    start = section_float(global_settings.get("section_start", "0"), 0.0)
    end = section_float(global_settings.get("section_end", ""), 0.0)
    suffix = f"{int(round(start * 1000)):010d}_{int(round(end * 1000)):010d}"
    return ROOT / "intermediate" / "source_sections" / f"{safe_stem(source.name)}_{suffix}{source.suffix or '.mp4'}"


def source_section_is_active(settings: dict) -> bool:
    global_settings = settings.get("global", {})
    start = section_float(global_settings.get("section_start", "0"), 0.0)
    end = section_float(global_settings.get("section_end", ""), 0.0)
    return end > start


def ensure_source_section_clip(settings: dict) -> str:
    global_settings = settings.get("global", {})
    source_text = global_settings.get("source", "")
    if not source_text or not source_section_is_active(settings):
        return source_text
    source = resolve_video_source(source_text)
    start = section_float(global_settings.get("section_start", "0"), 0.0)
    end = section_float(global_settings.get("section_end", ""), 0.0)
    if end <= start:
        return source_text
    output = source_section_output_for(settings)
    if output.exists() and output.stat().st_size > 0:
        return rel(output)
    ffmpeg = local_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Run install_windows.bat to install local FFmpeg for source section trimming.")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{max(0.041, end - start):.3f}",
        "-map",
        "0",
        "-c:v",
        "libx264",
        "-crf",
        "14",
        "-preset",
        "veryfast",
        "-c:a",
        "copy",
        str(partial),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg source section trim failed").strip())
    partial.replace(output)
    APP.log.append(f"Prepared source section clip: {rel(output)}")
    return rel(output)


def section_float(value: str, default: float) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default


def manifest_for_outpainted(outpainted_text: str) -> str:
    if not outpainted_text:
        return ""
    source = resolve(outpainted_text)
    stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in source.name.replace(" ", "_"))
    return rel(ROOT / "manifests" / "references" / f"colorize_manifest_{Path(stem).stem}_shots_auto.csv")


def outpaint_output_for(source_text: str, aspect: str, target_height_text: str = "720") -> str:
    if not source_text:
        return ""
    source = resolve_video_source(source_text)
    width, height = outpaint_size_for_source(source_text, aspect, target_height_text)
    values = APP.settings.get("outpaint", {}) if "APP" in globals() else {}
    crops = [int(float(values.get(key, "0") or 0)) for key in ("crop_left", "crop_right", "crop_top", "crop_bottom")]
    crop = "" if not any(crops) else f"_crop{crops[0]}-{crops[1]}-{crops[2]}-{crops[3]}"
    return rel(ROOT / "intermediate" / "outpainted" / f"{safe_stem(source.name)}_{aspect_slug(aspect)}_{width}x{height}{crop}_outpainted.mp4")


def outpaint_size_for(aspect: str, target_height_text: str = "720") -> tuple[int, int]:
    try:
        height = int(float(target_height_text or "720"))
    except ValueError:
        height = 720
    return even_int(height * parse_aspect(aspect)), even_int(height)


def source_video_height(source_text: str) -> int:
    try:
        source = resolve_video_source(source_text)
        metrics = video_metrics(source)
        return even_int(int(metrics.get("height") or 720))
    except Exception:
        return 720


def resolved_outpaint_height(source_text: str, target_height_text: str = "720") -> int:
    if str(target_height_text or "").strip().lower() in {"source", "source height", "original"}:
        return source_video_height(source_text)
    try:
        return even_int(int(float(target_height_text or "720")))
    except ValueError:
        return 720


def outpaint_size_for_source(source_text: str, aspect: str, target_height_text: str = "720") -> tuple[int, int]:
    height = resolved_outpaint_height(source_text, target_height_text)
    return even_int(height * parse_aspect(aspect)), height


def outpaint_crop_slug(values: dict[str, str]) -> str:
    crops = [int(float(values.get(key, "0") or 0)) for key in ("crop_left", "crop_right", "crop_top", "crop_bottom")]
    return "" if not any(crops) else f"_crop{crops[0]}-{crops[1]}-{crops[2]}-{crops[3]}"


def outpaint_chunk_dir_for(source_text: str, values: dict[str, str]) -> Path:
    source = resolve_video_source(source_text)
    aspect = values.get("target_aspect", "16:9")
    width, height = outpaint_size_for_source(source_text, aspect, values.get("target_height", "720"))
    return ROOT / ".cache" / "outpaint_chunks" / f"{safe_stem(source.name)}_{aspect_slug(aspect)}_{width}x{height}{outpaint_crop_slug(values)}"


def outpaint_chunk_manifest_for(source_text: str, values: dict[str, str]) -> str:
    if not source_text:
        return ""
    source = resolve_video_source(source_text)
    aspect = values.get("target_aspect", "16:9")
    width, height = outpaint_size_for_source(source_text, aspect, values.get("target_height", "720"))
    return rel(ROOT / "manifests" / "outpaint_chunks" / f"{safe_stem(source.name)}_{aspect_slug(aspect)}_{width}x{height}{outpaint_crop_slug(values)}_chunks.csv")


def outpaint_prepared_for(source_text: str, values: dict[str, str]) -> Path:
    source = resolve_video_source(source_text)
    aspect = values.get("target_aspect", "16:9")
    width, height = outpaint_size_for_source(source_text, aspect, values.get("target_height", "720"))
    return ROOT / "intermediate" / "outpaint_prepared" / f"{source.stem}_{width}x{height}_lifted.mp4"


def ensure_outpaint_prepared_canvas(source_text: str, values: dict[str, str]) -> Path:
    source = resolve_video_source(source_text)
    prepared = outpaint_prepared_for(source_text, values)
    if prepared.exists():
        return prepared

    cmd = [
        sys.executable,
        str(SCRIPTS / "prepare_outpaint_input.py"),
        "--source",
        str(source),
        "--target-aspect",
        values.get("target_aspect", "16:9"),
        "--black-lift",
        str(values.get("black_lift", "0.018") or "0.018"),
        "--gamma",
        str(values.get("gamma", "1.06") or "1.06"),
        "--output",
        str(prepared),
        "--crop-left",
        str(values.get("crop_left", "0") or "0"),
        "--crop-right",
        str(values.get("crop_right", "0") or "0"),
        "--crop-top",
        str(values.get("crop_top", "0") or "0"),
        "--crop-bottom",
        str(values.get("crop_bottom", "0") or "0"),
        "--target-height",
        str(resolved_outpaint_height(source_text, values.get("target_height", "720"))),
    ]
    APP.log.append(f"Preparing expanded canvas for guide frame: {rel(prepared)}")
    APP.log.append("> " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, check=False, capture_output=True, text=True)
    for line in (result.stdout or "").splitlines():
        APP.log.append(line)
    for line in (result.stderr or "").splitlines():
        APP.log.append(line)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "Could not prepare expanded outpaint canvas.")
    if not prepared.exists():
        raise RuntimeError(f"Prepared expanded canvas was not created: {prepared}")
    return prepared


def read_outpaint_chunk_rows(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {int(row["chunk_index"]): row for row in csv.DictReader(handle) if row.get("chunk_index", "").isdigit()}


def write_outpaint_chunk_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "chunk_index",
        "start_frame",
        "end_frame",
        "start_seconds",
        "end_seconds",
        "custom_seconds",
        "seed",
        "prompt_suffix",
        "negative_suffix",
        "anchor_image",
        "anchor_position",
        "anchor_seconds",
        "prepared_path",
        "raw_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def outpaint_chunks_state(settings: dict) -> dict:
    try:
        ensure_source_section_clip(settings)
    except Exception as exc:
        return {"manifest": "", "rows": [], "error": f"Could not prepare selected source section: {exc}"}

    source_text = pipeline_source_text(settings)
    if not source_text:
        return {"manifest": "", "rows": []}
    source = resolve_video_source(source_text)
    if not source.exists():
        return {"manifest": "", "rows": [], "error": f"Source material is not a readable file: {source}"}
    values = settings.get("outpaint", {})
    prepared = outpaint_prepared_for(source_text, values)
    range_source = prepared if prepared.exists() else source
    metrics = video_metrics(range_source)
    fps = metrics.get("fps") or 24.0
    total_frames = int(metrics.get("frames") or 0)
    if total_frames <= 0:
        message = f"Outpaint chunk preview skipped; could not count frames in: {range_source}"
        APP.log.append(message)
        return {"manifest": "", "rows": [], "error": message}
    try:
        chunk_seconds = float(values.get("chunk_seconds", "20") or 20)
    except ValueError:
        chunk_seconds = 20.0
    try:
        overlap_frames = int(float(values.get("overlap_frames", "8") or 8))
    except ValueError:
        overlap_frames = 8
    chunk_dir = outpaint_chunk_dir_for(source_text, values)
    manifest = resolve(outpaint_chunk_manifest_for(source_text, values))
    existing = read_outpaint_chunk_rows(manifest)
    ranges = outpaint_chunk_ranges(total_frames, fps, chunk_seconds, overlap_frames, existing)
    rows = []
    for index, start_frame, end_frame in ranges:
        row = dict(existing.get(index, {}))
        prepared = chunk_dir / f"prepared_{index:04d}_{start_frame:06d}_{end_frame:06d}.mp4"
        raw = chunk_dir / f"raw_{index:04d}_{start_frame:06d}_{end_frame:06d}.mp4"
        row.update({
            "chunk_index": str(index),
            "start_frame": str(start_frame),
            "end_frame": str(end_frame),
            "start_seconds": f"{start_frame / fps:.6f}",
            "end_seconds": f"{end_frame / fps:.6f}",
            "prepared_path": rel(prepared),
            "raw_path": rel(raw),
        })
        row.setdefault("custom_seconds", "")
        if not row.get("seed"):
            row["seed"] = str(42 + index)
        row.setdefault("prompt_suffix", "")
        row.setdefault("negative_suffix", "")
        row.setdefault("anchor_image", "")
        row.setdefault("anchor_position", "")
        row.setdefault("anchor_seconds", "")
        rows.append(row)
    write_outpaint_chunk_rows(manifest, rows)
    view_rows = []
    for row in rows:
        raw = resolve(row["raw_path"])
        prepared = resolve(row["prepared_path"])
        start_seconds = float(row["start_seconds"])
        end_seconds = float(row["end_seconds"])
        middle_seconds = (start_seconds + end_seconds) / 2
        anchor_path = resolve(row["anchor_image"]) if row.get("anchor_image") else None
        anchor_exists = bool(anchor_path and anchor_path.exists())
        try:
            anchor_seconds = float(row.get("anchor_seconds", "") or max(0.0, middle_seconds - start_seconds))
        except ValueError:
            anchor_seconds = max(0.0, middle_seconds - start_seconds)
        anchor_seconds = max(0.0, min(max(0.0, end_seconds - start_seconds), anchor_seconds))
        anchor_source_seconds = start_seconds + anchor_seconds
        view_rows.append(row | {
            "index": int(row["chunk_index"]),
            "start": float(row["start_seconds"]),
            "end": float(row["end_seconds"]),
            "fps": fps,
            "total_frames": total_frames,
            "length_frames": int(row["end_frame"]) - int(row["start_frame"]),
            "max_length_frames": max(1, total_frames - int(row["start_frame"])),
            "start_label": format_timecode(float(row["start_seconds"])),
            "end_label": format_timecode(float(row["end_seconds"])),
            "raw_exists": raw.exists(),
            "raw_mtime": int(raw.stat().st_mtime_ns) if raw.exists() else 0,
            "prepared_exists": prepared.exists(),
            "anchor_exists": anchor_exists,
            "anchor_mtime": int(anchor_path.stat().st_mtime_ns) if anchor_exists and anchor_path else 0,
            "anchor_seconds": f"{anchor_seconds:.6f}",
            "anchor_frame_preview": chunk_frame_preview(range_source, anchor_source_seconds, "source_guide"),
            "source_start_preview": chunk_frame_preview(range_source, start_seconds, "source_start"),
            "source_middle_preview": chunk_frame_preview(range_source, middle_seconds, "source_middle"),
            "source_end_preview": chunk_frame_preview(range_source, max(start_seconds, end_seconds - (1 / max(1.0, fps))), "source_end"),
            "raw_start_preview": chunk_frame_preview(raw, 0.0, "raw_start") if raw.exists() else "",
            "raw_middle_preview": chunk_frame_preview(raw, max(0.0, (end_seconds - start_seconds) / 2), "raw_middle") if raw.exists() else "",
            "raw_end_preview": chunk_frame_preview(raw, max(0.0, end_seconds - start_seconds - (1 / max(1.0, fps))), "raw_end") if raw.exists() else "",
        })
    return {"manifest": rel(manifest), "rows": view_rows}


def chunk_frame_preview(source: Path, seconds: float, suffix: str) -> str:
    if not source.exists():
        return ""
    return extract_video_frame_at(source, FILE_PREVIEW_DIR / "chunks", f"{suffix}_{int(seconds * 1000):010d}", seconds)


def outpaint_chunk_ranges(total_frames: int, fps: float, default_seconds: float, overlap_frames: int, existing: dict[int, dict[str, str]]) -> list[tuple[int, int, int]]:
    ranges = []
    start = 0
    index = 0
    while start < total_frames:
        seconds = default_seconds
        custom = existing.get(index, {}).get("custom_seconds", "")
        if custom:
            try:
                seconds = float(custom)
            except ValueError:
                seconds = default_seconds
        chunk_frames = total_frames if seconds <= 0 else max(1, int(round(seconds * fps)))
        end = min(total_frames, start + chunk_frames)
        ranges.append((index, start, end))
        if end >= total_frames:
            break
        overlap = max(0, min(overlap_frames, chunk_frames - 1))
        start += max(1, chunk_frames - overlap)
        index += 1
    return ranges


def update_outpaint_chunk(index: int, seed: str, prompt_suffix: str, custom_seconds: str = "", negative_suffix: str = "") -> None:
    state = outpaint_chunks_state(APP.settings)
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    rows = read_outpaint_chunk_rows(resolve(str(manifest_text)))
    if index not in rows:
        raise IndexError(f"Outpaint chunk not found: {index + 1}")
    row = rows[index]
    row["seed"] = str(int(float(seed or row.get("seed") or 42 + index)))
    row["prompt_suffix"] = prompt_suffix
    row["negative_suffix"] = negative_suffix
    if custom_seconds:
        row["custom_seconds"] = f"{max(0.1, float(custom_seconds)):.3f}"
    else:
        row["custom_seconds"] = ""
    ordered = [rows[key] for key in sorted(rows)]
    write_outpaint_chunk_rows(resolve(str(manifest_text)), ordered)
    APP.log.append(f"Saved outpaint chunk {index + 1}: seed {row['seed']}")


def remove_cached_file(path: Path) -> bool:
    removed = False
    for candidate in (path, path.with_suffix(path.suffix + ".sig.json"), path.with_suffix(path.suffix + ".partial")):
        try:
            if candidate.exists() and candidate.is_file():
                candidate.unlink()
                removed = True
        except PermissionError:
            APP.log.append(f"Could not delete cached file because it is open in another process: {rel(candidate)}")
        except OSError as exc:
            APP.log.append(f"Could not delete cached file {rel(candidate)}: {exc}")
    return removed


def clear_cached_outpaint_guides(manifest: Path, index: int) -> int:
    guide_dir = ROOT / "intermediate" / "outpaint_anchors" / manifest.stem
    if not guide_dir.exists():
        return 0
    removed = 0
    for path in guide_dir.glob(f"chunk_{index:04d}_*"):
        if path.is_file() and remove_cached_file(path):
            removed += 1
    return removed


def install_outpaint_anchor(index: int, seconds: str) -> dict[str, str]:
    state = outpaint_chunks_state(APP.settings)
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    manifest = resolve(str(manifest_text))
    rows = read_outpaint_chunk_rows(manifest)
    if index not in rows:
        raise IndexError(f"Outpaint chunk not found: {index + 1}")

    current = rows[index].get("anchor_image", "")
    selected = browse_path("image", current)
    if not selected:
        return {"selected": "", "anchor_image": current}

    source = resolve(selected)
    if source.suffix.lower() not in IMAGE_EXTS:
        raise RuntimeError("Choose a PNG or JPEG image for the outpaint anchor frame.")
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(source)

    try:
        guide_seconds = max(0.0, float(seconds or 0))
    except ValueError:
        guide_seconds = 0.0
    target_dir = ROOT / "intermediate" / "outpaint_anchors" / manifest.stem
    target = target_dir / f"chunk_{index:04d}_guide{source.suffix.lower()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

    rows[index]["anchor_image"] = rel(target)
    rows[index]["anchor_position"] = "guide"
    rows[index]["anchor_seconds"] = f"{guide_seconds:.6f}"
    write_outpaint_chunk_rows(manifest, [rows[key] for key in sorted(rows)])
    APP.log.append(f"Installed outpaint guide frame for chunk {index + 1}: {rel(target)}")
    return {"selected": selected, "anchor_image": rel(target), "anchor_position": "guide", "anchor_seconds": f"{guide_seconds:.6f}"}


def clear_outpaint_anchor(index: int) -> dict[str, str]:
    state = outpaint_chunks_state(APP.settings)
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    manifest = resolve(str(manifest_text))
    rows = read_outpaint_chunk_rows(manifest)
    if index not in rows:
        raise IndexError(f"Outpaint chunk not found: {index + 1}")
    removed = clear_cached_outpaint_guides(manifest, index)
    rows[index]["anchor_image"] = ""
    rows[index]["anchor_position"] = ""
    rows[index]["anchor_seconds"] = ""
    write_outpaint_chunk_rows(manifest, [rows[key] for key in sorted(rows)])
    suffix = f" and deleted {removed} cached file(s)" if removed else ""
    APP.log.append(f"Cleared outpaint guide frame for chunk {index + 1}{suffix}")
    return {"anchor_image": "", "anchor_position": "", "anchor_seconds": ""}


def update_outpaint_guide_time(index: int, seconds: str) -> dict[str, str]:
    state = outpaint_chunks_state(APP.settings)
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    manifest = resolve(str(manifest_text))
    rows = read_outpaint_chunk_rows(manifest)
    view_rows = state.get("rows", [])
    if index not in rows or index < 0 or index >= len(view_rows):
        raise IndexError(f"Outpaint chunk not found: {index + 1}")

    row = view_rows[index]
    try:
        guide_seconds = max(0.0, float(seconds or 0))
    except ValueError:
        guide_seconds = 0.0
    chunk_length = max(0.0, float(row.get("end", 0.0)) - float(row.get("start", 0.0)))
    guide_seconds = min(chunk_length, guide_seconds)
    rows[index]["anchor_seconds"] = f"{guide_seconds:.6f}"
    write_outpaint_chunk_rows(manifest, [rows[key] for key in sorted(rows)])
    return {"anchor_seconds": f"{guide_seconds:.6f}"}


def outpaint_anchor_generation_command(index: int, seconds: str, prompt: str) -> tuple[list[str], str]:
    state = outpaint_chunks_state(APP.settings)
    rows = state.get("rows", [])
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    if index < 0 or index >= len(rows):
        raise IndexError(f"Outpaint chunk not found: {index + 1}")

    row = rows[index]
    try:
        guide_seconds = max(0.0, float(seconds or row.get("anchor_seconds") or 0))
    except ValueError:
        guide_seconds = 0.0
    chunk_length = max(0.0, float(row.get("end", 0.0)) - float(row.get("start", 0.0)))
    guide_seconds = min(chunk_length, guide_seconds)
    source_text = pipeline_source_text(APP.settings)
    if not source_text:
        raise RuntimeError("No source material is selected.")
    range_source = ensure_outpaint_prepared_canvas(source_text, APP.settings.get("outpaint", {}))
    source = resolve(chunk_frame_preview(range_source, float(row.get("start", 0.0)) + guide_seconds, "source_guide_qwen"))
    if not source.exists():
        raise FileNotFoundError(source)

    manifest = resolve(str(manifest_text))
    output_dir = ROOT / "intermediate" / "outpaint_anchors" / manifest.stem
    output = output_dir / f"chunk_{index:04d}_guide_qwen.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_cached_file(output)

    stored = read_outpaint_chunk_rows(manifest)
    if index not in stored:
        raise IndexError(f"Outpaint chunk not found in manifest: {index + 1}")
    stored[index]["anchor_image"] = rel(output)
    stored[index]["anchor_position"] = "guide"
    stored[index]["anchor_seconds"] = f"{guide_seconds:.6f}"
    write_outpaint_chunk_rows(manifest, [stored[key] for key in sorted(stored)])

    values = APP.settings.get("references", {})
    config = current_config()
    workflow = values.get("workflow") or default_qwen_workflow(config)
    if not workflow:
        raise RuntimeError("No Qwen Image Edit workflow found. Install/configure ComfyUI first.")
    anchor_prompt = prompt.strip() or DEFAULT_ANCHOR_PROMPT
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPTS / "generate_single_reference.py"),
        "--source-image",
        str(source),
        "--output",
        str(output),
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
        anchor_prompt,
        "--prompt-suffix",
        "",
        "--load-image-node-id",
        values.get("load_image_node_id", "auto"),
        "--save-node-id",
        values.get("save_node_id", "auto"),
        "--no-normalize-to-source-size",
        "--force",
    ]
    if values.get("prompt_node_id"):
        cmd.extend(["--prompt-node-id", values["prompt_node_id"]])
    return cmd, rel(output)


def recomposition_output_for(outpainted_text: str) -> str:
    if not outpainted_text:
        return ""
    outpainted = resolve(outpainted_text)
    return rel(ROOT / "output" / "reassembled" / f"{safe_stem(outpainted.name)}_final.mp4")


def colorized_outputs_for_manifest(manifest_text: str, method: str = "deepexemplar") -> list[str]:
    if method == "both":
        return [path for path in (colorized_output_for_manifest(manifest_text, "deepexemplar"), colorized_output_for_manifest(manifest_text, "colormnet")) if path]
    output = colorized_output_for_manifest(manifest_text, method)
    return [output] if output else []


def colorized_output_for_manifest(manifest_text: str, method: str = "deepexemplar") -> str:
    if not manifest_text:
        return ""
    if method == "both":
        return ""
    suffix = "colormnet" if method == "colormnet" else "deepexemplar"
    manifest = resolve(manifest_text)
    source_video = manifest_source_video(manifest)
    if source_video:
        source = resolve(source_video)
        return rel(ROOT / "intermediate" / "outpainted_colorized" / f"{safe_stem(source.name)}_{suffix}_colorized.mp4")
    if manifest_text:
        stem = safe_stem(Path(manifest_text).stem.replace("colorize_manifest_", "").replace("_shots_auto", ""))
        return rel(ROOT / "intermediate" / "outpainted_colorized" / f"{stem}_{suffix}_colorized.mp4")
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
                "can_split": end - start >= 0.1,
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


def install_custom_color_reference(manifest_text: str, index: int) -> dict[str, str]:
    manifest = resolve(manifest_text)
    _source, _fields, rows = read_manifest_details(manifest)
    if index < 0 or index >= len(rows):
        raise IndexError("Shot index out of range.")

    selected = browse_path("file", rows[index].get("color_reference", "") or rows[index].get("source_reference", ""))
    if not selected:
        return {"selected": "", "color_reference": rows[index].get("color_reference", "")}

    source = resolve(selected)
    if source.suffix.lower() not in IMAGE_EXTS:
        raise RuntimeError("Choose a PNG or JPEG image for the custom color reference.")
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(source)

    current_target = rows[index].get("color_reference", "")
    if current_target:
        target_base = resolve(current_target)
        target = target_base.with_suffix(source.suffix.lower())
    elif rows[index].get("source_reference"):
        target = resolve(color_reference_for_source(rows[index]["source_reference"])).with_suffix(source.suffix.lower())
    else:
        target = ROOT / "intermediate" / "outpainted_references_color" / "custom" / f"shot_{index + 1:04d}{source.suffix.lower()}"

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    update_manifest_row(manifest, index, {"color_reference": rel(target)})
    APP.log.append(f"Installed custom color reference for shot {index + 1}: {rel(target)}")
    return {"selected": selected, "color_reference": rel(target)}


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


def aspect_preview_for_settings(settings: dict) -> str:
    source_text = preview_pipeline_source_text(settings)
    if not source_text:
        return ""
    signature = source_signature(source_text)
    if signature is None:
        return ""
    seconds = 0.0 if source_section_is_active(settings) else 10.0
    return aspect_preview_cached(
        signature[0],
        signature[1],
        signature[2],
        settings.get("outpaint", {}).get("target_aspect", "16:9"),
        current_crop_values(),
        seconds,
    )


def aspect_preview_at(source_text: str, aspect: str, seconds: float) -> str:
    signature = source_signature(source_text)
    if signature is None:
        return ""
    return aspect_preview_cached(signature[0], signature[1], signature[2], aspect, current_crop_values(), round(max(0.0, seconds), 3))


def aspect_preview_at_for_settings(settings: dict, seconds: float) -> str:
    source_text = preview_pipeline_source_text(settings)
    if not source_text:
        return ""
    relative_seconds = section_relative_seconds(settings, seconds)
    return aspect_preview_at(source_text, settings.get("outpaint", {}).get("target_aspect", "16:9"), relative_seconds)


def preview_pipeline_source_text(settings: dict) -> str:
    try:
        ensure_source_section_clip(settings)
    except Exception as exc:
        APP.log.append(f"Could not prepare selected source section for preview: {exc}")
    return pipeline_source_text(settings)


def section_relative_seconds(settings: dict, seconds: float) -> float:
    if not source_section_is_active(settings):
        return seconds
    start = section_float(settings.get("global", {}).get("section_start", "0"), 0.0)
    end = section_float(settings.get("global", {}).get("section_end", ""), 0.0)
    return max(0.0, min(end - start, seconds - start))


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
    try:
        if target.exists() and target.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            return rel(target)
    except OSError:
        pass
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


def video_metrics(source: Path) -> dict[str, float]:
    found = local_tool("ffprobe")
    if found:
        command = [
            str(found),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate,nb_frames,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(source),
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                stream = (data.get("streams") or [{}])[0]
                fmt = data.get("format") or {}
                fps = parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate")) or 24.0
                duration = float(stream.get("duration") or fmt.get("duration") or 0)
                frames = int(str(stream.get("nb_frames") or "0").replace(",", "") or "0")
                if frames <= 0 and duration > 0:
                    frames = int(round(duration * fps))
                if frames > 0:
                    return {"fps": fps, "frames": float(frames), "duration": duration or frames / fps}
            except (ValueError, TypeError, json.JSONDecodeError, IndexError):
                pass
    try:
        import cv2

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            return {}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return {"fps": fps, "frames": float(frames), "duration": frames / fps if frames and fps else 0.0}
    except Exception:
        return {}


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


def cache_categories() -> tuple[dict, ...]:
    return (
        {
            "key": "global",
            "title": "Overview",
            "description": "Source thumbnails, target-aspect previews, selected source sections, and small browser preview clips.",
            "folders": (
                PREVIEW_DIR,
                FILE_PREVIEW_DIR,
                ASPECT_PREVIEW_DIR,
                MEDIA_CLIP_DIR,
                ROOT / "intermediate" / "source_sections",
            ),
        },
        {
            "key": "outpaint",
            "title": "Outpainting",
            "description": "Prepared inputs, guide frames, per-chunk LTX renders, chunk manifests, and stitched outpainted videos.",
            "folders": (
                ROOT / ".cache" / "outpaint_chunks",
                ROOT / "intermediate" / "outpaint_anchors",
                ROOT / "intermediate" / "outpaint_prepared",
                ROOT / "intermediate" / "outpainted",
                ROOT / "manifests" / "outpaint_chunks",
            ),
        },
        {
            "key": "shots",
            "title": "Shot Detection",
            "description": "Shot manifests created by cut detection.",
            "folders": (ROOT / "manifests" / "references",),
        },
        {
            "key": "references",
            "title": "Reference Generation",
            "description": "Black-and-white shot screenshots and Qwen color reference stills.",
            "folders": (
                ROOT / "intermediate" / "outpainted_references",
                ROOT / "intermediate" / "outpainted_references_color",
            ),
        },
        {
            "key": "colour",
            "title": "Colorization",
            "description": "Per-shot colorized chunks and stitched Deep Exemplar colorized videos.",
            "folders": (
                ROOT / ".cache" / "colorized_chunks",
                ROOT / "intermediate" / "outpainted_colorized",
            ),
        },
        {
            "key": "recomp",
            "title": "Recomposition",
            "description": "Final recomposited movies created by the Recomposition tab.",
            "folders": (),
        },
        {
            "key": "output",
            "title": "Output",
            "description": "Finished output movies shown on the Output tab.",
            "folders": (ROOT / "output" / "reassembled",),
        },
    )


def cache_state() -> dict:
    categories = []
    grand_total = 0
    grand_count = 0

    for category in cache_categories():
        files = cache_category_files(category)
        total = sum(int(file["size"]) for file in files)
        grand_total += total
        grand_count += len(files)
        categories.append(
            {
                "key": category["key"],
                "title": category["title"],
                "description": category["description"],
                "count": len(files),
                "total": total,
                "total_label": human_size(total),
                "files": files,
            }
        )

    return {
        "count": grand_count,
        "total": grand_total,
        "total_label": human_size(grand_total),
        "categories": categories,
    }


def cache_category_files(category: dict) -> list[dict]:
    files = []
    for folder in category["folders"]:
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if not cache_file_is_listable(path):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append(
                {
                    "path": rel(path),
                    "size": stat.st_size,
                    "size_label": human_size(stat.st_size),
                    "mtime": int(stat.st_mtime),
                }
            )
    return sorted(files, key=lambda item: str(item["path"]).lower())


def cache_file_is_listable(path: Path) -> bool:
    return path.is_file() and path.name != ".gitkeep" and not path.name.endswith((".partial", ".tmp"))


def delete_cache_file(path_text: str) -> dict:
    path = resolve(path_text)
    category = cache_category_for_path(path)
    if category is None:
        raise ValueError("That file is not in an ARP cache/intermediate category.")
    try:
        if not path.is_file():
            return {"deleted": 0, "bytes": 0}
        size = path.stat().st_size
        path.unlink()
    except FileNotFoundError:
        return {"deleted": 0, "bytes": 0}
    clean_empty_cache_dirs(category)
    APP.log.append(f"Deleted cached file: {rel(path)}")
    return {"deleted": 1, "bytes": size}


def delete_cache_category(category_key: str) -> dict:
    if category_key == "all":
        total = {"deleted": 0, "bytes": 0}
        for category in cache_categories():
            result = delete_cache_category(category["key"])
            total["deleted"] += result["deleted"]
            total["bytes"] += result["bytes"]
        APP.log.append(f"Cleared all ARP cache categories: {total['deleted']} files, {human_size(total['bytes'])}.")
        return total

    category = next((item for item in cache_categories() if item["key"] == category_key), None)
    if category is None:
        raise ValueError("Unknown cache category.")

    deleted = 0
    bytes_deleted = 0
    for file in cache_category_files(category):
        path = resolve(str(file["path"]))
        try:
            size = path.stat().st_size
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            APP.log.append(f"Could not delete cached file {rel(path)}: {exc}")
            continue
        deleted += 1
        bytes_deleted += size

    clean_empty_cache_dirs(category)
    APP.log.append(f"Cleared {category['title']}: {deleted} files, {human_size(bytes_deleted)}.")
    return {"deleted": deleted, "bytes": bytes_deleted}


def cache_category_for_path(path: Path) -> dict | None:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path

    for category in cache_categories():
        for folder in category["folders"]:
            try:
                resolved.relative_to(folder.resolve())
                return category
            except ValueError:
                continue
    return None


def clean_empty_cache_dirs(category: dict) -> None:
    for folder in category["folders"]:
        if not folder.exists():
            continue
        for path in sorted((item for item in folder.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True):
            try:
                path.rmdir()
            except OSError:
                pass


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


def export_media_file(path_text: str) -> dict[str, str]:
    source = resolve(path_text)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(source)

    target_text = browse_path("save_image" if source.suffix.lower() in IMAGE_EXTS else "save", str(source))
    if not target_text:
        return {"saved": ""}
    target = resolve(target_text)
    if target.suffix == "":
        target = target.with_suffix(source.suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.resolve() != source.resolve():
        shutil.copy2(source, target)
    APP.log.append(f"Saved media file: {rel(source)} -> {target}")
    return {"saved": str(target)}


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
    initial = browse_initial_path(kind, current)
    if os.name == "nt":
        selected = browse_path_windows(kind, initial)
    elif sys.platform == "darwin":
        selected = browse_path_macos(kind, initial)
    else:
        selected = browse_path_linux(kind, initial)
    remember_browse_dir(selected)
    return selected


def browse_initial_path(kind: str, current: str = "") -> Path:
    save_kinds = {"save", "save_image", "project_save"}
    last_dir = last_browse_dir()
    if not current:
        return last_dir or ROOT

    current_path = resolve(current)
    if kind in save_kinds:
        if last_dir:
            return last_dir / (current_path.name or "output")
        if current_path.parent.exists():
            return current_path
        return current_path

    if last_dir:
        return last_dir
    if current_path.exists():
        return current_path if current_path.is_dir() else current_path.parent
    if current_path.parent.exists():
        return current_path.parent
    return ROOT


def remember_browse_dir(selected: str) -> None:
    if not selected or "APP" not in globals():
        return
    path = resolve(selected)
    folder = path if path.is_dir() else path.parent
    if not folder.exists():
        return
    APP.settings.setdefault("global", {})["last_browse_dir"] = str(folder)
    APP.save()


def browse_path_windows(kind: str, initial: Path) -> str:
    initial_dir = initial if initial.is_dir() else initial.parent
    initial_text = str(initial_dir).replace("'", "''")
    initial_file = "" if initial.is_dir() else initial.name.replace("'", "''")
    if kind == "folder":
        script = f"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.SelectedPath = '{initial_text}'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{ [Console]::Out.Write($dialog.SelectedPath) }}
"""
    elif kind in {"save", "save_image", "project_save"}:
        filter_text = (
            "ARP project files (*.arpp)|*.arpp|All files (*.*)|*.*"
            if kind == "project_save"
            else "Image files (*.png;*.jpg;*.jpeg;*.webp)|*.png;*.jpg;*.jpeg;*.webp|All files (*.*)|*.*"
            if kind == "save_image"
            else "Video files (*.mp4;*.mov;*.mkv;*.webm)|*.mp4;*.mov;*.mkv;*.webm|All files (*.*)|*.*"
        )
        script = f"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.SaveFileDialog
$dialog.InitialDirectory = '{initial_text}'
$dialog.FileName = '{initial_file}'
$dialog.Filter = '{filter_text}'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{ [Console]::Out.Write($dialog.FileName) }}
"""
    else:
        filter_text = (
            "ARP project files (*.arpp)|*.arpp|All files (*.*)|*.*"
            if kind == "project_open"
            else "Media/workflow files (*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v;*.png;*.jpg;*.jpeg;*.json;*.csv)|*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v;*.png;*.jpg;*.jpeg;*.json;*.csv|All files (*.*)|*.*"
        )
        script = f"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.InitialDirectory = '{initial_text}'
$dialog.Filter = '{filter_text}'
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
    initial_dir = initial if initial.is_dir() else initial.parent
    initial_script = applescript_quote(str(initial_dir))
    if kind == "folder":
        script = f'set chosen to choose folder with prompt "Choose folder" default location POSIX file {initial_script}\nPOSIX path of chosen'
    elif kind in {"save", "save_image", "project_save"}:
        default_name = applescript_quote("" if initial.is_dir() else initial.name)
        script = f'set chosen to choose file name with prompt "Choose output path" default location POSIX file {initial_script} default name {default_name}\nPOSIX path of chosen'
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
    save_kind = kind in {"save", "save_image", "project_save"}
    filename = str(initial if save_kind and not initial.is_dir() else initial) + ("" if save_kind and not initial.is_dir() else os.sep)
    command = ["zenity", "--file-selection", f"--filename={filename}"]
    if kind == "folder":
        command.append("--directory")
    elif kind in {"save", "save_image", "project_save"}:
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
    elif kind in {"save", "save_image", "project_save"}:
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
            self.send_static(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif parsed.path.startswith("/static/"):
            static_path = STATIC_DIR / unquote(parsed.path.removeprefix("/static/"))
            try:
                static_path.resolve().relative_to(STATIC_DIR.resolve())
            except ValueError:
                self.send_error(404)
                return
            self.send_static(static_path)
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
                path = aspect_preview_at_for_settings(APP.settings, float(query.get("time", ["0"])[0]))
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
                except FileNotFoundError:
                    self.send_error(404)
                    return
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
        elif parsed.path == "/api/shot-split":
            try:
                result = split_manifest_shot(str(data.get("manifest", "")), int(data.get("index", 0)))
                self.send_json({"ok": True, **result, "state": APP.state()})
            except Exception as exc:
                APP.log.append(f"Shot split failed: {exc}")
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
        elif parsed.path == "/api/reference-custom":
            try:
                result = install_custom_color_reference(str(data.get("manifest", "")), int(data.get("index", 0)))
                self.send_json({"ok": True, **result, "state": APP.state()})
            except Exception as exc:
                APP.log.append(f"Custom reference install failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/export-media":
            try:
                result = export_media_file(str(data.get("path", "")))
                self.send_json({"ok": True, **result})
            except Exception as exc:
                APP.log.append(f"Media export failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-chunk":
            try:
                update_outpaint_chunk(int(data.get("index", 0)), str(data.get("seed", "")), str(data.get("prompt_suffix", "")), str(data.get("custom_seconds", "")), str(data.get("negative_suffix", "")))
                self.send_json({"ok": True, "state": APP.state()})
            except Exception as exc:
                APP.log.append(f"Outpaint chunk save failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-chunk-regenerate":
            try:
                update_outpaint_chunk(int(data.get("index", 0)), str(data.get("seed", "")), str(data.get("prompt_suffix", "")), str(data.get("custom_seconds", "")), str(data.get("negative_suffix", "")))
                ok, message = APP.run_outpaint_chunk(int(data.get("index", 0)))
                self.send_json({"ok": ok, "message": message, "state": APP.state() if ok else None, "error": "" if ok else message})
            except Exception as exc:
                APP.log.append(f"Outpaint chunk regeneration failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-anchor":
            try:
                result = install_outpaint_anchor(int(data.get("index", 0)), str(data.get("seconds", "")))
                self.send_json({"ok": True, **result, "state": APP.state()})
            except Exception as exc:
                APP.log.append(f"Outpaint anchor install failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-anchor-clear":
            try:
                result = clear_outpaint_anchor(int(data.get("index", 0)))
                self.send_json({"ok": True, **result, "state": APP.state()})
            except Exception as exc:
                APP.log.append(f"Outpaint guide clear failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-guide-time":
            try:
                result = update_outpaint_guide_time(int(data.get("index", 0)), str(data.get("seconds", "")))
                self.send_json({"ok": True, **result, "state": APP.state()})
            except Exception as exc:
                APP.log.append(f"Outpaint guide time update failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-anchor-generate":
            try:
                ok, message = APP.run_outpaint_anchor_generation(
                    int(data.get("index", 0)),
                    str(data.get("seconds", "")),
                    str(data.get("prompt", "")),
                )
                self.send_json({"ok": ok, "message": message, "state": APP.state() if ok else None, "error": "" if ok else message})
            except Exception as exc:
                APP.log.append(f"Outpaint anchor generation failed: {exc}")
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
        elif parsed.path == "/api/project-save":
            try:
                result = APP.save_project(bool(data.get("save_as")))
                self.send_json({"ok": True, **result, "state": APP.state()})
            except Exception as exc:
                APP.log.append(f"Project save failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/project-load":
            try:
                result = APP.load_project()
                self.send_json({"ok": True, **result, "state": APP.state()})
            except Exception as exc:
                APP.log.append(f"Project load failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/cache-delete":
            try:
                if data.get("all"):
                    result = delete_cache_category("all")
                elif data.get("category"):
                    result = delete_cache_category(str(data.get("category", "")))
                else:
                    result = delete_cache_file(str(data.get("path", "")))
                self.send_json({"ok": True, **result, "state": APP.state()})
            except Exception as exc:
                APP.log.append(f"Cache delete failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
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
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def send_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_static(self, path: Path, content_type: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        mime = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_media(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        try:
            file_size = path.stat().st_size
        except FileNotFoundError:
            self.send_error(404)
            return
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
        except FileNotFoundError:
            return
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return



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


APP.normalize_loaded_source_state()


def main() -> int:
    os.chdir(ROOT)
    install_shutdown_handlers()
    if os.environ.get("AI_REMASTER_NO_COMFY_AUTOSTART") != "1":
        start_comfy_if_needed()
    host = "127.0.0.1"
    requested_port = int(os.environ.get("AI_REMASTER_GUI_PORT", "8765"))
    server = create_server(host, requested_port)
    url = f"http://{host}:{server.server_port}/"
    print(f"AI Remaster GUI {APP_VERSION} running at {url}")
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
