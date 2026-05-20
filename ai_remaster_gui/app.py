from __future__ import annotations

import csv
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
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
TEXT_EXTS = {".csv", ".json", ".txt", ".log", ".md"}


def load_config() -> dict[str, str]:
    config = {
        "comfy_dir": str(ROOT / "tools" / "comfyui"),
        "comfy_url": "http://127.0.0.1:8188",
        "comfy_host": "127.0.0.1",
        "comfy_port": "8188",
    }
    if CONFIG_FILE.exists():
        try:
            stored = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                config.update({key: str(value) for key, value in stored.items() if value is not None})
        except json.JSONDecodeError:
            pass
    return config


CONFIG = load_config()
STARTED_COMFY_PROCESS: subprocess.Popen | None = None


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
            ("shot_threshold", "Shot threshold", "number", "0.09"),
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
            ("workflow", "Qwen workflow", "file", ""),
            ("comfy_output_root", "Comfy output folder", "folder", "tools/comfyui/output"),
            ("comfy_url", "Comfy URL", "text", "http://127.0.0.1:8188"),
            ("prompt", "Prompt", "text", "Colorize this image."),
            ("prompt_suffix", "Prompt suffix", "text", "Natural period color, preserve lighting and composition."),
            ("load_image_node_id", "Load image node", "text", "1"),
            ("prompt_node_id", "Prompt node", "text", ""),
            ("save_node_id", "Save node", "text", ""),
            ("limit", "Limit rows", "number", ""),
        ),
        ("manifest", "workflow", "save_node_id"),
    ),
    Stage(
        "colour",
        "Colourisation",
        "Run your ComfyUI reference-video colorizer over the manifest.",
        ("intermediate/outpainted_references_color", "intermediate/outpainted_colorized", "manifests/references"),
        (
            ("manifest", "Manifest", "file", ""),
            ("comfy_runner", "Colorizer runner", "file", ""),
            ("method", "Method", "text", "DeepExemplar"),
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
    defaults["global"] = {"source": ""}
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
    for values in defaults.values():
        if outpainted and not values.get("outpainted_video"):
            values["outpainted_video"] = rel(outpainted)
        if manifest and not values.get("manifest"):
            values["manifest"] = rel(manifest)
        if colorized and not values.get("colorized_video"):
            values["colorized_video"] = rel(colorized)
    if outpainted and not defaults["recomp"].get("output"):
        defaults["recomp"]["output"] = rel(ROOT / "output" / "reassembled" / f"{outpainted.stem}_final.mp4")
    bundled_output = rel(ROOT / "tools" / "comfyui" / "output")
    if not defaults["references"].get("comfy_output_root") or (CONFIG_FILE.exists() and defaults["references"].get("comfy_output_root") == bundled_output):
        defaults["references"]["comfy_output_root"] = rel(Path(CONFIG["comfy_dir"]) / "output")
    if not defaults["references"].get("comfy_url"):
        defaults["references"]["comfy_url"] = CONFIG["comfy_url"]
    return defaults


class PipelineApp:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.log: list[str] = []
        self.process: subprocess.Popen[str] | None = None
        self.running_stage = ""
        self.running_stage_key = ""
        self.run_started_at = 0.0
        self.lock = threading.Lock()

    def save(self) -> None:
        SETTINGS_FILE.write_text(json.dumps(self.settings, indent=2) + "\n", encoding="utf-8")

    def files_for(self, stage: Stage) -> list[dict[str, str | int]]:
        exts = VIDEO_EXTS | IMAGE_EXTS | TEXT_EXTS
        out = []
        for folder_text in stage.folders:
            folder = ROOT / folder_text
            if not folder.exists():
                continue
            for path in folder.rglob("*"):
                if path.is_file() and path.suffix.lower() in exts:
                    stat = path.stat()
                    out.append({"path": rel(path), "size": stat.st_size, "mtime": int(stat.st_mtime), "preview": file_preview(path)})
        return sorted(out, key=lambda item: str(item["path"]).lower())

    def progress(self) -> list[dict[str, str]]:
        checks = {
            "Outpainting": newest(ROOT / "intermediate" / "outpainted", VIDEO_EXTS),
            "Shot Detection": newest(ROOT / "manifests" / "references", {".csv"}),
            "Reference Generation": newest(ROOT / "intermediate" / "outpainted_references_color", IMAGE_EXTS),
            "Colourisation": newest(ROOT / "intermediate" / "outpainted_colorized", VIDEO_EXTS),
            "Recomposition": newest(ROOT / "output" / "reassembled", VIDEO_EXTS),
        }
        return [{"stage": key, "status": "Ready" if value else "Waiting", "latest": rel(value) if value else ""} for key, value in checks.items()]

    def phase_progress(self) -> dict:
        current = self.estimate_running_progress()
        stages = []
        completed = 0.0
        for stage in STAGES:
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
        global_percent = int(round((completed / max(1, len(STAGES))) * 100))
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
            milestones = [
                ("reuse prepared outpaint input", 20, "Prepared input reused"),
                ("wrote prepared outpaint input", 20, "Prepared input written"),
                ("prepared comfy input", 25, "Prepared for ComfyUI"),
                ("waiting for comfyui", 30, "Waiting for ComfyUI"),
                ("queued comfyui prompt", 40, "Queued in ComfyUI"),
                ("wrote raw comfy render", 82, "Raw outpaint render written"),
                ("reuse raw comfy render", 82, "Raw outpaint render reused"),
                ("wrote outpainted video", 100, "Outpainted video written"),
            ]
            for token, value, text in milestones:
                if token in lower and value >= percent:
                    percent, label = value, text
        elif self.running_stage_key == "shots":
            if "detected " in lower:
                percent, label = max(percent, 75), "Shots detected"
            if "wrote manifest" in lower:
                percent, label = 100, "Manifest written"
        elif self.running_stage_key == "references":
            rows = first_int_after(log_text, "Rows:")
            done = count_lines_matching(log_text, ("Reuse ", "Wrote "))
            if rows:
                percent = min(99, int((done / rows) * 100))
                label = f"{done}/{rows} references"
        elif self.running_stage_key == "colour":
            if "reuse" in lower:
                percent, label = max(percent, 75), "Existing colorized video reused"
            if "wrote" in lower or "finished with exit code 0" in lower:
                percent, label = 100, "Colourisation complete"
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
                "stages": [stage.__dict__ | {"files": self.files_for(stage)} for stage in STAGES],
                "settings": self.settings,
                "progress": self.progress(),
                "phase_progress": self.phase_progress(),
                "expected_outputs": {stage.key: self.expected_outputs(stage.key) for stage in STAGES},
                "source_previews": previews,
                "source_info": info,
                "aspect_preview": aspect_preview(source_text, self.settings.get("outpaint", {}).get("target_aspect", "16:9")),
                "running": running,
                "running_stage": self.running_stage,
                "log": "\n".join(self.log[-800:]),
            }

    def update_settings(self, stage: str, values: dict[str, str]) -> None:
        self.settings.setdefault(stage, {}).update({key: str(value) for key, value in values.items()})
        if stage == "global" and "source" in values:
            self.log.append(f"Loading source material: {values.get('source')}")
        if stage == "shots" and "outpainted_video" in values:
            manifest = manifest_for_outpainted(values.get("outpainted_video", ""))
            self.settings.setdefault("references", {}).setdefault("manifest", manifest)
            self.settings.setdefault("colour", {}).setdefault("manifest", manifest)
        self.save()

    def hydrate_stage_inputs(self, completed_stage: str = "") -> None:
        expected_outpainted = resolve(self.expected_outputs("outpaint")[0]) if self.expected_outputs("outpaint") else None
        outpainted = expected_outpainted if expected_outpainted and expected_outpainted.exists() else newest(ROOT / "intermediate" / "outpainted", VIDEO_EXTS)
        if outpainted:
            outpainted_text = rel(outpainted)
            self.settings.setdefault("shots", {})["outpainted_video"] = outpainted_text
            self.settings.setdefault("recomp", {})["outpainted_video"] = outpainted_text
            manifest = manifest_for_outpainted(outpainted_text)
            self.settings.setdefault("references", {})["manifest"] = manifest
            self.settings.setdefault("colour", {})["manifest"] = manifest
            self.log.append(f"Updated Shot Detection input: {outpainted_text}")
        expected_manifest = resolve(self.expected_outputs("shots")[0]) if self.expected_outputs("shots") else None
        manifest = expected_manifest if expected_manifest and expected_manifest.exists() else newest(ROOT / "manifests" / "references", {".csv"})
        if manifest:
            manifest_text = rel(manifest)
            self.settings.setdefault("references", {})["manifest"] = manifest_text
            self.settings.setdefault("colour", {})["manifest"] = manifest_text
            self.log.append(f"Updated manifest inputs: {manifest_text}")
        colorized = newest(ROOT / "intermediate" / "outpainted_colorized", VIDEO_EXTS)
        if colorized:
            self.settings.setdefault("recomp", {})["colorized_video"] = rel(colorized)
        source = self.settings.get("global", {}).get("source")
        if source:
            self.settings.setdefault("recomp", {})["source"] = source
        output = recomposition_output_for(self.settings.get("recomp", {}).get("outpainted_video", ""))
        if output:
            self.settings.setdefault("recomp", {})["output"] = output
        self.save()

    def expected_outputs(self, stage_key: str) -> list[str]:
        values = self.settings.get(stage_key, {})
        if stage_key == "outpaint":
            source = self.settings.get("global", {}).get("source", "")
            return [outpaint_output_for(source, values.get("target_aspect", "16:9"))] if source else []
        if stage_key == "shots":
            return [manifest_for_outpainted(values.get("outpainted_video", ""))]
        if stage_key == "references":
            return color_reference_outputs(values.get("manifest", ""))
        if stage_key == "recomp":
            return [values.get("output") or recomposition_output_for(values.get("outpainted_video", ""))]
        return []

    def existing_outputs(self, stage_key: str) -> list[str]:
        return [path for path in self.expected_outputs(stage_key) if path and resolve(path).exists()]

    def command_for(self, stage_key: str) -> list[str]:
        values = self.settings[stage_key]
        py = sys.executable
        cmd = [py]
        add = cmd.extend
        if stage_key == "outpaint":
            cmd.append(str(SCRIPTS / "outpaint_video.py"))
            add(["--source", self.settings.get("global", {}).get("source", "")])
            add(["--target-aspect", values.get("target_aspect", "16:9")])
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
            add(["--manifest", values.get("manifest", ""), "--workflow", values.get("workflow", ""), "--comfy-url", values.get("comfy_url", "http://127.0.0.1:8188")])
            if values.get("comfy_output_root"):
                add(["--comfy-output-root", values["comfy_output_root"]])
            add(["--prompt", values.get("prompt", ""), "--prompt-suffix", values.get("prompt_suffix", ""), "--load-image-node-id", values.get("load_image_node_id", "1"), "--save-node-id", values.get("save_node_id", "")])
            if values.get("prompt_node_id"):
                add(["--prompt-node-id", values["prompt_node_id"]])
            if values.get("limit"):
                add(["--limit", values["limit"]])
        elif stage_key == "colour":
            cmd.append(str(SCRIPTS / "colorize_video.py"))
            add(["--manifest", values.get("manifest", "")])
            if values.get("comfy_runner"):
                add(["--comfy-runner", values["comfy_runner"]])
            if values.get("method"):
                add(["--method", values["method"]])
        elif stage_key == "recomp":
            cmd.append(str(SCRIPTS / "final_composite.py"))
            output = values.get("output") or recomposition_output_for(values.get("outpainted_video", ""))
            add(["--outpainted", values.get("outpainted_video", ""), "--source", values.get("source", ""), "--output", output])
            if values.get("colorized_video"):
                add(["--colorized", values["colorized_video"]])
            add(["--feather-pixels", values.get("feather_pixels", "80"), "--saturation", values.get("saturation", "0.82"), "--temperature", values.get("temperature", "-0.015"), "--color-opacity", values.get("color_opacity", "1.0"), "--encoder", values.get("encoder", "h264")])
        if values.get("force") == "true":
            cmd.append("--force")
        if values.get("dry_run") == "true":
            cmd.append("--dry-run")
        return [part for part in cmd if part != ""]

    def run_stage(self, stage_key: str) -> tuple[bool, str]:
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

    def run_all(self) -> tuple[bool, str]:
        threading.Thread(target=self._run_all_worker, daemon=True).start()
        return True, "Started whole remaster queue."

    def _run_all_worker(self) -> None:
        for stage in STAGES:
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
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if not line.startswith("#"):
                return list(csv.DictReader([line, *handle.readlines()]))
    return []


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["enabled", "end", "source_reference", "color_reference"], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})


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


def outpaint_output_for(source_text: str, aspect: str) -> str:
    if not source_text:
        return ""
    source = resolve(source_text)
    return rel(ROOT / "intermediate" / "outpainted" / f"{safe_stem(source.name)}_{aspect_slug(aspect)}_outpainted.mp4")


def recomposition_output_for(outpainted_text: str) -> str:
    if not outpainted_text:
        return ""
    outpainted = resolve(outpainted_text)
    return rel(ROOT / "output" / "reassembled" / f"{safe_stem(outpainted.name)}_final.mp4")


def color_reference_outputs(manifest_text: str) -> list[str]:
    if not manifest_text:
        return []
    rows = read_manifest(resolve(manifest_text))
    return [row.get("color_reference", "") for row in rows if row.get("color_reference")]


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


@lru_cache(maxsize=16)
def source_info_cached(source_path: str, size: int, _mtime_ns: int) -> tuple[tuple[str, str], ...]:
    source = Path(source_path)
    APP.log.append(f"Probing source file info: {source}")
    info: dict[str, str] = {"file": rel(source), "size": human_size(size)}
    info.update(ffprobe_info(source))
    return tuple(info.items())


def aspect_preview(source_text: str, aspect: str) -> str:
    signature = source_signature(source_text)
    if signature is None:
        return ""
    return aspect_preview_cached(signature[0], signature[1], signature[2], aspect)


@lru_cache(maxsize=32)
def aspect_preview_cached(source_path: str, _size: int, mtime_ns: int, aspect: str) -> str:
    source = Path(source_path)
    source_frame = extract_video_frame(source, ASPECT_PREVIEW_DIR / "frames", "aspect")
    if not source_frame:
        return ""
    target = ASPECT_PREVIEW_DIR / f"{safe_preview_name(source)}_{aspect_slug(aspect)}.jpg"
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
    ffmpeg = local_tool("ffmpeg")
    if not ffmpeg:
        return ""
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{safe_preview_name(source)}_{suffix}.jpg"
    command = [ffmpeg, "-y", "-ss", "10", "-i", str(source), "-frames:v", "1", "-q:v", "4", str(target)]
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
        elif parsed.path == "/api/manifest":
            path = resolve(parse_qs(parsed.query).get("path", [""])[0])
            self.send_json({"rows": read_manifest(path)})
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
        elif parsed.path == "/media":
            self.send_media(resolve(unquote(parse_qs(parsed.query).get("path", [""])[0])))
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
        elif parsed.path == "/api/manifest":
            write_manifest(resolve(str(data.get("path", ""))), data.get("rows", []))
            self.send_json({"ok": True})
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
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                self.wfile.write(chunk)

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
.source-info{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:12px 0}.source-info div{background:#11181d;border:1px solid var(--line);border-radius:6px;padding:8px}.source-info span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}.source-info strong{display:block;margin-top:2px;font-size:13px;word-break:break-word}
table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid var(--line);padding:8px;text-align:left}th{color:#b8cbd1;background:#202a31}.status-ready{color:#75d6b9}.status-waiting{color:var(--warn)}
.hidden{display:none}.command{font-size:12px;color:#c5d5da;word-break:break-all}.manifest td input{border:0;border-radius:0;background:#11181d}
</style>
</head>
<body>
<header><div class="brand"><img src="/media?path=assets/branding/arp-app-icon-192.png" alt=""><div><div class="brand-title">ARP</div><div class="brand-subtitle">AI Remaster Pipeline</div><div id="root" class="root"></div></div></div><div class="row" style="max-width:360px"><button onclick="refresh()">Refresh</button><button class="warn" onclick="stopRun()">Stop</button></div></header>
<nav id="tabs" class="tabs"></nav>
<main id="app"></main>
<script>
let state=null, active='global', selected={};
const media=p=>'/media?path='+encodeURIComponent(p);
async function api(path, opts={}){const r=await fetch(path,{headers:{'Content-Type':'application/json'},...opts});return await r.json();}
async function refresh(){const follow=logsNearBottom();state=await api('/api/state');document.getElementById('root').textContent=state.root+(state.running?'  |  Running: '+state.running_stage:'');drawTabs();draw(follow);}
function drawTabs(){const tabs=['global',...state.stages.map(s=>s.key),'manifest','comfy'];const names={global:'Global',manifest:'Manifests',comfy:'ComfyUI'};document.getElementById('tabs').innerHTML=tabs.map(t=>`<button class="tab ${active===t?'active':''}" onclick="active='${t}';draw()">${names[t]||stage(t).title}</button>`).join('');}
function stage(k){return state.stages.find(s=>s.key===k)}
function settings(k){return state.settings[k]||{}}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function draw(followLogs=false){if(active==='global')return drawGlobal(followLogs); if(active==='manifest')return drawManifest(); if(active==='comfy')return drawComfy(); return drawStage(stage(active),followLogs);}
function drawGlobal(followLogs=false){const src=(state.settings.global&&state.settings.global.source)||'';const thumbs=(state.source_previews||[]).map(p=>`<img src="${media(p)}" alt="">`).join('');const info=sourceInfoHtml(state.source_info||{});const gp=(state.phase_progress&&state.phase_progress.global)||{percent:0,label:'Waiting'};document.getElementById('app').innerHTML=`<section class="card"><img class="hero-logo" src="/media?path=assets/branding/arp-logo-wide.png" alt="ARP - AI Remaster Pipeline"><p class="hero">AI Remaster Pipeline</p><p>Choose the source material, then run or inspect each stage.</p><label>Source material</label><div class="field-row"><input id="globalSource" value="${esc(src)}"><button type="button" onclick="browseGlobalSource()">Browse</button></div>${thumbs?`<div class="filmstrip">${thumbs}</div>`:''}${info}${progressHtml(gp.percent,gp.label)}<div class="actions"><button class="primary" onclick="runAll()">Run Whole Remaster</button><button class="warn" onclick="stopRun()" ${state.running?'':'disabled'}>Stop</button></div><table><tr><th>Stage</th><th>Status</th><th>Progress</th><th>Latest output</th></tr>${state.progress.map(p=>{const sp=stageProgressByTitle(p.stage);return `<tr><td>${p.stage}</td><td class="status-${p.status.toLowerCase()}">${p.status}</td><td>${progressHtml(sp.percent,sp.label)}</td><td>${esc(p.latest)}</td></tr>`}).join('')}</table>${runLogHtml()}</section>`;document.getElementById('globalSource').addEventListener('change',saveGlobal);if(followLogs)scrollLogsToBottom()}
function sourceInfoHtml(info){const labels={resolution:'Resolution',aspect:'Aspect',duration:'Duration',frame_rate:'Frame rate',frames:'Frames',video_codec:'Video codec',pixel_format:'Pixel format',colour:'Colour',audio:'Audio',container:'Container',overall_bitrate:'Overall bitrate',video_bitrate:'Video bitrate',size:'File size',codec_note:'Note'};const keys=['resolution','aspect','duration','frame_rate','frames','video_codec','pixel_format','colour','audio','container','overall_bitrate','video_bitrate','size','codec_note'];const items=keys.filter(k=>info[k]).map(k=>`<div><span>${labels[k]||k}</span><strong>${esc(info[k])}</strong></div>`).join('');return items?`<div class="source-info">${items}</div>`:''}
function fieldHtml(st,[key,label,kind,def]){const v=settings(st.key)[key]??def??'';if(kind.startsWith('select:')){return `<label>${label}</label><select data-field="${key}">${kind.slice(7).split('|').map(o=>`<option ${v===o?'selected':''}>${o}</option>`).join('')}</select>`}const input=`<input data-field="${key}" data-kind="${kind}" type="${kind==='number'?'number':'text'}" step="any" value="${esc(v)}">`;if(['file','folder','save'].includes(kind)){return `<label>${label}</label><div class="field-row">${input}<button type="button" onclick="browseField('${st.key}','${key}','${kind}')">Browse</button></div>`}return `<label>${label}</label>${input}`}
function aspectPreviewHtml(st){if(st.key!=='outpaint')return '';const img=state.aspect_preview;const outputs=(state.expected_outputs&&state.expected_outputs.outpaint)||[];return `<h3>Target Preview</h3>${img?`<img src="${media(img)}" alt="Target aspect preview">`:'<p>Choose source material on the Global tab to preview the target frame.</p>'}${outputs.length?`<h3>Output Path</h3><ul class="output-list">${outputs.map(p=>`<li>${esc(p)}</li>`).join('')}</ul>`:''}`}
function fileRow(st,f){const thumb=f.preview?`<img class="file-thumb" src="${media(f.preview)}" alt="">`:'';return `<div class="file ${thumb?'':'no-thumb'}" onclick="selected['${st.key}']='${esc(f.path)}';draw()">${thumb}<div class="file-path">${esc(f.path)}</div></div>`}
function drawStage(st,followLogs=false){const s=settings(st.key);const file=selected[st.key];const expected=(state.expected_outputs&&state.expected_outputs[st.key])||[];const sp=stageProgress(st.key);document.getElementById('app').innerHTML=`<div class="grid"><section class="card"><h2>${st.title}</h2><p>${st.description}</p>${progressHtml(sp.percent,sp.label)}${st.fields.map(f=>fieldHtml(st,f)).join('')}${expected.length&&st.key!=='outpaint'?`<h3>Output Path</h3><ul class="output-list">${expected.map(p=>`<li>${esc(p)}</li>`).join('')}</ul>`:''}<div class="checks"><label><input data-field="force" type="checkbox" ${s.force==='true'?'checked':''}>Regenerate</label><label><input data-field="dry_run" type="checkbox" ${s.dry_run==='true'?'checked':''}>Dry run</label></div><div class="actions"><button class="primary" onclick="runStage('${st.key}')" ${state.running?'disabled':''}>Run ${st.title}</button><button class="warn" onclick="stopRun()" ${state.running?'':'disabled'}>Stop</button></div><div class="command" id="cmd"></div></section><section class="card files"><h3>Intermediate Files</h3>${st.files.map(f=>fileRow(st,f)).join('')||'<p>No files yet.</p>'}</section><section class="card preview">${aspectPreviewHtml(st)}<h3>${file?esc(file):'Preview'}</h3>${preview(file)}</section></div><section class="card" style="margin-top:16px">${runLogHtml()}</section>`;document.querySelectorAll('[data-field]').forEach(el=>el.addEventListener('change',()=>saveStage(st.key,true)));showCommand(st.key);if(followLogs)scrollLogsToBottom()}
function stageProgress(key){return ((state.phase_progress&&state.phase_progress.stages)||[]).find(p=>p.key===key)||{percent:0,label:'Waiting'}}
function stageProgressByTitle(title){return ((state.phase_progress&&state.phase_progress.stages)||[]).find(p=>p.stage===title)||{percent:0,label:'Waiting'}}
function progressHtml(percent,label){const p=Math.max(0,Math.min(100,Number(percent)||0));return `<div class="phase-progress"><div><span>${esc(label||'Waiting')}</span><span>${p}%</span></div><progress value="${p}" max="100"></progress></div>`}
function logsNearBottom(){const logs=[...document.querySelectorAll('pre.log')];return !logs.length||logs.some(el=>el.scrollHeight-el.scrollTop-el.clientHeight<32)}
function scrollLogsToBottom(){document.querySelectorAll('pre.log').forEach(el=>{el.scrollTop=el.scrollHeight})}
function runLogHtml(){return `<div class="log-heading"><h3>Run Log</h3><button type="button" onclick="copyRunLog()">Copy Log</button></div><pre class="log">${logHtml(state.log)}</pre>`}
function logHtml(text){return String(text||'').split('\n').map(line=>`<span class="${logClass(line)}">${esc(line)}</span>`).join('\n')}
function logClass(line){const lower=String(line).toLowerCase();if(/traceback|runtimeerror|exception|error|failed|refused|exit code [1-9]|filenotfound/.test(lower))return 'log-error';if(/warning|skipping|timed out/.test(lower))return 'log-warn';if(/ready|reuse|wrote|finished with exit code 0|started/.test(lower))return 'log-ok';return ''}
async function copyRunLog(){const text=state.log||'';try{await navigator.clipboard.writeText(text)}catch{const area=document.createElement('textarea');area.value=text;document.body.appendChild(area);area.select();document.execCommand('copy');area.remove()}}
function preview(p){if(!p)return '<p>Select an image, video, manifest, workflow, or log file.</p>';const ext=p.split('.').pop().toLowerCase();if(['png','jpg','jpeg','webp','tif','tiff'].includes(ext))return `<img src="${media(p)}">`;if(['mp4','mov','mkv','avi','webm','m4v'].includes(ext))return `<video src="${media(p)}" controls></video>`;return `<pre id="textPreview">Text preview opens via the browser media endpoint.</pre><p><a href="${media(p)}" target="_blank">Open file</a></p>`}
async function saveStage(k,redraw=false){const values={};document.querySelectorAll('[data-field]').forEach(el=>{values[el.dataset.field]=el.type==='checkbox'?String(el.checked):el.value});await api('/api/settings',{method:'POST',body:JSON.stringify({stage:k,values})});state=await api('/api/state');if(redraw)draw();showCommand(k)}
async function saveGlobal(){await api('/api/settings',{method:'POST',body:JSON.stringify({stage:'global',values:{source:document.getElementById('globalSource').value}})});state=await api('/api/state')}
async function browseGlobalSource(){const el=document.getElementById('globalSource');const r=await api('/api/browse-global-source',{method:'POST',body:JSON.stringify({current:el.value})});if(!r.ok){alert(r.error||'Browse failed');return}if(r.path){state=r.state;draw()}else{await refresh()}}
async function browseField(stageKey,fieldKey,kind){const el=document.querySelector(`[data-field="${fieldKey}"]`);const r=await api('/api/browse',{method:'POST',body:JSON.stringify({kind,current:el.value})});if(!r.ok){alert(r.error||'Browse failed');return}if(r.path){el.value=r.path;await saveStage(stageKey)}}
async function showCommand(k){const r=await api('/api/command?stage='+encodeURIComponent(k));const el=document.getElementById('cmd');if(el)el.textContent=r.command.join(' ')}
async function confirmOverwrite(k){const force=(settings(k).force==='true');if(!force&&k!=='shots')return true;const r=await api('/api/existing-outputs?stage='+encodeURIComponent(k));if(!r.paths||!r.paths.length)return true;const reason=force?'Regenerate is enabled':'Shot Detection rewrites its manifest';return confirm(reason+' and these output paths already exist:\n\n'+r.paths.join('\n')+'\n\nOverwrite them?')}
async function runStage(k){await saveStage(k);if(!(await confirmOverwrite(k)))return;const r=await api('/api/run',{method:'POST',body:JSON.stringify({stage:k})});if(!r.ok)alert(r.message);setTimeout(refresh,500)}
async function runAll(){for(const st of state.stages){if(!(await confirmOverwrite(st.key)))return}const r=await api('/api/run',{method:'POST',body:JSON.stringify({all:true})});if(!r.ok)alert(r.message);setTimeout(refresh,500)}
async function stopRun(){await api('/api/stop',{method:'POST',body:'{}'});refresh()}
function drawManifest(){document.getElementById('app').innerHTML=`<section class="card"><h2>Manifest Editor</h2><div class="row"><input id="manifestPath" placeholder="manifests/references/colorize_manifest_clip_shots_auto.csv"><button onclick="loadManifest()">Load</button><button onclick="saveManifest()">Save</button></div><div id="manifestRows"></div></section>`}
async function loadManifest(){const path=document.getElementById('manifestPath').value;const r=await api('/api/manifest?path='+encodeURIComponent(path));document.getElementById('manifestRows').innerHTML=`<table class="manifest"><tr><th>enabled</th><th>end</th><th>source_reference</th><th>color_reference</th></tr>${r.rows.map(row=>`<tr>${['enabled','end','source_reference','color_reference'].map(k=>`<td><input value="${esc(row[k]||'')}" data-col="${k}"></td>`).join('')}</tr>`).join('')}</table>`}
async function saveManifest(){const path=document.getElementById('manifestPath').value;const rows=[...document.querySelectorAll('.manifest tr')].slice(1).map(tr=>{const row={};tr.querySelectorAll('input').forEach(i=>row[i.dataset.col]=i.value);return row});await api('/api/manifest',{method:'POST',body:JSON.stringify({path,rows})});alert('Manifest saved.')}
function drawComfy(){document.getElementById('app').innerHTML=`<section class="card"><h2>ComfyUI</h2><div class="row"><input id="comfyUrl" value="http://127.0.0.1:8188"><button onclick="loadComfy()">Refresh Queue</button></div><pre class="log" id="queue"></pre><h3>Log file</h3><div class="row"><input id="comfyLog" placeholder="path/to/comfy.log"><button onclick="loadLogFile()">Load</button></div><pre class="log" id="comfyLogText"></pre></section>`}
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
    url = CONFIG.get("comfy_url", "http://127.0.0.1:8188")
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
    url = CONFIG.get("comfy_url", "http://127.0.0.1:8188")
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
    comfy_dir = Path(CONFIG.get("comfy_dir", ROOT / "tools" / "comfyui"))
    main_py = comfy_dir / "main.py"
    if not main_py.exists():
        if CONFIG_FILE.exists():
            startup_log(f"ComfyUI is configured but main.py was not found: {main_py}")
            startup_log("Run install_windows.bat again and choose your ComfyUI directory.")
        else:
            startup_log("ComfyUI is not configured yet.")
            startup_log("Run install_windows.bat again and choose whether to clone ComfyUI or use an existing ComfyUI directory.")
        return
    host = CONFIG.get("comfy_host", "127.0.0.1")
    port = str(CONFIG.get("comfy_port", "8188"))
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

