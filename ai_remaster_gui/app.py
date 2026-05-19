from __future__ import annotations

import csv
import html
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SETTINGS_FILE = ROOT / ".ai_remaster_gui.json"
CONFIG_FILE = ROOT / ".ai_remaster_config.json"
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
        "Prepare 4:3 footage for LTX outpainting, then restore the ComfyUI render.",
        ("input", "intermediate/outpaint_prepared", "intermediate/outpainted"),
        (
            ("source", "Source video", "file", ""),
            ("prepared_output", "Prepared output", "save", ""),
            ("comfy_outpaint_render", "Comfy outpaint render", "file", ""),
            ("restored_output", "Restored outpaint output", "save", ""),
            ("black_lift", "Black lift", "number", "0.018"),
            ("gamma", "Gamma", "number", "1.06"),
            ("target_height", "Target height", "number", ""),
            ("encoder", "Encoder", "select:h264|prores", "h264"),
        ),
        ("source",),
    ),
    Stage(
        "shots",
        "Shot Detection",
        "Detect cuts and extract one useful reference frame per shot.",
        ("intermediate/outpainted", "intermediate/outpainted_references", "manifests/references"),
        (
            ("outpainted_video", "Outpainted video", "file", ""),
            ("manifest", "Output manifest", "save", ""),
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
            ("output", "Final output", "save", ""),
            ("feather_pixels", "Feather pixels", "number", "80"),
            ("saturation", "Saturation", "number", "0.82"),
            ("temperature", "Temperature", "number", "-0.015"),
            ("color_opacity", "Color opacity", "number", "1.0"),
            ("encoder", "Encoder", "select:h264|prores", "h264"),
        ),
        ("outpainted_video", "source", "output"),
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
    for values in defaults.values():
        if source and not values.get("source"):
            values["source"] = rel(source)
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
                    out.append({"path": rel(path), "size": stat.st_size, "mtime": int(stat.st_mtime)})
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

    def state(self) -> dict:
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            return {
                "root": str(ROOT),
                "stages": [stage.__dict__ | {"files": self.files_for(stage)} for stage in STAGES],
                "settings": self.settings,
                "progress": self.progress(),
                "running": running,
                "running_stage": self.running_stage,
                "log": "\n".join(self.log[-800:]),
            }

    def update_settings(self, stage: str, values: dict[str, str]) -> None:
        self.settings.setdefault(stage, {}).update({key: str(value) for key, value in values.items()})
        self.save()

    def command_for(self, stage_key: str) -> list[str]:
        values = self.settings[stage_key]
        py = sys.executable
        cmd = [py]
        add = cmd.extend
        if stage_key == "outpaint":
            finalize = bool(values.get("comfy_outpaint_render"))
            script = "finalize_outpaint_output.py" if finalize else "prepare_outpaint_input.py"
            cmd.append(str(SCRIPTS / script))
            if finalize:
                add(["--source", values.get("comfy_outpaint_render", "")])
                if values.get("restored_output"):
                    add(["--output", values["restored_output"]])
            else:
                add(["--source", values.get("source", "")])
                if values.get("prepared_output"):
                    add(["--output", values["prepared_output"]])
                if values.get("target_height"):
                    add(["--target-height", values["target_height"]])
            add(["--black-lift", values.get("black_lift", "0.018"), "--gamma", values.get("gamma", "1.06"), "--encoder", values.get("encoder", "h264")])
        elif stage_key == "shots":
            cmd.append(str(SCRIPTS / "generate_references.py"))
            add(["--source-video", values.get("outpainted_video", "")])
            if values.get("manifest"):
                add(["--output-manifest", values["manifest"]])
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
            add(["--outpainted", values.get("outpainted_video", ""), "--source", values.get("source", ""), "--output", values.get("output", "")])
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
        if stage_key == "outpaint" and values.get("comfy_outpaint_render"):
            missing = []
        if missing:
            return False, "Missing settings: " + ", ".join(missing)
        with self.lock:
            if self.process and self.process.poll() is None:
                return False, "A command is already running."
            self.running_stage = stage.title
            cmd = self.command_for(stage_key)
            self.log.append("> " + " ".join(cmd))
            self.process = subprocess.Popen(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            threading.Thread(target=self._collect_output, daemon=True).start()
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

    def _collect_output(self) -> None:
        assert self.process and self.process.stdout
        for line in self.process.stdout:
            with self.lock:
                self.log.append(line.rstrip())
        code = self.process.wait()
        with self.lock:
            self.log.append(f"Process finished with exit code {code}.")
            self.running_stage = ""

    def stop(self) -> None:
        with self.lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                self.log.append("Stop requested.")


APP = PipelineApp()


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
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}.hero{font-size:32px;font-weight:800;margin:0 0 5px}.hero-logo{width:min(520px,100%);display:block;margin:2px auto 16px}
label{display:block;color:var(--muted);font-size:12px;margin:10px 0 5px}input,select,textarea{width:100%;background:#202930;color:var(--text);border:1px solid #3b4b55;border-radius:6px;padding:8px}
button{background:#26333b;color:var(--text);border:1px solid #43545f;border-radius:6px;padding:8px 12px;cursor:pointer}button:hover{background:#30404a}.primary{background:var(--accent);border-color:#44a995;font-weight:700}.warn{background:#5b4322;border-color:#8a6830}
.row{display:flex;gap:8px;align-items:center}.row>*{flex:1}.checks label{display:inline-flex;gap:6px;align-items:center;margin-right:12px}.checks input{width:auto}
.files{max-height:62vh;overflow:auto}.file{padding:8px;border-bottom:1px solid #27333a;cursor:pointer;color:#cfe0e5}.file:hover{background:#202a31}
.preview img,.preview video{width:100%;max-height:62vh;object-fit:contain;background:#050607;border-radius:8px}.preview pre,pre.log{white-space:pre-wrap;background:#0b0e10;border:1px solid var(--line);border-radius:8px;padding:10px;max-height:230px;overflow:auto}
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
async function refresh(){state=await api('/api/state');document.getElementById('root').textContent=state.root+(state.running?'  |  Running: '+state.running_stage:'');drawTabs();draw();}
function drawTabs(){const tabs=['global',...state.stages.map(s=>s.key),'manifest','comfy'];const names={global:'Global',manifest:'Manifests',comfy:'ComfyUI'};document.getElementById('tabs').innerHTML=tabs.map(t=>`<button class="tab ${active===t?'active':''}" onclick="active='${t}';draw()">${names[t]||stage(t).title}</button>`).join('');}
function stage(k){return state.stages.find(s=>s.key===k)}
function settings(k){return state.settings[k]||{}}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function draw(){if(active==='global')return drawGlobal(); if(active==='manifest')return drawManifest(); if(active==='comfy')return drawComfy(); return drawStage(stage(active));}
function drawGlobal(){const done=state.progress.filter(p=>p.status==='Ready').length;document.getElementById('app').innerHTML=`<section class="card"><img class="hero-logo" src="/media?path=assets/branding/arp-logo.png" alt="ARP - AI Remaster Pipeline"><p class="hero">AI Remaster Pipeline</p><p>Track the whole restoration and run the configured stages in sequence.</p><progress value="${done}" max="${state.progress.length}" style="width:100%;height:24px"></progress><p><button class="primary" onclick="runAll()">Run Whole Remaster</button></p><table><tr><th>Stage</th><th>Status</th><th>Latest output</th></tr>${state.progress.map(p=>`<tr><td>${p.stage}</td><td class="status-${p.status.toLowerCase()}">${p.status}</td><td>${esc(p.latest)}</td></tr>`).join('')}</table><h3>Run Log</h3><pre class="log">${esc(state.log)}</pre></section>`}
function fieldHtml(st,[key,label,kind,def]){const v=settings(st.key)[key]??def??'';if(kind.startsWith('select:')){return `<label>${label}</label><select data-field="${key}">${kind.slice(7).split('|').map(o=>`<option ${v===o?'selected':''}>${o}</option>`).join('')}</select>`}return `<label>${label}</label><input data-field="${key}" type="${kind==='number'?'number':'text'}" step="any" value="${esc(v)}">`}
function drawStage(st){const s=settings(st.key);const file=selected[st.key];document.getElementById('app').innerHTML=`<div class="grid"><section class="card"><h2>${st.title}</h2><p>${st.description}</p>${st.fields.map(f=>fieldHtml(st,f)).join('')}<div class="checks"><label><input data-field="force" type="checkbox" ${s.force==='true'?'checked':''}>Regenerate</label><label><input data-field="dry_run" type="checkbox" ${s.dry_run==='true'?'checked':''}>Dry run</label></div><p><button class="primary" onclick="runStage('${st.key}')">Run ${st.title}</button></p><div class="command" id="cmd"></div></section><section class="card files"><h3>Intermediate Files</h3>${st.files.map(f=>`<div class="file" onclick="selected['${st.key}']='${esc(f.path)}';draw()">${esc(f.path)}</div>`).join('')||'<p>No files yet.</p>'}</section><section class="card preview"><h3>${file?esc(file):'Preview'}</h3>${preview(file)}</section></div><section class="card" style="margin-top:16px"><h3>Run Log</h3><pre class="log">${esc(state.log)}</pre></section>`;document.querySelectorAll('[data-field]').forEach(el=>el.addEventListener('change',()=>saveStage(st.key)));showCommand(st.key)}
function preview(p){if(!p)return '<p>Select an image, video, manifest, workflow, or log file.</p>';const ext=p.split('.').pop().toLowerCase();if(['png','jpg','jpeg','webp','tif','tiff'].includes(ext))return `<img src="${media(p)}">`;if(['mp4','mov','mkv','avi','webm','m4v'].includes(ext))return `<video src="${media(p)}" controls></video>`;return `<pre id="textPreview">Text preview opens via the browser media endpoint.</pre><p><a href="${media(p)}" target="_blank">Open file</a></p>`}
async function saveStage(k){const values={};document.querySelectorAll('[data-field]').forEach(el=>{values[el.dataset.field]=el.type==='checkbox'?String(el.checked):el.value});await api('/api/settings',{method:'POST',body:JSON.stringify({stage:k,values})});state=await api('/api/state');showCommand(k)}
async function showCommand(k){const r=await api('/api/command?stage='+encodeURIComponent(k));const el=document.getElementById('cmd');if(el)el.textContent=r.command.join(' ')}
async function runStage(k){await saveStage(k);const r=await api('/api/run',{method:'POST',body:JSON.stringify({stage:k})});if(!r.ok)alert(r.message);setTimeout(refresh,500)}
async function runAll(){const r=await api('/api/run',{method:'POST',body:JSON.stringify({all:true})});if(!r.ok)alert(r.message);setTimeout(refresh,500)}
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


def start_comfy_if_needed() -> None:
    url = CONFIG.get("comfy_url", "http://127.0.0.1:8188")
    if comfy_is_running(url):
        print(f"ComfyUI already running at {url}")
        return
    comfy_dir = Path(CONFIG.get("comfy_dir", ROOT / "tools" / "comfyui"))
    main_py = comfy_dir / "main.py"
    if not main_py.exists():
        if CONFIG_FILE.exists():
            print(f"ComfyUI is configured but main.py was not found: {main_py}")
            print("Run install_windows.bat again and choose your ComfyUI directory.")
        else:
            print("ComfyUI is not configured yet.")
            print("Run install_windows.bat again and choose whether to clone ComfyUI or use an existing ComfyUI directory.")
        return
    host = CONFIG.get("comfy_host", "127.0.0.1")
    port = str(CONFIG.get("comfy_port", "8188"))
    command = [sys.executable, "main.py", "--listen", host, "--port", port]
    kwargs: dict = {"cwd": str(comfy_dir)}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    subprocess.Popen(command, **kwargs)
    print(f"Started ComfyUI in a new process at {url}")


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
    return 0
