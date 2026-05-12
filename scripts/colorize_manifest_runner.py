#!/usr/bin/env python3
"""Manifest-driven ComfyUI runner for reference-image video colorization.

CSV format:
    # source_video=outpainted/example_00.00.00to00.10.00_outpaint.mp4
    enabled,end,reference
    true,00:00:13,references/example/cut_0000.png

Rows use end times only. Each row begins where the previous row ended. Disabled
rows advance the clock but do not queue jobs.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOW = ROOT / "workflows" / "colorize" / "Colorize Video - Image Reference DeepEx Pipeline.json"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "colorized"
DEFAULT_PREFIX = "remaster_colorized"
DEFAULT_FFMPEG = Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe")
DEFAULT_FFPROBE = Path(r"C:\Program Files\ffmpeg\bin\ffprobe.exe")
METHODS = {
    "deepexemplar": ("DeepExemplar", "deepex"),
    "deepex": ("DeepExemplar", "deepex"),
    "deep-exemplar": ("DeepExemplar", "deepex"),
    "deep_exemplar": ("DeepExemplar", "deepex"),
    "colormnet": ("ColorMNet", "colormnet"),
}
DEFAULT_METHOD_ORDER = [("DeepExemplar", "deepex"), ("ColorMNet", "colormnet")]

NODE_IDS = {
    "load_video": "4",
    "video_info": "3",
    "load_reference": "2",
    "deepex": "344",
    "deepex_save": "343",
    "colormnet": "133",
    "colormnet_save": "131",
}

stop_after_current = False
interrupt_count = 0


@dataclass
class ManifestRow:
    index: int
    enabled: bool
    start: float
    end: float
    reference: str


@dataclass
class ChunkJob:
    chunk: int
    row: int
    start_frame: int
    frames: int
    start_time: float
    end_time: float
    reference: str
    output: str
    save_prefix: str
    trim_start_time: float | None = None
    trim_end_time: float | None = None
    trim_start_frame: int | None = None
    trim_frames: int | None = None
    status: str = "pending"
    prompt_id: str | None = None
    error: str | None = None


class WindowsSleepInhibitor:
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled and os.name == "nt"

    def __enter__(self) -> "WindowsSleepInhibitor":
        if self.enabled:
            ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED)
            print("Keeping Windows awake while the manifest runner is active.")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.enabled:
            ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
            print("Released Windows sleep hold.")


def parse_time(value: str) -> float:
    value = value.strip()
    parts = value.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Invalid time value: {value}")


def normalize_methods(value: str | None) -> list[tuple[str, str]]:
    raw = (value or "Both").strip().lower().replace(" ", "")
    if raw in {"both", "all"}:
        return list(DEFAULT_METHOD_ORDER)
    if raw not in METHODS:
        valid = "Both, DeepExemplar, ColorMNet"
        raise argparse.ArgumentTypeError(f"Unknown colorization method '{value}'. Use one of: {valid}")
    return [METHODS[raw]]


def format_time(seconds: float) -> str:
    total_millis = int(round(seconds * 1000))
    total = total_millis // 1000
    millis = total_millis % 1000
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if millis:
        return f"{hours}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours}:{minutes:02d}:{secs:02d}"


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}


def read_manifest(path: Path) -> tuple[str, list[ManifestRow]]:
    source_video: str | None = None
    data_lines: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            key, sep, value = stripped[1:].partition("=")
            if sep and key.strip() == "source_video":
                source_video = value.strip()
            continue
        if stripped:
            data_lines.append(line)

    if not source_video:
        raise ValueError(f"{path} is missing top-line metadata: # source_video=...")

    reader = csv.DictReader(data_lines)
    required = {"enabled", "end", "reference"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")

    rows: list[ManifestRow] = []
    start = 0.0
    for index, row in enumerate(reader):
        end = parse_time(row["end"])
        if end <= start:
            raise ValueError(f"Row {index} end time {row['end']} must be after {format_time(start)}")
        rows.append(
            ManifestRow(
                index=index,
                enabled=truthy(row["enabled"]),
                start=start,
                end=end,
                reference=row["reference"].strip().replace("\\", "/"),
            )
        )
        start = end
    return source_video, rows


def resolve_source_video(source_video: str) -> Path:
    path = Path(source_video)
    if path.is_absolute():
        return path
    return ROOT / "input" / source_video


def resolve_reference(reference: str) -> Path:
    path = Path(reference)
    if path.is_absolute():
        return path
    return ROOT / "input" / reference


def ffprobe_path(args: argparse.Namespace) -> str:
    if args.ffprobe:
        return args.ffprobe
    if args.ffmpeg:
        candidate = Path(args.ffmpeg).with_name("ffprobe.exe")
        if candidate.exists():
            return str(candidate)
    if DEFAULT_FFPROBE.exists():
        return str(DEFAULT_FFPROBE)
    return "ffprobe"


def ffmpeg_path(args: argparse.Namespace) -> str:
    if args.ffmpeg:
        return args.ffmpeg
    if DEFAULT_FFMPEG.exists():
        return str(DEFAULT_FFMPEG)
    return "ffmpeg"


def probe_video(path: Path, args: argparse.Namespace, fallback_duration: float) -> tuple[float, float | None]:
    command = [
        ffprobe_path(args),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        stream = json.loads(result.stdout)["streams"][0]
        num, den = stream.get("r_frame_rate", "24/1").split("/")
        fps = float(num) / float(den)
        duration = float(stream["duration"]) if stream.get("duration") else None
        return fps, duration
    except Exception as exc:
        print(f"Warning: ffprobe failed ({exc}); using --fps {args.fps}", file=sys.stderr)
        return args.fps, fallback_duration


def count_video_frames(path: Path, args: argparse.Namespace) -> int:
    command = [
        ffprobe_path(args),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found: {path}")
    stream = streams[0]
    frame_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    if frame_count in {None, "N/A"}:
        raise RuntimeError(f"Could not count video frames: {path}")
    return int(frame_count)


def build_jobs(
    manifest_stem: str,
    rows: list[ManifestRow],
    fps: float,
    max_frames: int,
    range_overlap_seconds: float,
    chunk_overlap_frames: int,
    source_duration: float | None,
    output_dir: Path,
    prefix: str,
) -> list[ChunkJob]:
    jobs: list[ChunkJob] = []
    for row in rows:
        if not row.enabled:
            continue
        padded_start = max(0.0, row.start - range_overlap_seconds)
        padded_end = row.end + range_overlap_seconds
        if source_duration is not None:
            padded_end = min(source_duration, padded_end)
        start_frame = int(round(padded_start * fps))
        end_frame = int(round(padded_end * fps))
        row_start_frame = int(round(row.start * fps))
        row_end_frame = int(round(row.end * fps))
        if source_duration is not None:
            row_end_frame = min(row_end_frame, int(round(source_duration * fps)))
        cursor = start_frame
        while cursor < end_frame:
            base_end = min(cursor + max_frames, end_frame)
            trim_start_frame = max(cursor, row_start_frame)
            trim_end_frame = min(base_end, row_end_frame)
            job_start = max(start_frame, cursor - chunk_overlap_frames)
            job_end = min(end_frame, base_end + chunk_overlap_frames)
            frames = job_end - job_start
            if trim_end_frame > trim_start_frame:
                chunk = len(jobs)
                stem = f"{prefix}_{manifest_stem}_chunk_{chunk:04d}"
                jobs.append(
                    ChunkJob(
                        chunk=chunk,
                        row=row.index,
                        start_frame=job_start,
                        frames=frames,
                        start_time=job_start / fps,
                        end_time=job_end / fps,
                        reference=row.reference,
                        output=str(output_dir / f"{stem}.mp4"),
                        save_prefix=stem + "_tmp",
                        trim_start_time=trim_start_frame / fps,
                        trim_end_time=trim_end_frame / fps,
                        trim_start_frame=trim_start_frame - job_start,
                        trim_frames=trim_end_frame - trim_start_frame,
                    )
                )
            cursor = base_end
    return jobs


def required_output_frames(job: ChunkJob) -> int:
    trim_start = job.trim_start_frame if job.trim_start_frame is not None else 0
    trim_frames = job.trim_frames if job.trim_frames is not None else job.frames
    return trim_start + trim_frames


def load_ledger(path: Path) -> dict[int, ChunkJob]:
    jobs: dict[int, ChunkJob] = {}
    if not path.exists():
        return jobs
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        jobs[int(data["chunk"])] = ChunkJob(**data)
    return jobs


def write_ledger(path: Path, jobs: list[ChunkJob]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(asdict(job), ensure_ascii=False) for job in jobs) + "\n", encoding="utf-8")
    tmp.replace(path)


def reconcile_jobs(jobs: list[ChunkJob], ledger_path: Path, args: argparse.Namespace) -> None:
    previous = load_ledger(ledger_path)
    for job in jobs:
        old = previous.get(job.chunk)
        output_path = Path(job.output)
        output_exists = output_path.exists() and output_path.stat().st_size > 0
        if output_exists:
            try:
                actual_frames = count_video_frames(output_path, args)
                needed_frames = required_output_frames(job)
                if actual_frames >= needed_frames:
                    job.status = "done"
                    if old:
                        job.prompt_id = old.prompt_id
                        job.error = old.error
                    continue
                print(
                    f"Redo chunk {job.chunk:04d}: existing output has {actual_frames} frames, "
                    f"but this manifest needs at least {needed_frames}."
                )
                if not getattr(args, "dry_run", False):
                    try:
                        output_path.unlink()
                    except OSError as exc:
                        print(f"Warning: could not remove stale output {output_path} ({exc}).")
            except Exception as exc:
                print(f"Redo chunk {job.chunk:04d}: could not validate existing output ({exc}).")
                if not getattr(args, "dry_run", False):
                    try:
                        output_path.unlink()
                    except OSError as unlink_exc:
                        print(f"Warning: could not remove invalid output {output_path} ({unlink_exc}).")
        if old and old.status in {"queued", "running", "failed"}:
            job.status = "pending"
        elif old and old.status == "done" and not output_exists:
            job.status = "pending"


def validate_inputs(source_video: str, rows: list[ManifestRow]) -> None:
    source_path = resolve_source_video(source_video)
    if not source_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_video}")
    for row in rows:
        if not row.enabled:
            continue
        ref_path = resolve_reference(row.reference)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference image not found: {row.reference}")


def node_by_id(workflow: dict[str, Any], node_id: str) -> dict[str, Any]:
    for node in workflow["nodes"]:
        if str(node["id"]) == str(node_id):
            return node
    raise KeyError(f"Workflow node not found: {node_id}")


def set_widget(node: dict[str, Any], key: str | int, value: Any) -> None:
    widgets = node.setdefault("widgets_values", {})
    if isinstance(widgets, dict):
        widgets[str(key)] = value
        return
    if not isinstance(widgets, list):
        widgets = [widgets]
        node["widgets_values"] = widgets
    index = int(key)
    while len(widgets) <= index:
        widgets.append(None)
    widgets[index] = value


def patch_workflow_for_job(
    workflow: dict[str, Any],
    source_video_path: Path,
    job: ChunkJob,
    method_engine: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    wf = json.loads(json.dumps(workflow))
    load_video = node_by_id(wf, NODE_IDS["load_video"])
    set_widget(load_video, "video", str(source_video_path))
    set_widget(load_video, "force_rate", 0)
    set_widget(load_video, "custom_width", 0)
    set_widget(load_video, "custom_height", 0)
    set_widget(load_video, "frame_load_cap", job.frames)
    set_widget(load_video, "skip_first_frames", job.start_frame)
    set_widget(load_video, "select_every_nth", 1)
    set_widget(load_video, "format", "AnimateDiff")

    set_widget(node_by_id(wf, NODE_IDS["load_reference"]), "image", str(resolve_reference(job.reference)))
    set_widget(node_by_id(wf, NODE_IDS["load_reference"]), "custom_width", 0)
    set_widget(node_by_id(wf, NODE_IDS["load_reference"]), "custom_height", 0)

    if method_engine == "deepex":
        deepex = node_by_id(wf, NODE_IDS["deepex"])
        deepex["mode"] = 0
        node_by_id(wf, NODE_IDS["deepex_save"])["mode"] = 0
        node_by_id(wf, NODE_IDS["colormnet"])["mode"] = 4
        node_by_id(wf, NODE_IDS["colormnet_save"])["mode"] = 4
        set_widget(deepex, 1, args.deepex_half_resolution)
        if args.target_width and args.target_height:
            for item in deepex.get("inputs", []):
                if item.get("name") in {"target_width", "target_height"}:
                    item["link"] = None
            set_widget(deepex, 2, args.target_width)
            set_widget(deepex, 3, args.target_height)
        save_id = NODE_IDS["deepex_save"]
    else:
        node_by_id(wf, NODE_IDS["deepex"])["mode"] = 4
        node_by_id(wf, NODE_IDS["deepex_save"])["mode"] = 4
        node_by_id(wf, NODE_IDS["colormnet"])["mode"] = 0
        node_by_id(wf, NODE_IDS["colormnet_save"])["mode"] = 0
        save_id = NODE_IDS["colormnet_save"]

    save = node_by_id(wf, save_id)
    set_widget(save, "filename_prefix", job.save_prefix)
    set_widget(save, "save_output", True)
    return wf, save_id


def workflow_to_prompt(workflow: dict[str, Any], output_node_id: str) -> dict[str, Any]:
    nodes = {str(node["id"]): node for node in workflow["nodes"] if int(node.get("mode", 0)) != 4}
    links = {int(link[0]): link for link in workflow.get("links", [])}
    needed: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in needed:
            return
        if node_id not in nodes:
            raise ValueError(f"Output path references disabled/missing node {node_id}")
        needed.add(node_id)
        for item in nodes[node_id].get("inputs", []):
            link_id = item.get("link")
            if link_id is not None:
                visit(str(links[int(link_id)][1]))

    visit(str(output_node_id))

    prompt: dict[str, Any] = {}
    for node_id in sorted(needed, key=lambda value: int(value)):
        node = nodes[node_id]
        inputs: dict[str, Any] = {}
        widget_values = node.get("widgets_values", [])
        widget_index = 0
        for item in node.get("inputs", []):
            name = item["name"]
            link_id = item.get("link")
            has_widget = "widget" in item
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
        prompt[node_id] = {"class_type": node["type"], "inputs": inputs}
        if node.get("title"):
            prompt[node_id]["_meta"] = {"title": node["title"]}
    return prompt


def http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to ComfyUI at {url}: {exc.reason}") from exc


def queue_prompt(comfy_url: str, prompt: dict[str, Any], client_id: str) -> str:
    response = http_json("POST", f"{comfy_url.rstrip('/')}/prompt", {"prompt": prompt, "client_id": client_id})
    prompt_id = response.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return prompt_id: {response}")
    return str(prompt_id)


def wait_for_prompt(comfy_url: str, prompt_id: str, poll_seconds: float) -> dict[str, Any]:
    while True:
        history = http_json("GET", f"{comfy_url.rstrip('/')}/history/{prompt_id}", timeout=30)
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("completed"):
                return entry
            for message in status.get("messages") or []:
                if isinstance(message, list) and message and message[0] == "execution_error":
                    raise RuntimeError(json.dumps(message[1], ensure_ascii=False))
        time.sleep(poll_seconds)


def extract_saved_video(history_entry: dict[str, Any]) -> Path:
    outputs = history_entry.get("outputs", {})
    candidates: list[Path] = []
    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        for key in ("videos", "gifs"):
            for item in output.get(key, []):
                filename = item.get("filename")
                if filename:
                    subfolder = item.get("subfolder") or ""
                    folder = item.get("type") or "output"
                    base = ROOT / folder
                    candidates.append(base / subfolder / filename)
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise RuntimeError(f"Could not find saved video in ComfyUI history: {candidates}")
    return max(existing, key=lambda path: path.stat().st_mtime)


def validate_output(path: Path) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"Output was not created or is empty: {path}")


def copy_output(source: Path, final: Path, attempts: int = 20, delay_seconds: float = 1.0) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(final.stem + ".partial.mp4")
    if tmp.exists():
        tmp.unlink()
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            shutil.copy2(str(source), str(tmp))
            validate_output(tmp)
            tmp.replace(final)
            return
        except PermissionError as exc:
            last_error = exc
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            print(f"Waiting for ComfyUI to release output file ({attempt}/{attempts}): {source}")
            time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Could not copy output file: {source}")


def run_ffmpeg(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def reassembled_output_path(output_dir: Path, manifest_stem: str, args: argparse.Namespace) -> Path:
    method_name = getattr(args, "method_name", "")
    method_part = f"_{method_name}" if method_name else ""
    return output_dir / f"{args.prefix}{method_part}_{manifest_stem}_reassembled.mp4"


def video_encode_args(args: argparse.Namespace) -> list[str]:
    if args.reassemble_encoder == "copy":
        return ["-c:v", "copy"]
    if args.reassemble_encoder == "prores":
        return ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]
    if args.reassemble_encoder == "h264":
        return ["-c:v", "libx264", "-preset", args.reassemble_preset, "-crf", str(args.reassemble_crf), "-pix_fmt", "yuv420p"]
    raise RuntimeError(f"Unsupported reassemble encoder: {args.reassemble_encoder}")


def reassemble_outputs(jobs: list[ChunkJob], output_dir: Path, manifest_stem: str, args: argparse.Namespace) -> None:
    if not jobs or any(job.status != "done" for job in jobs):
        return
    ffmpeg = ffmpeg_path(args)
    final = reassembled_output_path(output_dir, manifest_stem, args)
    partial = final.with_name(final.stem + ".partial.mp4")
    expected_frames = sum(job.trim_frames if job.trim_frames is not None else job.frames for job in jobs)
    print(f"Reassembling {len(jobs)} chunks into {expected_frames} frames at {args.reassemble_fps:.6g} fps.")

    if args.reassemble_encoder == "copy":
        list_path = output_dir / f"{manifest_stem}_reassemble_concat.txt"
        lines = []
        for job in jobs:
            path = Path(job.output).resolve()
            escaped = str(path).replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
            if job.trim_start_time is not None and job.trim_end_time is not None:
                inpoint = max(0.0, job.trim_start_time - job.start_time)
                outpoint = max(inpoint, job.trim_end_time - job.start_time)
                lines.append(f"inpoint {inpoint:.6f}")
                lines.append(f"outpoint {outpoint:.6f}")
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        run_ffmpeg([
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-map",
            "0",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            "-avoid_negative_ts",
            "make_zero",
            str(partial),
        ])
        try:
            actual_frames = count_video_frames(partial, args)
            if abs(actual_frames - expected_frames) > 1:
                print(
                    f"Warning: copy reassembly produced {actual_frames} frames; "
                    f"expected about {expected_frames}. Use h264/prores if Resolve dislikes this file."
                )
        except Exception as exc:
            print(f"Warning: could not validate copied reassembly frame count ({exc}).")
        partial.replace(final)
        print(f"Reassembled: {final}")
        return

    command = [ffmpeg, "-y"]
    filter_parts = []
    for job in jobs:
        command += ["-i", str(Path(job.output).resolve())]
    for index, job in enumerate(jobs):
        trim_start = job.trim_start_frame if job.trim_start_frame is not None else 0
        trim_frames = job.trim_frames if job.trim_frames is not None else job.frames
        trim_end = trim_start + trim_frames
        filter_parts.append(f"[{index}:v]trim=start_frame={trim_start}:end_frame={trim_end},setpts=PTS-STARTPTS[v{index}]")
    concat_inputs = "".join(f"[v{index}]" for index in range(len(jobs)))
    filter_complex = ";".join(
        filter_parts
        + [f"{concat_inputs}concat=n={len(jobs)}:v=1:a=0,setsar=1,setpts=N/({args.reassemble_fps:.12g}*TB)[vout]"]
    )
    run_ffmpeg(
        command
        + [
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-an",
        ]
        + video_encode_args(args)
        + ["-movflags", "+faststart", str(partial)]
    )
    actual_frames = count_video_frames(partial, args)
    if abs(actual_frames - expected_frames) > 1:
        raise RuntimeError(
            f"Reassembled frame count mismatch: expected {expected_frames}, got {actual_frames}. "
            "Keeping individual chunks untouched."
        )
    partial.replace(final)
    print(f"Reassembled: {final}")


def stitch_outputs(jobs: list[ChunkJob], output_dir: Path, manifest_stem: str, args: argparse.Namespace) -> None:
    reassemble_outputs(jobs, output_dir, manifest_stem, args)


def put_windows_to_sleep() -> None:
    print("Render complete. Asking Windows to sleep...")
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "Add-Type -AssemblyName System.Windows.Forms; "
            "[System.Windows.Forms.Application]::SetSuspendState('Suspend', $false, $false)",
        ],
        check=True,
    )


def handle_interrupt(signum: int, frame: Any) -> None:
    global stop_after_current, interrupt_count
    interrupt_count += 1
    if interrupt_count == 1:
        stop_after_current = True
        print("\nStopping after the current chunk. Press Ctrl+C again to exit immediately.")
    else:
        raise KeyboardInterrupt


def print_dry_run(source_video: str, rows: list[ManifestRow], jobs: list[ChunkJob], fps: float, method_name: str, manifest_stem: str, args: argparse.Namespace) -> None:
    print(f"Method: {method_name}")
    print(f"Source: {source_video}")
    print(f"FPS: {fps:.6g}")
    if jobs:
        print(f"Output dir: {Path(jobs[0].output).parent}")
        dry_args = argparse.Namespace(**vars(args))
        dry_args.method_name = method_name
        print(f"Reassembled mp4: {reassembled_output_path(Path(jobs[0].output).parent, manifest_stem, dry_args)}")
    print("Rows:")
    for row in rows:
        state = "enabled" if row.enabled else "disabled"
        print(f"  row {row.index:03d} {state:8s} {format_time(row.start)} -> {format_time(row.end)} ref={row.reference}")
    print("Jobs:")
    for job in jobs:
        print(
            f"  chunk {job.chunk:04d} row={job.row:03d} "
            f"frames={job.start_frame}+{job.frames} "
            f"{format_time(job.start_time)} -> {format_time(job.end_time)} "
            f"trim={format_time(job.trim_start_time or job.start_time)} -> {format_time(job.trim_end_time or job.end_time)} "
            f"trim_frames={job.trim_start_frame}+{job.trim_frames} "
            f"ref={job.reference} out={Path(job.output).name}"
        )


def run_jobs(args: argparse.Namespace, workflow: dict[str, Any], source_video_path: Path, jobs: list[ChunkJob], ledger_path: Path) -> None:
    signal.signal(signal.SIGINT, handle_interrupt)
    client_id = str(uuid.uuid4())

    for job in jobs:
        if job.status == "done":
            print(f"Skip chunk {job.chunk:04d}: already done")
            continue
        if (ledger_path.parent / "PAUSE").exists():
            print(f"Pause file found: {ledger_path.parent / 'PAUSE'}")
            break
        if stop_after_current:
            break

        print(f"Queue chunk {job.chunk:04d} @ {format_time(job.start_time)} ref={job.reference}")
        wf, output_node_id = patch_workflow_for_job(workflow, source_video_path, job, args.method_engine, args)
        prompt = workflow_to_prompt(wf, output_node_id)
        try:
            job.status = "queued"
            write_ledger(ledger_path, jobs)
            job.prompt_id = queue_prompt(args.comfy_url, prompt, client_id)
            job.status = "running"
            write_ledger(ledger_path, jobs)
            history = wait_for_prompt(args.comfy_url, job.prompt_id, args.poll_seconds)
            source_output = extract_saved_video(history)
            copy_output(source_output, Path(job.output))
            job.status = "done"
            job.error = None
            print(f"Done chunk {job.chunk:04d}: {job.output}")
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            write_ledger(ledger_path, jobs)
            print(f"Failed chunk {job.chunk:04d}: {exc}", file=sys.stderr)
            if not args.keep_going:
                raise
        finally:
            write_ledger(ledger_path, jobs)


def prepare_method_jobs(
    args: argparse.Namespace,
    method_name: str,
    method_engine: str,
    manifest_stem: str,
    rows: list[ManifestRow],
    fps: float,
    duration: float | None,
) -> tuple[list[ChunkJob], Path]:
    method_args = argparse.Namespace(**vars(args))
    method_args.method_name = method_name
    method_args.method_engine = method_engine
    method_slug = method_name
    run_output_dir = Path(args.output_dir) / method_slug / manifest_stem
    output_prefix = f"{args.prefix}_{method_slug}"
    jobs = build_jobs(
        manifest_stem,
        rows,
        fps,
        args.max_frames,
        args.range_overlap_seconds,
        args.chunk_overlap_frames,
        duration,
        run_output_dir,
        output_prefix,
    )
    ledger_path = run_output_dir / "colorize_jobs.jsonl"
    reconcile_jobs(jobs, ledger_path, args)
    return jobs, ledger_path


def run_method(
    args: argparse.Namespace,
    workflow: dict[str, Any],
    source_path: Path,
    manifest_stem: str,
    method_name: str,
    method_engine: str,
    rows: list[ManifestRow],
    fps: float,
    duration: float | None,
) -> int:
    method_args = argparse.Namespace(**vars(args))
    method_args.method_name = method_name
    method_args.method_engine = method_engine
    if method_args.reassemble_fps is None:
        method_args.reassemble_fps = fps
    jobs, ledger_path = prepare_method_jobs(method_args, method_name, method_engine, manifest_stem, rows, fps, duration)
    print(f"=== Method: {method_name} ===")
    write_ledger(ledger_path, jobs)
    run_jobs(method_args, workflow, source_path, jobs, ledger_path)
    if args.reassemble:
        reassemble_outputs(jobs, ledger_path.parent, manifest_stem, method_args)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Image-reference manifest CSV to execute.")
    parser.add_argument("--workflow", default=str(DEFAULT_WORKFLOW), help="ComfyUI workflow JSON.")
    parser.add_argument("--method", default="Both", help="Colorization method: Both, DeepExemplar, or ColorMNet.")
    parser.add_argument("--engine", choices=["deepex", "colormnet"], help="Deprecated alias for --method.")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--max-frames", type=int, default=1000, help="Maximum frames per ComfyUI job.")
    parser.add_argument(
        "--target-width",
        type=int,
        default=928,
        help="DeepEx target width. Use 0 with --target-height 0 for native source size.",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=512,
        help="DeepEx target height. Default is a VRAM-safe middle ground above half-res.",
    )
    parser.add_argument(
        "--deepex-half-resolution",
        action="store_true",
        help="Use DeepEx half-resolution mode for speed/VRAM. Off by default.",
    )
    parser.add_argument(
        "--chunk-overlap-frames",
        type=int,
        default=2,
        help="Extra frames rendered on both sides of automatic mini-chunk splits inside a long manifest row.",
    )
    parser.add_argument(
        "--range-overlap-seconds",
        type=float,
        default=2.0,
        help="Extra seconds rendered before and after every enabled manifest row.",
    )
    parser.add_argument("--fps", type=float, default=24.0, help="Fallback FPS if ffprobe is unavailable.")
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG"))
    parser.add_argument("--ffprobe", default=os.environ.get("FFPROBE"))
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="Continue after failed chunks.")
    parser.add_argument("--reassemble", dest="reassemble", action="store_true", default=True, help="Trim rendered overlap and concatenate completed chunks into a method-level MP4.")
    parser.add_argument("--no-reassemble", dest="reassemble", action="store_false", help="Keep individual chunks only; do not create the reassembled MP4.")
    parser.add_argument(
        "--reassemble-encoder",
        choices=["h264", "prores", "copy"],
        default="h264",
        help="Encoder for reassembled MP4. h264 is safest for Resolve; copy is lossless but may decode badly when trimming between keyframes.",
    )
    parser.add_argument("--reassemble-crf", type=int, default=12, help="CRF for --reassemble-encoder h264.")
    parser.add_argument("--reassemble-preset", default="slow", help="x264 preset for --reassemble-encoder h264.")
    parser.add_argument("--reassemble-fps", type=float, help="Output FPS for frame-exact reassembly. Defaults to the source video's FPS.")
    parser.add_argument("--stitch", action="store_true", help="Deprecated alias; reassembly now runs by default.")
    parser.add_argument("--no-stitch", action="store_true", help="Deprecated alias for --no-reassemble.")
    parser.add_argument(
        "--sleep-when-done",
        action="store_true",
        help="Put Windows to sleep after the runner exits, including after render failures.",
    )
    parser.add_argument(
        "--allow-sleep-while-running",
        action="store_true",
        help="Do not keep Windows awake while the runner is active.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.no_stitch:
        args.reassemble = False
    if args.engine and args.method == "Both":
        args.method = "ColorMNet" if args.engine == "colormnet" else "DeepExemplar"
    method_specs = normalize_methods(args.method)
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
        if not manifest.exists():
            candidate = ROOT / "manifests" / "colorize" / args.manifest
            if candidate.exists():
                manifest = candidate
        if not manifest.exists():
            candidate = ROOT / "examples" / "manifests" / args.manifest
            if candidate.exists():
                manifest = candidate
    workflow_path = Path(args.workflow)
    if not workflow_path.is_absolute():
        workflow_path = ROOT / workflow_path

    source_video, rows = read_manifest(manifest)
    validate_inputs(source_video, rows)
    source_path = resolve_source_video(source_video)
    fps, _duration = probe_video(source_path, args, rows[-1].end if rows else 0)

    manifest_stem = manifest.stem

    if args.dry_run:
        for method_name, method_engine in method_specs:
            jobs, _ledger_path = prepare_method_jobs(args, method_name, method_engine, manifest_stem, rows, fps, _duration)
            print_dry_run(source_video, rows, jobs, fps, method_name, manifest_stem, args)
        return 0

    exit_code = 0
    should_sleep = False
    with WindowsSleepInhibitor(enabled=not args.allow_sleep_while_running):
        try:
            workflow = json.loads(workflow_path.read_text(encoding="utf-8-sig"))
            for method_name, method_engine in method_specs:
                try:
                    run_method(args, workflow, source_path, manifest_stem, method_name, method_engine, rows, fps, _duration)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    exit_code = 1
                    print(f"{method_name} failed: {exc}", file=sys.stderr)
            pause_files = [Path(args.output_dir) / method_name / manifest_stem / "PAUSE" for method_name, _ in method_specs]
            should_sleep = args.sleep_when_done and interrupt_count == 0 and not any(path.exists() for path in pause_files)
        except KeyboardInterrupt:
            exit_code = 130
            print("Interrupted by user; not sleeping.")
        except Exception as exc:
            exit_code = 1
            print(f"Runner failed: {exc}", file=sys.stderr)
            should_sleep = args.sleep_when_done and interrupt_count == 0

    if should_sleep:
        put_windows_to_sleep()
    elif args.sleep_when_done and exit_code != 130:
        print("Not sleeping because the run was paused or interrupted before the queue could finish.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())



