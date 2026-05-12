#!/usr/bin/env python3
"""Manifest-driven ComfyUI runner for widescreen/outpainting passes.

CSV format:
    # source_video=source_4x3/my_clip.mp4
    enabled,end,prompt
    true,00:00:20,"restore the missing widescreen edges, preserve the original frame"

Rows use end times only. Each row begins where the previous row ended. Disabled
rows advance the timeline but do not queue jobs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import colorize_manifest_runner as base

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOW = ROOT / "workflows" / "widescreen" / "LTX-2.3 Video Outpainting.json"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "outpainted"
DEFAULT_PREFIX = "remaster_outpainted"
DEFAULT_PROMPT = (
    "Expand this black-and-white film clip from 4:3 to widescreen. Preserve the original center frame exactly. "
    "Generate only plausible missing left and right image area matching the lighting, grain, lens softness, set design, "
    "costumes, camera motion, and period atmosphere. Do not change faces, typography, props, or composition in the original frame."
)

NODE_IDS = {
    "load_video": "5060",
    "outpaint_subgraph": "5151",
    "prompt": "5138",
    "save_output": "5152",
}

stop_after_current = False
interrupt_count = 0


@dataclass
class ManifestRow:
    index: int
    enabled: bool
    start: float
    end: float
    prompt: str


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
    required = {"enabled", "end"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    rows: list[ManifestRow] = []
    start = 0.0
    for index, row in enumerate(reader):
        end = base.parse_time(row["end"])
        if end <= start:
            raise ValueError(f"Row {index} end time {row['end']} must be after {base.format_time(start)}")
        rows.append(
            ManifestRow(
                index=index,
                enabled=truthy(row["enabled"]),
                start=start,
                end=end,
                prompt=(row.get("prompt") or DEFAULT_PROMPT).strip() or DEFAULT_PROMPT,
            )
        )
        start = end
    return source_video, rows


def resolve_input_video(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / "input" / path_text


def build_jobs(
    manifest_stem: str,
    rows: list[ManifestRow],
    fps: float,
    max_frames: int,
    source_duration: float | None,
    output_dir: Path,
    prefix: str,
) -> list[base.ChunkJob]:
    jobs: list[base.ChunkJob] = []
    for row in rows:
        if not row.enabled:
            continue
        start_frame = int(round(row.start * fps))
        end_frame = int(round(row.end * fps))
        if source_duration is not None:
            end_frame = min(end_frame, int(round(source_duration * fps)))
        cursor = start_frame
        while cursor < end_frame:
            job_end = min(cursor + max_frames, end_frame)
            chunk = len(jobs)
            stem = f"{prefix}_{manifest_stem}_chunk_{chunk:04d}"
            job = base.ChunkJob(
                chunk=chunk,
                row=row.index,
                start_frame=cursor,
                frames=job_end - cursor,
                start_time=cursor / fps,
                end_time=job_end / fps,
                reference=row.prompt,
                output=str(output_dir / f"{stem}.mp4"),
                save_prefix=stem + "_tmp",
                trim_start_time=cursor / fps,
                trim_end_time=job_end / fps,
                trim_start_frame=0,
                trim_frames=job_end - cursor,
            )
            jobs.append(job)
            cursor = job_end
    return jobs


def set_widget_if_present(workflow: dict[str, Any], node_id: str, key: str | int, value: Any) -> None:
    try:
        base.set_widget(base.node_by_id(workflow, node_id), key, value)
    except KeyError:
        pass


def patch_workflow_for_job(workflow: dict[str, Any], source_video_path: Path, job: base.ChunkJob, args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    wf = copy.deepcopy(workflow)
    load_video = base.node_by_id(wf, NODE_IDS["load_video"])
    base.set_widget(load_video, "video", str(source_video_path))
    base.set_widget(load_video, "force_rate", 0)
    base.set_widget(load_video, "custom_width", 0)
    base.set_widget(load_video, "custom_height", 0)
    base.set_widget(load_video, "frame_load_cap", job.frames)
    base.set_widget(load_video, "skip_first_frames", job.start_frame)
    base.set_widget(load_video, "select_every_nth", 1)
    base.set_widget(load_video, "format", "AnimateDiff")

    # In the public LTX 2.3 outpainting workflow, target aspect is proxied through the subgraph node.
    set_widget_if_present(wf, NODE_IDS["outpaint_subgraph"], 1, args.aspect_width)
    set_widget_if_present(wf, NODE_IDS["outpaint_subgraph"], 2, args.aspect_height)
    set_widget_if_present(wf, NODE_IDS["prompt"], "text", job.reference)
    set_widget_if_present(wf, NODE_IDS["prompt"], 0, job.reference)

    save = base.node_by_id(wf, NODE_IDS["save_output"])
    base.set_widget(save, "filename_prefix", job.save_prefix)
    base.set_widget(save, "save_output", True)
    return wf, NODE_IDS["save_output"]


def handle_interrupt(signum: int, frame: Any) -> None:
    global stop_after_current, interrupt_count
    interrupt_count += 1
    if interrupt_count == 1:
        stop_after_current = True
        print("\nStopping after the current chunk. Press Ctrl+C again to exit immediately.")
    else:
        raise KeyboardInterrupt


def run_jobs(args: argparse.Namespace, workflow: dict[str, Any], source_video_path: Path, jobs: list[base.ChunkJob], ledger_path: Path) -> None:
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
        print(f"Queue outpaint chunk {job.chunk:04d} @ {base.format_time(job.start_time)}")
        wf, output_node_id = patch_workflow_for_job(workflow, source_video_path, job, args)
        prompt = base.workflow_to_prompt(wf, output_node_id)
        try:
            job.status = "queued"
            base.write_ledger(ledger_path, jobs)
            job.prompt_id = base.queue_prompt(args.comfy_url, prompt, client_id)
            job.status = "running"
            base.write_ledger(ledger_path, jobs)
            history = base.wait_for_prompt(args.comfy_url, job.prompt_id, args.poll_seconds)
            source_output = base.extract_saved_video(history)
            base.copy_output(source_output, Path(job.output))
            job.status = "done"
            job.error = None
            print(f"Done outpaint chunk {job.chunk:04d}: {job.output}")
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            base.write_ledger(ledger_path, jobs)
            print(f"Failed chunk {job.chunk:04d}: {exc}", file=sys.stderr)
            if not args.keep_going:
                raise
        finally:
            base.write_ledger(ledger_path, jobs)


def print_dry_run(source_video: str, rows: list[ManifestRow], jobs: list[base.ChunkJob], fps: float, output_dir: Path, args: argparse.Namespace) -> None:
    print(f"Source: {source_video}")
    print(f"FPS: {fps:.6g}")
    print(f"Output dir: {output_dir}")
    print(f"Aspect: {args.aspect_width}:{args.aspect_height}")
    print("Rows:")
    for row in rows:
        state = "enabled" if row.enabled else "disabled"
        print(f"  row {row.index:03d} {state:8s} {base.format_time(row.start)} -> {base.format_time(row.end)} prompt={row.prompt[:90]}")
    print("Jobs:")
    for job in jobs:
        print(f"  chunk {job.chunk:04d} row={job.row:03d} frames={job.start_frame}+{job.frames} {base.format_time(job.start_time)} -> {base.format_time(job.end_time)} out={Path(job.output).name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Outpaint manifest CSV to execute.")
    parser.add_argument("--workflow", default=str(DEFAULT_WORKFLOW), help="ComfyUI widescreen/outpainting workflow JSON.")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--max-frames", type=int, default=480, help="Maximum frames per ComfyUI job.")
    parser.add_argument("--aspect-width", type=int, default=16)
    parser.add_argument("--aspect-height", type=int, default=9)
    parser.add_argument("--fps", type=float, default=24.0, help="Fallback FPS if ffprobe is unavailable.")
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument("--ffprobe", default=None)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--reassemble", dest="reassemble", action="store_true", default=True)
    parser.add_argument("--no-reassemble", dest="reassemble", action="store_false")
    parser.add_argument("--reassemble-encoder", choices=["h264", "prores", "copy"], default="h264")
    parser.add_argument("--reassemble-crf", type=int, default=12)
    parser.add_argument("--reassemble-preset", default="slow")
    parser.add_argument("--reassemble-fps", type=float, help="Defaults to source FPS.")
    parser.add_argument("--sleep-when-done", action="store_true")
    parser.add_argument("--allow-sleep-while-running", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / "manifests" / "outpaint" / manifest
        if not manifest.exists():
            manifest = ROOT / args.manifest
        if not manifest.exists():
            candidate = ROOT / "examples" / "manifests" / args.manifest
            if candidate.exists():
                manifest = candidate
    workflow_path = Path(args.workflow)
    if not workflow_path.is_absolute():
        workflow_path = ROOT / workflow_path
    source_video, rows = read_manifest(manifest)
    source_path = resolve_input_video(source_video)
    if not source_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_path}")
    fps, duration = base.probe_video(source_path, args, rows[-1].end if rows else 0)
    if args.reassemble_fps is None:
        args.reassemble_fps = fps
    manifest_stem = manifest.stem
    output_dir = Path(args.output_dir) / manifest_stem
    jobs = build_jobs(manifest_stem, rows, fps, args.max_frames, duration, output_dir, args.prefix)
    ledger_path = output_dir / "outpaint_jobs.jsonl"
    base.reconcile_jobs(jobs, ledger_path, args)
    if args.dry_run:
        print_dry_run(source_video, rows, jobs, fps, output_dir, args)
        return 0
    exit_code = 0
    should_sleep = False
    with base.WindowsSleepInhibitor(enabled=not args.allow_sleep_while_running):
        try:
            workflow = json.loads(workflow_path.read_text(encoding="utf-8-sig"))
            base.write_ledger(ledger_path, jobs)
            run_jobs(args, workflow, source_path, jobs, ledger_path)
            if args.reassemble:
                base.reassemble_outputs(jobs, output_dir, manifest_stem, args)
            should_sleep = args.sleep_when_done and interrupt_count == 0 and not (output_dir / "PAUSE").exists()
        except KeyboardInterrupt:
            exit_code = 130
            print("Interrupted by user; not sleeping.")
        except Exception as exc:
            exit_code = 1
            print(f"Runner failed: {exc}", file=sys.stderr)
            should_sleep = args.sleep_when_done and interrupt_count == 0
    if should_sleep:
        base.put_windows_to_sleep()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

