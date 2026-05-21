from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from comfy_api import extract_output_files, queue_prompt, wait_for_comfy, wait_for_prompt
from common import ROOT, file_fingerprint, resolve_path, root_relative, safe_stem, resumable_output, video_info, write_signature

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def load_local_config() -> dict[str, str]:
    path = ROOT / ".ai_remaster_config.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return {str(key): str(value) for key, value in data.items() if value is not None}


def read_manifest(path: Path) -> tuple[str | None, list[dict[str, str]]]:
    source_video = None
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        while True:
            pos = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.startswith("#"):
                if line.startswith("# source_video="):
                    source_video = line.split("=", 1)[1].strip()
                continue
            handle.seek(pos)
            for row in csv.DictReader(handle):
                if row.get("enabled", "true").strip().lower() not in {"false", "0", "no", "off"}:
                    rows.append(row)
            break
    return source_video, rows


def parse_time(text: str) -> float:
    text = text.strip()
    if not text:
        return 0.0
    parts = text.split(":")
    if len(parts) == 1:
        return float(parts[0])
    seconds = float(parts[-1])
    minutes = int(parts[-2])
    hours = int(parts[-3]) if len(parts) > 2 else 0
    return hours * 3600 + minutes * 60 + seconds


def find_ffmpeg(explicit: str | None) -> str:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    exe = "ffmpeg.exe" if is_windows() else "ffmpeg"
    candidates.extend([ROOT / ".cache" / "tools" / "ffmpeg" / exe, Path("ffmpeg")])
    for candidate in candidates:
        try:
            subprocess.run([str(candidate), "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return str(candidate)
        except Exception:
            continue
    raise FileNotFoundError("ffmpeg was not found. Run install_windows.bat/install script again or pass --ffmpeg.")


def is_windows() -> bool:
    import os

    return os.name == "nt"


def copy_to_comfy_input(path: Path, comfy_dir: Path, subfolder: str) -> str:
    target_dir = comfy_dir / "input" / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / path.name
    if not target.exists() or target.stat().st_size != path.stat().st_size:
        shutil.copy2(path, target)
    return str(Path(subfolder) / target.name).replace("\\", "/")


def default_output(source_video: Path) -> Path:
    return ROOT / "intermediate" / "outpainted_colorized" / f"{safe_stem(source_video.name)}_deepexemplar_colorized.mp4"


def signature(args: argparse.Namespace, manifest: Path, source_video: Path) -> dict[str, Any]:
    return {
        "version": 1,
        "tool": "colorize_video.py",
        "manifest": root_relative(manifest),
        "manifest_fingerprint": file_fingerprint(manifest),
        "source_video": root_relative(source_video),
        "source_fingerprint": file_fingerprint(source_video),
        "method": "DeepExemplar",
        "frame_propagate": args.frame_propagate,
        "use_half_resolution": args.use_half_resolution,
        "use_torch_compile": args.use_torch_compile,
        "use_sage_attention": args.use_sage_attention,
    }


def row_reference(row: dict[str, str]) -> Path:
    ref = row.get("color_reference") or row.get("reference") or row.get("source_reference") or ""
    if not ref:
        raise RuntimeError("Manifest row has no color_reference/reference/source_reference.")
    path = resolve_path(ref)
    if not path.exists() and row.get("source_reference"):
        path = resolve_path(row["source_reference"])
    if not path.exists():
        raise FileNotFoundError(f"Reference image not found: {path}")
    return path


def build_prompt(
    video_name: str,
    ref_name: str,
    start_frame: int,
    frame_count: int,
    width: int,
    height: int,
    fps: float,
    args: argparse.Namespace,
    prefix: str,
) -> dict[str, Any]:
    return {
        "1": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": video_name,
                "force_rate": 0.0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": frame_count,
                "skip_first_frames": start_frame,
                "select_every_nth": 1,
                "format": "None",
            },
        },
        "2": {"class_type": "LoadImage", "inputs": {"image": ref_name}},
        "3": {
            "class_type": "DeepExColorVideoNode",
            "inputs": {
                "video_frames": ["1", 0],
                "reference_image": ["2", 0],
                "frame_propagate": args.frame_propagate,
                "use_half_resolution": args.use_half_resolution,
                "target_width": width,
                "target_height": height,
                "use_torch_compile": args.use_torch_compile,
                "use_sage_attention": args.use_sage_attention,
            },
        },
        "4": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["3", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": prefix,
                "format": args.video_format,
                "pix_fmt": "yuv420p",
                "crf": args.crf,
                "save_metadata": True,
                "pingpong": False,
                "save_output": True,
            },
        },
    }


def newest_output(files: list[Path]) -> Path:
    paths = [path for path in files if path.exists() and path.suffix.lower() in VIDEO_EXTS]
    if not paths:
        raise RuntimeError(f"Comfy completed but did not report a video output: {files}")
    return max(paths, key=lambda path: path.stat().st_mtime_ns)


def stitch(ffmpeg: str, chunks: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    concat = output.with_suffix(".concat.txt")
    concat.write_text("".join(f"file '{str(chunk).replace("'", "'\\''")}'\n" for chunk in chunks), encoding="utf-8")
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(partial)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    partial.replace(output)
    concat.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    config = load_local_config()
    parser = argparse.ArgumentParser(description="Colorize outpainted video shots with Deep Exemplar in ComfyUI.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-video", help="Override the # source_video from the manifest.")
    parser.add_argument("--output")
    parser.add_argument("--comfy-url", default=config.get("comfy_url", "http://127.0.0.1:8188"))
    parser.add_argument("--comfy-dir", default=config.get("comfy_dir", str(ROOT / "tools" / "comfyui")))
    parser.add_argument("--comfy-output-root", default="")
    parser.add_argument("--frame-propagate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-half-resolution", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-sage-attention", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--video-format", default="video/h264-mp4")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = resolve_path(args.manifest)
    source_from_manifest, rows = read_manifest(manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"No enabled rows in manifest: {manifest}")
    source_video = resolve_path(args.source_video or source_from_manifest or "")
    if not source_video.exists():
        raise FileNotFoundError(f"Source video not found for colourisation: {source_video}")
    output = resolve_path(args.output) if args.output else default_output(source_video)
    sig = signature(args, manifest, source_video)
    if not args.force and resumable_output(output, sig, video_like=source_video):
        print(f"Reuse colorized video: {output}", flush=True)
        return 0
    if args.dry_run:
        print(f"Would colorize {len(rows)} shot segment(s): {source_video} -> {output}", flush=True)
        return 0

    comfy_dir = resolve_path(args.comfy_dir)
    comfy_output_root = resolve_path(args.comfy_output_root) if args.comfy_output_root else comfy_dir / "output"
    ffmpeg = find_ffmpeg(args.ffmpeg)
    info = video_info(source_video)
    width, height, fps, total_frames = int(info["width"]), int(info["height"]), float(info["fps"]), int(info["frames"])
    video_name = copy_to_comfy_input(source_video, comfy_dir, "arp_colorize")
    wait_for_comfy(args.comfy_url, timeout_seconds=180, poll_seconds=args.poll_seconds)

    chunks: list[Path] = []
    start_frame = 0
    cache_dir = ROOT / ".cache" / "colorized_chunks" / safe_stem(source_video.name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        end_frame = min(total_frames, max(start_frame + 1, round(parse_time(row.get("end", "")) * fps)))
        if index == len(rows) - 1:
            end_frame = total_frames
        frame_count = max(1, end_frame - start_frame)
        ref_name = copy_to_comfy_input(row_reference(row), comfy_dir, "arp_colorize_refs")
        chunk = cache_dir / f"segment_{index:04d}_{start_frame:06d}_{end_frame:06d}.mp4"
        if chunk.exists() and not args.force:
            print(f"Reuse colorized segment {index + 1}/{len(rows)}: {chunk}", flush=True)
            chunks.append(chunk)
            start_frame = end_frame
            continue
        prefix = f"arp_colorize/{safe_stem(source_video.name)}_segment_{index:04d}_{start_frame:06d}_{end_frame:06d}"
        print(f"Colorize segment {index + 1}/{len(rows)}: frames {start_frame}-{end_frame} using {ref_name}", flush=True)
        prompt = build_prompt(video_name, ref_name, start_frame, frame_count, width, height, fps, args, prefix)
        prompt_id = queue_prompt(args.comfy_url, prompt)
        history = wait_for_prompt(args.comfy_url, prompt_id, args.poll_seconds)
        produced = newest_output(extract_output_files(history, comfy_output_root))
        shutil.copy2(produced, chunk)
        chunks.append(chunk)
        start_frame = end_frame

    stitch(ffmpeg, chunks, output)
    write_signature(output, sig)
    print(f"Wrote colorized video: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
