from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from comfy_api import ensure_node_types, extract_output_files, queue_prompt, wait_for_comfy, wait_for_prompt
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


def method_suffix(method: str) -> str:
    return "colormnet" if method == "colormnet" else "deepexemplar"


def default_output(source_video: Path, method: str) -> Path:
    return ROOT / "intermediate" / "outpainted_colorized" / f"{safe_stem(source_video.name)}_{method_suffix(method)}_colorized.mp4"


def reference_signature(row: dict[str, str]) -> dict[str, Any]:
    ref = row_reference(row)
    return {
        "start": row.get("start", ""),
        "end": row.get("end", ""),
        "start_frame": row.get("start_frame", ""),
        "end_frame": row.get("end_frame", ""),
        "reference": root_relative(ref),
        "reference_fingerprint": file_fingerprint(ref),
        "fade_to_next": row.get("fade_to_next", ""),
        "crossfade_seconds": row.get("crossfade_seconds", ""),
    }


def signature(args: argparse.Namespace, manifest: Path, source_video: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "version": 4,
        "tool": "colorize_video.py",
        "manifest": root_relative(manifest),
        "manifest_fingerprint": file_fingerprint(manifest),
        "source_video": root_relative(source_video),
        "source_fingerprint": file_fingerprint(source_video),
        "references": [reference_signature(row) for row in rows],
        "method": args.method,
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
    color_node = colorization_node(args, width, height)
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
        "3": color_node,
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


def colorization_node(args: argparse.Namespace, width: int, height: int) -> dict[str, Any]:
    if args.method == "colormnet":
        return {
            "class_type": "ColorMNetVideo",
            "inputs": {
                "video_frames": ["1", 0],
                "reference_image": ["2", 0],
                "target_width": width,
                "target_height": height,
                "memory_mode": args.colormnet_memory_mode,
                "feature_encoder": args.colormnet_feature_encoder,
                "use_fp16": True,
                "use_torch_compile": args.use_torch_compile,
                "text_guidance": args.colormnet_text_guidance,
                "text_guidance_weight": args.colormnet_text_guidance_weight,
            },
        }
    return {
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
    }


def newest_output(files: list[Path]) -> Path:
    paths = [path for path in files if path.exists() and path.suffix.lower() in VIDEO_EXTS]
    if not paths:
        raise RuntimeError(f"Comfy completed but did not report a video output: {files}")
    return max(paths, key=lambda path: path.stat().st_mtime_ns)


def segment_signature(
    args: argparse.Namespace,
    source_video: Path,
    row: dict[str, str],
    reference: Path,
    start_frame: int,
    end_frame: int,
    base_start_frame: int,
    base_end_frame: int,
    width: int,
    height: int,
    fps: float,
) -> dict[str, Any]:
    return {
        "version": 4,
        "tool": "colorize_video.py",
        "kind": f"{args.method} segment",
        "source_video": root_relative(source_video),
        "source_fingerprint": file_fingerprint(source_video),
        "reference": root_relative(reference),
        "reference_fingerprint": file_fingerprint(reference),
        "row_start": row.get("start", ""),
        "row_end": row.get("end", ""),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "base_start_frame": base_start_frame,
        "base_end_frame": base_end_frame,
        "fade_to_next": row.get("fade_to_next", ""),
        "crossfade_seconds": row.get("crossfade_seconds", ""),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_propagate": args.frame_propagate,
        "use_half_resolution": args.use_half_resolution,
        "use_torch_compile": args.use_torch_compile,
        "use_sage_attention": args.use_sage_attention,
        "colormnet_memory_mode": args.colormnet_memory_mode,
        "colormnet_feature_encoder": args.colormnet_feature_encoder,
        "colormnet_text_guidance": args.colormnet_text_guidance,
        "video_format": args.video_format,
        "crf": args.crf,
    }


def segment_resumable(chunk: Path, chunk_sig: dict[str, Any], width: int, height: int, expected_frames: int) -> bool:
    if not resumable_output(chunk, chunk_sig, width=width, height=height):
        return False
    try:
        info = video_info(chunk)
    except Exception:
        return False
    return abs(int(info["frames"]) - expected_frames) <= 3


def replace_with_retry(source: Path, target: Path, attempts: int = 12, delay: float = 0.5) -> None:
    last_exc: PermissionError | None = None
    for _attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def normalize_clip(ffmpeg: str, source: Path, output: Path, fps: float, expected_frames: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    vf = f"setpts=N/({fps:.8f}*TB),fps={fps:.8f},trim=end_frame={expected_frames},setpts=N/({fps:.8f}*TB),setsar=1"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vf",
        vf,
        "-an",
        "-r",
        f"{fps:.8f}",
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        str(partial),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    replace_with_retry(partial, output)


def truthy(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "on"}


def transition_seconds(row: dict[str, str]) -> float:
    if not truthy(row.get("fade_to_next", "")):
        return 0.0
    try:
        return max(0.0, float(row.get("crossfade_seconds", "") or 0.0))
    except ValueError:
        return 0.0


def shot_plan(rows: list[dict[str, str]], total_frames: int, fps: float) -> tuple[list[dict[str, int]], list[int]]:
    base: list[tuple[int, int]] = []
    start_frame = 0
    for index, row in enumerate(rows):
        end_frame = min(total_frames, max(start_frame + 1, round(parse_time(row.get("end", "")) * fps)))
        if index == len(rows) - 1:
            end_frame = total_frames
        base.append((start_frame, end_frame))
        start_frame = end_frame

    transitions = [0 for _ in rows]
    for index, row in enumerate(rows[:-1]):
        frames = int(round(transition_seconds(row) * fps))
        if frames <= 0:
            continue
        left = max(1, base[index][1] - base[index][0])
        right = max(1, base[index + 1][1] - base[index + 1][0])
        transitions[index] = max(1, min(frames, left, right))

    plan: list[dict[str, int]] = []
    for index, (base_start, base_end) in enumerate(base):
        prev_frames = transitions[index - 1] if index > 0 else 0
        next_frames = transitions[index] if index < len(transitions) else 0
        pre = prev_frames // 2
        post = next_frames - (next_frames // 2)
        actual_start = max(0, base_start - pre)
        actual_end = min(total_frames, base_end + post)
        plan.append(
            {
                "base_start": base_start,
                "base_end": base_end,
                "start": actual_start,
                "end": max(actual_start + 1, actual_end),
            }
        )
    return plan, transitions


def stitch(ffmpeg: str, chunks: list[Path], output: Path, fps: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    concat = output.with_suffix(".concat.txt")
    concat.write_text("".join("file '" + str(chunk).replace("'", "'\\''") + "'\n" for chunk in chunks), encoding="utf-8")
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat),
        "-vf",
        f"setpts=N/({fps:.8f}*TB),fps={fps:.8f},setsar=1",
        "-an",
        "-r",
        f"{fps:.8f}",
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        str(partial),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    replace_with_retry(partial, output)
    concat.unlink(missing_ok=True)


def transition_groups(transitions: list[int]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start = 0
    for index, frames in enumerate(transitions):
        if frames > 0:
            continue
        groups.append((start, index))
        start = index + 1
    if transitions:
        groups.append((start, len(transitions) - 1))
    elif not groups:
        groups.append((0, 0))
    return [(left, right) for left, right in groups if left <= right]


def xfade_group(ffmpeg: str, chunks: list[Path], transitions: list[int], output: Path, fps: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    cmd = [ffmpeg, "-y"]
    for chunk in chunks:
        cmd += ["-i", str(chunk)]

    filters: list[str] = []
    for index in range(len(chunks)):
        filters.append(f"[{index}:v]fps=fps={fps:.8f},setpts=PTS-STARTPTS[v{index}]")

    previous = "v0"
    accumulated = video_info(chunks[0])["frames"] / fps
    for index in range(1, len(chunks)):
        duration = max(1 / fps, transitions[index - 1] / fps)
        offset = max(0.0, accumulated - duration)
        current = f"x{index}"
        filters.append(f"[{previous}][v{index}]xfade=transition=fade:duration={duration:.8f}:offset={offset:.8f},setpts=PTS-STARTPTS[{current}]")
        previous = current
        accumulated += video_info(chunks[index])["frames"] / fps - duration
    filters.append(f"[{previous}]format=yuv420p[vout]")

    cmd += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[vout]",
        "-an",
        "-r",
        f"{fps:.8f}",
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        str(partial),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    replace_with_retry(partial, output)


def stitch_colorized(ffmpeg: str, chunks: list[Path], transitions: list[int], output: Path, fps: float, source_stem: str) -> None:
    if not any(transitions):
        stitch(ffmpeg, chunks, output, fps)
        return

    group_outputs: list[Path] = []
    group_dir = ROOT / ".cache" / "colorized_chunks" / "crossfaded" / source_stem
    group_dir.mkdir(parents=True, exist_ok=True)
    for group_index, (left, right) in enumerate(transition_groups(transitions)):
        group_chunks = chunks[left : right + 1]
        if len(group_chunks) == 1:
            group_outputs.append(group_chunks[0])
            continue
        group_transitions = transitions[left:right]
        group_output = group_dir / f"group_{group_index:04d}_{left:04d}_{right:04d}.mp4"
        xfade_group(ffmpeg, group_chunks, group_transitions, group_output, fps)
        group_outputs.append(group_output)
    stitch(ffmpeg, group_outputs, output, fps)


def build_parser() -> argparse.ArgumentParser:
    config = load_local_config()
    parser = argparse.ArgumentParser(description="Colorize outpainted video shots with reference-guided ComfyUI colorization.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-video", help="Override the # source_video from the manifest.")
    parser.add_argument("--output")
    parser.add_argument("--method", choices=["deepexemplar", "colormnet", "both"], default="deepexemplar")
    parser.add_argument("--comfy-url", default=config.get("comfy_url", "http://127.0.0.1:8188"))
    parser.add_argument("--comfy-dir", default=config.get("comfy_dir", str(ROOT / "tools" / "comfyui")))
    parser.add_argument("--comfy-output-root", default="")
    parser.add_argument("--frame-propagate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-half-resolution", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-sage-attention", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--colormnet-memory-mode", choices=["balanced", "low_memory", "high_quality"], default="balanced")
    parser.add_argument("--colormnet-feature-encoder", choices=["resnet50", "vgg19", "dinov2_vits", "dinov2_vitb", "dinov2_vitl", "clip_vitb"], default="resnet50")
    parser.add_argument("--colormnet-text-guidance", default="")
    parser.add_argument("--colormnet-text-guidance-weight", type=float, default=0.3)
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
    if args.method == "both":
        for method in ("deepexemplar", "colormnet"):
            child = argparse.Namespace(**vars(args))
            child.method = method
            child.output = ""
            run(child)
        return 0
    return run(args)


def run(args: argparse.Namespace) -> int:
    manifest = resolve_path(args.manifest)
    source_from_manifest, rows = read_manifest(manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"No enabled rows in manifest: {manifest}")
    source_video = resolve_path(args.source_video or source_from_manifest or "")
    if not source_video.exists():
        raise FileNotFoundError(f"Source video not found for colourisation: {source_video}")
    output = resolve_path(args.output) if args.output else default_output(source_video, args.method)
    sig = signature(args, manifest, source_video, rows)
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
    required_nodes = {
        "VHS_LoadVideo": "ComfyUI-VideoHelperSuite",
        "VHS_VideoCombine": "ComfyUI-VideoHelperSuite",
    }
    if args.method == "colormnet":
        required_nodes["ColorMNetVideo"] = "ComfyUI-Reference-Based-Video-Colorization"
    else:
        required_nodes["DeepExColorVideoNode"] = "ComfyUI-Reference-Based-Video-Colorization"
    ensure_node_types(args.comfy_url, required_nodes, f"{args.method} colourisation")

    chunks: list[Path] = []
    plan, transitions = shot_plan(rows, total_frames, fps)
    cache_dir = ROOT / ".cache" / "colorized_chunks" / method_suffix(args.method) / safe_stem(source_video.name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        item = plan[index]
        start_frame = item["start"]
        end_frame = item["end"]
        frame_count = max(1, end_frame - start_frame)
        reference = row_reference(row)
        ref_name = copy_to_comfy_input(reference, comfy_dir, "arp_colorize_refs")
        chunk = cache_dir / f"segment_{index:04d}_{start_frame:06d}_{end_frame:06d}.mp4"
        chunk_sig = segment_signature(args, source_video, row, reference, start_frame, end_frame, item["base_start"], item["base_end"], width, height, fps)
        if not args.force and segment_resumable(chunk, chunk_sig, width, height, frame_count):
            print(f"Reuse colorized segment {index + 1}/{len(rows)}: {chunk}", flush=True)
            chunks.append(chunk)
            continue
        prefix = f"arp_colorize/{method_suffix(args.method)}_{safe_stem(source_video.name)}_segment_{index:04d}_{start_frame:06d}_{end_frame:06d}"
        print(f"Colorize segment {index + 1}/{len(rows)} with {args.method}: frames {start_frame}-{end_frame} using {ref_name}", flush=True)
        prompt = build_prompt(video_name, ref_name, start_frame, frame_count, width, height, fps, args, prefix)
        prompt_id = queue_prompt(args.comfy_url, prompt)
        history = wait_for_prompt(args.comfy_url, prompt_id, args.poll_seconds)
        produced = newest_output(extract_output_files(history, comfy_output_root))
        normalize_clip(ffmpeg, produced, chunk, fps, frame_count)
        write_signature(chunk, chunk_sig)
        chunks.append(chunk)

    stitch_colorized(ffmpeg, chunks, transitions, output, fps, safe_stem(source_video.name))
    write_signature(output, sig)
    print(f"Wrote colorized video: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
