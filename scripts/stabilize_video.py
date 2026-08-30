#!/usr/bin/env python3
"""Scene-aware two-pass video stabilization using FFmpeg/libvidstab.

The source is divided at detected shot boundaries before motion analysis.  Each shot gets its own
transform file, preventing a cut or dissolve from being interpreted as camera movement.  The
intermediate and default output codec is FFV1 so this early pipeline phase does not add another
lossy H.264 generation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import artifact_ids as aid
from common import ROOT, file_fingerprint, find_ffmpeg, resolve_path, resumable_output, root_relative, write_signature
from generate_references import build_parser as shot_parser
from generate_references import detect_shots, probe_video, sample_video


DEFAULT_OUTPUT_ROOT = ROOT / "intermediate" / "stabilized"


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def stabilization_identity(source: Path, args: argparse.Namespace) -> dict[str, Any]:
    return aid.stabilize_identity(
        source.name,
        smoothing=int(args.smoothing),
        max_shift=int(args.max_shift),
        max_angle=float(args.max_angle),
        zoom=float(args.zoom),
        shot_threshold=float(args.shot_threshold),
        min_shot_seconds=float(args.min_shot_seconds),
        scene_aware=not bool(args.single_shot),
        encoder=str(args.encoder),
    )


def default_output(source: Path, args: argparse.Namespace) -> Path:
    extension = "mov" if args.encoder == "prores" else "mkv"
    return DEFAULT_OUTPUT_ROOT / aid.artifact_name(
        aid.source_word(source.name), "stabilized", stabilization_identity(source, args), extension
    )


def browser_preview_path(output: Path) -> Path:
    return output.with_name(output.stem + "_preview.mp4")


def output_signature(source: Path, args: argparse.Namespace, shots: list[tuple[int, int]]) -> dict[str, Any]:
    return {
        "tool": "stabilize_video.py",
        "version": 1,
        "source": root_relative(source),
        "source_fingerprint": file_fingerprint(source),
        "identity": stabilization_identity(source, args),
        "shots": [[int(start), int(end)] for start, end in shots],
    }


def detect_shot_ranges(source: Path, shot_threshold: float, min_shot_seconds: float) -> tuple[Any, list[tuple[int, int]]]:
    """Use the same detector as Reference Generation so both phases agree about cuts."""
    info = probe_video(source)
    detector_args = shot_parser().parse_args(["--source-video", str(source)])
    detector_args.shot_threshold = clamp(float(shot_threshold), 0.001, 2.0)
    detector_args.min_shot_seconds = max(0.1, float(min_shot_seconds))
    detector_args.sample_seconds = 0.0
    # Stabilization needs only genuine discontinuous cuts. The reference detector's
    # long-window dissolve and anchor heuristics deliberately react to accumulated scene
    # change, which misclassifies a sustained pan and resets the transform mid-move.
    detector_args.anchor_threshold = 0.0
    detector_args.dissolve_threshold = 0.0
    samples = sample_video(source, info, detector_args)
    detected = detect_shots(samples, info, detector_args)
    ranges = [(int(shot.start_frame), int(shot.end_frame)) for shot in detected if shot.end_frame > shot.start_frame]
    return info, ranges or [(0, int(info.frame_count))]


def manifest_shot_ranges(path: Path, frame_count: int) -> list[tuple[int, int]]:
    """Read user-reviewed shot spans when the project already has a reference manifest."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    ranges: list[tuple[int, int]] = []
    for row in rows:
        if str(row.get("enabled", "true")).strip().lower() not in {"1", "true", "yes", "on"}:
            continue
        try:
            start = max(0, int(float(row.get("start_frame", "0") or 0)))
            end = min(frame_count, int(float(row.get("end_frame", str(frame_count)) or frame_count)))
        except (TypeError, ValueError):
            continue
        if end > start:
            ranges.append((start, end))
    ranges.sort()
    if not ranges or ranges[0][0] != 0 or ranges[-1][1] != frame_count:
        return []
    if any(left[1] != right[0] for left, right in zip(ranges, ranges[1:])):
        return []
    return ranges


def filter_path(filename: str) -> str:
    # Transform files live in the subprocess working directory, so only a safe basename is needed
    # in the FFmpeg filter expression (avoids Windows drive-letter/filter escaping problems).
    return filename.replace("\\", "/").replace("'", "\\'")


def run(command: list[str], *, cwd: Path, label: str) -> None:
    print(label, flush=True)
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}.")


def encoder_args(encoder: str, pix_fmt: str) -> list[str]:
    if encoder == "prores":
        return ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]
    args = ["-c:v", "ffv1", "-level", "3", "-coder", "1", "-context", "1", "-g", "1"]
    if pix_fmt:
        args.extend(["-pix_fmt", pix_fmt])
    return args


def probe_streams(ffmpeg: str, source: Path) -> dict[str, Any]:
    ffprobe = str(Path(ffmpeg).with_name("ffprobe.exe" if Path(ffmpeg).suffix.lower() == ".exe" else "ffprobe"))
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "stream=index,codec_type,pix_fmt", "-of", "json", str(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout or "{}").get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    return {
        "pix_fmt": str(video.get("pix_fmt") or ""),
        "has_audio": any(stream.get("codec_type") == "audio" for stream in streams),
    }


def stabilization_pix_fmt(pix_fmt: str) -> str:
    # FFV1 supports the common planar/RGB formats used by archive sources. Hardware and palette
    # formats do not describe a useful lossless delivery format, so fall back conservatively.
    if not pix_fmt or pix_fmt.startswith(("cuda", "d3d", "vaapi", "qsv", "pal")):
        return "yuv422p10le"
    if pix_fmt.startswith(("gbr", "rgb", "bgr", "gray")):
        if "16" in pix_fmt or "12" in pix_fmt:
            return "yuv444p12le"
        if "10" in pix_fmt:
            return "yuv444p10le"
        return "yuv444p"
    return pix_fmt


def stabilize_shot(
    ffmpeg: str,
    source: Path,
    work_dir: Path,
    index: int,
    start: int,
    end: int,
    fps: float,
    args: argparse.Namespace,
    pix_fmt: str,
) -> Path:
    frames = max(1, end - start)
    smoothing = min(max(1, int(args.smoothing)), max(1, (frames - 1) // 2))
    transform_name = f"shot_{index:04d}.trf"
    chunk = work_dir / f"shot_{index:04d}.{'mov' if args.encoder == 'prores' else 'mkv'}"
    trim = f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,format={pix_fmt}"
    detect = (
        f"{trim},vidstabdetect=result='{filter_path(transform_name)}':"
        "shakiness=5:accuracy=15:stepsize=6:mincontrast=0.08"
    )
    run(
        [ffmpeg, "-hide_banner", "-y", "-i", str(source), "-map", "0:v:0", "-vf", detect, "-an", "-f", "null", "-"],
        cwd=work_dir,
        label=f"Shot {index + 1}: analysing frames {start}-{end - 1}",
    )

    max_shift = -1 if int(args.max_shift) <= 0 else int(args.max_shift)
    max_angle = -1.0 if float(args.max_angle) <= 0 else math.radians(float(args.max_angle))
    transform = (
        f"{trim},vidstabtransform=input='{filter_path(transform_name)}':smoothing={smoothing}:"
        f"maxshift={max_shift}:maxangle={max_angle:.8f}:crop=black:relative=1:"
        f"optzoom=0:zoom={float(args.zoom):.4f}:interpol=bicubic,"
        f"fps={fps:.10f},setsar=1"
    )
    command = [
        ffmpeg, "-hide_banner", "-y", "-i", str(source), "-map", "0:v:0", "-vf", transform,
        "-an", "-fps_mode", "cfr", "-r", f"{fps:.10f}", *encoder_args(args.encoder, pix_fmt), str(chunk),
    ]
    run(command, cwd=work_dir, label=f"Shot {index + 1}: stabilizing {frames} frames")
    return chunk


def concat_chunks(ffmpeg: str, source: Path, chunks: list[Path], output: Path, work_dir: Path, has_audio: bool) -> None:
    concat_file = work_dir / "shots.txt"
    concat_file.write_text(
        "".join(f"file '{chunk.as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for chunk in chunks),
        encoding="utf-8",
    )
    video_only = work_dir / ("stabilized.mov" if output.suffix.lower() == ".mov" else "stabilized.mkv")
    run(
        [ffmpeg, "-hide_banner", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c:v", "copy", "-an", str(video_only)],
        cwd=work_dir,
        label="Joining stabilized shots",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial" + output.suffix)
    partial.unlink(missing_ok=True)
    if has_audio:
        run(
            [
                ffmpeg, "-hide_banner", "-y", "-i", str(video_only), "-i", str(source),
                "-map", "0:v:0", "-map", "1:a?", "-c", "copy", "-shortest", str(partial),
            ],
            cwd=work_dir,
            label="Restoring source audio",
        )
    else:
        shutil.copy2(video_only, partial)
    partial.replace(output)


def ensure_browser_preview(ffmpeg: str, output: Path) -> Path:
    preview = browser_preview_path(output)
    if preview.is_file() and preview.stat().st_mtime_ns >= output.stat().st_mtime_ns:
        print(f"Reuse stabilization browser preview: {preview}", flush=True)
        return preview
    preview.parent.mkdir(parents=True, exist_ok=True)
    partial = preview.with_name(preview.stem + ".partial" + preview.suffix)
    partial.unlink(missing_ok=True)
    run(
        [
            ffmpeg, "-hide_banner", "-y", "-i", str(output), "-map", "0:v:0", "-an",
            "-vf", "scale=w='min(1280,iw)':h=-2:flags=lanczos,format=yuv420p",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-movflags", "+faststart", str(partial),
        ],
        cwd=output.parent,
        label="Creating stabilization browser preview",
    )
    partial.replace(preview)
    return preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scene-aware archive-film stabilization with FFmpeg/libvidstab.")
    parser.add_argument("--source", required=True, help="Input video. Relative paths resolve from the repository root.")
    parser.add_argument("--output", type=Path, help="Output path. Defaults to intermediate/stabilized.")
    parser.add_argument("--smoothing", type=int, default=12, help="Motion smoothing radius in frames (1-1000).")
    parser.add_argument("--max-shift", type=int, default=48, help="Maximum translation correction in pixels; 0 is unlimited.")
    parser.add_argument("--max-angle", type=float, default=3.0, help="Maximum rotation correction in degrees; 0 is unlimited.")
    parser.add_argument("--zoom", type=float, default=3.0, help="Fixed safety zoom percentage used to hide transformed edges.")
    parser.add_argument("--shot-threshold", type=float, default=0.075, help="Shot detector sensitivity threshold.")
    parser.add_argument("--min-shot-seconds", type=float, default=1.0, help="Minimum detected shot length.")
    parser.add_argument("--single-shot", action="store_true", help="Treat the entire clip as one continuous camera move and do not reset stabilization at detected cuts.")
    parser.add_argument("--encoder", choices=("ffv1", "prores"), default="ffv1")
    parser.add_argument("--shot-manifest", type=Path, help="Optional user-reviewed reference manifest whose frame spans override automatic detection.")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main_with_args(args: argparse.Namespace) -> int:
    source = resolve_path(args.source)
    if not source.is_file():
        raise FileNotFoundError(f"Stabilization source was not found: {source}")
    args.smoothing = int(clamp(int(args.smoothing), 1, 1000))
    args.max_shift = int(clamp(int(args.max_shift), 0, 500))
    args.max_angle = clamp(float(args.max_angle), 0.0, 180.0)
    args.zoom = clamp(float(args.zoom), 0.0, 20.0)
    output = resolve_path(args.output) if args.output else default_output(source, args)

    manifest = resolve_path(args.shot_manifest) if args.shot_manifest else None
    if args.single_shot:
        info = probe_video(source)
        shots = [(0, int(info.frame_count))]
        print(f"Continuous-shot mode: tracking all {info.frame_count} frames without transform resets.", flush=True)
    elif manifest and manifest.is_file():
        print(f"Analysing shot boundaries: {source}", flush=True)
        info = probe_video(source)
        shots = manifest_shot_ranges(manifest, int(info.frame_count))
        if shots:
            print(f"Using {len(shots)} user-reviewed shot span(s) from: {manifest}", flush=True)
        else:
            print(f"Shot manifest did not cover the source exactly; falling back to automatic detection: {manifest}", flush=True)
            info, shots = detect_shot_ranges(source, args.shot_threshold, args.min_shot_seconds)
    else:
        print(f"Analysing shot boundaries: {source}", flush=True)
        info, shots = detect_shot_ranges(source, args.shot_threshold, args.min_shot_seconds)
    signature = output_signature(source, args, shots)
    print(
        f"Video: {info.width}x{info.height}, {info.fps:.6g} fps, {info.frame_count} frames; "
        f"detected {len(shots)} shot(s).",
        flush=True,
    )
    if not args.force and resumable_output(output, signature, video_like=source):
        print(f"Reuse stabilized video: {output}", flush=True)
        ensure_browser_preview(find_ffmpeg(args.ffmpeg), output)
        return 0
    if args.dry_run:
        for index, (start, end) in enumerate(shots):
            print(f"Shot {index + 1}: frames {start}-{end - 1}", flush=True)
        print(f"Dry run; would write: {output}", flush=True)
        return 0

    ffmpeg = find_ffmpeg(args.ffmpeg)
    streams = probe_streams(ffmpeg, source)
    pix_fmt = "yuv422p10le" if args.encoder == "prores" else stabilization_pix_fmt(streams["pix_fmt"])
    cache_root = ROOT / ".cache" / "stabilize"
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{aid.source_word(source.name)}_", dir=cache_root) as temp_text:
        work_dir = Path(temp_text)
        chunks = [
            stabilize_shot(ffmpeg, source, work_dir, index, start, end, info.fps, args, pix_fmt)
            for index, (start, end) in enumerate(shots)
        ]
        concat_chunks(ffmpeg, source, chunks, output, work_dir, bool(streams["has_audio"]))

    write_signature(output, signature)
    ensure_browser_preview(ffmpeg, output)
    print(f"Wrote stabilized video: {output}", flush=True)
    return 0


def main() -> int:
    return main_with_args(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
