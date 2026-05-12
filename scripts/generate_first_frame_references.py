#!/usr/bin/env python3
"""Generate first-frame color references and a ComfyUI colorization manifest."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REF_ROOT = ROOT / "input" / "references" / "generated_first_frames"
DEFAULT_MODEL = "gpt-image-1.5"
DEFAULT_QUALITY = "medium"
DEFAULT_SIZE = "auto"
DEFAULT_PROMPT_SUFFIX = (
    "Colorize video. Modern clean period color, full-spectrum color across the whole frame, "
    "natural skin tones, warm tungsten interiors where appropriate, blue-grey mist and shadows "
    "where appropriate, no Technicolor, no sepia, no hand-tinted look, no monochrome background."
)
IMAGE_PROMPT = (
    "Colourise and restore this exact black-and-white film frame.\n"
    "Preserve composition, faces, clothing, set design, lighting, contrast, grain, and framing.\n"
    "Make it look like a 1939 production photographed today with a modern Sony A7IV or cinema camera and vintage lenses.\n"
    "Use modern clean natural color, full-spectrum color across the whole frame, realistic skin tones, "
    "warm tungsten interiors where appropriate, blue-grey mist and shadows where appropriate.\n"
    "Do not create a Technicolor look, hand-tinted look, sepia, WWII archive colorization, neon color, "
    "modern objects, or changed text."
)


@dataclass
class VideoInfo:
    width: int
    height: int
    duration: float


@dataclass
class Segment:
    index: int
    start: float
    end: float
    reference_path: Path
    reference_rel: str


def format_time(seconds: float, sep: str = ":") -> str:
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:d}{sep}{minutes:02d}{sep}{secs:02d}"
    return f"{minutes:d}{sep}{secs:02d}"


def format_stamp(seconds: float) -> str:
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}.{minutes:02d}.{secs:02d}"


def safe_stem(path: str) -> str:
    return Path(path).stem.replace(" ", "_")


def run(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return completed.stdout


def find_ffmpeg(executable: str) -> str:
    found = shutil.which(executable)
    if found:
        return found
    common = Path(r"C:\Program Files\ffmpeg\bin") / executable
    if common.exists():
        return str(common)
    raise FileNotFoundError(f"{executable} not found on PATH")


def ffprobe_video(path: Path, ffprobe: str) -> VideoInfo:
    raw = run([
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,duration",
        "-of",
        "json",
        str(path),
    ])
    data = json.loads(raw)
    stream = data["streams"][0]
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        duration=float(stream["duration"]),
    )


def build_segments(args: argparse.Namespace, info: VideoInfo) -> list[Segment]:
    ref_dir = args.reference_root / safe_stem(args.source_video)
    segments: list[Segment] = []
    start = 0.0
    index = 0
    while start < info.duration - 1e-6:
        end = min(start + args.segment_seconds, info.duration)
        ref_name = f"ref_{index:04d}_{format_stamp(start)}.png"
        ref_path = ref_dir / ref_name
        ref_rel = ref_path.relative_to(ROOT / "input").as_posix()
        segments.append(Segment(index=index, start=start, end=end, reference_path=ref_path, reference_rel=ref_rel))
        index += 1
        start += args.segment_seconds
        if args.limit is not None and index >= args.limit:
            break
    return segments


def extract_first_frame(source: Path, segment: Segment, destination: Path, ffmpeg: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    run([
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-ss",
        f"{segment.start:.6f}",
        "-frames:v",
        "1",
        str(destination),
    ])


def multipart_body(fields: dict[str, str], files: dict[str, Path]) -> tuple[bytes, str]:
    boundary = f"----codex-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")
    for name, path in files.items():
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n".encode("utf-8")
        )
        chunks.append(path.read_bytes())
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), boundary


def call_image_edit_api(args: argparse.Namespace, frame_path: Path) -> bytes:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    fields = {
        "model": args.model,
        "prompt": IMAGE_PROMPT,
        "quality": args.quality,
        "size": args.size,
        "output_format": "png",
    }
    body, boundary = multipart_body(fields, {"image[]": frame_path})
    request = urllib.request.Request(
        "https://api.openai.com/v1/images/edits",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=args.api_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI image edit failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI image edit connection failed: {exc.reason}") from exc

    item = (payload.get("data") or [{}])[0]
    if "b64_json" in item:
        return base64.b64decode(item["b64_json"])
    if "url" in item:
        with urllib.request.urlopen(item["url"], timeout=args.api_timeout) as response:
            return response.read()
    raise RuntimeError(f"OpenAI image edit response did not include b64_json or url: {payload}")


def resize_to_source(image_bytes: bytes, destination: Path, width: int, height: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = Path(tmp.name)
    try:
        with Image.open(tmp_path) as image:
            image = image.convert("RGB")
            src_w, src_h = image.size
            scale = max(width / src_w, height / src_h)
            resized = image.resize((round(src_w * scale), round(src_h * scale)), Image.Resampling.LANCZOS)
            left = max(0, (resized.width - width) // 2)
            top = max(0, (resized.height - height) // 2)
            cropped = resized.crop((left, top, left + width, top + height))
            cropped.save(destination, "PNG")
    finally:
        tmp_path.unlink(missing_ok=True)


def manifest_path_for(args: argparse.Namespace) -> Path:
    if args.manifest:
        return args.manifest
    return ROOT / "manifests" / "colorize" / f"colorize_auto_{safe_stem(args.source_video)}.csv"


def write_manifest(path: Path, source_video: str, segments: list[Segment]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# source_video={source_video}\n")
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["enabled", "end", "reference", "prompt_suffix"])
        for segment in segments:
            writer.writerow(["true", format_time(segment.end), segment.reference_rel, DEFAULT_PROMPT_SUFFIX])
    os.replace(tmp, path)


def print_dry_run(args: argparse.Namespace, info: VideoInfo, segments: list[Segment], manifest: Path) -> None:
    print(f"Source: {args.source_video}")
    print(f"Size: {info.width}x{info.height}, duration={info.duration:.3f}s")
    print(f"Manifest: {manifest}")
    for segment in segments:
        status = "exists" if segment.reference_path.exists() else "missing"
        print(
            f"{segment.index:04d} {format_time(segment.start)}-{format_time(segment.end)} "
            f"{segment.reference_rel} [{status}]"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate first-frame color references and an auto colorization manifest.")
    parser.add_argument("--source-video", required=True, help="Video path relative to this repository input folder.")
    parser.add_argument("--segment-seconds", type=float, default=20.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quality", default=DEFAULT_QUALITY)
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REF_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--api-timeout", type=int, default=180)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.segment_seconds <= 0:
        raise RuntimeError("--segment-seconds must be greater than zero")
    if args.limit is not None and args.limit <= 0:
        raise RuntimeError("--limit must be greater than zero")

    ffmpeg = find_ffmpeg(args.ffmpeg)
    ffprobe = find_ffmpeg(args.ffprobe)
    source = ROOT / "input" / args.source_video
    if not source.exists():
        raise FileNotFoundError(f"source video is not in this repository input folder: {source}")

    args.reference_root = args.reference_root.resolve()
    manifest = manifest_path_for(args).resolve()
    info = ffprobe_video(source, ffprobe)
    segments = build_segments(args, info)
    if args.dry_run:
        print_dry_run(args, info, segments, manifest)
        return 0

    with tempfile.TemporaryDirectory(prefix="remaster_refs_") as temp_dir:
        temp_root = Path(temp_dir)
        for segment in segments:
            if segment.reference_path.exists():
                print(f"Reuse reference {segment.index:04d}: {segment.reference_path}")
                continue
            frame_path = temp_root / f"frame_{segment.index:04d}.png"
            print(f"Extract frame {segment.index:04d} @ {format_time(segment.start)}")
            extract_first_frame(source, segment, frame_path, ffmpeg)
            print(f"Colorize reference {segment.index:04d} with {args.model} ({args.quality})")
            image_bytes = call_image_edit_api(args, frame_path)
            resize_to_source(image_bytes, segment.reference_path, info.width, info.height)
            print(f"Wrote reference: {segment.reference_path}")
            time.sleep(0.2)

    write_manifest(manifest, args.source_video, segments)
    print(f"Wrote manifest: {manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)



