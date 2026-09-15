from __future__ import annotations

"""Execute a stage and encode its video products in the selected intermediate master format."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import find_ffmpeg, resolve_path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def codec_args(profile: str) -> list[str]:
    if profile == "h264_high":
        return ["-c:v", "libx264", "-crf", "10", "-preset", "slow", "-pix_fmt", "yuv420p"]
    if profile == "hevc_high":
        return ["-c:v", "libx265", "-crf", "12", "-preset", "slow", "-pix_fmt", "yuv420p10le"]
    if profile == "hevc_lossless":
        return ["-c:v", "libx265", "-preset", "medium", "-x265-params", "lossless=1", "-pix_fmt", "yuv444p10le"]
    return ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]


def encode_output(ffmpeg: str, output: Path, profile: str) -> None:
    if not output.is_file() or output.suffix.lower() not in VIDEO_EXTS:
        return
    if profile == "h264_standard":
        return
    marker = Path(str(output) + ".master.json")
    try:
        previous = json.loads(marker.read_text(encoding="utf-8"))
        stat = output.stat()
        if previous == {"profile": profile, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
            print(f"Reuse {profile} intermediate master: {output}", flush=True)
            return
    except (OSError, json.JSONDecodeError):
        pass
    temporary = output.with_name(output.stem + ".master-partial" + output.suffix)
    print(f"Encoding {profile} intermediate master: {output.name}", flush=True)
    container_args: list[str] = []
    if output.suffix.lower() in {".mp4", ".m4v", ".mov"}:
        if profile.startswith("hevc"):
            container_args.extend(["-tag:v", "hvc1"])
        container_args.extend(["-movflags", "+faststart"])
    subprocess.run([
        ffmpeg, "-y", "-i", str(output), "-map", "0:v:0", "-map", "0:a?",
        *codec_args(profile), "-c:a", "copy", *container_args, str(temporary),
    ], check=True)
    temporary.replace(output)
    stat = output.stat()
    marker.write_text(json.dumps({"profile": profile, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    if not args.command:
        raise RuntimeError("No stage command supplied.")
    code = subprocess.run(args.command).returncode
    if code:
        return code
    videos = [resolve_path(text) for text in args.output]
    videos = [path for path in videos if path.is_file() and path.suffix.lower() in VIDEO_EXTS]
    if not videos or args.profile == "h264_standard":
        return 0
    ffmpeg = find_ffmpeg(args.ffmpeg)
    for output in videos:
        encode_output(ffmpeg, output, args.profile)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply an ARP intermediate master profile.")
    parser.add_argument("--profile", choices=["h264_standard", "h264_high", "hevc_high", "hevc_lossless"], default="hevc_high")
    parser.add_argument("--output", action="append", default=[])
    parser.add_argument("--ffmpeg", default="")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    return args


if __name__ == "__main__":
    raise SystemExit(run(build_parser()))
