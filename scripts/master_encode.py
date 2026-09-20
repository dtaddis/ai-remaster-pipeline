from __future__ import annotations

"""Execute a stage and encode its video products in the selected intermediate master format."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import find_ffmpeg, resolve_path
from intermediate_video import INTERMEDIATE_PROFILES, canonical_profile, codec_args as intermediate_codec_args, container_args

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def codec_args(profile: str) -> list[str]:
    return intermediate_codec_args(profile)


def encode_output(ffmpeg: str, output: Path, profile: str) -> None:
    if not output.is_file() or output.suffix.lower() not in VIDEO_EXTS:
        return
    profile = canonical_profile(profile)
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
    output_container_args = container_args(str(output), profile)
    subprocess.run([
        ffmpeg, "-y", "-i", str(output), "-map", "0:v:0", "-map", "0:a?",
        *codec_args(profile), "-c:a", "copy", *output_container_args, str(temporary),
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
    if not videos:
        return 0
    ffmpeg = find_ffmpeg(args.ffmpeg)
    for output in videos:
        encode_output(ffmpeg, output, args.profile)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply an ARP intermediate master profile.")
    parser.add_argument(
        "--profile",
        choices=[*INTERMEDIATE_PROFILES, "h264_standard", "h264_high", "hevc_high", "hevc_lossless"],
        default="high",
    )
    parser.add_argument("--output", action="append", default=[])
    parser.add_argument("--ffmpeg", default="")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    return args


if __name__ == "__main__":
    raise SystemExit(run(build_parser()))
