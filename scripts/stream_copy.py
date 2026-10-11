"""Join or trim working-codec clips without re-encoding them.

ARP's intermediates are already in the working codec, so joining clips end to end, or keeping the
first frames of one, only needs the packets copied. A re-encode there costs a full encode pass and
another generation of loss for nothing. Every helper here verifies its result (same stream format
in, exact packet count out) and returns False when it cannot guarantee it; callers then fall back
to their frame-accurate re-encode.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from common import ffprobe_for


def count_video_packets(ffmpeg: str, path: Path) -> int:
    """Exact frame count of a video, read from the container without decoding."""
    result = subprocess.run(
        [ffprobe_for(ffmpeg), "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    try:
        return int(result.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return -1


def stream_format(ffmpeg: str, path: Path) -> tuple:
    """What two clips must share for their packets to be concatenated."""
    result = subprocess.run(
        [ffprobe_for(ffmpeg), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,profile,width,height,pix_fmt,sample_aspect_ratio,r_frame_rate", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(result.stdout or "{}").get("streams") or [{}]
    stream = streams[0]
    return tuple(str(stream.get(key, "")) for key in ("codec_name", "profile", "width", "height", "pix_fmt", "sample_aspect_ratio", "r_frame_rate"))


def _verified(ffmpeg: str, output: Path, expected_frames: int, what: str) -> bool:
    frames = count_video_packets(ffmpeg, output)
    if frames == expected_frames:
        return True
    print(f"Stream-copied {what} has {frames} frame(s), expected {expected_frames}; re-encoding instead.", flush=True)
    output.unlink(missing_ok=True)
    return False


def copy_join(ffmpeg: str, clips: list[Path], output: Path, expected_frames: int, container: list[str] | None = None) -> bool:
    """Concatenate clips' video packets into output (no audio). False if they cannot be joined as-is."""
    if not clips:
        return False
    if len({stream_format(ffmpeg, clip) for clip in clips}) != 1:
        print("Clips differ in format; re-encoding the join.", flush=True)
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="arp_copy_join_") as tmp_text:
        listing = Path(tmp_text) / "clips.txt"
        listing.write_text("".join("file '" + clip.resolve().as_posix().replace("'", "'\\''") + "'\n" for clip in clips), encoding="utf-8")
        completed = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-map", "0:v:0", "-an", "-c", "copy", *(container or ["-f", "matroska"]), str(output)],
        )
    if completed.returncode != 0:
        output.unlink(missing_ok=True)
        return False
    return _verified(ffmpeg, output, expected_frames, f"join of {len(clips)} clip(s)")


def copy_head(ffmpeg: str, clip: Path, output: Path, frames: int, container: list[str] | None = None) -> bool:
    """The first frames of clip, copied (a clip starts on a keyframe, so only its end is cut)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", str(clip), "-map", "0:v:0", "-an", "-c", "copy", "-frames:v", str(frames), *(container or ["-f", "matroska"]), str(output)],
    )
    if completed.returncode != 0:
        output.unlink(missing_ok=True)
        return False
    return _verified(ffmpeg, output, frames, f"first {frames} frame(s) of {clip.name}")
