from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Callable

from .config import IMAGE_EXTS, ROOT
from .paths import rel, resolve, safe_stem

SEQUENCE_DIR = ROOT / "intermediate" / "source_sequences"
DEFAULT_SEQUENCE_FPS = 24.0
MIN_SEQUENCE_FPS = 1.0
MAX_SEQUENCE_FPS = 240.0


def natural_path_key(path: Path) -> tuple:
    """Sort numbered stills as frame2, frame10 instead of frame10, frame2."""
    return tuple((0, int(part)) if part.isdigit() else (1, part.casefold()) for part in re.split(r"(\d+)", path.name))


def normalize_image_paths(paths: list[str | Path]) -> list[Path]:
    images = [resolve(str(path)) for path in paths]
    if not images:
        raise RuntimeError("Select at least one image.")
    unsupported = [path for path in images if path.suffix.lower() not in IMAGE_EXTS]
    if unsupported:
        raise RuntimeError(f"Unsupported image type: {unsupported[0].suffix or unsupported[0].name}")
    missing = [path for path in images if not path.is_file()]
    if missing:
        raise RuntimeError(f"Selected image was not found: {missing[0]}")
    return sorted(dict.fromkeys(images), key=natural_path_key)


def parse_image_paths(value: str | list[str] | None) -> list[Path]:
    if isinstance(value, list):
        raw = value
    elif value:
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
        raw = decoded if isinstance(decoded, list) else []
    else:
        raw = []
    return [resolve(str(item)) for item in raw if str(item).strip()]


def serialize_image_paths(paths: list[Path]) -> str:
    return json.dumps([rel(path) for path in paths], ensure_ascii=False)


def sequence_fps(value: str | float | int | None) -> float:
    try:
        fps = float(value or DEFAULT_SEQUENCE_FPS)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Image-sequence frame rate must be a number.") from exc
    if not MIN_SEQUENCE_FPS <= fps <= MAX_SEQUENCE_FPS:
        raise RuntimeError(f"Image-sequence frame rate must be between {MIN_SEQUENCE_FPS:g} and {MAX_SEQUENCE_FPS:g} fps.")
    return fps


def sequence_identity(paths: list[Path], fps: float) -> dict:
    return {
        "version": 1,
        "fps": round(fps, 6),
        "images": [
            {"path": str(path.resolve()), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in paths
        ],
    }


def sequence_outputs(paths: list[Path], fps: float) -> tuple[Path, Path, Path]:
    identity = sequence_identity(paths, fps)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    stem = safe_stem(paths[0].stem) or "image_sequence"
    base = SEQUENCE_DIR / f"{stem}_sequence_{digest}"
    return base.with_suffix(".mkv"), Path(str(base) + "_preview.mp4"), Path(str(base) + ".json")


def ffconcat_quote(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "'\\''")


def ffconcat_text(paths: list[Path], fps: float) -> str:
    duration = 1.0 / fps
    lines = ["ffconcat version 1.0"]
    for path in paths:
        lines.extend((f"file '{ffconcat_quote(path)}'", f"duration {duration:.12f}"))
    # The concat demuxer ignores the final duration unless the last still is repeated.
    lines.append(f"file '{ffconcat_quote(paths[-1])}'")
    return "\n".join(lines) + "\n"


def prepare_image_sequence(
    paths: list[str | Path],
    fps_value: str | float | int | None,
    ffmpeg: str,
    progress: Callable[[int, str], None] | None = None,
) -> dict:
    images = normalize_image_paths(paths)
    decoder_types = {".jpg" if path.suffix.lower() in {".jpg", ".jpeg"} else path.suffix.lower() for path in images}
    if len(decoder_types) != 1:
        raise RuntimeError("All files in an image sequence must use the same image format.")
    fps = sequence_fps(fps_value)
    master, preview, sidecar = sequence_outputs(images, fps)
    if master.is_file() and preview.is_file() and sidecar.is_file():
        if progress:
            progress(100, f"Reusing converted image sequence ({len(images)} frames)")
        return sequence_result(images, fps, master, preview)

    master.parent.mkdir(parents=True, exist_ok=True)
    concat = master.with_suffix(".ffconcat")
    master_partial = master.with_name(master.stem + ".partial.mkv")
    preview_partial = preview.with_name(preview.stem + ".partial.mp4")
    concat.write_text(ffconcat_text(images, fps), encoding="utf-8")
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats_period",
        "0.5",
        "-progress",
        "pipe:1",
        "-nostats",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat),
        "-map",
        "0:v:0",
        "-an",
        "-r",
        f"{fps:.6f}",
        "-frames:v",
        str(len(images)),
        "-vf",
        "format=gbrp10le",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-coder",
        "1",
        "-context",
        "1",
        str(master_partial),
        "-map",
        "0:v:0",
        "-an",
        "-r",
        f"{fps:.6f}",
        "-frames:v",
        str(len(images)),
        "-vf",
        "format=yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "14",
        "-preset",
        "medium",
        "-movflags",
        "+faststart",
        str(preview_partial),
    ]
    if progress:
        progress(0, f"Converting image sequence (0/{len(images)} frames)")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    output: list[str] = []
    last_frame = 0
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.strip()
        if line.startswith("frame="):
            try:
                last_frame = max(last_frame, int(line.split("=", 1)[1].strip()))
            except ValueError:
                pass
            if progress:
                percent = min(99, int(last_frame * 100 / max(1, len(images))))
                progress(percent, f"Converting image sequence ({min(last_frame, len(images))}/{len(images)} frames)")
        elif line and not re.match(r"^[a-z_]+=", line, re.IGNORECASE):
            output.append(line)
    process.stdout.close()
    returncode = process.wait()
    if returncode != 0:
        for partial in (master_partial, preview_partial):
            partial.unlink(missing_ok=True)
        raise RuntimeError(("\n".join(output) or "FFmpeg could not import the image sequence.").strip())
    master_partial.replace(master)
    preview_partial.replace(preview)
    sidecar.write_text(json.dumps(sequence_identity(images, fps), indent=2) + "\n", encoding="utf-8")
    concat.unlink(missing_ok=True)
    if progress:
        progress(100, f"Converted image sequence ({len(images)}/{len(images)} frames)")
    return sequence_result(images, fps, master, preview)


def sequence_result(images: list[Path], fps: float, master: Path, preview: Path) -> dict:
    extensions = sorted({path.suffix.lower().lstrip(".").upper() for path in images})
    return {
        "source": rel(master),
        "playback": rel(preview),
        "images": [rel(path) for path in images],
        "images_json": serialize_image_paths(images),
        "count": len(images),
        "fps": f"{fps:g}",
        "formats": extensions,
        "first": rel(images[0]),
        "last": rel(images[-1]),
    }


def sequence_state(settings: dict) -> dict:
    values = settings.get("global", {})
    paths = parse_image_paths(values.get("source_images", ""))
    if not paths:
        return {}
    existing = [path for path in paths if path.is_file()]
    fps = sequence_fps(values.get("source_sequence_fps", DEFAULT_SEQUENCE_FPS))
    formats = sorted({path.suffix.lower().lstrip(".").upper() for path in paths})
    return {
        "count": len(paths),
        "available": len(existing),
        "fps": f"{fps:g}",
        "formats": formats,
        "first": rel(paths[0]),
        "last": rel(paths[-1]),
        "playback": values.get("source_sequence_preview", ""),
    }
