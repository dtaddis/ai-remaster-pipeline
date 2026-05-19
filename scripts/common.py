from __future__ import annotations

import hashlib
import math
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def root_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def signature_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sig.json")


def signature_matches(path: Path, signature: dict[str, Any]) -> bool:
    sig = signature_path(path)
    if not path.exists() or not sig.exists():
        return False
    try:
        return json.loads(sig.read_text(encoding="utf-8-sig")) == signature
    except Exception:
        return False


def video_info(path: Path) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, _frame = cap.read()
    cap.release()
    if width <= 0 or height <= 0 or frames <= 0 or not ok:
        raise RuntimeError(f"Video is not readable or has no frames: {path}")
    return {"width": width, "height": height, "fps": fps or 24.0, "frames": frames, "duration": frames / (fps or 24.0)}


def image_info(path: Path) -> dict[str, int]:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise RuntimeError(f"Image is not readable: {path}")
    height, width = image.shape[:2]
    return {"width": int(width), "height": int(height)}


def video_matches(
    path: Path,
    *,
    width: int | None = None,
    height: int | None = None,
    like: Path | None = None,
    duration_tolerance: float = 1.5,
    frame_tolerance: int = 3,
) -> bool:
    try:
        info = video_info(path)
        expected = video_info(like) if like else None
    except Exception:
        return False
    expected_width = width if width is not None else (expected["width"] if expected else None)
    expected_height = height if height is not None else (expected["height"] if expected else None)
    if expected_width is not None and info["width"] != expected_width:
        return False
    if expected_height is not None and info["height"] != expected_height:
        return False
    if expected:
        if abs(info["duration"] - expected["duration"]) > duration_tolerance:
            return False
        if abs(info["frames"] - expected["frames"]) > max(frame_tolerance, math.ceil(expected["fps"] * duration_tolerance)):
            return False
    return True


def image_matches(path: Path, *, like: Path | None = None, width: int | None = None, height: int | None = None) -> bool:
    try:
        info = image_info(path)
        expected = image_info(like) if like else None
    except Exception:
        return False
    expected_width = width if width is not None else (expected["width"] if expected else None)
    expected_height = height if height is not None else (expected["height"] if expected else None)
    return (expected_width is None or info["width"] == expected_width) and (expected_height is None or info["height"] == expected_height)


def resumable_output(path: Path, signature: dict[str, Any], *, video_like: Path | None = None, width: int | None = None, height: int | None = None, image_like: Path | None = None) -> bool:
    if not signature_matches(path, signature):
        return False
    if video_like or ((width is not None or height is not None) and path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}):
        return video_matches(path, like=video_like, width=width, height=height)
    if image_like or width is not None or height is not None:
        return image_matches(path, like=image_like, width=width, height=height)
    return True


def write_signature(path: Path, signature: dict[str, Any]) -> None:
    signature_path(path).write_text(json.dumps(signature, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def format_time(seconds: float) -> str:
    total_millis = int(round(seconds * 1000))
    total = total_millis // 1000
    millis = total_millis % 1000
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if millis:
        return f"{hours:d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def safe_stem(path_text: str) -> str:
    stem = Path(path_text).stem.replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
