from __future__ import annotations

import hashlib
import math
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# The single Qwen Image Edit model used everywhere we run Qwen (colour references, outpaint
# guide frames, and shot-change seed guides). Mirrors ai_remaster_gui.config.QWEN_IMAGE_EDIT_MODEL
# (the GUI and scripts are separate packages, so each keeps its own copy of the value).
QWEN_IMAGE_EDIT_MODEL = "qwen-image-edit-2511-Q4_K_M.gguf"


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def load_local_config() -> dict[str, str]:
    path = ROOT / ".ai_remaster_config.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return {str(key): str(value) for key, value in data.items() if value is not None}


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


def _without_mtime(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_mtime(item)
            for key, item in value.items()
            if key not in {"mtime_ns", "color_reference_previous"}
        }
    if isinstance(value, list):
        return [_without_mtime(item) for item in value]
    return value


def signature_matches(path: Path, signature: dict[str, Any]) -> bool:
    sig = signature_path(path)
    if not path.exists() or not sig.exists():
        return False
    try:
        # mtime_ns is recorded for diagnostics but ignored when matching: size+sha256 already
        # prove content identity, and project load rewrites identical bytes with fresh mtimes.
        return _without_mtime(json.loads(sig.read_text(encoding="utf-8-sig"))) == _without_mtime(signature)
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


def is_windows() -> bool:
    return os.name == "nt"


def find_ffmpeg(explicit: str | None = None) -> str:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    exe = "ffmpeg.exe" if is_windows() else "ffmpeg"
    candidates.extend([ROOT / ".cache" / "tools" / "ffmpeg" / exe, Path("ffmpeg")])
    if is_windows():
        candidates.append(Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"))
    for candidate in candidates:
        try:
            subprocess.run([str(candidate), "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return str(candidate)
        except Exception:
            continue
    raise FileNotFoundError("ffmpeg was not found. Run install_windows.bat/install script again or pass --ffmpeg.")


def copy_to_comfy_input(path: Path, comfy_dir: Path, subfolder: str) -> str:
    target_dir = comfy_dir / "input" / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / path.name
    # Same-named inputs from other chunk caches/projects collide here, so a matching size
    # alone is not proof the cached copy holds this file's content.
    if (
        not target.exists()
        or target.stat().st_size != path.stat().st_size
        or file_fingerprint(target)["sha256"] != file_fingerprint(path)["sha256"]
    ):
        shutil.copy2(path, target)
    return str(Path(subfolder) / target.name).replace("\\", "/")


def newest_output(
    files: list[Path],
    suffixes: set[str] | None = None,
    label: str = "output file",
    exist_attempts: int = 20,
    exist_delay: float = 0.25,
) -> Path:
    candidates = files
    if suffixes:
        wanted = {suffix.lower() for suffix in suffixes}
        candidates = [path for path in files if path.suffix.lower() in wanted]
    if not candidates:
        raise RuntimeError(f"ComfyUI completed but did not report a {label}.")
    for attempt in range(max(1, exist_attempts)):
        existing = [path for path in candidates if path.exists()]
        if existing:
            return max(existing, key=lambda path: path.stat().st_mtime_ns)
        if attempt < exist_attempts - 1:
            time.sleep(exist_delay)
    raise RuntimeError(f"ComfyUI reported {label}s, but none exist on disk after waiting: {candidates}")


def replace_with_retry(source: Path, target: Path, label: str | None = None, attempts: int = 20, delay: float = 0.5) -> None:
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt >= attempts - 1:
                raise
            if label:
                print(f"{label} is locked by another process; retrying in {delay:g}s ({attempt + 1}/{attempts})...", flush=True)
            time.sleep(delay)


def replace_unless_identical(partial: Path, target: Path, label: str | None = None) -> None:
    """Promote partial to target, but keep the existing target when the bytes are identical
    so resume signatures that fingerprint it stay valid."""
    if target.exists():
        try:
            if file_fingerprint(partial)["sha256"] == file_fingerprint(target)["sha256"]:
                partial.unlink()
                return
        except OSError:
            pass
    replace_with_retry(partial, target, label)


def split_sidecar_path(target: Path) -> Path:
    return target.with_suffix(target.suffix + ".src.json")


def split_matches_source(target: Path, source_fingerprint: dict[str, Any]) -> bool:
    """True when target was split from a source with this content (size+sha256). Chunk caches
    are keyed by file name, so a source re-rendered under the same name must invalidate its
    split files or stale frames get reused by the per-chunk resume signatures."""
    if not target.exists():
        return False
    try:
        stored = json.loads(split_sidecar_path(target).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False
    return stored.get("size") == source_fingerprint.get("size") and stored.get("sha256") == source_fingerprint.get("sha256")


def write_split_sidecar(target: Path, source: Path, source_fingerprint: dict[str, Any]) -> None:
    payload = {
        "source": root_relative(source),
        "size": source_fingerprint["size"],
        "sha256": source_fingerprint["sha256"],
    }
    split_sidecar_path(target).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
