"""Whole-video thumbnail sheets, so scrubbing shot boundaries never waits on ffmpeg.

One sequential decode tiles every frame of a video into small JPEG contact sheets. The
browser then scrubs by moving a background window over an already-loaded sheet. Frame N is
tile N % FRAMES_PER_SHEET of sheet N // FRAMES_PER_SHEET, in decode order, so thumbnails are
frame-exact without any seeking.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import threading
from pathlib import Path

from PIL import Image

from .config import PREVIEW_DIR
from .media import local_tool
from .paths import rel

COLUMNS = 5
ROWS = 5
FRAMES_PER_SHEET = COLUMNS * ROWS
# Sized for the shot cards (roughly 220-450 px wide); one decoded 5x5 sheet is about 8 MB.
TILE_WIDTH = 384
INFO_NAME = "sheets.json"
SHEETS_ROOT = PREVIEW_DIR / "shot_sheets"

_lock = threading.Lock()
_building: set[Path] = set()
_errors: dict[Path, str] = {}


def sheet_dir(source: Path) -> Path:
    stat = source.stat()
    key = hashlib.sha256(f"{source.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()).hexdigest()[:16]
    return SHEETS_ROOT / key


def sheet_status(source: Path) -> dict:
    """Ready sheet geometry, or start building them in the background and say so."""
    directory = sheet_dir(source)
    info_path = directory / INFO_NAME
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        return {"ready": True, "directory": rel(directory), **info}
    with _lock:
        if directory in _errors:
            return {"ready": False, "error": _errors[directory]}
        if directory not in _building:
            _building.add(directory)
            threading.Thread(target=_build, args=(source, directory), daemon=True).start()
    return {"ready": False, "building": True}


def _build(source: Path, directory: Path) -> None:
    partial = directory.with_name(directory.name + ".partial")
    try:
        ffmpeg = local_tool("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("Run install_windows.bat to install local FFmpeg for scrub previews.")
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        # passthrough keeps one output tile per decoded frame; timestamp-based frame
        # duplication or dropping would shift every later tile off its frame number.
        subprocess.run(
            [
                ffmpeg, "-v", "error", "-y", "-i", str(source), "-an", "-sn", "-dn",
                "-fps_mode", "passthrough",
                "-vf", f"scale={TILE_WIDTH}:-2:flags=bilinear,tile={COLUMNS}x{ROWS}",
                "-q:v", "5", "-start_number", "0", str(partial / "sheet_%05d.jpg"),
            ],
            check=True, capture_output=True, text=True,
        )
        sheets = sorted(partial.glob("sheet_*.jpg"))
        if not sheets:
            raise RuntimeError("FFmpeg produced no scrub preview sheets.")
        with Image.open(sheets[0]) as first:
            width, height = first.size
        info = {
            "source": str(source.resolve()),
            "columns": COLUMNS,
            "rows": ROWS,
            "frames_per_sheet": FRAMES_PER_SHEET,
            "tile_width": width // COLUMNS,
            "tile_height": height // ROWS,
            "sheet_count": len(sheets),
        }
        (partial / INFO_NAME).write_text(json.dumps(info), encoding="utf-8")
        partial.replace(directory)
        _prune_older_versions(directory, info["source"])
    except Exception as exc:
        shutil.rmtree(partial, ignore_errors=True)
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        with _lock:
            _errors[directory] = detail or "Could not build scrub previews."
    finally:
        with _lock:
            _building.discard(directory)


def _prune_older_versions(current: Path, source: str) -> None:
    """Drop sheets made from earlier versions of the same video."""
    for info_path in SHEETS_ROOT.glob(f"*/{INFO_NAME}"):
        directory = info_path.parent
        if directory == current:
            continue
        try:
            if json.loads(info_path.read_text(encoding="utf-8")).get("source") == source:
                shutil.rmtree(directory, ignore_errors=True)
        except (OSError, ValueError):
            continue
