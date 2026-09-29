"""Per-frame outpaint masks: single-frame patches layered on top of the global custom mask.

The global custom mask covers what is wrong on every frame (sprocket holes, edge damage).
Frame masks cover what is wrong on a few frames only: light leaks, manufacturer edge
lettering, dirt and scratches. Each masked frame is one binary PNG at the outpaint working
canvas size, and they are kept together in one zip so a project, a cloud upload and a
cache fingerprint all deal with a single file. Frames with nothing painted have no member.

Frame numbers are 0-based frames of the video Outpainting consumes, at its own frame rate.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Iterable

MEMBER_PATTERN = re.compile(r"^f(\d{6})\.png$")


def member_name(frame: int) -> str:
    return f"f{int(frame):06d}.png"


def frame_of(name: str) -> int | None:
    match = MEMBER_PATTERN.match(name)
    return int(match.group(1)) if match else None


def _members(path: Path) -> dict[int, zipfile.ZipInfo]:
    if not path or not Path(path).is_file():
        return {}
    with zipfile.ZipFile(path) as archive:
        return {frame: info for info in archive.infolist() if (frame := frame_of(info.filename)) is not None}


def frame_indices(path: Path) -> list[int]:
    return sorted(_members(path))


def read_frames(path: Path, frames: Iterable[int] | None = None) -> dict[int, bytes]:
    """Return the PNG bytes of each masked frame, optionally limited to `frames`."""
    if not path or not Path(path).is_file():
        return {}
    wanted = None if frames is None else set(int(frame) for frame in frames)
    result: dict[int, bytes] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            frame = frame_of(info.filename)
            if frame is not None and (wanted is None or frame in wanted):
                result[frame] = archive.read(info)
    return result


def read_frame(path: Path, frame: int) -> bytes | None:
    return read_frames(path, [frame]).get(int(frame))


def write_frames(path: Path, updates: dict[int, bytes | None]) -> None:
    """Replace, add or (with None) remove masked frames, rewriting the zip atomically.

    Zip members cannot be replaced in place. The PNGs are small and stored rather than
    deflated, so rewriting is a plain byte copy even for a long clip. An archive left
    with no frames is removed, so "has frame masks" is simply "the file exists".
    """
    path = Path(path)
    kept = read_frames(path)
    for frame, png in updates.items():
        if png:
            kept[int(frame)] = png
        else:
            kept.pop(int(frame), None)
    if not kept:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as archive:
        for frame in sorted(kept):
            archive.writestr(member_name(frame), kept[frame])
    partial.replace(path)


def digest(path: Path, first: int | None = None, end: int | None = None) -> str | None:
    """Content digest of the masked frames in [first, end), or None when there are none.

    Chunk signatures use a range so editing a patch re-renders only the chunks it touches.
    """
    members = _members(path)
    selected = [
        (frame, info) for frame, info in sorted(members.items())
        if (first is None or frame >= first) and (end is None or frame < end)
    ]
    if not selected:
        return None
    hasher = hashlib.sha1()
    for frame, info in selected:
        hasher.update(f"{frame}:{info.CRC:08x}:{info.file_size};".encode("ascii"))
    return hasher.hexdigest()


def decode_mask(png: bytes, width: int, height: int):
    """Decode one stored frame to a 0/255 uint8 array at the given canvas size.

    Uses the same reading rules as the global custom mask: the alpha channel when there is
    one, otherwise gray, with anything at 16 or above counting as masked.
    """
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("Could not decode a frame mask PNG.")
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, 3]
    elif image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(image >= 16, 255, 0).astype(np.uint8)


def source_frames_for(generation_frame: int, generation_fps: float, source_fps: float) -> range:
    """Source frames shown during one generation frame's display interval.

    LTX 2.5 can generate at a different frame rate from the source. A patch on one source
    frame must still reach every generated frame that overlaps it in time, so a generated
    frame takes the union of the source frames its interval covers.
    """
    if not generation_fps or not source_fps or abs(generation_fps - source_fps) < 1e-6:
        return range(int(generation_frame), int(generation_frame) + 1)
    ratio = float(source_fps) / float(generation_fps)
    first = math.floor(generation_frame * ratio + 1e-6)
    end = max(first + 1, math.ceil((generation_frame + 1) * ratio - 1e-6))
    return range(first, end)
