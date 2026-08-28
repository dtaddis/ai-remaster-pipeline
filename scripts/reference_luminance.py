from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

from common import resolve_path, root_relative


QUANTILES = (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
MAX_LUMA_SHIFT = 64.0


def parse_time(value: str) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    parts = text.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(text)
    except ValueError:
        return 0.0


def _frame_number(row: dict[str, str], key: str, time_key: str, fps: float) -> int:
    text = str(row.get(key, "") or "").strip()
    if text:
        try:
            return max(0, int(float(text)))
        except ValueError:
            pass
    return max(0, int(round(parse_time(row.get(time_key, "")) * fps)))


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if not line.startswith("#"):
                return list(csv.DictReader([line, *handle.readlines()]))
    return []


def _monotonic_curve(source_values: list[float], target_values: list[float], strength: float) -> list[list[float]]:
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    amount = max(0.0, min(1.0, float(strength)))
    for source, target in zip(source_values, target_values):
        source = max(0.0, min(255.0, float(source)))
        target = max(source - MAX_LUMA_SHIFT, min(source + MAX_LUMA_SHIFT, float(target)))
        matched = source + (target - source) * amount
        if source <= points[-1][0] + 0.5:
            if matched > points[-1][1]:
                points[-1] = (points[-1][0], matched)
            continue
        points.append((source, matched))
    points.append((255.0, 255.0))

    # FFmpeg curves must be monotonic. A generated reference can contain small local
    # histogram inversions, so clamp them rather than allowing a solarised tonal map.
    monotonic: list[list[float]] = []
    previous = 0.0
    for source, target in points:
        target = max(previous, min(255.0, target))
        monotonic.append([round(source / 255.0, 6), round(target / 255.0, 6)])
        previous = target
    monotonic[-1] = [1.0, 1.0]
    return monotonic


@lru_cache(maxsize=256)
def _curve_for_pair_cached(
    source_text: str,
    source_mtime_ns: int,
    color_text: str,
    color_mtime_ns: int,
    strength: float,
) -> tuple[tuple[float, float], ...]:
    del source_mtime_ns, color_mtime_ns  # cache-key-only values
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Reference luminance matching requires Pillow and NumPy.") from exc

    with Image.open(source_text) as source_image, Image.open(color_text) as color_image:
        source = source_image.convert("RGB")
        color = color_image.convert("RGB")
        max_edge = max(source.size)
        if max_edge > 1024:
            scale = 1024.0 / max_edge
            size = (max(2, round(source.width * scale)), max(2, round(source.height * scale)))
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            source = source.resize(size, resampling)
        if color.size != source.size:
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            color = color.resize(source.size, resampling)

        source_rgb = np.asarray(source, dtype=np.float32)
        color_rgb = np.asarray(color, dtype=np.float32)
        source_y = source_rgb[..., 0] * 0.299 + source_rgb[..., 1] * 0.587 + source_rgb[..., 2] * 0.114
        color_y = color_rgb[..., 0] * 0.299 + color_rgb[..., 1] * 0.587 + color_rgb[..., 2] * 0.114

        # Near-black/white pixels are usually presentation bars, borders, or clipped
        # damage. Excluding only those pixels keeps genuine shadows in the statistics
        # while preventing a 4:3 frame on a 16:9 canvas from dominating the curve.
        mask = (source_y > 4.0) & (source_y < 251.0)
        if int(mask.sum()) < 1024:
            mask = np.ones(source_y.shape, dtype=bool)
        source_samples = source_y[mask]
        color_samples = color_y[mask]
        source_values = np.quantile(source_samples, QUANTILES).tolist()
        target_values = np.quantile(color_samples, QUANTILES).tolist()
        curve = _monotonic_curve(source_values, target_values, strength)
        return tuple((point[0], point[1]) for point in curve)


def curve_for_pair(source: Path, color: Path, strength_percent: float = 70.0) -> list[list[float]]:
    source = source.resolve()
    color = color.resolve()
    if not source.is_file() or not color.is_file():
        return [[0.0, 0.0], [1.0, 1.0]]
    strength = max(0.0, min(100.0, float(strength_percent))) / 100.0
    result = _curve_for_pair_cached(
        str(source), source.stat().st_mtime_ns, str(color), color.stat().st_mtime_ns, round(strength, 4)
    )
    return [[x, y] for x, y in result]


def identity_curve(curve: list[list[float]], tolerance: float = 0.002) -> bool:
    return all(abs(float(x) - float(y)) <= tolerance for x, y in curve)


def ffmpeg_curve(curve: list[list[float]]) -> str:
    return " ".join(f"{float(x):.6f}/{float(y):.6f}" for x, y in curve)


def reference_luminance_plan(manifest: Path, fps: float, strength_percent: float = 70.0) -> list[dict[str, Any]]:
    """Build full-timeline, shot-constant tone curves from B&W/colour reference pairs.

    A constant curve per shot is intentional: matching each video frame independently would
    recreate exposure flicker. Boundaries come from reviewed manifest frame numbers and the
    final shot uses the manifest's reviewed end frame when present, preventing an
    overlay stream with a repeated tail frame from extending the composite.
    """
    manifest = manifest.resolve()
    if not manifest.is_file():
        return []
    rows = _manifest_rows(manifest)
    if not rows:
        return []

    shots: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        start = _frame_number(row, "start_frame", "start", fps)
        source = resolve_path(row.get("source_reference", ""))
        color = resolve_path(row.get("color_reference", ""))
        matched = source.is_file() and color.is_file()
        curve = curve_for_pair(source, color, strength_percent) if matched else [[0.0, 0.0], [1.0, 1.0]]
        shots.append(
            {
                "shot": index + 1,
                "start_frame": start,
                "manifest_end_frame": _frame_number(row, "end_frame", "end", fps),
                "source_reference": root_relative(source) if source.is_file() else row.get("source_reference", ""),
                "color_reference": root_relative(color) if color.is_file() else row.get("color_reference", ""),
                "matched": matched and not identity_curve(curve),
                "curve": curve,
            }
        )

    shots.sort(key=lambda item: int(item["start_frame"]))
    unique: list[dict[str, Any]] = []
    for shot in shots:
        if unique and int(shot["start_frame"]) == int(unique[-1]["start_frame"]):
            unique[-1] = shot
        else:
            unique.append(shot)
    if int(unique[0]["start_frame"]) > 0:
        unique.insert(
            0,
            {
                "shot": 0,
                "start_frame": 0,
                "manifest_end_frame": int(unique[0]["start_frame"]),
                "source_reference": "",
                "color_reference": "",
                "matched": False,
                "curve": [[0.0, 0.0], [1.0, 1.0]],
            },
        )
    for index, shot in enumerate(unique):
        if index + 1 < len(unique):
            shot["end_frame"] = int(unique[index + 1]["start_frame"])
        else:
            manifest_end = int(shot.get("manifest_end_frame", 0) or 0)
            shot["end_frame"] = manifest_end if manifest_end > int(shot["start_frame"]) else None
        shot.pop("manifest_end_frame", None)
        shot["start_seconds"] = round(int(shot["start_frame"]) / max(0.001, fps), 6)
        shot["end_seconds"] = (
            round(int(shot["end_frame"]) / max(0.001, fps), 6) if shot["end_frame"] is not None else None
        )
    return unique
