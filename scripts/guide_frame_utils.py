from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any


def guide_output_size_for_prepared(prepared_canvas: Path, canvas_w: int, canvas_h: int) -> tuple[int, int]:
    """Return the guide image size to store/use after Qwen.

    Guides should stay on the same model-safe canvas as the prepared video. LTX expects
    factor-of-32-friendly dimensions, so a 480p source-height workflow uses 864x480 and the
    720p delivery workflow keeps its established 1280x704 work canvas.
    """
    return canvas_w, canvas_h


def resize_frame_for_qwen(prepared_frame: Any, prepared_canvas: Path) -> Any:
    """Resize a prepared OpenCV frame to the guide/Qwen size for this canvas."""
    height, width = prepared_frame.shape[:2]
    guide_w, guide_h = guide_output_size_for_prepared(prepared_canvas, width, height)
    if (guide_w, guide_h) == (width, height):
        return prepared_frame
    import cv2

    return cv2.resize(prepared_frame, (guide_w, guide_h), interpolation=cv2.INTER_LANCZOS4)


def _edge_bbox_from_rgb_pixels(width: int, height: int, get_pixel, threshold: int = 4) -> tuple[int, int, int, int] | None:
    xs: list[int] = []
    ys: list[int] = []
    for y in range(height):
        for x in range(width):
            r, g, b = get_pixel(x, y)
            if r > threshold or g > threshold or b > threshold:
                xs.append(x)
                ys.append(y)
    if not xs or not ys:
        return None
    return min(xs), max(xs), min(ys), max(ys)


DEFAULT_EDGE_MASK_OVERLAP_PX = 10
DEFAULT_EDGE_MASK_FEATHER_PX = 10
DEFAULT_EDGE_MASK_WOBBLE_PX = 12


def _mask_seed(width: int, height: int, bbox: tuple[int, int, int, int] | None, target: Path) -> int:
    text = f"{width}x{height}|{bbox}|{target.as_posix()}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _smooth_wobble(length: int, amplitude: int, rng: random.Random) -> list[int]:
    if length <= 0 or amplitude <= 0:
        return [0] * max(0, length)
    step = 18
    controls = max(3, (length + step - 1) // step + 2)
    values = [rng.randint(-amplitude, amplitude) for _ in range(controls)]
    wobble: list[int] = []
    for i in range(length):
        pos = i / step
        left = min(controls - 2, int(pos))
        frac = pos - left
        value = values[left] * (1.0 - frac) + values[left + 1] * frac
        value += rng.uniform(-amplitude * 0.18, amplitude * 0.18)
        wobble.append(int(round(value)))
    return wobble


def _edge_opacity(distance_inside: int, feather_px: int) -> int:
    if distance_inside <= 0:
        return 0
    if feather_px <= 0 or distance_inside >= feather_px:
        return 255
    return max(0, min(255, int(round(255 * distance_inside / feather_px))))


def _save_edge_mask(
    width: int,
    height: int,
    bbox: tuple[int, int, int, int] | None,
    target: Path,
    overlap_px: int = DEFAULT_EDGE_MASK_OVERLAP_PX,
    feather_px: int = DEFAULT_EDGE_MASK_FEATHER_PX,
    wobble_px: int = DEFAULT_EDGE_MASK_WOBBLE_PX,
) -> Path:
    from PIL import Image as PILImage

    mask = PILImage.new("L", (width, height), 0)
    mask_pixels = mask.load()
    if bbox is None:
        for y in range(height):
            for x in range(width):
                mask_pixels[x, y] = 255
    else:
        left, right, top, bottom = bbox
        mask_left = left > 0
        mask_right = right < width - 1
        mask_top = top > 0
        mask_bottom = bottom < height - 1
        if not (mask_left or mask_right or mask_top or mask_bottom):
            for y in range(height):
                for x in range(width):
                    mask_pixels[x, y] = 255
        else:
            overlap = max(0, int(overlap_px))
            feather = max(0, int(feather_px))
            wobble = max(0, int(wobble_px))
            rng = random.Random(_mask_seed(width, height, bbox, target))
            left_wobble = _smooth_wobble(height, wobble, rng) if mask_left else []
            right_wobble = _smooth_wobble(height, wobble, rng) if mask_right else []
            top_wobble = _smooth_wobble(width, wobble, rng) if mask_top else []
            bottom_wobble = _smooth_wobble(width, wobble, rng) if mask_bottom else []
            for y in range(height):
                for x in range(width):
                    opacity = 0
                    if mask_left:
                        boundary = max(left + 1, min(width, left + overlap + left_wobble[y]))
                        opacity = max(opacity, _edge_opacity(boundary - x, feather))
                    if mask_right:
                        boundary = min(right - 1, max(-1, right - overlap + right_wobble[y]))
                        opacity = max(opacity, _edge_opacity(x - boundary, feather))
                    if mask_top:
                        boundary = max(top + 1, min(height, top + overlap + top_wobble[x]))
                        opacity = max(opacity, _edge_opacity(boundary - y, feather))
                    if mask_bottom:
                        boundary = min(bottom - 1, max(-1, bottom - overlap + bottom_wobble[x]))
                        opacity = max(opacity, _edge_opacity(y - boundary, feather))
                    mask_pixels[x, y] = opacity
    target.parent.mkdir(parents=True, exist_ok=True)
    mask.save(target, format="PNG")
    return target


def edge_region_found(source: Path) -> bool:
    """Whether save_edge_mask_for_image would find something to fill: black bars on at least one
    edge, or a wholly black frame. False for a frame with picture right up to every edge, where the
    edge mask would cover the whole image."""
    import numpy as np
    from PIL import Image as PILImage

    with PILImage.open(source) as img:
        content = (np.asarray(img.convert("RGB")) > 4).any(axis=2)
    if not content.any():
        return True
    rows = np.flatnonzero(content.any(axis=1))
    cols = np.flatnonzero(content.any(axis=0))
    height, width = content.shape
    return bool(cols[0] > 0 or cols[-1] < width - 1 or rows[0] > 0 or rows[-1] < height - 1)


def save_edge_mask_for_image(source: Path, target: Path, overlap_px: int = DEFAULT_EDGE_MASK_OVERLAP_PX) -> Path:
    """Create a mask for exact-black prepared-canvas edges with a small inward overlap."""
    from PIL import Image as PILImage

    with PILImage.open(source) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        pixels = rgb.load()
        bbox = _edge_bbox_from_rgb_pixels(width, height, lambda x, y: pixels[x, y])
    return _save_edge_mask(width, height, bbox, target, overlap_px)


def save_edge_mask_for_frame(frame: Any, target: Path, overlap_px: int = DEFAULT_EDGE_MASK_OVERLAP_PX) -> Path:
    """Create an edge mask for an OpenCV BGR frame."""
    height, width = frame.shape[:2]
    bbox = _edge_bbox_from_rgb_pixels(width, height, lambda x, y: frame[y, x])
    return _save_edge_mask(width, height, bbox, target, overlap_px)
