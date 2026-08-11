from __future__ import annotations

from dataclasses import dataclass


def even(value: float) -> int:
    number = int(round(value))
    return number if number % 2 == 0 else number + 1


def fit_size(width: int, height: int, target_width: int, target_height: int) -> tuple[int, int]:
    scale = min(target_width / width, target_height / height)
    out_width = min(target_width, max(2, even(width * scale)))
    out_height = min(target_height, max(2, even(height * scale)))
    return out_width, out_height


def crop_box(width: int, height: int, left: int, right: int, top: int, bottom: int) -> tuple[int, int, int, int, int, int]:
    left = min(max(0, int(left)), max(0, width - 2))
    right = min(max(0, int(right)), max(0, width - left - 2))
    top = min(max(0, int(top)), max(0, height - 2))
    bottom = min(max(0, int(bottom)), max(0, height - top - 2))
    crop_width = max(2, width - left - right)
    crop_height = max(2, height - top - bottom)
    crop_width = crop_width if crop_width % 2 == 0 else crop_width - 1
    crop_height = crop_height if crop_height % 2 == 0 else crop_height - 1
    return left, right, top, bottom, crop_width, crop_height


@dataclass(frozen=True)
class SourcePlacement:
    x: int
    y: int
    width: int
    height: int


def source_placement(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    crops: tuple[int, int, int, int],
    reference_width: int | None = None,
    reference_height: int | None = None,
) -> SourcePlacement:
    """Crop first, then fit the remaining frame into the requested canvas.

    Trimmed pixels are discarded geometry, never outpaint targets. The cropped
    frame is centered and scaled as one complete image, so the remaining mask is
    always a simple pillarbox, letterbox, or empty border.
    """
    reference_width = int(reference_width or target_width)
    reference_height = int(reference_height or target_height)
    _left, _right, _top, _bottom, crop_width, crop_height = crop_box(
        source_width, source_height, *crops
    )
    placed_width, placed_height = fit_size(
        crop_width,
        crop_height,
        reference_width,
        reference_height,
    )

    if (reference_width, reference_height) != (target_width, target_height):
        scale_x = target_width / reference_width
        scale_y = target_height / reference_height
        placed_width = min(target_width, max(2, even(placed_width * scale_x)))
        placed_height = min(target_height, max(2, even(placed_height * scale_y)))

    x = (target_width - placed_width) // 2
    y = (target_height - placed_height) // 2
    return SourcePlacement(x, y, placed_width, placed_height)
