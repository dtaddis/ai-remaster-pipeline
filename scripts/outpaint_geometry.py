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
    full_x: int
    full_y: int
    full_width: int
    full_height: int


def source_placement(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    crops: tuple[int, int, int, int],
    reference_width: int | None = None,
    reference_height: int | None = None,
) -> SourcePlacement:
    reference_width = int(reference_width or target_width)
    reference_height = int(reference_height or target_height)
    left, _right, top, _bottom, crop_width, crop_height = crop_box(source_width, source_height, *crops)
    full_width, full_height = fit_size(source_width, source_height, reference_width, reference_height)
    full_x = (reference_width - full_width) // 2
    full_y = (reference_height - full_height) // 2

    crop_x = full_x + int(round(left * full_width / source_width))
    crop_y = full_y + int(round(top * full_height / source_height))
    crop_end_x = full_x + int(round((left + crop_width) * full_width / source_width))
    crop_end_y = full_y + int(round((top + crop_height) * full_height / source_height))

    if (reference_width, reference_height) != (target_width, target_height):
        scale_x = target_width / reference_width
        scale_y = target_height / reference_height
        full_x = int(round(full_x * scale_x))
        full_y = int(round(full_y * scale_y))
        full_width = max(2, even(full_width * scale_x))
        full_height = max(2, even(full_height * scale_y))
        crop_x = int(round(crop_x * scale_x))
        crop_y = int(round(crop_y * scale_y))
        crop_end_x = int(round(crop_end_x * scale_x))
        crop_end_y = int(round(crop_end_y * scale_y))

    placed_width = max(2, crop_end_x - crop_x)
    placed_height = max(2, crop_end_y - crop_y)
    placed_width = placed_width if placed_width % 2 == 0 else placed_width - 1
    placed_height = placed_height if placed_height % 2 == 0 else placed_height - 1
    return SourcePlacement(crop_x, crop_y, placed_width, placed_height, full_x, full_y, full_width, full_height)
