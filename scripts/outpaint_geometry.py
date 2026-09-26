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


def source_envelope_size(
    source_width: int,
    source_height: int,
    adjustments: tuple[int, int, int, int],
) -> tuple[int, int]:
    """Return cropped source plus requested virtual outpaint border.

    Internal adjustment values retain the legacy crop convention: positive
    removes source pixels and negative adds virtual black canvas.  The GUI
    presents the friendlier inverse convention (negative trim, positive extend)
    and converts before calling the processing scripts.
    """

    left, right, top, bottom = (int(value) for value in adjustments)
    _crop_left, _crop_right, _crop_top, _crop_bottom, crop_width, crop_height = crop_box(
        source_width, source_height, left, right, top, bottom
    )
    return (
        crop_width + max(0, -left) + max(0, -right),
        crop_height + max(0, -top) + max(0, -bottom),
    )


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
    """Trim/extend first, then fit the source envelope into the canvas.

    Positive internal values trim pixels. Negative values add virtual border on
    that edge, shrinking and positioning the surviving source inside the fixed
    target-aspect canvas so LTX can outpaint in any combination of directions.
    """
    reference_width = int(reference_width or target_width)
    reference_height = int(reference_height or target_height)
    left, right, top, bottom = (int(value) for value in crops)
    _crop_left, _crop_right, _crop_top, _crop_bottom, crop_width, crop_height = crop_box(
        source_width, source_height, *crops
    )
    extend_left = max(0, -left)
    extend_right = max(0, -right)
    extend_top = max(0, -top)
    extend_bottom = max(0, -bottom)
    envelope_width = crop_width + extend_left + extend_right
    envelope_height = crop_height + extend_top + extend_bottom
    fitted_envelope_width, fitted_envelope_height = fit_size(
        envelope_width,
        envelope_height,
        reference_width,
        reference_height,
    )
    scale_x = fitted_envelope_width / envelope_width
    scale_y = fitted_envelope_height / envelope_height
    placed_width = min(reference_width, max(2, even(crop_width * scale_x)))
    placed_height = min(reference_height, max(2, even(crop_height * scale_y)))
    x = (reference_width - fitted_envelope_width) // 2 + int(round(extend_left * scale_x))
    y = (reference_height - fitted_envelope_height) // 2 + int(round(extend_top * scale_y))

    if (reference_width, reference_height) != (target_width, target_height):
        scale_x = target_width / reference_width
        scale_y = target_height / reference_height
        placed_width = min(target_width, max(2, even(placed_width * scale_x)))
        placed_height = min(target_height, max(2, even(placed_height * scale_y)))
        x = int(round(x * scale_x))
        y = int(round(y * scale_y))
    return SourcePlacement(x, y, placed_width, placed_height)


def native_source_layout(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    crops: tuple[int, int, int, int],
) -> tuple[tuple[int, int], SourcePlacement]:
    """Resize the outpaint canvas so the trimmed source lands on it 1:1.

    The source is placed on the target canvas exactly as ``source_placement``
    would, then the whole canvas is scaled by the factor that turns the placed
    source back into its native trimmed size. Only the outpainted base gets
    resampled; the source pixels pass through unscaled. Trims and extends are
    honoured because they are already baked into the working placement.
    """

    placement = source_placement(source_width, source_height, target_width, target_height, crops)
    *_offsets, crop_width, crop_height = crop_box(source_width, source_height, *crops)
    scale_x = crop_width / placement.width
    scale_y = crop_height / placement.height
    canvas_width = max(crop_width, even(target_width * scale_x))
    canvas_height = max(crop_height, even(target_height * scale_y))
    x = min(max(0, int(round(placement.x * scale_x))), canvas_width - crop_width)
    y = min(max(0, int(round(placement.y * scale_y))), canvas_height - crop_height)
    # Even offsets keep the source aligned with 4:2:0 chroma; all sizes here are even too.
    x -= x % 2
    y -= y % 2
    return (canvas_width, canvas_height), SourcePlacement(x, y, crop_width, crop_height)
