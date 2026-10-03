"""Edit one outpaint guide frame with the OpenAI Images API, changing nothing outside the mask.

GPT Image treats an edit mask as guidance rather than a hard stencil: it re-renders the whole
picture, shifts tone region by region, and only returns certain sizes. So the API result is never
used as-is. The source is fitted (scaled, then edge-padded) onto a canvas the model accepts, sent
with a matching alpha mask, and the result is cropped back to the source geometry, tone-matched to
the source locally from the kept pixels around the mask, and composited through the mask. Pixels
outside the mask are the source's own, and the output has the source's exact size.
"""
from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path

import numpy as np

from common import resolve_path
from openai_generate_reference import (
    DEFAULT_MODEL,
    MAX_IMAGE_EDGE,
    MAX_IMAGE_PIXELS,
    BOUNDARY,
    first_image_bytes,
    multipart_bytes,
    multipart_field,
    post_image_edit,
    supports_custom_size,
)

# GPT Image 2 / 2.5 also need at least this many pixels and an aspect within 1:3..3:1.
MIN_IMAGE_PIXELS = 655_360
MAX_ASPECT = 3.0
# Older GPT Image models only accept these sizes.
PRESET_SIZES = ((1024, 1024), (1536, 1024), (1024, 1536))
# Mask values at or below this count as "keep" (the edge mask's feather never reaches it).
KEEP_THRESHOLD = 0.02
# Composite feather and finest tone-matching radius, in pixels at a 704px short edge. Chosen on a
# real 1280x704 repair where these hid the join that a global tone match left plainly visible.
DEFAULT_FEATHER_PX = 10
DEFAULT_TONE_RADIUS_PX = 12.0

PROMPT_TEMPLATE = (
    "Edit only the masked (transparent) area of this film frame: {instruction} "
    "Everything outside the masked area must stay exactly as it is. Keep the same framing, camera "
    "position, perspective, lighting, colour grade, film grain and sharpness, and blend the edit "
    "seamlessly into its surroundings. Do not crop, zoom, shift or reframe the picture, and do not "
    "add borders, captions or text."
)


def _round16(value: float) -> int:
    return max(16, int(round(value / 16)) * 16)


def api_canvas_size(width: int, height: int, model: str) -> tuple[int, int]:
    """The API image size to fit a width x height source onto, as close to its aspect as allowed."""
    aspect = width / height
    if not supports_custom_size(model):
        return min(PRESET_SIZES, key=lambda size: abs(math.log((size[0] / size[1]) / aspect)))
    # Clamp the aspect first; the source is padded out to it rather than squashed.
    padded_w = max(width, math.ceil(height / MAX_ASPECT))
    padded_h = max(height, math.ceil(width / MAX_ASPECT))
    pixels = padded_w * padded_h
    scale = min(1.0, MAX_IMAGE_EDGE / padded_w, MAX_IMAGE_EDGE / padded_h, math.sqrt(MAX_IMAGE_PIXELS / pixels))
    if pixels * scale * scale < MIN_IMAGE_PIXELS:
        scale = math.sqrt(MIN_IMAGE_PIXELS / pixels)
    canvas_w = min(MAX_IMAGE_EDGE, _round16(padded_w * scale))
    canvas_h = min(MAX_IMAGE_EDGE, _round16(padded_h * scale))
    # Rounding to 16 can nudge the result just outside a limit; step 16px back inside, choosing
    # the edge that keeps the aspect closest to the source's.
    wide = padded_w / padded_h
    for _ in range(64):
        pixels = canvas_w * canvas_h
        if canvas_w > MAX_ASPECT * canvas_h or (pixels > MAX_IMAGE_PIXELS and canvas_w / canvas_h > wide):
            canvas_w -= 16
        elif canvas_h > MAX_ASPECT * canvas_w or pixels > MAX_IMAGE_PIXELS:
            canvas_h -= 16
        elif pixels < MIN_IMAGE_PIXELS:
            grow_h = canvas_w / canvas_h > wide
            if grow_h and canvas_h + 16 > MAX_ASPECT * canvas_w:
                grow_h = False
            elif not grow_h and canvas_w + 16 > MAX_ASPECT * canvas_h:
                grow_h = True
            if grow_h:
                canvas_h += 16
            else:
                canvas_w += 16
        else:
            break
    return canvas_w, canvas_h


def fit_box(width: int, height: int, canvas_w: int, canvas_h: int) -> tuple[int, int, int, int]:
    """Centred (x, y, w, h) of the source scaled to fit inside the canvas without distortion."""
    scale = min(canvas_w / width, canvas_h / height)
    fit_w = max(1, min(canvas_w, int(round(width * scale))))
    fit_h = max(1, min(canvas_h, int(round(height * scale))))
    return (canvas_w - fit_w) // 2, (canvas_h - fit_h) // 2, fit_w, fit_h


def _resize(array: np.ndarray, size: tuple[int, int], *, smooth: bool = True) -> np.ndarray:
    """Lanczos for pictures; nearest for masks, so a hard-edged mask stays hard-edged."""
    from PIL import Image

    if (array.shape[1], array.shape[0]) == size:
        return array
    resampling = getattr(Image, "Resampling", Image)
    image = Image.fromarray(array)
    return np.asarray(image.resize(size, resampling.LANCZOS if smooth else resampling.NEAREST))


def _png_bytes(array: np.ndarray, mode: str) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array, mode).save(buffer, format="PNG")
    return buffer.getvalue()


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    """Mask as float 0..1 (1 = edit) at the source size. The editor sends it at display size."""
    from PIL import Image

    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"))
    return _resize(gray, size, smooth=False).astype(np.float32) / 255.0


def api_inputs(source: np.ndarray, mask: np.ndarray, canvas: tuple[int, int], box: tuple[int, int, int, int], grow_px: int) -> tuple[bytes, bytes]:
    """PNG bytes for the API image (source fitted + edge-padded) and its alpha mask (0 = edit)."""
    import cv2

    canvas_w, canvas_h = canvas
    x, y, fit_w, fit_h = box
    fitted = _resize(source, (fit_w, fit_h))
    image = np.pad(fitted, ((y, canvas_h - fit_h - y), (x, canvas_w - fit_w - x), (0, 0)), mode="edge")
    edit = np.zeros((canvas_h, canvas_w), np.uint8)
    edit[y:y + fit_h, x:x + fit_w] = (_resize((mask * 255).astype(np.uint8), (fit_w, fit_h), smooth=False) > KEEP_THRESHOLD * 255)
    # Let the model repaint a little past the mask so the composite's feather blends into new pixels.
    if grow_px > 0:
        edit = cv2.dilate(edit, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1)))
    rgba = np.dstack([image, np.where(edit > 0, 0, 255).astype(np.uint8)])
    return _png_bytes(image, "RGB"), _png_bytes(rgba, "RGBA")


def result_to_source(result: np.ndarray, canvas: tuple[int, int], box: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    """Undo the fit: crop the source's box out of the API result and scale it to the source size."""
    result = _resize(result, canvas)
    x, y, fit_w, fit_h = box
    return _resize(np.ascontiguousarray(result[y:y + fit_h, x:x + fit_w]), size)


def _local_stats(source: np.ndarray, result: np.ndarray, weight: np.ndarray, sigma: float) -> tuple[np.ndarray, ...]:
    """Gaussian-windowed mean and variance of source and result over the kept pixels only, plus how
    much kept evidence each window had (0..1)."""
    import cv2

    def blur(values: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(values, (0, 0), sigma)

    weight_sum = blur(weight)
    denominator = np.maximum(weight_sum, 1e-6)
    src_mean = blur(source * weight) / denominator
    res_mean = blur(result * weight) / denominator
    src_var = np.maximum(blur(source * source * weight) / denominator - src_mean * src_mean, 0.0)
    res_var = np.maximum(blur(result * result * weight) / denominator - res_mean * res_mean, 0.0)
    support = np.clip(weight_sum / 0.1, 0.0, 1.0)
    return src_mean, res_mean, src_var, res_var, support


def match_tone(result: np.ndarray, source: np.ndarray, keep: np.ndarray, radius_px: float) -> np.ndarray:
    """Map the result's tones onto the source's, locally. GPT Image repaints the whole frame and
    shifts brightness and contrast region by region (it "restores" hazy areas, say), so one global
    correction leaves a seam at the mask. Per channel, a local gain/offset is measured where the two
    should agree (the kept pixels) and carried smoothly into the mask. Wide masks have no kept
    pixels nearby, so the estimate falls back through coarser windows to the whole frame."""
    if keep.mean() < 0.02:
        return result
    weight = keep.astype(np.float32)
    out = np.empty(result.shape, np.float32)
    for channel in range(3):
        src = source[..., channel].astype(np.float32)
        res = result[..., channel].astype(np.float32)
        kept_src, kept_res = src[keep], res[keep]
        src_mean = np.full(src.shape, kept_src.mean(), np.float32)
        res_mean = np.full(src.shape, kept_res.mean(), np.float32)
        src_var = np.full(src.shape, kept_src.var(), np.float32)
        res_var = np.full(src.shape, kept_res.var(), np.float32)
        for sigma in (radius_px * 16, radius_px * 4, radius_px):
            level_src_mean, level_res_mean, level_src_var, level_res_var, support = _local_stats(src, res, weight, sigma)
            src_mean += support * (level_src_mean - src_mean)
            res_mean += support * (level_res_mean - res_mean)
            src_var += support * (level_src_var - src_var)
            res_var += support * (level_res_var - res_var)
        # The +9 (a 3-level std) keeps flat areas from blowing the gain up on noise.
        gain = np.clip(np.sqrt((src_var + 9.0) / (res_var + 9.0)), 0.5, 2.0)
        out[..., channel] = (res - res_mean) * gain + src_mean
    return np.clip(out, 0, 255).round().astype(np.uint8)


def composite_alpha(mask: np.ndarray, feather_px: int) -> np.ndarray:
    """Blend weight for the edit: the mask, faded in over feather_px *inside* it so pixels outside
    the mask stay bit-exact source."""
    import cv2

    region = (mask > KEEP_THRESHOLD).astype(np.uint8)
    if feather_px <= 0:
        return mask * region
    inside = cv2.distanceTransform(region, cv2.DIST_L2, 5)
    return np.minimum(mask, np.clip(inside / float(feather_px), 0.0, 1.0)) * region


def edit_guide(args: argparse.Namespace) -> Path:
    from PIL import Image

    source_path = resolve_path(args.source_image)
    mask_path = resolve_path(args.mask)
    output = resolve_path(args.output)
    if not source_path.is_file():
        raise FileNotFoundError(f"Guide source image not found: {source_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Guide mask not found: {mask_path}")
    token = (args.api_key or "").strip()
    if not token and not args.dry_run:
        raise RuntimeError("Missing OpenAI API key. Add it in Settings -> OpenAI Reference Generation.")

    with Image.open(source_path) as image:
        source = np.asarray(image.convert("RGB"))
    height, width = source.shape[:2]
    mask = load_mask(mask_path, (width, height))
    if not (mask > KEEP_THRESHOLD).any():
        raise RuntimeError("The edit mask is empty; paint the area to change first.")

    canvas = api_canvas_size(width, height, args.model)
    box = fit_box(width, height, *canvas)
    image_png, mask_png = api_inputs(source, mask, canvas, box, args.grow_px)
    instruction = args.instruction.strip()
    if instruction and instruction[-1] not in ".!?":
        instruction += "."
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    print(f"OpenAI guide edit: {source_path} + {mask_path} -> {output}", flush=True)
    print(f"OpenAI model: {args.model}, quality={args.quality}", flush=True)
    print(f"OpenAI canvas: {canvas[0]}x{canvas[1]} (source {width}x{height} fitted at {box[2]}x{box[3]})", flush=True)
    print(f"OpenAI edit prompt: {prompt}", flush=True)
    if args.dry_run:
        return output

    body = b"".join([
        multipart_field("model", args.model),
        multipart_field("prompt", prompt),
        multipart_field("size", f"{canvas[0]}x{canvas[1]}"),
        multipart_field("quality", args.quality),
        multipart_bytes("image", "guide.png", image_png, "image/png"),
        multipart_bytes("mask", "mask.png", mask_png, "image/png"),
        f"--{BOUNDARY}--\r\n".encode("utf-8"),
    ])
    raw_bytes = first_image_bytes(post_image_edit(body, token, args.timeout, args.max_retries))
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_name(output.stem + "_raw" + output.suffix)
    raw_path.write_bytes(raw_bytes)
    with Image.open(io.BytesIO(raw_bytes)) as image:
        result = np.asarray(image.convert("RGB"))
    if (result.shape[1], result.shape[0]) != canvas:
        print(f"OpenAI returned {result.shape[1]}x{result.shape[0]} instead of {canvas[0]}x{canvas[1]}; rescaling.", flush=True)

    result = result_to_source(result, canvas, box, (width, height))
    # Radii were tuned on a 1280x704 guide; scale them with the frame.
    scale = min(width, height) / 704.0
    if not args.no_tone_match:
        # Unfilled black bars are no tone evidence: the model may paint them whatever the mask says.
        keep = (mask <= KEEP_THRESHOLD) & (source.max(axis=2) > 4)
        result = match_tone(result, source, keep, max(4.0, args.tone_radius_px * scale))
    feather_px = args.feather_px if args.feather_px >= 0 else max(2, round(DEFAULT_FEATHER_PX * scale))
    alpha = composite_alpha(mask, feather_px)[..., None]
    final = np.clip(source.astype(np.float32) * (1.0 - alpha) + result.astype(np.float32) * alpha, 0, 255).round().astype(np.uint8)

    temp = output.with_suffix(output.suffix + ".partial")
    Image.fromarray(final, "RGB").save(temp, format="PNG")
    temp.replace(output)
    print(f"Composited the OpenAI edit into the source through the mask ({width}x{height}).", flush=True)
    print(f"Wrote {output}", flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Edit one outpaint guide frame inside a mask with the OpenAI Images API.")
    parser.add_argument("--source-image", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path, help="Greyscale PNG, white = edit. Any size; it is scaled to the source.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quality", default="auto")
    parser.add_argument("--feather-px", type=int, default=-1, help=f"Inward feather of the composite in source pixels (default {DEFAULT_FEATHER_PX} at a 704px short edge, scaled).")
    parser.add_argument("--tone-radius-px", type=float, default=DEFAULT_TONE_RADIUS_PX, help="Finest tone-matching window (Gaussian sigma) at a 704px short edge, scaled with the frame.")
    parser.add_argument("--grow-px", type=int, default=8, help="How far past the mask the model may repaint (API canvas pixels).")
    parser.add_argument("--no-tone-match", action="store_true")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    edit_guide(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise
