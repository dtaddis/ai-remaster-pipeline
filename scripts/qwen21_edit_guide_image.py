"""Edit one outpaint guide frame with Qwen-Image-2.1 in ComfyUI, changing nothing outside the mask.

By default the transformer is Viggle's 6-step turbo distill of 2.1 (its "Lightning"), sampled on
Viggle's resolution-shifted schedule; --quality base runs the 40-step model instead.

The graph is ARP's Qwen 2511 inpaint workflow with 2.1's encoder swapped in: the source is
VAE-encoded and denoised only under a grown mask (SetLatentNoiseMask), while the conditioning
image has the masked content blurred away so the model is not anchored to what it should
replace. 2.1 is native up to about 2K, and its encoder wants sides in multiples of 32, so the
source is scaled onto such a canvas, edited there, scaled back, tone-matched from the kept pixels
and composited through the mask exactly as the OpenAI guide edit is. Pixels outside the mask are
the source's own, and the output has the source's exact size.

Qwen-Image-2.1 is under the Qwen Research License (research/evaluation use only); the GUI warns
when it is chosen, and its weights are fetched only on first use.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from comfy_api import ensure_node_types, extract_output_files, queue_prompt, wait_for_comfy, wait_for_prompt
from common import ROOT, copy_to_comfy_input, newest_output, resolve_path
from dependency_manager import (
    QWEN_IMAGE_21_DIFFUSION,
    QWEN_IMAGE_21_TEXT_ENCODER,
    QWEN_IMAGE_21_TURBO_DIFFUSION,
    QWEN_IMAGE_21_VAE,
    ensure_qwen_image_21_models,
)
from openai_edit_guide_image import (
    DEFAULT_FEATHER_PX,
    DEFAULT_TONE_RADIUS_PX,
    KEEP_THRESHOLD,
    _resize,
    composite_alpha,
    load_mask,
    match_tone,
)

# 2.1's native ceiling (2048x2048); larger guides are scaled down for the edit and back up after.
DEFAULT_MAX_PIXELS = 2048 * 2048
# "turbo" is Viggle's 6-step distill (v0.3, merged into int8 convrot), the default: about 5x faster
# than the base model. "base" is the 40-step model run at ComfyUI's template settings (euler, simple,
# 30 steps here; the official pipeline uses 40-50). Both run without CFG, so no negative prompt.
QUALITIES = ("turbo", "base")
BASE_STEPS = 30
# Viggle's raw student nodes for 6 steps. The pipeline shifts them by image size (turbo_sigmas);
# Viggle: change the count only at the high-noise end, keeping 0.875, 0.75, 0.5, 0.25.
TURBO_SIGMA_NODES = (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25)
SAVE_NODE_ID = "25"
REQUIRED_NODES = {
    "TextEncodeQwenImage21": "ComfyUI core (0.37 or newer)",
    "ManualSigmas": "ComfyUI core (0.37 or newer)",
}

PROMPT_TEMPLATE = (
    "{instruction} Edit only the blurred or black areas of <image1>; everything else stays exactly as "
    "it is. Keep the same framing, camera position, perspective, lighting, colour grade, film grain "
    "and sharpness, and blend the edit seamlessly into its surroundings. Do not add borders, "
    "captions or text."
)


def _round32(value: float) -> int:
    return max(32, int(round(value / 32)) * 32)


def canvas_size(width: int, height: int, max_pixels: int = DEFAULT_MAX_PIXELS) -> tuple[int, int]:
    """Edit size: the source shrunk to fit max_pixels (never enlarged), sides rounded to 32."""
    scale = min(1.0, math.sqrt(max_pixels / float(width * height)))
    return _round32(width * scale), _round32(height * scale)


def turbo_sigmas(width: int, height: int) -> list[float]:
    """Viggle's 6-step schedule at this canvas size, as diffusers' QwenImage21Pipeline runs it
    (and Viggle's ComfyUI sigma node): exponential time shift with mu = calculate_shift(tokens,
    256, 8192, 0.5, 0.9), tokens = (H/16)(W/16), no shift_terminal, ending at 0."""
    tokens = (height // 16) * (width // 16)
    mu = 0.5 + (0.9 - 0.5) * (tokens - 256) / (8192 - 256)
    return [math.exp(mu) / (math.exp(mu) + (1.0 / node - 1.0)) for node in TURBO_SIGMA_NODES] + [0.0]


def model_loader_nodes(quality: str = "turbo") -> dict[str, Any]:
    """Nodes 10-12: the 2.1 transformer (turbo or base), its Qwen3-VL text encoder and its VAE."""
    transformer = QWEN_IMAGE_21_TURBO_DIFFUSION if quality == "turbo" else QWEN_IMAGE_21_DIFFUSION
    return {
        "10": {"class_type": "UNETLoader", "inputs": {"unet_name": transformer, "weight_dtype": "default"}},
        "11": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN_IMAGE_21_TEXT_ENCODER, "type": "qwen_image", "device": "default"}},
        "12": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_IMAGE_21_VAE}},
    }


def sampler_nodes(latent: list[Any], *, quality: str, seed: int, canvas: tuple[int, int]) -> dict[str, Any]:
    """Node 23 (plus its helpers) denoising latent with conditioning node 20; output 0 is the latent.
    The turbo needs its own sigmas, so it samples through SamplerCustomAdvanced; the base model
    uses a plain KSampler. A noise mask on the latent is honoured by both."""
    if quality != "turbo":
        return {"23": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["10", 0], "positive": ["20", 0], "negative": ["20", 1], "latent_image": latent,
                "seed": int(seed), "steps": BASE_STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            },
        }}
    sigmas = ", ".join(f"{value:.6f}" for value in turbo_sigmas(*canvas))
    return {
        "26": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "27": {"class_type": "BasicGuider", "inputs": {"model": ["10", 0], "conditioning": ["20", 0]}},
        "28": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "29": {"class_type": "ManualSigmas", "inputs": {"sigmas": sigmas}},
        "23": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"noise": ["26", 0], "guider": ["27", 0], "sampler": ["28", 0], "sigmas": ["29", 0], "latent_image": latent},
        },
    }


def build_prompt(source_name: str, mask_name: str, prompt: str, *, seed: int, grow_px: int, prefix: str, canvas: tuple[int, int], quality: str = "turbo") -> dict[str, Any]:
    return {
        **model_loader_nodes(quality),
        **sampler_nodes(["22", 0], quality=quality, seed=seed, canvas=canvas),
        "1": {"class_type": "LoadImage", "inputs": {"image": source_name}, "_meta": {"title": "Source Image"}},
        "2": {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}, "_meta": {"title": "Mask Image"}},
        "3": {"class_type": "GrowMask", "inputs": {"mask": ["2", 0], "expand": int(grow_px), "tapered_corners": True}},
        # Remove the selected content from the conditioning image, as the 2511 inpaint workflow does.
        "4": {"class_type": "ImageBlur", "inputs": {"image": ["1", 0], "blur_radius": 31, "sigma": 10.0}},
        "5": {"class_type": "ImageCompositeMasked", "inputs": {"destination": ["1", 0], "source": ["4", 0], "x": 0, "y": 0, "resize_source": False, "mask": ["3", 0]}},
        # resolution 0 keeps the reference at the canvas size (already a multiple of 32), so its
        # latent lines up with the encoded source being denoised.
        "20": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {"clip": ["11", 0], "prompt": prompt, "negative_prompt": "", "vae": ["12", 0], "resolution": 0, "images.image_1": ["5", 0]},
        },
        "21": {"class_type": "VAEEncode", "inputs": {"pixels": ["1", 0], "vae": ["12", 0]}},
        "22": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["21", 0], "mask": ["3", 0]}},
        "24": {"class_type": "VAEDecode", "inputs": {"samples": ["23", 0], "vae": ["12", 0]}},
        SAVE_NODE_ID: {"class_type": "SaveImage", "inputs": {"images": ["24", 0], "filename_prefix": prefix}},
    }


def edit_guide(args: argparse.Namespace) -> Path:
    from PIL import Image

    source_path = resolve_path(args.source_image)
    mask_path = resolve_path(args.mask)
    output = resolve_path(args.output)
    if not source_path.is_file():
        raise FileNotFoundError(f"Guide source image not found: {source_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Guide mask not found: {mask_path}")

    with Image.open(source_path) as image:
        source = np.asarray(image.convert("RGB"))
    height, width = source.shape[:2]
    mask = load_mask(mask_path, (width, height))
    if not (mask > KEEP_THRESHOLD).any():
        raise RuntimeError("The edit mask is empty; paint the area to change first.")

    canvas = canvas_size(width, height, args.max_pixels)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas_source = output.with_name(output.stem + "_qwen21_source.png")
    canvas_mask = output.with_name(output.stem + "_qwen21_mask.png")
    Image.fromarray(_resize(source, canvas), "RGB").save(canvas_source, format="PNG")
    edit = ((mask > KEEP_THRESHOLD) * 255).astype(np.uint8)
    Image.fromarray(_resize(edit, canvas, smooth=False), "L").save(canvas_mask, format="PNG")

    instruction = args.instruction.strip()
    if instruction and instruction[-1] not in ".!?":
        instruction += "."
    prompt = PROMPT_TEMPLATE.format(instruction=instruction).strip()
    seed = args.seed if args.seed >= 0 else random.randint(0, 2**31 - 1)
    print(f"Qwen-Image-2.1 guide edit: {source_path} + {mask_path} -> {output}", flush=True)
    steps = len(TURBO_SIGMA_NODES) if args.quality == "turbo" else BASE_STEPS
    print(f"Qwen-Image-2.1 {args.quality}: canvas {canvas[0]}x{canvas[1]} (source {width}x{height}), {steps} steps, seed {seed}", flush=True)
    print(f"Qwen-Image-2.1 edit prompt: {prompt}", flush=True)
    if args.dry_run:
        return output

    comfy_dir = resolve_path(args.comfy_dir)
    ensure_qwen_image_21_models(comfy_dir, turbo=args.quality == "turbo")
    print(f"Waiting for ComfyUI at {args.comfy_url}...", flush=True)
    wait_for_comfy(args.comfy_url, timeout_seconds=180, poll_seconds=args.poll_seconds)
    ensure_node_types(args.comfy_url, REQUIRED_NODES, "Qwen-Image-2.1 guide edit", comfy_dir)
    payload = build_prompt(
        copy_to_comfy_input(canvas_source, comfy_dir, "arp_qwen21_guide_edits"),
        copy_to_comfy_input(canvas_mask, comfy_dir, "arp_qwen21_guide_masks"),
        prompt,
        seed=seed,
        grow_px=args.grow_px,
        prefix=str(Path("ai_remaster_qwen_edits") / output.stem).replace("\\", "/") + "_qwen21",
        canvas=canvas,
        quality=args.quality,
    )
    prompt_id = queue_prompt(args.comfy_url, payload)
    print(f"Queued ComfyUI prompt: {prompt_id}", flush=True)
    history = wait_for_prompt(args.comfy_url, prompt_id, args.poll_seconds)
    produced = newest_output(extract_output_files(history, resolve_path(args.comfy_output_root)))
    with Image.open(produced) as image:
        result = np.asarray(image.convert("RGB"))
    raw_path = output.with_name(output.stem + "_raw" + output.suffix)
    Image.fromarray(result, "RGB").save(raw_path, format="PNG")

    result = _resize(result, (width, height))
    # Radii were tuned on a 1280x704 guide; scale them with the frame.
    scale = min(width, height) / 704.0
    if not args.no_tone_match:
        # Kept pixels only drift by the VAE round trip and rescale; unfilled black bars are no evidence.
        keep = (mask <= KEEP_THRESHOLD) & (source.max(axis=2) > 4)
        result = match_tone(result, source, keep, max(4.0, args.tone_radius_px * scale))
    feather_px = args.feather_px if args.feather_px >= 0 else max(2, round(DEFAULT_FEATHER_PX * scale))
    alpha = composite_alpha(mask, feather_px)[..., None]
    final = np.clip(source.astype(np.float32) * (1.0 - alpha) + result.astype(np.float32) * alpha, 0, 255).round().astype(np.uint8)

    temp = output.with_suffix(output.suffix + ".partial")
    Image.fromarray(final, "RGB").save(temp, format="PNG")
    temp.replace(output)
    print(f"Composited the Qwen-Image-2.1 edit into the source through the mask ({width}x{height}).", flush=True)
    print(f"Wrote {output}", flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Edit one outpaint guide frame inside a mask with Qwen-Image-2.1 (ComfyUI).")
    parser.add_argument("--source-image", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path, help="Greyscale PNG, white = edit. Any size; it is scaled to the source.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-dir", type=Path, default=ROOT / "tools" / "comfyui")
    parser.add_argument("--comfy-output-root", type=Path, default=ROOT / "tools" / "comfyui" / "output")
    parser.add_argument("--quality", choices=QUALITIES, default="turbo", help=f"turbo: Viggle's 6-step distill (default). base: the 40-step model at {BASE_STEPS} steps, about 5x slower.")
    parser.add_argument("--seed", type=int, default=-1, help="Sampler seed; -1 picks one at random.")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help="Largest edit canvas; bigger guides are scaled down for the edit.")
    parser.add_argument("--grow-px", type=int, default=16, help="How far past the mask the model may repaint (canvas pixels).")
    parser.add_argument("--feather-px", type=int, default=-1, help=f"Inward feather of the composite in source pixels (default {DEFAULT_FEATHER_PX} at a 704px short edge, scaled).")
    parser.add_argument("--tone-radius-px", type=float, default=DEFAULT_TONE_RADIUS_PX, help="Finest tone-matching window (Gaussian sigma) at a 704px short edge, scaled with the frame.")
    parser.add_argument("--no-tone-match", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
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
