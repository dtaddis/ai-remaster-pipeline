"""Colourise reference stills with Qwen-Image-2.1 in ComfyUI.

The Qwen-Image-2.1 counterpart of openai_generate_reference.py, walking the same manifest rows. The
black-and-white still is <image1>, the edit target: the whole frame is regenerated in colour on a
canvas of up to about 2K, then scaled back to the still's own size. Like the guide edit, it runs
Viggle's 6-step turbo distill by default (--quality base for the 40-step model).

--describe has Qwen3-VL (2.1's own text encoder, through ComfyUI's TextGenerate) write two or
three sentences about each still (who is in frame, their poses, the framing, the background) into
the prompt. That description is what lets 2.1 take previous colour references for continuity:
with a short "colourise this" prompt, the only full picture of the scene the model has is the
colour reference, and it reproduces that instead of colouring its own frame. Tested 2026-10-09 on
eleven saloon shots of The Most Dangerous Game (shots kept / 11): references without a
description 1 (full size) or 6 (256-448 px); described with 1-3 full-size references 3; described
with two 448 px references 11. So --reference-count only takes effect with --describe, is held to
two, and the references go in at 448 px.

Even so, every still made with references is checked against its grey source (structure_score):
one that no longer has the source's composition is redone without references, so a copied frame
can never become the next still's reference.

Qwen-Image-2.1 is under the Qwen Research License (research/evaluation use only); the GUI warns
when it is chosen, and its weights are fetched only on first use.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from comfy_api import ensure_node_types, extract_output_files, queue_prompt, wait_for_comfy, wait_for_prompt
from common import ROOT, copy_to_comfy_input, newest_output, resolve_path
from dependency_manager import QWEN_IMAGE_21_TEXT_ENCODER, ensure_qwen_image_21_models
from openai_generate_reference import manifest_rows_to_generate, nearby_reference_images, row_source, row_target
from qwen21_edit_guide_image import BASE_STEPS, DEFAULT_MAX_PIXELS, QUALITIES, REQUIRED_NODES, TURBO_SIGMA_NODES, canvas_size, model_loader_nodes, sampler_nodes

# Previous colour references 2.1 is given (with --describe); more than two took over the still.
MAX_REFERENCE_IMAGES = 2
# References carry palette, not content: at 448 px they still steer colour without being copied.
REFERENCE_MAX_PIXELS = 448 * 448
# structure_score below this means the output no longer has the source still's composition.
# Measured: faithful colourisations 0.94-1.00, copied shots 0.0-0.4.
COPY_THRESHOLD = 0.8
SAVE_NODE_ID = "25"
DESCRIBE_NODE_ID = "52"
DESCRIBE_REQUIRED_NODES = {"TextGenerate": "ComfyUI core (0.37 or newer)"}
DESCRIBE_REQUEST = (
    "Describe this black-and-white film frame for a colourisation model in two or three plain sentences: "
    "how many people there are, their apparent age and clothing, where each one is in the frame and what they "
    "are doing, how the shot is framed (close-up, medium or wide), and the main background elements. Describe "
    "only what is visible and do not mention colour."
)


def describe_prompt(source_name: str) -> dict[str, Any]:
    """Graph that has Qwen3-VL describe the still; the text comes back from the PreviewAny node."""
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": source_name}, "_meta": {"title": "Source Image"}},
        "11": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN_IMAGE_21_TEXT_ENCODER, "type": "qwen_image", "device": "default"}},
        "50": {
            "class_type": "TextGenerate",
            "inputs": {
                "clip": ["11", 0], "prompt": DESCRIBE_REQUEST, "max_length": 256, "sampling_mode": "off",
                "image": ["1", 0], "thinking": False, "use_default_template": True, "mtp": "auto",
            },
        },
        DESCRIBE_NODE_ID: {"class_type": "PreviewAny", "inputs": {"source": ["50", 0]}},
    }


def describe_still(args: argparse.Namespace, source_name: str) -> str:
    prompt_id = queue_prompt(args.comfy_url, describe_prompt(source_name))
    history = wait_for_prompt(args.comfy_url, prompt_id, args.poll_seconds)
    entry = history.get(prompt_id, history)
    text = entry.get("outputs", {}).get(DESCRIBE_NODE_ID, {}).get("text", "")
    return " ".join((text[0] if isinstance(text, list) and text else str(text or "")).split())


def still_prompt(args: argparse.Namespace, description: str, reference_count: int) -> str:
    """The user's prompt, the still's own description, what the references are for, then the
    suffix and any per-shot note."""
    parts = [args.prompt]
    if description:
        parts.append(f"<image1> is a black-and-white film frame that shows: {description} Keep exactly these people, poses, framing and background.")
    if reference_count:
        tags = ", ".join(f"<image{index}>" for index in range(2, reference_count + 2))
        are = "is another shot" if reference_count == 1 else "are other shots"
        parts.append(
            f"{tags} {are} of the same scene: use them only for the colours of the set, furniture, costumes and "
            "skin tones; do not show their rooms, furniture or people."
        )
    parts += [args.prompt_suffix, args.add_prompt]
    return " ".join(part.strip() for part in parts if part and part.strip())


def build_prompt(source_name: str, reference_names: list[str], prompt: str, *, seed: int, prefix: str, canvas: tuple[int, int], quality: str = "turbo") -> dict[str, Any]:
    graph: dict[str, Any] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": source_name}, "_meta": {"title": "Source Image"}},
        **model_loader_nodes(quality),
        # The empty latent from the encoder, at the still's canvas size.
        **sampler_nodes(["20", 2], quality=quality, seed=seed, canvas=canvas),
    }
    images = {"images.image_1": ["1", 0]}
    for offset, name in enumerate(reference_names):
        node_id = str(30 + offset)
        graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}, "_meta": {"title": f"Colour reference {offset + 1}"}}
        images[f"images.image_{offset + 2}"] = [node_id, 0]
    graph.update({
        # resolution 0: every image keeps the size it was prepared at, and the latent follows image_1.
        "20": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {"clip": ["11", 0], "prompt": prompt, "negative_prompt": "", "vae": ["12", 0], "resolution": 0, **images},
        },
        "24": {"class_type": "VAEDecode", "inputs": {"samples": ["23", 0], "vae": ["12", 0]}},
        SAVE_NODE_ID: {"class_type": "SaveImage", "inputs": {"images": ["24", 0], "filename_prefix": prefix}},
    })
    return graph


def structure_score(result: np.ndarray, source: np.ndarray) -> float:
    """How well the result keeps the source's composition: correlation of their blurred,
    normalised brightness at 256 px wide. 1.0 is the same picture; a copied shot is near 0."""
    import cv2

    def normalised(rgb: np.ndarray) -> np.ndarray:
        grey = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2GRAY)
        height = max(16, round(256 * source.shape[0] / source.shape[1]))
        grey = cv2.GaussianBlur(cv2.resize(grey, (256, height), interpolation=cv2.INTER_AREA).astype(np.float32), (0, 0), 1.5)
        return (grey - grey.mean()) / (grey.std() + 1e-6)

    return float((normalised(result) * normalised(source)).mean())


def _prepared_copy(path: Path, target: Path, max_pixels: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Save path as an RGB PNG on a 32-multiple canvas within max_pixels; return (source size, canvas)."""
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
    size = canvas_size(rgb.width, rgb.height, max_pixels)
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    (rgb if rgb.size == size else rgb.resize(size, resampling)).save(target, format="PNG")
    return rgb.size, size


def _render(args: argparse.Namespace, source_name: str, reference_names: list[str], prompt: str, seed: int, canvas: tuple[int, int], prefix: str):
    from PIL import Image

    payload = build_prompt(source_name, reference_names, prompt, seed=seed, prefix=prefix, canvas=canvas, quality=args.quality)
    prompt_id = queue_prompt(args.comfy_url, payload)
    print(f"Queued ComfyUI prompt: {prompt_id}", flush=True)
    history = wait_for_prompt(args.comfy_url, prompt_id, args.poll_seconds)
    with Image.open(newest_output(extract_output_files(history, resolve_path(args.comfy_output_root)))) as image:
        return image.convert("RGB")


def generate(args: argparse.Namespace, references: list[Path]) -> Path:
    from PIL import Image

    source = resolve_path(args.source_image)
    output = resolve_path(args.output)
    if not source.is_file():
        raise FileNotFoundError(f"Reference source image not found: {source}")
    if output.exists() and not args.force:
        print(f"Reuse Qwen-Image-2.1 reference: {output}", flush=True)
        return output
    references = [path for path in references if path.is_file()][:MAX_REFERENCE_IMAGES] if args.describe else []
    seed = args.seed if args.seed >= 0 else random.randint(0, 2**31 - 1)
    steps = len(TURBO_SIGMA_NODES) if args.quality == "turbo" else BASE_STEPS
    print(f"Qwen-Image-2.1 {args.quality} reference ({steps} steps): {source} -> {output}", flush=True)
    if args.dry_run:
        print(f"Qwen-Image-2.1 prompt: {still_prompt(args, '<description>' if args.describe else '', len(references))}", flush=True)
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.with_name(output.stem + "_qwen21")
    comfy_dir = resolve_path(args.comfy_dir)
    source_copy = work.with_name(work.name + "_source.png")
    (width, height), canvas = _prepared_copy(source, source_copy, args.max_pixels)
    source_name = copy_to_comfy_input(source_copy, comfy_dir, "arp_qwen21_reference_sources")
    description = describe_still(args, source_name) if args.describe else ""
    if description:
        print(f"Qwen3-VL description: {description}", flush=True)
    elif args.describe:
        print("Qwen3-VL returned no description; colouring without previous references.", flush=True)
        references = []
    reference_names = []
    for index, reference in enumerate(references):
        copy = work.with_name(f"{work.name}_ref{index + 1}.png")
        _prepared_copy(reference, copy, args.reference_max_pixels)
        reference_names.append(copy_to_comfy_input(copy, comfy_dir, "arp_qwen21_reference_refs"))
    prefix = str(Path("ai_remaster_qwen_references") / output.stem).replace("\\", "/") + "_qwen21"
    prompt = still_prompt(args, description, len(reference_names))
    if reference_names:
        print(f"Qwen-Image-2.1 colour references: {len(reference_names)}", flush=True)
    print(f"Qwen-Image-2.1 prompt: {prompt}", flush=True)
    result = _render(args, source_name, reference_names, prompt, seed, canvas, prefix)
    if reference_names:
        with Image.open(source_copy) as image:
            grey_source = np.asarray(image.convert("RGB"))
        score = structure_score(np.asarray(result), grey_source)
        print(f"Composition check: {score:.2f} (copies score under {COPY_THRESHOLD})", flush=True)
        if score < COPY_THRESHOLD:
            print("The colour references took over this still; colouring it again without them.", flush=True)
            result = _render(args, source_name, [], still_prompt(args, description, 0), seed, canvas, prefix)
    # The format follows the manifest's colour_reference extension.
    temp = output.with_name(output.stem + ".partial" + output.suffix)
    if result.size != (width, height):
        result = result.resize((width, height), getattr(Image, "Resampling", Image).LANCZOS)
    result.save(temp)
    temp.replace(output)
    for leftover in [source_copy, *work.parent.glob(work.name + "_ref*.png")]:
        leftover.unlink(missing_ok=True)
    print(f"Wrote {output}", flush=True)
    return output


def generate_manifest(args: argparse.Namespace) -> None:
    rows, selected_rows = manifest_rows_to_generate(args)
    print(f"Manifest: {resolve_path(args.manifest)}", flush=True)
    print(f"Rows: {len(selected_rows)}", flush=True)
    if args.reference_count and not args.describe:
        print("Previous colour references need --describe with Qwen-Image-2.1 (without it they are copied); not sending any.", flush=True)
    for index, row in selected_rows:
        source = row_source(row)
        output = row_target(row)
        if not source or not output:
            raise RuntimeError(f"Manifest row {index} must have source_reference and color_reference.")
        child = argparse.Namespace(**vars(args))
        child.source_image = source
        child.output = output
        child.add_prompt = " ".join(part.strip() for part in (row.get("prompt", ""), args.add_prompt) if part and part.strip())
        count = max(0, min(MAX_REFERENCE_IMAGES, args.reference_count))
        generate(child, nearby_reference_images(rows, index, count))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colourise reference stills with Qwen-Image-2.1 (ComfyUI).")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--add-prompt", default="")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-dir", type=Path, default=ROOT / "tools" / "comfyui")
    parser.add_argument("--comfy-output-root", type=Path, default=ROOT / "tools" / "comfyui" / "output")
    parser.add_argument("--quality", choices=QUALITIES, default="turbo", help=f"turbo: Viggle's 6-step distill (default). base: the 40-step model at {BASE_STEPS} steps, about 5x slower.")
    parser.add_argument("--seed", type=int, default=-1, help="Sampler seed; -1 picks one at random per still.")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help="Largest canvas for the still being coloured.")
    parser.add_argument("--describe", action="store_true", help="Have Qwen3-VL describe each still into the prompt (about 4 s); needed for --reference-count.")
    parser.add_argument("--reference-count", type=int, default=0, help=f"With --describe: nearest existing colour references to send with each still (0-{MAX_REFERENCE_IMAGES}).")
    parser.add_argument("--reference-max-pixels", type=int, default=REFERENCE_MAX_PIXELS, help="Canvas for each colour reference.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--row-index", type=int)
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.dry_run:
        comfy_dir = resolve_path(args.comfy_dir)
        ensure_qwen_image_21_models(comfy_dir, turbo=args.quality == "turbo")
        print(f"Waiting for ComfyUI at {args.comfy_url}...", flush=True)
        wait_for_comfy(args.comfy_url, timeout_seconds=180, poll_seconds=args.poll_seconds)
        required = {**REQUIRED_NODES, **(DESCRIBE_REQUIRED_NODES if args.describe else {})}
        ensure_node_types(args.comfy_url, required, "Qwen-Image-2.1 reference generation", comfy_dir)
    generate_manifest(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise
