from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from comfy_api import extract_output_files, ensure_node_types, node_by_id, queue_prompt, set_widget, wait_for_comfy, wait_for_prompt, workflow_to_prompt
from common import ROOT, file_fingerprint, resolve_path, root_relative, resumable_output, write_signature, safe_stem
from dependency_manager import ensure_outpaint_models
from prepare_outpaint_input import default_output as default_prepared_output
from prepare_outpaint_input import even, parse_aspect, probe_video


DEFAULT_WORKFLOW = ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json"
DEFAULT_COMFY_DIR = ROOT / "tools" / "comfyui"
RECOMMENDED_OVERLAP_FRAMES = 8
MODEL_SIZE_MULTIPLE = 32
OUTPAINT_REQUIRED_NODES = {
    "LTXVImgToVideoConditionOnly": "ComfyUI-LTXVideo",
    "LTXAddVideoICLoRAGuide": "ComfyUI-LTXVideo",
    "LTXVPreprocess": "ComfyUI-LTXVideo",
}


def replace_with_retry(partial: Path, target: Path, label: str, attempts: int = 20, delay: float = 0.5) -> None:
    for attempt in range(attempts):
        try:
            partial.replace(target)
            return
        except PermissionError:
            if attempt >= attempts - 1:
                raise
            print(f"{label} is locked by another process; retrying in {delay:g}s ({attempt + 1}/{attempts})...", flush=True)
            time.sleep(delay)


def aspect_slug(value: str) -> str:
    return value.replace(":", "x").replace(".", "_")


def load_local_config() -> dict[str, str]:
    path = ROOT / ".ai_remaster_config.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return {str(key): str(value) for key, value in data.items() if value is not None}


def target_size(source: Path, aspect: str, target_height: int | None) -> tuple[int, int]:
    info = probe_video(source)
    height = even(target_height or 720)
    width = even(height * parse_aspect(aspect))
    return width, height


def model_safe(value: int, multiple: int = MODEL_SIZE_MULTIPLE) -> int:
    value = max(multiple, int(value))
    lower = max(multiple, (value // multiple) * multiple)
    upper = lower if lower == value else lower + multiple
    return lower if value - lower <= upper - value else upper


def model_safe_size(source: Path, aspect: str, target_height: int | None) -> tuple[int, int]:
    width, height = target_size(source, aspect, target_height)
    return model_safe(width), model_safe(height)


def crop_slug(args: Any) -> str:
    values = [int(getattr(args, key, 0)) for key in ("crop_left", "crop_right", "crop_top", "crop_bottom")]
    return "" if not any(values) else f"_crop{values[0]}-{values[1]}-{values[2]}-{values[3]}"


def default_output(source: Path, aspect: str, target_height: int | None, args: Any | None = None) -> Path:
    width, height = target_size(source, aspect, target_height)
    return ROOT / "intermediate" / "outpainted" / f"{safe_stem(source.name)}_{aspect_slug(aspect)}_{width}x{height}{crop_slug(args) if args else ''}_outpainted.mp4"


def default_raw_output(source: Path, aspect: str, target_height: int | None, args: Any | None = None) -> Path:
    width, height = model_safe_size(source, aspect, target_height)
    return ROOT / "intermediate" / "outpainted" / f"{safe_stem(source.name)}_{aspect_slug(aspect)}_{width}x{height}{crop_slug(args) if args else ''}_raw_comfy.mp4"


def prepared_for(source: Path, aspect: str, target_height: int | None, args: Any | None = None) -> Path:
    width, height = model_safe_size(source, aspect, target_height)
    prepared = default_prepared_output(source, width, height)
    return prepared.with_name(prepared.stem + (crop_slug(args) if args else "") + prepared.suffix)


def run_command(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def find_ffmpeg() -> str:
    candidates = [
        ROOT / ".cache" / "tools" / "ffmpeg" / "ffmpeg.exe",
        Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
        Path("ffmpeg"),
    ]
    for candidate in candidates:
        try:
            subprocess.run([str(candidate), "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return str(candidate)
        except Exception:
            continue
    raise FileNotFoundError("ffmpeg was not found. Re-run install_windows.bat to install ARP's local FFmpeg copy.")


def copy_to_comfy_input(source: Path, comfy_dir: Path) -> str:
    target_dir = comfy_dir / "input" / "arp_outpaint"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if not target.exists() or source.stat().st_mtime_ns != target.stat().st_mtime_ns or source.stat().st_size != target.stat().st_size:
        shutil.copy2(source, target)
    return f"arp_outpaint/{target.name}"


def copy_reference_frame_to_comfy_input(source: Path, comfy_dir: Path) -> str:
    import cv2

    target_dir = comfy_dir / "input" / "arp_outpaint"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{source.stem}_reference.png"
    if not target.exists() or source.stat().st_mtime_ns > target.stat().st_mtime_ns:
        cap = cv2.VideoCapture(str(source))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Could not extract reference frame from: {source}")
        cv2.imwrite(str(target), frame)
    return f"arp_outpaint/{target.name}"


def copy_guide_image_to_comfy_input(guide: Path, comfy_dir: Path, canvas_width: int = 0, canvas_height: int = 0) -> str:
    """Copy a guide image to ComfyUI's input folder, normalised to the LTX canvas size.

    When canvas dimensions are provided the guide is resized to exactly (canvas_width × canvas_height)
    before it reaches LTXVImgToVideoConditionOnly.  This avoids the zoom-out artefact that occurs
    when ResizeImageMaskNode center-crops a mismatched image.

    For small aspect-ratio differences (≤ 10%) — typical of Qwen output — the image is stretched
    directly to canvas size, reversing Qwen's uniform patch-quantisation compression with only a
    ~2% per-axis residual error, which is imperceptible to both humans and LTX's i2v conditioning.

    For large AR differences (> 10%) — e.g. a manually-uploaded portrait or square image — the
    image is letterboxed with black padding instead, so content is never badly distorted.
    """
    from PIL import Image as PILImage

    target_dir = comfy_dir / "input" / "arp_outpaint"
    target_dir.mkdir(parents=True, exist_ok=True)

    if canvas_width > 0 and canvas_height > 0:
        target = target_dir / f"guide_{guide.stem}_{canvas_width}x{canvas_height}.png"
    else:
        target = target_dir / f"guide_{guide.stem}{guide.suffix.lower()}"

    try:
        if target.exists() and target.stat().st_mtime_ns >= guide.stat().st_mtime_ns and target.stat().st_size > 0:
            return f"arp_outpaint/{target.name}"
    except OSError:
        pass

    if canvas_width > 0 and canvas_height > 0:
        with PILImage.open(guide) as img:
            img_w, img_h = img.size
            if img_w == canvas_width and img_h == canvas_height:
                shutil.copy2(guide, target)
            else:
                # Letterbox: scale to fit within the canvas preserving aspect ratio,
                # then pad the remainder with pure black.  This avoids geometric distortion
                # in the guide content — any black bands are treated as outpainting margins
                # by LTX and filled naturally.  Stretching to fill causes visible y-squish
                # in the first few generated frames, so we never do that.
                scale = min(canvas_width / img_w, canvas_height / img_h)
                new_w = max(1, int(img_w * scale))
                new_h = max(1, int(img_h * scale))
                resampling = getattr(PILImage, "Resampling", PILImage).LANCZOS
                resized = img.convert("RGB").resize((new_w, new_h), resampling)
                out = PILImage.new("RGB", (canvas_width, canvas_height), (0, 0, 0))
                out.paste(resized, ((canvas_width - new_w) // 2, (canvas_height - new_h) // 2))
                out.save(target, format="PNG")
                if new_w != canvas_width or new_h != canvas_height:
                    print(
                        f"Guide letterboxed {img_w}x{img_h} -> {new_w}x{new_h} "
                        f"(padded {canvas_width - new_w}px h, {canvas_height - new_h}px v to {canvas_width}x{canvas_height})",
                        flush=True,
                    )
    else:
        shutil.copy2(guide, target)

    return f"arp_outpaint/{target.name}"


def extract_last_frame_as_guide(previous_raw: Path, chunk_dir: Path) -> Path:
    """Extract the last frame of a finished raw chunk as a PNG for i2v guide conditioning."""
    import cv2

    target = chunk_dir / f"guide_prev_{safe_stem(previous_raw.name)}.png"
    if target.exists() and target.stat().st_mtime_ns >= previous_raw.stat().st_mtime_ns:
        return target
    cap = cv2.VideoCapture(str(previous_raw))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not extract last frame from previous chunk: {previous_raw}")
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), frame)
    return target


def set_widget_if_node(workflow: dict[str, Any], node_id: str | None, widget: str | int, value: Any) -> None:
    if not node_id:
        return
    set_widget(node_by_id(workflow, node_id), widget, value)


def add_or_replace_node(workflow: dict[str, Any], node: dict[str, Any]) -> None:
    nodes = workflow.setdefault("nodes", [])
    node_id = str(node["id"])
    for index, existing in enumerate(nodes):
        if str(existing.get("id")) == node_id:
            nodes[index] = node
            return
    nodes.append(node)


def patch_link(workflow: dict[str, Any], link_id: int, source_id: int, source_slot: int, target_id: int, target_slot: int, link_type: str) -> None:
    links = workflow.setdefault("links", [])
    for link in links:
        if int(link[0]) == link_id:
            link[1:6] = [source_id, source_slot, target_id, target_slot, link_type]
            return
    links.append([link_id, source_id, source_slot, target_id, target_slot, link_type])


def set_input_link(workflow: dict[str, Any], node_id: str, input_name: str, link_id: int) -> None:
    node = node_by_id(workflow, node_id)
    for item in node.get("inputs", []):
        if item.get("name") == input_name:
            item["link"] = link_id
            return
    node.setdefault("inputs", []).append({"name": input_name, "link": link_id})


def clear_input_link(workflow: dict[str, Any], node_id: str, input_name: str) -> None:
    node = node_by_id(workflow, node_id)
    for item in node.get("inputs", []):
        if item.get("name") == input_name:
            item["link"] = None
            return


def input_link(workflow: dict[str, Any], node_id: str, input_name: str) -> int | None:
    node = node_by_id(workflow, node_id)
    for item in node.get("inputs", []):
        if item.get("name") == input_name:
            link = item.get("link")
            return int(link) if link is not None else None
    return None


def ensure_widget_input(node: dict[str, Any], name: str, input_type: str = "COMBO") -> None:
    for item in node.setdefault("inputs", []):
        if item.get("name") == name:
            item.setdefault("widget", {"name": name})
            return
    node["inputs"].append({"name": name, "type": input_type, "widget": {"name": name}})


def bypass_optional_preview_nodes(workflow: dict[str, Any]) -> None:
    """Author workflows may include optional MTB color-correct nodes.

    They are behind Crystools switches, but Comfy validates every linked input in
    the API prompt. If MTB/Crystools are not installed, those unused linked
    branches still fail validation. Route downstream nodes through the plain
    image links so the LTX guide path still works on a minimal install.
    """
    try:
        plain_guide_link = input_link(workflow, "5087", "on_false")
        if plain_guide_link is not None:
            set_input_link(workflow, "5012", "image", plain_guide_link)
    except KeyError:
        pass

    try:
        plain_decode_link = input_link(workflow, "5089", "on_false")
        if plain_decode_link is not None:
            set_input_link(workflow, "5076", "images", plain_decode_link)
            set_input_link(workflow, "5067", "image1", plain_decode_link)
    except KeyError:
        pass


# The LTX example workflow is a frontend graph with stable-but-opaque node IDs.
# Keep ARP's edits explicit here so model/backend assumptions remain auditable.
def patch_lightweight_gguf(workflow: dict[str, Any], args) -> None:
    model_node = node_by_id(workflow, "3940")
    model_node["type"] = "UnetLoaderGGUF"
    model_node["title"] = "Unet Loader (GGUF)"
    model_node["inputs"] = [{"name": "unet_name", "type": "COMBO", "widget": {"name": "unet_name"}}]
    model_node["widgets_values"] = [args.gguf_model]
    model_node["outputs"] = [{"name": "MODEL", "type": "MODEL", "links": [13217]}]

    add_or_replace_node(
        workflow,
        {
            "id": 9001,
            "type": "VAELoader",
            "title": "LTX 2.3 Video VAE",
            "mode": 0,
            "inputs": [{"name": "vae_name", "type": "COMBO", "widget": {"name": "vae_name"}}],
            "outputs": [{"name": "VAE", "type": "VAE", "links": [13279, 13348, 13405]}],
            "widgets_values": [args.video_vae],
        },
    )
    patch_link(workflow, 13217, 3940, 0, 5011, 0, "MODEL")
    patch_link(workflow, 13279, 9001, 0, 3159, 0, "VAE")
    patch_link(workflow, 13348, 9001, 0, 4851, 1, "VAE")
    patch_link(workflow, 13405, 9001, 0, 5012, 2, "VAE")
    set_input_link(workflow, "5011", "model", 13217)
    lora_node = node_by_id(workflow, "5011")
    ensure_widget_input(lora_node, "lora_name")
    ensure_widget_input(lora_node, "strength_model", "FLOAT")
    set_widget(lora_node, "0", args.outpaint_lora)
    set_widget(lora_node, "1", 1.0)
    audio_vae_node = node_by_id(workflow, "4010")
    ensure_widget_input(audio_vae_node, "ckpt_name")
    set_widget(audio_vae_node, "0", args.audio_vae_checkpoint)
    text_node = node_by_id(workflow, "5023")
    ensure_widget_input(text_node, "text_encoder")
    ensure_widget_input(text_node, "ckpt_name")
    ensure_widget_input(text_node, "device")
    set_widget(text_node, "0", args.text_encoder)
    set_widget(text_node, "1", args.text_encoder_checkpoint)


def patch_workflow(args, workflow: dict[str, Any], prepared: Path, comfy_dir: Path, output_prefix: str, prompt_text: str, negative_text: str, seed: int | None, guide_image: Path | None = None) -> dict[str, Any]:
    video_name = copy_to_comfy_input(prepared, comfy_dir)
    prepared_info = probe_video(prepared)
    set_widget_if_node(workflow, args.load_video_node_id, args.video_widget, video_name)

    # Guide image path: LoadImage (2004) -> ResizeImageMaskNode (5090) -> LTXVPreprocess (3336)
    # -> LTXVImgToVideoConditionOnly (3159) -> LTXAddVideoICLoRAGuide (5012) latent input.
    # When a guide image is provided, enable i2v conditioning so LTX targets the guide appearance
    # at the start of the chunk.  When there is no guide, leave the bypass True so i2v has no
    # effect and LTX generates freely.
    canvas_width = int(prepared_info["width"])
    canvas_height = int(prepared_info["height"])
    if guide_image and guide_image.exists():
        image_name = copy_guide_image_to_comfy_input(guide_image, comfy_dir, canvas_width, canvas_height)
        # Node 5019 "bypass_i2v": False = run LTXVImgToVideoConditionOnly, True = bypass it.
        try:
            bypass_node = node_by_id(workflow, "5019")
            bypass_node["widgets_values"] = [False]
        except KeyError:
            pass
    else:
        image_name = copy_reference_frame_to_comfy_input(prepared, comfy_dir)
        # Keep bypass_i2v = True (default in workflow) so i2v has no effect.
        try:
            bypass_node = node_by_id(workflow, "5019")
            bypass_node["widgets_values"] = [True]
        except KeyError:
            pass

    try:
        image_node = node_by_id(workflow, "2004")
        ensure_widget_input(image_node, "image")
        set_widget(image_node, "0", image_name)
    except KeyError:
        pass

    set_widget_if_node(workflow, args.positive_node_id, args.prompt_widget, prompt_text)
    set_widget_if_node(workflow, args.negative_node_id, args.prompt_widget, negative_text)
    set_widget_if_node(workflow, args.save_node_id, args.save_prefix_widget, output_prefix)
    if seed is not None:
        set_widget_if_node(workflow, args.seed_node_id, args.seed_widget, int(seed))

    for node_id in args.extra_save_node_id:
        set_widget_if_node(workflow, node_id, args.save_prefix_widget, output_prefix)

    bypass_optional_preview_nodes(workflow)

    # Avoid depending on the optional ComfyMath CM_FloatToInt node; the audio latent node accepts
    # a normal integer widget value when its frame_rate link is cleared.
    try:
        clear_input_link(workflow, "3980", "frame_rate")
        audio_latent_node = node_by_id(workflow, "3980")
        set_widget(audio_latent_node, "1", int(round(float(prepared_info.get("fps") or 24))))
    except KeyError:
        pass

    try:
        latent_video_node = node_by_id(workflow, "3059")
        for input_name in ("width", "height", "length"):
            clear_input_link(workflow, "3059", input_name)
        set_widget(latent_video_node, "0", int(prepared_info["width"]))
        set_widget(latent_video_node, "1", int(prepared_info["height"]))
        set_widget(latent_video_node, "2", int(prepared_info["frames"]))
        set_widget(latent_video_node, "3", 1)
    except KeyError:
        pass

    try:
        if args.save_node_id != "5076":
            set_input_link(workflow, args.save_node_id, "images", 13594)
    except KeyError:
        pass

    if args.model_backend == "gguf":
        patch_lightweight_gguf(workflow, args)
    else:
        model_patches = {
            "3940": ("0", "ltx-2.3-22b-dev-fp8.safetensors"),
            "4010": ("0", "ltx-2.3-22b-dev-fp8.safetensors"),
            "5023": ("0", args.text_encoder),
            "5011": ("0", args.outpaint_lora),
            "4922": ("0", "ltx-2.3-22b-distilled-lora-384.safetensors"),
        }
        for node_id, (widget, value) in model_patches.items():
            try:
                set_widget(node_by_id(workflow, node_id), widget, value)
            except KeyError:
                pass

    # ARP prepares the black target canvas itself, so disable the workflow's demo padding node if present.
    try:
        pad_node = node_by_id(workflow, "5086")
        if isinstance(pad_node.get("widgets_values"), list) and len(pad_node["widgets_values"]) >= 4:
            pad_node["widgets_values"][0:4] = [0, 0, 0, 0]
    except KeyError:
        pass

    return workflow_to_prompt(workflow, args.output_node_id)


def raw_signature(args, workflow_path: Path, prepared: Path, seed: int | None = None, prompt_suffix: str = "", negative_suffix: str = "", guide_image: Path | None = None, auto_guide: bool = False, chunk_manifest: Path | None = None) -> dict[str, Any]:
    prompt_text = combine_prompt(args.prompt, prompt_suffix)
    negative_text = combine_prompt(args.negative_prompt, negative_suffix)
    return {
        "version": 22,
        "tool": "outpaint_video.py/raw_comfy",
        "prepared": root_relative(prepared),
        "prepared_fingerprint": file_fingerprint(prepared),
        "workflow": root_relative(workflow_path),
        "workflow_fingerprint": file_fingerprint(workflow_path),
        "target_aspect": args.target_aspect,
        "delivery_target_height": args.target_height,
        "model_size_multiple": MODEL_SIZE_MULTIPLE,
        "prompt": prompt_text,
        "prompt_suffix": prompt_suffix,
        "negative_suffix": negative_suffix,
        "guide_image": root_relative(guide_image) if guide_image else "",
        "guide_fingerprint": file_fingerprint(guide_image) if guide_image and guide_image.exists() else None,
        "guide_via_i2v_conditioning": bool(guide_image),
        "auto_guide_from_previous_chunk": auto_guide,
        "seed": seed,
        "negative_prompt": negative_text,
        "load_video_node_id": args.load_video_node_id,
        "save_node_id": args.save_node_id,
        "extra_save_node_id": args.extra_save_node_id,
        "output_node_id": args.output_node_id,
        "model_backend": args.model_backend,
        "gguf_model": args.gguf_model,
        "video_vae": args.video_vae,
        "outpaint_lora": args.outpaint_lora,
        "chunk_seconds": args.chunk_seconds,
        "overlap_frames": args.overlap_frames,
        "chunk_manifest": root_relative(chunk_manifest) if chunk_manifest else "",
        "chunk_manifest_fingerprint": file_fingerprint(chunk_manifest) if chunk_manifest and chunk_manifest.exists() else None,
    }


def newest_output(files: list[Path]) -> Path:
    videos = [path for path in files if path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}]
    candidates = videos or files
    if not candidates:
        raise RuntimeError("ComfyUI completed but did not report an output file.")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def chunk_ranges(prepared: Path, chunk_seconds: float, overlap_frames: int) -> list[tuple[int, int, int]]:
    info = probe_video(prepared)
    total_frames = int(info["frames"])
    if chunk_seconds <= 0 or total_frames <= 0:
        return [(0, 0, total_frames)]
    chunk_frames = max(1, int(round(chunk_seconds * info["fps"])))
    if chunk_frames >= total_frames:
        return [(0, 0, total_frames)]
    overlap = max(0, min(int(overlap_frames), chunk_frames - 1))
    step = max(1, chunk_frames - overlap)
    ranges: list[tuple[int, int, int]] = []
    start = 0
    while start < total_frames:
        end = min(total_frames, start + chunk_frames)
        ranges.append((len(ranges), start, end))
        if end >= total_frames:
            break
        start += step
    return ranges


def combine_prompt(prompt: str, suffix: str) -> str:
    return " ".join(part.strip() for part in (prompt, suffix) if part and part.strip())


def default_chunk_manifest(source: Path, aspect: str, width: int, height: int, args) -> Path:
    crops = [int(getattr(args, key, 0) or 0) for key in ("crop_left", "crop_right", "crop_top", "crop_bottom")]
    crop = "" if not any(crops) else f"_crop{crops[0]}-{crops[1]}-{crops[2]}-{crops[3]}"
    return ROOT / "manifests" / "outpaint_chunks" / f"{safe_stem(source.name)}_{aspect_slug(aspect)}_{width}x{height}{crop}_chunks.csv"


def read_chunk_manifest(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            if not row.get("chunk_index", "").isdigit():
                continue
            # Migrate old field names written before the anchor→guide rename.
            if "anchor_image" in row and "guide_image" not in row:
                row["guide_image"] = row.pop("anchor_image", "")
            row.pop("anchor_position", None)
            row.pop("anchor_seconds", None)
            rows[int(row["chunk_index"])] = row
        return rows


def write_chunk_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "chunk_index",
        "start_frame",
        "end_frame",
        "start_seconds",
        "end_seconds",
        "custom_seconds",
        "seed",
        "prompt_suffix",
        "negative_suffix",
        "guide_image",
        "prepared_path",
        "raw_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def chunk_ranges_from_manifest(total_frames: int, fps: float, default_seconds: float, overlap_frames: int, existing: dict[int, dict[str, str]]) -> list[tuple[int, int, int]]:
    if default_seconds <= 0 or total_frames <= 0:
        return [(0, 0, total_frames)]
    ranges: list[tuple[int, int, int]] = []
    start = 0
    index = 0
    while start < total_frames:
        seconds = default_seconds
        custom = existing.get(index, {}).get("custom_seconds", "")
        if custom:
            try:
                seconds = float(custom)
            except ValueError:
                seconds = default_seconds
        chunk_frames = max(1, int(round(seconds * fps)))
        end = min(total_frames, start + chunk_frames)
        ranges.append((index, start, end))
        if end >= total_frames:
            break
        overlap = max(0, min(int(overlap_frames), chunk_frames - 1))
        start += max(1, chunk_frames - overlap)
        index += 1
    return ranges


def sync_chunk_manifest(path: Path, ranges: list[tuple[int, int, int]], fps: float, chunk_dir: Path, default_seed: int) -> dict[int, dict[str, str]]:
    existing = read_chunk_manifest(path)
    rows: list[dict[str, str]] = []
    for chunk_index, start_frame, end_frame in ranges:
        row = dict(existing.get(chunk_index, {}))
        row.update(
            {
                "chunk_index": str(chunk_index),
                "start_frame": str(start_frame),
                "end_frame": str(end_frame),
                "start_seconds": f"{start_frame / fps:.6f}",
                "end_seconds": f"{end_frame / fps:.6f}",
                "prepared_path": root_relative(chunk_dir / f"prepared_{chunk_index:04d}_{start_frame:06d}_{end_frame:06d}.mp4"),
                "raw_path": root_relative(chunk_dir / f"raw_{chunk_index:04d}_{start_frame:06d}_{end_frame:06d}.mp4"),
            }
        )
        if not row.get("seed"):
            row["seed"] = str(default_seed + chunk_index)
        row.setdefault("prompt_suffix", "")
        row.setdefault("negative_suffix", "")
        row.setdefault("guide_image", "")
        row.setdefault("custom_seconds", "")
        rows.append(row)
    write_chunk_manifest(path, rows)
    return {int(row["chunk_index"]): row for row in rows}


def split_chunk(ffmpeg: str, prepared: Path, chunk_path: Path, start_frame: int, end_frame: int, fps: float, force: bool) -> None:
    if chunk_path.exists() and not force:
        return
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    partial = chunk_path.with_suffix(chunk_path.suffix + ".partial" + chunk_path.suffix)
    vf = f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=N/({fps:.8f}*TB),fps={fps:.8f},setsar=1"
    subprocess.run([ffmpeg, "-y", "-i", str(prepared), "-vf", vf, "-an", "-r", f"{fps:.8f}", "-fps_mode", "cfr", "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", str(partial)], check=True)
    replace_with_retry(partial, chunk_path, f"Prepared chunk {chunk_path.name}")




def make_piece(ffmpeg: str, source: Path, target: Path, start_frame: int, frame_count: int, fps: float) -> None:
    vf = f"trim=start_frame={start_frame}:end_frame={start_frame + frame_count},setpts=N/({fps:.8f}*TB),fps={fps:.8f},setsar=1"
    subprocess.run([ffmpeg, "-y", "-i", str(source), "-vf", vf, "-an", "-r", f"{fps:.8f}", "-fps_mode", "cfr", "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", str(target)], check=True)


def make_gap_piece(ffmpeg: str, source: Path, target: Path, frame_count: int, fps: float) -> None:
    if frame_count <= 0:
        return
    info = probe_video(source)
    last = max(0, int(info["frames"]) - 1)
    duration = frame_count / fps
    vf = f"trim=start_frame={last}:end_frame={last + 1},setpts=N/({fps:.8f}*TB),tpad=stop_mode=clone:stop_duration={duration:.8f},trim=end_frame={frame_count},fps={fps:.8f},setsar=1"
    subprocess.run([ffmpeg, "-y", "-i", str(source), "-vf", vf, "-an", "-r", f"{fps:.8f}", "-fps_mode", "cfr", "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", str(target)], check=True)


def stitch_chunks(ffmpeg: str, chunks: list[Path], ranges: list[tuple[int, int, int]], output: Path, fps: float, force: bool) -> None:
    if output.exists() and not force:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    if not chunks:
        raise RuntimeError("No outpaint chunks were produced.")
    if len(chunks) != len(ranges):
        raise RuntimeError(f"Chunk/range mismatch: {len(chunks)} chunks for {len(ranges)} ranges.")
    total_frames = ranges[-1][2]
    with tempfile.TemporaryDirectory(prefix="arp_stitch_") as tmp_text:
        tmp = Path(tmp_text)
        list_file = tmp / "chunks.txt"
        piece_paths: list[Path] = []
        cursor = 0
        previous_piece: Path | None = None
        for index, (chunk, (_chunk_index, start_frame, end_frame)) in enumerate(zip(chunks, ranges)):
            raw_frames = int(probe_video(chunk)["frames"])
            expected_frames = end_frame - start_frame
            print(f"Stitch chunk {index + 1}/{len(chunks)}: source frames {start_frame}-{end_frame}, expected {expected_frames}, got {raw_frames}", flush=True)
            if cursor < start_frame:
                gap = start_frame - cursor
                print(f"Outpaint chunk gap before chunk {index + 1}: filling {gap} frame(s) by holding the previous frame. Increase overlap to at least {gap + 1} to avoid this.", flush=True)
                if previous_piece is None:
                    raise RuntimeError(f"First outpaint chunk starts after frame 0: {start_frame}")
                gap_piece = tmp / f"gap_{index:04d}_{cursor:06d}_{start_frame:06d}.mp4"
                make_gap_piece(ffmpeg, previous_piece, gap_piece, gap, fps)
                piece_paths.append(gap_piece)
                previous_piece = gap_piece
                cursor = start_frame
            trim_start = max(0, cursor - start_frame)
            available = max(0, raw_frames - trim_start)
            if available <= 0:
                print(f"Skipping exhausted outpaint chunk {index + 1}: trim_start={trim_start}, raw_frames={raw_frames}", flush=True)
                continue
            piece = tmp / f"piece_{index:04d}_{cursor:06d}.mp4"
            make_piece(ffmpeg, chunk, piece, trim_start, available, fps)
            piece_paths.append(piece)
            previous_piece = piece
            cursor += available
        if cursor < total_frames:
            gap = total_frames - cursor
            print(f"Outpaint final gap: filling {gap} frame(s) by holding the last frame.", flush=True)
            if previous_piece is None:
                raise RuntimeError("No usable outpaint chunk frames were produced.")
            gap_piece = tmp / f"gap_final_{cursor:06d}_{total_frames:06d}.mp4"
            make_gap_piece(ffmpeg, previous_piece, gap_piece, gap, fps)
            piece_paths.append(gap_piece)
            cursor = total_frames
        list_file.write_text("".join(f"file '{path.as_posix()}'\n" for path in piece_paths), encoding="utf-8")
        partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
        subprocess.run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-vf", f"setpts=N/({fps:.8f}*TB),fps={fps:.8f},setsar=1", "-an", "-r", f"{fps:.8f}", "-fps_mode", "cfr", "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", str(partial)], check=True)
        replace_with_retry(partial, output, f"Stitched outpaint video {output.name}")


def build_parser() -> argparse.ArgumentParser:
    config = load_local_config()
    parser = argparse.ArgumentParser(description="Run the LTX IC-LoRA outpainting stage end to end.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target-aspect", default="16:9")
    parser.add_argument("--target-height", type=int, default=720)
    parser.add_argument("--crop-left", type=int, default=0)
    parser.add_argument("--crop-right", type=int, default=0)
    parser.add_argument("--crop-top", type=int, default=0)
    parser.add_argument("--crop-bottom", type=int, default=0)
    parser.add_argument("--chunk-seconds", type=float, default=20.0, help="Outpaint in chunks of roughly this many seconds. Use 0 to send the full clip.")
    parser.add_argument("--overlap-frames", type=int, default=8, help="Frames repeated between neighbouring chunks before stitching.")
    parser.add_argument("--chunk-manifest", help="CSV storing per-chunk seed, prompt, and guide image overrides.")
    parser.add_argument("--only-chunk", type=int, help="Regenerate only one outpaint chunk, then restitch from existing chunks.")
    parser.add_argument("--model-backend", choices=["gguf", "checkpoint"], default="gguf")
    parser.add_argument("--gguf-model", default="LTX-2.3-distilled-Q4_K_M.gguf")
    parser.add_argument("--video-vae", default="LTX23_video_vae_bf16.safetensors")
    parser.add_argument("--audio-vae-checkpoint", default="ltx-2.3-22b-dev-fp8.safetensors")
    parser.add_argument("--text-encoder", default="gemma_3_12B_it_fp8_scaled.safetensors")
    parser.add_argument("--text-encoder-checkpoint", default="ltx-2.3-22b-dev-fp8.safetensors")
    parser.add_argument("--outpaint-lora", default="ltx-2.3-22b-ic-lora-outpaint.safetensors")
    parser.add_argument("--output")
    parser.add_argument("--raw-output")
    parser.add_argument("--workflow", default=str(DEFAULT_WORKFLOW))
    parser.add_argument("--comfy-dir", default=config.get("comfy_dir", str(DEFAULT_COMFY_DIR)))
    parser.add_argument("--comfy-url", default=config.get("comfy_url", "http://127.0.0.1:8188"))
    parser.add_argument("--comfy-output-root")
    parser.add_argument("--load-video-node-id", default="5060")
    parser.add_argument("--video-widget", default="video")
    parser.add_argument("--save-node-id", default="5076")
    parser.add_argument("--extra-save-node-id", action="append", default=["5069"])
    parser.add_argument("--save-prefix-widget", default="filename_prefix")
    parser.add_argument("--output-node-id", default="5076")
    parser.add_argument("--positive-node-id", default="2483")
    parser.add_argument("--negative-node-id", default="2612")
    parser.add_argument("--prompt-widget", default="0")
    parser.add_argument("--prompt", default="naturalistic period film footage, coherent background extension, preserve camera motion, realistic cinematic lighting")
    parser.add_argument("--seed-node-id", default="4832")
    parser.add_argument("--seed-widget", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-prompt", default="cartoon, game, 3d render, still image, static, warped geometry, flicker, smeared details, extra fingers, broken fingers, deformed hands")
    parser.add_argument("--black-lift", type=float, default=0.018)
    parser.add_argument("--gamma", type=float, default=1.06)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = resolve_path(args.source)
    workflow_path = resolve_path(args.workflow)
    comfy_dir = resolve_path(args.comfy_dir)
    comfy_output_root = resolve_path(args.comfy_output_root) if args.comfy_output_root else comfy_dir / "output"

    if not source.exists():
        raise FileNotFoundError(f"Source video not found: {source}")
    if not workflow_path.exists():
        raise FileNotFoundError(f"Outpainting workflow not found: {workflow_path}")
    if not (comfy_dir / "main.py").exists():
        raise FileNotFoundError(f"ComfyUI main.py not found: {comfy_dir / 'main.py'}")
    if args.model_backend == "gguf" and not (comfy_dir / "custom_nodes" / "ComfyUI-GGUF").exists():
        raise FileNotFoundError(f"ComfyUI-GGUF is required for lightweight outpainting. Re-run install_windows.bat, then restart ComfyUI: {comfy_dir / 'custom_nodes' / 'ComfyUI-GGUF'}")

    output = resolve_path(args.output) if args.output else default_output(source, args.target_aspect, args.target_height, args)
    raw_output = resolve_path(args.raw_output) if args.raw_output else default_raw_output(source, args.target_aspect, args.target_height, args)
    delivery_width, delivery_height = target_size(source, args.target_aspect, args.target_height)
    work_width, work_height = model_safe_size(source, args.target_aspect, args.target_height)
    prepared = prepared_for(source, args.target_aspect, args.target_height, args)

    if not args.dry_run:
        ensure_outpaint_models(comfy_dir)
        print(f"Checking ComfyUI outpainting nodes at {args.comfy_url}...", flush=True)
        wait_for_comfy(args.comfy_url, timeout_seconds=180, poll_seconds=args.poll_seconds)
        required_nodes = dict(OUTPAINT_REQUIRED_NODES)
        if args.model_backend == "gguf":
            required_nodes["UnetLoaderGGUF"] = "ComfyUI-GGUF"
        ensure_node_types(args.comfy_url, required_nodes, "outpainting workflow")

    prepare_command = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_outpaint_input.py"),
        "--source",
        str(source),
        "--target-aspect",
        args.target_aspect,
        "--black-lift",
        str(args.black_lift),
        "--gamma",
        str(args.gamma),
        "--output",
        str(prepared),
        "--crop-left",
        str(args.crop_left),
        "--crop-right",
        str(args.crop_right),
        "--crop-top",
        str(args.crop_top),
        "--crop-bottom",
        str(args.crop_bottom),
    ]
    prepare_command += ["--target-height", str(work_height)]
    prepare_command += ["--target-width", str(work_width)]
    if args.force:
        prepare_command.append("--force")
    if args.dry_run:
        prepare_command.append("--dry-run")
    if (work_width, work_height) != (delivery_width, delivery_height):
        print(
            f"LTX working canvas rounded to {work_width}x{work_height} "
            f"from requested delivery {delivery_width}x{delivery_height}.",
            flush=True,
        )
    print(f"Preparing expanded outpaint canvas: {work_width}x{work_height}, aspect {args.target_aspect}, black_lift={args.black_lift}, gamma={args.gamma}", flush=True)
    run_command(prepare_command, False)

    output_prefix = f"arp_outpaint/{safe_stem(source.name)}_{aspect_slug(args.target_aspect)}_{work_width}x{work_height}"
    print(f"Prepared expanded canvas for ComfyUI: {prepared}", flush=True)
    if not args.dry_run:
        ffmpeg = find_ffmpeg()
        chunk_dir = ROOT / ".cache" / "outpaint_chunks" / f"{safe_stem(source.name)}_{aspect_slug(args.target_aspect)}_{work_width}x{work_height}{crop_slug(args)}"
        chunk_manifest = resolve_path(args.chunk_manifest) if args.chunk_manifest else default_chunk_manifest(source, args.target_aspect, work_width, work_height, args)
        prepared_info = probe_video(prepared)
        chunk_existing = read_chunk_manifest(chunk_manifest)
        ranges = chunk_ranges_from_manifest(int(prepared_info["frames"]), float(prepared_info["fps"]), args.chunk_seconds, args.overlap_frames, chunk_existing)
        chunk_overrides = sync_chunk_manifest(chunk_manifest, ranges, float(prepared_info["fps"]), chunk_dir, args.seed)
        print(f"Outpaint chunk manifest: {chunk_manifest}", flush=True)
        raw_sig = raw_signature(args, workflow_path, prepared, chunk_manifest=chunk_manifest)  # outer whole-run signature
        if args.only_chunk is None and not args.force and resumable_output(raw_output, raw_sig, video_like=prepared):
            print(f"Reuse raw Comfy render: {raw_output}", flush=True)
        else:
            print(f"ComfyUI is ready at {args.comfy_url}.", flush=True)
            print(f"Splitting prepared canvas into {len(ranges)} chunk(s): {args.chunk_seconds:g}s chunks, {args.overlap_frames} overlap frame(s)", flush=True)
            if len(ranges) > 1 and args.overlap_frames < RECOMMENDED_OVERLAP_FRAMES:
                print(
                    f"Warning: overlap is {args.overlap_frames} frame(s). LTX can return short chunks; "
                    f"{RECOMMENDED_OVERLAP_FRAMES}+ overlap frames is recommended to avoid held-frame seams.",
                    flush=True,
                )
            raw_chunks: list[Path] = []
            effective_ranges: list[tuple[int, int, int]] = []
            for range_index, (chunk_index, start_frame, end_frame) in enumerate(ranges):
                chunk_prepared = chunk_dir / f"prepared_{chunk_index:04d}_{start_frame:06d}_{end_frame:06d}.mp4"
                chunk_raw = chunk_dir / f"raw_{chunk_index:04d}_{start_frame:06d}_{end_frame:06d}.mp4"
                print(f"Outpaint chunk {chunk_index + 1}/{len(ranges)}: frames {start_frame}-{end_frame}", flush=True)
                split_chunk(ffmpeg, prepared, chunk_prepared, start_frame, end_frame, float(prepared_info["fps"] or 24.0), args.force)
                previous_raw = raw_chunks[-1] if raw_chunks else None
                chunk_row = chunk_overrides.get(chunk_index, {})
                chunk_seed = int(chunk_row.get("seed") or args.seed + chunk_index)
                chunk_prompt_suffix = chunk_row.get("prompt_suffix", "")
                chunk_negative_suffix = chunk_row.get("negative_suffix", "")

                # Determine guide image for i2v conditioning.
                # Priority: (1) explicit guide_image set in manifest, (2) auto-extract last
                # frame of previous raw chunk for temporal continuity, (3) no guide (first chunk).
                guide_text = chunk_row.get("guide_image", "")
                explicit_guide: Path | None = resolve_path(guide_text) if guide_text else None
                if explicit_guide and not explicit_guide.exists():
                    print(f"Warning: chunk {chunk_index + 1} guide image not found and will be ignored: {explicit_guide}", flush=True)
                    explicit_guide = None
                auto_guide: bool = False
                guide_image: Path | None = explicit_guide
                if guide_image is None and previous_raw is not None and previous_raw.exists():
                    try:
                        guide_image = extract_last_frame_as_guide(previous_raw, chunk_dir)
                        auto_guide = True
                        print(f"Chunk {chunk_index + 1}: auto-guide from last frame of chunk {chunk_index}", flush=True)
                    except Exception as exc:
                        print(f"Warning: could not extract auto-guide from previous chunk: {exc}", flush=True)

                chunk_sig = raw_signature(args, workflow_path, chunk_prepared, chunk_seed, chunk_prompt_suffix, chunk_negative_suffix, guide_image, auto_guide)
                if args.only_chunk is not None and chunk_index != args.only_chunk:
                    if not chunk_raw.exists():
                        if chunk_index < args.only_chunk:
                            # A chunk before the target is missing — we can't auto-guide or stitch correctly.
                            raise FileNotFoundError(
                                f"Cannot regenerate chunk {args.only_chunk + 1}; "
                                f"earlier chunk {chunk_index + 1} is missing: {chunk_raw}"
                            )
                        # A chunk after the target doesn't exist yet — skip it.
                        # Stitching will only cover frames up to the last available chunk.
                        print(f"Chunk {chunk_index + 1} not yet generated; stitching will stop at chunk {args.only_chunk + 1}.", flush=True)
                        continue
                    raw_chunks.append(chunk_raw)
                    effective_ranges.append((chunk_index, start_frame, end_frame))
                    continue
                if not args.force and resumable_output(chunk_raw, chunk_sig, video_like=chunk_prepared):
                    print(f"Reuse raw Comfy chunk: {chunk_raw}", flush=True)
                    raw_chunks.append(chunk_raw)
                    effective_ranges.append((chunk_index, start_frame, end_frame))
                    continue
                workflow = json.loads(workflow_path.read_text(encoding="utf-8-sig"))
                chunk_prefix = f"{output_prefix}_chunk_{chunk_index:04d}"
                prompt_text = combine_prompt(args.prompt, chunk_prompt_suffix)
                negative_text = combine_prompt(args.negative_prompt, chunk_negative_suffix)
                print(f"Chunk {chunk_index + 1} seed: {chunk_seed}", flush=True)
                if chunk_prompt_suffix:
                    print(f"Chunk {chunk_index + 1} prompt suffix: {chunk_prompt_suffix}", flush=True)
                if chunk_negative_suffix:
                    print(f"Chunk {chunk_index + 1} negative suffix: {chunk_negative_suffix}", flush=True)
                if guide_image:
                    source = "explicit" if explicit_guide else "auto"
                    print(f"Chunk {chunk_index + 1} guide frame ({source}): {guide_image}", flush=True)
                prompt = patch_workflow(args, workflow, chunk_prepared, comfy_dir, chunk_prefix, prompt_text, negative_text, chunk_seed, guide_image)
                prompt_id = queue_prompt(args.comfy_url, prompt)
                print(f"Queued ComfyUI prompt: {prompt_id}", flush=True)
                history = wait_for_prompt(args.comfy_url, prompt_id, args.poll_seconds)
                produced = newest_output(extract_output_files(history, comfy_output_root))
                chunk_raw.parent.mkdir(parents=True, exist_ok=True)
                chunk_tmp = chunk_raw.with_suffix(chunk_raw.suffix + ".partial")
                shutil.copy2(produced, chunk_tmp)
                replace_with_retry(chunk_tmp, chunk_raw, f"Outpaint chunk {chunk_index + 1}")
                write_signature(chunk_raw, chunk_sig)
                print(f"Wrote raw Comfy chunk: {chunk_raw}", flush=True)
                raw_chunks.append(chunk_raw)
                effective_ranges.append((chunk_index, start_frame, end_frame))
            restitched = True
            try:
                stitch_chunks(ffmpeg, raw_chunks, effective_ranges, raw_output, float(prepared_info["fps"] or 24.0), True)
            except PermissionError as exc:
                if args.only_chunk is None:
                    raise
                restitched = False
                print(
                    f"Warning: regenerated chunk {args.only_chunk + 1}, but could not replace the stitched raw outpaint video because it is open in another process: {raw_output}",
                    flush=True,
                )
                print("Close any preview/player using that video, then run Outpainting or regenerate the chunk again to restitch.", flush=True)
            if restitched:
                write_signature(raw_output, raw_sig)
                print(f"Wrote raw Comfy render: {raw_output}", flush=True)
            elif args.only_chunk is not None:
                return 0

    finalize_command = [
        sys.executable,
        str(ROOT / "scripts" / "finalize_outpaint_output.py"),
        "--source",
        str(raw_output),
        "--output",
        str(output),
        "--black-lift",
        str(args.black_lift),
        "--gamma",
        str(args.gamma),
        "--target-width",
        str(delivery_width),
        "--target-height",
        str(delivery_height),
    ]
    if args.force:
        finalize_command.append("--force")
    if args.dry_run:
        finalize_command.append("--dry-run")
    run_command(finalize_command, args.dry_run)
    print(f"Wrote outpainted video: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
