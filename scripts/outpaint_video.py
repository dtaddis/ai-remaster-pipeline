from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from comfy_api import extract_output_files, ensure_node_types, node_by_id, queue_prompt, set_widget, wait_for_comfy, wait_for_prompt, workflow_to_prompt
from common import (
    QWEN_IMAGE_EDIT_MODEL,
    ROOT,
    copy_to_comfy_input,
    file_fingerprint,
    find_ffmpeg,
    load_local_config,
    newest_output as newest_comfy_output,
    replace_unless_identical,
    replace_with_retry,
    resolve_path,
    root_relative,
    resumable_output,
    safe_stem,
    split_matches_source,
    write_signature,
    write_split_sidecar,
)
from dependency_manager import ensure_outpaint_models
from prepare_outpaint_input import default_output as default_prepared_output
from prepare_outpaint_input import even, parse_aspect, probe_video
from qwen_seed_guides import DEFAULT_SEED_PROMPT, seed_guides
import artifact_ids as aid


def _crop_black(args: Any | None) -> tuple[list[int], bool]:
    if args is None:
        return [0, 0, 0, 0], False
    crop = [int(getattr(args, key, 0) or 0) for key in ("crop_left", "crop_right", "crop_top", "crop_bottom")]
    return crop, bool(getattr(args, "outpaint_all_black_regions", False))


DEFAULT_WORKFLOW = ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json"
DEFAULT_COMFY_DIR = ROOT / "tools" / "comfyui"
DEFAULT_OUTPAINT_PROMPT = "outpaint"
RECOMMENDED_OVERLAP_FRAMES = 8
MODEL_SIZE_MULTIPLE = 32
OUTPAINT_REQUIRED_NODES = {
    "LTXVImgToVideoConditionOnly": "ComfyUI-LTXVideo",
    "LTXAddVideoICLoRAGuide": "ComfyUI-LTXVideo",
    "LTXVPreprocess": "ComfyUI-LTXVideo",
    "VHS_LoadVideo": "ComfyUI-VideoHelperSuite",
    "VHS_VideoCombine": "ComfyUI-VideoHelperSuite",
}


def aspect_slug(value: str) -> str:
    return value.replace(":", "x").replace(".", "_")


def target_size(source: Path, aspect: str, target_height: int | None) -> tuple[int, int]:
    # Delivery resolution from the (already-resolved) target height. Centralised in artifact_ids
    # so the GUI's outpaint_size_for_source and this stay identical.
    height = int(target_height or 720)
    return aid.delivery_size(height, aspect, str(height))


def model_safe(value: int, multiple: int = MODEL_SIZE_MULTIPLE) -> int:
    return aid.model_safe(value, multiple)


def model_safe_size(source: Path, aspect: str, target_height: int | None) -> tuple[int, int]:
    height = int(target_height or 720)
    return aid.work_size(height, aspect, str(height))


def crop_slug(args: Any) -> str:
    values = [int(getattr(args, key, 0)) for key in ("crop_left", "crop_right", "crop_top", "crop_bottom")]
    crop = "" if not any(values) else f"_crop{values[0]}-{values[1]}-{values[2]}-{values[3]}"
    mode = "_allblack" if getattr(args, "outpaint_all_black_regions", False) else ""
    return crop + mode


def default_output(source: Path, aspect: str, target_height: int | None, args: Any | None = None) -> Path:
    width, height = model_safe_size(source, aspect, target_height)
    crop, black = _crop_black(args)
    return ROOT / "intermediate" / "outpainted" / aid.outpaint_name(source.name, aspect, width, height, crop, black, "outpaint", "mp4")


def default_raw_output(source: Path, aspect: str, target_height: int | None, args: Any | None = None) -> Path:
    width, height = model_safe_size(source, aspect, target_height)
    crop, black = _crop_black(args)
    return ROOT / "intermediate" / "outpainted" / aid.outpaint_name(source.name, aspect, width, height, crop, black, "rawcomfy", "mp4")


def prepared_for(source: Path, aspect: str, target_height: int | None, args: Any | None = None) -> Path:
    # Prepare at model-safe dimensions so the canvas fed to LTX exactly matches the latent node
    # dimensions.  LTXVPreprocess crops (not scales) to fit the latent, so any mismatch would
    # silently remove rows/columns of pixels.  Recomposition later upscales from model-safe
    # dimensions (e.g. 704) to delivery resolution (e.g. 720) when producing the final master.
    work_w, work_h = model_safe_size(source, aspect, target_height)
    crop, black = _crop_black(args)
    return ROOT / "intermediate" / "outpaint_prepared" / aid.outpaint_name(source.name, aspect, work_w, work_h, crop, black, "prepared", "mp4")


def run_command(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


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


def _inpaint_black_corners(canvas_bgr: "np.ndarray", black_thresh: int = 4) -> "np.ndarray":
    """Fill any remaining near-black pixels in *canvas_bgr* using OpenCV inpainting.

    Uses a binary mask of pixels whose every channel is ≤ *black_thresh* (pure padding
    black), then applies cv2.INPAINT_TELEA which is fast and accurate for small regions.
    Returns the inpainted array (same shape/dtype as input).
    """
    import cv2
    import numpy as np

    mask = np.all(canvas_bgr <= black_thresh, axis=2).astype(np.uint8) * 255
    if not mask.any():
        return canvas_bgr
    inpainted = cv2.inpaint(canvas_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted


def copy_guide_image_to_comfy_input(
    guide: Path,
    comfy_dir: Path,
    canvas_width: int = 0,
    canvas_height: int = 0,
    source_frame: "np.ndarray | None" = None,
) -> str:
    """Copy a guide image to ComfyUI's input folder, stretched to exactly the LTX canvas size.

    Always stretches to (canvas_width × canvas_height) — no letterboxing, no cropping.
    This ensures the guide is pixel-aligned with the prepared canvas so LTXVImgToVideoConditionOnly
    receives a properly-sized image.

    A warning is printed when the aspect-ratio difference exceeds 10%, which typically
    indicates a mismatched source (e.g. a portrait image used as a landscape guide).
    The stretch still proceeds — the caller is responsible for supplying a sensible guide.
    """
    from PIL import Image as PILImage

    target_dir = comfy_dir / "input" / "arp_outpaint"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Content-keyed name: different guides can share a stem across chunk caches and projects
    # (e.g. guide_prev_raw_0000_000000_000480.png exists for every project with the same
    # chunking), so the digest is what stops a cached copy from feeding another guide's
    # frames to LTX.
    digest = file_fingerprint(guide)["sha256"][:12]
    if canvas_width > 0 and canvas_height > 0:
        target = target_dir / f"guide_{guide.stem}_{digest}_{canvas_width}x{canvas_height}.png"
    else:
        target = target_dir / f"guide_{guide.stem}_{digest}{guide.suffix.lower()}"

    try:
        if target.exists() and target.stat().st_size > 0:
            return f"arp_outpaint/{target.name}"
    except OSError:
        pass

    if canvas_width > 0 and canvas_height > 0:
        with PILImage.open(guide) as img:
            img_w, img_h = img.size
            if img_w == canvas_width and img_h == canvas_height:
                shutil.copy2(guide, target)
            else:
                resampling = getattr(PILImage, "Resampling", PILImage).LANCZOS
                img_ar = img_w / img_h
                canvas_ar = canvas_width / canvas_height
                ar_diff = abs(img_ar - canvas_ar) / canvas_ar
                if ar_diff > 0.10:
                    print(
                        f"Warning: guide AR {img_ar:.3f} ({img_w}x{img_h}) differs from canvas AR "
                        f"{canvas_ar:.3f} ({canvas_width}x{canvas_height}) by {ar_diff * 100:.1f}%. "
                        f"Stretching anyway — check that the correct guide image is being used.",
                        flush=True,
                    )
                else:
                    print(
                        f"Guide stretched {img_w}x{img_h} -> {canvas_width}x{canvas_height}",
                        flush=True,
                    )
                resized = img.convert("RGB").resize((canvas_width, canvas_height), resampling)
                resized.save(target, format="PNG")
    else:
        shutil.copy2(guide, target)

    return f"arp_outpaint/{target.name}"


def extract_last_frame_as_guide(previous_raw: Path, chunk_dir: Path) -> Path:
    """Extract the last frame of a finished raw chunk as a PNG for i2v guide conditioning.

    Raw chunks are at model-safe dimensions (e.g. 1280×704), which matches the canvas
    used for the next chunk.  No resize is performed; copy_guide_image_to_comfy_input
    handles any final stretching to canvas size.
    """
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

    # Dearchive's demo graph uses a Crystools latent switch to optionally let source audio drive
    # the video. Its default is the ordinary empty-audio latent, so route that branch directly
    # into the AV concat node and keep Clean Up runnable on ARP's minimal node installation.
    try:
        empty_audio_link = input_link(workflow, "5112", "on_false")
        if empty_audio_link is not None:
            links = {int(link[0]): link for link in workflow.get("links", [])}
            source_link = links.get(int(empty_audio_link))
            if source_link is not None:
                patch_link(workflow, 13649, int(source_link[1]), int(source_link[2]), 4528, 1, "LATENT")
                set_input_link(workflow, "4528", "audio_latent", 13649)
    except KeyError:
        pass


def bypass_demo_padding_node(workflow: dict[str, Any]) -> None:
    """Route around KJNodes' ImagePadKJ when ARP has already prepared the canvas."""
    try:
        source_link = input_link(workflow, "5086", "image")
        if source_link is not None:
            set_input_link(workflow, "5026", "input", source_link)
    except KeyError:
        pass


def bypass_conditioning_resize_nodes(workflow: dict[str, Any]) -> None:
    """Keep LTX conditioning on full-size model-safe images.

    ARP already copies guide images to the prepared canvas size. The workflow's
    demo resize branch halves the prepared source and rounds it back to a multiple of
    32, which can crop/zoom 864x480 or 1280x704 conditioning images. The i2v start
    guide should see the still guide image, while IC-LoRA must keep following the
    prepared video frames rather than the still guide.
    """
    try:
        patch_link(workflow, 13600, 2004, 0, 3336, 0, "IMAGE")
        set_input_link(workflow, "3336", "image", 13600)
    except KeyError:
        pass
    try:
        patch_link(workflow, 13589, 5060, 0, 5012, 4, "IMAGE")
        set_input_link(workflow, "5012", "image", 13589)
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
    # Author workflows have used different link IDs for the checkpoint's video-VAE outputs
    # (notably Dearchive uses 13619 for decode, while the outpaint workflow used 13348). Route
    # every former checkpoint VAE edge through the standalone VAE instead of relying only on a
    # fixed list of historical IDs.
    vae_links: list[int] = []
    for link in workflow.get("links", []):
        if int(link[1]) == 3940 and int(link[2]) == 2:
            link[1], link[2] = 9001, 0
            vae_links.append(int(link[0]))
    node_by_id(workflow, "9001")["outputs"][0]["links"] = sorted(set(vae_links) | {13279, 13348, 13405})
    patch_link(workflow, 13217, 3940, 0, 5011, 0, "MODEL")
    patch_link(workflow, 13279, 9001, 0, 3159, 0, "VAE")
    patch_link(workflow, 13348, 9001, 0, 4851, 1, "VAE")
    patch_link(workflow, 13405, 9001, 0, 5012, 2, "VAE")
    set_input_link(workflow, "5011", "model", 13217)
    lora_node = node_by_id(workflow, "5011")
    ensure_widget_input(lora_node, "lora_name")
    ensure_widget_input(lora_node, "strength_model", "FLOAT")
    set_widget(lora_node, "0", args.outpaint_lora)
    set_widget(lora_node, "1", float(getattr(args, "lora_strength", 1.0)))
    audio_vae_node = node_by_id(workflow, "4010")
    ensure_widget_input(audio_vae_node, "ckpt_name")
    set_widget(audio_vae_node, "0", args.audio_vae_checkpoint)
    text_node = node_by_id(workflow, "5023")
    ensure_widget_input(text_node, "text_encoder")
    ensure_widget_input(text_node, "ckpt_name")
    ensure_widget_input(text_node, "device")
    set_widget(text_node, "0", args.text_encoder)
    set_widget(text_node, "1", args.text_encoder_checkpoint)


def resolve_guide_coords(extra_guides: "list[dict]", num_pixel_frames: int) -> "list[dict]":
    """Resolve guide frame positions to explicit, collision-free pixel coordinates.

    LTXVAddGuide resolves a negative frame_idx by subtracting the number of keyframes
    already recorded in the conditioning, but it counts them by UNIQUE start coordinate
    (torch.unique), so any coordinate collision — two guides at the same position, or a
    guide landing on the IC-LoRA reference video's internal coordinates {0} ∪ {8m+1} —
    undercounts and shifts later negative guides past the end of the chunk, where they
    condition nothing, and makes LTXVCropGuides leave a stray guide latent in the output.
    Resolving negatives here from the chunk's true frame count, nudging 8m+1 coordinates
    down onto the free 8m grid, and dropping exact duplicates keeps the node's keyframe
    accounting exact.  Every skipped or moved guide is logged.
    """
    resolved: list[dict] = []
    seen: set[int] = set()
    for gf in extra_guides:
        fidx = int(gf.get("frame_idx", -1))
        if fidx < 0 and num_pixel_frames <= 0:
            # Chunk frame count unknown — let the Comfy node resolve the negative index.
            resolved.append(dict(gf))
            continue
        coord = fidx
        if fidx < 0:
            latent_count = (num_pixel_frames - 1) // 8 + 1
            coord = max((latent_count - 1) * 8 + 1 + fidx, 0)
        if coord > 0 and coord % 8 == 1:
            print(f"Guide frame_idx={fidx}: moved to frame {coord - 1} (coordinate {coord} collides with IC-LoRA reference conditioning)", flush=True)
            coord -= 1
        if coord == 0:
            print(f"Warning: guide frame_idx={fidx} resolves to frame 0; use the chunk's frame-0 start guide for that position. Skipping it.", flush=True)
            continue
        if coord in seen:
            print(f"Warning: guide frame_idx={fidx} duplicates another guide at frame {coord}; skipping the duplicate.", flush=True)
            continue
        seen.add(coord)
        resolved.append({**gf, "frame_idx": coord})
    return resolved


def _patch_extra_guides(
    workflow: dict[str, Any],
    args,
    extra_guides: "list[dict]",
    canvas_width: int,
    canvas_height: int,
    source_frame: "Any",
    num_pixel_frames: int = 0,
) -> None:
    """Chain one LTXVAddGuideAdvanced node per extra guide frame.

    Nodes are assigned IDs starting at 9050 (LoadImage) and 9060 (LTXVAddGuideAdvanced),
    incrementing by 1 per guide.  The first guide node redirects 5012's downstream
    consumers; each subsequent guide node redirects the previous guide node's consumers.
    """
    from pathlib import Path as _Path
    comfy_dir = _Path(args.comfy_dir)

    extra_guides = resolve_guide_coords(extra_guides, num_pixel_frames)
    if not extra_guides:
        return

    # Find the VAE source from node 5012's vae input (already resolved by GGUF patching).
    links_map = {lnk[0]: lnk for lnk in workflow.get("links", [])}
    vae_src_node: int = 3940
    vae_src_slot: int = 2
    try:
        ic_node = node_by_id(workflow, "5012")
        for inp in ic_node.get("inputs", []):
            if inp.get("name") == "vae" and inp.get("link") is not None:
                lnk = links_map[inp["link"]]
                vae_src_node, vae_src_slot = int(lnk[1]), int(lnk[2])
                break
    except KeyError:
        pass

    # These links carry 5012's outputs to downstream nodes.  The first guide node
    # will redirect them; subsequent nodes redirect the previous guide's outputs.
    REDIRECT_FROM_5012 = {13409, 13410, 13413, 13414, 13444}
    current_src = 5012  # the node whose outputs we redirect on each iteration

    new_links_all: list[list] = []
    reserved_ids = set()

    for i, gf in enumerate(extra_guides):
        load_id = 9050 + i
        guide_id = 9060 + i
        # Link ID base: 19050 + i*10
        lb = 19050 + i * 10
        img_link, vae_link, pos_link, neg_link, lat_link = lb, lb+1, lb+2, lb+3, lb+4
        reserved_ids.update({img_link, vae_link, pos_link, neg_link, lat_link})

        frame_idx = int(gf.get("frame_idx", -1))
        strength = float(gf.get("strength", 1.0))
        image_path = gf["image"]

        image_name = copy_guide_image_to_comfy_input(
            image_path, comfy_dir, canvas_width, canvas_height, source_frame=source_frame
        )

        add_or_replace_node(workflow, {
            "id": load_id,
            "type": "LoadImage",
            "title": f"Guide Frame {i} (idx={frame_idx})",
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [img_link]}],
            "widgets_values": [image_name, "image"],
        })

        add_or_replace_node(workflow, {
            "id": guide_id,
            "type": "LTXVAddGuideAdvanced",
            "title": f"Guide Frame {i} (idx={frame_idx})",
            "mode": 0,
            "inputs": [
                {"name": "positive", "type": "CONDITIONING", "link": pos_link},
                {"name": "negative", "type": "CONDITIONING", "link": neg_link},
                {"name": "vae",      "type": "VAE",          "link": vae_link},
                {"name": "latent",   "type": "LATENT",        "link": lat_link},
                {"name": "image",    "type": "IMAGE",         "link": img_link},
            ],
            "outputs": [
                {"name": "positive", "type": "CONDITIONING", "links": []},
                {"name": "negative", "type": "CONDITIONING", "links": []},
                {"name": "latent",   "type": "LATENT",        "links": []},
            ],
            "widgets_values": {
                "frame_idx": frame_idx,
                "strength": strength,
                "crf": 18,
                "blur_radius": 0,
                "interpolation": "lanczos",
                "crop": "disabled",
            },
        })

        new_links_all.extend([
            [img_link, load_id, 0, guide_id, 4, "IMAGE"],
            [vae_link, vae_src_node, vae_src_slot, guide_id, 2, "VAE"],
            [pos_link, current_src, 0, guide_id, 0, "CONDITIONING"],
            [neg_link, current_src, 1, guide_id, 1, "CONDITIONING"],
            [lat_link, current_src, 2, guide_id, 3, "LATENT"],
        ])

        # Redirect the PREVIOUS source's downstream consumers to this new node.
        prev_src = current_src
        redirect_ids = REDIRECT_FROM_5012 if prev_src == 5012 else set()
        # For nodes after the first, redirect any links that still point from prev_src
        # (these are the downstream links we haven't yet redirected).
        for lnk in workflow["links"]:
            if int(lnk[1]) == prev_src and lnk[0] not in reserved_ids and lnk[0] not in {l[0] for l in new_links_all}:
                if prev_src == 5012 and lnk[0] in redirect_ids:
                    lnk[1] = guide_id
                elif prev_src != 5012:
                    lnk[1] = guide_id

        if prev_src == 5012:
            for lnk in workflow["links"]:
                if lnk[0] in REDIRECT_FROM_5012 and int(lnk[1]) == 5012:
                    lnk[1] = guide_id

        current_src = guide_id

    # Merge new links, removing any stale entries with the same IDs.
    workflow["links"] = [l for l in workflow.get("links", []) if l[0] not in reserved_ids] + new_links_all


def patch_workflow(args, workflow: dict[str, Any], prepared: Path, comfy_dir: Path, output_prefix: str, prompt_text: str, negative_text: str, seed: int | None, guide_image: Path | None = None, extra_guides: "list[dict] | None" = None) -> dict[str, Any]:
    video_name = copy_to_comfy_input(prepared, comfy_dir, "arp_outpaint")
    prepared_info = probe_video(prepared)
    set_widget_if_node(workflow, args.load_video_node_id, args.video_widget, video_name)

    # Guide image path: LoadImage (2004) -> ResizeImageMaskNode (5090) -> LTXVPreprocess (3336)
    # -> LTXVImgToVideoConditionOnly (3159) -> LTXAddVideoICLoRAGuide (5012) latent input.
    # When a guide image is provided, enable i2v conditioning so LTX targets the guide appearance
    # at the start of the chunk.  When there is no guide, leave the bypass True so i2v has no
    # effect and LTX generates freely.
    #
    # The prepared video is at model-safe dimensions (e.g. 1280×704), so canvas, latent node,
    # and guide images all use the same dimensions — no mismatch, no crop.
    canvas_width = int(prepared_info["width"])
    canvas_height = int(prepared_info["height"])
    source_frame: "np.ndarray | None" = None
    if guide_image and guide_image.exists():
        # Extract the first frame of the prepared video to use as source fill for guide black bands.
        try:
            import cv2 as _cv2
            import numpy as _np
            _cap = _cv2.VideoCapture(str(prepared))
            _ok, _frame = _cap.read()
            _cap.release()
            if _ok and _frame is not None:
                source_frame = _frame
        except Exception as _e:
            print(f"Warning: could not extract source frame for guide compositing: {_e}", flush=True)

        image_name = copy_guide_image_to_comfy_input(guide_image, comfy_dir, canvas_width, canvas_height, source_frame=source_frame)
        # Node 5019 "bypass_i2v": False = run LTXVImgToVideoConditionOnly, True = bypass it.
        try:
            bypass_node = node_by_id(workflow, "5019")
            bypass_node["widgets_values"] = [False]
        except KeyError:
            pass
        # Apply start-guide strength to LTXVImgToVideoConditionOnly (node 3159) widget 0.
        guide_strength = getattr(args, "guide_strength", 0.7)
        try:
            i2v_node = node_by_id(workflow, "3159")
            if isinstance(i2v_node.get("widgets_values"), list) and i2v_node["widgets_values"]:
                i2v_node["widgets_values"][0] = float(guide_strength)
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
        set_widget(latent_video_node, "0", canvas_width)
        set_widget(latent_video_node, "1", canvas_height)
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

    # ARP prepares the black target canvas and guide images itself, so workflow demo resize/pad
    # nodes must not reinterpret guide geometry.
    bypass_demo_padding_node(workflow)
    bypass_conditioning_resize_nodes(workflow)

    # Clean Up uses the complete prepared clip as node 5012's video IC-LoRA guide. Keep this
    # separate from guide_strength, which controls the optional one-frame i2v guide at node 3159.
    # A value below 1 gives Dearchive room to repaint defects instead of copying them verbatim.
    source_fidelity = getattr(args, "source_fidelity", None)
    if source_fidelity is not None:
        fidelity = float(source_fidelity)
        if not 0.0 <= fidelity <= 1.0:
            raise ValueError("Source fidelity must be between 0 and 1.")
        try:
            set_widget(node_by_id(workflow, "5012"), "1", fidelity)
        except KeyError:
            pass

    # Extra guide frames via LTXVAddGuideAdvanced — inserted after GGUF patching so the VAE
    # source is already resolved.  Each guide is chained off the previous one.
    if extra_guides:
        _patch_extra_guides(workflow, args, extra_guides, canvas_width, canvas_height, source_frame, int(prepared_info.get("frames") or 0))

    return workflow_to_prompt(workflow, args.output_node_id)


def raw_signature(args, workflow_path: Path, prepared: Path, seed: int | None = None, prompt_suffix: str = "", negative_suffix: str = "", guide_image: Path | None = None, extra_guides: "list[dict] | None" = None, auto_guide: bool = False, chunk_manifest: Path | None = None) -> dict[str, Any]:
    prompt_text = combine_prompt(args.prompt, prompt_suffix)
    negative_text = combine_prompt(args.negative_prompt, negative_suffix)
    return {
        "version": 27,
        "tool": "outpaint_video.py/raw_comfy",
        "prepared": root_relative(prepared),
        "prepared_fingerprint": file_fingerprint(prepared),
        "workflow": root_relative(workflow_path),
        "workflow_fingerprint": file_fingerprint(workflow_path),
        "target_aspect": args.target_aspect,
        "delivery_target_height": args.target_height,
        "model_size_multiple": MODEL_SIZE_MULTIPLE,
        "outpaint_all_black_regions": bool(getattr(args, "outpaint_all_black_regions", False)),
        "prompt": prompt_text,
        "prompt_suffix": prompt_suffix,
        "negative_suffix": negative_suffix,
        "guide_image": root_relative(guide_image) if guide_image else "",
        "guide_fingerprint": file_fingerprint(guide_image) if guide_image and guide_image.exists() else None,
        "guide_strength": getattr(args, "guide_strength", 0.7),
        "extra_guides": [
            {
                "frame_idx": g["frame_idx"],
                "strength": g["strength"],
                "image": root_relative(g["image"]),
                # Guides regenerated at a stable path (qwen/seed guides) must invalidate the
                # chunk; the path alone does not change.
                "image_fingerprint": file_fingerprint(g["image"]),
            }
            for g in (extra_guides or [])
        ],
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
    videos = {".mp4", ".mov", ".mkv", ".webm"}
    return newest_comfy_output(files, videos if any(path.suffix.lower() in videos for path in files) else None, "output file")


def black_margin_warning(video: Path, sample_frame: int = 20, side_fraction: float = 0.125) -> str:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return ""
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(sample_frame, max(0, frame_count - 1))))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return ""

    height, width = frame.shape[:2]
    side_width = max(1, int(width * side_fraction))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    left_black = float((gray[:, :side_width] < 16).mean())
    right_black = float((gray[:, -side_width:] < 16).mean())
    if left_black > 0.9 and right_black > 0.9:
        return (
            f"Warning: {video.name} still has mostly black side margins after Comfy "
            f"(left {left_black:.0%}, right {right_black:.0%} below luma 16). "
            "The LTX outpaint job appears to have preserved the padding instead of filling it."
        )
    return ""


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
    base = (prompt or "").strip()
    extra = (suffix or "").strip()
    if not base:
        return extra
    if not extra:
        return base
    separator = " " if base.endswith((".", "!", "?", ":")) else ". "
    return f"{base}{separator}{extra}"


def default_chunk_manifest(source: Path, aspect: str, width: int, height: int, args) -> Path:
    crop, black = _crop_black(args)
    return ROOT / "manifests" / "outpaint_chunks" / aid.outpaint_name(source.name, aspect, width, height, crop, black, "chunks", "csv")


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
                row["guide_image"] = row.get("anchor_image", "")
            rows[int(row["chunk_index"])] = row
        return rows


def auto_start_guide_enabled(row: dict[str, str]) -> bool:
    value = str(row.get("auto_start_guide", "")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def write_chunk_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "chunk_index",
        "start_frame",
        "end_frame",
        "start_seconds",
        "end_seconds",
        "custom_seconds",
        "offset_x",
        "offset_y",
        "seed",
        "prompt_suffix",
        "negative_suffix",
        "guide_image",
        "guide_strength",
        "guide_end_image",
        "guide_end_strength",
        "guide_frames",
        "auto_start_guide",
        "anchor_image",
        "anchor_position",
        "anchor_seconds",
        "prepared_path",
        "raw_path",
    ]
    import io as _io
    buf = _io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    text = buf.getvalue()
    if path.exists():
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                if handle.read() == text:
                    return
        except OSError:
            pass
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


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
        offset_x = int(float(row.get("offset_x", "0") or 0))
        offset_y = int(float(row.get("offset_y", "0") or 0))
        offset_slug = "" if not (offset_x or offset_y) else f"_ox{offset_x:+d}_oy{offset_y:+d}"
        row.update(
            {
                "chunk_index": str(chunk_index),
                "start_frame": str(start_frame),
                "end_frame": str(end_frame),
                "start_seconds": f"{start_frame / fps:.6f}",
                "end_seconds": f"{end_frame / fps:.6f}",
                "prepared_path": root_relative(chunk_dir / f"prepared_{chunk_index:04d}_{start_frame:06d}_{end_frame:06d}{offset_slug}.mp4"),
                "raw_path": root_relative(chunk_dir / f"raw_{chunk_index:04d}_{start_frame:06d}_{end_frame:06d}{offset_slug}.mp4"),
            }
        )
        row["offset_x"] = str(offset_x)
        row["offset_y"] = str(offset_y)
        if not row.get("seed"):
            row["seed"] = str(default_seed + chunk_index)
        row.setdefault("prompt_suffix", "")
        row.setdefault("negative_suffix", "")
        row.setdefault("guide_image", "")
        row.setdefault("custom_seconds", "")
        row.setdefault("auto_start_guide", "true")
        rows.append(row)
    write_chunk_manifest(path, rows)
    return {int(row["chunk_index"]): row for row in rows}


def _guide_frames_from_row(row: dict[str, str]) -> list[dict]:
    raw = (row.get("guide_frames", "") or "").strip()
    if raw:
        try:
            frames = json.loads(raw)
            if isinstance(frames, list):
                return [frame for frame in frames if isinstance(frame, dict)]
        except (json.JSONDecodeError, ValueError):
            pass
    frames: list[dict] = []
    if row.get("guide_image"):
        frames.append({"frame_idx": 0, "image": row["guide_image"], "strength": row.get("guide_strength", "0.7")})
    if row.get("guide_end_image"):
        frames.append({"frame_idx": -1, "image": row["guide_end_image"], "strength": row.get("guide_end_strength", "1.0")})
    return frames


def select_chunk_guides(guide_frames_list: "list[dict]", chunk_index: int, chunk_frames: int) -> "tuple[Path | None, float | None, list[dict]]":
    """Split a chunk's guide_frames into the frame-0 i2v guide and extra guide frames.

    Returns (explicit_guide, explicit_strength, extra_guides); explicit_strength is None
    when there is no frame-0 guide.  Every skipped guide prints a warning so dropped
    guides are always visible in the render log: guides with no image set, a missing
    image file, a position outside the chunk, or a duplicate frame-0 entry.
    """
    explicit_guide: Path | None = None
    explicit_strength: float | None = None
    extra_guides: list[dict] = []
    for gf in guide_frames_list:
        fidx = int(gf.get("frame_idx", 0))
        if not gf.get("image"):
            print(f"Warning: chunk {chunk_index + 1} guide (frame_idx={fidx}) has no image set; skipping it.", flush=True)
            continue
        img_path = resolve_path(gf.get("image", ""))
        if not img_path.exists():
            print(f"Warning: chunk {chunk_index + 1} guide image (frame_idx={fidx}) not found, ignoring: {img_path}", flush=True)
            continue
        if chunk_frames > 0 and not (-chunk_frames <= fidx < chunk_frames):
            print(f"Warning: chunk {chunk_index + 1} guide frame_idx={fidx} is outside the chunk (frames 0-{chunk_frames - 1}); skipping it. Re-position it after changing chunk lengths.", flush=True)
            continue
        if fidx == 0:
            if explicit_guide is not None:
                print(f"Warning: chunk {chunk_index + 1} has more than one frame-0 guide; keeping the first and skipping the rest.", flush=True)
                continue
            explicit_guide = img_path
            explicit_strength = float(gf.get("strength", 0.7))
        else:
            extra_guides.append({"frame_idx": fidx, "strength": float(gf.get("strength", 1.0)), "image": img_path})
    return explicit_guide, explicit_strength, extra_guides


def _occupied_guide_frame_idxs(rows: dict[int, dict[str, str]]) -> dict[int, set[int]]:
    occupied: dict[int, set[int]] = {}
    for chunk_index, row in rows.items():
        for guide in _guide_frames_from_row(row):
            if not guide.get("image"):
                continue
            try:
                occupied.setdefault(chunk_index, set()).add(int(guide.get("frame_idx", 0)))
            except (TypeError, ValueError):
                continue
    return occupied


def apply_qwen_seed_guides(args, prepared: Path, ranges: list[tuple[int, int, int]], chunk_manifest: Path) -> dict[int, dict[str, str]]:
    """Generate Qwen guide frames at shot changes and merge them into the chunk manifest.

    Existing guide images are preserved, including previous seed guides and user edits.
    Seed generation only fills missing shot-boundary positions.
    """
    rows = read_chunk_manifest(chunk_manifest)
    occupied = _occupied_guide_frame_idxs(rows)
    comfy_output_root = resolve_path(args.comfy_output_root) if args.comfy_output_root else resolve_path(args.comfy_dir) / "output"
    qwen_args = {
        "workflow": args.qwen_workflow,
        "masked_workflow": args.qwen_masked_workflow,
        "comfy_url": args.comfy_url,
        "comfy_dir": str(resolve_path(args.comfy_dir)),
        "comfy_output_root": str(comfy_output_root),
        "model_backend": args.qwen_model_backend,
        "gguf_model": args.qwen_gguf_model,
        "prompt": args.qwen_prompt,
        "load_image_node_id": args.qwen_load_image_node_id,
        "save_node_id": args.qwen_save_node_id,
    }
    print(f"Seeding Qwen guide frames at shot changes (prompt: {json.dumps(args.qwen_prompt)})...", flush=True)
    seeded = seed_guides(
        prepared, ranges, safe_stem(chunk_manifest.stem), qwen_args,
        sample_seconds=args.seed_sample_seconds,
        shot_threshold=args.seed_shot_threshold,
        min_shot_seconds=args.seed_min_shot_seconds,
        start_strength=getattr(args, "guide_strength", 0.7),
        force=args.force,
        occupied_frame_idxs=occupied,
    )
    for chunk_index, guides in seeded.items():
        row = rows.get(chunk_index)
        if row is None:
            continue
        existing = _guide_frames_from_row(row)
        existing_frame_idxs = {int(g.get("frame_idx", 0)) for g in existing if isinstance(g, dict) and g.get("image")}
        merged = existing + [g for g in guides if int(g.get("frame_idx", 0)) not in existing_frame_idxs]
        merged.sort(key=lambda g: int(g.get("frame_idx", 0)))
        row["guide_frames"] = json.dumps(merged)
        rows[chunk_index] = row
    write_chunk_manifest(chunk_manifest, [rows[key] for key in sorted(rows)])
    return read_chunk_manifest(chunk_manifest)


def split_chunk(ffmpeg: str, prepared: Path, chunk_path: Path, start_frame: int, end_frame: int, fps: float, force: bool, offset_x: int = 0, offset_y: int = 0, prepared_fingerprint: dict[str, Any] | None = None) -> None:
    if chunk_path.exists() and not force and (prepared_fingerprint is None or split_matches_source(chunk_path, prepared_fingerprint)):
        return
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    partial = chunk_path.with_suffix(chunk_path.suffix + ".partial" + chunk_path.suffix)
    trim = f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=N/({fps:.8f}*TB),fps={fps:.8f},setsar=1"
    if offset_x or offset_y:
        info = probe_video(prepared)
        width = int(info["width"])
        height = int(info["height"])
        pad_width = width + abs(int(offset_x))
        pad_height = height + abs(int(offset_y))
        pad_x = max(0, int(offset_x))
        pad_y = max(0, int(offset_y))
        crop_x = max(0, -int(offset_x))
        crop_y = max(0, -int(offset_y))
        vf = (
            f"{trim},"
            f"pad={pad_width}:{pad_height}:{pad_x}:{pad_y}:black,"
            f"crop={width}:{height}:{crop_x}:{crop_y},setsar=1"
        )
    else:
        vf = trim
    subprocess.run([ffmpeg, "-y", "-i", str(prepared), "-vf", vf, "-an", "-r", f"{fps:.8f}", "-fps_mode", "cfr", "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", str(partial)], check=True)
    replace_unless_identical(partial, chunk_path, f"Prepared chunk {chunk_path.name}")
    if prepared_fingerprint is not None:
        write_split_sidecar(chunk_path, prepared, prepared_fingerprint)


def overlap_context_before_anchor(overlap_frames: int, anchor_seconds: str, fps: float, total_frames: int) -> int:
    try:
        anchor_frame = int(float(anchor_seconds or 0.0) * fps)
    except ValueError:
        anchor_frame = overlap_frames
    return max(0, min(int(overlap_frames), int(total_frames), anchor_frame))


def inject_overlap_context(ffmpeg: str, chunk: Path, previous_raw: Path, context_frames: int, fps: float, force: bool) -> Path:
    if context_frames <= 0 or not previous_raw.exists():
        return chunk
    output = chunk.with_name(f"{chunk.stem}_with_context{chunk.suffix}")
    if output.exists() and not force:
        return output

    chunk_info = probe_video(chunk)
    previous_info = probe_video(previous_raw)
    chunk_size = (int(chunk_info["width"]), int(chunk_info["height"]))
    previous_size = (int(previous_info["width"]), int(previous_info["height"]))
    if chunk_size != previous_size:
        raise RuntimeError(
            f"Outpaint chunk overlap geometry changed from {previous_size[0]}x{previous_size[1]} "
            f"to {chunk_size[0]}x{chunk_size[1]}; the working canvas should stay constant."
        )

    previous_frames = int(previous_info["frames"])
    start = max(0, previous_frames - int(context_frames) - 1)
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    vf = (
        f"[0:v]trim=start_frame={start}:end_frame={previous_frames},setpts=N/({fps:.8f}*TB)[ctx];"
        f"[1:v]trim=start_frame=1,setpts=N/({fps:.8f}*TB)[new];"
        f"[ctx][new]concat=n=2:v=1:a=0,fps={fps:.8f},setsar=1[v]"
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(previous_raw),
            "-i",
            str(chunk),
            "-filter_complex",
            vf,
            "-map",
            "[v]",
            "-an",
            "-r",
            f"{fps:.8f}",
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-preset",
            "veryfast",
            str(partial),
        ],
        check=True,
    )
    replace_with_retry(partial, output, f"Overlap context chunk {output.name}")
    return output




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
    parser.add_argument("--prompt", default=DEFAULT_OUTPAINT_PROMPT)
    parser.add_argument("--seed-node-id", default="4832")
    parser.add_argument("--seed-widget", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-prompt", default="cartoon, game, 3d render, still image, static, warped geometry, flicker, smeared details, extra fingers, broken fingers, deformed hands")
    parser.add_argument("--guide-strength", type=float, default=0.7, help="Conditioning strength for the start-frame guide (LTX i2v, 0–1). Default: 0.7.")
    parser.add_argument("--guide-end-strength", type=float, default=1.0, help="Conditioning strength for the end-frame guide (IC-LoRA, 0–1). Default: 1.0.")
    parser.add_argument("--black-lift", type=float, default=0.018)
    parser.add_argument("--gamma", type=float, default=1.06)
    parser.add_argument("--outpaint-all-black-regions", action="store_true", help="Leave source blacks untouched so black regions inside the source can be outpainted.")
    # Qwen guide seeding: when LTX won't outpaint, auto-generate filled guide frames at every
    # shot change with Qwen Image Edit so each shot anchors from a frame whose bars are filled.
    parser.add_argument("--seed-qwen-guides", action="store_true", help="Before rendering, detect shot changes and auto-add a Qwen-outpainted guide frame at each one.")
    parser.add_argument("--qwen-workflow", default=str(ROOT / "workflows" / "qwen_image_edit" / "Image Edit (Qwen 2511).json"))
    parser.add_argument("--qwen-masked-workflow", default=str(ROOT / "workflows" / "qwen_image_edit" / "Image Edit Inpaint (Qwen 2511).json"))
    parser.add_argument("--qwen-model-backend", default="gguf")
    parser.add_argument("--qwen-gguf-model", default=QWEN_IMAGE_EDIT_MODEL)
    parser.add_argument("--qwen-prompt", default=DEFAULT_SEED_PROMPT, help="Prompt sent to Qwen Image Edit for each seed guide frame.")
    parser.add_argument("--qwen-load-image-node-id", default="auto")
    parser.add_argument("--qwen-save-node-id", default="auto")
    parser.add_argument("--seed-sample-seconds", type=float, default=0.0, help="Shot-detection sampling interval for seeding (0 = every frame).")
    parser.add_argument("--seed-shot-threshold", type=float, default=0.075)
    parser.add_argument("--seed-min-shot-seconds", type=float, default=1.0)
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
        ensure_node_types(args.comfy_url, required_nodes, "outpainting workflow", comfy_dir)

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
    if args.outpaint_all_black_regions:
        prepare_command.append("--outpaint-all-black-regions")
    prepare_command += ["--target-height", str(work_height)]
    prepare_command += ["--target-width", str(work_width)]
    prepare_command += ["--delivery-width", str(delivery_width)]
    prepare_command += ["--delivery-height", str(delivery_height)]
    if args.force:
        prepare_command.append("--force")
    if args.dry_run:
        prepare_command.append("--dry-run")
    if (work_width, work_height) != (delivery_width, delivery_height):
        print(
            f"LTX working canvas: {work_width}x{work_height} (rounded to multiples of {MODEL_SIZE_MULTIPLE} "
            f"from delivery {delivery_width}x{delivery_height}). Recomposition will upscale back to delivery.",
            flush=True,
        )
    mode = "all black regions" if args.outpaint_all_black_regions else "protected source blacks"
    print(f"Preparing expanded outpaint canvas: {work_width}x{work_height}, aspect {args.target_aspect}, black_lift={args.black_lift}, gamma={args.gamma}, mode={mode}", flush=True)
    run_command(prepare_command, False)

    output_prefix = f"arp_outpaint/{safe_stem(source.name)}_{aspect_slug(args.target_aspect)}_{work_width}x{work_height}"
    print(f"Prepared expanded canvas for ComfyUI: {prepared}", flush=True)
    if not args.dry_run:
        ffmpeg = find_ffmpeg()
        chunk_crop, chunk_black = _crop_black(args)
        chunk_dir = ROOT / ".cache" / "outpaint_chunks" / aid.outpaint_basename(source.name, args.target_aspect, work_width, work_height, chunk_crop, chunk_black, "chunks")
        chunk_manifest = resolve_path(args.chunk_manifest) if args.chunk_manifest else default_chunk_manifest(source, args.target_aspect, work_width, work_height, args)
        prepared_info = probe_video(prepared)
        chunk_existing = read_chunk_manifest(chunk_manifest)
        ranges = chunk_ranges_from_manifest(int(prepared_info["frames"]), float(prepared_info["fps"]), args.chunk_seconds, args.overlap_frames, chunk_existing)
        chunk_overrides = sync_chunk_manifest(chunk_manifest, ranges, float(prepared_info["fps"]), chunk_dir, args.seed)
        print(f"Outpaint chunk manifest: {chunk_manifest}", flush=True)
        if args.seed_qwen_guides:
            chunk_overrides = apply_qwen_seed_guides(args, prepared, ranges, chunk_manifest)
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
            base_guide_strength = getattr(args, "guide_strength", 0.7)
            raw_chunks: list[Path] = []
            effective_ranges: list[tuple[int, int, int]] = []
            for range_index, (chunk_index, start_frame, end_frame) in enumerate(ranges):
                chunk_row = chunk_overrides.get(chunk_index, {})
                chunk_offset_x = int(float(chunk_row.get("offset_x", "0") or 0))
                chunk_offset_y = int(float(chunk_row.get("offset_y", "0") or 0))
                chunk_prepared = resolve_path(chunk_row.get("prepared_path", "")) if chunk_row.get("prepared_path") else chunk_dir / f"prepared_{chunk_index:04d}_{start_frame:06d}_{end_frame:06d}.mp4"
                chunk_raw = resolve_path(chunk_row.get("raw_path", "")) if chunk_row.get("raw_path") else chunk_dir / f"raw_{chunk_index:04d}_{start_frame:06d}_{end_frame:06d}.mp4"
                print(f"Outpaint chunk {chunk_index + 1}/{len(ranges)}: frames {start_frame}-{end_frame}", flush=True)
                force_this_split = args.force and (args.only_chunk is None or chunk_index == args.only_chunk)
                split_chunk(ffmpeg, prepared, chunk_prepared, start_frame, end_frame, float(prepared_info["fps"] or 24.0), force_this_split, chunk_offset_x, chunk_offset_y, raw_sig["prepared_fingerprint"])
                previous_raw = raw_chunks[-1] if raw_chunks else None
                chunk_seed = int(chunk_row.get("seed") or args.seed + chunk_index)
                chunk_prompt_suffix = chunk_row.get("prompt_suffix", "")
                chunk_negative_suffix = chunk_row.get("negative_suffix", "")

                # Parse guide frames list (JSON from manifest, or migrate legacy fields).
                raw_gf = chunk_row.get("guide_frames", "").strip()
                if raw_gf:
                    try:
                        guide_frames_list: list[dict] = json.loads(raw_gf)
                    except (json.JSONDecodeError, ValueError):
                        guide_frames_list = []
                else:
                    # Migrate from legacy guide_image / guide_end_image fields.
                    guide_frames_list = []
                    if chunk_row.get("guide_image"):
                        try:
                            s = float(chunk_row.get("guide_strength", "0.7") or "0.7")
                        except ValueError:
                            s = 0.7
                        guide_frames_list.append({"frame_idx": 0, "strength": s, "image": chunk_row["guide_image"]})
                    if chunk_row.get("guide_end_image"):
                        try:
                            s = float(chunk_row.get("guide_end_strength", "1.0") or "1.0")
                        except ValueError:
                            s = 1.0
                        guide_frames_list.append({"frame_idx": -1, "strength": s, "image": chunk_row["guide_end_image"]})

                # Frame-0 guide → i2v path (explicit_guide); others → LTXVAddGuideAdvanced.
                explicit_guide, explicit_strength, extra_guides = select_chunk_guides(guide_frames_list, chunk_index, end_frame - start_frame)
                # raw_signature and patch_workflow read guide strength off args; reset it each
                # chunk so a frame-0 guide's strength can't leak into later chunks' auto-guides.
                args.guide_strength = explicit_strength if explicit_strength is not None else base_guide_strength

                auto_guide: bool = False
                guide_image: Path | None = None
                use_auto_start_guide = auto_start_guide_enabled(chunk_row)
                if previous_raw is not None and previous_raw.exists() and use_auto_start_guide:
                    try:
                        guide_image = extract_last_frame_as_guide(previous_raw, chunk_dir)
                        auto_guide = True
                        print(f"Chunk {chunk_index + 1}: auto-guide from last frame of chunk {chunk_index}", flush=True)
                        if explicit_guide is not None:
                            print(f"Chunk {chunk_index + 1}: frame-0 guide overridden by previous-chunk start guide", flush=True)
                    except Exception as exc:
                        print(f"Warning: could not extract auto-guide from previous chunk: {exc}", flush=True)
                elif previous_raw is not None and previous_raw.exists() and not use_auto_start_guide:
                    print(f"Chunk {chunk_index + 1}: previous-chunk start guide disabled", flush=True)
                if guide_image is None:
                    guide_image = explicit_guide

                chunk_sig = raw_signature(args, workflow_path, chunk_prepared, chunk_seed, chunk_prompt_suffix, chunk_negative_suffix, guide_image, extra_guides, auto_guide)
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
                print(f"Chunk {chunk_index + 1} positive prompt sent to Comfy: {json.dumps(prompt_text, ensure_ascii=False)}", flush=True)
                print(f"Chunk {chunk_index + 1} negative prompt sent to Comfy: {json.dumps(negative_text, ensure_ascii=False)}", flush=True)
                if chunk_prompt_suffix:
                    print(f"Chunk {chunk_index + 1} prompt suffix: {chunk_prompt_suffix}", flush=True)
                if chunk_negative_suffix:
                    print(f"Chunk {chunk_index + 1} negative suffix: {chunk_negative_suffix}", flush=True)
                if guide_image:
                    source = "explicit" if explicit_guide else "auto"
                    print(f"Chunk {chunk_index + 1} start guide ({source}): {guide_image}", flush=True)
                for gf in extra_guides:
                    print(f"Chunk {chunk_index + 1} guide frame_idx={gf['frame_idx']}: {gf['image']}", flush=True)
                prompt = patch_workflow(args, workflow, chunk_prepared, comfy_dir, chunk_prefix, prompt_text, negative_text, chunk_seed, guide_image, extra_guides)
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
                warning = black_margin_warning(chunk_raw)
                if warning:
                    print(warning, flush=True)
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

    # Finalize restores the black/gamma lift but does NOT upscale to delivery resolution.
    # The outpainted and colorised layers stay at model-safe dimensions (e.g. 1280×704)
    # all the way through to recomposition, where final_composite.py scales up to delivery.
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
    ]
    if args.outpaint_all_black_regions:
        finalize_command.append("--skip-restore")
    if args.force:
        finalize_command.append("--force")
    if args.dry_run:
        finalize_command.append("--dry-run")
    run_command(finalize_command, args.dry_run)
    print(f"Wrote outpainted video: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
