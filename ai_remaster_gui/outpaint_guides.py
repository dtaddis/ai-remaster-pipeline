from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import state
from .config import (
    FILE_PREVIEW_DIR,
    IMAGE_EXTS,
    QWEN_IMAGE_EDIT_MODEL,
    ROOT,
    SCRIPTS,
    comfy_output_root_for,
    current_config,
)
from .file_dialogs import browse_path
from .manifests import read_outpaint_chunk_rows, write_outpaint_chunk_rows
from .media import extract_video_frame_at, pipeline_source_text
from .paths import rel, resolve
from .runtime_settings import qwen_masked_workflow_for
from .sam_masks import sam2_mask_for_image

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from guide_frame_utils import guide_output_size_for_prepared, save_edge_mask_for_image  # noqa: E402

# These few names live in server.py (a constant plus helpers wired into its outpaint internals);
# importing server here would be circular (see state.py), so server.py injects them at startup via
# bind_context. Declared here so the names resolve statically.
DEFAULT_ANCHOR_PROMPT = ""
ensure_outpaint_prepared_canvas = None
outpaint_chunks_state = None
remove_cached_file = None

_SERVER_OUTPAINT_OPS = (
    "DEFAULT_ANCHOR_PROMPT",
    "ensure_outpaint_prepared_canvas",
    "outpaint_chunks_state",
    "remove_cached_file",
)


def bind_context(context: dict) -> None:
    """Wire in the server.py-defined outpaint constant/helpers this module calls (see above)."""
    globals().update({name: context[name] for name in _SERVER_OUTPAINT_OPS})


def chunk_frame_preview(source: Path, seconds: float, suffix: str) -> str:
    if not source.exists():
        return ""
    return extract_video_frame_at(source, FILE_PREVIEW_DIR / "chunks", f"{suffix}_{int(seconds * 1000):010d}", seconds)

def _parse_guide_frames(row: dict[str, str]) -> list[dict]:
    """Return the guide_frames list for a manifest row, migrating from old fields if needed."""
    raw = row.get("guide_frames", "").strip()
    if raw:
        try:
            frames = json.loads(raw)
            if isinstance(frames, list):
                return frames
        except json.JSONDecodeError:
            pass
    # Migrate from legacy guide_image / guide_end_image fields.
    frames: list[dict] = []
    if row.get("guide_image"):
        try:
            strength = float(row.get("guide_strength", "0.7") or "0.7")
        except ValueError:
            strength = 0.7
        frames.append({"frame_idx": 0, "strength": round(strength, 3), "image": row["guide_image"]})
    if row.get("guide_end_image"):
        try:
            strength = float(row.get("guide_end_strength", "1.0") or "1.0")
        except ValueError:
            strength = 1.0
        frames.append({"frame_idx": -1, "strength": round(strength, 3), "image": row["guide_end_image"]})
    return frames

def _save_guide_frames(manifest: Path, chunk_index: int, frames: list[dict]) -> None:
    """Persist guide_frames JSON back to the manifest row."""
    rows = read_outpaint_chunk_rows(manifest)
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    rows[chunk_index]["guide_frames"] = json.dumps(frames)
    write_outpaint_chunk_rows(manifest, [rows[k] for k in sorted(rows)])

def _guide_source_seconds(row: dict, frame_idx: int, fps: float) -> float:
    """Convert a frame_idx (possibly negative) to absolute seconds in the prepared canvas."""
    start = float(row.get("start", 0.0))
    end = float(row.get("end", 0.0))
    length_frames = int(row.get("end_frame", 0)) - int(row.get("start_frame", 0))
    if frame_idx < 0:
        actual = max(0, length_frames + frame_idx)
    else:
        actual = frame_idx
    return max(start, min(end - (1.0 / max(1.0, fps)), start + actual / max(1.0, fps)))

def _build_guide_frames_view(
    row: dict,
    source_text: str,
    aspect: str,
    start_seconds: float,
    end_seconds: float,
    fps: float,
    length_frames: int,
) -> list[dict]:
    """Build the view list for guide frames, including thumbnail previews."""
    frames = _parse_guide_frames(row)
    view = []
    for i, gf in enumerate(frames):
        frame_idx = int(gf.get("frame_idx", 0))
        strength = float(gf.get("strength", 0.7))
        image_rel = gf.get("image", "")
        image_path = resolve(image_rel) if image_rel else None
        image_exists = bool(image_path and image_path.exists())
        source_secs = _guide_source_seconds(
            {"start": start_seconds, "end": end_seconds,
             "start_frame": str(int(start_seconds * fps)),
             "end_frame": str(int(end_seconds * fps))},
            frame_idx, fps,
        )
        view.append({
            "guide_index": i,
            "frame_idx": frame_idx,
            "strength": strength,
            "image": image_rel,
            "image_exists": image_exists,
            "image_mtime": int(image_path.stat().st_mtime_ns) if image_exists and image_path else 0,
            "source_preview": "",
        })
    return view

def guide_frame_generation_command(chunk_index: int, guide_index: int, frame_idx: int, prompt: str) -> tuple[list[str], str, Path, float]:
    """Build the Qwen generation command for any guide frame position."""
    chunks_state = outpaint_chunks_state(state.APP.settings)
    rows = chunks_state.get("rows", [])
    manifest_text = chunks_state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    if chunk_index < 0 or chunk_index >= len(rows):
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")

    row = rows[chunk_index]
    fps = float(row.get("fps", 24) or 24)
    source_seconds = _guide_source_seconds(row, frame_idx, fps)

    source_text = state.APP.outpaint_source_for()
    if not source_text:
        raise RuntimeError("No source material is selected.")
    range_source = ensure_outpaint_prepared_canvas(source_text, state.APP.settings.get("outpaint", {}))
    cache_key = f"gf_qwen_{int(source_seconds * 1000):010d}"
    preview_rel = chunk_frame_preview(range_source, source_seconds, cache_key)
    source_img = resolve(preview_rel) if preview_rel else Path("")
    if not source_img.is_file():
        raise FileNotFoundError(f"Could not extract source frame for Qwen guide at {source_seconds:.3f}s from {range_source}.")

    manifest = resolve(str(manifest_text))
    output_dir = ROOT / "intermediate" / "outpaint_guides" / manifest.stem
    output = output_dir / f"chunk_{chunk_index:04d}_guide_{guide_index:02d}_qwen.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_cached_file(output)
    source_img = save_qwen_input_copy(source_img, output.with_name(f"chunk_{chunk_index:04d}_guide_{guide_index:02d}_qwen_input{source_img.suffix.lower() or '.jpg'}"))

    # Pre-write the output path into guide_frames so the thumbnail updates immediately.
    stored = read_outpaint_chunk_rows(manifest)
    if chunk_index in stored:
        frames = _parse_guide_frames(stored[chunk_index])
        if 0 <= guide_index < len(frames):
            frames[guide_index]["image"] = rel(output)
            frames[guide_index].pop("seed", None)
        else:
            frames.append({"frame_idx": frame_idx, "strength": 0.7, "image": rel(output)})
        stored[chunk_index]["guide_frames"] = json.dumps(frames)
        write_outpaint_chunk_rows(manifest, [stored[k] for k in sorted(stored)])

    guide_prompt = prompt.strip() or DEFAULT_ANCHOR_PROMPT
    mask = save_edge_mask_for_image(source_img, output.with_name(f"chunk_{chunk_index:04d}_guide_{guide_index:02d}_qwen_edge_mask.png"))
    cmd = auto_masked_guide_command(source_img, output, guide_prompt, mask)
    return cmd, rel(output), resolve(range_source), source_seconds

def _composite_guide_in_place(output: Path, prepared_canvas: Path, source_seconds: float | None = None) -> None:
    """Composite a Qwen guide PNG with actual source content, then inpaint black corners, in-place.

    Steps:
      1. Scale the Qwen guide to the LTX work canvas size.
      2. Overlay actual source pixels from the prepared canvas wherever they are non-black
         (i.e. the source content area â€” e.g. 960Ã—704 centred in 1280Ã—704).  This ensures
         pixel-accurate alignment between the guide and the prepared canvas regardless of any
         sub-pixel shifts introduced by Qwen's internal patch processing.
      3. Inpaint any remaining near-black pixels (corners where both guide and source are black).
    Saves the result back over *output*.

    source_seconds: timestamp in the prepared canvas to use as the source frame.
    Defaults to t=0 (actual first frame).
    """
    import cv2
    import numpy as np
    from PIL import Image as PILImage

    # Read prepared canvas dimensions and extract the source frame.
    cap = cv2.VideoCapture(str(prepared_canvas))
    canvas_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    canvas_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    seek_ms = (source_seconds * 1000.0) if source_seconds is not None else 0.0
    cap.set(cv2.CAP_PROP_POS_MSEC, seek_ms)
    ok, src_frame = cap.read()
    if not ok or src_frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, src_frame = cap.read()
    cap.release()
    if not ok or src_frame is None:
        raise RuntimeError(f"Could not read frame from prepared canvas: {prepared_canvas}")
    if src_frame.shape[1] != canvas_w or src_frame.shape[0] != canvas_h:
        src_frame = cv2.resize(src_frame, (canvas_w, canvas_h), interpolation=cv2.INTER_LANCZOS4)

    with PILImage.open(output) as img:
        img_w, img_h = img.size
        guide_rgb = img.convert("RGB")

    # Preserve the raw Qwen output alongside the composited result for inspection.
    raw_copy = output.with_name(output.stem + "_raw" + output.suffix)
    if not raw_copy.exists():
        import shutil as _shutil
        _shutil.copy2(output, raw_copy)

    resampling = getattr(PILImage, "Resampling", PILImage).LANCZOS

    guide_w, guide_h = guide_output_size_for_prepared(prepared_canvas, canvas_w, canvas_h)
    if src_frame.shape[1] != guide_w or src_frame.shape[0] != guide_h:
        src_frame = cv2.resize(src_frame, (guide_w, guide_h), interpolation=cv2.INTER_LANCZOS4)

    # Step 1: fill-resize Qwen output to the model-safe guide canvas. Do not preserve
    # Qwen's AR here; the prepared video geometry is the authority.
    canvas_pil = guide_rgb.resize((guide_w, guide_h), resampling)

    canvas_bgr = cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)

    # Step 2: blend the source frame's content pixels over the centre with a soft edge.
    # black_lift raises all source pixels above 0; the padding margins are exact black (0,0,0).
    # A ~10px Gaussian feather at the content boundary avoids a hard seam where Qwen's
    # outpainting meets the composited source pixels.
    src_is_content = np.any(src_frame > 4, axis=2)
    if src_is_content.any():
        feather_px = 10
        alpha = cv2.GaussianBlur(
            src_is_content.astype(np.float32),
            (feather_px * 2 + 1, feather_px * 2 + 1),
            feather_px / 2,
        )
        # Mask the alpha back to zero outside the content area so the blur never
        # bleeds black pillar pixels into Qwen's outpainting â€” feather is inward only.
        alpha = (alpha * src_is_content.astype(np.float32))[:, :, np.newaxis]
        canvas_bgr = (
            src_frame.astype(np.float32) * alpha
            + canvas_bgr.astype(np.float32) * (1.0 - alpha)
        ).clip(0, 255).astype(np.uint8)

    # Step 3: inpaint the small corner triangles that remain black
    # (top/bottom strips outside both the Qwen letterbox and the source content area).
    still_black = np.all(canvas_bgr <= 4, axis=2).astype(np.uint8) * 255
    if still_black.any():
        canvas_bgr = cv2.inpaint(canvas_bgr, still_black, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    PILImage.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)).save(output, format="PNG")

def save_qwen_input_copy(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    shutil.copy2(source, target)
    return target

def auto_masked_guide_command(source: Path, output: Path, prompt: str, mask: Path) -> list[str]:
    values = state.APP.settings.get("references", {})
    config = current_config()
    workflow = qwen_masked_workflow_for(values, config)
    if not workflow:
        raise RuntimeError("Automatic guide generation needs a Qwen masked edit workflow.")
    if not resolve(workflow).is_file():
        raise FileNotFoundError(f"Masked edit workflow not found: {workflow}")
    cmd = [
        sys.executable, "-u",
        str(SCRIPTS / "edit_reference_image.py"),
        "--source-image", str(source),
        "--mask", rel(mask),
        "--output", rel(output),
        "--workflow", workflow,
        "--comfy-url", values.get("comfy_url") or config.get("comfy_url", "http://127.0.0.1:8188"),
        "--comfy-dir", config.get("comfy_dir", str(ROOT / "tools" / "comfyui")),
        "--comfy-output-root", comfy_output_root_for(config),
        "--model-backend", values.get("model_backend", "gguf"),
        "--gguf-model", values.get("gguf_model", QWEN_IMAGE_EDIT_MODEL),
        "--instruction", prompt,
        "--load-image-node-id", values.get("load_image_node_id", "auto"),
        "--save-node-id", values.get("save_node_id", "auto"),
        "--no-normalize-to-source-size",
        "--force",
    ]
    if values.get("prompt_node_id"):
        cmd.extend(["--prompt-node-id", values["prompt_node_id"]])
    return cmd

def outpaint_guide_generation_command(index: int, prompt: str) -> tuple[list[str], str, Path]:
    chunks_state = outpaint_chunks_state(state.APP.settings)
    rows = chunks_state.get("rows", [])
    manifest_text = chunks_state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    if index < 0 or index >= len(rows):
        raise IndexError(f"Outpaint chunk not found: {index + 1}")

    row = rows[index]
    fps = float(row.get("fps", 24) or 24)
    start_seconds = float(row.get("start", 0.0))
    guide_source_seconds = start_seconds
    source_text = state.APP.outpaint_source_for()
    if not source_text:
        raise RuntimeError("No source material is selected.")
    range_source = ensure_outpaint_prepared_canvas(source_text, state.APP.settings.get("outpaint", {}))
    preview_rel = chunk_frame_preview(range_source, guide_source_seconds, "source_guide_qwen")
    source = resolve(preview_rel) if preview_rel else Path("")
    if not source.is_file():
        raise FileNotFoundError(f"Could not extract source frame for Qwen guide at {guide_source_seconds:.3f}s from {range_source}.")

    manifest = resolve(str(manifest_text))
    output_dir = ROOT / "intermediate" / "outpaint_guides" / manifest.stem
    output = output_dir / f"chunk_{index:04d}_guide_qwen.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_cached_file(output)
    source = save_qwen_input_copy(source, output.with_name(f"chunk_{index:04d}_guide_qwen_input{source.suffix.lower() or '.jpg'}"))

    stored = read_outpaint_chunk_rows(manifest)
    if index not in stored:
        raise IndexError(f"Outpaint chunk not found in manifest: {index + 1}")
    stored[index]["guide_image"] = rel(output)
    write_outpaint_chunk_rows(manifest, [stored[key] for key in sorted(stored)])

    guide_prompt = prompt.strip() or DEFAULT_ANCHOR_PROMPT
    mask = save_edge_mask_for_image(source, output.with_name(f"chunk_{index:04d}_guide_qwen_edge_mask.png"))
    cmd = auto_masked_guide_command(source, output, guide_prompt, mask)
    return cmd, rel(output), resolve(range_source), guide_source_seconds

def outpaint_end_guide_generation_command(index: int, prompt: str) -> tuple[list[str], str, Path, float]:
    chunks_state = outpaint_chunks_state(state.APP.settings)
    rows = chunks_state.get("rows", [])
    manifest_text = chunks_state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    if index < 0 or index >= len(rows):
        raise IndexError(f"Outpaint chunk not found: {index + 1}")

    row = rows[index]
    fps = float(row.get("fps", 24) or 24)
    end_seconds = float(row.get("end", 0.0))
    # Use the last meaningful frame (end - 1/fps) as the Qwen source for the end guide.
    guide_source_seconds = max(float(row.get("start", 0.0)), end_seconds - (1.0 / max(1.0, fps)))
    source_text = state.APP.outpaint_source_for()
    if not source_text:
        raise RuntimeError("No source material is selected.")
    range_source = ensure_outpaint_prepared_canvas(source_text, state.APP.settings.get("outpaint", {}))
    preview_rel = chunk_frame_preview(range_source, guide_source_seconds, "source_guide_end_qwen")
    source = resolve(preview_rel) if preview_rel else Path("")
    if not source.is_file():
        raise FileNotFoundError(f"Could not extract source frame for Qwen end guide at {guide_source_seconds:.3f}s from {range_source}.")

    manifest = resolve(str(manifest_text))
    output_dir = ROOT / "intermediate" / "outpaint_guides" / manifest.stem
    output = output_dir / f"chunk_{index:04d}_guide_end_qwen.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_cached_file(output)
    source = save_qwen_input_copy(source, output.with_name(f"chunk_{index:04d}_guide_end_qwen_input{source.suffix.lower() or '.jpg'}"))

    stored = read_outpaint_chunk_rows(manifest)
    if index not in stored:
        raise IndexError(f"Outpaint chunk not found in manifest: {index + 1}")
    stored[index]["guide_end_image"] = rel(output)
    write_outpaint_chunk_rows(manifest, [stored[key] for key in sorted(stored)])

    guide_prompt = prompt.strip() or DEFAULT_ANCHOR_PROMPT
    mask = save_edge_mask_for_image(source, output.with_name(f"chunk_{index:04d}_guide_end_qwen_edge_mask.png"))
    cmd = auto_masked_guide_command(source, output, guide_prompt, mask)
    return cmd, rel(output), resolve(range_source), guide_source_seconds


def _get_guide_manifest() -> tuple[Path, dict[int, dict[str, str]], str]:
    chunks_state = outpaint_chunks_state(state.APP.settings)
    manifest_text = chunks_state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    manifest = resolve(str(manifest_text))
    rows = read_outpaint_chunk_rows(manifest)
    return manifest, rows, manifest_text

def add_guide_frame(chunk_index: int) -> dict:
    manifest, rows, _ = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    frames.append({"frame_idx": 0, "strength": 0.7, "image": ""})
    _save_guide_frames(manifest, chunk_index, frames)
    state.APP.log.append(f"Added guide frame to chunk {chunk_index + 1} (total: {len(frames)})")
    return {"guide_index": len(frames) - 1}

def remove_guide_frame(chunk_index: int, guide_index: int) -> dict:
    manifest, rows, _ = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    if guide_index < 0 or guide_index >= len(frames):
        raise IndexError(f"Guide frame {guide_index} not found in chunk {chunk_index + 1}")
    removed = frames.pop(guide_index)
    if removed.get("image"):
        remove_cached_file(resolve(removed["image"]))
    _save_guide_frames(manifest, chunk_index, frames)
    state.APP.log.append(f"Removed guide frame {guide_index} from chunk {chunk_index + 1}")
    return {"removed": guide_index}

def save_guide_frame(chunk_index: int, guide_index: int, frame_idx: int, strength: float) -> dict:
    manifest, rows, _ = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    if guide_index < 0 or guide_index >= len(frames):
        raise IndexError(f"Guide frame {guide_index} not found in chunk {chunk_index + 1}")
    frames[guide_index]["frame_idx"] = int(frame_idx)
    frames[guide_index]["strength"] = round(max(0.0, min(1.0, float(strength))), 3)
    _save_guide_frames(manifest, chunk_index, frames)
    state.APP.log.append(f"Saved guide frame {guide_index} for chunk {chunk_index + 1}: frame_idx={frame_idx}, strength={strength:.2f}")
    return {"frame_idx": frame_idx, "strength": frames[guide_index]["strength"]}

def upload_guide_frame_image(chunk_index: int, guide_index: int) -> dict:
    manifest, rows, _ = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    if guide_index < 0 or guide_index >= len(frames):
        raise IndexError(f"Guide frame {guide_index} not found in chunk {chunk_index + 1}")
    current = frames[guide_index].get("image", "")
    selected = browse_path("image", current)
    if not selected:
        return {"selected": "", "image": current}
    source = resolve(selected)
    if source.suffix.lower() not in IMAGE_EXTS:
        raise RuntimeError("Choose a PNG or JPEG image for the guide frame.")
    target_dir = ROOT / "intermediate" / "outpaint_guides" / manifest.stem
    target = target_dir / f"chunk_{chunk_index:04d}_guide_{guide_index:02d}{source.suffix.lower()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    frames[guide_index]["image"] = rel(target)
    frames[guide_index].pop("seed", None)
    _save_guide_frames(manifest, chunk_index, frames)
    state.APP.log.append(f"Uploaded guide frame {guide_index} for chunk {chunk_index + 1}: {rel(target)}")
    return {"selected": selected, "image": rel(target)}

def clear_guide_frame_image(chunk_index: int, guide_index: int) -> dict:
    manifest, rows, _ = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    if guide_index < 0 or guide_index >= len(frames):
        raise IndexError(f"Guide frame {guide_index} not found in chunk {chunk_index + 1}")
    current = frames[guide_index].get("image", "")
    if current:
        remove_cached_file(resolve(current))
    frames[guide_index]["image"] = ""
    _save_guide_frames(manifest, chunk_index, frames)
    state.APP.log.append(f"Cleared guide frame {guide_index} image for chunk {chunk_index + 1}")
    return {"image": ""}

def _guide_edit_dir(manifest: Path, chunk_index: int, guide_index: int) -> Path:
    return ROOT / "intermediate" / "outpaint_guides" / manifest.stem / "edits" / f"chunk_{chunk_index:04d}_guide_{guide_index:02d}"

def _next_guide_edit_output(manifest: Path, chunk_index: int, guide_index: int) -> Path:
    folder = _guide_edit_dir(manifest, chunk_index, guide_index)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = folder / f"edit_{stamp}.png"
    if not base.exists() and not base.with_suffix(base.suffix + ".json").exists():
        return base
    suffix = 1
    while True:
        candidate = folder / f"edit_{stamp}_{suffix:02d}.png"
        if not candidate.exists() and not candidate.with_suffix(candidate.suffix + ".json").exists():
            return candidate
        suffix += 1

def _save_guide_edit_mask(manifest: Path, chunk_index: int, guide_index: int, mask_data: str) -> str:
    if not mask_data:
        return ""
    import base64

    payload = mask_data.split(",", 1)[1] if "," in mask_data else mask_data
    raw = base64.b64decode(payload)
    folder = _guide_edit_dir(manifest, chunk_index, guide_index)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"mask_{time.strftime('%Y%m%d_%H%M%S')}.png"
    path.write_bytes(raw)
    return rel(path)

def _guide_edit_prompt(instruction: str, sampled_color: str = "") -> str:
    text = instruction.strip()
    color = sampled_color.strip()
    parts = [text]
    if color:
        parts.append(f"Use the sampled colour/value exactly where relevant: {color}, including its brightness.")
    if color:
        parts.append(
            "If the requested material colour is lighter than the current one, raise the masked area's "
            "luminance to the target value while preserving texture, shadows, edges, composition, and identity; "
            "do not merely darken or tint the existing colour."
        )
    return " ".join(part for part in parts if part).strip() or DEFAULT_ANCHOR_PROMPT

def normalize_guide_preview_to_source(output: Path, source: Path) -> None:
    """Fill-resize a Qwen guide-edit preview to the editor source image size."""
    if not output.is_file() or not source.is_file():
        return
    from PIL import Image as PILImage

    raw_copy = output.with_name(output.stem + "_raw" + output.suffix)
    if not raw_copy.exists():
        shutil.copy2(output, raw_copy)
    with PILImage.open(source) as src_img:
        target_size = src_img.size
    with PILImage.open(output) as out_img:
        if out_img.size == target_size:
            return
        resampling = getattr(PILImage, "Resampling", PILImage).LANCZOS
        out_img.convert("RGB").resize(target_size, resampling).save(output, format="PNG")

def _guide_editor_source(chunk_index: int, guide_index: int, frames: list[dict]) -> tuple[str, Path | None, float | None]:
    current = frames[guide_index].get("image", "")
    if current and resolve(current).is_file():
        return current, None, None
    chunks_state = outpaint_chunks_state(state.APP.settings)
    rows = chunks_state.get("rows", [])
    if chunk_index < 0 or chunk_index >= len(rows):
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    row = rows[chunk_index]
    fps = float(row.get("fps", 24) or 24)
    source_seconds = _guide_source_seconds(row, int(frames[guide_index].get("frame_idx", 0)), fps)
    source_text = state.APP.outpaint_source_for()
    if not source_text:
        raise RuntimeError("No source material is selected.")
    prepared = ensure_outpaint_prepared_canvas(source_text, state.APP.settings.get("outpaint", {}))
    preview_rel = chunk_frame_preview(prepared, source_seconds, f"guide_edit_{chunk_index}_{guide_index}")
    if not preview_rel or not resolve(preview_rel).is_file():
        raise FileNotFoundError("Could not prepare a guide image for editing.")
    return preview_rel, prepared, source_seconds

def guide_edit_preview_command(chunk_index: int, guide_index: int, instruction: str, mask_data: str = "", sampled_color: str = "") -> tuple[list[str], str]:
    manifest, rows, _manifest_text = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    if guide_index < 0 or guide_index >= len(frames):
        raise IndexError(f"Guide frame {guide_index} not found in chunk {chunk_index + 1}")
    source_rel, _prepared, _source_seconds = _guide_editor_source(chunk_index, guide_index, frames)
    source = resolve(source_rel)
    output = _next_guide_edit_output(manifest, chunk_index, guide_index)
    mask = _save_guide_edit_mask(manifest, chunk_index, guide_index, mask_data)
    if not mask:
        mask_path = _guide_edit_dir(manifest, chunk_index, guide_index) / f"mask_edge_{time.strftime('%Y%m%d_%H%M%S')}.png"
        mask = rel(save_edge_mask_for_image(source, mask_path))
    prompt = _guide_edit_prompt(instruction, sampled_color)
    values = state.APP.settings.get("references", {})
    config = current_config()
    comfy_dir = config.get("comfy_dir", str(ROOT / "tools" / "comfyui"))
    comfy_url = values.get("comfy_url") or config.get("comfy_url", "http://127.0.0.1:8188")
    comfy_output = comfy_output_root_for(config)
    workflow = qwen_masked_workflow_for(values, config)
    if not workflow:
        raise RuntimeError("Guide editing needs a Qwen masked edit workflow. ARP's bundled masked workflow was not found, and no custom workflow is set.")
    if not resolve(workflow).is_file():
        raise FileNotFoundError(f"Masked edit workflow not found: {workflow}")
    cmd = [
        sys.executable, "-u", str(SCRIPTS / "edit_reference_image.py"),
        "--source-image", str(source),
        "--mask", mask,
        "--output", rel(output),
        "--workflow", workflow,
        "--comfy-url", comfy_url,
        "--comfy-dir", comfy_dir,
        "--comfy-output-root", comfy_output,
        "--model-backend", values.get("model_backend", "gguf"),
        "--gguf-model", values.get("gguf_model", QWEN_IMAGE_EDIT_MODEL),
        "--instruction", prompt,
        "--no-normalize-to-source-size",
        "--force",
    ]
    if values.get("prompt_node_id"):
        cmd.extend(["--prompt-node-id", values["prompt_node_id"]])
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "chunk_index": chunk_index,
                "guide_index": guide_index,
                "source_image": rel(source),
                "mask": mask,
                "instruction": instruction,
                "sampled_color": sampled_color,
                "prompt": prompt,
                "output": rel(output),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return cmd, rel(output)

def sam_guide_mask(chunk_index: int, guide_index: int, points: list[dict], width: int, height: int, fallback_path: str = "") -> dict[str, str]:
    manifest, rows, _manifest_text = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    if guide_index < 0 or guide_index >= len(frames):
        raise IndexError(f"Guide frame {guide_index} not found in chunk {chunk_index + 1}")
    source_rel = frames[guide_index].get("image", "") or fallback_path
    if not source_rel:
        source_rel, _prepared, _source_seconds = _guide_editor_source(chunk_index, guide_index, frames)
    source = resolve(source_rel)
    if not source.is_file():
        raise FileNotFoundError(f"Guide image not found: {source_rel}")
    return sam2_mask_for_image(source, points, width, height)

def accept_guide_edit(chunk_index: int, guide_index: int, preview_path: str) -> dict:
    manifest, rows, _ = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    if guide_index < 0 or guide_index >= len(frames):
        raise IndexError(f"Guide frame {guide_index} not found in chunk {chunk_index + 1}")
    preview = resolve(preview_path)
    if not preview.is_file():
        raise FileNotFoundError(f"Edited guide not found: {preview}")
    previous = frames[guide_index].get("image", "")
    frames[guide_index]["image_previous"] = previous
    frames[guide_index]["image"] = rel(preview)
    frames[guide_index].pop("seed", None)
    _save_guide_frames(manifest, chunk_index, frames)
    state.APP.log.append(f"Accepted edited guide frame {guide_index + 1} for chunk {chunk_index + 1}: {rel(preview)}")
    return {"image": rel(preview), "previous": previous}

def save_guide_paint(chunk_index: int, guide_index: int, image_data: str) -> dict:
    if not image_data:
        raise ValueError("No painted image data was provided.")
    import base64

    payload = image_data.split(",", 1)[1] if "," in image_data else image_data
    raw = base64.b64decode(payload)
    manifest, rows, _ = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    if guide_index < 0 or guide_index >= len(frames):
        raise IndexError(f"Guide frame {guide_index} not found in chunk {chunk_index + 1}")
    output = _next_guide_edit_output(manifest, chunk_index, guide_index)
    output.write_bytes(raw)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "chunk_index": chunk_index,
                "guide_index": guide_index,
                "source_image": frames[guide_index].get("image", ""),
                "instruction": "Manual recolour paint",
                "output": rel(output),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return accept_guide_edit(chunk_index, guide_index, rel(output))

def revert_guide_edit(chunk_index: int, guide_index: int) -> dict:
    manifest, rows, _ = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    if guide_index < 0 or guide_index >= len(frames):
        raise IndexError(f"Guide frame {guide_index} not found in chunk {chunk_index + 1}")
    current = frames[guide_index].get("image", "")
    previous = frames[guide_index].get("image_previous", "")
    if not previous:
        raise RuntimeError("No previous guide image is recorded for this guide frame.")
    if not resolve(previous).is_file():
        raise FileNotFoundError(f"Previous guide image not found: {previous}")
    frames[guide_index]["image_previous"] = current
    frames[guide_index]["image"] = previous
    _save_guide_frames(manifest, chunk_index, frames)
    state.APP.log.append(f"Reverted guide frame {guide_index + 1} for chunk {chunk_index + 1}: {previous}")
    return {"image": previous, "previous": current}
