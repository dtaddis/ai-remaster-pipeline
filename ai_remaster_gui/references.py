from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

from . import state
from .config import IMAGE_EXTS, PREVIEW_DIR, QWEN_IMAGE_EDIT_MODEL, REFERENCE_PROMPT, REFERENCE_PROMPT_SUFFIX, ROOT, SCRIPTS, comfy_output_root_for, current_config
from .file_dialogs import browse_path
from .manifests import manifest_source_video, read_manifest, read_manifest_details, update_manifest_row, write_manifest_details
from .media import extract_video_frame_at, extract_video_frame_at_frame, ffprobe_info, local_tool, safe_preview_name
from .naming import manifest_for_outpainted
from .paths import format_timecode, rel, resolve, safe_stem
from .runtime_settings import qwen_masked_workflow_for, qwen_workflow_for
from .sam_masks import sam2_mask_for_image

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import artifact_ids as aid  # noqa: E402
from reference_sets import (  # noqa: E402
    EXTRA_REFERENCES_FIELD,
    additional_references,
    append_reference_item,
    encode_additional_references,
    reference_items,
    remove_reference_item,
    update_reference_item,
)


def recomposition_output_for(outpainted_text: str) -> str:
    if not outpainted_text:
        return ""
    outpainted = resolve(outpainted_text)
    ident = aid.recomp_identity(outpainted.stem)
    return rel(ROOT / "output" / "reassembled" / aid.artifact_name(aid.source_word(outpainted.name), "recomp", ident, "mp4"))

def colorized_outputs_for_manifest(manifest_text: str, method: str = "deepexemplar") -> list[str]:
    if method == "both":
        return [path for path in (colorized_output_for_manifest(manifest_text, "deepexemplar"), colorized_output_for_manifest(manifest_text, "colormnet")) if path]
    output = colorized_output_for_manifest(manifest_text, method)
    return [output] if output else []

def colorized_output_for_manifest(manifest_text: str, method: str = "deepexemplar") -> str:
    if not manifest_text:
        return ""
    if method == "both":
        return ""
    manifest = resolve(manifest_text)
    ident = aid.colorized_identity(manifest.stem, method)
    source_video = manifest_source_video(manifest)
    name_src = resolve(source_video).name if source_video else manifest.name
    return rel(ROOT / "intermediate" / "outpainted_colorized" / aid.artifact_name(aid.source_word(name_src), "color", ident, "mp4"))

def color_reference_outputs(manifest_text: str) -> list[str]:
    if not manifest_text:
        return []
    manifest = resolve(manifest_text)
    if not manifest.is_file():
        return []
    rows = read_manifest(manifest)
    return [item.get("color_reference", "") for row in rows for item in reference_items(row) if item.get("color_reference")]

def shot_views(settings: dict[str, dict[str, str]], generate_previews: bool = True) -> dict[str, object]:
    shots_settings = settings.get("shots", {})
    references_manifest = settings.get("references", {}).get("manifest", "")
    colour_manifest = settings.get("colour", {}).get("manifest", "") or references_manifest
    shots_manifest = manifest_for_outpainted(shots_settings.get("outpainted_video", ""))
    if not shots_manifest or not resolve(shots_manifest).is_file():
        for configured_manifest in (shots_settings.get("manifest", ""), references_manifest, colour_manifest):
            if configured_manifest and resolve(configured_manifest).is_file():
                shots_manifest = configured_manifest
                break
    upscale_manifest = colour_manifest or references_manifest or shots_manifest
    return {
        "shots_manifest": shots_manifest,
        "shots": shot_rows(shots_manifest, include_previews=True, generate_previews=generate_previews),
        "references_manifest": references_manifest,
        "references": shot_rows(references_manifest),
        "colour_manifest": colour_manifest,
        "colour": shot_rows(colour_manifest),
        "upscale_manifest": upscale_manifest,
        "upscale": shot_rows(upscale_manifest),
    }

def shot_rows(
    manifest_text: str,
    include_previews: bool = False,
    preview_indices: Iterable[int] | None = None,
    generate_previews: bool = True,
) -> list[dict[str, object]]:
    if not manifest_text:
        return []
    path = resolve(manifest_text)
    rows = read_manifest(path)
    fps = manifest_fps(path)
    out: list[dict[str, object]] = []
    preview_index_set = set(preview_indices) if preview_indices is not None else None
    cached_preview_source_text = manifest_source_video(path) if include_previews and not generate_previews else ""
    cached_preview_source = resolve(cached_preview_source_text) if cached_preview_source_text else None
    cached_preview_dir = PREVIEW_DIR / "shot_scrub" / safe_preview_name(path) if cached_preview_source else None
    start = 0.0
    start_frame = 0
    for index, row in enumerate(rows):
        row_start_frame = optional_int(row.get("start_frame"))
        row_end_frame = optional_int(row.get("end_frame"))
        if row_start_frame is not None:
            start_frame = max(0, row_start_frame)
            start = start_frame / fps
        end = parse_time_seconds(row.get("end", "")) or start
        if row_end_frame is not None:
            end_frame_exclusive = max(start_frame + 1, row_end_frame)
            end = end_frame_exclusive / fps
        else:
            end_frame_exclusive = max(start_frame + 1, int(round(end * fps)))
        selected_frame = optional_int(row.get("selected_frame"))
        selected = (selected_frame / fps) if selected_frame is not None else selected_seconds_from_reference(row.get("source_reference", "")) or ((start + end) / 2 if end > start else start)
        selected = max(start, min(end, selected))
        color_reference = row.get("color_reference", "")
        item_references = []
        for reference_index, reference in enumerate(reference_items(row)):
            reference_frame = optional_int(reference.get("selected_frame"))
            reference_seconds = (reference_frame / fps) if reference_frame is not None else selected
            reference_seconds = max(start, min(end, reference_seconds))
            item_color = reference.get("color_reference", "")
            item_references.append({
                "reference_index": reference_index,
                "selected_frame": reference_frame if reference_frame is not None else int(round(reference_seconds * fps)),
                "selected_time": round(reference_seconds, 3),
                "selected_label": format_timecode(reference_seconds),
                "source_reference": reference.get("source_reference", ""),
                "color_reference": item_color,
                "source_reference_mtime": file_mtime(reference.get("source_reference", "")),
                "color_reference_mtime": file_mtime(item_color),
                "prompt": reference.get("prompt", ""),
            })
        color_reference_versions = reference_edit_versions(manifest_text, index)
        item = {
                "index": index,
                "enabled": row.get("enabled", "true"),
                "start": round(start, 6),
                "end": round(end, 6),
                "fps": fps,
                "previous_start_frame": out[-1]["start_frame"] if out else 0,
                "start_frame": start_frame,
                "end_frame": max(start_frame, end_frame_exclusive - 1),
                "end_boundary_frame": end_frame_exclusive,
                "next_end_boundary_frame": optional_int(rows[index + 1].get("end_frame")) if index + 1 < len(rows) else end_frame_exclusive + 1,
                "duration": round(max(0.0, end - start), 3),
                "selected_time": round(selected, 3),
                "selected_frame": selected_frame if selected_frame is not None else int(round(selected * fps)),
                "start_label": format_timecode(start),
                "end_label": format_timecode(end),
                "selected_label": format_timecode(selected),
                "source_reference": row.get("source_reference", ""),
                "color_reference": color_reference,
                "source_reference_mtime": file_mtime(row.get("source_reference", "")),
                "color_reference_mtime": file_mtime(color_reference),
                "recent_color_references": recent_color_references(rows, index),
                "color_reference_versions": color_reference_versions,
                "color_reference_edited": bool(color_reference and color_reference in color_reference_versions),
                "masked_edit_available": bool(state.APP.settings.get("references", {}).get("masked_workflow", "")),
                "can_merge_next": index < len(rows) - 1,
                "can_split": end - start >= 0.1,
                "can_fade_next": index < len(rows) - 1,
                "fade_to_next": row.get("fade_to_next", "false"),
                "crossfade_seconds": row.get("crossfade_seconds", ""),
                "upscale_strength": row.get("upscale_strength", ""),
                "prompt": row.get("prompt", ""),
                "reference_items": item_references,
                "reference_count": len(item_references),
            }
        if include_previews and (preview_index_set is None or index in preview_index_set):
            last_frame = max(start_frame, end_frame_exclusive - 1)
            mid_frame = start_frame + max(0, (end_frame_exclusive - start_frame - 1) // 2)
            mid = mid_frame / fps
            end_preview = last_frame / fps
            for key, value, frame in (
                ("start_preview", start, start_frame),
                ("middle_preview", mid, mid_frame),
                ("end_preview", end_preview, last_frame),
            ):
                try:
                    item[key] = preview_reference_frame(manifest_text, index, value, frame=frame) if generate_previews else cached_preview_reference_frame_path(cached_preview_source, cached_preview_dir, index, frame)
                except Exception:
                    item[key] = ""
        out.append(item)
        start_frame = end_frame_exclusive
        start = end
    return out

def shot_rows_for_indices(manifest_text: str, indices: Iterable[int], include_previews: bool = True) -> list[dict[str, object]]:
    wanted = {index for index in indices if index >= 0}
    if not wanted:
        return []
    rows = shot_rows(manifest_text, include_previews=include_previews, preview_indices=wanted)
    return [row for row in rows if int(row.get("index", -1)) in wanted]

def recent_color_references(rows: list[dict[str, str]], row_index: int, limit: int = 8) -> list[str]:
    previous: list[tuple[int, str]] = []
    later: list[tuple[int, str]] = []
    for index, row in enumerate(rows):
        for reference_index, reference in enumerate(reference_items(row)):
            if index == row_index and reference_index == 0:
                continue
            candidate = reference.get("color_reference", "")
            if not candidate or not resolve(candidate).is_file():
                continue
            item = (abs(index - row_index), candidate)
            if index <= row_index:
                previous.append(item)
            else:
                later.append(item)
    ordered = [path for _distance, path in sorted(previous, key=lambda item: item[0])]
    ordered.extend(path for _distance, path in sorted(later, key=lambda item: item[0]))
    return ordered[:limit]

def reference_edit_dir(manifest_text: str, index: int) -> Path:
    stem = safe_stem(resolve(manifest_text).stem or "references")
    return ROOT / "intermediate" / "outpainted_references_color_edits" / stem / f"shot_{index:04d}"

def reference_edit_versions(manifest_text: str, index: int, limit: int = 8) -> list[str]:
    folder = reference_edit_dir(manifest_text, index)
    if not folder.exists():
        return []
    paths = sorted(
        (path for path in folder.glob("edit_*.png") if path.is_file()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    return [rel(path) for path in paths[:limit]]

def next_reference_edit_output(manifest_text: str, index: int) -> Path:
    folder = reference_edit_dir(manifest_text, index)
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

def save_reference_edit_mask(manifest_text: str, index: int, mask_data: str) -> str:
    if not mask_data:
        return ""
    import base64

    payload = mask_data.split(",", 1)[1] if "," in mask_data else mask_data
    raw = base64.b64decode(payload)
    folder = reference_edit_dir(manifest_text, index)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"mask_{time.strftime('%Y%m%d_%H%M%S')}.png"
    path.write_bytes(raw)
    return rel(path)

def reference_edit_prompt(instruction: str, sampled_color: str = "") -> str:
    text = instruction.strip()
    parts = [text]
    color = sampled_color.strip()
    if color:
        parts.append(f"Use the sampled colour exactly where relevant: {color}, including its brightness/value.")
    if color:
        parts.append(
            "If the requested material colour is lighter than the current one, raise the masked area's "
            "luminance to the target colour while preserving texture, shadows, edges, composition, and identity; "
            "do not merely darken or tint the existing colour."
        )
    prompt = " ".join(part for part in parts if part).strip()
    return prompt or "Refine this colour reference while preserving the original composition and identity."

def reference_edit_preview_command(manifest_text: str, index: int, instruction: str, mask_data: str = "", sampled_color: str = "") -> tuple[list[str], str]:
    _manifest, _row, _source, output = reference_row_io(manifest_text, index)
    current = output
    if not resolve(current).is_file():
        raise FileNotFoundError(f"Colour reference does not exist yet: {current}")
    values = state.APP.settings.get("references", {})
    config = current_config()
    mask = save_reference_edit_mask(manifest_text, index, mask_data)
    edit_output = next_reference_edit_output(manifest_text, index)
    prompt = reference_edit_prompt(instruction, sampled_color)
    comfy_dir = config.get("comfy_dir", str(ROOT / "tools" / "comfyui"))
    comfy_url = values.get("comfy_url") or config.get("comfy_url", "http://127.0.0.1:8188")
    comfy_output = comfy_output_root_for(config)
    if mask:
        workflow = qwen_masked_workflow_for(values, config)
        if not workflow:
            raise RuntimeError("Masked editing needs a Qwen masked edit workflow. ARP's bundled masked workflow was not found, and no custom workflow is set.")
        if not resolve(workflow).is_file():
            raise FileNotFoundError(f"Masked edit workflow not found: {workflow}")
        cmd = [
            sys.executable,
            "-u",
            str(SCRIPTS / "edit_reference_image.py"),
            "--source-image",
            current,
            "--mask",
            mask,
            "--output",
            rel(edit_output),
            "--workflow",
            workflow,
            "--comfy-url",
            comfy_url,
            "--comfy-dir",
            comfy_dir,
            "--comfy-output-root",
            comfy_output,
            "--model-backend",
            values.get("model_backend", "gguf"),
            "--gguf-model",
            values.get("gguf_model", QWEN_IMAGE_EDIT_MODEL),
            "--instruction",
            prompt,
            "--force",
        ]
    else:
        workflow = qwen_workflow_for(values, config)
        cmd = [
            sys.executable,
            "-u",
            str(SCRIPTS / "generate_single_reference.py"),
            "--source-image",
            current,
            "--output",
            rel(edit_output),
            "--workflow",
            workflow,
            "--comfy-url",
            comfy_url,
            "--comfy-dir",
            comfy_dir,
            "--comfy-output-root",
            comfy_output,
            "--model-backend",
            values.get("model_backend", "gguf"),
            "--gguf-model",
            values.get("gguf_model", QWEN_IMAGE_EDIT_MODEL),
            "--prompt",
            prompt,
            "--prompt-suffix",
            "",
            "--load-image-node-id",
            values.get("load_image_node_id", "auto"),
            "--save-node-id",
            values.get("save_node_id", "auto"),
            "--force",
        ]
        if values.get("prompt_node_id"):
            cmd.extend(["--prompt-node-id", values["prompt_node_id"]])
    meta = edit_output.with_suffix(edit_output.suffix + ".json")
    meta.write_text(
        json.dumps(
            {
                "manifest": rel(resolve(manifest_text)),
                "index": index,
                "source_image": current,
                "mask": mask,
                "instruction": instruction,
                "sampled_color": sampled_color,
                "prompt": prompt,
                "output": rel(edit_output),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return cmd, rel(edit_output)

def accept_reference_edit(manifest_text: str, index: int, preview_path: str) -> dict[str, str]:
    manifest, row, _source, current = reference_row_io(manifest_text, index)
    preview = resolve(preview_path)
    if not preview.is_file():
        raise FileNotFoundError(f"Edited reference not found: {preview}")
    update_manifest_row(manifest, index, {"color_reference": rel(preview), "color_reference_previous": current})
    state.APP.log.append(f"Accepted edited colour reference for shot {index + 1}: {rel(preview)}")
    return {"color_reference": rel(preview), "previous": current}

def revert_reference_edit(manifest_text: str, index: int) -> dict[str, str]:
    manifest, row, _source, current = reference_row_io(manifest_text, index)
    previous = row.get("color_reference_previous", "")
    if not previous:
        versions = reference_edit_versions(manifest_text, index, limit=20)
        candidates = [path for path in versions if path != current]
        previous = candidates[0] if candidates else ""
    if not previous:
        raise RuntimeError("No previous colour reference is recorded for this shot.")
    if not resolve(previous).is_file():
        raise FileNotFoundError(f"Previous colour reference not found: {previous}")
    update_manifest_row(manifest, index, {"color_reference": previous, "color_reference_previous": current})
    state.APP.log.append(f"Reverted edited colour reference for shot {index + 1}: {previous}")
    return {"color_reference": previous, "previous": current}

def save_reference_paint(manifest_text: str, index: int, image_data: str) -> dict[str, str]:
    if not image_data:
        raise ValueError("No painted image data was provided.")
    import base64

    payload = image_data.split(",", 1)[1] if "," in image_data else image_data
    raw = base64.b64decode(payload)
    _manifest, _row, _source, current = reference_row_io(manifest_text, index)
    output = next_reference_edit_output(manifest_text, index)
    output.write_bytes(raw)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "manifest": rel(resolve(manifest_text)),
                "index": index,
                "source_image": current,
                "instruction": "Manual recolour paint",
                "output": rel(output),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return accept_reference_edit(manifest_text, index, rel(output))

def sam_reference_mask(manifest_text: str, index: int, points: list[dict], width: int, height: int, tolerance: int = 10) -> dict[str, str]:
    _manifest, _row, _source, output = reference_row_io(manifest_text, index)
    source = resolve(output)
    if not source.is_file():
        raise FileNotFoundError(f"Colour reference does not exist yet: {output}")
    return sam2_mask_for_image(source, points, width, height)

def manifest_fps(path: Path) -> float:
    source = resolve(manifest_source_video(path))
    try:
        rate = ffprobe_info(source).get("frame_rate", "")
        if rate.endswith(" fps"):
            return float(rate[:-4])
    except Exception:
        pass
    return 24.0

def file_mtime(path_text: str) -> int:
    if not path_text:
        return 0
    path = resolve(path_text)
    try:
        return int(path.stat().st_mtime)
    except OSError:
        return 0

def parse_time_seconds(value: str) -> float:
    value = str(value or "").strip()
    if not value:
        return 0.0
    parts = value.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(value)
    except ValueError:
        return 0.0

def optional_int(value) -> int | None:
    try:
        text = str(value).strip()
        if text == "":
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None

def selected_seconds_from_reference(path_text: str) -> float:
    stem = Path(path_text).stem
    parts = stem.split("_")
    if len(parts) < 3 or parts[0] != "cut":
        return 0.0
    time_parts = parts[-1].split(".")
    try:
        if len(time_parts) >= 3:
            seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
            if len(time_parts) > 3:
                seconds += float("0." + "".join(time_parts[3:]))
            return seconds
    except ValueError:
        return 0.0
    return 0.0

def reference_name_for_time(index: int, seconds: float) -> str:
    return f"cut_{index:04d}_{format_timecode(seconds).replace(':', '.')}.png"

def color_reference_for_source(source_reference: str) -> str:
    source = resolve(source_reference)
    try:
        relative = source.relative_to(ROOT / "intermediate" / "outpainted_references")
        return rel(ROOT / "intermediate" / "outpainted_references_color" / relative)
    except ValueError:
        return rel(source.with_name(source.stem + "_color" + source.suffix))

def _write_reference_row(manifest: Path, source_video: str, fieldnames: list[str], rows: list[dict[str, str]], index: int) -> None:
    for key in rows[index]:
        if key not in fieldnames:
            fieldnames.append(key)
    write_manifest_details(manifest, source_video, fieldnames, rows)


def delete_color_reference(manifest_text: str, index: int, reference_index: int = 0) -> dict[str, str]:
    manifest = resolve(manifest_text)
    source_video, fieldnames, rows = read_manifest_details(manifest)
    if index < 0 or index >= len(rows):
        raise IndexError(f"Manifest row {index} is out of range.")
    items = reference_items(rows[index])
    if reference_index < 0 or reference_index >= len(items):
        raise IndexError(f"Reference {reference_index + 1} is out of range.")
    target = items[reference_index].get("color_reference", "")
    if not target:
        raise RuntimeError("Reference does not have a color_reference path.")
    path = resolve(target)
    sig = path.with_suffix(path.suffix + ".sig.json")
    deleted = []
    for item in (path, sig):
        if item.exists() and item.is_file():
            item.unlink()
            deleted.append(rel(item))
    update_reference_item(rows[index], reference_index, {"color_reference": target})
    _write_reference_row(manifest, source_video, fieldnames, rows, index)
    state.APP.log.append(f"Deleted colour reference {reference_index + 1} for shot {index + 1}: {target}")
    return {"deleted": ", ".join(deleted), "color_reference": target}

def install_custom_color_reference(manifest_text: str, index: int, reference_index: int = 0) -> dict[str, str]:
    manifest = resolve(manifest_text)
    source_video, fieldnames, rows = read_manifest_details(manifest)
    if index < 0 or index >= len(rows):
        raise IndexError("Shot index out of range.")
    items = reference_items(rows[index])
    if reference_index < 0 or reference_index >= len(items):
        raise IndexError(f"Reference {reference_index + 1} is out of range.")
    item = items[reference_index]

    selected = browse_path("file", item.get("color_reference", "") or item.get("source_reference", ""))
    if not selected:
        return {"selected": "", "color_reference": item.get("color_reference", "")}

    source = resolve(selected)
    if source.suffix.lower() not in IMAGE_EXTS:
        raise RuntimeError("Choose a PNG or JPEG image for the custom color reference.")
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(source)

    current_target = item.get("color_reference", "")
    if current_target:
        target_base = resolve(current_target)
        target = target_base.with_suffix(source.suffix.lower())
    elif item.get("source_reference"):
        target = resolve(color_reference_for_source(item["source_reference"])).with_suffix(source.suffix.lower())
    else:
        target = ROOT / "intermediate" / "outpainted_references_color" / "custom" / f"shot_{index + 1:04d}_ref_{reference_index + 1:02d}{source.suffix.lower()}"

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    update_reference_item(rows[index], reference_index, {"color_reference": rel(target)})
    _write_reference_row(manifest, source_video, fieldnames, rows, index)
    state.APP.log.append(f"Installed custom color reference {reference_index + 1} for shot {index + 1}: {rel(target)}")
    return {"selected": selected, "color_reference": rel(target)}

def extract_reference_frame(manifest_text: str, index: int, seconds: float, frame: int | None = None, reference_index: int = 0, append: bool = False) -> dict[str, str]:
    manifest = resolve(manifest_text)
    source_video, fieldnames, rows = read_manifest_details(manifest)
    if not source_video:
        raise RuntimeError("Manifest does not record a source_video, so ARP cannot rescrub this shot.")
    if index < 0 or index >= len(rows):
        raise IndexError(f"Manifest row {index} is out of range.")
    items = reference_items(rows[index])
    if append:
        reference_index = len(items)
        old_reference = items[0].get("source_reference", "")
    elif reference_index < 0 or reference_index >= len(items):
        raise IndexError(f"Reference {reference_index + 1} is out of range.")
    else:
        old_reference = items[reference_index].get("source_reference", "")
    if old_reference:
        folder = resolve(old_reference).parent
    else:
        source = resolve(source_video)
        folder = ROOT / "intermediate" / "outpainted_references" / safe_stem(source.name)
    new_source = folder / reference_name_for_time(index, seconds)
    ffmpeg = local_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Run install_windows.bat to install local FFmpeg for shot scrubbing.")
    new_source.parent.mkdir(parents=True, exist_ok=True)
    selected_frame = max(0, int(frame)) if frame is not None else int(round(max(0.0, seconds) * manifest_fps(manifest)))
    if frame is not None:
        vf = f"trim=start_frame={selected_frame}:end_frame={selected_frame + 1},setpts=PTS-STARTPTS"
        command = [ffmpeg, "-y", "-i", str(resolve(source_video)), "-vf", vf, "-frames:v", "1", "-q:v", "2", str(new_source)]
    else:
        command = [ffmpeg, "-y", "-ss", f"{seconds:.3f}", "-i", str(resolve(source_video)), "-frames:v", "1", "-q:v", "2", str(new_source)]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg failed").strip())
    new_color = color_reference_for_source(rel(new_source))
    values = {
        "source_reference": rel(new_source),
        "color_reference": new_color,
        "selected_frame": str(selected_frame),
    }
    if append:
        append_reference_item(rows[index], values)
    else:
        update_reference_item(rows[index], reference_index, values)
    _write_reference_row(manifest, source_video, fieldnames, rows, index)
    verb = "Added" if append else "Updated"
    state.APP.log.append(f"{verb} shot {index + 1} reference {reference_index + 1} at {format_timecode(seconds)}: {rel(new_source)}")
    return {**values, "reference_index": str(reference_index)}


def add_reference_frame(manifest_text: str, index: int, seconds: float, frame: int | None = None) -> dict[str, str]:
    return extract_reference_frame(manifest_text, index, seconds, frame, append=True)


def remove_additional_reference(manifest_text: str, index: int, reference_index: int) -> dict[str, str]:
    manifest = resolve(manifest_text)
    source_video, fieldnames, rows = read_manifest_details(manifest)
    if index < 0 or index >= len(rows):
        raise IndexError(f"Manifest row {index} is out of range.")
    removed = remove_reference_item(rows[index], reference_index)
    _write_reference_row(manifest, source_video, fieldnames, rows, index)
    state.APP.log.append(f"Removed reference {reference_index + 1} from shot {index + 1}.")
    return {"removed_source_reference": removed.get("source_reference", ""), "reference_index": str(reference_index)}

def preview_reference_frame(manifest_text: str, index: int, seconds: float, frame: int | None = None) -> str:
    manifest = resolve(manifest_text)
    source_video, _fields, rows = read_manifest_details(manifest)
    if not source_video:
        raise RuntimeError("Manifest does not record a source_video, so ARP cannot preview this shot.")
    if index < 0 or index >= len(rows):
        raise IndexError(f"Manifest row {index} is out of range.")
    source = resolve(source_video)
    target_dir = PREVIEW_DIR / "shot_scrub" / safe_preview_name(manifest)
    target_dir.mkdir(parents=True, exist_ok=True)
    if frame is not None:
        return extract_video_frame_at_frame(source, target_dir, f"shot_{index:04d}_frame_{max(0, int(frame)):010d}", int(frame))
    target = target_dir / f"shot_{index:04d}_{int(seconds * 1000):010d}.jpg"
    if target.exists():
        return rel(target)
    ffmpeg = local_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Run install_windows.bat to install local FFmpeg for shot previews.")
    command = [ffmpeg, "-y", "-ss", f"{seconds:.3f}", "-i", str(source), "-frames:v", "1", "-q:v", "4", str(target)]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg failed").strip())
    return rel(target)

def cached_preview_reference_frame(manifest_text: str, index: int, frame: int) -> str:
    manifest = resolve(manifest_text)
    source_video, _fields, rows = read_manifest_details(manifest)
    if not source_video or index < 0 or index >= len(rows):
        return ""
    return cached_preview_reference_frame_path(resolve(source_video), PREVIEW_DIR / "shot_scrub" / safe_preview_name(manifest), index, frame)

def cached_preview_reference_frame_path(source: Path | None, target_dir: Path | None, index: int, frame: int) -> str:
    if source is None or target_dir is None:
        return ""
    suffix = f"shot_{index:04d}_frame_{max(0, int(frame)):010d}"
    candidate = target_dir / f"{safe_preview_name(source)}_{suffix}.jpg"
    if len(str(candidate)) > 240:
        key = hashlib.sha256(f"{source}\0{suffix}\0{max(0, int(frame))}".encode()).hexdigest()[:24]
        candidate = target_dir / f"{key}.jpg"
    try:
        if candidate.exists() and candidate.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            return rel(candidate)
    except OSError:
        return ""
    return ""

def reference_row_io(manifest_text: str, index: int, reference_index: int = 0) -> tuple[Path, dict[str, str], str, str]:
    manifest = resolve(manifest_text)
    _source_video, _fields, rows = read_manifest_details(manifest)
    if index < 0 or index >= len(rows):
        raise IndexError(f"Manifest row {index} is out of range.")
    row = rows[index]
    items = reference_items(row)
    if reference_index < 0 or reference_index >= len(items):
        raise IndexError(f"Reference {reference_index + 1} is out of range.")
    reference = items[reference_index]
    source = reference.get("source_reference", "")
    output = reference.get("color_reference", "")
    if not source or not output:
        raise RuntimeError("Reference must have source_reference and color_reference.")
    effective_row = dict(row)
    effective_row.update(reference)
    return manifest, effective_row, source, output

def reference_regeneration_command(manifest_text: str, index: int, reference_index: int = 0) -> tuple[list[str], str]:
    _manifest, row, source, output = reference_row_io(manifest_text, index, reference_index)
    values = state.APP.settings.get("references", {})
    config = current_config()
    workflow = qwen_workflow_for(values, config)
    if not workflow:
        raise RuntimeError("No Qwen Image Edit workflow found. Install/configure ComfyUI first.")
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPTS / "generate_single_reference.py"),
        "--source-image",
        source,
        "--output",
        output,
        "--workflow",
        workflow,
        "--comfy-url",
        values.get("comfy_url") or config.get("comfy_url", "http://127.0.0.1:8188"),
        "--comfy-dir",
        config.get("comfy_dir", str(ROOT / "tools" / "comfyui")),
        "--comfy-output-root",
        comfy_output_root_for(config),
        "--model-backend",
        values.get("model_backend", "gguf"),
        "--gguf-model",
        values.get("gguf_model", QWEN_IMAGE_EDIT_MODEL),
        "--prompt",
        values.get("prompt", REFERENCE_PROMPT),
        "--prompt-suffix",
        values.get("prompt_suffix", REFERENCE_PROMPT_SUFFIX),
        "--load-image-node-id",
        values.get("load_image_node_id", "auto"),
        "--save-node-id",
        values.get("save_node_id", "auto"),
        "--force",
    ]
    if values.get("prompt_node_id"):
        cmd.extend(["--prompt-node-id", values["prompt_node_id"]])
    if row.get("prompt"):
        cmd.extend(["--add-prompt", row["prompt"]])
    return cmd, output

def openai_reference_regeneration_command(manifest_text: str, index: int, reference_index: int = 0) -> tuple[list[str], str]:
    manifest, _row, _source, output = reference_row_io(manifest_text, index, reference_index)
    values = state.APP.settings.get("references", {})
    token = values.get("openai_api_key", "").strip()
    if not token:
        raise RuntimeError("Add your OpenAI API key in Settings before generating with OpenAI.")
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPTS / "openai_generate_reference.py"),
        "--manifest",
        rel(manifest),
        "--row-index",
        str(index),
        "--reference-index",
        str(reference_index),
        "--api-key",
        token,
        "--model",
        values.get("openai_image_model", "gpt-image-2") or "gpt-image-2",
        "--prompt",
        values.get("prompt", REFERENCE_PROMPT),
        "--prompt-suffix",
        values.get("prompt_suffix", REFERENCE_PROMPT_SUFFIX),
    ]
    if values.get("openai_image_size"):
        cmd.extend(["--size", values["openai_image_size"]])
    if values.get("openai_image_quality"):
        cmd.extend(["--quality", values["openai_image_quality"]])
    if values.get("openai_send_references", "false") == "true":
        cmd.extend(["--reference-count", "3"])
    cmd.append("--force")
    return cmd, output

def regenerate_reference_image(manifest_text: str, index: int, reference_index: int = 0) -> dict[str, str]:
    cmd, output = reference_regeneration_command(manifest_text, index, reference_index)
    state.APP.log.append("> " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in result.stdout.splitlines():
        state.APP.log.append(line)
    if result.returncode != 0:
        raise RuntimeError(f"Reference regeneration failed with exit code {result.returncode}.")
    state.APP.log.append(f"Regenerated colour reference for shot {index + 1}: {output}")
    return {"color_reference": output}

def manifest_frame_spans(manifest: Path, rows: list[dict[str, str]], fps: float) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start_frame = 0
    for row in rows:
        row_start = optional_int(row.get("start_frame"))
        row_end = optional_int(row.get("end_frame"))
        if row_start is not None:
            start_frame = max(0, row_start)
        if row_end is None:
            end_seconds = parse_time_seconds(row.get("end", ""))
            row_end = max(start_frame + 1, int(round(end_seconds * fps)))
        end_frame = max(start_frame + 1, row_end)
        spans.append((start_frame, end_frame))
        start_frame = end_frame
    return spans

def ensure_frame_fields(fieldnames: list[str]) -> None:
    for key in ("start_frame", "end_frame"):
        if key not in fieldnames:
            fieldnames.append(key)

def set_row_span(row: dict[str, str], start_frame: int, end_frame: int, fps: float) -> None:
    row["start_frame"] = str(max(0, int(start_frame)))
    row["end_frame"] = str(max(int(start_frame) + 1, int(end_frame)))
    row["end"] = format_timecode(int(row["end_frame"]) / fps)


def merge_manifest_shots(manifest_text: str, index: int) -> dict[str, str]:
    manifest = resolve(manifest_text)
    source_video, fieldnames, rows = read_manifest_details(manifest)
    if index < 0 or index >= len(rows) - 1:
        raise IndexError(f"Shot {index + 1} cannot be merged because there is no following shot.")
    for key in ("fade_to_next", "crossfade_seconds"):
        if key not in fieldnames:
            fieldnames.append(key)
    fps = manifest_fps(manifest)
    spans = manifest_frame_spans(manifest, rows, fps)
    rows[index]["end"] = rows[index + 1].get("end", rows[index].get("end", ""))
    ensure_frame_fields(fieldnames)
    set_row_span(rows[index], spans[index][0], spans[index + 1][1], fps)
    merged_references = additional_references(rows[index])
    merged_references.extend(
        item for item in reference_items(rows[index + 1])
        if item.get("source_reference") or item.get("color_reference")
    )
    rows[index][EXTRA_REFERENCES_FIELD] = encode_additional_references(merged_references)
    if EXTRA_REFERENCES_FIELD not in fieldnames:
        fieldnames.append(EXTRA_REFERENCES_FIELD)
    rows[index]["fade_to_next"] = rows[index + 1].get("fade_to_next", "")
    rows[index]["crossfade_seconds"] = rows[index + 1].get("crossfade_seconds", "")
    removed = rows.pop(index + 1)
    write_manifest_details(manifest, source_video, fieldnames, rows)
    state.APP.log.append(f"Merged shot {index + 1} with shot {index + 2}; shared reference: {rows[index].get('source_reference', '')}")
    return {"manifest": rel(manifest), "removed_reference": removed.get("source_reference", ""), "new_end": rows[index].get("end", "")}

def split_manifest_shot(manifest_text: str, index: int, seconds: float | None = None) -> dict[str, str]:
    manifest = resolve(manifest_text)
    source_video, fieldnames, rows = read_manifest_details(manifest)
    if index < 0 or index >= len(rows):
        raise IndexError(f"Manifest row {index} is out of range.")

    for key in ("enabled", "end", "source_reference", "color_reference", "prompt", "fade_to_next", "crossfade_seconds", EXTRA_REFERENCES_FIELD):
        if key not in fieldnames:
            fieldnames.append(key)

    fps = manifest_fps(manifest)
    spans = manifest_frame_spans(manifest, rows, fps)
    start_frame, end_frame = spans[index]
    start = start_frame / fps
    end = end_frame / fps
    if end <= start:
        raise RuntimeError(f"Shot {index + 1} cannot be split because its duration is not valid.")

    split_at = (start + end) / 2 if seconds is None else float(seconds)
    split_at = max(start + 0.001, min(end - 0.001, split_at))
    if end - start < 0.1:
        raise RuntimeError(f"Shot {index + 1} is too short to split.")
    split_frame = max(start_frame + 1, min(end_frame - 1, int(round(split_at * fps))))
    split_at = split_frame / fps

    first = dict(rows[index])
    second = dict(rows[index])
    ensure_frame_fields(fieldnames)
    first["start_frame"] = str(start_frame)
    first["end_frame"] = str(split_frame)
    first["end"] = format_timecode(split_at)
    first["source_reference"] = ""
    first["color_reference"] = ""
    first["fade_to_next"] = "false"
    first["crossfade_seconds"] = ""
    first[EXTRA_REFERENCES_FIELD] = ""
    second["start_frame"] = str(split_frame)
    second["end_frame"] = str(end_frame)
    second["end"] = rows[index].get("end", "")
    second["source_reference"] = ""
    second["color_reference"] = ""
    second[EXTRA_REFERENCES_FIELD] = ""
    rows[index] = first
    rows.insert(index + 1, second)
    write_manifest_details(manifest, source_video, fieldnames, rows)
    state.APP.log.append(f"Split shot {index + 1} at {format_timecode(split_at)}")
    return {"manifest": rel(manifest), "split": format_timecode(split_at)}

def update_shot_boundary(manifest_text: str, index: int, edge: str, seconds: float, frame: int | None = None) -> dict[str, str]:
    manifest = resolve(manifest_text)
    source_video, fieldnames, rows = read_manifest_details(manifest)
    if index < 0 or index >= len(rows):
        raise IndexError(f"Manifest row {index} is out of range.")
    fps = manifest_fps(manifest)
    spans = manifest_frame_spans(manifest, rows, fps)
    ensure_frame_fields(fieldnames)
    requested = int(frame) if frame is not None else int(round(seconds * fps))
    if edge == "start":
        if index == 0:
            raise RuntimeError("The first shot must start at 00:00:00.")
        previous_start_frame = spans[index - 1][0]
        current_end_frame = spans[index][1]
        boundary = max(previous_start_frame + 1, min(current_end_frame - 1, requested))
        set_row_span(rows[index - 1], previous_start_frame, boundary, fps)
        set_row_span(rows[index], boundary, current_end_frame, fps)
    elif edge == "end":
        start_frame, current_end_frame = spans[index]
        next_end_frame = spans[index + 1][1] if index + 1 < len(spans) else max(current_end_frame, requested)
        upper = next_end_frame - 1 if index + 1 < len(spans) else max(start_frame + 1, requested)
        boundary = max(start_frame + 1, min(upper, requested))
        set_row_span(rows[index], start_frame, boundary, fps)
        if index + 1 < len(rows):
            set_row_span(rows[index + 1], boundary, next_end_frame, fps)
    else:
        raise RuntimeError("Boundary edge must be start or end.")
    write_manifest_details(manifest, source_video, fieldnames, rows)
    seconds = boundary / fps
    state.APP.log.append(f"Updated shot {index + 1} {edge} boundary to frame {boundary} ({format_timecode(seconds)})")
    return {"manifest": rel(manifest), "time": format_timecode(seconds), "frame": str(boundary)}

def update_shot_fade(manifest_text: str, index: int, enabled: bool, crossfade_seconds: str) -> dict[str, str]:
    manifest = resolve(manifest_text)
    source_video, fieldnames, rows = read_manifest_details(manifest)
    if index < 0 or index >= len(rows) - 1:
        raise IndexError(f"Shot {index + 1} does not have a following transition.")
    for key in ("fade_to_next", "crossfade_seconds"):
        if key not in fieldnames:
            fieldnames.append(key)
    try:
        seconds = max(0.0, float(crossfade_seconds or 0.0))
    except ValueError:
        seconds = 0.0
    rows[index]["fade_to_next"] = "true" if enabled and seconds > 0 else "false"
    rows[index]["crossfade_seconds"] = f"{seconds:.3f}".rstrip("0").rstrip(".") if seconds else ""
    write_manifest_details(manifest, source_video, fieldnames, rows)
    fade_state = "enabled" if rows[index]["fade_to_next"] == "true" else "disabled"
    state.APP.log.append(f"Fade transition after shot {index + 1} {fade_state}; crossfade {rows[index].get('crossfade_seconds') or '0'}s")
    return {"manifest": rel(manifest), "fade_to_next": rows[index]["fade_to_next"], "crossfade_seconds": rows[index]["crossfade_seconds"]}
