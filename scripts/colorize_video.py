from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import artifact_ids as aid
import openai_generate_reference as openai_images
from reference_sets import reference_items
from comfy_api import ensure_node_types, extract_output_files, queue_prompt, wait_for_comfy, wait_for_prompt
from common import (
    ROOT,
    copy_to_comfy_input,
    file_fingerprint,
    find_ffmpeg,
    load_local_config,
    newest_output as newest_comfy_output,
    replace_with_retry,
    resolve_path,
    root_relative,
    safe_stem,
    resumable_output,
    video_info,
    write_signature,
)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
REFERENCE_INPUT_COPY_STRATEGY = "content-keyed-v1"


def read_manifest(path: Path) -> tuple[str | None, list[dict[str, str]]]:
    source_video = None
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        while True:
            pos = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.startswith("#"):
                if line.startswith("# source_video="):
                    source_video = line.split("=", 1)[1].strip()
                continue
            handle.seek(pos)
            for row in csv.DictReader(handle):
                if row.get("enabled", "true").strip().lower() not in {"false", "0", "no", "off"}:
                    rows.append(row)
            break
    return source_video, rows


def parse_time(text: str) -> float:
    text = text.strip()
    if not text:
        return 0.0
    parts = text.split(":")
    if len(parts) == 1:
        return float(parts[0])
    seconds = float(parts[-1])
    minutes = int(parts[-2])
    hours = int(parts[-3]) if len(parts) > 2 else 0
    return hours * 3600 + minutes * 60 + seconds


def method_suffix(method: str) -> str:
    if method in {"colormnet", "cmnet2", "openai"}:
        return method
    return "deepexemplar"


def processing_dimensions(width: int, height: int, processing_height: str) -> tuple[int, int]:
    text = str(processing_height or "source").strip().lower()
    if text in {"", "source", "original", "0"}:
        return width, height
    try:
        target_h = int(float(text.rstrip("p")))
    except ValueError:
        return width, height
    if target_h <= 0 or height <= target_h:
        return width, height
    scale = target_h / height
    target_w = max(2, int(round(width * scale / 2) * 2))
    target_h = max(2, int(round(target_h / 2) * 2))
    return target_w, target_h


def downscaled_video_signature(source: Path, width: int, height: int) -> dict[str, Any]:
    return {
        "version": 2,
        "tool": "colorize_video.py",
        "kind": "grayscale processing input",
        "source": root_relative(source),
        "source_fingerprint": file_fingerprint(source),
        "width": width,
        "height": height,
        "grayscale": True,
    }


def prepare_processing_video(ffmpeg: str, source: Path, width: int, height: int, original_width: int, original_height: int) -> Path:
    """Create the shared, always-grayscale video input used by both colourisation backends.

    Dearchive can introduce colour, so this intermediate is produced even when the requested
    processing dimensions match the source. Shot ranges are selected from this file in ComfyUI.
    """
    digest = file_fingerprint(source)["sha256"][:12]
    output = ROOT / ".cache" / "colorize_inputs" / f"{safe_stem(source.name)}_{digest}_{width}x{height}_gray.mp4"
    sig = downscaled_video_signature(source, width, height)
    if resumable_output(output, sig, width=width, height=height):
        print(f"Reuse grayscale colourisation input: {output}", flush=True)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vf",
        f"scale={width}:{height}:flags=lanczos,hue=s=0,setsar=1",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        str(partial),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    replace_with_retry(partial, output)
    write_signature(output, sig)
    print(f"Wrote grayscale colourisation input: {output}", flush=True)
    return output


def prepare_processing_reference(reference: Path, width: int, height: int) -> Path:
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return reference

    try:
        with Image.open(reference) as image:
            if image.width == width and image.height == height:
                return reference
            digest = file_fingerprint(reference)["sha256"][:12]
            output = ROOT / ".cache" / "colorize_refs" / f"{safe_stem(reference.stem)}_{digest}_{width}x{height}.png"
            sig = {
                "version": 1,
                "tool": "colorize_video.py",
                "kind": "processing reference",
                "source": root_relative(reference),
                "source_fingerprint": file_fingerprint(reference),
                "width": width,
                "height": height,
            }
            if resumable_output(output, sig, width=width, height=height):
                return output
            output.parent.mkdir(parents=True, exist_ok=True)
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            canvas = Image.new("RGB", (width, height), (0, 0, 0))
            fitted = ImageOps.contain(image.convert("RGB"), (width, height), resampling)
            canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
            canvas.save(output, format="PNG")
            write_signature(output, sig)
            return output
    except Exception:
        return reference


def copy_reference_to_comfy_input(reference: Path, comfy_dir: Path, width: int | None = None, height: int | None = None) -> str:
    if width and height:
        reference = prepare_processing_reference(reference, width, height)
    target_dir = comfy_dir / "input" / "arp_colorize_refs"
    target_dir.mkdir(parents=True, exist_ok=True)
    digest = file_fingerprint(reference)["sha256"][:12]
    suffix = reference.suffix.lower() or ".png"
    target = target_dir / f"{safe_stem(reference.stem)}_{digest}{suffix}"
    if (
        not target.exists()
        or target.stat().st_size != reference.stat().st_size
        or file_fingerprint(target)["sha256"] != file_fingerprint(reference)["sha256"]
    ):
        shutil.copy2(reference, target)
    return str(Path("arp_colorize_refs") / target.name).replace("\\", "/")


def default_output(manifest: Path, manifest_source: str | None, method: str) -> Path:
    # Match the artifact name the GUI locates by (references.colorized_output_for_manifest), so the
    # method="both" path — the only path that reaches default_output, since single methods get an
    # explicit --output — writes files the UI can find. The old "<stem>_<method>_colorized.mp4" name
    # drifted from the artifact-id scheme and left "both" outputs invisible to the GUI.
    ident = aid.colorized_identity(manifest.stem, method)
    name_src = Path(manifest_source).name if manifest_source else manifest.name
    return ROOT / "intermediate" / "outpainted_colorized" / aid.artifact_name(aid.source_word(name_src), "color", ident, "mp4")


def reference_signature(row: dict[str, str]) -> dict[str, Any]:
    refs = row_references(row)
    return {
        "start": row.get("start", ""),
        "end": row.get("end", ""),
        "start_frame": row.get("start_frame", ""),
        "end_frame": row.get("end_frame", ""),
        "references": [
            {"selected_frame": item["selected_frame"], "reference": root_relative(item["path"]), "reference_fingerprint": file_fingerprint(item["path"])}
            for item in refs
        ],
        "fade_to_next": row.get("fade_to_next", ""),
        "crossfade_seconds": row.get("crossfade_seconds", ""),
    }


def row_path_signature(row: dict[str, str], key: str) -> dict[str, Any]:
    text = row.get(key, "")
    if not text:
        return {"path": ""}
    path = resolve_path(text)
    sig: dict[str, Any] = {"path": root_relative(path)}
    if path.exists():
        sig["fingerprint"] = file_fingerprint(path)
    else:
        sig["missing"] = True
    return sig


def shot_input_signature(row: dict[str, str]) -> dict[str, Any]:
    """Everything about a manifest row that can make a cached shot segment stale."""
    refs = row_references(row)
    unsigned_row_keys = {"color_reference_previous"}
    return {
        "row": {key: row.get(key, "") for key in sorted(row) if key not in unsigned_row_keys},
        "source_reference": row_path_signature(row, "source_reference"),
        "color_reference": row_path_signature(row, "color_reference"),
        "references": [
            {"selected_frame": item["selected_frame"], "reference": root_relative(item["path"]), "reference_fingerprint": file_fingerprint(item["path"])}
            for item in refs
        ],
    }


def method_settings_signature(args: argparse.Namespace) -> dict[str, Any]:
    settings = {
        "method": args.method,
        "use_torch_compile": args.use_torch_compile,
        "video_format": args.video_format,
        "crf": args.crf,
        "processing_height": getattr(args, "processing_height", "source"),
    }
    if args.method == "openai":
        settings.update(
            {
                "openai_model": args.openai_model,
                "openai_previous_frames": args.openai_previous_frames,
                "openai_prompt": args.openai_prompt,
                "openai_size": args.openai_size,
                "openai_quality": args.openai_quality,
            }
        )
    elif args.method == "colormnet":
        settings.update(
            {
                "colormnet_memory_mode": args.colormnet_memory_mode,
                "colormnet_feature_encoder": args.colormnet_feature_encoder,
                "colormnet_text_guidance": args.colormnet_text_guidance,
                "colormnet_text_guidance_weight": args.colormnet_text_guidance_weight,
            }
        )
    elif args.method == "cmnet2":
        settings.update(
            {
                "cmnet2_top_k": args.cmnet2_top_k,
                "cmnet2_mem_every": args.cmnet2_mem_every,
                "cmnet2_runtime": root_relative(resolve_path(args.cmnet2_dir)),
                "cmnet2_model": root_relative(resolve_path(args.cmnet2_model)) if args.cmnet2_model else "auto",
            }
        )
    else:
        settings.update(
            {
                "frame_propagate": args.frame_propagate,
                "use_half_resolution": args.use_half_resolution,
                "use_sage_attention": args.use_sage_attention,
            }
        )
    return settings


def signature(args: argparse.Namespace, manifest: Path, source_video: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "version": 11,
        "tool": "colorize_video.py",
        "reference_input_copy": REFERENCE_INPUT_COPY_STRATEGY,
        "manifest": root_relative(manifest),
        "manifest_fingerprint": file_fingerprint(manifest),
        "source_video": root_relative(source_video),
        "source_fingerprint": file_fingerprint(source_video),
        "grayscale_video_input": args.method != "openai",
        "references": [reference_signature(row) for row in rows],
        "shot_inputs": [shot_input_signature(row) for row in rows],
        "settings": method_settings_signature(args),
    }


def row_reference(row: dict[str, str]) -> Path:
    ref = row.get("color_reference") or row.get("reference") or row.get("source_reference") or ""
    if not ref:
        raise RuntimeError("Manifest row has no color_reference/reference/source_reference.")
    path = resolve_path(ref)
    if not path.exists() and row.get("source_reference"):
        path = resolve_path(row["source_reference"])
    if not path.exists():
        raise FileNotFoundError(f"Reference image not found: {path}")
    return path


def row_references(row: dict[str, str]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for item in reference_items(row):
        effective = dict(row)
        effective.update(item)
        values.append({
            "path": row_reference(effective),
            "selected_frame": optional_int(item.get("selected_frame")),
        })
    return values


def build_prompt(
    video_name: str,
    ref_names: str | list[str],
    start_frame: int,
    frame_count: int,
    width: int,
    height: int,
    fps: float,
    args: argparse.Namespace,
    prefix: str,
) -> dict[str, Any]:
    if isinstance(ref_names, str):
        ref_names = [ref_names]
    if not ref_names:
        raise RuntimeError("At least one reference image is required.")
    prompt: dict[str, Any] = {
        "1": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": video_name,
                "force_rate": 0.0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": frame_count,
                "skip_first_frames": start_frame,
                "select_every_nth": 1,
                "format": "None",
            },
        },
        "4": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["3", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": prefix,
                "format": args.video_format,
                "pix_fmt": "yuv420p",
                "crf": args.crf,
                "save_metadata": True,
                "pingpong": False,
                "save_output": True,
            },
        },
    }
    if len(ref_names) != 1:
        raise RuntimeError("ColorMNet and Deep Exemplar accept exactly one reference image per shot.")
    prompt["2"] = {"class_type": "LoadImage", "inputs": {"image": ref_names[0]}}
    prompt["3"] = colorization_node(args, width, height, ["2", 0])
    return prompt


def colorization_node(args: argparse.Namespace, width: int, height: int, reference_input: list[Any] | None = None) -> dict[str, Any]:
    reference_input = reference_input or ["2", 0]
    if args.method == "colormnet":
        return {
            "class_type": "ColorMNetVideo",
            "inputs": {
                "video_frames": ["1", 0],
                "reference_image": reference_input,
                "target_width": width,
                "target_height": height,
                "memory_mode": args.colormnet_memory_mode,
                "feature_encoder": args.colormnet_feature_encoder,
                "use_fp16": True,
                "use_torch_compile": args.use_torch_compile,
                "text_guidance": args.colormnet_text_guidance,
                "text_guidance_weight": args.colormnet_text_guidance_weight,
            },
        }
    return {
        "class_type": "DeepExColorVideoNode",
        "inputs": {
            "video_frames": ["1", 0],
            "reference_image": reference_input,
            "frame_propagate": args.frame_propagate,
            "use_half_resolution": args.use_half_resolution,
            "target_width": width,
            "target_height": height,
            "use_torch_compile": args.use_torch_compile,
            "use_sage_attention": args.use_sage_attention,
        },
    }


def newest_output(files: list[Path]) -> Path:
    return newest_comfy_output(files, VIDEO_EXTS, "video output")


def segment_signature(
    args: argparse.Namespace,
    source_video: Path,
    row: dict[str, str],
    references: list[dict[str, Any]],
    start_frame: int,
    end_frame: int,
    base_start_frame: int,
    base_end_frame: int,
    width: int,
    height: int,
    fps: float,
) -> dict[str, Any]:
    if isinstance(references, Path):
        references = [{"path": references, "selected_frame": optional_int(row.get("selected_frame"))}]
    return {
        "version": 10,
        "tool": "colorize_video.py",
        "kind": f"{args.method} segment",
        "reference_input_copy": REFERENCE_INPUT_COPY_STRATEGY,
        "source_video": root_relative(source_video),
        "source_fingerprint": file_fingerprint(source_video),
        "grayscale_video_input": True,
        "references": [
            {"selected_frame": item["selected_frame"], "reference": root_relative(item["path"]), "reference_fingerprint": file_fingerprint(item["path"])}
            for item in references
        ],
        "shot_input": shot_input_signature(row),
        "row_start": row.get("start", ""),
        "row_end": row.get("end", ""),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "base_start_frame": base_start_frame,
        "base_end_frame": base_end_frame,
        "fade_to_next": row.get("fade_to_next", ""),
        "crossfade_seconds": row.get("crossfade_seconds", ""),
        "width": width,
        "height": height,
        "fps": fps,
        "settings": method_settings_signature(args),
    }


def segment_resumable(chunk: Path, chunk_sig: dict[str, Any], width: int, height: int, expected_frames: int) -> bool:
    if not resumable_output(chunk, chunk_sig, width=width, height=height):
        return False
    try:
        info = video_info(chunk)
    except Exception:
        return False
    return abs(int(info["frames"]) - expected_frames) <= 3


def normalize_clip(ffmpeg: str, source: Path, output: Path, fps: float, expected_frames: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    vf = f"setpts=N/({fps:.8f}*TB),fps={fps:.8f},trim=end_frame={expected_frames},setpts=N/({fps:.8f}*TB),setsar=1"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vf",
        vf,
        "-an",
        "-r",
        f"{fps:.8f}",
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        str(partial),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    replace_with_retry(partial, output)


def truthy(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "on"}


def transition_seconds(row: dict[str, str]) -> float:
    if not truthy(row.get("fade_to_next", "")):
        return 0.0
    try:
        return max(0.0, float(row.get("crossfade_seconds", "") or 0.0))
    except ValueError:
        return 0.0

def optional_int(value: str | None) -> int | None:
    try:
        text = str(value or "").strip()
        if not text:
            return None
        return int(float(text))
    except ValueError:
        return None


def shot_plan(rows: list[dict[str, str]], total_frames: int, fps: float) -> tuple[list[dict[str, int]], list[int]]:
    base: list[tuple[int, int]] = []
    start_frame = 0
    for index, row in enumerate(rows):
        row_start = optional_int(row.get("start_frame"))
        row_end = optional_int(row.get("end_frame"))
        if row_start is not None:
            start_frame = max(0, min(total_frames - 1, row_start))
        if row_end is not None:
            end_frame = min(total_frames, max(start_frame + 1, row_end))
        else:
            end_frame = min(total_frames, max(start_frame + 1, round(parse_time(row.get("end", "")) * fps)))
        if index == len(rows) - 1:
            end_frame = total_frames
        base.append((start_frame, end_frame))
        start_frame = end_frame

    transitions = [0 for _ in rows]
    for index, row in enumerate(rows[:-1]):
        frames = int(round(transition_seconds(row) * fps))
        if frames <= 0:
            continue
        left = max(1, base[index][1] - base[index][0])
        right = max(1, base[index + 1][1] - base[index + 1][0])
        transitions[index] = max(1, min(frames, left, right))

    plan: list[dict[str, int]] = []
    for index, (base_start, base_end) in enumerate(base):
        prev_frames = transitions[index - 1] if index > 0 else 0
        next_frames = transitions[index] if index < len(transitions) else 0
        pre = prev_frames // 2
        post = next_frames - (next_frames // 2)
        actual_start = max(0, base_start - pre)
        actual_end = min(total_frames, base_end + post)
        plan.append(
            {
                "base_start": base_start,
                "base_end": base_end,
                "start": actual_start,
                "end": max(actual_start + 1, actual_end),
            }
        )
    return plan, transitions


def stitch(ffmpeg: str, chunks: list[Path], output: Path, fps: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    concat = output.with_suffix(".concat.txt")
    concat.write_text("".join("file '" + str(chunk).replace("'", "'\\''") + "'\n" for chunk in chunks), encoding="utf-8")
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat),
        "-vf",
        f"setpts=N/({fps:.8f}*TB),fps={fps:.8f},setsar=1",
        "-an",
        "-r",
        f"{fps:.8f}",
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        str(partial),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    replace_with_retry(partial, output)
    concat.unlink(missing_ok=True)


def transition_groups(transitions: list[int]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start = 0
    for index, frames in enumerate(transitions):
        if frames > 0:
            continue
        groups.append((start, index))
        start = index + 1
    if transitions:
        groups.append((start, len(transitions) - 1))
    elif not groups:
        groups.append((0, 0))
    return [(left, right) for left, right in groups if left <= right]


def xfade_group(ffmpeg: str, chunks: list[Path], transitions: list[int], output: Path, fps: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    cmd = [ffmpeg, "-y"]
    for chunk in chunks:
        cmd += ["-i", str(chunk)]

    filters: list[str] = []
    for index in range(len(chunks)):
        filters.append(f"[{index}:v]setpts=N/({fps:.8f}*TB),fps=fps={fps:.8f}[v{index}]")

    previous = "v0"
    accumulated = video_info(chunks[0])["frames"] / fps
    for index in range(1, len(chunks)):
        duration = max(1 / fps, transitions[index - 1] / fps)
        offset = max(0.0, accumulated - duration)
        current = f"x{index}"
        filters.append(f"[{previous}][v{index}]xfade=transition=fade:duration={duration:.8f}:offset={offset:.8f},setpts=PTS-STARTPTS[{current}]")
        previous = current
        accumulated += video_info(chunks[index])["frames"] / fps - duration
    filters.append(f"[{previous}]format=yuv420p[vout]")

    cmd += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[vout]",
        "-an",
        "-r",
        f"{fps:.8f}",
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        str(partial),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    replace_with_retry(partial, output)


def stitch_colorized(ffmpeg: str, chunks: list[Path], transitions: list[int], output: Path, fps: float, source_stem: str) -> None:
    if not any(transitions):
        stitch(ffmpeg, chunks, output, fps)
        return

    group_outputs: list[Path] = []
    group_dir = ROOT / ".cache" / "colorized_chunks" / "crossfaded" / source_stem
    group_dir.mkdir(parents=True, exist_ok=True)
    for group_index, (left, right) in enumerate(transition_groups(transitions)):
        group_chunks = chunks[left : right + 1]
        if len(group_chunks) == 1:
            group_outputs.append(group_chunks[0])
            continue
        group_transitions = transitions[left:right]
        group_output = group_dir / f"group_{group_index:04d}_{left:04d}_{right:04d}.mp4"
        xfade_group(ffmpeg, group_chunks, group_transitions, group_output, fps)
        group_outputs.append(group_output)
    stitch(ffmpeg, group_outputs, output, fps)


def prepare_openai_source_frames(
    ffmpeg: str,
    source: Path,
    width: int,
    height: int,
    total_frames: int,
) -> Path:
    """Extract colour-preserving source frames for cloud image editing.

    The directory is content-addressed, so it remains safe to reuse across interrupted jobs and
    never shares frames with the compulsory grayscale inputs used by the local colourizers.
    """
    digest = file_fingerprint(source)["sha256"][:16]
    frame_dir = ROOT / ".cache" / "openai_colorize" / f"{safe_stem(source.name)}_{digest}_{width}x{height}" / "source"
    expected = [frame_dir / f"frame_{index:08d}.png" for index in range(total_frames)]
    if expected and all(path.is_file() for path in expected):
        print(f"Reuse {total_frames} colour source frames for OpenAI Cloud: {frame_dir}", flush=True)
        return frame_dir
    frame_dir.mkdir(parents=True, exist_ok=True)
    pattern = frame_dir / "frame_%08d.png"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vf",
        f"scale={width}:{height}:flags=lanczos,setsar=1",
        "-an",
        "-vsync",
        "0",
        "-start_number",
        "0",
        str(pattern),
    ]
    print(f"Extracting {total_frames} colour-preserving frames for OpenAI Cloud...", flush=True)
    subprocess.run(cmd, check=True)
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"FFmpeg extracted an incomplete frame sequence; first missing frame: {missing[0]}")
    print(f"Extracted {total_frames} source frames: {frame_dir}", flush=True)
    return frame_dir


def openai_frame_prompt(previous_count: int, reference_count: int) -> tuple[str, str]:
    parts = ["The first image is the current source frame and is the only image to edit."]
    if previous_count:
        if previous_count > 1:
            parts.append(
                f"Images 2-{previous_count + 1} are preceding generated frames in chronological order, oldest first; "
                "use them only for temporal continuity of colours, texture, grain, lighting and identity."
            )
        else:
            parts.append(
                "Image 2 is the immediately preceding generated frame; use it only for temporal continuity "
                "of colours, texture, grain, lighting and identity."
            )
    reference_start = previous_count + 2
    reference_end = previous_count + reference_count + 1
    if reference_count:
        if reference_count > 1:
            parts.append(
                f"Images {reference_start}-{reference_end} are the approved colour references for this shot; "
                "use all of them to infer stable object, clothing, skin, architecture and landscape colours throughout the pan."
            )
        else:
            parts.append(
                f"Image {reference_start} is the approved colour reference for this shot; use it to infer "
                "stable object, clothing, skin, architecture and landscape colours throughout the pan."
            )
    parts.append(
        "Never copy composition, people, poses or geometry from a preceding frame or reference into the current frame."
    )
    instructions = " ".join(parts)
    return "", instructions


def openai_frame_signature(
    args: argparse.Namespace,
    source_frame: Path,
    previous_frames: list[Path],
    references: list[Path],
    shot_index: int,
    frame_index: int,
) -> dict[str, Any]:
    return {
        "version": 1,
        "tool": "colorize_video.py",
        "kind": "OpenAI Cloud colourized frame",
        "frame": frame_index,
        "shot": shot_index,
        "source": file_fingerprint(source_frame),
        "previous_generated_frames": [file_fingerprint(path) for path in previous_frames],
        "references": [
            {"path": root_relative(path), "fingerprint": file_fingerprint(path)} for path in references
        ],
        "model": args.openai_model,
        "prompt": args.openai_prompt,
        "size": args.openai_size,
        "quality": args.openai_quality,
    }


def assemble_openai_frames(ffmpeg: str, frame_dir: Path, output: Path, fps: float, total_frames: int, crf: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        f"{fps:.12g}",
        "-start_number",
        "0",
        "-i",
        str(frame_dir / "frame_%08d.png"),
        "-frames:v",
        str(total_frames),
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        str(partial),
    ]
    print("Assembling OpenAI Cloud frames into the colorized video...", flush=True)
    subprocess.run(cmd, check=True)
    replace_with_retry(partial, output)


def run_openai_colorization(
    args: argparse.Namespace,
    manifest: Path,
    source_video: Path,
    rows: list[dict[str, str]],
    output: Path,
    output_sig: dict[str, Any],
    ffmpeg: str,
    width: int,
    height: int,
    fps: float,
    total_frames: int,
) -> int:
    token = str(args.api_key or "").strip()
    if not token:
        raise RuntimeError("Missing OpenAI API key. Add it in ARP Settings before running OpenAI Cloud colorization.")
    previous_limit = max(0, min(12, int(args.openai_previous_frames)))
    plan, _transitions = shot_plan(rows, total_frames, fps)
    source_dir = prepare_openai_source_frames(ffmpeg, source_video, width, height, total_frames)
    # Keep the cache location stable when a project save rewrites identical files with new mtimes.
    # Per-frame signatures below still invalidate precisely when source, prompt, model, history or
    # reference content changes.
    cache_identity = {
        "source_sha256": output_sig["source_fingerprint"]["sha256"],
        "manifest": root_relative(manifest),
        "width": width,
        "height": height,
        "model": args.openai_model,
    }
    digest = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    generated_dir = ROOT / ".cache" / "openai_colorize" / f"{safe_stem(source_video.name)}_{digest}_{width}x{height}" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    row_references_cache = [[item["path"] for item in row_references(row)] for row in rows]
    print(
        f"OpenAI Cloud colorization: {total_frames} frames, model {args.openai_model}, "
        f"up to {previous_limit} preceding frame(s), all shot references.",
        flush=True,
    )
    print("Each completed frame is cached; this job can be stopped and resumed.", flush=True)
    history: list[Path] = []
    active_shot = -1
    reused = 0
    for frame_index in range(total_frames):
        shot_index = next(
            (index for index, item in enumerate(plan) if item["base_start"] <= frame_index < item["base_end"]),
            len(rows) - 1,
        )
        if shot_index != active_shot:
            history = []
            active_shot = shot_index
        previous = history[-previous_limit:] if previous_limit else []
        references = row_references_cache[shot_index]
        source_frame = source_dir / f"frame_{frame_index:08d}.png"
        generated = generated_dir / f"frame_{frame_index:08d}.png"
        frame_sig = openai_frame_signature(args, source_frame, previous, references, shot_index, frame_index)
        if not args.force and resumable_output(generated, frame_sig, width=width, height=height):
            reused += 1
        else:
            role_prompt, reference_instructions = openai_frame_prompt(len(previous), len(references))
            child = argparse.Namespace(
                source_image=source_frame,
                output=generated,
                api_key=token,
                model=args.openai_model,
                prompt=args.openai_prompt,
                prompt_suffix="",
                add_prompt=role_prompt,
                reference_instructions=reference_instructions,
                size=args.openai_size,
                quality=args.openai_quality,
                timeout=args.openai_timeout,
                max_retries=args.openai_max_retries,
                force=True,
                no_normalize_to_source_size=False,
                dry_run=False,
                quiet=True,
            )
            # Input order is significant and is described explicitly in the prompt.
            openai_images.generate(child, [*previous, *references])
            write_signature(generated, frame_sig)
        history.append(generated)
        print(
            f"OpenAI Cloud frame {frame_index + 1}/{total_frames}; "
            f"shot {shot_index + 1}/{len(rows)}; cached {reused}",
            flush=True,
        )
    assemble_openai_frames(ffmpeg, generated_dir, output, fps, total_frames, args.crf)
    write_signature(output, output_sig)
    print(f"Wrote OpenAI Cloud colorized video: {output}", flush=True)
    return 0


def run_cmnet2_colorization(
    args: argparse.Namespace,
    source_video: Path,
    rows: list[dict[str, str]],
    output: Path,
    output_sig: dict[str, Any],
    ffmpeg: str,
    width: int,
    height: int,
    fps: float,
    total_frames: int,
) -> int:
    """Colorize each true shot with every one of its timed CMNET2 reference anchors."""
    from cmnet2_runtime import CMNet2Session, resolve_model

    runtime_dir = resolve_path(args.cmnet2_dir)
    model_path = resolve_model(runtime_dir, args.cmnet2_model, args.comfy_dir)
    plan, transitions = shot_plan(rows, total_frames, fps)
    cache_dir = ROOT / ".cache" / "colorized_chunks" / "cmnet2" / safe_stem(source_video.name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    session: CMNet2Session | None = None
    chunks: list[Path] = []

    for index, row in enumerate(rows):
        item = plan[index]
        start_frame, end_frame = item["start"], item["end"]
        frame_count = max(1, end_frame - start_frame)
        references = row_references(row)
        prepared_references = [
            {
                **reference,
                "path": prepare_processing_reference(reference["path"], width, height),
            }
            for reference in references
        ]
        chunk = cache_dir / f"segment_{index:04d}_{start_frame:06d}_{end_frame:06d}.mp4"
        chunk_sig = segment_signature(
            args,
            source_video,
            row,
            references,
            start_frame,
            end_frame,
            item["base_start"],
            item["base_end"],
            width,
            height,
            fps,
        )
        if not args.force and segment_resumable(chunk, chunk_sig, width, height, frame_count):
            print(f"Reuse CMNET2 segment {index + 1}/{len(rows)}: {chunk}", flush=True)
            chunks.append(chunk)
            continue
        if session is None:
            session = CMNet2Session(
                runtime_dir,
                model_path,
                top_k=args.cmnet2_top_k,
                mem_every=args.cmnet2_mem_every,
                max_frames=total_frames,
            )
        frame_labels = ", ".join(
            str(reference["selected_frame"]) if reference["selected_frame"] is not None else "unspecified"
            for reference in references
        )
        print(
            f"Colorize segment {index + 1}/{len(rows)} with CMNET2: frames {start_frame}-{end_frame}; "
            f"{len(references)} permanent reference(s) at source frames {frame_labels}",
            flush=True,
        )
        session.colorize_segment(
            source_video,
            chunk,
            prepared_references,
            start_frame=start_frame,
            end_frame=end_frame,
            width=width,
            height=height,
            fps=fps,
            ffmpeg=ffmpeg,
            crf=args.crf,
        )
        write_signature(chunk, chunk_sig)
        chunks.append(chunk)

    stitch_colorized(ffmpeg, chunks, transitions, output, fps, safe_stem(source_video.name))
    write_signature(output, output_sig)
    print(f"Wrote CMNET2 colorized video: {output}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    config = load_local_config()
    parser = argparse.ArgumentParser(description="Colorize outpainted video shots with reference-guided ComfyUI colorization.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-video", help="Override the # source_video from the manifest.")
    parser.add_argument("--output")
    parser.add_argument("--method", choices=["deepexemplar", "colormnet", "cmnet2", "openai", "both"], default="deepexemplar")
    parser.add_argument("--comfy-url", default=config.get("comfy_url", "http://127.0.0.1:8188"))
    parser.add_argument("--comfy-dir", default=config.get("comfy_dir", str(ROOT / "tools" / "comfyui")))
    parser.add_argument("--comfy-output-root", default="")
    parser.add_argument("--processing-height", default="source", help="Downscale input video before ComfyUI processing. Use source/original or a target height such as 1080.")
    parser.add_argument("--frame-propagate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-half-resolution", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-sage-attention", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--colormnet-memory-mode", choices=["balanced", "low_memory", "high_quality"], default="balanced")
    parser.add_argument("--colormnet-feature-encoder", choices=["resnet50", "vgg19", "dinov2_vits", "dinov2_vitb", "dinov2_vitl", "clip_vitb"], default="resnet50")
    parser.add_argument("--colormnet-text-guidance", default="")
    parser.add_argument("--colormnet-text-guidance-weight", type=float, default=0.3)
    parser.add_argument("--cmnet2-dir", default=str(ROOT / "vendor" / "cmnet2"))
    parser.add_argument("--cmnet2-model", default="")
    parser.add_argument("--cmnet2-top-k", type=int, default=30)
    parser.add_argument("--cmnet2-mem-every", type=int, default=5)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--openai-model", default=openai_images.DEFAULT_MODEL)
    parser.add_argument("--openai-previous-frames", type=int, default=3)
    parser.add_argument(
        "--openai-prompt",
        default=(
            "Enhance and colourise the first image while preserving its exact composition, camera framing, "
            "geometry, identities, faces, hands, poses, objects, fine texture, film grain and luminance. "
            "Use its existing colour as evidence rather than removing it. Make the colour vivid, natural, "
            "period-appropriate and consistent with the supplied shot references. Do not add, remove, move, "
            "redesign or sharpen objects. Output one edited version of the first image only."
        ),
    )
    parser.add_argument("--openai-size", default="auto")
    parser.add_argument("--openai-quality", default="auto")
    parser.add_argument("--openai-timeout", type=float, default=900.0)
    parser.add_argument("--openai-max-retries", type=int, default=5)
    parser.add_argument("--video-format", default="video/h264-mp4")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.method == "both":
        for method in ("deepexemplar", "colormnet"):
            child = argparse.Namespace(**vars(args))
            child.method = method
            child.output = ""
            run(child)
        return 0
    return run(args)


def run(args: argparse.Namespace) -> int:
    manifest = resolve_path(args.manifest)
    source_from_manifest, rows = read_manifest(manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"No enabled rows in manifest: {manifest}")
    source_video = resolve_path(args.source_video or source_from_manifest or "")
    if not source_video.exists():
        raise FileNotFoundError(f"Source video not found for colourisation: {source_video}")
    output = resolve_path(args.output) if args.output else default_output(manifest, source_from_manifest, args.method)
    source_info = video_info(source_video)
    source_width, source_height = int(source_info["width"]), int(source_info["height"])
    target_width, target_height = processing_dimensions(source_width, source_height, args.processing_height)
    sig = signature(args, manifest, source_video, rows)
    if not args.force and resumable_output(output, sig, width=target_width, height=target_height):
        print(f"Reuse colorized video: {output}", flush=True)
        return 0
    if args.dry_run:
        if args.method == "openai":
            counts = [len(row_references(row)) for row in rows]
            print(
                f"Would send {int(source_info['frames'])} frame edit request(s) to {args.openai_model}: "
                f"{source_video} -> {output}; references per shot: {counts}; "
                f"up to {max(0, int(args.openai_previous_frames))} preceding frame(s).",
                flush=True,
            )
        elif args.method == "cmnet2":
            counts = [len(row_references(row)) for row in rows]
            print(
                f"Would colorize {len(rows)} shot segment(s) with CMNET2: {source_video} -> {output}; "
                f"permanent references per shot: {counts}.",
                flush=True,
            )
        else:
            print(f"Would colorize {len(rows)} shot segment(s): {source_video} -> {output}", flush=True)
        return 0

    ffmpeg = find_ffmpeg(args.ffmpeg)
    if args.method == "openai":
        return run_openai_colorization(
            args,
            manifest,
            source_video,
            rows,
            output,
            sig,
            ffmpeg,
            target_width,
            target_height,
            float(source_info["fps"]),
            int(source_info["frames"]),
        )

    processing_video = prepare_processing_video(ffmpeg, source_video, target_width, target_height, source_width, source_height)
    info = video_info(processing_video)
    width, height, fps, total_frames = int(info["width"]), int(info["height"]), float(info["fps"]), int(info["frames"])
    if args.method == "cmnet2":
        return run_cmnet2_colorization(
            args,
            processing_video,
            rows,
            output,
            sig,
            ffmpeg,
            width,
            height,
            fps,
            total_frames,
        )

    comfy_dir = resolve_path(args.comfy_dir)
    comfy_output_root = resolve_path(args.comfy_output_root) if args.comfy_output_root else comfy_dir / "output"
    video_name = copy_to_comfy_input(processing_video, comfy_dir, "arp_colorize")
    wait_for_comfy(args.comfy_url, timeout_seconds=180, poll_seconds=args.poll_seconds)
    required_nodes = {
        "VHS_LoadVideo": "ComfyUI-VideoHelperSuite",
        "VHS_VideoCombine": "ComfyUI-VideoHelperSuite",
    }
    if args.method == "colormnet":
        required_nodes["ColorMNetVideo"] = "ComfyUI-Reference-Based-Video-Colorization"
    else:
        required_nodes["DeepExColorVideoNode"] = "ComfyUI-Reference-Based-Video-Colorization"
    ensure_node_types(args.comfy_url, required_nodes, f"{args.method} colourisation")

    chunks: list[Path] = []
    plan, transitions = shot_plan(rows, total_frames, fps)
    cache_dir = ROOT / ".cache" / "colorized_chunks" / method_suffix(args.method) / safe_stem(source_video.name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        item = plan[index]
        start_frame = item["start"]
        end_frame = item["end"]
        frame_count = max(1, end_frame - start_frame)
        references = row_references(row)
        active_references = references[:1]
        if len(references) > 1:
            display_method = "ColorMNet" if args.method == "colormnet" else "Deep Exemplar"
            print(
                f"{display_method} supports one reference per shot; using reference 1 of {len(references)} "
                f"for shot {index + 1}. Choose CMNET2 or OpenAI Cloud to use every reference.",
                flush=True,
            )
        source_reference_indices = [
            max(start_frame, min(end_frame - 1, int(reference["selected_frame"] or start_frame)))
            for reference in active_references
        ]
        ref_names = [copy_reference_to_comfy_input(item["path"], comfy_dir, width, height) for item in active_references]
        chunk = cache_dir / f"segment_{index:04d}_{start_frame:06d}_{end_frame:06d}.mp4"
        chunk_sig = segment_signature(args, source_video, row, active_references, start_frame, end_frame, item["base_start"], item["base_end"], width, height, fps)
        if not args.force and segment_resumable(chunk, chunk_sig, width, height, frame_count):
            print(f"Reuse colorized segment {index + 1}/{len(rows)}: {chunk}", flush=True)
            chunks.append(chunk)
            continue
        prefix = f"arp_colorize/{method_suffix(args.method)}_{safe_stem(source_video.name)}_segment_{index:04d}_{start_frame:06d}_{end_frame:06d}"
        keyframe_note = ", ".join(str(value) for value in source_reference_indices)
        print(f"Colorize segment {index + 1}/{len(rows)} with {args.method}: frames {start_frame}-{end_frame} using {len(active_references)} reference(s) at source frames {keyframe_note}", flush=True)
        prompt = build_prompt(video_name, ref_names, start_frame, frame_count, width, height, fps, args, prefix)
        prompt_id = queue_prompt(args.comfy_url, prompt)
        history = wait_for_prompt(args.comfy_url, prompt_id, args.poll_seconds)
        produced = newest_output(extract_output_files(history, comfy_output_root))
        normalize_clip(ffmpeg, produced, chunk, fps, frame_count)
        write_signature(chunk, chunk_sig)
        chunks.append(chunk)

    stitch_colorized(ffmpeg, chunks, transitions, output, fps, safe_stem(source_video.name))
    write_signature(output, sig)
    print(f"Wrote colorized video: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
