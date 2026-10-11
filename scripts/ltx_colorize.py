"""Colourise shots with Lightricks' LTX-2.3 Colorization IC-LoRA.

Deep Exemplar and ColorMNet work in Lab and keep the film's lightness, so a near-white
garment that the reference paints red can only ever come out pale pink. This backend
generates the whole picture instead: the grayscale shot is the IC-LoRA's full-video
reference, the approved colour reference goes in as a guide frame at its own frame, and
the prompt follows the LoRA's trained "Reference shows ... COLORIZE ..." convention with
a Qwen3-VL description of the shot and of the reference's colours.

It runs Clean Up's Dearchive graph (the same single-stage distilled IC-LoRA graph) on the
Q4_K_M GGUF model, so it fits a 24 GB card. Shots longer than one window are split into
121-frame chunks (the LoRA's training length) that overlap. The chunk holding the
reference goes first; each chunk after it is guided by its neighbour's colourised frames
in the overlap, and the overlaps are crossfaded.

LTX's colour fades within a few dozen frames of a guide (tested on Steamboat Willie's title
card: vivid at the reference frame, washed out 60 frames away), so every 32 frames there is
also a colour-only guide: the reference carried onto that frame by optical flow, reduced to
coarse colour and brightness over the real grey frame's detail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from comfy_api import ensure_node_types, extract_output_files, free_comfy_models, queue_prompt, wait_for_comfy, wait_for_prompt
from common import ROOT, copy_to_comfy_input, file_fingerprint, newest_output, replace_with_retry, resolve_path, resumable_output, root_relative, safe_stem, write_signature
from dependency_manager import LTX_COLORIZE_LORA, QWEN_IMAGE_21_TEXT_ENCODER, ensure_ltx_colorize_models
from intermediate_video import codec_args, container_args, vhs_inputs_for_profile

VERSION = 2
MODEL_SIZE_MULTIPLE = 32
CHUNK_FRAMES = 121  # the LoRA's 960x544x121 training bucket
# Longer than the training bucket by default, so most shots are one chunk: a cut-free shot then
# gets one consistent pass instead of several joined at crossfades. Shots longer than this split.
DEFAULT_CHUNK_SECONDS = 15.0
DEFAULT_HEIGHT = 544
DEFAULT_OVERLAP = 17
DESCRIBE_NODE_ID = "52"
WARP_GUIDE_VERSION = 3
# The scene fills "Reference shows {scene}, rendered in grayscale", so it is asked for as a phrase.
SCENE_REQUEST = (
    "Describe this film frame as one plain noun phrase of up to 40 words for a colourisation model, starting with "
    "'a', 'an' or a character's name, for example 'Mickey Mouse turning a ship's wheel in a wheelhouse, with a "
    "river bank behind him'. Say who and what is in it, what they are doing, and the background. Do not mention "
    "colour, and do not say that it is black and white."
)
COLOUR_REQUEST = (
    "Describe the colours in this frame in two or three plain sentences for a colourisation model. Name the colour "
    "of each visible character's body, face, clothing and shoes, of each visible object, and of the background, "
    "for example 'Mickey has red shorts and yellow shoes'. Mention only things you can see and their colours."
)
SCENE_PREFIXES = ("this is ", "this image shows ", "this frame shows ", "the image shows ", "the frame shows ", "this shows ")
REQUIRED_NODES = {
    "LTXVImgToVideoConditionOnly": "ComfyUI-LTXVideo",
    "LTXAddVideoICLoRAGuide": "ComfyUI-LTXVideo",
    "LTXVAddGuideAdvanced": "ComfyUI-LTXVideo",
    "LTXVPreprocess": "ComfyUI-LTXVideo",
    "UnetLoaderGGUF": "ComfyUI-GGUF",
    "VHS_LoadVideo": "ComfyUI-VideoHelperSuite",
    "VHS_VideoCombine": "ComfyUI-VideoHelperSuite",
}
DESCRIBE_NODES = {"TextGenerate": "ComfyUI core (0.37 or newer)"}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("LTX Colorizer")
    group.add_argument("--ltx-height", type=int, default=DEFAULT_HEIGHT, help="Generation height; the LoRA was trained at 544 (960x544).")
    group.add_argument("--ltx-chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS, help="Longest LTX chunk; a shot up to this long is coloured in one pass, longer shots are split. Rounded to 8n+1 frames.")
    group.add_argument("--ltx-overlap-frames", type=int, default=DEFAULT_OVERLAP, help="Minimum frames shared by neighbouring chunks of a long shot.")
    group.add_argument("--ltx-lora-strength", type=float, default=1.0)
    group.add_argument("--ltx-reference-strength", type=float, default=1.0, help="Guide strength of the colour reference at its own frame.")
    group.add_argument("--ltx-continuity-strength", type=float, default=0.9, help="Guide strength of a neighbouring chunk's colourised overlap frames.")
    group.add_argument("--ltx-warp-interval", type=int, default=32, help="Frames between optical-flow-warped reference guides; 0 turns them off.")
    group.add_argument("--ltx-warp-strength", type=float, default=0.6, help="Guide strength of a warped reference (colour and broad brightness only; the shape is always the grey frame's).")
    group.add_argument("--ltx-describe", action=argparse.BooleanOptionalAction, default=True, help="Describe each shot and its colour reference with Qwen3-VL for the prompt.")
    group.add_argument("--ltx-seed", type=int, default=42)


def settings_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "ltx_version": VERSION,
        "ltx_height": args.ltx_height,
        "ltx_overlap_frames": args.ltx_overlap_frames,
        "ltx_chunk_seconds": args.ltx_chunk_seconds,
        "ltx_lora_strength": args.ltx_lora_strength,
        "ltx_reference_strength": args.ltx_reference_strength,
        "ltx_continuity_strength": args.ltx_continuity_strength,
        "ltx_warp_interval": args.ltx_warp_interval,
        "ltx_warp_strength": args.ltx_warp_strength,
        "ltx_describe": args.ltx_describe,
        "ltx_seed": args.ltx_seed,
    }


def ltx_dimensions(width: int, height: int, target_height: int = DEFAULT_HEIGHT) -> tuple[int, int]:
    """The generation canvas: target height and the source's aspect, both on LTX's 32-pixel grid."""
    out_h = max(MODEL_SIZE_MULTIPLE, round(target_height / MODEL_SIZE_MULTIPLE) * MODEL_SIZE_MULTIPLE)
    out_w = max(MODEL_SIZE_MULTIPLE, round(width * out_h / height / MODEL_SIZE_MULTIPLE) * MODEL_SIZE_MULTIPLE)
    return out_w, out_h


def chunk_frames_for(seconds: float, fps: float) -> int:
    """The longest chunk in frames for a duration: LTX needs 8n + 1, at least the 9-frame minimum."""
    return max(1, round((max(0.5, float(seconds)) * fps - 1) / 8)) * 8 + 1


def shot_chunks(start: int, end: int, chunk_frames: int = CHUNK_FRAMES, overlap: int = DEFAULT_OVERLAP) -> list[tuple[int, int]]:
    """Split [start, end) into windows of chunk_frames, spread evenly, each sharing >= overlap frames."""
    length = end - start
    if length <= chunk_frames:
        return [(start, end)]
    step = max(1, chunk_frames - overlap)
    count = max(2, math.ceil((length - overlap) / step))
    span = length - chunk_frames
    starts = [start + round(index * span / (count - 1)) for index in range(count)]
    return [(first, first + chunk_frames) for first in starts]


def processing_order(chunk_count: int, anchor: int) -> list[int]:
    """The anchor chunk, then outwards: forwards to the end, then backwards to the start."""
    return [anchor, *range(anchor + 1, chunk_count), *range(anchor - 1, -1, -1)]


def anchor_chunk(chunks: list[tuple[int, int]], frames: list[int]) -> int:
    for frame in frames:
        for index, (first, last) in enumerate(chunks):
            if first <= frame < last:
                return index
    return 0


def guide_grid(first_local: int, last_local: int) -> list[int]:
    """LTX guide positions (multiples of 8) inside [first_local, last_local]."""
    return [index for index in range(0, last_local + 1, 8) if index >= first_local]


def build_prompt(scene: str, colours: str) -> str:
    """The LoRA's trained prompt convention (see its model card)."""
    scene = scene.strip().rstrip(".")
    for prefix in SCENE_PREFIXES:
        if scene.lower().startswith(prefix):
            scene = scene[len(prefix):]
    if scene.split(" ", 1)[0] in {"A", "An", "The"}:
        scene = scene[0].lower() + scene[1:]
    scene = scene or "a scene from a black-and-white cartoon"
    colours = colours.strip().rstrip(".") or "the same scene with natural, period-appropriate colours"
    return (
        f"Reference shows {scene}, rendered in grayscale. Edited shows the same scene with natural colors restored. "
        f"COLORIZE {colours}. Subject identity, framing, and background geometry are identical to the reference; "
        "only color information differs between reference and edited."
    )


def describe_image(args: argparse.Namespace, image: Path, request: str) -> str:
    """Qwen3-VL (ComfyUI TextGenerate) description of one still, cached by content and request."""
    key = file_fingerprint(image)["sha256"][:16] + "_" + hashlib.sha256(request.encode("utf-8")).hexdigest()[:8]
    cache = ROOT / ".cache" / "ltx_colorize" / "descriptions" / f"{key}.json"
    if cache.is_file():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("request") == request:
            return str(cached.get("text", ""))
    name = copy_to_comfy_input(image, resolve_path(args.comfy_dir), "arp_ltx_colorize")
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": name}},
        "11": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN_IMAGE_21_TEXT_ENCODER, "type": "qwen_image", "device": "default"}},
        "50": {
            "class_type": "TextGenerate",
            "inputs": {
                "clip": ["11", 0], "prompt": request, "max_length": 256, "sampling_mode": "off",
                "image": ["1", 0], "thinking": False, "use_default_template": True, "mtp": "auto",
            },
        },
        DESCRIBE_NODE_ID: {"class_type": "PreviewAny", "inputs": {"source": ["50", 0]}},
    }
    prompt_id = queue_prompt(args.comfy_url, graph)
    history = wait_for_prompt(args.comfy_url, prompt_id, args.poll_seconds)
    entry = history.get(prompt_id, history)
    text = entry.get("outputs", {}).get(DESCRIBE_NODE_ID, {}).get("text", "")
    text = " ".join((text[0] if isinstance(text, list) and text else str(text or "")).split())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"request": request, "image": root_relative(image), "text": text}, indent=2), encoding="utf-8")
    return text


def shot_prompt(args: argparse.Namespace, row: dict[str, str], references: list[dict[str, Any]]) -> str:
    if not args.ltx_describe:
        return build_prompt("", "")
    scene_image = resolve_path(row["source_reference"]) if row.get("source_reference") else None
    scene = describe_image(args, scene_image, SCENE_REQUEST) if scene_image and scene_image.is_file() else ""
    colours = " ".join(describe_image(args, item["path"], COLOUR_REQUEST) for item in references[:2])
    return build_prompt(scene, colours)


def cleanup_graph_args(args: argparse.Namespace, prepared: Path) -> argparse.Namespace:
    """Clean Up's Dearchive graph settings, retargeted at the Colorization LoRA."""
    import cleanup_video

    graph_args = cleanup_video.build_parser().parse_args(["--source", str(prepared)])
    graph_args.cleanup_lora = graph_args.outpaint_lora = LTX_COLORIZE_LORA
    graph_args.lora_strength = float(args.ltx_lora_strength)
    graph_args.comfy_dir = str(resolve_path(args.comfy_dir))
    graph_args.comfy_url = args.comfy_url
    # The distilled recipe has no CFG; patch_workflow raises Dearchive's guider to 4.0
    # only when a source fidelity is set, so leave it unset.
    del graph_args.source_fidelity
    return graph_args


def run_chunk(
    args: argparse.Namespace,
    prepared: Path,
    output: Path,
    prompt_text: str,
    seed: int,
    start_guide: dict[str, Any] | None,
    extra_guides: list[dict[str, Any]],
    width: int,
    height: int,
    fps: float,
    frames: int,
) -> None:
    from outpaint_video import patch_workflow

    import cleanup_video

    comfy_dir = resolve_path(args.comfy_dir)
    graph_args = cleanup_graph_args(args, prepared)
    graph_args.guide_strength = float(start_guide["strength"]) if start_guide else 0.0
    workflow = json.loads(cleanup_video.DEFAULT_WORKFLOW.read_text(encoding="utf-8-sig"))
    prefix = f"arp_ltx_colorize/{output.stem}"
    prompt = patch_workflow(
        graph_args, workflow, prepared, comfy_dir, prefix, prompt_text, "", seed,
        guide_image=start_guide["image"] if start_guide else None,
        extra_guides=extra_guides or None,
    )
    save = prompt[graph_args.output_node_id]
    save["inputs"] = vhs_inputs_for_profile(save["inputs"], args.intermediate_profile)
    prompt_id = queue_prompt(args.comfy_url, prompt)
    history = wait_for_prompt(args.comfy_url, prompt_id, args.poll_seconds)
    output_root = resolve_path(args.comfy_output_root) if args.comfy_output_root else comfy_dir / "output"
    produced = newest_output(extract_output_files(history, output_root), {".mp4", ".mov", ".mkv", ".webm"}, "LTX Colorizer output")
    cleanup_video.normalize_chunk(args.ffmpeg_path, produced, output, width, height, fps, frames, args.intermediate_profile)


def extract_frame(ffmpeg: str, video: Path, index: int, target: Path) -> Path:
    target = target.with_name(f"{target.stem}_{file_fingerprint(video)['sha256'][:10]}{target.suffix}")
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.stem + ".partial.png")
    subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-i", str(video), "-vf", f"select=eq(n\\,{index})", "-frames:v", "1", str(partial)],
        check=True,
    )
    replace_with_retry(partial, target)
    return target


class FrameReader:
    """Sequential RGB frames from a video through an ffmpeg pipe."""

    def __init__(self, ffmpeg: str, video: Path, width: int, height: int) -> None:
        self.size = width * height * 3
        self.shape = (height, width, 3)
        self.process = subprocess.Popen(
            [ffmpeg, "-v", "error", "-i", str(video), "-vf", f"scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE,
        )
        self.last: np.ndarray | None = None

    def read(self) -> np.ndarray:
        data = self.process.stdout.read(self.size) if self.process.stdout else b""
        if len(data) == self.size:
            self.last = np.frombuffer(data, np.uint8).reshape(self.shape)
        if self.last is None:
            raise RuntimeError("LTX Colorizer chunk produced no frames.")
        return self.last  # a short chunk holds its final frame

    def close(self) -> None:
        if self.process.stdout:
            self.process.stdout.close()
        self.process.wait()


def assemble_shot(
    ffmpeg: str,
    chunks: list[tuple[int, int]],
    renders: list[Path],
    output: Path,
    width: int,
    height: int,
    fps: float,
    ramp: int,
    profile: str,
) -> None:
    """Join a shot's chunk renders, crossfading wherever they overlap."""
    start, end = chunks[0][0], chunks[-1][1]
    partial = output.with_name(output.stem + ".partial" + output.suffix)
    writer = subprocess.Popen(
        [
            ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", f"{fps:.8f}", "-i", "-", "-an", *codec_args(profile), *container_args(str(partial), profile), str(partial),
        ],
        stdin=subprocess.PIPE,
    )
    readers: dict[int, FrameReader] = {}
    try:
        for frame in range(start, end):
            total = np.zeros((height, width, 3), np.float32)
            weight_sum = 0.0
            for index, (first, last) in enumerate(chunks):
                if not first <= frame < last:
                    if index in readers and frame >= last:
                        readers.pop(index).close()
                    continue
                reader = readers.get(index) or readers.setdefault(index, FrameReader(ffmpeg, renders[index], width, height))
                image = reader.read()
                weight = max(1e-3, min(1.0, (frame - first + 1) / ramp, (last - frame) / ramp))
                total += image.astype(np.float32) * weight
                weight_sum += weight
            writer.stdin.write(np.clip(total / weight_sum + 0.5, 0, 255).astype(np.uint8).tobytes())
    finally:
        for reader in readers.values():
            reader.close()
        writer.stdin.close()
        writer.wait()
    if writer.returncode:
        raise RuntimeError(f"Encoding the LTX Colorizer shot failed: {output}")
    replace_with_retry(partial, output)


def read_frames(ffmpeg: str, video: Path, indices: list[int], width: int, height: int, gray: bool = True) -> dict[int, np.ndarray]:
    """Selected frames of a (short) video, as gray or RGB arrays keyed by frame index."""
    wanted = sorted(set(indices))
    if not wanted:
        return {}
    select = "+".join(f"eq(n\\,{index})" for index in wanted)
    pix_fmt, channels = ("gray", 1) if gray else ("rgb24", 3)
    raw = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(video), "-vf", f"select='{select}',scale={width}:{height}", "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
        check=True, capture_output=True,
    ).stdout
    size = width * height * channels
    frames = [np.frombuffer(raw[offset:offset + size], np.uint8).reshape((height, width) if gray else (height, width, 3)) for offset in range(0, len(raw) - size + 1, size)]
    return dict(zip(wanted, frames))


class WarpSource:
    """A shot's colour reference and the grey frame it was painted from, ready to warp."""

    def __init__(self, ffmpeg: str, reference: Path, prepared: Path, local_frame: int, width: int, height: int) -> None:
        import cv2

        self.reference = reference
        self.fingerprint = file_fingerprint(reference)["sha256"][:12]
        colour = cv2.cvtColor(cv2.imread(str(reference), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        self.colour = cv2.resize(colour, (width, height), interpolation=cv2.INTER_AREA)
        self.gray = read_frames(ffmpeg, prepared, [local_frame], width, height)[local_frame]


def match_score(warped: np.ndarray, target: np.ndarray) -> float:
    """Correlation of the blurred greys inside the picture area (black bars would inflate it)."""
    import cv2

    columns = np.flatnonzero(target.mean(axis=0) > 8)
    rows = np.flatnonzero(target.mean(axis=1) > 8)
    if len(columns) < 16 or len(rows) < 16:
        return 0.0
    box = (slice(rows[0], rows[-1] + 1), slice(columns[0], columns[-1] + 1))

    def normalised(image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image[box].astype(np.float32), (0, 0), 2.0)
        return (blurred - blurred.mean()) / (blurred.std() + 1e-6)

    return float((normalised(warped) * normalised(target)).mean())


def colour_only_guide(warped_colour: np.ndarray, warped_gray: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The target grey frame, tinted and shaded by the warped reference at a coarse scale only.

    Flow warps squash and smear moving characters, and an LTX guide is copied closely, so the
    warp never supplies shape: the guide's detail is the real grey frame, and only blurred colour
    and brightness come from the warp. Where the warped grey disagrees with the target (the warp
    is wrong there), the warp's share falls to zero and the guide stays grey.
    """
    import cv2

    def blur(plane: np.ndarray, sigma: float = 3.0) -> np.ndarray:
        return cv2.GaussianBlur(plane.astype(np.float32), (0, 0), sigma)

    target_f = target.astype(np.float32)
    error = np.abs(blur(warped_gray, 2.0) - blur(target_f, 2.0))
    confidence = blur(np.clip(1.0 - (error - 6.0) / 15.0, 0.0, 1.0), 3.0) ** 2
    ycc = cv2.cvtColor(warped_colour, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    luma = target_f + confidence * (blur(ycc[..., 0]) - blur(target_f))
    chroma = [128.0 + confidence * (blur(ycc[..., channel]) - 128.0) for channel in (1, 2)]
    guide = np.clip(np.dstack([luma, *chroma]) + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(guide, cv2.COLOR_YCrCb2RGB)


def warped_reference_guides(
    args: argparse.Namespace,
    ffmpeg: str,
    source: WarpSource | None,
    prepared: Path,
    first: int,
    frames: int,
    taken: list[int],
    width: int,
    height: int,
    work: Path,
) -> list[dict[str, Any]]:
    """Guides every ltx_warp_interval frames: the reference flowed onto that grey frame.

    DIS optical flow from each target grey frame to the reference's grey frame says where each
    pixel came from; the colour reference is remapped through it. A frame the warp does not
    explain (a cut inside the shot, a fast move) scores low and gets no guide.
    """
    import cv2

    if source is None:
        return []
    interval = max(8, int(args.ltx_warp_interval) // 8 * 8)
    locals_ = [local for local in range(interval // 2 // 8 * 8 or 8, frames, interval) if all(abs(local - other) >= 12 for other in taken)]
    targets = read_frames(ffmpeg, prepared, locals_, width, height)
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    guides: list[dict[str, Any]] = []
    for local, target in targets.items():
        motion = flow.calc(target, source.gray, None)
        map_x, map_y = grid_x + motion[..., 0], grid_y + motion[..., 1]
        warped_gray = cv2.remap(source.gray, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        score = match_score(warped_gray, target)
        if score < 0.7:  # a cut inside the shot, or a different view
            print(f"  warped reference at frame {first + local}: match {score:.2f}, too low; no guide", flush=True)
            continue
        image = work / f"warp{WARP_GUIDE_VERSION}_{source.fingerprint}_{first + local:06d}.png"
        if not image.is_file():
            warped = cv2.remap(source.colour, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            guide = colour_only_guide(warped, warped_gray, target)
            cv2.imwrite(str(image), cv2.cvtColor(guide, cv2.COLOR_RGB2BGR))
        guides.append({"image": image, "frame_idx": local, "strength": float(args.ltx_warp_strength), "match": round(score, 3)})
    return guides


def run_ltx_colorization(
    args: argparse.Namespace,
    processing_video: Path,
    rows: list[dict[str, str]],
    plan: list[dict[str, int]],
    row_references,
    ffmpeg: str,
    width: int,
    height: int,
    fps: float,
) -> list[Path]:
    """Colourise every planned shot; returns one segment per shot, in order, for stitching."""
    from outpaint_video import split_chunk

    args.ffmpeg_path = ffmpeg
    comfy_dir = resolve_path(args.comfy_dir)
    ensure_ltx_colorize_models(comfy_dir)
    wait_for_comfy(args.comfy_url, timeout_seconds=180, poll_seconds=args.poll_seconds)
    ensure_node_types(args.comfy_url, {**REQUIRED_NODES, **(DESCRIBE_NODES if args.ltx_describe else {})}, "LTX Colorizer", comfy_dir)
    source_sig = file_fingerprint(processing_video)
    work = ROOT / ".cache" / "ltx_colorize" / f"{safe_stem(processing_video.stem)}_{source_sig['sha256'][:12]}"
    work.mkdir(parents=True, exist_ok=True)
    settings = settings_signature(args)

    # Describe every shot first, so Qwen3-VL and LTX never compete for VRAM mid-run.
    references = [row_references(row) for row in rows]
    prompts = [shot_prompt(args, row, refs) for row, refs in zip(rows, references)]
    free_comfy_models(args.comfy_url)

    segments: list[Path] = []
    for shot, (row, item) in enumerate(zip(rows, plan)):
        start, end = item["start"], item["end"]
        chunks = shot_chunks(start, end, chunk_frames_for(args.ltx_chunk_seconds, fps), args.ltx_overlap_frames)
        anchors = [
            ref for ref in references[shot]
            if ref["selected_frame"] is not None and start <= ref["selected_frame"] < end
        ]
        if not anchors and references[shot]:
            middle = (start + end) // 2
            print(f"Shot {shot + 1}: its reference has no frame number inside the shot; guiding at frame {middle}.", flush=True)
            anchors = [{**references[shot][0], "selected_frame": middle}]
        anchor = anchor_chunk(chunks, [ref["selected_frame"] for ref in anchors])
        warp_source = None
        if anchors and args.ltx_warp_interval > 0:
            first_anchor = anchors[0]
            anchor_first, anchor_last = chunks[anchor]
            anchor_prepared = work / f"gray_{anchor_first:06d}_{anchor_last:06d}.mkv"
            split_chunk(ffmpeg, processing_video, anchor_prepared, anchor_first, anchor_last, fps, False, prepared_fingerprint=source_sig, intermediate_profile=args.intermediate_profile)
            warp_source = WarpSource(ffmpeg, Path(first_anchor["path"]), anchor_prepared, first_anchor["selected_frame"] - anchor_first, width, height)
        segment = work / f"shot_{shot:04d}_{start:06d}_{end:06d}.mkv"
        renders = [work / f"shot_{shot:04d}_chunk_{index:02d}_{first:06d}_{last:06d}.mkv" for index, (first, last) in enumerate(chunks)]
        print(f"LTX Colorizer shot {shot + 1}/{len(rows)}: frames {start}-{end}, {len(chunks)} chunk(s); prompt: {prompts[shot]}", flush=True)

        for index in processing_order(len(chunks), anchor):
            first, last = chunks[index]
            frames = last - first
            start_guide: dict[str, Any] | None = None
            extra: list[dict[str, Any]] = []
            for ref in anchors:
                local = ref["selected_frame"] - first
                if 0 <= local < frames:
                    guide = {"image": Path(ref["path"]), "frame_idx": local, "strength": args.ltx_reference_strength}
                    if local == 0:
                        start_guide = guide
                    else:
                        extra.append(guide)
            if index > anchor:  # forwards: lead in from the previous chunk's colours
                previous_last = chunks[index - 1][1]
                for local in guide_grid(0, min(previous_last, last) - first - 1):
                    frame_png = extract_frame(ffmpeg, renders[index - 1], first + local - chunks[index - 1][0], work / f"{renders[index - 1].stem}_f{first + local:06d}.png")
                    guide = {"image": frame_png, "frame_idx": local, "strength": args.ltx_continuity_strength}
                    if local == 0:
                        start_guide = start_guide or guide
                    else:
                        extra.append(guide)
            elif index < anchor:  # backwards: run out into the next chunk's colours
                next_first = chunks[index + 1][0]
                for local in guide_grid(max(1, next_first - first), frames - 1):
                    frame_png = extract_frame(ffmpeg, renders[index + 1], first + local - next_first, work / f"{renders[index + 1].stem}_f{first + local:06d}.png")
                    extra.append({"image": frame_png, "frame_idx": local, "strength": args.ltx_continuity_strength})
            extra = [guide for guide in extra if guide["frame_idx"] != 0]

            prepared = work / f"gray_{first:06d}_{last:06d}.mkv"
            split_chunk(ffmpeg, processing_video, prepared, first, last, fps, False, prepared_fingerprint=source_sig, intermediate_profile=args.intermediate_profile)
            # LTX's colour fades within a few dozen frames of a guide, so fill the rest of the
            # chunk with the reference carried onto later frames by optical flow.
            if anchors and args.ltx_warp_interval > 0:
                taken = [guide["frame_idx"] for guide in extra] + ([0] if start_guide else [])
                extra += warped_reference_guides(args, ffmpeg, warp_source, prepared, first, frames, taken, width, height, work)
            seed = args.ltx_seed + first
            chunk_sig = {
                "version": VERSION,
                "kind": "LTX Colorizer chunk",
                "source": source_sig,
                "frames": [first, last],
                "prompt": prompts[shot],
                "seed": seed,
                "start_guide": guide_signature(start_guide),
                "guides": [guide_signature(guide) for guide in extra],
                "settings": settings,
                "profile": args.intermediate_profile,
            }
            render = renders[index]
            if not args.force and resumable_output(render, chunk_sig, width=width, height=height):
                print(f"Reuse LTX Colorizer chunk {index + 1}/{len(chunks)} of shot {shot + 1}: {render.name}", flush=True)
                continue
            guide_text = ", ".join(f"{guide['frame_idx']}@{guide['strength']:g}" for guide in ([start_guide] if start_guide else []) + extra) or "none"
            print(f"LTX Colorizer shot {shot + 1}/{len(rows)} chunk {index + 1}/{len(chunks)}: frames {first}-{last}, guides {guide_text}", flush=True)
            run_chunk(args, prepared, render, prompts[shot], seed, start_guide, extra, width, height, fps, frames)
            write_signature(render, chunk_sig)

        segment_sig = {
            "version": VERSION,
            "kind": "LTX Colorizer shot",
            "chunks": [file_fingerprint(render) for render in renders],
            "ranges": chunks,
        }
        if args.force or not resumable_output(segment, segment_sig, width=width, height=height):
            assemble_shot(ffmpeg, chunks, renders, segment, width, height, fps, max(1, args.ltx_overlap_frames), args.intermediate_profile)
            write_signature(segment, segment_sig)
        segments.append(segment)
    return segments


def guide_signature(guide: dict[str, Any] | None) -> dict[str, Any] | None:
    if not guide:
        return None
    return {"image": file_fingerprint(Path(guide["image"])), "frame_idx": guide["frame_idx"], "strength": guide["strength"]}
