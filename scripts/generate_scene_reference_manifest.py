#!/usr/bin/env python3
"""Detect cuts, generate color references, and write a manifest."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REF_ROOT = ROOT / "input" / "references" / "generated_scene_refs"
DEFAULT_MODEL = "gpt-image-2"
DEFAULT_QUALITY = "medium"
DEFAULT_SIZE = "auto"
IMAGE_PROMPT = (
    "Colourise and restore this black-and-white public-domain film frame.\n\n"
    "This is a preservation task, not a redesign. Keep the input image as the master plate. Add only plausible colour and gentle restoration.\n\n"
    "Absolute preservation rules: keep the exact same composition, geometry, silhouettes, people, objects, logo shapes, typography, lettering, framing, camera angle, "
    "lighting pattern, shadows, grain, and lens softness. Do not add people. Do not remove people. Do not add props, scenery, symbols, text, decoration, faces, "
    "buildings, animals, vehicles, or background detail. Do not reinterpret logos or title cards. If there is text or a logo, preserve its shapes exactly and only tint the existing ink, metal, or paper.\n\n"
    "Style goal: believable modern cinematic colour restoration while remaining historically authentic. The result should feel like a pristine colour restoration of the same frame, "
    "not an AI repainting, remake, illustration, fantasy artwork, or newly generated scene.\n\n"
    "Colour grading: use clear but realistic cinematic colour. Give existing materials believable colour identity: warm indoor light, cool blue-grey fog and moonlight outdoors, "
    "polished dark wood, brass highlights, burgundy and deep green fabrics, cream walls, patterned wallpaper and upholstery, natural skin tones, deep charcoal blacks, and preserved shadow detail. "
    "The image should be fully colour-restored, not grey, monochrome, sepia, or muddy, but avoid neon colour, cartoon colour, over-saturation, teal/orange blockbuster grading, hand-tinting, or HDR.\n\n"
    "Film look: high-quality restoration scanned from vintage film. Preserve contrast, subtle grain, gentle optical bloom around practical lights, and authentic period lens softness.\n\n"
    "People should remain recognisable with realistic skin texture and subtle facial tones. Preserve original makeup and costume styling. Avoid beautification, glamour retouching, altered writing, altered faces, modern objects, or changed composition.\n\n"
    "Final result: the same image, faithfully preserved, with natural historically plausible colour added."
)


class ImageEditError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, detail: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float


@dataclass
class Sample:
    frame: int
    time: float
    mean_luma: float
    black_ratio: float
    sharpness: float
    hist: np.ndarray
    color_hist: np.ndarray
    edge_hist: np.ndarray
    dhash: np.ndarray
    gray_small: np.ndarray


@dataclass
class Shot:
    index: int
    start_frame: int
    end_frame: int
    samples: list[Sample]


@dataclass
class Scene:
    index: int
    start_frame: int
    end_frame: int
    shots: list[Shot]
    selected_frame: int
    selected_time: float
    source_frame_path: Path
    reference_path: Path
    reference_rel: str
    selected_sample: Sample
    reused_from: int | None = None
    reused_existing_from: str | None = None
    own_reference_path: Path | None = None


@dataclass
class ExistingReference:
    source_path: Path
    reference_path: Path
    reference_rel: str
    sample: Sample


def resolve_input_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / "input" / path_text


def input_relative(path: Path) -> str:
    return path.resolve().relative_to((ROOT / "input").resolve()).as_posix()


def safe_stem(path_text: str) -> str:
    return Path(path_text).stem.replace(" ", "_")


def manifest_id_from_source(path_text: str) -> str:
    stem = safe_stem(path_text)
    if stem.endswith("_outpaint"):
        stem = stem.removesuffix("_outpaint")
    parts = stem.split("_")
    for part in parts:
        if "to" in part and any(char.isdigit() for char in part):
            return part
    return stem


def format_time(seconds: float) -> str:
    total_millis = int(round(seconds * 1000))
    total = total_millis // 1000
    millis = total_millis % 1000
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if millis:
        return f"{hours:d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def format_stamp(seconds: float) -> str:
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}.{minutes:02d}.{secs:02d}"


def probe_video(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Could not read video metadata: {path}")
    return VideoInfo(width=width, height=height, fps=fps, frame_count=frame_count, duration=frame_count / fps)


def frame_hist(gray_small: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([gray_small], [0], None, [32], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def color_hist(frame_bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def edge_hist(gray_small: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray_small, 60, 140)
    hist = cv2.calcHist([edges], [0], None, [16], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def dhash(gray_small: np.ndarray) -> np.ndarray:
    tiny = cv2.resize(gray_small, (17, 16), interpolation=cv2.INTER_AREA)
    return (tiny[:, 1:] > tiny[:, :-1]).astype(np.uint8).flatten()


def hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    corr = cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CORREL)
    if np.isnan(corr):
        return 1.0
    return float(max(0.0, min(2.0, 1.0 - corr)))


def array_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def hash_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a != b))


def analyze_frame(frame_bgr: np.ndarray, frame_index: int, fps: float) -> Sample:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
    mean_luma = float(np.mean(gray_small))
    black_ratio = float(np.mean(gray_small < 18))
    sharpness = float(cv2.Laplacian(gray_small, cv2.CV_64F).var())
    return Sample(
        frame=frame_index,
        time=frame_index / fps,
        mean_luma=mean_luma,
        black_ratio=black_ratio,
        sharpness=sharpness,
        hist=frame_hist(gray_small),
        color_hist=color_hist(frame_bgr),
        edge_hist=edge_hist(gray_small),
        dhash=dhash(gray_small),
        gray_small=gray_small,
    )


def analyze_image(path: Path) -> Sample:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read image: {path}")
    return analyze_frame(frame, 0, 1.0)


def sample_video(path: Path, info: VideoInfo, sample_seconds: float) -> list[Sample]:
    step = 1 if sample_seconds <= 0 else max(1, int(round(info.fps * sample_seconds)))
    capture = cv2.VideoCapture(str(path))
    samples: list[Sample] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % step == 0:
            samples.append(analyze_frame(frame, frame_index, info.fps))
        frame_index += 1
    capture.release()
    if not samples:
        raise RuntimeError("No video samples could be read")
    return samples


def transition_score(prev: Sample, cur: Sample) -> float:
    gray_hist = hist_distance(prev.hist, cur.hist)
    hsv_hist = hist_distance(prev.color_hist, cur.color_hist)
    edge = hist_distance(prev.edge_hist, cur.edge_hist)
    image = array_distance(prev.gray_small, cur.gray_small)
    perceptual = hash_distance(prev.dhash, cur.dhash)
    luma = abs(prev.mean_luma - cur.mean_luma) / 255.0
    black_change = abs(prev.black_ratio - cur.black_ratio)
    return (
        0.28 * hsv_hist
        + 0.22 * image
        + 0.18 * perceptual
        + 0.14 * gray_hist
        + 0.08 * edge
        + 0.06 * luma
        + 0.04 * black_change
    )


def reuse_similarity_score(a: Sample, b: Sample) -> float:
    gray_hist = hist_distance(a.hist, b.hist)
    hsv_hist = hist_distance(a.color_hist, b.color_hist)
    edge = hist_distance(a.edge_hist, b.edge_hist)
    image = array_distance(a.gray_small, b.gray_small)
    perceptual = hash_distance(a.dhash, b.dhash)
    luma = abs(a.mean_luma - b.mean_luma) / 255.0
    black_change = abs(a.black_ratio - b.black_ratio)
    return (
        0.26 * image
        + 0.22 * perceptual
        + 0.18 * gray_hist
        + 0.14 * hsv_hist
        + 0.10 * edge
        + 0.06 * luma
        + 0.04 * black_change
    )


def is_fade_frame(sample: Sample, args: argparse.Namespace) -> bool:
    return sample.black_ratio >= args.fade_black_ratio or sample.mean_luma <= args.fade_luma


def refine_boundary(samples: list[Sample], start_index: int, end_index: int) -> int:
    if end_index <= start_index:
        return samples[end_index].frame
    best_index = max(
        range(start_index + 1, end_index + 1),
        key=lambda i: transition_score(samples[i - 1], samples[i]),
    )
    return samples[best_index].frame


def dissolve_score(samples: list[Sample], index: int, window: int) -> float:
    before = max(0, index - window)
    after = min(len(samples) - 1, index + window)
    if before == index or after == index:
        return 0.0
    return transition_score(samples[before], samples[after])


def near_existing_boundary(boundaries: list[int], frame: int, min_gap_frames: int) -> bool:
    return any(abs(frame - boundary) < min_gap_frames for boundary in boundaries)


def detect_shots(samples: list[Sample], info: VideoInfo, args: argparse.Namespace) -> list[Shot]:
    min_frames = max(1, int(round(args.min_shot_seconds * info.fps)))
    boundary_dedupe_frames = max(1, int(round(args.boundary_dedupe_seconds * info.fps)))
    dissolve_window = max(1, int(round(args.dissolve_window_seconds * info.fps)))
    dissolve_gap_frames = max(min_frames, int(round(args.dissolve_min_gap_seconds * info.fps)))
    boundaries = [0]
    last_boundary = 0
    anchor = samples[0]
    fade_start_index: int | None = None
    scores = [0.0] + [transition_score(samples[i - 1], samples[i]) for i in range(1, len(samples))]
    dissolve_scores = [dissolve_score(samples, i, dissolve_window) for i in range(len(samples))]
    nonzero_scores = np.array([score for score in scores[1:] if score > 0], dtype=np.float32)
    if nonzero_scores.size:
        median = float(np.median(nonzero_scores))
        mad = float(np.median(np.abs(nonzero_scores - median)))
        adaptive_threshold = median + args.dynamic_threshold_scale * max(mad * 1.4826, 0.001)
    else:
        adaptive_threshold = args.shot_threshold
    shot_threshold = max(args.shot_threshold, adaptive_threshold)

    for i in range(1, len(samples)):
        prev = samples[i - 1]
        cur = samples[i]
        far_enough = cur.frame - last_boundary >= min_frames
        prev_fade = is_fade_frame(prev, args)
        cur_fade = is_fade_frame(cur, args)
        adjacent = scores[i]
        prev_score = scores[i - 1] if i > 1 else 0.0
        next_score = scores[i + 1] if i + 1 < len(scores) else 0.0
        peak_margin = adjacent - max(prev_score, next_score)
        local_peak = adjacent >= prev_score and adjacent >= next_score
        dissolve = dissolve_scores[i]
        prev_dissolve = dissolve_scores[i - 1] if i > 1 else 0.0
        next_dissolve = dissolve_scores[i + 1] if i + 1 < len(dissolve_scores) else 0.0
        dissolve_peak = dissolve >= prev_dissolve and dissolve >= next_dissolve

        if cur_fade and not prev_fade and fade_start_index is None:
            fade_start_index = i
        if fade_start_index is not None and prev_fade and not cur_fade:
            boundary = refine_boundary(samples, fade_start_index, i)
            if boundary - last_boundary >= min_frames and not near_existing_boundary(boundaries, boundary, boundary_dedupe_frames):
                boundaries.append(boundary)
                last_boundary = boundary
                anchor = cur
            fade_start_index = None
            continue

        anchor_delta = transition_score(anchor, cur) if args.anchor_threshold > 0 else 0.0
        anchor_far_enough = cur.frame - last_boundary >= int(round(args.anchor_min_seconds * info.fps))
        hard_cut = local_peak and adjacent >= shot_threshold and peak_margin >= args.peak_margin
        gradual_change = (
            args.anchor_threshold > 0
            and anchor_far_enough
            and adjacent >= args.anchor_adjacent_floor
            and anchor_delta >= args.anchor_threshold
        )
        cross_fade = (
            args.dissolve_threshold > 0
            and dissolve_peak
            and dissolve >= args.dissolve_threshold
            and cur.frame - last_boundary >= dissolve_gap_frames
            and not near_existing_boundary(boundaries, cur.frame, dissolve_gap_frames)
        )
        if far_enough and (hard_cut or gradual_change or cross_fade):
            if hard_cut:
                boundary = cur.frame
            elif cross_fade:
                boundary = cur.frame
            else:
                boundary = refine_boundary(samples, max(0, i - 3), i)
            if boundary - last_boundary >= min_frames and not near_existing_boundary(boundaries, boundary, boundary_dedupe_frames):
                boundaries.append(boundary)
                last_boundary = boundary
                anchor = cur
                fade_start_index = None
    boundaries.append(info.frame_count)

    shots: list[Shot] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        shot_samples = [sample for sample in samples if start <= sample.frame < end]
        shots.append(Shot(index=index, start_frame=start, end_frame=end, samples=shot_samples))
    return shots


def representative_sample(samples: list[Sample], start_frame: int, end_frame: int, fps: float) -> Sample:
    if not samples:
        mid_frame = (start_frame + end_frame) // 2
        gray_small = np.full((90, 160), 128, dtype=np.uint8)
        return Sample(
            frame=mid_frame,
            time=mid_frame / fps,
            mean_luma=128.0,
            black_ratio=0.0,
            sharpness=0.0,
            hist=frame_hist(gray_small),
            color_hist=np.zeros(24 * 16, dtype=np.float32),
            edge_hist=edge_hist(gray_small),
            dhash=dhash(gray_small),
            gray_small=gray_small,
        )
    usable = [
        sample
        for sample in samples
        if sample.black_ratio < 0.82 and 14.0 <= sample.mean_luma <= 242.0
    ]
    candidates = usable or samples
    midpoint = (start_frame + end_frame) / 2
    max_sharp = max((sample.sharpness for sample in candidates), default=1.0) or 1.0
    best = max(
        candidates,
        key=lambda sample: (
            (sample.sharpness / max_sharp)
            - 1.8 * sample.black_ratio
            - 0.35 * abs(sample.mean_luma - 92.0) / 255.0
            - 0.65 * abs(sample.frame - midpoint) / max(1, end_frame - start_frame)
        ),
    )
    return best


def should_merge(group: list[Shot], candidate: Shot, info: VideoInfo, args: argparse.Namespace) -> bool:
    group_start = group[0].start_frame
    group_end = group[-1].end_frame
    combined_seconds = (candidate.end_frame - group_start) / info.fps
    if combined_seconds > args.max_scene_seconds:
        return False
    candidate_seconds = (candidate.end_frame - candidate.start_frame) / info.fps
    group_seconds = (group_end - group_start) / info.fps
    if candidate_seconds < args.min_scene_seconds or group_seconds < args.min_scene_seconds:
        return True
    group_sample = representative_sample(
        [sample for shot in group for sample in shot.samples],
        group_start,
        group_end,
        info.fps,
    )
    candidate_sample = representative_sample(candidate.samples, candidate.start_frame, candidate.end_frame, info.fps)
    return transition_score(group_sample, candidate_sample) <= args.merge_threshold


def group_scenes(shots: list[Shot], info: VideoInfo, args: argparse.Namespace) -> list[list[Shot]]:
    groups: list[list[Shot]] = []
    current: list[Shot] = []
    for shot in shots:
        if not current:
            current = [shot]
        elif should_merge(current, shot, info, args):
            current.append(shot)
        else:
            groups.append(current)
            current = [shot]
    if current:
        groups.append(current)
    return groups


def shot_groups(shots: list[Shot]) -> list[list[Shot]]:
    return [[shot] for shot in shots]


def read_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {path}")
    return frame


def write_png_bgr(path: Path, frame_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame_bgr):
        raise RuntimeError(f"Could not write image: {path}")


def build_scenes(source_video: str, groups: list[list[Shot]], info: VideoInfo, args: argparse.Namespace) -> list[Scene]:
    ref_dir = args.reference_root / args.reference_set_name
    source_frame_dir = ref_dir / "_source_frames"
    scenes: list[Scene] = []
    for index, group in enumerate(groups):
        start_frame = group[0].start_frame
        end_frame = group[-1].end_frame
        samples = [sample for shot in group for sample in shot.samples]
        selected = representative_sample(samples, start_frame, end_frame, info.fps)
        name = f"cut_{index:04d}_{format_stamp(selected.time)}.png"
        reference_path = ref_dir / name
        source_frame_path = source_frame_dir / name
        scenes.append(
            Scene(
                index=index,
                start_frame=start_frame,
                end_frame=end_frame,
                shots=group,
                selected_frame=selected.frame,
                selected_time=selected.time,
                source_frame_path=source_frame_path,
                reference_path=reference_path,
                reference_rel=input_relative(reference_path),
                selected_sample=selected,
                own_reference_path=reference_path,
            )
        )
        if args.limit is not None and len(scenes) >= args.limit:
            break
    return scenes


def apply_reference_reuse(scenes: list[Scene], args: argparse.Namespace) -> None:
    if not args.reuse_similar:
        return
    accepted: list[Scene] = []
    for scene in scenes:
        if scene.reused_existing_from is not None:
            accepted.append(scene)
            continue
        best_scene: Scene | None = None
        best_score = float("inf")
        for candidate in accepted[-args.reuse_window:]:
            score = reuse_similarity_score(scene.selected_sample, candidate.selected_sample)
            if score < best_score:
                best_score = score
                best_scene = candidate
        if best_scene is not None and best_score <= args.reuse_threshold:
            scene.reused_from = best_scene.index
            scene.reference_path = best_scene.reference_path
            scene.reference_rel = best_scene.reference_rel
        else:
            accepted.append(scene)


def existing_reference_candidates(args: argparse.Namespace) -> list[ExistingReference]:
    if not args.reuse_existing_references:
        return []
    candidates: list[ExistingReference] = []
    root = args.reference_root
    if not root.exists():
        return candidates
    for ref_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        source_dir = ref_dir / "_source_frames"
        if not source_dir.exists():
            continue
        for source_path in sorted(source_dir.glob("*.png")):
            reference_path = ref_dir / source_path.name
            if not reference_path.exists():
                continue
            try:
                sample = analyze_image(source_path)
            except RuntimeError:
                continue
            candidates.append(
                ExistingReference(
                    source_path=source_path,
                    reference_path=reference_path,
                    reference_rel=input_relative(reference_path),
                    sample=sample,
                )
            )
    return candidates


def apply_existing_reference_reuse(scenes: list[Scene], candidates: list[ExistingReference], args: argparse.Namespace) -> None:
    if not candidates:
        return
    for scene in scenes:
        if scene.reused_from is not None or scene.reference_path.exists():
            continue
        best: ExistingReference | None = None
        best_score = float("inf")
        for candidate in candidates:
            score = reuse_similarity_score(scene.selected_sample, candidate.sample)
            if score < best_score:
                best_score = score
                best = candidate
        if best is not None and best_score <= args.existing_reuse_threshold:
            scene.reused_existing_from = str(best.reference_path.relative_to(args.reference_root))
            scene.reference_path = best.reference_path
            scene.reference_rel = best.reference_rel


def multipart_body(fields: dict[str, str], files: dict[str, Path]) -> tuple[bytes, str]:
    boundary = f"----codex-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")
    for name, path in files.items():
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n".encode("utf-8")
        )
        chunks.append(path.read_bytes())
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), boundary


def image_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8").strip()
    return IMAGE_PROMPT


def call_image_edit_api(args: argparse.Namespace, frame_path: Path) -> bytes:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    fields = {
        "model": args.model,
        "prompt": image_prompt(args),
        "quality": args.quality,
        "size": args.size,
        "output_format": "png",
    }
    body, boundary = multipart_body(fields, {"image[]": frame_path})
    request = urllib.request.Request(
        "https://api.openai.com/v1/images/edits",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=args.api_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ImageEditError(f"OpenAI image edit failed with HTTP {exc.code}: {detail}", exc.code, detail) from exc
    except urllib.error.URLError as exc:
        raise ImageEditError(f"OpenAI image edit connection failed: {exc.reason}") from exc

    item = (payload.get("data") or [{}])[0]
    if "b64_json" in item:
        return base64.b64decode(item["b64_json"])
    if "url" in item:
        with urllib.request.urlopen(item["url"], timeout=args.api_timeout) as response:
            return response.read()
    raise RuntimeError(f"OpenAI image edit response did not include b64_json or url: {payload}")


def is_retryable_image_error(exc: ImageEditError) -> bool:
    detail = exc.detail.lower()
    if exc.status in {408, 409, 429, 500, 502, 503, 504}:
        return True
    if exc.status == 403 and "must be verified" in detail:
        return True
    if exc.status is None:
        return True
    return False


def call_image_edit_api_with_retries(args: argparse.Namespace, frame_path: Path) -> bytes:
    last_error: ImageEditError | None = None
    for attempt in range(1, args.api_retries + 2):
        try:
            return call_image_edit_api(args, frame_path)
        except ImageEditError as exc:
            last_error = exc
            if not is_retryable_image_error(exc) or attempt > args.api_retries:
                raise
            delay = min(args.api_retry_max_seconds, args.api_retry_seconds * (2 ** (attempt - 1)))
            if exc.status == 403 and "must be verified" in exc.detail.lower():
                delay = max(delay, args.api_verification_retry_seconds)
            print(f"API attempt {attempt} failed; retrying in {delay:.0f}s: {exc}")
            time.sleep(delay)
    if last_error:
        raise last_error
    raise RuntimeError("Image edit retry loop ended unexpectedly")


def resize_to_source(image_bytes: bytes, destination: Path, width: int, height: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = Path(tmp.name)
    try:
        with Image.open(tmp_path) as image:
            image = image.convert("RGB")
            src_w, src_h = image.size
            scale = max(width / src_w, height / src_h)
            resized = image.resize((round(src_w * scale), round(src_h * scale)), Image.Resampling.LANCZOS)
            left = max(0, (resized.width - width) // 2)
            top = max(0, (resized.height - height) // 2)
            cropped = resized.crop((left, top, left + width, top + height))
            cropped.save(destination, "PNG")
    finally:
        tmp_path.unlink(missing_ok=True)


def write_manifest(path: Path, source_video: str, scenes: list[Scene], info: VideoInfo) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# source_video={source_video}\n")
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["enabled", "end", "reference"])
        for scene in scenes:
            writer.writerow(["true", format_time(min(scene.end_frame / info.fps, info.duration)), scene.reference_rel])
    os.replace(tmp, path)


def manifest_path_for(args: argparse.Namespace) -> Path:
    if args.output_manifest:
        return args.output_manifest
    return ROOT / "manifests" / "colorize" / f"colorize_manifest_{manifest_id_from_source(args.source_video)}_shots_auto.csv"


def reference_set_name(args: argparse.Namespace) -> str:
    if args.reference_set:
        return args.reference_set.replace(" ", "_")
    if args.output_manifest:
        return args.output_manifest.stem.replace(" ", "_")
    return manifest_path_for(args).stem.replace(" ", "_")


def print_dry_run(args: argparse.Namespace, info: VideoInfo, shots: list[Shot], scenes: list[Scene], manifest: Path) -> None:
    unique_paths = {scene.reference_path for scene in scenes}
    unique_refs = len(unique_paths)
    reused_refs = len(scenes) - unique_refs
    estimated_api_calls = sum(1 for path in unique_paths if not path.exists())
    print(f"Source: {args.source_video}")
    print(f"Size: {info.width}x{info.height}, fps={info.fps:.6g}, duration={info.duration:.3f}s")
    print(f"Detected shots: {len(shots)}")
    print(f"Manifest cut entries: {len(scenes)}")
    print(f"Unique references: {unique_refs}")
    print(f"Reused references: {reused_refs}")
    print(f"Estimated API calls: {estimated_api_calls}")
    print(f"Manifest: {manifest}")
    for scene in scenes:
        status = "exists" if scene.reference_path.exists() else "missing"
        reuse_note = ""
        if scene.reused_from is not None:
            reuse_note = f" reuse=cut_{scene.index:04d}->cut_{scene.reused_from:04d}"
        elif scene.reused_existing_from is not None:
            reuse_note = f" reuse_existing={scene.reused_existing_from}"
        print(
            f"{scene.index:04d} end={format_time(scene.end_frame / info.fps)} "
            f"shot_count={len(scene.shots)} selected={format_time(scene.selected_time)} "
            f"ref={scene.reference_rel} [{status}]{reuse_note}"
        )


def extract_source_frames(args: argparse.Namespace, source_path: Path, scenes: list[Scene]) -> None:
    for scene in scenes:
        if args.force or not scene.source_frame_path.exists():
            frame = read_frame(source_path, scene.selected_frame)
            write_png_bgr(scene.source_frame_path, frame)
            print(f"Wrote source frame {scene.index:04d}: {scene.source_frame_path}")
        if args.extract_only and scene.reused_from is None and (args.force or not scene.reference_path.exists()):
            frame = read_frame(source_path, scene.selected_frame)
            write_png_bgr(scene.reference_path, frame)
            print(f"Wrote manual reference seed {scene.index:04d}: {scene.reference_path}")


def generate_references(args: argparse.Namespace, source_path: Path, info: VideoInfo, scenes: list[Scene]) -> None:
    extract_source_frames(args, source_path, scenes)
    if args.extract_only:
        print("Extract-only mode: skipped OpenAI image edits.")
        print(f"Colorize these source frames manually, then save finished references in: {args.reference_root / args.reference_set_name}")
        return
    for scene in scenes:
        if scene.reused_from is not None:
            print(f"Reuse reference {scene.index:04d} -> {scene.reused_from:04d}: {scene.reference_path}")
            continue
        if scene.reused_existing_from is not None:
            print(f"Reuse existing reference {scene.index:04d}: {scene.reference_path}")
            continue
        if scene.reference_path.exists() and not args.force:
            print(f"Reuse reference {scene.index:04d}: {scene.reference_path}")
            continue
        print(f"Colorize cut {scene.index:04d} @ {format_time(scene.selected_time)} with {args.model} ({args.quality})")
        image_bytes = call_image_edit_api_with_retries(args, scene.source_frame_path)
        resize_to_source(image_bytes, scene.reference_path, info.width, info.height)
        print(f"Wrote reference: {scene.reference_path}")
        time.sleep(args.api_pause_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", required=True, help="Video path relative to this repository input folder, or absolute path.")
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quality", default=DEFAULT_QUALITY)
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--prompt-file", type=Path, default=None, help="Optional UTF-8 text file containing the OpenAI image edit prompt.")
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REF_ROOT)
    parser.add_argument("--reference-set", default=None, help="Subfolder under --reference-root. Defaults to the output manifest stem.")
    reuse_group = parser.add_mutually_exclusive_group()
    reuse_group.add_argument("--reuse-similar", dest="reuse_similar", action="store_true", default=True, help="Reuse recent visually similar references.")
    reuse_group.add_argument("--no-reuse-similar", dest="reuse_similar", action="store_false", help="Disable visually similar reference reuse.")
    parser.add_argument("--reuse-window", type=int, default=6, help="Number of recent unique references to compare against.")
    parser.add_argument("--reuse-threshold", type=float, default=0.07, help="Lower is stricter; reused when similarity score is at or below this value.")
    existing_reuse_group = parser.add_mutually_exclusive_group()
    existing_reuse_group.add_argument("--reuse-existing-references", dest="reuse_existing_references", action="store_true", default=True, help="Reuse already colorized references by matching their saved source frames.")
    existing_reuse_group.add_argument("--no-reuse-existing-references", dest="reuse_existing_references", action="store_false", help="Ignore references from previous generator runs.")
    parser.add_argument("--existing-reuse-threshold", type=float, default=0.025, help="Lower is stricter; previous references are reused when source-frame similarity is at or below this value.")
    parser.add_argument("--sample-seconds", type=float, default=0.0, help="0 means inspect every frame for hard cuts.")
    parser.add_argument("--shot-threshold", type=float, default=0.09)
    parser.add_argument("--dynamic-threshold-scale", type=float, default=3.2)
    parser.add_argument("--peak-margin", type=float, default=0.01)
    parser.add_argument("--anchor-threshold", type=float, default=0.36, help="Cumulative-change threshold for dissolves/fades. Set 0 to disable.")
    parser.add_argument("--anchor-min-seconds", type=float, default=4.0)
    parser.add_argument("--anchor-adjacent-floor", type=float, default=0.004)
    parser.add_argument("--dissolve-threshold", type=float, default=0.20, help="Wide-window cross-fade detector threshold. Set 0 to disable.")
    parser.add_argument("--dissolve-window-seconds", type=float, default=1.5, help="Seconds before/after a candidate frame to compare for cross-fades.")
    parser.add_argument("--dissolve-min-gap-seconds", type=float, default=4.0, help="Minimum spacing between detected cross-fade boundaries.")
    parser.add_argument("--boundary-dedupe-seconds", type=float, default=1.5, help="Suppress duplicate boundaries fired by multiple detectors around the same transition.")
    parser.add_argument("--merge-threshold", type=float, default=0.14)
    parser.add_argument("--min-shot-seconds", type=float, default=1.0)
    parser.add_argument("--min-scene-seconds", type=float, default=6.0)
    parser.add_argument("--max-scene-seconds", type=float, default=60.0)
    parser.add_argument("--fade-black-ratio", type=float, default=0.72)
    parser.add_argument("--fade-luma", type=float, default=18.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extract-only", action="store_true", help="Extract selected source frames and write the manifest without calling OpenAI.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--api-timeout", type=int, default=180)
    parser.add_argument("--api-pause-seconds", type=float, default=0.2)
    parser.add_argument("--api-retries", type=int, default=8)
    parser.add_argument("--api-retry-seconds", type=float, default=20.0)
    parser.add_argument("--api-retry-max-seconds", type=float, default=300.0)
    parser.add_argument("--api-verification-retry-seconds", type=float, default=180.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.sample_seconds < 0:
        raise RuntimeError("--sample-seconds must be zero or greater")
    if args.limit is not None and args.limit <= 0:
        raise RuntimeError("--limit must be greater than zero")
    if args.reuse_window <= 0:
        raise RuntimeError("--reuse-window must be greater than zero")

    source_path = resolve_input_path(args.source_video)
    if not source_path.exists():
        raise FileNotFoundError(f"source video not found: {source_path}")
    args.reference_root = args.reference_root.resolve()
    if args.output_manifest and not args.output_manifest.is_absolute():
        args.output_manifest = ROOT / args.output_manifest
    args.reference_set_name = reference_set_name(args)

    info = probe_video(source_path)
    samples = sample_video(source_path, info, args.sample_seconds)
    shots = detect_shots(samples, info, args)
    groups = shot_groups(shots)
    scenes = build_scenes(args.source_video, groups, info, args)
    apply_existing_reference_reuse(scenes, existing_reference_candidates(args), args)
    apply_reference_reuse(scenes, args)
    manifest = manifest_path_for(args).resolve()

    if args.dry_run:
        print_dry_run(args, info, shots, scenes, manifest)
        return 0

    generate_references(args, source_path, info, scenes)
    write_manifest(manifest, args.source_video, scenes, info)
    print(f"Wrote manifest: {manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=os.sys.stderr)
        raise SystemExit(1)


