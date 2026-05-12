#!/usr/bin/env python3
"""Generate color-reference manifests using TransNetV2 shot detection."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import generate_scene_reference_manifest as legacy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRANSNET_ROOT = ROOT / "tools" / "TransNetV2"
DEFAULT_TRANSNET_WEIGHTS = ROOT / "models" / "transnetv2" / "transnetv2-pytorch-weights.pth"


def load_transnet_module(transnet_root: Path):
    module_path = transnet_root / "inference-pytorch" / "transnetv2_pytorch.py"
    if not module_path.exists():
        raise RuntimeError(
            "TransNetV2 PyTorch module not found. Expected: "
            f"{module_path}\nClone https://github.com/soCzech/TransNetV2 into {transnet_root}"
        )
    spec = importlib.util.spec_from_file_location("transnetv2_pytorch", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import TransNetV2 module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["transnetv2_pytorch"] = module
    spec.loader.exec_module(module)
    return module


def load_transnet_frames(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(frame_rgb, (48, 27), interpolation=cv2.INTER_AREA))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames could be read for TransNetV2: {path}")
    return np.stack(frames).astype(np.uint8)


def transnet_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def predict_transnet(args: argparse.Namespace, source_path: Path) -> np.ndarray:
    import torch

    transnet_root = args.transnet_root.resolve()
    weights_path = args.transnet_weights.resolve()
    if not weights_path.exists():
        raise RuntimeError(
            "TransNetV2 PyTorch weights not found. Expected: "
            f"{weights_path}\nCreate this file by converting the upstream weights, or place an existing "
            "transnetv2-pytorch-weights.pth there."
        )

    module = load_transnet_module(transnet_root)
    device = transnet_device(args.transnet_device)
    model = module.TransNetV2()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    frames = load_transnet_frames(source_path)
    total = len(frames)
    window = args.transnet_window_frames
    overlap = args.transnet_overlap_frames
    if overlap >= window:
        raise RuntimeError("--transnet-overlap-frames must be smaller than --transnet-window-frames")

    scores = np.zeros(total, dtype=np.float32)
    counts = np.zeros(total, dtype=np.float32)
    step = window - overlap

    with torch.no_grad():
        for start in range(0, total, step):
            end = min(total, start + window)
            chunk = frames[start:end]
            actual = len(chunk)
            if actual < window:
                pad = np.repeat(chunk[-1:,:,:,:], window - actual, axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            tensor = torch.from_numpy(chunk[None]).to(device)
            single, many = model(tensor)
            single_scores = torch.sigmoid(single).squeeze().detach().cpu().numpy()
            many_scores = torch.sigmoid(many["many_hot"]).squeeze().detach().cpu().numpy()
            if args.transnet_head == "single":
                chunk_scores = single_scores
            elif args.transnet_head == "many_hot":
                chunk_scores = many_scores
            else:
                chunk_scores = np.maximum(single_scores, many_scores)
            scores[start:end] += chunk_scores[:actual]
            counts[start:end] += 1.0
            if end >= total:
                break

    counts[counts == 0] = 1.0
    return scores / counts


def boundary_peaks(scores: np.ndarray, threshold: float, min_gap_frames: int) -> list[tuple[int, float]]:
    peaks: list[tuple[int, float]] = []
    i = 0
    while i < len(scores):
        if scores[i] < threshold:
            i += 1
            continue
        start = i
        while i < len(scores) and scores[i] >= threshold:
            i += 1
        end = i
        local = scores[start:end]
        peak = start + int(np.argmax(local))
        peak_score = float(scores[peak])
        if not peaks or peak - peaks[-1][0] >= min_gap_frames:
            peaks.append((peak, peak_score))
        elif peak_score > peaks[-1][1]:
            peaks[-1] = (peak, peak_score)
    return peaks


def shots_from_boundaries(boundaries: list[int], samples: list[legacy.Sample], frame_count: int) -> list[legacy.Shot]:
    cleaned = [0]
    for boundary in sorted(set(boundaries)):
        if 0 < boundary < frame_count and boundary > cleaned[-1]:
            cleaned.append(boundary)
    if cleaned[-1] != frame_count:
        cleaned.append(frame_count)

    shots: list[legacy.Shot] = []
    for index, (start, end) in enumerate(zip(cleaned, cleaned[1:])):
        shot_samples = [sample for sample in samples if start <= sample.frame < end]
        shots.append(legacy.Shot(index=index, start_frame=start, end_frame=end, samples=shot_samples))
    return shots


def shot_similarity(a: legacy.Shot, b: legacy.Shot, info: legacy.VideoInfo) -> float:
    a_sample = legacy.representative_sample(a.samples, a.start_frame, a.end_frame, info.fps)
    b_sample = legacy.representative_sample(b.samples, b.start_frame, b.end_frame, info.fps)
    return legacy.reuse_similarity_score(a_sample, b_sample)


def merge_short_similar_shots(shots: list[legacy.Shot], info: legacy.VideoInfo, args: argparse.Namespace) -> list[list[legacy.Shot]]:
    if not shots:
        return []
    groups: list[list[legacy.Shot]] = [[shots[0]]]
    for shot in shots[1:]:
        current = groups[-1]
        current_span = (current[-1].end_frame - current[0].start_frame) / info.fps
        shot_span = (shot.end_frame - shot.start_frame) / info.fps
        previous = current[-1]
        similarity = shot_similarity(previous, shot, info)
        shot_sample = legacy.representative_sample(shot.samples, shot.start_frame, shot.end_frame, info.fps)
        fade_fragment = (
            shot_span < args.merge_shorter_than_seconds
            and (shot_sample.black_ratio >= args.merge_fade_black_ratio or shot_sample.mean_luma <= args.merge_fade_luma)
        )
        should_merge = (
            fade_fragment
            or (
                (current_span < args.merge_shorter_than_seconds or shot_span < args.merge_shorter_than_seconds)
                and similarity <= args.merge_similar_threshold
            )
        )
        if should_merge:
            current.append(shot)
        else:
            groups.append([shot])
    return groups


def transnet_shots(args: argparse.Namespace, source_path: Path, info: legacy.VideoInfo, samples: list[legacy.Sample]) -> tuple[list[legacy.Shot], list[tuple[int, float]]]:
    scores = predict_transnet(args, source_path)
    min_gap_frames = max(1, int(round(args.min_shot_seconds * info.fps)))
    peaks = boundary_peaks(scores, args.transnet_threshold, min_gap_frames)
    boundaries = [frame for frame, _score in peaks]
    return shots_from_boundaries(boundaries, samples, info.frame_count), peaks


def manifest_path_for(args: argparse.Namespace) -> Path:
    if args.output_manifest:
        return args.output_manifest
    return ROOT / "manifests" / "colorize" / f"colorize_manifest_{legacy.safe_stem(args.source_video)}_transnet.csv"


def reference_set_name(args: argparse.Namespace) -> str:
    if args.reference_set:
        return args.reference_set.replace(" ", "_")
    if args.output_manifest:
        return args.output_manifest.stem.replace(" ", "_")
    return f"{legacy.safe_stem(args.source_video)}_transnet"


def print_dry_run(args: argparse.Namespace, info: legacy.VideoInfo, raw_shots: list[legacy.Shot], scenes: list[legacy.Scene], manifest: Path, peaks: list[tuple[int, float]]) -> None:
    unique_paths = {scene.reference_path for scene in scenes}
    estimated_api_calls = sum(1 for path in unique_paths if not path.exists())
    print(f"Source: {args.source_video}")
    print(f"Size: {info.width}x{info.height}, fps={info.fps:.6g}, duration={info.duration:.3f}s")
    print(f"TransNet raw boundaries: {len(peaks)}")
    print(f"TransNet raw shots: {len(raw_shots)}")
    print(f"Post-merged manifest cut entries: {len(scenes)}")
    print(f"Unique references: {len(unique_paths)}")
    print(f"Reused references: {len(scenes) - len(unique_paths)}")
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
            f"{scene.index:04d} end={legacy.format_time(scene.end_frame / info.fps)} "
            f"shot_count={len(scene.shots)} selected={legacy.format_time(scene.selected_time)} "
            f"ref={scene.reference_rel} [{status}]{reuse_note}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", required=True, help="Video path relative to this repository input folder, or absolute path.")
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--model", default=legacy.DEFAULT_MODEL)
    parser.add_argument("--quality", default=legacy.DEFAULT_QUALITY)
    parser.add_argument("--size", default=legacy.DEFAULT_SIZE)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--reference-root", type=Path, default=legacy.DEFAULT_REF_ROOT)
    parser.add_argument("--reference-set", default=None)
    parser.add_argument("--transnet-root", type=Path, default=DEFAULT_TRANSNET_ROOT)
    parser.add_argument("--transnet-weights", type=Path, default=DEFAULT_TRANSNET_WEIGHTS)
    parser.add_argument("--transnet-threshold", type=float, default=0.5)
    parser.add_argument("--transnet-head", choices=["max", "single", "many_hot"], default="max")
    parser.add_argument("--transnet-device", default="auto")
    parser.add_argument("--transnet-window-frames", type=int, default=100)
    parser.add_argument("--transnet-overlap-frames", type=int, default=50)
    parser.add_argument("--merge-shorter-than-seconds", type=float, default=6.0)
    parser.add_argument("--merge-similar-threshold", type=float, default=0.07)
    parser.add_argument("--merge-fade-black-ratio", type=float, default=0.55)
    parser.add_argument("--merge-fade-luma", type=float, default=35.0)
    reuse_group = parser.add_mutually_exclusive_group()
    reuse_group.add_argument("--reuse-similar", dest="reuse_similar", action="store_true", default=True)
    reuse_group.add_argument("--no-reuse-similar", dest="reuse_similar", action="store_false")
    parser.add_argument("--reuse-window", type=int, default=6)
    parser.add_argument("--reuse-threshold", type=float, default=0.07)
    existing_reuse_group = parser.add_mutually_exclusive_group()
    existing_reuse_group.add_argument("--reuse-existing-references", dest="reuse_existing_references", action="store_true", default=True)
    existing_reuse_group.add_argument("--no-reuse-existing-references", dest="reuse_existing_references", action="store_false")
    parser.add_argument("--existing-reuse-threshold", type=float, default=0.025)
    parser.add_argument("--sample-seconds", type=float, default=0.0, help="0 means keep every frame available for representative selection.")
    parser.add_argument("--min-shot-seconds", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
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
    if args.limit is not None and args.limit <= 0:
        raise RuntimeError("--limit must be greater than zero")
    if args.sample_seconds < 0:
        raise RuntimeError("--sample-seconds must be zero or greater")
    if args.reuse_window <= 0:
        raise RuntimeError("--reuse-window must be greater than zero")

    source_path = legacy.resolve_input_path(args.source_video)
    if not source_path.exists():
        raise FileNotFoundError(f"source video not found: {source_path}")
    args.reference_root = args.reference_root.resolve()
    args.transnet_root = args.transnet_root if args.transnet_root.is_absolute() else ROOT / args.transnet_root
    args.transnet_weights = args.transnet_weights if args.transnet_weights.is_absolute() else ROOT / args.transnet_weights
    if args.output_manifest and not args.output_manifest.is_absolute():
        args.output_manifest = ROOT / args.output_manifest
    args.reference_set_name = reference_set_name(args)

    info = legacy.probe_video(source_path)
    samples = legacy.sample_video(source_path, info, args.sample_seconds)
    raw_shots, peaks = transnet_shots(args, source_path, info, samples)
    groups = merge_short_similar_shots(raw_shots, info, args)
    scenes = legacy.build_scenes(args.source_video, groups, info, args)
    legacy.apply_existing_reference_reuse(scenes, legacy.existing_reference_candidates(args), args)
    legacy.apply_reference_reuse(scenes, args)
    manifest = manifest_path_for(args).resolve()

    if args.dry_run:
        print_dry_run(args, info, raw_shots, scenes, manifest, peaks)
        return 0

    legacy.generate_references(args, source_path, info, scenes)
    legacy.write_manifest(manifest, args.source_video, scenes, info)
    print(f"Wrote manifest: {manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=os.sys.stderr)
        raise SystemExit(1)


