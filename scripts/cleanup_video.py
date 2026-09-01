"""Restore archive footage with the LTX 2.3 Dearchive IC-LoRA.

The model works on model-safe dimensions internally, but the delivered file is normalized back to
the source's exact width, height, frame rate, and frame count. Source audio is muxed back in after
the generated chunks are stitched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from comfy_api import extract_output_files, ensure_node_types, queue_prompt, wait_for_comfy, wait_for_prompt
from common import (
    ROOT,
    file_fingerprint,
    find_ffmpeg,
    load_local_config,
    newest_output,
    replace_with_retry,
    resolve_path,
    resumable_output,
    root_relative,
    safe_stem,
    write_signature,
)
from dependency_manager import ensure_cleanup_models
from outpaint_video import chunk_ranges, patch_workflow, split_chunk, stitch_chunks
from prepare_outpaint_input import probe_video
import artifact_ids as aid


DEFAULT_WORKFLOW = ROOT / "workflows" / "cleanup_ltx" / "DeArchive.json"
DEFAULT_CHUNK_SECONDS = 4.04
MIN_CHUNK_SECONDS = 2.0
MAX_CHUNK_SECONDS = 20.0
DEFAULT_COMFY_DIR = ROOT / "tools" / "comfyui"
DEFAULT_PROMPT = (
    "A modern, high-resolution video shot in vivid color, sharp detail, clean tonality, "
    "and contemporary cinematography."
)
DEFAULT_NEGATIVE = "monochrome"
REQUIRED_NODES = {
    "LTXVImgToVideoConditionOnly": "ComfyUI-LTXVideo",
    "LTXAddVideoICLoRAGuide": "ComfyUI-LTXVideo",
    "LTXVPreprocess": "ComfyUI-LTXVideo",
    "VHS_LoadVideo": "ComfyUI-VideoHelperSuite",
    "VHS_VideoCombine": "ComfyUI-VideoHelperSuite",
}
AI_DESCRATCH_NODES = {
    "ProPainterInpaint": "ComfyUI_ProPainter_Nodes",
    "VHS_LoadVideoPath": "ComfyUI-VideoHelperSuite",
    "VHS_VideoCombine": "ComfyUI-VideoHelperSuite",
    "ImageToMask": "ComfyUI core",
}
DEVIGNETTE_VERSION = 1
AI_DESCRATCH_VERSION = 3


def cleanup_identity(source: Path, args: argparse.Namespace) -> dict[str, Any]:
    return aid.cleanup_identity(
        source.name,
        args.cleanup_lora,
        args.prompt,
        args.negative_prompt,
        args.lora_strength,
        args.seed,
        source_fidelity=args.source_fidelity,
        ai_descratch=args.ai_descratch,
        ai_descratch_height=args.ai_descratch_height,
        scratch_sensitivity=args.scratch_sensitivity,
        scratch_mask_dilate=args.scratch_mask_dilate,
        ai_chunk_frames=args.ai_chunk_frames,
        devignette=args.devignette,
        dearchive=args.dearchive,
        dearchive_height=args.dearchive_height,
    )


def default_output(source: Path, args: argparse.Namespace) -> Path:
    ident = cleanup_identity(source, args)
    return ROOT / "intermediate" / "cleaned" / aid.artifact_name(aid.source_word(source.name), "cleanup", ident, "mp4")


def model_dimensions(width: int, height: int, processing_height: int | str = 720) -> tuple[int, int]:
    text = str(processing_height or "720").strip().lower()
    if text in {"source", "original", "0"}:
        return aid.model_safe(width), aid.model_safe(height)
    target_height = aid.model_safe(min(height, max(64, int(float(text)))))
    target_width = aid.model_safe(round(width * target_height / max(1, height)))
    return target_width, target_height


def _contiguous_true_runs(values: Any) -> list[tuple[int, int]]:
    """Return half-open runs in a one-dimensional boolean NumPy array."""
    import numpy as np

    padded = np.pad(np.asarray(values, dtype=np.uint8), (1, 1))
    changes = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(start), int(end)) for start, end in changes.reshape(-1, 2)]


def radial_coordinates(width: int, height: int) -> Any:
    import numpy as np

    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x = (x - (width - 1) * 0.5) / max(width * 0.5, 1.0)
    y = (y - (height - 1) * 0.5) / max(height * 0.5, 1.0)
    return np.sqrt(x * x + y * y)


@dataclass(frozen=True)
class VignetteCorrection:
    gain: Any
    bias: Any
    # Fractions of the full frame: left, top, right, bottom. Black presentation bars sit outside.
    bounds: tuple[float, float, float, float]
    edge_low: float
    edge_high: float


def active_picture_bounds(frames: list[Any]) -> tuple[int, int, int, int]:
    """Find a persistent active-picture rectangle while excluding letter/pillarbox bars."""
    import cv2
    import numpy as np

    height, width = frames[0].shape[:2]
    luminance = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]).astype(np.float32)
    # The high temporal percentile reveals picture content that changes between shots; a black
    # presentation bar remains black in every sample.
    composite = np.percentile(luminance, 85, axis=0)
    peak = float(np.percentile(composite, 95))
    threshold = max(5.0, peak * 0.045)
    columns = np.percentile(composite, 80, axis=0) > threshold
    rows = np.percentile(composite, 80, axis=1) > threshold

    def limits(active: Any, size: int) -> tuple[int, int]:
        indices = np.flatnonzero(active)
        if not indices.size:
            return 0, size
        start, end = int(indices[0]), int(indices[-1]) + 1
        # Tiny dark edges are ordinary image content, not presentation bars.
        if start < size * 0.02:
            start = 0
        if size - end < size * 0.02:
            end = size
        return start, end

    left, right = limits(columns, width)
    top, bottom = limits(rows, height)
    if right - left < width * 0.45 or bottom - top < height * 0.45:
        return 0, 0, width, height
    return left, top, right, bottom


def vignette_profile(frames: list[Any]) -> tuple[Any | None, float]:
    """Estimate a stationary radial illumination field from representative BGR frames.

    Each frame contributes a normalized ring profile, so changing shots and exposure do not
    dominate the estimate. A correction is enabled only when the aggregate edge/centre difference
    is large enough to look like a vignette; its strength is deliberately bounded.
    """
    import cv2
    import numpy as np

    if not frames:
        return None, 1.0
    full_height, full_width = frames[0].shape[:2]
    left, top, right, bottom = active_picture_bounds(frames)
    width, height = right - left, bottom - top
    radius = radial_coordinates(width, height)
    fields: list[Any] = []
    low_grids: list[Any] = []
    high_grids: list[Any] = []
    grid_y, grid_x = 10, 14
    for frame in frames:
        gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY).astype(np.float32) + 1.0
        centre_pixels = gray[radius <= 0.28]
        normalizer = float(np.median(centre_pixels)) if centre_pixels.size else float(np.median(gray))
        if normalizer < 8.0:
            continue
        sigma = max(5.0, min(width, height) / 12.0)
        fields.append(cv2.GaussianBlur(gray / normalizer, (0, 0), sigmaX=sigma, sigmaY=sigma))
        lows = np.zeros((grid_y, grid_x), dtype=np.float32)
        highs = np.zeros((grid_y, grid_x), dtype=np.float32)
        for gy in range(grid_y):
            y0, y1 = gy * height // grid_y, (gy + 1) * height // grid_y
            for gx in range(grid_x):
                x0, x1 = gx * width // grid_x, (gx + 1) * width // grid_x
                tile = gray[y0:y1, x0:x1]
                lows[gy, gx], highs[gy, gx] = np.percentile(tile, (12, 88))
        low_grids.append(lows)
        high_grids.append(highs)
    if not fields:
        return None, 1.0

    field = np.median(np.stack(fields), axis=0).astype(np.float32)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(3.0, width / 32.0), sigmaY=max(3.0, height / 32.0))
    central = float(np.median(field[radius <= 0.3]))
    if central <= 0:
        return None, 1.0
    field /= central
    base_gain = 1.0 / np.maximum(field, 0.05)
    bias = np.zeros_like(base_gain)

    # A pale vignette is commonly an additive veil rather than a simple brightening. Estimate a
    # local black/white range over a coarse grid so raised black levels can be subtracted as well as
    # dark multiplicative falloff being lifted. Multiple sampled shots make this much less sensitive
    # to the contents of any one frame.
    lows = np.median(np.stack(low_grids), axis=0)
    highs = np.median(np.stack(high_grids), axis=0)
    grid_radius = radial_coordinates(grid_x, grid_y)
    centre_low = float(np.median(lows[grid_radius <= 0.36]))
    centre_high = float(np.median(highs[grid_radius <= 0.36]))
    centre_contrast = centre_high - centre_low
    if centre_contrast >= 12.0:
        local_contrast = np.maximum(highs - lows, 8.0)
        affine_gain = np.clip(centre_contrast / local_contrast, 0.72, 1.38)
        affine_bias = np.clip(centre_low - lows * affine_gain, -42.0, 42.0)
        affine_gain = cv2.resize(affine_gain, (width, height), interpolation=cv2.INTER_CUBIC)
        affine_bias = cv2.resize(affine_bias, (width, height), interpolation=cv2.INTER_CUBIC)
        affine_gain = cv2.GaussianBlur(affine_gain, (0, 0), sigmaX=max(3.0, width / 28.0), sigmaY=max(3.0, height / 28.0))
        affine_bias = cv2.GaussianBlur(affine_bias, (0, 0), sigmaX=max(3.0, width / 28.0), sigmaY=max(3.0, height / 28.0))
        # Blend with the robust low-frequency gain rather than trusting tile statistics outright.
        base_gain = base_gain * 0.35 + affine_gain * 0.65
        bias = affine_bias

    # Only correct the outer picture. This keeps intentional centre-of-frame lighting and grading
    # intact while allowing different edges to have light and dark contamination simultaneously.
    feather = np.clip((radius - 0.48) / 0.50, 0.0, 1.0)
    gain = 1.0 + (base_gain - 1.0) * feather
    bias *= feather
    edge_values = field[(radius >= 0.82) & (radius <= 1.22)]
    edge_ratio = float(np.median(edge_values))
    edge_low = float(np.percentile(edge_values, 15))
    edge_high = float(np.percentile(edge_values, 85))
    correction_strength = np.maximum(np.abs(gain - 1.0), np.abs(bias) / 64.0)
    edge_strength = float(np.percentile(correction_strength[(radius >= 0.82) & (radius <= 1.22)], 70))
    if edge_strength < 0.09:
        return None, edge_ratio
    # Keep automatic correction conservative even when a sampled shot has extreme edge lighting.
    gain = np.clip(gain, 0.72, 1.38)
    bias = np.clip(bias, -42.0, 42.0)
    bounds = (left / full_width, top / full_height, right / full_width, bottom / full_height)
    return VignetteCorrection(gain, bias, bounds, edge_low, edge_high), edge_ratio


def apply_vignette_correction(frame: Any, profile: Any | None) -> Any:
    import cv2
    import numpy as np

    if profile is None:
        return frame
    full_height, full_width = frame.shape[:2]
    left = max(0, min(full_width - 1, int(round(profile.bounds[0] * full_width))))
    top = max(0, min(full_height - 1, int(round(profile.bounds[1] * full_height))))
    right = max(left + 1, min(full_width, int(round(profile.bounds[2] * full_width))))
    bottom = max(top + 1, min(full_height, int(round(profile.bounds[3] * full_height))))
    height, width = bottom - top, right - left
    gain = cv2.resize(profile.gain, (width, height), interpolation=cv2.INTER_LINEAR)
    bias = cv2.resize(profile.bias, (width, height), interpolation=cv2.INTER_LINEAR)
    corrected = frame.copy()
    active = frame[top:bottom, left:right].astype(np.float32) * gain[:, :, None] + bias[:, :, None]
    corrected[top:bottom, left:right] = np.clip(active, 0, 255).astype(np.uint8)
    return corrected


class TorchDeVignetteProcessor:
    """Batched CUDA implementation of the DeVignette correction."""

    def __init__(
        self,
        width: int,
        height: int,
        vignette: VignetteCorrection | None,
        device: str = "cuda",
    ) -> None:
        import cv2
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.width = width
        self.height = height
        self.vignette = vignette

        self.gain = self.bias = self.active_bounds = None
        if vignette is not None:
            left = max(0, min(width - 1, int(round(vignette.bounds[0] * width))))
            top = max(0, min(height - 1, int(round(vignette.bounds[1] * height))))
            right = max(left + 1, min(width, int(round(vignette.bounds[2] * width))))
            bottom = max(top + 1, min(height, int(round(vignette.bounds[3] * height))))
            active_width, active_height = right - left, bottom - top
            gain = cv2.resize(vignette.gain, (active_width, active_height), interpolation=cv2.INTER_LINEAR)
            bias = cv2.resize(vignette.bias, (active_width, active_height), interpolation=cv2.INTER_LINEAR)
            self.gain = torch.from_numpy(gain).to(self.device, dtype=torch.float32)
            self.bias = torch.from_numpy(bias).to(self.device, dtype=torch.float32)
            self.active_bounds = (left, top, right, bottom)

    @classmethod
    def available(cls) -> tuple[bool, str]:
        try:
            import torch

            if not torch.cuda.is_available():
                return False, "PyTorch is installed but CUDA is not available"
            return True, torch.cuda.get_device_name(0)
        except Exception as exc:
            return False, f"PyTorch CUDA could not be loaded: {exc}"

    def suggested_batch_size(self) -> int:
        torch = self.torch
        pixels = self.width * self.height
        cap = 8 if pixels <= 1_000_000 else 4 if pixels <= 2_500_000 else 1
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(self.device)
            per_frame = max(1, pixels * 4 * 8)
            memory_batch = max(1, int((free_bytes * 0.30) // per_frame))
            return max(1, min(cap, memory_batch))
        except Exception:
            return max(1, min(cap, 2))

    def process(self, frames: list[Any]) -> list[Any]:
        import numpy as np

        torch = self.torch
        values = torch.from_numpy(np.stack(frames)).to(self.device, dtype=torch.float32)

        if self.active_bounds is not None and self.gain is not None and self.bias is not None:
            left, top, right, bottom = self.active_bounds
            active = values[:, top:bottom, left:right]
            values[:, top:bottom, left:right] = active * self.gain[None, :, :, None] + self.bias[None, :, :, None]

        output = values.clamp(0, 255).to(torch.uint8).cpu().numpy()
        return [frame for frame in output]


def sample_vignette_frames(source: Path, frame_count: int, width: int, height: int, samples: int = 48) -> list[Any]:
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return []
    analysis_width = min(320, width)
    analysis_height = max(2, int(round(height * analysis_width / max(width, 1))))
    indices = np.linspace(0, max(0, frame_count - 1), min(samples, max(1, frame_count)), dtype=int)
    frames: list[Any] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            frames.append(cv2.resize(frame, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA))
    capture.release()
    return frames


def prepass_signature(source: Path, args: argparse.Namespace, info: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 2,
        "tool": "cleanup_video.py/prepasses",
        "source": root_relative(source),
        "source_fingerprint": file_fingerprint(source),
        "width": int(info["width"]),
        "height": int(info["height"]),
        "fps": float(info.get("fps") or 24.0),
        "frames": int(info.get("frames") or 0),
        "devignette": bool(args.devignette),
        "devignette_version": DEVIGNETTE_VERSION if args.devignette else 0,
    }


def _read_raw_frame(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def format_elapsed(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def prepare_devignette(
    ffmpeg: str,
    source: Path,
    work_dir: Path,
    info: dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    """Stream DeVignette through OpenCV without changing source timing or geometry."""
    import numpy as np

    if not args.devignette:
        return source
    width, height = int(info["width"]), int(info["height"])
    fps = float(info.get("fps") or 24.0)
    frames = int(info.get("frames") or 0)
    signature = prepass_signature(source, args, info)
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    output = work_dir / f"devignette_{key}.mp4"
    if not args.force and resumable_output(output, signature, width=width, height=height, video_like=source):
        print(f"Reuse DeVignette: {output}", flush=True)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.mp4")
    partial.unlink(missing_ok=True)

    analysis_started = time.perf_counter()
    print(
        f"DeVignette analysis: source {width}x{height}, {fps:g} fps, "
        f"{frames or '?'} frames. Sampling the clip...",
        flush=True,
    )
    analysis_frames = sample_vignette_frames(source, frames, width, height, samples=48)

    profile, edge_ratio = vignette_profile(analysis_frames)
    if profile is None:
        print(
            f"DeVignette: no stable vignette detected (edge/centre {edge_ratio:.3f}); "
            "leaving illumination unchanged.",
            flush=True,
        )
    else:
        if profile.edge_low < 0.91 and profile.edge_high > 1.09:
            polarity = "mixed light/dark"
        elif profile.edge_low < 0.91:
            polarity = "dark"
        else:
            polarity = "light"
        print(
            f"DeVignette: detected {polarity} vignette (edge range {profile.edge_low:.3f}-"
            f"{profile.edge_high:.3f}, median {edge_ratio:.3f}); applying bounded edge correction.",
            flush=True,
        )
    print(
        f"DeVignette analysis complete in {format_elapsed(time.perf_counter() - analysis_started)} "
        f"using {len(analysis_frames)} sampled frame(s).",
        flush=True,
    )
    if profile is None:
        return source

    requested_device = str(getattr(args, "repair_device", "auto") or "auto").lower()
    gpu_processor: TorchDeVignetteProcessor | None = None
    batch_size = 1
    available, gpu_detail = (False, "not requested")
    if requested_device in {"auto", "cuda"}:
        available, gpu_detail = TorchDeVignetteProcessor.available()
    if requested_device in {"auto", "cuda"} and available:
        try:
            gpu_processor = TorchDeVignetteProcessor(width, height, profile, device="cuda")
            batch_size = gpu_processor.suggested_batch_size()
            print(
                f"DeVignette processor: CUDA on {gpu_detail}; adaptive batch size {batch_size}. "
                "This uses ARP's PyTorch CUDA stack directly (the same GPU stack used by ComfyUI).",
                flush=True,
            )
        except Exception as exc:
            print(f"WARNING: CUDA DeVignette could not initialize ({exc}); falling back to CPU.", flush=True)
    elif requested_device == "cuda":
        print(f"WARNING: CUDA DeVignette requested but unavailable ({gpu_detail}); falling back to CPU.", flush=True)
    if gpu_processor is None:
        print("DeVignette processor: CPU (OpenCV/NumPy), one frame at a time.", flush=True)

    decoder = subprocess.Popen(
        [ffmpeg, "-v", "error", "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
         "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    encoder = subprocess.Popen(
        [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s:v", f"{width}x{height}",
         "-r", f"{fps:.8f}", "-i", "pipe:0", "-an", "-r", f"{fps:.8f}", "-fps_mode", "cfr",
         "-c:v", "libx264", "-crf", "14", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(partial)],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if decoder.stdout is None or encoder.stdin is None:
        decoder.kill()
        encoder.kill()
        raise RuntimeError("Could not open FFmpeg pipes for DeVignette.")
    frame_bytes = width * height * 3
    processed = 0
    processing_error: BaseException | None = None
    processing_started = time.perf_counter()
    last_report = processing_started
    try:
        while frames <= 0 or processed < frames:
            source_batch: list[Any] = []
            while len(source_batch) < batch_size and (frames <= 0 or processed + len(source_batch) < frames):
                raw = _read_raw_frame(decoder.stdout, frame_bytes)
                if len(raw) != frame_bytes:
                    break
                source_batch.append(np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy())
            if not source_batch:
                break
            if gpu_processor is not None:
                try:
                    repaired_batch = gpu_processor.process(source_batch)
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    print(
                        "WARNING: CUDA ran out of memory; clearing its cache and continuing on CPU.",
                        flush=True,
                    )
                    gpu_processor.torch.cuda.empty_cache()
                    gpu_processor = None
                    batch_size = 1
                    repaired_batch = source_batch
            else:
                repaired_batch = source_batch
            if gpu_processor is None:
                repaired_batch = [apply_vignette_correction(frame, profile) for frame in repaired_batch]
            for frame in repaired_batch:
                encoder.stdin.write(frame.tobytes())
            processed += len(repaired_batch)
            now = time.perf_counter()
            complete = bool(frames and processed >= frames)
            if complete or now - last_report >= 2.0:
                elapsed = now - processing_started
                rate = processed / max(elapsed, 1e-6)
                eta = (frames - processed) / rate if frames and rate > 0 else None
                percent = f"{processed / frames * 100:5.1f}%" if frames else "  ?.?%"
                backend = "CUDA" if gpu_processor is not None else "CPU"
                print(
                    f"DeVignette: {processed:,}/{frames or '?'} frames ({percent}) | "
                    f"{rate:.2f} fps | elapsed {format_elapsed(elapsed)} | ETA {format_elapsed(eta)} | {backend}",
                    flush=True,
                )
                last_report = now
    except BaseException as exc:
        processing_error = exc
    finally:
        try:
            decoder.stdout.close()
        except OSError:
            pass
        try:
            encoder.stdin.close()
        except OSError:
            pass

    decoder_error = (decoder.stderr.read() if decoder.stderr else b"").decode("utf-8", errors="replace").strip()
    encoder_error = (encoder.stderr.read() if encoder.stderr else b"").decode("utf-8", errors="replace").strip()
    if decoder.stderr:
        decoder.stderr.close()
    if encoder.stderr:
        encoder.stderr.close()
    decoder_code = decoder.wait()
    encoder_code = encoder.wait()
    if processing_error is not None:
        partial.unlink(missing_ok=True)
        raise processing_error
    if decoder_code or encoder_code or not partial.exists() or processed == 0:
        partial.unlink(missing_ok=True)
        detail = encoder_error or decoder_error or "FFmpeg DeVignette pass failed."
        raise RuntimeError(detail)
    replace_with_retry(partial, output, "DeVignette")
    write_signature(output, signature)
    total_elapsed = time.perf_counter() - processing_started
    print(
        f"DeVignette complete: {processed:,} frames in {format_elapsed(total_elapsed)} "
        f"({processed / max(total_elapsed, 1e-6):.2f} fps).",
        flush=True,
    )
    return output


def ai_descratch_dimensions(info: dict[str, Any], requested_height: str | int) -> tuple[int, int]:
    """Return an aspect-preserving, ProPainter-safe working size divisible by eight."""
    source_width, source_height = int(info["width"]), int(info["height"])
    if str(requested_height).lower() == "source":
        height = source_height
    else:
        height = min(source_height, max(64, int(requested_height)))
    height = max(8, height - height % 8)
    width = max(8, int(round(source_width * height / max(source_height, 1))))
    width -= width % 8
    return width, height


def ai_scratch_mask(frame: Any, sensitivity: float = 0.65) -> Any:
    """Detect long, narrow light/dark damage without granting an AI access to the full frame.

    The detector looks for horizontal luminance outliers which remain coherent through a tall
    vertical window. Genuine scene edges are usually wider, change direction, or terminate; film
    scratches remain narrow and vertically persistent. A per-frame robust threshold adapts to
    grain and contrast, while a hard coverage cap prevents a pathological frame from becoming a
    full-frame generative request.
    """
    import cv2
    import numpy as np

    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = gray.shape
    horizontal_base = cv2.GaussianBlur(gray, (9, 1), sigmaX=2.1, sigmaY=0)
    detail = gray - horizontal_base
    vertical_window = max(15, min(101, (height // 10) | 1))
    coherent = cv2.GaussianBlur(
        detail, (1, vertical_window), sigmaX=0, sigmaY=max(2.0, vertical_window / 5.0)
    )
    strength = np.abs(coherent)
    centre = float(np.median(strength))
    mad = float(np.median(np.abs(strength - centre)))
    # 0.0 is intentionally conservative; 1.0 admits faint scratches. The absolute floor avoids
    # selecting ordinary grain in very flat material.
    sigma = 5.2 - 3.4 * sensitivity
    threshold = max(0.85, centre + sigma * max(mad, 0.12))
    detail_floor = max(1.5, float(np.percentile(np.abs(detail), 45)) * (0.72 - 0.22 * sensitivity))
    mask = ((strength >= threshold) & (np.abs(detail) >= detail_floor)).astype(np.uint8) * 255

    # Require vertical continuity, reconnect short gaps, and reject broad blocks that are much
    # more likely to be picture content than a scratch.
    open_length = max(5, min(19, (height // 72) | 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_length, 1), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((max(3, open_length // 2), 1), np.uint8))
    components, labels, stats, _centres = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    # At 720p a film scratch is normally only a handful of pixels wide. A much broader component
    # is usually a person, column, curtain fold, or picture edge and must never become an AI mask.
    # Gaussian response extends a one-pixel line several pixels to either side, so this is the
    # detector-response width rather than the literal scratch width.
    max_width = max(10, min(13, int(round(width * 0.009))))
    min_height = max(12, int(round(height * 0.07)))
    for label in range(1, components):
        x, y, component_width, component_height, area = stats[label]
        if (
            component_width <= max_width
            and component_height >= min_height
            and component_height >= component_width * 7
            and area >= min_height
        ):
            clean[labels == label] = 255

    # Whole-height column evidence catches scratches interrupted by similarly toned picture areas.
    column = np.percentile(strength, 70, axis=0)
    col_centre = float(np.median(column))
    col_mad = float(np.median(np.abs(column - col_centre)))
    column_threshold = max(threshold * 0.72, col_centre + (4.8 - 2.8 * sensitivity) * max(col_mad, 0.12))
    candidates = column >= column_threshold
    for start, end in _contiguous_true_runs(candidates):
        if end - start <= max_width:
            clean[:, start:end] = 255

    # Do not let presentation bars or a detector failure turn into invented imagery.
    picture = gray > max(4.0, float(np.percentile(gray, 92)) * 0.025)
    clean[picture == 0] = 0
    coverage = float(np.count_nonzero(clean)) / max(clean.size, 1)
    max_coverage = 0.28
    if coverage > max_coverage:
        selected_strength = strength[clean > 0]
        cutoff = float(np.quantile(selected_strength, 1.0 - max_coverage / coverage))
        clean[(clean > 0) & (strength < cutoff)] = 0
    return clean


def ai_mask_signature(source: Path, args: argparse.Namespace, width: int, height: int) -> dict[str, Any]:
    return {
        "version": AI_DESCRATCH_VERSION,
        "tool": "cleanup_video.py/ai_scratch_mask",
        "source": root_relative(source),
        "source_fingerprint": file_fingerprint(source),
        "width": width,
        "height": height,
        "sensitivity": float(args.scratch_sensitivity),
    }


def prepare_ai_descratch_input(
    ffmpeg: str,
    source: Path,
    work_dir: Path,
    info: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Path, Path, int, int]:
    """Create matching ProPainter input and binary-mask videos at the requested working size."""
    import cv2
    import numpy as np

    width, height = ai_descratch_dimensions(info, args.ai_descratch_height)
    fps = float(info.get("fps") or 24.0)
    frames = int(info.get("frames") or 0)
    prepared = work_dir / f"ai_input_{width}x{height}.mp4"
    prepared_sig = processing_signature(source, width, height, fps, frames)
    if args.force or not resumable_output(prepared, prepared_sig, width=width, height=height, video_like=source):
        prepared.parent.mkdir(parents=True, exist_ok=True)
        partial = prepared.with_suffix(".partial.mp4")
        partial.unlink(missing_ok=True)
        vf = (
            f"scale={width}:{height}:flags=lanczos,trim=end_frame={frames},"
            f"setpts=N/({fps:.8f}*TB),fps={fps:.8f},setsar=1"
        )
        subprocess.run(
            [
                ffmpeg, "-y", "-i", str(source), "-vf", vf, "-an", "-r", f"{fps:.8f}",
                "-fps_mode", "cfr", "-c:v", "libx264", "-crf", "10", "-preset", "veryfast",
                str(partial),
            ],
            check=True,
        )
        replace_with_retry(partial, prepared, "AI DeScratch input")
        write_signature(prepared, prepared_sig)

    mask = work_dir / f"ai_scratch_mask_{width}x{height}.mp4"
    signature = ai_mask_signature(source, args, width, height)
    if not args.force and resumable_output(mask, signature, width=width, height=height, video_like=prepared):
        print(f"Reuse AI DeScratch mask: {mask}", flush=True)
        return prepared, mask, width, height

    partial_mask = mask.with_suffix(".partial.mp4")
    partial_mask.unlink(missing_ok=True)
    capture = cv2.VideoCapture(str(prepared))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open AI DeScratch input: {prepared}")
    encoder = subprocess.Popen(
        [
            ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", f"{fps:.8f}", "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "0",
            "-preset", "veryfast", "-pix_fmt", "yuv420p", str(partial_mask),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if encoder.stdin is None:
        capture.release()
        raise RuntimeError("Could not open FFmpeg mask encoder.")
    processed = selected = 0
    started = time.perf_counter()
    last_report = started
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_mask = ai_scratch_mask(frame, args.scratch_sensitivity)
            selected += int(np.count_nonzero(frame_mask))
            mask_bgr = cv2.cvtColor(frame_mask, cv2.COLOR_GRAY2BGR)
            encoder.stdin.write(mask_bgr.tobytes())
            processed += 1
            now = time.perf_counter()
            if now - last_report >= 2.0 or (frames and processed >= frames):
                coverage = selected / max(processed * width * height, 1)
                print(
                    f"AI DeScratch mask: {processed:,}/{frames or '?'} frames | "
                    f"{coverage:.1%} average coverage | {processed / max(now - started, 1e-6):.2f} fps",
                    flush=True,
                )
                last_report = now
    finally:
        capture.release()
        try:
            encoder.stdin.close()
        except OSError:
            pass
    error = (encoder.stderr.read() if encoder.stderr else b"").decode("utf-8", errors="replace")
    if encoder.stderr:
        encoder.stderr.close()
    code = encoder.wait()
    if code or processed == 0 or not partial_mask.exists():
        partial_mask.unlink(missing_ok=True)
        raise RuntimeError(error.strip() or "AI DeScratch mask encoding failed.")
    replace_with_retry(partial_mask, mask, "AI DeScratch mask")
    write_signature(mask, signature)
    print(
        f"AI DeScratch mask complete: {processed:,} frames, "
        f"{selected / max(processed * width * height, 1):.1%} average coverage.",
        flush=True,
    )
    return prepared, mask, width, height


def propainter_prompt(
    video: Path,
    mask: Path,
    width: int,
    height: int,
    fps: float,
    output_prefix: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    loader_defaults = {
        "force_rate": 0,
        "custom_width": 0,
        "custom_height": 0,
        "frame_load_cap": 0,
        "skip_first_frames": 0,
        "select_every_nth": 1,
    }
    return {
        "1": {
            "class_type": "VHS_LoadVideoPath",
            "inputs": {"video": str(video), **loader_defaults},
        },
        "2": {
            "class_type": "VHS_LoadVideoPath",
            "inputs": {"video": str(mask), **loader_defaults},
        },
        "3": {
            "class_type": "ImageToMask",
            "inputs": {"image": ["2", 0], "channel": "red"},
        },
        "4": {
            "class_type": "ProPainterInpaint",
            "inputs": {
                "image": ["1", 0],
                "mask": ["3", 0],
                "width": width,
                "height": height,
                "mask_dilates": int(args.scratch_mask_dilate),
                "flow_mask_dilates": max(4, int(args.scratch_mask_dilate) + 4),
                "ref_stride": 10,
                "neighbor_length": 10,
                "subvideo_length": min(49, int(args.ai_chunk_frames)),
                "raft_iter": 20,
                "fp16": "enable",
            },
        },
        "5": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["4", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": output_prefix,
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
                "pix_fmt": "yuv420p",
                "crf": 10,
                "save_metadata": False,
            },
        },
    }


def run_propainter_chunk(
    comfy_url: str,
    comfy_output_root: Path,
    video: Path,
    mask: Path,
    output: Path,
    width: int,
    height: int,
    fps: float,
    args: argparse.Namespace,
    output_prefix: str,
) -> None:
    prompt = propainter_prompt(video, mask, width, height, fps, output_prefix, args)
    prompt_id = queue_prompt(comfy_url, prompt)
    print(f"Queued AI DeScratch ProPainter job {prompt_id}.", flush=True)
    history = wait_for_prompt(comfy_url, prompt_id, args.poll_seconds)
    candidates = [
        path for path in extract_output_files(history, comfy_output_root)
        if path.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm"} and path.is_file()
    ]
    if not candidates:
        raise RuntimeError(f"ProPainter prompt {prompt_id} completed without a video output.")
    generated = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.mp4")
    partial.unlink(missing_ok=True)
    shutil.copy2(generated, partial)
    replace_with_retry(partial, output, "AI DeScratch ProPainter chunk")


def composite_ai_descratch(
    ffmpeg: str,
    source: Path,
    repaired: Path,
    mask: Path,
    output: Path,
    info: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Composite only the dilated scratch mask, keeping every other source pixel untouched."""
    import cv2
    import numpy as np

    width, height = int(info["width"]), int(info["height"])
    fps = float(info.get("fps") or 24.0)
    frames = int(info.get("frames") or 0)
    source_capture = cv2.VideoCapture(str(source))
    repair_capture = cv2.VideoCapture(str(repaired))
    mask_capture = cv2.VideoCapture(str(mask))
    if not all(capture.isOpened() for capture in (source_capture, repair_capture, mask_capture)):
        for capture in (source_capture, repair_capture, mask_capture):
            capture.release()
        raise RuntimeError("Could not open all AI DeScratch composite inputs.")
    partial = output.with_suffix(".partial.mp4")
    partial.unlink(missing_ok=True)
    encoder = subprocess.Popen(
        [
            ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", f"{fps:.8f}", "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "10",
            "-preset", "veryfast", "-pix_fmt", "yuv420p", str(partial),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if encoder.stdin is None:
        raise RuntimeError("Could not open AI DeScratch composite encoder.")
    processed = 0
    started = time.perf_counter()
    last_report = started
    try:
        while True:
            source_ok, source_frame = source_capture.read()
            repair_ok, repaired_frame = repair_capture.read()
            mask_ok, mask_frame = mask_capture.read()
            if not source_ok:
                break
            if not repair_ok or not mask_ok:
                raise RuntimeError(f"AI DeScratch repair/mask ended early at frame {processed}.")
            repaired_frame = cv2.resize(repaired_frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
            mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
            if args.scratch_mask_dilate:
                mask_gray = cv2.dilate(
                    mask_gray,
                    np.ones((3, 3), np.uint8),
                    iterations=int(args.scratch_mask_dilate),
                )
            mask_gray = cv2.resize(mask_gray, (width, height), interpolation=cv2.INTER_LINEAR)
            alpha = cv2.GaussianBlur(mask_gray.astype(np.float32) / 255.0, (0, 0), sigmaX=0.7)
            alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
            composed = (
                source_frame.astype(np.float32) * (1.0 - alpha)
                + repaired_frame.astype(np.float32) * alpha
            )
            encoder.stdin.write(np.clip(composed, 0, 255).astype(np.uint8).tobytes())
            processed += 1
            now = time.perf_counter()
            if now - last_report >= 2.0 or (frames and processed >= frames):
                print(
                    f"AI DeScratch composite: {processed:,}/{frames or '?'} frames | "
                    f"{processed / max(now - started, 1e-6):.2f} fps",
                    flush=True,
                )
                last_report = now
    finally:
        for capture in (source_capture, repair_capture, mask_capture):
            capture.release()
        try:
            encoder.stdin.close()
        except OSError:
            pass
    error = (encoder.stderr.read() if encoder.stderr else b"").decode("utf-8", errors="replace")
    if encoder.stderr:
        encoder.stderr.close()
    code = encoder.wait()
    if code or processed == 0 or not partial.exists():
        partial.unlink(missing_ok=True)
        raise RuntimeError(error.strip() or "AI DeScratch composite failed.")
    replace_with_retry(partial, output, "AI DeScratch composite")


def prepare_ai_descratch(
    ffmpeg: str,
    source: Path,
    work_dir: Path,
    info: dict[str, Any],
    args: argparse.Namespace,
    comfy_dir: Path,
    comfy_output_root: Path,
) -> tuple[Path, Path]:
    """Run masked ProPainter in chunks and return the source-sized composite plus its mask."""
    prepared, mask, width, height = prepare_ai_descratch_input(ffmpeg, source, work_dir, info, args)
    fps = float(info.get("fps") or 24.0)
    total_frames = int(info.get("frames") or 0)
    overlap = min(8, max(0, int(args.ai_chunk_frames) - 2))
    ranges = chunk_ranges(prepared, int(args.ai_chunk_frames) / max(fps, 0.001), overlap)
    print(
        f"AI DeScratch: ProPainter at {width}x{height}, {len(ranges)} chunk(s) of up to "
        f"{int(args.ai_chunk_frames)} frames with {overlap} overlap; sensitivity "
        f"{float(args.scratch_sensitivity):.2f}, mask expansion {int(args.scratch_mask_dilate)} px.",
        flush=True,
    )
    wait_for_comfy(args.comfy_url, timeout_seconds=180, poll_seconds=args.poll_seconds)
    ensure_node_types(args.comfy_url, AI_DESCRATCH_NODES, "AI DeScratch", comfy_dir)
    raw_chunks: list[Path] = []
    for index, start, end in ranges:
        video_chunk = work_dir / f"ai_video_{index:04d}_{start:06d}_{end:06d}.mp4"
        mask_chunk = work_dir / f"ai_mask_{index:04d}_{start:06d}_{end:06d}.mp4"
        raw = work_dir / f"ai_raw_{index:04d}_{start:06d}_{end:06d}.mp4"
        normalized = work_dir / f"ai_normalized_{index:04d}_{start:06d}_{end:06d}.mp4"
        split_chunk(ffmpeg, prepared, video_chunk, start, end, fps, args.force)
        split_chunk(ffmpeg, mask, mask_chunk, start, end, fps, args.force)
        expected = end - start
        chunk_sig = {
            "v": AI_DESCRATCH_VERSION,
            "source": file_fingerprint(video_chunk),
            "mask": file_fingerprint(mask_chunk),
            "mask_dilate": int(args.scratch_mask_dilate),
            "frames": int(args.ai_chunk_frames),
        }
        if args.force or not resumable_output(raw, chunk_sig, width=width, height=height):
            run_propainter_chunk(
                args.comfy_url,
                comfy_output_root,
                video_chunk,
                mask_chunk,
                raw,
                width,
                height,
                fps,
                args,
                f"arp_ai_descratch/{safe_stem(source.name)}_{index:04d}",
            )
            write_signature(raw, chunk_sig)
        normalize_chunk(ffmpeg, raw, normalized, width, height, fps, expected)
        raw_chunks.append(normalized)
        print(f"AI DeScratch chunk {index + 1}/{len(ranges)} complete.", flush=True)
    stitched = work_dir / f"ai_propainter_stitched_v{AI_DESCRATCH_VERSION}_{width}x{height}.mp4"
    stitch_chunks(ffmpeg, raw_chunks, ranges, stitched, fps, args.force)
    if int(probe_video(stitched).get("frames") or 0) < total_frames:
        raise RuntimeError("AI DeScratch stitched output is shorter than the source.")
    composite = work_dir / "ai_descratch_composite.mp4"
    composite_ai_descratch(ffmpeg, source, stitched, mask, composite, info, args)
    return composite, mask


def save_scratch_mask_preview(
    ffmpeg: str,
    mask: Path,
    output: Path,
    info: dict[str, Any],
) -> Path:
    width, height = int(info["width"]), int(info["height"])
    fps = float(info.get("fps") or 24.0)
    frames = int(info.get("frames") or 0)
    preview = output.with_name(f"{output.stem}_scratch_mask.mp4")
    partial = preview.with_suffix(".partial.mp4")
    vf = (
        f"scale={width}:{height}:flags=neighbor,trim=end_frame={frames},"
        f"setpts=N/({fps:.8f}*TB),fps={fps:.8f},setsar=1"
    )
    subprocess.run(
        [
            ffmpeg, "-y", "-i", str(mask), "-vf", vf, "-an", "-r", f"{fps:.8f}",
            "-fps_mode", "cfr", "-c:v", "libx264", "-crf", "12", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", str(partial),
        ],
        check=True,
    )
    replace_with_retry(partial, preview, "AI DeScratch mask preview")
    print(f"Wrote AI DeScratch mask preview: {preview}", flush=True)
    return preview


def processing_signature(source: Path, width: int, height: int, fps: float, frames: int) -> dict[str, Any]:
    return {
        "version": 1,
        "tool": "cleanup_video.py/model_input",
        "source": root_relative(source),
        "source_fingerprint": file_fingerprint(source),
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
    }


def prepare_model_input(ffmpeg: str, source: Path, work_dir: Path, info: dict[str, Any], processing_height: int | str, force: bool) -> Path:
    width, height = model_dimensions(int(info["width"]), int(info["height"]), processing_height)
    fps = float(info.get("fps") or 24.0)
    frames = int(info.get("frames") or 0)
    output = work_dir / f"model_input_{width}x{height}.mp4"
    sig = processing_signature(source, width, height, fps, frames)
    if not force and resumable_output(output, sig, width=width, height=height, video_like=source):
        print(f"Reuse Clean Up model input: {output}", flush=True)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    vf = (
        f"scale={width}:{height}:flags=lanczos,"
        f"trim=end_frame={frames},setpts=N/({fps:.8f}*TB),fps={fps:.8f},setsar=1"
    )
    subprocess.run(
        [ffmpeg, "-y", "-i", str(source), "-vf", vf, "-an", "-r", f"{fps:.8f}",
         "-fps_mode", "cfr", "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", str(partial)],
        check=True,
    )
    replace_with_retry(partial, output, "Clean Up model input")
    write_signature(output, sig)
    print(f"Prepared Clean Up model input: {output} ({width}x{height} internally)", flush=True)
    return output


def normalize_chunk(ffmpeg: str, source: Path, output: Path, width: int, height: int, fps: float, frames: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    duration = max(1, frames) / max(fps, 0.001)
    vf = (
        f"scale={width}:{height}:flags=lanczos,trim=end_frame={frames},"
        f"setpts=N/({fps:.8f}*TB),tpad=stop_mode=clone:stop_duration={duration:.8f},"
        f"trim=end_frame={frames},fps={fps:.8f},setsar=1"
    )
    subprocess.run(
        [ffmpeg, "-y", "-i", str(source), "-vf", vf, "-an", "-r", f"{fps:.8f}",
         "-fps_mode", "cfr", "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", str(partial)],
        check=True,
    )
    replace_with_retry(partial, output, f"Clean Up chunk {output.name}")


def probe_sample_aspect_ratio(ffmpeg: str, source: Path) -> str:
    ffmpeg_path = Path(ffmpeg)
    ffprobe_name = "ffprobe.exe" if ffmpeg_path.suffix.lower() == ".exe" else "ffprobe"
    ffprobe = str(ffmpeg_path.with_name(ffprobe_name)) if ffmpeg_path.parent != Path(".") else ffprobe_name
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=sample_aspect_ratio", "-of", "default=noprint_wrappers=1:nokey=1", str(source)],
            check=True,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        numerator, denominator = value.split(":", 1)
        if int(numerator) > 0 and int(denominator) > 0:
            return f"{int(numerator)}/{int(denominator)}"
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError):
        pass
    return "1"


def first_audio_packet_stream(ffmpeg: str, source: Path) -> int | None:
    """Return the first audio stream that actually contains a packet.

    Optional FFmpeg maps still create an output stream when a container advertises an empty audio
    track.  Mapping every track also makes malformed multichannel layouts capable of aborting the
    completed video encode.  Probe one real packet and map only that stream during delivery.
    """
    ffmpeg_path = Path(ffmpeg)
    ffprobe_name = "ffprobe.exe" if ffmpeg_path.suffix.lower() == ".exe" else "ffprobe"
    ffprobe = str(ffmpeg_path.with_name(ffprobe_name)) if ffmpeg_path.parent != Path(".") else ffprobe_name
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-read_intervals", "%+#1", "-select_streams", "a",
             "-show_entries", "packet=stream_index", "-of", "json", str(source)],
            check=True,
            capture_output=True,
            text=True,
        )
        packets = json.loads(result.stdout or "{}").get("packets") or []
        if packets and packets[0].get("stream_index") is not None:
            return int(packets[0]["stream_index"])
    except (FileNotFoundError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError):
        pass
    return None


def finalize_output(
    ffmpeg: str,
    generated: Path,
    source: Path,
    output: Path,
    info: dict[str, Any],
    output_dimensions: tuple[int, int] | None = None,
) -> None:
    width, height = output_dimensions or (int(info["width"]), int(info["height"]))
    fps = float(info.get("fps") or 24.0)
    frames = int(info.get("frames") or 0)
    duration = max(1, frames) / max(fps, 0.001)
    sample_aspect_ratio = probe_sample_aspect_ratio(ffmpeg, source)
    audio_stream = first_audio_packet_stream(ffmpeg, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    vf = (
        f"scale={width}:{height}:flags=lanczos,trim=end_frame={frames},"
        f"setpts=N/({fps:.8f}*TB),tpad=stop_mode=clone:stop_duration={duration:.8f},"
        f"trim=end_frame={frames},fps={fps:.8f},setsar={sample_aspect_ratio}:max=100000"
    )
    command = [
        ffmpeg, "-y", "-i", str(generated), "-i", str(source), "-filter:v", vf,
        "-map", "0:v:0", "-r", f"{fps:.8f}", "-fps_mode", "cfr",
        "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p",
    ]
    if audio_stream is None:
        command.append("-an")
    else:
        command.extend([
            "-map", f"1:{audio_stream}", "-af", "aresample=async=1:first_pts=0,apad",
            "-ac", "2", "-c:a", "aac", "-b:a", "192k", "-shortest",
        ])
    command.extend(["-movflags", "+faststart", str(partial)])
    subprocess.run(command, check=True)
    replace_with_retry(partial, output, "Clean Up output")


def run_signature(source: Path, workflow: Path, args: argparse.Namespace, info: dict[str, Any]) -> dict[str, Any]:
    signature = {
        "version": 6,
        "tool": "cleanup_video.py",
        "source": root_relative(source),
        "source_fingerprint": file_fingerprint(source),
        "identity": cleanup_identity(source, args),
        "source_geometry": {
            "width": int(info["width"]), "height": int(info["height"]),
            "fps": float(info.get("fps") or 24.0), "frames": int(info.get("frames") or 0),
        },
        "ai_descratch_version": AI_DESCRATCH_VERSION if args.ai_descratch else 0,
        "devignette_version": DEVIGNETTE_VERSION if args.devignette else 0,
        "chunk_seconds": args.chunk_seconds,
        "overlap_frames": args.overlap_frames,
    }
    if args.dearchive:
        signature.update({
            "workflow": root_relative(workflow),
            "workflow_fingerprint": file_fingerprint(workflow),
            "model_backend": args.model_backend,
            "gguf_model": args.gguf_model,
            "video_vae": args.video_vae,
            "text_encoder": args.text_encoder,
        })
    return signature


def chunk_signature(run_sig: dict[str, Any], prepared: Path, index: int, start: int, end: int, seed: int) -> dict[str, Any]:
    return {
        "version": 1,
        "tool": "cleanup_video.py/chunk",
        "run": run_sig,
        "prepared": root_relative(prepared),
        "prepared_fingerprint": file_fingerprint(prepared),
        "chunk": index,
        "start_frame": start,
        "end_frame": end,
        "seed": seed,
    }


def parse_chunk_seconds(value: str) -> float:
    """Parse the user-facing duration while keeping Dearchive jobs in a useful range."""
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not MIN_CHUNK_SECONDS <= seconds <= MAX_CHUNK_SECONDS:
        raise argparse.ArgumentTypeError(
            f"must be between {MIN_CHUNK_SECONDS:g} and {MAX_CHUNK_SECONDS:g} seconds"
        )
    return seconds


def parse_source_fidelity(value: str) -> float:
    try:
        fidelity = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not 0.0 <= fidelity <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return fidelity


def dearchive_chunk_frames(chunk_seconds: float, fps: float) -> int:
    """Round a requested duration to the nearest LTX-valid ``8n + 1`` frame count.

    Ties are rounded down so the default remains the model author's 97-frame window at
    frame rates where the requested duration falls exactly between two valid sizes.
    """
    requested_frames = max(1.0, float(chunk_seconds) * max(float(fps), 0.001))
    multiple = max(0, math.floor(((requested_frames - 1.0) / 8.0) + 0.5 - 1e-12))
    return multiple * 8 + 1


def dearchive_chunk_ranges(
    prepared: Path, chunk_seconds: float, overlap_frames: int
) -> tuple[list[tuple[int, int, int]], int, float]:
    """Return source ranges using the quantized Dearchive frame window."""
    info = probe_video(prepared)
    fps = float(info.get("fps") or 24.0)
    chunk_frames = dearchive_chunk_frames(chunk_seconds, fps)
    effective_seconds = chunk_frames / max(fps, 0.001)
    return chunk_ranges(prepared, effective_seconds, overlap_frames), chunk_frames, effective_seconds


def build_parser() -> argparse.ArgumentParser:
    config = load_local_config()
    parser = argparse.ArgumentParser(description="Clean archive footage with the LTX 2.3 Dearchive IC-LoRA.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output")
    parser.add_argument("--workflow", default=str(DEFAULT_WORKFLOW))
    parser.add_argument(
        "--ai-descratch",
        action="store_true",
        help="Repair a detected scratch mask with ProPainter before Dearchive (non-commercial model licence).",
    )
    parser.add_argument(
        "--scratch-sensitivity",
        type=parse_source_fidelity,
        default=0.65,
        help="Scratch-mask sensitivity from 0 (conservative) to 1 (aggressive).",
    )
    parser.add_argument(
        "--scratch-mask-dilate",
        type=int,
        choices=range(0, 13),
        default=3,
        help="Working-resolution pixels added around detected scratches before ProPainter.",
    )
    parser.add_argument(
        "--ai-descratch-height",
        choices=["540", "720", "1080", "source"],
        default="720",
        help="ProPainter working height. The composite is returned at source resolution.",
    )
    parser.add_argument(
        "--ai-chunk-frames",
        type=int,
        choices=[25, 33, 41, 49],
        default=41,
        help="Frames per ProPainter window; larger values use more VRAM.",
    )
    parser.add_argument(
        "--save-scratch-mask",
        action="store_true",
        help="Write a companion mask-preview video beside the Clean Up output.",
    )
    parser.add_argument("--devignette", action="store_true", help="Automatically correct a stable light or dark vignette before Dearchive.")
    parser.add_argument(
        "--repair-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Processor for DeVignette. Auto prefers PyTorch CUDA and falls back to CPU.",
    )
    dearchive_group = parser.add_mutually_exclusive_group()
    dearchive_group.add_argument("--dearchive", dest="dearchive", action="store_true", help="Run the Dearchive LoRA (default).")
    dearchive_group.add_argument(
        "--no-dearchive", dest="dearchive", action="store_false",
        help="Skip Dearchive and deliver only the selected DeVignette/AI DeScratch repairs.",
    )
    parser.set_defaults(dearchive=True)
    parser.add_argument(
        "--chunk-seconds",
        type=parse_chunk_seconds,
        default=DEFAULT_CHUNK_SECONDS,
        metavar="SECONDS",
        help="Dearchive chunk duration from 2 to 20 seconds; rounded to the nearest 8n+1 frames.",
    )
    parser.add_argument("--overlap-frames", type=int, default=8)
    parser.add_argument(
        "--dearchive-height",
        choices=["540", "720", "1080", "source"],
        default="720",
        help="Dearchive processing and output height. Values are rounded to an LTX-safe multiple of 64.",
    )
    parser.add_argument(
        "--source-fidelity",
        type=parse_source_fidelity,
        default=1.0,
        help="Strength of the complete source-video IC-LoRA guide. Lower values permit more repainting.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--model-backend", choices=["gguf", "checkpoint"], default="gguf")
    parser.add_argument("--gguf-model", default="LTX-2.3-distilled-Q4_K_M.gguf")
    parser.add_argument("--video-vae", default="LTX23_video_vae_bf16.safetensors")
    parser.add_argument("--audio-vae-checkpoint", default="ltx-2.3-22b-dev-fp8.safetensors")
    parser.add_argument("--text-encoder", default="gemma_3_12B_it_fp8_scaled.safetensors")
    parser.add_argument("--text-encoder-checkpoint", default="ltx-2.3-22b-dev-fp8.safetensors")
    parser.add_argument("--cleanup-lora", default="ltx-2.3-dearchive-lora.safetensors")
    parser.add_argument("--comfy-dir", default=config.get("comfy_dir", str(DEFAULT_COMFY_DIR)))
    parser.add_argument("--comfy-url", default=config.get("comfy_url", "http://127.0.0.1:8188"))
    parser.add_argument("--comfy-output-root")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    # Stable node/widget IDs in the bundled author workflow; consumed by outpaint_video.patch_workflow.
    parser.set_defaults(
        load_video_node_id="5060", video_widget="video", save_node_id="5076",
        extra_save_node_id=[], save_prefix_widget="filename_prefix", output_node_id="5076",
        positive_node_id="2483", negative_node_id="2612", prompt_widget="0",
        seed_node_id="4832", seed_widget="0", outpaint_lora="ltx-2.3-dearchive-lora.safetensors",
        guide_strength=0.7,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    # patch_workflow uses the historical outpaint_lora attribute name internally.
    args.outpaint_lora = args.cleanup_lora
    source = resolve_path(args.source)
    workflow_path = resolve_path(args.workflow)
    comfy_dir = resolve_path(args.comfy_dir)
    comfy_output_root = resolve_path(args.comfy_output_root) if args.comfy_output_root else comfy_dir / "output"
    if not source.is_file():
        raise FileNotFoundError(f"Clean Up source video not found: {source}")
    if not (args.ai_descratch or args.devignette or args.dearchive):
        raise ValueError("Select at least one Clean Up operation: AI DeScratch, DeVignette, or Dearchive.")
    if args.dearchive and not workflow_path.is_file():
        raise FileNotFoundError(f"Clean Up workflow not found: {workflow_path}")
    info = probe_video(source)
    output = resolve_path(args.output) if args.output else default_output(source, args)
    sig = run_signature(source, workflow_path, args, info)
    delivery_width, delivery_height = (
        model_dimensions(int(info["width"]), int(info["height"]), args.dearchive_height)
        if args.dearchive else (int(info["width"]), int(info["height"]))
    )
    if not args.force and resumable_output(
        output, sig, width=delivery_width, height=delivery_height, video_like=source
    ):
        print(f"Reuse Clean Up output: {output}", flush=True)
        return 0
    if args.dry_run:
        print(
            f"Would clean {source} -> {output} at "
            f"{delivery_width}x{delivery_height} and {float(info.get('fps') or 24.0):g} fps",
            flush=True,
        )
        return 0
    ffmpeg = find_ffmpeg()
    identity_key = aid.artifact_key(cleanup_identity(source, args))
    work_dir = ROOT / ".cache" / "cleanup_chunks" / f"{safe_stem(source.name)}_{identity_key}"
    repaired_source = prepare_devignette(ffmpeg, source, work_dir, info, args)
    scratch_mask_preview: Path | None = None
    if args.ai_descratch:
        if not (comfy_dir / "main.py").is_file():
            raise FileNotFoundError(f"ComfyUI main.py not found: {comfy_dir / 'main.py'}")
        repaired_source, scratch_mask_preview = prepare_ai_descratch(
            ffmpeg,
            repaired_source,
            work_dir,
            info,
            args,
            comfy_dir,
            comfy_output_root,
        )
    if not args.dearchive:
        finalize_output(ffmpeg, repaired_source, source, output, info)
        if args.save_scratch_mask and scratch_mask_preview is not None:
            save_scratch_mask_preview(ffmpeg, scratch_mask_preview, output, info)
        write_signature(output, sig)
        print(f"Wrote Clean Up video: {output}", flush=True)
        return 0

    if not (comfy_dir / "main.py").is_file():
        raise FileNotFoundError(f"ComfyUI main.py not found: {comfy_dir / 'main.py'}")
    ensure_cleanup_models(comfy_dir)
    wait_for_comfy(args.comfy_url, timeout_seconds=180, poll_seconds=args.poll_seconds)
    required = dict(REQUIRED_NODES)
    if args.model_backend == "gguf":
        required["UnetLoaderGGUF"] = "ComfyUI-GGUF"
    ensure_node_types(args.comfy_url, required, "Clean Up workflow", comfy_dir)
    print(
        f"Dearchive source fidelity: {args.source_fidelity:.2f} "
        f"({'exact source control' if args.source_fidelity == 1.0 else 'weakened control; LTX may repaint source detail'}).",
        flush=True,
    )

    model_input = prepare_model_input(
        ffmpeg, repaired_source, work_dir, info, args.dearchive_height, args.force
    )
    model_info = probe_video(model_input)
    fps = float(model_info.get("fps") or info.get("fps") or 24.0)
    width, height = int(model_info["width"]), int(model_info["height"])
    ranges, chunk_frames, effective_chunk_seconds = dearchive_chunk_ranges(
        model_input, args.chunk_seconds, args.overlap_frames
    )
    print(
        f"Clean Up: requested {args.chunk_seconds:g}s chunks; using {chunk_frames} frames "
        f"({effective_chunk_seconds:.3f}s at {fps:g} fps, frame count = 8n + 1). "
        f"{len(ranges)} LTX chunk(s); Dearchive delivery stays at the model resolution "
        f"{width}x{height} and {float(info.get('fps') or fps):g} fps",
        flush=True,
    )
    raw_chunks: list[Path] = []
    for index, start, end in ranges:
        prepared = work_dir / f"prepared_{index:04d}_{start:06d}_{end:06d}.mp4"
        raw = work_dir / f"raw_{index:04d}_{start:06d}_{end:06d}.mp4"
        seed = args.seed + index
        split_chunk(ffmpeg, model_input, prepared, start, end, fps, args.force, prepared_fingerprint=sig["source_fingerprint"])
        chunk_sig = chunk_signature(sig, prepared, index, start, end, seed)
        if not args.force and resumable_output(raw, chunk_sig, width=width, height=height):
            print(f"Reuse Clean Up chunk {index + 1}/{len(ranges)}: {raw}", flush=True)
            raw_chunks.append(raw)
            continue
        workflow = json.loads(workflow_path.read_text(encoding="utf-8-sig"))
        prefix = f"arp_cleanup/{safe_stem(source.name)}_{identity_key}_chunk_{index:04d}"
        print(f"Clean Up chunk {index + 1}/{len(ranges)}: frames {start}-{end}, seed {seed}", flush=True)
        prompt = patch_workflow(
            args, workflow, prepared, comfy_dir, prefix, args.prompt, args.negative_prompt, seed
        )
        prompt_id = queue_prompt(args.comfy_url, prompt)
        print(f"Queued ComfyUI prompt: {prompt_id}", flush=True)
        history = wait_for_prompt(args.comfy_url, prompt_id, args.poll_seconds)
        produced = newest_output(extract_output_files(history, comfy_output_root), {".mp4", ".mov", ".mkv", ".webm"}, "Clean Up output")
        normalize_chunk(ffmpeg, produced, raw, width, height, fps, end - start)
        write_signature(raw, chunk_sig)
        raw_chunks.append(raw)

    stitched = work_dir / "stitched.mp4"
    stitch_chunks(ffmpeg, raw_chunks, ranges, stitched, fps, True)
    finalize_output(ffmpeg, stitched, source, output, info, (width, height))
    if args.save_scratch_mask and scratch_mask_preview is not None:
        save_scratch_mask_preview(ffmpeg, scratch_mask_preview, output, info)
    write_signature(output, sig)
    print(f"Wrote Clean Up video: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
