from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

from comfy_api import ensure_node_types, extract_output_files, object_info, queue_prompt_with_progress, wait_for_comfy
from common import ROOT, copy_to_comfy_input, ffprobe_for, file_fingerprint, find_ffmpeg, load_local_config, newest_output as newest_comfy_output, replace_unless_identical, replace_with_retry, resolve_path, root_relative, safe_stem, resumable_output, split_matches_source, video_info, write_signature, write_split_sidecar
from dependency_manager import (
    LTX25_CQ_ENHANCER_LORA,
    LTX25_DEV_GGUF_MODEL,
    LTX25_DISTILLED_LORA,
    LTX25_GGUF_MODEL,
    LTX25_PIXEL_UPSCALER_LORA,
    LTX25_TEXT_ENCODER,
    LTX25_VIDEO_VAE,
    ensure_ltx25_cq_enhancer_models,
    ensure_ltx25_upscale_models,
)


config = load_local_config()

SOURCE_SECTION_STEM_RE = re.compile(r"^(?P<prefix>.+)_(?P<start>\d{10})_(?P<end>\d{10})$")
ARP_ARTIFACT_STEM_RE = re.compile(r"^(?P<prefix>.+)_(?P<tag>outpaint|recomp|color|audio)_[0-9a-f]{8}$")
LTX25_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
DEFAULT_LTX25_PROMPT = (
    "A modern high-resolution live-action video, professionally photographed with a contemporary "
    "cinema camera and a sharp modern lens. Re-render the scene with crisp natural facial features, "
    "realistic skin texture, finely resolved hair and fabric, clean optical detail, rich micro-contrast, "
    "controlled highlights, natural depth, and stable coherent motion. Preserve the original people, "
    "action, composition, timing, clothing, objects, location, lighting intent, and camera movement."
)
DEFAULT_LTX25_NEGATIVE_PROMPT = (
    "soft focus, blurry, smeared detail, waxy faces, plastic skin, painterly, illustration, cartoon, "
    "video game, CGI, oversharpened, halos, ringing, warped faces, changed identity, changed clothing, "
    "changed objects, altered composition, duplicate limbs, temporal inconsistency, flicker, invented "
    "text, compression artifacts, film damage, dust, scratches"
)
from intermediate_video import codec_args as intermediate_codec_args, comfy_video_combine_inputs, container_args, migrate_profile_name

UPSCALE_METHODS = {"flashvsr", "seedvr2", "ltx25", "ltx25cq"}
# The CQ Enhancer restores at its working size; one of these then takes it to delivery size.
CQ_FINISH_METHODS = ("flashvsr", "seedvr2", "ltx25", "lanczos")
# The CQ LoRA was trained on 30 fps clips; its author warns other rates cause artefacts.
CQ_FPS = 30.0
CQ_DISTILLED_LORA_STRENGTH = 0.5
DEFAULT_SEEDVR2_MODEL = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
CURRENT_INTERMEDIATE_PROFILE = "high"


def working_codec_args(*, fast: bool = False) -> list[str]:
    return intermediate_codec_args(CURRENT_INTERMEDIATE_PROFILE, fast=fast)


def working_container_args(path: Path) -> list[str]:
    return container_args(str(path), CURRENT_INTERMEDIATE_PROFILE)


def delivery_codec_args() -> list[str]:
    return ["-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p"]


def delivery_audio_args(ffmpeg: str, media: Path) -> list[str]:
    """Copy an AAC track (such as the soundtrack stage's) instead of encoding it a second time."""
    probe = subprocess.run(
        [ffprobe_for(ffmpeg), "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(media)],
        capture_output=True, text=True, check=False,
    )
    return ["-c:a", "copy"] if probe.stdout.strip() == "aac" else ["-c:a", "aac", "-b:a", "320k"]


def encode_upscale_delivery(ffmpeg: str, ai_upscale: Path, output: Path) -> None:
    partial = output.with_suffix(output.suffix + ".full.partial" + output.suffix)
    subprocess.run([
        ffmpeg, "-y", "-i", str(ai_upscale), "-map", "0:v:0", "-map", "0:a?",
        *delivery_codec_args(), *delivery_audio_args(ffmpeg, ai_upscale), "-movflags", "+faststart", str(partial),
    ], check=True)
    replace_with_retry(partial, output, "Full-strength upscaled output")


def upscale_method(args: argparse.Namespace) -> str:
    method = str(getattr(args, "method", "flashvsr")).lower()
    return method if method in UPSCALE_METHODS else "flashvsr"


def cq_finish_method(args: argparse.Namespace) -> str:
    finish = str(getattr(args, "cq_finish", "flashvsr")).lower()
    return finish if finish in CQ_FINISH_METHODS else "flashvsr"


def finish_args(args: argparse.Namespace) -> argparse.Namespace:
    """The arguments the finishing upscaler of a CQ-enhanced render runs with."""
    return argparse.Namespace(**{**vars(args), "method": cq_finish_method(args)})


def delivery_dimensions(args: argparse.Namespace, output_width: int, output_height: int) -> tuple[int, int]:
    """The LTX Pixel Spatial LoRA delivers the nearest model-safe size, not the exact request."""
    method = upscale_method(args)
    if method == "ltx25cq":
        method = cq_finish_method(args)
    if method == "ltx25":
        return ltx25_generation_dimensions(output_width, output_height)
    return output_width, output_height


def default_output(source: Path, width: int, height: int, method: str = "flashvsr") -> Path:
    backend = method if method in UPSCALE_METHODS else "flashvsr"
    suffix = f"{backend}_{width}x{height}" if width and height else backend
    return ROOT / "output" / "upscaled" / f"{safe_stem(source.name)}_{suffix}.mp4"


def upscale_chunk_source_key(source: Path) -> str:
    stem = safe_stem(source.name)
    section = SOURCE_SECTION_STEM_RE.match(Path(source.name).stem)
    if section:
        return safe_stem(f"{section.group('prefix')}_{section.group('start')}")
    artifact = ARP_ARTIFACT_STEM_RE.match(Path(source.name).stem)
    if artifact:
        return safe_stem(f"{artifact.group('prefix')}_{artifact.group('tag')}")
    return stem


def upscale_chunk_dir(args: argparse.Namespace, source: Path, output_width: int, output_height: int) -> Path:
    source_key = upscale_chunk_source_key(source)
    backend = upscale_method(args)
    return ROOT / ".cache" / "upscale_chunks" / f"{source_key}_{backend}_{output_width}x{output_height}_{int(args.chunk_seconds * 1000)}ms_ov{max(0, args.overlap_frames)}"


def chunk_signature_match_view(signature: dict[str, Any]) -> dict[str, Any]:
    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items() if key != "mtime_ns"}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    return scrub(signature)


def compatible_upscale_chunk_signature(stored: dict[str, Any], expected: dict[str, Any]) -> bool:
    stored_view = chunk_signature_match_view(stored)
    expected_view = chunk_signature_match_view(expected)
    if stored_view == expected_view:
        return True
    stored_without_path = dict(stored_view)
    expected_without_path = dict(expected_view)
    stored_without_path.pop("source", None)
    expected_without_path.pop("source", None)
    return stored_without_path == expected_without_path


def resumable_upscale_chunk(path: Path, expected_signature: dict[str, Any], width: int, height: int) -> bool:
    from common import signature_path

    if not path.exists() or not signature_path(path).exists():
        return False
    try:
        import json

        stored = json.loads(signature_path(path).read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    if not compatible_upscale_chunk_signature(stored, expected_signature):
        return False
    return resumable_output(path, stored, width=width, height=height)


def find_reusable_upscale_chunk(chunk_dir: Path, chunk_name: str, chunk_sig: dict[str, Any], output_width: int, output_height: int) -> Path | None:
    for candidate in (ROOT / ".cache" / "upscale_chunks").glob(f"*/{chunk_name}"):
        if candidate.parent == chunk_dir:
            continue
        if resumable_upscale_chunk(candidate, chunk_sig, output_width, output_height):
            return candidate
    return None


# The legacy single-node workflow hardcoded these values, so leaving a knob at its
# default produces the same pixels as before the advanced node was adopted.
ADVANCED_DEFAULTS = {
    "flashvsr_color_fix": True,
    "flashvsr_tile_size": 256,
    "flashvsr_tile_overlap": 24,
    "flashvsr_vae_tile_multiplier": 1,
    "flashvsr_sparse_ratio": 2.0,
    "flashvsr_kv_ratio": 3.0,
    "flashvsr_local_range": 11,
}


def cq_signature(args: argparse.Namespace, source: Path, width: int, height: int) -> dict[str, Any]:
    """Identity of the CQ enhance pass alone, shared by its chunks and its stitched render."""
    return {
        "version": 1,
        "tool": "upscale_video.py",
        "method": "ltx25_cq_enhancer",
        "source": root_relative(source),
        "source_fingerprint": file_fingerprint(source),
        "enhance_width": width,
        "enhance_height": height,
        "comfy_dir": root_relative(resolve_path(args.comfy_dir)),
        "comfy_url": args.comfy_url,
        "base": args.cq_base,
        "model": cq_base_model(args),
        "text_encoder": args.ltx25_text_encoder,
        "video_vae": args.ltx25_video_vae,
        "lora": args.cq_lora,
        "lora_strength": args.cq_lora_strength,
        "guide_strength": args.cq_guide_strength,
        "frame_rate_mode": args.cq_frame_rate,
        "prompt": args.cq_prompt,
        "seed": args.cq_seed,
        "chunk_seconds": args.cq_chunk_seconds,
        "overlap_frames": args.overlap_frames,
        "fps": args.fps,
        "intermediate_profile": getattr(args, "intermediate_profile", "high"),
    }


def signature(args: argparse.Namespace, source: Path, output_width: int, output_height: int) -> dict[str, Any]:
    method = upscale_method(args)
    if method == "ltx25cq":
        finish = cq_finish_method(args)
        if finish == "lanczos":
            finish_settings: dict[str, Any] = {"method": "lanczos"}
        else:
            finish_settings = signature(finish_args(args), source, output_width, output_height)
        # The finishing pass reads the enhanced render, not the source, so only its settings count here.
        for key in ("tool", "source", "source_fingerprint", "target_width", "target_height"):
            finish_settings.pop(key, None)
        enhance = cq_output_signature(args, source)
        for key in ("tool", "source", "source_fingerprint"):
            enhance.pop(key, None)
        return {
            "version": 1,
            "tool": "upscale_video.py",
            "method": "ltx25_cq_enhancer_then_upscale",
            "source": root_relative(source),
            "source_fingerprint": file_fingerprint(source),
            "target_width": output_width,
            "target_height": output_height,
            "enhance": enhance,
            "finish": finish,
            "finish_settings": finish_settings,
        }
    if method == "ltx25":
        return {
            "version": 1,
            "tool": "upscale_video.py",
            "method": "ltx25_pixel_spatial_ic_lora_x2",
            "source": root_relative(source),
            "source_fingerprint": file_fingerprint(source),
            "target_width": output_width,
            "target_height": output_height,
            "comfy_dir": root_relative(resolve_path(args.comfy_dir)),
            "comfy_url": args.comfy_url,
            "model": args.ltx25_model,
            "text_encoder": args.ltx25_text_encoder,
            "video_vae": args.ltx25_video_vae,
            "lora": args.ltx25_upscaler_lora,
            "lora_strength": args.ltx25_lora_strength,
            "source_fidelity": args.ltx25_source_fidelity,
            "guidance_scale": args.ltx25_guidance_scale,
            "prompt": args.ltx25_prompt,
            "negative_prompt": args.ltx25_negative_prompt,
            "seed": args.ltx25_seed,
            "chunk_seconds": args.chunk_seconds,
            "overlap_frames": args.overlap_frames,
            "fps": args.fps,
            "intermediate_profile": getattr(args, "intermediate_profile", "high"),
        }
    if method == "seedvr2":
        return {
            "version": 1,
            "tool": "upscale_video.py",
            "method": "seedvr2_video_upscaler",
            "source": root_relative(source),
            "source_fingerprint": file_fingerprint(source),
            "target_width": output_width,
            "target_height": output_height,
            "comfy_dir": root_relative(resolve_path(args.comfy_dir)),
            "comfy_url": args.comfy_url,
            "model": args.seedvr2_model,
            "batch_size": args.seedvr2_batch_size,
            "color_correction": args.seedvr2_color_correction,
            "input_noise_scale": args.seedvr2_input_noise_scale,
            "latent_noise_scale": args.seedvr2_latent_noise_scale,
            "tiled_vae": args.seedvr2_tiled_vae,
            "vae_tile_size": args.seedvr2_vae_tile_size,
            "vae_tile_overlap": args.seedvr2_vae_tile_overlap,
            "preserve_vram": args.seedvr2_preserve_vram,
            "cache_model": args.seedvr2_cache_model,
            "blocks_to_swap": args.seedvr2_blocks_to_swap,
            "offload_io_components": args.seedvr2_offload_io_components,
            "seed": args.seedvr2_seed,
            "chunk_seconds": args.chunk_seconds,
            "overlap_frames": args.overlap_frames,
            "fps": args.fps,
            "intermediate_profile": getattr(args, "intermediate_profile", "high"),
        }
    sig = {
        "version": 6,
        "tool": "upscale_video.py",
        "method": "flashvsr_ultra_fast",
        "source": root_relative(source),
        "source_fingerprint": file_fingerprint(source),
        "target_width": output_width,
        "target_height": output_height,
        "flashvsr_pre_downscale": args.flashvsr_pre_downscale,
        "intermediate_profile": getattr(args, "intermediate_profile", "high"),
        "comfy_dir": root_relative(resolve_path(args.comfy_dir)),
        "comfy_url": args.comfy_url,
        "flashvsr_model": args.flashvsr_model,
        "flashvsr_mode": args.flashvsr_mode,
        "flashvsr_scale": args.flashvsr_scale,
        "flashvsr_tiled_vae": args.flashvsr_tiled_vae,
        "flashvsr_tiled_dit": args.flashvsr_tiled_dit,
        "flashvsr_unload_dit": args.flashvsr_unload_dit,
        "flashvsr_seed": args.flashvsr_seed,
        "chunk_seconds": args.chunk_seconds,
        "overlap_frames": args.overlap_frames,
        "fps": args.fps,
    }
    # Only record advanced knobs that differ from the legacy behaviour so outputs and
    # cached chunks rendered before these flags existed remain reusable.
    for name, default in ADVANCED_DEFAULTS.items():
        value = getattr(args, name)
        if value != default:
            sig[name] = value
    return sig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upscale an ARP video.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--target-width", type=int, default=3840)
    parser.add_argument("--target-height", type=int, default=2160)
    parser.add_argument("--comfy-dir", default=config.get("comfy_dir", str(ROOT / "tools" / "comfyui")))
    parser.add_argument("--comfy-url", default=config.get("comfy_url", "http://127.0.0.1:8188"))
    parser.add_argument("--comfy-output-root", default="")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--method", choices=sorted(UPSCALE_METHODS), default="flashvsr")
    parser.add_argument("--flashvsr-model", choices=["FlashVSR", "FlashVSR-v1.1"], default="FlashVSR-v1.1")
    parser.add_argument("--flashvsr-mode", choices=["tiny", "tiny-long", "full"], default="tiny")
    parser.add_argument("--flashvsr-scale", type=int, default=2)
    parser.add_argument("--flashvsr-pre-downscale", dest="flashvsr_pre_downscale", action="store_true", default=False, help="Downscale input before FlashVSR to the smallest model-compatible size that upscales cleanly to the requested target aspect.")
    parser.add_argument("--no-flashvsr-pre-downscale", dest="flashvsr_pre_downscale", action="store_false")
    parser.add_argument("--flashvsr-tiled-vae", dest="flashvsr_tiled_vae", action="store_true", default=True)
    parser.add_argument("--no-flashvsr-tiled-vae", dest="flashvsr_tiled_vae", action="store_false")
    parser.add_argument("--flashvsr-tiled-dit", dest="flashvsr_tiled_dit", action="store_true", default=True)
    parser.add_argument("--no-flashvsr-tiled-dit", dest="flashvsr_tiled_dit", action="store_false")
    parser.add_argument("--flashvsr-unload-dit", dest="flashvsr_unload_dit", action="store_true", default=False)
    parser.add_argument("--no-flashvsr-unload-dit", dest="flashvsr_unload_dit", action="store_false")
    parser.add_argument("--flashvsr-color-fix", dest="flashvsr_color_fix", action="store_true", default=True, help="Wavelet color correction that matches output colors back to the source.")
    parser.add_argument("--no-flashvsr-color-fix", dest="flashvsr_color_fix", action="store_false")
    parser.add_argument("--flashvsr-tile-size", type=int, default=256, help="DiT tile edge in pixels when tiled DiT is on (multiple of 32, max 1024). Larger tiles give faces more surrounding context at the cost of VRAM.")
    parser.add_argument("--flashvsr-tile-overlap", type=int, default=24, help="Feathered overlap between DiT tiles in pixels; raise it if tile seams are visible.")
    parser.add_argument("--flashvsr-vae-tile-multiplier", type=int, default=1, help="Multiplies the full-model tiled VAE decode tile size. Larger values can greatly speed decode if VRAM allows.")
    parser.add_argument("--flashvsr-sparse-ratio", type=float, default=2.0, help="Sparse attention density, 1.5-2.0: 2.0 is most stable, 1.5 is faster.")
    parser.add_argument("--flashvsr-kv-ratio", type=float, default=3.0, help="Attention memory budget, 1.0-3.0: 3.0 is highest quality, lower saves VRAM.")
    parser.add_argument("--flashvsr-local-range", type=int, choices=[9, 11], default=11, help="Temporal attention window: 11 is more stable, 9 keeps more fine motion (lips) and detail.")
    parser.add_argument("--flashvsr-seed", type=int, default=0)
    parser.add_argument("--seedvr2-model", default=DEFAULT_SEEDVR2_MODEL)
    parser.add_argument("--seedvr2-batch-size", type=int, default=5, help="Frames per SeedVR2 batch. Values in the 4n+1 sequence (1, 5, 9, ...) are supported; at least 5 enables temporal consistency.")
    parser.add_argument("--seedvr2-color-correction", choices=["wavelet", "adain", "none"], default="wavelet")
    parser.add_argument("--seedvr2-input-noise-scale", type=float, default=0.0)
    parser.add_argument("--seedvr2-latent-noise-scale", type=float, default=0.0)
    parser.add_argument("--seedvr2-tiled-vae", dest="seedvr2_tiled_vae", action="store_true", default=True)
    parser.add_argument("--no-seedvr2-tiled-vae", dest="seedvr2_tiled_vae", action="store_false")
    parser.add_argument("--seedvr2-vae-tile-size", type=int, default=512)
    parser.add_argument("--seedvr2-vae-tile-overlap", type=int, default=64)
    parser.add_argument("--seedvr2-preserve-vram", dest="seedvr2_preserve_vram", action="store_true", default=True)
    parser.add_argument("--no-seedvr2-preserve-vram", dest="seedvr2_preserve_vram", action="store_false")
    parser.add_argument("--seedvr2-cache-model", dest="seedvr2_cache_model", action="store_true", default=False)
    parser.add_argument("--no-seedvr2-cache-model", dest="seedvr2_cache_model", action="store_false")
    parser.add_argument("--seedvr2-blocks-to-swap", type=int, default=16)
    parser.add_argument("--seedvr2-offload-io-components", dest="seedvr2_offload_io_components", action="store_true", default=False)
    parser.add_argument("--no-seedvr2-offload-io-components", dest="seedvr2_offload_io_components", action="store_false")
    parser.add_argument("--seedvr2-seed", type=int, default=100)
    parser.add_argument("--ltx25-model", default=LTX25_GGUF_MODEL)
    parser.add_argument("--ltx25-text-encoder", default=LTX25_TEXT_ENCODER)
    parser.add_argument("--ltx25-video-vae", default=LTX25_VIDEO_VAE)
    parser.add_argument("--ltx25-upscaler-lora", default=LTX25_PIXEL_UPSCALER_LORA)
    parser.add_argument("--ltx25-lora-strength", type=float, default=1.0)
    parser.add_argument("--ltx25-source-fidelity", type=float, default=0.85)
    parser.add_argument(
        "--ltx25-guidance-scale",
        type=float,
        default=1.0,
        help="Text guidance for LTX reconstruction. Keep 1 for the fast distilled path; values above 1 use slower classifier-free guidance.",
    )
    parser.add_argument("--ltx25-seed", type=int, default=42)
    parser.add_argument("--ltx25-prompt", default=DEFAULT_LTX25_PROMPT)
    parser.add_argument("--ltx25-negative-prompt", default=DEFAULT_LTX25_NEGATIVE_PROMPT)
    parser.add_argument("--cq-finish", choices=CQ_FINISH_METHODS, default="flashvsr", help="Upscaler that takes the CQ-enhanced render to the delivery size, using its own settings.")
    parser.add_argument("--cq-base", choices=["distilled", "dev"], default="distilled", help="distilled reuses the local distilled transformer; dev is the LoRA author's recipe (dev transformer + distilled LoRA at 0.5, about 25 GB of extra downloads).")
    parser.add_argument("--cq-lora", default=LTX25_CQ_ENHANCER_LORA)
    parser.add_argument("--cq-lora-strength", type=float, default=1.0)
    parser.add_argument("--cq-guide-strength", type=float, default=1.0, help="How strongly the source video conditions the CQ render; the author's workflow uses 1.")
    parser.add_argument("--cq-short-edge", type=int, default=720, help="Short edge the CQ Enhancer works at; the author's workflow uses 720.")
    parser.add_argument("--cq-frame-rate", choices=["resample", "retime"], default="resample", help="resample converts to 30 fps by repeating frames, as the author's workflow does; retime plays the original frames at 30 fps. Either way every source frame comes back once.")
    parser.add_argument("--cq-chunk-seconds", type=float, default=8.0, help="Longest source span per CQ render; chunks also split at --shot-manifest cuts, and longer shots split evenly. The author's workflow uses 153 frames at 30 fps (about 5 s).")
    parser.add_argument("--cq-colour", choices=["model", "source"], default="model", help="model keeps the CQ render's colour (it can colourise black-and-white film); source keeps the source's colour and takes only brightness detail from the CQ render.")
    parser.add_argument("--cq-seed", type=int, default=42)
    parser.add_argument("--cq-prompt", default="", help="The CQ LoRA needs no prompt. It samples without CFG, so there is no negative prompt.")
    parser.add_argument("--chunk-seconds", type=float, default=6.0, help="Upscale in chunks of roughly this many seconds. Use 0 to send the whole clip.")
    parser.add_argument("--overlap-frames", type=int, default=8, help="Frames repeated before each chunk, then trimmed before stitching.")
    parser.add_argument("--blend-strength", type=float, default=100.0, help="AI upscale contribution in the final delivery. 0 is a conventional Lanczos resize; 100 is the full AI upscale.")
    parser.add_argument("--shot-manifest", default="", help="Optional shot CSV containing per-row upscale_strength plus fade_to_next/crossfade_seconds tween metadata.")
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--ffmpeg", default="")
    parser.add_argument(
        "--intermediate-profile",
        type=migrate_profile_name,
        choices=["low", "medium", "high", "lossless"],
        default="high",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def normalized_blend_strength(value: Any, default: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number / 100.0))


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "on"}


def optional_int(value: Any) -> int | None:
    try:
        text = str(value or "").strip()
        return int(float(text)) if text else None
    except (TypeError, ValueError):
        return None


def read_upscale_shots(manifest: Path | None, total_frames: int, fps: float, default_strength: float) -> list[dict[str, Any]]:
    if manifest is None or not manifest.is_file():
        return []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    rows = list(csv.DictReader(lines)) if lines else []
    shots: list[dict[str, Any]] = []
    start_frame = 0
    for index, row in enumerate(rows):
        explicit_start = optional_int(row.get("start_frame"))
        if explicit_start is not None:
            start_frame = max(0, min(max(0, total_frames - 1), explicit_start))
        end_frame = optional_int(row.get("end_frame"))
        if end_frame is None:
            try:
                end_text = str(row.get("end", "0")).strip()
                parts = [float(part) for part in end_text.split(":")]
                seconds = sum(part * (60 ** power) for power, part in enumerate(reversed(parts)))
            except (TypeError, ValueError):
                seconds = start_frame / max(fps, 1e-6)
            end_frame = int(round(seconds * fps))
        end_frame = min(total_frames, max(start_frame + 1, end_frame))
        if index == len(rows) - 1:
            end_frame = total_frames
        shots.append({
            "start": start_frame,
            "end": end_frame,
            "strength": normalized_blend_strength(row.get("upscale_strength", ""), default_strength),
            "fade": truthy(row.get("fade_to_next")),
            "crossfade_seconds": row.get("crossfade_seconds", ""),
        })
        start_frame = end_frame
    return shots


def blend_weight_expression(shots: list[dict[str, Any]], fps: float, default_strength: float) -> str:
    if not shots:
        return f"{default_strength:.8f}"
    expression = f"{float(shots[-1]['strength']):.8f}"
    for index in range(len(shots) - 2, -1, -1):
        current = float(shots[index]["strength"])
        following = float(shots[index + 1]["strength"])
        boundary = int(shots[index]["end"])
        transition_frames = 0
        if shots[index].get("fade"):
            try:
                transition_frames = max(0, int(round(float(shots[index].get("crossfade_seconds") or 0.0) * fps)))
            except (TypeError, ValueError):
                transition_frames = 0
        transition_frames = min(
            transition_frames,
            max(0, int(shots[index]["end"]) - int(shots[index]["start"])),
            max(0, int(shots[index + 1]["end"]) - int(shots[index + 1]["start"])),
        )
        if transition_frames <= 0 or abs(current - following) < 1e-9:
            expression = f"if(lt(N,{boundary}),{current:.8f},{expression})"
            continue
        ramp_start = boundary - transition_frames // 2
        ramp_end = ramp_start + transition_frames
        ramp = f"{current:.8f}+({following - current:.8f})*(N-{ramp_start})/{max(1, transition_frames)}"
        expression = f"if(lt(N,{ramp_start}),{current:.8f},if(lt(N,{ramp_end}),{ramp},{expression}))"
    return expression


def blend_upscale_delivery(
    ffmpeg: str,
    source: Path,
    ai_upscale: Path,
    output: Path,
    width: int,
    height: int,
    fps: float,
    shots: list[dict[str, Any]],
    default_strength: float,
) -> None:
    weight = blend_weight_expression(shots, fps, default_strength)
    partial = output.with_suffix(output.suffix + ".blend.partial" + output.suffix)
    filters = (
        f"[0:v]setpts=N/({fps:.8f}*TB),fps={fps:.8f},scale={width}:{height}:flags=lanczos,setsar=1[src];"
        f"[1:v]setpts=N/({fps:.8f}*TB),fps={fps:.8f},scale={width}:{height}:flags=lanczos,setsar=1[ai];"
        f"[src][ai]blend=all_expr='A*(1-({weight}))+B*({weight})',"
        "format=yuv420p[vout]"
    )
    command = [
        ffmpeg, "-y", "-i", str(source), "-i", str(ai_upscale),
        "-filter_complex", filters, "-map", "[vout]", "-map", "0:a?", "-shortest",
        "-r", f"{fps:.8f}", "-fps_mode", "cfr", *delivery_codec_args(),
        *delivery_audio_args(ffmpeg, source), "-movflags", "+faststart", str(partial),
    ]
    print("Blending source motion/detail with AI upscale by shot", flush=True)
    subprocess.run(command, check=True)
    replace_with_retry(partial, output, "Blended upscaled output")


def default_from_spec(spec: Any) -> Any:
    if isinstance(spec, (list, tuple)):
        if len(spec) > 1 and isinstance(spec[1], dict) and "default" in spec[1]:
            return spec[1]["default"]
        if spec and isinstance(spec[0], list) and spec[0]:
            return spec[0][0]
    return None


def comfy_node_inputs(class_type: str, info: dict[str, Any], values: dict[str, Any], connections: dict[str, list[Any]]) -> dict[str, Any]:
    inputs: dict[str, Any] = dict(connections)
    input_info = info.get(class_type, {}).get("input", {})
    accepted: set[str] = set()
    for group in ("required", "optional"):
        for name, spec in (input_info.get(group) or {}).items():
            accepted.add(name)
            if name in connections:
                continue
            value = default_from_spec(spec)
            if value is not None:
                inputs[name] = value

    # Use the Ultra Fast node's real input names so ARP logs and saved settings map
    # directly to the Comfy node.
    for name, value in values.items():
        if not accepted or name in accepted:
            inputs[name] = value
    return inputs


def flashvsr_prompt(video_name: str, fps: float, args: argparse.Namespace, prefix: str, info: dict[str, Any]) -> dict[str, Any]:
    init_values = {
        "model": args.flashvsr_model,
        "mode": args.flashvsr_mode,
        "force_offload": False,
        # The basic FlashVSRNode ran fp16; keep it so existing renders stay reproducible.
        "precision": "fp16",
    }
    adv_values = {
        "scale": args.flashvsr_scale,
        "color_fix": bool(args.flashvsr_color_fix),
        "tiled_vae": bool(args.flashvsr_tiled_vae),
        "tiled_dit": bool(args.flashvsr_tiled_dit),
        "tile_size": args.flashvsr_tile_size,
        "tile_overlap": args.flashvsr_tile_overlap,
        "vae_tile_multiplier": args.flashvsr_vae_tile_multiplier,
        "unload_dit": bool(args.flashvsr_unload_dit),
        "sparse_ratio": args.flashvsr_sparse_ratio,
        "kv_ratio": args.flashvsr_kv_ratio,
        "local_range": args.flashvsr_local_range,
        "seed": args.flashvsr_seed,
    }
    return {
        "1": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": video_name,
                "force_rate": 0.0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
                "format": "None",
            },
        },
        "2": {"class_type": "FlashVSRInitPipe", "inputs": comfy_node_inputs("FlashVSRInitPipe", info, init_values, {})},
        "3": {"class_type": "FlashVSRNodeAdv", "inputs": comfy_node_inputs("FlashVSRNodeAdv", info, adv_values, {"pipe": ["2", 0], "frames": ["1", 0]})},
        "4": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["3", 0],
                # Upscale chunks are intentionally encoded without audio.  Asking
                # VHS_LoadVideo for its lazy audio output makes silent chunks fail
                # before FlashVSR runs; the source audio is muxed back after the
                # scaled chunks are stitched.
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": prefix,
                **comfy_video_combine_inputs(CURRENT_INTERMEDIATE_PROFILE),
                "save_metadata": True,
                "pingpong": False,
                "save_output": True,
            },
        },
    }


def flashvsr_run(args: argparse.Namespace, source: Path, partial: Path, output_width: int, output_height: int) -> Path:
    comfy_dir = resolve_path(args.comfy_dir)
    comfy_output_root = resolve_path(args.comfy_output_root) if args.comfy_output_root else comfy_dir / "output"
    if not (comfy_dir / "main.py").exists():
        raise FileNotFoundError(f"ComfyUI main.py not found: {comfy_dir / 'main.py'}")
    wait_for_comfy(args.comfy_url, timeout_seconds=180, poll_seconds=args.poll_seconds)
    required_nodes = {
        "VHS_LoadVideo": "ComfyUI-VideoHelperSuite",
        "VHS_VideoCombine": "ComfyUI-VideoHelperSuite",
        "FlashVSRInitPipe": "ComfyUI-FlashVSR_Ultra_Fast",
        "FlashVSRNodeAdv": "ComfyUI-FlashVSR_Ultra_Fast",
    }
    ensure_node_types(args.comfy_url, required_nodes, "FlashVSR upscaling")
    info = object_info(args.comfy_url)
    video_name = copy_to_comfy_input(source, comfy_dir, "arp_upscale")
    fps = args.fps or float(video_info(source)["fps"])
    prefix = f"arp_upscale/{safe_stem(source.name)}_flashvsr_{output_width}x{output_height}"
    prompt = flashvsr_prompt(video_name, fps, args, prefix, info)
    history = queue_prompt_with_progress(args.comfy_url, prompt, args.poll_seconds, {"3"})
    produced = newest_comfy_output(extract_output_files(history, comfy_output_root), {".mp4", ".mov", ".mkv", ".webm"}, "FlashVSR video")
    partial.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, partial)
    return partial


def seedvr2_generation_resolution(output_width: int, output_height: int) -> int:
    """Return the nearest SeedVR2-safe output short edge (a multiple of 16)."""
    return max(16, int(round(min(output_width, output_height) / 16.0)) * 16)


def seedvr2_prompt(
    video_name: str,
    fps: float,
    output_width: int,
    output_height: int,
    args: argparse.Namespace,
    prefix: str,
    info: dict[str, Any],
) -> dict[str, Any]:
    """Build a SeedVR2 prompt while adapting to the installed node version."""
    new_node_types = {"SeedVR2LoadDiTModel", "SeedVR2LoadVAEModel", "SeedVR2VideoUpscaler"}
    if new_node_types.issubset(info):
        preserve_vram = bool(args.seedvr2_preserve_vram)
        cache_model = bool(args.seedvr2_cache_model)
        dit_offload = "cpu" if preserve_vram or cache_model or args.seedvr2_blocks_to_swap or args.seedvr2_offload_io_components else "none"
        vae_offload = "cpu" if preserve_vram or cache_model else "none"
        dit_values = {
            "model": args.seedvr2_model,
            "blocks_to_swap": int(args.seedvr2_blocks_to_swap),
            "swap_io_components": bool(args.seedvr2_offload_io_components),
            "offload_device": dit_offload,
            "cache_model": cache_model,
            "attention_mode": "sdpa",
        }
        vae_values = {
            "encode_tiled": bool(args.seedvr2_tiled_vae),
            "encode_tile_size": int(args.seedvr2_vae_tile_size),
            "encode_tile_overlap": int(args.seedvr2_vae_tile_overlap),
            "decode_tiled": bool(args.seedvr2_tiled_vae),
            "decode_tile_size": int(args.seedvr2_vae_tile_size),
            "decode_tile_overlap": int(args.seedvr2_vae_tile_overlap),
            "tile_debug": "false",
            "offload_device": vae_offload,
            "cache_model": cache_model,
        }
        upscale_values = {
            "seed": int(args.seedvr2_seed),
            "resolution": seedvr2_generation_resolution(output_width, output_height),
            "max_resolution": 0,
            "batch_size": int(args.seedvr2_batch_size),
            "uniform_batch_size": False,
            "temporal_overlap": 0,
            "prepend_frames": 0,
            "color_correction": args.seedvr2_color_correction,
            "input_noise_scale": float(args.seedvr2_input_noise_scale),
            "latent_noise_scale": float(args.seedvr2_latent_noise_scale),
            "offload_device": "cpu" if preserve_vram else "none",
            "enable_debug": False,
        }
        return {
            "1": {
                "class_type": "VHS_LoadVideo",
                "inputs": {
                    "video": video_name,
                    "force_rate": 0.0,
                    "custom_width": 0,
                    "custom_height": 0,
                    "frame_load_cap": 0,
                    "skip_first_frames": 0,
                    "select_every_nth": 1,
                    "format": "None",
                },
            },
            "2": {
                "class_type": "SeedVR2LoadDiTModel",
                "inputs": comfy_node_inputs("SeedVR2LoadDiTModel", info, dit_values, {}),
            },
            "3": {
                "class_type": "SeedVR2LoadVAEModel",
                "inputs": comfy_node_inputs("SeedVR2LoadVAEModel", info, vae_values, {}),
            },
            "4": {
                "class_type": "SeedVR2VideoUpscaler",
                "inputs": comfy_node_inputs(
                    "SeedVR2VideoUpscaler",
                    info,
                    upscale_values,
                    {"image": ["1", 0], "dit": ["2", 0], "vae": ["3", 0]},
                ),
            },
            "5": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["4", 0],
                    "frame_rate": fps,
                    "loop_count": 0,
                    "filename_prefix": prefix,
                    **comfy_video_combine_inputs(CURRENT_INTERMEDIATE_PROFILE),
                    "save_metadata": True,
                    "pingpong": False,
                    "save_output": True,
                },
            },
        }

    # SeedVR2 releases before v2.5 used a single upscaler plus two settings nodes.
    block_swap_values = {
        "blocks_to_swap": int(args.seedvr2_blocks_to_swap),
        "offload_io_components": bool(args.seedvr2_offload_io_components),
    }
    extra_values = {
        "tiled_vae": bool(args.seedvr2_tiled_vae),
        "vae_tile_size": int(args.seedvr2_vae_tile_size),
        "vae_tile_overlap": int(args.seedvr2_vae_tile_overlap),
        "preserve_vram": bool(args.seedvr2_preserve_vram),
        "cache_model": bool(args.seedvr2_cache_model),
        "enable_debug": False,
    }
    upscale_values = {
        "model": args.seedvr2_model,
        "seed": int(args.seedvr2_seed),
        # SeedVR2 defines this as the output's short edge and preserves aspect ratio.
        # ARP normalizes the result to the exact requested delivery size afterwards.
        "new_resolution": seedvr2_generation_resolution(output_width, output_height),
        "batch_size": int(args.seedvr2_batch_size),
        "color_correction": args.seedvr2_color_correction,
        "input_noise_scale": float(args.seedvr2_input_noise_scale),
        "latent_noise_scale": float(args.seedvr2_latent_noise_scale),
    }
    return {
        "1": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": video_name,
                "force_rate": 0.0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
                "format": "None",
            },
        },
        "2": {
            "class_type": "SeedVR2BlockSwap",
            "inputs": comfy_node_inputs("SeedVR2BlockSwap", info, block_swap_values, {}),
        },
        "3": {
            "class_type": "SeedVR2ExtraArgs",
            "inputs": comfy_node_inputs("SeedVR2ExtraArgs", info, extra_values, {}),
        },
        "4": {
            "class_type": "SeedVR2",
            "inputs": comfy_node_inputs(
                "SeedVR2",
                info,
                upscale_values,
                {"images": ["1", 0], "block_swap_config": ["2", 0], "extra_args": ["3", 0]},
            ),
        },
        "5": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["4", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": prefix,
                **comfy_video_combine_inputs(CURRENT_INTERMEDIATE_PROFILE),
                "save_metadata": True,
                "pingpong": False,
                "save_output": True,
            },
        },
    }


def seedvr2_run(args: argparse.Namespace, source: Path, partial: Path, output_width: int, output_height: int) -> Path:
    comfy_dir = resolve_path(args.comfy_dir)
    comfy_output_root = resolve_path(args.comfy_output_root) if args.comfy_output_root else comfy_dir / "output"
    if not (comfy_dir / "main.py").exists():
        raise FileNotFoundError(f"ComfyUI main.py not found: {comfy_dir / 'main.py'}")
    wait_for_comfy(args.comfy_url, timeout_seconds=180, poll_seconds=args.poll_seconds)
    info = object_info(args.comfy_url)
    common_nodes = {
        "VHS_LoadVideo": "ComfyUI-VideoHelperSuite",
        "VHS_VideoCombine": "ComfyUI-VideoHelperSuite",
    }
    new_nodes = {
        "SeedVR2LoadDiTModel": "ComfyUI-SeedVR2_VideoUpscaler",
        "SeedVR2LoadVAEModel": "ComfyUI-SeedVR2_VideoUpscaler",
        "SeedVR2VideoUpscaler": "ComfyUI-SeedVR2_VideoUpscaler",
    }
    legacy_nodes = {
        "SeedVR2": "ComfyUI-SeedVR2_VideoUpscaler",
        "SeedVR2BlockSwap": "ComfyUI-SeedVR2_VideoUpscaler",
        "SeedVR2ExtraArgs": "ComfyUI-SeedVR2_VideoUpscaler",
    }
    required_nodes = common_nodes | (new_nodes if set(new_nodes).issubset(info) else legacy_nodes)
    ensure_node_types(args.comfy_url, required_nodes, "SeedVR2 upscaling", comfy_dir=comfy_dir)
    video_name = copy_to_comfy_input(source, comfy_dir, "arp_upscale_seedvr2")
    fps = args.fps or float(video_info(source)["fps"])
    prefix = f"arp_upscale/{safe_stem(source.name)}_seedvr2_{output_width}x{output_height}"
    prompt = seedvr2_prompt(video_name, fps, output_width, output_height, args, prefix, info)
    print(f"Sending SeedVR2 prompt nodes: {sorted(node['class_type'] for node in prompt.values())}", flush=True)
    history = queue_prompt_with_progress(args.comfy_url, prompt, args.poll_seconds, {"4"})
    produced = newest_comfy_output(
        extract_output_files(history, comfy_output_root),
        {".mp4", ".mov", ".mkv", ".webm"},
        "SeedVR2 video",
    )
    partial.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, partial)
    return partial


def ltx25_generation_dimensions(output_width: int, output_height: int) -> tuple[int, int]:
    """Return the nearest LTX-safe 2x canvas; delivery is normalized afterwards."""
    return (
        max(64, int(round(output_width / 64.0)) * 64),
        max(64, int(round(output_height / 64.0)) * 64),
    )


def ltx25_prompt(
    video_name: str,
    fps: float,
    frames: int,
    width: int,
    height: int,
    args: argparse.Namespace,
    prefix: str,
) -> dict[str, Any]:
    """Build a video-only LTX 2.5 Pixel Spatial Upscaler IC-LoRA prompt."""
    return ltx_ic_lora_prompt(
        video_name, fps, frames, width, height, prefix,
        unet=args.ltx25_model,
        lora=args.ltx25_upscaler_lora,
        lora_strength=args.ltx25_lora_strength,
        text_encoder=args.ltx25_text_encoder,
        vae=args.ltx25_video_vae,
        prompt=args.ltx25_prompt,
        negative_prompt=args.ltx25_negative_prompt,
        guide_strength=args.ltx25_source_fidelity,
        cfg=args.ltx25_guidance_scale,
        seed=args.ltx25_seed,
    )


def ltx_ic_lora_prompt(
    video_name: str,
    fps: float,
    frames: int,
    width: int,
    height: int,
    prefix: str,
    *,
    unet: str,
    lora: str,
    lora_strength: float,
    text_encoder: str,
    vae: str,
    prompt: str,
    negative_prompt: str,
    guide_strength: float,
    cfg: float,
    seed: int,
    distilled_lora: str = "",
    distilled_lora_strength: float = CQ_DISTILLED_LORA_STRENGTH,
) -> dict[str, Any]:
    """Build a video-only LTX 2.5 prompt that re-renders the loaded video through an IC-LoRA.

    The LoRA's reference_downscale_factor metadata sets the guide scale, so the same graph
    serves the 2x Pixel Spatial upscaler and the same-size CQ Enhancer.
    """
    graph = {
        "1": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": video_name,
                "force_rate": 0.0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": frames,
                "skip_first_frames": 0,
                "select_every_nth": 1,
                "format": "None",
            },
        },
        "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet}},
        "3": {
            "class_type": "ARPLTXVideoOnlyICLoRALoader",
            "inputs": {
                "model": ["2", 0],
                "lora_name": lora,
                "strength_model": float(lora_strength),
            },
        },
        "4": {
            "class_type": "CLIPLoaderGGUF",
            "inputs": {"clip_name": text_encoder, "type": "ltxv"},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 0], "text": prompt},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 0], "text": negative_prompt},
        },
        "7": {
            "class_type": "LTXVConditioning",
            "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": fps},
        },
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "9": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": width, "height": height, "length": frames, "batch_size": 1},
        },
        "10": {
            "class_type": "LTXAddVideoICLoRAGuide",
            "inputs": {
                "positive": ["7", 0],
                "negative": ["7", 1],
                "vae": ["8", 0],
                "latent": ["9", 0],
                "image": ["1", 0],
                "frame_idx": 0,
                "strength": float(guide_strength),
                "latent_downscale_factor": ["3", 1],
                "crop": "disabled",
                "use_tiled_encode": True,
                "tile_size": 256,
                "tile_overlap": 64,
            },
        },
        "11": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["3", 0],
                "positive": ["10", 0],
                "negative": ["10", 1],
                "cfg": float(cfg),
            },
        },
        "12": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed), "control_after_generate": "fixed"}},
        "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "14": {"class_type": "ManualSigmas", "inputs": {"sigmas": LTX25_SIGMAS}},
        "15": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["12", 0],
                "guider": ["11", 0],
                "sampler": ["13", 0],
                "sigmas": ["14", 0],
                "latent_image": ["10", 2],
            },
        },
        "16": {
            "class_type": "LTXVCropGuides",
            "inputs": {"positive": ["10", 0], "negative": ["10", 1], "latent": ["15", 0]},
        },
        "17": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["16", 2],
                "vae": ["8", 0],
                "tile_size": 512,
                "overlap": 64,
                "temporal_size": 512,
                "temporal_overlap": 4,
            },
        },
        "18": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["17", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": prefix,
                **comfy_video_combine_inputs(CURRENT_INTERMEDIATE_PROFILE),
                "save_metadata": True,
                "pingpong": False,
                "save_output": True,
            },
        },
    }
    if distilled_lora:
        graph["19"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["2", 0], "lora_name": distilled_lora, "strength_model": float(distilled_lora_strength)},
        }
        graph["3"]["inputs"]["model"] = ["19", 0]
    return graph


LTX_IC_LORA_NODE_TYPES = {
    "VHS_LoadVideo": "ComfyUI-VideoHelperSuite",
    "VHS_VideoCombine": "ComfyUI-VideoHelperSuite",
    "UnetLoaderGGUF": "ComfyUI-GGUF",
    "CLIPLoaderGGUF": "ComfyUI-GGUF",
    "ARPLTXVideoOnlyICLoRALoader": "ComfyUI-ARP",
    "LTXVConditioning": "ComfyUI-LTXVideo",
    "LTXAddVideoICLoRAGuide": "ComfyUI-LTXVideo",
    "LTXVCropGuides": "ComfyUI-LTXVideo",
}


def ltx25_run(args: argparse.Namespace, source: Path, partial: Path, output_width: int, output_height: int) -> Path:
    comfy_dir = resolve_path(args.comfy_dir)
    comfy_output_root = resolve_path(args.comfy_output_root) if args.comfy_output_root else comfy_dir / "output"
    if not (comfy_dir / "main.py").exists():
        raise FileNotFoundError(f"ComfyUI main.py not found: {comfy_dir / 'main.py'}")
    ensure_ltx25_upscale_models(comfy_dir)
    wait_for_comfy(args.comfy_url, timeout_seconds=180, poll_seconds=args.poll_seconds)
    ensure_node_types(args.comfy_url, LTX_IC_LORA_NODE_TYPES, "LTX 2.5 Pixel Spatial upscaling")
    source_info = video_info(source)
    video_name = copy_to_comfy_input(source, comfy_dir, "arp_upscale_ltx25")
    fps = args.fps or float(source_info["fps"])
    frames = int(source_info["frames"])
    generation_width, generation_height = ltx25_generation_dimensions(output_width, output_height)
    prefix = f"arp_upscale/{safe_stem(source.name)}_ltx25_{generation_width}x{generation_height}"
    prompt = ltx25_prompt(video_name, fps, frames, generation_width, generation_height, args, prefix)
    print(f"Sending LTX 2.5 upscale prompt nodes: {sorted(node['class_type'] for node in prompt.values())}", flush=True)
    history = queue_prompt_with_progress(args.comfy_url, prompt, args.poll_seconds, {"15", "17"})
    produced = newest_comfy_output(
        extract_output_files(history, comfy_output_root),
        {".mp4", ".mov", ".mkv", ".webm"},
        "LTX 2.5 upscaled video",
    )
    partial.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, partial)
    return partial


def cq_base_model(args: argparse.Namespace) -> str:
    return LTX25_DEV_GGUF_MODEL if args.cq_base == "dev" else args.ltx25_model


def cq_output_size(width: int, height: int, short_edge: int) -> tuple[int, int]:
    """The CQ render's delivered size: the source aspect with its short edge at the working size."""
    scale = max(64, int(short_edge)) / max(1, min(width, height))
    return max(2, int(round(width * scale / 2.0)) * 2), max(2, int(round(height * scale / 2.0)) * 2)


def cq_generation_size(width: int, height: int) -> tuple[int, int]:
    """LTX renders on a 32 px grid, so 720 becomes 704; the render is resized back afterwards."""
    return max(64, int(round(width / 32.0)) * 32), max(64, int(round(height / 32.0)) * 32)


def cq_dimensions(args: argparse.Namespace, source: Path) -> tuple[tuple[int, int], tuple[int, int]]:
    """(delivered size, LTX render size) of the CQ pass for this source."""
    info = video_info(source)
    output = cq_output_size(int(info["width"]), int(info["height"]), args.cq_short_edge)
    return output, cq_generation_size(*output)


def cq_render_path(source: Path, width: int, height: int) -> Path:
    """Where the stitched CQ render of ``source`` at its delivered size is cached (the GUI previews it)."""
    return ROOT / ".cache" / "upscale_cq" / f"{safe_stem(source.name)}_cq_{width}x{height}.mkv"


def cq_frame_rate_mode(args: argparse.Namespace, fps: float) -> str:
    # Resampling above 30 fps would drop source frames, so faster footage is always retimed.
    return "resample" if args.cq_frame_rate == "resample" and fps < CQ_FPS else "retime"


def cq_chunk_plan(
    total_frames: int, fps: float, chunk_seconds: float, overlap_frames: int, mode: str,
    shot_cuts: list[int] | tuple[int, ...] = (),
) -> list[tuple[int, int, int, int]]:
    """Source windows as (start, end, trim_start, model_frames), model_frames being the 30 fps 8n+1 render.

    Chunks never span a shot cut, so every frame of a shot is restored by the same generation
    where possible. Each shot starts a fresh chunk with no lead-in (the source cuts there anyway).
    A shot longer than chunk_seconds is split into equal pieces, each after the first carrying an
    overlap lead-in from the same shot to dissolve across.
    """
    if total_frames <= 0:
        return []
    limit = max(1, int(round(chunk_seconds * fps))) if chunk_seconds > 0 and fps > 0 else total_frames
    overlap = max(0, min(int(overlap_frames), limit - 1))
    bounds = [0, *sorted({int(cut) for cut in shot_cuts if 0 < int(cut) < total_frames}), total_frames]
    plan: list[tuple[int, int, int, int]] = []
    for shot_start, shot_end in zip(bounds, bounds[1:]):
        length = shot_end - shot_start
        pieces = max(1, math.ceil(length / limit))
        for piece in range(pieces):
            start = shot_start + length * piece // pieces
            end = shot_start + length * (piece + 1) // pieces
            lead = min(overlap, start - shot_start)
            span = end - start + lead
            working = span if mode == "retime" else int(math.ceil(span * CQ_FPS / max(fps, 0.001)))
            plan.append((start - lead, end, lead, max(9, int(math.ceil((working - 1) / 8.0)) * 8 + 1)))
    return plan


def cq_shot_cuts(args: argparse.Namespace, source: Path) -> list[int]:
    """Frames of ``source`` where a new shot starts, from the shot list; empty without one."""
    manifest = resolve_path(args.shot_manifest) if getattr(args, "shot_manifest", "") else None
    if manifest is None or not manifest.is_file():
        return []
    info = video_info(source)
    frames = int(info["frames"])
    shots = read_upscale_shots(manifest, frames, args.fps or float(info["fps"]), 1.0)
    # Shots past the end of a shorter input (an upscale preview clip) are clamped onto its last
    # frame; that is not a real cut, and a one-frame chunk there would be a wasted generation.
    return sorted({int(shot["start"]) for shot in shots if 0 < int(shot["start"]) < frames - 1})


def prepare_cq_chunk(
    ffmpeg: str,
    source: Path,
    target: Path,
    start_frame: int,
    end_frame: int,
    model_frames: int,
    fps: float,
    width: int,
    height: int,
    mode: str,
    force: bool,
    source_fingerprint: dict[str, Any],
) -> None:
    if target.exists() and not force and split_matches_source(target, source_fingerprint):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial" + target.suffix)
    # Retiming declares each source frame 1/30 s long; the fps filter then only snaps those
    # timestamps onto the exact 30 fps grid, which the millisecond timebase cannot hold.
    to_working_rate = f"fps={CQ_FPS:g}" if mode == "resample" else f"setpts=N/({CQ_FPS:g}*TB),fps={CQ_FPS:g}"
    vf = (
        f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=N/({fps:.8f}*TB),"
        f"scale={width}:{height}:flags=lanczos,setsar=1,{to_working_rate},"
        f"tpad=stop_mode=clone:stop={model_frames},trim=end_frame={model_frames}"
    )
    subprocess.run(
        [
            ffmpeg, "-y", "-i", str(source), "-vf", vf, "-an", "-r", f"{CQ_FPS:g}", "-fps_mode", "cfr",
            # A retimed last frame keeps its source duration; cfr would pad it with a duplicate.
            "-frames:v", str(model_frames),
            *working_codec_args(fast=True), *working_container_args(partial), str(partial),
        ],
        check=True,
    )
    replace_unless_identical(partial, target, f"CQ prepared chunk {target.name}")
    write_split_sidecar(target, source, source_fingerprint)


def normalize_cq_chunk(
    ffmpeg: str,
    source: Path,
    target: Path,
    width: int,
    height: int,
    trim_start: int,
    keep_frames: int,
    fps: float,
    mode: str,
) -> None:
    """Return a 30 fps CQ render to the source rate, one output frame per source frame."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial" + target.suffix)
    # Resampling repeated some source frames; the nearest working frame to each source
    # timestamp is always a render of that source frame, because 30 fps is the faster rate.
    to_source_rate = f"setpts=N/({CQ_FPS:g}*TB),fps={fps:.8f}" if mode == "resample" else f"setpts=N/({fps:.8f}*TB),fps={fps:.8f}"
    vf = (
        f"{to_source_rate},trim=start_frame={trim_start}:end_frame={trim_start + keep_frames},"
        f"setpts=N/({fps:.8f}*TB),scale={width}:{height}:flags=lanczos,setsar=1"
    )
    subprocess.run(
        [
            ffmpeg, "-y", "-i", str(source), "-vf", vf, "-an", "-r", f"{fps:.8f}", "-fps_mode", "cfr",
            "-frames:v", str(keep_frames), *working_codec_args(), *working_container_args(partial), str(partial),
        ],
        check=True,
    )
    replace_with_retry(partial, target, f"CQ normalized chunk {target.name}")


def cq_run(args: argparse.Namespace, source: Path, partial: Path, width: int, height: int, frames: int) -> Path:
    comfy_dir = resolve_path(args.comfy_dir)
    comfy_output_root = resolve_path(args.comfy_output_root) if args.comfy_output_root else comfy_dir / "output"
    if not (comfy_dir / "main.py").exists():
        raise FileNotFoundError(f"ComfyUI main.py not found: {comfy_dir / 'main.py'}")
    dev_base = args.cq_base == "dev"
    ensure_ltx25_cq_enhancer_models(comfy_dir, dev_base=dev_base)
    wait_for_comfy(args.comfy_url, timeout_seconds=180, poll_seconds=args.poll_seconds)
    ensure_node_types(args.comfy_url, LTX_IC_LORA_NODE_TYPES, "LTX 2.5 CQ enhancement")
    video_name = copy_to_comfy_input(source, comfy_dir, "arp_upscale_cq")
    prefix = f"arp_upscale/{safe_stem(source.name)}_cq_{width}x{height}"
    prompt = ltx_ic_lora_prompt(
        video_name, CQ_FPS, frames, width, height, prefix,
        unet=cq_base_model(args),
        distilled_lora=LTX25_DISTILLED_LORA if dev_base else "",
        lora=args.cq_lora,
        lora_strength=args.cq_lora_strength,
        text_encoder=args.ltx25_text_encoder,
        vae=args.ltx25_video_vae,
        prompt=args.cq_prompt,
        negative_prompt="",
        guide_strength=args.cq_guide_strength,
        cfg=1.0,
        seed=args.cq_seed,
    )
    print(f"Sending LTX 2.5 CQ enhancer prompt nodes: {sorted(node['class_type'] for node in prompt.values())}", flush=True)
    history = queue_prompt_with_progress(args.comfy_url, prompt, args.poll_seconds, {"15", "17"})
    produced = newest_comfy_output(
        extract_output_files(history, comfy_output_root),
        {".mp4", ".mov", ".mkv", ".webm"},
        "LTX 2.5 CQ enhanced video",
    )
    partial.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, partial)
    return partial


def cq_enhance(args: argparse.Namespace, source: Path, info: dict[str, Any], source_fingerprint: dict[str, Any]) -> Path:
    """Restore the source at the CQ working size; the result keeps the source's frame rate and audio."""
    # Chunks render at the 32 px grid size (width x height); the stitched render is
    # resized to the aspect-correct output size.
    (out_width, out_height), (width, height) = cq_dimensions(args, source)
    target = cq_render_path(source, out_width, out_height)
    # Colour and output size are applied after stitching, so changing them reuses the rendered chunks.
    sig = cq_output_signature(args, source)
    if not args.force and resumable_output(target, sig, video_like=source, width=out_width, height=out_height):
        print(f"Reuse CQ-enhanced render: {target}", flush=True)
        return target
    ffmpeg = find_ffmpeg(args.ffmpeg)
    fps = args.fps or float(info["fps"])
    mode = cq_frame_rate_mode(args, fps)
    shot_cuts = cq_shot_cuts(args, source)
    plan = cq_chunk_plan(int(info["frames"]), fps, args.cq_chunk_seconds, args.overlap_frames, mode, shot_cuts)
    chunk_dir = ROOT / ".cache" / "upscale_chunks" / (
        f"{upscale_chunk_source_key(source)}_ltx25cq_{width}x{height}"
        f"_{int(args.cq_chunk_seconds * 1000)}ms_ov{max(0, args.overlap_frames)}"
    )
    chunk_dir.mkdir(parents=True, exist_ok=True)
    shots = f" across {len(shot_cuts) + 1} shot(s)" if shot_cuts else ""
    print(
        f"Enhancing with LTX 2.5 CQ in {len(plan)} chunk(s){shots}: {info['width']}x{info['height']} -> "
        f"{out_width}x{out_height} (rendered at {width}x{height}) at {CQ_FPS:g} fps ({mode} from {fps:g} fps)",
        flush=True,
    )
    # Within a shot, each chunk keeps its overlap lead-in: both neighbours rendered those frames,
    # and dissolving between the two renders hides the jump between independent generations.
    # A chunk starting a shot has no lead-in and simply cuts, as the source does.
    chunks: list[tuple[Path, int]] = []
    digits = max(4, int(math.log10(len(plan))) + 1)
    for index, (start_frame, end_frame, trim_start, model_frames) in enumerate(plan):
        name = f"{index:0{digits}d}_{start_frame:06d}_{end_frame:06d}"
        chunk_input = chunk_dir / f"cq_input_{name}.mkv"
        chunk_raw = chunk_dir / f"cq_raw_{name}.mp4"
        chunk_final = chunk_dir / f"cq_render_{name}.mkv"
        print(
            f"CQ enhance chunk {index + 1}/{len(plan)}: frames {start_frame}-{end_frame}, "
            f"crossfade {trim_start}, LTX window {model_frames}",
            flush=True,
        )
        prepare_cq_chunk(
            ffmpeg, source, chunk_input, start_frame, end_frame, model_frames, fps,
            width, height, mode, args.force, source_fingerprint,
        )
        chunk_sig = cq_signature(args, chunk_input, width, height)
        if not args.force and resumable_upscale_chunk(chunk_final, chunk_sig, width, height):
            print(f"Reuse CQ enhanced chunk: {chunk_final}", flush=True)
            chunks.append((chunk_final, trim_start))
            continue
        reusable = None if args.force else find_reusable_upscale_chunk(chunk_dir, chunk_final.name, chunk_sig, width, height)
        if reusable:
            shutil.copy2(reusable, chunk_final)
            shutil.copy2(reusable.with_suffix(reusable.suffix + ".sig.json"), chunk_final.with_suffix(chunk_final.suffix + ".sig.json"))
            print(f"Reuse CQ enhanced chunk from compatible cache: {reusable}", flush=True)
            chunks.append((chunk_final, trim_start))
            continue
        cq_run(args, chunk_input, chunk_raw, width, height, model_frames)
        normalize_cq_chunk(ffmpeg, chunk_raw, chunk_final, width, height, 0, end_frame - start_frame, fps, mode)
        write_signature(chunk_final, chunk_sig)
        print(f"Wrote CQ enhanced chunk: {chunk_final}", flush=True)
        chunk_raw.unlink(missing_ok=True)
        chunks.append((chunk_final, trim_start))
    frames = int(info["frames"])
    if args.cq_colour == "source":
        stitched = target.with_suffix(".stitched.mkv")
        crossfade_chunks(ffmpeg, chunks, fps, frames, source, stitched, out_width, out_height)
        restore_source_colour(ffmpeg, stitched, source, target, out_width, out_height, fps, frames)
        stitched.unlink(missing_ok=True)
    else:
        crossfade_chunks(ffmpeg, chunks, fps, frames, source, target, out_width, out_height)
    write_signature(target, sig)
    return target


def cq_output_signature(args: argparse.Namespace, source: Path) -> dict[str, Any]:
    """Identity of the stitched CQ render: the chunk identity plus what is applied after stitching."""
    (out_width, out_height), (width, height) = cq_dimensions(args, source)
    return {
        **cq_signature(args, source, width, height),
        "shot_cuts": cq_shot_cuts(args, source),
        "colour": args.cq_colour,
        "output_width": out_width,
        "output_height": out_height,
    }


def frame_clock(fps: float) -> str:
    """Filters that put a stream on an exact 1/fps timebase, one tick per frame.

    setpts alone leaves millisecond (MKV) and MP4 timebases slightly apart, and filters
    that sync two inputs then repeat frames to cover the gaps.
    """
    rate = Fraction(fps).limit_denominator(1001)
    return f"settb={rate.denominator}/{rate.numerator},setpts=N,setsar=1"


def crossfade_chunks(
    ffmpeg: str,
    chunks: list[tuple[Path, int]],
    fps: float,
    frames: int,
    audio_source: Path,
    output: Path,
    width: int = 0,
    height: int = 0,
) -> None:
    """Join chunk renders, dissolving across each chunk's lead-in instead of cutting.

    A chunk's lead-in repeats the previous chunk's last frames, so a dissolve over them
    keeps every source frame exactly once. A width and height resize the joined video.
    """
    if not chunks:
        raise RuntimeError("No CQ chunks were produced.")
    rate = Fraction(fps).limit_denominator(1001)
    seconds_per_frame = rate.denominator / rate.numerator
    inputs: list[str] = []
    filters = []
    for index, (chunk, _lead) in enumerate(chunks):
        inputs += ["-i", str(chunk)]
        filters.append(f"[{index}:v]{frame_clock(fps)}[c{index}]")
    current, length = "c0", video_info(chunks[0][0])["frames"]
    for index, (chunk, lead) in enumerate(chunks[1:], start=1):
        chunk_frames = video_info(chunk)["frames"]
        joined = f"j{index}"
        if lead > 0:
            filters.append(
                f"[{current}][c{index}]xfade=transition=fade:duration={lead * seconds_per_frame:.9f}:"
                f"offset={(length - lead) * seconds_per_frame:.9f}[{joined}]"
            )
            length += chunk_frames - lead
        else:
            filters.append(f"[{current}][c{index}]concat=n=2:v=1:a=0[{joined}]")
            length += chunk_frames
        current = joined
    if width and height:
        filters.append(f"[{current}]scale={width}:{height}:flags=lanczos,setsar=1[sized]")
        current = "sized"
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".video.partial" + output.suffix)
    print(f"Stitching upscaled chunks: {len(chunks)} chunk(s), crossfading overlaps", flush=True)
    subprocess.run([
        ffmpeg, "-y", *inputs, "-filter_complex", ";".join(filters), "-map", f"[{current}]", "-an",
        "-r", f"{fps:.8f}", "-fps_mode", "cfr", "-frames:v", str(frames),
        *working_codec_args(), *working_container_args(partial), str(partial),
    ], check=True)
    mux_audio(ffmpeg, partial, audio_source, output)
    partial.unlink(missing_ok=True)


def restore_source_colour(
    ffmpeg: str, enhanced: Path, source: Path, output: Path, width: int, height: int, fps: float, frames: int,
) -> None:
    """Keep the CQ render's luma but the source's chroma.

    The LoRA invents colour on black-and-white film and shifts hues on colour film; ARP's
    colour decisions belong to the Colorize stage, so only brightness detail is taken from CQ.
    """
    partial = output.with_suffix(output.suffix + ".colour.partial" + output.suffix)
    # Both inputs need the same exact frame clock, or the merge repeats early frames and
    # delays the picture; mergeplanes also refuses inputs whose sample aspect ratios differ.
    filters = (
        f"[0:v]{frame_clock(fps)},format=yuv444p10le,extractplanes=y[luma];"
        f"[1:v]{frame_clock(fps)},scale={width}:{height}:flags=lanczos,setsar=1,"
        "format=yuv444p10le,extractplanes=u+v[cb][cr];"
        "[luma][cb][cr]mergeplanes=format=yuv444p10le:map0s=0:map0p=0:map1s=1:map1p=0:map2s=2:map2p=0,setsar=1[vout]"
    )
    print(f"Restoring source colour under the CQ render: {enhanced}", flush=True)
    subprocess.run([
        ffmpeg, "-y", "-i", str(enhanced), "-i", str(source), "-filter_complex", filters,
        # Syncing two inputs repeats the last frame to cover stream durations; keep exactly one per source frame.
        "-map", "[vout]", "-an", "-r", f"{fps:.8f}", "-fps_mode", "cfr", "-frames:v", str(frames),
        *working_codec_args(), *working_container_args(partial), str(partial),
    ], check=True)
    mux_audio(ffmpeg, partial, source, output)
    partial.unlink(missing_ok=True)


def lanczos_upscale_run(ffmpeg: str, source: Path, output: Path, width: int, height: int, audio_source: Path) -> None:
    partial = output.with_suffix(output.suffix + ".video.partial" + output.suffix)
    print(f"Lanczos-scaling to {width}x{height}: {source}", flush=True)
    subprocess.run([
        ffmpeg, "-y", "-i", str(source), "-map", "0:v:0", "-vf", f"scale={width}:{height}:flags=lanczos,setsar=1",
        "-an", *working_codec_args(), *working_container_args(partial), str(partial),
    ], check=True)
    mux_audio(ffmpeg, partial, audio_source, output)
    partial.unlink(missing_ok=True)


def chunk_ranges(total_frames: int, fps: float, chunk_seconds: float, overlap_frames: int) -> list[tuple[int, int, int]]:
    if total_frames <= 0 or fps <= 0 or chunk_seconds <= 0:
        return [(0, 0, total_frames)]
    chunk_frames = max(1, int(round(chunk_seconds * fps)))
    if chunk_frames >= total_frames:
        return [(0, 0, total_frames)]
    overlap = max(0, min(int(overlap_frames), chunk_frames - 1))
    ranges: list[tuple[int, int, int]] = []
    start = 0
    while start < total_frames:
        end = min(total_frames, start + chunk_frames)
        source_start = max(0, start - overlap)
        trim_start = start - source_start
        ranges.append((source_start, end, trim_start))
        if end >= total_frames:
            break
        start = end
    return ranges


def ltx25_chunk_ranges(
    total_frames: int, fps: float, chunk_seconds: float, overlap_frames: int
) -> list[tuple[int, int, int, int, int]]:
    """Create LTX-valid 8n+1 windows while preserving every delivery frame once."""
    if total_frames <= 0:
        return []
    if chunk_seconds <= 0:
        model_frames = max(9, int(math.ceil(max(0, total_frames - 1) / 8.0)) * 8 + 1)
    else:
        requested = max(9, int(round(chunk_seconds * max(fps, 0.001))))
        model_frames = max(9, int(round((requested - 1) / 8.0)) * 8 + 1)
    overlap = max(0, min(int(overlap_frames), model_frames - 1))
    ranges: list[tuple[int, int, int, int, int]] = []
    delivery_start = 0
    while delivery_start < total_frames:
        source_start = max(0, delivery_start - overlap)
        source_end = min(total_frames, source_start + model_frames)
        trim_start = delivery_start - source_start
        keep_frames = source_end - delivery_start
        ranges.append((source_start, source_end, trim_start, keep_frames, model_frames))
        if source_end >= total_frames:
            break
        delivery_start = source_end
    return ranges


def split_video_chunk(ffmpeg: str, source: Path, target: Path, start_frame: int, end_frame: int, fps: float, force: bool, source_fingerprint: dict[str, Any]) -> None:
    if target.exists() and not force and split_matches_source(target, source_fingerprint):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial" + target.suffix)
    vf = f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=N/({fps:.8f}*TB),fps={fps:.8f},setsar=1"
    command = [
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
        *working_codec_args(fast=True),
        *working_container_args(partial),
        str(partial),
    ]
    subprocess.run(command, check=True)
    replace_unless_identical(partial, target, f"Upscale prepared chunk {target.name}")
    write_split_sidecar(target, source, source_fingerprint)


def prepare_ltx25_chunk(
    ffmpeg: str,
    source: Path,
    target: Path,
    start_frame: int,
    end_frame: int,
    model_frames: int,
    fps: float,
    width: int,
    height: int,
    force: bool,
    source_fingerprint: dict[str, Any],
) -> None:
    if target.exists() and not force and split_matches_source(target, source_fingerprint):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial" + target.suffix)
    duration = model_frames / max(fps, 0.001)
    vf = (
        f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=N/({fps:.8f}*TB),"
        f"scale={width}:{height}:flags=lanczos,setsar=1,"
        f"tpad=stop_mode=clone:stop_duration={duration:.8f},trim=end_frame={model_frames},fps={fps:.8f}"
    )
    subprocess.run(
        [
            ffmpeg, "-y", "-i", str(source), "-vf", vf, "-an", "-r", f"{fps:.8f}",
            "-fps_mode", "cfr", *working_codec_args(fast=True), *working_container_args(partial), str(partial),
        ],
        check=True,
    )
    replace_unless_identical(partial, target, f"LTX 2.5 prepared chunk {target.name}")
    write_split_sidecar(target, source, source_fingerprint)


def normalize_ltx25_chunk(
    ffmpeg: str,
    source: Path,
    target: Path,
    width: int,
    height: int,
    trim_start: int,
    keep_frames: int,
    fps: float,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial" + target.suffix)
    trim_end = trim_start + keep_frames
    vf = (
        f"trim=start_frame={trim_start}:end_frame={trim_end},setpts=N/({fps:.8f}*TB),"
        f"fps={fps:.8f},scale={width}:{height}:flags=lanczos,setsar=1"
    )
    subprocess.run(
        [
            ffmpeg, "-y", "-i", str(source), "-vf", vf, "-an", "-r", f"{fps:.8f}",
            "-fps_mode", "cfr", *working_codec_args(), *working_container_args(partial), str(partial),
        ],
        check=True,
    )
    replace_with_retry(partial, target, f"LTX 2.5 normalized chunk {target.name}")


def normalize_chunk(ffmpeg: str, source: Path, target: Path, width: int, height: int, trim_start: int, fps: float, force: bool) -> None:
    if target.exists() and not force:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial" + target.suffix)
    filters = []
    if trim_start > 0:
        filters.append(f"trim=start_frame={trim_start}")
    filters.append(f"setpts=N/({fps:.8f}*TB)")
    filters.append(f"fps={fps:.8f}")
    filters.append(f"scale={width}:{height}:flags=lanczos,setsar=1")
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vf",
        ",".join(filters),
        "-an",
        "-r",
        f"{fps:.8f}",
        "-fps_mode",
        "cfr",
        *working_codec_args(),
        *working_container_args(partial),
        str(partial),
    ]
    subprocess.run(command, check=True)
    replace_with_retry(partial, target, f"Upscale normalized chunk {target.name}")


def mux_audio(ffmpeg: str, video_source: Path, audio_source: Path, output: Path) -> None:
    partial = output.with_suffix(output.suffix + ".mux.partial" + output.suffix)
    print("Muxing original audio into upscaled video", flush=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_source),
        "-i",
        str(audio_source),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        *working_container_args(partial),
        str(partial),
    ]
    subprocess.run(command, check=True)
    replace_with_retry(partial, output, "Upscaled output")


def stitch_chunks(ffmpeg: str, chunks: list[Path], audio_source: Path, output: Path) -> None:
    if not chunks:
        raise RuntimeError("No upscale chunks were produced.")
    output.parent.mkdir(parents=True, exist_ok=True)
    video_partial = output.with_suffix(output.suffix + ".video.partial" + output.suffix)
    with tempfile.TemporaryDirectory(prefix="arp_upscale_concat_") as tmp_text:
        list_file = Path(tmp_text) / "chunks.txt"
        list_file.write_text("".join(f"file '{chunk.as_posix()}'\n" for chunk in chunks), encoding="utf-8")
        print(f"Stitching upscaled chunks: {len(chunks)} chunk(s)", flush=True)
        subprocess.run([
            ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-an", *working_codec_args(), *working_container_args(video_partial), str(video_partial),
        ], check=True)
    mux_audio(ffmpeg, video_partial, audio_source, output)
    video_partial.unlink(missing_ok=True)


def chunked_standard_upscale_run(
    args: argparse.Namespace,
    source: Path,
    output: Path,
    output_width: int,
    output_height: int,
    info: dict[str, Any],
    source_fingerprint: dict[str, Any],
    method: str,
    audio_source: Path | None = None,
) -> None:
    if method not in {"flashvsr", "seedvr2"}:
        raise ValueError(f"Unsupported standard upscale method: {method}")
    runner = flashvsr_run if method == "flashvsr" else seedvr2_run
    label = "FlashVSR" if method == "flashvsr" else "SeedVR2"
    ffmpeg = find_ffmpeg(args.ffmpeg)
    audio_source = audio_source or source
    fps = args.fps or float(info["fps"])
    ranges = chunk_ranges(int(info["frames"]), fps, args.chunk_seconds, args.overlap_frames)
    if len(ranges) <= 1:
        raw_partial = output.with_suffix(output.suffix + f".{method}.partial" + output.suffix)
        final_partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
        for path in (raw_partial, final_partial):
            if path.exists():
                path.unlink()
        print(f"Upscale chunk 1/1: frames 0-{int(info['frames'])}, trim 0", flush=True)
        print(f"Queueing {label} in ComfyUI: {source}", flush=True)
        runner(args, source, raw_partial, output_width, output_height)
        if not raw_partial.exists():
            raise RuntimeError(f"{label} finished but did not create expected output: {raw_partial}")
        raw_info = video_info(raw_partial)
        if raw_info["width"] == output_width and raw_info["height"] == output_height:
            replace_with_retry(raw_partial, final_partial, "Upscaled preview")
        else:
            scale_video(ffmpeg, raw_partial, final_partial, output_width, output_height)
            raw_partial.unlink(missing_ok=True)
        print(f"Wrote upscaled chunk: {final_partial}", flush=True)
        if output.exists():
            output.unlink()
        mux_audio(ffmpeg, final_partial, audio_source, output)
        final_partial.unlink(missing_ok=True)
        return

    chunk_dir = upscale_chunk_dir(args, source, output_width, output_height)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    print(f"Splitting upscaling into {len(ranges)} chunk(s): {args.chunk_seconds:g}s chunks, {max(0, args.overlap_frames)} overlap frame(s)", flush=True)
    normalized_chunks: list[Path] = []
    digits = max(4, int(math.log10(len(ranges))) + 1)
    for index, (start_frame, end_frame, trim_start) in enumerate(ranges):
        chunk_input = chunk_dir / f"input_{index:0{digits}d}_{start_frame:06d}_{end_frame:06d}.mkv"
        chunk_raw = chunk_dir / f"raw_{index:0{digits}d}_{start_frame:06d}_{end_frame:06d}.mp4"
        chunk_final = chunk_dir / f"final_{index:0{digits}d}_{start_frame:06d}_{end_frame:06d}.mkv"
        print(f"Upscale chunk {index + 1}/{len(ranges)}: frames {start_frame}-{end_frame}, trim {trim_start}", flush=True)
        split_video_chunk(ffmpeg, source, chunk_input, start_frame, end_frame, fps, args.force, source_fingerprint)
        chunk_sig = signature(args, chunk_input, output_width, output_height)
        if not args.force and resumable_upscale_chunk(chunk_final, chunk_sig, output_width, output_height):
            print(f"Reuse upscaled chunk: {chunk_final}", flush=True)
            normalized_chunks.append(chunk_final)
            continue
        reusable = None if args.force else find_reusable_upscale_chunk(chunk_dir, chunk_final.name, chunk_sig, output_width, output_height)
        if reusable:
            shutil.copy2(reusable, chunk_final)
            shutil.copy2(reusable.with_suffix(reusable.suffix + ".sig.json"), chunk_final.with_suffix(chunk_final.suffix + ".sig.json"))
            print(f"Reuse upscaled chunk from compatible cache: {reusable}", flush=True)
            normalized_chunks.append(chunk_final)
            continue
        runner(args, chunk_input, chunk_raw, output_width, output_height)
        normalize_chunk(ffmpeg, chunk_raw, chunk_final, output_width, output_height, trim_start, fps, True)
        write_signature(chunk_final, chunk_sig)
        print(f"Wrote upscaled chunk: {chunk_final}", flush=True)
        chunk_raw.unlink(missing_ok=True)
        normalized_chunks.append(chunk_final)
    if output.exists():
        output.unlink()
    stitch_chunks(ffmpeg, normalized_chunks, audio_source, output)


def chunked_ltx25_run(
    args: argparse.Namespace,
    source: Path,
    output: Path,
    output_width: int,
    output_height: int,
    info: dict[str, Any],
    source_fingerprint: dict[str, Any],
) -> None:
    ffmpeg = find_ffmpeg(args.ffmpeg)
    fps = args.fps or float(info["fps"])
    generation_width, generation_height = ltx25_generation_dimensions(output_width, output_height)
    reference_width, reference_height = generation_width // 2, generation_height // 2
    ranges = ltx25_chunk_ranges(
        int(info["frames"]), fps, args.chunk_seconds, args.overlap_frames
    )
    if not ranges:
        raise RuntimeError("The source has no frames to upscale.")
    chunk_dir = upscale_chunk_dir(args, source, output_width, output_height)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    requested_note = (
        ""
        if (generation_width, generation_height) == (output_width, output_height)
        else f" (nearest model-safe size to requested {output_width}x{output_height})"
    )
    print(
        f"Splitting LTX 2.5 upscaling into {len(ranges)} chunk(s): "
        f"{reference_width}x{reference_height} reference -> "
        f"{generation_width}x{generation_height} native delivery{requested_note}",
        flush=True,
    )
    normalized_chunks: list[Path] = []
    digits = max(4, int(math.log10(len(ranges))) + 1)
    for index, (start_frame, end_frame, trim_start, keep_frames, model_frames) in enumerate(ranges):
        chunk_input = chunk_dir / f"ltx25_input_{index:0{digits}d}_{start_frame:06d}_{end_frame:06d}.mkv"
        chunk_raw = chunk_dir / f"ltx25_raw_{index:0{digits}d}_{start_frame:06d}_{end_frame:06d}.mp4"
        chunk_final = chunk_dir / f"ltx25_final_{index:0{digits}d}_{start_frame:06d}_{end_frame:06d}.mkv"
        print(
            f"Upscale chunk {index + 1}/{len(ranges)}: frames {start_frame}-{end_frame}, "
            f"trim {trim_start}, keep {keep_frames}, LTX window {model_frames}",
            flush=True,
        )
        prepare_ltx25_chunk(
            ffmpeg, source, chunk_input, start_frame, end_frame, model_frames, fps,
            reference_width, reference_height, args.force, source_fingerprint,
        )
        chunk_sig = signature(args, chunk_input, output_width, output_height)
        if not args.force and resumable_upscale_chunk(
            chunk_final, chunk_sig, generation_width, generation_height
        ):
            print(f"Reuse LTX 2.5 upscaled chunk: {chunk_final}", flush=True)
            normalized_chunks.append(chunk_final)
            continue
        reusable = None if args.force else find_reusable_upscale_chunk(
            chunk_dir, chunk_final.name, chunk_sig, generation_width, generation_height
        )
        if reusable:
            shutil.copy2(reusable, chunk_final)
            shutil.copy2(
                reusable.with_suffix(reusable.suffix + ".sig.json"),
                chunk_final.with_suffix(chunk_final.suffix + ".sig.json"),
            )
            print(f"Reuse LTX 2.5 chunk from compatible cache: {reusable}", flush=True)
            normalized_chunks.append(chunk_final)
            continue
        ltx25_run(args, chunk_input, chunk_raw, generation_width, generation_height)
        normalize_ltx25_chunk(
            ffmpeg, chunk_raw, chunk_final, generation_width, generation_height,
            trim_start, keep_frames, fps,
        )
        write_signature(chunk_final, chunk_sig)
        print(f"Wrote LTX 2.5 upscaled chunk: {chunk_final}", flush=True)
        chunk_raw.unlink(missing_ok=True)
        normalized_chunks.append(chunk_final)
    if output.exists():
        output.unlink()
    stitch_chunks(ffmpeg, normalized_chunks, source, output)


def fit_dimensions(source_width: int, source_height: int, target_width: int, target_height: int) -> tuple[int, int]:
    if target_width <= 0 and target_height <= 0:
        return source_width * 4, source_height * 4
    if target_width <= 0:
        target_width = round(target_height * source_width / source_height)
    if target_height <= 0:
        target_height = round(target_width * source_height / source_width)
    return max(2, target_width // 2 * 2), max(2, target_height // 2 * 2)


def scale_video(ffmpeg: str, source: Path, output: Path, width: int, height: int) -> None:
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vf",
        f"scale={width}:{height}:flags=lanczos,setsar=1",
        *working_codec_args(),
        *working_container_args(output),
        str(output),
    ]
    subprocess.run(command, check=True)


def pre_downscale_dimensions(output_width: int, output_height: int, flashvsr_scale: int) -> tuple[int, int]:
    scale = max(1, int(flashvsr_scale or 1))
    multiple = 128
    aspect = output_width / max(1, output_height)
    min_h_units = max(1, math.ceil(output_height / multiple))
    candidates: list[tuple[float, int, int, int]] = []
    for h_units in range(min_h_units, min_h_units + 32):
        raw_height = h_units * multiple
        raw_width = math.ceil((raw_height * aspect) / multiple) * multiple
        if raw_width < output_width:
            raw_width = math.ceil(output_width / multiple) * multiple
        aspect_error = abs((raw_width / raw_height) - aspect) / aspect
        candidates.append((aspect_error, raw_width * raw_height, raw_width, raw_height))
    _error, _area, raw_width, raw_height = min(candidates, key=lambda item: (item[0], item[1]))
    width = max(2, math.ceil(raw_width / scale))
    height = max(2, math.ceil(raw_height / scale))
    width = width + (width % 2)
    height = height + (height % 2)
    return width, height


def pre_downscale_source(ffmpeg: str, args: argparse.Namespace, source: Path, output_width: int, output_height: int, source_info: dict[str, Any]) -> Path:
    width, height = pre_downscale_dimensions(output_width, output_height, args.flashvsr_scale)
    target = ROOT / ".cache" / "upscale_sources" / f"{safe_stem(source.name)}_flashvsr_input_{width}x{height}_s{args.flashvsr_scale}.mkv"
    sig = {
        "version": 1,
        "tool": "upscale_video.py",
        "method": "flashvsr_pre_downscale",
        "source": root_relative(source),
        "source_fingerprint": file_fingerprint(source),
        "target_width": output_width,
        "target_height": output_height,
        "flashvsr_scale": args.flashvsr_scale,
        "processing_width": width,
        "processing_height": height,
        "fps": args.fps,
    }
    if not args.force and resumable_output(target, sig, video_like=source, width=width, height=height):
        print(f"Reuse FlashVSR pre-downscaled input: {target}", flush=True)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial" + target.suffix)
    fps = args.fps or float(source_info["fps"])
    print(f"Pre-downscaling FlashVSR input to {width}x{height}: {source}", flush=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vf",
        f"scale={width}:{height}:flags=lanczos,setsar=1,fps={fps:.8f}",
        "-an",
        "-r",
        f"{fps:.8f}",
        "-fps_mode",
        "cfr",
        *working_codec_args(),
        *working_container_args(partial),
        str(partial),
    ]
    subprocess.run(command, check=True)
    replace_with_retry(partial, target, "FlashVSR pre-downscaled input")
    write_signature(target, sig)
    return target


BACKEND_LABELS = {
    "flashvsr": "FlashVSR",
    "seedvr2": "SeedVR2",
    "ltx25": "LTX 2.5 Pixel Spatial IC-LoRA",
    "lanczos": "Lanczos",
}


def run(args: argparse.Namespace) -> int:
    global CURRENT_INTERMEDIATE_PROFILE
    CURRENT_INTERMEDIATE_PROFILE = getattr(args, "intermediate_profile", "high")
    source = resolve_path(args.input)
    if not source.exists():
        raise FileNotFoundError(f"Input video not found for upscaling: {source}")

    info = video_info(source)
    output_width, output_height = fit_dimensions(int(info["width"]), int(info["height"]), args.target_width, args.target_height)
    method = upscale_method(args)
    # A CQ-enhanced render is finished by one of the other backends, with that backend's settings.
    backend = cq_finish_method(args) if method == "ltx25cq" else method
    backend_args = finish_args(args) if method == "ltx25cq" else args
    if backend == "seedvr2":
        if args.seedvr2_batch_size < 1 or (args.seedvr2_batch_size - 1) % 4 != 0:
            raise ValueError("SeedVR2 batch size must use the 4n+1 sequence: 1, 5, 9, 13, ...")
        if args.seedvr2_vae_tile_overlap >= args.seedvr2_vae_tile_size:
            raise ValueError("SeedVR2 VAE tile overlap must be smaller than the tile size.")
    delivery_width, delivery_height = delivery_dimensions(args, output_width, output_height)
    output = resolve_path(args.output) if args.output else default_output(source, output_width, output_height, method)
    sig = signature(args, source, output_width, output_height)
    manifest = resolve_path(args.shot_manifest) if args.shot_manifest else None
    final_sig = dict(sig)
    final_sig.update({
        "version": 7,
        "blend_strength": float(args.blend_strength),
        "shot_manifest": root_relative(manifest) if manifest else "",
        "shot_manifest_fingerprint": file_fingerprint(manifest) if manifest and manifest.is_file() else None,
    })

    if not args.force and resumable_output(
        output, final_sig, video_like=source, width=delivery_width, height=delivery_height
    ):
        print(f"Reuse upscaled video: {output}", flush=True)
        return 0
    if args.dry_run:
        if method == "ltx25cq":
            (enhance_width, enhance_height), (render_width, render_height) = cq_dimensions(args, source)
            print(
                f"Would enhance with LTX 2.5 CQ to {enhance_width}x{enhance_height} "
                f"(rendered at {render_width}x{render_height}), {CQ_FPS:g} fps",
                flush=True,
            )
        if backend == "flashvsr" and args.flashvsr_pre_downscale:
            processing_width, processing_height = pre_downscale_dimensions(output_width, output_height, args.flashvsr_scale)
            print(f"Would pre-downscale FlashVSR input to {processing_width}x{processing_height}", flush=True)
        print(
            f"Would upscale {source} -> {output} at {delivery_width}x{delivery_height} "
            f"using {BACKEND_LABELS[backend]} in ComfyUI at {args.comfy_url} ({args.comfy_dir})",
            flush=True,
        )
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    ai_output = ROOT / ".cache" / "upscale_ai" / f"{safe_stem(output.stem)}_{method}.mkv"
    ai_output.parent.mkdir(parents=True, exist_ok=True)
    if args.force or not resumable_output(
        ai_output, sig, video_like=source, width=delivery_width, height=delivery_height
    ):
        processing_source = source
        processing_info = info
        processing_fingerprint = sig["source_fingerprint"]
        if method == "ltx25cq":
            processing_source = cq_enhance(args, source, info, sig["source_fingerprint"])
            processing_info = video_info(processing_source)
            processing_fingerprint = file_fingerprint(processing_source)
            print(f"Finishing CQ-enhanced video with {BACKEND_LABELS[backend]}: {processing_source}", flush=True)
        if backend == "flashvsr" and args.flashvsr_pre_downscale:
            processing_source = pre_downscale_source(
                find_ffmpeg(args.ffmpeg), backend_args, processing_source, output_width, output_height, processing_info,
            )
            processing_info = video_info(processing_source)
            processing_fingerprint = file_fingerprint(processing_source)
        if backend == "ltx25":
            chunked_ltx25_run(
                backend_args, processing_source, ai_output, output_width, output_height, processing_info,
                processing_fingerprint,
            )
        elif backend == "lanczos":
            lanczos_upscale_run(find_ffmpeg(args.ffmpeg), processing_source, ai_output, output_width, output_height, source)
        else:
            chunked_standard_upscale_run(
                backend_args, processing_source, ai_output, output_width, output_height,
                processing_info, processing_fingerprint, backend, audio_source=source,
            )
        write_signature(ai_output, sig)
    else:
        print(f"Reuse full-strength {method} render: {ai_output}", flush=True)
    ffmpeg = find_ffmpeg(args.ffmpeg)
    default_strength = normalized_blend_strength(args.blend_strength)
    shots = read_upscale_shots(manifest, int(info["frames"]), float(info["fps"]), default_strength)
    strengths = [float(shot["strength"]) for shot in shots] or [default_strength]
    if all(abs(strength - 1.0) < 1e-9 for strength in strengths):
        encode_upscale_delivery(ffmpeg, ai_output, output)
    else:
        blend_upscale_delivery(
            ffmpeg, source, ai_output, output, delivery_width, delivery_height,
            float(info["fps"]), shots, default_strength,
        )
    write_signature(output, final_sig)
    print(f"Wrote upscaled video: {output}", flush=True)
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
