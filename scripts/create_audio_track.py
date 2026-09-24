#!/usr/bin/env python3
"""Create an audio track (music score and/or synchronized sound effects) for a silent film
and mux it onto the supplied render without re-encoding the video.

Pipeline:
  1. Probe the input video (duration / fps).
  2. Music (optional): detect scenes, caption a representative frame per scene with a local
     Qwen-VL node, generate a Stable Audio cue per scene, trim/pad each to its scene length,
     and concatenate into a full-length music stem.
  3. SFX (optional): split the video into short windows, build a low-res proxy (MMAudio gains
     nothing above ~384px on the short side), run MMAudio per window, and concatenate into a
     full-length effects stem.
  4. Mix the stems (music ducked under SFX via sidechain compression).
  5. Mux the mixed audio onto the input video with ``-c:v copy`` (no video re-encode).

The model steps run through ComfyUI (see ``audio_models.py``). The ffmpeg orchestration here
is self-contained.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from comfy_api import wait_for_comfy
from dependency_manager import ensure_audio_models
from common import (
    ROOT,
    file_fingerprint,
    find_ffmpeg,
    load_local_config,
    resolve_path,
    resumable_output,
    root_relative,
    safe_stem,
    video_info,
    write_signature,
)

import audio_models

config = load_local_config()

DEFAULT_MUSIC_PROMPT = "Gentle cinematic orchestral score, melodic, instrumental, period-appropriate, soft dynamics."
DEFAULT_SFX_PROMPT = "Natural ambient sound and foley matching the on-screen action."

# MMAudio's Synchformer branch consumes frames at exactly 25 fps and its node slices
# the tensor by frame count, not timestamps; proxies must be retimed to this rate or
# the generated audio is time-warped and truncated to frames/25 seconds.
MMAUDIO_FPS = 25


# ── ffmpeg helpers ────────────────────────────────────────────────────────────


def run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def extract_frame(ffmpeg: str, source: Path, seconds: float, target: Path, short_side: int = 512) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        ffmpeg, "-y", "-ss", f"{max(0.0, seconds):.3f}", "-i", str(source),
        "-frames:v", "1", "-update", "1",
        "-vf", f"scale=-2:'min({short_side},ih)':flags=lanczos",
        "-q:v", "2", str(target),
    ])
    return target


def make_sfx_proxy(ffmpeg: str, source: Path, start: float, duration: float, short_side: int, target: Path) -> Path:
    """A muted, low-resolution, 25fps chunk (short side capped at ``short_side``) for MMAudio."""
    target.parent.mkdir(parents=True, exist_ok=True)
    short = max(64, int(short_side) // 2 * 2)
    scale = f"scale='if(gt(iw,ih),-2,{short})':'if(gt(iw,ih),{short},-2)':flags=bicubic"
    run_ffmpeg([
        ffmpeg, "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source),
        "-an", "-vf", f"{scale},fps={MMAUDIO_FPS}", "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
    ])
    return target


def fit_cue_to_length(ffmpeg: str, raw: Path, length: float, target: Path) -> Path:
    """Trim/pad a generated audio cue to exactly ``length`` seconds with short fades."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fade_out_start = max(0.0, length - 0.5)
    af = (
        f"apad=whole_dur={length:.3f},atrim=end={length:.3f},asetpts=N/SR/TB,"
        f"afade=t=in:st=0:d=0.3,afade=t=out:st={fade_out_start:.3f}:d=0.5"
    )
    run_ffmpeg([
        ffmpeg, "-y", "-i", str(raw), "-af", af,
        "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(target),
    ])
    return target


def normalize_pcm(ffmpeg: str, raw: Path, target: Path, length: float | None = None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-i", str(raw)]
    if length is not None:
        cmd += ["-af", f"apad=whole_dur={length:.3f},atrim=end={length:.3f},asetpts=N/SR/TB"]
    cmd += ["-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(target)]
    run_ffmpeg(cmd)
    return target


def concat_audio(ffmpeg: str, parts: list[Path], target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if len(parts) == 1:
        normalize_pcm(ffmpeg, parts[0], target)
        return target
    with tempfile.TemporaryDirectory(prefix="arp_audio_concat_") as tmp:
        list_file = Path(tmp) / "parts.txt"
        list_file.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
        run_ffmpeg([
            ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(target),
        ])
    return target


def mix_stems(ffmpeg: str, music: Path | None, sfx: Path | None, music_db: float, sfx_db: float, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if music and sfx:
        filt = (
            f"[0:a]aformat=channel_layouts=stereo:sample_rates=44100,volume={music_db}dB[m];"
            f"[1:a]aformat=channel_layouts=stereo:sample_rates=44100,volume={sfx_db}dB[s];"
            f"[s]asplit=2[s_out][s_sc];"
            f"[m][s_sc]sidechaincompress=threshold=0.03:ratio=6:attack=20:release=350[mduck];"
            f"[mduck][s_out]amix=inputs=2:duration=longest:normalize=0[mix]"
        )
        run_ffmpeg([
            ffmpeg, "-y", "-i", str(music), "-i", str(sfx),
            "-filter_complex", filt, "-map", "[mix]",
            "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(target),
        ])
    else:
        stem = music or sfx
        gain = music_db if music else sfx_db
        run_ffmpeg([
            ffmpeg, "-y", "-i", str(stem),
            "-af", f"aformat=channel_layouts=stereo:sample_rates=44100,volume={gain}dB",
            "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(target),
        ])
    return target


def mux_audio(ffmpeg: str, video: Path, audio: Path, output: Path) -> Path:
    """Attach the audio track, copying the video stream so it is never re-encoded."""
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.stem}.partial{output.suffix}")
    run_ffmpeg([
        ffmpeg, "-y", "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
        "-shortest", "-movflags", "+faststart", str(partial),
    ])
    if output.exists():
        output.unlink()
    partial.replace(output)
    return output


# ── Scene detection (lightweight, self-contained) ─────────────────────────────


def detect_scenes(source: Path, duration: float, cue_seconds: float, min_len: float | None = None) -> list[tuple[float, float]]:
    """Return contiguous (start, end) scene spans tiling [0, duration].

    Samples frames ~every 1.5s, places a boundary on large colour-histogram jumps, then
    enforces a minimum/maximum cue length so music does not change too often or drift.
    ``min_len`` overrides the music-oriented minimum span (SFX windows can be shorter).
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0 or duration <= 0:
        cap.release()
        return [(0.0, max(0.1, duration))]

    step = max(1, int(round(1.5 * fps)))
    threshold = 0.45
    min_len = max(6.0, cue_seconds * 0.5) if min_len is None else max(1.0, float(min_len))
    max_len = max(min_len + 1.0, cue_seconds)

    boundaries = [0.0]
    prev_hist = None
    last_boundary = 0.0
    idx = 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype("float32")
        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            dist = 1.0 if math.isnan(corr) else max(0.0, 1.0 - corr)
            if dist >= threshold and (t - last_boundary) >= min_len:
                boundaries.append(t)
                last_boundary = t
        prev_hist = hist
        idx += step
    cap.release()

    # Build spans, then split any span longer than max_len into roughly equal cues.
    edges = sorted(set(boundaries + [duration]))
    spans: list[tuple[float, float]] = []
    for start, end in zip(edges, edges[1:]):
        if end - start <= 0:
            continue
        length = end - start
        if length <= max_len:
            spans.append((start, end))
            continue
        pieces = max(1, int(math.ceil(length / max_len)))
        piece = length / pieces
        for i in range(pieces):
            spans.append((start + i * piece, start + (i + 1) * piece if i < pieces - 1 else end))
    # Merge any too-short span into its neighbour so cues never become degenerate.
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if merged and (end - start) < 2.0:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    if len(merged) >= 2 and (merged[0][1] - merged[0][0]) < 2.0:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)
    return merged or [(0.0, duration)]


# ── Stage builders ────────────────────────────────────────────────────────────


MUSIC_CAPTION_QUESTION = (
    "Describe this film scene in ONE short phrase covering mood, setting, and era "
    "(e.g. 'tense, shadowy industrial laboratory, 1920s'). Answer with the phrase only - "
    "no song or artist suggestions, no lists, no explanations."
)
SFX_CAPTION_QUESTION = (
    "You are a foley artist. List the concrete sounds this film moment would produce, as one short "
    "comma-separated line (e.g. 'heavy machinery clanking, steam hiss, footsteps on metal'). "
    "Name sounds caused by visible movement and action first, then background ambience. "
    "Only sounds - no music, no dialogue, no commentary."
)


def resolve_captioner(args: argparse.Namespace):
    """Return (label, fn(image_path, question) -> str) for the best available caption backend.

    Preference order: the user-specified ComfyUI node, then a vision model on the local
    Ollama server. Returns None when neither is available (captioning is skipped).
    """
    if args.caption_node:
        def comfy_captioner(image_path: Path, question: str) -> str:
            return audio_models.run_caption(
                args.comfy_url, resolve_path(args.comfy_dir),
                image_path=image_path, node_class=args.caption_node,
                question=question, poll_seconds=args.poll_seconds,
            )
        return f"ComfyUI node '{args.caption_node}'", comfy_captioner
    choice = (args.ollama_vision_model or "").strip()
    if choice.lower() in ("off", "none"):
        return None
    model = audio_models.pick_ollama_vision_model(args.ollama_url) if choice.lower() in ("", "auto") else choice
    if not model:
        return None

    def ollama_captioner(image_path: Path, question: str) -> str:
        return audio_models.run_ollama_caption(args.ollama_url, model, image_path=image_path, question=question)

    return f"Ollama model '{model}'", ollama_captioner


def combine_music_prompt(caption: str, style_hint: str) -> str:
    parts = [p.strip() for p in (caption, style_hint) if p and p.strip()]
    if not parts:
        return DEFAULT_MUSIC_PROMPT
    body = ". ".join(parts)
    return f"{body}. Instrumental score, no vocals."


def combine_sfx_prompt(caption: str, style_hint: str) -> str:
    parts = [p.strip() for p in (caption, style_hint) if p and p.strip()]
    return ". ".join(parts) if parts else DEFAULT_SFX_PROMPT


def caption_midpoint_frame(ffmpeg: str, source: Path, start: float, end: float, target: Path, captioner, question: str) -> str:
    """Best-effort caption of the span's middle frame; returns "" when unavailable/failed."""
    if captioner is None:
        return ""
    label, fn = captioner
    try:
        frame = extract_frame(ffmpeg, source, (start + end) / 2.0, target)
        return fn(frame, question)
    except Exception as exc:
        print(f"Warning: captioning failed ({label}: {exc}); using the prompt hint instead.", flush=True)
        return ""


def build_music_stem(args: argparse.Namespace, ffmpeg: str, source: Path, duration: float, work_dir: Path, captioner) -> Path:
    print("Detecting scenes for the music score", flush=True)
    cue_seconds = float(args.music_cue_seconds)
    if cue_seconds > 46.0:
        # Stable Audio Open 1.0 generates at most ~47s of audio; each cue is composed at
        # scene length + 1s of trim headroom, so scenes must stay within 46s.
        print(f"Note: capping music cue length at 46s (Stable Audio Open's limit); {cue_seconds:.0f}s was requested.", flush=True)
        cue_seconds = 46.0
    scenes = detect_scenes(source, duration, cue_seconds)
    print(f"Detected {len(scenes)} music scene(s).", flush=True)
    cue_dir = work_dir / "music_cues"
    cue_dir.mkdir(parents=True, exist_ok=True)
    fixed_cues: list[Path] = []
    for index, (start, end) in enumerate(scenes):
        length = max(1.0, end - start)
        if captioner is not None:
            print(f"Captioning scene {index + 1}/{len(scenes)}", flush=True)
        caption = caption_midpoint_frame(
            ffmpeg, source, start, end, cue_dir / f"scene_{index:04d}.png", captioner, MUSIC_CAPTION_QUESTION
        )
        if caption:
            print(f"  scene {index + 1} caption: {caption}", flush=True)
        prompt = combine_music_prompt(caption, args.music_prompt)
        print(f"Composing music cue {index + 1}/{len(scenes)} ({length:.1f}s)", flush=True)
        raw = audio_models.run_music_cue(
            args.comfy_url, resolve_path(args.comfy_output_root),
            checkpoint=args.music_checkpoint, text_encoder=args.music_text_encoder,
            prompt=prompt, negative=args.music_negative,
            seconds=length + 1.0, steps=args.music_steps, cfg=args.music_cfg,
            seed=args.seed + index, prefix=f"arp_audio/music_{safe_stem(source.name)}_{index:04d}",
            poll_seconds=args.poll_seconds,
        )
        fixed_cues.append(fit_cue_to_length(ffmpeg, raw, length, cue_dir / f"cue_{index:04d}.wav"))
    stem = work_dir / "music_stem.wav"
    concat_audio(ffmpeg, fixed_cues, stem)
    print(f"Wrote music stem: {root_relative(stem)}", flush=True)
    return stem


def build_sfx_stem(args: argparse.Namespace, ffmpeg: str, source: Path, duration: float, work_dir: Path, captioner) -> Path:
    chunk = max(2.0, float(args.sfx_chunk_seconds))
    if chunk > 12.0:
        print(
            f"Warning: SFX chunks of {chunk:.0f}s are far beyond MMAudio's 8s training length; "
            "effects tend to dissolve into vague ambience. 8-10s chunks track the picture best.",
            flush=True,
        )
    if (args.sfx_scene_align or "on").lower() != "off":
        # Align window boundaries to scene cuts so MMAudio never has to average two unrelated
        # shots; long scenes are still tiled into <= chunk-second pieces by detect_scenes.
        windows = detect_scenes(source, duration, chunk, min_len=max(2.0, chunk / 4.0))
        print(f"Aligned {len(windows)} SFX window(s) to scene cuts.", flush=True)
    else:
        count = max(1, int(math.ceil(duration / chunk)))
        windows = [(i * chunk, min((i + 1) * chunk, duration)) for i in range(count) if duration - i * chunk > 0.05]
        print(f"Tiling {len(windows)} fixed SFX window(s).", flush=True)
    proxy_dir = work_dir / "sfx_chunks"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    for index, (start, end) in enumerate(windows):
        length = max(1.0, end - start)
        if captioner is not None:
            print(f"Captioning SFX window {index + 1}/{len(windows)}", flush=True)
        caption = caption_midpoint_frame(
            ffmpeg, source, start, end, proxy_dir / f"frame_{index:04d}.png", captioner, SFX_CAPTION_QUESTION
        )
        if caption:
            print(f"  window {index + 1} sounds: {caption}", flush=True)
        prompt = combine_sfx_prompt(caption, args.sfx_prompt)
        print(f"Preparing SFX proxy chunk {index + 1}/{len(windows)}", flush=True)
        proxy = make_sfx_proxy(ffmpeg, source, start, length, int(args.sfx_short_side), proxy_dir / f"proxy_{index:04d}.mp4")
        print(f"Generating SFX chunk {index + 1}/{len(windows)} (MMAudio, {length:.1f}s)", flush=True)
        raw = audio_models.run_sfx_chunk(
            args.comfy_url, resolve_path(args.comfy_dir), resolve_path(args.comfy_output_root),
            proxy_video=proxy, prompt=prompt, negative=args.sfx_negative,
            seconds=length, steps=args.sfx_steps, cfg=args.sfx_cfg,
            seed=args.seed + index, prefix=f"arp_audio/sfx_{safe_stem(source.name)}_{index:04d}",
            poll_seconds=args.poll_seconds,
        )
        parts.append(normalize_pcm(ffmpeg, raw, proxy_dir / f"sfx_{index:04d}.wav", length=length))
    stem = work_dir / "sfx_stem.wav"
    concat_audio(ffmpeg, parts, stem)
    print(f"Wrote sfx stem: {root_relative(stem)}", flush=True)
    return stem


# ── Driver ────────────────────────────────────────────────────────────────────


def ensure_music_checkpoint_file(comfy_dir: Path, checkpoint: str) -> None:
    target = comfy_dir / "models" / "checkpoints" / checkpoint
    if target.exists():
        return
    raise FileNotFoundError(
        f"Stable Audio checkpoint is missing: {target}. "
        "Open https://huggingface.co/stabilityai/stable-audio-open-1.0 in your browser, sign in, accept the gated licence, "
        "authenticate with 'hf auth login' or HF_TOKEN, then retry so ARP can download it. "
        "You can also place the downloaded model.safetensors at that path manually."
    )


def ensure_music_text_encoder_file(comfy_dir: Path, text_encoder: str) -> None:
    target = comfy_dir / "models" / "text_encoders" / text_encoder
    if target.exists():
        return
    raise FileNotFoundError(
        f"Stable Audio text encoder is missing: {target}. "
        "ARP normally downloads this public T5-base file automatically from google-t5/t5-base. "
        "Check network access and retry, or place google-t5/t5-base model.safetensors at that path manually."
    )


def signature(args: argparse.Namespace, source: Path, caption_backend: str = "") -> dict[str, Any]:
    return {
        "version": 2,
        "tool": "create_audio_track.py",
        "caption_backend": caption_backend,
        "source": root_relative(source),
        "source_fingerprint": file_fingerprint(source),
        "music": bool(args.music),
        "sfx": bool(args.sfx),
        "music_prompt": args.music_prompt,
        "music_negative": args.music_negative,
        "music_cue_seconds": args.music_cue_seconds,
        "music_checkpoint": args.music_checkpoint,
        "music_text_encoder": args.music_text_encoder,
        "music_steps": args.music_steps,
        "music_cfg": args.music_cfg,
        "sfx_prompt": args.sfx_prompt,
        "sfx_negative": args.sfx_negative,
        "sfx_chunk_seconds": args.sfx_chunk_seconds,
        "sfx_scene_align": (args.sfx_scene_align or "on").lower(),
        "sfx_short_side": args.sfx_short_side,
        "sfx_steps": args.sfx_steps,
        "sfx_cfg": args.sfx_cfg,
        "music_gain_db": args.music_gain_db,
        "sfx_gain_db": args.sfx_gain_db,
        "seed": args.seed,
        "caption_node": args.caption_node,
    }


def run(args: argparse.Namespace) -> int:
    if not args.music and not args.sfx:
        raise SystemExit("Nothing to do: enable --music and/or --sfx.")
    source = resolve_path(args.input)
    if not source.exists():
        raise FileNotFoundError(f"Input video not found for soundtrack: {source}")
    output = resolve_path(args.output)
    captioner = resolve_captioner(args)
    caption_backend = captioner[0] if captioner else ""
    sig = signature(args, source, caption_backend)
    if not args.force and resumable_output(output, sig, video_like=source):
        print(f"Reuse soundtrack video: {output}", flush=True)
        return 0
    if args.dry_run:
        print(f"Would create a soundtrack for {source} -> {output} (music={args.music}, sfx={args.sfx})", flush=True)
        return 0
    if captioner:
        print(f"Scene captioning: {caption_backend}", flush=True)
    else:
        print("Scene captioning unavailable (no caption node configured and no Ollama vision model found); using prompt hints only.", flush=True)

    info = video_info(source)
    duration = float(info["duration"])
    print(f"Source video: {info['width']}x{info['height']}, {info['fps']:.4g} fps, {duration:.2f}s", flush=True)

    ffmpeg = find_ffmpeg(args.ffmpeg)
    comfy_dir = resolve_path(args.comfy_dir)
    ensure_audio_models(comfy_dir, music=bool(args.music), sfx=bool(args.sfx))
    if args.music:
        ensure_music_checkpoint_file(comfy_dir, args.music_checkpoint)
        ensure_music_text_encoder_file(comfy_dir, args.music_text_encoder)
    print("Waiting for ComfyUI", flush=True)
    wait_for_comfy(args.comfy_url, timeout_seconds=180, poll_seconds=args.poll_seconds)

    work_dir = ROOT / ".cache" / "audio" / safe_stem(source.name)
    work_dir.mkdir(parents=True, exist_ok=True)

    music_stem = build_music_stem(args, ffmpeg, source, duration, work_dir, captioner) if args.music else None
    sfx_stem = build_sfx_stem(args, ffmpeg, source, duration, work_dir, captioner) if args.sfx else None

    print("Mixing audio stems", flush=True)
    mixed = mix_stems(ffmpeg, music_stem, sfx_stem, float(args.music_gain_db), float(args.sfx_gain_db), work_dir / "mixed.wav")

    print("Muxing soundtrack onto the video (copying the video stream)", flush=True)
    mux_audio(ffmpeg, source, mixed, output)
    write_signature(output, sig)
    print(f"Wrote soundtrack video: {output}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a music/SFX soundtrack for a silent film and mux it onto the video.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--comfy-dir", default=config.get("comfy_dir", str(ROOT / "tools" / "comfyui")))
    parser.add_argument("--comfy-url", default=config.get("comfy_url", "http://127.0.0.1:8188"))
    parser.add_argument("--comfy-output-root", default=str(Path(config.get("comfy_dir", str(ROOT / "tools" / "comfyui"))) / "output"))
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--music", action="store_true", help="Generate a musical score.")
    parser.add_argument("--sfx", action="store_true", help="Generate synchronized sound effects with MMAudio.")
    parser.add_argument("--music-prompt", default="", help="Optional global style hint added to per-scene captions.")
    parser.add_argument("--music-negative", default="low quality, distorted, noisy, clipping")
    parser.add_argument("--music-cue-seconds", type=float, default=30.0)
    parser.add_argument("--music-checkpoint", default="stable_audio_open_1.0.safetensors")
    parser.add_argument("--music-text-encoder", default="t5_base.safetensors")
    parser.add_argument("--music-steps", type=int, default=80)
    parser.add_argument("--music-cfg", type=float, default=6.0)
    parser.add_argument("--sfx-prompt", default="")
    parser.add_argument("--sfx-negative", default="music, song, singing, speech, voice")
    parser.add_argument("--sfx-chunk-seconds", type=float, default=8.0)
    parser.add_argument("--sfx-scene-align", default="on", help="Align SFX windows to scene cuts ('off' tiles fixed windows).")
    parser.add_argument("--sfx-short-side", type=int, default=384)
    parser.add_argument("--sfx-steps", type=int, default=25)
    parser.add_argument("--sfx-cfg", type=float, default=4.5)
    parser.add_argument("--music-gain-db", type=float, default=-9.0)
    parser.add_argument("--sfx-gain-db", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--caption-node", default="", help="ComfyUI node class for local Qwen-VL captioning (optional; overrides Ollama).")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Local Ollama server used for scene captioning.")
    parser.add_argument(
        "--ollama-vision-model", default="auto",
        help="Ollama vision model for scene captions ('auto' picks the first vision-capable model; 'off' disables).",
    )
    parser.add_argument("--ffmpeg", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
