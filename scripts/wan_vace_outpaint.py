"""Wan 2.1 VACE outpainting backend.

VACE is conditioned on a *control video* plus a per-frame *control mask*: masked pixels are
generated, unmasked pixels are kept. ARP's LTX backends render a whole chunk (often 20s) in one
pass because LTX compresses 32x spatially and 8x temporally. Wan only compresses 8x/4x and is
trained on 81-frame clips, so a long chunk would be both far too large and far outside its
training length. Each ARP chunk is therefore rendered as a sequence of overlapping VACE
windows. Every window after the first starts with frames that are already finished, fed
with mask 0 so VACE continues their motion and content instead of starting afresh. The same
mechanism carries the previous chunk's overlap frames into the next chunk.

This module is deliberately free of ComfyUI I/O: rendering a window is delegated to a callback,
so the planning, masking and compositing logic can be tested without a GPU.
"""

from __future__ import annotations

import math
import re
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from common import replace_with_retry
from dependency_manager import WAN_DISTILL_LORA, WAN_TEXT_ENCODER, WAN_VACE_GGUF_MODEL, WAN_VAE
from intermediate_video import audio_codec_args, codec_args, comfy_video_combine_inputs, container_args

# Wan is trained on 81 frames, but at 720p one 161-frame window (6.7s of look-ahead at 24 fps)
# fits in 24 GB (21.6 GB peak) and kept invented surroundings steadier than 81-frame windows
# in side-by-side tests, for about 1.6x the render time.
WAN_WINDOW_FRAMES = 161
WAN_NATIVE_WINDOW_FRAMES = 81
# Windows are capped by the largest one measured to fit in 24 GB, so higher resolutions get
# shorter windows instead of running out of VRAM. Wan's native 81 frames is the floor.
WAN_MAX_WINDOW_PIXELS = 161 * 1280 * 704
WAN_CONTEXT_FRAMES = 13
# Context-window passes hold the whole span in system RAM as float frames (control, mask and
# VACE's inactive/reactive copies). Measured on a 481-frame 1280x704 pass (20s at 24 fps):
# about 24 GB of frame tensors on top of the model weights. Passes are capped by that pixel
# budget, so 1080p passes are shorter rather than exhausting a 64 GB machine.
WAN_MAX_PASS_FRAMES = 481
WAN_MAX_PASS_PIXELS = 481 * 1280 * 704
# ComfyUI's ImagePadForOutpaint fills new canvas with 0.5 grey. WanVaceToVideo centres the
# control video on 0.5, so grey is the "no information" value in both VACE control streams.
WAN_UNKNOWN_GREY = 128
# Lightricks' blend dilates on a mask whose long side is 64 cells; one "Mask seam blend" step
# therefore spans long_side / 64 pixels. Keep the same meaning for Wan's seam feather.
SEAM_BLEND_CELLS = 64

WAN_VACE_REQUIRED_NODES = {
    "UnetLoaderGGUF": "ComfyUI-GGUF",
    "VHS_VideoCombine": "ComfyUI-VideoHelperSuite",
    "WanVaceToVideo": "ComfyUI core",
    "TrimVideoLatent": "ComfyUI core",
    "LoraLoaderModelOnly": "ComfyUI core",
    "ModelSamplingSD3": "ComfyUI core",
    "CLIPLoader": "ComfyUI core",
    "CLIPTextEncode": "ComfyUI core",
    "VAELoader": "ComfyUI core",
    "KSampler": "ComfyUI core",
    "VAEDecode": "ComfyUI core",
    "LoadVideo": "ComfyUI core",
    "GetVideoComponents": "ComfyUI core",
    "ImageToMask": "ComfyUI core",
    "WanContextWindowsManual": "ComfyUI core",
}


@dataclass(frozen=True)
class WanSettings:
    steps: int = 6
    cfg: float = 1.0
    shift: float = 8.0
    sampler: str = "euler"
    scheduler: str = "simple"
    lora_strength: float = 1.0
    control_strength: float = 1.0
    window_frames: int = WAN_WINDOW_FRAMES
    context_frames: int = WAN_CONTEXT_FRAMES
    # "sequential": each window is its own ComfyUI pass, continuing from the previous one.
    # "context": a whole shot is one pass; ComfyUI's Wan context windows sample overlapping
    # windows and blend them at every step, so each frame sees the entire shot (look-ahead)
    # while VRAM stays at one window's worth.
    sampling: str = "sequential"
    context_overlap: int = 30
    max_pass_frames: int = WAN_MAX_PASS_FRAMES

    def window_frames_for(self, width: int, height: int) -> int:
        """Attention window at this canvas size (4n + 1 frames), within the VRAM budget."""
        by_pixels = WAN_MAX_WINDOW_PIXELS // max(1, int(width) * int(height))
        floor = min(int(self.window_frames), WAN_NATIVE_WINDOW_FRAMES)
        frames = max(floor, min(int(self.window_frames), by_pixels))
        return ((frames - 1) // 4) * 4 + 1

    def pass_frames(self, width: int, height: int) -> int:
        """Longest span sampled in one ComfyUI pass at this canvas size (4n + 1 frames)."""
        window = self.window_frames_for(width, height)
        if self.sampling != "context":
            return window
        by_pixels = WAN_MAX_PASS_PIXELS // max(1, int(width) * int(height))
        frames = max(window, min(int(self.max_pass_frames), by_pixels))
        return ((frames - 1) // 4) * 4 + 1

    def signature(self) -> dict[str, Any]:
        extra = (
            {"sampling": "context", "context_overlap": self.context_overlap, "max_pass_frames": self.max_pass_frames}
            if self.sampling == "context"
            else {}
        )
        return {
            **extra,
            "model": WAN_VACE_GGUF_MODEL,
            "text_encoder": WAN_TEXT_ENCODER,
            "vae": WAN_VAE,
            "distill_lora": WAN_DISTILL_LORA,
            "steps": self.steps,
            "cfg": self.cfg,
            "shift": self.shift,
            "sampler": self.sampler,
            "scheduler": self.scheduler,
            "lora_strength": self.lora_strength,
            "control_strength": self.control_strength,
            "window_frames": self.window_frames,
            "context_frames": self.context_frames,
        }


def wan_prompt(prompt: str) -> str:
    """Drop the LTX IC-LoRA trigger word. To Wan "outpaint" is just English for painting, and it
    painted a giant hand holding a brush into the bars of a Metropolis test shot."""
    cleaned = re.sub(r"(?i)\boutpaint(?:ing)?\b[\s,.;:]*", "", prompt or "")
    return cleaned.strip(" ,.;:")


def wan_valid_length(frame_count: int) -> int:
    """Round up to Wan's temporal length (4n + 1)."""
    requested = max(1, int(frame_count))
    return ((requested - 1 + 3) // 4) * 4 + 1


def plan_windows(
    total_frames: int,
    known_prefix: int,
    window_frames: int,
    context_frames: int,
    cuts: "list[int] | tuple[int, ...]" = (),
) -> list[tuple[int, int]]:
    """Return (start, end) windows that generate every frame from known_prefix onward.

    Each window begins with up to context_frames already-finished frames of the same shot.
    Windows never span a cut and never take context from before one: VACE continues whatever
    its context and window contain, so a window crossing a cut paints the previous shot into
    the new shot's bars. A shot uses the fewest windows that fit, sized evenly: filling each
    window greedily would leave a final window that samples a full window's frames to add
    only a handful of new ones.
    """
    window = max(5, int(window_frames))
    context = max(0, min(int(context_frames), window // 2))
    written = max(0, min(int(known_prefix), total_frames))
    shot_starts = sorted({0, *(int(cut) for cut in cuts if 0 < int(cut) < total_frames)})
    windows: list[tuple[int, int]] = []
    while written < total_frames:
        shot_start = max(cut for cut in shot_starts if cut <= written)
        shot_end = min([cut for cut in shot_starts if cut > written] + [total_frames])
        start = max(shot_start, written - context)
        leading_context = written - start
        remaining = shot_end - written
        first_capacity = window - leading_context
        count = 1 if remaining <= first_capacity else 1 + math.ceil((remaining - first_capacity) / (window - context))
        length = math.ceil((remaining + leading_context + (count - 1) * context) / count)
        end = min(shot_end, start + min(window, length))
        windows.append((start, end))
        written = end
    return windows


def seam_alpha(exact_mask: "np.ndarray", feather_px: float) -> "np.ndarray":
    """Blend weight for generated pixels: 1 inside the exact mask, ramping to 0 over feather_px
    of protected source. The exact mask itself is never softened, so every requested pixel
    stays generated."""
    import cv2
    import numpy as np

    inside = exact_mask > 127
    if feather_px <= 0 or not inside.any():
        return inside.astype(np.float32)
    distance = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(1.0 - distance / float(feather_px), 0.0, 1.0).astype(np.float32)
    alpha[inside] = 1.0
    return alpha


def seam_feather_pixels(mask_blend_dilation: int, width: int, height: int) -> float:
    return max(0, int(mask_blend_dilation)) * max(width, height) / SEAM_BLEND_CELLS


class StaticMasks:
    """Geometric bands (plus any custom regions): one mask pair for every frame."""

    def __init__(self, generation: "np.ndarray", exact: "np.ndarray", feather_px: float) -> None:
        self.generation = generation > 127
        self.alpha = seam_alpha(exact, feather_px)

    def for_frame(self, _rgb: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
        return self.generation, self.alpha


class BlackRegionMasks:
    """'Outpaint all black regions': the mask follows each frame's own black pixels."""

    def __init__(self, threshold: int, custom: "np.ndarray | None", overlap_px: int, feather_px: float) -> None:
        import numpy as np

        self.threshold = int(threshold)
        self.custom = custom > 127 if custom is not None else None
        radius = max(0, int(overlap_px))
        self.kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8) if radius else None
        self.feather_px = feather_px

    def for_frame(self, rgb: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
        import cv2
        import numpy as np

        exact = rgb.max(axis=2) <= self.threshold
        if self.custom is not None:
            exact |= self.custom
        exact_u8 = exact.astype(np.uint8) * 255
        generation = cv2.dilate(exact_u8, self.kernel) > 127 if self.kernel is not None else exact
        return generation, seam_alpha(exact_u8, self.feather_px)


def load_guide_rgb(path: Path, width: int, height: int) -> "np.ndarray":
    """Load a full-canvas guide, stretched exactly onto the working canvas like the LTX path."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (width, height):
            rgb = rgb.resize((width, height), getattr(Image, "Resampling", Image).LANCZOS)
        return np.asarray(rgb, dtype=np.uint8).copy()


class FrameReader:
    """Sequential RGB frames from any video ffmpeg can decode."""

    def __init__(self, ffmpeg: str, path: Path, width: int, height: int, start_frame: int = 0) -> None:
        self.width, self.height = width, height
        self.frame_bytes = width * height * 3
        vf = f"select=gte(n\\,{int(start_frame)})," if start_frame else ""
        self.process = subprocess.Popen(
            [ffmpeg, "-v", "error", "-i", str(path), "-vf", f"{vf}scale={width}:{height}:flags=neighbor",
             "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE,
        )

    def read(self) -> "np.ndarray | None":
        import numpy as np

        data = self.process.stdout.read(self.frame_bytes) if self.process.stdout else b""
        if len(data) < self.frame_bytes:
            return None
        return np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)

    def __iter__(self) -> Iterator["np.ndarray"]:
        while (frame := self.read()) is not None:
            yield frame

    def close(self) -> None:
        # Callers often stop early (e.g. skipping a window's padding frames). Stop ffmpeg
        # before closing its pipe, or it reports a spurious "Broken pipe" error.
        if self.process.poll() is None:
            self.process.kill()
        if self.process.stdout:
            self.process.stdout.close()
        self.process.wait()

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class FrameWriter:
    """Pipe RGB frames into ffmpeg. ``codec`` is the output argument list after the input."""

    def __init__(self, ffmpeg: str, path: Path, width: int, height: int, fps: float, output_args: list[str], extra_inputs: list[str] | None = None) -> None:
        self.path = path
        self.process = subprocess.Popen(
            [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", f"{fps:.8f}", "-i", "-", *(extra_inputs or []), *output_args, str(path)],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: "np.ndarray") -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed while writing {self.path}")


def lossless_control_args() -> list[str]:
    # PNG frames decode as rgb24, which ComfyUI's LoadVideo maps to exactly value / 255.
    return ["-c:v", "png", "-pix_fmt", "rgb24", "-f", "matroska"]


def build_wan_vace_prompt(
    control_name: str,
    mask_name: str,
    width: int,
    height: int,
    length: int,
    fps: float,
    prompt_text: str,
    negative_text: str,
    seed: int,
    settings: WanSettings,
    output_prefix: str,
) -> dict[str, Any]:
    """API prompt for one VACE pass, following ComfyUI's own Wan VACE outpainting template."""
    prompt: dict[str, Any] = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": control_name}, "_meta": {"title": "ARP Wan control video"}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "LoadVideo", "inputs": {"file": mask_name}, "_meta": {"title": "ARP Wan control mask"}},
        "4": {"class_type": "GetVideoComponents", "inputs": {"video": ["3", 0]}},
        "5": {"class_type": "ImageToMask", "inputs": {"image": ["4", 0], "channel": "red"}},
        "10": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": WAN_VACE_GGUF_MODEL}},
        "11": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["10", 0], "lora_name": WAN_DISTILL_LORA, "strength_model": float(settings.lora_strength)},
        },
        "12": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["11", 0], "shift": float(settings.shift)}},
        "13": {"class_type": "CLIPLoader", "inputs": {"clip_name": WAN_TEXT_ENCODER, "type": "wan", "device": "default"}},
        "14": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt_text, "clip": ["13", 0]}},
        "15": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_text, "clip": ["13", 0]}},
        "16": {"class_type": "VAELoader", "inputs": {"vae_name": WAN_VAE}},
        "17": {
            "class_type": "WanVaceToVideo",
            "inputs": {
                "positive": ["14", 0],
                "negative": ["15", 0],
                "vae": ["16", 0],
                "width": int(width),
                "height": int(height),
                "length": int(length),
                "batch_size": 1,
                "strength": float(settings.control_strength),
                "control_video": ["2", 0],
                "control_masks": ["5", 0],
            },
        },
        "18": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["12", 0],
                "seed": int(seed),
                "steps": int(settings.steps),
                "cfg": float(settings.cfg),
                "sampler_name": settings.sampler,
                "scheduler": settings.scheduler,
                "positive": ["17", 0],
                "negative": ["17", 1],
                "latent_image": ["17", 2],
                "denoise": 1.0,
            },
        },
        "19": {"class_type": "TrimVideoLatent", "inputs": {"samples": ["18", 0], "trim_amount": ["17", 3]}},
        "20": {"class_type": "VAEDecode", "inputs": {"samples": ["19", 0], "vae": ["16", 0]}},
        "21": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["20", 0],
                "frame_rate": float(fps),
                "loop_count": 0,
                "filename_prefix": output_prefix,
                # Lossless: these frames are composited again before the chunk is encoded.
                **comfy_video_combine_inputs("lossless"),
                "save_metadata": False,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        },
    }
    window = settings.window_frames_for(width, height)
    if settings.sampling == "context" and length > window:
        prompt["22"] = {
            "class_type": "WanContextWindowsManual",
            "inputs": {
                "model": ["12", 0],
                "context_length": int(window),
                "context_overlap": int(settings.context_overlap),
                "context_schedule": "standard_uniform",
                "context_stride": 1,
                "closed_loop": False,
                "fuse_method": "pyramid",
                "freenoise": True,
                "retain_first_frame": False,
                "split_conds_to_windows": False,
            },
            "_meta": {"title": "ARP Wan context windows (whole-shot look-ahead)"},
        }
        prompt["18"]["inputs"]["model"] = ["22", 0]
    return prompt


WAN_OUTPUT_NODE_ID = "21"

# (control file, mask file, length, window index) -> rendered window video
RenderWindow = Callable[[Path, Path, int, int], Path]


def render_chunk(
    *,
    ffmpeg: str,
    chunk_video: Path,
    output: Path,
    width: int,
    height: int,
    fps: float,
    total_frames: int,
    masks: "StaticMasks | BlackRegionMasks",
    known_prefix: list["np.ndarray"],
    guides: dict[int, tuple["np.ndarray", float]],
    render_window: RenderWindow,
    work_dir: Path,
    settings: WanSettings,
    intermediate_profile: str = "high",
    has_audio: bool = False,
    cuts: "list[int] | tuple[int, ...]" = (),
    log: Callable[[str], None] = lambda message: print(message, flush=True),
) -> None:
    """Render one ARP chunk as consecutive VACE windows and encode it to ``output``.

    known_prefix holds finished frames for the start of the chunk (the previous chunk's
    overlap); they are written verbatim and serve as the first window's context. guides maps
    a chunk frame index to (full-canvas RGB, strength): the guide fills that frame's generated
    region, and strength sets how firmly VACE keeps it (1 = kept as given).
    """
    import numpy as np

    prefix = known_prefix[:total_frames]
    pass_frames = settings.pass_frames(width, height)
    windows = plan_windows(total_frames, len(prefix), pass_frames, settings.context_frames, cuts)
    inner_cuts = sorted(int(cut) for cut in cuts if 0 < int(cut) < total_frames)
    window = settings.window_frames_for(width, height)
    if window < settings.window_frames:
        log(f"Wan VACE: {width}x{height} limits windows to {window} frames (requested {settings.window_frames}) to stay within 24 GB of VRAM.")
    log(
        f"Wan VACE ({settings.sampling}): {total_frames} frame(s) as {len(windows)} pass(es) of up to "
        f"{pass_frames} frames, {settings.context_frames} context frame(s) each"
        + (f", sampled in {window}-frame context windows overlapping by {settings.context_overlap}"
           if settings.sampling == "context" else "")
        + (f"; {len(prefix)} frame(s) continue the previous chunk" if prefix else "")
        + (f"; new shot(s) start at chunk frame(s) {', '.join(map(str, inner_cuts))}" if inner_cuts else "")
    )
    for index in sorted(guides):
        if index < len(prefix):
            log(f"Wan VACE: guide at chunk frame {index} falls inside the previous chunk's overlap and is ignored.")

    grey = np.full((height, width, 3), WAN_UNKNOWN_GREY, dtype=np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial" + output.suffix)
    output_args = ["-map", "0:v:0"]
    extra_inputs: list[str] = []
    if has_audio:
        # Keep the chunk's own audio in the raw render, as the LTX graph does for previews.
        extra_inputs = ["-i", str(chunk_video)]
        output_args += ["-map", "1:a:0?", *audio_codec_args(intermediate_profile, bitrate="192k")]
    output_args += [
        "-frames:v", str(total_frames), "-fps_mode", "cfr",
        *codec_args(intermediate_profile, fast=True), *container_args(str(partial), intermediate_profile),
    ]
    writer = FrameWriter(ffmpeg, partial, width, height, fps, output_args, extra_inputs)
    # Finished frames still needed as context: plan_windows never takes more than context_frames.
    finished: deque[tuple[int, "np.ndarray"]] = deque(maxlen=max(settings.context_frames, 1))
    try:
        for index, frame in enumerate(prefix):
            writer.write(frame)
            finished.append((index, frame))
        written = len(prefix)
        with FrameReader(ffmpeg, chunk_video, width, height, start_frame=written) as source:
            last_source: "np.ndarray | None" = None
            for window_index, (start, end) in enumerate(windows):
                length = wan_valid_length(end - start)
                known = {index: frame for index, frame in finished if start <= index < written}
                new_sources: list["np.ndarray"] = []
                new_masks: list[tuple["np.ndarray", "np.ndarray"]] = []
                control_path = work_dir / f"{output.stem}_w{window_index:02d}_control.mkv"
                mask_path = work_dir / f"{output.stem}_w{window_index:02d}_mask.mkv"
                control_writer = FrameWriter(ffmpeg, control_path, width, height, fps, lossless_control_args())
                mask_writer = FrameWriter(ffmpeg, mask_path, width, height, fps, lossless_control_args())
                try:
                    control = mask_rgb = None
                    for index in range(start, start + length):
                        if index in known:
                            control = known[index]
                            mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
                        elif index < end:
                            frame = source.read()
                            if frame is None:
                                frame = last_source if last_source is not None else grey
                            last_source = frame
                            generation, alpha = masks.for_frame(frame)
                            new_sources.append(frame)
                            new_masks.append((generation, alpha))
                            control = np.where(generation[..., None], grey, frame)
                            mask_value = generation.astype(np.float32)
                            if index in guides:
                                guide, strength = guides[index]
                                control = np.where(generation[..., None], guide, frame)
                                mask_value *= 1.0 - max(0.0, min(1.0, float(strength)))
                                log(f"Wan VACE: guide at chunk frame {index}, strength {float(strength):g}")
                            mask_rgb = np.repeat(np.round(mask_value * 255).astype(np.uint8)[..., None], 3, axis=2)
                        # Padding past `end` repeats the last control frame and mask.
                        control_writer.write(control)
                        mask_writer.write(mask_rgb)
                finally:
                    control_writer.close()
                    mask_writer.close()
                log(
                    f"Wan VACE window {window_index + 1}/{len(windows)}: chunk frames {start}-{end - 1} "
                    f"({written - start} context, {end - written} new, {length} sampled)"
                )
                rendered = render_window(control_path, mask_path, length, window_index)
                with FrameReader(ffmpeg, rendered, width, height, start_frame=written - start) as generated_frames:
                    for offset, (frame, (_generation, alpha)) in enumerate(zip(new_sources, new_masks)):
                        generated = generated_frames.read()
                        if generated is None:
                            raise RuntimeError(f"Wan VACE window {window_index + 1} returned too few frames: {rendered}")
                        weight = alpha[..., None]
                        blended = (generated.astype(np.float32) * weight + frame.astype(np.float32) * (1.0 - weight))
                        composite = np.clip(np.round(blended), 0, 255).astype(np.uint8)
                        writer.write(composite)
                        finished.append((written + offset, composite))
                written = end
                for path in (control_path, mask_path, rendered):
                    path.unlink(missing_ok=True)
    except BaseException:
        try:
            writer.close()
        except RuntimeError:
            pass
        partial.unlink(missing_ok=True)
        raise
    writer.close()
    replace_with_retry(partial, output, f"Wan VACE chunk {output.name}")
