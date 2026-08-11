from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from . import state
from .cache import human_size
from .config import ASPECT_PREVIEW_DIR, FILE_PREVIEW_DIR, IMAGE_EXTS, MEDIA_CLIP_DIR, PREVIEW_DIR, ROOT, SCRIPTS, VIDEO_EXTS
from .file_dialogs import browse_path, parse_duration
from .paths import even_int, format_timecode, parse_aspect, rel, resolve, resolve_video_source, safe_stem
from .process_utils import format_duration
from .project_io import source_analysis_key, source_signature

SOURCE_PREVIEW_COUNT = 3
ASPECT_PREVIEW_STYLE_VERSION = 6

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import artifact_ids as aid  # noqa: E402
from outpaint_geometry import crop_box, source_placement  # noqa: E402


def aspect_preview_identity(source: Path, size: int, mtime_ns: int, aspect: str, crops: tuple[int, int, int, int], seconds: float, offset_x: int = 0, offset_y: int = 0) -> dict:
    return {
        "v": 1,
        "kind": "aspect_preview",
        "source": aid.safe_stem(source.name),
        "source_path": str(source.resolve()),
        "source_size": int(size),
        "source_mtime_ns": int(mtime_ns),
        "aspect": str(aspect),
        "crop": [int(v) for v in crops],
        "seconds_ms": int(round(seconds * 1000)),
        "offset": [int(offset_x), int(offset_y)],
        "style": ASPECT_PREVIEW_STYLE_VERSION,
    }


def source_previews(source_text: str) -> list[str]:
    signature = source_signature(source_text)
    if signature is None:
        if source_text:
            source = resolve(source_text)
            state.APP.log.append(f"Source preview skipped; file was not found or is not a supported video: {source}")
        return []
    return list(source_previews_cached(*signature))

def source_previews_for_analysis(signature: tuple[str, int, int], info: dict[str, str], progress: Callable[[int, str], None]) -> tuple[str, ...]:
    source = Path(signature[0])
    _source_path, _size, mtime_ns = signature
    safe = safe_preview_name(source)
    target_dir = PREVIEW_DIR / safe
    frames = [target_dir / f"preview_{index}.jpg" for index in range(SOURCE_PREVIEW_COUNT)]
    try:
        if all(frame.exists() and frame.stat().st_mtime_ns >= mtime_ns for frame in frames):
            return tuple(rel(frame) for frame in frames)
        state.APP.log.append(f"Generating source previews from: {source}")
        generate_video_previews(source, target_dir, progress, parse_duration(info.get("duration")))
        state.APP.log.append(f"Generated source previews in: {target_dir}")
        return tuple(rel(frame) for frame in frames if frame.exists())
    except Exception as exc:
        state.APP.log.append(f"Could not generate source previews: {exc}")
        return ()

def source_previews_cached(source_path: str, _size: int, mtime_ns: int) -> tuple[str, ...]:
    source = Path(source_path)
    safe = safe_preview_name(source)
    target_dir = PREVIEW_DIR / safe
    frames = [target_dir / f"preview_{index}.jpg" for index in range(SOURCE_PREVIEW_COUNT)]
    try:
        if all(frame.exists() and frame.stat().st_mtime_ns >= mtime_ns for frame in frames):
            return tuple(rel(frame) for frame in frames)
        state.APP.log.append(f"Generating {SOURCE_PREVIEW_COUNT} source previews from: {source}")
        generate_video_previews(source, target_dir)
        state.APP.log.append(f"Generated source previews in: {target_dir}")
        return tuple(rel(frame) for frame in frames if frame.exists())
    except Exception as exc:
        state.APP.log.append(f"Could not generate source previews: {exc}")
        return ()

def source_info(source_text: str) -> dict[str, str]:
    signature = source_signature(source_text)
    if signature is None:
        if source_text:
            source = resolve(source_text)
            state.APP.log.append(f"Source info skipped; file was not found or is not a supported video: {source}")
        return {}
    return dict(source_info_cached(*signature))

def source_monochrome(source_text: str) -> bool:
    signature = source_signature(source_text)
    if signature is None:
        return True
    return source_monochrome_cached(*signature)

def source_monochrome_cached(source_path: str, size: int, mtime_ns: int) -> bool:
    try:
        from PIL import Image, ImageChops, ImageStat
    except ModuleNotFoundError:
        return True
    previews = source_previews_cached(source_path, size, mtime_ns)
    if not previews:
        return True
    scores = []
    for preview_path in previews:
        try:
            image = Image.open(resolve(preview_path)).convert("RGB").resize((160, 90))
            r, g, b = image.split()
            rg = ImageStat.Stat(ImageChops.difference(r, g)).mean[0]
            rb = ImageStat.Stat(ImageChops.difference(r, b)).mean[0]
            gb = ImageStat.Stat(ImageChops.difference(g, b)).mean[0]
            scores.append((rg + rb + gb) / 3)
        except Exception:
            continue
    return (sum(scores) / max(1, len(scores))) < 2.5

def source_info_cached(source_path: str, size: int, _mtime_ns: int) -> tuple[tuple[str, str], ...]:
    source = Path(source_path)
    state.APP.log.append(f"Probing source file info: {source}")
    info: dict[str, str] = {"file": rel(source), "size": human_size(size)}
    info.update(ffprobe_info(source))
    return tuple(info.items())

def current_crop_values() -> tuple[int, int, int, int]:
    values = state.APP.settings.get("outpaint", {}) if state.APP is not None else {}
    return tuple(max(0, int(float(values.get(key, "0") or 0))) for key in ("crop_left", "crop_right", "crop_top", "crop_bottom"))  # type: ignore[return-value]

def aspect_preview(source_text: str, aspect: str) -> str:
    signature = source_signature(source_text)
    if signature is None:
        return ""
    return aspect_preview_cached(signature[0], signature[1], signature[2], aspect, current_crop_values(), 10.0)

def aspect_preview_for_settings(settings: dict) -> str:
    source_text = preview_pipeline_source_text(settings)
    if not source_text:
        return ""
    signature = source_signature(source_text)
    if signature is None:
        return ""
    seconds = 0.0 if source_section_is_active(settings) else 10.0
    return aspect_preview_cached(
        signature[0],
        signature[1],
        signature[2],
        settings.get("outpaint", {}).get("target_aspect", "16:9"),
        current_crop_values(),
        seconds,
    )

def aspect_preview_at(source_text: str, aspect: str, seconds: float, offset_x: int = 0, offset_y: int = 0) -> str:
    signature = source_signature(source_text)
    if signature is None:
        return ""
    return aspect_preview_cached(signature[0], signature[1], signature[2], aspect, current_crop_values(), round(max(0.0, seconds), 3), offset_x, offset_y)

def aspect_preview_at_for_settings(settings: dict, seconds: float) -> str:
    source_text = preview_pipeline_source_text(settings)
    if not source_text:
        return ""
    relative_seconds = section_relative_seconds(settings, seconds)
    return aspect_preview_at(source_text, settings.get("outpaint", {}).get("target_aspect", "16:9"), relative_seconds)

def auto_crop_for_settings(settings: dict, seconds: float) -> dict[str, str | int]:
    source_text = preview_pipeline_source_text(settings)
    if not source_text:
        raise RuntimeError("Choose source material before using Auto Crop.")
    relative_seconds = section_relative_seconds(settings, seconds)
    signature = source_signature(source_text)
    if signature is None:
        raise RuntimeError("Source material is not a readable video.")
    source = Path(signature[0])
    frame = extract_video_frame_at(source, ASPECT_PREVIEW_DIR / "frames", f"autocrop_{int(relative_seconds * 1000):010d}", relative_seconds)
    if not frame:
        raise RuntimeError("Could not extract the current preview frame.")
    from PIL import Image

    with Image.open(resolve(frame)).convert("RGB") as image:
        left, right, top, bottom = detect_letterbox_crop(image)
    values = {
        "crop_left": str(left),
        "crop_right": str(right),
        "crop_top": str(top),
        "crop_bottom": str(bottom),
    }
    state.APP.update_settings("outpaint", values)
    state.APP.log.append(f"Auto Crop set source crop to left {left}, right {right}, top {top}, bottom {bottom}.")
    return {**values, "preview": aspect_preview_at_for_settings(settings, seconds)}

def detect_letterbox_crop(image) -> tuple[int, int, int, int]:
    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.load()
    threshold = 18
    min_content_fraction = 0.035
    max_x = int(width * 0.45)
    max_y = int(height * 0.45)

    def col_has_content(x: int) -> bool:
        count = 0
        for y in range(height):
            if pixels[x, y] > threshold:
                count += 1
        return (count / max(1, height)) >= min_content_fraction

    def row_has_content(y: int) -> bool:
        count = 0
        for x in range(width):
            if pixels[x, y] > threshold:
                count += 1
        return (count / max(1, width)) >= min_content_fraction

    left = next((x for x in range(max_x) if col_has_content(x)), 0)
    right_edge = next((x for x in range(width - 1, width - max_x - 1, -1) if col_has_content(x)), width - 1)
    top = next((y for y in range(max_y) if row_has_content(y)), 0)
    bottom_edge = next((y for y in range(height - 1, height - max_y - 1, -1) if row_has_content(y)), height - 1)
    right = max(0, width - 1 - right_edge)
    bottom = max(0, height - 1 - bottom_edge)

    # Ignore tiny border noise; users can fine-tune those manually.
    floor_x = max(4, width // 200)
    floor_y = max(4, height // 200)
    left = 0 if left < floor_x else left
    right = 0 if right < floor_x else right
    top = 0 if top < floor_y else top
    bottom = 0 if bottom < floor_y else bottom
    return left, right, top, bottom

def preview_pipeline_source_text(settings: dict) -> str:
    try:
        ensure_source_section_clip(settings)
    except Exception as exc:
        state.APP.log.append(f"Could not prepare selected source section for preview: {exc}")
    return pipeline_source_text(settings)

def section_relative_seconds(settings: dict, seconds: float) -> float:
    if not source_section_is_active(settings):
        return seconds
    start = section_float(settings.get("global", {}).get("section_start", "0"), 0.0)
    end = section_float(settings.get("global", {}).get("section_end", ""), 0.0)
    return max(0.0, min(end - start, seconds - start))

def aspect_preview_cached(source_path: str, size: int, mtime_ns: int, aspect: str, crops: tuple[int, int, int, int], seconds: float, offset_x: int = 0, offset_y: int = 0) -> str:
    source = Path(source_path)
    source_frame = extract_video_frame_at(source, ASPECT_PREVIEW_DIR / "frames", f"aspect_{int(seconds * 1000):010d}", seconds)
    if not source_frame:
        return ""
    identity = aspect_preview_identity(source, size, mtime_ns, aspect, crops, seconds, offset_x, offset_y)
    target = aid.artifact_path(ASPECT_PREVIEW_DIR, aid.source_word(source.name), "aspectpreview", identity, "jpg")
    if target.exists() and target.stat().st_mtime_ns >= mtime_ns:
        if not aid.sig_path(target).exists():
            aid.write_identity(target, identity, label="Aspect preview")
        return rel(target)
    try:
        from PIL import Image, ImageOps
    except ModuleNotFoundError:
        state.APP.log.append("Pillow is not available; using FFmpeg for the aspect preview.")
        preview = ffmpeg_aspect_preview(source, target, aspect, mtime_ns, crops, offset_x, offset_y)
        if preview:
            aid.write_identity(target, identity, label="Aspect preview")
        return preview or source_frame
    ratio = parse_aspect(aspect)
    image = Image.open(resolve(source_frame)).convert("RGB")
    width, height = image.size
    left, right, top, bottom = crops
    crop_left, _crop_right, crop_top, _crop_bottom, crop_width, crop_height = crop_box(width, height, left, right, top, bottom)
    cropped = image.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))
    if crop_width / crop_height < ratio:
        target_h = crop_height
        target_w = int(round(crop_height * ratio))
    else:
        target_w = crop_width
        target_h = int(round(crop_width / ratio))
    placement = source_placement(width, height, target_w, target_h, crops)
    canvas = patterned_canvas(target_w, target_h)
    paste_xy = (placement.x + int(offset_x), placement.y + int(offset_y))
    canvas.paste(cropped.resize((placement.width, placement.height), Image.Resampling.LANCZOS), paste_xy)
    draw_source_frame_border(canvas, paste_xy, (placement.width, placement.height))
    preview = ImageOps.contain(canvas, (960, 540), Image.Resampling.LANCZOS)
    target.parent.mkdir(parents=True, exist_ok=True)
    preview.save(target, quality=90)
    aid.write_identity(target, identity, label="Aspect preview")
    return rel(target)

def draw_source_frame_border(canvas, origin: tuple[int, int], size: tuple[int, int]) -> None:
    from PIL import ImageDraw

    x, y = origin
    width, height = size
    if width <= 2 or height <= 2:
        return
    draw = ImageDraw.Draw(canvas)
    box = (x, y, x + width - 1, y + height - 1)
    inner = (x + 1, y + 1, x + width - 2, y + height - 2)
    outer = (x - 1, y - 1, x + width, y + height)
    draw.rectangle(outer, outline=(0, 0, 0), width=2)
    draw.rectangle(box, outline=(235, 212, 134), width=max(2, min(width, height) // 180))
    draw.rectangle(inner, outline=(16, 19, 22), width=1)

def patterned_canvas(width: int, height: int):
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (width, height), (18, 31, 35))
    draw = ImageDraw.Draw(canvas)
    spacing = max(14, min(width, height) // 28)
    line_color = (68, 157, 145)
    accent = (219, 174, 66)
    for offset in range(-height, width, spacing):
        draw.line((offset, 0, offset + height, height), fill=line_color, width=max(2, spacing // 8))
    for offset in range(0, width + height, spacing * 3):
        draw.line((offset, 0, offset - height, height), fill=accent, width=max(2, spacing // 10))
    return canvas

def ffmpeg_aspect_preview(source: Path, target: Path, aspect: str, mtime_ns: int, crops: tuple[int, int, int, int] = (0, 0, 0, 0), offset_x: int = 0, offset_y: int = 0) -> str:
    ffmpeg = local_tool("ffmpeg")
    dims = video_dimensions(source)
    if not ffmpeg or not dims:
        return ""
    source_w, source_h = dims
    ratio = parse_aspect(aspect)
    left, _right, top, _bottom, crop_width, crop_height = crop_box(source_w, source_h, *crops)
    if crop_width / crop_height < ratio:
        canvas_h = crop_height
        canvas_w = int(round(crop_height * ratio))
    else:
        canvas_w = crop_width
        canvas_h = int(round(crop_width / ratio))
    scale = min(960 / canvas_w, 540 / canvas_h, 1.0)
    out_w = max(2, even_int(canvas_w * scale))
    out_h = max(2, even_int(canvas_h * scale))
    placement = source_placement(source_w, source_h, canvas_w, canvas_h, crops)
    placed_x = int(round(placement.x * scale))
    placed_y = int(round(placement.y * scale))
    placed_w = max(2, even_int(placement.width * scale))
    placed_h = max(2, even_int(placement.height * scale))
    scaled_offset_x = int(round(offset_x * scale))
    scaled_offset_y = int(round(offset_y * scale))
    target.parent.mkdir(parents=True, exist_ok=True)
    filter_text = (
        f"crop=w={crop_width}:h={crop_height}:x={left}:y={top},scale={placed_w}:{placed_h}[src];"
        f"color=c=0x15272b:s={out_w}x{out_h}[bg];"
        f"[bg]geq=r='34+34*mod(floor((X+Y)/18)\\,2)':g='62+48*mod(floor((X+Y)/18)\\,2)':b='67+40*mod(floor((X+Y)/18)\\,2)'[pat];"
        f"[pat][src]overlay={placed_x + scaled_offset_x}:{placed_y + scaled_offset_y}"
    )
    command = [ffmpeg, "-y", "-ss", "10", "-i", str(source), "-frames:v", "1", "-vf", filter_text, "-q:v", "3", str(target)]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        command[3] = "0"
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        state.APP.log.append(f"Could not generate aspect preview: {(result.stderr or result.stdout).strip()}")
    return rel(target) if result.returncode == 0 and target.exists() and target.stat().st_mtime_ns >= mtime_ns else ""

def video_dimensions(source: Path) -> tuple[int, int] | None:
    resolution = ffprobe_info(source).get("resolution", "")
    if "x" not in resolution:
        return None
    left, right = resolution.split("x", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None

def file_preview(path: Path) -> str:
    if path.suffix.lower() in IMAGE_EXTS:
        return rel(path)
    if path.suffix.lower() in VIDEO_EXTS:
        signature = source_signature(str(path))
        if signature is None:
            return ""
        return file_preview_cached(*signature)
    return ""

def file_preview_cached(source_path: str, _size: int, _mtime_ns: int) -> str:
    return extract_video_frame(Path(source_path), FILE_PREVIEW_DIR, "thumb")

def extract_video_frame(source: Path, target_dir: Path, suffix: str) -> str:
    return extract_video_frame_at(source, target_dir, suffix, 10.0)

def extract_video_frame_at(source: Path, target_dir: Path, suffix: str, seconds: float) -> str:
    ffmpeg = local_tool("ffmpeg")
    if not ffmpeg:
        return ""
    target_dir.mkdir(parents=True, exist_ok=True)
    candidate = target_dir / f"{safe_preview_name(source)}_{suffix}.jpg"
    # Fall back to a hash-based name if the path would exceed Windows MAX_PATH (260 chars).
    if len(str(candidate)) > 240:
        key = hashlib.sha256(f"{source}\0{suffix}".encode()).hexdigest()[:24]
        candidate = target_dir / f"{key}.jpg"
    target = candidate
    try:
        if target.exists() and target.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            return rel(target)
    except OSError:
        pass
    command = [ffmpeg, "-y", "-ss", f"{max(0.0, seconds):.3f}", "-i", str(source), "-frames:v", "1", "-q:v", "4", str(target)]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        command = [ffmpeg, "-y", "-ss", "0", "-i", str(source), "-frames:v", "1", "-q:v", "4", str(target)]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    return rel(target) if result.returncode == 0 and target.exists() else ""

def extract_video_frame_at_frame(source: Path, target_dir: Path, suffix: str, frame_index: int, quality: int = 4) -> str:
    ffmpeg = local_tool("ffmpeg")
    if not ffmpeg:
        return ""
    target_dir.mkdir(parents=True, exist_ok=True)
    frame = max(0, int(frame_index))
    candidate = target_dir / f"{safe_preview_name(source)}_{suffix}.jpg"
    if len(str(candidate)) > 240:
        key = hashlib.sha256(f"{source}\0{suffix}\0{frame}".encode()).hexdigest()[:24]
        candidate = target_dir / f"{key}.jpg"
    target = candidate
    try:
        if target.exists() and target.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            return rel(target)
    except OSError:
        pass
    vf = f"trim=start_frame={frame}:end_frame={frame + 1},setpts=PTS-STARTPTS"
    command = [ffmpeg, "-y", "-i", str(source), "-vf", vf, "-frames:v", "1", "-q:v", str(quality), str(target)]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0 and frame != 0:
        return extract_video_frame_at_frame(source, target_dir, suffix, 0, quality)
    return rel(target) if result.returncode == 0 and target.exists() else ""

def ffprobe_basic_info(source: Path) -> dict[str, str]:
    found = local_tool("ffprobe")
    if not found:
        return {"codec_note": "Run install_windows.bat to install local FFmpeg/ffprobe for codec and colour metadata."}
    command = [
        str(found),
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,nb_frames,codec_name,pix_fmt,color_space,color_transfer,color_primaries,color_range,bit_rate,sample_rate,channels:format=duration,format_name,bit_rate",
        "-of",
        "json",
        str(source),
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return {"codec_note": "Basic ffprobe metadata timed out; previews may still load."}
    if result.returncode != 0:
        return {"codec_note": (result.stderr or "ffprobe failed").strip()}
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"codec_note": "ffprobe returned invalid JSON."}
    return ffprobe_info_from_data(data)

# ffprobe is slow to spawn (≈50-100ms each) and shot views probe the same source
# dozens of times per state build. Cache successful probes by file identity so they
# only re-run when the underlying file actually changes.
_FFPROBE_CACHE: dict[tuple[str, int, int], dict[str, str]] = {}


def ffprobe_info(source: Path) -> dict[str, str]:
    cache_key: tuple[str, int, int] | None = None
    try:
        stat = source.stat()
        cache_key = (str(source), stat.st_mtime_ns, stat.st_size)
    except OSError:
        cache_key = None
    if cache_key is not None:
        cached = _FFPROBE_CACHE.get(cache_key)
        if cached is not None:
            return dict(cached)
    found = local_tool("ffprobe")
    if not found:
        return {"codec_note": "Run install_windows.bat to install local FFmpeg/ffprobe for codec and colour metadata."}
    command = [
        str(found),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(source),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return {"codec_note": (result.stderr or "ffprobe failed").strip()}
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"codec_note": "ffprobe returned invalid JSON."}
    info = ffprobe_info_from_data(data)
    if cache_key is not None:
        _FFPROBE_CACHE[cache_key] = dict(info)
    return info

def ffprobe_info_from_data(data: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    streams = data.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if video:
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
        if width and height:
            out["resolution"] = f"{width}x{height}"
            out["aspect"] = f"{width / height:.3f}:1"
        fps = parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate"))
        if fps:
            out["frame_rate"] = f"{fps:.3f} fps"
        if video.get("nb_frames"):
            out["frames"] = f"{int(video['nb_frames']):,}"
        if video.get("codec_name"):
            out["video_codec"] = str(video["codec_name"])
        if video.get("pix_fmt"):
            out["pixel_format"] = str(video["pix_fmt"])
        color_parts = [video.get("color_space"), video.get("color_transfer"), video.get("color_primaries"), video.get("color_range")]
        color = " / ".join(str(part) for part in color_parts if part)
        if color:
            out["colour"] = color
        if video.get("bit_rate"):
            out["video_bitrate"] = human_bitrate(video["bit_rate"])
    if audio:
        audio_parts = [audio.get("codec_name"), audio.get("sample_rate"), audio.get("channels")]
        values = [str(part) for part in audio_parts if part]
        if values:
            out["audio"] = ", ".join(values)
    fmt = data.get("format") or {}
    if fmt.get("duration"):
        try:
            out["duration"] = format_duration(float(fmt["duration"]))
        except ValueError:
            pass
    if fmt.get("format_name"):
        out["container"] = str(fmt["format_name"])
    if fmt.get("bit_rate"):
        out["overall_bitrate"] = human_bitrate(fmt["bit_rate"])
    return out

def video_metrics(source: Path) -> dict[str, float]:
    found = local_tool("ffprobe")
    if found:
        command = [
            str(found),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(source),
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                stream = (data.get("streams") or [{}])[0]
                fmt = data.get("format") or {}
                fps = parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate")) or 24.0
                duration = float(stream.get("duration") or fmt.get("duration") or 0)
                frames = int(str(stream.get("nb_frames") or "0").replace(",", "") or "0")
                width = int(stream.get("width") or 0)
                height = int(stream.get("height") or 0)
                if frames <= 0 and duration > 0:
                    frames = int(round(duration * fps))
                if frames > 0:
                    return {"fps": fps, "frames": float(frames), "duration": duration or frames / fps, "width": width, "height": height}
            except (ValueError, TypeError, json.JSONDecodeError, IndexError):
                pass
    try:
        import cv2

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            return {}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        return {"fps": fps, "frames": float(frames), "duration": frames / fps if frames and fps else 0.0, "width": width, "height": height}
    except Exception:
        return {}

def local_tool(name: str) -> str | None:
    exe = f"{name}.exe" if os.name == "nt" else name
    local = ROOT / ".cache" / "tools" / "ffmpeg" / exe
    if local.exists():
        return str(local)
    return shutil.which(name)

def parse_rate(value: str | None) -> float | None:
    if not value:
        return None
    if "/" in value:
        left, right = value.split("/", 1)
        try:
            denominator = float(right)
            return float(left) / denominator if denominator else None
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def human_bitrate(value: str | int | float) -> str:
    try:
        bits = float(value)
    except (TypeError, ValueError):
        return str(value)
    if bits >= 1_000_000:
        return f"{bits / 1_000_000:.2f} Mbps"
    if bits >= 1_000:
        return f"{bits / 1_000:.1f} Kbps"
    return f"{bits:.0f} bps"

def safe_preview_name(path: Path) -> str:
    stem = safe_stem(path.name) or "media"
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{stem[:64]}_{digest}"

def media_clip_path(source: Path, start: float, end: float, key: str = "") -> Path:
    ffmpeg = local_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Run install_windows.bat to install local FFmpeg for shot video previews.")
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(source)
    duration = max(0.041, end - start)
    stat = source.stat()
    digest = hashlib.sha1(
        f"{source.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{start:.3f}|{end:.3f}|{key}".encode("utf-8", errors="ignore")
    ).hexdigest()[:20]
    target = MEDIA_CLIP_DIR / f"{safe_preview_name(source)[:80]}_{digest}.mp4"
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.mp4")
    if partial.exists():
        partial.unlink()
    command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{max(0.0, start):.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-an",
        "-vf",
        "setpts=PTS-STARTPTS,setsar=1",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(partial),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        if partial.exists():
            partial.unlink()
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg clip extraction failed").strip())
    partial.replace(target)
    return target

def export_media_file(path_text: str) -> dict[str, str]:
    source = resolve(path_text)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(source)

    target_text = browse_path("save_image" if source.suffix.lower() in IMAGE_EXTS else "save", str(source))
    if not target_text:
        return {"saved": ""}
    target = resolve(target_text)
    if target.suffix == "":
        target = target.with_suffix(source.suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.resolve() != source.resolve():
        shutil.copy2(source, target)
    state.APP.log.append(f"Saved media file: {rel(source)} -> {target}")
    return {"saved": str(target)}

def generate_video_previews(source: Path, target_dir: Path, progress: Callable[[int, str], None] | None = None, duration: float | None = None) -> None:
    ffmpeg = local_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Run install_windows.bat to install local FFmpeg for source previews.")
    target_dir.mkdir(parents=True, exist_ok=True)
    if duration and duration > 45:
        positions = [0, min(10, duration * 0.08), min(30, duration * 0.18)]
    elif duration and duration > 0:
        positions = [0, duration * 0.33, duration * 0.66]
    else:
        positions = [0, 10, 30]
    for index, seconds in enumerate(positions):
        if progress:
            percent = 35 + int(((index + 1) / len(positions)) * 30)
            progress(percent, f"Generating source preview frame {index + 1}/{len(positions)}")
        out = target_dir / f"preview_{index}.jpg"
        command = [
            ffmpeg,
            "-y",
            "-ss",
            f"{seconds:.3f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(out),
        ]
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            state.APP.log.append(f"Preview frame {index + 1} timed out at {seconds:.3f}s; skipping it.")
            continue
        if result.returncode != 0:
            state.APP.log.append(f"Preview frame {index + 1} failed: {(result.stderr or result.stdout).strip()}")


def pipeline_source_text(settings: dict) -> str:
    global_settings = settings.get("global", {})
    source_text = global_settings.get("source", "")
    if not source_text or not source_section_is_active(settings):
        return source_text
    return rel(source_section_output_for(settings))

def source_section_state(settings: dict) -> dict:
    global_settings = settings.get("global", {})
    source_text = global_settings.get("source", "")
    start = section_float(global_settings.get("section_start", "0"), 0.0)
    end = section_float(global_settings.get("section_end", ""), 0.0)
    enabled = source_section_is_active(settings)
    output = source_section_output_for(settings) if source_text and enabled else None
    return {
        "enabled": enabled,
        "start": start,
        "end": end,
        "start_label": format_timecode(start),
        "end_label": format_timecode(end) if end > 0 else "",
        "output": rel(output) if output else "",
        "output_exists": bool(output and output.exists()),
    }

def source_section_output_for(settings: dict) -> Path:
    global_settings = settings.get("global", {})
    source = resolve_video_source(global_settings.get("source", ""))
    start = section_float(global_settings.get("section_start", "0"), 0.0)
    end = section_float(global_settings.get("section_end", ""), 0.0)
    suffix = f"{int(round(start * 1000)):010d}_{int(round(end * 1000)):010d}"
    # Section clips are transcoded to H.264/AAC below, so they must use an MP4
    # container even when the original source is WebM, MKV, or another format.
    return ROOT / "intermediate" / "source_sections" / f"{safe_stem(source.name)}_{suffix}.mp4"

def source_section_is_active(settings: dict) -> bool:
    global_settings = settings.get("global", {})
    start = section_float(global_settings.get("section_start", "0"), 0.0)
    end = section_float(global_settings.get("section_end", ""), 0.0)
    return end > start

def ensure_source_section_clip(settings: dict) -> str:
    global_settings = settings.get("global", {})
    source_text = global_settings.get("source", "")
    if not source_text or not source_section_is_active(settings):
        return source_text
    source = resolve_video_source(source_text)
    start = section_float(global_settings.get("section_start", "0"), 0.0)
    end = section_float(global_settings.get("section_end", ""), 0.0)
    if end <= start:
        return source_text
    output = source_section_output_for(settings)
    ffmpeg = local_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Run install_windows.bat to install local FFmpeg for source section trimming.")
    expected_duration = max(0.041, end - start)
    if source_section_clip_is_valid(output, ffmpeg, expected_duration):
        return rel(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    has_audio = source_has_audio_stream(ffmpeg, source)
    command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{max(0.041, end - start):.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        "setpts=PTS-STARTPTS",
        "-c:v",
        "libx264",
        "-crf",
        "14",
        "-preset",
        "veryfast",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-movflags",
        "+faststart",
        str(partial),
    ]
    if has_audio:
        audio_args = ["-af", "asetpts=PTS-STARTPTS", "-c:a", "aac", "-b:a", "192k"]
        command[command.index("-sn"):command.index("-sn")] = audio_args
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg source section trim failed").strip())
    partial.replace(output)
    log_media_message(f"Prepared source section clip: {rel(output)}")
    return rel(output)

def source_section_clip_is_valid(output: Path, ffmpeg: str, expected_duration: float) -> bool:
    if not output.exists() or output.stat().st_size <= 0:
        return False
    try:
        timing = source_section_timing(ffmpeg, output)
    except Exception:
        return True
    first_pts = timing.get("first_pts")
    duration = timing.get("duration")
    if first_pts is not None and abs(first_pts) > 0.25:
        log_media_message(f"Regenerating source section with non-zero first video timestamp: {rel(output)} ({first_pts:.3f}s)")
        return False
    if duration is not None and expected_duration > 0 and abs(duration - expected_duration) > 1.0:
        log_media_message(
            f"Regenerating source section with mismatched duration: {rel(output)} "
            f"({duration:.3f}s, expected {expected_duration:.3f}s)"
        )
        return False
    return True

def source_section_timing(ffmpeg: str, output: Path) -> dict[str, float | None]:
    ffprobe = ffprobe_for_ffmpeg(ffmpeg)
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-read_intervals",
            "%+#1",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=start_time,duration:packet=pts_time",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout or "{}")
    packets = payload.get("packets") or []
    streams = payload.get("streams") or [{}]
    stream = streams[0] if streams else {}
    first_pts = float(packets[0]["pts_time"]) if packets and packets[0].get("pts_time") is not None else _optional_float(stream.get("start_time"))
    return {"first_pts": first_pts, "duration": _optional_float(stream.get("duration"))}

def source_has_audio_stream(ffmpeg: str, source: Path) -> bool:
    try:
        ffprobe = ffprobe_for_ffmpeg(ffmpeg)
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=index",
                "-of",
                "json",
                str(source),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(json.loads(result.stdout or "{}").get("streams"))
    except Exception:
        return True

def ffprobe_for_ffmpeg(ffmpeg: str) -> str:
    path = Path(ffmpeg)
    if path.suffix.lower() == ".exe":
        return str(path.with_name("ffprobe.exe"))
    return "ffprobe"

def _optional_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def log_media_message(message: str) -> None:
    app = getattr(state, "APP", None)
    if app is not None:
        app.log.append(message)

def section_float(value: str, default: float) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default
