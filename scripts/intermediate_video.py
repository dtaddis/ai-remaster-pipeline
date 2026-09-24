from __future__ import annotations

"""Shared open-codec profiles for files that remain in the processing chain."""

from pathlib import Path

INTERMEDIATE_PROFILES = ("low", "medium", "high", "lossless")

_LEGACY_PROFILES = {
    "h264_standard": "low",
    "h264_high": "high",
    "hevc_high": "high",
    "hevc_lossless": "lossless",
}


def migrate_profile_name(profile: str | None) -> str:
    value = str(profile or "high").strip().lower()
    return _LEGACY_PROFILES.get(value, value)


def canonical_profile(profile: str | None) -> str:
    value = migrate_profile_name(profile)
    return value if value in INTERMEDIATE_PROFILES else "high"


def codec_args(profile: str | None, *, fast: bool = False) -> list[str]:
    """Return a scale-independent Matroska working codec.

    VP9/libvpx is used for the three lossy profiles and FFV1 v3 for lossless.
    Neither path uses HEVC/x265. CRF is preferable to a fixed bitrate here: a
    704p model input and a 4K intermediate receive comparable visual quality
    without either starving the larger frame or wasting space on the smaller one.
    """

    selected = canonical_profile(profile)
    cpu_used = "4" if fast else "2"
    if selected == "low":
        return ["-c:v", "libvpx-vp9", "-crf", "36", "-b:v", "0", "-deadline", "good", "-cpu-used", cpu_used, "-row-mt", "1", "-pix_fmt", "yuv420p10le"]
    if selected == "medium":
        return ["-c:v", "libvpx-vp9", "-crf", "27", "-b:v", "0", "-deadline", "good", "-cpu-used", cpu_used, "-row-mt", "1", "-pix_fmt", "yuv420p10le"]
    if selected == "lossless":
        return [
            "-c:v", "ffv1", "-level", "3", "-coder", "1", "-context", "1",
            "-g", "1", "-slicecrc", "1",
        ]
    return ["-c:v", "libvpx-vp9", "-crf", "18", "-b:v", "0", "-deadline", "good", "-cpu-used", cpu_used, "-row-mt", "1", "-pix_fmt", "yuv420p10le"]


# ARP re-encodes every ComfyUI render into the working codec, so lossy profiles save
# those renders a step above their own quality and the profile's encode stays the
# only visible generation loss.
_COMFY_H264_CRF = {"low": 20, "medium": 16, "high": 12}


def comfy_video_combine_inputs(profile: str | None) -> dict[str, object]:
    """VHS_VideoCombine format inputs for a ComfyUI render that feeds the processing chain."""

    selected = canonical_profile(profile)
    if selected == "lossless":
        return {
            "format": "video/ffv1-mkv", "level": "3", "coder": "1", "context": "1",
            "gop_size": 1, "slices": "16", "slicecrc": "1", "pix_fmt": "bgra",
        }
    return {"format": "video/h264-mp4", "pix_fmt": "yuv420p10le", "crf": _COMFY_H264_CRF[selected]}


_VHS_ENCODER_INPUTS = {
    "format", "pix_fmt", "crf", "bitrate", "megabit",
    "level", "coder", "context", "gop_size", "slices", "slicecrc",
}


def vhs_inputs_for_profile(inputs: dict, profile: str | None) -> dict:
    """Replace a VHS_VideoCombine API node's encoder settings with the profile's."""

    kept = {key: value for key, value in inputs.items() if key not in _VHS_ENCODER_INPUTS}
    return {**kept, **comfy_video_combine_inputs(profile)}


def existing_comfy_render(path: Path) -> Path:
    """A cached ComfyUI render is .mkv (lossless profile) or .mp4 (lossy profiles, older caches)."""

    other = path.with_suffix(".mp4" if path.suffix.lower() == ".mkv" else ".mkv")
    return other if not path.exists() and other.exists() else path


def container_args(path: str, profile: str | None) -> list[str]:
    # Force Matroska even for legacy cache filenames whose suffix predates the
    # profile migration. Newly created user-visible intermediates use .mkv.
    return ["-f", "matroska"]


def extension(_profile: str | None = None) -> str:
    return "mkv"


def audio_codec_args(profile: str | None, *, bitrate: str = "192k") -> list[str]:
    if canonical_profile(profile) == "lossless":
        return ["-c:a", "flac"]
    return ["-c:a", "libopus", "-b:a", bitrate]
