from __future__ import annotations

"""Shared open-codec profiles for files that remain in the processing chain."""

INTERMEDIATE_PROFILES = ("low", "medium", "high", "lossless")

_LEGACY_PROFILES = {
    "h264_standard": "low",
    "h264_high": "high",
    "hevc_high": "high",
    "hevc_lossless": "lossless",
}


def canonical_profile(profile: str | None) -> str:
    value = str(profile or "high").strip().lower()
    value = _LEGACY_PROFILES.get(value, value)
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
