from __future__ import annotations

from dataclasses import dataclass

from .config import REFERENCE_PROMPT, REFERENCE_PROMPT_SUFFIX

COLORIZE_STAGE_KEYS = {"shots", "references", "colour"}


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    description: str
    folders: tuple[str, ...]
    fields: tuple[tuple[str, str, str, str], ...]
    required: tuple[str, ...]


STAGES = (
    Stage(
        "outpaint",
        "Outpainting",
        "Prepare the source clip chosen on the Global tab for LTX outpainting.",
        ("input", "intermediate/outpaint_prepared", "intermediate/outpainted"),
        (
            ("target_aspect", "Target aspect ratio", "select:16:9|9:16|4:3|3:4|1:1|21:9|2.39:1|2.35:1|1.85:1|3:2|2:3|5:4|4:5", "16:9"),
            ("target_height", "Output height", "select:544|576|720|768|1080", "720"),
            ("chunk_seconds", "Chunk seconds", "number", "4"),
            ("overlap_frames", "Overlap frames", "range:0|48|1", "8"),
            ("crop_left", "Crop left", "range:0|240|1", "0"),
            ("crop_right", "Crop right", "range:0|240|1", "0"),
            ("crop_top", "Crop top", "range:0|240|1", "0"),
            ("crop_bottom", "Crop bottom", "range:0|240|1", "0"),
        ),
        (),
    ),
    Stage(
        "output",
        "Output",
        "Preview the final remastered movie once recomposition has finished.",
        ("output/reassembled",),
        (("output", "Final output", "file", ""),),
        (),
    ),
    Stage(
        "shots",
        "Shot Detection",
        "Detect cuts and extract one useful reference frame per shot.",
        ("intermediate/outpainted", "intermediate/outpainted_references", "manifests/references"),
        (
            ("outpainted_video", "Outpainted video", "file", ""),
            ("sample_seconds", "Sample seconds", "number", "0"),
            ("shot_threshold", "Shot threshold", "number", "0.075"),
            ("min_shot_seconds", "Minimum shot seconds", "number", "1.0"),
            ("limit", "Limit rows", "number", ""),
        ),
        ("outpainted_video",),
    ),
    Stage(
        "references",
        "Reference Generation",
        "Colorize extracted stills through a Qwen Image Edit ComfyUI workflow.",
        ("intermediate/outpainted_references", "intermediate/outpainted_references_color", "manifests/references"),
        (
            ("manifest", "Manifest", "file", ""),
            ("prompt", "Prompt", "text", REFERENCE_PROMPT),
            ("prompt_suffix", "Prompt suffix", "text", REFERENCE_PROMPT_SUFFIX),
            ("limit", "Limit rows", "number", ""),
        ),
        ("manifest",),
    ),
    Stage(
        "colour",
        "Colorization",
        "Run Deep Exemplar in ComfyUI over the outpainted video, using the generated color references.",
        ("intermediate/outpainted_references_color", "intermediate/outpainted_colorized", "manifests/references"),
        (
            ("manifest", "Manifest", "file", ""),
            ("frame_propagate", "Frame propagation", "select:true|false", "true"),
            ("use_half_resolution", "Half-resolution processing", "select:true|false", "true"),
            ("use_torch_compile", "Torch compile", "select:false|true", "false"),
            ("use_sage_attention", "SageAttention", "select:false|true", "false"),
            ("crf", "CRF", "number", "18"),
        ),
        ("manifest",),
    ),
    Stage(
        "recomp",
        "Recomposition",
        "Composite outpainted video, original centre footage, and optional colorized video.",
        ("input", "intermediate/outpainted", "intermediate/outpainted_colorized", "output/reassembled"),
        (
            ("outpainted_video", "Outpainted video", "file", ""),
            ("source", "Original source", "file", ""),
            ("colorized_video", "Colorized video", "file", ""),
            ("feather_pixels", "Feather pixels", "number", "80"),
            ("saturation", "Saturation", "number", "0.82"),
            ("temperature", "Temperature", "number", "-0.015"),
            ("color_opacity", "Color opacity", "number", "1.0"),
            ("encoder", "Encoder", "select:h264|prores", "h264"),
        ),
        ("outpainted_video", "source"),
    ),
)


def output_stage() -> Stage:
    return next(stage for stage in STAGES if stage.key == "output")
