from __future__ import annotations

from dataclasses import dataclass

from .config import OUTPAINT_PROMPT, REFERENCE_PROMPT, REFERENCE_PROMPT_SUFFIX

COLORIZE_STAGE_KEYS = {"shots", "references", "colour"}

CLEANUP_PROMPT = (
    "Clean, restored archive footage. Remove vertical scratches, emulsion damage, dirt, dust, "
    "blotches, gate weave, and flicker. Reconstruct clean continuous surfaces and fine natural "
    "detail while preserving the original people, faces, hands, clothing, sets, camera motion, "
    "composition, lighting, contrast, frame timing, and period cinematography. Natural film "
    "texture and stable temporal consistency."
)
CLEANUP_NEGATIVE_PROMPT = (
    "vertical scratches, film scratches, vertical streaks, emulsion damage, dirt, dust, blotches, "
    "gate weave, flicker, frame jitter, torn film, smeared details, warped geometry, altered faces, "
    "altered hands, duplicate limbs, temporal inconsistency, cartoon, game, 3d render, "
    "oversaturated color, color bleeding"
)


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
        "cleanup",
        "Clean Up",
        "Reconstruct masked scratches with ProPainter, correct vignettes, then optionally restore archive footage with Dearchive. Source geometry is preserved.",
        ("intermediate/cleaned",),
        (
            ("ai_descratch", "AI DeScratch (ProPainter)", "checkbox", "false"),
            ("scratch_sensitivity", "Scratch detection sensitivity", "range:0|1|0.05", "0.65"),
            ("scratch_mask_dilate", "Scratch mask expansion", "range:0|12|1", "3"),
            ("ai_descratch_height", "AI DeScratch resolution", "select:540|720|1080|source", "720"),
            ("ai_chunk_frames", "AI DeScratch chunk frames", "select:25|33|41|49", "41"),
            ("save_scratch_mask", "Save scratch-mask preview", "checkbox", "true"),
            ("devignette", "DeVignette", "checkbox", "false"),
            ("dearchive", "Dearchive (LTX 2.3 LoRA)", "checkbox", "true"),
            ("repair_device", "DeVignette processor", "select:auto|cuda|cpu", "auto"),
            ("chunk_seconds", "Dearchive chunk length", "range:2|20|0.01", "4.04"),
            ("overlap_frames", "Overlap frames", "number", "8"),
            ("source_fidelity", "Source Fidelity", "range:0|1|0.05", "1.0"),
            ("prompt", "Prompt", "text", CLEANUP_PROMPT),
            ("negative_prompt", "Negative prompt", "text", CLEANUP_NEGATIVE_PROMPT),
            ("lora_strength", "Dearchive LoRA strength", "number", "1.0"),
            ("seed", "Seed", "number", "42"),
        ),
        (),
    ),
    Stage(
        "outpaint",
        "Outpainting",
        "Prepare the source clip chosen on the Global tab for LTX outpainting.",
        ("input", "intermediate/outpaint_prepared", "intermediate/outpainted"),
        (
            ("target_aspect", "Target aspect ratio", "select:16:9|9:16|4:3|3:4|1:1|21:9|2.39:1|2.35:1|1.85:1|3:2|2:3|5:4|4:5", "16:9"),
            ("target_height", "Output height", "select:source|480|544|576|720|768|1080", "source"),
            ("chunk_seconds", "Chunk seconds", "number", "20"),
            ("overlap_frames", "Overlap frames", "range:0|48|1", "8"),
            ("seed_qwen_guides", "Seed with Qwen guide frames", "checkbox", "false"),
            ("outpaint_all_black_regions", "Outpaint all black regions", "checkbox", "false"),
            ("prompt", "Prompt", "text", OUTPAINT_PROMPT),
            ("negative_prompt", "Negative prompt", "text", "cartoon, game, 3d render, still image, static, warped geometry, flicker, smeared details, extra fingers, broken fingers, deformed hands"),
            ("crop_left", "Crop left", "range:0|960|1", "0"),
            ("crop_right", "Crop right", "range:0|960|1", "0"),
            ("crop_top", "Crop top", "range:0|960|1", "0"),
            ("crop_bottom", "Crop bottom", "range:0|960|1", "0"),
        ),
        (),
    ),
    Stage(
        "shots",
        "Shot Detection",
        "Detect cuts and divide the video into sections for independent colorization.",
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
            ("method", "Method", "select:qwen|openai", "qwen"),
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
        "Run reference-guided video colorization over the outpainted video.",
        ("intermediate/outpainted_references_color", "intermediate/outpainted_colorized", "manifests/references"),
        (
            ("manifest", "Manifest", "file", ""),
            ("method", "Method", "select:deepexemplar|colormnet|both", "deepexemplar"),
            ("processing_height", "Processing scale", "select:source|2160|1440|1080|720|540", "source"),
            ("frame_propagate", "Frame propagation", "select:true|false", "true"),
            ("use_half_resolution", "Half-resolution processing", "checkbox", "true"),
            ("use_torch_compile", "Torch compile", "select:false|true", "false"),
            ("use_sage_attention", "SageAttention", "select:false|true", "false"),
            ("colormnet_memory_mode", "ColorMNet memory", "select:balanced|low_memory|high_quality", "balanced"),
            ("colormnet_feature_encoder", "ColorMNet encoder", "select:resnet50|vgg19|dinov2_vits|dinov2_vitb|dinov2_vitl|clip_vitb", "resnet50"),
            ("colormnet_text_guidance", "ColorMNet text guidance", "text", ""),
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
            ("colorization_method", "Colorization layer", "select:deepexemplar|colormnet", "deepexemplar"),
            ("colorized_video", "Colorized video", "file", ""),
            ("feather_pixels", "Feather pixels", "range:0|240|1", "80"),
            ("saturation", "Saturation", "range:0|200|1", "82"),
            ("temperature", "Temperature", "range:2500|9500|1", "6500"),
            ("color_opacity", "Color opacity", "range:0|100|1", "100"),
            ("encoder", "Encoder", "select:h264|prores", "h264"),
        ),
        ("source",),
    ),
    Stage(
        "audio",
        "Create Audio Track",
        "Generate a musical score and/or synchronized sound effects for a silent film and mux them onto the latest render.",
        ("output/reassembled", "intermediate/audio", "output/with_soundtrack"),
        (
            ("input_video", "Input video", "file", ""),
            ("create_music", "Create Music", "checkbox", "true"),
            ("create_sfx", "Create Sound Effects", "checkbox", "true"),
            ("music_prompt", "Music style hint", "text", ""),
            ("music_negative_prompt", "Music negative", "text", "low quality, distorted, noisy, clipping"),
            ("music_cue_seconds", "Music cue seconds", "number", "30"),
            ("music_checkpoint", "Stable Audio checkpoint", "text", "stable_audio_open_1.0.safetensors"),
            ("sfx_prompt", "Sound effects hint", "text", ""),
            ("sfx_negative_prompt", "Sound effects negative", "text", "music, song, singing, speech, voice"),
            ("sfx_chunk_seconds", "SFX chunk seconds", "number", "8"),
            ("sfx_short_side", "MMAudio analysis short side", "number", "384"),
            ("music_gain_db", "Music level (dB)", "number", "-9"),
            ("sfx_gain_db", "Sound effects level (dB)", "number", "0"),
            ("seed", "Seed", "number", "42"),
            ("caption_node", "Qwen-VL caption node (advanced)", "text", ""),
            ("ollama_vision_model", "Ollama caption model (auto/off/name)", "text", "auto"),
        ),
        (),
    ),
    Stage(
        "upscale",
        "Upscaling",
        "Optionally upscale the composited render or selected source section.",
        ("output/reassembled", "output/upscaled"),
        (
            ("input_video", "Input video", "file", ""),
            ("target_width", "Target width", "number", "3840"),
            ("target_height", "Target height", "number", "2160"),
            ("output", "Upscaled output", "save", ""),
            ("flashvsr_model", "FlashVSR model", "select:FlashVSR|FlashVSR-v1.1", "FlashVSR-v1.1"),
            ("flashvsr_mode", "FlashVSR mode", "select:tiny|tiny-long|full", "tiny"),
            ("flashvsr_scale", "FlashVSR scale", "select:2|3|4", "2"),
            ("flashvsr_pre_downscale", "Pre-downscale input", "checkbox", "false"),
            ("flashvsr_tiled_dit", "Tiled diffusion (tiled_dit)", "checkbox", "true"),
            ("flashvsr_tile_size", "Tile size (px)", "number", "256"),
            ("flashvsr_tile_overlap", "Tile overlap (px)", "number", "24"),
            ("flashvsr_vae_tile_multiplier", "VAE tile multiplier", "number", "1"),
            ("flashvsr_local_range", "Temporal window (local_range)", "select:9|11", "11"),
            ("flashvsr_sparse_ratio", "Attention density (sparse_ratio)", "number", "2.0"),
            ("flashvsr_kv_ratio", "Attention memory (kv_ratio)", "number", "3.0"),
            ("flashvsr_color_fix", "Wavelet color fix (color_fix)", "checkbox", "true"),
            ("flashvsr_tiled_vae", "Tiled decode (tiled_vae)", "checkbox", "true"),
            ("flashvsr_unload_dit", "Unload before decode (unload_dit)", "checkbox", "false"),
            ("flashvsr_seed", "FlashVSR seed", "number", "0"),
            ("blend_strength", "Default AI upscale strength", "range:0|100|1", "100"),
            ("chunk_seconds", "Chunk seconds", "number", "6"),
            ("overlap_frames", "Overlap frames", "number", "8"),
            ("preview_seconds", "Preview seconds", "number", "6"),
        ),
        (),
    ),
    Stage(
        "output",
        "Output",
        "Preview the best available render once processing has finished.",
        ("output/reassembled", "output/upscaled"),
        (("output", "Selected output", "file", ""),),
        (),
    ),
)


def output_stage() -> Stage:
    return next(stage for stage in STAGES if stage.key == "output")
