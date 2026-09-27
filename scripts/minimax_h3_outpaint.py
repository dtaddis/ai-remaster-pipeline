"""MiniMax H3 outpainting backend.

H3's Fun ControlNet-Union patch has a video-inpainting mode: a source video plus a mask where 1
marks pixels to regenerate. That is the same contract as Wan VACE's control video and mask, so
H3 reuses ARP's masked-pass chunk renderer (wan_vace_outpaint.render_chunk): the same masks,
previous-chunk context frames, guides, shot-cut splitting and seam composite. Only the graph,
the frame grid (17k + 5 instead of 4n + 1) and the pass limits differ.

The graph follows ComfyUI's own video_minimax_h3_fun_controlnet_union template: an FL2VA
transformer with the Fun ControlNet applied as a model patch, MiniMaxH3ImageToVideo for the
text conditioning and AV latent, BasicGuider at guidance 1 and res_multistep sampling. With PDD
the model also goes through ModelSamplingMiniMaxH3 (12/3) and MiniMaxH3PDDAccApply, whose
trained 8-step sigmas drive an euler sampler (the pack's required recipe).

The open weights are licensed under the MiniMax H3 Community License, which excludes the EU, UK,
South Korea and USA; outpaint_video.py refuses to download or run them until the user confirms
they are licensed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dependency_manager import H3_DIFFUSION_GGUF, H3_FUN_CONTROLNET, H3_PDD_ACC, H3_TEXT_ENCODER_GGUF, H3_VIDEO_VAE
from intermediate_video import comfy_video_combine_inputs

# H3 is trained on 124-362 frames (about 5-15 s at 24 fps) and renders a chunk in one pass, so
# chunks are capped at the top of that range; Chunk seconds chooses anything shorter.
H3_MAX_CHUNK_FRAMES = 362
# The longest pass that stays on a 24 GB card: measured on an RTX 4090 (2026-09-27, Q5_K_M
# transformer + Fun ControlNet, 20 steps), 124 frames at 1536x640 ran at full power in ~17 min;
# 192 frames started offloading weights to system RAM (~95 W, VRAM pinned) and crawled for hours.
H3_MAX_PASS_PIXELS = 124 * 1536 * 640
# The shortest pass worth running: the 22 context frames plus one latent group of new ones.
H3_MIN_PASS_FRAMES = 39
# Finished frames each pass opens with: one latent group past the first (5 + 17), so the kept
# context ends on a latent boundary instead of splitting one.
H3_CONTEXT_FRAMES = 22
H3_OUTPUT_NODE_ID = "23"
# PDD's heads are trained on 8 evaluations at sigma shift 12 (video) / 3 (audio); the pack
# refuses any other shift and any sampler that evaluates between its block boundaries.
PDD_STEPS = 8
PDD_SHIFT_VIDEO = 12.0
PDD_SHIFT_AUDIO = 3.0
# H3 is steered almost entirely by its prompt (guidance 1 runs no negative), so the default
# "outpaint" prompt, which is stripped for Wan and H3, is replaced by a plain scene continuation.
H3_DEFAULT_PROMPT = (
    "The same scene continues naturally beyond the original frame edges, filmed with the same "
    "camera, lens, lighting and film stock; one continuous shot filling the whole frame."
)

H3_REQUIRED_NODES = {
    "UnetLoaderGGUF": "ComfyUI-GGUF",
    "CLIPLoaderGGUF": "ComfyUI-GGUF",
    "VHS_VideoCombine": "ComfyUI-VideoHelperSuite",
    "MiniMaxH3ImageToVideo": "ComfyUI core (0.30+)",
    "MiniMaxH3FunControlNetApply": "ComfyUI core (0.30+)",
    "MiniMaxH3SigmaShift": "ComfyUI core (0.30+)",
    "ModelPatchLoader": "ComfyUI core",
    "VAELoader": "ComfyUI core",
    "BasicGuider": "ComfyUI core",
    "BasicScheduler": "ComfyUI core",
    "KSamplerSelect": "ComfyUI core",
    "RandomNoise": "ComfyUI core",
    "SamplerCustomAdvanced": "ComfyUI core",
    "VAEDecode": "ComfyUI core",
    "LoadVideo": "ComfyUI core",
    "GetVideoComponents": "ComfyUI core",
    "ImageToMask": "ComfyUI core",
    "ImageFromBatch": "ComfyUI core",
    "MiniMaxH3AddGuide": "ComfyUI core (0.30+)",
}


def h3_valid_length(frame_count: int) -> int:
    """Round up to H3's temporal grid (17k + 5 frames), like ComfyUI's align_frame_count."""
    count = max(5, int(frame_count))
    return count + (5 - count % 17) % 17


@dataclass(frozen=True)
class H3Settings:
    steps: int = 20
    sampler: str = "res_multistep"
    scheduler: str = "simple"
    control_strength: float = 1.0
    # Alibaba PAI's PDD 8-step distillation (what Wan2GP offers as "PDD 8-Step"): 2.5x faster.
    # Distilled models can keep the unmasked source less faithfully, so compare with full steps.
    pdd: bool = False
    chunk_frames: int = H3_MAX_CHUNK_FRAMES
    context_frames: int = H3_CONTEXT_FRAMES
    # render_chunk's interface; H3 always renders a chunk in one sequential pass.
    sampling = "sequential"
    context_overlap = 0
    label = "MiniMax H3"

    def valid_length(self, frame_count: int) -> int:
        return h3_valid_length(frame_count)

    def chunk_frames_for(self, width: int, height: int) -> int:
        """Frames in one pass at this canvas size, on the 17k + 5 grid and within 24 GB.

        The previous chunk's context frames are anchored as an extra guide clip that the model
        attends to alongside the pass, so they count against the budget too: with them a
        124-frame pass at 1536x640 began offloading and took twice as long.
        """
        by_pixels = H3_MAX_PASS_PIXELS // max(1, int(width) * int(height)) - int(self.context_frames)
        frames = max(H3_MIN_PASS_FRAMES, min(int(self.chunk_frames), by_pixels))
        return frames - (frames - 5) % 17

    def pass_frames(self, width: int, height: int) -> int:
        return self.chunk_frames_for(width, height)

    def sampling_steps(self) -> int:
        return PDD_STEPS if self.pdd else int(self.steps)

    def signature(self) -> dict[str, Any]:
        return {
            "model": H3_DIFFUSION_GGUF,
            "text_encoder": H3_TEXT_ENCODER_GGUF,
            "vae": H3_VIDEO_VAE,
            "fun_controlnet": H3_FUN_CONTROLNET,
            "pdd_acc": H3_PDD_ACC if self.pdd else "",
            "steps": self.sampling_steps(),
            "sampler": "euler" if self.pdd else self.sampler,
            "scheduler": "pdd" if self.pdd else self.scheduler,
            "control_strength": self.control_strength,
            "chunk_frames": self.chunk_frames,
            "context_frames": self.context_frames,
            # Bump when the graph changes how a pass is conditioned, so cached chunks re-render.
            "graph": "fun_inpaint+context_guide_v2",
        }


def build_h3_prompt(
    control_name: str,
    mask_name: str,
    width: int,
    height: int,
    length: int,
    fps: float,
    prompt_text: str,
    seed: int,
    settings: H3Settings,
    output_prefix: str,
    context_frames: int = 0,
) -> dict[str, Any]:
    """API prompt for one H3 pass. The control video is the source with generated regions in
    mid-grey; the Fun ControlNet re-blanks masked pixels itself, so the grey only has to agree.

    The pass's first context_frames frames are finished frames of the previous chunk. The Fun
    ControlNet alone treats them only as a hint, so a new chunk opened with a visible jump; they
    are also anchored with MiniMaxH3AddGuide, whose keyframe latents are re-injected every step
    and never denoised, so the pass continues from them exactly.
    """
    model_source = ["10", 0]
    prompt: dict[str, Any] = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": control_name}, "_meta": {"title": "ARP H3 source video"}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "LoadVideo", "inputs": {"file": mask_name}, "_meta": {"title": "ARP H3 regenerate mask"}},
        "4": {"class_type": "GetVideoComponents", "inputs": {"video": ["3", 0]}},
        "5": {"class_type": "ImageToMask", "inputs": {"image": ["4", 0], "channel": "red"}},
        "10": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": H3_DIFFUSION_GGUF}},
        "12": {"class_type": "ModelPatchLoader", "inputs": {"name": H3_FUN_CONTROLNET}},
        "13": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": H3_TEXT_ENCODER_GGUF, "type": "minimax"}},
        "14": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VIDEO_VAE}},
    }
    if settings.pdd:
        prompt["11"] = {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {"model": ["10", 0], "shift_video": PDD_SHIFT_VIDEO, "shift_audio": PDD_SHIFT_AUDIO},
        }
        prompt["24"] = {
            "class_type": "MiniMaxH3PDDAccApply",
            "inputs": {
                "model": ["11", 0],
                "pdd_file": H3_PDD_ACC,
                "nfe": str(PDD_STEPS),
                "lora_strength": 1.0,
                "head_strength": 1.0,
                "on_off_grid": "error",
            },
        }
        model_source = ["24", 0]
    prompt.update({
        "15": {
            "class_type": "MiniMaxH3FunControlNetApply",
            "inputs": {
                "model": model_source,
                "model_patch": ["12", 0],
                "vae": ["14", 0],
                "strength": float(settings.control_strength),
                "start_percent": 0.0,
                "end_percent": 1.0,
                "mask": ["5", 0],
                "source_video": ["2", 0],
            },
        },
        "16": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["13", 0],
                "vae": ["14", 0],
                "prompt": prompt_text,
                "width": int(width),
                "height": int(height),
                "length": int(length),
            },
        },
        "17": {"class_type": "BasicGuider", "inputs": {"model": ["15", 0], "conditioning": ["16", 0]}},
        "18": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler" if settings.pdd else settings.sampler}},
        "20": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "21": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["20", 0],
                "guider": ["17", 0],
                "sampler": ["18", 0],
                # PDD supplies its trained block-boundary sigmas; otherwise a normal schedule.
                "sigmas": ["24", 1] if settings.pdd else ["19", 0],
                "latent_image": ["16", 1],
            },
        },
        "22": {"class_type": "VAEDecode", "inputs": {"samples": ["21", 0], "vae": ["14", 0]}},
        H3_OUTPUT_NODE_ID: {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["22", 0],
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
    })
    # AddGuide anchors clips of 17k + 5 frames (it crops longer batches); under 5 it would keep
    # only one frame, which is not a continuation.
    if context_frames >= 5:
        prompt["25"] = {
            "class_type": "ImageFromBatch",
            "inputs": {"image": ["2", 0], "batch_index": 0, "length": int(context_frames)},
        }
        prompt["26"] = {
            "class_type": "MiniMaxH3AddGuide",
            "inputs": {"positive": ["16", 0], "vae": ["14", 0], "latent": ["16", 1], "image": ["25", 0], "frame_idx": 0},
        }
        prompt["17"]["inputs"]["conditioning"] = ["26", 0]
    if not settings.pdd:
        prompt["19"] = {
            "class_type": "BasicScheduler",
            "inputs": {"model": ["15", 0], "scheduler": settings.scheduler, "steps": settings.sampling_steps(), "denoise": 1.0},
        }
    return prompt
