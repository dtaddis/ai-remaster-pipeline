"""ARP-owned ComfyUI nodes.

The LTX execution patch is intentionally installed only when ARP's dedicated
video-only IC-LoRA loader executes. Importing ComfyUI or ComfyUI-LTXVideo does
not alter global model behavior.
"""

from __future__ import annotations

import gc
import logging

from aiohttp import web
import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
import torch
from server import PromptServer

from .ltx_video_only_patch import (
    install_sparse_guide_attention_patch,
    ltx_runtime_capabilities,
    prune_ltxav_audio_transformer_blocks,
)


LOGGER = logging.getLogger(__name__)


@PromptServer.instance.routes.get("/arp/ltx-compatibility")
async def arp_ltx_compatibility(_request):
    """Report whether this live Comfy/LTX combination supports ARP's adapter."""

    try:
        return web.json_response(ltx_runtime_capabilities())
    except Exception as exc:
        LOGGER.exception("ARP LTX compatibility preflight failed")
        return web.json_response(
            {"compatible": False, "adapter": "none", "error": str(exc)}
        )


class ARPLTXVideoOnlyICLoRALoader:
    """Load ARP's outpaint LoRA and opt this model into video-only execution."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "FLOAT")
    RETURN_NAMES = ("model", "latent_downscale_factor")
    FUNCTION = "load_lora"
    CATEGORY = "ARP/LTX"
    DESCRIPTION = (
        "ARP-scoped LTX IC-LoRA loader for video-only outpainting. It lazily "
        "enables exact guide attention and bounded feed-forward execution."
    )

    def load_lora(self, model, lora_name, strength_model):
        # Validate and install before mutating the model. Incompatible ComfyUI
        # updates fail here with an actionable error instead of silently using
        # different attention semantics or hanging in an unsupported kernel.
        install_sparse_guide_attention_patch()

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora, metadata = comfy.utils.load_torch_file(
            lora_path, safe_load=True, return_metadata=True
        )
        try:
            latent_downscale_factor = float(metadata["reference_downscale_factor"])
        except (KeyError, ValueError, TypeError):
            latent_downscale_factor = 1.0
            LOGGER.warning(
                "Failed to extract reference_downscale_factor from %s; using 1.0",
                lora_path,
            )

        model_lora = model
        if strength_model != 0:
            model_lora, _ = comfy.sd.load_lora_for_models(
                model, None, lora, strength_model, 0
            )
        video_only_model = prune_ltxav_audio_transformer_blocks(model_lora)
        install_sparse_guide_attention_patch(video_only_model)
        return video_only_model, latent_downscale_factor


class ARPLTXEphemeralTextEncode:
    """Encode both prompts, then release the large LTX text encoder.

    Returning a CLIP object from a conventional loader makes Comfy's execution
    cache retain Gemma for the whole diffusion run.  On a 64 GB / RTX 4090
    workstation that leaves too little host RAM for GGUF offload and guide
    attention.  This combined node keeps only the small conditioning tensors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": (folder_paths.get_filename_list("text_encoders"),),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "device": (["default", "cpu"], {"default": "cpu"}),
                "positive": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "negative": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "ARP/LTX"
    DESCRIPTION = (
        "Encodes both LTX prompts and immediately releases Gemma before video "
        "diffusion, preventing the text encoder from consuming host RAM for the run."
    )

    def encode(self, text_encoder, ckpt_name, device, positive, negative):
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", text_encoder)
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        model_options = {}
        if device == "cpu":
            cpu = torch.device("cpu")
            model_options["load_device"] = cpu
            model_options["offload_device"] = cpu

        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path, ckpt_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.LTXV,
            model_options=model_options,
        )
        try:
            positive_conditioning = clip.encode_from_tokens_scheduled(
                clip.tokenize(positive)
            )
            negative_conditioning = clip.encode_from_tokens_scheduled(
                clip.tokenize(negative)
            )
        finally:
            # The conditionings are tensors and do not need the CLIP wrapper. Once
            # this last strong reference is gone, Comfy's weak loaded-model entry
            # can be removed and the encoder's CPU tensors reclaimed.
            del clip
            gc.collect()
            comfy.model_management.cleanup_models()
            gc.collect()

        LOGGER.info("ARP released the LTX text encoder after prompt encoding")
        return positive_conditioning, negative_conditioning


NODE_CLASS_MAPPINGS = {
    "ARPLTXVideoOnlyICLoRALoader": ARPLTXVideoOnlyICLoRALoader,
    "ARPLTXEphemeralTextEncode": ARPLTXEphemeralTextEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ARPLTXVideoOnlyICLoRALoader": "ARP LTX Video-Only IC-LoRA Loader",
    "ARPLTXEphemeralTextEncode": "ARP LTX Ephemeral Text Encode",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
