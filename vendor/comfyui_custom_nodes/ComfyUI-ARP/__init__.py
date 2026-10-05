"""ARP-owned ComfyUI nodes.

The LTX execution patch is intentionally installed only when ARP's dedicated
video-only IC-LoRA loader executes. Importing ComfyUI or ComfyUI-LTXVideo does
not alter global model behavior.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
from pathlib import Path

from aiohttp import web
import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
import torch
from server import PromptServer

from .ltx_video_only_patch import (
    ATTENTION_BACKENDS,
    install_attention_backend,
    install_frozen_reference,
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
            },
            "optional": {
                "attention_backend": (
                    ["default", *ATTENTION_BACKENDS],
                    {
                        "default": "default",
                        "tooltip": "int8 uses ComfyUI's bundled int8 attention kernel for this model only.",
                    },
                ),
                "frozen_reference": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Experimental: the IC reference attends only to itself, so its "
                        "keys/values are computed once and reused on later steps.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "FLOAT")
    RETURN_NAMES = ("model", "latent_downscale_factor")
    FUNCTION = "load_lora"
    CATEGORY = "ARP/LTX"
    DESCRIPTION = (
        "ARP-scoped LTX IC-LoRA loader for video-only outpainting. It lazily "
        "enables exact guide attention and bounded feed-forward execution."
    )

    def load_lora(self, model, lora_name, strength_model, attention_backend="default", frozen_reference=False):
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
        install_attention_backend(video_only_model, attention_backend)
        if frozen_reference:
            install_frozen_reference(video_only_model)
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
        cache_path = _conditioning_cache_path(clip_path, ckpt_path, device, positive, negative)
        cached = _load_cached_conditioning(cache_path)
        if cached is not None:
            LOGGER.info("ARP reused cached LTX prompt encoding (%s)", cache_path.name)
            return cached
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
        _save_cached_conditioning(cache_path, (positive_conditioning, negative_conditioning))
        return positive_conditioning, negative_conditioning


# Encoding both prompts loads ~15 GB of Gemma and runs it on the CPU: minutes per
# outpaint run. The result depends only on the encoder files, device and the two
# prompt texts, so keep it on disk.
CONDITIONING_CACHE_LIMIT = 64


def _conditioning_cache_dir() -> Path:
    return Path(folder_paths.get_user_directory()) / "arp_cache" / "ltx_prompt_encodings"


def _file_identity(path: str) -> list:
    stat = os.stat(path)
    return [os.path.basename(path), stat.st_size, stat.st_mtime_ns]


def _conditioning_cache_path(clip_path: str, ckpt_path: str, device: str, positive: str, negative: str) -> Path:
    identity = json.dumps(
        {
            "version": 1,
            "text_encoder": _file_identity(clip_path),
            "checkpoint": _file_identity(ckpt_path),
            "device": device,
            "positive": positive,
            "negative": negative,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    return _conditioning_cache_dir() / f"{digest}.pt"


def _load_cached_conditioning(path: Path):
    if not path.is_file():
        return None
    try:
        positive, negative = torch.load(path, map_location="cpu", weights_only=True)
        os.utime(path)  # most recently used survives pruning
        return positive, negative
    except Exception as exc:
        LOGGER.warning("ARP ignored an unreadable cached prompt encoding %s: %s", path, exc)
        return None


def _save_cached_conditioning(path: Path, conditioning) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(conditioning, temporary)
        os.replace(temporary, path)
        entries = sorted(path.parent.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
        for stale in entries[CONDITIONING_CACHE_LIMIT:]:
            stale.unlink(missing_ok=True)
    except Exception as exc:
        LOGGER.warning("ARP could not cache the prompt encoding: %s", exc)


NODE_CLASS_MAPPINGS = {
    "ARPLTXVideoOnlyICLoRALoader": ARPLTXVideoOnlyICLoRALoader,
    "ARPLTXEphemeralTextEncode": ARPLTXEphemeralTextEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ARPLTXVideoOnlyICLoRALoader": "ARP LTX Video-Only IC-LoRA Loader",
    "ARPLTXEphemeralTextEncode": "ARP LTX Ephemeral Text Encode",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
