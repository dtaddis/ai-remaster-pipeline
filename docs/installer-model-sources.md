# Installer Model Sources

Model downloads are handled on demand by the pipeline stage that needs them. `install_windows.ps1 -DownloadModels` can still prefetch the default Windows/NVIDIA model set.

## LTX 2.3

- Base checkpoint: `Lightricks/LTX-2.3-fp8/ltx-2.3-22b-dev-fp8.safetensors`
- Dearchive/outpainting transformer: `QuantStack/LTX-2.3-GGUF/LTX-2.3-distilled/LTX-2.3-distilled-Q4_K_M.gguf` → `models/unet/LTX-2.3-distilled-Q4_K_M.gguf`
- Text encoder: `Comfy-Org/ltx-2/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors`
- Audio VAE: `Kijai/LTX2.3_comfy/vae/LTX23_audio_vae_bf16.safetensors`
- Distilled LoRA for the full dev-model backend: `Lightricks/LTX-2.3/ltx-2.3-22b-distilled-lora-384-1.1.safetensors` → `models/loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-384-1.1.safetensors`. The default distilled-GGUF backend bypasses this 7.6 GB LoRA because its weights are already distilled.
- Clean Up / Dearchive LoRA: `oumoumad/ltx-2.3-dearchive-lora/lora_weights_step_05000.safetensors` → `models/loras/ltx-2.3-dearchive-lora.safetensors`
- Outpainting LoRA (default): `oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint/ltx-2.3-22b-ic-lora-outpaint.safetensors` → `models/loras/ltx-2.3-22b-ic-lora-outpaint.safetensors`
- In/Outpainting LoRA: `Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors` (gated; accept/request browser access for the same individual account used by ARP, then run `hf auth login --force` if the saved token still receives a 403)

## LTX 2.5

- Distilled transformer: `vantagewithai/LTX-2.5-GGUF/distilled/ltx-2.5-22b-distilled-transformer-Q4_K_M.gguf`
- Text encoder: `elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF/gemma4-12b-with-proj-ltx-2.5-Q5_K_M.gguf`
- Video/audio VAEs and latent upscaler: `Lightricks/LTX-2.5`
- In/Outpainting LoRA: the gated LTX 2.3 model listed above

The quantized text encoder is also gated. Accept its terms in the browser, then authenticate ARP's
local downloader with `hf auth login --force` from the activated ARP environment (or set `HF_TOKEN`)
using the same account. Browser approval by itself does not provide credentials to the local process.

## Wan 2.1 VACE (outpainting)

Downloaded on first use when the Wan 2.1 VACE outpaint model is selected; none are gated.

- Transformer: `QuantStack/Wan2.1_14B_VACE-GGUF/Wan2.1_14B_VACE-Q4_K_M.gguf` → `models/diffusion_models/Wan2.1_14B_VACE-Q4_K_M.gguf`
- Text encoder: `Comfy-Org/Wan_2.1_ComfyUI_repackaged/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`
- VAE: `Comfy-Org/Wan_2.1_ComfyUI_repackaged/split_files/vae/wan_2.1_vae.safetensors`
- Step-distill LoRA: `Kijai/WanVideo_comfy/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors` → `models/loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors`

## SeedVR2

The `numz/ComfyUI-SeedVR2_VideoUpscaler` custom node downloads the selected model and shared VAE
into `models/SEEDVR2` on first use. SeedVR2 is not included in the installer's eager
`-DownloadModels` set because the selectable 3B/7B and precision variants are large and mutually
exclusive.

- 3B/7B FP16 and FP8 models plus `ema_vae_fp16.safetensors`: `numz/SeedVR2_comfyUI`
- 3B/7B Q4_K_M GGUF and mixed/sharp variants: `AInVFX/SeedVR2_comfyUI`

## ProPainter (AI DeScratch)

The installer adds `daniabib/ComfyUI_ProPainter_Nodes`. On the first AI DeScratch run that node
downloads `raft-things.pth`, `recurrent_flow_completion.pth`, and `ProPainter.pth` from the official
`sczhou/ProPainter` GitHub release into the node's `weights` folder.

ProPainter is covered by the NTU S-Lab License 1.0 and is limited to non-commercial use. The GUI
labels this restriction; do not enable AI DeScratch for commercial processing without obtaining
separate permission from the model authors.

## Qwen Image Edit

- Diffusion model: `unsloth/Qwen-Image-Edit-2511-GGUF/qwen-image-edit-2511-Q4_K_M.gguf`
- Text encoder: `Comfy-Org/Qwen-Image_ComfyUI/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`
- VAE: `Comfy-Org/Qwen-Image_ComfyUI/split_files/vae/qwen_image_vae.safetensors`
- Lightning LoRA: `lightx2v/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`

## Soundtrack (Create Audio Track)

Fetched on demand when the Create Audio Track stage first runs (and prefetched by
`install_windows.ps1 -DownloadModels`). These are downloaded in **soft mode** — a failure
logs guidance and continues instead of aborting the stage.

### Music — Stable Audio Open (ComfyUI core audio nodes)
- Checkpoint: `stabilityai/stable-audio-open-1.0/model.safetensors` → `models/checkpoints/stable_audio_open_1.0.safetensors`
- Text encoder: `google-t5/t5-base/model.safetensors` → `models/text_encoders/t5_base.safetensors`
- **Gated model.** Accept the licence at https://huggingface.co/stabilityai/stable-audio-open-1.0
  and authenticate (`hf auth login`, or set `HF_TOKEN`) before the download can succeed. If it
  is skipped, place the file manually at the destination above.

### Sound effects — MMAudio (`kijai/ComfyUI-MMAudio`)
Downloaded into `models/mmaudio/`:
- `Kijai/MMAudio_safetensors/mmaudio_large_44k_v2_fp16.safetensors`
- `Kijai/MMAudio_safetensors/mmaudio_vae_44k_fp16.safetensors`
- `Kijai/MMAudio_safetensors/mmaudio_synchformer_fp16.safetensors`
- `Kijai/MMAudio_safetensors/apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors`

(Confirm the exact filenames against the repo if a download 404s; update `SFX_MODELS` in
`scripts/dependency_manager.py` to match.)

These are large files. The downloader skips already-present destination files and keeps Hugging Face
cache files in the configured cache location (ARP's `.cache/huggingface` by default).

## Custom cache and model locations

The Hugging Face cache location is selected in this order:

1. `ARP_HF_CACHE_DIR`
2. `HF_HUB_CACHE`
3. the standard `hub` directory below `HF_HOME`
4. ARP's `.cache/huggingface` directory

When ComfyUI has an `extra_model_paths.yaml`, ARP reuses models found in its configured search
paths. New downloads use `download_model_base` (or the relevant category path) from the section
marked `is_default`. This keeps ARP and ComfyUI/ComfyUI-Manager pointed at the same centralized
model store.
