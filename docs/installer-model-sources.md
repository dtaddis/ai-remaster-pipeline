# Installer Model Sources

Model downloads are handled on demand by the pipeline stage that needs them. `install_windows.ps1 -DownloadModels` can still prefetch the default Windows/NVIDIA model set.

## LTX 2.3

- Base checkpoint: `Lightricks/LTX-2.3-fp8/ltx-2.3-22b-dev-fp8.safetensors`
- Text encoder: `Comfy-Org/ltx-2/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors`
- Audio VAE: `Kijai/LTX2.3_comfy/vae/LTX23_audio_vae_bf16.safetensors`
- Distilled LoRA: `Lightricks/LTX-2.3/ltx-2.3-22b-distilled-lora-384.safetensors`
- Clean Up / Dearchive LoRA: `oumoumad/ltx-2.3-dearchive-lora/lora_weights_step_05000.safetensors` → `models/loras/ltx-2.3-dearchive-lora.safetensors`
- Outpainting LoRA: `oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint/ltx-2.3-22b-ic-lora-outpaint.safetensors`

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
