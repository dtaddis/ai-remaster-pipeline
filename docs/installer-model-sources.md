# Installer Model Sources

`install_windows.ps1` downloads the default Windows/NVIDIA model set with the modern Hugging Face `hf download` command.

## LTX 2.3

- Base checkpoint: `Lightricks/LTX-2.3-fp8/ltx-2.3-22b-dev-fp8.safetensors`
- Text encoder: `Comfy-Org/ltx-2/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors`
- Audio VAE: `Kijai/LTX2.3_comfy/vae/LTX23_audio_vae_bf16.safetensors`
- Outpainting LoRA: `oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint/ltx-2.3-22b-ic-lora-outpaint.safetensors`

## Qwen Image Edit

- Diffusion model: `Comfy-Org/Qwen-Image-Edit_ComfyUI/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors`
- Text encoder: `Comfy-Org/Qwen-Image_ComfyUI/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`
- VAE: `Comfy-Org/Qwen-Image_ComfyUI/split_files/vae/qwen_image_vae.safetensors`
- Lightning LoRA: `lightx2v/Qwen-Image-Lightning/Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors`

These are large files. The installer skips already-present destination files and keeps partial Hugging Face downloads under `.cache/huggingface` while downloading.
