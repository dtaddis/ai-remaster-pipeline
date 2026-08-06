This directory is for ComfyUI LTX 2.3 IC outpainting workflows.

`outpaint_LTX-IC.json` is Lightricks' official `LTX-2.3_ICLoRA_Outpaint_Two_Stage_Distilled` workflow for the v0.9 in/outpainting IC-LoRA. `scripts/outpaint_video.py` uses it as a node template but routes execution through one full-resolution masked pass, bypassing the half-resolution draft, Lanczos enlargement, and second sampler.

At runtime ARP patches this workflow to use the LTX 2.3 distilled GGUF Q4_K_M model through ComfyUI-GGUF, a separate LTX 2.3 video VAE, and the official IC-LoRA. ARP supplies an expanded, latent-safe copy of its binary mask to the full-resolution inpaint preprocessor, while the final Laplacian blend receives the exact requested mask and untouched prepared source so the green conditioning sentinel and hidden overlap cannot leak into protected pixels.

ARP preserves each chunk's time-aligned source audio for the official AV conditioning path, normalizing codecs such as WebM/Opus to AAC for reliable ComfyUI decoding. If a source is genuinely silent, ARP supplies a frozen empty LTX audio latent instead.

Suggested pattern:
- input video from `input/`
- output target-aspect clip to `intermediate/outpainted/`
- keep model paths configurable in ComfyUI rather than committing local absolute paths
