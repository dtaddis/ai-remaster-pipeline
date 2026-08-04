This directory is for ComfyUI LTX 2.3 IC outpainting workflows.

`outpaint_LTX-IC.json` is the bundled workflow used by `scripts/outpaint_video.py`. It is Lightricks' official `LTX-2.3_ICLoRA_Outpaint_Two_Stage_Distilled` workflow for the v0.9 in/outpainting IC-LoRA.

At runtime ARP patches this workflow to use the LTX 2.3 distilled GGUF Q4_K_M model through ComfyUI-GGUF, a separate LTX 2.3 video VAE, and the official IC-LoRA. ARP supplies a binary video mask derived from its prepared canvas. The workflow performs a coarse first pass, a boundary-refinement second pass, and Laplacian blending of the protected source region.

ARP preserves each chunk's time-aligned source audio for the official AV conditioning path, normalizing codecs such as WebM/Opus to AAC for reliable ComfyUI decoding. If a source is genuinely silent, ARP supplies a frozen empty LTX audio latent instead.

Suggested pattern:
- input video from `input/`
- output target-aspect clip to `intermediate/outpainted/`
- keep model paths configurable in ComfyUI rather than committing local absolute paths
