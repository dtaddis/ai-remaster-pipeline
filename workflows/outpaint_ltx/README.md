This directory is for ComfyUI LTX 2.3 IC outpainting workflows.

`outpaint_LTX-IC.json` is the bundled workflow used by `scripts/outpaint_video.py`. It is based on the workflow linked from the `oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint` Hugging Face model card.

At runtime ARP patches this workflow to use the LTX 2.3 distilled GGUF Q4_K_M model through ComfyUI-GGUF, a separate LTX 2.3 video VAE, and the IC outpainting LoRA. The original checkpoint nodes remain in the workflow file so it still opens sensibly in ComfyUI, but the generated API prompt defaults to the lightweight GGUF path.

Suggested pattern:
- input video from `input/`
- output target-aspect clip to `intermediate/outpainted/`
- keep model paths configurable in ComfyUI rather than committing local absolute paths
