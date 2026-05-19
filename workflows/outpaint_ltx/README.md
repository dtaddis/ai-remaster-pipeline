This directory is for ComfyUI LTX 2.3 IC outpainting workflows.

`outpaint_LTX-IC.json` is the bundled workflow used by `scripts/outpaint_video.py`. It is based on the workflow linked from the `oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint` Hugging Face model card.

Suggested pattern:
- input video from `input/`
- output target-aspect clip to `intermediate/outpainted/`
- keep model paths configurable in ComfyUI rather than committing local absolute paths
