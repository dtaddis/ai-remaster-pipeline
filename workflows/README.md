ARP bundles the workflow JSON files it needs to queue ComfyUI jobs. A fresh
checkout should not require matching workflows to already exist in the user's
ComfyUI blueprints folder.

Bundled workflows:

- `outpaint_ltx/outpaint_LTX-IC.json` - LTX IC outpainting.
- `qwen_image_edit/Image Edit (Qwen 2509).json` - Qwen Image Edit reference
  frame colourisation and outpaint guide-frame generation.

Video colourisation through Deep Exemplar or ColorMNet is built as a ComfyUI API
prompt in `scripts/colorize_video.py`, so there is no separate JSON workflow to
bundle for that stage.
