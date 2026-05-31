This folder contains ARP's bundled Qwen Image Edit workflow.

`Image Edit (Qwen 2509).json` is the default workflow for colour reference
generation and outpaint guide-frame generation. ARP should be usable without
depending on a matching workflow already being present in the user's ComfyUI
blueprints folder.

The wrapper can patch arbitrary node IDs for:
- load image
- prompt text
- save image prefix

User-provided ComfyUI blueprints are still supported as overrides through the
References workflow setting.

See `docs/qwen-image-edit-workflow.md`.
