This folder contains ARP's bundled Qwen Image Edit workflow.

`Image Edit (Qwen 2511).json` is the default workflow for colour reference
generation and outpaint guide-frame generation. ARP should be usable without
depending on a matching workflow already being present in the user's ComfyUI
blueprints folder.

`Image Edit Inpaint (Qwen 2511).json` is the default masked-edit workflow used
by the advanced reference/guide editor when a SAM2/brush/wand mask is present.
It uses the Qwen Image Edit 2511 four-step Lightning LoRA plus ComfyUI's latent
`SetLatentNoiseMask` path. The generation mask is expanded by 32 pixels
so small brush selections remain large enough to affect Qwen's latent image.
The same area is blurred in Qwen's conditioning image, preventing the model
from simply copying the selected object back into the generated pixels.
The decoded result is finally composited over the source with an 8-pixel mask
safety margin, preserving the rest of the image while avoiding a fringe of the
original object around the edit.

The wrapper can patch arbitrary node IDs for:
- load image
- mask image
- prompt text
- save image prefix

User-provided ComfyUI blueprints are still supported as overrides through the
References workflow setting.

See `docs/qwen-image-edit-workflow.md`.
