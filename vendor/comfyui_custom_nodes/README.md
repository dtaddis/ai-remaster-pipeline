This directory contains the ComfyUI custom-node packages ARP needs at runtime,
excluding ComfyUI itself and model weights.

The Windows installer copies these folders into `ComfyUI/custom_nodes` before it
falls back to cloning from GitHub. That keeps ARP usable when a matching node
pack is not already installed in the user's ComfyUI checkout.

Bundled node packs:

- `ComfyUI-GGUF`
  - Upstream: https://github.com/city96/ComfyUI-GGUF
  - Source revision: `6ea2651e7df66d7585f6ffee804b20e92fb38b8a`
  - License: Apache-2.0
  - Used for: `UnetLoaderGGUF`
- `ComfyUI-VideoHelperSuite`
  - Upstream: https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
  - Source revision: local non-Git install; refresh from upstream before release
  - License: MIT
  - Used for: `VHS_LoadVideo`, `VHS_VideoCombine`
- `reference-video-colorization`
  - Upstream: https://github.com/jonstreeter/ComfyUI-Reference-Based-Video-Colorization
  - Source revision: `644680e9dfad09bca3358119f48eb12a5ad920cd`
  - License: MIT for the node pack; bundled third-party subcomponents retain
    their own licenses.
  - Used for: `DeepExColorVideoNode`, `ColorMNetVideo`
  - Excludes: large checkpoint/model files and sample assets.
- `ComfyUI-LTXVideo`
  - Upstream: https://github.com/Lightricks/ComfyUI-LTXVideo
  - Source revision: `229437c6b65796d6a7a63ae34be2bd5ba31fa543`, with the
    ComfyUI rope compatibility fix from `548a39373bc2fdb880a233944e4621efb1a8950e`
    backported.
  - License: LTX-2 Community License Agreement.
  - Used for: `LTXVImgToVideoConditionOnly`, `LTXAddVideoICLoRAGuideAdvanced`,
    `LTXVInpaintPreprocess`, `LTXVLaplacianPyramidBlend`, and related LTX helpers.
  - ARP patch: singleton masks are broadcast per blend chunk instead of expanded
    into a full video-sized mask tensor.
- `ComfyUI-MMAudio`
  - Upstream: https://github.com/kijai/ComfyUI-MMAudio
  - Source revision: `8eaeb72edc3aaf2059b57f2d96a1f6f689f19ae2`
  - License: MIT
  - Used for: `MMAudioModelLoader`, `MMAudioFeatureUtilsLoader`,
    `MMAudioSampler`.
  - Excludes: large checkpoint/model files.

Before publishing a release, re-check upstream licenses and replace any
non-redistributable package with installer-only download instructions.
