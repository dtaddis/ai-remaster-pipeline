# ComfyUI Setup

This project expects ComfyUI to be running locally, usually at:

```text
http://127.0.0.1:8188
```

## Recommended Folder Layout

```text
ai-remaster-pipeline/
  tools/
    ComfyUI/
      main.py
      venv/
```

Clone ComfyUI into `tools/ComfyUI`, create its virtual environment, and install your ComfyUI requirements there. The wrapper scripts automatically prefer `tools/ComfyUI/venv/Scripts/python.exe` when it exists.

A portable install can also be placed at:

```text
tools/ComfyUI_windows_portable/
```

## Custom Nodes

The included workflows are helper workflows, not bundled custom nodes. Install the nodes required by the workflows you use.

Typical colourisation workflow requirements:

- ComfyUI-VideoHelperSuite
- ComfyUI-Reference-Based-Video-Colorization
- its DeepExemplar / ColorMNet dependencies

Typical LTX outpainting workflow requirements:

- ComfyUI-LTXVideo
- ComfyUI-VideoHelperSuite
- comfyui-kjnodes, if using the included public outpainting workflow

ComfyUI-Manager is the easiest way to install missing nodes, but it is not required by these scripts.

## Models

This repository does not include model weights. Place model files wherever your ComfyUI install expects them, normally under `tools/ComfyUI/models/`.

Useful model families for this pipeline:

- LTX 2.3 video models and compatible LoRAs for widescreen/outpainting.
- DeepExemplar / ColorMNet weights for reference-based colourisation.
- Optional TransNetV2 weights if you want to experiment with the TransNet generator.

## Running Against Another ComfyUI

If ComfyUI is already running somewhere else, pass:

```bat
--comfy-url http://127.0.0.1:8188
```

The scripts send absolute file paths to ComfyUI where possible, so ComfyUI must be running on the same machine and able to read this repository's `input/` files.
