# Create Audio Track (music + sound effects)

This phase generates a soundtrack for a silent film and muxes it onto the latest render
**without re-encoding the video** (`-c:v copy`). It runs after Recomposition and before
Upscaling; when Upscaling is enabled, the selected backend carries the new audio track through
via the shared audio-mux step.

Unlike the other phases, the ComfyUI graphs are built in code (see
`scripts/audio_models.py`) rather than loaded from JSON, so there are no `.json` files to
install here. The script discovers each node's inputs via ComfyUI's `/object_info`, so model
filenames are auto-selected from whatever you have installed.

## Models — downloaded automatically

Like the other phases, the models are fetched on demand the first time **Create Audio Track**
runs (`scripts/dependency_manager.py` → `ensure_audio_models`), and can be prefetched with
`install_windows.ps1 -DownloadModels`. See `docs/installer-model-sources.md` for the exact
repos/filenames. Audio models download in **soft mode**: a failure logs guidance and continues
rather than aborting the stage.

### Music — Stable Audio Open (ComfyUI core)
No custom nodes needed; these ship with ComfyUI:
`CheckpointLoaderSimple`, `CLIPTextEncode`, `EmptyLatentAudio`, `KSampler`,
`VAEDecodeAudio`, `SaveAudio`.

Auto-downloaded to `ComfyUI/models/checkpoints/stable_audio_open_1.0.safetensors`.
The separate T5-base text encoder is auto-downloaded to
`ComfyUI/models/text_encoders/t5_base.safetensors`.
**This is a gated Hugging Face model** — accept the licence at
https://huggingface.co/stabilityai/stable-audio-open-1.0 and run `hf auth login` (or set
`HF_TOKEN`) so the download can authenticate; otherwise place the file there manually.
Override the filename on the Audio tab ("Stable Audio checkpoint") if yours differs.

### Sound effects — MMAudio (video → synchronized audio)
Installed by `install_windows.bat`:
- `ComfyUI-MMAudio` → https://github.com/kijai/ComfyUI-MMAudio
  (provides `MMAudioModelLoader`, `MMAudioFeatureUtilsLoader`, `MMAudioSampler`)

Plus `ComfyUI-VideoHelperSuite` (already used by outpaint/upscale) for `VHS_LoadVideo`.

The MMAudio weights (model + VAE + Synchformer + CLIP) auto-download into
`ComfyUI/models/mmaudio/`. MMAudio analyses a low-resolution proxy — the short side is capped
at **384px** by default, since higher resolution only increases processing time without
improving the audio. The proxy is retimed to **25 fps** (Synchformer's native rate); MMAudio's
sampler slices frames by count, so other rates would time-warp and truncate the audio.

The default SFX window is **8 seconds** — MMAudio's training length. Longer windows do not
improve quality: the model generalizes poorly past ~10s and the effects dissolve into vague
ambience, so resist the urge to raise it to "cover more action". VRAM is rarely the limit at
these settings.

### Captioning (automatic when available) — local Ollama or a ComfyUI node
Music cues are mood-matched and SFX windows are sound-matched by captioning a representative
frame per scene/window. The caption backend is resolved in this order:

1. **A ComfyUI caption node**, if its class name is set in the Audio tab field
   **"Qwen-VL caption node (advanced)"** (a `ShowText|pysssss` node, if installed, is used to
   capture the caption text from the graph).
2. **A local Ollama server** (`http://127.0.0.1:11434`) with any vision-capable model pulled
   (e.g. `qwen2.5vl`, `llava`, `moondream`) — picked automatically; override with
   `--ollama-vision-model <name>` or disable with `--ollama-vision-model off`.

If neither is available, captioning is skipped and the **Music style hint** / **Sound effects
hint** (or sensible defaults) are used for every cue/window. Captions matter most for SFX:
MMAudio takes its semantics largely from the text prompt, so per-window captions
("machinery clanking, steam hiss…") track the picture far better than one generic prompt.

SFX windows are aligned to detected scene cuts (and long scenes tiled to the chunk length),
so a window never straddles two unrelated shots.

## How the stems are assembled
- **Music**: scenes are detected from colour-histogram jumps (min/max cue length derived from
  "Music cue seconds"); one Stable Audio cue per scene, trimmed/padded to the exact scene
  length with short fades, then concatenated → a full-length music stem.
- **SFX**: the video is split into "SFX chunk seconds" windows; each is downscaled to a proxy
  and fed to MMAudio; the per-window audio is concatenated → a full-length effects stem.
- **Mix**: music is ducked under the effects via `sidechaincompress`, with per-stem gain set
  by "Music level (dB)" / "Sound effects level (dB)", then muxed onto the video as AAC.
