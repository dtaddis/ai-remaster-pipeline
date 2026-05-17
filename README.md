# AI Movie Remaster Pipeline

A staged toolkit for remastering public-domain or licensed film material with ComfyUI-assisted outpainting, shot-reference extraction, optional Qwen Image Edit still colorization, optional reference-based video colorization, and final Resolve-friendly compositing.

The pipeline is deliberately modular. Each stage writes obvious intermediate files, and the scripts use sidecar signatures so reruns skip unchanged work and regenerate outputs when their source material, prompt, workflow, or settings change.

## Folder Layout

```text
input/                                  Original movie exports or clips to process
intermediate/outpaint_prepared/        Gamma/black-lifted 16:9 clips prepared for LTX outpainting
intermediate/outpainted/                 Restored widescreen/outpainted clips from ComfyUI LTX
intermediate/outpainted_references/      B&W source reference stills selected from cuts
intermediate/outpainted_references_color/ Qwen/Comfy colorized reference stills
intermediate/outpainted_colorized/       Deep Exemplar or ColorMNet colorized video chunks
manifests/references/                    Shot/reference manifests
manifests/colorize/                      Video colorization manifests if you keep them separate
output/reassembled/                      Final reassembled/composited masters
workflows/                               ComfyUI workflow templates/placeholders
wrappers/                                Batch-file entry points
```

Runtime media and generated manifests are git-ignored by default. Commit workflow templates, scripts, docs, and small examples, not a whole movie.

## 1. Put A Movie In `input`

Place your source movie or working segment in `input`. Keep filenames descriptive, for example:

```bat
input\Metropolis_00.00.00to00.10.00_source.mp4
```

## 2. Widescreen Conversion

The LTX 2.3 outpainting LoRA fills pure black pixels. To stop it from treating genuine black pixels in the original movie as mask, prepare the input first:

```bat
prepare_outpaint_input.bat ^
  --source input\Metropolis_00.00.00to00.10.00_source.mp4 ^
  --output intermediate\outpaint_prepared\Metropolis_00.00.00to00.10.00_16x9_lifted.mp4
```

This lifts the source image away from pure black, centres it in a 16:9 canvas, and leaves only the new left/right regions as exact black. Use that prepared clip in ComfyUI with the LTX 2.3 IC outpainting workflow.

After ComfyUI renders the outpainted clip, restore the lift:

```bat
finalize_outpaint_output.bat ^
  --source path\to\comfy_outpaint_render.mp4 ^
  --output intermediate\outpainted\Metropolis_00.00.00to00.10.00_outpaint.mp4
```

See `docs/ltx-outpainting-prep.md` for the black-lift/gamma details.

This repo does not lock you to a single ComfyUI install. A good arrangement is:

```text
ai-remaster-pipeline/
  tools/comfyui/      optional local ComfyUI checkout or portable install
```

The `tools/` folder is ignored, so users can bring their own ComfyUI without polluting the repo.

## 3. Detect Cuts And Extract Reference Stills

```bat
generate_references.bat --source-video intermediate\outpainted\Metropolis_00.00.00to00.10.00_outpaint.mp4
```

This writes:

```text
intermediate/outpainted_references/<clip-stem>/cut_XXXX_00.MM.SS.png
manifests/references/colorize_manifest_<clip-stem>_shots_auto.csv
```

The detector compares frame structure, histograms, edges, fades, and dissolve-like changes. It writes only meaningful shot references, prunes stale screenshots by default, and can reuse existing color references if the selected frame is effectively the same after a later rerun.

Useful options:

```bat
generate_references.bat --source-video intermediate\outpainted\clip.mp4 --dry-run
generate_references.bat --source-video intermediate\outpainted\clip.mp4 --limit 10
generate_references.bat --source-video intermediate\outpainted\clip.mp4 --keep-existing-source-frames
generate_references.bat --source-video intermediate\outpainted\clip.mp4 --no-reuse-existing-references
```

## 4. Optional Still Colorization With Qwen Image Edit

Create or export a ComfyUI workflow that loads one image, runs Qwen Image Edit, and saves one image. Then run:

```bat
qwen_colorize_references.bat ^
  --manifest manifests\references\colorize_manifest_Metropolis_00.00.00to00.10.00_outpaint_shots_auto.csv ^
  --workflow workflows\qwen_image_edit\Qwen Image Edit Reference Colorize.json ^
  --load-image-node-id 1 ^
  --prompt-node-id 2 ^
  --save-node-id 9 ^
  --prompt "Colorize this image." ^
  --prompt-suffix "Modern clean restoration, natural period color, preserve composition and text."
```

The exact node IDs depend on your workflow. For normal exported ComfyUI workflows, widget selectors are usually numeric indexes like `0`; for API-format workflows, they can be input names. If ComfyUI is not under `tools\comfyui`, add `--comfy-output-root D:\dtaddis\ComfyUI\output` or wherever your Comfy output folder lives. The script patches the input image, prompt, and save prefix, then copies ComfyUI's result to the `color_reference` path in the manifest.

## 5. Optional Video Colorization

The repository provides a lightweight wrapper for the reference-video colorization runner rather than hard-coding one person's ComfyUI graph:

```bat
colorize_video.bat --manifest manifests\references\colorize_manifest_clip_shots_auto.csv --method DeepExemplar
```

By default it looks for:

```text
tools/comfyui/scripts/colorize_manifest_runner.py
```

You can point it at your own known-good runner with `--comfy-runner`.

## 6. Final Composite / Reassembly

The finishing idea is:

1. Bottom: widescreen outpainted video.
2. Middle: original source video centered over it, feathered at the left/right edges to preserve as much real footage as possible.
3. Top: optional colorized video as a color/detail overlay with adjustable saturation and temperature.

```bat
final_composite.bat ^
  --outpainted intermediate\outpainted\clip_outpaint.mp4 ^
  --source input\clip_source.mp4 ^
  --colorized intermediate\outpainted_colorized\clip_colorized.mp4 ^
  --output output\reassembled\clip_final.mp4 ^
  --feather-pixels 80 ^
  --saturation 0.82 ^
  --temperature -0.015
```

The FFmpeg blend is an approximation of a Resolve-style color layer. For final grading, Resolve is still the more comfortable place to finesse saturation, coolness, grain, and masks.

## Resume Behavior

Scripts write `.sig.json` sidecars beside outputs. If inputs and settings match, a rerun reuses the existing output. If a source video, source frame, prompt, workflow, or relevant parameter changes, the dependent output is regenerated.

## Installation

For a full Windows setup, run:

```bat
install_windows.bat
```

That script creates `tools\comfyui`, sets up a CUDA PyTorch ComfyUI venv, installs ComfyUI Manager, LTXVideo nodes, Deep Exemplar / ColorMNet reference colorization nodes, creates this repo's `.venv`, and downloads the default model set:

- LTX 2.3 FP8 checkpoint.
- LTX 2.3 text encoder and audio VAE.
- LTX 2.3 outpainting IC-LoRA.
- Qwen Image Edit 2509 FP8 diffusion model.
- Qwen image text encoder and VAE.
- Qwen Image Edit Lightning 4-step LoRA.

Useful installer options:

```bat
install_windows.bat -SkipModelDownloads
install_windows.bat -SkipDeepExemplar
install_windows.bat -ComfyDir D:\somewhere\comfyui
install_windows.bat -TorchIndexUrl https://download.pytorch.org/whl/cu128
```

The model downloads are huge and resumable. If a file already exists at the expected destination, the installer skips it. See `docs/installer-model-sources.md` for the exact Hugging Face repo/file mapping.

You also need FFmpeg available on PATH, or pass `--ffmpeg` to `final_composite.py`.
## Licensing Notes

Check the licenses for every model and workflow you use. This repo is only orchestration code; it does not grant commercial rights to source films, LoRAs, Qwen models, Deep Exemplar, ColorMNet, or any other model weights.







