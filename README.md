<p align="center">
  <img src="assets/branding/arp-logo.png" alt="ARP - AI Remaster Pipeline" width="520">
</p>

# ARP - AI Remaster Pipeline

ARP is a local GUI app for remastering public-domain or properly licensed film material with ComfyUI-powered AI tools.

It is built around an old-film workflow: choose source material, optionally clean archive damage, outpaint it to a wider aspect ratio, detect shots, generate color reference stills, colorize the video from those references, and finally recomposite the result so the original center footage stays as faithful as possible.

The app is still alpha software, but the goal is simple: you should be able to run the whole remaster from the GUI, then inspect and adjust each stage when the AI needs a little human steering.

<p align="center">
  <img src="assets/screenshots/walkthrough/arp-walkthrough-overview.jpg" alt="ARP Overview tab showing source video metadata, preview frames, and whole-pipeline progress">
</p>

## What It Does

- Outpaints 4:3 or similar archive footage into common target aspect ratios such as `16:9`, `9:16`, `4:3`, `3:4`, `1:1`, `21:9`, `2.39:1`, and `1.85:1`.
- Splits video into shots and lets you review, merge, enable, disable, and adjust shot boundaries.
- Generates per-shot reference frames and colorizes them with Qwen Image Edit.
- Optionally reconstructs vertical scratches with masked ProPainter inpainting, corrects light/dark vignettes, then optionally restores archive footage with the LTX 2.3 Dearchive LoRA while preserving source geometry.
- Uses reference-guided video colorization for the outpainted footage.
- Recombines the original, outpainted, and colorized layers into a final output.
- Keeps intermediate files deterministic and resumable, so reruns can reuse valid existing work.

## Install

### Requirements

- Windows with an NVIDIA GPU is the currently supported installer path. The installer defaults to CUDA 12.8 PyTorch wheels.
- Python 3.13 is required. On Windows, install it from [python.org](https://www.python.org/downloads/) with the Python Launcher option enabled, or make sure `python.exe` is on `PATH`.
- Git is required so the installer can clone and update ARP's managed ComfyUI runtime and required custom nodes.
- Internet access is required during installation for Python packages, ComfyUI, FFmpeg, and optional model downloads.
- ComfyUI is required as the AI backend. The installer can clone it for you, or you can point ARP at an existing ComfyUI checkout.

### Windows

Run:

```bat
install_windows.bat
```

The installer creates this repo's `.venv`, installs FFmpeg locally, and uses an ARP-managed ComfyUI checkout at `tools\comfyui`.

That managed checkout is the recommended path. Re-running `install_windows.bat` refreshes ComfyUI and the required custom nodes with fast-forward Git updates, while preserving downloaded models and generated outputs.

If the installer cannot find Python 3.13, it will prompt you to install it and retry detection.

If Python 3.13 is installed somewhere custom, point the installer at the actual executable. Common python.org install paths are:

```bat
install_windows.bat -PythonLauncher "%LocalAppData%\Programs\Python\Python313\python.exe"
install_windows.bat -PythonLauncher "C:\Program Files\Python313\python.exe"
```

If you intentionally want to use an existing ComfyUI checkout instead:

```bat
install_windows.bat -ComfyDir D:\path\to\ComfyUI
```

When you pass `-ComfyDir`, ARP treats that checkout as external: it can install/update required custom nodes there, but it will not update ComfyUI core itself. Keep external ComfyUI checkouts current yourself with `git pull` and `pip install -r requirements.txt`.

Python packages are installed into ARP's `.venv`, not into a separate ComfyUI virtual environment. The managed ComfyUI checkout is used as the AI backend.

Models and LoRAs are downloaded on demand when a stage first needs them. If a large Hugging Face download is interrupted, rerun the same stage and the download should resume. You can prefetch the main model set with:

```bat
install_windows.bat -DownloadModels
```

Useful installer options:

```bat
install_windows.bat -NonInteractive
install_windows.bat -SkipDeepExemplar
install_windows.bat -InstallCorrelationExtension
install_windows.bat -TorchIndexUrl https://download.pytorch.org/whl/cu128
```

ColorMNet uses a PyTorch fallback by default with the same output quality. `-InstallCorrelationExtension` attempts the optional faster CUDA correlation extension; it requires Visual Studio C++ Build Tools and a local CUDA Toolkit matching the installed PyTorch CUDA build. If the extension cannot build, installation continues in fallback mode.

See [docs/installer-model-sources.md](docs/installer-model-sources.md) for the exact model and LoRA sources.

### macOS And Linux

The full installer is currently Windows-focused. On macOS or Linux, set up a Python virtual environment, install `requirements.txt`, configure an existing ComfyUI directory in `.ai_remaster_config.json`, and launch the GUI with `./launch_gui.sh` or `python -m ai_remaster_gui`.

Cross-platform packaging is planned, but not polished yet.

## Launch The GUI

On Windows:

```bat
launch_gui.bat
```

On macOS or Linux:

```sh
./launch_gui.sh
```

Or from an activated environment:

```sh
python -m ai_remaster_gui
```

The GUI opens as a local web app. It checks ComfyUI at `http://127.0.0.1:8188`; if ComfyUI is configured but not running, ARP can start it in its own console window so you can watch ComfyUI load and render in real time.

ComfyUI runs in that separate console window, where its startup, progress, and any import/custom-node errors are visible live. ARP also records the launch command to `output/logs/comfyui-startup.log` (viewable from Settings → Log file).

Set `AI_REMASTER_NO_COMFY_AUTOSTART=1` if you want to manage ComfyUI yourself.

## Basic Workflow

1. Open the Overview tab.
2. Choose your source material with Browse.
3. Check the preview frames, duration, resolution, frame rate, codecs, and color/monochrome guess.
4. Enable Clean Up when the source needs archive restoration; it defaults off. Inside the phase, select any combination of AI DeScratch, DeVignette, and Dearchive. Leave Colorize enabled for black-and-white sources, or turn it off if you only want cleanup/outpainting.
5. Click Run Whole Remaster for a first pass.
6. Use the stage tabs to inspect or rerun individual phases.

Every stage writes predictable intermediate files under `intermediate/`, manifests under `manifests/`, and final renders under `output/reassembled/`.

## Tabs

### Overview

Choose the source video, see useful media info, and run the whole pipeline.

![Overview tab](assets/screenshots/walkthrough/arp-walkthrough-overview.jpg)

### Clean Up

Optional and off by default. Clean Up has three independent operations. **AI DeScratch** builds a visible scratch-only mask, reconstructs only those pixels across neighbouring frames with ProPainter, and composites them back over the untouched source; it is also off by default. DeVignette estimates stationary dark falloff or a pale additive edge veil while preserving black presentation bars. Dearchive applies the LTX 2.3 IC-LoRA and defaults on.

The order is DeVignette, AI DeScratch, then Dearchive. AI DeScratch defaults to a 720p working copy and 41-frame windows on a 24 GB GPU, but always returns the source resolution and timing. Sensitivity controls how faint a vertical mark may be before it enters the mask; mask expansion includes damaged edges around each detected line. Enable the mask preview to write a companion `_scratch_mask.mp4`; white shows the detected base mask, before the selected expansion margin. ProPainter's model and upstream code use the NTU S-Lab License 1.0 and are restricted to non-commercial use.

DeVignette defaults to **Auto (prefer GPU)**. ARP uses its installed PyTorch CUDA stack directly and processes frames in adaptive batches when an NVIDIA GPU is available; otherwise it logs the reason and falls back to OpenCV/NumPy on the CPU. The log reports the selected processor, GPU name, batch size, sampled-frame analysis time, frames per second, elapsed time, and ETA.

When Dearchive is enabled it defaults to 4.04-second chunks (97 frames at 24 fps); the UI accepts 2 to 20 seconds and rounds the requested duration at the source frame rate to the nearest LTX-valid `8n + 1` frame count. **Source Fidelity** controls the strength of the complete input-video IC-LoRA guide. Its safe default of `1.0` preserves the source most exactly, while lower values let Dearchive repaint more damage at increasing risk of changes to faces, hands, motion, and fine period detail. Every combination returns a video with the source resolution, sample aspect ratio, frame rate, frame count, and audio. The passes preserve colour; aspect-ratio expansion and delivery scaling happen only in later phases.

### Outpainting

Set the target aspect ratio, output height, chunk length, overlap frames, and source crop. The target preview helps you see where ARP will add new canvas before LTX fills it.

Outpainting is chunked so longer movies can be processed without requiring a huge single ComfyUI job. ARP defaults to 8 overlap frames because LTX can return short chunks; lower values may still work, but the app warns you when the overlap is risky.

Outpainting is the slowest stage. On local GPUs, a 20 second 720p-ish LTX chunk can still take several minutes, and 10 minutes is not automatically a sign that something is broken. Very short chunk lengths multiply the number of ComfyUI jobs, so use the default 20 seconds unless you need a cut at a precise point.

If outpainting fails immediately with missing `LTXVImgToVideoConditionOnly` or `LTXAddVideoICLoRAGuide` nodes, fully close ComfyUI, re-run `install_windows.bat`, choose the same ComfyUI directory, then restart ARP/ComfyUI. These nodes come from [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo), which should live in `ComfyUI\custom_nodes\ComfyUI-LTXVideo`. ARP also uses [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF), installed to `ComfyUI\custom_nodes\ComfyUI-GGUF`, for the lightweight GGUF models.

If you use ComfyUI portable, select either the inner folder that contains `main.py` or the portable parent folder; ARP will look for `ComfyUI\main.py` inside it.

### Shot Detection

Review the detected shots, inspect start/middle/end frames, merge shots that should share a reference, and nudge shot boundaries frame by frame.

![Shot Detection tab](assets/screenshots/walkthrough/arp-walkthrough-shot-detection.jpg)

### Reference Generation

Pick the reference time inside each shot, regenerate individual Qwen color references, delete references you do not like, and add short per-shot prompt notes.

![Reference Generation tab](assets/screenshots/walkthrough/arp-walkthrough-reference-generation.jpg)

### Colorization

Review each shot's color reference alongside the corresponding colorized video segment. This stage uses the generated references to guide video colorization.

Because Dearchive can introduce colour of its own, ARP converts extracted source stills and the video stream back to neutral grayscale before reference generation and before Deep Exemplar/ColorMNet. The generated colour reference stays in colour and guides the selected model as usual.

![Colorization tab](assets/screenshots/walkthrough/arp-walkthrough-colorization.jpg)

### Recomposition

Preview and tune the final blend: outpainted video at the bottom, original source in the center with feathered edges, and the colorized layer contributing chroma on top.

![Recomposition tab](assets/screenshots/walkthrough/arp-walkthrough-recomposition.jpg)

### Output

Once recomposition finishes, the Output tab plays the final render.

![Output tab](assets/screenshots/walkthrough/arp-walkthrough-output.jpg)

### Settings

Settings contains ComfyUI connection details, queue/log inspection, and global pipeline defaults that are useful but too noisy for the main stage tabs.

## ComfyUI And Models

ARP uses ComfyUI as the backend for the AI-heavy stages. The current intended stack includes:

- LTX 2.3 distilled GGUF Q4_K_M for lightweight outpainting.
- CUDA-accelerated automatic light/dark DeVignette correction for optional archive repair.
- Masked ProPainter video inpainting for optional AI DeScratch (non-commercial NTU S-Lab licence).
- LTX 2.3 Dearchive IC-LoRA for optional generative archive cleanup.
- LTX 2.3 IC outpainting LoRA.
- Qwen Image Edit 2511 GGUF Q4_K_M for still reference colorization.
- Qwen Image Edit Lightning LoRA.
- Deep Exemplar reference-guided video colorization.

The repo stores orchestration code, GUI code, workflows, wrappers, docs, and small assets. Runtime media, model caches, ComfyUI installs, and generated outputs are ignored by Git.

ARP bundles the ComfyUI workflow JSONs it needs to queue jobs:

- `workflows/cleanup_ltx/DeArchive.json` for LTX Dearchive cleanup.
- `workflows/outpaint_ltx/outpaint_LTX-IC.json` for LTX IC outpainting.
- `workflows/qwen_image_edit/Image Edit (Qwen 2511).json` for Qwen Image Edit reference frames and outpaint guide frames.

Deep Exemplar and ColorMNet video colourisation do not use saved workflow JSON files; ARP builds those ComfyUI API prompts directly from the selected source, manifest, and method. ComfyUI itself and model weights remain external dependencies. ARP bundles the required custom-node packages where their licenses allow redistribution; see `vendor/comfyui_custom_nodes/README.md`.

## Folder Layout

```text
input/                                   Optional source clips
intermediate/cleaned/                    Geometry-preserving Clean Up outputs
intermediate/outpaint_prepared/          Expanded/lifted clips prepared for LTX
intermediate/outpainted/                 Widescreen/outpainted clips
intermediate/outpainted_references/      Per-shot black-and-white reference stills
intermediate/outpainted_references_color/ Qwen colorized reference stills
intermediate/outpainted_colorized/       Reference-guided colorized video
manifests/references/                    Shot/reference manifests
output/reassembled/                      Final composited masters
workflows/                               Bundled ComfyUI workflows used by ARP
vendor/comfyui_custom_nodes/             Bundled ComfyUI custom nodes used by ARP
wrappers/                                Batch/shell entry points
assets/branding/                         Logo, icons, and GitHub artwork
assets/screenshots/                      README screenshots
```

## Resume Behavior

ARP writes `.sig.json` sidecars beside generated outputs. If inputs and settings still match, a rerun can reuse existing work. If the source, prompt, workflow, crop, aspect ratio, or other relevant setting changes, the dependent output is regenerated.

The GUI is also designed around deterministic intermediate paths. When one stage completes, the next stage's input fields are populated automatically.

## Direct Script Use

The GUI is the recommended way to use ARP, but the backend scripts are still normal command-line tools. If you want to wire ARP into your own pipeline, look in `wrappers/` for entry points such as `cleanup_video.bat`, `outpaint_video.bat`, `generate_references.bat`, `qwen_colorize_references.bat`, `colorize_video.bat`, and `final_composite.bat`.

Those scripts are what the GUI calls internally, and the GUI shows the equivalent command before running a stage.

## Licensing Notes

Check the licenses for every model, workflow, and source film you use. This repo is orchestration software; it does not grant rights to source films, model weights, LoRAs, ComfyUI custom nodes, Qwen models, Deep Exemplar, or other third-party components.
