# AI Movie Remaster Pipeline

Tools to use AI models to expand classic 4:3 films to widescreen and colourise them in resumable, reviewable stages.

The pipeline is built around a simple idea: keep every expensive AI stage separate, save the intermediate files, and use manifests as the source of truth. That makes long films practical to process overnight, patch in small sections, and assemble in an editor without losing your mind.

## Pipeline

```text
source 4:3 B&W
  -> widescreen / outpaint pass
  -> shot and fade detection
  -> reference-frame generation or manual references
  -> reference-based colourisation
  -> reassembled preview MP4s
  -> final edit / grade / upscale
```

The scripts are helpers around ComfyUI. They do not replace ComfyUI and they do not include model weights.

## Repository Layout

```text
input/
  source_4x3/        Put original or prepared 4:3 clips here.
  outpainted/        Put completed widescreen clips here, or copy outputs here.
  references/        Put manual and generated colour references here.

manifests/
  outpaint/          End-time manifests for widescreen conversion.
  colorize/          End-time manifests for colourisation.

output/
  outpainted/        Outpaint chunks and reassembled widescreen previews.
  colorized/         Colourised chunks and reassembled previews.
  reassembled/       Optional editorial/export staging.
  logs/              Your own logs/notes.

workflows/
  widescreen/        ComfyUI workflows for LTX outpainting.
  colorize/          ComfyUI workflows for DeepExemplar / ColorMNet.

scripts/             Python pipeline runners.
wrappers/            Windows .bat launchers.
tools/               Optional local ComfyUI / TransNetV2 checkouts.
```

## ComfyUI Setup

Recommended layout:

```text
ai-remaster-pipeline/
  tools/
    ComfyUI/
      main.py
      venv/
```

You can either clone ComfyUI there, or place a portable ComfyUI package under `tools/ComfyUI_windows_portable`. You can also run an existing ComfyUI elsewhere and pass `--comfy-url` to the runners.

Start ComfyUI:

```bat
wrappers\run_comfyui.bat
```

Then open ComfyUI at:

```text
http://127.0.0.1:8188
```

See [docs/comfyui-setup.md](docs/comfyui-setup.md) for model and custom-node notes.

## Stage 1: Widescreen / Outpaint

Create an outpaint manifest under `manifests/outpaint/`:

```csv
# source_video=source_4x3/example_00.00.00to00.10.00.mp4
enabled,end,prompt
true,0:00:20,"Expand to 16:9. Preserve the original center frame exactly."
true,0:00:40,"Continue the same restrained widescreen restoration."
false,0:10:00,"Disabled placeholder for later."
```

Dry-run it:

```bat
wrappers\run_outpaint_manifest.bat --manifest outpaint_manifest_example.csv --dry-run
```

Render it:

```bat
wrappers\run_outpaint_manifest.bat --manifest outpaint_manifest_example.csv
```

Outputs go to `output/outpainted/<manifest-stem>/`. Copy or move the final widescreen clips you want to colourise into `input/outpainted/`.

## Stage 2: Generate Colour References And Manifest

The main generator detects hard cuts, fades, and cross-fades. It writes audit source frames for every detected cut, can call the OpenAI Image API to colourise references, and writes a colourisation manifest.

Dry-run first:

```bat
wrappers\run_generate_scene_reference_manifest.bat ^
  --source-video outpainted/example_00.00.00to00.10.00_outpaint.mp4 ^
  --dry-run
```

Generate references and manifest:

```bat
wrappers\run_generate_scene_reference_manifest.bat ^
  --source-video outpainted/example_00.00.00to00.10.00_outpaint.mp4
```

By default the manifest is written to `manifests/colorize/colorize_manifest_<clip-id>_shots_auto.csv` and generated references go under `input/references/generated_scene_refs/`.

Set your API key before using API generation:

```bat
set OPENAI_API_KEY=sk-...
```

Use `--extract-only` if you want to colourise the selected source frames manually instead.

## Stage 3: Colourise

Dry-run:

```bat
wrappers\run_colorize_manifest.bat --manifest colorize_manifest_example.csv --dry-run
```

Render:

```bat
wrappers\run_colorize_manifest.bat --manifest colorize_manifest_example.csv
```

The default method is `Both`, which renders DeepExemplar and ColorMNet into separate output folders:

```text
output/colorized/DeepExemplar/<manifest-stem>/
output/colorized/ColorMNet/<manifest-stem>/
```

Render one method only:

```bat
wrappers\run_colorize_manifest.bat --manifest colorize_manifest_example.csv --method DeepExemplar
wrappers\run_colorize_manifest.bat --manifest colorize_manifest_example.csv --method ColorMNet
```

Each run keeps individual chunks and also creates a frame-validated reassembled MP4 for quick editorial placement.

## Resume And Repair

The runners keep JSONL ledgers beside their outputs. Rerunning the same command skips valid completed chunks and regenerates missing, failed, stale, or too-short chunks.

Useful controls:

```bat
--dry-run                 Show planned jobs without queueing ComfyUI.
--keep-going              Continue after a failed chunk.
--sleep-when-done         Keep Windows awake while running, then sleep afterwards.
--no-reassemble           Keep chunks only.
```

Pause behavior:

- Press `Ctrl+C` once to stop after the current chunk.
- Press `Ctrl+C` twice to exit immediately.
- Create a `PAUSE` file in the output folder to stop queueing after the current chunk.

## Licensing And Media

The code in this repository is licensed under Apache-2.0. Third-party tools, ComfyUI workflows, custom nodes, model weights, and generated/remastered media have their own licenses. See [docs/licensing-notes.md](docs/licensing-notes.md).

Do not commit film footage, generated movie frames, model weights, or API keys.
