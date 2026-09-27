# LTX Outpainting Preparation

ARP's current LTX 2.3 outpainting workflow supplies the official IC-LoRA with an explicit, frame-aligned mask. Real black pixels inside the original movie are therefore protected by geometry rather than inferred from pixel brightness.

Source crop values deliberately create additional outpaint regions on those same edges. The remaining source is cropped first and then fitted as one complete image into the requested canvas.

The pipeline now:

1. Crops and fits the unmodified source into the requested canvas.
2. Builds an exact geometric generation mask for the added margins and requested crop borders.
3. Runs the official LTX IC-LoRA masked workflow.
4. Uses the untouched prepared source for the protected centre of the final Laplacian blend.
5. Applies only a modest detail recovery to generated regions during finalization.

There is no black-floor lift or gamma-restore pass. Those belonged to the older brightness-inferred masking path and could darken the generated margins differently from the centre.

## Run The Full Outpainting Stage

The normal ARP entry point is:

```bat
outpaint_video.bat ^
  --source input\movie_4x3.mp4 ^
  --target-aspect 16:9
```

This prepares the input, queues the bundled ComfyUI workflow from `workflows/outpaint_ltx/outpaint_LTX-IC.json`, copies the raw ComfyUI render, and finalizes it into `intermediate/outpainted`.

To use the optional LTX 2.5 two-stage workflow at its recommended cadence:

```bat
outpaint_video.bat ^
  --source input\movie_4x3.mp4 ^
  --target-aspect 16:9 ^
  --ltx-version 2.5 ^
  --generation-fps 24
```

ARP uses the Q4_K_M distilled transformer on 24 GB GPUs and motion-interpolates lower-rate input without changing duration. LTX 2.5 retains the official 2.3 in/outpainting IC-LoRA and adds its official latent two-stage upscaler.

To use Wan 2.1 VACE instead of LTX:

```bat
outpaint_video.bat ^
  --source input\movie_4x3.mp4 ^
  --target-aspect 16:9 ^
  --outpaint-backend wan-vace
```

Wan builds its ComfyUI graph in `scripts/wan_vace_outpaint.py` (following ComfyUI's own Wan VACE outpainting template) rather than from a workflow file. Each chunk is rendered in one VACE pass, so Wan chunks are capped at 161 frames, the measured 24 GB budget at 1280x704, and shrink automatically at higher resolutions (never below Wan's native 81). Chunks overlap by at least `--wan-context-frames` (13) so each continues from that many finished frames of the previous chunk. A chunk containing a shot change is split into one pass per shot.

`--wan-sampling context` is experimental: it lets chunks run to 20 seconds at 720p and samples each in one pass using ComfyUI's Wan context windows, blending overlapping 161-frame windows at every step. It fits in 24 GB of VRAM but needs roughly 24 GB of system RAM for frames at 720p, and in testing the blended windows sometimes morphed invented surroundings between different objects, so sequential chunks remain the default. The GUI plans sequential chunks, so run it from the command line. `--wan-steps`, `--wan-cfg`, `--wan-shift`, `--wan-sampler`, and `--wan-scheduler` tune sampling; the defaults suit the lightx2v step-distill LoRA.

MiniMax H3 (`--outpaint-backend h3`) builds its graph in `scripts/minimax_h3_outpaint.py`, following ComfyUI's `video_minimax_h3_fun_controlnet_union` template: the FL2VA transformer patched with `MiniMaxH3FunControlNetApply` (mask = regenerate, source video behind it), `MiniMaxH3ImageToVideo` for the text conditioning and AV latent, `BasicGuider` at guidance 1 and `res_multistep` sampling. It reuses Wan's masked-pass chunk renderer with H3's 17k + 5 frame grid, 22 context frames and a pass cap from a measured 24 GB budget (`H3_MAX_PASS_PIXELS`, 124 frames at 1536x640, less the 22 context frames that each continuing pass anchors with `MiniMaxH3AddGuide`), never beyond 362 frames. `--h3-license-confirmed` is required before any download or run. `--h3-pdd` swaps the 20-step schedule for PDD 8-step: ModelSamplingMiniMaxH3 at 12/3, then MiniMaxH3PDDAccApply (ComfyUI-MiniMax-H3-PDD-Acc) supplies the trained sigmas to an euler sampler; `--h3-steps` sets the full-step count.

## Prepare The Comfy Input Manually

```bat
prepare_outpaint_input.bat ^
  --source input\movie_4x3.mp4 ^
  --output intermediate\outpaint_prepared\movie_16x9.mp4 ^
  --target-aspect 16:9
```

## Run LTX Outpainting

Use the prepared clip and ARP's generated mask as the ComfyUI inputs. The IC-LoRA fills only the requested mask regions.

## Finalize The Render

```bat
finalize_outpaint_output.bat ^
  --source path\to\comfy_outpaint_render.mp4 ^
  --output intermediate\outpainted\movie_16x9_outpainted.mp4
```

Finalization preserves the render's tone. When the protected source rectangle is known, a restrained edge-only sharpening pass reduces the model's softness without sharpening or altering the original centre.
