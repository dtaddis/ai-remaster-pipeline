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
