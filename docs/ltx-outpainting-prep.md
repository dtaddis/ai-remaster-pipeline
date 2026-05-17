# LTX Outpainting Prep And Restore

The LTX 2.3 outpainting IC-LoRA only treats pure black pixels as the region to fill. That creates a nasty edge case: real black pixels inside the original movie can be interpreted as outpaint mask.

The pipeline handles this with a reversible-ish lift:

1. Lift the source movie image away from pure black.
2. Place that lifted source in the centre of a 16:9 canvas.
3. Leave only the new left/right margins as exact black.
4. Run LTX outpainting in ComfyUI.
5. Apply the inverse lift to the outpainted result.

It is technically a black-floor lift plus optional gamma lift. Pure gamma alone does not protect exact black because `0` stays `0`.

## Prepare The Comfy Input

```bat
prepare_outpaint_input.bat ^
  --source input\movie_4x3.mp4 ^
  --output intermediate\outpaint_prepared\movie_16x9_lifted.mp4 ^
  --target-aspect 16:9 ^
  --black-lift 0.018 ^
  --gamma 1.06
```

The default `--black-lift 0.018` is about 5 code values in 8-bit video. It is enough to protect black clothing, shadows, titles, and fades from being seen as mask, while keeping the side margins exactly black.

## Run LTX Outpainting

Use the prepared clip as the ComfyUI input. The IC-LoRA should see only the side margins as pure black and fill those regions.

## Restore The Render

```bat
finalize_outpaint_output.bat ^
  --source path\to\comfy_outpaint_render.mp4 ^
  --output intermediate\outpainted\movie_16x9_outpainted.mp4 ^
  --black-lift 0.018 ^
  --gamma 1.06
```

Use the same `--black-lift` and `--gamma` values as the prep step. The restore is global, so it also slightly darkens the generated side material. In practice this helps the generated edges sit closer to the original centre image.

## Alternatives

A more elaborate workflow could use an explicit mask instead of pure-black detection. If your ComfyUI graph supports an explicit mask input for the outpaint LoRA, that is cleaner. This prep/restore method exists because it works with simple IC-LoRA workflows that infer the mask from black pixels.
