# Final Composite Notes

The composite script is meant to reproduce the Resolve stack in a command-line form for batch assembly:

- Outpainted widescreen plate at the bottom.
- Original source centered over it, scaled to the same height and feathered at the left/right edges.
- Optional colorized layer blended over the result.

Example:

```bat
final_composite.bat ^
  --outpainted intermediate\outpainted\clip_outpaint.mp4 ^
  --source input\clip_source.mp4 ^
  --colorized intermediate\outpainted_colorized\clip_colorized.mp4 ^
  --output output\reassembled\clip_composited.mp4
```

Useful parameters:

- `--feather-pixels 50` to `100`: softer or harder transition from the real source to generated sides.
- `--saturation`: reduce colorization intensity before blending.
- `--temperature`: negative cools, positive warms.
- `--encoder prores`: bigger intermediate, friendlier for editors.

The blend is an FFmpeg approximation. Resolve remains better for shot-specific mask tweaks, Color blend mode, grain, and finishing grade.
