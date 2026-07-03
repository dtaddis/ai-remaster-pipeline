# Final Composite Notes

The final composite script is meant to reproduce the Resolve stack in a command-line form for batch assembly:

- Outpainted widescreen plate at the bottom.
- Original source centered over it, scaled to the same height and feathered at the left/right edges.
- Optional colorized layer blended over the result.

Example:

```bat
final_composite.bat ^
  --outpainted intermediate\outpainted\clip_outpaint.mp4 ^
  --source input\clip_source.mp4 ^
  --colorized intermediate\outpainted_colorized\clip_colorized.mp4 ^
  --output output\reassembled\clip_final.mp4
```

Useful parameters:

- `--feather-pixels 50` to `100`: softer or harder transition from the real source to generated sides.
- `--saturation`: colorization intensity as a percentage before blending.
- `--temperature`: color temperature in Kelvin. 6500 is neutral; lower warms, higher cools.
- `--source-black-transparent`: let the outpainted plate show through near-black pixels in the original source layer. The GUI enables this automatically when Outpainting used "Outpaint all black regions".
- `--source-black-threshold`: RGB threshold for that transparency mask. The default is `24`.
- `--source-black-matte-shrink-pixels`: removes a small rim around detected black source regions so compressed or resampled black edges do not reappear over the outpainted plate. The default is `2`.
- `--encoder prores`: bigger intermediate, friendlier for editors.

The blend is an FFmpeg approximation. Resolve remains better for shot-specific mask tweaks, Color blend mode, grain, and final grade.
