# Manifest Format

Manifests are CSV files with a required first metadata line:

```csv
# source_video=outpainted/example.mp4
```

Paths are relative to the repository `input/` folder unless absolute paths are used.

## End-Time Rows

Rows use end times only. Each row starts where the previous row ended.

```csv
enabled,end,reference
true,0:00:20,references/example/cut_0000.png
true,0:00:40,references/example/cut_0001.png
false,0:10:00,references/example/placeholder.png
```

This avoids off-by-one gaps between `start` and `end` columns. The runner converts times to frame windows using the source video FPS.

## Disabled Rows

Disabled rows advance the timeline but do not render. They are useful when you only want to test the first few minutes of a reel while preserving the intended overall range.

## Outpaint Manifests

Outpaint manifests use a `prompt` column instead of `reference`:

```csv
# source_video=source_4x3/example.mp4
enabled,end,prompt
true,0:00:20,"Expand to 16:9 and preserve the original center frame."
```

## Colourisation Manifests

Colourisation manifests use a `reference` column:

```csv
# source_video=outpainted/example_outpaint.mp4
enabled,end,reference
true,0:00:20,references/generated_scene_refs/example/cut_0000.png
```

Generated manifests are ready for `wrappers/run_colorize_manifest.bat`.
