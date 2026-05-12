# Editorial Workflow Notes

The runners keep two kinds of output:

- Individual chunk files, useful for repairs and manual crossfades.
- Reassembled preview MP4s, useful for dropping a whole section onto a Resolve timeline.

For final finishing, keep the individual chunks. The reassembled file is a convenience layer, not a prison.

Suggested Resolve workflow:

1. Place the original restoration master on a lower track.
2. Place the outpainted or colourised reassembled preview above it.
3. Use individual chunks for repairs where fades or reference changes need hand blending.
4. Apply final grain, grade, upscale, and audio sync at the end.

When a section fails or looks wrong, delete only that chunk from `output/` and rerun the same manifest. The runner validates existing chunk frame counts and regenerates stale outputs.
