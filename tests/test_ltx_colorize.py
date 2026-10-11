from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import final_composite  # noqa: E402
import ltx_colorize  # noqa: E402
from common import find_ffmpeg  # noqa: E402


class ShotChunkTests(unittest.TestCase):
    def test_short_shot_is_one_chunk(self) -> None:
        self.assertEqual(ltx_colorize.shot_chunks(100, 180), [(100, 180)])

    def test_long_shot_chunks_cover_it_with_the_minimum_overlap(self) -> None:
        for length in (122, 200, 300, 576, 1000):
            chunks = ltx_colorize.shot_chunks(50, 50 + length, 121, 17)
            self.assertEqual(chunks[0][0], 50)
            self.assertEqual(chunks[-1][1], 50 + length)
            for first, last in chunks:
                self.assertEqual(last - first, 121)
            for (_, previous_last), (first, _) in zip(chunks, chunks[1:]):
                self.assertGreaterEqual(previous_last - first, 17, length)

    def test_processing_starts_at_the_reference_chunk_and_works_outwards(self) -> None:
        chunks = ltx_colorize.shot_chunks(0, 400, 121, 17)
        anchor = ltx_colorize.anchor_chunk(chunks, [250])
        self.assertTrue(chunks[anchor][0] <= 250 < chunks[anchor][1])
        order = ltx_colorize.processing_order(len(chunks), anchor)
        self.assertEqual(order[0], anchor)
        self.assertEqual(sorted(order), list(range(len(chunks))))
        # Every later chunk has a finished neighbour on the anchor's side when it runs.
        done = {anchor}
        for index in order[1:]:
            self.assertTrue(index - 1 in done or index + 1 in done)
            done.add(index)

    def test_guides_sit_on_the_latent_grid(self) -> None:
        self.assertEqual(ltx_colorize.guide_grid(0, 16), [0, 8, 16])
        self.assertEqual(ltx_colorize.guide_grid(95, 120), [96, 104, 112, 120])

    def test_dimensions_follow_the_source_aspect_on_the_32_pixel_grid(self) -> None:
        self.assertEqual(ltx_colorize.ltx_dimensions(1280, 704), (992, 544))
        self.assertEqual(ltx_colorize.ltx_dimensions(1920, 1080), (960, 544))

    def test_prompt_uses_the_trained_convention(self) -> None:
        prompt = ltx_colorize.build_prompt("a mouse steering a boat.", "red shorts and yellow shoes.")
        self.assertTrue(prompt.startswith("Reference shows a mouse steering a boat, rendered in grayscale."))
        self.assertIn("COLORIZE red shorts and yellow shoes.", prompt)
        self.assertIn("only color information differs", prompt)


class WarpedGuideTests(unittest.TestCase):
    def test_guide_takes_colour_where_the_warp_agrees_and_stays_grey_where_it_does_not(self) -> None:
        target = np.full((64, 96), 230, np.uint8)
        warped_gray = target.copy()
        warped_gray[:, 64:] = 40  # the warp put something dark where the target is white
        warped_colour = np.zeros((64, 96, 3), np.uint8)
        warped_colour[:] = (200, 20, 20)  # red, much darker than the near-white grey

        guide = ltx_colorize.colour_only_guide(warped_colour, warped_gray, target).astype(int)

        red, green, blue = guide[32, 16]
        self.assertGreater(red, green + 100)  # coloured and darkened where the warp is trusted
        self.assertLess(0.299 * red + 0.587 * green + 0.114 * blue, 120)
        self.assertLess(int(np.ptp(guide[32, 90])), 12)  # neutral where the warp is wrong
        self.assertGreater(int(guide[32, 90].mean()), 200)

    def test_match_score_ignores_black_side_bars(self) -> None:
        rng = np.random.default_rng(1)
        target = np.zeros((64, 96), np.uint8)
        target[:, 16:80] = rng.integers(30, 220, (64, 64))
        other = target.copy()
        other[:, 16:80] = rng.integers(30, 220, (64, 64))
        self.assertGreater(ltx_colorize.match_score(target, target), 0.99)
        self.assertLess(ltx_colorize.match_score(other, target), 0.5)


class ColourLuminanceCompositeTests(unittest.TestCase):
    def _composite(self, root: Path, luminance: float) -> np.ndarray:
        ffmpeg = find_ffmpeg()
        source, colour, output = root / "src.mp4", root / "col.mp4", root / f"out_{luminance:g}.mp4"
        base = "color=c=0xF0F0F0:s=160x120:d=0.25:r=24"
        if not source.exists():
            subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", base, "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", str(source)], check=True)
            subprocess.run(
                [ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", base + ",drawbox=x=40:y=30:w=80:h=60:c=0xCC1010:t=fill",
                 "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", str(colour)],
                check=True,
            )
        subprocess.run(
            [sys.executable, str(SCRIPTS / "final_composite.py"), "--source", str(source), "--colorized", str(colour),
             "--output", str(output), "--saturation", "100", "--color-luminance", f"{luminance:g}", "--ffmpeg", ffmpeg],
            check=True, capture_output=True,
        )
        raw = subprocess.run([ffmpeg, "-v", "error", "-i", str(output), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
        return np.frombuffer(raw, np.uint8).reshape(120, 160, 3)

    def test_colour_layer_brightness_turns_near_white_into_red(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            kept = self._composite(root, 0)
            moved = self._composite(root, 100)
        # Source luminance kept: the most a red can be at near-white brightness is pale pink.
        self.assertGreater(int(kept[60, 80, 1]), 150)
        # Colour-layer luminance: a real red, while the untouched white stays white.
        self.assertGreater(int(moved[60, 80, 0]), 180)
        self.assertLess(int(moved[60, 80, 1]), 70)
        self.assertGreater(int(moved[5, 5].min()), 225)

    def test_zero_share_leaves_the_filter_and_signature_unchanged(self) -> None:
        args = argparse.Namespace(color_luminance=0.0)
        self.assertEqual(final_composite.color_luminance_share(args), 0.0)
        self.assertEqual(final_composite.color_luminance_share(argparse.Namespace(color_luminance=3)), 0.03)


if __name__ == "__main__":
    unittest.main()
