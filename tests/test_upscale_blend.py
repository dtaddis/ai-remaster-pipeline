from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from upscale_video import blend_weight_expression, normalized_blend_strength, read_upscale_shots  # noqa: E402


class UpscaleBlendTests(unittest.TestCase):
    def test_strength_values_are_percentages(self) -> None:
        self.assertEqual(normalized_blend_strength(0), 0.0)
        self.assertEqual(normalized_blend_strength(1), 0.01)
        self.assertEqual(normalized_blend_strength(100), 1.0)
        self.assertEqual(normalized_blend_strength(250), 1.0)

    def test_manifest_strength_and_fade_build_a_centered_tween(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "shots.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["start_frame", "end_frame", "upscale_strength", "fade_to_next", "crossfade_seconds"],
                )
                writer.writeheader()
                writer.writerow({"start_frame": 0, "end_frame": 100, "upscale_strength": 25, "fade_to_next": "true", "crossfade_seconds": 2})
                writer.writerow({"start_frame": 100, "end_frame": 200, "upscale_strength": 75})
            shots = read_upscale_shots(path, 200, 20.0, 1.0)

        self.assertEqual([shot["strength"] for shot in shots], [0.25, 0.75])
        expression = blend_weight_expression(shots, 20.0, 1.0)
        self.assertIn("lt(N,80)", expression)
        self.assertIn("lt(N,120)", expression)
        self.assertIn("(N-80)/40", expression)

    def test_no_fade_changes_strength_at_the_cut(self) -> None:
        shots = [
            {"start": 0, "end": 12, "strength": 0.2, "fade": False, "crossfade_seconds": ""},
            {"start": 12, "end": 24, "strength": 0.8, "fade": False, "crossfade_seconds": ""},
        ]
        self.assertEqual(blend_weight_expression(shots, 24.0, 1.0), "if(lt(N,12),0.20000000,0.80000000)")


if __name__ == "__main__":
    unittest.main()
