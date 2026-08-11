from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import final_composite  # noqa: E402
from outpaint_geometry import source_placement  # noqa: E402


class OutpaintGeometryTests(unittest.TestCase):
    def test_crop_regions_are_removed_before_the_source_is_fitted(self) -> None:
        placement = source_placement(1440, 1080, 1280, 720, (20, 20, 10, 10))

        self.assertEqual((placement.x, placement.y, placement.width, placement.height), (164, 0, 952, 720))

    def test_one_sided_crop_recenters_the_post_crop_frame(self) -> None:
        placement = source_placement(1440, 1080, 1280, 720, (20, 0, 10, 0))

        self.assertEqual((placement.x, placement.y), (162, 0))
        self.assertEqual((placement.x + placement.width, placement.y + placement.height), (1118, 720))

    def test_model_safe_canvas_preserves_crop_first_delivery_placement(self) -> None:
        placement = source_placement(1440, 1080, 1280, 704, (20, 20, 10, 10), 1280, 720)

        self.assertEqual((placement.x, placement.y, placement.width, placement.height), (164, 0, 952, 704))

    def test_trimmed_source_produces_only_a_pillarbox(self) -> None:
        placement = source_placement(1456, 1080, 1280, 704, (10, 10, 6, 6), 1280, 720)

        self.assertEqual((placement.x, placement.y, placement.width, placement.height), (156, 0, 968, 704))

    def test_final_composite_does_not_expand_cropped_source_over_regenerated_pixels(self) -> None:
        args = final_composite.build_parser().parse_args(
            [
                "--outpainted", "outpainted.mp4",
                "--source", "source.mp4",
                "--output", "final.mp4",
                "--crop-left", "20",
                "--crop-right", "20",
                "--crop-top", "10",
                "--crop-bottom", "10",
                "--output-width", "1280",
                "--output-height", "720",
            ]
        )

        filter_text = final_composite.build_filter(
            args,
            has_color=False,
            fps=24.0,
            source_size=(1440, 1080),
            base_size=(1280, 720),
        )

        self.assertIn("crop=w=iw-20-20:h=ih-10-10:x=20:y=10,scale=952:720", filter_text)
        self.assertIn("[base][srcm]overlay=x=164:y=0[merged]", filter_text)
        self.assertIn("lt(X,80)", filter_text)
        self.assertNotIn("lt(Y,80)", filter_text)
        self.assertNotIn("gt(Y,H-80)", filter_text)
        self.assertNotIn("scale2ref", filter_text)

    def test_final_composite_feathers_only_letterbox_edges(self) -> None:
        args = final_composite.build_parser().parse_args(
            [
                "--outpainted", "outpainted.mp4",
                "--source", "source.mp4",
                "--output", "final.mp4",
                "--output-width", "1280",
                "--output-height", "720",
            ]
        )

        filter_text = final_composite.build_filter(
            args,
            has_color=False,
            fps=24.0,
            source_size=(1920, 800),
            base_size=(1280, 720),
        )

        self.assertIn("scale=1280:534", filter_text)
        self.assertIn("overlay=x=0:y=93", filter_text)
        self.assertIn("lt(Y,80)", filter_text)
        self.assertNotIn("lt(X,80)", filter_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
