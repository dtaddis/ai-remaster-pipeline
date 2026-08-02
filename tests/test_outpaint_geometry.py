from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import final_composite  # noqa: E402
from outpaint_geometry import source_placement  # noqa: E402


class OutpaintGeometryTests(unittest.TestCase):
    def test_crop_regions_keep_full_source_scale_and_position(self) -> None:
        placement = source_placement(1440, 1080, 1280, 720, (20, 20, 10, 10))

        self.assertEqual((placement.full_x, placement.full_y, placement.full_width, placement.full_height), (160, 0, 960, 720))
        self.assertEqual((placement.x, placement.y, placement.width, placement.height), (173, 7, 934, 706))

    def test_one_sided_crop_stays_on_the_cropped_side(self) -> None:
        placement = source_placement(1440, 1080, 1280, 720, (20, 0, 10, 0))

        self.assertEqual((placement.x, placement.y), (173, 7))
        self.assertEqual((placement.x + placement.width, placement.y + placement.height), (1119, 719))

    def test_model_safe_canvas_preserves_delivery_crop_placement(self) -> None:
        placement = source_placement(1440, 1080, 1280, 704, (20, 20, 10, 10), 1280, 720)

        self.assertEqual((placement.full_x, placement.full_y, placement.full_width, placement.full_height), (160, 0, 960, 704))
        self.assertEqual((placement.x, placement.y, placement.width, placement.height), (173, 7, 934, 690))

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

        self.assertIn("crop=w=iw-20-20:h=ih-10-10:x=20:y=10,scale=934:706", filter_text)
        self.assertIn("[base][srcm]overlay=x=173:y=7[merged]", filter_text)
        self.assertIn("lt(Y,80)", filter_text)
        self.assertIn("gt(Y,H-80)", filter_text)
        self.assertNotIn("scale2ref", filter_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
