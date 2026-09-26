from __future__ import annotations

import argparse
import sys
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import final_composite  # noqa: E402
from outpaint_geometry import native_source_layout, source_envelope_size, source_placement  # noqa: E402


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

    def test_equal_vertical_extensions_create_generation_room_on_all_four_sides(self) -> None:
        # Backend geometry keeps its legacy sign convention: negative means extend.
        placement = source_placement(1440, 1080, 1920, 1080, (0, 0, -180, -180))

        self.assertEqual(source_envelope_size(1440, 1080, (0, 0, -180, -180)), (1440, 1440))
        self.assertEqual((placement.x, placement.y, placement.width, placement.height), (420, 135, 1080, 810))

    def test_one_sided_extension_positions_source_asymmetrically(self) -> None:
        placement = source_placement(1440, 1080, 1920, 1080, (-240, 0, 0, 0))

        self.assertEqual((placement.x, placement.y, placement.width, placement.height), (360, 0, 1440, 1080))

    def test_extension_placement_maps_to_model_safe_canvas(self) -> None:
        placement = source_placement(1440, 1080, 1920, 1088, (0, 0, -180, -180), 1920, 1080)

        self.assertEqual((placement.x, placement.y, placement.width, placement.height), (420, 136, 1080, 816))

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

    def test_final_composite_preserves_extended_source_placement(self) -> None:
        args = final_composite.build_parser().parse_args(
            [
                "--outpainted", "outpainted.mp4",
                "--source", "source.mp4",
                "--output", "final.mp4",
                "--crop-top", "-180",
                "--crop-bottom", "-180",
                "--output-width", "1920",
                "--output-height", "1080",
            ]
        )

        filter_text = final_composite.build_filter(
            args,
            has_color=False,
            fps=24.0,
            source_size=(1440, 1080),
            base_size=(1920, 1080),
        )

        self.assertIn("scale=1080:810", filter_text)
        self.assertIn("overlay=x=420:y=135", filter_text)
        self.assertNotIn("crop=w=", filter_text)

    def test_native_layout_scales_the_canvas_around_an_unscaled_source(self) -> None:
        # 1440x1080 outpainted to 1280x720: the source sat at 960x720, so the canvas
        # grows by 1.5x and the source goes back on at its own size.
        canvas, placement = native_source_layout(1440, 1080, 1280, 720, (0, 0, 0, 0))

        self.assertEqual(canvas, (1920, 1080))
        self.assertEqual((placement.x, placement.y, placement.width, placement.height), (240, 0, 1440, 1080))

    def test_native_layout_compensates_for_trims_and_extends(self) -> None:
        trimmed_canvas, trimmed = native_source_layout(1440, 1080, 1280, 720, (20, 20, 10, 10))
        extended_canvas, extended = native_source_layout(1440, 1080, 1920, 1080, (0, 0, -180, -180))

        # The trimmed source keeps its native trimmed size and the canvas follows it.
        self.assertEqual((trimmed.width, trimmed.height), (1400, 1060))
        self.assertEqual(trimmed_canvas, (1882, 1060))
        self.assertEqual((trimmed.x % 2, trimmed.y % 2), (0, 0))
        # Virtual border added around the source scales up with the rest of the canvas.
        self.assertEqual(extended_canvas, (2560, 1440))
        self.assertEqual((extended.x, extended.y, extended.width, extended.height), (560, 180, 1440, 1080))

    def test_final_composite_native_resolution_overlays_the_source_without_scaling_it(self) -> None:
        args = final_composite.build_parser().parse_args(
            [
                "--outpainted", "outpainted.mp4",
                "--source", "source.mp4",
                "--output", "final.mp4",
                "--crop-left", "20",
                "--crop-right", "20",
                "--output-width", "1280",
                "--output-height", "720",
                "--native-source-resolution",
            ]
        )

        filter_text = final_composite.build_filter(
            args,
            has_color=False,
            fps=24.0,
            source_size=(1440, 1080),
            base_size=(1280, 720),
        )

        src_chain = next(part for part in filter_text.split(";") if part.endswith("[src]"))
        self.assertIn("crop=w=1400:h=1080:x=20:y=0", src_chain)
        self.assertNotIn("scale=", src_chain)
        self.assertIn("scale=1920:1080:flags=lanczos[base]", filter_text)
        self.assertIn("overlay=x=258:y=0", filter_text)
        # 80px of feather at 720p becomes 120px at 1080p so the blend looks the same.
        self.assertIn("lt(X,120)", filter_text)

    def test_native_resolution_flag_stays_out_of_the_signature_when_off(self) -> None:
        # Composites made before the option existed must keep matching their signature.
        base = ["--outpainted", "o.mp4", "--source", "s.mp4", "--output", "f.mp4"]
        with mock.patch.object(final_composite, "file_fingerprint", return_value={}):
            off = final_composite.signature(final_composite.build_parser().parse_args(base))
            on = final_composite.signature(final_composite.build_parser().parse_args(base + ["--native-source-resolution"]))

        self.assertNotIn("native_source_resolution", off)
        self.assertTrue(on["native_source_resolution"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
