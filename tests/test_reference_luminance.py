from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import final_composite  # noqa: E402
import reference_luminance  # noqa: E402
from common import video_info  # noqa: E402


def gradient_image(path: Path, low: int, high: int) -> None:
    image = Image.new("RGB", (96, 64), "black")
    pixels = image.load()
    for y in range(8, 56):
        for x in range(12, 84):
            value = round(low + (high - low) * ((x - 12) / 71))
            pixels[x, y] = (value, value, value)
    image.save(path)


class ReferenceLuminanceTests(unittest.TestCase):
    def test_curve_is_bounded_monotonic_and_ignores_black_bars(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.png"
            color = root / "color.png"
            gradient_image(source, 30, 170)
            gradient_image(color, 65, 220)

            curve = reference_luminance.curve_for_pair(source, color, 70)

            self.assertEqual(curve[0], [0.0, 0.0])
            self.assertEqual(curve[-1], [1.0, 1.0])
            self.assertTrue(all(curve[index][0] < curve[index + 1][0] for index in range(len(curve) - 1)))
            self.assertTrue(all(curve[index][1] <= curve[index + 1][1] for index in range(len(curve) - 1)))
            midpoint = min(curve, key=lambda point: abs(point[0] - 0.5))
            self.assertGreater(midpoint[1], midpoint[0])
            self.assertLessEqual(max(abs(x - y) for x, y in curve), reference_luminance.MAX_LUMA_SHIFT / 255.0 + 0.01)

    def test_plan_uses_reviewed_frame_boundaries_including_final_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.png"
            color = root / "color.png"
            manifest = root / "shots.csv"
            gradient_image(source, 30, 170)
            gradient_image(color, 65, 220)
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["start_frame", "end_frame", "source_reference", "color_reference"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"start_frame": "0", "end_frame": "8", "source_reference": str(source), "color_reference": str(color)},
                        {"start_frame": "8", "end_frame": "16", "source_reference": str(source), "color_reference": str(color)},
                    ]
                )

            plan = reference_luminance.reference_luminance_plan(manifest, fps=4, strength_percent=70)

            self.assertEqual([(item["start_frame"], item["end_frame"]) for item in plan], [(0, 8), (8, 16)])
            self.assertEqual([(item["start_seconds"], item["end_seconds"]) for item in plan], [(0.0, 2.0), (2.0, 4.0)])
            self.assertTrue(all(item["matched"] for item in plan))

    def test_ffmpeg_compositor_accepts_multi_shot_reference_curve_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ffmpeg = final_composite.find_ffmpeg(None)
            source_video = root / "source.mp4"
            color_video = root / "color.mp4"
            output = root / "output.mp4"
            source_ref = root / "source.png"
            color_ref = root / "color.png"
            manifest = root / "shots.csv"
            gradient_image(source_ref, 30, 170)
            gradient_image(color_ref, 65, 220)
            subprocess.run(
                [ffmpeg, "-y", "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=4:duration=2", "-vf", "hue=s=0", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source_video)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [ffmpeg, "-y", "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=4:duration=2", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(color_video)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["start_frame", "end_frame", "source_reference", "color_reference"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"start_frame": "0", "end_frame": "4", "source_reference": str(source_ref), "color_reference": str(color_ref)},
                        {"start_frame": "4", "end_frame": "8", "source_reference": str(source_ref), "color_reference": str(color_ref)},
                    ]
                )
            args = final_composite.build_parser().parse_args(
                [
                    "--source", str(source_video),
                    "--colorized", str(color_video),
                    "--manifest", str(manifest),
                    "--reference-luminance-match",
                    "--reference-luminance-strength", "70",
                    "--output", str(output),
                    "--ffmpeg", ffmpeg,
                ]
            )

            self.assertEqual(final_composite.run(args), 0)
            self.assertTrue(output.is_file())
            self.assertEqual(int(video_info(output)["frames"]), 8)


if __name__ == "__main__":
    unittest.main()
