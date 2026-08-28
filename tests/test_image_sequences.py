from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_remaster_gui import image_sequences
from ai_remaster_gui.media import local_tool


class ImageSequenceTests(unittest.TestCase):
    def test_natural_sort_orders_numbered_frames(self) -> None:
        with tempfile.TemporaryDirectory(dir=image_sequences.ROOT) as tmp_text:
            folder = Path(tmp_text)
            paths = [folder / name for name in ("frame10.png", "frame2.png", "frame1.png")]
            for path in paths:
                path.write_bytes(b"image")

            ordered = image_sequences.normalize_image_paths(paths)

        self.assertEqual([path.name for path in ordered], ["frame1.png", "frame2.png", "frame10.png"])

    def test_sequence_state_round_trips_selected_files(self) -> None:
        with tempfile.TemporaryDirectory(dir=image_sequences.ROOT) as tmp_text:
            folder = Path(tmp_text)
            first = folder / "scan0001.dpx"
            second = folder / "scan0002.dpx"
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            settings = {
                "global": {
                    "source_images": image_sequences.serialize_image_paths([first, second]),
                    "source_sequence_fps": "23.976",
                    "source_sequence_preview": "intermediate/source_sequences/preview.mp4",
                }
            }

            state = image_sequences.sequence_state(settings)

        self.assertEqual(state["count"], 2)
        self.assertEqual(state["available"], 2)
        self.assertEqual(state["fps"], "23.976")
        self.assertEqual(state["formats"], ["DPX"])

    def test_required_image_formats_are_supported(self) -> None:
        self.assertTrue({".png", ".bmp", ".jpg", ".jpeg", ".dpx"}.issubset(image_sequences.IMAGE_EXTS))

    def test_ffmpeg_imports_dpx_sequence(self) -> None:
        ffmpeg = local_tool("ffmpeg")
        if not ffmpeg:
            self.skipTest("FFmpeg is not installed")
        try:
            from PIL import Image
        except ModuleNotFoundError:
            self.skipTest("Pillow is not installed")

        with tempfile.TemporaryDirectory(dir=image_sequences.ROOT) as tmp_text:
            folder = Path(tmp_text)
            output_dir = folder / "outputs"
            png1 = folder / "source1.png"
            png2 = folder / "source2.png"
            dpx1 = folder / "frame0010.dpx"
            dpx2 = folder / "frame0002.dpx"
            Image.new("RGB", (64, 48), (200, 20, 10)).save(png1)
            Image.new("RGB", (64, 48), (10, 200, 20)).save(png2)
            subprocess.run([ffmpeg, "-y", "-i", str(png1), "-frames:v", "1", str(dpx1)], check=True, capture_output=True)
            subprocess.run([ffmpeg, "-y", "-i", str(png2), "-frames:v", "1", str(dpx2)], check=True, capture_output=True)

            progress: list[tuple[int, str]] = []
            with mock.patch.object(image_sequences, "SEQUENCE_DIR", output_dir):
                result = image_sequences.prepare_image_sequence(
                    [dpx1, dpx2],
                    "24",
                    ffmpeg,
                    progress=lambda percent, label: progress.append((percent, label)),
                )

            master = image_sequences.resolve(result["source"])
            preview = image_sequences.resolve(result["playback"])
            self.assertTrue(master.is_file())
            self.assertTrue(preview.is_file())
            self.assertEqual(result["count"], 2)
            self.assertEqual([Path(path).name for path in result["images"]], ["frame0002.dpx", "frame0010.dpx"])
            self.assertEqual(progress[0][0], 0)
            self.assertEqual(progress[-1][0], 100)
            self.assertTrue(any("2/2 frames" in label for _percent, label in progress))
            sidecar = json.loads(next(output_dir.glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(len(sidecar["images"]), 2)
            ffprobe = str(Path(ffmpeg).with_name("ffprobe.exe" if Path(ffmpeg).suffix.lower() == ".exe" else "ffprobe"))
            probe = subprocess.run(
                [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames,r_frame_rate", "-of", "json", str(master)],
                check=True,
                capture_output=True,
                text=True,
            )
            stream = json.loads(probe.stdout)["streams"][0]
            self.assertEqual(stream["nb_read_frames"], "2")
            self.assertEqual(stream["r_frame_rate"], "24/1")


if __name__ == "__main__":
    unittest.main()
