from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import intermediate_video as iv  # noqa: E402


class ComfyRenderProfileTests(unittest.TestCase):
    def test_lossless_profile_saves_ffv1_rgb(self) -> None:
        inputs = iv.comfy_video_combine_inputs("lossless")
        self.assertEqual(inputs["format"], "video/ffv1-mkv")
        self.assertEqual(inputs["pix_fmt"], "bgra")

    def test_lossy_profiles_save_10_bit_above_their_own_quality(self) -> None:
        crfs = [iv.comfy_video_combine_inputs(profile)["crf"] for profile in ("low", "medium", "high")]
        self.assertEqual(crfs, sorted(crfs, reverse=True))
        self.assertEqual(iv.comfy_video_combine_inputs("high")["pix_fmt"], "yuv420p10le")

    def test_existing_vhs_encoder_settings_are_replaced(self) -> None:
        nvenc = {
            "images": ["4851", 0], "frame_rate": 24.0, "filename_prefix": "remaster",
            "format": "video/nvenc_h264-mp4", "pix_fmt": "yuv420p", "bitrate": 10, "megabit": True,
        }
        inputs = iv.vhs_inputs_for_profile(nvenc, "lossless")
        self.assertEqual(inputs["format"], "video/ffv1-mkv")
        self.assertNotIn("bitrate", inputs)
        self.assertNotIn("megabit", inputs)
        self.assertEqual(inputs["images"], ["4851", 0])
        self.assertEqual(inputs["filename_prefix"], "remaster")

    def test_cached_render_falls_back_between_mkv_and_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mp4 = Path(tmp) / "raw_0000.mp4"
            mkv = mp4.with_suffix(".mkv")
            self.assertEqual(iv.existing_comfy_render(mkv), mkv)
            mp4.write_bytes(b"x")
            self.assertEqual(iv.existing_comfy_render(mkv), mp4)
            mkv.write_bytes(b"x")
            self.assertEqual(iv.existing_comfy_render(mkv), mkv)
            mp4.unlink()
            self.assertEqual(iv.existing_comfy_render(mp4), mkv)


if __name__ == "__main__":
    unittest.main()
