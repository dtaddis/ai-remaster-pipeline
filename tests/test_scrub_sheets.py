from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai_remaster_gui import media, scrub_sheets  # noqa: E402

FFMPEG = media.local_tool("ffmpeg")


@unittest.skipUnless(FFMPEG, "local FFmpeg is not installed")
class ScrubSheetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        # 60 frames at 24 fps, each visibly different (testsrc draws a moving pattern and counter).
        self.video = self.folder / "clip.mkv"
        subprocess.run(
            [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x176:rate=24:duration=2.5",
             "-c:v", "libx264", "-g", "12", "-pix_fmt", "yuv420p", str(self.video)],
            check=True,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_sheets_tile_every_frame_in_order(self) -> None:
        with mock.patch.object(scrub_sheets, "SHEETS_ROOT", self.folder / "sheets"):
            directory = scrub_sheets.sheet_dir(self.video)
            scrub_sheets._build(self.video, directory)
            status = scrub_sheets.sheet_status(self.video)

        self.assertTrue(status["ready"])
        self.assertEqual(status["sheet_count"], 3)  # 60 frames / 25 per sheet
        self.assertEqual(status["tile_width"], scrub_sheets.TILE_WIDTH)
        self.assertEqual(sorted(p.name for p in directory.glob("sheet_*.jpg")), ["sheet_00000.jpg", "sheet_00001.jpg", "sheet_00002.jpg"])
        self.assertEqual(json.loads((directory / scrub_sheets.INFO_NAME).read_text())["frames_per_sheet"], 25)

    def test_rebuilding_for_a_changed_video_prunes_the_old_sheets(self) -> None:
        with mock.patch.object(scrub_sheets, "SHEETS_ROOT", self.folder / "sheets"):
            first = scrub_sheets.sheet_dir(self.video)
            scrub_sheets._build(self.video, first)
            self.video.write_bytes(self.video.read_bytes() + b"\0")
            second = scrub_sheets.sheet_dir(self.video)
            scrub_sheets._build(self.video, second)

        self.assertNotEqual(first, second)
        self.assertFalse(first.exists())
        self.assertTrue(second.exists())

    def test_seeking_frame_extraction_matches_decoding_from_the_start(self) -> None:
        for frame in (0, 1, 13, 25, 59):
            fast = media.extract_video_frame_at_frame(self.video, self.folder / "fast", f"f{frame}", frame)
            exact = self.folder / f"exact_{frame}.jpg"
            subprocess.run(
                [FFMPEG, "-v", "error", "-y", "-i", str(self.video),
                 "-vf", f"trim=start_frame={frame}:end_frame={frame + 1},setpts=PTS-STARTPTS",
                 "-frames:v", "1", "-q:v", "4", str(exact)],
                check=True,
            )
            self.assertEqual(media.resolve(fast).read_bytes(), exact.read_bytes(), f"frame {frame}")


if __name__ == "__main__":
    unittest.main()
