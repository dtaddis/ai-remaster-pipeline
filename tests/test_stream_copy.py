from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import stream_copy  # noqa: E402
from common import find_ffmpeg  # noqa: E402
from intermediate_video import codec_args  # noqa: E402


def make_clip(ffmpeg: str, path: Path, frames: int, size: str = "64x48", colour: str = "red") -> Path:
    subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c={colour}:s={size}:r=24", "-frames:v", str(frames),
         *codec_args("high", fast=True), "-f", "matroska", str(path)],
        check=True,
    )
    return path


class StreamCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ffmpeg = find_ffmpeg()

    def test_same_format_clips_join_without_an_encode(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = make_clip(self.ffmpeg, root / "a.mkv", 10, colour="red")
            second = make_clip(self.ffmpeg, root / "b.mkv", 7, colour="blue")
            joined = root / "joined.mkv"
            self.assertTrue(stream_copy.copy_join(self.ffmpeg, [first, second], joined, 17))
            self.assertEqual(stream_copy.count_video_packets(self.ffmpeg, joined), 17)

    def test_clips_of_different_sizes_are_left_to_the_re_encode(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = make_clip(self.ffmpeg, root / "a.mkv", 5)
            second = make_clip(self.ffmpeg, root / "b.mkv", 5, size="96x48")
            joined = root / "joined.mkv"
            self.assertFalse(stream_copy.copy_join(self.ffmpeg, [first, second], joined, 10))
            self.assertFalse(joined.exists())

    def test_a_wrong_frame_count_is_refused_and_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = make_clip(self.ffmpeg, root / "a.mkv", 6)
            joined = root / "joined.mkv"
            self.assertFalse(stream_copy.copy_join(self.ffmpeg, [first], joined, 9))
            self.assertFalse(joined.exists())

    def test_head_of_a_clip_is_copied_to_the_exact_length(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clip = make_clip(self.ffmpeg, root / "a.mkv", 17)
            head = root / "head.mkv"
            self.assertTrue(stream_copy.copy_head(self.ffmpeg, clip, head, 12))
            self.assertEqual(stream_copy.count_video_packets(self.ffmpeg, head), 12)


if __name__ == "__main__":
    unittest.main()
