from __future__ import annotations

import io
import unittest

from ai_remaster_gui.console_log import ConsoleLog


class TtyBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


class ConsoleLogTests(unittest.TestCase):
    def test_messages_are_retained_and_printed_immediately(self) -> None:
        stream = TtyBuffer()
        log = ConsoleLog(stream=stream)

        log.append("Loading source material")

        self.assertEqual(log, ["Loading source material"])
        self.assertEqual(stream.getvalue(), "Loading source material\n")

    def test_percentage_updates_render_in_place(self) -> None:
        stream = TtyBuffer()
        log = ConsoleLog(stream=stream)

        log.append("Download progress: 10%")
        log.append("Download progress: 20%")
        log.append("Download complete")

        output = stream.getvalue()
        self.assertIn("\r[", output)
        self.assertIn(" 10% Download progress: 10%", output)
        self.assertIn(" 20% Download progress: 20%", output)
        self.assertTrue(output.endswith("Download complete\n"))
        self.assertEqual(len(log), 3)

    def test_chunk_updates_are_converted_to_progress(self) -> None:
        stream = TtyBuffer()
        log = ConsoleLog(stream=stream)

        log.append("Upscale chunk 2/4: frames 100-199")

        self.assertIn(" 50% Upscale chunk 2/4", stream.getvalue())

    def test_ffmpeg_banner_noise_is_retained_but_not_printed(self) -> None:
        stream = TtyBuffer()
        log = ConsoleLog(stream=stream)

        log.append("ffmpeg version 8.1 Copyright FFmpeg")
        log.append("Error opening input file broken.mov")

        self.assertEqual(len(log), 2)
        self.assertNotIn("ffmpeg version", stream.getvalue())
        self.assertIn("Error opening input file broken.mov", stream.getvalue())

    def test_multiline_tool_error_filters_banner_but_keeps_cause(self) -> None:
        stream = TtyBuffer()
        log = ConsoleLog(stream=stream)

        log.append("Preview failed\nffmpeg version 8.1\nconfiguration: very long\nError opening input")

        self.assertEqual(len(log), 1)
        self.assertEqual(stream.getvalue(), "Preview failed\nError opening input\n")


if __name__ == "__main__":
    unittest.main()
