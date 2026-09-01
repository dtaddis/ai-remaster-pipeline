from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import stabilize_video  # noqa: E402
from common import find_ffmpeg  # noqa: E402


class StabilizeVideoTests(unittest.TestCase):
    def test_scene_ranges_are_preserved_in_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            source = Path(tmp_text) / "source.mp4"
            source.write_bytes(b"fixture")
            args = stabilize_video.build_parser().parse_args(["--source", str(source)])
            signature = stabilize_video.output_signature(source, args, [(0, 12), (12, 24)])

        self.assertEqual(signature["shots"], [[0, 12], [12, 24]])
        self.assertEqual(signature["identity"]["kind"], "stabilize")

    def test_smoothing_is_capped_for_short_shots(self) -> None:
        args = stabilize_video.build_parser().parse_args(["--source", "source.mp4", "--smoothing", "30"])
        commands = []
        with tempfile.TemporaryDirectory() as tmp_text, mock.patch.object(stabilize_video, "run", side_effect=lambda command, **kwargs: commands.append(command)):
            stabilize_video.stabilize_shot(
                "ffmpeg", Path("source.mp4"), Path(tmp_text), 0, 0, 9, 24.0, args, "yuv420p"
            )

        transform_command = commands[1]
        filter_graph = transform_command[transform_command.index("-vf") + 1]
        self.assertIn("smoothing=4", filter_graph)
        self.assertIn("maxangle=0.05235988", filter_graph)

    def test_motion_guard_rejects_large_failed_global_fit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            report = Path(tmp_text) / "global_motions.trf"
            report.write_text(
                "0 0 0 0 0 1\n"
                "# no fields\n"
                "0 0.4 -0.8 0.0002 0 0\n"
                "# 4.2 3\n"
                "0 1.8 -71.6 -0.0081 0.39 1\n"
                "# 698.4 3\n",
                encoding="utf-8",
            )

            reason = stabilize_video.unreliable_motion_report(report, 108)

        self.assertIn("71.6px", reason)

    def test_motion_guard_allows_confident_fast_camera_move(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            report = Path(tmp_text) / "global_motions.trf"
            report.write_text(
                "0 0 0 0 0 1\n"
                "# no fields\n"
                "0 18 2 0.002 0 0\n"
                "# 2.0 3\n",
                encoding="utf-8",
            )

            reason = stabilize_video.unreliable_motion_report(report, 48)

        self.assertEqual(reason, "")

    def test_unreliable_shot_is_reencoded_without_transform(self) -> None:
        args = stabilize_video.build_parser().parse_args(["--source", "source.mp4"])
        commands = []
        with tempfile.TemporaryDirectory() as tmp_text:
            work_dir = Path(tmp_text)

            def fake_run(command, **_kwargs):
                commands.append(command)
                if len(commands) == 2:
                    (work_dir / "global_motions.trf").write_text(
                        "0 0 0 0 0 1\n0 0 -64 0 0 1\n",
                        encoding="utf-8",
                    )

            with mock.patch.object(stabilize_video, "run", side_effect=fake_run):
                stabilize_video.stabilize_shot(
                    "ffmpeg", Path("source.mp4"), work_dir, 0, 0, 24, 24.0, args, "yuv420p"
                )

        self.assertEqual(len(commands), 3)
        fallback_filter = commands[2][commands[2].index("-vf") + 1]
        self.assertNotIn("vidstabtransform", fallback_filter)

    def test_audio_mux_does_not_trim_the_authoritative_video_stream(self) -> None:
        commands = []
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            source.write_bytes(b"source")
            chunk = folder / "shot.mkv"
            chunk.write_bytes(b"chunk")
            output = folder / "output.mkv"

            def fake_run(command, **_kwargs):
                commands.append(command)
                Path(command[-1]).write_bytes(b"result")

            with mock.patch.object(stabilize_video, "run", side_effect=fake_run):
                stabilize_video.concat_chunks("ffmpeg", source, [chunk], output, folder, has_audio=True)

        self.assertNotIn("-shortest", commands[1])

    def test_user_reviewed_manifest_provides_exact_contiguous_shots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "shots.csv"
            manifest.write_text(
                "# source_video=source.mkv\n"
                "enabled,start_frame,end_frame,selected_frame,end,source_reference,color_reference,prompt,fade_to_next,crossfade_seconds\n"
                "true,0,76,34,00:00:03.167,a.png,b.png,,,\n"
                "true,76,148,108,00:00:06.167,c.png,d.png,,,\n"
                "true,148,294,220,00:00:12.250,e.png,f.png,,,\n",
                encoding="utf-8",
            )
            self.assertEqual(stabilize_video.manifest_shot_ranges(manifest, 294), [(0, 76), (76, 148), (148, 294)])

    def test_stabilization_detector_disables_pan_sensitive_long_window_heuristics(self) -> None:
        info = SimpleNamespace(width=1280, height=704, fps=16.0, frame_count=294)
        with (
            mock.patch.object(stabilize_video, "probe_video", return_value=info),
            mock.patch.object(stabilize_video, "sample_video", return_value=[object()]),
            mock.patch.object(stabilize_video, "detect_shots", return_value=[]) as detect,
        ):
            _info, ranges = stabilize_video.detect_shot_ranges(Path("continuous-pan.mp4"), 0.075, 1.0)

        detector_args = detect.call_args.args[2]
        self.assertEqual(detector_args.anchor_threshold, 0.0)
        self.assertEqual(detector_args.dissolve_threshold, 0.0)
        self.assertEqual(ranges, [(0, 294)])

    def test_single_shot_mode_bypasses_cut_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            source = Path(tmp_text) / "continuous-pan.mp4"
            source.write_bytes(b"fixture")
            args = stabilize_video.build_parser().parse_args([
                "--source", str(source), "--single-shot", "--dry-run",
            ])
            info = SimpleNamespace(width=1280, height=704, fps=16.0, frame_count=294)
            with (
                mock.patch.object(stabilize_video, "probe_video", return_value=info),
                mock.patch.object(stabilize_video, "detect_shot_ranges") as detect,
            ):
                self.assertEqual(stabilize_video.main_with_args(args), 0)

            detect.assert_not_called()
            self.assertFalse(stabilize_video.stabilization_identity(source, args)["scene_aware"])

    def test_end_to_end_keeps_exact_frame_count_and_writes_lossless_output(self) -> None:
        ffmpeg = find_ffmpeg()
        ffprobe = str(Path(ffmpeg).with_name("ffprobe.exe" if Path(ffmpeg).suffix.lower() == ".exe" else "ffprobe"))
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            output = folder / "stabilized.mkv"
            subprocess.run(
                [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=160x96:rate=12:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
                ],
                check=True,
            )
            args = stabilize_video.build_parser().parse_args(
                ["--source", str(source), "--output", str(output), "--smoothing", "3", "--zoom", "1"]
            )
            info = SimpleNamespace(width=160, height=96, fps=12.0, frame_count=24)
            with mock.patch.object(stabilize_video, "detect_shot_ranges", return_value=(info, [(0, 12), (12, 24)])):
                self.assertEqual(stabilize_video.main_with_args(args), 0)

            result = subprocess.run(
                [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=codec_name,nb_read_frames", "-of", "json", str(output)],
                check=True,
                capture_output=True,
                text=True,
            )
            stream = json.loads(result.stdout)["streams"][0]
            self.assertEqual(stream["codec_name"], "ffv1")
            self.assertEqual(int(stream["nb_read_frames"]), 24)
            self.assertTrue(output.with_suffix(output.suffix + ".sig.json").is_file())
            preview = stabilize_video.browser_preview_path(output)
            preview_result = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "json", str(preview)],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads(preview_result.stdout)["streams"][0]["codec_name"], "h264")


if __name__ == "__main__":
    unittest.main()
