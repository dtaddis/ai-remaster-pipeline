from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_remaster_gui import references, state
from ai_remaster_gui.manifests import read_manifest, write_manifest_details


class ReferenceScrubTests(unittest.TestCase):
    def test_extract_reference_frame_persists_selected_frame(self) -> None:
        previous_app = state.APP
        state.APP = SimpleNamespace(log=[], settings={"references": {}})
        try:
            with tempfile.TemporaryDirectory(dir=references.ROOT) as tmp_text:
                folder = Path(tmp_text)
                source = folder / "source.mp4"
                source.write_bytes(b"video placeholder")
                manifest = folder / "shots.csv"
                write_manifest_details(
                    manifest,
                    references.rel(source),
                    ["start", "end", "source_reference", "color_reference"],
                    [{
                        "start": "00:00:00.000",
                        "end": "00:00:02.000",
                        "source_reference": "",
                        "color_reference": "",
                    }],
                )

                fake_result = SimpleNamespace(returncode=0, stderr="", stdout="")
                with (
                    mock.patch.object(references, "local_tool", return_value="ffmpeg"),
                    mock.patch.object(references, "ffprobe_info", return_value={"frame_rate": "25.000 fps"}),
                    mock.patch.object(references.subprocess, "run", return_value=fake_result),
                ):
                    result = references.extract_reference_frame(references.rel(manifest), 0, 1.24)

                row = read_manifest(manifest)[0]
                self.assertEqual(row["selected_frame"], "31")
                self.assertEqual(result["selected_frame"], "31")
                self.assertEqual(row["source_reference"], result["source_reference"])
        finally:
            state.APP = previous_app

    def test_extract_reference_frame_uses_exact_frame_when_provided(self) -> None:
        previous_app = state.APP
        state.APP = SimpleNamespace(log=[], settings={"references": {}})
        try:
            with tempfile.TemporaryDirectory(dir=references.ROOT) as tmp_text:
                folder = Path(tmp_text)
                source = folder / "source.mp4"
                source.write_bytes(b"video placeholder")
                manifest = folder / "shots.csv"
                write_manifest_details(
                    manifest,
                    references.rel(source),
                    ["start", "end", "source_reference", "color_reference"],
                    [{
                        "start": "00:00:00.000",
                        "end": "00:00:02.000",
                        "source_reference": "",
                        "color_reference": "",
                    }],
                )

                fake_result = SimpleNamespace(returncode=0, stderr="", stdout="")
                with (
                    mock.patch.object(references, "local_tool", return_value="ffmpeg"),
                    mock.patch.object(references.subprocess, "run", return_value=fake_result) as run,
                ):
                    result = references.extract_reference_frame(references.rel(manifest), 0, 1.251, frame=30)

                command = run.call_args.args[0]
                self.assertEqual(command[command.index("-vf") + 1], "trim=start_frame=30:end_frame=31,setpts=PTS-STARTPTS")
                self.assertEqual(read_manifest(manifest)[0]["selected_frame"], "30")
                self.assertEqual(result["selected_frame"], "30")
        finally:
            state.APP = previous_app

    def test_additional_reference_round_trips_inside_one_shot(self) -> None:
        previous_app = state.APP
        state.APP = SimpleNamespace(log=[], settings={"references": {}})
        try:
            with tempfile.TemporaryDirectory(dir=references.ROOT) as tmp_text:
                folder = Path(tmp_text)
                source = folder / "source.mp4"
                source.write_bytes(b"video placeholder")
                primary = folder / "primary.png"
                primary.write_bytes(b"primary")
                manifest = folder / "shots.csv"
                write_manifest_details(
                    manifest,
                    references.rel(source),
                    ["start_frame", "end_frame", "selected_frame", "source_reference", "color_reference"],
                    [{
                        "start_frame": "0",
                        "end_frame": "100",
                        "selected_frame": "20",
                        "source_reference": references.rel(primary),
                        "color_reference": references.rel(primary),
                    }],
                )

                fake_result = SimpleNamespace(returncode=0, stderr="", stdout="")
                with (
                    mock.patch.object(references, "local_tool", return_value="ffmpeg"),
                    mock.patch.object(references, "manifest_fps", return_value=25.0),
                    mock.patch.object(references.subprocess, "run", return_value=fake_result),
                ):
                    added = references.add_reference_frame(references.rel(manifest), 0, 2.4, frame=60)

                row = read_manifest(manifest)[0]
                items = references.reference_items(row)
                self.assertEqual(added["reference_index"], "1")
                self.assertEqual([item["selected_frame"] for item in items], ["20", "60"])
                view = references.shot_rows(references.rel(manifest))[0]
                self.assertEqual(view["reference_count"], 2)
                self.assertEqual(view["reference_items"][1]["selected_frame"], 60)

                references.remove_additional_reference(references.rel(manifest), 0, 1)
                self.assertEqual(len(references.reference_items(read_manifest(manifest)[0])), 1)
        finally:
            state.APP = previous_app

    def test_merging_shots_preserves_the_following_shot_as_a_keyframe(self) -> None:
        previous_app = state.APP
        state.APP = SimpleNamespace(log=[], settings={"references": {}})
        try:
            with tempfile.TemporaryDirectory(dir=references.ROOT) as tmp_text:
                manifest = Path(tmp_text) / "shots.csv"
                write_manifest_details(
                    manifest,
                    "source.mp4",
                    ["start_frame", "end_frame", "selected_frame", "end", "source_reference", "color_reference"],
                    [
                        {"start_frame": "0", "end_frame": "50", "selected_frame": "20", "end": "00:00:02.000", "source_reference": "a.png", "color_reference": "a_color.png"},
                        {"start_frame": "50", "end_frame": "100", "selected_frame": "70", "end": "00:00:04.000", "source_reference": "b.png", "color_reference": "b_color.png"},
                    ],
                )
                with mock.patch.object(references, "manifest_fps", return_value=25.0):
                    references.merge_manifest_shots(references.rel(manifest), 0)

                rows = read_manifest(manifest)
                self.assertEqual(len(rows), 1)
                items = references.reference_items(rows[0])
                self.assertEqual([item["selected_frame"] for item in items], ["20", "70"])
                self.assertEqual([item["color_reference"] for item in items], ["a_color.png", "b_color.png"])
        finally:
            state.APP = previous_app

    def test_preview_reference_frame_can_use_exact_frame(self) -> None:
        with tempfile.TemporaryDirectory(dir=references.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            source.write_bytes(b"video placeholder")
            manifest = folder / "shots.csv"
            write_manifest_details(
                manifest,
                references.rel(source),
                ["start", "end", "source_reference", "color_reference"],
                [{"start": "00:00:00.000", "end": "00:00:02.000", "source_reference": "", "color_reference": ""}],
            )

            with mock.patch.object(references, "extract_video_frame_at_frame", return_value="preview.jpg") as exact:
                path = references.preview_reference_frame(references.rel(manifest), 0, 1.251, frame=30)

        self.assertEqual(path, "preview.jpg")
        self.assertEqual(exact.call_args.args[3], 30)

    def test_shot_rows_can_limit_preview_generation_to_selected_indices(self) -> None:
        previous_app = state.APP
        state.APP = SimpleNamespace(settings={"references": {}})
        try:
            with tempfile.TemporaryDirectory(dir=references.ROOT) as tmp_text:
                folder = Path(tmp_text)
                source = folder / "source.mp4"
                source.write_bytes(b"video placeholder")
                manifest = folder / "shots.csv"
                write_manifest_details(
                    manifest,
                    references.rel(source),
                    ["start_frame", "end_frame", "end", "source_reference", "color_reference"],
                    [
                        {"start_frame": "0", "end_frame": "24", "end": "00:00:01.000", "source_reference": "", "color_reference": ""},
                        {"start_frame": "24", "end_frame": "48", "end": "00:00:02.000", "source_reference": "", "color_reference": ""},
                        {"start_frame": "48", "end_frame": "72", "end": "00:00:03.000", "source_reference": "", "color_reference": ""},
                    ],
                )

                calls = []

                def fake_preview(_manifest, index, _seconds, frame=None):
                    calls.append((index, frame))
                    return f"preview-{index}-{frame}.jpg"

                with (
                    mock.patch.object(references, "manifest_fps", return_value=24.0),
                    mock.patch.object(references, "preview_reference_frame", side_effect=fake_preview),
                ):
                    rows = references.shot_rows(references.rel(manifest), include_previews=True, preview_indices={1})

            self.assertNotIn("start_preview", rows[0])
            self.assertEqual(rows[1]["start_preview"], "preview-1-24.jpg")
            self.assertNotIn("start_preview", rows[2])
            self.assertEqual([index for index, _frame in calls], [1, 1, 1])
        finally:
            state.APP = previous_app


if __name__ == "__main__":
    unittest.main()
