from __future__ import annotations

import copy
import csv
import json
import tempfile
import threading
import urllib.request
import unittest
from unittest import mock
from pathlib import Path

from ai_remaster_gui import app


class GuiSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._settings = copy.deepcopy(app.APP.settings)

    def tearDown(self) -> None:
        app.APP.settings = self._settings

    def test_source_resolver_accepts_ascii_pipe_for_full_width_pipe_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            real = folder / "King Kong Scene Pack ｜ King Kong [0JgMh4I2UjY].mp4"
            real.write_bytes(b"not a real video")
            typed = folder / "King Kong Scene Pack | King Kong [0JgMh4I2UjY].mp4"

            self.assertEqual(app.resolve_video_source(str(typed)), real)

    def test_deterministic_outpaint_output_path_uses_selected_source(self) -> None:
        app.APP.settings.setdefault("outpaint", {}).update(
            {
                "target_aspect": "16:9",
                "target_height": "720",
                "crop_left": "0",
                "crop_right": "0",
                "crop_top": "0",
                "crop_bottom": "0",
            }
        )

        output = app.outpaint_output_for("input/My Source.mp4", "16:9", "720")

        self.assertEqual(output, "intermediate/outpainted/My_Source_16x9_1280x720_outpainted.mp4")

    def test_outpaint_chunk_rows_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "chunks.csv"
            rows = [
                {
                    "chunk_index": "0",
                    "start_frame": "0",
                    "end_frame": "10",
                    "start_seconds": "0.000000",
                    "end_seconds": "0.416667",
                    "seed": "42",
                    "prompt_suffix": "",
                    "prepared_path": "prepared.mp4",
                    "raw_path": "raw.mp4",
                }
            ]

            app.write_outpaint_chunk_rows(manifest, rows)

            self.assertEqual(app.read_outpaint_chunk_rows(manifest)[0]["raw_path"], "raw.mp4")

    def test_reference_manifest_read_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "refs.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                handle.write("# source_video=input/example.mp4\n")
                writer = csv.DictWriter(handle, fieldnames=["enabled", "end", "source_reference", "color_reference", "prompt"])
                writer.writeheader()
                writer.writerow(
                    {
                        "enabled": "true",
                        "end": "00:00:01.000",
                        "source_reference": "bw.png",
                        "color_reference": "color.png",
                        "prompt": "",
                    }
                )

            source, fields, rows = app.read_manifest_details(manifest)

            self.assertEqual(source, "input/example.mp4")
            self.assertIn("color_reference", fields)
            self.assertEqual(rows[0]["source_reference"], "bw.png")

    def test_command_construction_for_outpaint_uses_overview_source(self) -> None:
        app.APP.settings["global"]["source"] = "input/example.mp4"
        app.APP.settings["outpaint"].update(
            {
                "target_aspect": "16:9",
                "target_height": "720",
                "chunk_seconds": "20",
                "overlap_frames": "8",
                "crop_left": "0",
                "crop_right": "0",
                "crop_top": "0",
                "crop_bottom": "0",
            }
        )

        command = app.APP.command_for("outpaint")

        self.assertIn("--source", command)
        self.assertIn("input/example.mp4", command)
        self.assertIn("--chunk-manifest", command)

    def test_media_clip_rejects_missing_source(self) -> None:
        with self.assertRaises(FileNotFoundError):
            app.media_clip_path(app.ROOT / "does-not-exist.mp4", 0, 1, "smoke")

    def test_files_for_skips_files_deleted_during_refresh(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            rel_folder = app.rel(folder)
            disappearing = folder / "vanishing.txt"
            disappearing.write_text("briefly here", encoding="utf-8")
            stage = app.Stage("smoke", "Smoke", "", (rel_folder,), (), ())
            real_stat = Path.stat
            calls = {"target": 0}

            def stat_once_then_missing(path: Path, *args, **kwargs):
                if path == disappearing:
                    calls["target"] += 1
                    if calls["target"] >= 2:
                        raise FileNotFoundError(str(path))
                return real_stat(path, *args, **kwargs)

            with mock.patch.object(Path, "stat", stat_once_then_missing):
                self.assertEqual(app.APP.files_for(stage), [])

    def test_state_endpoint_returns_json(self) -> None:
        server = app.create_server("127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/state", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        finally:
            server.shutdown()
            server.server_close()

        self.assertIn("stages", payload)
        self.assertIn("settings", payload)

    def test_root_serves_static_frontend_shell(self) -> None:
        server = app.create_server("127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=5) as response:
                html = response.read().decode("utf-8")
        finally:
            server.shutdown()
            server.server_close()

        self.assertIn('/static/styles.css', html)
        self.assertIn('/static/js/core.js', html)
        self.assertIn('/static/js/render-cache.js', html)
        self.assertIn('/static/js/app.js', html)


if __name__ == "__main__":
    unittest.main()
