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
from ai_remaster_gui import server


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
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "0", "section_end": ""})
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

    def test_source_section_names_include_trim_points(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "12", "section_end": "24"})

        first = app.source_section_output_for(app.APP.settings)
        app.APP.settings["global"].update({"section_start": "45", "section_end": "60"})
        second = app.source_section_output_for(app.APP.settings)

        self.assertNotEqual(first, second)
        self.assertIn("0000012000_0000024000", first.name)
        self.assertIn("0000045000_0000060000", second.name)

    def test_pipeline_source_uses_section_when_trim_points_are_set(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "12", "section_end": "24"})

        self.assertIn("source_sections", app.pipeline_source_text(app.APP.settings))

    def test_project_payload_round_trips_settings_with_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            path = Path(tmp_text) / "demo.arpp"
            app.APP.settings["global"].update({"source": "input/example.mp4"})
            path.write_text(json.dumps(app.project_payload(app.APP.settings)), encoding="utf-8")

            loaded = app.read_project_file(path)

        self.assertEqual(loaded["global"]["source"], "input/example.mp4")
        self.assertIn("schema_version", app.project_payload(app.APP.settings))

    def test_project_save_suggestion_uses_last_browse_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            app.APP.settings["global"].update({"source": "input/example.mp4", "last_browse_dir": str(folder)})

            suggestion = app.project_save_suggestion(app.APP.settings)

        self.assertEqual(suggestion.parent, folder)
        self.assertEqual(suggestion.name, "example.arpp")

    def test_browse_initial_path_uses_last_browse_dir_without_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            app.APP.settings["global"]["last_browse_dir"] = str(folder)

            self.assertEqual(app.browse_initial_path("project_open", ""), folder)

    def test_browse_initial_path_prefers_last_dir_over_existing_current_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text, tempfile.TemporaryDirectory() as old_text:
            remembered = Path(tmp_text)
            old = Path(old_text)
            current = old / "layer.mp4"
            current.write_bytes(b"placeholder")
            app.APP.settings["global"]["last_browse_dir"] = str(remembered)

            self.assertEqual(app.browse_initial_path("save", str(current)), remembered / "layer.mp4")
            self.assertEqual(app.browse_initial_path("file", str(current)), remembered)

    def test_colorized_outputs_include_both_methods(self) -> None:
        outputs = app.colorized_outputs_for_manifest("manifests/references/colorize_manifest_demo_shots_auto.csv", "both")

        self.assertEqual(len(outputs), 2)
        self.assertTrue(outputs[0].endswith("_deepexemplar_colorized.mp4"))
        self.assertTrue(outputs[1].endswith("_colormnet_colorized.mp4"))

    def test_colorization_command_can_request_both_methods(self) -> None:
        app.APP.settings["colour"].update({"manifest": "manifests/references/colorize_manifest_demo_shots_auto.csv", "method": "both"})

        command = app.APP.command_for("colour")

        self.assertIn("--method", command)
        self.assertIn("both", command)
        self.assertNotIn("--output", command)

    def test_recomposition_output_path_uses_composited_suffix(self) -> None:
        output = app.recomposition_output_for("intermediate/outpainted/demo_outpainted.mp4")

        self.assertEqual(output, "output/reassembled/demo_outpainted_composited.mp4")

    def test_upscale_output_path_uses_recomposite_name(self) -> None:
        output = app.upscale_output_for("output/reassembled/demo_composited.mp4", {"method": "realbasicvsr", "scale": "4"})

        self.assertEqual(output, "output/upscaled/demo_composited_realbasicvsr_x4.mp4")

    def test_upscale_preview_output_path_uses_preview_seconds(self) -> None:
        output = app.upscale_preview_output_for(
            "output/reassembled/demo_composited.mp4",
            {"method": "realbasicvsr", "scale": "4", "preview_seconds": "8"},
        )

        self.assertEqual(output, "output/upscaled/previews/demo_composited_realbasicvsr_x4_preview_8s.mp4")

    def test_command_construction_for_upscale_uses_realbasicvsr(self) -> None:
        app.APP.settings["upscale"].update(
            {
                "input_video": "output/reassembled/demo_composited.mp4",
                "method": "realbasicvsr",
                "scale": "4",
                "output": "output/upscaled/demo_composited_realbasicvsr_x4.mp4",
                "realbasicvsr_repo": "tools/realbasicvsr",
                "max_seq_len": "0",
            }
        )

        command = app.APP.command_for("upscale")

        self.assertIn(str(app.SCRIPTS / "upscale_video.py"), command)
        self.assertIn("--input", command)
        self.assertIn("output/reassembled/demo_composited.mp4", command)
        self.assertIn("--realbasicvsr-repo", command)

    def test_upscale_preview_state_reports_expected_paths(self) -> None:
        app.APP.settings["upscale"].update(
            {
                "input_video": "output/reassembled/demo_composited.mp4",
                "method": "realbasicvsr",
                "scale": "4",
                "preview_seconds": "6",
            }
        )

        state = app.upscale_preview_state(app.APP.settings)

        self.assertEqual(state["input"], "output/reassembled/demo_composited.mp4")
        self.assertEqual(state["output"], "output/upscaled/previews/demo_composited_realbasicvsr_x4_preview_6s.mp4")

    def test_output_selection_prefers_existing_upscale(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            composited = folder / "demo_composited.mp4"
            upscaled = folder / "demo_composited_realbasicvsr_x4.mp4"
            composited.write_bytes(b"composited")
            upscaled.write_bytes(b"upscaled")
            app.APP.settings["recomp"]["output"] = app.rel(composited)
            app.APP.settings["upscale"]["output"] = app.rel(upscaled)

            selection = app.output_selection_state(app.APP.settings)

        self.assertEqual(selection["kind"], "upscaled")
        self.assertEqual(selection["path"], app.rel(upscaled))

    def test_output_selection_falls_back_to_composited(self) -> None:
        app.APP.settings["recomp"]["output"] = "output/reassembled/demo_composited.mp4"
        app.APP.settings["upscale"]["output"] = "output/upscaled/demo_composited_realbasicvsr_x4.mp4"

        selection = app.output_selection_state(app.APP.settings)

        self.assertEqual(selection["kind"], "composited")
        self.assertEqual(selection["path"], "output/reassembled/demo_composited.mp4")

    def test_section_preview_times_are_relative_to_trim_start(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "12", "section_end": "24"})

        self.assertAlmostEqual(app.section_relative_seconds(app.APP.settings, 12), 0.0)
        self.assertAlmostEqual(app.section_relative_seconds(app.APP.settings, 18.5), 6.5)
        self.assertAlmostEqual(app.section_relative_seconds(app.APP.settings, 30), 12.0)

    def test_outpaint_chunks_prepares_section_before_reading_it(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "12", "section_end": "24"})

        with mock.patch.object(server, "ensure_source_section_clip") as ensure, mock.patch.object(server, "resolve_video_source") as resolve_source:
            resolve_source.return_value = Path("missing-section.mp4")
            state = app.outpaint_chunks_state(app.APP.settings)

        ensure.assert_called_once_with(app.APP.settings)
        self.assertIn("not a readable file", state["error"])

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
