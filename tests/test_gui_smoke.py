from __future__ import annotations

import copy
import csv
import json
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import unittest
import sys
import argparse
import importlib.util
import os
import zipfile
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import comfy_api  # noqa: E402
import audio_models  # noqa: E402
import common  # noqa: E402
import cleanup_video  # noqa: E402
import colorize_video  # noqa: E402
import create_audio_track  # noqa: E402
import generate_single_reference  # noqa: E402
import generate_references  # noqa: E402
import guide_frame_utils  # noqa: E402
import edit_reference_image  # noqa: E402
import final_composite  # noqa: E402
import finalize_outpaint_output  # noqa: E402
import openai_generate_reference  # noqa: E402
import outpaint_video  # noqa: E402
import prepare_outpaint_input  # noqa: E402
import qwen_colorize_references  # noqa: E402
import upscale_video  # noqa: E402

from ai_remaster_gui import app
from ai_remaster_gui import cache
from ai_remaster_gui import config
from ai_remaster_gui import lifecycle
from ai_remaster_gui import media
from ai_remaster_gui import outpaint_guides
from ai_remaster_gui import project_io
from ai_remaster_gui import sam_masks
from ai_remaster_gui import server
from ai_remaster_gui import system_status


class GuiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Isolate the suite from the developer's live .ai_remaster_gui.json regardless of how it is
        # launched. In particular `unittest discover -s tests` imports sibling test modules (which
        # pull in ai_remaster_gui.config) before tests/__init__.py runs, so the ARP_SETTINGS_FILE
        # redirect there can lose the import race and app.APP would load real settings. Redirect the
        # load and save paths to a throwaway file here and rebuild settings from defaults, then
        # snapshot that as the per-test baseline (setUp resets to a deep copy of it each test).
        tmp_settings = Path(tempfile.mkdtemp(prefix="arp-test-settings-")) / "settings.json"
        for target in (app, server):
            patcher = mock.patch.object(target, "SETTINGS_FILE", tmp_settings)
            patcher.start()
            cls.addClassCleanup(patcher.stop)
        app.APP.settings = server.load_settings()
        cls._pristine_settings = copy.deepcopy(app.APP.settings)

    def setUp(self) -> None:
        app.APP.settings = copy.deepcopy(self._pristine_settings)
        # Default the soundtrack phase off so stage-order / upscale-chaining tests are not
        # affected by whatever add_soundtrack happens to be in the loaded settings.
        app.APP.settings.setdefault("global", {})["add_soundtrack"] = "false"
        app.APP.quitting = False

    def tearDown(self) -> None:
        app.APP.quitting = False

    def _populate_full_pipeline_settings(self) -> None:
        """Populate settings so command_for can build a full command for every stage. Used by the
        stage-dispatch characterization tests below, which guard the per-stage routing while it is
        being consolidated out of the if/elif chains."""
        app.APP.settings["global"].update({
            "source": "input/fixture_clip.mp4", "section_start": "0", "section_end": "",
            "expand_outpaint": "true", "colorize": "true", "upscale": "true", "add_soundtrack": "true",
        })
        app.APP.settings["shots"]["outpainted_video"] = "intermediate/outpainted/Fixture_outpaint.mp4"
        app.APP.settings["references"].update({"manifest": "manifests/references/Fixture_shots.csv", "method": "qwen"})
        app.APP.settings["colour"]["manifest"] = "manifests/references/Fixture_shots.csv"
        app.APP.settings["recomp"].update({
            "outpainted_video": "intermediate/outpainted/Fixture_outpaint.mp4",
            "source": "input/fixture_clip.mp4",
            "colorized_video": "intermediate/outpainted_colorized/Fixture_color.mp4",
        })
        app.APP.settings["audio"]["input_video"] = "output/reassembled/Fixture_recomp.mp4"
        app.APP.settings["upscale"]["input_video"] = "output/reassembled/Fixture_recomp.mp4"

    def test_stage_commands_route_to_expected_scripts(self) -> None:
        # Characterization guard: every stage's command_for must launch its producer script via
        # `python -u <script>`. Protects the dispatch while it is consolidated into a stage registry.
        self._populate_full_pipeline_settings()
        expected = {
            "cleanup": "cleanup_video.py",
            "outpaint": "outpaint_video.py",
            "shots": "generate_references.py",
            "references": "qwen_colorize_references.py",
            "colour": "colorize_video.py",
            "recomp": "final_composite.py",
            "audio": "create_audio_track.py",
            "upscale": "upscale_video.py",
        }
        for key, script in expected.items():
            cmd = app.APP.command_for(key)
            self.assertTrue(cmd, f"{key} produced an empty command")
            self.assertEqual(cmd[:2], [sys.executable, "-u"], key)
            self.assertEqual([Path(p).name for p in cmd if p.endswith(".py")], [script], key)
        # The reference stage swaps scripts by method.
        app.APP.settings["references"]["method"] = "openai"
        self.assertIn("openai_generate_reference.py", [Path(p).name for p in app.APP.command_for("references")])

    def test_active_stages_for_global_flag_combinations(self) -> None:
        # Characterization guard for which phases run per global-toggle combination. Recomposition
        # is implied by outpaint OR colorize; audio and upscale are independent tail stages.
        def active(expand: str, colorize: str, soundtrack: str, upscale: str) -> list[str]:
            app.APP.settings["global"].update({
                "expand_outpaint": expand, "colorize": colorize,
                "add_soundtrack": soundtrack, "upscale": upscale,
            })
            return [stage.key for stage in app.APP.active_stages()]

        T, F = "true", "false"
        self.assertEqual(active(T, T, T, T), ["outpaint", "shots", "references", "colour", "recomp", "audio", "upscale"])
        self.assertEqual(active(T, F, F, F), ["outpaint", "recomp"])
        self.assertEqual(active(F, T, F, F), ["shots", "references", "colour", "recomp"])
        self.assertEqual(active(F, F, F, T), ["upscale"])
        self.assertEqual(active(F, F, T, F), ["audio"])
        self.assertEqual(active(F, F, F, F), [])

    def test_cleanup_is_optional_and_runs_before_outpainting(self) -> None:
        app.APP.settings["global"].update(
            {"cleanup": "true", "expand_outpaint": "true", "colorize": "true", "upscale": "false", "add_soundtrack": "false"}
        )

        self.assertEqual(
            [stage.key for stage in app.APP.active_stages()],
            ["cleanup", "outpaint", "shots", "references", "colour", "recomp"],
        )

    def test_cleanup_output_name_matches_gui_and_producer(self) -> None:
        source = app.resolve_video_source("input/example.mp4")
        values = app.APP.settings["cleanup"]
        args = cleanup_video.build_parser().parse_args(["--source", str(source)])

        self.assertEqual(app.cleanup_output_for(str(source), values), app.rel(cleanup_video.default_output(source, args)))

    def test_cleanup_output_routes_to_outpaint_when_ready(self) -> None:
        app.APP.settings["global"].update(
            {"source": "input/example.mp4", "cleanup": "true", "expand_outpaint": "true", "colorize": "false"}
        )
        cleaned = app.cleanup_output_for("input/example.mp4", app.APP.settings["cleanup"])
        cleaned_path = app.resolve(cleaned)
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_path.write_bytes(b"placeholder")
        try:
            app.APP.hydrate_stage_inputs("cleanup")
            command = app.APP.command_for("outpaint")
        finally:
            cleaned_path.unlink(missing_ok=True)

        self.assertEqual(command[command.index("--source") + 1], cleaned)
        self.assertEqual(app.APP.settings["recomp"]["source"], cleaned)

    def test_cleanup_output_routes_directly_to_colorization_when_outpainting_is_off(self) -> None:
        app.APP.settings["global"].update(
            {"source": "input/example.mp4", "cleanup": "true", "expand_outpaint": "false", "colorize": "true"}
        )
        cleaned = app.cleanup_output_for("input/example.mp4", app.APP.settings["cleanup"])
        cleaned_path = app.resolve(cleaned)
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_path.write_bytes(b"placeholder")
        try:
            app.APP.hydrate_stage_inputs("cleanup")
            shots_command = app.APP.command_for("shots")
        finally:
            cleaned_path.unlink(missing_ok=True)

        self.assertEqual(app.APP.settings["shots"]["outpainted_video"], cleaned)
        self.assertEqual(shots_command[shots_command.index("--source-video") + 1], cleaned)
        self.assertEqual(app.APP.settings["recomp"]["source"], cleaned)

    def test_cleanup_only_can_feed_upscaling(self) -> None:
        app.APP.settings["global"].update(
            {"source": "input/example.mp4", "cleanup": "true", "expand_outpaint": "false", "colorize": "false", "upscale": "true"}
        )
        cleaned = app.cleanup_output_for("input/example.mp4", app.APP.settings["cleanup"])
        cleaned_path = app.resolve(cleaned)
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_path.write_bytes(b"placeholder")
        try:
            app.APP.hydrate_stage_inputs("cleanup")
            command = app.APP.command_for("upscale")
        finally:
            cleaned_path.unlink(missing_ok=True)

        self.assertEqual([stage.key for stage in app.APP.active_stages()], ["cleanup", "upscale"])
        self.assertEqual(command[command.index("--input") + 1], cleaned)

    def test_cleanup_comparison_pairs_pipeline_source_with_finished_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            source = Path(tmp_text) / "comparison-source.mp4"
            source.write_bytes(b"placeholder")
            app.APP.settings["global"].update(
                {"source": app.rel(source), "cleanup": "true", "section_start": "0", "section_end": ""}
            )
            output = app.resolve(app.cleanup_output_for(app.rel(source), app.APP.settings["cleanup"]))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"placeholder")
            try:
                comparison = app.APP.cleanup_comparison_state()
            finally:
                output.unlink(missing_ok=True)

        self.assertEqual(comparison["source"], app.rel(source))
        self.assertEqual(comparison["output"], app.rel(output))
        self.assertEqual(comparison["source_exists"], "true")
        self.assertEqual(comparison["exists"], "true")

    def test_cleanup_page_wires_the_shared_before_after_player(self) -> None:
        helpers = (app.ROOT / "ai_remaster_gui" / "static" / "js" / "render-helpers.js").read_text(encoding="utf-8")
        comparison = (app.ROOT / "ai_remaster_gui" / "static" / "js" / "render-recomp.js").read_text(encoding="utf-8")

        self.assertIn("function drawCleanup", helpers)
        self.assertIn('id="cleanupCompareSlider"', helpers)
        self.assertIn("bindCleanupComparison()", helpers)
        self.assertIn("bindVideoComparison('cleanupCompareSlider')", comparison)

    def test_upscale_page_uses_a_full_width_per_shot_comparison_panel(self) -> None:
        comparison = (app.ROOT / "ai_remaster_gui" / "static" / "js" / "render-recomp.js").read_text(encoding="utf-8")

        self.assertIn('class="card upscale-shot-panel"', comparison)
        self.assertIn('data-upscale-shot-comparison', comparison)
        self.assertIn('/api/upscale-shot-comparison?', comparison)
        self.assertIn('bindUpscaleShotComparisons()', comparison)

    def test_upscale_shot_comparison_uses_latest_full_output_and_marks_it_stale(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "comparison-source.mp4"
            manifest = folder / "shots.csv"
            source.write_bytes(b"source")
            manifest.write_text("end\n1.0\n", encoding="utf-8")
            app.APP.settings["global"].update({
                "source": app.rel(source),
                "expand_outpaint": "false",
                "colorize": "false",
                "upscale": "true",
                "section_start": "0",
                "section_end": "",
            })
            app.APP.settings["colour"]["manifest"] = app.rel(manifest)
            output = app.resolve(app.upscale_output_for(app.rel(source), app.APP.settings["upscale"]))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"output")
            old = time.time() - 10
            os.utime(output, (old, old))

            try:
                with mock.patch.object(server, "extract_video_frame_at", side_effect=["before.jpg", "after.jpg"]) as extract:
                    result = app.APP.upscale_shot_comparison_frames(2, 4.25)
            finally:
                output.unlink(missing_ok=True)

        self.assertEqual(result, {"before": "before.jpg", "after": "after.jpg", "stale": "true"})
        self.assertEqual(extract.call_count, 2)
        self.assertEqual(extract.call_args_list[0].args[0], source)
        self.assertEqual(extract.call_args_list[1].args[0], output)
        self.assertEqual(extract.call_args_list[0].args[3], 4.25)

    def test_cleanup_chunk_duration_defaults_to_97_frames_and_valid_ltx_sizes(self) -> None:
        args = cleanup_video.build_parser().parse_args(["--source", "input/example.mp4"])

        self.assertEqual(args.chunk_seconds, 4.04)
        self.assertEqual(cleanup_video.dearchive_chunk_frames(args.chunk_seconds, 24.0), 97)
        self.assertEqual(cleanup_video.dearchive_chunk_frames(args.chunk_seconds, 23.976), 97)
        self.assertEqual(cleanup_video.dearchive_chunk_frames(args.chunk_seconds, 25.0), 97)
        self.assertEqual(cleanup_video.dearchive_chunk_frames(2.0, 24.0), 49)
        self.assertEqual(cleanup_video.dearchive_chunk_frames(20.0, 24.0), 481)
        for seconds in (2.0, 3.25, 4.04, 10.0, 20.0):
            self.assertEqual(cleanup_video.dearchive_chunk_frames(seconds, 24.0) % 8, 1)

    def test_cleanup_chunk_duration_is_bounded_to_two_through_twenty_seconds(self) -> None:
        parser = cleanup_video.build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["--source", "input/example.mp4", "--chunk-seconds", "1.99"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--source", "input/example.mp4", "--chunk-seconds", "20.01"])

    def test_cleanup_source_fidelity_defaults_to_exact_control_and_is_bounded(self) -> None:
        parser = cleanup_video.build_parser()

        self.assertEqual(parser.parse_args(["--source", "input/example.mp4"]).source_fidelity, 1.0)
        self.assertEqual(
            parser.parse_args(["--source", "input/example.mp4", "--source-fidelity", "0.55"]).source_fidelity,
            0.55,
        )
        with self.assertRaises(SystemExit):
            parser.parse_args(["--source", "input/example.mp4", "--source-fidelity", "-0.01"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--source", "input/example.mp4", "--source-fidelity", "1.01"])

    def test_cleanup_ui_exposes_bounded_chunk_duration_and_rounding_help(self) -> None:
        cleanup_stage = next(stage for stage in app.STAGES if stage.key == "cleanup")
        chunk_field = next(field for field in cleanup_stage.fields if field[0] == "chunk_seconds")
        helpers = (app.ROOT / "ai_remaster_gui" / "static" / "js" / "render-helpers.js").read_text(encoding="utf-8")

        self.assertEqual(chunk_field, ("chunk_seconds", "Dearchive chunk length", "range:2|20|0.01", "4.04"))
        self.assertIn("frames % 8 = 1 (8n + 1)", helpers)
        self.assertIn("4.04 seconds becomes 97 frames at 24 fps", helpers)
        self.assertIn("const RANGE_NUDGE_FIELDS = new Set(['chunk_seconds'", helpers)

    def test_cleanup_exposes_independent_repairs_and_optional_dearchive(self) -> None:
        cleanup_stage = next(stage for stage in app.STAGES if stage.key == "cleanup")
        fields = {field[0]: field for field in cleanup_stage.fields}

        self.assertNotIn("descratch", fields)
        self.assertEqual(fields["ai_descratch"], ("ai_descratch", "AI DeScratch (ProPainter)", "checkbox", "false"))
        self.assertEqual(fields["ai_descratch_height"], ("ai_descratch_height", "AI DeScratch resolution", "select:540|720|1080|source", "720"))
        self.assertEqual(fields["ai_chunk_frames"], ("ai_chunk_frames", "AI DeScratch chunk frames", "select:25|33|41|49", "41"))
        self.assertEqual(fields["devignette"], ("devignette", "DeVignette", "checkbox", "false"))
        self.assertEqual(fields["dearchive"], ("dearchive", "Dearchive (LTX 2.3 LoRA)", "checkbox", "true"))
        self.assertEqual(fields["repair_device"], ("repair_device", "DeVignette processor", "select:auto|cuda|cpu", "auto"))
        self.assertEqual(fields["source_fidelity"], ("source_fidelity", "Source Fidelity", "range:0|1|0.05", "1.0"))
        helpers = (app.ROOT / "ai_remaster_gui" / "static" / "js" / "render-helpers.js").read_text(encoding="utf-8")
        self.assertIn("Lower values give Dearchive more freedom to repaint damage", helpers)
        self.assertIn("non-commercial use only", helpers)
        self.assertIn("${cleanupLicenseWarning(s)}", helpers)
        self.assertIn("AI DeScratch is non-commercial only.", helpers)
        self.assertIn("s.ai_descratch !== 'true'", helpers)
        self.assertIn("github.com/sczhou/ProPainter/blob/main/LICENSE", helpers)
        parser = cleanup_video.build_parser()
        defaults = parser.parse_args(["--source", "input/example.mp4"])
        self.assertTrue(defaults.dearchive)
        self.assertFalse(defaults.ai_descratch)
        self.assertEqual(defaults.ai_descratch_height, "720")
        self.assertEqual(defaults.ai_chunk_frames, 41)
        self.assertAlmostEqual(defaults.scratch_sensitivity, 0.65)
        self.assertEqual(defaults.repair_device, "auto")
        local_only = parser.parse_args([
            "--source", "input/example.mp4", "--devignette", "--no-dearchive",
        ])
        self.assertTrue(local_only.devignette)
        self.assertFalse(local_only.dearchive)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--source", "input/example.mp4", "--descratch"])

    def test_cleanup_command_passes_independent_operation_flags(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "cleanup": "true"})
        app.APP.settings["cleanup"].update({
            "ai_descratch": "true", "devignette": "true",
            "dearchive": "false", "source_fidelity": "0.55",
        })

        command = app.APP.command_for("cleanup")

        self.assertNotIn("--descratch", command)
        self.assertIn("--ai-descratch", command)
        self.assertIn("--save-scratch-mask", command)
        self.assertIn("--devignette", command)
        self.assertIn("--no-dearchive", command)
        self.assertEqual(command[command.index("--repair-device") + 1], "auto")
        self.assertEqual(command[command.index("--source-fidelity") + 1], "0.55")
        self.assertEqual(command[command.index("--ai-descratch-height") + 1], "720")
        self.assertEqual(command[command.index("--ai-chunk-frames") + 1], "41")

    def test_cleanup_identity_changes_with_each_operation(self) -> None:
        source = "input/example.mp4"
        base = dict(app.APP.settings["cleanup"])
        paths = set()
        for ai_descratch, devignette, dearchive in (
            ("true", "false", "false"),
            ("false", "true", "false"),
            ("true", "true", "false"),
            ("false", "false", "true"),
            ("true", "true", "true"),
        ):
            values = dict(
                base,
                ai_descratch=ai_descratch,
                devignette=devignette,
                dearchive=dearchive,
            )
            paths.add(app.cleanup_output_for(source, values))

        self.assertEqual(len(paths), 5)

        weak = app.cleanup_output_for(source, dict(base, dearchive="true", source_fidelity="0.55"))
        exact = app.cleanup_output_for(source, dict(base, dearchive="true", source_fidelity="1.0"))
        self.assertNotEqual(weak, exact)
        ai = app.cleanup_output_for(source, dict(base, ai_descratch="true", dearchive="false"))
        local = app.cleanup_output_for(source, dict(base, ai_descratch="false", dearchive="false"))
        self.assertNotEqual(ai, local)

    def test_ai_descratch_mask_selects_vertical_damage_not_horizontal_picture_edges(self) -> None:
        import numpy as np

        height, width = 192, 320
        frame = np.full((height, width, 3), 128, dtype=np.uint8)
        frame[:, 91:93] = 28
        frame[:, 206] = 230
        frame[96:100, :] = 35

        mask = cleanup_video.ai_scratch_mask(frame, 0.65)

        self.assertGreater(np.count_nonzero(mask[:, 90:94]), height)
        self.assertGreater(np.count_nonzero(mask[:, 204:208]), height // 2)
        horizontal_only = np.concatenate(
            (mask[94:102, 16:80], mask[94:102, 110:190], mask[94:102, 220:304]),
            axis=1,
        )
        self.assertLess(np.count_nonzero(horizontal_only), 20)

    def test_ai_descratch_prompt_is_masked_and_source_geometry_is_separate(self) -> None:
        args = cleanup_video.build_parser().parse_args([
            "--source", "input/example.mp4", "--ai-descratch", "--no-dearchive",
        ])
        prompt = cleanup_video.propainter_prompt(
            Path("source.mp4"), Path("mask.mp4"), 1280, 720, 24.0, "test/repair", args
        )

        self.assertEqual(prompt["4"]["class_type"], "ProPainterInpaint")
        self.assertEqual(prompt["4"]["inputs"]["image"], ["1", 0])
        self.assertEqual(prompt["4"]["inputs"]["mask"], ["3", 0])
        self.assertEqual(prompt["4"]["inputs"]["width"], 1280)
        self.assertEqual(prompt["4"]["inputs"]["height"], 720)
        self.assertEqual(prompt["4"]["inputs"]["subvideo_length"], 41)
        self.assertTrue(prompt["5"]["inputs"]["save_output"])

    def test_devignette_corrects_both_dark_and_light_edges(self) -> None:
        import numpy as np

        height, width = 120, 160
        radius = cleanup_video.radial_coordinates(width, height)
        centre = radius < 0.25
        edge = (radius > 0.9) & (radius < 1.1)
        for field in (1.0 - 0.34 * np.minimum(radius, 1.0) ** 2, 1.0 + 0.30 * np.minimum(radius, 1.0) ** 2):
            frame = np.clip(128.0 * field[:, :, None] * np.ones((1, 1, 3)), 0, 255).astype(np.uint8)
            profile, ratio = cleanup_video.vignette_profile([frame] * 4)
            corrected = cleanup_video.apply_vignette_correction(frame, profile)
            before = abs(float(frame[edge].mean()) - float(frame[centre].mean()))
            after = abs(float(corrected[edge].mean()) - float(corrected[centre].mean()))

            self.assertIsNotNone(profile)
            self.assertTrue(ratio < 0.91 or ratio > 1.09)
            self.assertLess(after, before * 0.35)

    def test_devignette_excludes_and_preserves_black_presentation_bars(self) -> None:
        import numpy as np

        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        active = np.full((120, 160, 3), 128, dtype=np.float32)
        radius = cleanup_video.radial_coordinates(160, 120)
        active *= (1.0 - 0.30 * np.minimum(radius, 1.0) ** 2)[:, :, None]
        frame[:, 20:180] = np.clip(active, 0, 255).astype(np.uint8)

        profile, _ratio = cleanup_video.vignette_profile([frame] * 4)
        corrected = cleanup_video.apply_vignette_correction(frame, profile)

        self.assertIsNotNone(profile)
        self.assertAlmostEqual(profile.bounds[0], 0.1, places=2)
        self.assertAlmostEqual(profile.bounds[2], 0.9, places=2)
        self.assertEqual(int(corrected[:, :20].max()), 0)
        self.assertEqual(int(corrected[:, 180:].max()), 0)

    def test_cuda_devignette_processor_handles_vignette(self) -> None:
        import numpy as np

        available, reason = cleanup_video.TorchDeVignetteProcessor.available()
        if not available:
            self.skipTest(reason)
        height, width = 96, 256
        radius = cleanup_video.radial_coordinates(width, height)
        field = 1.0 - 0.25 * np.minimum(radius, 1.0) ** 2
        frame = np.clip(128.0 * field[:, :, None], 0, 255)
        frame = np.repeat(frame, 3, axis=2).astype(np.uint8)
        profile, _ratio = cleanup_video.vignette_profile([frame] * 4)
        processor = cleanup_video.TorchDeVignetteProcessor(width, height, profile, "cuda")

        repaired = processor.process([frame, frame])
        centre = radius < 0.25
        edge = (radius > 0.9) & (radius < 1.1)
        before = abs(float(frame[edge].mean()) - float(frame[centre].mean()))
        after = abs(float(repaired[0][edge].mean()) - float(repaired[0][centre].mean()))

        self.assertEqual(len(repaired), 2)
        self.assertEqual(repaired[0].shape, frame.shape)
        self.assertLess(after, before * 0.45)

    def test_devignette_streams_a_geometry_preserving_video_without_comfy(self) -> None:
        ffmpeg = common.find_ffmpeg()
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            source = folder / "damaged.mp4"
            work = folder / "work"
            subprocess.run(
                [
                    ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=gray:s=96x64:r=8:d=0.5",
                    "-vf", "drawbox=x=47:y=0:w=1:h=64:color=black:t=fill",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
                ],
                check=True,
                capture_output=True,
            )
            args = cleanup_video.build_parser().parse_args([
                "--source", str(source), "--devignette", "--no-dearchive",
            ])
            info = cleanup_video.probe_video(source)

            repaired = cleanup_video.prepare_devignette(ffmpeg, source, work, info, args)
            result = cleanup_video.probe_video(repaired)

        self.assertEqual((result["width"], result["height"]), (96, 64))
        self.assertAlmostEqual(result["fps"], 8.0, places=3)
        self.assertEqual(result["frames"], info["frames"])

    def test_cleanup_workflow_uses_dearchive_lora_and_standalone_video_vae(self) -> None:
        args = cleanup_video.build_parser().parse_args(["--source", "input/example.mp4"])
        args.outpaint_lora = args.cleanup_lora
        workflow = json.loads(cleanup_video.DEFAULT_WORKFLOW.read_text(encoding="utf-8-sig"))
        with (
            mock.patch.object(outpaint_video, "copy_to_comfy_input", return_value="arp_cleanup/example.mp4"),
            mock.patch.object(outpaint_video, "copy_reference_frame_to_comfy_input", return_value="arp_cleanup/frame.png"),
            mock.patch.object(outpaint_video, "probe_video", return_value={"width": 864, "height": 480, "frames": 97, "fps": 24.0}),
        ):
            prompt = outpaint_video.patch_workflow(
                args,
                workflow,
                Path("input/example.mp4"),
                Path("tools/comfyui"),
                "arp_cleanup/example",
                args.prompt,
                args.negative_prompt,
                args.seed,
            )

        self.assertEqual(prompt["5011"]["inputs"]["lora_name"], "ltx-2.3-dearchive-lora.safetensors")
        self.assertEqual(prompt["5011"]["inputs"]["strength_model"], 1.0)
        self.assertEqual(prompt["5012"]["inputs"]["strength"], 1.0)
        self.assertEqual(prompt["9001"]["class_type"], "VAELoader")
        self.assertEqual(prompt["4851"]["inputs"]["vae"], ["9001", 0])
        self.assertNotIn("Switch latent [Crystools]", {node["class_type"] for node in prompt.values()})

    def test_cleanup_source_fidelity_controls_the_complete_video_guide(self) -> None:
        args = cleanup_video.build_parser().parse_args([
            "--source", "input/example.mp4", "--source-fidelity", "0.55",
        ])
        args.outpaint_lora = args.cleanup_lora
        workflow = json.loads(cleanup_video.DEFAULT_WORKFLOW.read_text(encoding="utf-8-sig"))
        with (
            mock.patch.object(outpaint_video, "copy_to_comfy_input", return_value="arp_cleanup/example.mp4"),
            mock.patch.object(outpaint_video, "copy_reference_frame_to_comfy_input", return_value="arp_cleanup/frame.png"),
            mock.patch.object(outpaint_video, "probe_video", return_value={"width": 864, "height": 480, "frames": 97, "fps": 24.0}),
        ):
            prompt = outpaint_video.patch_workflow(
                args, workflow, Path("input/example.mp4"), Path("tools/comfyui"),
                "arp_cleanup/example", args.prompt, args.negative_prompt, args.seed,
            )

        self.assertEqual(prompt["5012"]["inputs"]["image"], ["5060", 0])
        self.assertEqual(prompt["5012"]["inputs"]["strength"], 0.55)
        self.assertTrue(prompt["5019"]["inputs"]["value"])

    def test_cleanup_finalize_restores_exact_source_geometry_and_timing(self) -> None:
        ffmpeg = common.find_ffmpeg()
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            generated = folder / "generated.mp4"
            output = folder / "cleaned.mp4"
            subprocess.run(
                [ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=gray:s=86x48:r=12:d=1",
                 "-f", "lavfi", "-i", "anullsrc=channel_layout=5.1:sample_rate=48000:d=1",
                 "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-map", "0:v:0",
                 "-map", "1:a:0", "-map", "2:a:0", "-shortest", "-vf", "setsar=1280/1281:max=100000",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source)],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=blue:s=96x64:r=12:d=0.75",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(generated)],
                check=True,
                capture_output=True,
            )
            info = cleanup_video.probe_video(source)

            cleanup_video.finalize_output(ffmpeg, generated, source, output, info)
            result = cleanup_video.probe_video(output)
            output_sar = cleanup_video.probe_sample_aspect_ratio(ffmpeg, output)
            ffprobe = str(Path(ffmpeg).with_name("ffprobe.exe" if Path(ffmpeg).suffix.lower() == ".exe" else "ffprobe"))
            audio_probe = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "a", "-show_entries", "stream=channels,channel_layout",
                 "-of", "json", str(output)], check=True, capture_output=True, text=True,
            )
            audio_streams = json.loads(audio_probe.stdout)["streams"]

        self.assertEqual((result["width"], result["height"]), (86, 48))
        self.assertAlmostEqual(result["fps"], 12.0, places=3)
        self.assertEqual(result["frames"], info["frames"])
        self.assertEqual(output_sar, "1280/1281")
        self.assertEqual(audio_streams, [{"channels": 2, "channel_layout": "stereo"}])

    def test_cleanup_finalize_handles_a_source_with_no_audio_packets(self) -> None:
        ffmpeg = common.find_ffmpeg()
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            source = folder / "silent-source.mp4"
            generated = folder / "generated.mp4"
            output = folder / "cleaned.mp4"
            for path, colour in ((source, "gray"), (generated, "blue")):
                subprocess.run(
                    [ffmpeg, "-y", "-f", "lavfi", "-i", f"color=c={colour}:s=86x48:r=12:d=1",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                    check=True,
                    capture_output=True,
                )

            self.assertIsNone(cleanup_video.first_audio_packet_stream(ffmpeg, source))
            cleanup_video.finalize_output(ffmpeg, generated, source, output, cleanup_video.probe_video(source))
            result = cleanup_video.probe_video(output)

        self.assertEqual((result["width"], result["height"], result["frames"]), (86, 48, 12))

    def test_colorization_always_builds_a_grayscale_model_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            source.write_bytes(b"fixture")
            with (
                mock.patch.object(colorize_video, "ROOT", folder),
                mock.patch.object(colorize_video, "resumable_output", return_value=False),
                mock.patch.object(colorize_video.subprocess, "run") as run,
                mock.patch.object(colorize_video, "replace_with_retry"),
                mock.patch.object(colorize_video, "write_signature"),
            ):
                output = colorize_video.prepare_processing_video("ffmpeg", source, 640, 360, 640, 360)

        command = run.call_args.args[0]
        self.assertIn("hue=s=0", command[command.index("-vf") + 1])
        self.assertTrue(output.name.endswith("_gray.mp4"))
        self.assertNotEqual(output, source)

    def test_both_colorization_backends_consume_the_guarded_video_input(self) -> None:
        for method, expected_node in (
            ("deepexemplar", "DeepExColorVideoNode"),
            ("colormnet", "ColorMNetVideo"),
        ):
            with self.subTest(method=method):
                args = colorize_video.build_parser().parse_args(["--manifest", "shots.csv", "--method", method])
                prompt = colorize_video.build_prompt(
                    "arp_colorize/guarded_gray.mp4", "reference.png", 12, 24, 640, 360, 24.0, args, "test/output"
                )

                self.assertEqual(prompt["1"]["inputs"]["video"], "arp_colorize/guarded_gray.mp4")
                self.assertEqual(prompt["3"]["class_type"], expected_node)
                self.assertEqual(prompt["3"]["inputs"]["video_frames"], ["1", 0])

    def test_source_resolver_accepts_ascii_pipe_for_full_width_pipe_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            real = folder / "King Kong Scene Pack ｜ King Kong [0JgMh4I2UjY].mp4"
            real.write_bytes(b"not a real video")
            typed = folder / "King Kong Scene Pack | King Kong [0JgMh4I2UjY].mp4"

            self.assertEqual(app.resolve_video_source(str(typed)), real)

    def test_default_comfy_dir_is_treated_as_managed_even_with_stale_flag(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            root = Path(tmp_text)
            comfy_dir = root / "tools" / "comfyui"
            comfy_dir.mkdir(parents=True)
            (comfy_dir / "main.py").write_text("# fake comfy", encoding="utf-8")
            config_file = root / ".ai_remaster_config.json"
            config_file.write_text(
                json.dumps({"comfy_dir": str(comfy_dir), "comfy_managed_by_arp": "false"}),
                encoding="utf-8",
            )

            with mock.patch.object(config, "ROOT", root), mock.patch.object(config, "CONFIG_FILE", config_file):
                loaded = config.load_config()

        self.assertEqual(loaded["comfy_dir"], str(comfy_dir))
        self.assertEqual(loaded["comfy_managed_by_arp"], "true")

    def test_run_all_waits_for_stage_hydration_before_next_stage(self) -> None:
        class DoneProcess:
            returncode = 0

            def poll(self):
                return 0

        stages = [stage for stage in app.STAGES if stage.key in {"outpaint", "shots"}]
        seen: list[tuple[str, str]] = []

        def fake_run_stage(stage_key: str) -> tuple[bool, str]:
            seen.append((stage_key, app.APP.settings.get("shots", {}).get("outpainted_video", "")))
            app.APP.process = DoneProcess()
            app.APP.running_stage_key = stage_key
            if stage_key == "outpaint":
                def finish_hydration() -> None:
                    time.sleep(0.1)
                    app.APP.settings.setdefault("shots", {})["outpainted_video"] = "intermediate/outpainted/movie.mp4"
                    app.APP.running_stage_key = ""

                threading.Thread(target=finish_hydration).start()
            else:
                app.APP.running_stage_key = ""
            return True, "started"

        with mock.patch.object(app.APP, "active_stages", return_value=tuple(stages)), mock.patch.object(app.APP, "run_stage", side_effect=fake_run_stage):
            app.APP._run_all_worker()

        self.assertEqual([key for key, _ in seen], ["outpaint", "shots"])
        self.assertEqual(seen[1][1], "intermediate/outpainted/movie.mp4")

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

        # Identity-keyed short name: <sourceword>_<tag>_<key>.mp4 under intermediate/outpainted/.
        self.assertTrue(output.startswith("intermediate/outpainted/My_outpaint_"), output)
        self.assertTrue(output.endswith(".mp4"))
        # The GUI locator and the producer script must name the file identically (no drift).
        args = argparse.Namespace(crop_left=0, crop_right=0, crop_top=0, crop_bottom=0, outpaint_all_black_regions=False)
        producer = outpaint_video.default_output(app.resolve_video_source("input/My Source.mp4"), "16:9", 720, args)
        self.assertEqual(Path(output).name, producer.name)

    def test_outpaint_ltx_working_paths_use_model_safe_size(self) -> None:
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

        prepared = app.outpaint_prepared_for("input/My Source.mp4", app.APP.settings["outpaint"])
        manifest = app.outpaint_chunk_manifest_for("input/My Source.mp4", app.APP.settings["outpaint"])

        self.assertEqual(prepared.parent.name, "outpaint_prepared")
        self.assertTrue(prepared.name.startswith("My_prepared_"), prepared.name)
        self.assertTrue(manifest.startswith("manifests/outpaint_chunks/My_chunks_"), manifest)
        self.assertTrue(manifest.endswith(".csv"))
        # GUI and producer must agree on the prepared-canvas and outpaint output names.
        args = argparse.Namespace(crop_left=0, crop_right=0, crop_top=0, crop_bottom=0, outpaint_all_black_regions=False)
        source = app.resolve_video_source("input/My Source.mp4")
        self.assertEqual(prepared.name, outpaint_video.prepared_for(source, "16:9", 720, args).name)
        self.assertEqual(
            Path(app.outpaint_output_for("input/My Source.mp4", "16:9", "720")).name,
            outpaint_video.default_output(source, "16:9", 720, args).name,
        )

    def test_source_height_outpaint_option_uses_video_height(self) -> None:
        app.APP.settings.setdefault("outpaint", {}).update(
            {
                "crop_left": "0",
                "crop_right": "0",
                "crop_top": "0",
                "crop_bottom": "0",
            }
        )
        with mock.patch.object(server, "video_metrics", return_value={"height": 480}):
            output = app.outpaint_output_for("input/My Source.mp4", "16:9", "source")
        output_720 = app.outpaint_output_for("input/My Source.mp4", "16:9", "720")

        self.assertTrue(output.startswith("intermediate/outpainted/My_outpaint_"), output)
        # "Source height" (480 -> work 864x480) must differ from a fixed 720 (-> 1280x704).
        self.assertNotEqual(output, output_720)
        # Agrees with the producer at the resolved height (480).
        args = argparse.Namespace(crop_left=0, crop_right=0, crop_top=0, crop_bottom=0, outpaint_all_black_regions=False)
        producer = outpaint_video.default_output(app.resolve_video_source("input/My Source.mp4"), "16:9", 480, args)
        self.assertEqual(Path(output).name, producer.name)

    def test_outpaint_source_crop_preserves_original_source_scale(self) -> None:
        args = argparse.Namespace(
            delivery_width=1280,
            delivery_height=720,
            crop_left=0,
            crop_right=0,
            crop_top=270,
            crop_bottom=270,
            black_lift=0.018,
            gamma=1.06,
            outpaint_all_black_regions=False,
        )
        info = {"width": 1440, "height": 1080, "fps": 24.0}

        self.assertEqual(prepare_outpaint_input.source_placement_size(args, info, 1280, 704), (960, 360, 960, 352))

        filter_text = prepare_outpaint_input.build_filter(args, info, 1280, 704)

        self.assertIn("trim=start_frame=0,setpts=N/(24.00000000*TB),fps=24.00000000", filter_text)
        self.assertIn("crop=w=1440:h=540:x=0:y=270,scale=w=960:h=360:flags=lanczos,scale=w=960:h=352:flags=lanczos", filter_text)
        self.assertNotIn("force_original_aspect_ratio=decrease", filter_text)

    def test_outpaint_all_black_regions_bypasses_source_lift(self) -> None:
        args = argparse.Namespace(
            delivery_width=1280,
            delivery_height=720,
            crop_left=0,
            crop_right=0,
            crop_top=0,
            crop_bottom=0,
            black_lift=0.018,
            gamma=1.06,
            outpaint_all_black_regions=True,
        )
        info = {"width": 960, "height": 720, "fps": 24.0}

        filter_text = prepare_outpaint_input.build_filter(args, info, 1280, 704)

        self.assertNotIn("lutrgb=", filter_text)
        self.assertIn("color=c=black:s=1280x704", filter_text)

    def test_standard_outpaint_preparation_lifts_source_but_keeps_padding_pure_black(self) -> None:
        args = argparse.Namespace(
            delivery_width=1280,
            delivery_height=720,
            crop_left=0,
            crop_right=0,
            crop_top=0,
            crop_bottom=0,
            black_lift=0.018,
            gamma=1.06,
            outpaint_all_black_regions=False,
        )
        info = {"width": 960, "height": 720, "fps": 24.0}

        filter_text = prepare_outpaint_input.build_filter(args, info, 1280, 704)

        self.assertIn("color=c=black:s=1280x704", filter_text)
        self.assertIn("lutrgb=", filter_text)
        self.assertIn("0.018", filter_text)
        self.assertIn("1/1.06", filter_text)

    def test_prepare_outpaint_partial_output_paths_are_per_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            output = Path(tmp_text) / "Example_prepared_12345678.mp4"

            first = prepare_outpaint_input.partial_output_path(output)
            second = prepare_outpaint_input.partial_output_path(output)

        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, output.parent)
        self.assertTrue(first.name.startswith("Example_prepared_12345678.partial."))
        self.assertTrue(first.name.endswith(".mp4"))

    def test_outpaint_all_black_regions_changes_output_paths_and_command(self) -> None:
        app.APP.settings.setdefault("outpaint", {}).update(
            {
                "target_aspect": "16:9",
                "target_height": "720",
                "crop_left": "0",
                "crop_right": "0",
                "crop_top": "0",
                "crop_bottom": "0",
                "outpaint_all_black_regions": "true",
                "black_mask_threshold": "12",
            }
        )
        app.APP.settings["global"].update({"source": "input/My Source.mp4", "section_start": "0", "section_end": ""})

        output_black = app.outpaint_output_for("input/My Source.mp4", "16:9", "720")
        prepared_black = app.outpaint_prepared_for("input/My Source.mp4", app.APP.settings["outpaint"])
        command = app.APP.command_for("outpaint")
        # The all-black variant must be a distinct artifact from the protected-blacks variant.
        app.APP.settings["outpaint"]["outpaint_all_black_regions"] = "false"
        output_plain = app.outpaint_output_for("input/My Source.mp4", "16:9", "720")
        app.APP.settings["outpaint"]["outpaint_all_black_regions"] = "true"

        self.assertTrue(output_black.startswith("intermediate/outpainted/My_outpaint_"), output_black)
        self.assertNotEqual(output_black, output_plain)
        self.assertTrue(prepared_black.name.startswith("My_prepared_"), prepared_black.name)
        self.assertIn("--outpaint-all-black-regions", command)
        self.assertEqual(command[command.index("--black-mask-threshold") + 1], "12")

    def test_final_composite_can_make_source_black_transparent(self) -> None:
        args = final_composite.build_parser().parse_args(
            [
                "--outpainted", "outpainted.mp4",
                "--source", "source.mp4",
                "--output", "final.mp4",
                "--source-black-transparent",
                "--source-black-threshold", "12",
            ]
        )

        filter_text = final_composite.build_filter(args, has_color=False, fps=24.0)

        self.assertIn("[0:v]setpts=N/(24.00000000*TB),fps=fps=24.00000000", filter_text)
        self.assertIn("[1:v]setpts=N/(24.00000000*TB),fps=fps=24.00000000", filter_text)
        self.assertIn("a='if(lte(min(", filter_text)
        self.assertIn("min(max(X-2,0),W-1)", filter_text)
        self.assertIn(",12),0,", filter_text)
        self.assertIn("[base][srcm]overlay", filter_text)

    def test_recomposition_command_punches_source_black_when_outpainting_all_black_regions(self) -> None:
        app.APP.settings["global"].update({"source": "input/My Source.mp4", "section_start": "0", "section_end": ""})
        app.APP.settings["outpaint"].update(
            {
                "target_aspect": "16:9",
                "target_height": "720",
                "crop_left": "0",
                "crop_right": "0",
                "crop_top": "0",
                "crop_bottom": "0",
                "outpaint_all_black_regions": "true",
            }
        )
        app.APP.settings["recomp"].update(
            {
                "outpainted_video": "intermediate/outpainted/My_outpaint.mp4",
                "source": "input/My Source.mp4",
                "output": "output/reassembled/My_recomp.mp4",
            }
        )

        command = app.APP.command_for("recomp")
        app.APP.settings["outpaint"]["outpaint_all_black_regions"] = "false"
        protected_command = app.APP.command_for("recomp")

        self.assertIn("--source-black-transparent", command)
        self.assertEqual(command[command.index("--source-black-threshold") + 1], "12")
        self.assertNotIn("--source-black-transparent", protected_command)

    def test_portable_comfy_parent_resolves_to_inner_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            portable = Path(tmp_text)
            inner = portable / "ComfyUI"
            inner.mkdir()
            (inner / "main.py").write_text("# comfy\n", encoding="utf-8")

            self.assertEqual(config.resolve_comfy_dir(str(portable)), inner)

    def test_comfy_startup_tail_reports_recent_log_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            log_path = Path(tmp_text) / "comfyui-startup.log"
            log_path.write_text("\n".join(f"line {index}" for index in range(120)), encoding="utf-8")

            with mock.patch.object(lifecycle, "COMFY_STARTUP_LOG", log_path):
                tail = lifecycle.tail_text(lifecycle.comfy_startup_log_path(), max_lines=3)

        self.assertEqual(tail.splitlines(), ["line 117", "line 118", "line 119"])

    def test_start_comfy_opens_console_window_and_records_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            root = Path(tmp_text)
            comfy = root / "tools" / "comfyui"
            comfy.mkdir(parents=True)
            (comfy / "main.py").write_text("# comfy\n", encoding="utf-8")
            log_path = root / "output" / "logs" / "comfyui-startup.log"

            class FakeProcess:
                returncode = None

                def poll(self):
                    return None

            popen_calls: list[tuple[list[str], dict]] = []

            def fake_popen(command, **kwargs):
                popen_calls.append((command, kwargs))
                return FakeProcess()

            with (
                mock.patch.object(lifecycle, "STARTED_COMFY_PROCESS", None),
                mock.patch.object(lifecycle, "COMFY_STARTUP_LOG", log_path),
                mock.patch.object(lifecycle, "current_config", return_value={
                    "comfy_dir": str(comfy),
                    "comfy_url": "http://127.0.0.1:8188",
                    "comfy_host": "127.0.0.1",
                    "comfy_port": "8188",
                    "comfy_managed_by_arp": "true",
                }),
                mock.patch.object(lifecycle, "discover_comfy_instances", return_value=[]),
                mock.patch.object(lifecycle.subprocess, "Popen", side_effect=fake_popen),
                mock.patch.object(lifecycle.threading.Thread, "start", lambda _self: None),
            ):
                lifecycle.start_comfy_if_needed()

            self.assertEqual(len(popen_calls), 1)
            _command, kwargs = popen_calls[0]
            self.assertEqual(kwargs["cwd"], str(comfy))
            # ComfyUI runs in its own visible console window, so its output must NOT be redirected
            # away to a file (that would blank the window). Only the launch banner is recorded.
            self.assertNotIn("stdout", kwargs)
            self.assertNotIn("stderr", kwargs)
            if os.name == "nt":
                self.assertTrue(kwargs["creationflags"] & subprocess.CREATE_NEW_CONSOLE)
            self.assertTrue(log_path.exists())
            self.assertIn("Starting ComfyUI:", log_path.read_text(encoding="utf-8"))

    def test_stage_comfy_gate_waits_for_existing_launch(self) -> None:
        class FakeProcess:
            returncode = None

            def poll(self):
                return None

        fake_process = FakeProcess()
        with (
            mock.patch.object(lifecycle, "STARTED_COMFY_PROCESS", fake_process),
            mock.patch.object(lifecycle, "current_config", return_value={"comfy_url": "http://127.0.0.1:8188"}),
            mock.patch.object(lifecycle, "discover_comfy_instances", return_value=[]),
            mock.patch.object(lifecycle, "comfy_is_running", side_effect=[False, True]) as is_running,
            mock.patch.object(lifecycle.time, "sleep"),
        ):
            ok, message = lifecycle.ensure_comfy_available_for_stage("Outpainting")

        self.assertTrue(ok)
        self.assertEqual(message, "")
        self.assertEqual(is_running.call_count, 2)

    def test_wait_for_comfy_ready_clears_dead_launch_handle(self) -> None:
        class ExitedProcess:
            returncode = 1

            def poll(self):
                return 1

        process = ExitedProcess()
        with (
            mock.patch.object(lifecycle, "STARTED_COMFY_PROCESS", process),
            mock.patch.object(lifecycle, "comfy_is_running", return_value=False),
        ):
            ready = lifecycle.wait_for_comfy_ready("http://127.0.0.1:8188", process, timeout_seconds=1)
            self.assertIsNone(lifecycle.STARTED_COMFY_PROCESS)

        self.assertFalse(ready)

    def test_required_comfy_workflows_are_bundled(self) -> None:
        outpaint = app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json"
        qwen = app.ROOT / "workflows" / "qwen_image_edit" / "Image Edit (Qwen 2511).json"

        for workflow in (outpaint, qwen):
            with self.subTest(workflow=workflow.name):
                self.assertTrue(workflow.exists(), f"Missing bundled workflow: {workflow}")
                json.loads(workflow.read_text(encoding="utf-8-sig"))

        self.assertEqual(server.default_qwen_workflow({}), app.rel(qwen))
        self.assertEqual(server.qwen_workflow_for({"workflow": "D:/missing/blueprints/Qwen Custom.json"}, {}), app.rel(qwen))
        self.assertEqual(
            server.qwen_workflow_for(
                {
                    "workflow": (
                        "D:/ComfyUI/venv/Lib/site-packages/"
                        "comfyui_workflow_templates_media_image/templates/image_qwen_image_edit_2511.json"
                    )
                },
                {},
            ),
            app.rel(qwen),
        )

    def test_required_custom_nodes_are_bundled(self) -> None:
        required = {
            "ComfyUI-ARP": ("ARPLTXVideoOnlyICLoRALoader",),
            "ComfyUI-LTXVideo": ("LTXVImgToVideoConditionOnly", "LTXAddVideoICLoRAGuideAdvanced", "LTXVInpaintPreprocess", "LTXVLaplacianPyramidBlend"),
            "ComfyUI-GGUF": ("UnetLoaderGGUF",),
            "ComfyUI-VideoHelperSuite": ("VHS_LoadVideo", "VHS_VideoCombine"),
            "ComfyUI-FlashVSR_Ultra_Fast": ("FlashVSRInitPipe", "FlashVSRNodeAdv"),
            "reference-video-colorization": ("DeepExColorVideoNode", "ColorMNetVideo"),
        }
        vendor_root = app.ROOT / "vendor" / "comfyui_custom_nodes"
        for folder, symbols in required.items():
            with self.subTest(folder=folder):
                package = vendor_root / folder
                self.assertTrue(package.is_dir(), f"Missing bundled custom node package: {package}")
                self.assertTrue((package / "LICENSE").exists(), f"Missing bundled custom node license: {package}")
                texts = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in package.rglob("*.py"))
                for symbol in symbols:
                    self.assertIn(symbol, texts)

        ltx_connector = vendor_root / "ComfyUI-LTXVideo" / "embeddings_connector.py"
        connector_text = ltx_connector.read_text(encoding="utf-8")
        self.assertIn("from comfy.ldm.lightricks.model import freqs_cis_matrix", connector_text)
        self.assertIn("_USE_FREQS_CIS_MATRIX = True", connector_text)
        upstream_init = (vendor_root / "ComfyUI-LTXVideo" / "__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("guide_attention_patch", upstream_init)
        arp_init = (vendor_root / "ComfyUI-ARP" / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("install_sparse_guide_attention_patch()", arp_init)

    def test_wait_for_prompt_retries_transient_polling_errors(self) -> None:
        calls = {"count": 0}

        def fake_http_json(method, url, timeout=30):
            calls["count"] += 1
            if calls["count"] < 3:
                raise RuntimeError("Timed out waiting for ComfyUI")
            return {"prompt-id": {"status": {"completed": True}, "outputs": {}}}

        with mock.patch.object(comfy_api, "http_json", side_effect=fake_http_json), mock.patch.object(comfy_api.time, "sleep"):
            history = comfy_api.wait_for_prompt("http://127.0.0.1:8188", "prompt-id", 0.01, transient_timeout_seconds=30)

        self.assertEqual(calls["count"], 3)
        self.assertEqual(history["status"]["completed"], True)

    def test_bundled_outpaint_template_contains_official_two_stage_nodes(self) -> None:
        workflow = json.loads((app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json").read_text(encoding="utf-8-sig"))

        prompt = comfy_api.workflow_to_prompt(workflow, "5228")

        class_types = {node["class_type"] for node in prompt.values()}
        self.assertNotIn("ImagePadKJ", class_types)
        self.assertIn("LoadVideo", class_types)
        self.assertIn("SaveVideo", class_types)
        self.assertEqual(sum(node["class_type"] == "SamplerCustomAdvanced" for node in prompt.values()), 2)
        self.assertEqual(sum(node["class_type"] == "LTXVLaplacianPyramidBlend" for node in prompt.values()), 2)

    def test_outpaint_prompt_sent_to_ic_lora_guide_combines_global_and_chunk_suffix(self) -> None:
        workflow = json.loads((app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json").read_text(encoding="utf-8-sig"))
        args = outpaint_video.build_parser().parse_args(
            [
                "--source",
                "input/example.mp4",
                "--comfy-dir",
                str(app.ROOT),
                "--prompt",
                "outpaint with natural film grain",
                "--dry-run",
            ]
        )

        with (
            mock.patch.object(outpaint_video, "copy_to_comfy_input", return_value="arp_outpaint/prepared.mp4"),
            mock.patch.object(outpaint_video, "copy_reference_frame_to_comfy_input", return_value="arp_outpaint/reference.png"),
            mock.patch.object(outpaint_video, "official_mask_image", return_value=app.ROOT / "mask.png"),
            mock.patch.object(outpaint_video, "generation_mask_image", return_value=app.ROOT / "generation-mask.png"),
            mock.patch.object(outpaint_video, "probe_video", return_value={"width": 1280, "height": 704, "frames": 24, "fps": 24.0}),
        ):
            prompt = outpaint_video.patch_workflow(
                args,
                workflow,
                app.ROOT / "prepared.mp4",
                app.ROOT,
                "arp_outpaint/test",
                outpaint_video.combine_prompt(args.prompt, "continue the wallpaper"),
                args.negative_prompt,
                42,
            )

        self.assertEqual(prompt["2483"]["inputs"]["text"], "outpaint with natural film grain. continue the wallpaper")
        self.assertEqual(prompt["5114"]["inputs"]["positive"], ["1241", 0])
        self.assertEqual(prompt["1241"]["inputs"]["positive"], ["2483", 0])

    def test_outpaint_conditioning_bypasses_resize_without_replacing_video_control(self) -> None:
        workflow = json.loads((app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json").read_text(encoding="utf-8-sig"))
        args = outpaint_video.build_parser().parse_args(["--source", "input/example.mp4", "--comfy-dir", str(app.ROOT), "--dry-run"])

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            guide = Path(tmp_text) / "guide.png"
            guide.write_bytes(b"guide")
            with (
                mock.patch.object(outpaint_video, "copy_to_comfy_input", return_value="arp_outpaint/prepared.mp4"),
                mock.patch.object(outpaint_video, "copy_guide_image_to_comfy_input", return_value="arp_outpaint/guide_864x480.png"),
                mock.patch.object(outpaint_video, "official_mask_image", return_value=app.ROOT / "mask.png"),
                mock.patch.object(outpaint_video, "generation_mask_image", return_value=app.ROOT / "generation-mask.png"),
                mock.patch.object(outpaint_video, "probe_video", return_value={"width": 864, "height": 480, "frames": 24, "fps": 24.0}),
            ):
                prompt = outpaint_video.patch_workflow(
                    args,
                    workflow,
                    app.ROOT / "prepared.mp4",
                    app.ROOT,
                    "arp_outpaint/test",
                    args.prompt,
                    args.negative_prompt,
                    42,
                    guide,
                )

        self.assertEqual(prompt["5358"]["inputs"]["images"], ["5168", 0])
        self.assertEqual(prompt["5358"]["inputs"]["mask"], ["9104", 0])
        self.assertEqual(prompt["5114"]["inputs"]["image"], ["5358", 0])
        self.assertEqual(prompt["3159"]["inputs"]["image"], ["2004", 0])
        self.assertEqual(prompt["2004"]["inputs"]["image"], "arp_outpaint/guide_864x480.png")
        self.assertNotIn("4922", prompt)
        self.assertEqual(prompt["5011"]["inputs"]["model"], ["3940", 0])
        self.assertEqual(prompt["5011"]["class_type"], "ARPLTXVideoOnlyICLoRALoader")
        self.assertNotIn("video_only", prompt["5011"]["inputs"])
        self.assertEqual(prompt["5368"]["inputs"]["file"], "arp_outpaint/prepared.mp4")
        self.assertEqual(prompt["9100"]["class_type"], "LoadImage")
        self.assertEqual(prompt["9101"]["class_type"], "ImageToMask")
        self.assertEqual(prompt["9103"]["class_type"], "LoadImage")
        self.assertEqual(prompt["9104"]["class_type"], "ImageToMask")
        self.assertEqual(prompt["9104"]["inputs"]["image"], ["9103", 0])
        self.assertEqual(prompt["3059"]["inputs"]["width"], ["5054", 0])
        self.assertEqual(prompt["3059"]["inputs"]["height"], ["5054", 1])
        self.assertEqual(prompt["3059"]["inputs"]["length"], ["5054", 2])
        self.assertEqual(prompt["5023"]["inputs"]["device"], "cpu")
        self.assertEqual(prompt["5093"]["inputs"]["latent_image"], ["5114", 2])
        self.assertEqual(prompt["5013"]["inputs"]["latent"], ["5093", 1])
        for audio_diffusion_node in ("5382", "5383", "5385", "5386", "5389", "5390", "9110"):
            self.assertNotIn(audio_diffusion_node, prompt)
        self.assertEqual(prompt["5266"]["inputs"]["mask"], ["9101", 0])
        self.assertEqual(prompt["5266"]["inputs"]["image_b"], ["5168", 0])
        self.assertEqual(prompt["5266"]["inputs"]["mask_low_res_dilation"], 5)
        self.assertEqual(prompt["5227"]["inputs"]["images"], ["5266", 0])
        self.assertEqual(prompt["5227"]["inputs"]["audio"], ["5168", 1])
        self.assertEqual(sum(node["class_type"] == "SamplerCustomAdvanced" for node in prompt.values()), 1)
        for bypassed_node in ("5211", "5214", "5226", "5357", "5360", "5361", "5364", "5365", "5379", "5380"):
            self.assertNotIn(bypassed_node, prompt)

    def test_outpaint_official_graph_remuxes_source_audio_without_audio_diffusion(self) -> None:
        workflow = json.loads((app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json").read_text(encoding="utf-8-sig"))
        args = outpaint_video.build_parser().parse_args(["--source", "input/example.mp4", "--comfy-dir", str(app.ROOT), "--dry-run"])
        with (
            mock.patch.object(outpaint_video, "copy_to_comfy_input", return_value="arp_outpaint/prepared.mp4"),
            mock.patch.object(outpaint_video, "copy_reference_frame_to_comfy_input", return_value="arp_outpaint/reference.png"),
            mock.patch.object(outpaint_video, "official_mask_image", return_value=app.ROOT / "mask.png"),
            mock.patch.object(outpaint_video, "generation_mask_image", return_value=app.ROOT / "generation-mask.png"),
            mock.patch.object(outpaint_video, "probe_video", return_value={"width": 864, "height": 480, "frames": 24, "fps": 24.0}),
        ):
            prompt = outpaint_video.patch_workflow(
                args, workflow, app.ROOT / "prepared.mp4", app.ROOT, "arp_outpaint/test",
                args.prompt, args.negative_prompt, 42,
            )

        self.assertEqual(prompt["5227"]["inputs"]["audio"], ["5168", 1])
        self.assertEqual(prompt["5093"]["inputs"]["latent_image"], ["5114", 2])
        self.assertEqual(prompt["5013"]["inputs"]["latent"], ["5093", 1])
        for audio_diffusion_node in ("5382", "5383", "5385", "5386", "5389", "5390", "9110"):
            self.assertNotIn(audio_diffusion_node, prompt)

    def test_outpaint_extra_guides_remain_in_video_only_sampler_chain(self) -> None:
        workflow = json.loads((app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json").read_text(encoding="utf-8-sig"))
        args = outpaint_video.build_parser().parse_args(["--source", "input/example.mp4", "--comfy-dir", str(app.ROOT), "--dry-run"])
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            guide = Path(tmp_text) / "guide.png"
            guide.write_bytes(b"guide")
            with (
                mock.patch.object(outpaint_video, "copy_to_comfy_input", return_value="arp_outpaint/prepared.mp4"),
                mock.patch.object(outpaint_video, "copy_reference_frame_to_comfy_input", return_value="arp_outpaint/reference.png"),
                mock.patch.object(outpaint_video, "copy_guide_image_to_comfy_input", return_value="arp_outpaint/extra.png"),
                mock.patch.object(outpaint_video, "official_mask_image", return_value=app.ROOT / "mask.png"),
                mock.patch.object(outpaint_video, "generation_mask_image", return_value=app.ROOT / "generation-mask.png"),
                mock.patch.object(outpaint_video, "probe_video", return_value={"width": 864, "height": 480, "frames": 24, "fps": 24.0}),
            ):
                prompt = outpaint_video.patch_workflow(
                    args, workflow, app.ROOT / "prepared.mp4", app.ROOT, "arp_outpaint/test",
                    args.prompt, args.negative_prompt, 42,
                    extra_guides=[{"frame_idx": 16, "strength": 0.5, "image": guide}],
                )

        self.assertEqual(prompt["5093"]["inputs"]["latent_image"], ["9060", 2])
        self.assertEqual(prompt["5013"]["inputs"]["latent"], ["5093", 1])
        self.assertEqual(prompt["9060"]["inputs"]["latent"], ["5114", 2])
        self.assertNotIn("5383", prompt)
        self.assertNotIn("5386", prompt)

    def test_outpaint_monochrome_finish_removes_residual_chroma(self) -> None:
        args = finalize_outpaint_output.build_parser().parse_args(["--source", "input.mp4", "--monochrome"])
        filter_graph = finalize_outpaint_output.inverse_filter(args, {"width": 1280, "height": 704})

        self.assertIn(",hue=s=0,", filter_graph)

    def test_outpaint_all_black_mode_reuses_source_frames_for_dynamic_mask(self) -> None:
        workflow = json.loads((app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json").read_text(encoding="utf-8-sig"))
        args = outpaint_video.build_parser().parse_args(
            ["--source", "input/example.mp4", "--comfy-dir", str(app.ROOT), "--outpaint-all-black-regions", "--black-mask-threshold", "17", "--dry-run"]
        )
        with (
            mock.patch.object(outpaint_video, "copy_to_comfy_input", return_value="arp_outpaint/prepared.mp4"),
            mock.patch.object(outpaint_video, "copy_reference_frame_to_comfy_input", return_value="arp_outpaint/reference.png"),
            mock.patch.object(outpaint_video, "probe_video", return_value={"width": 864, "height": 480, "frames": 24, "fps": 24.0}),
            mock.patch.object(outpaint_video, "video_has_audio", return_value=False),
            mock.patch.object(outpaint_video, "official_mask_image") as static_mask,
        ):
            prompt = outpaint_video.patch_workflow(
                args, workflow, app.ROOT / "prepared.mp4", app.ROOT, "arp_outpaint/test",
                args.prompt, args.negative_prompt, 42,
            )

        static_mask.assert_not_called()
        self.assertEqual(prompt["9100"]["class_type"], "ImageToMask")
        self.assertEqual(prompt["9100"]["inputs"]["image"], ["5168", 0])
        self.assertEqual(prompt["9101"]["class_type"], "ThresholdMask")
        self.assertAlmostEqual(prompt["9101"]["inputs"]["value"], 17 / 255)
        self.assertEqual(prompt["9102"]["class_type"], "InvertMask")
        self.assertEqual(prompt["5358"]["inputs"]["mask"], ["9105", 0])
        self.assertEqual(prompt["5266"]["inputs"]["mask"], ["9102", 0])
        self.assertEqual(prompt["5266"]["inputs"]["image_b"], ["5168", 0])

    def test_outpaint_static_mask_is_one_geometric_frame(self) -> None:
        import cv2
        import numpy as np

        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        frame[:, 16:48] = 5
        capture = mock.Mock()
        capture.get.return_value = 1
        capture.read.return_value = (True, frame)
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            root = Path(tmp_text)
            prepared = root / "prepared.mp4"
            prepared.write_bytes(b"prepared")
            args = argparse.Namespace(force=True)
            with (
                mock.patch.object(outpaint_video, "ROOT", root),
                mock.patch.object(outpaint_video, "file_fingerprint", return_value={"size": 8, "mtime_ns": 1, "sha256": "test"}),
                mock.patch.object(cv2, "VideoCapture", return_value=capture),
            ):
                mask_path = outpaint_video.official_mask_image(prepared, args, 64, 32)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        self.assertEqual(mask.shape, (32, 64))
        self.assertTrue(np.all(mask[:, :16] == 255))
        self.assertTrue(np.all(mask[:, 16:48] == 0))
        self.assertTrue(np.all(mask[:, 48:] == 255))

    def test_outpaint_generation_mask_only_grows_thin_edge_bands_to_one_latent_cell(self) -> None:
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            exact_path = Path(tmp_text) / "exact.png"
            exact = np.full((704, 1280), 255, dtype=np.uint8)
            exact[4:700, 161:1119] = 0
            self.assertTrue(cv2.imwrite(str(exact_path), exact))
            args = argparse.Namespace(force=True, generation_mask_overlap=64)

            generation_path = outpaint_video.generation_mask_image(exact_path, args)
            generation = cv2.imread(str(generation_path), cv2.IMREAD_GRAYSCALE)
            exact_after = cv2.imread(str(exact_path), cv2.IMREAD_GRAYSCALE)

        self.assertTrue(np.array_equal(exact_after, exact))
        self.assertTrue(np.all(generation[:32] == 255))
        self.assertTrue(np.all(generation[32:672, :161] == 255))
        self.assertTrue(np.all(generation[32:672, 161:1119] == 0))
        self.assertTrue(np.all(generation[672:] == 255))

    def test_ltx_laplacian_blend_keeps_fully_masked_thin_bands_exact(self) -> None:
        import importlib
        import sys
        import types

        import torch

        package_name = "arp_test_ltxvideo"
        comfy_dir = app.ROOT / "tools" / "comfyui"
        package_dir = app.ROOT / "tools" / "comfyui" / "custom_nodes" / "ComfyUI-LTXVideo"
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_dir)]
        sys.modules[package_name] = package
        original_sys_path = list(sys.path)
        scripts_dir = str(app.ROOT / "scripts")
        sys.path[:] = [str(comfy_dir)] + [entry for entry in sys.path if entry != scripts_dir]
        script_comfy_api = sys.modules.pop("comfy_api", None)
        try:
            blending = importlib.import_module(f"{package_name}.pyramid_blending")
        finally:
            sys.path[:] = original_sys_path
            if script_comfy_api is not None:
                sys.modules["comfy_api"] = script_comfy_api

        generated = torch.ones((1, 3, 32, 64), dtype=torch.float32)
        prepared = torch.zeros_like(generated)
        exact_mask = torch.zeros((1, 1, 32, 64), dtype=torch.float32)
        exact_mask[:, :, :2, :] = 1.0
        pyramid_mask = blending._apply_low_res_mask_dilation(exact_mask, 5)

        result = blending._pyramid_blend(
            generated,
            prepared,
            pyramid_mask,
            max_level=5,
            exact_image1_mask=exact_mask,
        )

        self.assertTrue(torch.equal(result[:, :, :2, :], generated[:, :, :2, :]))

    def test_outpaint_chunk_transcodes_opus_audio_to_aac(self) -> None:
        ffmpeg = outpaint_video.find_ffmpeg()
        ffprobe = str(Path(ffmpeg).with_name("ffprobe.exe" if Path(ffmpeg).suffix.lower() == ".exe" else "ffprobe"))
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source_with_opus.mkv"
            chunk = folder / "chunk.mp4"
            subprocess.run(
                [
                    ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=gray:s=96x64:r=8:d=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "libopus", "-shortest", str(source),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            outpaint_video.split_chunk(ffmpeg, source, chunk, 4, 12, 8.0, True)
            result = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1", str(chunk)],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.stdout.strip(), "aac")

    def test_qwen_seed_guides_do_not_overwrite_existing_set_guides(self) -> None:
        args = argparse.Namespace(
            comfy_output_root="",
            comfy_dir=str(app.ROOT),
            qwen_workflow="workflow.json",
            qwen_masked_workflow="masked.json",
            comfy_url="http://127.0.0.1:8188",
            qwen_model_backend="gguf",
            qwen_gguf_model="model.gguf",
            qwen_prompt="Replace the black bars.",
            qwen_load_image_node_id="auto",
            qwen_save_node_id="auto",
            seed_sample_seconds=0.0,
            seed_shot_threshold=None,
            seed_min_shot_seconds=None,
            guide_strength=0.7,
            force=False,
        )
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "chunks.csv"
            guide = folder / "existing.png"
            guide.write_bytes(b"guide")
            existing = [{"frame_idx": 0, "strength": 0.7, "image": app.rel(guide), "seed": True}]
            outpaint_video.write_chunk_manifest(
                manifest,
                [
                    {
                        "chunk_index": "0",
                        "start_frame": "0",
                        "end_frame": "24",
                        "guide_frames": json.dumps(existing),
                    }
                ],
            )

            with mock.patch.object(outpaint_video, "seed_guides", return_value={}) as seed_mock:
                rows = outpaint_video.apply_qwen_seed_guides(args, folder / "prepared.mp4", [(0, 0, 24)], manifest)

        self.assertEqual(json.loads(rows[0]["guide_frames"]), existing)
        self.assertEqual(seed_mock.call_args.kwargs["occupied_frame_idxs"], {0: {0}})

    def test_outpaint_command_uses_global_prompt(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "0", "section_end": ""})
        app.APP.settings["outpaint"].update(
            {
                "target_aspect": "16:9",
                "target_height": "720",
                "chunk_seconds": "20",
                "overlap_frames": "8",
                "prompt": "outpaint with restrained natural edges",
                "crop_left": "0",
                "crop_right": "0",
                "crop_top": "0",
                "crop_bottom": "0",
            }
        )

        command = app.APP.command_for("outpaint")

        self.assertEqual(command[command.index("--prompt") + 1], "outpaint with restrained natural edges")

    def test_outpaint_command_falls_back_to_activation_prompt_when_global_prompt_blank(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "0", "section_end": ""})
        app.APP.settings["outpaint"].update(
            {
                "target_aspect": "16:9",
                "target_height": "720",
                "chunk_seconds": "20",
                "overlap_frames": "8",
                "prompt": "",
                "crop_left": "0",
                "crop_right": "0",
                "crop_top": "0",
                "crop_bottom": "0",
            }
        )

        command = app.APP.command_for("outpaint")

        self.assertEqual(command[command.index("--prompt") + 1], "outpaint")

    def test_outpaint_prompt_combiner_adds_sentence_separator_for_chunk_suffix(self) -> None:
        self.assertEqual(outpaint_video.combine_prompt("outpaint", "continue the room"), "outpaint. continue the room")
        self.assertEqual(outpaint_video.combine_prompt("outpaint.", "continue the room"), "outpaint. continue the room")
        self.assertEqual(outpaint_video.combine_prompt("", "continue the room"), "continue the room")

    def test_outpaint_manifest_sync_preserves_chunk_prompt_suffixes(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "chunks.csv"
            outpaint_video.write_chunk_manifest(
                manifest,
                [
                    {
                        "chunk_index": "0",
                        "start_frame": "0",
                        "end_frame": "10",
                        "seed": "42",
                        "prompt_suffix": "stale per-chunk direction",
                    }
                ],
            )

            rows = outpaint_video.sync_chunk_manifest(manifest, [(0, 0, 10)], 24.0, folder, 42)

            self.assertEqual(rows[0]["prompt_suffix"], "stale per-chunk direction")

    def test_outpaint_manifest_sync_uses_offset_specific_chunk_paths(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "chunks.csv"
            outpaint_video.write_chunk_manifest(
                manifest,
                [
                    {
                        "chunk_index": "0",
                        "start_frame": "0",
                        "end_frame": "10",
                        "seed": "42",
                        "offset_x": "12",
                        "offset_y": "-4",
                    }
                ],
            )

            rows = outpaint_video.sync_chunk_manifest(manifest, [(0, 0, 10)], 24.0, folder, 42)

            self.assertIn("_ox+12_oy-4", rows[0]["prepared_path"])
            self.assertIn("_ox+12_oy-4", rows[0]["raw_path"])

    def test_outpaint_auto_start_guide_defaults_on_and_can_be_disabled(self) -> None:
        self.assertTrue(outpaint_video.auto_start_guide_enabled({}))
        self.assertTrue(outpaint_video.auto_start_guide_enabled({"auto_start_guide": "true"}))
        self.assertFalse(outpaint_video.auto_start_guide_enabled({"auto_start_guide": "false"}))

    def test_outpaint_manifest_sync_preserves_auto_start_guide_override(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "chunks.csv"
            outpaint_video.write_chunk_manifest(
                manifest,
                [
                    {
                        "chunk_index": "0",
                        "start_frame": "0",
                        "end_frame": "10",
                        "seed": "42",
                        "auto_start_guide": "false",
                    }
                ],
            )

            rows = outpaint_video.sync_chunk_manifest(manifest, [(0, 0, 10), (1, 8, 18)], 24.0, folder, 42)

            self.assertEqual(rows[0]["auto_start_guide"], "false")
            self.assertEqual(rows[1]["auto_start_guide"], "true")

    def test_colormnet_correlation_extension_install_is_opt_in(self) -> None:
        downloader_path = app.ROOT / "vendor" / "comfyui_custom_nodes" / "reference-video-colorization" / "colormnet" / "downloader.py"
        spec = importlib.util.spec_from_file_location("colormnet_downloader_under_test", downloader_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(module.optional_correlation_install_enabled())

        for value in ("1", "true", "yes", "on"):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"COLORMNET_INSTALL_CORRELATION_EXTENSION": value}, clear=True):
                self.assertTrue(module.optional_correlation_install_enabled())

        self.assertTrue(module.cuda_versions_match("13.0", "13.0"))
        self.assertFalse(module.cuda_versions_match("12.4", "13.0"))

    def test_single_reference_rejects_missing_source_before_qwen_startup(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            missing = Path(tmp_text) / "missing.png"
            output = Path(tmp_text) / "output.png"
            argv = [
                "generate_single_reference.py",
                "--source-image",
                str(missing),
                "--output",
                str(output),
                "--workflow",
                str(app.ROOT / "workflows" / "qwen_image_edit" / "Image Edit (Qwen 2511).json"),
                "--dry-run",
            ]

            with mock.patch.object(sys, "argv", argv), mock.patch.object(generate_single_reference.qwen, "main_with_args") as qwen_main:
                with self.assertRaisesRegex(FileNotFoundError, "Reference source image not found"):
                    generate_single_reference.main()

            qwen_main.assert_not_called()

    def test_bundled_qwen_2511_subgraph_patches_to_gguf_by_default(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.png"
            source.write_bytes(b"placeholder")
            comfy_dir = folder / "comfy"
            (comfy_dir / "input").mkdir(parents=True)
            workflow = qwen_colorize_references.load_workflow(app.ROOT / "workflows" / "qwen_image_edit" / "Image Edit (Qwen 2511).json")
            args = argparse.Namespace(comfy_dir=comfy_dir, model_backend="gguf", gguf_model="qwen-image-edit-2511-Q4_K_M.gguf")

            qwen_colorize_references.patch_qwen_model_backend(args, workflow)
            prompt = qwen_colorize_references.patch_workflow(args, workflow, source, folder / "output.png", "Colorize this image.")

            self.assertEqual(prompt["169"]["inputs"]["seed"], 1)
            self.assertEqual(prompt["169"]["inputs"]["control_after_generate"], "fixed")
            self.assertEqual(prompt["161"]["class_type"], "UnetLoaderGGUF")
            self.assertEqual(prompt["161"]["inputs"]["unet_name"], "qwen-image-edit-2511-Q4_K_M.gguf")

    def test_qwen_completion_copies_produced_image_to_requested_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.png"
            output = folder / "output.png"
            produced = folder / "comfy-output.png"
            workflow = folder / "workflow.json"
            manifest = folder / "manifest.csv"
            comfy_dir = folder / "comfy"
            comfy_output = folder / "comfy-output"
            source.write_bytes(b"source image bytes")
            produced.write_bytes(b"produced image bytes")
            workflow.write_text("{}", encoding="utf-8")
            comfy_dir.mkdir()
            comfy_output.mkdir()
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["source_reference", "color_reference"])
                writer.writeheader()
                writer.writerow({"source_reference": str(source), "color_reference": str(output)})

            args = qwen_colorize_references.build_parser().parse_args(
                [
                    "--manifest",
                    str(manifest),
                    "--workflow",
                    str(workflow),
                    "--comfy-dir",
                    str(comfy_dir),
                    "--comfy-output-root",
                    str(comfy_output),
                    "--model-backend",
                    "safetensors",
                    "--no-normalize-to-source-size",
                    "--force",
                ]
            )

            with (
                mock.patch.object(qwen_colorize_references, "ensure_qwen_image_edit_models"),
                mock.patch.object(qwen_colorize_references, "wait_for_comfy"),
                mock.patch.object(qwen_colorize_references, "patch_qwen_model_backend"),
                mock.patch.object(qwen_colorize_references, "patch_workflow", return_value={}),
                mock.patch.object(qwen_colorize_references, "queue_prompt", return_value="prompt-id"),
                mock.patch.object(qwen_colorize_references, "wait_for_prompt", return_value={}),
                mock.patch.object(qwen_colorize_references, "extract_output_files", return_value=[produced]),
                mock.patch.object(qwen_colorize_references, "newest_output", return_value=produced),
            ):
                self.assertEqual(qwen_colorize_references.main_with_args(args), 0)

            self.assertEqual(output.read_bytes(), b"produced image bytes")

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

    def test_outpaint_chunk_rows_do_not_rewrite_identical_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "chunks.csv"
            rows = [{"chunk_index": "0", "start_frame": "0", "end_frame": "10", "seed": "42"}]

            app.write_outpaint_chunk_rows(manifest, rows)
            first_mtime = manifest.stat().st_mtime_ns
            app.write_outpaint_chunk_rows(manifest, rows)

            self.assertEqual(manifest.stat().st_mtime_ns, first_mtime)

    def test_outpaint_chunk_save_can_clear_custom_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "chunks.csv"
            rows = [
                {
                    "chunk_index": "0",
                    "start_frame": "0",
                    "end_frame": "120",
                    "start_seconds": "0.000000",
                    "end_seconds": "5.000000",
                    "custom_seconds": "5.000",
                    "seed": "42",
                }
            ]
            app.write_outpaint_chunk_rows(manifest, rows)

            with mock.patch.object(server, "outpaint_chunks_state", return_value={"manifest": str(manifest)}):
                app.update_outpaint_chunk(
                    0,
                    seed="43",
                    prompt_suffix="",
                    custom_seconds="5.000",
                    custom_length=False,
                )

            stored = app.read_outpaint_chunk_rows(manifest)[0]
            self.assertEqual(stored["seed"], "43")
            self.assertEqual(stored["custom_seconds"], "")

    def test_outpaint_chunk_save_persists_prompt_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "chunks.csv"
            rows = [
                {
                    "chunk_index": "0",
                    "start_frame": "0",
                    "end_frame": "120",
                    "start_seconds": "0.000000",
                    "end_seconds": "5.000000",
                    "seed": "42",
                    "prompt_suffix": "",
                    "negative_suffix": "",
                }
            ]
            app.write_outpaint_chunk_rows(manifest, rows)

            with mock.patch.object(server, "outpaint_chunks_state", return_value={"manifest": str(manifest)}):
                app.update_outpaint_chunk(
                    0,
                    seed="43",
                    prompt_suffix="avoid changing the actor's hands",
                    negative_suffix="extra fingers",
                )

            stored = app.read_outpaint_chunk_rows(manifest)[0]
            self.assertEqual(stored["prompt_suffix"], "avoid changing the actor's hands")
            self.assertEqual(stored["negative_suffix"], "extra fingers")

    def test_outpaint_chunk_save_persists_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "chunks.csv"
            rows = [{"chunk_index": "0", "start_frame": "0", "end_frame": "120", "seed": "42"}]
            app.write_outpaint_chunk_rows(manifest, rows)

            with mock.patch.object(server, "outpaint_chunks_state", return_value={"manifest": str(manifest)}):
                app.update_outpaint_chunk(0, seed="42", prompt_suffix="", offset_x="-11", offset_y="7")

            stored = app.read_outpaint_chunk_rows(manifest)[0]
            self.assertEqual(stored["offset_x"], "-11")
            self.assertEqual(stored["offset_y"], "7")

    def test_outpaint_chunk_preview_passes_chunk_offsets(self) -> None:
        settings = {"outpaint": {"target_aspect": "16:9"}}
        fake_state = {
            "rows": [
                {
                    "index": 0,
                    "start": 1.0,
                    "end": 2.0,
                    "fps": 24.0,
                    "offset_x": "13",
                    "offset_y": "-5",
                    "raw_path": "",
                }
            ]
        }
        with (
            mock.patch.object(server, "outpaint_chunks_state", return_value=fake_state),
            mock.patch.object(server, "pipeline_source_text", return_value="input/example.mp4"),
            mock.patch.object(server, "aspect_preview_at", return_value="preview.jpg") as preview,
        ):
            self.assertEqual(app.outpaint_chunk_preview(settings, 0, "source", "middle"), "preview.jpg")

        preview.assert_called_once_with("input/example.mp4", "16:9", 1.5, 13, -5)


    def test_outpaint_chunk_state_reports_exact_prompts_sent_to_comfy(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            source.write_bytes(b"video")
            manifest = folder / "chunks.csv"
            app.write_outpaint_chunk_rows(
                manifest,
                [
                    {
                        "chunk_index": "0",
                        "start_frame": "0",
                        "end_frame": "24",
                        "seed": "42",
                        "prompt_suffix": "extend the wallpaper",
                        "negative_suffix": "extra hands",
                    }
                ],
            )
            settings = {
                "global": {"source": app.rel(source), "section_start": "0", "section_end": ""},
                "outpaint": {
                    "target_aspect": "16:9",
                    "target_height": "720",
                    "chunk_seconds": "1",
                    "overlap_frames": "0",
                    "prompt": "outpaint with natural edges",
                    "negative_prompt": "text",
                },
            }
            with (
                mock.patch.object(server, "ensure_source_section_clip"),
                mock.patch.object(server, "resolve_video_source", return_value=source),
                mock.patch.object(server, "video_metrics", return_value={"fps": 24.0, "frames": 24}),
                mock.patch.object(server, "outpaint_chunk_manifest_for", return_value=app.rel(manifest)),
                mock.patch.object(server, "outpaint_chunk_dir_for", return_value=folder),
            ):
                state = app.outpaint_chunks_state(settings)
                with mock.patch.object(server, "write_outpaint_chunk_rows") as rewrite:
                    app.outpaint_chunks_state(settings)

        row = state["rows"][0]
        self.assertEqual(row["effective_prompt"], "outpaint with natural edges. extend the wallpaper")
        self.assertEqual(row["effective_negative_prompt"], "text. extra hands")
        rewrite.assert_not_called()


    def test_clearing_outpaint_guide_deletes_cached_chunk_guides(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "demo_chunks.csv"
            guide_dir = app.ROOT / "intermediate" / "outpaint_anchors" / manifest.stem
            guide_dir.mkdir(parents=True, exist_ok=True)
            current = guide_dir / "chunk_0000_guide_qwen.png"
            older = guide_dir / "chunk_0000_middle_qwen.png"
            other = guide_dir / "chunk_0001_guide_qwen.png"
            for path in (current, older, other, current.with_suffix(".png.sig.json")):
                path.write_bytes(b"cached")
            rows = [
                {
                    "chunk_index": "0",
                    "start_frame": "0",
                    "end_frame": "10",
                    "start_seconds": "0",
                    "end_seconds": "1",
                    "seed": "42",
                    "anchor_image": app.rel(current),
                    "anchor_position": "guide",
                    "anchor_seconds": "0.5",
                }
            ]
            app.write_outpaint_chunk_rows(manifest, rows)

            try:
                with mock.patch.object(server, "outpaint_chunks_state", return_value={"manifest": str(manifest)}):
                    app.clear_outpaint_anchor(0)

                self.assertFalse(current.exists())
                self.assertFalse(older.exists())
                self.assertFalse(current.with_suffix(".png.sig.json").exists())
                self.assertTrue(other.exists())
                self.assertEqual(app.read_outpaint_chunk_rows(manifest)[0]["anchor_image"], "")
            finally:
                shutil.rmtree(guide_dir, ignore_errors=True)

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

    def test_shot_rows_marks_only_active_edit_reference_as_edited(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / f"{folder.name}_refs.csv"
            original = folder / "color.png"
            source = folder / "bw.png"
            original.write_bytes(b"original")
            source.write_bytes(b"source")
            edit_dir = app.ROOT / "intermediate" / "outpainted_references_color_edits" / app.safe_stem(app.resolve(manifest).stem) / "shot_0000"
            edit_dir.mkdir(parents=True, exist_ok=True)
            edit = edit_dir / "edit_20260619_120000.png"
            edit.write_bytes(b"edited")
            try:
                app.write_manifest_details(
                    manifest,
                    "input/example.mp4",
                    ["enabled", "end", "source_reference", "color_reference", "color_reference_previous"],
                    [
                        {
                            "enabled": "true",
                            "end": "00:00:01.000",
                            "source_reference": app.rel(source),
                            "color_reference": app.rel(original),
                            "color_reference_previous": app.rel(edit),
                        }
                    ],
                )

                normal = app.shot_rows(str(manifest))[0]
                app.update_manifest_row(manifest, 0, {"color_reference": app.rel(edit), "color_reference_previous": app.rel(original)})
                edited = app.shot_rows(str(manifest))[0]
            finally:
                shutil.rmtree(edit_dir.parent, ignore_errors=True)

        self.assertTrue(normal["color_reference_versions"])
        self.assertFalse(normal["color_reference_edited"])
        self.assertTrue(edited["color_reference_edited"])

    def test_reference_paint_save_installs_painted_image_with_revert_history(self) -> None:
        import base64

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "paint_smoke_refs.csv"
            original = folder / "color.png"
            original.write_bytes(b"original image bytes")
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["enabled", "end", "source_reference", "color_reference", "prompt"])
                writer.writeheader()
                writer.writerow(
                    {
                        "enabled": "true",
                        "end": "00:00:01.000",
                        "source_reference": "bw.png",
                        "color_reference": app.rel(original),
                        "prompt": "",
                    }
                )

            painted = b"\x89PNG\r\n\x1a\npainted pixels"
            data_url = "data:image/png;base64," + base64.b64encode(painted).decode("ascii")
            edit_root = app.ROOT / "intermediate" / "outpainted_references_color_edits" / "paint_smoke_refs"
            try:
                result = app.save_reference_paint(str(manifest), 0, data_url)

                installed = app.resolve(result["color_reference"])
                self.assertEqual(installed.read_bytes(), painted)
                _source, _fields, rows = app.read_manifest_details(manifest)
                self.assertEqual(rows[0]["color_reference"], result["color_reference"])
                self.assertEqual(rows[0]["color_reference_previous"], app.rel(original))
            finally:
                shutil.rmtree(edit_root, ignore_errors=True)

    def test_guide_paint_save_installs_painted_guide_image(self) -> None:
        import base64

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "paint_smoke_chunks.csv"
            guides = [{"frame_idx": 0, "strength": 0.7, "image": ""}]
            outpaint_video.write_chunk_manifest(
                manifest,
                [
                    {
                        "chunk_index": "0",
                        "start_frame": "0",
                        "end_frame": "24",
                        "guide_frames": json.dumps(guides),
                    }
                ],
            )

            painted = b"\x89PNG\r\n\x1a\npainted guide pixels"
            data_url = "data:image/png;base64," + base64.b64encode(painted).decode("ascii")
            edit_root = app.ROOT / "intermediate" / "outpaint_guides" / "paint_smoke_chunks"
            try:
                with mock.patch.object(outpaint_guides, "outpaint_chunks_state", return_value={"manifest": str(manifest)}):
                    result = outpaint_guides.save_guide_paint(0, 0, data_url)

                installed = app.resolve(result["image"])
                self.assertEqual(installed.read_bytes(), painted)
                stored = app.read_outpaint_chunk_rows(manifest)[0]
                frames = outpaint_guides._parse_guide_frames(stored)
                self.assertEqual(frames[0]["image"], result["image"])
                self.assertEqual(frames[0]["image_previous"], "")
            finally:
                shutil.rmtree(edit_root, ignore_errors=True)

    def test_shot_fade_marker_round_trips_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "refs.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["enabled", "end", "source_reference", "color_reference", "prompt"])
                writer.writeheader()
                writer.writerow({"enabled": "true", "end": "00:00:01.000", "source_reference": "a.png", "color_reference": "a_color.png", "prompt": ""})
                writer.writerow({"enabled": "true", "end": "00:00:02.000", "source_reference": "b.png", "color_reference": "b_color.png", "prompt": ""})

            app.update_shot_fade(str(manifest), 0, True, "0.5")
            _source, fields, rows = app.read_manifest_details(manifest)

            self.assertIn("fade_to_next", fields)
            self.assertEqual(rows[0]["fade_to_next"], "true")
            self.assertEqual(rows[0]["crossfade_seconds"], "0.5")

    def test_colorize_plan_extends_fading_transition_chunks(self) -> None:
        rows = [
            {"end": "00:00:01.000", "fade_to_next": "true", "crossfade_seconds": "0.5"},
            {"end": "00:00:02.000"},
        ]

        plan, transitions = colorize_video.shot_plan(rows, total_frames=48, fps=24.0)

        self.assertEqual(transitions[0], 12)
        self.assertEqual(plan[0]["start"], 0)
        self.assertEqual(plan[0]["end"], 30)
        self.assertEqual(plan[1]["start"], 18)
        self.assertEqual(plan[1]["end"], 48)

    def test_shot_manifest_writes_integer_frame_spans(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            manifest = folder / "shots.csv"
            info = generate_references.VideoInfo(width=16, height=9, fps=24.0, frame_count=20, duration=20 / 24.0)
            rows = [
                generate_references.ReferenceRow(0, 0, 7, 3, 3 / 24.0, folder / "a.png", folder / "a_color.png"),
                generate_references.ReferenceRow(1, 7, 20, 9, 9 / 24.0, folder / "b.png", folder / "b_color.png"),
            ]

            generate_references.write_manifest(manifest, source, rows, info)
            _source, fields, read_rows = app.read_manifest_details(manifest)

        self.assertIn("start_frame", fields)
        self.assertIn("end_frame", fields)
        self.assertIn("selected_frame", fields)
        self.assertEqual(read_rows[0]["start_frame"], "0")
        self.assertEqual(read_rows[0]["end_frame"], "7")
        self.assertEqual(read_rows[1]["start_frame"], "7")

    def test_shot_rows_prefer_manifest_frames_over_rounded_seconds(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            manifest = Path(tmp_text) / "shots.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["enabled", "start_frame", "end_frame", "selected_frame", "end", "source_reference", "color_reference"],
                )
                writer.writeheader()
                writer.writerow({"enabled": "true", "start_frame": "0", "end_frame": "7", "selected_frame": "3", "end": "0.250"})
                writer.writerow({"enabled": "true", "start_frame": "7", "end_frame": "20", "selected_frame": "9", "end": "0.833"})

            rows = app.shot_rows(str(manifest))

        self.assertEqual(rows[0]["start_frame"], 0)
        self.assertEqual(rows[0]["end_boundary_frame"], 7)
        self.assertEqual(rows[1]["start_frame"], 7)
        self.assertEqual(rows[1]["selected_frame"], 9)

    def test_colorize_plan_prefers_manifest_frames_over_rounded_seconds(self) -> None:
        rows = [
            {"start_frame": "0", "end_frame": "7", "end": "0.250"},
            {"start_frame": "7", "end_frame": "20", "end": "0.833"},
        ]

        plan, transitions = colorize_video.shot_plan(rows, total_frames=20, fps=24.0)

        self.assertEqual(transitions, [0, 0])
        self.assertEqual(plan[0]["base_start"], 0)
        self.assertEqual(plan[0]["base_end"], 7)
        self.assertEqual(plan[1]["base_start"], 7)
        self.assertEqual(plan[1]["base_end"], 20)

    def test_colorize_segment_signature_tracks_full_shot_inputs_for_each_method(self) -> None:
        def args_for(method: str) -> argparse.Namespace:
            return argparse.Namespace(
                method=method,
                frame_propagate=True,
                use_half_resolution=True,
                use_torch_compile=False,
                use_sage_attention=False,
                colormnet_memory_mode="balanced",
                colormnet_feature_encoder="resnet50",
                colormnet_text_guidance="",
                colormnet_text_guidance_weight=0.3,
                video_format="video/h264-mp4",
                crf=18,
            )

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            source_reference = folder / "source.png"
            color_reference = folder / "color.png"
            source.write_bytes(b"video bytes")
            source_reference.write_bytes(b"source reference")
            color_reference.write_bytes(b"color reference")
            row = {
                "enabled": "true",
                "start_frame": "0",
                "end_frame": "10",
                "selected_frame": "4",
                "end": "00:00:00.417",
                "source_reference": app.rel(source_reference),
                "color_reference": app.rel(color_reference),
                "prompt": "",
                "fade_to_next": "false",
                "crossfade_seconds": "",
            }

            deep = colorize_video.segment_signature(args_for("deepexemplar"), source, row, color_reference, 0, 10, 0, 10, 640, 360, 24.0)
            color = colorize_video.segment_signature(args_for("colormnet"), source, row, color_reference, 0, 10, 0, 10, 640, 360, 24.0)
            prompted = dict(row)
            prompted["prompt"] = "warmer faces"
            deep_prompted = colorize_video.segment_signature(args_for("deepexemplar"), source, prompted, color_reference, 0, 10, 0, 10, 640, 360, 24.0)
            color_prompted = colorize_video.segment_signature(args_for("colormnet"), source, prompted, color_reference, 0, 10, 0, 10, 640, 360, 24.0)
            with_previous = dict(row)
            with_previous["color_reference_previous"] = ""
            deep_with_previous = colorize_video.segment_signature(args_for("deepexemplar"), source, with_previous, color_reference, 0, 10, 0, 10, 640, 360, 24.0)
            color_with_previous = colorize_video.segment_signature(args_for("colormnet"), source, with_previous, color_reference, 0, 10, 0, 10, 640, 360, 24.0)
            color_reference.write_bytes(b"new color reference")
            deep_changed_ref = colorize_video.segment_signature(args_for("deepexemplar"), source, row, color_reference, 0, 10, 0, 10, 640, 360, 24.0)
            color_changed_ref = colorize_video.segment_signature(args_for("colormnet"), source, row, color_reference, 0, 10, 0, 10, 640, 360, 24.0)

        self.assertEqual(deep["shot_input"], color["shot_input"])
        self.assertNotEqual(deep["shot_input"], deep_prompted["shot_input"])
        self.assertNotEqual(color["shot_input"], color_prompted["shot_input"])
        self.assertEqual(deep["shot_input"], deep_with_previous["shot_input"])
        self.assertEqual(color["shot_input"], color_with_previous["shot_input"])
        self.assertNotEqual(deep["shot_input"], deep_changed_ref["shot_input"])
        self.assertNotEqual(color["shot_input"], color_changed_ref["shot_input"])

    def test_colorize_crossfade_rebuilds_frame_timestamps_before_fps(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            chunks = [folder / "a.mp4", folder / "b.mp4"]
            for chunk in chunks:
                chunk.write_bytes(b"placeholder")
            output = folder / "xfade.mp4"

            with (
                mock.patch.object(colorize_video, "video_info", return_value={"frames": 24}),
                mock.patch.object(colorize_video.subprocess, "run") as run,
                mock.patch.object(colorize_video, "replace_with_retry"),
            ):
                colorize_video.xfade_group("ffmpeg", chunks, [6], output, 24.0)

        command = run.call_args.args[0]
        filter_text = command[command.index("-filter_complex") + 1]
        self.assertIn("[0:v]setpts=N/(24.00000000*TB),fps=fps=24.00000000[v0]", filter_text)
        self.assertIn("[1:v]setpts=N/(24.00000000*TB),fps=fps=24.00000000[v1]", filter_text)

    def test_outpaint_overlap_context_stops_before_guide_inside_overlap(self) -> None:
        self.assertEqual(outpaint_video.overlap_context_before_anchor(8, "0.125", 24.0, 100), 3)
        self.assertEqual(outpaint_video.overlap_context_before_anchor(8, "1.0", 24.0, 100), 8)
        self.assertEqual(outpaint_video.overlap_context_before_anchor(8, "0", 24.0, 100), 0)

    def test_outpaint_overlap_context_rejects_mixed_geometry(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            chunk = folder / "chunk.mp4"
            previous = folder / "previous.mp4"

            with (
                mock.patch.object(outpaint_video, "probe_video") as probe,
                mock.patch.object(outpaint_video.subprocess, "run") as run,
                mock.patch.object(outpaint_video, "replace_with_retry") as replace,
            ):
                probe.side_effect = [
                    {"width": 1280, "height": 720, "frames": 8},
                    {"width": 1280, "height": 704, "frames": 8},
                ]
                previous.write_bytes(b"placeholder")

                with self.assertRaisesRegex(RuntimeError, "working canvas should stay"):
                    outpaint_video.inject_overlap_context("ffmpeg", chunk, previous, 3, 24.0, False)

            run.assert_not_called()
            replace.assert_not_called()

    def test_outpaint_overlap_context_skips_first_new_frame_on_matching_geometry(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            chunk = folder / "chunk.mp4"
            previous = folder / "previous.mp4"

            with (
                mock.patch.object(outpaint_video, "probe_video") as probe,
                mock.patch.object(outpaint_video.subprocess, "run") as run,
                mock.patch.object(outpaint_video, "replace_with_retry") as replace,
            ):
                probe.side_effect = [
                    {"width": 1280, "height": 704, "frames": 8},
                    {"width": 1280, "height": 704, "frames": 8},
                ]
                previous.write_bytes(b"placeholder")

                result = outpaint_video.inject_overlap_context("ffmpeg", chunk, previous, 3, 24.0, False)

            self.assertNotEqual(result, chunk)
            filter_text = run.call_args.args[0][7]
            self.assertNotIn("scale=1280:720", filter_text)
            self.assertIn("trim=start_frame=4:end_frame=8", filter_text)
            replace.assert_called_once()

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

    def test_outpaint_stage_defaults_to_longer_chunks(self) -> None:
        outpaint_stage = next(stage for stage in app.STAGES if stage.key == "outpaint")
        chunk_field = next(field for field in outpaint_stage.fields if field[0] == "chunk_seconds")

        self.assertEqual(chunk_field[3], "20")

    def test_comfy_node_check_reports_missing_custom_nodes(self) -> None:
        with mock.patch.object(comfy_api, "object_info", return_value={"UnetLoaderGGUF": {}}):
            with self.assertRaisesRegex(RuntimeError, "ComfyUI-LTXVideo"):
                comfy_api.ensure_node_types(
                    "http://127.0.0.1:8188",
                    {"LTXVImgToVideoConditionOnly": "ComfyUI-LTXVideo", "UnetLoaderGGUF": "ComfyUI-GGUF"},
                    "outpainting workflow",
                )

    def test_mmaudio_graph_selects_distinct_model_files(self) -> None:
        model_files = [
            "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors",
            "mmaudio_large_44k_v2_fp16.safetensors",
            "mmaudio_synchformer_fp16.safetensors",
            "mmaudio_vae_44k_fp16.safetensors",
        ]
        info = {
            "MMAudioModelLoader": {
                "input": {
                    "required": {
                        "mmaudio_model": (model_files, {}),
                        "base_precision": (["fp16", "fp32"], {"default": "fp16"}),
                    },
                },
            },
            "MMAudioFeatureUtilsLoader": {
                "input": {
                    "required": {
                        "vae_model": (model_files, {}),
                        "synchformer_model": (model_files, {}),
                        "clip_model": (model_files, {}),
                    },
                    "optional": {
                        "mode": (["16k", "44k"], {"default": "44k"}),
                        "precision": (["fp16", "fp32"], {"default": "fp16"}),
                    },
                },
            },
            "MMAudioSampler": {"input": {"required": {"images": ("IMAGE",)}}},
        }

        graph = audio_models.sfx_prompt_graph(
            info,
            video_name="proxy.mp4",
            prompt="machinery",
            negative="music",
            seconds=8,
            steps=25,
            cfg=4.5,
            seed=42,
            prefix="arp_audio_sfx/test",
        )

        self.assertEqual(graph["1"]["inputs"]["mmaudio_model"], "mmaudio_large_44k_v2_fp16.safetensors")
        self.assertEqual(graph["2"]["inputs"]["vae_model"], "mmaudio_vae_44k_fp16.safetensors")
        self.assertEqual(graph["2"]["inputs"]["synchformer_model"], "mmaudio_synchformer_fp16.safetensors")
        self.assertEqual(graph["2"]["inputs"]["clip_model"], "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors")

    def test_music_checkpoint_validation_reports_available_choices(self) -> None:
        info = {
            "CheckpointLoaderSimple": {
                "input": {
                    "required": {
                        "ckpt_name": (["ltx-2.3-22b-dev-fp8.safetensors"], {}),
                    },
                },
            },
        }

        with self.assertRaisesRegex(RuntimeError, "stable_audio_open_1.0.safetensors"):
            audio_models.ensure_checkpoint_choice(info, "stable_audio_open_1.0.safetensors")

    def test_stable_audio_music_graph_loads_separate_t5_encoder(self) -> None:
        info = {
            "KSampler": {
                "input": {
                    "required": {
                        "sampler_name": (["euler"], {}),
                        "scheduler": (["simple"], {}),
                    },
                },
            },
            "EmptyLatentAudio": {"input": {"required": {"seconds": ("FLOAT", {"default": 30.0}), "batch_size": ("INT", {"default": 1})}}},
        }

        graph = audio_models.music_prompt_graph(
            info,
            checkpoint="stable_audio_open_1.0.safetensors",
            text_encoder="t5_base.safetensors",
            prompt="orchestral",
            negative="noise",
            seconds=12.0,
            steps=20,
            cfg=6.0,
            seed=42,
            prefix="arp_audio/music_test",
        )

        self.assertEqual(graph["2"]["class_type"], "CLIPLoader")
        self.assertEqual(graph["2"]["inputs"]["clip_name"], "t5_base.safetensors")
        self.assertEqual(graph["2"]["inputs"]["type"], "stable_audio")
        self.assertEqual(graph["5"]["class_type"], "ConditioningStableAudio")
        self.assertEqual(graph["7"]["inputs"]["positive"], ["5", 0])
        self.assertEqual(graph["7"]["inputs"]["negative"], ["5", 1])
        self.assertEqual(graph["8"]["inputs"]["samples"], ["7", 0])

    def test_music_checkpoint_file_preflight_reports_gated_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            with self.assertRaisesRegex(FileNotFoundError, "stabilityai/stable-audio-open-1.0"):
                create_audio_track.ensure_music_checkpoint_file(Path(tmp_text), "stable_audio_open_1.0.safetensors")

    def test_audio_music_missing_checkpoint_opens_huggingface_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            source = Path(tmp_text) / "source.mp4"
            comfy = Path(tmp_text) / "ComfyUI"
            source.write_bytes(b"placeholder")
            app.APP.settings["global"].update(
                {
                    "source": str(source),
                    "expand_outpaint": "false",
                    "colorize": "false",
                    "add_soundtrack": "true",
                    "section_start": "0",
                    "section_end": "",
                }
            )
            app.APP.settings["audio"].update(
                {
                    "create_music": "true",
                    "create_sfx": "false",
                    "music_checkpoint": "stable_audio_open_1.0.safetensors",
                }
            )

            with (
                mock.patch.object(server, "current_config", return_value={"comfy_dir": str(comfy)}),
                mock.patch.object(server, "ROOT", Path(tmp_text)),
                mock.patch.object(server.webbrowser, "open", return_value=True) as open_browser,
                mock.patch.object(server, "ensure_comfy_available_for_stage") as ensure_comfy,
            ):
                ok, message = app.APP.run_stage("audio")

        self.assertFalse(ok)
        self.assertIn("Hugging Face license acceptance", message)
        open_browser.assert_called_once_with(server.STABLE_AUDIO_LICENSE_URL)
        ensure_comfy.assert_not_called()

    def test_audio_music_second_click_after_handoff_attempts_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            root = Path(tmp_text)
            source = root / "source.mp4"
            comfy = root / "ComfyUI"
            source.write_bytes(b"placeholder")
            app.APP.settings["global"].update(
                {
                    "source": str(source),
                    "expand_outpaint": "false",
                    "colorize": "false",
                    "add_soundtrack": "true",
                    "section_start": "0",
                    "section_end": "",
                }
            )
            app.APP.settings["audio"].update(
                {
                    "create_music": "true",
                    "create_sfx": "false",
                    "music_checkpoint": "stable_audio_open_1.0.safetensors",
                }
            )

            with (
                mock.patch.object(server, "current_config", return_value={"comfy_dir": str(comfy)}),
                mock.patch.object(server, "ROOT", root),
                mock.patch.object(server.webbrowser, "open", return_value=True) as open_browser,
            ):
                marker = server.stable_audio_handoff_marker_path("stable_audio_open_1.0.safetensors")
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("{}", encoding="utf-8")
                ok, message = server.stable_audio_browser_handoff("stable_audio_open_1.0.safetensors")

        self.assertTrue(ok)
        self.assertEqual(message, "")
        open_browser.assert_not_called()

    def test_outpaint_missing_official_lora_opens_huggingface_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            root = Path(tmp_text)
            comfy = root / "ComfyUI"
            comfy.mkdir()
            with (
                mock.patch.object(server, "current_config", return_value={"comfy_dir": str(comfy)}),
                mock.patch.object(server, "ROOT", root),
                mock.patch.object(server.webbrowser, "open", return_value=True) as open_browser,
            ):
                ok, message = server.outpaint_browser_handoff()

        self.assertFalse(ok)
        self.assertIn("Approve the official LTX outpainting model download", message)
        self.assertIn("opened the approval page", message)
        open_browser.assert_called_once_with(server.OUTPAINT_LICENSE_URL)

    def test_outpaint_403_reopens_approval_page_with_short_message(self) -> None:
        error = outpaint_video.HuggingFaceAccessError("long recovery instructions")
        with mock.patch.object(outpaint_video.webbrowser, "open", return_value=True) as open_browser:
            message = outpaint_video.outpaint_access_error_message(error)

        self.assertEqual(
            message,
            "Approve the official LTX outpainting model download in the Hugging Face tab ARP just opened, then run Outpainting again.",
        )
        open_browser.assert_called_once_with(outpaint_video.OUTPAINT_ACCESS_URL)

    def test_audio_state_exposes_lossless_stems(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            root = Path(tmp_text)
            source = root / "input" / "Silent Movie.mp4"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"not a real video")
            stem_dir = root / ".cache" / "audio" / "Silent_Movie"
            stem_dir.mkdir(parents=True)
            (stem_dir / "music_stem.wav").write_bytes(b"music")
            (stem_dir / "sfx_stem.wav").write_bytes(b"sfx")

            app.APP.settings["global"].update(
                {
                    "source": str(source),
                    "expand_outpaint": "false",
                    "colorize": "false",
                    "add_soundtrack": "true",
                    "section_start": "0",
                    "section_end": "",
                }
            )
            with mock.patch.object(server, "ROOT", root):
                payload = app.APP.state("audio")

            stems = {row["key"]: row for row in payload["audio_stems"]}
            self.assertEqual(set(stems), {"music", "sfx", "mixed"})
            self.assertTrue(stems["music"]["exists"])
            self.assertTrue(stems["sfx"]["exists"])
            self.assertFalse(stems["mixed"]["exists"])
            self.assertEqual(stems["music"]["path"], ".cache/audio/Silent_Movie/music_stem.wav")
            self.assertEqual(stems["sfx"]["path"], ".cache/audio/Silent_Movie/sfx_stem.wav")

    def test_source_section_names_include_trim_points(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "12", "section_end": "24"})

        first = app.source_section_output_for(app.APP.settings)
        app.APP.settings["global"].update({"section_start": "45", "section_end": "60"})
        second = app.source_section_output_for(app.APP.settings)

        self.assertNotEqual(first, second)
        self.assertIn("0000012000_0000024000", first.name)
        self.assertIn("0000045000_0000060000", second.name)

    def test_webm_source_section_uses_mp4_container_for_h264_aac(self) -> None:
        app.APP.settings["global"].update(
            {"source": "input/example.webm", "section_start": "12", "section_end": "24"}
        )

        output = app.source_section_output_for(app.APP.settings)

        self.assertEqual(output.suffix, ".mp4")

    def test_pipeline_source_uses_section_when_trim_points_are_set(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "12", "section_end": "24"})

        self.assertIn("source_sections", app.pipeline_source_text(app.APP.settings))

    def test_existing_source_section_with_late_first_pts_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            clip = Path(tmp_text) / "section.mp4"
            clip.write_bytes(b"placeholder")

            with mock.patch.object(media, "source_section_timing", return_value={"first_pts": 5.249, "duration": 68.351}):
                self.assertFalse(media.source_section_clip_is_valid(clip, "ffmpeg", 63.062))

    def test_source_section_trim_resets_video_and_audio_timestamps(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            root = Path(tmp_text)
            source = root / "input" / "example.mkv"
            source.parent.mkdir()
            source.write_bytes(b"source")
            settings = {
                "global": {
                    "source": str(source),
                    "section_start": "12",
                    "section_end": "24",
                }
            }
            ffmpeg_commands: list[list[str]] = []

            def fake_run(command, **_kwargs):
                if command[0] == "ffprobe":
                    return subprocess.CompletedProcess(command, 0, stdout='{"streams":[{"index":1}]}', stderr="")
                ffmpeg_commands.append(command)
                Path(command[-1]).write_bytes(b"trimmed")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with (
                mock.patch.object(media, "ROOT", root),
                mock.patch.object(media, "local_tool", return_value="ffmpeg"),
                mock.patch.object(media.subprocess, "run", side_effect=fake_run),
            ):
                output = media.ensure_source_section_clip(settings)

            self.assertIn("intermediate/source_sections/example_", output)
            self.assertEqual(len(ffmpeg_commands), 1)
            command = ffmpeg_commands[0]
            self.assertIn("-vf", command)
            self.assertEqual(command[command.index("-vf") + 1], "setpts=PTS-STARTPTS")
            self.assertIn("-af", command)
            self.assertEqual(command[command.index("-af") + 1], "asetpts=PTS-STARTPTS")
            self.assertEqual(command[command.index("-c:a") + 1], "aac")
            self.assertNotIn("copy", command)

    def test_opening_source_resets_trim_to_source_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            source = Path(tmp_text) / "new-source.mp4"
            source.write_bytes(b"placeholder")
            app.APP.settings["global"].update({"source": "input/old.mp4", "section_start": "10", "section_end": "20"})

            with (
                mock.patch.object(server, "video_metrics", return_value={"duration": 123.456}),
                mock.patch.object(server, "ffprobe_basic_info", return_value={"resolution": "1920x1080"}),
            ):
                app.APP.update_settings("global", {"source": str(source)})

        self.assertEqual(app.APP.settings["global"]["section_start"], "0")
        self.assertEqual(app.APP.settings["global"]["section_end"], "123.456")

    def test_opening_squareish_sd_source_enables_outpaint_and_upscale_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            source = Path(tmp_text) / "new-source.mp4"
            source.write_bytes(b"placeholder")
            app.APP.settings["global"].update({"source": "input/old.mp4", "expand_outpaint": "false", "colorize": "false", "upscale": "false"})

            with (
                mock.patch.object(server, "video_metrics", return_value={"duration": 60.0}),
                mock.patch.object(server, "ffprobe_basic_info", return_value={"resolution": "960x720"}),
            ):
                app.APP.update_settings("global", {"source": str(source)})

        self.assertEqual(app.APP.settings["global"]["expand_outpaint"], "true")
        self.assertEqual(app.APP.settings["global"]["upscale"], "true")
        self.assertEqual(app.APP.settings["outpaint"]["target_aspect"], "16:9")
        self.assertEqual(app.APP.settings["outpaint"]["target_height"], "source")
        self.assertEqual(app.APP.settings["outpaint"]["seed_qwen_guides"], "false")
        self.assertEqual(app.APP.settings["upscale"]["target_width"], "1920")
        self.assertEqual(app.APP.settings["upscale"]["target_height"], "1080")

    def test_opening_1080p_widescreen_source_disables_outpaint_and_upscale_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            source = Path(tmp_text) / "new-source.mp4"
            source.write_bytes(b"placeholder")
            app.APP.settings["global"].update({"source": "input/old.mp4", "expand_outpaint": "true", "colorize": "true", "upscale": "true"})

            with (
                mock.patch.object(server, "video_metrics", return_value={"duration": 60.0}),
                mock.patch.object(server, "ffprobe_basic_info", return_value={"resolution": "1920x1080"}),
            ):
                app.APP.update_settings("global", {"source": str(source)})

        self.assertEqual(app.APP.settings["global"]["expand_outpaint"], "false")
        self.assertEqual(app.APP.settings["global"]["upscale"], "false")
        self.assertEqual(app.APP.settings["outpaint"]["target_height"], "720")
        self.assertEqual(app.APP.settings["outpaint"]["seed_qwen_guides"], "false")

    def test_load_settings_never_persists_qwen_seed_guides_as_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            settings_file = Path(tmp_text) / "settings.json"
            settings_file.write_text(json.dumps({"outpaint": {"seed_qwen_guides": "true"}}), encoding="utf-8")
            with mock.patch.object(app, "SETTINGS_FILE", settings_file), mock.patch.object(app, "newest", return_value=None):
                settings = app.load_settings()

        self.assertEqual(settings["outpaint"]["seed_qwen_guides"], "false")

    def test_load_settings_discards_removed_legacy_descratch_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            settings_file = Path(tmp_text) / "settings.json"
            settings_file.write_text(json.dumps({"cleanup": {"descratch": "true"}}), encoding="utf-8")
            with mock.patch.object(app, "SETTINGS_FILE", settings_file), mock.patch.object(app, "newest", return_value=None):
                settings = app.load_settings()

        self.assertNotIn("descratch", settings["cleanup"])

    def test_clear_overview_resets_project_ui_settings_to_defaults(self) -> None:
        app.APP.settings["global"].update({
            "source": "input/old.mp4",
            "expand_outpaint": "false",
            "colorize": "false",
            "upscale": "true",
            "add_soundtrack": "true",
            "section_start": "12",
            "section_end": "34",
            "last_browse_dir": "D:/media",
        })
        app.APP.settings["outpaint"].update({
            "outpaint_all_black_regions": "true",
            "prompt": "custom outpaint prompt",
            "crop_left": "88",
        })
        app.APP.settings["references"].update({
            "method": "openai",
            "manifest": "manifests/references/old.csv",
            "prompt": "custom reference prompt",
            "prompt_suffix": "custom suffix",
            "openai_api_key": "sk-test",
        })
        app.APP.settings["recomp"].update({"feather_pixels": "12", "colorized_video": "old.mp4"})

        app.APP.clear_overview()

        self.assertEqual(app.APP.settings["global"]["source"], "")
        self.assertEqual(app.APP.settings["global"]["expand_outpaint"], "true")
        self.assertEqual(app.APP.settings["global"]["colorize"], "true")
        self.assertEqual(app.APP.settings["global"]["upscale"], "false")
        self.assertEqual(app.APP.settings["global"]["add_soundtrack"], "false")
        self.assertEqual(app.APP.settings["global"]["section_start"], "0")
        self.assertEqual(app.APP.settings["global"]["section_end"], "")
        self.assertEqual(app.APP.settings["global"]["last_browse_dir"], "D:/media")
        self.assertEqual(app.APP.settings["outpaint"]["outpaint_all_black_regions"], "false")
        self.assertEqual(app.APP.settings["outpaint"]["prompt"], config.OUTPAINT_PROMPT)
        self.assertEqual(app.APP.settings["outpaint"]["crop_left"], "0")
        self.assertEqual(app.APP.settings["references"]["method"], "qwen")
        self.assertEqual(app.APP.settings["references"]["manifest"], "")
        self.assertEqual(app.APP.settings["references"]["prompt"], config.REFERENCE_PROMPT)
        self.assertEqual(app.APP.settings["references"]["prompt_suffix"], config.REFERENCE_PROMPT_SUFFIX)
        self.assertEqual(app.APP.settings["references"]["openai_api_key"], "")
        self.assertEqual(app.APP.settings["recomp"]["feather_pixels"], "80")
        self.assertEqual(app.APP.settings["recomp"]["colorized_video"], "")

    def test_detected_monochrome_source_sets_colorize_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            source = Path(tmp_text) / "new-source.mp4"
            source.write_bytes(b"placeholder")
            app.APP.settings["global"].update({"source": str(source), "colorize": "false"})

            app.APP.apply_detected_source_tone(str(source), True)
            self.assertEqual(app.APP.settings["global"]["colorize"], "true")

            app.APP.apply_detected_source_tone(str(source), False)
            self.assertEqual(app.APP.settings["global"]["colorize"], "false")

    def test_project_payload_round_trips_settings_with_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            path = Path(tmp_text) / "demo.arpp"
            app.APP.settings["global"].update({"source": "input/example.mp4"})
            path.write_text(json.dumps(app.project_payload(app.APP.settings)), encoding="utf-8")

            loaded = app.read_project_file(path)

        self.assertEqual(loaded["global"]["source"], "input/example.mp4")
        self.assertIn("schema_version", app.project_payload(app.APP.settings))

    def test_project_bundle_includes_reference_assets_without_openai_key(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "refs.csv"
            source = folder / "bw.png"
            color = folder / "color.png"
            project = folder / "demo.arpp"
            source.write_bytes(b"bw")
            color.write_bytes(b"color")
            app.write_manifest_details(
                manifest,
                "input/example.mp4",
                ["enabled", "end", "source_reference", "color_reference"],
                [{"enabled": "true", "end": "00:00:01.000", "source_reference": app.rel(source), "color_reference": app.rel(color)}],
            )
            settings = copy.deepcopy(app.APP.settings)
            settings["references"].update({"manifest": app.rel(manifest), "openai_api_key": "sk-secret"})

            app.write_project_file(project, settings)

            with zipfile.ZipFile(project) as archive:
                names = set(archive.namelist())
                payload = json.loads(archive.read("project.json").decode("utf-8"))

            self.assertIn(app.rel(manifest).replace("\\", "/"), names)
            self.assertIn(app.rel(source).replace("\\", "/"), names)
            self.assertIn(app.rel(color).replace("\\", "/"), names)
            self.assertNotIn("openai_api_key", payload["settings"]["references"])

    def test_cache_categories_do_not_include_project_state(self) -> None:
        folders_by_key = {
            category["key"]: {app.rel(folder).replace("\\", "/") for folder in category["folders"]}
            for category in cache.cache_categories()
        }

        self.assertNotIn("intermediate/outpaint_guides", folders_by_key["outpaint"])
        self.assertNotIn("intermediate/outpaint_anchors", folders_by_key["outpaint"])
        self.assertNotIn("manifests/outpaint_chunks", folders_by_key["outpaint"])
        self.assertNotIn("manifests/references", folders_by_key["shots"])
        self.assertNotIn("intermediate/outpainted_references", folders_by_key["references"])
        self.assertNotIn("intermediate/outpainted_references_color", folders_by_key["references"])

    def test_project_bundle_includes_outpaint_chunk_manifest_and_guides(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "chunks.csv"
            guide = folder / "guide.png"
            guide_previous = folder / "guide_previous.png"
            project = folder / "demo.arpp"
            guide.write_bytes(b"guide")
            guide_previous.write_bytes(b"previous")
            Path(str(guide) + ".json").write_text("{}", encoding="utf-8")
            app.write_outpaint_chunk_rows(
                manifest,
                [
                    {
                        "chunk_index": "0",
                        "start_frame": "0",
                        "end_frame": "24",
                        "guide_image": app.rel(guide),
                        "guide_frames": json.dumps([{"image": app.rel(guide), "image_previous": app.rel(guide_previous)}]),
                    }
                ],
            )
            settings = copy.deepcopy(app.APP.settings)
            settings.setdefault("outpaint", {})["manifest"] = app.rel(manifest)

            app.write_project_file(project, settings)

            with zipfile.ZipFile(project) as archive:
                names = set(archive.namelist())

            self.assertIn(app.rel(manifest).replace("\\", "/"), names)
            self.assertIn(app.rel(guide).replace("\\", "/"), names)
            self.assertIn(app.rel(guide_previous).replace("\\", "/"), names)
            self.assertIn((app.rel(guide) + ".json").replace("\\", "/"), names)

    def test_project_load_skips_assets_already_on_disk(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            asset = folder / "guide.txt"
            asset.write_bytes(b"guide bytes")
            bundle = folder / "demo.arpp"
            with zipfile.ZipFile(bundle, "w") as archive:
                archive.writestr(app.rel(asset).replace("\\", "/"), b"guide bytes")
            first_mtime = asset.stat().st_mtime_ns

            with zipfile.ZipFile(bundle) as archive:
                project_io.extract_project_assets(archive)

            # Identical content must keep its mtime so resume signatures stay valid.
            self.assertEqual(asset.stat().st_mtime_ns, first_mtime)

            asset.write_bytes(b"stale bytes")
            with zipfile.ZipFile(bundle) as archive:
                project_io.extract_project_assets(archive)

            self.assertEqual(asset.read_bytes(), b"guide bytes")

    def test_signature_match_ignores_fingerprint_mtime_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            output = folder / "chunk.mp4"
            guide = folder / "guide.png"
            output.write_bytes(b"rendered")
            guide.write_bytes(b"guide bytes")
            signature = {
                "version": 1,
                "guide_fingerprint": common.file_fingerprint(guide),
                "row": {"color_reference": "ref.png", "color_reference_previous": None},
            }
            common.write_signature(output, signature)

            touched = copy.deepcopy(signature)
            touched["guide_fingerprint"]["mtime_ns"] += 1
            self.assertTrue(common.signature_matches(output, touched))

            bookkeeping = copy.deepcopy(signature)
            bookkeeping["row"] = {"color_reference": "ref.png"}
            self.assertTrue(common.signature_matches(output, bookkeeping))

            changed = copy.deepcopy(signature)
            changed["guide_fingerprint"]["sha256"] = "0" * 64
            self.assertFalse(common.signature_matches(output, changed))

    def test_signature_match_does_not_ignore_model_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            output = Path(tmp_text) / "chunk.mp4"
            output.write_bytes(b"chunk")
            common.write_signature(output, {"outpaint_lora": config.DEFAULT_OUTPAINT_LORA, "seed": 42})

            selected = {"outpaint_lora": "different-model.safetensors", "seed": 42}

            self.assertFalse(common.signature_matches(output, selected))

    def test_replace_unless_identical_keeps_matching_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            target = folder / "chunk.mp4"
            target.write_bytes(b"same bytes")
            first_mtime = target.stat().st_mtime_ns
            partial = folder / "chunk.mp4.partial.mp4"
            partial.write_bytes(b"same bytes")

            common.replace_unless_identical(partial, target, "chunk")

            self.assertFalse(partial.exists())
            self.assertEqual(target.stat().st_mtime_ns, first_mtime)

            partial.write_bytes(b"new bytes!")
            common.replace_unless_identical(partial, target, "chunk")

            self.assertEqual(target.read_bytes(), b"new bytes!")

    def test_split_video_chunk_resplits_when_source_content_changes(self) -> None:
        ffmpeg = common.find_ffmpeg()
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            source = folder / "src.mp4"
            target = folder / "input_0000.mp4"

            def encode_source(colour: str) -> None:
                subprocess.run(
                    [ffmpeg, "-y", "-f", "lavfi", "-i", f"color=c={colour}:s=64x64:d=0.5:r=8", "-pix_fmt", "yuv420p", str(source)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )

            encode_source("red")
            fingerprint = common.file_fingerprint(source)
            upscale_video.split_video_chunk(ffmpeg, source, target, 0, 4, 8.0, False, fingerprint)
            first = common.file_fingerprint(target)

            # Same source content: the split is fresh and must not be redone.
            upscale_video.split_video_chunk(ffmpeg, source, target, 0, 4, 8.0, False, fingerprint)
            self.assertEqual(target.stat().st_mtime_ns, first["mtime_ns"])

            # New content under the same file name must replace the stale chunk input.
            encode_source("blue")
            upscale_video.split_video_chunk(ffmpeg, source, target, 0, 4, 8.0, False, common.file_fingerprint(source))
            self.assertNotEqual(common.file_fingerprint(target)["sha256"], first["sha256"])

    def test_upscale_chunk_split_uses_frame_index_trim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            source = folder / "src.mp4"
            target = folder / "input_0000.mp4"
            source.write_bytes(b"placeholder")
            fingerprint = common.file_fingerprint(source)

            with (
                mock.patch.object(upscale_video, "split_matches_source", return_value=False),
                mock.patch.object(upscale_video.subprocess, "run") as run,
                mock.patch.object(upscale_video, "replace_unless_identical"),
                mock.patch.object(upscale_video, "write_split_sidecar"),
            ):
                upscale_video.split_video_chunk("ffmpeg", source, target, 7, 19, 23.97602398, False, fingerprint)

        command = run.call_args.args[0]
        self.assertNotIn("-ss", command)
        self.assertNotIn("-t", command)
        self.assertEqual(command[command.index("-vf") + 1], "trim=start_frame=7:end_frame=19,setpts=N/(23.97602398*TB),fps=23.97602398,setsar=1")

    def test_upscale_normalize_chunk_rebuilds_frame_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            source = folder / "raw.mp4"
            target = folder / "final.mp4"
            source.write_bytes(b"placeholder")

            with (
                mock.patch.object(upscale_video.subprocess, "run") as run,
                mock.patch.object(upscale_video, "replace_with_retry"),
            ):
                upscale_video.normalize_chunk("ffmpeg", source, target, 1920, 1080, 4, 23.97602398, True)

        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-vf") + 1], "trim=start_frame=4,setpts=N/(23.97602398*TB),fps=23.97602398,scale=1920:1080:flags=lanczos,setsar=1")
        self.assertIn("-fps_mode", command)

    def test_upscale_mux_audio_uses_distinct_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            output = folder / "preview.mp4"
            video_source = output.with_suffix(output.suffix + ".partial" + output.suffix)
            audio_source = folder / "source.mp4"

            with (
                mock.patch.object(upscale_video.subprocess, "run") as run,
                mock.patch.object(upscale_video, "replace_with_retry") as replace,
            ):
                upscale_video.mux_audio("ffmpeg", video_source, audio_source, output)

        expected_partial = output.with_suffix(output.suffix + ".mux.partial" + output.suffix)
        command = run.call_args.args[0]
        self.assertEqual(Path(command[-1]), expected_partial)
        self.assertNotEqual(Path(command[-1]), video_source)
        replace.assert_called_once_with(expected_partial, output, "Upscaled output")

    def test_write_manifest_details_skips_identical_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "refs.csv"
            rows = [{"enabled": "true", "end": "00:00:01.000"}]

            app.write_manifest_details(manifest, "input/example.mp4", ["enabled", "end"], rows)
            first_mtime = manifest.stat().st_mtime_ns
            app.write_manifest_details(manifest, "input/example.mp4", ["enabled", "end"], rows)

            self.assertEqual(manifest.stat().st_mtime_ns, first_mtime)

            app.write_manifest_details(manifest, "input/example.mp4", ["enabled", "end"], [{"enabled": "false", "end": "00:00:01.000"}])
            self.assertIn("false", manifest.read_text(encoding="utf-8"))

    def test_copy_to_comfy_input_replaces_same_size_different_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            source = folder / "prepared_0000.mp4"
            source.write_bytes(b"metropolis bytes")

            name = common.copy_to_comfy_input(source, folder, "arp_outpaint")
            target = folder / "input" / Path(name)

            # Another project's same-named, same-sized input must not survive as a stale copy.
            target.write_bytes(b"baskerville16gtd")
            common.copy_to_comfy_input(source, folder, "arp_outpaint")

            self.assertEqual(target.read_bytes(), b"metropolis bytes")

    def test_colorize_reference_copy_names_are_content_keyed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            reference = folder / "shot_ref.png"
            reference.write_bytes(b"first reference")

            first = colorize_video.copy_reference_to_comfy_input(reference, folder)
            again = colorize_video.copy_reference_to_comfy_input(reference, folder)
            reference.write_bytes(b"changed reference")
            second = colorize_video.copy_reference_to_comfy_input(reference, folder)

            self.assertEqual(first, again)
            self.assertNotEqual(first, second)
            self.assertEqual((folder / "input" / Path(first)).read_bytes(), b"first reference")
            self.assertEqual((folder / "input" / Path(second)).read_bytes(), b"changed reference")

    def test_guide_copy_names_are_content_keyed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            metropolis = folder / "metropolis" / "guide_prev_raw_0000.png"
            baskervilles = folder / "baskervilles" / "guide_prev_raw_0000.png"
            metropolis.parent.mkdir()
            baskervilles.parent.mkdir()
            metropolis.write_bytes(b"metropolis frame")
            baskervilles.write_bytes(b"baskervilles frame")

            first = outpaint_video.copy_guide_image_to_comfy_input(metropolis, folder)
            second = outpaint_video.copy_guide_image_to_comfy_input(baskervilles, folder)
            again = outpaint_video.copy_guide_image_to_comfy_input(metropolis, folder)

            # Same stem, different content: each project keeps its own Comfy input copy.
            self.assertNotEqual(first, second)
            self.assertEqual(first, again)
            self.assertEqual((folder / "input" / Path(first)).read_bytes(), b"metropolis frame")
            self.assertEqual((folder / "input" / Path(second)).read_bytes(), b"baskervilles frame")

    def test_outpaint_signature_fingerprints_extra_guide_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            prepared = folder / "prepared.mp4"
            workflow = folder / "workflow.json"
            guide = folder / "extra_guide.png"
            prepared.write_bytes(b"prepared")
            workflow.write_bytes(b"{}")
            guide.write_bytes(b"guide bytes")
            args = argparse.Namespace(
                target_aspect="16:9", target_height=480, outpaint_all_black_regions=False,
                prompt="outpaint", negative_prompt="", guide_strength=0.7,
                load_video_node_id="5060", save_node_id="5076", extra_save_node_id=["5069"],
                output_node_id="5076", model_backend="gguf", gguf_model="model.gguf",
                video_vae="vae.safetensors", outpaint_lora="lora.safetensors",
                chunk_seconds=20.0, overlap_frames=8,
            )

            sig = outpaint_video.raw_signature(args, workflow, prepared, extra_guides=[{"frame_idx": 5, "strength": 1.0, "image": guide}])

            self.assertEqual(sig["extra_guides"][0]["image_fingerprint"]["sha256"], common.file_fingerprint(guide)["sha256"])

    def test_unchanged_auto_guide_pixels_preserve_resume_signature(self) -> None:
        import cv2
        import numpy as np

        class FakeCapture:
            def __init__(self, frame):
                self.frame = frame

            def get(self, _property):
                return 2

            def set(self, _property, _value):
                return True

            def read(self):
                return True, self.frame.copy()

            def release(self):
                return None

        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            previous_raw = folder / "raw_0000.mp4"
            previous_raw.write_bytes(b"regenerated chunk")
            target = folder / "guide_prev_raw_0000.png"
            next_chunk = folder / "raw_0001.mp4"
            next_chunk.write_bytes(b"cached next chunk")
            original = np.full((3, 4, 3), 42, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(target), original))
            os.utime(target, ns=(1_000_000_000, 1_000_000_000))
            os.utime(previous_raw, ns=(2_000_000_000, 2_000_000_000))

            before = common.file_fingerprint(target)
            common.write_signature(next_chunk, {"guide_fingerprint": before})
            with mock.patch("cv2.VideoCapture", return_value=FakeCapture(original)):
                extracted = outpaint_video.extract_last_frame_as_guide(previous_raw, folder)
            after = common.file_fingerprint(extracted)

            self.assertEqual(after, before)
            self.assertTrue(common.signature_matches(next_chunk, {"guide_fingerprint": after}))

            changed = original.copy()
            changed[0, 0, 0] += 1
            os.utime(previous_raw, ns=(3_000_000_000, 3_000_000_000))
            with mock.patch("cv2.VideoCapture", return_value=FakeCapture(changed)):
                outpaint_video.extract_last_frame_as_guide(previous_raw, folder)

            changed_fingerprint = common.file_fingerprint(target)
            self.assertNotEqual(changed_fingerprint["sha256"], before["sha256"])
            self.assertFalse(common.signature_matches(next_chunk, {"guide_fingerprint": changed_fingerprint}))

    def test_openai_reference_command_requires_key_and_uses_selected_model(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "refs.csv"
            source = folder / "bw.png"
            color = folder / "color.png"
            source.write_bytes(b"bw")
            app.write_manifest_details(
                manifest,
                "input/example.mp4",
                ["enabled", "end", "source_reference", "color_reference", "prompt"],
                [{"enabled": "true", "end": "00:00:01.000", "source_reference": app.rel(source), "color_reference": app.rel(color), "prompt": "warm highlights"}],
            )
            app.APP.settings["references"].update({"openai_api_key": "", "openai_image_model": "gpt-image-2"})

            with self.assertRaisesRegex(RuntimeError, "OpenAI API key"):
                app.openai_reference_regeneration_command(str(manifest), 0)

            app.APP.settings["references"].update({"openai_api_key": "sk-test", "openai_image_model": "gpt-image-1"})
            command, output = app.openai_reference_regeneration_command(str(manifest), 0)

            self.assertIn("openai_generate_reference.py", " ".join(command))
            self.assertIn("--manifest", command)
            self.assertIn("--row-index", command)
            self.assertEqual(command[command.index("--model") + 1], "gpt-image-1")
            self.assertEqual(output, app.rel(color))

    def test_openai_manifest_command_can_send_nearby_reference_images(self) -> None:
        app.APP.settings["references"].update(
            {
                "manifest": "manifests/references/demo.csv",
                "method": "openai",
                "openai_api_key": "sk-test",
                "openai_image_model": "gpt-image-2",
                "openai_send_references": "true",
            }
        )

        command = app.APP.command_for("references")

        self.assertIn("openai_generate_reference.py", " ".join(command))
        self.assertIn("--reference-count", command)
        self.assertNotIn("qwen_colorize_references.py", " ".join(command))

    def test_openai_nearby_references_prefer_previous_then_later(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            refs = []
            rows = []
            for index in range(5):
                color = folder / f"color_{index}.png"
                if index in {0, 2, 3, 4}:
                    color.write_bytes(b"ref")
                refs.append(color)
                rows.append({"color_reference": app.rel(color)})

            chosen = openai_generate_reference.nearby_reference_images(rows, 1, 3)
            early = openai_generate_reference.nearby_reference_images(rows, 0, 3)

        self.assertEqual(chosen, [refs[0], refs[2], refs[3]])
        self.assertEqual(early, [refs[2], refs[3], refs[4]])

    def test_reference_edit_recent_references_prefer_previous_then_later(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            refs = []
            rows = []
            for index in range(5):
                color = folder / f"color_{index}.png"
                if index in {0, 2, 3, 4}:
                    color.write_bytes(b"ref")
                refs.append(app.rel(color))
                rows.append({"color_reference": app.rel(color)})

            chosen = app.recent_color_references(rows, 1, 3)
            early = app.recent_color_references(rows, 0, 3)

        self.assertEqual(chosen, [refs[0], refs[2], refs[3]])
        self.assertEqual(early, [refs[2], refs[3], refs[4]])

    def test_sam2_mask_helper_uses_point_prompts_and_returns_mask_data_url(self) -> None:
        import numpy as np
        from PIL import Image

        class FakePredictor:
            def __init__(self) -> None:
                self.image_shape = None
                self.point_coords = None
                self.point_labels = None

            def set_image(self, image) -> None:
                self.image_shape = image.shape

            def predict(self, point_coords=None, point_labels=None, multimask_output=True):
                self.point_coords = point_coords
                self.point_labels = point_labels
                masks = np.zeros((2, 4, 4), dtype=np.uint8)
                masks[1, 1:3, 1:3] = 1
                return masks, np.array([0.2, 0.9]), None

        predictor = FakePredictor()
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            image = Path(tmp_text) / "source.png"
            Image.new("RGB", (4, 4), (20, 30, 40)).save(image)
            with mock.patch.object(sam_masks, "_sam2_predictor", return_value=predictor):
                result = sam_masks.sam2_mask_for_image(
                    image,
                    [{"x": 1, "y": 1, "label": "add"}, {"x": 2, "y": 2, "label": "subtract"}],
                    4,
                    4,
                )

        self.assertTrue(result["mask"].startswith("data:image/png;base64,"))
        self.assertIn("SAM 2.1 Hiera Large", result["provider"])
        self.assertEqual(predictor.image_shape, (4, 4, 3))
        self.assertEqual(predictor.point_labels.tolist(), [1, 0])
        self.assertEqual(predictor.point_coords.tolist(), [[1.0, 1.0], [2.0, 2.0]])

    def test_reference_edit_preview_command_selects_masked_and_unmasked_runners(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "refs.csv"
            source = folder / "bw.png"
            color = folder / "color.png"
            masked_workflow = folder / "masked.json"
            source.write_bytes(b"bw")
            color.write_bytes(b"color")
            masked_workflow.write_text("{}", encoding="utf-8")
            app.write_manifest_details(
                manifest,
                "input/example.mp4",
                ["enabled", "end", "source_reference", "color_reference", "prompt"],
                [{"enabled": "true", "end": "00:00:01.000", "source_reference": app.rel(source), "color_reference": app.rel(color), "prompt": ""}],
            )
            app.APP.settings["references"].update(
                {
                    "workflow": "workflows/qwen_image_edit/Image Edit (Qwen 2511).json",
                    "masked_workflow": app.rel(masked_workflow),
                }
            )
            unmasked, unmasked_output = app.reference_edit_preview_command(app.rel(manifest), 0, "make it green")
            masked, masked_output = app.reference_edit_preview_command(app.rel(manifest), 0, "make it green", "iVBORw0KGgo=")

        self.assertIn("generate_single_reference.py", " ".join(unmasked))
        self.assertIn("edit_reference_image.py", " ".join(masked))
        self.assertIn("--mask", masked)
        self.assertIn("outpainted_references_color_edits", unmasked_output)
        self.assertIn("outpainted_references_color_edits", masked_output)

    def test_reference_edit_prompt_preserves_lighter_sampled_values(self) -> None:
        prompt = app.reference_edit_prompt("make the selected suit light grey", "#c8c8c8")

        self.assertIn("#c8c8c8", prompt)
        self.assertIn("brightness/value", prompt)
        self.assertIn("raise the masked area's luminance", prompt)
        self.assertIn("do not merely darken", prompt)

    def test_semantic_edit_prompt_does_not_add_colour_instructions_without_a_sample(self) -> None:
        reference_prompt = app.reference_edit_prompt("remove the masked text", "")
        guide_prompt = outpaint_guides._guide_edit_prompt("remove the masked text", "")

        self.assertEqual(reference_prompt, "remove the masked text")
        self.assertEqual(guide_prompt, "remove the masked text")

    def test_reference_edit_accept_and_revert_updates_manifest(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "refs.csv"
            source = folder / "bw.png"
            color = folder / "color.png"
            edit = folder / "edit.png"
            source.write_bytes(b"bw")
            color.write_bytes(b"color")
            edit.write_bytes(b"edit")
            app.write_manifest_details(
                manifest,
                "input/example.mp4",
                ["enabled", "end", "source_reference", "color_reference", "prompt"],
                [{"enabled": "true", "end": "00:00:01.000", "source_reference": app.rel(source), "color_reference": app.rel(color), "prompt": ""}],
            )

            accepted = app.accept_reference_edit(app.rel(manifest), 0, app.rel(edit))
            after_accept = app.read_manifest(manifest)[0]
            reverted = app.revert_reference_edit(app.rel(manifest), 0)
            after_revert = app.read_manifest(manifest)[0]

        self.assertEqual(accepted["color_reference"], app.rel(edit))
        self.assertEqual(after_accept["color_reference_previous"], app.rel(color))
        self.assertEqual(reverted["color_reference"], app.rel(color))
        self.assertEqual(after_revert["color_reference_previous"], app.rel(edit))

    def test_guide_edit_preview_command_always_uses_masked_runner(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "chunks.csv"
            guide = folder / "guide.png"
            prepared = folder / "prepared.mp4"
            prepared_preview = folder / "prepared_preview.png"
            masked_workflow = folder / "masked.json"
            Image.new("RGB", (16, 9), (32, 32, 32)).save(guide)
            Image.new("RGB", (16, 9), (0, 0, 0)).save(prepared_preview)
            prepared.write_bytes(b"prepared")
            masked_workflow.write_text("{}", encoding="utf-8")
            rows = [
                {
                    "chunk_index": "0",
                    "start_frame": "0",
                    "end_frame": "24",
                    "start_seconds": "0",
                    "end_seconds": "1",
                    "guide_frames": json.dumps([{"frame_idx": 0, "strength": 0.7, "image": app.rel(guide), "seed": True}]),
                }
            ]
            app.write_outpaint_chunk_rows(manifest, rows)
            app.APP.settings["references"].update(
                {
                    "workflow": "workflows/qwen_image_edit/Image Edit (Qwen 2511).json",
                    "masked_workflow": app.rel(masked_workflow),
                }
            )
            fake_state = {"manifest": app.rel(manifest), "rows": [{"index": 0}]}
            with (
                mock.patch.object(outpaint_guides, "outpaint_chunks_state", return_value=fake_state),
                mock.patch.object(outpaint_guides, "ensure_outpaint_prepared_canvas", return_value=prepared),
                mock.patch.object(outpaint_guides, "chunk_frame_preview", return_value=app.rel(prepared_preview)),
            ):
                unmasked, unmasked_output = app.guide_edit_preview_command(0, 0, "replace detail")
                defaulted, _defaulted_output = app.guide_edit_preview_command(0, 0, "")
                masked, masked_output = app.guide_edit_preview_command(0, 0, "replace detail", "iVBORw0KGgo=")

        self.assertIn("edit_reference_image.py", " ".join(unmasked))
        self.assertIn("edit_reference_image.py", " ".join(masked))
        self.assertEqual(Path(unmasked[unmasked.index("--source-image") + 1]), app.resolve(app.rel(guide)))
        self.assertEqual(defaulted[defaulted.index("--instruction") + 1], "Replace the black bars.")
        self.assertIn("--mask", unmasked)
        self.assertIn("--mask", masked)
        self.assertIn("outpaint_guides", unmasked_output)
        self.assertIn("outpaint_guides", masked_output)

    def test_outpaint_guide_generation_defaults_to_replace_black_bars(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "chunks.csv"
            prepared = folder / "prepared.mp4"
            source_frame = folder / "source.jpg"
            masked_workflow = folder / "masked.json"
            prepared.write_bytes(b"prepared")
            from PIL import Image

            Image.new("RGB", (16, 9), (0, 0, 0)).save(source_frame)
            masked_workflow.write_text("{}", encoding="utf-8")
            rows = [
                {
                    "chunk_index": "0",
                    "start_frame": "0",
                    "end_frame": "24",
                    "start_seconds": "0",
                    "end_seconds": "1",
                }
            ]
            app.write_outpaint_chunk_rows(manifest, rows)
            app.APP.settings["references"].update({"masked_workflow": app.rel(masked_workflow)})
            fake_state = {"manifest": app.rel(manifest), "rows": [{"index": 0, "fps": 24, "start": 0.0, "end": 1.0}]}

            with (
                mock.patch.object(outpaint_guides, "outpaint_chunks_state", return_value=fake_state),
                mock.patch.object(outpaint_guides, "pipeline_source_text", return_value="input/example.mp4"),
                mock.patch.object(outpaint_guides, "ensure_outpaint_prepared_canvas", return_value=prepared),
                mock.patch.object(outpaint_guides, "chunk_frame_preview", return_value=app.rel(source_frame)),
            ):
                command, _output, _canvas, _seconds = app.outpaint_guide_generation_command(0, "")

        self.assertIn("edit_reference_image.py", " ".join(command))
        self.assertEqual(command[command.index("--instruction") + 1], "Replace the black bars.")
        self.assertIn("--mask", command)

    def test_auto_edge_mask_uses_sloppy_feathered_boundary(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "pillarbox.png"
            mask_path = folder / "mask.png"
            image = Image.new("RGB", (96, 48), (0, 0, 0))
            for y in range(48):
                for x in range(24, 72):
                    image.putpixel((x, y), (64, 64, 64))
            image.save(source)

            guide_frame_utils.save_edge_mask_for_image(source, mask_path)

            with Image.open(mask_path) as mask:
                pixels = mask.convert("L").load()
                left_edges = []
                for y in range(mask.height):
                    xs = [x for x in range(mask.width // 2) if pixels[x, y] > 0]
                    left_edges.append(max(xs))

                self.assertEqual(pixels[0, mask.height // 2], 255)
                self.assertEqual(pixels[mask.width // 2, mask.height // 2], 0)
                self.assertGreater(len(set(left_edges)), 4)
                self.assertTrue(any(0 < pixels[x, mask.height // 2] < 255 for x in range(mask.width // 2)))

    def test_masked_edit_uses_bundled_workflow_when_setting_is_empty(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "refs.csv"
            source = folder / "bw.png"
            color = folder / "color.png"
            source.write_bytes(b"bw")
            color.write_bytes(b"color")
            app.write_manifest_details(
                manifest,
                "input/example.mp4",
                ["enabled", "end", "source_reference", "color_reference", "prompt"],
                [{"enabled": "true", "end": "00:00:01.000", "source_reference": app.rel(source), "color_reference": app.rel(color), "prompt": ""}],
            )
            app.APP.settings["references"].update({"masked_workflow": ""})

            command, _output = app.reference_edit_preview_command(app.rel(manifest), 0, "make it green", "iVBORw0KGgo=")

        self.assertIn("edit_reference_image.py", " ".join(command))
        self.assertIn("workflows/qwen_image_edit/Image Edit Inpaint (Qwen 2511).json", command)

    def test_masked_edit_workflow_patches_user_instruction_into_positive_prompt(self) -> None:
        import argparse

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            comfy_dir = folder / "comfy"
            source = folder / "source.png"
            mask = folder / "mask.png"
            source.write_bytes(b"source")
            mask.write_bytes(b"mask")
            workflow = qwen_colorize_references.load_workflow(app.ROOT / "workflows" / "qwen_image_edit" / "Image Edit Inpaint (Qwen 2511).json")
            args = argparse.Namespace(
                comfy_dir=comfy_dir,
                load_image_node_id="auto",
                mask_image_node_id="auto",
                save_node_id="auto",
                prompt_node_id=None,
                load_image_widget="0",
                mask_image_widget="0",
                prompt_widget="0",
                save_prefix_widget="0",
                model_backend="gguf",
                gguf_model="qwen-image-edit-2511-Q4_K_M.gguf",
            )
            qwen_colorize_references.patch_qwen_model_backend(args, workflow)

            prompt = edit_reference_image.patch_masked_workflow(
                args,
                workflow,
                source,
                mask,
                folder / "output.png",
                "Replace the masked hat with a bright green hat.",
            )

        self.assertEqual(prompt["1"]["inputs"]["prompt"], "Replace the masked hat with a bright green hat.")
        self.assertEqual(prompt["12"]["inputs"]["image"], "arp_qwen_ref_masks/mask.png")
        self.assertEqual(prompt["11"]["inputs"]["image"], "arp_qwen_ref_edits/source.png")
        self.assertEqual(prompt["152"]["inputs"]["model"], ["31", 0])
        self.assertEqual(prompt["152"]["inputs"]["steps"], 4)
        self.assertEqual(prompt["152"]["inputs"]["cfg"], 1)
        self.assertEqual(prompt["24"]["inputs"]["lora_name"], "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
        self.assertEqual(prompt["31"]["class_type"], "CFGNorm")
        self.assertEqual(prompt["32"]["class_type"], "UnetLoaderGGUF")
        self.assertEqual(prompt["161"]["inputs"]["mask"], ["12", 0])
        self.assertEqual(prompt["161"]["inputs"]["expand"], 32)
        self.assertEqual(prompt["162"]["inputs"]["mask"], ["12", 0])
        self.assertEqual(prompt["162"]["inputs"]["expand"], 8)
        self.assertEqual(prompt["163"]["inputs"]["image"], ["11", 0])
        self.assertEqual(prompt["163"]["inputs"]["blur_radius"], 31)
        self.assertEqual(prompt["164"]["inputs"]["destination"], ["11", 0])
        self.assertEqual(prompt["164"]["inputs"]["source"], ["163", 0])
        self.assertEqual(prompt["164"]["inputs"]["mask"], ["161", 0])
        self.assertEqual(prompt["1"]["inputs"]["image1"], ["164", 0])
        self.assertEqual(prompt["39"]["inputs"]["image1"], ["164", 0])
        self.assertEqual(prompt["126"]["inputs"]["mask"], ["161", 0])
        self.assertEqual(prompt["160"]["inputs"]["destination"], ["11", 0])
        self.assertEqual(prompt["160"]["inputs"]["source"], ["13", 0])
        self.assertEqual(prompt["160"]["inputs"]["mask"], ["162", 0])
        self.assertEqual(prompt["14"]["inputs"]["images"], ["160", 0])

    def test_guide_edit_accept_and_revert_updates_guide_frames(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "chunks.csv"
            guide = folder / "guide.png"
            edit = folder / "edit.png"
            guide.write_bytes(b"guide")
            edit.write_bytes(b"edit")
            rows = [
                {
                    "chunk_index": "0",
                    "start_frame": "0",
                    "end_frame": "24",
                    "start_seconds": "0",
                    "end_seconds": "1",
                    "guide_frames": json.dumps([{"frame_idx": 0, "strength": 0.7, "image": app.rel(guide)}]),
                }
            ]
            app.write_outpaint_chunk_rows(manifest, rows)
            fake_state = {"manifest": app.rel(manifest), "rows": [{"index": 0}]}
            with mock.patch.object(outpaint_guides, "outpaint_chunks_state", return_value=fake_state):
                accepted = app.accept_guide_edit(0, 0, app.rel(edit))
                accepted_frame = outpaint_guides._parse_guide_frames(app.read_outpaint_chunk_rows(manifest)[0])[0]
                reverted = app.revert_guide_edit(0, 0)
                reverted_frame = outpaint_guides._parse_guide_frames(app.read_outpaint_chunk_rows(manifest)[0])[0]

        self.assertEqual(accepted["image"], app.rel(edit))
        self.assertEqual(accepted_frame["image_previous"], app.rel(guide))
        self.assertNotIn("seed", accepted_frame)
        self.assertEqual(reverted["image"], app.rel(guide))
        self.assertEqual(reverted_frame["image_previous"], app.rel(edit))

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
        # deepexemplar and colormnet are distinct identity-keyed artifacts.
        self.assertNotEqual(outputs[0], outputs[1])
        for output in outputs:
            self.assertTrue(output.startswith("intermediate/outpainted_colorized/"), output)
            self.assertTrue(output.endswith(".mp4"))

    def test_colorization_command_can_request_both_methods(self) -> None:
        app.APP.settings["colour"].update({"manifest": "manifests/references/colorize_manifest_demo_shots_auto.csv", "method": "both", "processing_height": "1080"})

        command = app.APP.command_for("colour")

        self.assertIn("--method", command)
        self.assertIn("both", command)
        self.assertEqual(command[command.index("--processing-height") + 1], "1080")
        self.assertNotIn("--output", command)

    def test_skip_outpainting_uses_pipeline_source_for_shot_detection(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "true", "section_start": "0", "section_end": ""})

        app.APP.hydrate_stage_inputs("global")
        stage_keys = [stage.key for stage in app.APP.active_stages()]
        command = app.APP.command_for("shots")

        self.assertNotIn("outpaint", stage_keys)
        self.assertIn("shots", stage_keys)
        self.assertEqual(app.APP.settings["shots"]["outpainted_video"], "input/example.mp4")
        self.assertEqual(app.APP.settings["recomp"]["outpainted_video"], "")
        self.assertEqual(app.APP.settings["recomp"]["source"], "input/example.mp4")
        self.assertIn("input/example.mp4", command)

    def test_colour_only_recomposition_command_uses_source_as_base(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "true", "section_start": "0", "section_end": ""})
        app.APP.settings["recomp"].update(
            {
                "outpainted_video": "",
                "source": "input/example.mp4",
                "colorized_video": "intermediate/outpainted_colorized/example_color.mp4",
                "output": "output/reassembled/example_recomp.mp4",
            }
        )

        command = app.APP.command_for("recomp")

        self.assertNotIn("--outpainted", command)
        self.assertEqual(command[command.index("--source") + 1], "input/example.mp4")
        self.assertEqual(command[command.index("--colorized") + 1], "intermediate/outpainted_colorized/example_color.mp4")

    def test_no_overview_steps_selected_leaves_only_output_tab(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "false", "upscale": "false", "section_start": "0", "section_end": ""})

        self.assertEqual(app.APP.active_stages(), ())
        self.assertEqual(app.APP.phase_progress()["stages"], [])

    def test_optional_phase_combinations_have_expected_stage_order(self) -> None:
        cases = [
            (False, False, False, []),
            (True, False, False, ["outpaint", "recomp"]),
            (False, True, False, ["shots", "references", "colour", "recomp"]),
            (False, False, True, ["upscale"]),
            (True, True, False, ["outpaint", "shots", "references", "colour", "recomp"]),
            (True, False, True, ["outpaint", "recomp", "upscale"]),
            (False, True, True, ["shots", "references", "colour", "recomp", "upscale"]),
            (True, True, True, ["outpaint", "shots", "references", "colour", "recomp", "upscale"]),
        ]
        for outpaint, colorize, upscale, expected in cases:
            with self.subTest(outpaint=outpaint, colorize=colorize, upscale=upscale):
                app.APP.settings["global"].update(
                    {
                        "source": "input/example.mp4",
                        "expand_outpaint": "true" if outpaint else "false",
                        "colorize": "true" if colorize else "false",
                        "upscale": "true" if upscale else "false",
                        "section_start": "0",
                        "section_end": "",
                    }
                )

                self.assertEqual([stage.key for stage in app.APP.active_stages()], expected)

    def test_outpaint_without_colour_can_feed_upscale_after_recomposition(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "true", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
        app.APP.settings["outpaint"].update({"target_aspect": "16:9", "target_height": "720", "crop_left": "0", "crop_right": "0", "crop_top": "0", "crop_bottom": "0"})

        outpainted = app.outpaint_output_for("input/example.mp4", "16:9", "720")
        outpainted_path = app.resolve(outpainted)
        outpainted_path.parent.mkdir(parents=True, exist_ok=True)
        outpainted_path.write_bytes(b"placeholder")
        try:
            app.APP.hydrate_stage_inputs("outpaint")
            recomp_output = app.recomposition_output_for(outpainted)
            command = app.APP.command_for("upscale")
        finally:
            outpainted_path.unlink(missing_ok=True)

        self.assertEqual([stage.key for stage in app.APP.active_stages()], ["outpaint", "recomp", "upscale"])
        self.assertEqual(app.APP.settings["upscale"]["input_video"], recomp_output)
        self.assertEqual(command[command.index("--input") + 1], recomp_output)

    def test_upscale_after_earlier_phase_waits_for_recomposition_output(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "true", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
        app.APP.settings["recomp"]["output"] = "output/reassembled/missing_recomposition.mp4"

        ok, message = app.APP.run_stage("upscale")

        self.assertFalse(ok)
        self.assertIn("Run Recomposition first", message)

    def test_upscale_stage_stops_before_comfy_on_unsupported_flashvsr_gpu(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})

        with (
            mock.patch.object(server, "flashvsr_hardware_warning", return_value="Unsupported FlashVSR GPU"),
            mock.patch.object(server, "ensure_comfy_available_for_stage") as ensure_comfy,
        ):
            ok, message = app.APP.run_stage("upscale")

        self.assertFalse(ok)
        self.assertEqual(message, "Unsupported FlashVSR GPU")
        ensure_comfy.assert_not_called()

    def test_upscale_preview_stops_on_unsupported_flashvsr_gpu(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})

        with mock.patch.object(server, "flashvsr_hardware_warning", return_value="Unsupported FlashVSR GPU"):
            ok, message = app.APP.run_upscale_preview()

        self.assertFalse(ok)
        self.assertEqual(message, "Unsupported FlashVSR GPU")

    def test_flashvsr_hardware_warning_flags_pascal_gpu(self) -> None:
        message = system_status.flashvsr_hardware_warning("NVIDIA GeForce GTX 1050", (6, 1))

        self.assertIn("compute capability 6.1", message)
        self.assertIn("FlashVSR upscaling requires NVIDIA compute capability 7.5+", message)
        self.assertEqual(system_status.flashvsr_hardware_warning("NVIDIA GeForce RTX 2080", (7, 5)), "")

    def test_outpaint_hydration_does_not_pick_stale_newest_output_for_new_source(self) -> None:
        app.APP.settings["global"].update({"source": "input/new-source.mp4", "expand_outpaint": "true", "colorize": "false", "upscale": "false", "section_start": "0", "section_end": ""})
        stale = app.ROOT / "intermediate" / "outpainted" / "old-source_16x9_1280x704_outpainted.mp4"

        with mock.patch.object(server, "newest", return_value=stale):
            app.APP.hydrate_stage_inputs("upscale")

        self.assertEqual(app.APP.settings["recomp"]["outpainted_video"], "")

    def test_upscale_only_uses_pipeline_source(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
        # Pin every asserted field: settings are read from the live .ai_remaster_gui.json,
        # so values left over from a real project would otherwise leak into the command.
        app.APP.settings["upscale"].update(
            {
                "target_width": "1920",
                "target_height": "1080",
                "flashvsr_model": "FlashVSR-v1.1",
                "flashvsr_mode": "tiny",
                "flashvsr_tiled_vae": "true",
                "flashvsr_tiled_dit": "true",
                "flashvsr_color_fix": "true",
                "flashvsr_unload_dit": "false",
                "flashvsr_tile_size": "256",
                "flashvsr_tile_overlap": "24",
                "flashvsr_local_range": "11",
                "flashvsr_sparse_ratio": "2.0",
                "flashvsr_kv_ratio": "3.0",
                "blend_strength": "65",
                "chunk_seconds": "6",
                "overlap_frames": "8",
            }
        )

        app.APP.hydrate_stage_inputs("global")
        app.APP.settings["colour"]["manifest"] = "work/shots.csv"
        stage_keys = [stage.key for stage in app.APP.active_stages()]
        command = app.APP.command_for("upscale")

        self.assertEqual(stage_keys, ["upscale"])
        self.assertEqual(app.APP.settings["upscale"]["input_video"], "input/example.mp4")
        self.assertNotIn("--method", command)
        self.assertIn("--target-width", command)
        self.assertIn("1920", command)
        self.assertIn("--comfy-url", command)
        self.assertEqual(command[command.index("--flashvsr-model") + 1], "FlashVSR-v1.1")
        self.assertEqual(command[command.index("--flashvsr-mode") + 1], "tiny")
        self.assertIn("--flashvsr-tiled-vae", command)
        self.assertIn("--flashvsr-tiled-dit", command)
        self.assertIn("--flashvsr-color-fix", command)
        self.assertNotIn("--flashvsr-unload-dit", command)
        self.assertEqual(command[command.index("--flashvsr-tile-size") + 1], "256")
        self.assertEqual(command[command.index("--flashvsr-tile-overlap") + 1], "24")
        self.assertEqual(command[command.index("--flashvsr-local-range") + 1], "11")
        self.assertEqual(command[command.index("--flashvsr-sparse-ratio") + 1], "2.0")
        self.assertEqual(command[command.index("--flashvsr-kv-ratio") + 1], "3.0")
        self.assertEqual(command[command.index("--blend-strength") + 1], "65")
        self.assertEqual(command[command.index("--shot-manifest") + 1], "work/shots.csv")
        self.assertEqual(command[command.index("--chunk-seconds") + 1], "6")
        self.assertEqual(command[command.index("--overlap-frames") + 1], "8")
        self.assertIn("scripts\\upscale_video.py", " ".join(command))

    def test_upscale_can_request_flashvsr_unload_dit(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
        app.APP.settings["upscale"].update({"target_width": "1920", "target_height": "1080", "flashvsr_unload_dit": "true"})

        command = app.APP.command_for("upscale")

        self.assertIn("--flashvsr-unload-dit", command)

    def test_upscale_preview_clips_directly_from_selected_source_section(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            source = Path(tmp_text) / "source.mp4"
            source.write_bytes(b"placeholder")
            app.APP.settings["global"].update(
                {
                    "source": app.rel(source),
                    "expand_outpaint": "false",
                    "colorize": "false",
                    "upscale": "true",
                    "section_start": "1",
                    "section_end": "2",
                }
            )
            app.APP.settings["upscale"].update({"target_width": "1920", "target_height": "1080"})

            with (
                mock.patch.object(server, "ensure_source_section_clip") as ensure,
                mock.patch.object(server, "media_clip_path", side_effect=RuntimeError("clip probe")) as clip,
            ):
                ok, message = app.APP.run_upscale_preview()

        self.assertFalse(ok)
        ensure.assert_not_called()
        clip.assert_called_once()
        self.assertEqual(clip.call_args.args[0], source)
        self.assertAlmostEqual(clip.call_args.args[1], 1.0)
        self.assertAlmostEqual(clip.call_args.args[2], 2.0)
        self.assertIn("clip probe", message)
        self.assertNotIn("input does not exist", message)

    def test_upscale_preview_completion_does_not_hydrate_shot_detection(self) -> None:
        class FakeProcess:
            stdout = ["Wrote upscaled video: output/upscaled/previews/example.mp4\n"]

            def wait(self) -> int:
                return 0

        original_process = app.APP.process
        original_log = app.APP.log
        app.APP.process = FakeProcess()
        app.APP.log = []
        app.APP.running_stage = "Upscale Preview"
        app.APP.running_stage_key = "upscale"

        try:
            with mock.patch.object(app.APP, "hydrate_stage_inputs") as hydrate:
                app.APP._collect_output("upscale_preview")
        finally:
            app.APP.process = original_process
            log = app.APP.log
            app.APP.log = original_log
            app.APP.running_stage = ""
            app.APP.running_stage_key = ""

        self.assertNotIn(mock.call("shots"), hydrate.call_args_list)
        self.assertIn("Upscale preview ready.", log)
        self.assertFalse(any("Updated Shot Detection input" in line for line in log))

    def test_upscale_preview_state_uses_generated_clip_output_pair(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            source = Path(tmp_text) / "source.mp4"
            clip = Path(tmp_text) / "preview-clip.mp4"
            output = app.ROOT / "output" / "upscaled" / "previews" / "preview-clip_flashvsr_1920x1080_preview_6s.mp4"
            source.write_bytes(b"placeholder")
            clip.write_bytes(b"placeholder")
            app.APP.settings["global"].update({"source": app.rel(source), "expand_outpaint": "false", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"placeholder")
            app.APP.settings["upscale"].update({"preview_source": app.rel(clip), "preview_output": app.rel(output)})

            try:
                preview = app.APP.upscale_preview_state()
            finally:
                output.unlink(missing_ok=True)

        self.assertEqual(preview["source"], app.rel(clip))
        self.assertEqual(preview["output"], app.rel(output))
        self.assertEqual(preview["exists"], "true")

    def test_upscale_preview_state_promotes_finished_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            source = Path(tmp_text) / "source.mp4"
            preview_clip = Path(tmp_text) / "preview-clip.mp4"
            source.write_bytes(b"placeholder")
            preview_clip.write_bytes(b"placeholder")
            app.APP.settings["global"].update({"source": app.rel(source), "expand_outpaint": "false", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
            app.APP.settings["upscale"].update(
                {
                    "target_width": "1920",
                    "target_height": "1080",
                    "preview_source": app.rel(preview_clip),
                    "preview_output": "output/upscaled/previews/stale_preview.mp4",
                }
            )
            output = app.resolve(app.upscale_output_for(app.rel(source), app.APP.settings["upscale"]))
            preview_output = app.resolve(app.APP.settings["upscale"]["preview_output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            preview_output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"placeholder")
            preview_output.write_bytes(b"placeholder")

            try:
                preview = app.APP.upscale_preview_state()
            finally:
                output.unlink(missing_ok=True)
                preview_output.unlink(missing_ok=True)

        self.assertEqual(preview["source"], app.rel(source))
        self.assertEqual(preview["output"], app.rel(output))
        self.assertEqual(preview["kind"], "output")
        self.assertEqual(preview["title"], "Upscale Output")

    def test_upscale_preview_start_records_actual_clip_and_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            source = Path(tmp_text) / "source.mp4"
            clip = Path(tmp_text) / "clip.mp4"
            source.write_bytes(b"placeholder")
            clip.write_bytes(b"placeholder")
            app.APP.settings["global"].update(
                {
                    "source": app.rel(source),
                    "expand_outpaint": "false",
                    "colorize": "false",
                    "upscale": "true",
                    "section_start": "0",
                    "section_end": "",
                }
            )
            app.APP.settings["upscale"].update({"target_width": "1920", "target_height": "1080", "preview_seconds": "6"})

            class FakeProcess:
                stdout = []

                def poll(self):
                    return None

            with (
                mock.patch.object(server, "media_clip_path", return_value=clip),
                mock.patch.object(server.subprocess, "Popen", return_value=FakeProcess()),
                mock.patch.object(server.threading.Thread, "start", lambda _self: None),
            ):
                ok, message = app.APP.run_upscale_preview()

        self.assertTrue(ok, message)
        self.assertEqual(app.APP.settings["upscale"]["preview_source"], app.rel(clip))
        self.assertEqual(
            app.APP.settings["upscale"]["preview_output"],
            app.upscale_preview_output_for(app.rel(clip), app.APP.settings["upscale"]),
        )
        self.assertTrue(app.APP.settings["upscale"]["preview_output"].startswith("output/upscaled/previews/clip_upscalepreview_"))

    def test_upscale_only_ignores_stale_recomposition_input(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
        app.APP.settings["upscale"].update(
            {
                "method": "realbasicvsr",
                "input_video": "output/reassembled/old_intermediate_final.mp4",
                "output": "output/upscaled/old_intermediate_final_realbasicvsr_3840x2160.mp4",
                "target_width": "1920",
                "target_height": "1080",
            }
        )

        command = app.APP.command_for("upscale")

        self.assertIn("--input", command)
        self.assertEqual(command[command.index("--input") + 1], "input/example.mp4")
        self.assertEqual(command[command.index("--output") + 1], app.upscale_output_for("input/example.mp4", app.APP.settings["upscale"]))
        self.assertNotIn("--method", command)

    def test_upscale_ignores_stale_backend_method_setting(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
        app.APP.settings["upscale"].update({"method": "realbasicvsr", "target_width": "1920", "target_height": "1080"})

        command = app.APP.command_for("upscale")

        self.assertNotIn("--method", command)
        self.assertNotIn("--realbasicvsr-repo", command)
        self.assertIn("--comfy-url", command)
        self.assertEqual(command[command.index("--output") + 1], app.upscale_output_for("input/example.mp4", app.APP.settings["upscale"]))

    def test_flashvsr_prompt_uses_video_helper_load_and_combine_nodes(self) -> None:
        args = argparse.Namespace(
            flashvsr_model="FlashVSR-v1.1",
            flashvsr_mode="tiny",
            flashvsr_scale=2,
            flashvsr_tiled_vae=True,
            flashvsr_tiled_dit=True,
            flashvsr_unload_dit=True,
            flashvsr_color_fix=True,
            flashvsr_tile_size=512,
            flashvsr_tile_overlap=24,
            flashvsr_sparse_ratio=2.0,
            flashvsr_kv_ratio=3.0,
            flashvsr_local_range=9,
            flashvsr_seed=123,
        )
        info = {
            "FlashVSRInitPipe": {
                "input": {
                    "required": {
                        "model": (["FlashVSR", "FlashVSR-v1.1"],),
                        "mode": (["tiny", "tiny-long", "full"],),
                        "alt_vae": (["none"], {"default": "none"}),
                        "force_offload": ("BOOLEAN", {"default": True}),
                        "precision": (["fp16", "bf16"], {"default": "bf16"}),
                        "device": (["auto"], {"default": "auto"}),
                        "attention_mode": (["sparse_sage_attention", "block_sparse_attention"], {"default": "sparse_sage_attention"}),
                    }
                }
            },
            "FlashVSRNodeAdv": {
                "input": {
                    "required": {
                        "pipe": ("PIPE",),
                        "frames": ("IMAGE",),
                        "scale": ("INT", {"default": 4}),
                        "color_fix": ("BOOLEAN", {"default": True}),
                        "tiled_vae": ("BOOLEAN", {"default": True}),
                        "tiled_dit": ("BOOLEAN", {"default": True}),
                        "tile_size": ("INT", {"default": 256}),
                        "tile_overlap": ("INT", {"default": 24}),
                        "unload_dit": ("BOOLEAN", {"default": False}),
                        "sparse_ratio": ("FLOAT", {"default": 2.0}),
                        "kv_ratio": ("FLOAT", {"default": 3.0}),
                        "local_range": ("INT", {"default": 11}),
                        "seed": ("INT", {"default": 0}),
                    }
                }
            },
        }

        prompt = upscale_video.flashvsr_prompt("example.mp4", 24.0, args, "arp_upscale/example", info)

        self.assertEqual(prompt["1"]["class_type"], "VHS_LoadVideo")
        self.assertEqual(prompt["2"]["class_type"], "FlashVSRInitPipe")
        self.assertEqual(prompt["2"]["inputs"]["model"], "FlashVSR-v1.1")
        self.assertEqual(prompt["2"]["inputs"]["mode"], "tiny")
        self.assertEqual(prompt["2"]["inputs"]["precision"], "fp16")
        self.assertEqual(prompt["2"]["inputs"]["force_offload"], True)
        self.assertEqual(prompt["3"]["class_type"], "FlashVSRNodeAdv")
        self.assertEqual(prompt["3"]["inputs"]["pipe"], ["2", 0])
        self.assertEqual(prompt["3"]["inputs"]["frames"], ["1", 0])
        self.assertEqual(prompt["3"]["inputs"]["scale"], 2)
        self.assertEqual(prompt["3"]["inputs"]["color_fix"], True)
        self.assertEqual(prompt["3"]["inputs"]["tiled_vae"], True)
        self.assertEqual(prompt["3"]["inputs"]["tiled_dit"], True)
        self.assertEqual(prompt["3"]["inputs"]["tile_size"], 512)
        self.assertEqual(prompt["3"]["inputs"]["tile_overlap"], 24)
        self.assertEqual(prompt["3"]["inputs"]["unload_dit"], True)
        self.assertEqual(prompt["3"]["inputs"]["sparse_ratio"], 2.0)
        self.assertEqual(prompt["3"]["inputs"]["kv_ratio"], 3.0)
        self.assertEqual(prompt["3"]["inputs"]["local_range"], 9)
        self.assertEqual(prompt["3"]["inputs"]["seed"], 123)
        self.assertEqual(prompt["4"]["class_type"], "VHS_VideoCombine")
        self.assertEqual(prompt["4"]["inputs"]["images"], ["3", 0])
        self.assertNotIn("audio", prompt["4"]["inputs"])

    def test_upscale_signature_only_records_advanced_knobs_when_changed(self) -> None:
        parser = upscale_video.build_parser()
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            source = Path(tmp_text) / "source.mp4"
            source.write_bytes(b"placeholder")

            defaults = upscale_video.signature(parser.parse_args(["--input", str(source)]), source, 1920, 1080)
            tweaked = upscale_video.signature(parser.parse_args(["--input", str(source), "--flashvsr-local-range", "9", "--flashvsr-tile-size", "512"]), source, 1920, 1080)

        # Advanced knobs at their defaults reproduce the legacy single-node output, so
        # they must not invalidate signatures written before the knobs existed.
        for name in upscale_video.ADVANCED_DEFAULTS:
            self.assertNotIn(name, defaults)
        self.assertEqual(tweaked["flashvsr_local_range"], 9)
        self.assertEqual(tweaked["flashvsr_tile_size"], 512)
        self.assertNotIn("flashvsr_kv_ratio", tweaked)

    def test_upscale_chunk_ranges_overlap_without_dropping_frames(self) -> None:
        ranges = upscale_video.chunk_ranges(total_frames=100, fps=10.0, chunk_seconds=3.0, overlap_frames=4)

        self.assertEqual(ranges, [(0, 30, 0), (26, 60, 4), (56, 90, 4), (86, 100, 4)])

    def test_upscale_chunk_ranges_use_single_prompt_when_clip_fits(self) -> None:
        self.assertEqual(upscale_video.chunk_ranges(total_frames=100, fps=25.0, chunk_seconds=6.0, overlap_frames=8), [(0, 0, 100)])

    def test_upscale_chunk_cache_key_keeps_same_section_start_when_end_extends(self) -> None:
        first = Path("intermediate/source_sections/example_0000012000_0000024000.mp4")
        extended = Path("intermediate/source_sections/example_0000012000_0000060000.mp4")
        moved_start = Path("intermediate/source_sections/example_0000045000_0000060000.mp4")

        self.assertEqual(upscale_video.upscale_chunk_source_key(first), upscale_video.upscale_chunk_source_key(extended))
        self.assertNotEqual(upscale_video.upscale_chunk_source_key(first), upscale_video.upscale_chunk_source_key(moved_start))

    def test_upscale_chunk_signature_compatibility_ignores_cache_path_and_mtime_only(self) -> None:
        parser = upscale_video.build_parser()
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            old_chunk = folder / "old" / "input_0000.mp4"
            new_chunk = folder / "new" / "input_0000.mp4"
            old_chunk.parent.mkdir()
            new_chunk.parent.mkdir()
            old_chunk.write_bytes(b"same")
            new_chunk.write_bytes(b"same")
            args = parser.parse_args(["--input", str(new_chunk)])
            old_sig = upscale_video.signature(args, old_chunk, 1920, 1080)
            new_sig = upscale_video.signature(args, new_chunk, 1920, 1080)
            old_sig["source_fingerprint"]["mtime_ns"] = 1
            new_sig["source_fingerprint"]["mtime_ns"] = 2

        self.assertTrue(upscale_video.compatible_upscale_chunk_signature(old_sig, new_sig))
        changed = copy.deepcopy(new_sig)
        changed["source_fingerprint"]["sha256"] = "different"
        self.assertFalse(upscale_video.compatible_upscale_chunk_signature(old_sig, changed))

    def test_upscale_runs_after_recomposition_when_processing_is_enabled(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "true", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
        app.APP.settings["outpaint"].update({"target_aspect": "16:9", "target_height": "720", "crop_left": "0", "crop_right": "0", "crop_top": "0", "crop_bottom": "0"})

        with mock.patch.object(server, "newest", return_value=None):
            app.APP.hydrate_stage_inputs("global")
        stage_keys = [stage.key for stage in app.APP.active_stages()]
        outpainted = app.outpaint_output_for("input/example.mp4", "16:9", "720")
        outpainted_path = app.resolve(outpainted)
        outpainted_path.parent.mkdir(parents=True, exist_ok=True)
        outpainted_path.write_bytes(b"placeholder")
        try:
            app.APP.hydrate_stage_inputs("outpaint")
        finally:
            outpainted_path.unlink(missing_ok=True)

        self.assertEqual(stage_keys, ["outpaint", "recomp", "upscale"])
        self.assertTrue(app.APP.settings["upscale"]["input_video"].startswith("output/reassembled/"))

    def test_new_outpaint_source_does_not_hydrate_empty_manifest_as_repo_root(self) -> None:
        app.APP.settings["global"].update({"source": "input/new-source.mp4", "expand_outpaint": "true", "colorize": "true", "section_start": "0", "section_end": ""})
        app.APP.settings.setdefault("outpaint", {}).update({"target_aspect": "16:9", "target_height": "480"})

        with mock.patch.object(server, "newest", return_value=None):
            app.APP.hydrate_stage_inputs("global")

        self.assertEqual(app.APP.settings["shots"]["outpainted_video"], "")
        self.assertEqual(app.APP.settings["references"]["manifest"], "")
        self.assertEqual(app.APP.settings["colour"]["manifest"], "")
        self.assertEqual(app.APP.settings["recomp"]["manifest"], "")
        self.assertEqual(app.APP.expected_outputs("shots"), [])

    def test_outpaint_progress_surfaces_active_comfy_chunk_globally(self) -> None:
        original_log = app.APP.log
        app.APP.settings["global"].update({"expand_outpaint": "true", "colorize": "false", "upscale": "false"})
        app.APP.running_stage_key = "outpaint"
        app.APP.running_stage = "Outpainting"
        app.APP.run_started_at = time.time() - 120
        app.APP.log = [
            "Outpaint chunk 1/3: frames 0-480",
            "Wrote raw Comfy chunk 1: chunk_0000.mp4",
            "Outpaint chunk 2/3: frames 472-952",
            "Sending prompt nodes: {'5076': 'VHS_VideoCombine'}",
            "Queued ComfyUI prompt: prompt-id",
        ]

        try:
            stage_progress = app.APP.estimate_running_progress()
            global_progress = app.APP.phase_progress()["global"]
        finally:
            app.APP.running_stage_key = ""
            app.APP.running_stage = ""
            app.APP.run_started_at = 0.0
            app.APP.log = original_log

        self.assertIn("Chunk 2/3 rendering in ComfyUI (1 done), ETA", stage_progress["label"])
        self.assertIn("Chunk 2/3 rendering in ComfyUI", global_progress["label"])

    def test_upscale_progress_surfaces_active_chunk_with_eta(self) -> None:
        original_log = app.APP.log
        app.APP.settings["global"].update({"expand_outpaint": "false", "colorize": "false", "upscale": "true"})
        app.APP.running_stage_key = "upscale"
        app.APP.running_stage = "Upscaling"
        app.APP.run_started_at = time.time() - 180
        app.APP.log = [
            "Splitting upscaling into 12 chunk(s): 6s chunks, 8 overlap frame(s)",
            "Upscale chunk 1/12: frames 0-144, trim 0",
            "Wrote upscaled chunk: chunk_0001.mp4",
            "Upscale chunk 2/12: frames 136-288, trim 8",
            "Queued ComfyUI prompt: prompt-id",
        ]

        try:
            progress = app.APP.estimate_running_progress()
        finally:
            app.APP.running_stage_key = ""
            app.APP.running_stage = ""
            app.APP.run_started_at = 0.0
            app.APP.log = original_log

        self.assertIn("Upscale chunk 2/12 rendering in ComfyUI (1 done), ETA", progress["label"])
        self.assertGreater(progress["percent"], 10)
        self.assertLess(progress["percent"], 100)

    def test_upscale_progress_reports_stitching_after_chunks_complete(self) -> None:
        original_log = app.APP.log
        app.APP.running_stage_key = "upscale"
        app.APP.running_stage = "Upscaling"
        app.APP.run_started_at = time.time() - 300
        app.APP.log = [
            "Splitting upscaling into 2 chunk(s): 6s chunks, 8 overlap frame(s)",
            "Upscale chunk 1/2: frames 0-144, trim 0",
            "Wrote upscaled chunk: chunk_0001.mp4",
            "Upscale chunk 2/2: frames 136-288, trim 8",
            "Wrote upscaled chunk: chunk_0002.mp4",
            "Stitching upscaled chunks: 2 chunk(s)",
        ]

        try:
            progress = app.APP.estimate_running_progress()
        finally:
            app.APP.running_stage_key = ""
            app.APP.running_stage = ""
            app.APP.run_started_at = 0.0
            app.APP.log = original_log

        self.assertEqual(progress["label"], "Upscale chunks complete, stitching")
        self.assertLess(progress["percent"], 100)

    def test_model_download_progress_surfaces_percent(self) -> None:
        original_log = app.APP.log
        app.APP.running_stage_key = "outpaint"
        app.APP.running_stage = "Outpainting"
        app.APP.run_started_at = time.time() - 30
        app.APP.log = [
            "Checking model: repo/model.safetensors",
            "Downloading model: repo/model.safetensors (4.0 GB)",
            "Download progress: 42%, ETA 3:15",
        ]

        try:
            progress = app.APP.estimate_running_progress()
        finally:
            app.APP.running_stage_key = ""
            app.APP.running_stage = ""
            app.APP.run_started_at = 0.0
            app.APP.log = original_log

        self.assertEqual(progress["label"], "Downloading model 42%, ETA 3:15")
        self.assertGreater(progress["percent"], 10)
        self.assertLess(progress["percent"], 35)

    def test_reference_regeneration_model_download_progress_surfaces_eta(self) -> None:
        original_log = app.APP.log
        app.APP.running_stage_key = "references"
        app.APP.running_stage = "Reference Generation"
        app.APP.running_reference_index = 0
        app.APP.run_started_at = time.time() - 30
        app.APP.log = [
            "Qwen mode: gguf; one source image only; extra references are converted to text guidance when enabled.",
            "Checking model: unsloth/Qwen-Image-Edit-2511-GGUF/qwen-image-edit-2511-Q4_K_M.gguf",
            "Downloading model: unsloth/Qwen-Image-Edit-2511-GGUF/qwen-image-edit-2511-Q4_K_M.gguf (18.2 GB)",
            "Download progress: 7%, ETA 12:04",
        ]

        try:
            progress = app.APP.estimate_running_progress()
        finally:
            app.APP.running_stage_key = ""
            app.APP.running_stage = ""
            app.APP.running_reference_index = None
            app.APP.run_started_at = 0.0
            app.APP.log = original_log

        self.assertEqual(progress["label"], "Downloading model 7%, ETA 12:04")
        self.assertGreater(progress["percent"], 5)
        self.assertLess(progress["percent"], 30)

    def test_reference_regeneration_model_download_heartbeat_replaces_stale_eta(self) -> None:
        original_log = app.APP.log
        app.APP.running_stage_key = "references"
        app.APP.running_stage = "Reference Generation"
        app.APP.running_reference_index = 0
        app.APP.run_started_at = time.time() - 180
        app.APP.log = [
            "Downloading model: Comfy-Org/Qwen-Image_ComfyUI/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors (8.7 GB)",
            "Download progress: 80%, ETA 0:01",
            "Download progress: 80%, still working",
        ]

        try:
            progress = app.APP.estimate_running_progress()
        finally:
            app.APP.running_stage_key = ""
            app.APP.running_stage = ""
            app.APP.running_reference_index = None
            app.APP.run_started_at = 0.0
            app.APP.log = original_log

        self.assertEqual(progress["label"], "Downloading model 80%, still working")

    def test_reference_regeneration_model_install_progress_surfaces_percent(self) -> None:
        original_log = app.APP.log
        app.APP.running_stage_key = "references"
        app.APP.running_stage = "Reference Generation"
        app.APP.running_reference_index = 0
        app.APP.run_started_at = time.time() - 180
        app.APP.log = [
            "Downloading model: Comfy-Org/Qwen-Image_ComfyUI/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors (8.7 GB)",
            "Download progress: 100%",
            "Installing model: D:\\dtaddis\\ai-remaster-pipeline\\tools\\comfyui\\models\\text_encoders\\qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "Install progress: 52%",
        ]

        try:
            progress = app.APP.estimate_running_progress()
        finally:
            app.APP.running_stage_key = ""
            app.APP.running_stage = ""
            app.APP.running_reference_index = None
            app.APP.run_started_at = 0.0
            app.APP.log = original_log

        self.assertEqual(progress["label"], "Installing model 52%")

    def test_reference_progress_ignores_previous_stage_writes(self) -> None:
        original_log = app.APP.log
        app.APP.running_stage_key = "references"
        app.APP.running_stage = "Reference Generation"
        app.APP.run_started_at = time.time() - 60
        app.APP.log = [
            "Wrote source frame 0000: source.png",
            "Wrote source frame 0001: source.png",
            "Wrote manifest: refs.csv",
            r"> python scripts\qwen_colorize_references.py --manifest refs.csv",
            "Rows: 11",
            "Colorize 0000: source.png -> color.png",
            "Queued ComfyUI prompt: prompt-id",
            "Wrote color.png",
        ]

        try:
            progress = app.APP.estimate_running_progress()
        finally:
            app.APP.running_stage_key = ""
            app.APP.running_stage = ""
            app.APP.run_started_at = 0.0
            app.APP.log = original_log

        self.assertEqual(progress["label"], "1/11 references")

    def test_colour_progress_ignores_previous_process_finished_lines(self) -> None:
        original_log = app.APP.log
        app.APP.running_stage_key = "colour"
        app.APP.running_stage = "Colorization"
        app.APP.run_started_at = time.time() - 60
        app.APP.log = [
            r"> python scripts\qwen_colorize_references.py --manifest refs.csv",
            "Wrote reference.png",
            "Process finished with exit code 0.",
            r"> python scripts\colorize_video.py --manifest refs.csv --method both",
            "Colorize segment 11/11 with deepexemplar: frames 1343-1440 using ref.png",
            "Wrote colorized video: deepexemplar.mp4",
            "Colorize segment 4/11 with colormnet: frames 527-632 using ref.png",
            "Sending prompt nodes: {'1': 'VHS_LoadVideo'}",
        ]

        try:
            progress = app.APP.estimate_running_progress()
        finally:
            app.APP.running_stage_key = ""
            app.APP.running_stage = ""
            app.APP.run_started_at = 0.0
            app.APP.log = original_log

        self.assertEqual(progress["label"], "Colorizing segment 4/11")

    def test_blank_project_defaults_outpainting_visible(self) -> None:
        with mock.patch.object(app, "SETTINGS_FILE", Path("missing-settings.json")), mock.patch.object(app, "newest", return_value=None):
            settings = app.load_settings()

        self.assertEqual(settings["global"]["expand_outpaint"], "true")
        self.assertEqual(settings["outpaint"]["target_height"], "source")
        self.assertEqual(settings["outpaint"]["seed_qwen_guides"], "false")
        self.assertNotIn("outpaint_lora", settings["outpaint"])

    def test_blank_loaded_project_defaults_outpainting_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            path = Path(tmp_text) / "blank.arpp"
            payload = app.project_payload(app.APP.settings)
            payload["settings"] = {"global": {"source": "", "expand_outpaint": "false", "colorize": "true"}}
            path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = app.read_project_file(path)

        self.assertEqual(loaded["global"]["expand_outpaint"], "true")

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

    def test_preview_cache_name_uses_short_stem_and_hash(self) -> None:
        source = Path(r"C:\Users\mdamberger\AppData\Local\Programs\ai-remaster-pipeline\intermediate\source_sections\DrWho_Wheel_in_space_0000000000_0000154040.mp4")

        identity = app.aspect_preview_identity(source, 123, 456, "16:9", (8, 7, 0, 0), 0.0)
        preview = Path(r"C:\Users\mdamberger\AppData\Local\Programs\ai-remaster-pipeline\.cache\aspect_previews") / app.aid.artifact_name(
            app.aid.source_word(source.name),
            "aspectpreview",
            identity,
            "jpg",
        )

        self.assertTrue(preview.name.startswith("DrWho_aspectpreview_"))
        self.assertNotIn("16x9", preview.name)
        self.assertNotIn("crop", preview.name)
        self.assertLess(len(str(preview)), 240)

    def test_aspect_preview_cache_writes_signature_sidecar(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            source.write_bytes(b"video placeholder")
            preview_dir = folder / "aspect_previews"
            identity = app.aspect_preview_identity(source, source.stat().st_size, source.stat().st_mtime_ns, "16:9", (8, 7, 0, 0), 0.0)
            target = preview_dir / app.aid.artifact_name(app.aid.source_word(source.name), "aspectpreview", identity, "jpg")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"preview")

            with mock.patch.dict(
                app.aspect_preview_cached.__globals__,
                {"ASPECT_PREVIEW_DIR": preview_dir, "extract_video_frame_at": mock.Mock(return_value=app.rel(target))},
            ):
                preview = app.aspect_preview_cached(str(source), source.stat().st_size, source.stat().st_mtime_ns, "16:9", (8, 7, 0, 0), 0.0)
                sidecar = json.loads((Path(app.resolve(preview)).with_suffix(".jpg.sig.json")).read_text(encoding="utf-8-sig"))

        self.assertTrue(Path(preview).name.startswith("source_aspectpreview_"))
        self.assertNotIn("16x9", Path(preview).name)
        self.assertEqual(sidecar["identity"]["aspect"], "16:9")
        self.assertEqual(sidecar["identity"]["crop"], [8, 7, 0, 0])

    def test_source_preview_analysis_regenerates_without_cache_clear_attribute(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            source.write_bytes(b"video placeholder")
            preview_dir = folder / "previews"
            signature = (str(source), source.stat().st_size, source.stat().st_mtime_ns)

            def fake_generate_video_previews(_source, target_dir, progress=None, duration=None):
                target_dir.mkdir(parents=True, exist_ok=True)
                for index in range(app.source_previews_for_analysis.__globals__["SOURCE_PREVIEW_COUNT"]):
                    (target_dir / f"preview_{index}.jpg").write_bytes(b"preview")

            with mock.patch.dict(
                app.source_previews_for_analysis.__globals__,
                {"PREVIEW_DIR": preview_dir, "generate_video_previews": fake_generate_video_previews},
            ):
                previews = app.source_previews_for_analysis(signature, {"duration": "1"}, lambda _percent, _message: None)

        self.assertEqual(len(previews), app.source_previews_for_analysis.__globals__["SOURCE_PREVIEW_COUNT"])

    def test_frame_preview_reuses_fresh_existing_thumbnail(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            source.write_bytes(b"video placeholder")
            target_dir = folder / "previews"
            target_dir.mkdir()
            target = target_dir / f"{app.safe_preview_name(source)}_thumb.jpg"
            target.write_bytes(b"cached thumbnail")

            with mock.patch.object(server, "local_tool", return_value="ffmpeg"), mock.patch.object(server.subprocess, "run") as run:
                self.assertEqual(app.extract_video_frame_at(source, target_dir, "thumb", 0), app.rel(target))

            run.assert_not_called()

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

    def test_media_status_endpoint_reports_existing_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            media_file = Path(tmp_text) / "preview.png"
            media_file.write_bytes(b"preview")
            server = app.create_server("127.0.0.1", 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/api/media-status?path={urllib.parse.quote(app.rel(media_file))}"
                with urllib.request.urlopen(url, timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()

        self.assertTrue(payload["ok"])
        self.assertTrue(payload["exists"])
        self.assertGreater(payload["mtime"], 0)

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

    def test_quit_endpoint_acknowledges_and_stops_server(self) -> None:
        server = app.create_server("127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/api/quit",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertTrue(payload["ok"])
            # request_quit() shuts the server down a moment after answering.
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        finally:
            server.shutdown()
            server.server_close()

    def test_quit_stops_running_stage_and_blocks_relaunch(self) -> None:
        class FakeRunningProcess:
            returncode = None

            def poll(self):
                return None

        previous_process = app.APP.process
        previous_quitting = app.APP.quitting
        app.APP.process = FakeRunningProcess()
        app.APP.quitting = False
        try:
            with mock.patch.object(server, "terminate_process_tree") as kill:
                app.APP.stop_for_quit()

            kill.assert_called_once_with(app.APP.process)
            self.assertTrue(app.APP.quitting)
            # While quitting, a Run All queue (or any caller) must not relaunch a stage.
            ok, message = app.APP.run_stage("outpaint")
            self.assertFalse(ok)
            self.assertIn("shutting down", message)
        finally:
            app.APP.process = previous_process
            app.APP.quitting = previous_quitting


if __name__ == "__main__":
    unittest.main()
