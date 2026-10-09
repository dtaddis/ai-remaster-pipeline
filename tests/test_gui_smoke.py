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
import artifact_ids as aid  # noqa: E402
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
import master_encode  # noqa: E402
import stabilize_video  # noqa: E402
import upscale_video  # noqa: E402

from ai_remaster_gui import app
from ai_remaster_gui import cache
from ai_remaster_gui import config
from ai_remaster_gui import lifecycle
from ai_remaster_gui import media
from ai_remaster_gui import outpaint_guides
from ai_remaster_gui import paths
from ai_remaster_gui import project_io
from ai_remaster_gui import references
from ai_remaster_gui import runtime_settings
from ai_remaster_gui import sam_masks
from ai_remaster_gui import server
from ai_remaster_gui import stage_stamps
from ai_remaster_gui import system_status


def fake_section_ffprobe(command, **_kwargs):
    """ffprobe stand-in for section trims: one audio stream, no readable timing."""
    return subprocess.CompletedProcess(command, 0, stdout='{"streams":[{"index":1}]}', stderr="")


def fake_section_trim(commands: list[list[str]], release: threading.Event | None = None, fail: bool = False):
    """Popen stand-in for the section trim: reports halfway, then writes the partial clip."""

    class FakeTrim:
        def __init__(self, command, **_kwargs):
            commands.append(command)
            self.command = command
            self.returncode = None

        @property
        def stdout(self):
            yield "out_time_us=N/A\n"
            yield "out_time_us=6000000\n"
            if release is not None:
                release.wait(5)
            if fail:
                yield Path(self.command[self.command.index("-i") + 1]).name + ": Invalid data found\n"
            else:
                Path(self.command[-1]).write_bytes(b"trimmed")
            yield "progress=end\n"

        def wait(self):
            self.returncode = 1 if fail else 0
            return self.returncode

        def poll(self):
            return self.returncode

    return FakeTrim


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
        # Stage runs record completion stamps; keep them out of the real .cache.
        patcher = mock.patch.object(stage_stamps, "STAMP_DIR", tmp_settings.parent / "stage_stamps")
        patcher.start()
        cls.addClassCleanup(patcher.stop)
        app.APP.settings = server.load_settings()
        cls._pristine_settings = copy.deepcopy(app.APP.settings)

    def setUp(self) -> None:
        app.APP.settings = copy.deepcopy(self._pristine_settings)
        with app.APP.source_analysis_lock:
            app.APP.source_tone_default_keys.clear()
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
            "stabilize": "stabilize_video.py",
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
        self.assertEqual(active(F, F, F, T), ["shots", "upscale"])
        self.assertEqual(active(F, F, T, F), ["audio"])
        self.assertEqual(active(F, F, F, F), [])

    def test_every_processing_stage_offers_runpod_compute(self) -> None:
        defaults = app.default_settings()
        for key in ("cleanup", "stabilize", "outpaint", "shots", "references", "colour", "recomp", "audio", "upscale"):
            self.assertEqual(defaults[key]["compute"], "local", key)
            stage = next(item for item in app.STAGES if item.key == key)
            field = next(item for item in stage.fields if item[0] == "compute")
            self.assertIn("runpod", field[2], key)

    def test_runpod_compute_wraps_stage_and_redacts_secrets(self) -> None:
        self._populate_full_pipeline_settings()
        app.APP.settings["cloud"].update({"runpod_api_key": "secret-runpod", "huggingface_token": "secret-hf"})
        app.APP.settings["cleanup"]["compute"] = "runpod"
        command = app.APP.command_for("cleanup")
        self.assertEqual(Path(command[2]).name, "runpod_stage.py")
        script_names = [Path(part).name for part in command if part.endswith(".py")]
        self.assertIn("cleanup_video.py", script_names)
        self.assertNotIn("master_encode.py", script_names)
        self.assertIn("--intermediate-profile", command)
        self.assertNotIn("secret-runpod", command)
        self.assertNotIn("secret-hf", command)
        self.assertIn("--settings-file", command)

    def test_modern_intermediate_master_profiles(self) -> None:
        self.assertEqual(app.default_settings()["cloud"]["intermediate_format"], "high")
        for profile, crf in (("low", "36"), ("medium", "27"), ("high", "18")):
            args = master_encode.codec_args(profile)
            self.assertIn("libvpx-vp9", args)
            self.assertIn(crf, args)
            self.assertNotIn("libx265", args)
        self.assertIn("ffv1", master_encode.codec_args("lossless"))
        self.assertNotIn("libx265", master_encode.codec_args("lossless"))
        self.assertEqual(master_encode.codec_args("hevc_high"), master_encode.codec_args("high"))

    def test_intermediate_profile_is_passed_into_video_producers(self) -> None:
        self._populate_full_pipeline_settings()
        app.APP.settings["cloud"]["intermediate_format"] = "lossless"
        self.assertNotIn("encoder", app.default_settings()["stabilize"])
        for stage in ("cleanup", "stabilize", "outpaint", "colour", "upscale"):
            command = app.APP.command_for(stage)
            self.assertIn("--intermediate-profile", command, stage)
            self.assertEqual(command[command.index("--intermediate-profile") + 1], "lossless", stage)
        self.assertNotIn("--intermediate-profile", app.APP.command_for("recomp"))

    def test_legacy_intermediate_profile_is_canonicalized_for_video_producers(self) -> None:
        self._populate_full_pipeline_settings()
        app.APP.settings["cloud"]["intermediate_format"] = "hevc_high"
        for stage in ("cleanup", "stabilize", "outpaint", "colour", "upscale"):
            command = app.APP.command_for(stage)
            self.assertEqual(command[command.index("--intermediate-profile") + 1], "high", stage)
        args = upscale_video.build_parser().parse_args(["--input", "input/example.mp4", "--intermediate-profile", "hevc_high"])
        self.assertEqual(args.intermediate_profile, "high")

    def test_cleanup_is_optional_and_runs_before_outpainting(self) -> None:
        app.APP.settings["global"].update(
            {"cleanup": "true", "expand_outpaint": "true", "colorize": "true", "upscale": "false", "add_soundtrack": "false"}
        )

        self.assertEqual(
            [stage.key for stage in app.APP.active_stages()],
            ["cleanup", "outpaint", "shots", "references", "colour", "recomp"],
        )

    def test_stabilization_runs_after_cleanup_and_before_outpainting(self) -> None:
        app.APP.settings["global"].update(
            {
                "source": "input/example.mp4",
                "cleanup": "true",
                "stabilize": "true",
                "expand_outpaint": "true",
                "colorize": "true",
                "upscale": "false",
                "add_soundtrack": "false",
            }
        )

        self.assertEqual(
            [stage.key for stage in app.APP.active_stages()],
            ["cleanup", "stabilize", "outpaint", "shots", "references", "colour", "recomp"],
        )

    def test_stabilization_uses_source_directly_when_cleanup_is_disabled(self) -> None:
        app.APP.settings["global"].update(
            {
                "source": "input/example.mp4",
                "cleanup": "false",
                "stabilize": "true",
                "section_start": "0",
                "section_end": "",
            }
        )

        command = app.APP.command_for("stabilize")

        self.assertEqual(command[command.index("--source") + 1], "input/example.mp4")
        self.assertEqual(command[command.index("--intermediate-profile") + 1], "high")

    def test_stabilization_output_name_matches_gui_and_producer(self) -> None:
        source = app.resolve_video_source("input/example.mp4")
        values = app.APP.settings["stabilize"]
        args = stabilize_video.build_parser().parse_args(["--source", str(source)])

        self.assertEqual(app.stabilize_output_for(str(source), values), app.rel(stabilize_video.default_output(source, args)))

    def test_continuous_stabilization_adds_single_shot_flag_and_changes_output_identity(self) -> None:
        values = dict(app.APP.settings["stabilize"])
        scene_output = app.stabilize_output_for("input/example.mp4", values)
        values["scene_aware"] = "false"
        continuous_output = app.stabilize_output_for("input/example.mp4", values)
        app.APP.settings["stabilize"].update(values)

        command = app.APP.command_for("stabilize")

        self.assertIn("--single-shot", command)
        self.assertNotEqual(scene_output, continuous_output)

    def test_finished_stabilization_routes_to_outpaint(self) -> None:
        app.APP.settings["global"].update(
            {"source": "input/example.mp4", "cleanup": "false", "stabilize": "true", "expand_outpaint": "true", "colorize": "false"}
        )
        stabilized = app.stabilize_output_for("input/example.mp4", app.APP.settings["stabilize"])
        stabilized_path = app.resolve(stabilized)
        stabilized_path.parent.mkdir(parents=True, exist_ok=True)
        stabilized_path.write_bytes(b"placeholder")
        try:
            app.APP.hydrate_stage_inputs("stabilize")
            command = app.APP.command_for("outpaint")
        finally:
            stabilized_path.unlink(missing_ok=True)

        self.assertEqual(command[command.index("--source") + 1], stabilized)
        self.assertEqual(app.APP.settings["recomp"]["source"], stabilized)

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

        self.assertEqual([stage.key for stage in app.APP.active_stages()], ["cleanup", "shots", "upscale"])
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

    def test_stabilization_viewer_falls_back_to_source_when_cleanup_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "sequence-master.mkv"
            preview = folder / "sequence-preview.mp4"
            source.write_bytes(b"master")
            preview.write_bytes(b"preview")
            app.APP.settings["global"].update(
                {
                    "source": app.rel(source),
                    "source_sequence_preview": app.rel(preview),
                    "cleanup": "false",
                    "stabilize": "true",
                    "section_start": "1",
                    "section_end": "2",
                }
            )

            comparison = app.APP.stabilization_comparison_state()

        self.assertEqual(comparison["source"], app.rel(preview))
        self.assertEqual(comparison["master_source"], app.rel(source))
        self.assertEqual(comparison["source_exists"], "true")
        self.assertEqual(comparison["exists"], "false")

    def test_stabilization_page_preserves_its_video_during_polling(self) -> None:
        core = (app.ROOT / "ai_remaster_gui" / "static" / "js" / "core.js").read_text(encoding="utf-8")

        self.assertEqual(core.count("['cleanup', 'stabilize', 'outpaint'"), 2)

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

    def test_upscale_page_exposes_method_specific_controls(self) -> None:
        comparison = (app.ROOT / "ai_remaster_gui" / "static" / "js" / "render-recomp.js").read_text(encoding="utf-8")

        self.assertIn("const fieldKeys = ['method']", comparison)
        self.assertIn("if (method === 'ltx25')", comparison)
        self.assertIn("'ltx25_source_fidelity', 'ltx25_lora_strength', 'ltx25_guidance_scale', 'ltx25_seed'", comparison)
        self.assertIn("'ltx25_prompt', 'ltx25_negative_prompt'", comparison)

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

    def test_cleanup_migrates_stale_dearchive_prompt_variants(self) -> None:
        settings = app.default_settings()
        settings["cleanup"]["prompt"] = (
            "A modern, high-resolution video shot in vivid color, sharp detail, contemporary cinematography."
        )
        settings["cleanup"]["negative_prompt"] = "black and white, old, grainy, desaturated, sepia"

        normalized = runtime_settings.normalize_settings(settings, include_newest_source=False)

        self.assertEqual(normalized["cleanup"]["prompt"], app.CLEANUP_PROMPT)
        self.assertEqual(normalized["cleanup"]["negative_prompt"], app.CLEANUP_NEGATIVE_PROMPT)

    def test_outpaint_ui_exposes_a_persistent_custom_output_height(self) -> None:
        outpaint_stage = next(stage for stage in app.STAGES if stage.key == "outpaint")
        height_field = next(field for field in outpaint_stage.fields if field[0] == "target_height")
        helpers = (app.ROOT / "ai_remaster_gui" / "static" / "js" / "render-helpers.js").read_text(encoding="utf-8")

        self.assertEqual(
            height_field,
            ("target_height", "Output height", "select:source|480|544|576|720|768|1080|custom", "source"),
        )
        self.assertEqual(app.default_settings()["outpaint"]["custom_target_height"], "960")
        self.assertIn('id="outpaintCustomHeightField"', helpers)
        self.assertIn('data-field="custom_target_height"', helpers)
        self.assertIn("function selectOutpaintHeightMode(mode)", helpers)
        self.assertIn("function syncCustomOutpaintHeight(value)", helpers)

        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "0", "section_end": ""})
        app.APP.settings["outpaint"].update({"target_aspect": "21:9", "target_height": "960"})
        command = app.APP.command_for("outpaint")

        self.assertEqual(app.outpaint_work_size_for_source("input/example.mp4", "21:9", "960"), (2240, 960))
        self.assertEqual(command[command.index("--target-height") + 1], "960")

    def test_legacy_outpaint_crop_settings_migrate_to_negative_trim_values(self) -> None:
        settings = app.default_settings()
        settings["outpaint"].update({
            "crop_left": "88",
            "crop_right": "12",
            "crop_top": "6",
            "crop_bottom": "0",
        })

        normalized = runtime_settings.normalize_settings(settings, include_newest_source=False)

        self.assertEqual(
            [normalized["outpaint"][key] for key in ("edge_left", "edge_right", "edge_top", "edge_bottom")],
            ["-88", "-12", "-6", "0"],
        )
        self.assertNotIn("crop_left", normalized["outpaint"])

    def test_upscale_migrates_literal_ltx_prompt_to_modern_reconstruction(self) -> None:
        settings = app.default_settings()
        settings["upscale"]["ltx25_prompt"] = (
            "The exact same video, faithfully reconstructed at twice the spatial resolution with "
            "natural fine detail, stable motion, unchanged people, faces, clothing, objects, framing, "
            "lighting, colour, film texture, and camera movement."
        )
        settings["upscale"]["ltx25_negative_prompt"] = (
            "changed identity, changed face, changed hands, changed objects, altered composition, "
            "warped geometry, duplicate limbs, temporal inconsistency, flicker, oversharpening, halos, "
            "plastic skin, invented text, compression artifacts"
        )

        normalized = runtime_settings.normalize_settings(settings, include_newest_source=False)

        self.assertEqual(normalized["upscale"]["ltx25_prompt"], runtime_settings.LTX25_UPSCALE_PROMPT)
        self.assertEqual(
            normalized["upscale"]["ltx25_negative_prompt"],
            runtime_settings.LTX25_UPSCALE_NEGATIVE_PROMPT,
        )
        self.assertEqual(normalized["upscale"]["ltx25_source_fidelity"], "85")
        self.assertEqual(normalized["upscale"]["ltx25_guidance_scale"], "1.0")

    def test_cleanup_exposes_independent_repairs_and_optional_dearchive(self) -> None:
        cleanup_stage = next(stage for stage in app.STAGES if stage.key == "cleanup")
        fields = {field[0]: field for field in cleanup_stage.fields}

        self.assertNotIn("descratch", fields)
        self.assertEqual(fields["ai_descratch"], ("ai_descratch", "AI DeScratch (ProPainter)", "checkbox", "false"))
        self.assertEqual(fields["ai_descratch_height"], ("ai_descratch_height", "AI DeScratch resolution", "select:540|720|1080|source", "720"))
        self.assertEqual(fields["ai_chunk_frames"], ("ai_chunk_frames", "AI DeScratch chunk frames", "select:25|33|41|49", "41"))
        self.assertEqual(fields["devignette"], ("devignette", "DeVignette", "checkbox", "false"))
        self.assertEqual(fields["dearchive"], ("dearchive", "Dearchive (LTX 2.3 LoRA)", "checkbox", "true"))
        self.assertEqual(fields["dearchive_height"], ("dearchive_height", "Dearchive resolution", "select:540|720|1080|source", "720"))
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
        self.assertEqual(defaults.dearchive_height, "720")
        self.assertEqual(defaults.gguf_model, "LTX-2.3-distilled-Q4_K_M.gguf")
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
        self.assertEqual(command[command.index("--dearchive-height") + 1], "720")
        self.assertEqual(command[command.index("--ai-chunk-frames") + 1], "41")

    def test_dearchive_processing_resolution_preserves_aspect_on_model_safe_grid(self) -> None:
        self.assertEqual(cleanup_video.model_dimensions(4096, 3112, "1080"), (1440, 1088))
        self.assertEqual(cleanup_video.model_dimensions(4096, 3112, "720"), (928, 704))
        self.assertEqual(cleanup_video.model_dimensions(4096, 3112, "source"), (4096, 3104))

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
        self.assertEqual(prompt["2483"]["inputs"]["text"], cleanup_video.DEFAULT_PROMPT)
        self.assertEqual(prompt["2612"]["inputs"]["text"], cleanup_video.DEFAULT_NEGATIVE)
        self.assertEqual(prompt["3940"]["inputs"]["unet_name"], "LTX-2.3-distilled-Q4_K_M.gguf")
        self.assertEqual(prompt["5025"]["class_type"], "ManualSigmas")
        self.assertEqual(len(prompt["5025"]["inputs"]["sigmas"].split(",")), 9)
        self.assertEqual(prompt["4828"]["inputs"]["cfg"], 4.0)
        self.assertEqual(prompt["4829"]["inputs"]["sigmas"], ["5025", 0])
        self.assertEqual(prompt["9001"]["class_type"], "VAELoader")
        self.assertEqual(prompt["4851"]["inputs"]["vae"], ["9001", 0])
        self.assertEqual(prompt["4829"]["inputs"]["latent_image"], ["5012", 2])
        self.assertEqual(prompt["5013"]["inputs"]["latent"], ["4829", 0])
        self.assertNotIn("audio", prompt["5076"]["inputs"])
        class_types = {node["class_type"] for node in prompt.values()}
        self.assertNotIn("CM_FloatToInt", class_types)
        self.assertNotIn("Switch latent [Crystools]", class_types)
        self.assertNotIn("LTXVEmptyLatentAudio", class_types)
        self.assertNotIn("LTXVAudioVAELoader", class_types)
        self.assertNotIn("LTXVConcatAVLatent", class_types)
        self.assertNotIn("LTXVSeparateAVLatent", class_types)

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
        # VP9/Matroska may normalize a nearly-square SAR to 1:1 because its
        # display dimensions are integral; the displayed geometry remains equal.
        sar_num, sar_den = (int(part) for part in output_sar.split("/", 1))
        self.assertAlmostEqual((result["width"] * sar_num / sar_den) / result["height"], (86 * 1280 / 1281) / 48, places=2)
        self.assertEqual(audio_streams, [{"channels": 2, "channel_layout": "stereo"}])

    def test_cleanup_finalize_keeps_model_resolution_and_handles_no_audio_packets(self) -> None:
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
            cleanup_video.finalize_output(
                ffmpeg, generated, source, output, cleanup_video.probe_video(source), (96, 64)
            )
            result = cleanup_video.probe_video(output)

        self.assertEqual((result["width"], result["height"], result["frames"]), (96, 64, 12))

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
        self.assertTrue(output.name.endswith("_gray.mkv"))
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

    def test_openai_colorization_keeps_colour_input_and_labels_all_image_roles(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "shots.csv"
            source = folder / "source.mp4"
            reference_a = folder / "reference_a.png"
            reference_b = folder / "reference_b.png"
            source.write_bytes(b"video")
            Image.new("RGB", (16, 8), (20, 40, 60)).save(reference_a)
            Image.new("RGB", (16, 8), (60, 40, 20)).save(reference_b)
            manifest.write_text("enabled,color_reference,end\ntrue," + str(reference_a) + ",1\n", encoding="utf-8")
            args = colorize_video.build_parser().parse_args(
                ["--manifest", str(manifest), "--method", "openai", "--api-key", "sk-test"]
            )
            sig = colorize_video.signature(
                args,
                manifest,
                source,
                [{"color_reference": str(reference_a), "end": "1"}],
            )
            _extra, instructions = colorize_video.openai_frame_prompt(3, 2)

        self.assertFalse(sig["grayscale_video_input"])
        self.assertIn("Images 2-4", instructions)
        self.assertIn("Images 5-6", instructions)
        self.assertIn("only image to edit", instructions)

    def test_openai_colorization_sends_previous_frames_before_every_shot_reference(self) -> None:
        from PIL import Image
        from reference_sets import encode_additional_references

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source_dir = folder / "source_frames"
            source_dir.mkdir()
            for index in range(4):
                Image.new("RGB", (16, 8), (index * 20, 30, 40)).save(source_dir / f"frame_{index:08d}.png")
            source_video = folder / "source.mp4"
            source_video.write_bytes(b"video")
            references = [folder / "reference_a.png", folder / "reference_b.png"]
            for index, reference in enumerate(references):
                Image.new("RGB", (16, 8), (80 + index, 40, 20)).save(reference)
            row = {
                "start_frame": "0",
                "end_frame": "4",
                "color_reference": str(references[0]),
                "additional_references": encode_additional_references([{"color_reference": str(references[1])}]),
            }
            args = colorize_video.build_parser().parse_args(
                ["--manifest", "shots.csv", "--method", "openai", "--api-key", "sk-test", "--openai-previous-frames", "3"]
            )
            calls = []

            def fake_generate(child, inputs):
                calls.append(list(inputs))
                shutil.copy2(child.source_image, child.output)
                return child.output

            with (
                mock.patch.object(colorize_video, "ROOT", folder),
                mock.patch.object(colorize_video, "prepare_openai_source_frames", return_value=source_dir),
                mock.patch.object(colorize_video.openai_images, "generate", side_effect=fake_generate),
                mock.patch.object(colorize_video, "resumable_output", return_value=False),
                mock.patch.object(colorize_video, "assemble_openai_frames"),
            ):
                colorize_video.run_openai_colorization(
                    args, folder / "shots.csv", source_video, [row], folder / "output.mp4",
                    {"source_fingerprint": colorize_video.file_fingerprint(source_video)},
                    "ffmpeg", 16, 8, 4.0, 4,
                )

        self.assertEqual(calls[0], references)
        self.assertEqual(calls[1][-2:], references)
        self.assertEqual(calls[2][-2:], references)
        self.assertEqual(calls[3][-2:], references)
        self.assertEqual([len(inputs) for inputs in calls], [2, 3, 4, 5])
        self.assertEqual(calls[1][0].name, "frame_00000000.png")

    def test_colormnet_prompt_rejects_unpacked_multi_reference_batch(self) -> None:
        args = colorize_video.build_parser().parse_args(["--manifest", "shots.csv", "--method", "colormnet"])
        with self.assertRaisesRegex(RuntimeError, "exactly one reference"):
            colorize_video.build_prompt(
                "arp_colorize/guarded_gray.mp4", ["reference_1.png", "reference_2.png"],
                0, 240, 640, 360, 24.0, args, "test/output",
            )

    def test_cmnet2_orders_real_reference_anchors_by_source_frame(self) -> None:
        import cmnet2_runtime

        references = [
            {"path": Path("late.png"), "selected_frame": 200},
            {"path": Path("early.png"), "selected_frame": 20},
            {"path": Path("middle.png"), "selected_frame": 90},
        ]

        ordered = cmnet2_runtime.ordered_references(references)

        self.assertEqual([item["selected_frame"] for item in ordered], [20, 90, 200])

    def test_cmnet2_preloads_every_reference_for_the_shot(self) -> None:
        import cmnet2_runtime
        from reference_sets import encode_additional_references

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "gray.mp4"
            source.write_bytes(b"video")
            refs = [folder / "early.png", folder / "late.png"]
            for index, reference in enumerate(refs):
                reference.write_bytes(f"ref {index}".encode())
            row = {
                "start_frame": "0",
                "end_frame": "10",
                "selected_frame": "2",
                "color_reference": str(refs[0]),
                "additional_references": encode_additional_references(
                    [{"selected_frame": "8", "color_reference": str(refs[1])}]
                ),
            }
            args = colorize_video.build_parser().parse_args(
                ["--manifest", "shots.csv", "--method", "cmnet2", "--cmnet2-dir", str(folder)]
            )
            session = mock.Mock()
            session_type = mock.Mock(return_value=session)
            with (
                mock.patch.object(cmnet2_runtime, "resolve_model", return_value=folder / "model.pth"),
                mock.patch.object(cmnet2_runtime, "CMNet2Session", session_type),
                mock.patch.object(colorize_video, "resumable_output", return_value=False),
                mock.patch.object(colorize_video, "prepare_processing_reference", side_effect=lambda path, _w, _h: path),
                mock.patch.object(colorize_video, "stitch_colorized"),
                mock.patch.object(colorize_video, "write_signature"),
            ):
                colorize_video.run_cmnet2_colorization(
                    args, source, [row], folder / "output.mp4", {"settings": {}},
                    "ffmpeg", 640, 360, 24.0, 10,
                )

        supplied = session.colorize_segment.call_args.args[2]
        self.assertEqual([item["selected_frame"] for item in supplied], [2, 8])
        self.assertEqual([item["path"] for item in supplied], refs)

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

    def test_run_all_stops_when_an_upstream_stage_cannot_start(self) -> None:
        stages = [stage for stage in app.STAGES if stage.key in {"outpaint", "shots"}]
        with (
            mock.patch.object(app.APP, "active_stages", return_value=tuple(stages)),
            mock.patch.object(app.APP, "run_stage", return_value=(False, "missing input")) as run_stage,
        ):
            app.APP._run_all_worker()

        run_stage.assert_called_once_with("outpaint")
        self.assertIn("Whole remaster stopped before Outpainting", app.APP.log[-1])

    def test_run_all_skips_stages_that_are_up_to_date(self) -> None:
        class DoneProcess:
            returncode = 0

            def poll(self):
                return 0

        stages = [stage for stage in app.STAGES if stage.key in {"outpaint", "shots", "references"}]
        started: list[str] = []

        def fake_run_stage(stage_key: str) -> tuple[bool, str]:
            started.append(stage_key)
            app.APP.process = DoneProcess()
            app.APP.running_stage_key = ""
            return True, "started"

        with (
            mock.patch.object(app.APP, "active_stages", return_value=tuple(stages)),
            mock.patch.object(app.APP, "stage_is_current", side_effect=lambda key: key in {"outpaint", "shots"}),
            mock.patch.object(app.APP, "hydrate_stage_inputs"),
            mock.patch.object(app.APP, "run_stage", side_effect=fake_run_stage),
        ):
            app.APP._run_all_worker()

        self.assertEqual(started, ["references"])
        log = "\n".join(app.APP.log)
        self.assertIn("Outpainting is up to date", log)
        self.assertIn("Shot Detection is up to date", log)

    def test_run_all_keep_shots_skips_shot_detection_and_references_that_exist(self) -> None:
        class DoneProcess:
            returncode = 0

            def poll(self):
                return 0

        stages = [stage for stage in app.STAGES if stage.key in {"outpaint", "shots", "references", "colour"}]
        started: list[str] = []

        def fake_run_stage(stage_key: str) -> tuple[bool, str]:
            started.append(stage_key)
            app.APP.process = DoneProcess()
            app.APP.running_stage_key = ""
            return True, "started"

        outputs = {"shots": ["shots.csv"], "references": ["a.png", "b.png"]}
        with (
            mock.patch.object(app.APP, "active_stages", return_value=tuple(stages)),
            mock.patch.object(app.APP, "stage_is_current", return_value=False),
            mock.patch.object(app.APP, "expected_outputs", side_effect=lambda key: outputs.get(key, [])),
            mock.patch.object(app.APP, "existing_outputs", side_effect=lambda key: outputs.get(key, [])),
            mock.patch.object(app.APP, "hydrate_stage_inputs"),
            mock.patch.object(app.APP, "run_stage", side_effect=fake_run_stage),
        ):
            app.APP._run_all_worker(keep_shots=True)
            self.assertEqual(started, ["outpaint", "colour"])
            self.assertIn("Keeping the existing Shot Detection results", "\n".join(app.APP.log))

            # A missing reference image still runs Reference Generation, which fills in only that one.
            started.clear()
            with mock.patch.object(app.APP, "existing_outputs", side_effect=lambda key: outputs.get(key, [])[:1]):
                app.APP._run_all_worker(keep_shots=True)
            self.assertEqual(started, ["outpaint", "references", "colour"])

            started.clear()
            app.APP._run_all_worker()
            self.assertEqual(started, ["outpaint", "shots", "references", "colour"])

    def test_switching_outpaint_model_carries_the_shot_list_and_references_over(self) -> None:
        old_clip = "intermediate/outpainted/Zzcarry_outpaint_0123abcd.mkv"
        new_clip = "intermediate/outpainted/Zzcarry_outpaint25_0123abcd.mkv"
        self.assertIn(old_clip, app.other_model_outpaints(new_clip))
        old_manifest = app.resolve(app.manifest_for_outpainted(old_clip))
        new_manifest = app.resolve(app.manifest_for_outpainted(new_clip))
        self.assertNotEqual(old_manifest, new_manifest)
        folders = [app.ROOT / "intermediate" / root / stem for root in ("outpainted_references", "outpainted_references_color") for stem in ("Zzcarry_outpaint_0123abcd", "Zzcarry_outpaint25_0123abcd")]
        try:
            colour = app.ROOT / "intermediate/outpainted_references_color/Zzcarry_outpaint_0123abcd/cut_0000.png"
            colour.parent.mkdir(parents=True)
            colour.write_bytes(b"png")
            app.write_manifest_details(old_manifest, old_clip, ["enabled", "color_reference", "prompt"], [
                {"enabled": "true", "color_reference": "intermediate/outpainted_references_color/Zzcarry_outpaint_0123abcd/cut_0000.png", "prompt": "a red car"},
            ])
            old_manifest.with_name(old_manifest.name + ".sig.json").write_text("{}", encoding="utf-8")

            self.assertEqual(app.APP.adopt_other_model_shot_list(new_clip), app.rel(new_manifest))

            source_video, _fields, rows = app.read_manifest_details(new_manifest)
            self.assertEqual(source_video, new_clip)
            self.assertEqual(rows[0]["prompt"], "a red car")
            self.assertEqual(rows[0]["color_reference"], "intermediate/outpainted_references_color/Zzcarry_outpaint25_0123abcd/cut_0000.png")
            self.assertTrue((app.ROOT / rows[0]["color_reference"]).is_file())
            self.assertTrue(new_manifest.with_name(new_manifest.name + ".sig.json").is_file())
            # The other model's list is untouched, and an existing list is never replaced.
            self.assertEqual(app.read_manifest_details(old_manifest)[0], old_clip)
            self.assertEqual(app.APP.adopt_other_model_shot_list(new_clip), "")
        finally:
            for manifest in (old_manifest, new_manifest):
                manifest.unlink(missing_ok=True)
                manifest.with_name(manifest.name + ".sig.json").unlink(missing_ok=True)
            for folder in folders:
                shutil.rmtree(folder, ignore_errors=True)

    def test_stage_is_current_only_while_inputs_and_settings_match_the_last_finished_run(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            video = folder / "outpainted.mp4"
            video.write_bytes(b"video")
            manifest = folder / "shots.csv"
            manifest.write_text("enabled,prompt\ntrue,\n", encoding="utf-8")
            command = [sys.executable, "-u", str(app.SCRIPTS / "generate_references.py"), "--source-video", str(video), "--output-manifest", str(manifest), "--shot-threshold", "0.075"]
            values = app.APP.settings.setdefault("shots", {})
            with (
                mock.patch.object(app.APP, "expected_outputs", return_value=[str(manifest)]),
                mock.patch.object(app.APP, "local_command_for", side_effect=lambda _key, run_flags=True: list(command)),
            ):
                self.assertFalse(app.APP.stage_is_current("shots"))  # never finished

                launch = app.APP.stage_stamp("shots")
                app.APP.record_stage_stamp("shots", launch)
                self.assertTrue(app.APP.stage_is_current("shots"))

                manifest.write_text("enabled,prompt\nfalse,a user edit\n", encoding="utf-8")
                self.assertTrue(app.APP.stage_is_current("shots"))  # its own output is not an input

                values["force"] = "true"
                self.assertFalse(app.APP.stage_is_current("shots"))  # Regenerate always runs
                values["force"] = "false"

                command[-1] = "0.2"
                self.assertFalse(app.APP.stage_is_current("shots"))  # a setting changed
                command[-1] = "0.075"
                self.assertTrue(app.APP.stage_is_current("shots"))

                video.write_bytes(b"re-rendered video")
                self.assertFalse(app.APP.stage_is_current("shots"))  # the input changed

                # An input edited while the stage ran: the output may predate it, so no stamp.
                launch = app.APP.stage_stamp("shots")
                video.write_bytes(b"edited mid-run")
                app.APP.record_stage_stamp("shots", launch)
                self.assertFalse(app.APP.stage_is_current("shots"))

                manifest.unlink()
                app.APP.record_stage_stamp("shots", app.APP.stage_stamp("shots"))
                self.assertFalse(app.APP.stage_is_current("shots"))  # output missing

    def test_stage_stamp_follows_files_a_manifest_points_at(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            guide = folder / "guide.png"
            raw = folder / "raw_0000.mkv"
            for path in (source, guide, raw):
                path.write_bytes(b"x")
            chunks = folder / "chunks.csv"
            guide_frames = json.dumps([{"frame_idx": 0, "strength": 0.7, "image": str(guide)}])
            with chunks.open("w", encoding="utf-8", newline="") as handle:
                handle.write(f"# source_video={source}\n")
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["chunk_index", "guide_frames", "raw_path"])
                writer.writerow(["0", guide_frames, str(raw)])
            command = ["python", str(app.SCRIPTS / "outpaint_video.py"), "--chunk-manifest", str(chunks)]

            def stamp() -> str:
                return stage_stamps.compute_stamp(command, [str(folder / "out.mp4")], " ".join(command))

            before = stamp()
            raw.write_bytes(b"a regenerated chunk render")
            self.assertEqual(stamp(), before)  # the stage's own chunk renders are not inputs
            guide.write_bytes(b"a new guide")
            after_guide = stamp()
            self.assertNotEqual(after_guide, before)
            source.write_bytes(b"a new source")
            self.assertNotEqual(stamp(), after_guide)

    def test_outpaint_chunk_regeneration_forgets_the_outpaint_stamp(self) -> None:
        stage_stamps.write_stamp("outpaint", ["intermediate/outpainted/movie.mp4"], "digest")
        app.APP.settings["global"]["source"] = "input/movie.mp4"

        class RunningProcess:
            stdout: list[str] = []

            def poll(self):
                return None

        with (
            mock.patch.object(app.APP, "ensure_pipeline_source"),
            mock.patch.object(server, "ensure_comfy_available_for_stage", return_value=(True, "")),
            mock.patch.object(app.APP, "command_for", return_value=["python", "outpaint_video.py"]),
            mock.patch.object(server.subprocess, "Popen", return_value=RunningProcess()),
            mock.patch.object(server, "keep_awake"),
            mock.patch.object(server.threading, "Thread"),
        ):
            ok, _message = app.APP.run_outpaint_chunk(0)

        app.APP.process = None
        app.APP.running_stage_key = ""
        self.assertTrue(ok)
        self.assertEqual(stage_stamps.read_stamp("outpaint", ["intermediate/outpainted/movie.mp4"]), "")

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

        # Identity-keyed short name: <sourceword>_<tag>_<key>.mkv under intermediate/outpainted/.
        self.assertTrue(output.startswith("intermediate/outpainted/My_outpaint_"), output)
        self.assertTrue(output.endswith(".mkv"))
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

    def test_outpaint_models_share_user_chunk_manifest_but_not_render_cache(self) -> None:
        base = {
            "target_aspect": "16:9",
            "target_height": "720",
            "crop_left": "0",
            "crop_right": "0",
            "crop_top": "0",
            "crop_bottom": "0",
            "outpaint_all_black_regions": "false",
        }
        official = {**base, "outpaint_model": "official"}
        ltx25 = {**base, "outpaint_model": "ltx25"}
        wanvace = {**base, "outpaint_model": "wanvace"}
        self.assertEqual(
            app.outpaint_chunk_manifest_for("input/My Source.mp4", official),
            app.outpaint_chunk_manifest_for("input/My Source.mp4", wanvace),
        )
        self.assertIn("_chunkswan_", app.outpaint_chunk_dir_for("input/My Source.mp4", wanvace).name)

        official_manifest = app.outpaint_chunk_manifest_for("input/My Source.mp4", official)
        ltx25_manifest = app.outpaint_chunk_manifest_for("input/My Source.mp4", ltx25)

        self.assertEqual(official_manifest, ltx25_manifest)
        self.assertIn("_chunks_", Path(official_manifest).name)
        self.assertNotEqual(
            app.outpaint_chunk_dir_for("input/My Source.mp4", official),
            app.outpaint_chunk_dir_for("input/My Source.mp4", ltx25),
        )

        source = app.resolve_video_source("input/My Source.mp4")
        official_args = argparse.Namespace(
            crop_left=0, crop_right=0, crop_top=0, crop_bottom=0,
            outpaint_all_black_regions=False, ltx_version="2.3",
        )
        ltx25_args = argparse.Namespace(
            crop_left=0, crop_right=0, crop_top=0, crop_bottom=0,
            outpaint_all_black_regions=False, ltx_version="2.5",
        )
        self.assertEqual(
            outpaint_video.default_chunk_manifest(source, "16:9", 1280, 704, official_args),
            outpaint_video.default_chunk_manifest(source, "16:9", 1280, 704, ltx25_args),
        )

    def test_outpaint_manifest_migration_copies_legacy_ltx25_plan_non_destructively(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            legacy = folder / "movie_chunks25_deadbeef.csv"
            shared = folder / "movie_chunks_deadbeef.csv"
            legacy.write_text("chunk_index,guide_frames\n0,valuable-user-guide\n", encoding="utf-8")
            args = argparse.Namespace(
                crop_left=0, crop_right=0, crop_top=0, crop_bottom=0,
                outpaint_all_black_regions=False, ltx_version="2.5",
            )
            with mock.patch.object(outpaint_video, "legacy_ltx25_chunk_manifest", return_value=legacy):
                outpaint_video.migrate_legacy_chunk_manifest(
                    shared, folder / "source.mp4", "16:9", 1280, 704, args
                )

            self.assertEqual(shared.read_bytes(), legacy.read_bytes())
            self.assertTrue(legacy.exists())

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

    def test_outpaint_source_crop_is_fitted_as_the_complete_source(self) -> None:
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

        self.assertEqual(prepare_outpaint_input.source_placement_size(args, info, 1280, 704), (1280, 480, 1280, 470))

        filter_text = prepare_outpaint_input.build_filter(args, info, 1280, 704)

        self.assertIn("trim=start_frame=0,setpts=N/(24.00000000*TB),fps=24.00000000", filter_text)
        self.assertIn("crop=w=1440:h=540:x=0:y=270,scale=w=1280:h=480:flags=lanczos,scale=w=1280:h=470:flags=lanczos", filter_text)
        self.assertNotIn("force_original_aspect_ratio=decrease", filter_text)

    def test_outpaint_source_extension_reserves_canvas_inside_selected_aspect(self) -> None:
        args = argparse.Namespace(
            delivery_width=1920,
            delivery_height=1080,
            crop_left=0,
            crop_right=0,
            crop_top=-180,
            crop_bottom=-180,
            black_lift=0.018,
            gamma=1.06,
            outpaint_all_black_regions=False,
        )
        info = {"width": 1440, "height": 1080, "fps": 24.0}

        self.assertEqual(prepare_outpaint_input.source_placement_size(args, info, 1920, 1088), (1080, 810, 1080, 816))
        filter_text = prepare_outpaint_input.build_filter(args, info, 1920, 1088)

        self.assertIn("crop=w=1440:h=1080:x=0:y=0,scale=w=1080:h=810:flags=lanczos,scale=w=1080:h=816:flags=lanczos", filter_text)
        self.assertIn("overlay=x=420:y=136", filter_text)

    def test_outpaint_all_black_regions_uses_explicit_mask_without_source_lift(self) -> None:
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

    def test_standard_outpaint_preparation_keeps_source_tone_unchanged(self) -> None:
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
        self.assertNotIn("lutrgb=", filter_text)

    def test_oumoumad_preparation_uses_legacy_source_lift(self) -> None:
        args = argparse.Namespace(
            delivery_width=1280,
            delivery_height=720,
            crop_left=0,
            crop_right=0,
            crop_top=0,
            crop_bottom=0,
            black_lift=0.018,
            gamma=1.06,
            legacy_black_mask=True,
            outpaint_all_black_regions=False,
        )
        info = {"width": 960, "height": 720, "fps": 24.0}

        filter_text = prepare_outpaint_input.build_filter(args, info, 1280, 704)

        self.assertIn("lutrgb=", filter_text)
        self.assertIn("color=c=black:s=1280x704", filter_text)

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
        # Native-resolution recomposition is opt-in: off unless the box is ticked.
        self.assertNotIn("--native-source-resolution", command)
        app.APP.settings["recomp"]["native_source_resolution"] = "true"
        self.assertIn("--native-source-resolution", app.APP.command_for("recomp"))

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
                mock.patch.object(lifecycle, "ensure_bindable_managed_comfy_endpoint", side_effect=lambda value: value),
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

    def test_start_comfy_uses_pytorch_attention_on_newer_gpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            root = Path(tmp_text)
            comfy = root / "tools" / "comfyui"
            comfy.mkdir(parents=True)
            (comfy / "main.py").write_text("# comfy\n", encoding="utf-8")

            class FakeProcess:
                returncode = None

                def poll(self):
                    return None

            popen_calls: list[list[str]] = []

            def fake_popen(command, **_kwargs):
                popen_calls.append(command)
                return FakeProcess()

            with (
                mock.patch.object(lifecycle, "STARTED_COMFY_PROCESS", None),
                mock.patch.object(lifecycle, "current_config", return_value={
                    "comfy_dir": str(comfy),
                    "comfy_url": "http://127.0.0.1:8188",
                    "comfy_host": "127.0.0.1",
                    "comfy_port": "8188",
                    "comfy_managed_by_arp": "true",
                }),
                mock.patch.object(lifecycle, "discover_comfy_instances", return_value=[]),
                mock.patch.object(lifecycle, "ensure_bindable_managed_comfy_endpoint", side_effect=lambda value: value),
                mock.patch.object(lifecycle, "comfy_requires_pytorch_attention", return_value=True),
                mock.patch.object(lifecycle.subprocess, "Popen", side_effect=fake_popen),
                mock.patch.object(lifecycle.threading.Thread, "start", lambda _self: None),
            ):
                lifecycle.start_comfy_if_needed()

        self.assertEqual(len(popen_calls), 1)
        self.assertIn("--use-pytorch-cross-attention", popen_calls[0])

    def test_managed_comfy_moves_from_an_unbindable_port_and_persists_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            config_path = Path(tmp_text) / ".ai_remaster_config.json"
            config_path.write_text(json.dumps({"keep": "value"}), encoding="utf-8")
            active = {
                "comfy_url": "http://127.0.0.1:8188",
                "comfy_host": "127.0.0.1",
                "comfy_port": "8188",
                "comfy_managed_by_arp": "true",
            }

            with (
                mock.patch.object(lifecycle, "CONFIG_FILE", config_path),
                mock.patch.object(lifecycle, "port_can_bind", side_effect=lambda _host, port: port == 8797),
            ):
                updated = lifecycle.ensure_bindable_managed_comfy_endpoint(active)

            stored = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(updated["comfy_port"], "8797")
        self.assertEqual(updated["comfy_url"], "http://127.0.0.1:8797")
        self.assertEqual(stored["keep"], "value")
        self.assertEqual(stored["comfy_port"], "8797")

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
            "ComfyUI-ARP": ("ARPLTXVideoOnlyICLoRALoader", "ARPLTXEphemeralTextEncode"),
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
        gguf_loader = (vendor_root / "ComfyUI-GGUF" / "loader.py").read_text(encoding="utf-8")
        self.assertRegex(gguf_loader, r"TXT_ARCH_LIST\s*=.*[\"']gemma4[\"']")
        self.assertIn("audio_embeddings_connector.learnable_registers", gguf_loader)
        installer = (app.ROOT / "install_windows.ps1").read_text(encoding="utf-8")
        self.assertIn("Ensure-ComfyUIGGUFGemma4Support", installer)

    def test_windows_installer_defaults_to_cuda_13_pytorch_wheels(self) -> None:
        installer = (app.ROOT / "install_windows.ps1").read_text(encoding="utf-8")

        self.assertIn("https://download.pytorch.org/whl/cu130", installer)
        self.assertNotIn("https://download.pytorch.org/whl/cu128", installer)

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

    def test_comfy_progress_event_reports_selected_upscale_node(self) -> None:
        message = json.dumps({
            "type": "progress_state",
            "data": {
                "prompt_id": "prompt-id",
                "nodes": {
                    "2": {"state": "finished", "value": 1, "max": 1},
                    "3": {"state": "running", "value": 17, "max": 50},
                },
            },
        })

        self.assertEqual(comfy_api.prompt_progress_fraction(message, "prompt-id", {"3"}), (17.0, 50.0))
        self.assertIsNone(comfy_api.prompt_progress_fraction(message, "other-prompt", {"3"}))

    def test_bundled_outpaint_template_contains_official_two_stage_nodes(self) -> None:
        workflow = json.loads((app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json").read_text(encoding="utf-8-sig"))

        prompt = comfy_api.workflow_to_prompt(workflow, "5228")

        class_types = {node["class_type"] for node in prompt.values()}
        self.assertNotIn("ImagePadKJ", class_types)
        self.assertIn("LoadVideo", class_types)
        self.assertIn("SaveVideo", class_types)
        self.assertEqual(sum(node["class_type"] == "SamplerCustomAdvanced" for node in prompt.values()), 2)
        self.assertEqual(sum(node["class_type"] == "LTXVLaplacianPyramidBlend" for node in prompt.values()), 2)

    def test_bundled_ltx25_template_patches_to_local_quantized_models(self) -> None:
        workflow = json.loads((app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-2.5.json").read_text(encoding="utf-8-sig"))
        args = outpaint_video.build_parser().parse_args([
            "--source", "input/example.mp4", "--comfy-dir", str(app.ROOT),
            "--ltx-version", "2.5", "--dry-run",
        ])

        with (
            mock.patch.object(outpaint_video, "copy_to_comfy_input", return_value="arp_outpaint_ltx25/prepared.mp4"),
            mock.patch.object(outpaint_video, "copy_reference_frame_to_comfy_input", return_value="arp_outpaint_ltx25/frame.png"),
            mock.patch.object(outpaint_video, "official_mask_image", return_value=app.ROOT / "exact-mask.png"),
            mock.patch.object(outpaint_video, "generation_mask_image", return_value=app.ROOT / "generation-mask.png"),
            mock.patch.object(outpaint_video, "probe_video", return_value={"width": 1280, "height": 704, "frames": 49, "fps": 24.0}),
        ):
            prompt = outpaint_video.patch_workflow(
                args, workflow, app.ROOT / "prepared.mp4", app.ROOT, "arp_outpaint_ltx25/test",
                "preserve the archival street scene", args.negative_prompt, 42,
            )

        classes = [node["class_type"] for node in prompt.values()]
        self.assertIn("UnetLoaderGGUF", classes)
        self.assertEqual(classes.count("UnetLoaderGGUF"), 1)
        self.assertEqual(classes.count("CLIPLoaderGGUF"), 1)
        self.assertNotIn("ImagePadForOutpaintTargetSize", classes)
        self.assertNotIn("OpenAIDalle3", classes)
        self.assertNotIn("GeminiImage2Node", classes)
        self.assertEqual(sum(node.get("inputs", {}).get("unet_name") == outpaint_video.LTX25_GGUF_MODEL for node in prompt.values()), 1)
        self.assertEqual(sum(node.get("inputs", {}).get("clip_name") == outpaint_video.LTX25_TEXT_ENCODER for node in prompt.values()), 1)
        self.assertEqual(prompt["9013"]["inputs"]["images"], ["9069", 0])
        self.assertEqual(prompt["9066"]["inputs"]["input"], ["9056", 0])
        self.assertEqual(prompt["9067"]["inputs"]["input"], ["9191", 0])
        self.assertEqual(prompt["9191"]["inputs"]["image"], ["9190", 0])

    def test_ltx25_default_uses_the_full_resolution_official_graph(self) -> None:
        workflow = json.loads((app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json").read_text(encoding="utf-8-sig"))
        args = outpaint_video.build_parser().parse_args([
            "--source", "input/example.mp4", "--comfy-dir", str(app.ROOT),
            "--ltx-version", "2.5", "--dry-run",
        ])
        outpaint_video.configure_ltx25_model_args(args)

        with (
            mock.patch.object(outpaint_video, "copy_to_comfy_input", side_effect=[
                "arp_outpaint_ltx25/prepared.mp4",
                "arp_outpaint_mask/mask.png",
                "arp_outpaint_generation_mask/mask.png",
            ]),
            mock.patch.object(outpaint_video, "copy_reference_frame_to_comfy_input", return_value="arp_outpaint_ltx25/frame.png"),
            mock.patch.object(outpaint_video, "official_mask_image", return_value=app.ROOT / "exact-mask.png"),
            mock.patch.object(outpaint_video, "generation_mask_image", return_value=app.ROOT / "generation-mask.png"),
            mock.patch.object(outpaint_video, "probe_video", return_value={"width": 2240, "height": 960, "frames": 294, "fps": 24.0}),
        ):
            prompt = outpaint_video.patch_workflow(
                args, workflow, app.ROOT / "prepared.mp4", app.ROOT, "arp_outpaint_ltx25/test",
                "preserve the archival street scene", args.negative_prompt, 42,
            )

        classes = [node["class_type"] for node in prompt.values()]
        self.assertEqual(args.gguf_model, outpaint_video.LTX25_GGUF_MODEL)
        self.assertEqual(prompt["3940"]["inputs"]["unet_name"], outpaint_video.LTX25_GGUF_MODEL)
        self.assertEqual(prompt["9001"]["inputs"]["vae_name"], outpaint_video.LTX25_VIDEO_VAE)
        self.assertEqual(prompt["5023"]["class_type"], "CLIPLoaderGGUF")
        self.assertEqual(prompt["5023"]["inputs"]["clip_name"], outpaint_video.LTX25_TEXT_ENCODER)
        self.assertEqual(classes.count("SamplerCustomAdvanced"), 1)
        self.assertEqual(classes.count("LTXVLaplacianPyramidBlend"), 1)
        self.assertNotIn("LatentUpscaleModelLoader", classes)
        self.assertEqual(prompt["5227"]["inputs"]["images"], ["5266", 0])

    def test_ltx25_uses_the_cached_gguf_prompt_encoder_when_comfy_has_it(self) -> None:
        workflow = json.loads((app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json").read_text(encoding="utf-8-sig"))
        args = outpaint_video.build_parser().parse_args([
            "--source", "input/example.mp4", "--comfy-dir", str(app.ROOT),
            "--ltx-version", "2.5", "--dry-run",
        ])
        outpaint_video.configure_ltx25_model_args(args)
        args.cached_gguf_text_encode = True

        with (
            mock.patch.object(outpaint_video, "copy_to_comfy_input", side_effect=[
                "arp_outpaint_ltx25/prepared.mp4",
                "arp_outpaint_mask/mask.png",
                "arp_outpaint_generation_mask/mask.png",
            ]),
            mock.patch.object(outpaint_video, "copy_reference_frame_to_comfy_input", return_value="arp_outpaint_ltx25/frame.png"),
            mock.patch.object(outpaint_video, "official_mask_image", return_value=app.ROOT / "exact-mask.png"),
            mock.patch.object(outpaint_video, "generation_mask_image", return_value=app.ROOT / "generation-mask.png"),
            mock.patch.object(outpaint_video, "probe_video", return_value={"width": 2240, "height": 960, "frames": 294, "fps": 24.0}),
        ):
            prompt = outpaint_video.patch_workflow(
                args, workflow, app.ROOT / "prepared.mp4", app.ROOT, "arp_outpaint_ltx25/test",
                "preserve the archival street scene", args.negative_prompt, 42,
            )

        classes = [node["class_type"] for node in prompt.values()]
        encoder = prompt["9199"]
        self.assertEqual(encoder["class_type"], "ARPLTXCachedGGUFTextEncode")
        self.assertEqual(encoder["inputs"]["clip_name"], outpaint_video.LTX25_TEXT_ENCODER)
        self.assertEqual(encoder["inputs"]["positive"], "preserve the archival street scene")
        self.assertEqual(encoder["inputs"]["negative"], args.negative_prompt)
        self.assertNotIn("CLIPLoaderGGUF", classes)
        self.assertNotIn("CLIPTextEncode", classes)
        links = [value for node in prompt.values() for value in node["inputs"].values() if isinstance(value, list) and len(value) == 2]
        self.assertIn(["9199", 0], links)
        self.assertIn(["9199", 1], links)
        self.assertTrue(all(str(source) in prompt for source, _slot in links))

    def test_outpaint_model_labels_name_model_and_lora(self) -> None:
        helpers = (app.ROOT / "ai_remaster_gui" / "static" / "js" / "render-helpers.js").read_text(encoding="utf-8")

        self.assertIn("'2.3 - Oumoumad LoRA'", helpers)
        self.assertIn("'2.3 - Official LoRA'", helpers)
        self.assertIn("'2.5 - Official LoRA'", helpers)
        # Wan and H3 stay available but are flagged: neither produced usable archive outpaints.
        self.assertIn("'Wan 2.1 VACE (14B) - experimental'", helpers)
        self.assertIn("'MiniMax H3 - experimental, licence required'", helpers)

    def test_ltx25_frame_rate_preparation_adds_silence_for_archival_video(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            prepared = Path(tmp_text) / "prepared.mp4"
            prepared.write_bytes(b"video")
            with (
                mock.patch.object(outpaint_video, "probe_video", side_effect=[{"fps": 16.0, "frames": 33}, {"frames": 50}]),
                mock.patch.object(outpaint_video, "video_has_audio", return_value=False),
                mock.patch.object(outpaint_video, "resumable_output", return_value=False),
                mock.patch.object(outpaint_video.subprocess, "run") as run,
                mock.patch.object(outpaint_video, "replace_with_retry"),
            ):
                output = outpaint_video.prepare_ltx25_frame_rate("ffmpeg", prepared, "24")

        command = run.call_args.args[0]
        self.assertEqual(output.name, "prepared_ltx25_24fps.mkv")
        self.assertIn("anullsrc=channel_layout=stereo:sample_rate=48000", command)
        self.assertIn("minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,tpad=stop_mode=clone:stop_duration=1", command)
        self.assertEqual(command[command.index("-frames:v") + 1], "50")
        self.assertIn("-shortest", command)

    def test_ltx25_fast_frame_rate_retimes_only_existing_frames(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            prepared = Path(tmp_text) / "prepared.mp4"
            prepared.write_bytes(b"video")
            with (
                mock.patch.object(outpaint_video, "probe_video", side_effect=[{"fps": 16.0, "frames": 294}, {"frames": 294}]),
                mock.patch.object(outpaint_video, "video_has_audio", return_value=False),
                mock.patch.object(outpaint_video, "resumable_output", return_value=False),
                mock.patch.object(outpaint_video.subprocess, "run") as run,
                mock.patch.object(outpaint_video, "replace_with_retry"),
            ):
                output = outpaint_video.prepare_ltx25_frame_rate("ffmpeg", prepared, "24-fast")

        command = run.call_args.args[0]
        self.assertEqual(output.name, "prepared_ltx25_24_fastfps.mkv")
        self.assertIn("trim=end_frame=294,setpts=N/(24*TB),fps=24", command)
        self.assertNotIn("minterpolate", " ".join(command))
        self.assertEqual(command[command.index("-frames:v") + 1], "294")
        self.assertEqual(command[command.index("-r") + 1], "24.00000000")

    def test_outpaint_chunk_lengths_follow_ltx_8n_plus_1_rule(self) -> None:
        self.assertEqual(outpaint_video.ltx_valid_frame_count(2.0, 24.0), 49)
        self.assertEqual(outpaint_video.ltx_valid_frame_count(4.04, 24.0), 97)
        self.assertEqual(outpaint_video.ltx_padded_frame_count(1), 1)
        self.assertEqual(outpaint_video.ltx_padded_frame_count(8), 9)
        self.assertEqual(outpaint_video.ltx_padded_frame_count(9), 9)
        self.assertEqual(outpaint_video.ltx_padded_frame_count(10), 17)
        ranges = outpaint_video.chunk_ranges_from_manifest(100, 24.0, 2.0, 8, {})
        self.assertEqual(ranges, [(0, 0, 49), (1, 41, 90), (2, 75, 100)])
        self.assertTrue(all((end - start) % 8 == 1 for _, start, end in ranges))

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

        self.assertNotIn("2483", prompt)
        self.assertNotIn("2612", prompt)
        self.assertNotIn("5023", prompt)
        self.assertEqual(
            prompt["9199"]["inputs"]["positive"],
            "outpaint with natural film grain. continue the wallpaper",
        )
        self.assertEqual(
            prompt["9199"]["inputs"]["negative"],
            args.negative_prompt,
        )
        self.assertEqual(prompt["5114"]["inputs"]["positive"], ["1241", 0])
        self.assertEqual(prompt["1241"]["inputs"]["positive"], ["9199", 0])
        self.assertEqual(prompt["1241"]["inputs"]["negative"], ["9199", 1])

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
        self.assertNotIn("5023", prompt)
        self.assertEqual(prompt["9199"]["inputs"]["device"], "cpu")
        self.assertEqual(prompt["9199"]["class_type"], "ARPLTXEphemeralTextEncode")
        self.assertEqual(prompt["5093"]["inputs"]["latent_image"], ["5114", 2])
        self.assertEqual(prompt["5013"]["inputs"]["latent"], ["5093", 1])
        for audio_diffusion_node in ("5382", "5383", "5385", "5386", "5389", "5390", "9110"):
            self.assertNotIn(audio_diffusion_node, prompt)
        self.assertEqual(prompt["5266"]["inputs"]["mask"], ["9101", 0])
        self.assertEqual(prompt["5266"]["inputs"]["image_b"], ["5168", 0])
        self.assertEqual(prompt["5266"]["inputs"]["mask_low_res_dilation"], 2)
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

    def test_oumoumad_outpaint_uses_legacy_black_guide_without_official_masks(self) -> None:
        workflow = json.loads((app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json").read_text(encoding="utf-8-sig"))
        args = outpaint_video.build_parser().parse_args([
            "--source", "input/example.mp4",
            "--comfy-dir", str(app.ROOT),
            "--outpaint-lora", outpaint_video.OUMOUMAD_OUTPAINT_LORA,
            "--dry-run",
        ])
        with (
            mock.patch.object(outpaint_video, "copy_to_comfy_input", return_value="arp_outpaint/prepared.mp4"),
            mock.patch.object(outpaint_video, "copy_reference_frame_to_comfy_input", return_value="arp_outpaint/reference.png"),
            mock.patch.object(outpaint_video, "official_mask_image") as official_mask,
            mock.patch.object(outpaint_video, "generation_mask_image") as generation_mask,
            mock.patch.object(outpaint_video, "probe_video", return_value={"width": 864, "height": 480, "frames": 24, "fps": 24.0}),
        ):
            prompt = outpaint_video.patch_workflow(
                args, workflow, app.ROOT / "prepared.mp4", app.ROOT, "arp_outpaint/test",
                args.prompt, args.negative_prompt, 42,
            )

        official_mask.assert_not_called()
        generation_mask.assert_not_called()
        self.assertEqual(prompt["5114"]["class_type"], "LTXAddVideoICLoRAGuide")
        self.assertEqual(prompt["5114"]["inputs"]["image"], ["5168", 0])
        self.assertEqual(prompt["5227"]["inputs"]["images"], ["4851", 0])
        self.assertEqual(prompt["5227"]["inputs"]["audio"], ["5168", 1])
        self.assertNotIn("9100", prompt)

    def test_oumoumad_custom_mask_is_applied_as_black_guide_pixels(self) -> None:
        workflow = json.loads((app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json").read_text(encoding="utf-8-sig"))
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            custom_mask = Path(tmp_text) / "custom-mask.png"
            custom_mask.write_bytes(b"mask")
            args = outpaint_video.build_parser().parse_args([
                "--source", "input/example.mp4",
                "--comfy-dir", str(app.ROOT),
                "--outpaint-lora", outpaint_video.OUMOUMAD_OUTPAINT_LORA,
                "--custom-mask", str(custom_mask),
                "--dry-run",
            ])
            with (
                mock.patch.object(outpaint_video, "copy_to_comfy_input", side_effect=["arp_outpaint/prepared.mp4", "arp_outpaint_custom_mask/mask.png"]),
                mock.patch.object(outpaint_video, "copy_reference_frame_to_comfy_input", return_value="arp_outpaint/reference.png"),
                mock.patch.object(outpaint_video, "probe_video", return_value={"width": 864, "height": 480, "frames": 24, "fps": 24.0}),
            ):
                prompt = outpaint_video.patch_workflow(
                    args, workflow, app.ROOT / "prepared.mp4", app.ROOT, "arp_outpaint/test",
                    args.prompt, args.negative_prompt, 42,
                )

        self.assertEqual(prompt["9120"]["class_type"], "LoadImage")
        self.assertEqual(prompt["9120"]["inputs"]["image"], "arp_outpaint_custom_mask/mask.png")
        self.assertEqual(prompt["9121"]["inputs"]["image"], ["9120", 0])
        self.assertEqual(prompt["9122"]["inputs"], {"width": 864, "height": 480, "batch_size": 1, "color": 0})
        self.assertEqual(prompt["9123"]["inputs"]["destination"], ["5168", 0])
        self.assertEqual(prompt["9123"]["inputs"]["source"], ["9122", 0])
        self.assertEqual(prompt["9123"]["inputs"]["mask"], ["9121", 0])
        self.assertEqual(prompt["5114"]["inputs"]["image"], ["9123", 0])
        self.assertEqual(prompt["5054"]["inputs"]["image"], ["5168", 0])
        # The full-canvas guide keeps its own edge: sharing a link ID with the custom
        # mask made ImageToMask read the guide frame and black out the whole picture.
        self.assertEqual(prompt["3159"]["inputs"]["image"], ["2004", 0])

    def test_official_black_region_custom_mask_adds_to_the_dynamic_mask(self) -> None:
        workflow = json.loads((app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json").read_text(encoding="utf-8-sig"))
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            custom_mask = Path(tmp_text) / "custom-mask.png"
            custom_mask.write_bytes(b"mask")
            args = outpaint_video.build_parser().parse_args([
                "--source", "input/example.mp4",
                "--comfy-dir", str(app.ROOT),
                "--outpaint-all-black-regions",
                "--custom-mask", str(custom_mask),
                "--generation-mask-overlap", "0",
                "--dry-run",
            ])
            with (
                mock.patch.object(outpaint_video, "copy_to_comfy_input", side_effect=["arp_outpaint/prepared.mp4", "arp_outpaint_custom_mask/mask.png"]),
                mock.patch.object(outpaint_video, "copy_reference_frame_to_comfy_input", return_value="arp_outpaint/reference.png"),
                mock.patch.object(outpaint_video, "probe_video", return_value={"width": 864, "height": 480, "frames": 24, "fps": 24.0}),
            ):
                prompt = outpaint_video.patch_workflow(
                    args, workflow, app.ROOT / "prepared.mp4", app.ROOT, "arp_outpaint/test",
                    args.prompt, args.negative_prompt, 42,
                )

        self.assertEqual(prompt["9120"]["inputs"]["image"], "arp_outpaint_custom_mask/mask.png")
        self.assertEqual(prompt["9121"]["inputs"]["image"], ["9120", 0])
        self.assertEqual(prompt["9122"]["class_type"], "MaskComposite")
        self.assertEqual(prompt["9122"]["inputs"]["destination"], ["9102", 0])
        self.assertEqual(prompt["9122"]["inputs"]["source"], ["9121", 0])
        # Both the sentinel preprocessor and the final blend see the combined mask.
        self.assertEqual(prompt["5358"]["inputs"]["mask"], ["9122", 0])
        self.assertEqual(prompt["5266"]["inputs"]["mask"], ["9122", 0])
        self.assertEqual(prompt["3159"]["inputs"]["image"], ["2004", 0])

    def test_many_extra_guides_do_not_overwrite_the_mask_wiring(self) -> None:
        # Extra guides reserve ten link IDs each and delete whatever already holds them.
        # Six guides used to land on the mask chain's own IDs and strip it from the graph.
        workflow = json.loads((app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json").read_text(encoding="utf-8-sig"))
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            guide = Path(tmp_text) / "guide.png"
            guide.write_bytes(b"guide")
            custom_mask = Path(tmp_text) / "custom-mask.png"
            custom_mask.write_bytes(b"mask")
            args = outpaint_video.build_parser().parse_args([
                "--source", "input/example.mp4",
                "--comfy-dir", str(app.ROOT),
                "--custom-mask", str(custom_mask),
                "--dry-run",
            ])
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
                    extra_guides=[
                        {"frame_idx": idx, "strength": 0.5, "image": guide}
                        for idx in (8, 16, 24, 32, 40, 48)
                    ],
                )

        self.assertEqual(prompt["5358"]["inputs"]["mask"], ["9104", 0])
        self.assertEqual(prompt["5266"]["inputs"]["mask"], ["9101", 0])
        self.assertEqual(prompt["9101"]["inputs"]["image"], ["9100", 0])
        self.assertEqual(prompt["9104"]["inputs"]["image"], ["9103", 0])
        self.assertEqual(prompt["3159"]["inputs"]["image"], ["2004", 0])
        guide_nodes = sorted(nid for nid, n in prompt.items() if n["class_type"] == "LTXVAddGuideAdvanced")
        self.assertEqual(len(guide_nodes), 6)

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

        self.assertIn("hue=s=0", filter_graph)

    def test_outpaint_finish_preserves_source_tone_and_sharpens_generated_edges(self) -> None:
        args = finalize_outpaint_output.build_parser().parse_args([
            "--source", "input.mp4", "--source-x", "168", "--source-y", "0",
            "--source-width", "944", "--source-height", "704",
        ])

        filter_graph = finalize_outpaint_output.inverse_filter(args, {"width": 1280, "height": 704})

        self.assertIn("[generated][centre][sourcemask]maskedmerge", filter_graph)
        self.assertIn("unsharp=5:5:0.3500", filter_graph)
        self.assertIn("between(X\\,168\\,1111)", filter_graph)
        self.assertNotIn("lutrgb", filter_graph)

    def test_oumoumad_finish_restores_tone_across_generated_edges_too(self) -> None:
        args = finalize_outpaint_output.build_parser().parse_args([
            "--source", "input.mp4", "--restore-tone", "--black-lift", "0.018", "--gamma", "1.06",
            "--source-x", "168", "--source-y", "0", "--source-width", "944", "--source-height", "704",
        ])

        filter_graph = finalize_outpaint_output.inverse_filter(args, {"width": 1280, "height": 704})

        # LTX paints the edges to match the lifted source, so the restore precedes the split.
        self.assertTrue(filter_graph.startswith("[0:v]format=rgb24,lutrgb="))
        self.assertEqual(filter_graph.count("lutrgb="), 1)
        self.assertIn("split=3[raw][centre][maskbase]", filter_graph)
        self.assertIn("[generated][centre][sourcemask]maskedmerge", filter_graph)

    def test_oumoumad_lifts_user_guides_but_not_previous_chunk_guides(self) -> None:
        workflow_path = app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json"
        args = outpaint_video.build_parser().parse_args([
            "--source", "input/example.mp4",
            "--comfy-dir", str(app.ROOT),
            "--outpaint-lora", outpaint_video.OUMOUMAD_OUTPAINT_LORA,
            "--dry-run",
        ])
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            guide = Path(tmp_text) / "guide.png"
            guide.write_bytes(b"guide")
            lifts = {}
            for auto_guide in (False, True):
                workflow = json.loads(workflow_path.read_text(encoding="utf-8-sig"))
                with (
                    mock.patch.object(outpaint_video, "copy_to_comfy_input", return_value="arp_outpaint/prepared.mp4"),
                    mock.patch.object(outpaint_video, "copy_guide_image_to_comfy_input", return_value="arp_outpaint/guide.png") as copy_guide,
                    mock.patch.object(outpaint_video, "probe_video", return_value={"width": 864, "height": 480, "frames": 24, "fps": 24.0}),
                ):
                    outpaint_video.patch_workflow(
                        args, workflow, app.ROOT / "prepared.mp4", app.ROOT, "arp_outpaint/test",
                        args.prompt, args.negative_prompt, 42, guide, None, auto_guide,
                    )
                lifts[auto_guide] = copy_guide.call_args.kwargs["tone_lift"]

        self.assertEqual(lifts, {False: True, True: False})

    def test_guide_tone_lift_raises_black_off_the_outpaint_trigger(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            guide = folder / "guide.png"
            Image.new("RGB", (8, 4), (0, 0, 0)).save(guide)

            plain = outpaint_video.copy_guide_image_to_comfy_input(guide, folder, 8, 4)
            lifted = outpaint_video.copy_guide_image_to_comfy_input(guide, folder, 8, 4, tone_lift=True)

            self.assertNotEqual(plain, lifted)
            with Image.open(folder / "input" / plain) as image:
                self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
            with Image.open(folder / "input" / lifted) as image:
                self.assertEqual(image.getpixel((0, 0)), (5, 5, 5))

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
        self.assertEqual(prompt["5358"]["inputs"]["mask"], ["9103", 0])
        self.assertEqual(prompt["5266"]["inputs"]["mask"], ["9102", 0])
        self.assertEqual(prompt["5266"]["inputs"]["image_b"], ["5168", 0])

    def test_outpaint_static_mask_is_one_geometric_frame(self) -> None:
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            root = Path(tmp_text)
            prepared = root / "prepared.mp4"
            prepared.write_bytes(b"prepared")
            outpaint_video.signature_path(prepared).write_text(
                json.dumps({
                    "tool": "prepare_outpaint_input.py",
                    "geometry": "crop_then_fit_v1",
                    "source_width": 1456,
                    "source_height": 1080,
                    "target_width": 1280,
                    "target_height": 704,
                    "delivery_width": 1280,
                    "delivery_height": 720,
                    "crop_left": 10,
                    "crop_right": 10,
                    "crop_top": 6,
                    "crop_bottom": 6,
                }),
                encoding="utf-8",
            )
            args = argparse.Namespace(force=True)
            with (
                mock.patch.object(outpaint_video, "ROOT", root),
                mock.patch.object(outpaint_video, "file_fingerprint", return_value={"size": 8, "mtime_ns": 1, "sha256": "test"}),
                mock.patch.object(cv2, "VideoCapture", side_effect=AssertionError("pixel detection is forbidden")),
            ):
                mask_path = outpaint_video.official_mask_image(prepared, args, 1280, 704)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        self.assertEqual(mask.shape, (704, 1280))
        self.assertTrue(np.all(mask[:, :156] == 255))
        self.assertTrue(np.all(mask[:, 156:1124] == 0))
        self.assertTrue(np.all(mask[:, 1124:] == 255))

    def test_outpaint_static_mask_adds_custom_source_regions(self) -> None:
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            root = Path(tmp_text)
            prepared = root / "prepared.mp4"
            custom = root / "custom.png"
            prepared.write_bytes(b"prepared")
            outpaint_video.signature_path(prepared).write_text(
                json.dumps({
                    "tool": "prepare_outpaint_input.py", "geometry": "crop_then_fit_v1",
                    "source_width": 640, "source_height": 480, "target_width": 864,
                    "target_height": 480, "delivery_width": 854, "delivery_height": 480,
                    "crop_left": 0, "crop_right": 0, "crop_top": 0, "crop_bottom": 0,
                }), encoding="utf-8",
            )
            painted = np.zeros((480, 864), dtype=np.uint8)
            painted[:40, 112:152] = 255
            self.assertTrue(cv2.imwrite(str(custom), painted))
            args = argparse.Namespace(force=True, custom_mask=str(custom))
            with mock.patch.object(outpaint_video, "ROOT", root):
                mask_path = outpaint_video.official_mask_image(prepared, args, 864, 480)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        self.assertTrue(np.all(mask[:40, 112:152] == 255))
        self.assertTrue(np.all(mask[80:400, 152:752] == 0))

    def test_recomposition_custom_mask_keeps_generated_pixels_visible(self) -> None:
        args = final_composite.build_parser().parse_args([
            "--outpainted", "outpainted.mp4", "--source", "source.mp4",
            "--output", "output.mp4", "--custom-mask", "mask.png",
            "--output-width", "1920", "--output-height", "1080",
        ])
        graph = final_composite.build_filter(
            args, False, 24.0, True, (1440, 1080), (1920, 1080), [],
        )

        # The mask input is the pre-feathered keep-alpha, so it is used as-is: not negated.
        self.assertIn("[2:v]setpts=N/(24.00000000*TB),scale=1920:1080:flags=bilinear", graph)
        self.assertNotIn("negate", graph)
        # The custom mask multiplies into the source alpha before it is merged, so the
        # masked-out regions stay transparent and the generated pixels below show through.
        self.assertIn("[srcalphamask][customkeep]blend=all_mode=multiply[srcalphacustom]", graph)
        self.assertIn("[srcrgba][srcalphacustom]alphamerge[srcm]", graph)
        # The alpha is built as its own stream; the source is never split and
        # re-alphaextracted per frame just to reach it.
        self.assertNotIn("alphaextract", graph)

    def test_recomposition_custom_mask_feathers_outward_into_the_source(self) -> None:
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mask_path = root / "mask.png"
            mask = np.zeros((90, 160), dtype=np.uint8)
            mask[:, 60:70] = 255  # a vertical strip, like a sprocket-hole column
            cv2.imwrite(str(mask_path), mask)
            with mock.patch.object(final_composite, "ROOT", root):
                keep_path = final_composite.feathered_custom_keep_mask(mask_path, (320, 180), 20)
                keep = cv2.imread(str(keep_path), cv2.IMREAD_GRAYSCALE)

                # Masked pixels stay fully transparent, so the generated fill shows whole.
                self.assertTrue(np.all(keep[:, 120:140] == 0))
                # The source fades back in linearly over the feather, outside the mask only.
                self.assertTrue(0 < keep[90, 145] < keep[90, 150] < 255)
                self.assertTrue(0 < keep[90, 115] < 255)
                self.assertTrue(np.all(keep[:, :100] == 255))
                self.assertTrue(np.all(keep[:, 160:] == 255))
                self.assertEqual(final_composite.feathered_custom_keep_mask(mask_path, (320, 180), 20), keep_path)

    def test_recomposition_custom_mask_invalidates_only_masked_composites(self) -> None:
        base = ["--outpainted", "o.mp4", "--source", "s.mp4", "--output", "out.mp4"]
        with mock.patch.object(final_composite, "file_fingerprint", return_value={}):
            plain = final_composite.signature(final_composite.build_parser().parse_args(base))
            masked = final_composite.signature(final_composite.build_parser().parse_args(base + ["--custom-mask", "m.png"]))
        self.assertNotIn("custom_mask_feather", plain)
        self.assertIn("custom_mask_feather", masked)

    def test_recomposition_progress_tracks_the_ffmpeg_frame_counter(self) -> None:
        from ai_remaster_gui.process_utils import recomp_progress

        # The composite is one long ffmpeg pass, so the bar is driven by the frame
        # count the script announces plus ffmpeg's own running counter.
        log = "\n".join([
            "Composite frames: 175 at 25.000000 fps",
            "frame=   13 fps=0.0 q=-0.0 size=N/A time=00:00:00.52",
            "frame=   96 fps= 28 q=-0.0 size=N/A time=00:00:03.84",
        ])
        self.assertEqual(recomp_progress(log), {"current": 96, "total": 175})

        # ffmpeg writes status updates with carriage returns, so several can land on
        # one line: take the furthest along, not the first.
        packed = "Composite frames: 175 at 25 fps\nframe=  10 q=1 frame=  99 q=1 frame= 140 q=2"
        self.assertEqual(recomp_progress(packed), {"current": 140, "total": 175})

        self.assertEqual(recomp_progress("nothing yet"), {"current": 0, "total": 0})

    def test_recomposition_custom_mask_input_is_never_looped(self) -> None:
        # -loop 1 on the mask makes it an endless input, so blend's frame sync keeps
        # producing frames after the video ends and the encode never terminates: a 7s
        # clip grew past 500MB. -shortest cannot bound it when the source is silent.
        inputs, audio_input = final_composite.input_args(
            Path("outpainted.mp4"), Path("source.mp4"), None, Path("mask.png"),
        )

        self.assertNotIn("-loop", inputs)
        self.assertEqual(inputs.count("-i"), 3)
        self.assertEqual(inputs[-1], "mask.png")
        self.assertEqual(audio_input, "1:a?")

    def test_recomposition_alpha_avoids_per_frame_geq_when_matte_is_static(self) -> None:
        args = final_composite.build_parser().parse_args([
            "--outpainted", "outpainted.mp4", "--source", "source.mp4",
            "--output", "output.mp4",
            "--output-width", "1920", "--output-height", "1080",
        ])
        graph = final_composite.build_filter(
            args, False, 24.0, True, (1440, 1080), (1920, 1080), [],
        )

        # A feather-only matte depends on X/Y alone, so geq runs once on a still frame
        # that alphamerge reuses, instead of once per pixel per plane per frame.
        self.assertIn("trim=end_frame=1,format=gray,geq=lum=", graph)
        self.assertNotIn("geq=r=", graph)

    def test_recomposition_black_matte_runs_on_a_single_luma_plane(self) -> None:
        args = final_composite.build_parser().parse_args([
            "--outpainted", "outpainted.mp4", "--source", "source.mp4",
            "--output", "output.mp4", "--source-black-transparent",
            "--source-black-threshold", "12",
            "--output-width", "1920", "--output-height", "1080",
        ])
        graph = final_composite.build_filter(
            args, False, 24.0, True, (1440, 1080), (1920, 1080), [],
        )

        # max(r,g,b) is folded down by native filters first, so the content-dependent
        # matte walks one plane sampling lum() rather than four sampling r()/g()/b().
        self.assertIn("blend=all_mode=lighten", graph)
        self.assertIn("lum(", graph)
        self.assertNotIn("geq=r=", graph)

    def test_ltx25_blend_protects_with_clean_frames_not_the_green_sentinel(self) -> None:
        # LTXVLaplacianPyramidBlend mixes a Gaussian pyramid of image_b at every scale, so
        # a #66ff00 sentinel there washes green across the whole picture, not just the mask.
        workflow = json.loads((app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-2.5.json").read_text(encoding="utf-8-sig"))
        args = outpaint_video.build_parser().parse_args([
            "--source", "input/example.mp4",
            "--comfy-dir", str(app.ROOT),
            "--ltx-version", "2.5",
            "--dry-run",
        ])
        with (
            mock.patch.object(outpaint_video, "copy_to_comfy_input", return_value="arp_outpaint/prepared.mkv"),
            mock.patch.object(outpaint_video, "copy_reference_frame_to_comfy_input", return_value="arp_outpaint/reference.png"),
            mock.patch.object(outpaint_video, "official_mask_image", return_value=app.ROOT / "mask.png"),
            mock.patch.object(outpaint_video, "generation_mask_image", return_value=app.ROOT / "generation-mask.png"),
            mock.patch.object(outpaint_video, "probe_video", return_value={"width": 1280, "height": 704, "frames": 33, "fps": 24.0}),
        ):
            prompt = outpaint_video.patch_workflow(
                args, workflow, app.ROOT / "prepared.mkv", app.ROOT, "arp_outpaint/test",
                args.prompt, args.negative_prompt, 42,
            )

        blends = {nid: node for nid, node in prompt.items() if node["class_type"] == "LTXVLaplacianPyramidBlend"}
        self.assertTrue(blends)
        for node in blends.values():
            protected = node["inputs"]["image_b"]
            source = prompt.get(str(protected[0]), {})
            self.assertNotEqual(source.get("class_type"), "LTXVInpaintPreprocess")
        # The sampler still receives the sentinel conditioning it needs to inpaint.
        guides = [n for n in prompt.values() if n["class_type"] == "LTXAddVideoICLoRAGuideAdvanced"]
        self.assertTrue(guides)
        self.assertEqual(
            {prompt[str(g["inputs"]["image"][0])]["class_type"] for g in guides},
            {"LTXVInpaintPreprocess"},
        )

    def test_ltx25_chunk_recovers_geometry_through_frame_rate_intermediate(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            root = Path(tmp_text)
            prepared = root / "prepared.mp4"
            prepared.write_bytes(b"prepared")
            outpaint_video.signature_path(prepared).write_text(
                json.dumps({
                    "tool": "prepare_outpaint_input.py",
                    "geometry": "crop_then_fit_v1",
                    "source_width": 4096,
                    "source_height": 3112,
                    "target_width": 1280,
                    "target_height": 704,
                    "delivery_width": 1280,
                    "delivery_height": 720,
                    "crop_left": 10,
                    "crop_right": 10,
                    "crop_top": 0,
                    "crop_bottom": 0,
                }),
                encoding="utf-8",
            )
            converted = root / "prepared_ltx25_24fps.mp4"
            converted.write_bytes(b"converted")
            outpaint_video.signature_path(converted).write_text(
                json.dumps({"tool": "outpaint_video.py/ltx25_fps", "source": str(prepared)}),
                encoding="utf-8",
            )
            chunk = root / "prepared_0000_000000_000097.mp4"
            chunk.write_bytes(b"chunk")
            outpaint_video.split_sidecar_path(chunk).write_text(
                json.dumps({"source": str(converted), "offset_x": 0, "offset_y": 0}),
                encoding="utf-8",
            )

            with mock.patch.object(outpaint_video, "ROOT", root):
                rectangle = outpaint_video.prepared_source_rectangle(chunk, 1280, 704)

        self.assertEqual(rectangle, (168, 0, 1112, 704))

    @staticmethod
    def _pillarbox_prepared(root: Path) -> Path:
        """A prepared input whose signature places 1456x1080 source at columns 156-1124."""
        prepared = root / "prepared.mp4"
        prepared.write_bytes(b"prepared")
        outpaint_video.signature_path(prepared).write_text(
            json.dumps({
                "tool": "prepare_outpaint_input.py", "geometry": "crop_then_fit_v1",
                "source_width": 1456, "source_height": 1080,
                "target_width": 1280, "target_height": 704,
                "delivery_width": 1280, "delivery_height": 720,
                "crop_left": 10, "crop_right": 10, "crop_top": 6, "crop_bottom": 6,
            }),
            encoding="utf-8",
        )
        return prepared

    def test_outpaint_generation_mask_adds_overlap_only_to_pillarbox(self) -> None:
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            root = Path(tmp_text)
            prepared = self._pillarbox_prepared(root)
            exact_path = root / "exact.png"
            exact = np.full((704, 1280), 255, dtype=np.uint8)
            exact[:, 156:1124] = 0
            self.assertTrue(cv2.imwrite(str(exact_path), exact))
            args = argparse.Namespace(force=True, generation_mask_overlap=64, custom_mask="")

            with mock.patch.object(outpaint_video, "ROOT", root):
                generation_path = outpaint_video.generation_mask_image(exact_path, args, prepared, 1280, 704)
            generation = cv2.imread(str(generation_path), cv2.IMREAD_GRAYSCALE)
            exact_after = cv2.imread(str(exact_path), cv2.IMREAD_GRAYSCALE)

        self.assertTrue(np.array_equal(exact_after, exact))
        self.assertTrue(np.all(generation[:, :220] == 255))
        self.assertTrue(np.all(generation[:, 220:1060] == 0))
        self.assertTrue(np.all(generation[:, 1060:] == 255))

    def test_outpaint_generation_mask_keeps_custom_regions(self) -> None:
        # The generation mask drives what LTX actually paints. A custom region that
        # survives only in the exact delivery mask is never generated, so the user sees
        # the untouched sprocket hole (LTX 2.5) or a resurrected source patch (LTX 2.3).
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            root = Path(tmp_text)
            prepared = self._pillarbox_prepared(root)
            custom_path = root / "custom.png"
            painted = np.zeros((704, 1280), dtype=np.uint8)
            painted[300:340, 400:440] = 255
            self.assertTrue(cv2.imwrite(str(custom_path), painted))
            args = argparse.Namespace(force=True, generation_mask_overlap=8, custom_mask=str(custom_path))

            with mock.patch.object(outpaint_video, "ROOT", root):
                exact_path = outpaint_video.official_mask_image(prepared, args, 1280, 704)
                generation_path = outpaint_video.generation_mask_image(exact_path, args, prepared, 1280, 704)
            exact = cv2.imread(str(exact_path), cv2.IMREAD_GRAYSCALE)
            generation = cv2.imread(str(generation_path), cv2.IMREAD_GRAYSCALE)

        self.assertTrue(np.all(exact[300:340, 400:440] == 255))
        # Generated everywhere the delivery mask reveals generation, plus latent headroom.
        self.assertTrue(np.all(generation[300:340, 400:440] == 255))
        self.assertTrue(np.all(generation[292:348, 392:448] == 255))
        # Source outside the painted region and the pillarbox stays protected.
        self.assertTrue(np.all(generation[300:340, 500:900] == 0))

    def test_outpaint_masks_allow_bands_on_both_axes(self) -> None:
        # Signed trim/extend geometry can add border on every edge at once; the mask
        # builders must not reject it as impossible crop-first geometry.
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            root = Path(tmp_text)
            prepared = root / "prepared.mp4"
            prepared.write_bytes(b"prepared")
            outpaint_video.signature_path(prepared).write_text(
                json.dumps({
                    "tool": "prepare_outpaint_input.py", "geometry": "trim_extend_then_fit_v2",
                    "source_width": 640, "source_height": 480,
                    "target_width": 1280, "target_height": 704,
                    "delivery_width": 1280, "delivery_height": 704,
                    "crop_left": -64, "crop_right": -64, "crop_top": -32, "crop_bottom": -32,
                }),
                encoding="utf-8",
            )
            args = argparse.Namespace(force=True, generation_mask_overlap=8, custom_mask="")
            with mock.patch.object(outpaint_video, "ROOT", root):
                exact_path = outpaint_video.official_mask_image(prepared, args, 1280, 704)
                generation_path = outpaint_video.generation_mask_image(exact_path, args, prepared, 1280, 704)
            exact = cv2.imread(str(exact_path), cv2.IMREAD_GRAYSCALE)
            generation = cv2.imread(str(generation_path), cv2.IMREAD_GRAYSCALE)

        for mask in (exact, generation):
            self.assertTrue(np.all(mask[0, :] == 255))
            self.assertTrue(np.all(mask[-1, :] == 255))
            self.assertTrue(np.all(mask[:, 0] == 255))
            self.assertTrue(np.all(mask[:, -1] == 255))
            self.assertTrue(np.any(mask == 0))

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
            frame_result = subprocess.run(
                [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(chunk)],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.stdout.strip(), "opus")
        self.assertEqual(frame_result.stdout.strip(), "9")

    def test_outpaint_stitch_trims_ltx_padding_back_to_real_frame_count(self) -> None:
        ffmpeg = outpaint_video.find_ffmpeg()
        ffprobe = str(Path(ffmpeg).with_name("ffprobe.exe" if Path(ffmpeg).suffix.lower() == ".exe" else "ffprobe"))
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mkv"
            padded = folder / "padded.mkv"
            stitched = folder / "stitched.mkv"
            subprocess.run(
                [
                    ffmpeg, "-y", "-f", "lavfi", "-i", "testsrc2=s=96x64:r=8:d=1.25",
                    "-frames:v", "10", "-c:v", "ffv1", str(source),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            outpaint_video.split_chunk(ffmpeg, source, padded, 0, 10, 8.0, True, intermediate_profile="lossless")
            outpaint_video.stitch_chunks(ffmpeg, [padded], [(0, 0, 10)], stitched, 8.0, True, "lossless")
            counts = []
            for video in (padded, stitched):
                result = subprocess.run(
                    [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(video)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                counts.append(int(result.stdout.strip()))

        self.assertEqual(counts, [17, 10])

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

    def test_outpaint_command_converts_gui_trim_extend_signs_for_backend(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "0", "section_end": ""})
        app.APP.settings["outpaint"].update({
            "edge_left": "120",     # extend in the GUI
            "edge_right": "0",
            "edge_top": "-24",      # trim in the GUI
            "edge_bottom": "48",
        })

        command = app.APP.command_for("outpaint")

        self.assertEqual(command[command.index("--crop-left") + 1], "-120")
        self.assertEqual(command[command.index("--crop-right") + 1], "0")
        self.assertEqual(command[command.index("--crop-top") + 1], "24")
        self.assertEqual(command[command.index("--crop-bottom") + 1], "-48")

    def test_trim_extend_geometry_change_selects_a_fresh_custom_mask(self) -> None:
        base = dict(app.APP.settings["outpaint"])
        base.update({"target_aspect": "16:9", "target_height": "720", "edge_left": "0"})
        extended = {**base, "edge_left": "120"}

        with mock.patch.object(server, "outpaint_work_size_for_source", return_value=(1280, 704)):
            original = server.custom_outpaint_mask_for("input/example.mp4", base)
            changed = server.custom_outpaint_mask_for("input/example.mp4", extended)

        self.assertNotEqual(original, changed)

    def test_outpaint_command_includes_saved_custom_mask(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "0", "section_end": ""})
        with tempfile.TemporaryDirectory(dir=app.ROOT) as folder_text:
            custom = Path(folder_text) / "custom.png"
            custom.write_bytes(b"mask")
            with mock.patch.object(server, "custom_outpaint_mask_for", return_value=custom):
                command = app.APP.command_for("outpaint")

        self.assertEqual(command[command.index("--custom-mask") + 1], app.rel(custom))

    def test_outpaint_and_recomp_commands_pass_saved_frame_masks(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "0", "section_end": ""})
        app.APP.settings["recomp"].update({"outpainted_video": "intermediate/outpainted/example_outpaint.mp4", "source": "input/example.mp4"})
        with tempfile.TemporaryDirectory(dir=app.ROOT) as folder_text:
            archive = Path(folder_text) / "frames.zip"
            with mock.patch.object(server, "frame_outpaint_masks_for", return_value=archive):
                self.assertNotIn("--frame-masks", app.APP.command_for("outpaint"))
                archive.write_bytes(b"zip")
                outpaint = app.APP.command_for("outpaint")
                recomp = app.APP.command_for("recomp")

        self.assertEqual(outpaint[outpaint.index("--frame-masks") + 1], app.rel(archive))
        self.assertEqual(recomp[recomp.index("--frame-masks") + 1], app.rel(archive))

    def test_frame_mask_is_saved_at_working_size_and_an_empty_frame_is_removed(self) -> None:
        import base64
        import io

        from PIL import Image

        import frame_masks

        def data_url(image) -> str:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

        painted = Image.new("RGBA", (32, 16), (0, 0, 0, 0))
        painted.paste((224, 92, 73, 158), (8, 4, 16, 12))
        with tempfile.TemporaryDirectory(dir=app.ROOT) as folder_text:
            folder = Path(folder_text)
            source = folder / "source.mp4"
            source.write_bytes(b"video")
            archive = folder / "frames.zip"
            with (
                mock.patch.object(app.APP, "outpaint_source_for", return_value=app.rel(source)),
                mock.patch.object(server, "frame_outpaint_masks_for", return_value=archive),
                mock.patch.object(server, "outpaint_work_size_for_source", return_value=(64, 32)),
            ):
                server.save_frame_outpaint_mask(7, data_url(painted))
                stored = Image.open(io.BytesIO(frame_masks.read_frame(archive, 7)))
                self.assertEqual((stored.mode, stored.size), ("L", (64, 32)))
                self.assertEqual(stored.getbbox(), (16, 8, 32, 24))
                self.assertTrue(server.frame_outpaint_mask_image(7)["image"].startswith("data:image/png;base64,"))
                self.assertEqual(server.frame_outpaint_mask_image(8)["image"], "")

                server.save_frame_outpaint_mask(7, data_url(Image.new("RGBA", (32, 16), (0, 0, 0, 0))))
                self.assertFalse(archive.exists())

    def test_project_bundles_frame_mask_zips_only_from_the_mask_folder(self) -> None:
        self.assertTrue(project_io.bundle_name_allowed("manifests/outpaint_masks/clip_framemasks_ab12.zip"))
        self.assertFalse(project_io.bundle_name_allowed("scripts/anything.zip"))
        self.assertFalse(project_io.bundle_name_allowed("manifests/outpaint_masks/clip.exe"))
        self.assertTrue(project_io.bundle_name_allowed("manifests/outpaint_masks/clip_custommask_ab12.png"))

    def test_1080p_runpod_outpaint_filters_24gb_gpu_choices(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "0", "section_end": ""})
        app.APP.settings["outpaint"].update({"compute": "runpod", "target_aspect": "16:9", "target_height": "1080"})
        app.APP.settings["cloud"]["runpod_gpu_types"] = "NVIDIA RTX 6000 Ada Generation,NVIDIA GeForce RTX 4090"
        with mock.patch.object(server, "outpaint_work_size_for_source", return_value=(1920, 1088)):
            command = app.APP.command_for("outpaint")

        self.assertEqual(command[command.index("--target-height") + 1], "1080")
        self.assertEqual(command[command.index("--gpu-types") + 1], "NVIDIA RTX 6000 Ada Generation")

    def test_raw_outpaint_signature_fingerprints_custom_mask(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as folder_text:
            folder = Path(folder_text)
            prepared = folder / "prepared.mp4"
            workflow = folder / "workflow.json"
            mask = folder / "mask.png"
            prepared.write_bytes(b"prepared")
            workflow.write_text("{}", encoding="utf-8")
            mask.write_bytes(b"first")
            args = outpaint_video.build_parser().parse_args([
                "--source", "input/example.mp4", "--custom-mask", str(mask), "--dry-run",
            ])
            first = outpaint_video.raw_signature(args, workflow, prepared)
            mask.write_bytes(b"changed")
            second = outpaint_video.raw_signature(args, workflow, prepared)

        self.assertNotEqual(first["custom_mask_fingerprint"], second["custom_mask_fingerprint"])

    def test_outpaint_command_selects_oumoumad_lora(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "0", "section_end": ""})
        app.APP.settings["outpaint"]["outpaint_model"] = "oumoumad"

        command = app.APP.command_for("outpaint")

        self.assertEqual(
            command[command.index("--outpaint-lora") + 1],
            outpaint_video.OUMOUMAD_OUTPAINT_LORA,
        )

    def test_outpaint_command_selects_ltx25_and_generation_fps(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "0", "section_end": ""})
        app.APP.settings["outpaint"].update({"outpaint_model": "ltx25", "generation_fps": "24"})

        command = app.APP.command_for("outpaint")

        self.assertEqual(command[command.index("--ltx-version") + 1], "2.5")
        self.assertEqual(command[command.index("--generation-fps") + 1], "24")
        self.assertIn("outpaint25", Path(app.APP.expected_outputs("outpaint")[0]).name)

    def test_outpaint_command_selects_wan_vace_backend(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "0", "section_end": ""})
        app.APP.settings["outpaint"].update({"outpaint_model": "wanvace"})

        command = app.APP.command_for("outpaint")

        self.assertEqual(command[command.index("--outpaint-backend") + 1], "wan-vace")
        self.assertNotIn("--ltx-version", command)
        self.assertEqual(command[command.index("--outpaint-lora") + 1], outpaint_video.DEFAULT_OUTPAINT_LORA)
        # The GUI locates exactly the files outpaint_video.py writes for this backend.
        expected = Path(app.APP.expected_outputs("outpaint")[0]).name
        self.assertIn("outpaintwan", expected)
        args = outpaint_video.build_parser().parse_args(command[3:])
        self.assertEqual(outpaint_video.outpaint_artifact_tag(args, "outpaint"), "outpaintwan")
        # The GUI plans exactly the chunks the script renders: capped at one Wan pass and
        # overlapping by a pass's context, whatever Chunk seconds and Overlap say.
        values = app.APP.settings["outpaint"] | {"overlap_frames": "8"}
        with mock.patch.object(server, "outpaint_work_size_for_source", return_value=(1280, 704)):
            gui_limits = server.outpaint_chunk_limits("input/example.mp4", values, 8)
        args.overlap_frames = 8
        self.assertEqual(gui_limits, (outpaint_video.max_chunk_frames(args, 1280, 704), outpaint_video.chunk_overlap_frames(args)))
        self.assertEqual(gui_limits, (161, 13))
        # A wider canvas gets shorter chunks from Wan's VRAM budget, in both places.
        with mock.patch.object(server, "outpaint_work_size_for_source", return_value=(1856, 800)):
            wide_limits = server.outpaint_chunk_limits("input/example.mp4", values, 8)
        self.assertEqual(wide_limits[0], outpaint_video.max_chunk_frames(args, *outpaint_video.render_size(args, 1856, 800)))
        self.assertEqual(wide_limits[0], 97)

    def test_outpaint_command_selects_minimax_h3_and_gates_it_on_a_licence(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "0", "section_end": "", "cleanup": "false", "stabilize": "false", "expand_outpaint": "true"})
        app.APP.settings["outpaint"].update({"outpaint_model": "h3", "h3_license_confirmed": "false", "h3_pdd": "true"})

        command = app.APP.command_for("outpaint")
        self.assertEqual(command[command.index("--outpaint-backend") + 1], "h3")
        self.assertNotIn("--h3-license-confirmed", command)
        self.assertIn("--h3-pdd", command)
        self.assertIn("outpainth3", Path(app.APP.expected_outputs("outpaint")[0]).name)
        # H3's open weights exclude the UK, EU, South Korea and USA: nothing runs until the user confirms a licence.
        ok, message = app.APP.run_stage("outpaint")
        self.assertFalse(ok)
        self.assertIn("licence", message)

        app.APP.settings["outpaint"]["h3_license_confirmed"] = "true"
        command = app.APP.command_for("outpaint")
        self.assertIn("--h3-license-confirmed", command)
        args = outpaint_video.build_parser().parse_args(command[3:])
        self.assertTrue(outpaint_video.uses_h3(args))
        self.assertEqual(outpaint_video.outpaint_artifact_tag(args, "outpaint"), "outpainth3")
        # The GUI plans the same chunks as the script: H3's one-pass cap on H3's canvas.
        values = app.APP.settings["outpaint"] | {"overlap_frames": "8"}
        with mock.patch.object(server, "outpaint_work_size_for_source", return_value=(1856, 800)):
            gui_limits = server.outpaint_chunk_limits("input/example.mp4", values, 8)
        args.overlap_frames = 8
        self.assertEqual(gui_limits, (outpaint_video.max_chunk_frames(args, *outpaint_video.render_size(args, 1856, 800)), outpaint_video.chunk_overlap_frames(args)))
        self.assertEqual(gui_limits, (90, 22))

    def test_new_projects_default_to_oumoumad_and_ltx_chunks_are_uncapped(self) -> None:
        defaults = runtime_settings.base_settings()["outpaint"]

        self.assertEqual(defaults["outpaint_model"], "oumoumad")
        self.assertNotIn("wan_window_frames", defaults)
        self.assertEqual(server.outpaint_chunk_limits("input/example.mp4", defaults, 8), (0, 8))
        args = outpaint_video.build_parser().parse_args(["--source", "input/example.mp4"])
        self.assertEqual((outpaint_video.max_chunk_frames(args, 1280, 704), outpaint_video.chunk_overlap_frames(args)), (0, args.overlap_frames))

    def test_switching_outpaint_model_rechunks_and_keeps_guides_on_their_frame(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            source.write_bytes(b"video")
            manifest = folder / "chunks.csv"
            guide = {"frame_idx": 300, "image": "intermediate/outpaint_anchors/demo/g.png", "strength": 1.0}
            app.write_outpaint_chunk_rows(manifest, [
                {"chunk_index": "0", "start_frame": "0", "end_frame": "481", "seed": "42", "prompt_suffix": "a sunny street", "guide_frames": json.dumps([guide])},
                {"chunk_index": "1", "start_frame": "473", "end_frame": "720", "seed": "43"},
            ])
            settings = {
                "global": {"source": app.rel(source), "section_start": "0", "section_end": ""},
                "outpaint": {"target_aspect": "16:9", "target_height": "720", "chunk_seconds": "20", "overlap_frames": "8", "outpaint_model": "wanvace"},
            }
            with (
                mock.patch.object(server, "ensure_source_section_clip"),
                mock.patch.object(server, "resolve_video_source", return_value=source),
                mock.patch.object(server, "video_metrics", return_value={"fps": 24.0, "frames": 720}),
                mock.patch.object(server, "outpaint_work_size_for_source", return_value=(1280, 704)),
                mock.patch.object(server, "outpaint_chunk_manifest_for", return_value=app.rel(manifest)),
                mock.patch.object(server, "outpaint_chunk_dir_for", return_value=folder),
            ):
                rows = app.outpaint_chunks_state(settings)["rows"]
                stored = app.read_outpaint_chunk_rows(manifest)

        self.assertEqual([(int(r["start_frame"]), int(r["end_frame"])) for r in rows[:2]], [(0, 161), (148, 309)])
        self.assertTrue(all(r["max_chunk_frames"] == 161 for r in rows))
        # Frame 300 is now frame 152 of the second chunk, and the stretch keeps its prompt.
        self.assertEqual(json.loads(stored[1]["guide_frames"])[0]["frame_idx"], 152)
        self.assertEqual(stored[0]["guide_frames"], "")
        self.assertEqual(stored[1]["prompt_suffix"], "a sunny street")

    def test_outpaint_command_uses_whole_video_offsets(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "0", "section_end": ""})
        app.APP.settings["outpaint"].update({"offset_x": "-14", "offset_y": "23"})

        command = app.APP.command_for("outpaint")

        self.assertEqual(command[command.index("--offset-x") + 1], "-14")
        self.assertEqual(command[command.index("--offset-y") + 1], "23")

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

    def test_outpaint_manifest_sync_inherits_global_offsets_and_preserves_legacy_overrides(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "chunks.csv"
            outpaint_video.write_chunk_manifest(
                manifest,
                [
                    {"chunk_index": "0", "start_frame": "0", "end_frame": "10", "seed": "42"},
                    {"chunk_index": "1", "start_frame": "8", "end_frame": "18", "seed": "43", "offset_x": "12", "offset_y": "-4"},
                ],
            )

            rows = outpaint_video.sync_chunk_manifest(
                manifest,
                [(0, 0, 10), (1, 8, 18)],
                24.0,
                folder,
                42,
                default_offset_x=-6,
                default_offset_y=9,
            )

            self.assertEqual((rows[0]["offset_mode"], rows[0]["offset_x"], rows[0]["offset_y"]), ("inherit", "-6", "9"))
            self.assertEqual((rows[1]["offset_mode"], rows[1]["offset_x"], rows[1]["offset_y"]), ("custom", "12", "-4"))

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
            self.assertEqual(stored["offset_mode"], "custom")
            self.assertEqual(stored["offset_x"], "-11")
            self.assertEqual(stored["offset_y"], "7")

    def test_outpaint_chunk_can_return_to_whole_video_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "chunks.csv"
            rows = [{"chunk_index": "0", "start_frame": "0", "end_frame": "120", "seed": "42", "offset_mode": "custom", "offset_x": "4", "offset_y": "5"}]
            app.write_outpaint_chunk_rows(manifest, rows)

            with (
                mock.patch.object(server, "outpaint_chunks_state", return_value={"manifest": str(manifest)}),
                mock.patch.dict(app.APP.settings["outpaint"], {"offset_x": "-8", "offset_y": "17"}, clear=False),
            ):
                app.update_outpaint_chunk(0, seed="42", prompt_suffix="", offset_x="4", offset_y="5", offset_override=False)

            stored = app.read_outpaint_chunk_rows(manifest)[0]
            self.assertEqual((stored["offset_mode"], stored["offset_x"], stored["offset_y"]), ("inherit", "-8", "17"))

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

    def test_outpaint_source_for_settings_includes_stabilization(self) -> None:
        settings = {
            "global": {"source": "input/example.mp4", "cleanup": "false", "stabilize": "true"},
            "stabilize": {"smoothing": "12"},
        }
        with (
            mock.patch.object(server, "pipeline_source_text", return_value="input/example.mp4"),
            mock.patch.object(server, "stabilize_output_for", return_value="intermediate/stabilized/example.mkv") as stabilized,
        ):
            source = app.outpaint_source_for_settings(settings)

        self.assertEqual(source, "intermediate/stabilized/example.mkv")
        stabilized.assert_called_once_with("input/example.mp4", settings["stabilize"], "high")

    def test_outpaint_chunk_preview_falls_back_to_finished_render(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            missing_chunk = folder / "missing_chunk.mp4"
            finished = folder / "finished_outpaint.mp4"
            finished.write_bytes(b"video")
            settings = {"outpaint": {"target_aspect": "16:9"}}
            fake_state = {
                "rows": [{
                    "index": 2,
                    "start": 10.0,
                    "end": 14.0,
                    "fps": 24.0,
                    "raw_path": app.rel(missing_chunk),
                }]
            }
            with (
                mock.patch.object(server, "outpaint_chunks_state", return_value=fake_state),
                mock.patch.object(server, "outpaint_source_for_settings", return_value="input/example.mp4"),
                mock.patch.object(server, "outpaint_render_outputs_for_settings", return_value=[finished]),
                mock.patch.object(server, "video_metrics", return_value={"fps": 24.0, "duration": 30.0}),
                mock.patch.object(server, "chunk_frame_preview", return_value="preview.jpg") as preview,
            ):
                result = app.outpaint_chunk_preview(settings, 2, "raw", "middle")

        self.assertEqual(result, "preview.jpg")
        self.assertEqual(preview.call_args.args[0], finished)
        self.assertEqual(preview.call_args.args[1], 12.0)

    def test_outpaint_chunk_state_surfaces_finished_render_when_chunk_cache_is_missing(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "stabilized.mkv"
            finished = folder / "finished_outpaint.mp4"
            source.write_bytes(b"source")
            finished.write_bytes(b"finished")
            finished_mtime = finished.stat().st_mtime_ns
            settings = {
                "global": {"source": "input/example.mp4", "stabilize": "true"},
                "outpaint": {"target_aspect": "16:9", "chunk_seconds": "1", "overlap_frames": "0"},
            }
            with (
                mock.patch.object(server, "ensure_source_section_clip"),
                mock.patch.object(server, "outpaint_source_for_settings", return_value=app.rel(source)),
                mock.patch.object(server, "resolve_video_source", return_value=source),
                mock.patch.object(server, "video_metrics", return_value={"fps": 24.0, "frames": 24}),
                mock.patch.object(server, "outpaint_chunk_manifest_for", return_value=app.rel(folder / "chunks.csv")),
                mock.patch.object(server, "outpaint_chunk_dir_for", return_value=folder / "missing-cache"),
                mock.patch.object(server, "outpaint_render_outputs_for_settings", return_value=[finished]),
            ):
                chunk_state = app.outpaint_chunks_state(settings)

        self.assertTrue(chunk_state["rows"][0]["raw_exists"])
        self.assertEqual(chunk_state["rows"][0]["raw_mtime"], finished_mtime)

    def test_colour_video_wiring_function_is_present_after_static_refactor(self) -> None:
        shots_js = (app.ROOT / "ai_remaster_gui" / "static" / "js" / "render-shots.js").read_text(encoding="utf-8")

        self.assertIn("function wireColourShotVideos()", shots_js)


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

    def test_shot_redetection_keeps_edits_on_unchanged_shot_spans(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            manifest = folder / "shots.csv"
            info = generate_references.VideoInfo(width=16, height=9, fps=24.0, frame_count=30, duration=30 / 24.0)
            previous = [
                generate_references.ReferenceRow(0, 0, 10, 4, 4 / 24.0, folder / "old_a.png", folder / "chosen_a.png"),
                generate_references.ReferenceRow(1, 10, 30, 20, 20 / 24.0, folder / "old_b.png", folder / "old_b_color.png"),
            ]
            generate_references.write_manifest(manifest, source, previous, info)
            _source, fields, rows = app.read_manifest_details(manifest)
            rows[0].update({"enabled": "false", "prompt": "a red coat", "fade_to_next": "true", "crossfade_seconds": "0.5", "note": "mine"})
            app.write_manifest_details(manifest, _source, [*fields, "note"], rows)

            # Re-detection keeps shot 0's span and splits the old shot 1 in two.
            detected = [
                generate_references.ReferenceRow(0, 0, 10, 5, 5 / 24.0, folder / "new_a.png", folder / "new_a_color.png"),
                generate_references.ReferenceRow(1, 10, 18, 12, 12 / 24.0, folder / "new_b.png", folder / "new_b_color.png"),
                generate_references.ReferenceRow(2, 18, 30, 24, 24 / 24.0, folder / "new_c.png", folder / "new_c_color.png"),
            ]
            _previous_source, previous_fields, previous_rows = generate_references.read_manifest(manifest)
            kept = generate_references.carry_over_rows(detected, previous_rows, info.fps)
            generate_references.write_manifest(manifest, source, detected, info, kept, previous_fields)
            _source, fields, rows = app.read_manifest_details(manifest)

        self.assertEqual(list(kept), [0])
        self.assertEqual(detected[0].selected_frame, 4)  # the frame the kept row's reference was made from
        self.assertEqual(Path(rows[0]["source_reference"]).name, "old_a.png")
        self.assertEqual(Path(rows[0]["color_reference"]).name, "chosen_a.png")
        self.assertEqual(
            {key: rows[0][key] for key in ("enabled", "prompt", "fade_to_next", "crossfade_seconds", "note")},
            {"enabled": "false", "prompt": "a red coat", "fade_to_next": "true", "crossfade_seconds": "0.5", "note": "mine"},
        )
        self.assertIn("note", fields)
        self.assertEqual([(row["start_frame"], row["end_frame"]) for row in rows], [("0", "10"), ("10", "18"), ("18", "30")])
        self.assertEqual((rows[1]["enabled"], rows[1]["prompt"]), ("true", ""))

    def test_shot_manifest_rewrite_with_identical_rows_keeps_the_file_untouched(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            manifest = folder / "shots.csv"
            info = generate_references.VideoInfo(width=16, height=9, fps=24.0, frame_count=20, duration=20 / 24.0)
            rows = [generate_references.ReferenceRow(0, 0, 20, 3, 3 / 24.0, folder / "a.png", folder / "a_color.png")]
            generate_references.write_manifest(manifest, source, rows, info)
            os.utime(manifest, ns=(1_000_000_000, 1_000_000_000))

            generate_references.write_manifest(manifest, source, rows, info)

            self.assertEqual(manifest.stat().st_mtime_ns, 1_000_000_000)

    def test_shot_detection_reuses_a_manifest_for_the_same_video_and_settings(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            video = folder / "outpainted.mp4"
            video.write_bytes(b"video")
            frame = folder / "refs" / "cut_0000.png"
            frame.parent.mkdir()
            frame.write_bytes(b"png")
            manifest = folder / "shots.csv"
            manifest.write_text(
                f"# source_video={video}\nenabled,start_frame,end_frame,selected_frame,end,source_reference,color_reference,prompt\n"
                f"true,0,20,3,00:00:01.000,{frame},{folder / 'color.png'},an edit\n",
                encoding="utf-8",
            )
            os.utime(video, ns=(1_000_000_000, 1_000_000_000))
            argv = ["generate_references.py", "--source-video", str(video), "--output-manifest", str(manifest)]

            frames_read: list[int] = []

            def read_frame(_path, index):
                frames_read.append(index)
                return generate_references.np.zeros((9, 16, 3), dtype=generate_references.np.uint8)

            def run(*extra: str, frame_count: int = 20) -> None:
                info = generate_references.VideoInfo(width=16, height=9, fps=24.0, frame_count=frame_count, duration=frame_count / 24.0)
                with (
                    mock.patch.object(sys, "argv", [*argv, *extra]),
                    mock.patch.object(generate_references, "probe_video", return_value=info),
                    mock.patch.object(generate_references, "read_frame", side_effect=read_frame),
                    mock.patch.object(generate_references, "sample_video", side_effect=AssertionError("re-detected")),
                ):
                    generate_references.main()

            # Written before manifests were signed, for this video at this length: kept and signed.
            run()
            self.assertTrue(common.signature_path(manifest).exists())
            frames_read.clear()
            run()  # signed and current: not even the frames are re-read
            self.assertEqual(frames_read, [])
            video.write_bytes(b"re-rendered video")
            run()  # new pixels, same length: every shot and edit kept, the frame refreshed
            self.assertEqual(frames_read, [3])
            self.assertIn("an edit", manifest.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(AssertionError, "re-detected"):
                run("--shot-threshold", "0.2")  # a detection setting changed
            video.write_bytes(b"a longer re-render")
            with self.assertRaisesRegex(AssertionError, "re-detected"):
                run(frame_count=30)  # the video's length changed
            self.assertIn("an edit", manifest.read_text(encoding="utf-8"))

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
        self.assertIn("hf auth login --force", message)
        open_browser.assert_called_once_with(server.OUTPAINT_LICENSE_URL)

    def test_outpaint_403_opens_failing_repository_and_keeps_auth_guidance(self) -> None:
        model_url = "https://huggingface.co/owner/gated-model"
        error = outpaint_video.HuggingFaceAccessError(
            "Approve access, then run `hf auth login --force` using the same account.",
            model_url=model_url,
        )
        with mock.patch.object(outpaint_video.webbrowser, "open", return_value=True) as open_browser:
            message = outpaint_video.outpaint_access_error_message(error)

        self.assertIn("hf auth login --force", message)
        self.assertIn("opened the required model repository", message)
        open_browser.assert_called_once_with(model_url)

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

    def test_webm_source_section_uses_mkv_processing_master(self) -> None:
        app.APP.settings["global"].update(
            {"source": "input/example.webm", "section_start": "12", "section_end": "24"}
        )

        output = app.source_section_output_for(app.APP.settings)

        self.assertEqual(output.suffix, ".mkv")

    def test_mjpeg_avi_gets_cached_h264_browser_proxy(self) -> None:
        ffmpeg = common.find_ffmpeg()
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "archival transfer.avi"
            cache = folder / "playback-cache"
            subprocess.run(
                [
                    ffmpeg, "-y", "-f", "lavfi", "-i", "testsrc2=s=160x120:r=17:d=0.25",
                    "-c:v", "mjpeg", "-q:v", "3", str(source),
                ],
                check=True,
                capture_output=True,
            )
            with mock.patch.object(media, "SOURCE_PLAYBACK_DIR", cache):
                untouched = folder / "no proxy.avi"
                shutil.copy2(source, untouched)
                first = media.browser_playback_for(str(source), create=True)
                second = media.browser_playback_for(str(source), create=False)
                # Every tab's /media request resolves to the proxy without re-probing.
                served = media.existing_browser_playback(source)
                unproxied = media.existing_browser_playback(untouched)

            proxy = app.resolve(first)
            self.assertEqual(served, proxy)
            self.assertIsNone(unproxied)
            ffprobe = str(Path(ffmpeg).with_name("ffprobe.exe" if Path(ffmpeg).suffix.lower() == ".exe" else "ffprobe"))
            result = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1", str(proxy)],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertEqual(first, second)
        self.assertEqual(proxy.suffix, ".mp4")
        self.assertEqual(result.stdout.strip(), "h264")

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
            progress: list[int] = []

            with (
                mock.patch.object(media, "ROOT", root),
                mock.patch.object(media, "local_tool", return_value="ffmpeg"),
                mock.patch.object(media.subprocess, "run", side_effect=fake_section_ffprobe),
                mock.patch.object(media.subprocess, "Popen", side_effect=fake_section_trim(ffmpeg_commands)),
            ):
                output = media.ensure_source_section_clip(settings, progress=lambda percent, _label: progress.append(percent))

            self.assertIn("intermediate/source_sections/example_", output)
            self.assertEqual(progress, [0, 50, 100])
            self.assertEqual(len(ffmpeg_commands), 1)
            command = ffmpeg_commands[0]
            self.assertIn("-vf", command)
            self.assertEqual(command[command.index("-vf") + 1], "setpts=PTS-STARTPTS")
            self.assertIn("-af", command)
            self.assertEqual(command[command.index("-af") + 1], "asetpts=PTS-STARTPTS")
            self.assertEqual(command[command.index("-c:a") + 1], "libopus")
            self.assertNotIn("copy", command)

    def test_source_section_job_trims_in_background_and_reports_progress(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            root = Path(tmp_text)
            source = root / "input" / "example.mkv"
            source.parent.mkdir()
            source.write_bytes(b"source")
            settings = {"global": {"source": str(source), "section_start": "12", "section_end": "24"}}
            release = threading.Event()
            ffmpeg_commands: list[list[str]] = []

            with (
                mock.patch.object(media, "ROOT", root),
                mock.patch.object(media, "local_tool", return_value="ffmpeg"),
                mock.patch.object(media.subprocess, "run", side_effect=fake_section_ffprobe),
                mock.patch.object(media.subprocess, "Popen", side_effect=fake_section_trim(ffmpeg_commands, release)),
            ):
                first = media.source_section_job_state(settings)
                again = media.source_section_job_state(settings)
                release.set()
                deadline = time.monotonic() + 5
                done = again
                while done["pending"] and time.monotonic() < deadline:
                    time.sleep(0.02)
                    done = media.source_section_job_state(settings)
                output = media.source_section_output_for(settings)

            self.assertTrue(first["pending"])
            self.assertTrue(again["pending"])
            self.assertFalse(done["pending"])
            self.assertEqual(done["error"], "")
            self.assertEqual(len(ffmpeg_commands), 1, "a second poll must not start a second trim")
            self.assertTrue(output.exists())

    def test_failed_source_section_job_waits_for_retry(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            root = Path(tmp_text)
            source = root / "input" / "broken.mkv"
            source.parent.mkdir()
            source.write_bytes(b"source")
            settings = {"global": {"source": str(source), "section_start": "1", "section_end": "3"}}
            ffmpeg_commands: list[list[str]] = []

            def wait_for_job() -> dict:
                deadline = time.monotonic() + 5
                job = media.source_section_job_state(settings)
                while job["pending"] and time.monotonic() < deadline:
                    time.sleep(0.02)
                    job = media.source_section_job_state(settings)
                return job

            with (
                mock.patch.object(media, "ROOT", root),
                mock.patch.object(media, "local_tool", return_value="ffmpeg"),
                mock.patch.object(media.subprocess, "run", side_effect=fake_section_ffprobe),
                mock.patch.object(media.subprocess, "Popen", side_effect=fake_section_trim(ffmpeg_commands, fail=True)),
            ):
                failed = wait_for_job()
                polled = media.source_section_job_state(settings)
                retried = media.source_section_job_state(settings, retry=True)
                wait_for_job()

            self.assertEqual(failed["error"], "broken.mkv: Invalid data found")
            self.assertFalse(polled["pending"])
            self.assertEqual(polled["error"], failed["error"])
            self.assertTrue(retried["pending"])
            self.assertEqual(len(ffmpeg_commands), 2)

    def test_outpaint_state_skips_chunks_while_section_trims(self) -> None:
        pending = {"pending": True, "percent": 40, "label": "Trimming source section", "error": ""}
        with (
            mock.patch.object(server, "source_section_job_state", return_value=pending),
            mock.patch.object(server, "outpaint_chunks_state") as chunks,
            mock.patch.object(server, "aspect_preview_frame_for_settings") as aspect,
        ):
            payload = server.APP.state("outpaint")

        self.assertEqual(payload["source_section_job"], pending)
        self.assertEqual(payload["outpaint_chunks"]["rows"], [])
        chunks.assert_not_called()
        aspect.assert_not_called()

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
            "edge_left": "-88",
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
        self.assertEqual(app.APP.settings["outpaint"]["edge_left"], "0")
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

    def test_background_source_analysis_preserves_saved_recolorization_choice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            source = Path(tmp_text) / "already-colour.mp4"
            source.write_bytes(b"placeholder")
            signature = (str(source), source.stat().st_size, source.stat().st_mtime_ns)
            key = app.source_analysis_key(signature)
            app.APP.settings["global"].update({"source": str(source), "colorize": "true"})
            with (
                mock.patch.object(server, "ffprobe_basic_info", return_value={}),
                mock.patch.object(server, "source_previews_for_analysis", return_value=[]),
                mock.patch.object(server, "source_monochrome_cached", return_value=False),
            ):
                app.APP.analyze_source_media(signature, key)

        self.assertEqual(app.APP.settings["global"]["colorize"], "true")

    def test_background_source_analysis_defaults_new_colour_source_to_no_colorization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            source = Path(tmp_text) / "new-colour.mp4"
            source.write_bytes(b"placeholder")
            signature = (str(source), source.stat().st_size, source.stat().st_mtime_ns)
            key = app.source_analysis_key(signature)
            app.APP.settings["global"].update({"source": str(source), "colorize": "true"})
            with app.APP.source_analysis_lock:
                app.APP.source_tone_default_keys.add(key)
            with (
                mock.patch.object(server, "ffprobe_basic_info", return_value={}),
                mock.patch.object(server, "source_previews_for_analysis", return_value=[]),
                mock.patch.object(server, "source_monochrome_cached", return_value=False),
            ):
                app.APP.analyze_source_media(signature, key)

        self.assertEqual(app.APP.settings["global"]["colorize"], "false")

    def test_project_payload_round_trips_settings_with_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            path = Path(tmp_text) / "demo.arpp"
            app.APP.settings["global"].update({"source": "input/example.mp4"})
            path.write_text(json.dumps(app.project_payload(app.APP.settings)), encoding="utf-8")

            loaded = app.read_project_file(path)

        self.assertEqual(loaded["global"]["source"], "input/example.mp4")
        self.assertIn("schema_version", app.project_payload(app.APP.settings))

    def test_legacy_video_project_does_not_inherit_active_image_sequence(self) -> None:
        active = copy.deepcopy(app.APP.settings)
        active["global"].update({
            "source": "intermediate/source_sequences/berlin.mkv",
            "source_images": json.dumps(["G:/Berlin/frame_0001.dpx", "G:/Berlin/frame_0002.dpx"]),
            "source_sequence_preview": "intermediate/source_sequences/berlin_preview.mp4",
            "source_sequence_fps": "18",
        })
        active["references"]["openai_api_key"] = "sk-machine-local"
        legacy_project = {
            "schema_version": 2,
            "settings": {
                "global": {
                    "source": "G:/Hiroshima/Hiroshima.mov",
                    "colorize": "true",
                    "upscale": "true",
                }
            },
        }

        with mock.patch.object(project_io, "load_settings", return_value=active):
            loaded = project_io.load_project_payload(legacy_project)

        self.assertEqual(loaded["global"]["source"], "G:/Hiroshima/Hiroshima.mov")
        self.assertEqual(loaded["global"]["source_images"], "")
        self.assertEqual(loaded["global"]["source_sequence_preview"], "")
        self.assertEqual(loaded["global"]["source_sequence_fps"], "24")
        self.assertEqual(loaded["references"]["openai_api_key"], "sk-machine-local")

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
            settings["cloud"].update({
                "runpod_api_key": "rp-secret",
                "minimax_api_key": "mm-secret",
                "kie_api_key": "kie-secret",
                "huggingface_token": "hf-secret",
            })

            app.write_project_file(project, settings)

            with zipfile.ZipFile(project) as archive:
                names = set(archive.namelist())
                payload = json.loads(archive.read("project.json").decode("utf-8"))

            self.assertIn(app.rel(manifest).replace("\\", "/"), names)
            self.assertIn(app.rel(source).replace("\\", "/"), names)
            self.assertIn(app.rel(color).replace("\\", "/"), names)
            self.assertNotIn("openai_api_key", payload["settings"]["references"])
            for secret in ("runpod_api_key", "minimax_api_key", "kie_api_key", "huggingface_token"):
                self.assertNotIn(secret, payload["settings"]["cloud"])

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

    def test_loading_older_project_adopts_its_work_under_current_names(self) -> None:
        chunk_header = "chunk_index,start_frame,end_frame,seed,guide_frames,auto_start_guide\n"
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            old_chunks, new_chunks = folder / "The_chunks_old.csv", folder / "The_chunks_new.csv"
            old_video, new_video = folder / "The_outpaint_old.mkv", folder / "The_outpaint_new.mkv"
            old_shots, new_shots = folder / "The_shots_old.csv", folder / "The_shots_new.csv"
            old_chunks.write_text(chunk_header + '0,0,480,42,"[{""frame_idx"": 9, ""image"": ""g.png""}]",true\n', encoding="utf-8")
            # A default plan written under the new name by an earlier open of the project.
            new_chunks.write_text(chunk_header + "0,0,480,42,,true\n", encoding="utf-8")
            old_video.write_bytes(b"finished outpaint")
            old_shots.write_text(f"# source_video={server.rel(old_video)}\nenabled,start_frame\ntrue,0\n", encoding="utf-8")

            settings = copy.deepcopy(app.APP.settings)
            settings["outpaint"]["manifest"] = server.rel(old_chunks)
            settings["recomp"]["outpainted_video"] = server.rel(old_video)
            settings["references"]["manifest"] = server.rel(old_shots)
            app.APP.settings = settings
            with (
                mock.patch.object(app.APP, "outpaint_enabled", return_value=True),
                mock.patch.object(app.APP, "outpaint_source_for", return_value="input/section.mkv"),
                mock.patch.object(server, "outpaint_chunk_manifest_for", return_value=server.rel(new_chunks)),
                mock.patch.object(server, "outpaint_output_for", return_value=server.rel(new_video)),
                mock.patch.object(server, "manifest_for_outpainted", return_value=server.rel(new_shots)),
                mock.patch.object(server, "colorized_outputs_for_manifest", return_value=[]),
            ):
                app.APP.adopt_saved_project_artifacts()

            self.assertIn("g.png", new_chunks.read_text(encoding="utf-8"))
            self.assertIn(",42,,true", (folder / "The_chunks_new.csv.bak").read_text(encoding="utf-8"))
            self.assertEqual(new_video.read_bytes(), b"finished outpaint")
            self.assertEqual(server.manifest_source_video(new_shots), server.rel(new_video))
            self.assertEqual(server.read_manifest(new_shots), [{"enabled": "true", "start_frame": "0"}])

    def test_loading_project_without_plan_path_finds_plan_under_older_name(self) -> None:
        chunk_header = "chunk_index,start_frame,end_frame,seed,guide_frames,auto_start_guide\n"
        plans = app.ROOT / "manifests" / "outpaint_chunks"
        source = "intermediate/source_sections/Zzarptest_0000000000_0000001000_high.mkv"
        values = {"target_aspect": "16:9", "target_height": "720", "edge_left": "-10", "edge_right": "-10"}
        crop = [10, 10, 0, 0]
        old_section_name = "Zzarptest_0000000000_0000001000.mp4"
        edited = plans / f"{aid.legacy_outpaint_basenames(old_section_name, '16:9', 1280, 704, crop, False, 'chunks')[0]}.csv"
        untouched = plans / f"{aid.legacy_outpaint_basenames(Path(source).name, '16:9', 1280, 704, crop, False, 'chunks')[0]}.csv"
        current = plans / f"{aid.outpaint_basename(Path(source).name, '16:9', 1280, 704, crop, False, 'chunks')}.csv"
        created = [edited, untouched, current, Path(str(current) + ".bak")]
        try:
            plans.mkdir(parents=True, exist_ok=True)
            edited.write_text(chunk_header + '0,0,480,42,"[{""frame_idx"": 9, ""image"": ""g.png""}]",true\n', encoding="utf-8")
            untouched.write_text(chunk_header + "0,0,480,42,,true\n", encoding="utf-8")
            current.write_text(chunk_header + "0,0,480,42,,true\n", encoding="utf-8")
            with (
                mock.patch.object(server, "resolve_video_source", side_effect=lambda text: app.ROOT / text),
                mock.patch.object(server, "outpaint_work_size_for_source", return_value=(1280, 704)),
            ):
                self.assertEqual(server.legacy_outpaint_chunk_manifests(source, values), [edited, untouched])

                settings = copy.deepcopy(app.APP.settings)
                settings["outpaint"].update(values, manifest="")
                app.APP.settings = settings
                with (
                    mock.patch.object(app.APP, "outpaint_enabled", return_value=True),
                    mock.patch.object(app.APP, "outpaint_source_for", return_value=source),
                    mock.patch.object(server, "outpaint_output_for", return_value=""),
                    mock.patch.object(server, "manifest_for_outpainted", return_value=""),
                    mock.patch.object(server, "colorized_outputs_for_manifest", return_value=[]),
                ):
                    app.APP.adopt_saved_project_artifacts()

            self.assertIn("g.png", current.read_text(encoding="utf-8"))
            self.assertTrue(Path(str(current) + ".bak").is_file())
        finally:
            for path in created:
                path.unlink(missing_ok=True)

    def test_guide_images_outside_arp_are_copied_in(self) -> None:
        chunk_header = "chunk_index,start_frame,end_frame,seed,guide_image,guide_frames,auto_start_guide\n"
        with tempfile.TemporaryDirectory() as outside_text, tempfile.TemporaryDirectory(dir=app.ROOT) as inside_text:
            outside, inside = Path(outside_text), Path(inside_text)
            manifest = inside / "Zzarptest_chunks_import.csv"
            imported_dir = app.ROOT / "intermediate" / "outpaint_guides" / manifest.stem
            try:
                (outside / "a").mkdir()
                (outside / "b").mkdir()
                # Same name, different content: each must keep its own copy.
                (outside / "a" / "guide.png").write_bytes(b"first guide")
                (outside / "b" / "guide.png").write_bytes(b"second guide")
                (outside / "a" / "guide.png.sig.json").write_text("{}", encoding="utf-8")
                local = inside / "local.png"
                local.write_bytes(b"already inside")
                frames = [
                    {"frame_idx": 0, "image": str(outside / "a" / "guide.png"), "image_previous": server.rel(local)},
                    {"frame_idx": 9, "image": str(outside / "missing.png")},
                ]
                with manifest.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    handle.write(chunk_header)
                    writer.writerow([0, 0, 480, 42, str(outside / "b" / "guide.png"), json.dumps(frames), "true"])

                self.assertEqual(server.import_outside_guide_images(manifest), 2)
                row = server.read_outpaint_chunk_rows(manifest)[0]
                guides = json.loads(row["guide_frames"])
                first, second = server.resolve(guides[0]["image"]), server.resolve(row["guide_image"])
                self.assertTrue(paths.is_within(first, imported_dir) and paths.is_within(second, imported_dir))
                self.assertEqual(first.read_bytes(), b"first guide")
                self.assertEqual(second.read_bytes(), b"second guide")
                self.assertTrue(Path(str(first) + ".sig.json").is_file())
                self.assertEqual(guides[0]["image_previous"], server.rel(local))
                self.assertEqual(guides[1]["image"], str(outside / "missing.png"))
                # Already imported: nothing changes and the plan is left alone.
                self.assertEqual(server.import_outside_guide_images(manifest), 0)
            finally:
                shutil.rmtree(imported_dir, ignore_errors=True)

    def test_adopting_older_work_never_overwrites_edited_current_files(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            saved, current = folder / "old.csv", folder / "new.csv"
            saved.write_text("older", encoding="utf-8")
            current.write_text("newer", encoding="utf-8")

            self.assertEqual(server.adopt_saved_artifact(server.rel(saved), server.rel(current)), "")
            self.assertEqual(current.read_text(encoding="utf-8"), "newer")
            self.assertEqual(server.adopt_saved_artifact(server.rel(folder / "missing.csv"), server.rel(folder / "other.csv")), "")
            self.assertFalse((folder / "other.csv").exists())

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

    def test_outpaint_chunk_reused_when_identical_guides_move_folders(self) -> None:
        # A chunk rendered with guides outside ARP's folder, then recovered into imported/ (same
        # bytes, new path), was re-rendered because the signature compared guide paths.
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            prepared = folder / "prepared.mp4"
            workflow = folder / "workflow.json"
            outside = folder / "elsewhere" / "chunk_0000_guide_00.png"
            imported = folder / "imported" / "chunk_0000_guide_00_cf5b91d7.png"
            start = folder / "elsewhere" / "guide_prev.png"
            moved_start = folder / "imported" / "guide_prev.png"
            output = folder / "raw_0000.mp4"
            for path, data in ((prepared, b"prepared"), (workflow, b"{}"), (outside, b"guide bytes"), (start, b"start")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            imported.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(outside, imported)
            shutil.copy2(start, moved_start)
            output.write_bytes(b"rendered")
            args = argparse.Namespace(
                target_aspect="16:9", target_height=480, outpaint_all_black_regions=False,
                prompt="outpaint", negative_prompt="", guide_strength=0.7,
                load_video_node_id="5060", save_node_id="5076", extra_save_node_id=["5069"],
                output_node_id="5076", model_backend="gguf", gguf_model="model.gguf",
                video_vae="vae.safetensors", outpaint_lora="lora.safetensors",
                chunk_seconds=20.0, overlap_frames=8,
            )

            def signature(guide: Path, start_guide: Path) -> dict:
                return outpaint_video.raw_signature(args, workflow, prepared, 42, guide_image=start_guide,
                                                    extra_guides=[{"frame_idx": 237, "strength": 0.95, "image": guide}])

            common.write_signature(output, signature(outside, start))
            self.assertTrue(common.resumable_output(output, signature(imported, moved_start)))

            imported.write_bytes(b"edited guide")  # different content still re-renders
            self.assertFalse(common.resumable_output(output, signature(imported, moved_start)))
            moved_start.unlink()  # a path with no fingerprint (file gone) is still compared
            imported.write_bytes(b"guide bytes")
            self.assertFalse(common.resumable_output(output, signature(imported, moved_start)))

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
            self.assertIn("--no-normalize-to-source-size", command)
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
        self.assertEqual(command[command.index("--reference-count") + 1], "3")  # the default count
        self.assertIn("--no-normalize-to-source-size", command)
        self.assertNotIn("qwen_colorize_references.py", " ".join(command))

        app.APP.settings["references"]["previous_reference_count"] = "5"
        command = app.APP.command_for("references")
        self.assertEqual(command[command.index("--reference-count") + 1], "5")
        app.APP.settings["references"]["openai_send_references"] = "false"
        self.assertNotIn("--reference-count", app.APP.command_for("references"))

    def test_previous_reference_count_is_ticked_editable_and_clamped(self) -> None:
        count = server.previous_reference_count
        self.assertEqual(count({}), 0)
        self.assertEqual(count({"openai_send_references": "false", "previous_reference_count": "6"}), 0)
        self.assertEqual(count({"openai_send_references": "true"}), 3)
        self.assertEqual(count({"openai_send_references": "true", "previous_reference_count": "6"}), 6)
        self.assertEqual(count({"openai_send_references": "true", "previous_reference_count": "40"}), 9)
        self.assertEqual(count({"openai_send_references": "true", "previous_reference_count": "-2"}), 0)
        self.assertEqual(count({"openai_send_references": "true", "previous_reference_count": "lots"}), 3)

    def test_qwen21_reference_generation_describes_stills_and_sends_references_only_with_descriptions(self) -> None:
        refs = app.APP.settings["references"]
        # Defaults: describe each still, with two previous references. The OpenAI count is separate.
        for key in ("qwen21_describe", "qwen21_send_references", "qwen21_reference_count"):
            refs.pop(key, None)
        refs.update({"manifest": "manifests/references/demo.csv", "method": "qwen21", "openai_send_references": "true", "previous_reference_count": "6"})
        command = app.APP.command_for("references")
        self.assertIn("qwen21_generate_reference.py", " ".join(command))
        self.assertNotIn("qwen_colorize_references.py", " ".join(command))
        self.assertIn("--describe", command)
        self.assertEqual(command[command.index("--reference-count") + 1], "2")

        self.assertEqual(server.qwen21_reference_args({"qwen21_reference_count": "9"}), ["--describe", "--reference-count", "2"])  # held to two
        self.assertEqual(server.qwen21_reference_args({"qwen21_reference_count": "1"}), ["--describe", "--reference-count", "1"])
        self.assertEqual(server.qwen21_reference_args({"qwen21_send_references": "false"}), ["--describe"])
        # Without descriptions 2.1 copies the references, so none are sent.
        self.assertEqual(server.qwen21_reference_args({"qwen21_describe": "false", "qwen21_send_references": "true"}), [])
        refs.update({"qwen21_describe": "true", "qwen21_send_references": "true", "qwen21_reference_count": "1"})

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "refs.csv"
            source = folder / "bw.png"
            color = folder / "color.png"
            source.write_bytes(b"bw")
            app.write_manifest_details(
                manifest, "input/example.mp4", ["enabled", "end", "source_reference", "color_reference"],
                [{"enabled": "true", "end": "00:00:01.000", "source_reference": app.rel(source), "color_reference": app.rel(color)}],
            )
            single, output = server.qwen21_reference_regeneration_command(str(manifest), 0)
        self.assertIn("qwen21_generate_reference.py", " ".join(single))
        self.assertIn("--describe", single)
        self.assertEqual(single[single.index("--reference-count") + 1], "1")
        self.assertEqual(single[single.index("--row-index") + 1], "0")
        self.assertEqual(output, app.rel(color))

    def test_qwen21_reference_graph_feeds_previous_references_as_extra_images(self) -> None:
        import qwen21_generate_reference as qwen21_ref

        graph = qwen21_ref.build_prompt("still.png", ["ref1.png", "ref2.png"], "colour it", seed=3, prefix="p", canvas=(1856, 800))
        encode = graph["20"]["inputs"]
        self.assertEqual(encode["images.image_1"], ["1", 0])
        self.assertEqual(graph[encode["images.image_2"][0]]["inputs"]["image"], "ref1.png")
        self.assertEqual(graph[encode["images.image_3"][0]]["inputs"]["image"], "ref2.png")
        self.assertNotIn("images.image_4", encode)
        self.assertEqual(graph["23"]["inputs"]["latent_image"], ["20", 2])  # latent follows the still
        self.assertEqual(graph["23"]["class_type"], "SamplerCustomAdvanced")  # turbo by default
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    self.assertIn(value[0], graph)

    def test_qwen21_reference_prompt_carries_the_description_and_the_copy_check_spots_copies(self) -> None:
        import argparse

        import numpy as np

        import qwen21_generate_reference as qwen21_ref

        args = argparse.Namespace(prompt="Colorize this image.", prompt_suffix="Preserve detail.", add_prompt="Warm lamps.")
        described = qwen21_ref.still_prompt(args, "Two men talk by a lamp.", 2)
        self.assertTrue(described.startswith("Colorize this image. <image1> is a black-and-white film frame that shows: Two men talk by a lamp."))
        self.assertIn("<image2>, <image3> are other shots of the same scene", described)
        self.assertTrue(described.endswith("Preserve detail. Warm lamps."))
        self.assertIn("<image2> is another shot", qwen21_ref.still_prompt(args, "A man.", 1))
        plain = qwen21_ref.still_prompt(args, "", 0)
        self.assertEqual(plain, "Colorize this image. Preserve detail. Warm lamps.")

        graph = qwen21_ref.describe_prompt("still.png")
        self.assertEqual(graph["50"]["class_type"], "TextGenerate")
        self.assertEqual(graph["50"]["inputs"]["image"], ["1", 0])
        self.assertEqual(graph[qwen21_ref.DESCRIBE_NODE_ID]["class_type"], "PreviewAny")

        rng = np.random.default_rng(0)
        scene = (rng.random((90, 160)) * 255).astype(np.uint8)
        other = (rng.random((90, 160)) * 255).astype(np.uint8)
        grey = np.stack([scene] * 3, axis=2)
        tinted = np.clip(np.stack([scene * 1.1, scene * 0.95, scene * 0.7], axis=2), 0, 255).astype(np.uint8)
        self.assertGreater(qwen21_ref.structure_score(tinted, grey), 0.95)  # coloured, same picture
        self.assertLess(qwen21_ref.structure_score(np.stack([other] * 3, axis=2), grey), qwen21_ref.COPY_THRESHOLD)
        # A different size still compares on the source's grid.
        bigger = np.repeat(np.repeat(tinted, 2, axis=0), 2, axis=1)
        self.assertGreater(qwen21_ref.structure_score(bigger, grey), 0.9)

    def test_openai_reference_max_size_preserves_aspect_and_api_limits(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            source = Path(tmp_text) / "source.png"
            Image.new("RGB", (1200, 700)).save(source)

            resolved = openai_generate_reference.resolved_output_size(
                "max", source, "gpt-image-2.5-sunburst"
            )

        width, height = (int(part) for part in resolved.split("x"))
        self.assertEqual(width % 16, 0)
        self.assertEqual(height % 16, 0)
        self.assertLessEqual(max(width, height), 3840)
        self.assertLessEqual(width * height, 8_294_400)
        self.assertGreater(width * height, 8_000_000)
        self.assertAlmostEqual(width / height, 1200 / 700, delta=0.01)

    def test_openai_reference_size_and_quality_default_to_auto_and_keep_the_users_choice(self) -> None:
        values = runtime_settings.base_settings()
        values["references"].update(
            {
                "openai_image_model": "gpt-image-2.5-sunburst",
                "openai_image_size": "auto",
                "openai_image_quality": "high",
            }
        )
        normalized = runtime_settings.normalize_settings(values, include_newest_source=False)
        self.assertEqual(normalized["references"]["openai_image_size"], "auto")
        self.assertEqual(normalized["references"]["openai_image_quality"], "high")

        # Settings an earlier build forced up to Maximum go back to Auto once...
        forced = runtime_settings.base_settings()
        forced["references"].update({"openai_image_size": "max", "openai_image_quality": "max"})
        forced["references"].pop("openai_cost_defaults", None)
        normalized = runtime_settings.normalize_settings(forced, include_newest_source=False)
        self.assertEqual(normalized["references"]["openai_image_size"], "auto")
        self.assertEqual(normalized["references"]["openai_image_quality"], "auto")

        # ...but Maximum chosen afterwards is kept.
        normalized["references"].update({"openai_image_size": "max", "openai_image_quality": "max"})
        again = runtime_settings.normalize_settings(normalized, include_newest_source=False)
        self.assertEqual(again["references"]["openai_image_size"], "max")
        self.assertEqual(again["references"]["openai_image_quality"], "max")

    def test_openai_reference_command_defaults_to_auto(self) -> None:
        app.APP.settings["references"].update(
            {"method": "openai", "manifest": "manifests/references/demo.csv", "openai_api_key": "sk-test",
             "openai_image_size": "", "openai_image_quality": ""}
        )
        command = app.APP.command_for("references")
        self.assertEqual(command[command.index("--size") + 1], "auto")
        self.assertEqual(command[command.index("--quality") + 1], "auto")
        args = openai_generate_reference.build_parser().parse_args(["--manifest", "m.csv", "--api-key", "k", "--prompt", "p"])
        self.assertEqual((args.size, args.quality), ("auto", "auto"))

    def test_openai_cloud_colorization_command_uses_shared_key_and_cloud_controls(self) -> None:
        app.APP.settings["references"].update({"openai_api_key": "sk-test"})
        app.APP.settings["colour"].update(
            {
                "manifest": "manifests/references/demo.csv",
                "method": "openai",
                "openai_image_model": "gpt-image-2",
                "openai_previous_frames": "3",
                "openai_image_size": "auto",
                "openai_image_quality": "high",
                "openai_prompt": "Enhance colour without changing geometry.",
            }
        )

        command = app.APP.command_for("colour")

        self.assertEqual(command[command.index("--method") + 1], "openai")
        self.assertEqual(command[command.index("--openai-previous-frames") + 1], "3")
        self.assertEqual(command[command.index("--openai-model") + 1], "gpt-image-2")
        self.assertIn("--api-key", command)
        self.assertNotIn("--comfy-url", command)

    def test_cmnet2_colorization_command_uses_local_runtime_without_comfy_server(self) -> None:
        app.APP.settings["colour"].update(
            {
                "manifest": "manifests/references/demo.csv",
                "method": "cmnet2",
                "processing_height": "source",
            }
        )

        command = app.APP.command_for("colour")

        self.assertEqual(command[command.index("--method") + 1], "cmnet2")
        self.assertIn("--cmnet2-dir", command)
        self.assertIn("--comfy-dir", command)  # checkpoint fallback only; no HTTP dependency
        self.assertNotIn("--comfy-url", command)

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
            barred = Image.new("RGB", (16, 9), (32, 32, 32))
            barred.paste((0, 0, 0), (0, 0, 3, 9))  # a black bar down the left edge
            barred.save(guide)
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
        self.assertEqual(defaulted[defaulted.index("--instruction") + 1], outpaint_guides.GUIDE_EDIT_PROMPT)
        self.assertIn("--mask", unmasked)
        self.assertIn("--mask", masked)
        self.assertIn("outpaint_guides", unmasked_output)
        self.assertIn("outpaint_guides", masked_output)

    def test_guide_edit_preview_command_routes_to_openai_with_the_shared_key(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "chunks.csv"
            guide = folder / "guide.png"
            Image.new("RGB", (16, 9), (32, 32, 32)).save(guide)
            rows = [
                {
                    "chunk_index": "0",
                    "start_frame": "0",
                    "end_frame": "24",
                    "guide_frames": json.dumps([{"frame_idx": 0, "strength": 0.7, "image": app.rel(guide)}]),
                }
            ]
            app.write_outpaint_chunk_rows(manifest, rows)
            app.APP.settings["references"].update(
                {"openai_api_key": "", "openai_image_model": "gpt-image-2.5-sunburst", "openai_image_quality": "high"}
            )
            app.APP.settings["outpaint"].pop("guide_edit_method", None)
            fake_state = {"manifest": app.rel(manifest), "rows": [{"index": 0}]}
            with mock.patch.object(outpaint_guides, "outpaint_chunks_state", return_value=fake_state):
                self.assertEqual(outpaint_guides.guide_edit_engine(), "qwen")
                with self.assertRaisesRegex(RuntimeError, "OpenAI API key"):
                    app.guide_edit_preview_command(0, 0, "make the coat green", "iVBORw0KGgo=", "", "openai")
                app.APP.settings["references"]["openai_api_key"] = "sk-test"
                command, output = app.guide_edit_preview_command(0, 0, "make the coat green", "iVBORw0KGgo=", "", "openai")
                app.APP.settings["outpaint"]["guide_edit_method"] = "openai"
                remembered, _output = app.guide_edit_preview_command(0, 0, "make the coat green", "iVBORw0KGgo=")
                sidecar = json.loads(app.resolve(output + ".json").read_text(encoding="utf-8"))
            app.APP.settings["outpaint"].pop("guide_edit_method", None)

        self.assertIn("openai_edit_guide_image.py", " ".join(command))
        self.assertNotIn("edit_reference_image.py", " ".join(command))
        self.assertEqual(command[command.index("--api-key") + 1], "sk-test")
        self.assertEqual(command[command.index("--model") + 1], "gpt-image-2.5-sunburst")
        self.assertEqual(command[command.index("--quality") + 1], "high")
        self.assertIn("--mask", command)
        self.assertIn("openai_edit_guide_image.py", " ".join(remembered))
        self.assertEqual(sidecar["engine"], "openai")
        self.assertNotIn("sk-test", json.dumps(sidecar))

    def test_guide_edit_preview_command_routes_to_qwen_image_21(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "chunks.csv"
            guide = folder / "guide.png"
            Image.new("RGB", (16, 9), (32, 32, 32)).save(guide)
            rows = [{"chunk_index": "0", "start_frame": "0", "end_frame": "24",
                     "guide_frames": json.dumps([{"frame_idx": 0, "strength": 0.7, "image": app.rel(guide)}])}]
            app.write_outpaint_chunk_rows(manifest, rows)
            app.APP.settings["outpaint"]["guide_edit_method"] = "qwen21"
            fake_state = {"manifest": app.rel(manifest), "rows": [{"index": 0}]}
            try:
                with mock.patch.object(outpaint_guides, "outpaint_chunks_state", return_value=fake_state):
                    self.assertEqual(outpaint_guides.guide_edit_engine(), "qwen21")
                    command, output = app.guide_edit_preview_command(0, 0, "fill the bars", "iVBORw0KGgo=")
                    sidecar = json.loads(app.resolve(output + ".json").read_text(encoding="utf-8"))
            finally:
                app.APP.settings["outpaint"].pop("guide_edit_method", None)

        joined = " ".join(command)
        self.assertIn("qwen21_edit_guide_image.py", joined)
        self.assertNotIn("edit_reference_image.py", joined)
        self.assertIn("--mask", command)
        self.assertIn("--comfy-url", command)
        self.assertEqual(sidecar["engine"], "qwen21")
        # The 2.1 engine renders in ComfyUI, so the runner starts it like the 2511 engine.
        self.assertIn("qwen21", outpaint_guides.COMFY_GUIDE_EDIT_ENGINES)
        self.assertNotIn("openai", outpaint_guides.COMFY_GUIDE_EDIT_ENGINES)

    def test_qwen21_guide_edit_canvas_and_graph(self) -> None:
        import qwen21_edit_guide_image as qwen21

        self.assertEqual(qwen21.canvas_size(1280, 704), (1280, 704))
        for width, height in ((1866, 800), (3840, 1646), (4096, 1716), (1920, 1080), (300, 170)):
            canvas_w, canvas_h = qwen21.canvas_size(width, height)
            self.assertEqual((canvas_w % 32, canvas_h % 32), (0, 0))
            self.assertLessEqual(canvas_w * canvas_h, qwen21.DEFAULT_MAX_PIXELS * 1.05)
            self.assertLessEqual(canvas_w, max(32, round(width / 32) * 32))  # never enlarged
            self.assertAlmostEqual(canvas_w / canvas_h, width / height, delta=0.06)

        # Turbo (default): Viggle's 6-step distill on its resolution-shifted schedule.
        graph = qwen21.build_prompt("src.png", "mask.png", "fill", seed=7, grow_px=16, prefix="p", canvas=(1024, 1024))
        encode = graph["20"]["inputs"]
        self.assertEqual(graph["20"]["class_type"], "TextEncodeQwenImage21")
        self.assertEqual(encode["images.image_1"], ["5", 0])
        self.assertEqual(encode["resolution"], 0)
        self.assertEqual(graph["10"]["inputs"]["unet_name"], qwen21.QWEN_IMAGE_21_TURBO_DIFFUSION)
        sampler = graph["23"]
        self.assertEqual(sampler["class_type"], "SamplerCustomAdvanced")
        self.assertEqual(sampler["inputs"]["latent_image"], ["22", 0])  # denoised only under the noise mask
        self.assertEqual(graph["26"]["inputs"]["noise_seed"], 7)
        sigmas = [float(value) for value in graph["29"]["inputs"]["sigmas"].split(",")]
        self.assertEqual(len(sigmas), 7)  # six steps, ending at 0
        self.assertEqual((sigmas[0], sigmas[-1]), (1.0, 0.0))
        self.assertTrue(all(a > b for a, b in zip(sigmas, sigmas[1:])))
        # Viggle's shift at 1024x1024: mu = 0.6935 lifts the 0.25 node to about 0.40.
        self.assertAlmostEqual(sigmas[5], 0.4001, places=3)
        self.assertGreater(qwen21.turbo_sigmas(1856, 800)[5], qwen21.turbo_sigmas(512, 512)[5])  # more pixels, more shift
        self.assertEqual(graph[qwen21.SAVE_NODE_ID]["class_type"], "SaveImage")
        # The base model keeps a plain 30-step KSampler.
        base = qwen21.build_prompt("src.png", "mask.png", "fill", seed=7, grow_px=16, prefix="p", canvas=(1024, 1024), quality="base")
        self.assertEqual(base["10"]["inputs"]["unet_name"], qwen21.QWEN_IMAGE_21_DIFFUSION)
        self.assertEqual(base["23"]["class_type"], "KSampler")
        self.assertEqual((base["23"]["inputs"]["cfg"], base["23"]["inputs"]["steps"]), (1.0, qwen21.BASE_STEPS))
        self.assertNotIn("29", base)
        # Every link points at a node in the graph.
        for built in (graph, base):
            for node in built.values():
                for value in node["inputs"].values():
                    if isinstance(value, list):
                        self.assertIn(value[0], built)

    def test_guide_edit_without_mask_or_black_bars_is_refused(self) -> None:
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "chunks.csv"
            filled = folder / "filled.png"
            barred = folder / "barred.png"
            black = folder / "black.png"
            picture = np.full((90, 160, 3), 40, np.uint8)
            picture[0, 0] = (3, 3, 3)  # one dark corner pixel is not a bar
            Image.fromarray(picture).save(filled)
            picture[:, :20] = 0
            Image.fromarray(picture).save(barred)
            Image.new("RGB", (160, 90)).save(black)
            self.assertFalse(guide_frame_utils.edge_region_found(filled))
            self.assertTrue(guide_frame_utils.edge_region_found(barred))
            self.assertTrue(guide_frame_utils.edge_region_found(black))

            rows = [{"chunk_index": "0", "start_frame": "0", "end_frame": "24",
                     "guide_frames": json.dumps([{"frame_idx": 0, "strength": 0.7, "image": app.rel(filled)}])}]
            app.write_outpaint_chunk_rows(manifest, rows)
            app.APP.settings["references"]["openai_api_key"] = "sk-test"
            fake_state = {"manifest": app.rel(manifest), "rows": [{"index": 0}]}
            with mock.patch.object(outpaint_guides, "outpaint_chunks_state", return_value=fake_state):
                for engine in ("openai", "qwen"):
                    with self.assertRaisesRegex(outpaint_guides.NoEditRegionError, "No mask has been drawn"):
                        app.guide_edit_preview_command(0, 0, "", "", "", engine)
                # A painted mask is always enough.
                command, _output = app.guide_edit_preview_command(0, 0, "", "iVBORw0KGgo=", "", "openai")
                self.assertIn("openai_edit_guide_image.py", " ".join(command))
                # The runner lets this error reach the HTTP layer, before any ComfyUI start.
                with mock.patch.object(server, "ensure_comfy_available_for_stage") as comfy:
                    with self.assertRaises(outpaint_guides.NoEditRegionError):
                        app.APP.run_guide_edit_preview(0, 0, "", "", "", "qwen")
                    comfy.assert_not_called()

    def test_openai_guide_canvas_keeps_the_source_aspect_within_api_limits(self) -> None:
        import openai_edit_guide_image as guide_edit

        self.assertEqual(guide_edit.api_canvas_size(1280, 704, "gpt-image-2.5-sunburst"), (1280, 704))
        for width, height in ((1920, 1080), (864, 480), (4096, 1716), (640, 352), (3000, 600)):
            canvas_w, canvas_h = guide_edit.api_canvas_size(width, height, "gpt-image-2")
            self.assertEqual((canvas_w % 16, canvas_h % 16), (0, 0))
            self.assertLessEqual(max(canvas_w, canvas_h), 3840)
            self.assertLessEqual(canvas_w * canvas_h, 8_294_400)
            self.assertGreaterEqual(canvas_w * canvas_h, 655_360)
            self.assertLessEqual(max(canvas_w / canvas_h, canvas_h / canvas_w), 3.0)
            x, y, fit_w, fit_h = guide_edit.fit_box(width, height, canvas_w, canvas_h)
            self.assertAlmostEqual(fit_w / fit_h, width / height, delta=0.02)
            self.assertLessEqual(x + fit_w, canvas_w)
            self.assertLessEqual(y + fit_h, canvas_h)
        # Older models only take the presets: pick the nearest aspect and pad.
        self.assertEqual(guide_edit.api_canvas_size(1280, 544, "gpt-image-1"), (1536, 1024))
        self.assertEqual(guide_edit.api_canvas_size(704, 1280, "gpt-image-1.5"), (1024, 1536))

    def test_openai_guide_edit_changes_only_the_masked_area_at_source_size(self) -> None:
        import io
        import base64
        import numpy as np
        import openai_edit_guide_image as guide_edit
        from PIL import Image

        sent: dict = {}
        rng = np.random.default_rng(7)
        # Mid-grey texture, so a brightness shift has room in both directions.
        pixels = rng.integers(70, 150, (300, 500, 3), dtype=np.uint8)

        def fake_post(body, token, timeout, max_retries):
            sent["body"] = body
            sent["token"] = token
            # GPT Image repaints everything: the left part of the frame comes back much brighter
            # (a local shift, as on real footage), and the masked area turns red.
            match = body.split(b'name="size"\r\n\r\n', 1)[1].split(b"\r\n", 1)[0].decode()
            canvas_w, canvas_h = (int(part) for part in match.split("x"))
            result = np.asarray(Image.fromarray(pixels).resize((canvas_w, canvas_h), Image.LANCZOS)).astype(np.float32)
            result[:, : canvas_w // 3] += 45
            sx, sy = canvas_w / 500, canvas_h / 300
            result[int(100 * sy):int(200 * sy), int(80 * sx):int(160 * sx)] = (220, 40, 40)
            buffer = io.BytesIO()
            Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), "RGB").save(buffer, format="PNG")
            return {"data": [{"b64_json": base64.b64encode(buffer.getvalue()).decode()}]}

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "guide.png"
            mask = folder / "mask.png"
            output = folder / "edit.png"
            Image.fromarray(pixels, "RGB").save(source)
            # The editor sends its mask at display size; white = edit, left part only.
            mask_pixels = np.zeros((150, 250), np.uint8)
            mask_pixels[30:120, 20:100] = 255
            Image.fromarray(mask_pixels, "L").save(mask)
            args = guide_edit.build_parser().parse_args([
                "--source-image", str(source), "--mask", str(mask), "--output", str(output),
                "--instruction", "make the coat red", "--api-key", "sk-test", "--model", "gpt-image-2",
            ])
            with mock.patch.object(guide_edit, "post_image_edit", side_effect=fake_post):
                guide_edit.edit_guide(args)
            result = np.asarray(Image.open(output).convert("RGB"))
            raw_exists = output.with_name("edit_raw.png").is_file()

        self.assertEqual(sent["token"], "sk-test")
        self.assertIn(b'name="mask"', sent["body"])
        self.assertIn(b"make the coat red", sent["body"])
        self.assertTrue(raw_exists)
        self.assertEqual(result.shape, pixels.shape)
        full_mask = np.asarray(Image.fromarray(mask_pixels).resize((500, 300), Image.NEAREST)) > 0
        np.testing.assert_array_equal(result[~full_mask], pixels[~full_mask])
        inner = result[110:190, 90:150].astype(int)
        self.assertGreater(inner[..., 0].mean() - inner[..., 2].mean(), 40)
        # The local brightening is undone where the edit meets the untouched frame: the masked
        # area's unchanged (non-red) margin is as bright as the frame just outside it.
        margin = result[64:72, 50:150].astype(float).mean()
        outside = result[50:58, 50:150].astype(float).mean()
        self.assertLess(abs(margin - outside), 12)

    def test_openai_guide_tone_match_falls_back_inside_wide_masks(self) -> None:
        import numpy as np
        import openai_edit_guide_image as guide_edit

        rng = np.random.default_rng(3)
        source = rng.integers(60, 120, (200, 400, 3), dtype=np.uint8)
        result = np.clip(source.astype(int) + 30, 0, 255).astype(np.uint8)
        keep = np.ones((200, 400), bool)
        keep[:, :300] = False  # a mask far wider than the finest window
        matched = guide_edit.match_tone(result, source, keep, 6.0)
        self.assertTrue(np.isfinite(matched).all())
        self.assertLess(abs(matched[:, :300].astype(float).mean() - source[:, :300].astype(float).mean()), 3)
        unmatched = np.ones((200, 400), bool)
        unmatched[:] = False
        np.testing.assert_array_equal(guide_edit.match_tone(result, source, unmatched, 6.0), result)

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
            self.assertTrue(output.endswith(".mkv"))

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

    def test_recomposition_command_passes_reference_luminance_controls(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "colorize": "true"})
        app.APP.settings["recomp"].update(
            {
                "source": "input/example.mp4",
                "colorized_video": "intermediate/outpainted_colorized/example_color.mp4",
                "manifest": "manifests/references/example_shots.csv",
                "reference_luminance_match": "true",
                "reference_luminance_strength": "76",
            }
        )

        command = app.APP.command_for("recomp")

        self.assertEqual(command[command.index("--manifest") + 1], "manifests/references/example_shots.csv")
        self.assertIn("--reference-luminance-match", command)
        self.assertEqual(command[command.index("--reference-luminance-strength") + 1], "76")

    def test_no_overview_steps_selected_leaves_only_output_tab(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "false", "upscale": "false", "section_start": "0", "section_end": ""})

        self.assertEqual(app.APP.active_stages(), ())
        self.assertEqual(app.APP.phase_progress()["stages"], [])

    def test_optional_phase_combinations_have_expected_stage_order(self) -> None:
        cases = [
            (False, False, False, []),
            (True, False, False, ["outpaint", "recomp"]),
            (False, True, False, ["shots", "references", "colour", "recomp"]),
            (False, False, True, ["shots", "upscale"]),
            (True, True, False, ["outpaint", "shots", "references", "colour", "recomp"]),
            (True, False, True, ["outpaint", "shots", "recomp", "upscale"]),
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

        self.assertEqual([stage.key for stage in app.APP.active_stages()], ["outpaint", "shots", "recomp", "upscale"])
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

    def test_soundtrack_output_keeps_a_prores_recomposition_in_mov(self) -> None:
        values = {"create_music": "true", "create_sfx": "true"}
        self.assertTrue(app.soundtrack_output_for("output/reassembled/film_recomp_abc.mov", values).endswith(".mov"))
        self.assertTrue(app.soundtrack_output_for("output/reassembled/film_recomp_abc.mp4", values).endswith(".mp4"))

    def test_outpaint_raw_chunk_prefers_mkv_and_keeps_finished_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mkv = Path(tmp) / "raw_0000_000000_000100.mkv"
            self.assertEqual(app.outpaint_raw_chunk(mkv), mkv)
            mkv.with_suffix(".mp4").write_bytes(b"x")
            self.assertEqual(app.outpaint_raw_chunk(mkv), mkv.with_suffix(".mp4"))

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

        self.assertEqual(stage_keys, ["shots", "upscale"])
        self.assertEqual(app.APP.settings["upscale"]["input_video"], "input/example.mp4")
        self.assertEqual(command[command.index("--method") + 1], "flashvsr")
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
        self.assertEqual(command[command.index("--method") + 1], "flashvsr")

    def test_upscale_ignores_stale_backend_method_setting(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
        app.APP.settings["upscale"].update({"method": "realbasicvsr", "target_width": "1920", "target_height": "1080"})

        command = app.APP.command_for("upscale")

        self.assertEqual(command[command.index("--method") + 1], "flashvsr")
        self.assertNotIn("--realbasicvsr-repo", command)
        self.assertIn("--comfy-url", command)
        self.assertEqual(command[command.index("--output") + 1], app.upscale_output_for("input/example.mp4", app.APP.settings["upscale"]))

    def test_upscale_can_select_ltx25_pixel_spatial_backend(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
        app.APP.settings["upscale"].update({
            "method": "ltx25",
            "target_width": "1920",
            "target_height": "1080",
            "ltx25_source_fidelity": "95",
            "ltx25_lora_strength": "1.0",
            "ltx25_guidance_scale": "3.5",
            "ltx25_seed": "77",
            "ltx25_prompt": "faithful upscale",
            "ltx25_negative_prompt": "changed identity",
        })

        command = app.APP.command_for("upscale")

        self.assertEqual(command[command.index("--method") + 1], "ltx25")
        self.assertEqual(command[command.index("--ltx25-source-fidelity") + 1], "0.95")
        self.assertEqual(command[command.index("--ltx25-lora-strength") + 1], "1.0")
        self.assertEqual(command[command.index("--ltx25-guidance-scale") + 1], "3.5")
        self.assertEqual(command[command.index("--ltx25-seed") + 1], "77")
        self.assertEqual(command[command.index("--ltx25-prompt") + 1], "faithful upscale")
        ltx_output = app.upscale_output_for("input/example.mp4", app.APP.settings["upscale"])
        flash_values = dict(app.APP.settings["upscale"], method="flashvsr")
        self.assertNotEqual(ltx_output, app.upscale_output_for("input/example.mp4", flash_values))

    def test_upscale_can_select_ltx25_cq_enhancer_with_a_finishing_backend(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
        app.APP.settings["upscale"].update({
            "method": "ltx25cq",
            "target_width": "3840",
            "target_height": "2160",
            "cq_finish": "seedvr2",
            "cq_base": "dev",
            "cq_short_edge": "720",
            "cq_frame_rate": "retime",
            "cq_colour": "model",
            "cq_guide_strength": "90",
            "cq_lora_strength": "0.8",
            "cq_chunk_seconds": "4",
            "cq_continuity": "60",
            "cq_seed": "9",
            "cq_prompt": "",
        })

        command = app.APP.command_for("upscale")

        self.assertEqual(command[command.index("--method") + 1], "ltx25cq")
        self.assertEqual(command[command.index("--cq-finish") + 1], "seedvr2")
        self.assertEqual(command[command.index("--cq-base") + 1], "dev")
        self.assertEqual(command[command.index("--cq-frame-rate") + 1], "retime")
        self.assertEqual(command[command.index("--cq-colour") + 1], "model")
        self.assertEqual(command[command.index("--cq-guide-strength") + 1], "0.9")
        self.assertEqual(command[command.index("--cq-lora-strength") + 1], "0.8")
        self.assertEqual(command[command.index("--cq-chunk-seconds") + 1], "4")
        self.assertEqual(command[command.index("--cq-continuity") + 1], "0.6")
        self.assertEqual(command[command.index("--cq-seed") + 1], "9")
        # The script's parser must accept everything the GUI sends.
        upscale_video.build_parser().parse_args(command[3:])
        values = app.APP.settings["upscale"]
        cq_seedvr2 = app.upscale_output_for("input/example.mp4", values)
        self.assertEqual(command[command.index("--output") + 1], cq_seedvr2)
        self.assertNotEqual(cq_seedvr2, app.upscale_output_for("input/example.mp4", dict(values, cq_finish="flashvsr")))
        self.assertNotEqual(cq_seedvr2, app.upscale_output_for("input/example.mp4", dict(values, method="seedvr2")))
        self.assertEqual(app.upscale_backend(values), "seedvr2")
        self.assertEqual(app.upscale_backend(dict(values, cq_finish="bogus")), "flashvsr")

    def test_non_cq_upscale_commands_leave_out_cq_flags(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
        app.APP.settings["upscale"].update({"method": "flashvsr", "cq_finish": "seedvr2"})

        command = app.APP.command_for("upscale")

        self.assertNotIn("--cq-finish", command)
        self.assertEqual(app.upscale_identity_model(app.APP.settings["upscale"])[0], "flashvsr")

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
            flashvsr_vae_tile_multiplier=1,
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
        self.assertEqual(prompt["2"]["inputs"]["force_offload"], False)
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

    def test_upscale_can_select_seedvr2_backend(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
        app.APP.settings["upscale"].update({
            "method": "seedvr2",
            "target_width": "1920",
            "target_height": "1080",
            "seedvr2_model": "seedvr2_ema_7b-Q4_K_M.gguf",
            "seedvr2_batch_size": "9",
            "seedvr2_color_correction": "adain",
            "seedvr2_input_noise_scale": "0.05",
            "seedvr2_latent_noise_scale": "0.1",
            "seedvr2_tiled_vae": "true",
            "seedvr2_vae_tile_size": "768",
            "seedvr2_vae_tile_overlap": "96",
            "seedvr2_preserve_vram": "true",
            "seedvr2_cache_model": "true",
            "seedvr2_blocks_to_swap": "24",
            "seedvr2_offload_io_components": "true",
            "seedvr2_seed": "321",
        })

        command = app.APP.command_for("upscale")

        self.assertEqual(command[command.index("--method") + 1], "seedvr2")
        self.assertEqual(command[command.index("--seedvr2-model") + 1], "seedvr2_ema_7b-Q4_K_M.gguf")
        self.assertEqual(command[command.index("--seedvr2-batch-size") + 1], "9")
        self.assertEqual(command[command.index("--seedvr2-color-correction") + 1], "adain")
        self.assertEqual(command[command.index("--seedvr2-input-noise-scale") + 1], "0.05")
        self.assertEqual(command[command.index("--seedvr2-latent-noise-scale") + 1], "0.1")
        self.assertEqual(command[command.index("--seedvr2-vae-tile-size") + 1], "768")
        self.assertEqual(command[command.index("--seedvr2-vae-tile-overlap") + 1], "96")
        self.assertEqual(command[command.index("--seedvr2-blocks-to-swap") + 1], "24")
        self.assertEqual(command[command.index("--seedvr2-seed") + 1], "321")
        self.assertIn("--seedvr2-tiled-vae", command)
        self.assertIn("--seedvr2-preserve-vram", command)
        self.assertIn("--seedvr2-cache-model", command)
        self.assertIn("--seedvr2-offload-io-components", command)
        seed_output = app.upscale_output_for("input/example.mp4", app.APP.settings["upscale"])
        flash_values = dict(app.APP.settings["upscale"], method="flashvsr")
        self.assertNotEqual(seed_output, app.upscale_output_for("input/example.mp4", flash_values))

    def test_seedvr2_prompt_uses_memory_controls_and_video_helper_nodes(self) -> None:
        args = upscale_video.build_parser().parse_args([
            "--input", "input/example.mp4",
            "--method", "seedvr2",
            "--seedvr2-model", "seedvr2_ema_3b-Q4_K_M.gguf",
            "--seedvr2-batch-size", "9",
            "--seedvr2-color-correction", "adain",
            "--seedvr2-input-noise-scale", "0.05",
            "--seedvr2-latent-noise-scale", "0.1",
            "--seedvr2-vae-tile-size", "768",
            "--seedvr2-vae-tile-overlap", "96",
            "--seedvr2-blocks-to-swap", "24",
            "--seedvr2-offload-io-components",
            "--seedvr2-seed", "321",
        ])
        info = {
            "SeedVR2BlockSwap": {"input": {"required": {
                "blocks_to_swap": ("INT", {"default": 16}),
                "offload_io_components": ("BOOLEAN", {"default": False}),
            }}},
            "SeedVR2ExtraArgs": {"input": {"required": {
                "tiled_vae": ("BOOLEAN", {"default": False}),
                "vae_tile_size": ("INT", {"default": 512}),
                "vae_tile_overlap": ("INT", {"default": 64}),
                "preserve_vram": ("BOOLEAN", {"default": False}),
                "cache_model": ("BOOLEAN", {"default": False}),
                "enable_debug": ("BOOLEAN", {"default": False}),
                "device": (["cuda:0"], {"default": "cuda:0"}),
            }}},
            "SeedVR2": {"input": {
                "required": {
                    "images": ("IMAGE",),
                    "model": (["seedvr2_ema_3b-Q4_K_M.gguf"],),
                    "seed": ("INT", {"default": 100}),
                    "new_resolution": ("INT", {"default": 1072}),
                    "batch_size": ("INT", {"default": 5}),
                    "color_correction": (["wavelet", "adain", "none"], {"default": "wavelet"}),
                    "input_noise_scale": ("FLOAT", {"default": 0.0}),
                    "latent_noise_scale": ("FLOAT", {"default": 0.0}),
                },
                "optional": {
                    "block_swap_config": ("block_swap_config",),
                    "extra_args": ("extra_args",),
                },
            }},
        }

        prompt = upscale_video.seedvr2_prompt("example.mp4", 24.0, 1920, 1080, args, "arp_upscale/example", info)

        self.assertEqual(prompt["1"]["class_type"], "VHS_LoadVideo")
        self.assertEqual(prompt["2"]["inputs"]["blocks_to_swap"], 24)
        self.assertTrue(prompt["2"]["inputs"]["offload_io_components"])
        self.assertTrue(prompt["3"]["inputs"]["tiled_vae"])
        self.assertTrue(prompt["3"]["inputs"]["preserve_vram"])
        self.assertEqual(prompt["3"]["inputs"]["vae_tile_size"], 768)
        self.assertEqual(prompt["3"]["inputs"]["vae_tile_overlap"], 96)
        self.assertEqual(prompt["3"]["inputs"]["device"], "cuda:0")
        self.assertEqual(prompt["4"]["class_type"], "SeedVR2")
        self.assertEqual(prompt["4"]["inputs"]["images"], ["1", 0])
        self.assertEqual(prompt["4"]["inputs"]["block_swap_config"], ["2", 0])
        self.assertEqual(prompt["4"]["inputs"]["extra_args"], ["3", 0])
        self.assertEqual(prompt["4"]["inputs"]["model"], "seedvr2_ema_3b-Q4_K_M.gguf")
        self.assertEqual(prompt["4"]["inputs"]["new_resolution"], 1088)
        self.assertEqual(prompt["4"]["inputs"]["batch_size"], 9)
        self.assertEqual(prompt["4"]["inputs"]["color_correction"], "adain")
        self.assertEqual(prompt["4"]["inputs"]["seed"], 321)
        self.assertEqual(prompt["5"]["class_type"], "VHS_VideoCombine")
        self.assertEqual(prompt["5"]["inputs"]["images"], ["4", 0])
        self.assertNotIn("audio", prompt["5"]["inputs"])

    def test_seedvr2_prompt_supports_v25_modular_nodes(self) -> None:
        args = upscale_video.build_parser().parse_args([
            "--input", "input/example.mp4",
            "--method", "seedvr2",
            "--seedvr2-model", "seedvr2_ema_3b-Q4_K_M.gguf",
            "--seedvr2-batch-size", "9",
            "--seedvr2-color-correction", "adain",
            "--seedvr2-input-noise-scale", "0.05",
            "--seedvr2-latent-noise-scale", "0.1",
            "--seedvr2-vae-tile-size", "768",
            "--seedvr2-vae-tile-overlap", "96",
            "--seedvr2-blocks-to-swap", "24",
            "--seedvr2-offload-io-components",
            "--seedvr2-cache-model",
            "--seedvr2-seed", "321",
        ])
        info = {
            "SeedVR2LoadDiTModel": {"input": {"required": {
                "model": (["seedvr2_ema_3b-Q4_K_M.gguf"],),
                "device": (["cuda:0"], {"default": "cuda:0"}),
            }, "optional": {
                "blocks_to_swap": ("INT", {"default": 0}),
                "swap_io_components": ("BOOLEAN", {"default": False}),
                "offload_device": (["none", "cpu"], {"default": "none"}),
                "cache_model": ("BOOLEAN", {"default": False}),
                "attention_mode": (["sdpa"], {"default": "sdpa"}),
            }}},
            "SeedVR2LoadVAEModel": {"input": {"required": {
                "model": (["ema_vae_fp16.safetensors"], {"default": "ema_vae_fp16.safetensors"}),
                "device": (["cuda:0"], {"default": "cuda:0"}),
            }, "optional": {
                "encode_tiled": ("BOOLEAN", {"default": False}),
                "encode_tile_size": ("INT", {"default": 1024}),
                "encode_tile_overlap": ("INT", {"default": 128}),
                "decode_tiled": ("BOOLEAN", {"default": False}),
                "decode_tile_size": ("INT", {"default": 1024}),
                "decode_tile_overlap": ("INT", {"default": 128}),
                "tile_debug": (["false"], {"default": "false"}),
                "offload_device": (["none", "cpu"], {"default": "none"}),
                "cache_model": ("BOOLEAN", {"default": False}),
            }}},
            "SeedVR2VideoUpscaler": {"input": {"required": {
                "image": ("IMAGE",),
                "dit": ("SEEDVR2_DIT",),
                "vae": ("SEEDVR2_VAE",),
                "seed": ("INT", {"default": 42}),
                "resolution": ("INT", {"default": 1080}),
                "max_resolution": ("INT", {"default": 0}),
                "batch_size": ("INT", {"default": 5}),
                "uniform_batch_size": ("BOOLEAN", {"default": False}),
                "color_correction": (["lab", "wavelet", "adain", "none"], {"default": "lab"}),
            }, "optional": {
                "input_noise_scale": ("FLOAT", {"default": 0.0}),
                "latent_noise_scale": ("FLOAT", {"default": 0.0}),
                "offload_device": (["none", "cpu"], {"default": "cpu"}),
                "enable_debug": ("BOOLEAN", {"default": False}),
            }}},
        }

        prompt = upscale_video.seedvr2_prompt("example.mp4", 24.0, 1920, 1080, args, "arp_upscale/example", info)

        self.assertEqual(prompt["2"]["class_type"], "SeedVR2LoadDiTModel")
        self.assertEqual(prompt["2"]["inputs"]["model"], "seedvr2_ema_3b-Q4_K_M.gguf")
        self.assertEqual(prompt["2"]["inputs"]["blocks_to_swap"], 24)
        self.assertTrue(prompt["2"]["inputs"]["swap_io_components"])
        self.assertEqual(prompt["2"]["inputs"]["offload_device"], "cpu")
        self.assertTrue(prompt["2"]["inputs"]["cache_model"])
        self.assertEqual(prompt["3"]["class_type"], "SeedVR2LoadVAEModel")
        self.assertEqual(prompt["3"]["inputs"]["model"], "ema_vae_fp16.safetensors")
        self.assertTrue(prompt["3"]["inputs"]["encode_tiled"])
        self.assertTrue(prompt["3"]["inputs"]["decode_tiled"])
        self.assertEqual(prompt["3"]["inputs"]["decode_tile_size"], 768)
        self.assertEqual(prompt["3"]["inputs"]["decode_tile_overlap"], 96)
        self.assertEqual(prompt["3"]["inputs"]["offload_device"], "cpu")
        self.assertEqual(prompt["4"]["class_type"], "SeedVR2VideoUpscaler")
        self.assertEqual(prompt["4"]["inputs"]["image"], ["1", 0])
        self.assertEqual(prompt["4"]["inputs"]["dit"], ["2", 0])
        self.assertEqual(prompt["4"]["inputs"]["vae"], ["3", 0])
        self.assertEqual(prompt["4"]["inputs"]["resolution"], 1088)
        self.assertEqual(prompt["4"]["inputs"]["offload_device"], "cpu")
        self.assertEqual(prompt["5"]["inputs"]["images"], ["4", 0])

    def test_ltx25_upscale_prompt_is_video_only_and_uses_x2_ic_lora(self) -> None:
        args = upscale_video.build_parser().parse_args(["--input", "input/example.mp4", "--method", "ltx25"])

        prompt = upscale_video.ltx25_prompt(
            "example.mp4", 24.0, 97, 1920, 1088, args, "arp_upscale/example"
        )

        self.assertEqual(prompt["2"]["inputs"]["unet_name"], upscale_video.LTX25_GGUF_MODEL)
        self.assertEqual(prompt["3"]["inputs"]["lora_name"], upscale_video.LTX25_PIXEL_UPSCALER_LORA)
        self.assertEqual(prompt["3"]["inputs"]["strength_model"], 1.0)
        self.assertEqual(prompt["10"]["inputs"]["strength"], 0.85)
        self.assertEqual(prompt["11"]["inputs"]["cfg"], 1.0)
        self.assertEqual(prompt["11"]["inputs"]["model"], ["3", 0])
        self.assertEqual(prompt["10"]["inputs"]["latent_downscale_factor"], ["3", 1])
        self.assertEqual(prompt["10"]["inputs"]["latent"], ["9", 0])
        self.assertEqual(prompt["15"]["inputs"]["latent_image"], ["10", 2])
        self.assertEqual(prompt["16"]["inputs"]["latent"], ["15", 0])
        self.assertEqual(prompt["17"]["inputs"]["samples"], ["16", 2])
        self.assertNotIn("audio", prompt["18"]["inputs"])
        class_types = {node["class_type"] for node in prompt.values()}
        self.assertNotIn("LTXVConcatAVLatent", class_types)
        self.assertNotIn("LTXVSeparateAVLatent", class_types)

    def test_ltx25_upscale_chunks_keep_every_frame_in_valid_windows(self) -> None:
        ranges = upscale_video.ltx25_chunk_ranges(294, 24.0, 4.04, 8)

        self.assertEqual(ranges, [
            (0, 97, 0, 97, 97),
            (89, 186, 8, 89, 97),
            (178, 275, 8, 89, 97),
            (267, 294, 8, 19, 97),
        ])
        self.assertEqual(sum(item[3] for item in ranges), 294)
        self.assertTrue(all(item[4] % 8 == 1 for item in ranges))

    def test_cq_enhancer_prompt_is_video_only_at_30fps_without_prompt_or_cfg(self) -> None:
        args = upscale_video.build_parser().parse_args(["--input", "input/example.mp4", "--method", "ltx25cq"])

        prompt = upscale_video.ltx_ic_lora_prompt(
            "example.mp4", upscale_video.CQ_FPS, 153, 1280, 704, "arp_upscale/example",
            unet=upscale_video.cq_base_model(args), lora=args.cq_lora, lora_strength=args.cq_lora_strength,
            text_encoder=args.ltx25_text_encoder, vae=args.ltx25_video_vae, prompt=args.cq_prompt,
            negative_prompt="", guide_strength=args.cq_guide_strength, cfg=1.0, seed=args.cq_seed,
        )

        self.assertEqual(prompt["2"]["inputs"]["unet_name"], upscale_video.LTX25_GGUF_MODEL)
        self.assertEqual(prompt["3"]["inputs"]["model"], ["2", 0])
        self.assertEqual(prompt["3"]["inputs"]["lora_name"], upscale_video.LTX25_CQ_ENHANCER_LORA)
        self.assertEqual(prompt["5"]["inputs"]["text"], "")
        self.assertEqual(prompt["7"]["inputs"]["frame_rate"], 30.0)
        self.assertEqual(prompt["9"]["inputs"], {"width": 1280, "height": 704, "length": 153, "batch_size": 1})
        self.assertEqual(prompt["10"]["inputs"]["strength"], 1.0)
        self.assertEqual(prompt["11"]["inputs"]["cfg"], 1.0)
        self.assertNotIn("19", prompt)
        class_types = {node["class_type"] for node in prompt.values()}
        self.assertNotIn("LTXVConcatAVLatent", class_types)

    def test_cq_enhancer_dev_base_stacks_the_distilled_lora_at_half_strength(self) -> None:
        args = upscale_video.build_parser().parse_args(["--input", "input/example.mp4", "--method", "ltx25cq", "--cq-base", "dev"])

        prompt = upscale_video.ltx_ic_lora_prompt(
            "example.mp4", upscale_video.CQ_FPS, 153, 1280, 704, "arp_upscale/example",
            unet=upscale_video.cq_base_model(args), distilled_lora=upscale_video.LTX25_DISTILLED_LORA,
            lora=args.cq_lora, lora_strength=1.0, text_encoder=args.ltx25_text_encoder, vae=args.ltx25_video_vae,
            prompt="", negative_prompt="", guide_strength=1.0, cfg=1.0, seed=1,
        )

        self.assertEqual(prompt["2"]["inputs"]["unet_name"], upscale_video.LTX25_DEV_GGUF_MODEL)
        self.assertEqual(prompt["19"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(prompt["19"]["inputs"]["model"], ["2", 0])
        self.assertEqual(prompt["19"]["inputs"]["lora_name"], upscale_video.LTX25_DISTILLED_LORA)
        self.assertEqual(prompt["19"]["inputs"]["strength_model"], 0.5)
        self.assertEqual(prompt["3"]["inputs"]["model"], ["19", 0])

    def test_cq_chunk_plan_renders_valid_30fps_windows_that_keep_every_frame(self) -> None:
        plan = upscale_video.cq_chunk_plan(294, 24.0, 5.0, 8, "resample")

        # 294 frames over a 120-frame limit: three equal pieces, not two full ones and a short tail.
        self.assertEqual(plan, [(0, 98, 0, 129), (90, 196, 8, 137), (188, 294, 8, 137)])
        self.assertEqual(sum(end - start - trim for start, end, trim, _model in plan), 294)
        self.assertTrue(all(model % 8 == 1 for *_rest, model in plan))
        # Resampling 24 -> 30 fps needs 5 working frames per 4 source frames.
        self.assertTrue(all(model >= (end - start) * 30 / 24 for start, end, _trim, model in plan))
        self.assertEqual(upscale_video.cq_chunk_plan(294, 24.0, 5.0, 8, "retime")[0], (0, 98, 0, 105))
        self.assertEqual(upscale_video.cq_chunk_plan(50, 24.0, 5.0, 8, "resample"), [(0, 50, 0, 65)])
        self.assertEqual(upscale_video.cq_chunk_plan(50, 24.0, 0.0, 8, "resample"), [(0, 50, 0, 65)])

    def test_cq_chunk_plan_restarts_at_every_shot_and_splits_long_shots(self) -> None:
        # Shots: 0-40, 40-60, 60-300 (10 s at 24 fps, over the 8 s limit), plus cuts outside the clip.
        plan = upscale_video.cq_chunk_plan(300, 24.0, 8.0, 8, "retime", [0, 40, 60, 300, 450])

        self.assertEqual([(start, end, trim) for start, end, trim, _model in plan], [
            (0, 40, 0), (40, 60, 0), (60, 180, 0), (172, 300, 8),
        ])
        self.assertEqual(sum(end - start - trim for start, end, trim, _model in plan), 300)
        # No chunk's window (lead-in included) spans a cut.
        self.assertFalse(any(start < cut < end for start, end, _trim, _model in plan for cut in (40, 60)))

    def test_cq_shot_cuts_come_from_the_shot_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "shots.csv"
            manifest.write_text("start_frame,end_frame\n0,40\n40,60\n60,500\n", encoding="utf-8")
            args = upscale_video.build_parser().parse_args(
                ["--input", "input/example.mp4", "--method", "ltx25cq", "--shot-manifest", str(manifest)])
            with mock.patch.object(upscale_video, "video_info", return_value={"frames": 300, "fps": 24.0}):
                self.assertEqual(upscale_video.cq_shot_cuts(args, Path("clip.mp4")), [40, 60])
                # A preview clip shorter than the film only sees the cuts inside it.
                with mock.patch.object(upscale_video, "video_info", return_value={"frames": 50, "fps": 24.0}):
                    self.assertEqual(upscale_video.cq_shot_cuts(args, Path("clip.mp4")), [40])
            args.shot_manifest = ""
            self.assertEqual(upscale_video.cq_shot_cuts(args, Path("clip.mp4")), [])

    def test_cq_frame_rate_never_resamples_footage_faster_than_30fps(self) -> None:
        args = upscale_video.build_parser().parse_args(["--input", "input/example.mp4", "--method", "ltx25cq"])

        self.assertEqual(upscale_video.cq_frame_rate_mode(args, 24.0), "resample")
        self.assertEqual(upscale_video.cq_frame_rate_mode(args, 50.0), "retime")
        args.cq_frame_rate = "retime"
        self.assertEqual(upscale_video.cq_frame_rate_mode(args, 24.0), "retime")

    def test_cq_frame_rate_round_trip_returns_each_source_frame_once(self) -> None:
        try:
            ffmpeg = common.find_ffmpeg("")
        except Exception:
            self.skipTest("FFmpeg is not available")
        import cv2

        def frame_levels(path: Path) -> list[int]:
            capture = cv2.VideoCapture(str(path))
            levels = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                levels.append(int(round(frame.mean())))
            capture.release()
            return levels

        frames = 40
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            for fps in (18.0, 24.0):
                source = folder / f"source_{fps:g}.mkv"
                # Frame i is flat grey 5*i, so every frame stays identifiable through lossy coding.
                raw = b"".join(bytes([5 * index]) * (64 * 48 * 3) for index in range(frames))
                subprocess.run(
                    [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "64x48",
                     "-r", f"{fps:g}", "-i", "-", "-c:v", "ffv1", str(source)],
                    input=raw, check=True,
                )
                for mode in ("resample", "retime"):
                    levels: list[int] = []
                    for index, (start, end, trim, model) in enumerate(upscale_video.cq_chunk_plan(frames, fps, 1.0, 4, mode)):
                        prepared = folder / f"prepared_{fps:g}_{mode}_{index}.mkv"
                        restored = folder / f"restored_{fps:g}_{mode}_{index}.mkv"
                        upscale_video.prepare_cq_chunk(
                            ffmpeg, source, prepared, start, end, model, fps, 64, 48, mode, True,
                            common.file_fingerprint(source),
                        )
                        self.assertEqual(len(frame_levels(prepared)), model)
                        # An identity "render": the prepared 30 fps clip goes straight back.
                        upscale_video.normalize_cq_chunk(ffmpeg, prepared, restored, 64, 48, trim, end - start - trim, fps, mode)
                        levels.extend(frame_levels(restored))
                    expected = [5 * index for index in range(frames)]
                    self.assertEqual(len(levels), frames, (fps, mode))
                    self.assertTrue(all(abs(got - want) <= 1 for got, want in zip(levels, expected)), (fps, mode, levels))

    def test_cq_continuity_frames_line_up_with_the_next_chunks_lead_in(self) -> None:
        try:
            ffmpeg = common.find_ffmpeg("")
        except Exception:
            self.skipTest("FFmpeg is not available")
        import cv2

        def frame_levels(path: Path) -> list[int]:
            capture = cv2.VideoCapture(str(path))
            levels = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                levels.append(int(round(frame.mean())))
            capture.release()
            return levels

        frames = 48
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            for fps in (24.0, 25.0):
                source = folder / f"source_{fps:g}.mkv"
                raw = b"".join(bytes([5 * index]) * (64 * 48 * 3) for index in range(frames))
                subprocess.run(
                    [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "64x48",
                     "-r", f"{fps:g}", "-i", "-", "-c:v", "ffv1", str(source)],
                    input=raw, check=True,
                )
                for mode in ("resample", "retime"):
                    plan = upscale_video.cq_chunk_plan(frames, fps, 1.0, 12, mode)
                    (first_start, first_end, _, _), (start, end, lead, model) = plan[0], plan[1]
                    count = upscale_video.cq_continuity_frames(lead, fps, mode)
                    self.assertEqual(count % 8, 1)
                    # An identity "render" of the first chunk, at the source rate.
                    previous = folder / f"previous_{fps:g}_{mode}.mkv"
                    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(source), "-vf",
                                    f"trim=start_frame={first_start}:end_frame={first_end},setpts=N/({fps:g}*TB)",
                                    "-c:v", "ffv1", str(previous)], check=True)
                    guide = folder / f"guide_{fps:g}_{mode}.mkv"
                    previous_frames = first_end - first_start
                    upscale_video.prepare_cq_working_frames(
                        ffmpeg, previous, guide, previous_frames - lead, previous_frames, count, fps, mode, 64, 48,
                    )
                    prepared = folder / f"prepared_{fps:g}_{mode}.mkv"
                    upscale_video.prepare_cq_chunk(
                        ffmpeg, source, prepared, start, end, model, fps, 64, 48, mode, True, common.file_fingerprint(source),
                    )
                    guide_levels = frame_levels(guide)
                    self.assertEqual(len(guide_levels), count, (fps, mode))
                    self.assertEqual(guide_levels, frame_levels(prepared)[:count], (fps, mode))

    def test_cq_guide_shots_render_first_and_their_frames_are_trimmed_back_off(self) -> None:
        try:
            ffmpeg = common.find_ffmpeg("")
        except Exception:
            self.skipTest("FFmpeg is not available")
        import cv2

        def frame_levels(path: Path) -> list[int]:
            capture = cv2.VideoCapture(str(path))
            levels = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                levels.append(int(round(frame.mean())))
            capture.release()
            return levels

        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mkv"
            raw = b"".join(bytes([4 * index]) * (64 * 48 * 3) for index in range(60))
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "64x48",
                 "-r", "24", "-i", "-", "-c:v", "ffv1", str(source)],
                input=raw, check=True,
            )
            manifest = folder / "shots.csv"
            # Shot 1 is guided by shot 3, so it renders last.
            manifest.write_text("start_frame,end_frame,cq_guide_shots\n0,20,40\n20,40,\n40,60,\n", encoding="utf-8")
            args = upscale_video.build_parser().parse_args([
                "--input", str(source), "--method", "ltx25cq", "--cq-short-edge", "64", "--shot-manifest", str(manifest),
            ])
            calls: list[dict] = []

            def identity_render(_args, chunk_input, partial, _width, _height, frames, locked=None):
                calls.append({
                    "input": frame_levels(chunk_input),
                    "frames": frames,
                    "locked": frame_levels(locked[0]) if locked else [],
                })
                shutil.copy2(chunk_input, partial)
                return partial

            info = common.video_info(source)
            with mock.patch.object(upscale_video, "ROOT", folder), \
                    mock.patch.object(upscale_video, "cq_run", side_effect=identity_render):
                stitched = upscale_video.cq_enhance(args, source, info, common.file_fingerprint(source))
                levels = frame_levels(stitched)

        self.assertEqual(len(calls), 3)
        # Shots 2 and 3 render plainly, then shot 1 after 9 frames of shot 3's render.
        self.assertAlmostEqual(calls[0]["input"][0], 80, delta=1)
        self.assertAlmostEqual(calls[1]["input"][0], 160, delta=1)
        guided = calls[2]
        self.assertEqual(len(guided["locked"]), 9)
        self.assertEqual(guided["frames"] % 8, 1)
        self.assertEqual(guided["input"][:9], guided["locked"])
        self.assertTrue(all(159 <= level < 240 for level in guided["locked"]), guided["locked"])
        self.assertAlmostEqual(guided["input"][9], 0, delta=1)
        # The identity "render" comes back as the source: the guide frames are trimmed off exactly.
        self.assertEqual(len(levels), 60)
        self.assertTrue(all(abs(level - 4 * index) <= 1 for index, level in enumerate(levels)), levels)

    def test_cq_thumbnails_cover_exactly_the_asked_frames(self) -> None:
        try:
            ffmpeg = common.find_ffmpeg("")
        except Exception:
            self.skipTest("FFmpeg is not available")
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp_text:
            source = Path(tmp_text) / "source.mkv"
            frames = []
            for index in range(40):
                # A bright bar that moves across the frame, so every frame looks different.
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                frame[:, index:index + 16] = 255
                frames.append(frame.tobytes())
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "64x48",
                 "-r", "24", "-i", "-", "-c:v", "ffv1", str(source)],
                input=b"".join(frames), check=True,
            )
            first = upscale_video.cq_thumbnails(ffmpeg, source, 30, 31)
            shot = upscale_video.cq_thumbnails(ffmpeg, source, 10, 35)

        self.assertEqual(first.shape[0], 1)
        self.assertEqual(shot.shape[0], 25)
        self.assertEqual(int(np.abs(shot - first[0]).mean(axis=1).argmin()), 20)

    def test_cq_shot_chunk_order_spreads_out_from_the_anchor(self) -> None:
        self.assertEqual(upscale_video.cq_shot_chunk_order(0, 1), [])
        self.assertEqual(upscale_video.cq_shot_chunk_order(1, 3), [(0, None)])
        self.assertEqual(upscale_video.cq_shot_chunk_order(3, 1), [(0, None), (1, "forward"), (2, "forward")])
        self.assertEqual(
            upscale_video.cq_shot_chunk_order(4, 2),
            [(1, None), (2, "forward"), (3, "forward"), (0, "backward")],
        )
        self.assertEqual(upscale_video.cq_shot_chunk_order(3, 9), [(2, None), (1, "backward"), (0, "backward")])

    def test_cq_chunks_before_the_anchor_render_backwards_from_it(self) -> None:
        try:
            ffmpeg = common.find_ffmpeg("")
        except Exception:
            self.skipTest("FFmpeg is not available")
        import cv2

        def frame_levels(path: Path) -> list[int]:
            capture = cv2.VideoCapture(str(path))
            levels = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                levels.append(int(round(frame.mean())))
            capture.release()
            return levels

        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mkv"
            raw = b"".join(bytes([4 * index]) * (64 * 48 * 3) for index in range(60))
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "64x48",
                 "-r", "24", "-i", "-", "-c:v", "ffv1", str(source)],
                input=raw, check=True,
            )
            manifest = folder / "shots.csv"
            # One 60-frame shot in three 1-second chunks, taking its look from the last.
            manifest.write_text("start_frame,end_frame,cq_anchor_chunk\n0,60,3\n", encoding="utf-8")
            args = upscale_video.build_parser().parse_args([
                "--input", str(source), "--method", "ltx25cq", "--cq-short-edge", "64", "--cq-chunk-seconds", "1",
                "--shot-manifest", str(manifest),
            ])
            calls: list[dict] = []

            def identity_render(_args, chunk_input, partial, _width, _height, _frames, locked=None):
                calls.append({"input": frame_levels(chunk_input), "locked": frame_levels(locked[0]) if locked else []})
                shutil.copy2(chunk_input, partial)
                return partial

            info = common.video_info(source)
            with mock.patch.object(upscale_video, "ROOT", folder), \
                    mock.patch.object(upscale_video, "cq_run", side_effect=identity_render):
                levels = frame_levels(upscale_video.cq_enhance(args, source, info, common.file_fingerprint(source)))

        self.assertEqual(len(calls), 3)
        # The last chunk renders first, forwards; then the middle and first chunks, backwards,
        # each starting from the frames it shares with the chunk after it.
        self.assertLess(calls[0]["input"][0], calls[0]["input"][5])
        self.assertEqual(calls[0]["locked"], [])
        for call in calls[1:]:
            self.assertGreater(call["input"][0], call["input"][5])
            self.assertEqual(len(call["locked"]), 9)
            self.assertTrue(all(abs(a - b) <= 1 for a, b in zip(call["locked"], call["input"])), call)
        self.assertGreater(calls[1]["input"][0], calls[2]["input"][0])
        # Rendered backwards and turned round again, every frame comes back in order.
        self.assertEqual(len(levels), 60)
        self.assertTrue(all(abs(level - 4 * index) <= 1 for index, level in enumerate(levels)), levels)

    def test_cq_guide_change_re_renders_only_the_guided_shot(self) -> None:
        try:
            ffmpeg = common.find_ffmpeg("")
        except Exception:
            self.skipTest("FFmpeg is not available")

        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mkv"
            raw = b"".join(bytes([4 * index]) * (64 * 48 * 3) for index in range(60))
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "64x48",
                 "-r", "24", "-i", "-", "-c:v", "ffv1", str(source)],
                input=raw, check=True,
            )
            manifest = folder / "shots.csv"
            manifest.write_text("start_frame,end_frame,cq_guide_shots\n0,20,\n20,40,\n40,60,\n", encoding="utf-8")
            args = upscale_video.build_parser().parse_args([
                "--input", str(source), "--method", "ltx25cq", "--cq-short-edge", "64", "--shot-manifest", str(manifest),
            ])
            rendered: list[int] = []

            def identity_render(_args, chunk_input, partial, _width, _height, _frames, locked=None):
                rendered.append(len(rendered))
                shutil.copy2(chunk_input, partial)
                return partial

            info = common.video_info(source)
            with mock.patch.object(upscale_video, "ROOT", folder), \
                    mock.patch.object(upscale_video, "cq_run", side_effect=identity_render):
                first = upscale_video.cq_enhance(args, source, info, common.file_fingerprint(source))
                before = upscale_video.cq_range_fingerprint(first, common.file_fingerprint(first), {})
                first_ranges = [before(0, 20), before(20, 40), before(40, 60)]
                renders_first = len(rendered)
                # Shot 3 now follows shot 1: only shot 3 is rendered again.
                manifest.write_text("start_frame,end_frame,cq_guide_shots\n0,20,\n20,40,\n40,60,0\n", encoding="utf-8")
                second = upscale_video.cq_enhance(args, source, info, common.file_fingerprint(source))
                after = upscale_video.cq_range_fingerprint(second, common.file_fingerprint(second), {})
                second_ranges = [after(0, 20), after(20, 40), after(40, 60)]
                # A render without a chunk map falls back to keying on the whole file.
                no_map = upscale_video.cq_range_fingerprint(source, common.file_fingerprint(source), {})

        self.assertEqual(renders_first, 3)
        self.assertEqual(len(rendered), 4)
        # The finisher keys its splits on these, so it redoes only shot 3's frames.
        self.assertEqual(first_ranges[:2], second_ranges[:2])
        self.assertNotEqual(first_ranges[2], second_ranges[2])
        self.assertIsNone(no_map)

    def test_upscale_only_shot_list_columns_do_not_stale_other_stages(self) -> None:
        sys.path.insert(0, str(app.ROOT / "scripts"))
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            output = folder / "out.mp4"
            output.write_bytes(b"video")
            row = {"start_frame": "0", "end_frame": "20", "prompt": "a", "cq_guide_shots": "", "upscale_strength": ""}
            common.write_signature(output, {"manifest_fingerprint": {"sha256": "old"}, "shot_input": {"row": dict(row)}})
            edited = dict(row, cq_guide_shots="40", upscale_strength="60", cq_anchor_chunk="2")
            # Colorize signs each shot's row: upscale-only columns and the whole-file hash are ignored.
            self.assertTrue(common.signature_matches(output, {"manifest_fingerprint": {"sha256": "new"}, "shot_input": {"row": edited}}))
            self.assertFalse(common.signature_matches(output, {"manifest_fingerprint": {"sha256": "new"}, "shot_input": {"row": dict(edited, prompt="b")}}))

            manifest = folder / "shots.csv"
            command = ["python", "colorize_video.py", str(manifest)]

            def stamps(text: str) -> tuple[str, str]:
                manifest.write_text(text, encoding="utf-8")
                return (
                    stage_stamps.compute_stamp(command, [], "label", stage_stamps.stage_ignored_columns("colour")),
                    stage_stamps.compute_stamp(command, [], "label", stage_stamps.stage_ignored_columns("upscale")),
                )

            plain = stamps("start_frame,end_frame,prompt,cq_guide_shots\n0,20,a,\n")
            guided = stamps("start_frame,end_frame,prompt,cq_guide_shots\n0,20,a,40\n")
            prompted = stamps("start_frame,end_frame,prompt,cq_guide_shots\n0,20,b,40\n")
        self.assertEqual(plain[0], guided[0])
        self.assertNotEqual(plain[1], guided[1])
        self.assertNotEqual(guided[0], prompted[0])

    def test_replace_unless_identical_keeps_a_re_encode_of_the_same_frames(self) -> None:
        try:
            ffmpeg = common.find_ffmpeg("")
        except Exception:
            self.skipTest("FFmpeg is not available")
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mkv"
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=24:duration=1",
                 "-c:v", "ffv1", str(source)],
                check=True,
            )
            target, partial, other = folder / "split.mkv", folder / "split.partial.mkv", folder / "other.partial.mkv"
            for path in (target, partial):
                subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(source), "-c:v", "ffv1", "-f", "matroska", str(path)], check=True)
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(source), "-vf", "negate", "-c:v", "ffv1", str(other)], check=True)
            original = common.file_fingerprint(target)["sha256"]
            self.assertNotEqual(original, common.file_fingerprint(partial)["sha256"])

            common.replace_unless_identical(partial, target)
            kept = common.file_fingerprint(target)["sha256"]
            common.replace_unless_identical(other, target)
            replaced = common.file_fingerprint(target)["sha256"]

        self.assertEqual(kept, original)
        self.assertNotEqual(replaced, original)

    def test_cq_render_order_puts_guide_shots_first_and_breaks_loops(self) -> None:
        self.assertEqual(upscale_video.cq_render_order(3, {}), ([0, 1, 2], {0: [], 1: [], 2: []}))
        order, usable = upscale_video.cq_render_order(4, {0: [2], 1: [3, 0]})
        self.assertEqual(order, [2, 0, 3, 1])
        self.assertEqual(usable[0], [2])
        self.assertEqual(usable[1], [3, 0])
        # Shots 0 and 1 guide each other: the earlier one renders first, without its guide.
        order, usable = upscale_video.cq_render_order(2, {0: [1], 1: [0]})
        self.assertEqual(order, [0, 1])
        self.assertEqual(usable, {0: [], 1: [0]})

    def test_cq_guide_window_ends_on_the_best_match_inside_one_chunk(self) -> None:
        distances = [5.0] * 30
        distances[12] = 1.0
        distances[2] = 0.5  # too early for a whole window
        self.assertEqual(upscale_video.cq_guide_window(distances, 100, [(100, 130)], 8), (105, 113))
        # A best match straddling two chunks is out; the window must sit inside one render.
        self.assertEqual(upscale_video.cq_guide_window(distances, 100, [(100, 110), (110, 130)], 8), (100, 108))
        self.assertIsNone(upscale_video.cq_guide_window([1.0] * 5, 0, [(0, 5)], 8))

    def test_cq_guide_shots_come_from_the_shot_list_as_shot_starts(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            manifest = Path(tmp_text) / "shots.csv"
            manifest.write_text(
                "start_frame,end_frame,cq_guide_shots\n0,20,45;25\n20,40,\n40,60,5;0;20;40;55\n60,96,999\n",
                encoding="utf-8",
            )
            args = upscale_video.build_parser().parse_args(["--input", "input/example.mp4", "--method", "ltx25cq", "--shot-manifest", str(manifest)])
            source = app.ROOT / "input" / "example.mp4"
            with mock.patch.object(upscale_video, "video_info", return_value={"width": 64, "height": 48, "fps": 24.0, "frames": 96}):
                guides = upscale_video.cq_guide_shots(args, source)
        # Frames resolve to the shots holding them; a shot never guides itself, repeats collapse,
        # and a frame past the end lands on the last shot (which then guides nothing new).
        self.assertEqual(guides, {0: [40, 20], 40: [0, 20]})

    def test_cq_continuity_frame_counts_fit_the_lead_in(self) -> None:
        self.assertEqual(upscale_video.cq_continuity_frames(0, 24.0, "resample"), 0)
        self.assertEqual(upscale_video.cq_continuity_frames(8, 24.0, "resample"), 9)
        self.assertEqual(upscale_video.cq_continuity_frames(8, 25.0, "resample"), 9)
        self.assertEqual(upscale_video.cq_continuity_frames(8, 24.0, "retime"), 1)
        self.assertEqual(upscale_video.cq_continuity_frames(24, 24.0, "resample"), 25)
        self.assertEqual(upscale_video.cq_continuity_frames(17, 24.0, "retime"), 17)

    def test_cq_continuity_writes_the_start_frames_over_the_first_latents(self) -> None:
        args = upscale_video.build_parser().parse_args(["--input", "input/example.mp4", "--method", "ltx25cq"])
        common_inputs = dict(
            unet=upscale_video.cq_base_model(args), lora=args.cq_lora, lora_strength=args.cq_lora_strength,
            text_encoder=args.ltx25_text_encoder, vae=args.ltx25_video_vae, prompt="", negative_prompt="",
            guide_strength=1.0, cfg=1.0, seed=42,
        )
        plain = upscale_video.ltx_ic_lora_prompt("example.mp4", 30.0, 153, 1280, 704, "p", **common_inputs)
        continued = upscale_video.ltx_ic_lora_prompt(
            "example.mp4", 30.0, 153, 1280, 704, "p", **common_inputs,
            start_video="previous.mkv", start_frames=9, start_strength=0.8,
        )

        self.assertNotIn("21", plain)
        self.assertEqual(plain["10"]["inputs"]["latent"], ["9", 0])
        self.assertEqual(continued["20"]["class_type"], "VHS_LoadVideo")
        self.assertEqual(continued["20"]["inputs"]["video"], "previous.mkv")
        self.assertEqual(continued["20"]["inputs"]["frame_load_cap"], 9)
        self.assertEqual(continued["21"]["class_type"], "LTXVImgToVideoInplace")
        self.assertEqual(continued["21"]["inputs"]["latent"], ["9", 0])
        self.assertEqual(continued["21"]["inputs"]["image"], ["20", 0])
        self.assertEqual(continued["21"]["inputs"]["strength"], 0.8)
        self.assertEqual(continued["10"]["inputs"]["latent"], ["21", 0])

    def test_cq_luma_only_shots_keep_source_colour_at_delivery(self) -> None:
        try:
            ffmpeg = common.find_ffmpeg("")
        except Exception:
            self.skipTest("FFmpeg is not available")
        import cv2

        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            source = folder / "grey_source.mp4"
            upscaled = folder / "colour_upscale.mkv"
            output = folder / "delivery.mp4"
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0x808080:s=64x48:r=24",
                            "-frames:v", "48", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)], check=True)
            # A brighter, strongly coloured "CQ + finisher" upscale at twice the size.
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0xE08040:s=128x96:r=24",
                            "-frames:v", "48", "-c:v", "ffv1", str(upscaled)], check=True)
            shots = [
                {"start": 0, "end": 24, "strength": 1.0, "fade": False, "crossfade_seconds": "", "luma_only": True},
                {"start": 24, "end": 48, "strength": 1.0, "fade": False, "crossfade_seconds": "", "luma_only": False},
            ]

            upscale_video.blend_upscale_delivery(ffmpeg, source, upscaled, output, 128, 96, 24.0, shots, 1.0)

            capture = cv2.VideoCapture(str(output))
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            capture.release()
        self.assertEqual(len(frames), 48)
        self.assertEqual(frames[0].shape[:2], (96, 128))
        blue, green, red = (float(frames[12][..., channel].mean()) for channel in range(3))
        # The luma-only shot keeps the grey source's colour...
        self.assertLess(max(blue, green, red) - min(blue, green, red), 6)
        # ...but its brightness comes from the upscale (0xE08040 is brighter than mid grey).
        self.assertGreater(green, 135)
        blue, green, red = (float(frames[36][..., channel].mean()) for channel in range(3))
        # The next shot keeps the upscale's own colour.
        self.assertGreater(red - blue, 100)

    def test_cq_luma_only_is_a_shot_list_column_inheriting_the_cq_colour_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "shots.csv"
            manifest.write_text("start_frame,end_frame,cq_luma_only\n0,10,\n10,20,true\n20,30,false\n", encoding="utf-8")
            shots = upscale_video.read_upscale_shots(manifest, 30, 24.0, 1.0)
            self.assertEqual([shot["luma_only"] for shot in shots], [False, True, False])
            shots = upscale_video.read_upscale_shots(manifest, 30, 24.0, 1.0, default_luma_only=True)
            self.assertEqual([shot["luma_only"] for shot in shots], [True, True, False])
        args = upscale_video.build_parser().parse_args(["--input", "input/example.mp4", "--method", "ltx25cq", "--cq-colour", "source"])
        self.assertTrue(upscale_video.cq_default_luma_only(args))
        args.method = "flashvsr"
        self.assertFalse(upscale_video.cq_default_luma_only(args))

    def test_cq_chunks_dissolve_across_their_overlap_and_keep_every_frame(self) -> None:
        try:
            ffmpeg = common.find_ffmpeg("")
        except Exception:
            self.skipTest("FFmpeg is not available")
        import cv2

        def clip(path: Path, levels: list[int]) -> None:
            raw = b"".join(bytes([level]) * (64 * 48 * 3) for level in levels)
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "64x48",
                            "-r", "24", "-i", "-", "-c:v", "ffv1", str(path)], input=raw, check=True)

        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            first, second, output = folder / "first.mkv", folder / "second.mkv", folder / "joined.mkv"
            # Source frames 0-19, then 16-39 with a 4-frame lead-in rendered 40 levels brighter.
            clip(first, [5 * index for index in range(20)])
            clip(second, [5 * index + 40 for index in range(16, 40)])

            upscale_video.crossfade_chunks(ffmpeg, [(first, 0), (second, 4)], 24.0, 40, first, output)

            capture = cv2.VideoCapture(str(output))
            levels = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                levels.append(frame.mean())
            capture.release()
        self.assertEqual(len(levels), 40)
        offsets = [level - 5 * index for index, level in enumerate(levels)]
        self.assertTrue(all(abs(offset) <= 2 for offset in offsets[:16]), offsets)
        self.assertTrue(all(abs(offset - 40) <= 2 for offset in offsets[20:]), offsets)
        # The lead-in frames ramp from the first render towards the second (xfade weights 0, 1/4, 1/2, 3/4).
        self.assertTrue(all(2 < offset < 38 for offset in offsets[17:20]), offsets)
        self.assertEqual(offsets[16:21], sorted(offsets[16:21]))

    def test_cq_chunks_cut_at_a_shot_change_then_dissolve_within_the_next_shot(self) -> None:
        try:
            ffmpeg = common.find_ffmpeg("")
        except Exception:
            self.skipTest("FFmpeg is not available")
        import cv2

        def clip(path: Path, levels: list[int]) -> None:
            raw = b"".join(bytes([level]) * (64 * 48 * 3) for level in levels)
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "64x48",
                            "-r", "24", "-i", "-", "-c:v", "ffv1", str(path)], input=raw, check=True)

        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            shot_one, shot_two_a, shot_two_b = folder / "a.mkv", folder / "b.mkv", folder / "c.mkv"
            output = folder / "joined.mkv"
            # Shot one is frames 0-9; shot two (10-39) is split in two with a 4-frame lead-in.
            # The concat at the cut used to leave a microsecond timebase that xfade refused.
            clip(shot_one, [5 * index for index in range(10)])
            clip(shot_two_a, [5 * index for index in range(10, 25)])
            clip(shot_two_b, [5 * index for index in range(21, 40)])

            upscale_video.crossfade_chunks(
                ffmpeg, [(shot_one, 0), (shot_two_a, 0), (shot_two_b, 4)], 24.0, 40, shot_one, output,
            )

            capture = cv2.VideoCapture(str(output))
            levels = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                levels.append(frame.mean())
            capture.release()
        self.assertEqual(len(levels), 40)
        self.assertTrue(all(abs(level - 5 * index) <= 2 for index, level in enumerate(levels)), levels)

    def test_upscale_delivery_blend_keeps_render_frames_aligned_with_an_mp4_source(self) -> None:
        try:
            ffmpeg = common.find_ffmpeg("")
        except Exception:
            self.skipTest("FFmpeg is not available")
        import cv2

        frames = 30
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            enhanced = folder / "render.mkv"
            output = folder / "delivery.mp4"
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=64x48:r=24",
                            "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)], check=True)
            # Frame i of the render is grey 6*i; a timebase slip would repeat or delay levels.
            raw = b"".join(bytes([6 * index]) * (64 * 48 * 3) for index in range(frames))
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "64x48",
                            "-r", "24", "-i", "-", "-c:v", "ffv1", str(enhanced)], input=raw, check=True)

            # Luma-only forces the blend path; its luma is all render, so levels track the render.
            upscale_video.blend_upscale_delivery(ffmpeg, source, enhanced, output, 64, 48, 24.0, [], 1.0, True)

            capture = cv2.VideoCapture(str(output))
            levels = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                levels.append(int(round(frame.mean())))
            capture.release()
        self.assertEqual(len(levels), frames)
        self.assertTrue(all(abs(got - 6 * index) <= 2 for index, got in enumerate(levels)), levels)

    def test_cq_delivers_the_source_aspect_but_renders_on_the_32px_grid(self) -> None:
        self.assertEqual(upscale_video.cq_output_size(854, 480, 720), (1280, 720))
        self.assertEqual(upscale_video.cq_generation_size(1280, 720), (1280, 704))
        self.assertEqual(upscale_video.cq_output_size(4096, 3112, 720), (948, 720))
        self.assertEqual(upscale_video.cq_generation_size(948, 720), (960, 704))
        self.assertEqual(upscale_video.cq_output_size(1920, 800, 720), (1728, 720))

    def test_auto_upscale_target_is_the_flashvsr_input_times_its_scale(self) -> None:
        values = {"method": "flashvsr", "auto_target_size": "true", "flashvsr_scale": "3", "cq_short_edge": "720", "cq_finish": "flashvsr"}
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            source = Path(tmp_text) / "input.mp4"
            source.write_bytes(b"video")
            with mock.patch("ai_remaster_gui.server.video_metrics", return_value={"width": 854, "height": 480}):
                self.assertEqual(app.auto_upscale_target(values, str(source)), (2562, 1440))
                # With CQ in front, FlashVSR's input is CQ's 720p output.
                self.assertEqual(app.auto_upscale_target(dict(values, method="ltx25cq"), str(source)), (3840, 2160))
                self.assertEqual(app.auto_upscale_target(dict(values, method="ltx25cq", flashvsr_scale="2"), str(source)), (2560, 1440))
                self.assertIsNone(app.auto_upscale_target(dict(values, auto_target_size="false"), str(source)))
                self.assertIsNone(app.auto_upscale_target(dict(values, method="seedvr2"), str(source)))
                self.assertIsNone(app.auto_upscale_target(dict(values, method="ltx25cq", cq_finish="lanczos"), str(source)))
                # With no finisher the CQ render is delivered, so its size is the target even with Auto off.
                no_finish = dict(values, method="ltx25cq", cq_finish="none", auto_target_size="false")
                self.assertEqual(app.auto_upscale_target(no_finish, str(source)), (1280, 720))
            self.assertIsNone(app.auto_upscale_target(values, str(Path(tmp_text) / "missing.mp4")))

    def test_upscale_preview_state_reports_the_cq_render_of_its_input(self) -> None:
        values = {"method": "ltx25cq", "cq_finish": "flashvsr", "cq_short_edge": "720"}
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            clip = Path(tmp_text) / "cqpreviewclip.mp4"
            clip.write_bytes(b"video")
            render = upscale_video.cq_render_path(clip, 1280, 720)
            with mock.patch("ai_remaster_gui.server.video_metrics", return_value={"width": 1920, "height": 1080}):
                self.assertEqual(app.upscale_cq_render_for(app.rel(clip), values), "")
                render.parent.mkdir(parents=True, exist_ok=True)
                render.write_bytes(b"cq")
                try:
                    self.assertEqual(app.upscale_cq_render_for(app.rel(clip), values), app.rel(render))
                    self.assertEqual(app.upscale_cq_render_for(app.rel(clip), dict(values, method="flashvsr")), "")
                    # Mid-preview: CQ has rendered, the finisher has not, so the CQ pass is shown on its own.
                    app.APP.settings["upscale"].update({
                        **values, "preview_source": app.rel(clip),
                        "preview_output": "output/upscaled/previews/cq_unfinished_preview.mp4",
                    })
                    with mock.patch.object(app.APP, "upscale_input_for", return_value=""):
                        preview = app.APP.upscale_preview_state()
                finally:
                    render.unlink(missing_ok=True)
        self.assertEqual(preview["source"], app.rel(clip))
        self.assertEqual(preview["cq"], app.rel(render))
        self.assertEqual(preview["exists"], "false")

    def test_auto_upscale_target_fills_the_target_fields_on_upscale_changes(self) -> None:
        app.APP.settings["upscale"].update({"method": "ltx25cq", "cq_finish": "flashvsr", "flashvsr_scale": "2", "target_width": "3840", "target_height": "2160"})
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            source = Path(tmp_text) / "recomp.mp4"
            source.write_bytes(b"video")
            with mock.patch.object(app.APP, "upscale_input_for", return_value=str(source)), \
                    mock.patch("ai_remaster_gui.server.video_metrics", return_value={"width": 1920, "height": 800}), \
                    mock.patch.object(app.APP, "save"):
                app.APP.update_settings("upscale", {"auto_target_size": "true"})
                filled = (app.APP.settings["upscale"]["target_width"], app.APP.settings["upscale"]["target_height"])
                app.APP.update_settings("upscale", {"auto_target_size": "false", "target_width": "3840", "target_height": "2160"})
                manual = (app.APP.settings["upscale"]["target_width"], app.APP.settings["upscale"]["target_height"])

        self.assertEqual(filled, ("3456", "1440"))
        self.assertEqual(manual, ("3840", "2160"))

    def test_cq_upscale_signature_tracks_its_finishing_backend(self) -> None:
        args = upscale_video.build_parser().parse_args(["--input", "input/example.mp4", "--method", "ltx25cq"])
        source = app.ROOT / "input" / "example.mp4"
        with mock.patch.object(upscale_video, "file_fingerprint", return_value={"size": 1}), \
                mock.patch.object(upscale_video, "video_info", return_value={"width": 854, "height": 480, "fps": 24.0, "frames": 96}):
            flash = upscale_video.signature(args, source, 3840, 2160)
            args.cq_finish = "lanczos"
            lanczos = upscale_video.signature(args, source, 3840, 2160)
            args.cq_seed = 7
            reseeded = upscale_video.signature(args, source, 3840, 2160)

        self.assertEqual(flash["finish"], "flashvsr")
        self.assertEqual(flash["finish_settings"]["method"], "flashvsr_ultra_fast")
        self.assertNotIn("source_fingerprint", flash["finish_settings"])
        self.assertEqual(flash["enhance"]["enhance_width"], 1280)
        self.assertEqual(flash["enhance"]["enhance_height"], 704)
        self.assertEqual(lanczos["finish_settings"], {"method": "lanczos"})
        self.assertEqual(flash["enhance"]["colour"], "model")
        self.assertEqual((flash["enhance"]["output_width"], flash["enhance"]["output_height"]), (1280, 720))
        self.assertNotEqual(lanczos, reseeded)
        self.assertEqual(upscale_video.delivery_dimensions(args, 3840, 2160), (3840, 2160))
        args.cq_finish = "ltx25"
        self.assertEqual(upscale_video.delivery_dimensions(args, 3840, 2160), (3840, 2176))
        args.cq_finish = "none"
        with mock.patch.object(upscale_video, "file_fingerprint", return_value={"size": 1}),                 mock.patch.object(upscale_video, "video_info", return_value={"width": 854, "height": 480, "fps": 24.0, "frames": 96}):
            self.assertEqual(upscale_video.signature(args, source, 1280, 720)["finish_settings"], {"method": "none"})

    def test_cq_upscale_with_no_finisher_delivers_the_cq_render(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            source = Path(tmp_text) / "source.mp4"
            source.write_bytes(b"video")
            output = Path(tmp_text) / "out.mp4"
            render = Path(tmp_text) / "cq.mkv"
            args = upscale_video.build_parser().parse_args([
                "--input", str(source), "--output", str(output), "--method", "ltx25cq", "--cq-finish", "none",
                "--target-width", "3840", "--target-height", "2160",
            ])
            with mock.patch.object(upscale_video, "file_fingerprint", return_value={"size": 1}),                     mock.patch.object(upscale_video, "video_info", return_value={"width": 854, "height": 480, "fps": 24.0, "frames": 96}),                     mock.patch.object(upscale_video, "find_ffmpeg", return_value="ffmpeg"),                     mock.patch.object(upscale_video, "cq_enhance", return_value=render) as enhance,                     mock.patch.object(upscale_video, "lanczos_upscale_run") as lanczos,                     mock.patch.object(upscale_video, "chunked_standard_upscale_run") as standard,                     mock.patch.object(upscale_video, "encode_upscale_delivery") as deliver,                     mock.patch.object(upscale_video, "write_signature") as write_signature:
                self.assertEqual(upscale_video.run(args), 0)

        enhance.assert_called_once()
        lanczos.assert_not_called()
        standard.assert_not_called()
        deliver.assert_called_once_with("ffmpeg", render, output)
        final_sig = write_signature.call_args[0][1]
        # The 3840x2160 target is ignored: CQ's own size is what is delivered.
        self.assertEqual((final_sig["target_width"], final_sig["target_height"]), (1280, 720))

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

        self.assertEqual(stage_keys, ["outpaint", "shots", "recomp", "upscale"])
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

        self.assertIn("Upscale chunk 2/12 rendering in ComfyUI (1 done", progress["label"])
        self.assertIn("ETA", progress["label"])
        self.assertGreater(progress["percent"], 10)
        self.assertLess(progress["percent"], 25)

    def test_upscale_progress_does_not_jump_to_ninety_from_elapsed_time(self) -> None:
        original_log = app.APP.log
        app.APP.settings["upscale"].update({
            "method": "flashvsr",
            "target_width": "4412",
            "target_height": "1888",
            "flashvsr_tiled_dit": "true",
            "flashvsr_tile_size": "512",
            "flashvsr_tile_overlap": "64",
        })
        app.APP.running_stage_key = "upscale"
        app.APP.running_stage = "Upscaling"
        app.APP.run_started_at = time.time() - 3600
        app.APP.log = [
            "Splitting upscaling into 12 chunk(s): 6s chunks, 8 overlap frame(s)",
            "Upscale chunk 1/12: frames 0-144, trim 0",
            "Queued ComfyUI prompt: prompt-id",
        ]

        try:
            progress = app.APP.estimate_running_progress()
        finally:
            app.APP.running_stage_key = ""
            app.APP.running_stage = ""
            app.APP.run_started_at = 0.0
            app.APP.log = original_log

        self.assertLess(progress["percent"], 20)
        self.assertIn("10x5 spatial tiles/pass", progress["label"])

    def test_ltx_upscale_chunk_completions_are_counted(self) -> None:
        chunk = server.upscale_chunk_progress("\n".join([
            "Upscale chunk 1/3: frames 0-97, trim 0",
            "Wrote LTX 2.5 upscaled chunk: first.mkv",
            "Upscale chunk 2/3: frames 89-186, trim 8",
            "Reuse LTX 2.5 chunk from compatible cache: second.mkv",
        ]))

        self.assertEqual(chunk, {
            "done": 2, "current": 2, "total": 3,
            "active_percent": 0, "active_value": 0, "active_total": 0,
        })

    def test_upscale_progress_uses_live_comfy_tile_progress_inside_chunk(self) -> None:
        chunk = server.upscale_chunk_progress("\n".join([
            "Upscale chunk 1/3: frames 0-144, trim 0",
            "ComfyUI upscale pass: 17/50 (34%)",
        ]))

        self.assertEqual(chunk["active_percent"], 34)
        self.assertEqual((chunk["active_value"], chunk["active_total"]), (17, 50))

    def test_upscale_progress_ignores_completed_chunks_from_an_older_run(self) -> None:
        original_log = app.APP.log
        app.APP.running_stage_key = "upscale"
        app.APP.running_stage = "Upscaling"
        app.APP.run_started_at = time.time() - 60
        app.APP.log = [
            "Upscale chunk 1/1: frames 0-24, trim 0",
            "Wrote upscaled chunk: old.mkv",
            "Wrote upscaled video: old.mp4",
            "> python -u scripts\\upscale_video.py --input new.mp4",
            "Upscale chunk 1/12: frames 0-144, trim 0",
            "Queued ComfyUI prompt: new-prompt-id",
        ]

        try:
            progress = app.APP.estimate_running_progress()
        finally:
            app.APP.running_stage_key = ""
            app.APP.running_stage = ""
            app.APP.run_started_at = 0.0
            app.APP.log = original_log

        self.assertLess(progress["percent"], 20)
        self.assertIn("Upscale chunk 1/12", progress["label"])

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

    def test_cq_upscale_progress_splits_enhance_and_finish_passes(self) -> None:
        original_log = app.APP.log
        app.APP.running_stage_key = "upscale"
        app.APP.running_stage = "Upscaling"
        app.APP.run_started_at = time.time() - 300
        enhance_log = [
            "scripts/upscale_video.py --method ltx25cq",
            "Enhancing with LTX 2.5 CQ in 2 chunk(s): 854x480 -> 1280x704 at 30 fps (resample from 24 fps)",
            "CQ enhance chunk 1/2: frames 0-120, trim 0, keep 120, LTX window 153",
            "Wrote CQ enhanced chunk: cq_final_0000.mkv",
            "CQ enhance chunk 2/2: frames 112-240, trim 8, keep 120, LTX window 161",
        ]
        finish_log = [
            "Stitching upscaled chunks: 2 chunk(s)",
            "Muxing original audio into upscaled video",
            "Finishing CQ-enhanced video with FlashVSR: enhanced.mkv",
            "Splitting upscaling into 4 chunk(s): 6s chunks, 8 overlap frame(s)",
            "Upscale chunk 1/4: frames 0-144, trim 0",
            "Wrote upscaled chunk: chunk_0001.mp4",
            "Upscale chunk 2/4: frames 136-288, trim 8",
        ]

        try:
            app.APP.log = enhance_log
            enhancing = app.APP.estimate_running_progress()
            app.APP.log = enhance_log + finish_log
            finishing = app.APP.estimate_running_progress()
        finally:
            app.APP.running_stage_key = ""
            app.APP.running_stage = ""
            app.APP.run_started_at = 0.0
            app.APP.log = original_log

        self.assertTrue(enhancing["label"].startswith("CQ enhance: chunk 2/2"), enhancing["label"])
        self.assertLess(enhancing["percent"], 50)
        self.assertTrue(finishing["label"].startswith("Finishing: Upscale chunk 2/4"), finishing["label"])
        self.assertGreaterEqual(finishing["percent"], 50)
        self.assertLess(finishing["percent"], 75)

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

    def test_chunk_length_checkbox_only_disables_the_length_controls(self) -> None:
        # This once disabled every button under the slider's parent, which meant the
        # nearby +/-1 frame nudges. Those went, so the selector started hitting Save and
        # Regenerate instead: clearing "Custom length" could never be saved.
        source = (app.ROOT / "ai_remaster_gui" / "static" / "js" / "render-outpaint.js").read_text(encoding="utf-8")
        body = source.split("function toggleChunkLength(", 1)[1].split("\nfunction ", 1)[0]
        self.assertIn("slider.disabled = !enabled", body)
        self.assertIn("input.disabled = !enabled", body)
        self.assertNotIn("shot-tools", body)
        self.assertNotIn("button.disabled", body)

    def test_aspect_preview_overlay_is_boxed_like_the_preview_image(self) -> None:
        # ".preview img" and ".preview.compact img" out-specify the frame rule and never
        # match the mask overlay, so the picture and its overlay were sized by different
        # rules and the coral mask covered the whole pane.
        css = (app.ROOT / "ai_remaster_gui" / "static" / "styles.css").read_text(encoding="utf-8")
        frame = next(rule for rule in css.split("}") if rule.strip().startswith(".aspect-preview-frame{"))
        self.assertIn("grid-template-rows", frame)
        override = "".join(rule for rule in css.split("}") if "max-height:none" in rule)
        for selector in (
            ".preview .aspect-preview-frame img",
            ".preview .aspect-preview-frame canvas",
            ".preview.compact .aspect-preview-frame img",
            ".preview.compact .aspect-preview-frame canvas",
        ):
            self.assertIn(selector, override)

    def test_shot_cq_guides_are_saved_as_frames_and_shown_as_shot_numbers(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            manifest = Path(tmp_text) / "shots.csv"
            manifest.write_text("start_frame,end_frame\n0,10\n10,20\n20,30\n30,40\n", encoding="utf-8")
            manifest_text = app.rel(manifest)

            numbers = references.update_shot_cq_guides(manifest_text, 3, "3, 1, 3")
            with manifest.open(encoding="utf-8", newline="") as handle:
                stored = [row.get("cq_guide_shots", "") for row in csv.DictReader(handle)]
            # Splitting shot 1 renumbers the later shots, but the guides follow their frames.
            references.split_manifest_shot(manifest_text, 0, 5 / 24)
            renumbered = references.shot_rows(manifest_text)[-1]["cq_guide_shot_numbers"]
            with self.assertRaises(ValueError):
                references.update_shot_cq_guides(manifest_text, 0, "1")
            with self.assertRaises(ValueError):
                references.update_shot_cq_guides(manifest_text, 0, "9")
            cleared = references.update_shot_cq_guides(manifest_text, 4, "")

        self.assertEqual(numbers, [3, 1])
        self.assertEqual(stored, ["", "", "", "20;0"])
        self.assertEqual(renumbered, [4, 1])
        self.assertEqual(cleared, [])

    def test_shot_cq_anchor_endpoint_writes_the_shot_list_column(self) -> None:
        server = app.create_server("127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
                manifest = Path(tmp_text) / "shots.csv"
                manifest.write_text("start_frame,end_frame\n0,400\n400,420\n", encoding="utf-8")

                def post(value: str) -> dict:
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{server.server_port}/api/shot-cq-anchor",
                        data=json.dumps({"manifest": app.rel(manifest), "index": 0, "chunk": value}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=10) as response:
                        return json.loads(response.read().decode("utf-8"))

                def column() -> list[str]:
                    with manifest.open(encoding="utf-8", newline="") as handle:
                        return [row.get("cq_anchor_chunk", "") for row in csv.DictReader(handle)]

                self.assertTrue(post("3")["ok"])
                self.assertEqual(column(), ["3", ""])
                post("1")  # the first chunk is the default, so nothing is stored
                self.assertEqual(column(), ["", ""])
                post("bogus")
                self.assertEqual(column(), ["", ""])
        finally:
            server.shutdown()

    def test_shot_cq_luma_only_endpoint_writes_the_shot_list_column(self) -> None:
        server = app.create_server("127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
                manifest = Path(tmp_text) / "shots.csv"
                manifest.write_text("start_frame,end_frame,upscale_strength\n0,10,80\n10,20,\n", encoding="utf-8")

                def post(value: str) -> dict:
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{server.server_port}/api/shot-cq-luma-only",
                        data=json.dumps({"manifest": app.rel(manifest), "index": 1, "luma_only": value}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=10) as response:
                        return json.loads(response.read().decode("utf-8"))

                def column() -> list[str]:
                    with manifest.open(encoding="utf-8", newline="") as handle:
                        return [row.get("cq_luma_only", "") for row in csv.DictReader(handle)]

                self.assertTrue(post("true")["ok"])
                self.assertEqual(column(), ["", "true"])
                post("bogus")  # anything else falls back to inheriting the project setting
                self.assertEqual(column(), ["", ""])
                with manifest.open(encoding="utf-8", newline="") as handle:
                    self.assertEqual(next(csv.DictReader(handle))["upscale_strength"], "80")
        finally:
            server.shutdown()
            server.server_close()

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
