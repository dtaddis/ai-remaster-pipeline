from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import common  # noqa: E402
import outpaint_video  # noqa: E402
import wan_vace_outpaint as wan  # noqa: E402


class WanWindowPlanTests(unittest.TestCase):
    def assert_plan_covers(self, windows, total, known_prefix, window, context, cuts=()) -> None:
        written = known_prefix
        for start, end in windows:
            self.assertLessEqual(start, written)
            self.assertGreater(end, written)
            self.assertLessEqual(end - start, window)
            self.assertLessEqual(written - start, context)
            self.assertFalse(any(start < cut < end for cut in cuts), (start, end, cuts))
            written = end
        self.assertEqual(written, total)

    def test_windows_cover_every_frame_with_even_lengths(self) -> None:
        windows = wan.plan_windows(481, 0, 81, 13)

        self.assert_plan_covers(windows, 481, 0, 81, 13)
        # Same window count as greedy filling, but no near-empty tail window.
        self.assertEqual(len(windows), 7)
        lengths = [end - start for start, end in windows]
        self.assertLessEqual(max(lengths) - min(lengths), 2)

    def test_short_tail_is_spread_instead_of_resampling_a_full_window(self) -> None:
        # Greedy planning sampled 81 + 81 + 81 frames here to add only 12 in the last window.
        self.assertEqual(wan.plan_windows(161, 0, 81, 13), [(0, 63), (50, 112), (99, 161)])
        self.assertEqual(wan.plan_windows(161, 0, 161, 13), [(0, 161)])

    def test_known_prefix_is_context_not_regenerated(self) -> None:
        windows = wan.plan_windows(200, 8, 81, 13)

        self.assertEqual(windows[0][0], 0)
        self.assert_plan_covers(windows, 200, 8, 81, 13)
        self.assertEqual(wan.plan_windows(200, 40, 81, 13)[0][0], 27)
        self.assertEqual(wan.plan_windows(8, 8, 81, 13), [])

    def test_windows_never_span_or_take_context_across_a_cut(self) -> None:
        windows = wan.plan_windows(200, 0, 81, 13, cuts=[50, 120])

        self.assertEqual(windows, [(0, 50), (50, 120), (120, 200)])
        # A cut inside the previous chunk's overlap: context starts at the cut.
        self.assertEqual(wan.plan_windows(100, 8, 81, 13, cuts=[5])[0][0], 5)
        # A cut right after the overlap: that shot starts without the overlap as context.
        windows = wan.plan_windows(100, 8, 81, 13, cuts=[8])
        self.assertEqual(windows[0][0], 8)
        self.assert_plan_covers(windows, 100, 8, 81, 13, cuts=[8])

    def test_default_windows_are_161_frames_within_the_measured_vram_budget(self) -> None:
        self.assertEqual(wan.WanSettings().window_frames_for(1280, 704), 161)
        self.assertEqual(wan.WanSettings().window_frames_for(864, 480), 161)
        # 1080p exceeds what was measured to fit in 24 GB: fall back to Wan's native 81.
        self.assertEqual(wan.WanSettings().window_frames_for(1920, 1088), 81)
        self.assertEqual(wan.WanSettings(window_frames=81).window_frames_for(1280, 704), 81)

    def test_ltx_trigger_word_is_removed_from_wan_prompts(self) -> None:
        self.assertEqual(wan.wan_prompt("outpaint"), "")
        self.assertEqual(wan.wan_prompt("Outpaint, a crowded hall"), "a crowded hall")
        self.assertEqual(wan.wan_prompt("a street, outpaint"), "a street")
        self.assertEqual(wan.wan_prompt("the outpainter's studio"), "the outpainter's studio")

    def test_lengths_round_up_to_wan_temporal_grid(self) -> None:
        self.assertEqual([wan.wan_valid_length(n) for n in (1, 2, 5, 6, 81, 82)], [1, 5, 5, 9, 81, 85])


class WanMaskTests(unittest.TestCase):
    def test_seam_alpha_keeps_exact_mask_and_ramps_into_source(self) -> None:
        exact = np.zeros((8, 40), dtype=np.uint8)
        exact[:, :10] = 255

        alpha = wan.seam_alpha(exact, 10)

        self.assertTrue((alpha[:, :10] == 1).all())
        self.assertGreater(alpha[0, 12], alpha[0, 16])
        self.assertEqual(alpha[0, 30], 0)
        self.assertTrue((wan.seam_alpha(exact, 0) == (exact > 0)).all())

    def test_black_region_masks_follow_each_frame(self) -> None:
        masks = wan.BlackRegionMasks(threshold=12, custom=None, overlap_px=2, feather_px=0)
        frame = np.full((10, 10, 3), 100, dtype=np.uint8)
        frame[:, :3] = 5

        generation, alpha = masks.for_frame(frame)

        self.assertTrue(alpha[:, :3].all())
        self.assertFalse(alpha[:, 3:].any())
        # Generation grows under protected pixels; the composite mask does not.
        self.assertTrue(generation[:, :5].all())
        self.assertFalse(generation[:, 6:].any())


class WanPromptTests(unittest.TestCase):
    def test_prompt_follows_comfy_vace_template_with_arp_models(self) -> None:
        prompt = wan.build_wan_vace_prompt(
            "arp_outpaint_wan/c.mkv", "arp_outpaint_wan/m.mkv", 1280, 704, 81, 24.0,
            "a street", "blur", 7, wan.WanSettings(), "arp_outpaint/x_w00",
        )

        classes = {node["class_type"] for node in prompt.values()}
        self.assertTrue(classes <= set(wan.WAN_VACE_REQUIRED_NODES), classes - set(wan.WAN_VACE_REQUIRED_NODES))
        vace = prompt["17"]["inputs"]
        self.assertEqual((vace["width"], vace["height"], vace["length"]), (1280, 704, 81))
        self.assertEqual(vace["control_video"], ["2", 0])
        self.assertEqual(vace["control_masks"], ["5", 0])
        self.assertEqual(prompt["10"]["inputs"]["unet_name"], outpaint_video.WAN_VACE_GGUF_MODEL)
        self.assertEqual(prompt["13"]["inputs"]["type"], "wan")
        self.assertEqual(prompt["18"]["inputs"]["seed"], 7)
        self.assertEqual(prompt[wan.WAN_OUTPUT_NODE_ID]["inputs"]["format"], "video/ffv1-mkv")
        # Every link targets a node that exists.
        for node in prompt.values():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    self.assertIn(value[0], prompt)


    def test_context_sampling_patches_the_model_only_for_passes_longer_than_a_window(self) -> None:
        settings = wan.WanSettings(sampling="context", window_frames=81, context_overlap=30)
        args = ("c.mkv", "m.mkv", 1280, 704)
        tail = (24.0, "", "", 1, settings, "x")

        long_pass = wan.build_wan_vace_prompt(*args, 481, *tail)
        short_pass = wan.build_wan_vace_prompt(*args, 61, *tail)

        self.assertEqual(long_pass["18"]["inputs"]["model"], ["22", 0])
        self.assertEqual(long_pass["22"]["class_type"], "WanContextWindowsManual")
        self.assertEqual(long_pass["22"]["inputs"]["context_length"], 81)
        self.assertNotIn("22", short_pass)
        self.assertEqual(settings.pass_frames(1280, 704), 481)
        # 1080p passes shrink to the same RAM budget instead of growing 2.3x.
        self.assertEqual(settings.pass_frames(1920, 1088), 205)
        self.assertEqual(wan.WanSettings().pass_frames(1920, 1088), 81)
        # One pass per shot, never across a cut.
        self.assertEqual(wan.plan_windows(481, 0, settings.pass_frames(1280, 704), 13, cuts=[208]), [(0, 208), (208, 481)])


def write_video(ffmpeg: str, path: Path, frames: list[np.ndarray], fps: float = 24.0) -> None:
    height, width = frames[0].shape[:2]
    writer = wan.FrameWriter(ffmpeg, path, width, height, fps, wan.lossless_control_args())
    for frame in frames:
        writer.write(frame)
    writer.close()


def read_video(ffmpeg: str, path: Path, width: int, height: int) -> list[np.ndarray]:
    with wan.FrameReader(ffmpeg, path, width, height) as reader:
        return [frame.copy() for frame in reader]


class WanRenderChunkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ffmpeg = common.find_ffmpeg()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.width, self.height = 32, 16

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_windows_continue_from_finished_frames_and_composite_exactly(self) -> None:
        width, height = self.width, self.height
        # Source: left quarter is the black bar to outpaint; the picture carries its frame index.
        sources = []
        for index in range(20):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, 8:] = 40 + index * 5
            sources.append(frame)
        chunk = self.root / "prepared.mkv"
        write_video(self.ffmpeg, chunk, sources)
        exact = np.zeros((height, width), dtype=np.uint8)
        exact[:, :8] = 255
        masks = wan.StaticMasks(exact, exact, feather_px=0)
        prefix = [np.full((height, width, 3), 250, dtype=np.uint8) for _ in range(2)]
        guide = np.full((height, width, 3), 77, dtype=np.uint8)
        seen: list[tuple[list[np.ndarray], list[np.ndarray]]] = []

        def render_window(control: Path, mask: Path, length: int, window_index: int) -> Path:
            controls = read_video(self.ffmpeg, control, width, height)
            mask_frames = read_video(self.ffmpeg, mask, width, height)
            self.assertEqual(len(controls), length)
            seen.append((controls, mask_frames))
            rendered = self.root / f"rendered_{window_index}.mkv"
            # A fake model that paints every generated pixel 200 + window index.
            write_video(self.ffmpeg, rendered, [np.full((height, width, 3), 200 + window_index, dtype=np.uint8)] * length)
            return rendered

        output = self.root / "raw.mkv"
        wan.render_chunk(
            ffmpeg=self.ffmpeg, chunk_video=chunk, output=output, width=width, height=height, fps=24.0,
            total_frames=20, masks=masks, known_prefix=prefix, guides={10: (guide, 1.0)},
            render_window=render_window, work_dir=self.root / "work",
            settings=wan.WanSettings(window_frames=9, context_frames=3), intermediate_profile="lossless",
        )

        result = read_video(self.ffmpeg, output, width, height)
        self.assertEqual(len(result), 20)
        self.assertTrue((result[0] == 250).all() and (result[1] == 250).all())
        for index in range(2, 20):
            np.testing.assert_array_equal(result[index][:, 8:], sources[index][:, 8:])
            self.assertGreaterEqual(int(result[index][0, 0, 0]), 200)
        # Window 1 opens with the previous chunk's frames, fully kept (mask 0).
        first_controls, first_masks = seen[0]
        self.assertTrue((first_controls[0] == 250).all())
        self.assertFalse(first_masks[0].any())
        # New frames: grey (unknown) in the bar, source elsewhere, bar masked.
        self.assertTrue((first_controls[2][:, :8] == wan.WAN_UNKNOWN_GREY).all())
        np.testing.assert_array_equal(first_controls[2][:, 8:], sources[2][:, 8:])
        self.assertTrue((first_masks[2][:, :8] == 255).all())
        # Later windows start from the composited frames of the previous window.
        second_controls, second_masks = seen[1]
        np.testing.assert_array_equal(second_controls[0], result[6])
        self.assertFalse(second_masks[0].any())
        # The guide fills its frame's bar and a strength-1 guide is kept as given.
        guide_window = next(i for i, (start, end) in enumerate(wan.plan_windows(20, 2, 9, 3)) if start <= 10 < end)
        start = wan.plan_windows(20, 2, 9, 3)[guide_window][0]
        controls, mask_frames = seen[guide_window]
        self.assertTrue((controls[10 - start][:, :8] == 77).all())
        self.assertFalse(mask_frames[10 - start].any())
        self.assertEqual(list((self.root / "work").iterdir()), [])


class WanBackendWiringTests(unittest.TestCase):
    def args(self, **extra) -> argparse.Namespace:
        args = outpaint_video.build_parser().parse_args(["--source", "input/example.mp4", *extra.pop("argv", [])])
        for key, value in extra.items():
            setattr(args, key, value)
        return args

    def test_wan_renders_use_their_own_cache_names(self) -> None:
        ltx = self.args()
        wan_args = self.args(argv=["--outpaint-backend", "wan-vace"])
        source = Path("input/example.mp4")

        self.assertIn("outpaintwan", outpaint_video.default_output(source, "16:9", 720, wan_args).name)
        self.assertIn("rawcomfywan", outpaint_video.default_raw_output(source, "16:9", 720, wan_args).name)
        self.assertNotIn("wan", outpaint_video.default_output(source, "16:9", 720, ltx).name)
        # The user's chunk plan is shared between every backend.
        self.assertEqual(
            outpaint_video.default_chunk_manifest(source, "16:9", 1280, 704, ltx),
            outpaint_video.default_chunk_manifest(source, "16:9", 1280, 704, wan_args),
        )

    def test_signature_identifies_wan_without_changing_ltx_signatures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prepared = Path(tmp) / "prepared.mkv"
            prepared.write_bytes(b"prepared")
            workflow = outpaint_video.DEFAULT_WORKFLOW
            ltx = outpaint_video.raw_signature(self.args(), workflow, prepared)
            wan_sig = outpaint_video.raw_signature(self.args(argv=["--outpaint-backend", "wan-vace", "--wan-steps", "8"]), workflow, prepared)

        self.assertNotIn("wan_vace", ltx)
        self.assertEqual(ltx["outpaint_pipeline"], "crop_first_pillarbox_letterbox_video_only_v1")
        self.assertEqual(wan_sig["outpaint_pipeline"], "wan21_vace14b_windowed_v1")
        self.assertEqual(wan_sig["wan_vace"]["steps"], 8)
        self.assertEqual(wan_sig["gguf_model"], outpaint_video.WAN_VACE_GGUF_MODEL)
        self.assertIsNone(wan_sig["workflow_fingerprint"])


if __name__ == "__main__":
    unittest.main()
