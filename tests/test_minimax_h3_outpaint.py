from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import artifact_ids as aid  # noqa: E402
import common  # noqa: E402
import dependency_manager  # noqa: E402
import minimax_h3_outpaint as h3  # noqa: E402
import outpaint_video  # noqa: E402
import wan_vace_outpaint as wan  # noqa: E402


def write_video(ffmpeg: str, path: Path, frames: list[np.ndarray], fps: float = 24.0) -> None:
    height, width = frames[0].shape[:2]
    writer = wan.FrameWriter(ffmpeg, path, width, height, fps, wan.lossless_control_args())
    for frame in frames:
        writer.write(frame)
    writer.close()


def read_video(ffmpeg: str, path: Path, width: int, height: int) -> list[np.ndarray]:
    with wan.FrameReader(ffmpeg, path, width, height) as reader:
        return [frame.copy() for frame in reader]


class H3GridTests(unittest.TestCase):
    def test_lengths_snap_up_to_the_17k_plus_5_grid(self) -> None:
        self.assertEqual([h3.h3_valid_length(n) for n in (1, 5, 6, 22, 23, 124, 125, 361)], [5, 5, 22, 22, 39, 124, 141, 362])

    def test_passes_stay_within_the_measured_24gb_budget(self) -> None:
        settings = h3.H3Settings()
        # 124 frames at 1536x640 fitted on a 4090; 192 started offloading to system RAM. The 22
        # anchored context frames count against that budget, leaving 90 on the 17k + 5 grid.
        self.assertEqual(settings.pass_frames(1536, 640), 90)
        self.assertEqual(settings.pass_frames(1536, 672), 90)
        # Small canvases run longer passes, never past the top of H3's trained range.
        self.assertEqual(settings.pass_frames(864, 480), 260)
        self.assertEqual(settings.pass_frames(480, 272), 362)
        self.assertEqual(h3.H3Settings(chunk_frames=100).pass_frames(864, 480), 90)
        # The overlap context ends on a latent group boundary (5 + 17 frames).
        self.assertEqual(settings.context_frames, 22)

    def test_h3_renders_inside_its_trained_canvas(self) -> None:
        self.assertEqual(aid.h3_render_size(1856, 800), (1536, 672))
        self.assertEqual(aid.h3_render_size(1920, 1088), (1344, 768))
        self.assertEqual(aid.h3_render_size(1088, 1920), (768, 1344))
        self.assertEqual(aid.h3_render_size(1280, 704), (1280, 704))
        for size in ((1856, 800), (2560, 1088), (4096, 1760)):
            width, height = aid.h3_render_size(*size)
            self.assertLessEqual(width * height, aid.H3_MAX_PIXELS)
            self.assertEqual((width % 32, height % 32), (0, 0))


class H3CanvasTests(unittest.TestCase):
    def test_a_smaller_h3_canvas_keeps_shared_names_but_prepares_its_own_canvas(self) -> None:
        h3_args = outpaint_video.build_parser().parse_args(["--source", "input/film.mp4", "--outpaint-backend", "h3"])
        ltx_args = outpaint_video.build_parser().parse_args(["--source", "input/film.mp4"])
        source = Path("input/film.mp4")

        self.assertEqual(outpaint_video.render_size(h3_args, 1856, 800), (1536, 672))
        self.assertEqual(outpaint_video.render_size(ltx_args, 1856, 800), (1856, 800))
        # The GUI's guide editor keeps the working-size canvas; H3 writes a separate one.
        self.assertIn("preparedh3", outpaint_video.prepared_for(source, "21:9", 800, h3_args).name)
        self.assertIn("prepared_", outpaint_video.prepared_for(source, "21:9", 800, ltx_args).name)
        self.assertIn("prepared_", outpaint_video.prepared_for(source, "16:9", 480, h3_args).name)
        # The chunk plan (and so guides and prompts) stays shared between the models.
        self.assertEqual(
            outpaint_video.default_chunk_manifest(source, "21:9", 1856, 800, h3_args),
            outpaint_video.default_chunk_manifest(source, "21:9", 1856, 800, ltx_args),
        )


class H3PromptTests(unittest.TestCase):
    def build(self, **settings):
        return h3.build_h3_prompt(
            "arp_outpaint_h3/c.mkv", "arp_outpaint_h3/m.mkv", 1536, 672, 362, 24.0,
            "a street", 7, h3.H3Settings(**settings), "arp_outpaint/x_p00",
        )

    def test_graph_follows_the_fun_controlnet_inpaint_template(self) -> None:
        prompt = self.build()
        classes = {node["class_type"] for node in prompt.values()}
        self.assertTrue(classes <= set(h3.H3_REQUIRED_NODES), classes - set(h3.H3_REQUIRED_NODES))
        control = prompt["15"]["inputs"]
        # The regenerate mask and the source video drive the Fun ControlNet's inpaint mode.
        self.assertEqual(control["mask"], ["5", 0])
        self.assertEqual(control["source_video"], ["2", 0])
        self.assertEqual(control["model"], ["10", 0])
        self.assertEqual(prompt["13"]["inputs"]["type"], "minimax")
        self.assertEqual(prompt["16"]["inputs"]["length"], 362)
        self.assertEqual(prompt["19"]["inputs"]["steps"], 20)
        self.assertEqual(prompt["18"]["inputs"]["sampler_name"], "res_multistep")
        self.assertEqual(prompt[h3.H3_OUTPUT_NODE_ID]["class_type"], "VHS_VideoCombine")
        self.assertNotIn("11", prompt)

    def test_previous_chunk_frames_are_anchored_as_a_guide_clip(self) -> None:
        prompt = h3.build_h3_prompt(
            "arp_outpaint_h3/c.mkv", "arp_outpaint_h3/m.mkv", 1536, 640, 124, 24.0,
            "a street", 7, h3.H3Settings(), "arp_outpaint/x_p00", context_frames=22,
        )

        # The finished frames lead the source video; they become a never-denoised keyframe clip.
        self.assertEqual(prompt["25"]["inputs"], {"image": ["2", 0], "batch_index": 0, "length": 22})
        self.assertEqual(prompt["26"]["class_type"], "MiniMaxH3AddGuide")
        self.assertEqual(prompt["26"]["inputs"]["frame_idx"], 0)
        self.assertEqual(prompt["17"]["inputs"]["conditioning"], ["26", 0])
        # A pass with no context (first chunk, or after a cut) has no guide.
        self.assertNotIn("26", self.build())

    def test_pdd_follows_the_pack_recipe(self) -> None:
        prompt = self.build(pdd=True)

        # Shift 12/3 before PDD, PDD's trained sigmas into an euler sampler, and the inpaint
        # patch applied on top of the distilled model.
        self.assertEqual(prompt["11"]["class_type"], "MiniMaxH3SigmaShift")
        self.assertEqual((prompt["11"]["inputs"]["shift_video"], prompt["11"]["inputs"]["shift_audio"]), (12.0, 3.0))
        self.assertEqual(prompt["24"]["inputs"]["pdd_file"], dependency_manager.H3_PDD_ACC)
        self.assertEqual(prompt["24"]["inputs"]["model"], ["11", 0])
        self.assertEqual(prompt["24"]["inputs"]["nfe"], "8")
        self.assertEqual(prompt["15"]["inputs"]["model"], ["24", 0])
        self.assertEqual(prompt["21"]["inputs"]["sigmas"], ["24", 1])
        self.assertEqual(prompt["18"]["inputs"]["sampler_name"], "euler")
        self.assertNotIn("19", prompt)


class H3LicenceTests(unittest.TestCase):
    def test_models_are_never_downloaded_without_a_confirmed_licence(self) -> None:
        with mock.patch.object(dependency_manager, "ensure_hf_models") as download:
            with self.assertRaises(RuntimeError) as raised:
                dependency_manager.ensure_h3_outpaint_models(Path("comfy"), license_confirmed=False)
            download.assert_not_called()
            self.assertIn("UK", str(raised.exception))
            dependency_manager.ensure_h3_outpaint_models(Path("comfy"), license_confirmed=True, pdd=True)
        downloaded = [model.file for model in download.call_args.args[1]]
        self.assertIn(dependency_manager.H3_DIFFUSION_GGUF, downloaded)
        self.assertIn(dependency_manager.H3_PDD_ACC, downloaded)

    def test_the_script_refuses_to_start_without_a_confirmed_licence(self) -> None:
        with mock.patch.object(sys, "argv", ["outpaint_video.py", "--source", "input/film.mp4", "--outpaint-backend", "h3", "--dry-run"]):
            with self.assertRaises(SystemExit) as raised:
                outpaint_video.main()
        self.assertIn("licence", str(raised.exception.code))


class H3BackendWiringTests(unittest.TestCase):
    def test_h3_shares_the_masked_pass_pipeline_with_its_own_limits(self) -> None:
        args = outpaint_video.build_parser().parse_args(["--source", "input/film.mp4", "--outpaint-backend", "h3", "--overlap-frames", "8"])

        self.assertTrue(outpaint_video.uses_masked_pass(args))
        self.assertEqual(outpaint_video.outpaint_artifact_tag(args, "outpaint"), "outpainth3")
        self.assertEqual(outpaint_video.render_size(args, 1856, 800), (1536, 672))
        self.assertEqual(outpaint_video.max_chunk_frames(args, 1536, 672), 90)
        self.assertEqual(outpaint_video.chunk_overlap_frames(args), 22)
        self.assertEqual(outpaint_video.masked_pass_settings(args).label, "MiniMax H3")


class H3RenderChunkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ffmpeg = common.find_ffmpeg()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.width, self.height = 32, 16

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_passes_are_padded_to_the_h3_grid_and_composited_exactly(self) -> None:
        width, height = self.width, self.height
        sources = []
        for index in range(30):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, 8:] = 40 + index * 5
            sources.append(frame)
        chunk = self.root / "prepared.mkv"
        write_video(self.ffmpeg, chunk, sources)
        exact = np.zeros((height, width), dtype=np.uint8)
        exact[:, :8] = 255
        lengths: list[int] = []

        def render_pass(control: Path, mask: Path, length: int, pass_index: int, context_frames: int) -> Path:
            lengths.append(length)
            self.assertEqual(len(read_video(self.ffmpeg, control, width, height)), length)
            rendered = self.root / f"rendered_{pass_index}.mkv"
            write_video(self.ffmpeg, rendered, [np.full((height, width, 3), 200, dtype=np.uint8)] * length)
            return rendered

        output = self.root / "raw.mkv"
        wan.render_chunk(
            ffmpeg=self.ffmpeg, chunk_video=chunk, output=output, width=width, height=height, fps=24.0,
            total_frames=30, masks=wan.StaticMasks(exact, exact, feather_px=0), known_prefix=[],
            guides={}, render_pass=render_pass, work_dir=self.root / "work",
            settings=h3.H3Settings(), intermediate_profile="lossless", log=lambda message: None,
        )

        # One pass for the whole chunk, sampled on the 17k + 5 grid (30 -> 39 frames).
        self.assertEqual(lengths, [39])
        result = read_video(self.ffmpeg, output, width, height)
        self.assertEqual(len(result), 30)
        for index in range(30):
            np.testing.assert_array_equal(result[index][:, 8:], sources[index][:, 8:])
            self.assertTrue((result[index][:, :8] == 200).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
