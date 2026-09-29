from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import final_composite  # noqa: E402
import frame_masks  # noqa: E402
import outpaint_video  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def mask_png(width: int, height: int, box: tuple[int, int, int, int] | None) -> bytes:
    import cv2
    import numpy as np

    mask = np.zeros((height, width), dtype=np.uint8)
    if box:
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = 255
    ok, encoded = cv2.imencode(".png", mask)
    assert ok
    return encoded.tobytes()


def decode_gray_video(ffmpeg: str, video: Path, width: int, height: int):
    import numpy as np

    raw = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width)


class FrameMaskArchiveTests(unittest.TestCase):
    def test_frames_round_trip_and_an_emptied_archive_disappears(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "masks.zip"
            frame_masks.write_frames(archive, {3: b"three", 10: b"ten"})
            self.assertEqual(frame_masks.frame_indices(archive), [3, 10])
            self.assertEqual(frame_masks.read_frame(archive, 10), b"ten")
            self.assertIsNone(frame_masks.read_frame(archive, 4))

            frame_masks.write_frames(archive, {3: None, 10: b"TEN"})
            self.assertEqual(frame_masks.read_frames(archive), {10: b"TEN"})

            frame_masks.write_frames(archive, {10: None})
            self.assertFalse(archive.exists())
            self.assertEqual(frame_masks.frame_indices(archive), [])

    def test_range_digest_changes_only_for_the_edited_range(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "masks.zip"
            frame_masks.write_frames(archive, {5: b"a", 50: b"b"})
            early, late = frame_masks.digest(archive, 0, 25), frame_masks.digest(archive, 25, 75)
            self.assertIsNone(frame_masks.digest(archive, 100, 200))

            frame_masks.write_frames(archive, {50: b"edited"})
            self.assertEqual(frame_masks.digest(archive, 0, 25), early)
            self.assertNotEqual(frame_masks.digest(archive, 25, 75), late)

    def test_a_generated_frame_takes_every_source_frame_it_overlaps(self) -> None:
        self.assertEqual(list(frame_masks.source_frames_for(7, 24.0, 24.0)), [7])
        # 25fps source generated at 24fps: generated frame 23 spans source 23.96-24.99.
        self.assertEqual(list(frame_masks.source_frames_for(23, 24.0, 25.0)), [23, 24])
        # 24fps source generated at 48fps: two generated frames per source frame.
        self.assertEqual(list(frame_masks.source_frames_for(3, 48.0, 24.0)), [1])


class OutpaintFrameMaskTests(unittest.TestCase):
    def test_chunk_mask_video_ors_global_and_frame_patches_and_pads_like_the_chunk(self) -> None:
        import cv2
        import numpy as np

        ffmpeg = outpaint_video.find_ffmpeg()
        width, height = 64, 32
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            folder = Path(temp)
            archive = folder / "frames.zip"
            frame_masks.write_frames(archive, {
                12: mask_png(width, height, (40, 10, 50, 20)),
                18: mask_png(width, height, (5, 5, 10, 10)),   # outside the chunk
            })
            custom = folder / "custom.png"
            cv2.imwrite(str(custom), np.pad(np.full((height, 4), 255, np.uint8), ((0, 0), (0, width - 4))))
            args = outpaint_video.build_parser().parse_args([
                "--source", "input/example.mp4", "--custom-mask", str(custom), "--frame-masks", str(archive),
            ])
            chunk = folder / "prepared_0000.mkv"
            # Frames 10-16 are 6 real frames, padded to LTX's 8n+1 = 9 by repeating the last.
            video = outpaint_video.chunk_frame_mask_video(args, chunk, 10, 16, 24.0, 24.0, width, height)
            frames = decode_gray_video(ffmpeg, video, width, height)

            self.assertEqual(len(frames), 9)
            self.assertTrue(np.all(frames[:, :, :4] == 255), "global mask on every frame")
            # Source frame 12 is chunk frame 2, in LTX latent group 1 (chunk frames 1-8). The
            # whole group carries the patch, so LTX sees one consistent hole across the latent.
            self.assertTrue(np.all(frames[1:9, 10:20, 40:50] == 255), "patch across its latent group")
            self.assertFalse(np.any(frames[0, :, 4:] == 255), "frame 0 is its own latent group")
            self.assertFalse(np.any(frames[:, 5:10, 5:10] == 255), "patch outside the chunk stays out")

            self.assertIsNone(outpaint_video.chunk_frame_mask_video(args, chunk, 20, 30, 24.0, 24.0, width, height))

    def test_latent_groups_match_the_ltx_vae(self) -> None:
        groups = [outpaint_video.ltx_latent_group(index) for index in range(18)]
        self.assertEqual(groups, [0] + [1] * 8 + [2] * 8 + [3])

    def test_chunk_signature_covers_only_the_chunk_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "frames.zip"
            frame_masks.write_frames(archive, {100: b"x"})
            args = outpaint_video.build_parser().parse_args(["--source", "input/example.mp4", "--frame-masks", str(archive)])
            self.assertIsNone(outpaint_video.chunk_frame_mask_digest(args, 0, 50, 24.0, 24.0))
            self.assertIsNotNone(outpaint_video.chunk_frame_mask_digest(args, 90, 110, 24.0, 24.0))

    def test_oumoumad_reads_the_chunk_mask_video_per_frame(self) -> None:
        workflow = json.loads((ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json").read_text(encoding="utf-8-sig"))
        args = outpaint_video.build_parser().parse_args([
            "--source", "input/example.mp4", "--comfy-dir", str(ROOT),
            "--outpaint-lora", outpaint_video.OUMOUMAD_OUTPAINT_LORA, "--dry-run",
        ])
        args.chunk_frame_mask_videos = {"sentinel": (ROOT / "chunk_framemask.mkv", None)}
        with (
            mock.patch.object(outpaint_video, "copy_to_comfy_input", side_effect=["arp_outpaint/prepared.mp4", "arp_outpaint_frame_mask/mask.mkv"]),
            mock.patch.object(outpaint_video, "copy_reference_frame_to_comfy_input", return_value="arp_outpaint/reference.png"),
            mock.patch.object(outpaint_video, "probe_video", return_value={"width": 864, "height": 480, "frames": 25, "fps": 24.0}),
        ):
            prompt = outpaint_video.patch_workflow(
                args, workflow, ROOT / "prepared.mp4", ROOT, "arp_outpaint/test", args.prompt, args.negative_prompt, 42,
            )

        self.assertEqual(prompt["9120"]["class_type"], "LoadVideo")
        self.assertEqual(prompt["9120"]["inputs"]["file"], "arp_outpaint_frame_mask/mask.mkv")
        self.assertEqual(prompt["9124"]["inputs"]["video"], ["9120", 0])
        self.assertEqual(prompt["9121"]["inputs"]["image"], ["9124", 0])
        # Binarised before it cuts the sentinel, so the black stays pure black.
        self.assertEqual(prompt["9125"]["inputs"]["mask"], ["9121", 0])
        self.assertEqual(prompt["9123"]["inputs"]["mask"], ["9125", 0])
        self.assertEqual(prompt["5114"]["inputs"]["image"], ["9123", 0])


def write_video(ffmpeg: str, path: Path, frames, pix_fmt: str, codec: list[str]) -> None:
    import numpy as np

    height, width = frames[0].shape[:2]
    process = subprocess.Popen(
        [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{width}x{height}", "-r", "24", "-i", "-", *codec, str(path)],
        stdin=subprocess.PIPE,
    )
    for frame in frames:
        process.stdin.write(np.ascontiguousarray(frame).tobytes())
    process.stdin.close()
    assert process.wait() == 0


class UnfilledPatchTests(unittest.TestCase):
    """Oumoumad sometimes hands a patch hole back as flat sentinel black."""

    def setUp(self) -> None:
        import numpy as np

        self.ffmpeg = outpaint_video.find_ffmpeg()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.width, self.height = 128, 64
        hole = np.zeros((self.height, self.width), np.uint8)
        hole[20:40, 40:70] = 255
        self.base = np.zeros_like(hole)
        # Nine frames: frame 0 alone, then one 8-frame latent group carrying the hole.
        self.mask = self.root / "mask.mkv"
        write_video(self.ffmpeg, self.mask, [np.zeros_like(hole)] + [hole] * 8, "gray", ["-c:v", "ffv1"])
        self.hole = hole > 0

    def tearDown(self) -> None:
        self.temp.cleanup()

    def render(self, name: str, fill: int, dark_background: bool = False) -> Path:
        import numpy as np

        frame = np.full((self.height, self.width, 3), 30 if dark_background else 150, np.uint8)
        frame[self.hole] = fill
        path = self.root / name
        write_video(self.ffmpeg, path, [frame] * 9, "rgb24", ["-c:v", "ffv1"])
        return path

    def test_black_blob_on_a_bright_scene_is_unfilled_but_a_dark_fill_on_a_dark_scene_is_not(self) -> None:
        blob = self.render("blob.mkv", 4)
        self.assertEqual(outpaint_video.unfilled_patch_frames(blob, self.mask, self.base, self.width, self.height), list(range(1, 9)))
        painted = self.render("painted.mkv", 140)
        self.assertEqual(outpaint_video.unfilled_patch_frames(painted, self.mask, self.base, self.width, self.height), [])
        shadow = self.render("shadow.mkv", 18, dark_background=True)
        self.assertEqual(outpaint_video.unfilled_patch_frames(shadow, self.mask, self.base, self.width, self.height), [])

    def test_retry_splices_only_the_failed_holes(self) -> None:
        first = self.render("first.mkv", 4)
        retry_output = self.render("retry.mkv", 140)
        args = outpaint_video.build_parser().parse_args([
            "--source", "input/example.mp4", "--intermediate-profile", "lossless",
            "--outpaint-lora", outpaint_video.OUMOUMAD_OUTPAINT_LORA,
        ])
        seeds = []

        def render(seed: int) -> Path:
            seeds.append(seed)
            return retry_output

        args.frame_patch_retries = 2
        result = outpaint_video.retry_unfilled_patches(args, first, (self.mask, self.base), 0, 100, self.width, self.height, render)
        frames = decode_gray_video(self.ffmpeg, result, self.width, self.height)

        self.assertEqual(seeds, [100 + 7919])  # one retry was enough
        self.assertGreater(int(frames[4][30, 55]), 100, "hole now painted")
        self.assertEqual(outpaint_video.unfilled_patch_frames(result, self.mask, self.base, self.width, self.height), [])

    def test_recomposition_skips_patches_the_report_says_are_unfilled(self) -> None:
        archive = self.root / "frames.zip"
        frame_masks.write_frames(archive, {3: b"a", 4: b"b"})
        outpainted = self.root / "out.mkv"
        report = final_composite.frame_patch_report(outpainted)
        report.write_text(json.dumps({"frame_masks_digest": frame_masks.digest(archive), "unfilled_frames": [4]}), encoding="utf-8")
        self.assertEqual(final_composite.unfilled_patch_frames(outpainted, archive), {4})
        # Once the patches are edited, the report no longer describes them: skip nothing.
        frame_masks.write_frames(archive, {4: b"edited"})
        self.assertEqual(final_composite.unfilled_patch_frames(outpainted, archive), set())


class OfficialFrameMaskTests(unittest.TestCase):
    """The official 2.3 / 2.5 graph takes explicit masks, so patches go in per frame there too."""

    def patched_prompt(self, extra_args: list[str], videos: dict, copies: list[str]) -> dict:
        workflow = json.loads((ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json").read_text(encoding="utf-8-sig"))
        args = outpaint_video.build_parser().parse_args(["--source", "input/example.mp4", "--comfy-dir", str(ROOT), "--dry-run", *extra_args])
        args.chunk_frame_mask_videos = videos
        with (
            mock.patch.object(outpaint_video, "copy_to_comfy_input", side_effect=copies),
            mock.patch.object(outpaint_video, "copy_reference_frame_to_comfy_input", return_value="arp_outpaint/reference.png"),
            mock.patch.object(outpaint_video, "probe_video", return_value={"width": 864, "height": 480, "frames": 25, "fps": 24.0}),
        ):
            return outpaint_video.patch_workflow(args, workflow, ROOT / "prepared.mp4", ROOT, "arp_outpaint/test", args.prompt, args.negative_prompt, 42)

    def test_exact_and_generation_masks_are_loaded_per_frame(self) -> None:
        prompt = self.patched_prompt(
            [], {"exact": (ROOT / "exact.mkv", None), "generation": (ROOT / "generation.mkv", None)},
            ["arp_outpaint/prepared.mp4", "arp_outpaint_frame_mask/exact.mkv", "arp_outpaint_frame_mask/generation.mkv"],
        )
        self.assertEqual(prompt["9100"]["class_type"], "LoadVideo")
        self.assertEqual(prompt["9100"]["inputs"]["file"], "arp_outpaint_frame_mask/exact.mkv")
        self.assertEqual(prompt["9103"]["inputs"]["file"], "arp_outpaint_frame_mask/generation.mkv")
        # The final blend reveals by the exact mask; LTX paints by the generation mask.
        # Both arrive binarised, so no trace of the picture stays inside a hole.
        self.assertEqual(prompt["5266"]["inputs"]["mask"], ["9106", 0])
        self.assertEqual(prompt["5358"]["inputs"]["mask"], ["9108", 0])
        self.assertEqual(prompt["9106"]["class_type"], "ThresholdMask")
        self.assertEqual(prompt["9106"]["inputs"]["mask"], ["9101", 0])
        self.assertEqual(prompt["9101"]["inputs"]["image"], ["9105", 0])
        self.assertEqual(prompt["9105"]["inputs"]["video"], ["9100", 0])
        self.assertEqual(prompt["9108"]["inputs"]["mask"], ["9104", 0])

    def test_black_region_mode_adds_the_per_frame_custom_mask(self) -> None:
        prompt = self.patched_prompt(
            ["--outpaint-all-black-regions", "--generation-mask-overlap", "0"], {"sentinel": (ROOT / "sentinel.mkv", None)},
            ["arp_outpaint/prepared.mp4", "arp_outpaint_frame_mask/sentinel.mkv"],
        )
        self.assertEqual(prompt["9120"]["class_type"], "LoadVideo")
        self.assertEqual(prompt["9122"]["inputs"]["destination"], ["9102", 0])
        self.assertEqual(prompt["9122"]["inputs"]["source"], ["9125", 0])
        self.assertEqual(prompt["5266"]["inputs"]["mask"], ["9122", 0])

    def test_green_sentinel_left_in_a_hole_counts_as_unfilled(self) -> None:
        import numpy as np

        ffmpeg = outpaint_video.find_ffmpeg()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hole = np.zeros((64, 128), np.uint8)
            hole[20:40, 40:70] = 255
            mask = root / "mask.mkv"
            write_video(ffmpeg, mask, [hole] * 9, "gray", ["-c:v", "ffv1"])
            base = np.zeros_like(hole)
            for name, fill, expected in (("green.mkv", (102, 255, 0), 9), ("grass.mkv", (70, 110, 50), 0)):
                frame = np.full((64, 128, 3), 120, np.uint8)
                frame[hole > 0] = fill
                video = root / name
                write_video(ffmpeg, video, [frame] * 9, "rgb24", ["-c:v", "ffv1"])
                failed = outpaint_video.unfilled_patch_frames(video, mask, base, 128, 64, "green")
                self.assertEqual(len(failed), expected, name)


class RecompFrameMaskTests(unittest.TestCase):
    def test_keep_video_feathers_each_patch_on_its_own_frame_then_falls_back_to_global(self) -> None:
        import numpy as np

        ffmpeg = final_composite.find_ffmpeg(None)
        width, height = 80, 40
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "frames.zip"
            frame_masks.write_frames(archive, {2: mask_png(width, height, (30, 10, 40, 20))})
            with mock.patch.object(final_composite, "ROOT", root):
                video = final_composite.feathered_frame_keep_video(archive, None, (width, height), 5, 24.0, ffmpeg)
            frames = decode_gray_video(ffmpeg, video, width, height)

        # Frames 0-3: unpatched, patched, and the trailing global-only frame that frame sync repeats.
        self.assertEqual(len(frames), 4)
        self.assertTrue(np.all(frames[[0, 1, 3]] == 255))
        self.assertTrue(np.all(frames[2, 10:20, 30:40] == 0))
        self.assertTrue(0 < frames[2, 15, 42] < 255, "ramp outside the patch")
        self.assertEqual(frames[2, 15, 50], 255)

    def test_signature_leaves_frame_masks_out_unless_given(self) -> None:
        base = ["--outpainted", "o.mp4", "--source", "s.mp4", "--output", "out.mp4"]
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "frames.zip"
            frame_masks.write_frames(archive, {0: b"x"})
            with mock.patch.object(final_composite, "file_fingerprint", return_value={}):
                plain = final_composite.signature(final_composite.build_parser().parse_args(base))
                masked = final_composite.signature(final_composite.build_parser().parse_args(base + ["--frame-masks", str(archive)]))
        self.assertNotIn("frame_masks", plain)
        self.assertIn("frame_masks_digest", masked)

    def test_composite_reveals_the_outpaint_only_on_the_patched_frame(self) -> None:
        import cv2
        import numpy as np

        ffmpeg = final_composite.find_ffmpeg(None)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outpainted, source, output = root / "out.mp4", root / "src.mp4", root / "comp.mp4"
            for path, colour, size in ((outpainted, "black", "320x180"), (source, "white", "240x180")):
                subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={colour}:s={size}:r=24:d=0.5",
                                "-c:v", "libx264", "-pix_fmt", "yuv444p", "-qp", "0", str(path)], check=True)
            archive = root / "frames.zip"
            frame_masks.write_frames(archive, {5: mask_png(320, 180, (150, 60, 170, 120))})
            args = final_composite.build_parser().parse_args([
                "--outpainted", str(outpainted), "--source", str(source), "--output", str(output),
                "--frame-masks", str(archive), "--feather-pixels", "10", "--crf", "0", "--force",
            ])
            with mock.patch.object(final_composite, "ROOT", root), mock.patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(final_composite.run(args), 0)
            frames = decode_gray_video(ffmpeg, output, 320, 180)

        self.assertEqual(len(frames), 12)
        self.assertLess(int(frames[5, 90, 160]), 20, "patched frame shows the outpaint")
        for frame in (0, 4, 6, 11):
            self.assertGreater(int(frames[frame, 90, 160]), 235, f"frame {frame} keeps the source")


if __name__ == "__main__":
    unittest.main()
