"""Script-level tests for outpaint guide frame handling.

Covers guide selection (silent-skip paths must log), coordinate resolution
(negative indices, IC-LoRA coordinate collisions, duplicates), and end-to-end
workflow patching with out-of-order guide lists.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import outpaint_video as ov  # noqa: E402
from comfy_api import workflow_to_prompt  # noqa: E402

WORKFLOW = ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json"


def coords(guides: list[dict], frames: int) -> list[int]:
    with contextlib.redirect_stdout(io.StringIO()):
        return [g["frame_idx"] for g in ov.resolve_guide_coords(guides, frames)]


class ResolveGuideCoordsTests(unittest.TestCase):
    def test_out_of_order_positions_are_kept(self) -> None:
        guides = [{"frame_idx": 200, "image": "a"}, {"frame_idx": 56, "image": "b"}, {"frame_idx": 120, "image": "c"}]
        self.assertEqual(coords(guides, 481), [200, 56, 120])

    def test_negative_index_resolves_to_last_frame(self) -> None:
        # 481 frames -> 61 latent frames -> -1 lands on frame 480, matching the Comfy
        # node's own resolution when its keyframe accounting is intact.
        self.assertEqual(coords([{"frame_idx": -1, "image": "a"}], 481), [480])
        # Non-8k+1 chunk: 480 frames -> 60 latent frames -> -1 lands on 472.
        self.assertEqual(coords([{"frame_idx": -1, "image": "a"}], 480), [472])

    def test_ic_lora_colliding_coordinates_shift_down(self) -> None:
        # Coordinates of 1 above a multiple of 8 collide with the IC-LoRA reference
        # video's internal start coordinates {0} u {8m+1} and shift down by 1.
        self.assertEqual(coords([{"frame_idx": 9, "image": "a"}, {"frame_idx": 17, "image": "b"}], 481), [8, 16])

    def test_duplicate_coordinates_are_dropped(self) -> None:
        self.assertEqual(coords([{"frame_idx": 16, "image": "a"}, {"frame_idx": 16, "image": "b"}], 481), [16])
        # 9 shifts to 8 and then collides with the explicit 8.
        self.assertEqual(coords([{"frame_idx": 8, "image": "a"}, {"frame_idx": 9, "image": "b"}], 481), [8])

    def test_coordinate_zero_is_skipped(self) -> None:
        # -9 on a 9-frame chunk resolves to frame 0, which belongs to the i2v start guide.
        self.assertEqual(coords([{"frame_idx": -9, "image": "a"}], 9), [])

    def test_negative_index_passes_through_when_frame_count_unknown(self) -> None:
        self.assertEqual(coords([{"frame_idx": -1, "image": "a"}], 0), [-1])


class SelectChunkGuidesTests(unittest.TestCase):
    def select(self, guides: list[dict], chunk_frames: int = 481):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            explicit, strength, extra = ov.select_chunk_guides(guides, 0, chunk_frames)
        return explicit, strength, extra, buffer.getvalue()

    def test_skips_empty_and_missing_images_with_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.png"
            good.write_bytes(b"x")
            guides = [
                {"frame_idx": 8, "strength": 0.5, "image": ""},
                {"frame_idx": 16, "strength": 0.5, "image": str(Path(tmp) / "missing.png")},
                {"frame_idx": 24, "strength": 0.5, "image": str(good)},
            ]
            explicit, strength, extra, out = self.select(guides)
        self.assertIsNone(explicit)
        self.assertIsNone(strength)
        self.assertEqual([g["frame_idx"] for g in extra], [24])
        self.assertIn("has no image set", out)
        self.assertIn("not found", out)

    def test_skips_out_of_range_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "g.png"
            img.write_bytes(b"x")
            guides = [
                {"frame_idx": 500, "strength": 0.5, "image": str(img)},
                {"frame_idx": -500, "strength": 0.5, "image": str(img)},
                {"frame_idx": 480, "strength": 0.5, "image": str(img)},
            ]
            _, _, extra, out = self.select(guides)
        self.assertEqual([g["frame_idx"] for g in extra], [480])
        self.assertEqual(out.count("outside the chunk"), 2)

    def test_keeps_first_frame_zero_guide(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.png"
            second = Path(tmp) / "second.png"
            first.write_bytes(b"x")
            second.write_bytes(b"x")
            guides = [
                {"frame_idx": 0, "strength": 0.4, "image": str(first)},
                {"frame_idx": 0, "strength": 0.9, "image": str(second)},
            ]
            explicit, strength, extra, out = self.select(guides)
            self.assertEqual(explicit, first)
        self.assertEqual(strength, 0.4)
        self.assertEqual(extra, [])
        self.assertIn("more than one frame-0 guide", out)


class PatchExtraGuidesTests(unittest.TestCase):
    def patch_and_convert(self, extra_guides: list[dict], frames: int = 481) -> dict:
        workflow = json.loads(WORKFLOW.read_text(encoding="utf-8-sig"))
        args = SimpleNamespace(comfy_dir=str(ROOT))
        stub = lambda guide, comfy_dir, w, h, source_frame=None: f"stub_{Path(str(guide)).stem}.png"  # noqa: E731
        with mock.patch.object(ov, "copy_guide_image_to_comfy_input", stub), contextlib.redirect_stdout(io.StringIO()):
            ov._patch_extra_guides(workflow, args, extra_guides, 1280, 704, None, frames)
        return workflow_to_prompt(workflow, "5228")

    def consumers_of(self, prompt: dict, node_id: str, exclude: set[str] = frozenset()) -> set[str]:
        return {
            node["class_type"]
            for nid, node in prompt.items()
            if nid not in exclude
            for val in node["inputs"].values()
            if isinstance(val, list) and len(val) == 2 and str(val[0]) == node_id
        }

    def test_out_of_order_guides_all_reach_the_prompt(self) -> None:
        extra = [
            {"frame_idx": 200, "strength": 0.9, "image": "imgA.png"},
            {"frame_idx": 56, "strength": 0.8, "image": "imgB.png"},
            {"frame_idx": -1, "strength": 1.0, "image": "imgC.png"},
            {"frame_idx": 120, "strength": 0.7, "image": "imgD.png"},
        ]
        prompt = self.patch_and_convert(extra)
        nodes = {nid: n for nid, n in prompt.items() if n["class_type"] == "LTXVAddGuideAdvanced"}
        self.assertEqual(sorted(nodes), ["9060", "9061", "9062", "9063"])
        self.assertEqual([nodes[nid]["inputs"]["frame_idx"] for nid in sorted(nodes)], [200, 56, 480, 120])
        # Chain: official IC guide -> 9060 -> 9061 -> 9062 -> 9063
        self.assertEqual(nodes["9060"]["inputs"]["positive"][0], "5114")
        self.assertEqual(nodes["9061"]["inputs"]["positive"][0], "9060")
        self.assertEqual(nodes["9062"]["inputs"]["positive"][0], "9061")
        self.assertEqual(nodes["9063"]["inputs"]["positive"][0], "9062")
        # Downstream consumers (sampler/crop) must take conditioning from the last guide node.
        consumers = self.consumers_of(prompt, "9063", exclude=set(nodes))
        self.assertIn("CFGGuider", consumers)
        self.assertIn("LTXVCropGuides", consumers)

    def test_all_guides_skipped_leaves_workflow_untouched(self) -> None:
        # A lone guide resolving to frame 0 is dropped entirely; the prompt must keep
        # The official IC guide's direct wiring instead of dangling guide nodes.
        prompt = self.patch_and_convert([{"frame_idx": -9, "strength": 1.0, "image": "imgA.png"}], frames=9)
        self.assertFalse(any(n["class_type"] == "LTXVAddGuideAdvanced" for n in prompt.values()))
        self.assertIn("CFGGuider", self.consumers_of(prompt, "5114"))


if __name__ == "__main__":
    unittest.main()
