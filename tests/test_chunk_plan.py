from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import chunk_plan  # noqa: E402


def rows_for(ranges, per_index=None):
    """Manifest rows as the planner would have written them for these ranges."""
    rows = {}
    for index, start, end in ranges:
        row = {"chunk_index": str(index), "start_frame": str(start), "end_frame": str(end), "seed": str(42 + index), "auto_start_guide": "true"}
        row.update((per_index or {}).get(index, {}))
        rows[index] = row
    return rows


def absolute_guides(rows, ranges):
    found = []
    for index, start, _end in ranges:
        for guide in json.loads(rows[index].get("guide_frames") or "[]"):
            found.append((start + int(guide["frame_idx"]), guide["image"]))
    return sorted(found)


class ChunkRangeTests(unittest.TestCase):
    def test_ltx_chunks_are_8n_plus_1_frames(self) -> None:
        ranges = chunk_plan.chunk_ranges(1000, 24.0, 20.0, 8, {})

        self.assertEqual(ranges[:2], [(0, 0, 481), (1, 473, 954)])
        self.assertTrue(all((end - start) % 8 == 1 for _index, start, end in ranges))

    def test_a_cap_limits_every_chunk_including_custom_and_whole_clip_lengths(self) -> None:
        capped = chunk_plan.chunk_ranges(1000, 24.0, 20.0, 13, {1: {"custom_seconds": "12"}}, max_frames=161)
        whole = chunk_plan.chunk_ranges(400, 24.0, 0.0, 13, {}, max_frames=161)

        self.assertEqual(capped[:3], [(0, 0, 161), (1, 148, 309), (2, 296, 457)])
        self.assertTrue(all(end - start <= 161 for _index, start, end in capped))
        self.assertEqual(capped[-1][2], 1000)
        self.assertEqual(len(whole), 3)
        self.assertEqual(chunk_plan.chunk_ranges(400, 24.0, 0.0, 13, {}), [(0, 0, 400)])

    def test_a_cap_that_is_not_8n_plus_1_rounds_down_so_the_final_chunk_fits(self) -> None:
        ranges = chunk_plan.chunk_ranges(1000, 24.0, 20.0, 13, {}, max_frames=117)

        self.assertTrue(all(end - start <= 113 for _index, start, end in ranges))


class RemapChunkRowsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ltx = chunk_plan.chunk_ranges(1000, 24.0, 20.0, 13, {})
        self.wan = chunk_plan.chunk_ranges(1000, 24.0, 20.0, 13, {}, max_frames=161)

    def test_an_unchanged_layout_is_returned_untouched(self) -> None:
        rows = rows_for(self.ltx)

        self.assertIs(chunk_plan.remap_chunk_rows(rows, self.ltx), rows)

    def test_switching_to_wan_keeps_every_guide_on_its_frame(self) -> None:
        rows = rows_for(
            self.ltx,
            {
                0: {"guide_frames": json.dumps([{"frame_idx": 300, "image": "a.png", "strength": 1.0}])},
                1: {"guide_frames": json.dumps([{"frame_idx": -1, "image": "end.png", "strength": 1.0}]), "guide_image": "start.png"},
            },
        )
        before = sorted([(300, "a.png"), (self.ltx[1][2] - 1, "end.png")])

        remapped = chunk_plan.remap_chunk_rows(rows, self.wan)

        self.assertEqual(absolute_guides(remapped, self.wan), before)
        self.assertEqual(remapped[2]["guide_image"], "")
        # Switching back to LTX keeps them there too.
        synced = {index: dict(remapped[index], start_frame=str(start), end_frame=str(end)) for index, start, end in self.wan}
        self.assertEqual(absolute_guides(chunk_plan.remap_chunk_rows(synced, self.ltx), self.ltx), before)

    def test_a_guide_in_an_overlap_goes_to_the_chunk_that_generates_it(self) -> None:
        # Frame 150 lies in both Wan chunk 0 (0-160) and chunk 1 (148-308). Chunk 1 opens with
        # chunk 0's finished frames there, so only chunk 0 would actually use the guide.
        rows = rows_for(self.ltx, {0: {"guide_frames": json.dumps([{"frame_idx": 150, "image": "g.png"}])}})

        remapped = chunk_plan.remap_chunk_rows(rows, self.wan)

        self.assertEqual(json.loads(remapped[0]["guide_frames"]), [{"frame_idx": 150, "image": "g.png"}])
        self.assertEqual(remapped[1]["guide_frames"], "")

    def test_span_settings_spread_but_seed_and_start_guide_stay_with_one_chunk(self) -> None:
        rows = rows_for(self.ltx, {0: {"prompt_suffix": "a sunny street", "offset_x": "12", "seed": "999", "auto_start_guide": "false"}})

        remapped = chunk_plan.remap_chunk_rows(rows, self.wan)

        inside_first = [index for index, start, end in self.wan if end <= self.ltx[0][2]]
        self.assertGreater(len(inside_first), 1)
        self.assertTrue(all(remapped[i]["prompt_suffix"] == "a sunny street" for i in inside_first))
        self.assertTrue(all(remapped[i]["offset_x"] == "12" for i in inside_first))
        self.assertEqual(remapped[0]["seed"], "999")
        self.assertEqual(remapped[0]["auto_start_guide"], "false")
        # Later pieces of the same stretch fall back to the defaults.
        self.assertNotIn("seed", remapped[1])
        self.assertNotIn("auto_start_guide", remapped[1])

    def test_custom_lengths_stay_with_their_chunk_number(self) -> None:
        rows = rows_for(self.ltx, {1: {"custom_seconds": "12"}})

        remapped = chunk_plan.remap_chunk_rows(rows, self.wan)

        self.assertEqual(remapped[1]["custom_seconds"], "12")
        self.assertEqual(remapped[2]["custom_seconds"], "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
