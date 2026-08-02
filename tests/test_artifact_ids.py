from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import artifact_ids as aid  # noqa: E402


class SizeMathTests(unittest.TestCase):
    def test_source_height_matches_old_values(self) -> None:
        # 854x480 source at "source" height -> delivery 854x480, work (model-safe) 864x480.
        self.assertEqual(aid.delivery_size(480, "16:9", "source"), (854, 480))
        self.assertEqual(aid.work_size(480, "16:9", "source"), (864, 480))

    def test_fixed_720_matches_existing_files(self) -> None:
        # The pre-existing renders are named 1280x704: delivery 1280x720 -> work 1280x704.
        self.assertEqual(aid.delivery_size(480, "16:9", "720"), (1280, 720))
        self.assertEqual(aid.work_size(480, "16:9", "720"), (1280, 704))

    def test_model_safe_rounds_to_32(self) -> None:
        self.assertEqual(aid.model_safe(720), 704)
        self.assertEqual(aid.model_safe(854), 864)
        self.assertEqual(aid.model_safe(1080), 1088)


class IdentityKeyTests(unittest.TestCase):
    def base(self, **over):
        params = dict(source_name="Metropolis_sec.mp4", aspect="16:9", work_w=864, work_h=480, crop=[108, 108, 0, 0], black=False)
        params.update(over)
        return aid.outpaint_identity(**params)

    def test_key_is_stable_and_short(self) -> None:
        k = aid.artifact_key(self.base())
        self.assertEqual(k, aid.artifact_key(self.base()))
        self.assertEqual(len(k), aid.KEY_LEN)

    def test_key_distinguishes_variants(self) -> None:
        keys = {
            aid.artifact_key(self.base()),
            aid.artifact_key(self.base(work_h=704, work_w=1280)),  # different size
            aid.artifact_key(self.base(crop=[107, 107, 0, 0])),     # different crop
            aid.artifact_key(self.base(black=True)),                # all-black mode
            aid.artifact_key(self.base(aspect="4:3")),              # different aspect
            aid.artifact_key(self.base(source_name="Other.mp4")),   # different source
        }
        self.assertEqual(len(keys), 6, "each distinguishing param must change the key")

    def test_name_shape(self) -> None:
        name = aid.artifact_name("Metropolis", "outpaint", self.base(), "mp4")
        self.assertTrue(name.startswith("Metropolis_outpaint_"))
        self.assertTrue(name.endswith(".mp4"))

    def test_cropped_outputs_use_regeneration_identity(self) -> None:
        self.assertEqual(self.base()["crop_fill"], 2)
        self.assertNotIn("crop_fill", self.base(crop=[0, 0, 0, 0]))


class FindTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="arp_aid_")
        self.dir = Path(self._tmp.name)
        self.identity = aid.outpaint_identity("Metropolis_sec.mp4", "16:9", 864, 480, [108, 108, 0, 0], False)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_finds_by_deterministic_name(self) -> None:
        path = aid.artifact_path(self.dir, "Metropolis", "outpaint", self.identity, "mp4")
        path.write_bytes(b"x")
        aid.write_identity(path, self.identity, label="test")
        self.assertEqual(aid.find(self.dir, "Metropolis", "outpaint", self.identity, "mp4"), path)

    def test_finds_by_signature_when_name_differs(self) -> None:
        # A file whose on-disk name does NOT match the computed key, but whose sidecar identity does.
        legacy = self.dir / "legacy_name.mp4"
        legacy.write_bytes(b"x")
        aid.write_identity(legacy, self.identity, label="legacy")
        self.assertEqual(aid.find(self.dir, "Metropolis", "outpaint", self.identity, "mp4"), legacy)

    def test_returns_none_when_absent(self) -> None:
        self.assertIsNone(aid.find(self.dir, "Metropolis", "outpaint", self.identity, "mp4"))

    def test_different_identity_not_matched(self) -> None:
        path = aid.artifact_path(self.dir, "Metropolis", "outpaint", self.identity, "mp4")
        path.write_bytes(b"x")
        aid.write_identity(path, self.identity)
        other = aid.outpaint_identity("Metropolis_sec.mp4", "16:9", 1280, 704, [108, 108, 0, 0], False)
        self.assertIsNone(aid.find(self.dir, "Metropolis", "outpaint", other, "mp4"))


class PathLengthTests(unittest.TestCase):
    def test_guide_edit_path_well_under_max_path(self) -> None:
        identity = aid.outpaint_identity("The_Hound_of_the_Baskervilles__0001195926_0001236260.mp4", "16:9", 1280, 704, [108, 108, 0, 0], False)
        word = aid.source_word("The Hound of the Baskervilles (1939).mp4")
        guide_dir = f"{word}_guides_{aid.artifact_key(identity)}"
        rel = f"intermediate/outpaint_guides/{guide_dir}/g00/edit_01.png.sig.json"
        # Even with a long absolute root this must stay clear of the 260-char Windows limit.
        self.assertLess(len(rel), 130, rel)


if __name__ == "__main__":
    unittest.main(verbosity=2)
