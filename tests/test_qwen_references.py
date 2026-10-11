from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import qwen_colorize_references as qwen  # noqa: E402

WORKFLOW = ROOT / "workflows" / "qwen_image_edit" / "Image Edit (Qwen 2511).json"


class Qwen2511ReferenceGraphTests(unittest.TestCase):
    def build(self, reference_names: list[str]) -> dict:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "grey.png"
            source.write_bytes(b"png")
            args = qwen.build_parser().parse_args(["--manifest", "x.csv", "--workflow", str(WORKFLOW), "--comfy-dir", temp])
            args.active_seed = 5
            args.active_reference_names = reference_names
            workflow = qwen.load_workflow(WORKFLOW)
            qwen.patch_qwen_model_backend(args, workflow)
            return qwen.patch_workflow(args, workflow, source, Path(temp) / "out.png", "colour it")

    def test_without_references_the_graph_is_unchanged(self) -> None:
        prompt = self.build([])
        encoders = [node for node in prompt.values() if node["class_type"] == "TextEncodeQwenImageEditPlus"]
        self.assertTrue(all("vae" in node["inputs"] and "image2" not in node["inputs"] for node in encoders))
        self.assertFalse(any(node["class_type"] == "ReferenceLatent" for node in prompt.values()))

    def test_references_are_seen_but_only_the_grey_frame_is_a_reference_picture(self) -> None:
        prompt = self.build(["arp_qwen_ref_images/a.png", "arp_qwen_ref_images/b.png"])
        encoders = {key: node for key, node in prompt.items() if node["class_type"] == "TextEncodeQwenImageEditPlus"}
        references = {key: node for key, node in prompt.items() if node["class_type"] == "ReferenceLatent"}
        loads = {key: node["inputs"]["image"] for key, node in prompt.items() if node["class_type"] == "LoadImage"}
        encode = next(key for key, node in prompt.items() if node["class_type"] == "VAEEncode")
        for node in encoders.values():
            self.assertNotIn("vae", node["inputs"])  # every image is vision tokens only
            self.assertEqual(loads[node["inputs"]["image2"][0]], "arp_qwen_ref_images/a.png")
            self.assertEqual(loads[node["inputs"]["image3"][0]], "arp_qwen_ref_images/b.png")
        self.assertEqual(len(references), len(encoders))
        for node in references.values():
            self.assertIn(node["inputs"]["conditioning"][0], encoders)
            self.assertEqual(node["inputs"]["latent"][0], encode)  # the grey frame's own latent
        # The sampler's conditioning now passes through the ReferenceLatent nodes.
        consumers = [node for node in prompt.values() if node["class_type"] == "FluxKontextMultiReferenceLatentMethod"]
        self.assertTrue(all(node["inputs"]["conditioning"][0] in references for node in consumers))


if __name__ == "__main__":
    unittest.main()
