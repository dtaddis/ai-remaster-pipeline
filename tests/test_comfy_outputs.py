from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import comfy_api  # noqa: E402
import common  # noqa: E402
from ai_remaster_gui.config import comfy_output_root_for  # noqa: E402


class ComfyOutputTests(unittest.TestCase):
    def test_gui_comfy_output_root_follows_active_comfy_dir(self) -> None:
        config = {"comfy_dir": r"D:\dtaddis\ai-remaster-pipeline\tools\comfyui"}

        self.assertEqual(
            comfy_output_root_for(config),
            r"D:\dtaddis\ai-remaster-pipeline\tools\comfyui\output",
        )

    def test_extractor_keeps_reported_paths_before_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            root = Path(tmp_text)
            history = {
                "outputs": {
                    "14": {
                        "images": [
                            {
                                "filename": "guide_00001_.png",
                                "subfolder": "ai_remaster_qwen_edits",
                                "type": "output",
                            }
                        ]
                    }
                }
            }

            files = comfy_api.extract_output_files(history, root / "output")

        self.assertEqual(files, [root / "output" / "ai_remaster_qwen_edits" / "guide_00001_.png"])

    def test_newest_output_waits_for_reported_file_to_appear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            produced = Path(tmp_text) / "guide.png"
            sleeps = {"count": 0}

            def fake_sleep(_delay: float) -> None:
                sleeps["count"] += 1
                produced.write_bytes(b"done")

            with mock.patch.object(common.time, "sleep", side_effect=fake_sleep):
                found = common.newest_output([produced], exist_attempts=2, exist_delay=0.01)

            self.assertEqual(found, produced)
            self.assertEqual(sleeps["count"], 1)

    def test_missing_live_node_mentions_import_failure_when_package_defines_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            comfy = Path(tmp_text) / "comfy"
            package = comfy / "custom_nodes" / "ComfyUI-LTXVideo"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text('NODE_CLASS_MAPPINGS = {"LTXAddVideoICLoRAGuide": object}\n', encoding="utf-8")

            with mock.patch.object(comfy_api, "object_info", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "failed to import"):
                    comfy_api.ensure_node_types(
                        "http://127.0.0.1:8188",
                        {"LTXAddVideoICLoRAGuide": "ComfyUI-LTXVideo"},
                        "outpainting workflow",
                        comfy,
                    )

    def test_arp_ltx_compatibility_returns_adapter_report(self) -> None:
        report = {"compatible": True, "adapter": "model_scoped_ltx_guide_v1"}
        with mock.patch.object(comfy_api, "http_json", return_value=report):
            self.assertEqual(
                comfy_api.ensure_arp_ltx_compatible("http://127.0.0.1:8797"),
                report,
            )

    def test_arp_ltx_compatibility_propagates_actionable_failure(self) -> None:
        report = {"compatible": False, "adapter": "none", "error": "signature changed"}
        with mock.patch.object(comfy_api, "http_json", return_value=report):
            with self.assertRaisesRegex(RuntimeError, "signature changed"):
                comfy_api.ensure_arp_ltx_compatible("http://127.0.0.1:8797")


if __name__ == "__main__":
    unittest.main()
