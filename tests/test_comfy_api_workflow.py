from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from comfy_api import workflow_to_prompt  # noqa: E402


class WorkflowToPromptTests(unittest.TestCase):
    def test_flattens_frontend_subgraph_instance_and_exposed_widget(self) -> None:
        workflow = {
            "nodes": [
                {
                    "id": 1,
                    "type": "subgraph-id",
                    "inputs": [{"name": "amount", "widget": {"name": "amount"}}],
                    "outputs": [{"name": "value", "links": [10]}],
                    "widgets_values": [7],
                },
                {"id": 2, "type": "Sink", "inputs": [{"name": "value", "link": 10}]},
            ],
            "links": [[10, 1, 0, 2, 0, "INT"]],
            "definitions": {
                "subgraphs": [{
                    "id": "subgraph-id",
                    "name": "Number source",
                    "inputs": [{"name": "amount"}],
                    "outputs": [{"name": "value"}],
                    "nodes": [{
                        "id": 5,
                        "type": "PrimitiveInt",
                        "inputs": [{"name": "value", "link": 20, "widget": {"name": "value"}}],
                        "widgets_values": [1],
                    }],
                    "links": [
                        {"id": 20, "origin_id": -10, "origin_slot": 0, "target_id": 5, "target_slot": 0, "type": "INT"},
                        {"id": 21, "origin_id": 5, "origin_slot": 0, "target_id": -20, "target_slot": 0, "type": "INT"},
                    ],
                }],
            },
        }

        prompt = workflow_to_prompt(workflow, "2")

        source_id = next(node_id for node_id, node in prompt.items() if node["class_type"] == "PrimitiveInt")
        self.assertEqual(prompt[source_id]["inputs"]["value"], 7)
        self.assertEqual(prompt["2"]["inputs"]["value"], [source_id, 0])

    def test_core_load_video_uses_file_widget_name(self) -> None:
        workflow = {
            "nodes": [{"id": 1, "type": "LoadVideo", "inputs": [], "widgets_values": ["clip.mp4", "image"]}],
            "links": [],
        }

        prompt = workflow_to_prompt(workflow, "1")

        self.assertEqual(prompt["1"]["inputs"], {"file": "clip.mp4"})

    def test_resize_longer_dimension_emits_dynamic_combo_value(self) -> None:
        workflow = {
            "nodes": [{"id": 1, "type": "ResizeImageMaskNode", "inputs": [], "widgets_values": ["scale longer dimension", 1024, "lanczos"]}],
            "links": [],
        }

        prompt = workflow_to_prompt(workflow, "1")

        self.assertEqual(
            prompt["1"]["inputs"],
            {"resize_type": "scale longer dimension", "resize_type.longer_size": 1024, "scale_method": "lanczos"},
        )

    def test_collapses_frontend_reroute_nodes(self) -> None:
        workflow = {
            "nodes": [
                {"id": 1, "type": "Source", "inputs": []},
                {"id": 2, "type": "Reroute", "inputs": [{"name": "", "type": "*", "link": 10}]},
                {"id": 3, "type": "Sink", "inputs": [{"name": "audio", "type": "AUDIO", "link": 11}]},
            ],
            "links": [
                [10, 1, 2, 2, 0, "AUDIO"],
                [11, 2, 0, 3, 0, "AUDIO"],
            ],
        }

        prompt = workflow_to_prompt(workflow, "3")

        self.assertEqual(set(prompt), {"1", "3"})
        self.assertEqual(prompt["3"]["inputs"]["audio"], ["1", 2])

    def test_rejects_reroute_cycles(self) -> None:
        workflow = {
            "nodes": [
                {"id": 1, "type": "Reroute", "inputs": [{"name": "", "link": 10}]},
                {"id": 2, "type": "Reroute", "inputs": [{"name": "", "link": 11}]},
                {"id": 3, "type": "Sink", "inputs": [{"name": "value", "link": 12}]},
            ],
            "links": [
                [10, 2, 0, 1, 0, "*"],
                [11, 1, 0, 2, 0, "*"],
                [12, 1, 0, 3, 0, "*"],
            ],
        }

        with self.assertRaisesRegex(ValueError, "Reroute cycle"):
            workflow_to_prompt(workflow, "3")


if __name__ == "__main__":
    unittest.main()
