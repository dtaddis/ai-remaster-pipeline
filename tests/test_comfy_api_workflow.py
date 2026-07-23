from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from comfy_api import workflow_to_prompt  # noqa: E402


class WorkflowToPromptTests(unittest.TestCase):
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
