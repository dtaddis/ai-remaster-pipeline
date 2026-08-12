from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from ltx_outpaint_workflow_adapter import (  # noqa: E402
    ADAPTER_ID,
    is_official_outpaint_template,
    validate_official_outpaint_workflow,
)


class LTXOutpaintWorkflowAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        path = ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json"
        self.workflow = json.loads(path.read_text(encoding="utf-8-sig"))

    def test_bundled_workflow_matches_explicit_contract(self) -> None:
        self.assertTrue(is_official_outpaint_template(self.workflow))
        self.assertEqual(validate_official_outpaint_workflow(self.workflow), ADAPTER_ID)

    def test_changed_required_node_fails_before_rewiring(self) -> None:
        node = next(node for node in self.workflow["nodes"] if node["id"] == 5358)
        node["type"] = "FutureLTXInpaintNode"
        with self.assertRaisesRegex(RuntimeError, "node 5358"):
            validate_official_outpaint_workflow(self.workflow)


if __name__ == "__main__":
    unittest.main()
