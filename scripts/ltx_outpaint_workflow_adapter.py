"""Compatibility boundary for ARP's bundled official LTX outpaint graph.

The frontend workflow uses numeric node identifiers.  Keep the assumptions we
make about those identifiers explicit and validate them before graph surgery so
an updated/replaced workflow fails with a useful message rather than being
silently rewired incorrectly.
"""

from __future__ import annotations

from typing import Any


ADAPTER_ID = "official_ltx23_outpaint_graph_v1"

# Only nodes whose identity is required by ARP's structural rewiring belong in
# this contract. Optional prompt, preview, and guide nodes remain feature-tested
# by the caller.
OFFICIAL_NODE_CONTRACT: dict[str, str] = {
    "3940": "CheckpointLoaderSimple",
    "2004": "LoadImage",
    "3159": "LTXVImgToVideoConditionOnly",
    "5011": "LTXICLoRALoaderModelOnly",
    "5013": "LTXVCropGuides",
    "5093": "SamplerCustomAdvanced",
    "5114": "LTXAddVideoICLoRAGuideAdvanced",
    "5168": "GetVideoComponents",
    "5226": "LTXVLaplacianPyramidBlend",
    "5227": "CreateVideo",
    "5266": "LTXVLaplacianPyramidBlend",
    "5358": "LTXVInpaintPreprocess",
}

_OFFICIAL_SENTINELS = ("5114", "5226", "5266")


def _nodes_by_id(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(node.get("id")): node
        for node in workflow.get("nodes", [])
        if isinstance(node, dict) and node.get("id") is not None
    }


def is_official_outpaint_template(workflow: dict[str, Any]) -> bool:
    nodes = _nodes_by_id(workflow)
    return all(node_id in nodes for node_id in _OFFICIAL_SENTINELS)


def validate_official_outpaint_workflow(workflow: dict[str, Any]) -> str:
    """Validate the graph schema used by ARP and return this adapter's ID."""

    nodes = _nodes_by_id(workflow)
    problems: list[str] = []
    for node_id, expected_type in OFFICIAL_NODE_CONTRACT.items():
        node = nodes.get(node_id)
        if node is None:
            problems.append(f"missing node {node_id} ({expected_type})")
            continue
        actual_type = str(node.get("type") or "")
        if actual_type != expected_type:
            problems.append(
                f"node {node_id} is {actual_type or '<unknown>'}, expected {expected_type}"
            )
    if problems:
        details = "; ".join(problems)
        raise RuntimeError(
            "The selected LTX outpaint workflow does not match ARP's supported "
            f"official graph contract ({ADAPTER_ID}): {details}. Restore the "
            "workflow bundled with this ARP version or update the adapter explicitly."
        )
    return ADAPTER_ID
