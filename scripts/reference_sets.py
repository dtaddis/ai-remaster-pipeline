from __future__ import annotations

import json
from typing import Any


EXTRA_REFERENCES_FIELD = "additional_references"
REFERENCE_KEYS = ("selected_frame", "source_reference", "color_reference", "prompt")


def _clean_item(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {key: str(value.get(key, "") or "") for key in REFERENCE_KEYS}


def additional_references(row: dict[str, str]) -> list[dict[str, str]]:
    raw = str(row.get(EXTRA_REFERENCES_FIELD, "") or "").strip()
    if not raw:
        return []
    try:
        values = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(values, list):
        return []
    return [item for value in values if (item := _clean_item(value))]


def reference_items(row: dict[str, str]) -> list[dict[str, str]]:
    primary = {
        "selected_frame": str(row.get("selected_frame", "") or ""),
        "source_reference": str(row.get("source_reference") or row.get("reference") or ""),
        "color_reference": str(row.get("color_reference") or row.get("reference") or ""),
        "prompt": str(row.get("prompt", "") or ""),
    }
    return [primary, *additional_references(row)]


def encode_additional_references(items: list[dict[str, str]]) -> str:
    cleaned = [_clean_item(item) for item in items]
    return json.dumps(cleaned, ensure_ascii=False, separators=(",", ":")) if cleaned else ""


def update_reference_item(row: dict[str, str], reference_index: int, values: dict[str, Any]) -> None:
    reference_index = int(reference_index)
    updates = {key: str(value or "") for key, value in values.items() if key in REFERENCE_KEYS}
    if reference_index == 0:
        row.update(updates)
        return
    extras = additional_references(row)
    extra_index = reference_index - 1
    if extra_index < 0 or extra_index >= len(extras):
        raise IndexError(f"Reference {reference_index + 1} is out of range.")
    extras[extra_index].update(updates)
    row[EXTRA_REFERENCES_FIELD] = encode_additional_references(extras)


def append_reference_item(row: dict[str, str], item: dict[str, Any]) -> int:
    extras = additional_references(row)
    extras.append(_clean_item(item))
    row[EXTRA_REFERENCES_FIELD] = encode_additional_references(extras)
    return len(extras)


def remove_reference_item(row: dict[str, str], reference_index: int) -> dict[str, str]:
    if int(reference_index) <= 0:
        raise RuntimeError("The primary shot reference cannot be removed; choose another frame instead.")
    extras = additional_references(row)
    extra_index = int(reference_index) - 1
    if extra_index < 0 or extra_index >= len(extras):
        raise IndexError(f"Reference {reference_index + 1} is out of range.")
    removed = extras.pop(extra_index)
    row[EXTRA_REFERENCES_FIELD] = encode_additional_references(extras)
    return removed


def expanded_reference_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    expanded: list[dict[str, str]] = []
    for shot_index, row in enumerate(rows):
        for reference_index, item in enumerate(reference_items(row)):
            value = dict(row)
            value.update(item)
            value["_shot_index"] = str(shot_index)
            value["_reference_index"] = str(reference_index)
            expanded.append(value)
    return expanded
