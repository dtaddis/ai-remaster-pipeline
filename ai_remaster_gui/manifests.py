from __future__ import annotations

import csv
import io
from pathlib import Path


def read_manifest(path: Path) -> list[dict[str, str]]:
    _source, _fields, rows = read_manifest_details(path)
    return rows


def read_manifest_details(path: Path) -> tuple[str, list[str], list[dict[str, str]]]:
    if not path.exists() or not path.is_file():
        return "", [], []
    source_video = ""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if line.startswith("#"):
                if line.startswith("# source_video="):
                    source_video = line.split("=", 1)[1].strip()
                continue
            reader = csv.DictReader([line, *handle.readlines()])
            return source_video, list(reader.fieldnames or []), list(reader)
    return source_video, [], []


def manifest_source_video(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if line.startswith("# source_video="):
                return line.split("=", 1)[1].strip()
            if line and not line.startswith("#"):
                return ""
    return ""


def update_manifest_row(path: Path, index: int, values: dict[str, str]) -> None:
    source_video, fieldnames, rows = read_manifest_details(path)
    if index < 0 or index >= len(rows):
        raise IndexError(f"Manifest row {index} is out of range.")
    for key in values:
        if key not in fieldnames:
            fieldnames.append(key)
    rows[index].update(values)
    write_manifest_details(path, source_video, fieldnames, rows)


def write_manifest_details(path: Path, source_video: str, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if source_video:
            handle.write(f"# source_video={source_video}\n")
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_outpaint_chunk_rows(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            if not row.get("chunk_index", "").isdigit():
                continue
            if "anchor_image" in row and "guide_image" not in row:
                row["guide_image"] = row.get("anchor_image", "")
            rows[int(row["chunk_index"])] = row
        return rows


def write_outpaint_chunk_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "chunk_index",
        "start_frame",
        "end_frame",
        "start_seconds",
        "end_seconds",
        "custom_seconds",
        "seed",
        "prompt_suffix",
        "negative_suffix",
        "guide_image",
        "guide_strength",
        "guide_end_image",
        "guide_end_strength",
        "guide_frames",
        "anchor_image",
        "anchor_position",
        "anchor_seconds",
        "prepared_path",
        "raw_path",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    text = buffer.getvalue()
    if path.exists():
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                if handle.read() == text:
                    return
        except OSError:
            pass
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
