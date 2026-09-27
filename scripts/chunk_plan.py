"""Outpaint chunk planning, shared by outpaint_video.py and the GUI.

Both sides must cut the video into exactly the same chunks: the chunk manifest stores guide
frames relative to their chunk, so any disagreement over where a chunk starts moves every
guide in it. Keep all chunk-range arithmetic here.
"""

from __future__ import annotations

import json


def ltx_valid_frame_count(seconds: float, fps: float) -> int:
    """Return the closest positive LTX temporal length (8n + 1)."""
    requested = max(1, int(round(float(seconds) * float(fps))))
    lower = max(1, ((requested - 1) // 8) * 8 + 1)
    upper = lower + 8
    return lower if requested - lower <= upper - requested else upper


def valid_final_chunk_start(total_frames: int, start: int, ranges: list[tuple[int, int, int]]) -> int:
    """Grow the final overlap just enough to make its temporal length 8n + 1."""
    length = total_frames - start
    grow = (1 - length) % 8
    adjusted = max(0, start - grow)
    if ranges and adjusted <= ranges[-1][1]:
        return start
    return adjusted


def chunk_ranges(
    total_frames: int,
    fps: float,
    default_seconds: float,
    overlap_frames: int,
    existing: dict[int, dict[str, str]],
    max_frames: int = 0,
) -> list[tuple[int, int, int]]:
    """Return (chunk_index, start_frame, end_frame) for every chunk.

    Chunks are default_seconds long unless their manifest row sets custom_seconds, and never
    longer than max_frames when that is set (Wan VACE renders a chunk in one pass, so its
    chunks are capped by what one pass can hold). 0 seconds means one chunk for the whole
    clip, still subject to max_frames.
    """
    if max_frames > 0:
        # Stay on an 8n + 1 length so growing the final chunk can never pass the cap.
        max_frames = max(1, ((int(max_frames) - 1) // 8) * 8 + 1)
    if total_frames <= 0 or (default_seconds <= 0 and max_frames <= 0):
        return [(0, 0, total_frames)]
    ranges: list[tuple[int, int, int]] = []
    start = 0
    index = 0
    while start < total_frames:
        seconds = default_seconds
        custom = existing.get(index, {}).get("custom_seconds", "")
        if custom:
            try:
                seconds = float(custom)
            except ValueError:
                seconds = default_seconds
        chunk_frames = total_frames if seconds <= 0 else ltx_valid_frame_count(seconds, fps)
        if max_frames > 0:
            chunk_frames = min(chunk_frames, max_frames)
        end = min(total_frames, start + chunk_frames)
        if end == total_frames:
            start = valid_final_chunk_start(total_frames, start, ranges)
        ranges.append((index, start, end))
        if end >= total_frames:
            break
        overlap = max(0, min(int(overlap_frames), chunk_frames - 1))
        start += max(1, chunk_frames - overlap)
        index += 1
    return ranges


def guide_frames_from_row(row: dict[str, str]) -> list[dict]:
    """A chunk row's guide frames, migrating the legacy guide_image / guide_end_image fields."""
    raw = (row.get("guide_frames", "") or "").strip()
    if raw:
        try:
            frames = json.loads(raw)
            if isinstance(frames, list):
                return [frame for frame in frames if isinstance(frame, dict)]
        except (json.JSONDecodeError, ValueError):
            pass
    frames: list[dict] = []
    if row.get("guide_image"):
        frames.append({"frame_idx": 0, "image": row["guide_image"], "strength": row.get("guide_strength", "0.7")})
    if row.get("guide_end_image"):
        frames.append({"frame_idx": -1, "image": row["guide_end_image"], "strength": row.get("guide_end_strength", "1.0")})
    return frames


def _row_span(row: dict[str, str]) -> tuple[int, int] | None:
    try:
        return int(row["start_frame"]), int(row["end_frame"])
    except (KeyError, TypeError, ValueError):
        return None


# Settings describing a stretch of the film: every new chunk inside it inherits them.
SPAN_FIELDS = ("prompt_suffix", "negative_suffix", "offset_mode", "offset_x", "offset_y")
# Settings describing one chunk's own render or its join: only one new chunk may keep them.
START_FIELDS = ("seed", "auto_start_guide")


def remap_chunk_rows(
    existing: dict[int, dict[str, str]],
    ranges: list[tuple[int, int, int]],
) -> dict[int, dict[str, str]]:
    """Carry chunk rows onto a new chunk layout, keeping guides on the same picture.

    Returns ``existing`` itself when the layout is unchanged. Otherwise (a different Chunk
    seconds, a custom length upstream, or switching between LTX and Wan's shorter chunks)
    each guide frame moves to the first new chunk that contains its absolute frame, since a
    guide pins what one particular frame looks like. The new chunk that overlaps an old
    chunk most inherits its prompt suffixes and offsets; seed and the previous-chunk start
    guide stay with the first new chunk taken from each old one, the rest use defaults.
    custom_seconds stays with its chunk number, because chunk_ranges reads it that way.
    """
    old: list[tuple[int, int, int, dict[str, str]]] = []
    for index, row in sorted(existing.items()):
        span = _row_span(row)
        if span is None:
            return existing
        old.append((index, span[0], span[1], row))
    new_spans = {index: (start, end) for index, start, end in ranges}
    if {index: (start, end) for index, start, end, _row in old} == new_spans:
        return existing

    guides: list[tuple[int, dict]] = []
    for _index, start, end, row in old:
        for guide in guide_frames_from_row(row):
            try:
                offset = int(guide.get("frame_idx", 0))
            except (TypeError, ValueError):
                continue
            guides.append((start + (offset if offset >= 0 else (end - start) + offset), guide))

    remapped: dict[int, dict[str, str]] = {}
    claimed: set[int] = set()
    for index, start, end in ranges:
        row: dict[str, str] = {"custom_seconds": existing.get(index, {}).get("custom_seconds", "")}
        overlaps = [(min(end, old_end) - max(start, old_start), old_index, old_row) for old_index, old_start, old_end, old_row in old]
        overlap, source_index, source = max(overlaps, key=lambda item: item[0], default=(0, -1, {}))
        if overlap > 0:
            row.update({key: source[key] for key in SPAN_FIELDS if key in source})
            if source_index not in claimed:
                claimed.add(source_index)
                row.update({key: source[key] for key in START_FIELDS if key in source})
        remapped[index] = row

    placed: dict[int, list[dict]] = {index: [] for index in remapped}
    for frame, guide in guides:
        target = next(((index, start) for index, start, end in ranges if start <= frame < end), None)
        if target is None:
            continue
        placed[target[0]].append(dict(guide, frame_idx=frame - target[1]))
    for index, row in remapped.items():
        frames = sorted(placed[index], key=lambda guide: int(guide["frame_idx"]))
        row.update({"guide_frames": json.dumps(frames) if frames else "", "guide_image": "", "guide_end_image": ""})
    return remapped
