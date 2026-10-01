"""Completion stamps that let Run Whole Remaster skip a phase whose inputs have not changed.

A stamp is a digest of the phase's command line plus the size and modification time of every
file it reads: files named on the command line, and files that a manifest on the command line
points at (guide images, reference frames, the manifest's source video). Size+mtime can only err
towards "changed" (identical bytes rewritten), which just sends the phase back to its script,
whose own content signatures then reuse the cached work.

The phase's own outputs are left out, so editing them afterwards (shot manifest prompts, a
regenerated reference image) does not make the phase look stale. Code is not an input either:
updating ARP does not re-run finished phases; the phase's own Run button always runs its script.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .config import ROOT
from .paths import resolve

STAMP_DIR = ROOT / ".cache" / "stage_stamps"

# Manifest columns that are bookkeeping, not inputs: the outpaint chunk manifest's own chunk
# renders, and times the GUI and the scripts each recompute from the frame columns. The GUI and
# outpaint_video.py write these slightly differently (.mp4 vs .mkv raw chunk, last-digit
# rounding), so every run and every page poll would otherwise make the manifest look changed.
DERIVED_COLUMNS = {"prepared_path", "raw_path", "start_seconds", "end_seconds", "end"}
# Program files on the command line are code, not inputs.
CODE_SUFFIXES = {".py", ".pyw", ".exe", ".bat", ".cmd"}
SCANNED_SUFFIXES = {".csv", ".json"}
MAX_SCANNED_BYTES = 8 * 1024 * 1024


def compute_stamp(command: list[str], outputs: list[str], label: str) -> str:
    """Digest of a phase's inputs. `label` is the command as logged (secrets redacted)."""
    excluded = {_identity(resolve(path)) for path in outputs if path}
    files: dict[str, list[int]] = {}
    for arg in command:
        path = _input_file(arg)
        if path is None or _identity(path) in excluded:
            continue
        files[_identity(path)] = _csv_content(path) if path.suffix.lower() == ".csv" else _stat(path)
        if path.suffix.lower() in SCANNED_SUFFIXES:
            for ref in _referenced_files(path):
                if _identity(ref) not in excluded:
                    files[_identity(ref)] = _stat(ref)
    payload = json.dumps({"command": label, "files": files}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stamp_path(stage_key: str, outputs: list[str]) -> Path:
    key = hashlib.sha1(json.dumps(sorted(_identity(resolve(p)) for p in outputs)).encode("utf-8")).hexdigest()[:16]
    return STAMP_DIR / f"{stage_key}_{key}.json"


def read_stamp(stage_key: str, outputs: list[str]) -> str:
    try:
        return str(json.loads(stamp_path(stage_key, outputs).read_text(encoding="utf-8")).get("digest", ""))
    except (OSError, ValueError, AttributeError):
        return ""


def write_stamp(stage_key: str, outputs: list[str], digest: str) -> None:
    path = stamp_path(stage_key, outputs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stage": stage_key, "outputs": outputs, "digest": digest}, indent=2) + "\n", encoding="utf-8")


def clear_stamps(stage_key: str) -> None:
    """Forget every completion stamp of a phase, e.g. after a partial re-render of its output."""
    if STAMP_DIR.is_dir():
        for path in STAMP_DIR.glob(f"{stage_key}_*.json"):
            path.unlink(missing_ok=True)


def _identity(path: Path) -> str:
    try:
        return str(path.resolve()).casefold()
    except OSError:
        return str(path).casefold()


def _stat(path: Path) -> list[int]:
    try:
        stat = path.stat()
        return [stat.st_size, stat.st_mtime_ns]
    except OSError:
        return [-1, -1]


def _manifest_lines(path: Path) -> list[str] | None:
    try:
        if path.stat().st_size > MAX_SCANNED_BYTES:
            return None
        return path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError):
        return None


def _manifest_rows(lines: list[str]) -> list[dict[str, str]]:
    rows = csv.DictReader(line for line in lines if not line.startswith("#"))
    return [{key: value for key, value in row.items() if key not in DERIVED_COLUMNS} for row in rows]


def _csv_content(path: Path):
    """A manifest's inputs by content (headers + rows without bookkeeping columns), not its mtime."""
    lines = _manifest_lines(path)
    if lines is None:
        return _stat(path)
    headers = [line for line in lines if line.startswith("#")]
    payload = json.dumps([headers, _manifest_rows(lines)], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _input_file(text: str) -> Path | None:
    """The existing data file an argument or manifest cell names, if it names one."""
    text = text.strip()
    if not text or len(text) > 1024 or "\n" in text or text.startswith("-"):
        return None
    try:
        path = resolve(text)
        if path.suffix.lower() in CODE_SUFFIXES or not path.is_file():
            return None
    except (OSError, ValueError):
        return None
    return path


def _referenced_files(manifest: Path) -> list[Path]:
    lines = _manifest_lines(manifest)
    if lines is None:
        return []
    strings: list[str] = []
    if manifest.suffix.lower() == ".json":
        try:
            _collect_strings(json.loads("\n".join(lines)), strings)
        except ValueError:
            return []
    else:
        # A "# source_video=<path>" header names the video the rows were cut from.
        strings.extend(line.split("=", 1)[1] for line in lines if line.startswith("#") and "=" in line)
        for row in _manifest_rows(lines):
            for cell in row.values():
                if not isinstance(cell, str):
                    continue
                strings.append(cell)
                if cell.strip()[:1] in {"[", "{"}:
                    # Guide frames are stored as JSON inside one cell.
                    try:
                        _collect_strings(json.loads(cell), strings)
                    except ValueError:
                        pass
    return [path for path in map(_input_file, strings) if path is not None]


def _collect_strings(value, out: list[str]) -> None:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_strings(item, out)
    elif isinstance(value, list):
        for item in value:
            _collect_strings(item, out)
