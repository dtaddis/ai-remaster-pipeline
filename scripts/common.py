from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def root_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def signature_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sig.json")


def signature_matches(path: Path, signature: dict[str, Any]) -> bool:
    sig = signature_path(path)
    if not path.exists() or not sig.exists():
        return False
    try:
        return json.loads(sig.read_text(encoding="utf-8-sig")) == signature
    except Exception:
        return False


def write_signature(path: Path, signature: dict[str, Any]) -> None:
    signature_path(path).write_text(json.dumps(signature, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def format_time(seconds: float) -> str:
    total_millis = int(round(seconds * 1000))
    total = total_millis // 1000
    millis = total_millis % 1000
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if millis:
        return f"{hours:d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def safe_stem(path_text: str) -> str:
    stem = Path(path_text).stem.replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
