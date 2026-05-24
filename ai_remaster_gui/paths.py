from __future__ import annotations

from pathlib import Path

from .config import ROOT, VIDEO_EXTS


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve(text: str) -> Path:
    path = Path(text).expanduser()
    return path if path.is_absolute() else ROOT / path


def resolve_video_source(text: str) -> Path:
    path = resolve(text)
    if path.exists():
        return path
    if path.suffix.lower() not in VIDEO_EXTS:
        return path
    name = path.name
    if not any(ch in name for ch in '<>:"|?*'):
        return path
    parent = path.parent
    if not parent.exists():
        return path

    def comparable(filename: str) -> str:
        stripped = "".join("" if ch in '<>:"|?*\uff5c\xa6' else ch for ch in filename)
        return " ".join(stripped.lower().split())

    wanted = comparable(name)
    matches = sorted(
        (
            candidate
            for candidate in parent.iterdir()
            if candidate.is_file()
            and candidate.suffix.lower() == path.suffix.lower()
            and comparable(candidate.name) == wanted
        ),
        key=lambda candidate: candidate.name.lower(),
    )
    return matches[0] if len(matches) == 1 else path


def newest(folder: Path, exts: set[str]) -> Path | None:
    if not folder.exists():
        return None
    files = [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in exts]
    return max(files, key=lambda path: path.stat().st_mtime_ns) if files else None


def aspect_slug(value: str) -> str:
    return value.replace(":", "x").replace(".", "_")


def safe_stem(path_text: str) -> str:
    stem = Path(path_text).stem.replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)


def even_int(value: float) -> int:
    return max(2, int(round(value / 2)) * 2)


def parse_aspect(value: str) -> float:
    if ":" in value:
        left, right = value.split(":", 1)
        return float(left) / float(right)
    return float(value)
