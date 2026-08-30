from __future__ import annotations

import json
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

from .config import IMAGE_EXTS, ROOT, VIDEO_EXTS
from . import state
from .manifests import read_manifest, read_outpaint_chunk_rows
from .paths import resolve, resolve_video_source, safe_stem
from .runtime_settings import default_settings, load_settings

PROJECT_SCHEMA_VERSION = 2
PROJECT_JSON_NAME = "project.json"

# Extensions allowed inside a project bundle. .json is included so outpaint guide sidecars
# (resume signatures, edit metadata) travel with the guides and aren't treated as stale on load.
BUNDLE_EXTS = IMAGE_EXTS | {".csv", ".txt", ".json"}


def bind_context(context: dict) -> None:
    """Bind GUI helpers used to identify the active outpaint chunk manifest."""
    globals().update({key: context[key] for key in ("outpaint_chunk_manifest_for", "pipeline_source_text") if key in context})


def source_signature(source_text: str) -> tuple[str, int, int] | None:
    if not source_text:
        return None
    source = resolve_video_source(source_text)
    if not source.exists() or source.suffix.lower() not in VIDEO_EXTS:
        return None
    stat = source.stat()
    return str(source), stat.st_size, stat.st_mtime_ns

def source_analysis_key(signature: tuple[str, int, int]) -> str:
    return "\0".join(str(part) for part in signature)

def project_payload(settings: dict[str, dict[str, str]]) -> dict:
    stored_settings = {
        stage: {key: value for key, value in values.items() if key != "openai_api_key"}
        for stage, values in settings.items()
    }
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "app": "AI Remaster Pipeline",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "settings": stored_settings,
    }

def write_project_file(path: Path, settings: dict[str, dict[str, str]]) -> None:
    payload = project_payload(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(PROJECT_JSON_NAME, json.dumps(payload, indent=2) + "\n")
        for asset in project_asset_paths(settings):
            archive.write(asset, asset.relative_to(ROOT).as_posix())

def read_project_file(path: Path) -> dict[str, dict[str, str]]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            try:
                data = json.loads(archive.read(PROJECT_JSON_NAME).decode("utf-8-sig"))
            except KeyError as exc:
                raise RuntimeError("Project bundle does not contain project.json.") from exc
            extract_project_assets(archive)
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Project file is not valid JSON: {exc}") from exc
    return load_project_payload(data)

def load_project_payload(data: dict) -> dict[str, dict[str, str]]:
    try:
        version = int(data.get("schema_version", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Project file has an invalid schema_version.") from exc
    if version < 1:
        raise RuntimeError("Project file does not include a supported schema_version.")
    if version > PROJECT_SCHEMA_VERSION:
        raise RuntimeError(f"Project schema {version} is newer than this ARP build supports.")
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise RuntimeError("Project file does not contain settings.")
    # A project must hydrate from clean defaults, not from the settings of the project that is
    # currently open.  In particular, projects saved before image-sequence support do not contain
    # source_images/source_sequence_preview; inheriting those fields could make their conventional
    # video source get replaced by the previously-open sequence during restore_image_sequence_source.
    # Keep only machine-local secrets which are deliberately excluded from project bundles.
    current = load_settings()
    loaded = default_settings(include_newest_source=False)
    loaded["references"]["openai_api_key"] = current.get("references", {}).get("openai_api_key", "")
    for stage, values in settings.items():
        if stage in loaded and isinstance(values, dict):
            loaded[stage].update({str(key): str(value) for key, value in values.items()})
    if not loaded.get("global", {}).get("source"):
        loaded.setdefault("global", {})["expand_outpaint"] = "true"
    loaded.setdefault("global", {}).setdefault("upscale", "false")
    return loaded

def project_asset_paths(settings: dict[str, dict[str, str]]) -> list[Path]:
    candidates: list[str] = []
    for key in ("outpaint", "shots", "references", "colour", "recomp"):
        manifest = settings.get(key, {}).get("manifest", "")
        if manifest:
            candidates.append(manifest)
    assets: list[Path] = []
    seen: set[Path] = set()
    for text in candidates:
        manifest = resolve(text)
        if not project_asset_is_bundleable(manifest) or manifest in seen:
            continue
        seen.add(manifest)
        assets.append(manifest)
        for row in read_manifest(manifest):
            reference_values = [row]
            try:
                extras = json.loads(row.get("additional_references", "") or "[]")
                if isinstance(extras, list):
                    reference_values.extend(item for item in extras if isinstance(item, dict))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            for reference in reference_values:
                for field in ("source_reference", "color_reference"):
                    image = resolve(str(reference.get(field, "") or ""))
                    if project_asset_is_bundleable(image) and image not in seen:
                        seen.add(image)
                        assets.append(image)
    # Outpaint chunk manifests and the hand-edited guide frames they reference, so a project
    # keeps the user's manual guide work for safekeeping.
    for asset in outpaint_guide_asset_paths(settings):
        if asset not in seen:
            seen.add(asset)
            assets.append(asset)
    return assets

def chunk_guide_image_texts(manifest: Path) -> list[str]:
    """Guide image paths referenced by a chunk manifest's rows (the active guide per frame plus
    one undo step), covering both hand-edited guides (outpaint_guides/) and auto seed guides
    (outpaint_seed_guides/). Mirrors outpaint_guides._parse_guide_frames, including the legacy
    guide_image / guide_end_image fields, without importing that module."""
    texts: list[str] = []
    try:
        rows = read_outpaint_chunk_rows(manifest)
    except Exception:
        return texts
    for row in rows.values():
        frames: list = []
        raw = (row.get("guide_frames") or "").strip()
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    frames = parsed
            except (json.JSONDecodeError, TypeError):
                frames = []
        for legacy in ("guide_image", "guide_end_image"):
            if row.get(legacy):
                frames.append({"image": row[legacy]})
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            for key in ("image", "image_previous"):
                value = frame.get(key)
                if value:
                    texts.append(str(value))
    return texts

def outpaint_guide_asset_paths(settings: dict[str, dict[str, str]]) -> list[Path]:
    # The outpaint chunk manifest is user-authored project state: custom lengths, seeds, offsets,
    # guide strengths, and guide image references all live there. Prefer any explicit settings
    # pointer, then use the same GUI helper that names the active manifest so Save Project can
    # preserve guide work even when the cache itself has been cleared.
    manifests: list[Path] = []
    seen_manifests: set[Path] = set()
    explicit = settings.get("outpaint", {}).get("manifest", "")
    if explicit:
        manifest = resolve(explicit)
        if project_asset_is_bundleable(manifest):
            seen_manifests.add(manifest)
            manifests.append(manifest)

    chunk_manifest_for = globals().get("outpaint_chunk_manifest_for")
    pipeline_source = globals().get("pipeline_source_text")
    if chunk_manifest_for and pipeline_source:
        try:
            manifest_text = chunk_manifest_for(pipeline_source(settings), settings.get("outpaint", {}))
            manifest = resolve(manifest_text) if manifest_text else None
            if manifest and project_asset_is_bundleable(manifest) and manifest not in seen_manifests:
                seen_manifests.add(manifest)
                manifests.append(manifest)
        except Exception:
            pass

    assets: list[Path] = []
    seen_assets: set[Path] = set()
    for manifest in manifests:
        if manifest not in seen_assets:
            seen_assets.add(manifest)
            assets.append(manifest)
        for image_text in chunk_guide_image_texts(manifest):
            image = resolve(image_text)
            # Bundle the guide image and its resume/edit sidecars so reloaded guides are used
            # as-is and not treated as stale.
            for candidate in (image, Path(str(image) + ".sig.json"), Path(str(image) + ".json")):
                if candidate not in seen_assets and project_asset_is_bundleable(candidate):
                    seen_assets.add(candidate)
                    assets.append(candidate)
            seen_assets.add(image)
    return assets

def project_asset_is_bundleable(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if path.suffix.lower() not in BUNDLE_EXTS:
        return False
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return False
    return True

def asset_already_on_disk(info: zipfile.ZipInfo, target: Path) -> bool:
    """True when the target file already holds the archived bytes. Restoring it anyway would
    only refresh its mtime, which used to invalidate resume signatures of dependent outputs
    (e.g. outpaint chunks re-rendering after every project load)."""
    try:
        if not target.is_file() or target.stat().st_size != info.file_size:
            return False
        return zlib.crc32(target.read_bytes()) == info.CRC
    except OSError:
        return False

def extract_project_assets(archive: zipfile.ZipFile) -> None:
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        if name == PROJECT_JSON_NAME or name.startswith("/") or ".." in Path(name).parts:
            continue
        if Path(name).suffix.lower() not in BUNDLE_EXTS:
            continue
        target = ROOT / name
        try:
            target.resolve().relative_to(ROOT.resolve())
        except ValueError:
            continue
        # Restore each asset independently so one failure (e.g. a Windows MAX_PATH limit on a
        # deeply nested guide-edit file) can't abort restoring the rest of the project.
        try:
            if asset_already_on_disk(info, target):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as dest:
                dest.write(source.read())
        except OSError as exc:
            print(f"Warning: could not restore project asset {name}: {exc}", flush=True)

def project_default_path(settings: dict[str, dict[str, str]]) -> Path:
    source = resolve_video_source(settings.get("global", {}).get("source", ""))
    stem = safe_stem(source.name if source.name else "arp_project")
    return ROOT / "projects" / f"{stem}.arpp"

def project_save_suggestion(settings: dict[str, dict[str, str]], project_path: Path | None = None) -> Path:
    if project_path:
        return project_path
    default_path = project_default_path(settings)
    last_dir = last_browse_dir(settings)
    return (last_dir / default_path.name) if last_dir else default_path

def last_browse_dir(settings: dict[str, dict[str, str]] | None = None) -> Path | None:
    values = settings or (state.APP.settings if state.APP is not None else {})
    text = values.get("global", {}).get("last_browse_dir", "") if isinstance(values, dict) else ""
    if not text:
        return None
    path = resolve(str(text))
    return path if path.exists() and path.is_dir() else None
