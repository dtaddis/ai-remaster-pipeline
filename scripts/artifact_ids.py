"""Single source of truth for artifact identity, naming, sizing, and signature I/O.

Why this module exists
----------------------
The pipeline used to encode an artifact's identity in a long descriptive filename
(`<source>_<aspect>_<WxH><crop>_outpainted.mp4`) and locate files by reconstructing or
prefix-matching that string. Producer (a `scripts/*.py`) and locator (the GUI) each built the
name independently, so they could drift — e.g. the GUI computed a delivery-size name while the
script wrote a model-safe (work) size name, so a finished render looked "missing" and was
re-rendered. The long names also pushed guide-edit paths toward the Windows MAX_PATH limit.

This module makes one definition compute both the identity signature and a short on-disk name,
so producer and locator can never disagree. On-disk names are
``<sourceword>_<tag>_<key>.<ext>`` (e.g. ``Metropolis_outpaint_a1b2c3d4.mp4``) where ``key`` is a
short hash of the canonical identity. The full identity is written into the ``.sig.json`` sidecar
and is the authoritative record; :func:`find` locates an artifact by matching that signature.

It is intentionally stdlib-only (no cv2/torch) so both the ``scripts`` package and the
``ai_remaster_gui`` package can import it (the GUI puts ``scripts/`` on ``sys.path``).

Key vs fingerprint
------------------
The ``key`` (and therefore the path) is derived from *cheap* identity params — the source's
filename stem plus the knobs that distinguish one variant from another (aspect, work size, crop,
…). It deliberately does NOT hash file contents, so the GUI can recompute it on every poll
without reading multi-gigabyte sources. Content fingerprints (size+mtime+sha256) live in the
sidecar's ``inputs`` and are what the producer scripts compare for resume/staleness, exactly as
before.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

SIG_SUFFIX = ".sig.json"
SCHEMA = 1
MODEL_SIZE_MULTIPLE = 32
KEY_LEN = 8


# ── filename helpers ──────────────────────────────────────────────────────────


def safe_stem(path_text: str) -> str:
    """Sanitise a filename stem: spaces -> _, keep alnum/._-, everything else -> _.
    Matches paths.safe_stem / common.safe_stem so identities computed from a name agree."""
    stem = Path(path_text).stem.replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)


def source_word(name: str) -> str:
    """First alphanumeric token of a source filename, capped, for a human-readable name prefix.
    e.g. 'Metropolis 1927 BDrip...' -> 'Metropolis'."""
    tokens = re.findall(r"[A-Za-z0-9]+", Path(name).stem)
    return (tokens[0] if tokens else "src")[:20]


# ── size math (the part that used to be duplicated and drifted) ───────────────


def even(value: float) -> int:
    return max(2, int(round(value / 2)) * 2)


def parse_aspect(value: str) -> float:
    if ":" in str(value):
        left, right = str(value).split(":", 1)
        return float(left) / float(right)
    return float(value)


def model_safe(value: int, multiple: int = MODEL_SIZE_MULTIPLE) -> int:
    value = max(multiple, int(value))
    lower = max(multiple, (value // multiple) * multiple)
    upper = lower if lower == value else lower + multiple
    return lower if value - lower <= upper - value else upper


def resolved_height(source_height: int, target_height_text: str) -> int:
    if str(target_height_text or "").strip().lower() in {"source", "source height", "original"}:
        return even(int(source_height or 720))
    try:
        return even(int(float(target_height_text)))
    except (TypeError, ValueError):
        return 720


def delivery_size(source_height: int, aspect: str, target_height_text: str) -> tuple[int, int]:
    """Final delivery resolution (the size the master is intended to be)."""
    height = resolved_height(source_height, target_height_text)
    return even(height * parse_aspect(aspect)), height


def work_size(source_height: int, aspect: str, target_height_text: str) -> tuple[int, int]:
    """Model-safe working resolution LTX actually renders at (delivery rounded to multiples of 32).
    This is what the outpaint output file is named with."""
    width, height = delivery_size(source_height, aspect, target_height_text)
    return model_safe(width), model_safe(height)


# ── identity + naming ─────────────────────────────────────────────────────────


def canonical(identity: dict[str, Any]) -> str:
    return json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def artifact_key(identity: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(identity).encode("utf-8")).hexdigest()[:KEY_LEN]


def artifact_basename(word: str, tag: str, identity: dict[str, Any]) -> str:
    """Stem shared by an artifact and its sidecar: ``<word>_<tag>_<key>``."""
    return f"{word}_{tag}_{artifact_key(identity)}"


def artifact_name(word: str, tag: str, identity: dict[str, Any], ext: str) -> str:
    base = artifact_basename(word, tag, identity)
    ext = str(ext).lstrip(".")
    return f"{base}.{ext}" if ext else base


def artifact_path(directory: Path | str, word: str, tag: str, identity: dict[str, Any], ext: str) -> Path:
    return Path(directory) / artifact_name(word, tag, identity, ext)


# ── identity builders (one definition per artifact kind) ──────────────────────
# Builders take the *final* work sizes (computed via work_size) so producer and locator,
# which both call work_size, always produce byte-identical identities.


def outpaint_identity(source_name: str, aspect: str, work_w: int, work_h: int, crop: Iterable[int], black: bool) -> dict:
    """Identity shared by the whole outpaint family: outpaint output, raw comfy render, prepared
    canvas, chunk manifest, guide directory and seed guides. Includes every input that changes
    the crop-first canvas geometry or masking mode."""
    left, right, top, bottom = ([int(v) for v in crop] + [0, 0, 0, 0])[:4]
    identity = {
        "v": 2,
        "kind": "outpaint",
        "geometry": "crop_then_fit_v1",
        "source": safe_stem(source_name),
        "aspect": str(aspect),
        "w": int(work_w),
        "h": int(work_h),
        "crop": [left, right, top, bottom],
        "black": bool(black),
    }
    return identity


def outpaint_basename(source_name: str, aspect: str, work_w: int, work_h: int, crop: Iterable[int], black: bool, tag: str) -> str:
    """Shared stem for every outpaint-family artifact (output/rawcomfy/prepared/chunks). Call from
    both the GUI locator and the producer scripts so their names are byte-identical."""
    ident = outpaint_identity(source_name, aspect, work_w, work_h, crop, black)
    return artifact_basename(source_word(source_name), tag, ident)


def outpaint_name(source_name: str, aspect: str, work_w: int, work_h: int, crop: Iterable[int], black: bool, tag: str, ext: str) -> str:
    base = outpaint_basename(source_name, aspect, work_w, work_h, crop, black, tag)
    ext = str(ext).lstrip(".")
    return f"{base}.{ext}" if ext else base


def cleanup_identity(
    source_name: str,
    lora: str,
    prompt: str,
    negative_prompt: str,
    strength: float,
    seed: int,
    *,
    source_fidelity: float = 1.0,
    ai_descratch: bool = False,
    ai_descratch_height: int | str = 720,
    scratch_sensitivity: float = 0.65,
    scratch_mask_dilate: int = 3,
    ai_chunk_frames: int = 41,
    devignette: bool = False,
    dearchive: bool = True,
) -> dict:
    """Identity for the geometry-preserving archive clean-up render.

    Chunk size and overlap are deliberately resume settings rather than identity settings: changing
    how a movie is divided should rebuild the same logical output path, while model/prompt changes
    describe a genuinely different restoration.
    """
    identity = {
        "v": 5,
        "kind": "cleanup",
        "source": safe_stem(source_name),
        "ai_descratch": bool(ai_descratch),
        "devignette": bool(devignette),
        "dearchive": bool(dearchive),
    }
    if ai_descratch:
        identity.update({
            "ai_model": "propainter",
            "ai_height": str(ai_descratch_height),
            "scratch_sensitivity": float(scratch_sensitivity),
            "scratch_mask_dilate": int(scratch_mask_dilate),
            "ai_chunk_frames": int(ai_chunk_frames),
        })
    if dearchive:
        identity.update({
            "lora": Path(str(lora)).name,
            "prompt": str(prompt).strip(),
            "negative_prompt": str(negative_prompt).strip(),
            "strength": float(strength),
            "source_fidelity": float(source_fidelity),
            "seed": int(seed),
        })
    return identity


def stabilize_identity(
    source_name: str,
    *,
    smoothing: int = 12,
    max_shift: int = 48,
    max_angle: float = 3.0,
    zoom: float = 3.0,
    shot_threshold: float = 0.075,
    min_shot_seconds: float = 1.0,
    scene_aware: bool = True,
    encoder: str = "ffv1",
) -> dict:
    """Identity for scene-aware, geometry-preserving stabilization."""
    return {
        "v": 3,
        "kind": "stabilize",
        "source": safe_stem(source_name),
        "smoothing": int(smoothing),
        "max_shift": int(max_shift),
        "max_angle": float(max_angle),
        "zoom": float(zoom),
        "shot_threshold": float(shot_threshold),
        "min_shot_seconds": float(min_shot_seconds),
        "scene_aware": bool(scene_aware),
        "cut_detection": "hard_cuts_only_v1",
        "encoder": str(encoder),
    }


def shots_identity(outpaint_stem: str) -> dict:
    """Shot manifest is keyed on the outpainted video only, so it stays stable and user-editable
    across detection-parameter tweaks (threshold etc. live in the resume sig, not the identity)."""
    return {"v": 1, "kind": "shots", "outpaint": safe_stem(outpaint_stem)}


def references_identity(outpaint_stem: str) -> dict:
    return {"v": 1, "kind": "references", "outpaint": safe_stem(outpaint_stem)}


def colorized_identity(manifest_stem: str, method: str) -> dict:
    return {"v": 1, "kind": "color", "manifest": safe_stem(manifest_stem), "method": str(method)}


def recomp_identity(outpaint_stem: str) -> dict:
    return {"v": 1, "kind": "recomp", "outpaint": safe_stem(outpaint_stem)}


def upscale_identity(input_stem: str, target_w: int, target_h: int, model: str, pre_downscale: bool = False, scale: int | str = 0) -> dict:
    ident = {"v": 1, "kind": "upscale", "input": safe_stem(input_stem), "w": int(target_w), "h": int(target_h), "model": str(model)}
    if pre_downscale:
        ident.update({"pre_downscale": True, "scale": int(scale or 0)})
    return ident


def upscale_preview_identity(input_stem: str, target_w: int, target_h: int, model: str, seconds: str, pre_downscale: bool = False, scale: int | str = 0) -> dict:
    ident = upscale_identity(input_stem, target_w, target_h, model, pre_downscale, scale)
    ident.update({"kind": "upscalepreview", "seconds": str(seconds)})
    return ident


def soundtrack_identity(input_stem: str, music: bool, sfx: bool) -> dict:
    return {"v": 1, "kind": "audio", "input": safe_stem(input_stem), "music": bool(music), "sfx": bool(sfx)}


# ── signature sidecar I/O ─────────────────────────────────────────────────────


def sig_path(path: Path | str) -> Path:
    return Path(str(path) + SIG_SUFFIX)


def write_identity(path: Path | str, identity: dict[str, Any], label: str = "", inputs: dict[str, Any] | None = None) -> Path:
    """Write ``<path>.sig.json`` = identity + human-readable label + input content fingerprints."""
    payload = {
        "schema": SCHEMA,
        "kind": identity.get("kind", ""),
        "identity": identity,
        "key": artifact_key(identity),
        "label": label,
        "inputs": inputs or {},
    }
    target = sig_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def read_identity(path: Path | str) -> dict[str, Any] | None:
    target = sig_path(path)
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def find(directory: Path | str, word: str, tag: str, identity: dict[str, Any], ext: str) -> Path | None:
    """Locate the artifact for ``identity``.

    The sidecar is authoritative: the deterministic ``<word>_<tag>_<key>.<ext>`` candidate is
    accepted only if its sidecar's identity matches (or it has no sidecar, for back-compat); then
    we fall back to scanning the directory's sidecars for an identity match. This is robust if the
    naming scheme or key length later changes.
    """
    directory = Path(directory)
    candidate = artifact_path(directory, word, tag, identity, ext)
    if candidate.exists():
        # The deterministic name already encodes the identity (key = hash of identity), so accept
        # the candidate unless its sidecar records a *conflicting* identity (collision / stale file).
        # A sidecar that is a plain resume signature (no identity field) is fine.
        sig = read_identity(candidate)
        sig_identity = sig.get("identity") if sig else None
        if sig_identity is None or sig_identity == identity:
            return candidate
    if directory.exists():
        suffix = f".{str(ext).lstrip('.')}" if ext else ""
        for sidecar in sorted(directory.glob("*" + SIG_SUFFIX)):
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("identity") != identity:
                continue
            target = Path(str(sidecar)[: -len(SIG_SUFFIX)])
            if (not suffix or target.suffix == suffix) and target.exists():
                return target
    return None
