from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from common import ROOT, resolve_path
from reference_sets import expanded_reference_rows, reference_items

DEFAULT_MODEL = "gpt-image-2"
BOUNDARY = "----arp-openai-image-edit"


def multipart_field(name: str, value: str) -> bytes:
    return (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
        f"{value}\r\n"
    ).encode("utf-8")


def multipart_file(name: str, path: Path) -> bytes:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    header = (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8")
    return header + path.read_bytes() + b"\r\n"


def build_body(args: argparse.Namespace, source: Path, references: list[Path] | None = None) -> bytes:
    image_paths = [source, *(references or [])]
    image_field = "image[]" if len(image_paths) > 1 else "image"
    parts = [
        multipart_field("model", args.model),
        multipart_field("prompt", args.prompt),
        multipart_field("size", args.size),
        multipart_field("quality", args.quality),
        *(multipart_file(image_field, path) for path in image_paths),
        f"--{BOUNDARY}--\r\n".encode("utf-8"),
    ]
    return b"".join(parts)


def normalize_to_source_size(path: Path, source_path: Path, *, quiet: bool = False) -> None:
    try:
        from PIL import Image
    except Exception:
        return
    with Image.open(source_path) as source_image:
        target_size = source_image.size
    with Image.open(path) as image:
        if image.size == target_size:
            return
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image.convert("RGB").resize(target_size, resampling).save(path, format="PNG")
    if not quiet:
        print(f"Normalized OpenAI output to source size: {path} ({target_size[0]}x{target_size[1]})", flush=True)


def generate(args: argparse.Namespace, references: list[Path] | None = None) -> Path:
    source = resolve_path(args.source_image)
    output = resolve_path(args.output)
    if not source.is_file():
        raise FileNotFoundError(f"Reference source image not found: {source}")
    if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise RuntimeError("OpenAI image edits require a PNG, JPEG, or WebP source image.")
    token = args.api_key.strip()
    if not token:
        raise RuntimeError("Missing OpenAI API key.")
    if output.exists() and not args.force:
        print(f"Reuse OpenAI reference: {output}", flush=True)
        return output

    reference_paths = [resolve_path(path) for path in (references or []) if resolve_path(path).is_file()]
    continuity = ""
    if reference_paths:
        continuity = getattr(args, "reference_instructions", "") or (
            "Use the additional colour reference images only to preserve palette, material colours, "
            "skin tones, lighting temperature, and overall colour continuity. The first image is the "
            "black-and-white shot to colourise."
        )
    prompt = " ".join(part.strip() for part in (args.prompt, args.prompt_suffix, continuity, args.add_prompt) if part and part.strip())
    args.prompt = prompt
    body = build_body(args, source, reference_paths)
    request = urllib.request.Request(
        "https://api.openai.com/v1/images/edits",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={BOUNDARY}",
        },
        method="POST",
    )
    quiet = bool(getattr(args, "quiet", False))
    if not quiet:
        print(f"OpenAI image edit: {source} -> {output}", flush=True)
        print(f"OpenAI model: {args.model}", flush=True)
        if reference_paths:
            print(f"OpenAI reference images: {len(reference_paths)}", flush=True)
    if args.dry_run:
        return output
    max_retries = max(0, int(getattr(args, "max_retries", 5)))
    payload = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=args.timeout, context=ssl.create_default_context()) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt >= max_retries:
                raise RuntimeError(f"OpenAI API error {exc.code}: {error_text}") from exc
            retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
            try:
                delay = max(1.0, float(retry_after))
            except (TypeError, ValueError):
                delay = min(60.0, 2.0 ** attempt)
            print(f"OpenAI API temporarily unavailable ({exc.code}); retrying in {delay:.0f}s...", flush=True)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt >= max_retries:
                raise RuntimeError(f"OpenAI API request failed: {exc}") from exc
            delay = min(60.0, 2.0 ** attempt)
            print(f"OpenAI connection interrupted; retrying in {delay:.0f}s...", flush=True)
            time.sleep(delay)

    if payload is None:
        raise RuntimeError("OpenAI API request failed without a response.")

    images = payload.get("data") or []
    if not images or not images[0].get("b64_json"):
        raise RuntimeError("OpenAI API response did not contain an image.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".partial")
    temp.write_bytes(base64.b64decode(images[0]["b64_json"]))
    if not args.no_normalize_to_source_size:
        normalize_to_source_size(temp, source, quiet=quiet)
    temp.replace(output)
    if not quiet:
        print(f"Wrote {output}", flush=True)
    return output


def read_manifest(path: Path, enabled_only: bool = True) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        while True:
            pos = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.startswith("#"):
                continue
            handle.seek(pos)
            reader = csv.DictReader(handle)
            for row in reader:
                if enabled_only and row.get("enabled", "true").strip().lower() in {"false", "0", "no", "off"}:
                    continue
                rows.append(row)
            break
    return rows


def row_source(row: dict[str, str]) -> str:
    return row.get("source_reference") or row.get("reference") or ""


def row_target(row: dict[str, str]) -> str:
    return row.get("color_reference") or row.get("reference") or ""


def nearby_reference_images(rows: list[dict[str, str]], row_index: int, count: int) -> list[Path]:
    if count <= 0:
        return []
    previous = []
    later = []
    for index, row in enumerate(rows):
        if index == row_index:
            continue
        candidate = resolve_path(row_target(row))
        if not candidate.is_file():
            continue
        item = (abs(index - row_index), candidate)
        if index < row_index:
            previous.append(item)
        else:
            later.append(item)
    ordered = [path for _distance, path in sorted(previous, key=lambda item: item[0])]
    ordered.extend(path for _distance, path in sorted(later, key=lambda item: item[0]))
    return ordered[:count]


def generate_manifest(args: argparse.Namespace) -> None:
    manifest = resolve_path(args.manifest)
    shot_rows = read_manifest(manifest, enabled_only=args.row_index is None)
    if args.row_index is not None:
        if args.row_index < 0 or args.row_index >= len(shot_rows):
            raise IndexError(f"Manifest row {args.row_index} is out of range.")
        items = reference_items(shot_rows[args.row_index])
        if args.reference_index < 0 or args.reference_index >= len(items):
            raise IndexError(f"Reference {args.reference_index + 1} is out of range for manifest row {args.row_index}.")
        rows = expanded_reference_rows(shot_rows)
        selected_index = next(
            index for index, row in enumerate(rows)
            if row.get("_shot_index") == str(args.row_index) and row.get("_reference_index") == str(args.reference_index)
        )
        selected_rows = [(selected_index, rows[selected_index])]
    else:
        rows = expanded_reference_rows(shot_rows)
    if args.row_index is None and args.limit is not None:
        rows = rows[: args.limit]
        selected_rows = list(enumerate(rows))
    elif args.row_index is None:
        selected_rows = list(enumerate(rows))
    print(f"Manifest: {manifest}", flush=True)
    print(f"Rows: {len(selected_rows)}", flush=True)
    for index, row in selected_rows:
        source = row_source(row)
        output = row_target(row)
        if not source or not output:
            raise RuntimeError(f"Manifest row {index} must have source_reference and color_reference.")
        child = argparse.Namespace(**vars(args))
        child.source_image = source
        child.output = output
        child.add_prompt = " ".join(part.strip() for part in (row.get("prompt", ""), args.add_prompt) if part and part.strip())
        refs = nearby_reference_images(rows, index, args.reference_count)
        generate(child, refs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colorize one reference image with the OpenAI Images API.")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-image", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--add-prompt", default="")
    parser.add_argument("--size", default="auto")
    parser.add_argument("--quality", default="auto")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--row-index", type=int)
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--reference-count", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-normalize-to-source-size", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.manifest:
        generate_manifest(args)
    elif args.source_image and args.output:
        generate(args)
    else:
        raise RuntimeError("Provide either --manifest, or both --source-image and --output.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise
