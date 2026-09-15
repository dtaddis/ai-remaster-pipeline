from __future__ import annotations

import base64
import csv
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def manifest_assets(path: Path, root: Path) -> set[Path]:
    assets: set[Path] = set()
    if path.suffix.lower() != ".csv" or not path.is_file():
        return assets
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    for row in csv.DictReader(lines):
        for value in row.values():
            values = [value]
            try:
                parsed = json.loads(value or "")
                if isinstance(parsed, list):
                    values.extend(str(item.get(k, "")) for item in parsed if isinstance(item, dict) for k in ("source_reference", "color_reference", "image"))
            except (json.JSONDecodeError, TypeError):
                pass
            for text in values:
                candidate = Path(str(text or ""))
                if not candidate.is_absolute():
                    candidate = root / candidate
                if candidate.is_file():
                    assets.add(candidate)
    return assets


def main(encoded: str) -> int:
    job = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")))
    root = Path(job["root"])
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "ARP_COMFY_DIR": job["runtime"] + "/ComfyUI",
        "ARP_COMFY_URL": "http://127.0.0.1:8188",
    })
    display: list[str] = []
    hide_next = False
    for part in job["command"]:
        if hide_next:
            display.append("[redacted]")
            hide_next = False
            continue
        display.append(part)
        hide_next = part in {"--api-key", "--openai-api-key", "--h3-2k-api-key", "--huggingface-token"}
    print("Remote command:", " ".join(display), flush=True)
    code = subprocess.run(job["command"], cwd=root, env=env).returncode
    if code:
        return code
    outputs: set[Path] = set()
    for text in job["outputs"]:
        path = Path(text)
        if path.is_file():
            outputs.add(path)
            outputs.update(manifest_assets(path, root))
            for sidecar in (Path(str(path) + ".sig.json"), Path(str(path) + ".json")):
                if sidecar.is_file():
                    outputs.add(sidecar)
    with zipfile.ZipFile("/tmp/arp-result.zip", "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(outputs):
            archive.write(path, path.relative_to(root).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
