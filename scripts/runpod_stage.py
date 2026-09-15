from __future__ import annotations

"""Run one ordinary ARP stage on a disposable or reusable RunPod Pod.

Only Python's standard library plus OpenSSH are required locally.  The worker mirrors the small
ARP source tree, transfers just the files referenced by the command/manifest, bootstraps ComfyUI
once on the persistent /workspace volume, executes the unchanged stage CLI, and returns outputs.
"""

import argparse
import base64
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/workspace/ai-remaster-pipeline"
REMOTE_RUNTIME = "/workspace/arp-runtime"
RUNPOD_API = "https://rest.runpod.io/v1"
SOURCE_ITEMS = (
    "ai_remaster_gui", "scripts", "vendor/comfyui_custom_nodes", "workflows",
    "wrappers", "requirements.txt", "VERSION",
)
PATH_FLAGS = {
    "--source", "--input", "--output", "--manifest", "--outpainted", "--colorized",
    "--source-video", "--output-manifest", "--workflow", "--qwen-workflow",
    "--qwen-masked-workflow", "--chunk-manifest", "--comfy-dir", "--comfy-output-root",
    "--cmnet2-dir",
}


def api_request(api_key: str, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        RUNPOD_API + path,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AI-Remaster-Pipeline/1.0",
        },
    )
    attempts = 3 if method.upper() == "GET" else 1
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"RunPod API {method} {path} failed ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt >= attempts:
                raise RuntimeError(f"RunPod API {method} {path} was temporarily unreachable: {exc}") from exc
            print(f"RunPod API read timed out; retrying ({attempt}/{attempts})...", flush=True)
            time.sleep(2)
    return json.loads(raw) if raw else {}


def ensure_ssh_key() -> tuple[Path, str]:
    key = ROOT / ".cache" / "runpod" / "id_ed25519"
    public = Path(str(key) + ".pub")
    if not public.is_file():
        key.parent.mkdir(parents=True, exist_ok=True)
        tool = shutil.which("ssh-keygen")
        if not tool:
            raise RuntimeError("OpenSSH ssh-keygen is required for automatic RunPod setup.")
        subprocess.run([tool, "-q", "-t", "ed25519", "-N", "", "-C", "arp-runpod", "-f", str(key)], check=True)
    return key, public.read_text(encoding="utf-8").strip()


def create_pod(args: argparse.Namespace, public_key: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": "AI Remaster Pipeline",
        "imageName": args.image,
        "gpuCount": 1,
        "gpuTypeIds": [item.strip() for item in args.gpu_types.split(",") if item.strip()],
        # Availability priority is important for long-running media work: a stopped Pod's
        # original host may be full when the next stage begins.  RunPod can choose whichever
        # compatible card is actually free instead of failing on the first item in the list.
        "gpuTypePriority": "availability",
        "cloudType": "SECURE",
        "computeType": "GPU",
        "interruptible": False,
        "containerDiskInGb": 50,
        "volumeInGb": int(args.volume_gb),
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp", "8188/http"],
        "supportPublicIp": True,
        "globalNetworking": True,
        "allowedCudaVersions": ["13.0"],
        "env": {"PUBLIC_KEY": public_key, "SSH_PUBLIC_KEY": public_key, "ARP_CUDA_VERSION": "13.0"},
    }
    if args.network_volume_id:
        payload["networkVolumeId"] = args.network_volume_id
        payload.pop("volumeInGb", None)
    pod = api_request(args.api_key, "POST", "/pods", payload)
    print(f"Created RunPod Pod {pod.get('id', '<unknown>')}; CUDA 13 secure cloud requested.", flush=True)
    return pod


def remember_cloud_value(key: str, value: str, settings_file: str = "") -> None:
    settings_path = Path(settings_file or os.environ.get("ARP_SETTINGS_FILE") or ROOT / ".ai_remaster_gui.json")
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.is_file() else {}
        data.setdefault("cloud", {})[key] = value
        temporary = settings_path.with_suffix(settings_path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temporary.replace(settings_path)
    except (OSError, json.JSONDecodeError):
        print(f"Note: could not remember RunPod setting {key}; copy it into Settings to reuse it.", flush=True)


def ensure_network_volume(args: argparse.Namespace) -> None:
    """Create portable storage once when the user selected automatic network storage."""

    if args.storage_mode != "network" or args.network_volume_id:
        return
    size = int(args.volume_gb)
    if size < 50:
        raise RuntimeError("RunPod network storage must be at least 50 GB for the ARP runtime.")
    payload = {
        "dataCenterId": args.data_center_id,
        "name": "AI Remaster Pipeline",
        "size": size,
    }
    volume = api_request(args.api_key, "POST", "/networkvolumes", payload)
    volume_id = str(volume.get("id") or "").strip()
    if not volume_id:
        raise RuntimeError("RunPod created network storage but did not return its volume ID.")
    args.network_volume_id = volume_id
    remember_cloud_value("runpod_network_volume_id", volume_id, args.settings_file)
    monthly = size * 0.07
    print(
        f"Created portable RunPod network volume {volume_id} in {args.data_center_id} "
        f"({size} GB standard storage, approximately ${monthly:.2f}/month).",
        flush=True,
    )


def pod_ready(pod: dict[str, Any]) -> tuple[str, int] | None:
    address = str(pod.get("publicIp") or "").strip()
    mappings = pod.get("portMappings") or {}
    port = mappings.get("22") or mappings.get(22)
    if address and port:
        return address, int(port)
    return None


def wait_for_pod(args: argparse.Namespace, public_key: str) -> tuple[dict[str, Any], str, int]:
    pod = api_request(args.api_key, "GET", f"/pods/{args.pod_id}") if args.pod_id else {}
    if pod and args.network_volume_id:
        # Current REST responses expose the attachment as networkVolumeId.  Keep
        # accepting the older nested GraphQL-style shape for saved configurations.
        attached = str(
            pod.get("networkVolumeId")
            or (pod.get("networkVolume") or {}).get("id")
            or ""
        )
        if attached != args.network_volume_id:
            print(
                f"Saved Pod {args.pod_id} does not use the selected portable volume; "
                "creating a replacement while preserving the old Pod and its data.",
                flush=True,
            )
            pod = {}
            args.pod_id = ""
    if not pod:
        pod = create_pod(args, public_key)
    pod_id = str(pod.get("id") or args.pod_id)
    deadline = time.monotonic() + 900
    start_retry_at = 0.0
    while time.monotonic() < deadline:
        ready = pod_ready(pod)
        if ready:
            return pod, ready[0], ready[1]
        status = pod.get("desiredStatus") or pod.get("status") or "starting"
        if str(status).upper() in {"STOPPED", "EXITED"} and time.monotonic() >= start_retry_at:
            print(f"Starting saved RunPod Pod {pod_id}...", flush=True)
            try:
                api_request(args.api_key, "POST", f"/pods/{pod_id}/start")
            except RuntimeError as exc:
                if "not enough free gpus" not in str(exc).lower():
                    raise
                print("Saved RunPod host is temporarily full; retrying without creating a new billed worker...", flush=True)
                start_retry_at = time.monotonic() + 20
        print(f"Waiting for RunPod {pod_id}: {status}", flush=True)
        time.sleep(5)
        pod = api_request(args.api_key, "GET", f"/pods/{pod_id}")
    raise TimeoutError(f"RunPod {pod_id} did not expose SSH within 15 minutes.")


def remember_pod_id(pod_id: str, settings_file: str = "") -> None:
    remember_cloud_value("runpod_pod_id", pod_id, settings_file)


def ssh_base(key: Path, host: str, port: int) -> list[str]:
    return [
        "ssh", "-i", str(key), "-p", str(port),
        "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30", f"root@{host}",
    ]


def scp_base(key: Path, host: str, port: int) -> list[str]:
    return [
        "scp", "-i", str(key), "-P", str(port),
        "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30",
    ]


def wait_for_ssh(key: Path, host: str, port: int) -> None:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        result = subprocess.run(
            [*ssh_base(key, host, port), "true"], capture_output=True, text=True,
        )
        if result.returncode == 0:
            return
        print("RunPod is allocated; waiting for its SSH service...", flush=True)
        time.sleep(5)
    raise TimeoutError("The RunPod SSH service did not become ready within 10 minutes.")


def add_zip_path(archive: zipfile.ZipFile, path: Path) -> None:
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_file() and "__pycache__" not in child.parts:
                archive.write(child, child.relative_to(ROOT).as_posix())
    elif path.is_file():
        archive.write(path, path.relative_to(ROOT).as_posix())


def resolve_local(text: str) -> Path | None:
    if not text or text.startswith("-") or "://" in text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def archive_name(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(ROOT.resolve(strict=False)).as_posix()
    except (OSError, ValueError):
        digest = hashlib.sha256(str(path.resolve(strict=False)).encode("utf-8")).hexdigest()[:12]
        return f"input/cloud_imports/{digest}_{path.name}"


def referenced_manifest_files(path: Path) -> set[Path]:
    found: set[Path] = set()
    if path.suffix.lower() != ".csv" or not path.is_file():
        return found
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    for row in csv.DictReader(lines):
        for value in row.values():
            candidates = [value]
            try:
                parsed = json.loads(value or "")
                if isinstance(parsed, list):
                    candidates.extend(str(item.get(key, "")) for item in parsed if isinstance(item, dict) for key in ("source_reference", "color_reference", "image"))
            except (json.JSONDecodeError, TypeError):
                pass
            for candidate in candidates:
                resolved = resolve_local(str(candidate or ""))
                if resolved and resolved.is_file():
                    found.add(resolved)
    return found


def make_job_archive(command: list[str], target: Path) -> None:
    inputs: set[Path] = set()
    for value in command:
        path = resolve_local(value)
        if path and path.is_file():
            inputs.add(path)
            inputs.update(referenced_manifest_files(path))
            for sidecar in (Path(str(path) + ".sig.json"), Path(str(path) + ".json")):
                if sidecar.is_file():
                    inputs.add(sidecar)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        source_items = list(SOURCE_ITEMS)
        # CMNet2's vendored weights are roughly 700 MB. They are useful for that one
        # colourizer, but should not be uploaded before every unrelated cloud stage.
        if "--cmnet2-dir" in command:
            source_items.append("vendor/cmnet2")
        for item in source_items:
            add_zip_path(archive, ROOT / item)
        for path in sorted(inputs):
            resolved = path.resolve(strict=False)
            if any(
                resolved == (ROOT / item).resolve(strict=False)
                or (ROOT / item).is_dir() and (ROOT / item).resolve(strict=False) in resolved.parents
                for item in source_items
            ):
                continue
            archive.write(path, archive_name(path))


def remote_path(text: str) -> str:
    path = resolve_local(text)
    if path is None:
        return text
    return f"{REMOTE_ROOT}/{archive_name(path)}"


def remote_command(command: list[str]) -> list[str]:
    converted: list[str] = []
    for index, value in enumerate(command):
        if index == 0:
            converted.append(f"{REMOTE_RUNTIME}/venv/bin/python")
        elif index > 0 and command[index - 1] == "--comfy-dir":
            converted.append(f"{REMOTE_RUNTIME}/ComfyUI")
        elif index > 0 and command[index - 1] == "--comfy-output-root":
            converted.append(f"{REMOTE_RUNTIME}/ComfyUI/output")
        elif index > 0 and command[index - 1] in PATH_FLAGS:
            converted.append(remote_path(value))
        elif value.lower().endswith(".py") and resolve_local(value):
            converted.append(remote_path(value))
        else:
            converted.append(value)
    return converted


def encoded_remote_job(command: list[str], outputs: list[str]) -> str:
    payload = {
        "command": remote_command(command),
        "outputs": [remote_path(value) for value in outputs],
        "root": REMOTE_ROOT,
        "runtime": REMOTE_RUNTIME,
    }
    return base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


def load_local_cloud_secrets(args: argparse.Namespace) -> None:
    """Load machine-local credentials and reusable IDs omitted from process arguments."""

    settings_path = Path(args.settings_file)
    try:
        stored = json.loads(settings_path.read_text(encoding="utf-8-sig"))
        cloud = stored.get("cloud", {}) if isinstance(stored, dict) else {}
    except (OSError, json.JSONDecodeError):
        cloud = {}
    if not args.api_key.strip():
        args.api_key = str(cloud.get("runpod_api_key", "") or os.environ.get("RUNPOD_API_KEY", ""))
    if not args.huggingface_token.strip():
        args.huggingface_token = str(cloud.get("huggingface_token", "") or os.environ.get("HF_TOKEN", ""))
    # The GUI normally supplies these explicitly, but direct/diagnostic runner calls
    # should still reuse the saved worker and portable volume instead of allocating a
    # duplicate billed Pod merely because --pod-id was omitted.
    if not args.pod_id.strip():
        args.pod_id = str(cloud.get("runpod_pod_id", "") or "").strip()
    if not args.network_volume_id.strip():
        args.network_volume_id = str(cloud.get("runpod_network_volume_id", "") or "").strip()


def run(args: argparse.Namespace) -> int:
    load_local_cloud_secrets(args)
    if not args.api_key.strip():
        raise RuntimeError("Add a RunPod API key in Settings before selecting RunPod compute.")
    if not args.command:
        raise RuntimeError("No ARP stage command was supplied to the RunPod runner.")
    ensure_network_volume(args)
    key, public_key = ensure_ssh_key()
    pod, host, port = wait_for_pod(args, public_key)
    pod_id = str(pod.get("id") or args.pod_id)
    remember_pod_id(pod_id, args.settings_file)
    print(f"RunPod worker: {pod_id} ({host}:{port})", flush=True)
    wait_for_ssh(key, host, port)
    try:
        with tempfile.TemporaryDirectory(prefix="arp-runpod-") as folder_text:
            folder = Path(folder_text)
            archive = folder / "job.zip"
            result = folder / "result.zip"
            make_job_archive(args.command, archive)
            print(f"Uploading ARP job bundle ({archive.stat().st_size / 1048576:.1f} MiB)...", flush=True)
            subprocess.run([*scp_base(key, host, port), str(archive), f"root@{host}:/tmp/arp-job.zip"], check=True)
            token_setup = ""
            if args.huggingface_token:
                # Send secrets over SSH stdin, never in a local/remote process command line or
                # persistent Pod environment. The shell removes the root-only temp file on exit.
                subprocess.run(
                    [*ssh_base(key, host, port), "umask 077; cat > /tmp/.arp-hf-token"],
                    input=args.huggingface_token.encode("utf-8"),
                    check=True,
                )
                token_setup = "trap 'rm -f /tmp/.arp-hf-token' EXIT; export HF_TOKEN=\"$(cat /tmp/.arp-hf-token)\"; "
            payload = encoded_remote_job(args.command, args.expected_output)
            remote = (
                token_setup
                + f"mkdir -p {REMOTE_ROOT} {REMOTE_RUNTIME} && "
                f"python3 -m zipfile -e /tmp/arp-job.zip {REMOTE_ROOT} && "
                f"bash {REMOTE_ROOT}/scripts/bootstrap_runpod.sh {REMOTE_RUNTIME} {args.stage} && "
                f"{REMOTE_RUNTIME}/venv/bin/python -u {REMOTE_ROOT}/scripts/runpod_remote_job.py {payload}"
            )
            subprocess.run([*ssh_base(key, host, port), remote], check=True)
            subprocess.run([*scp_base(key, host, port), f"root@{host}:/tmp/arp-result.zip", str(result)], check=True)
            with zipfile.ZipFile(result) as bundle:
                bundle.extractall(ROOT)
            # Deterministic stage outputs normally live below ROOT. If the user chose an
            # external Save path, remote_path placed it in cloud_imports for transport;
            # copy that downloaded artifact back to the exact requested destination.
            for output_text in args.expected_output:
                local_output = resolve_local(output_text)
                if local_output is None:
                    continue
                try:
                    local_output.resolve(strict=False).relative_to(ROOT.resolve(strict=False))
                    continue
                except (OSError, ValueError):
                    pass
                downloaded = ROOT / archive_name(local_output)
                if not downloaded.is_file():
                    continue
                local_output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(downloaded, local_output)
                for suffix in (".sig.json", ".json"):
                    sidecar = Path(str(downloaded) + suffix)
                    if sidecar.is_file():
                        shutil.copy2(sidecar, Path(str(local_output) + suffix))
    finally:
        if args.idle_minutes <= 0:
            print(f"Stopping RunPod {pod_id} to end GPU billing...", flush=True)
            api_request(args.api_key, "POST", f"/pods/{pod_id}/stop")
    print(f"Downloaded {len(args.expected_output)} expected output(s) from RunPod {pod_id}.", flush=True)
    if args.idle_minutes > 0:
        print(f"RunPod {pod_id} remains running by request; stop it from the RunPod console when finished.", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute an ARP stage on a managed RunPod worker.")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--api-key", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--settings-file",
        default=os.environ.get("ARP_SETTINGS_FILE", str(ROOT / ".ai_remaster_gui.json")),
        help="Machine-local ARP settings used to load cloud credentials securely.",
    )
    parser.add_argument("--pod-id", default="")
    parser.add_argument(
        "--gpu-types",
        default=(
            "NVIDIA RTX PRO 4500 Blackwell,NVIDIA GeForce RTX 5090,"
            "NVIDIA RTX 6000 Ada Generation,NVIDIA L40S,NVIDIA A40,"
            "NVIDIA RTX A6000,NVIDIA GeForce RTX 4090"
        ),
    )
    parser.add_argument("--image", default="runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404")
    parser.add_argument("--volume-gb", type=int, default=100)
    parser.add_argument("--storage-mode", choices=("network", "pod"), default="network")
    parser.add_argument("--data-center-id", default="EU-RO-1")
    parser.add_argument("--network-volume-id", default="")
    parser.add_argument("--idle-minutes", type=int, default=0)
    parser.add_argument("--huggingface-token", default="", help=argparse.SUPPRESS)
    parser.add_argument("--expected-output", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    return args


if __name__ == "__main__":
    raise SystemExit(run(build_parser()))
