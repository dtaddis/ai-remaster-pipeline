from __future__ import annotations

"""Cloud routing shared by the GUI.

The GUI deliberately keeps provider credentials in the machine-local settings file.  A project
bundle may contain the selected route, but never the keys used to execute it.
"""

from pathlib import Path
import urllib.request

from .config import ROOT, SCRIPTS, SETTINGS_FILE


CLOUD_STAGE_KEYS = {
    "cleanup", "stabilize", "outpaint", "shots", "references", "colour",
    "recomp", "audio", "upscale",
}

SECRET_SETTING_KEYS = {
    "openai_api_key",
    "runpod_api_key",
    # Keep protecting credentials written by feature branches even when this
    # branch does not expose their provider UI.
    "minimax_api_key",
    "kie_api_key",
    "huggingface_token",
}


def stage_uses_runpod(stage_key: str, values: dict[str, str]) -> bool:
    return values.get("compute", "local") == "runpod"


def wrap_runpod_command(
    command: list[str],
    *,
    stage_key: str,
    expected_outputs: list[str],
    cloud: dict[str, str],
) -> list[str]:
    wrapped = [
        command[0], "-u", str(SCRIPTS / "runpod_stage.py"),
        "--stage", stage_key,
        "--settings-file", str(SETTINGS_FILE),
        "--pod-id", cloud.get("runpod_pod_id", ""),
        "--gpu-types", cloud.get(
            "runpod_gpu_types",
            "NVIDIA RTX PRO 4500 Blackwell,NVIDIA GeForce RTX 5090,NVIDIA RTX 6000 Ada Generation,NVIDIA L40S,NVIDIA A40,NVIDIA RTX A6000,NVIDIA GeForce RTX 4090",
        ),
        "--image", cloud.get("runpod_image", "runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404"),
        "--storage-mode", cloud.get("runpod_storage_mode", "network"),
        "--volume-gb", cloud.get("runpod_volume_gb", "100"),
        "--data-center-id", cloud.get("runpod_data_center_id", "EU-RO-1"),
        "--idle-minutes", cloud.get("runpod_idle_minutes", "0"),
    ]
    if cloud.get("runpod_network_volume_id", ""):
        wrapped.extend(["--network-volume-id", cloud["runpod_network_volume_id"]])
    for output in expected_outputs:
        if output:
            wrapped.extend(["--expected-output", output])
    wrapped.extend(["--", *command])
    return wrapped


def wrap_master_command(command: list[str], outputs: list[str], profile: str) -> list[str]:
    wrapped = [command[0], "-u", str(SCRIPTS / "master_encode.py"), "--profile", profile]
    for output in outputs:
        if output:
            wrapped.extend(["--output", output])
    wrapped.extend(["--", *command])
    return wrapped


def stop_runpod_pod(api_key: str, pod_id: str) -> None:
    """Best-effort billing stop used when the GUI force-stops a cloud subprocess."""
    request = urllib.request.Request(
        f"https://rest.runpod.io/v1/pods/{pod_id}/stop",
        data=b"{}",
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AI-Remaster-Pipeline/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()


def is_workspace_path(path: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(ROOT.resolve(strict=False))
        return True
    except (OSError, ValueError):
        return False
