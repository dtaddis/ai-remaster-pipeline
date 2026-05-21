from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from common import ROOT


FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"


@dataclass(frozen=True)
class HfModel:
    repo: str
    file: str
    destination: str


OUTPAINT_MODELS = [
    HfModel("QuantStack/LTX-2.3-GGUF", "LTX-2.3-distilled/LTX-2.3-distilled-Q4_K_M.gguf", "models/unet/LTX-2.3-distilled-Q4_K_M.gguf"),
    HfModel("Lightricks/LTX-2.3-fp8", "ltx-2.3-22b-dev-fp8.safetensors", "models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors"),
    HfModel("Comfy-Org/ltx-2", "split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors", "models/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors"),
    HfModel("Kijai/LTX2.3_comfy", "vae/LTX23_video_vae_bf16.safetensors", "models/vae/LTX23_video_vae_bf16.safetensors"),
    HfModel("Kijai/LTX2.3_comfy", "vae/LTX23_audio_vae_bf16.safetensors", "models/vae/LTX23_audio_vae_bf16.safetensors"),
    HfModel("oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint", "ltx-2.3-22b-ic-lora-outpaint.safetensors", "models/loras/ltx-2.3-22b-ic-lora-outpaint.safetensors"),
]

QWEN_IMAGE_EDIT_MODELS = [
    HfModel("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors", "models/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"),
    HfModel("Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    HfModel("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", "models/vae/qwen_image_vae.safetensors"),
    HfModel("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors", "models/loras/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"),
]


def ensure_ffmpeg_tools() -> tuple[Path, Path]:
    tool_dir = ROOT / ".cache" / "tools" / "ffmpeg"
    ffmpeg = tool_dir / "ffmpeg.exe"
    ffprobe = tool_dir / "ffprobe.exe"
    if ffmpeg.exists() and ffprobe.exists():
        return ffmpeg, ffprobe
    if os.name != "nt":
        found_ffmpeg = shutil.which("ffmpeg")
        found_ffprobe = shutil.which("ffprobe")
        if found_ffmpeg and found_ffprobe:
            return Path(found_ffmpeg), Path(found_ffprobe)
        raise FileNotFoundError("ffmpeg/ffprobe were not found. Automatic FFmpeg download is currently implemented for Windows.")

    archive = ROOT / ".cache" / "downloads" / "ffmpeg-release-essentials.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    tool_dir.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print(f"Downloading FFmpeg essentials from {FFMPEG_URL}")
        urllib.request.urlretrieve(FFMPEG_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            name = Path(member).name.lower()
            if name in {"ffmpeg.exe", "ffprobe.exe"} and "/bin/" in member.replace("\\", "/").lower():
                target = tool_dir / Path(member).name
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    if not ffmpeg.exists() or not ffprobe.exists():
        raise FileNotFoundError("Downloaded FFmpeg archive did not contain ffmpeg.exe and ffprobe.exe.")
    return ffmpeg, ffprobe


def ensure_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
        return
    except ImportError:
        pass
    print("Installing huggingface_hub for on-demand model downloads.")
    subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)


def ensure_hf_models(comfy_dir: Path, models: list[HfModel]) -> None:
    ensure_huggingface_hub()
    from huggingface_hub import hf_hub_download, hf_hub_url
    import urllib.request

    cache_root = ROOT / ".cache" / "huggingface"
    cache_root.mkdir(parents=True, exist_ok=True)
    old_python_utf8 = os.environ.get("PYTHONUTF8")
    old_python_io = os.environ.get("PYTHONIOENCODING")
    old_progress = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        for model in models:
            destination = comfy_dir / model.destination
            print(f"Checking model: {model.repo}/{model.file}", flush=True)
            if destination.exists():
                print(f"Model already exists: {destination}", flush=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            size = remote_file_size(model.repo, model.file)
            size_text = f" ({format_bytes(size)})" if size else ""
            print(f"Downloading model: {model.repo}/{model.file}{size_text}", flush=True)
            downloaded = Path(hf_hub_download(repo_id=model.repo, filename=model.file, local_dir=cache_root))
            shutil.copy2(downloaded, destination)
            print(f"Downloaded: {destination}", flush=True)
    finally:
        restore_env("PYTHONUTF8", old_python_utf8)
        restore_env("PYTHONIOENCODING", old_python_io)
        restore_env("HF_HUB_DISABLE_PROGRESS_BARS", old_progress)


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def remote_file_size(repo: str, filename: str) -> int:
    try:
        request = urllib.request.Request(hf_hub_url(repo, filename), method="HEAD")
        with urllib.request.urlopen(request, timeout=10) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else 0
    except Exception:
        return 0


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def ensure_outpaint_models(comfy_dir: Path) -> None:
    ensure_hf_models(comfy_dir, OUTPAINT_MODELS)


def ensure_qwen_image_edit_models(comfy_dir: Path) -> None:
    ensure_hf_models(comfy_dir, QWEN_IMAGE_EDIT_MODELS)
