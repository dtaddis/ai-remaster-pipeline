from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from common import QWEN_IMAGE_EDIT_MODEL, ROOT
from model_paths import huggingface_cache_dir, resolve_comfy_model_path


FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"


@dataclass(frozen=True)
class HfModel:
    repo: str
    file: str
    destination: str


class HuggingFaceAccessError(RuntimeError):
    """A gated Hugging Face model could not be read with the active credentials."""


def huggingface_access_denied(exc: BaseException) -> bool:
    """Recognize Hub access failures without depending on a particular hub/httpx version."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        if getattr(response, "status_code", None) in {401, 403}:
            return True
        text = f"{type(current).__name__}: {current}".lower()
        if any(marker in text for marker in ("gatedrepoerror", "401 unauthorized", "403 forbidden", "access to model", "gated repo")):
            return True
        current = current.__cause__ or current.__context__
    return False


LTX_BASE_MODELS = [
    HfModel("QuantStack/LTX-2.3-GGUF", "LTX-2.3-distilled/LTX-2.3-distilled-Q4_K_M.gguf", "models/unet/LTX-2.3-distilled-Q4_K_M.gguf"),
    HfModel("Lightricks/LTX-2.3-fp8", "ltx-2.3-22b-dev-fp8.safetensors", "models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors"),
    HfModel("Comfy-Org/ltx-2", "split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors", "models/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors"),
    HfModel("Kijai/LTX2.3_comfy", "vae/LTX23_video_vae_bf16.safetensors", "models/vae/LTX23_video_vae_bf16.safetensors"),
    HfModel("Kijai/LTX2.3_comfy", "vae/LTX23_audio_vae_bf16.safetensors", "models/vae/LTX23_audio_vae_bf16.safetensors"),
]

OUTPAINT_MODELS = [
    *LTX_BASE_MODELS,
    HfModel("Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting", "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors", "models/loras/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"),
    HfModel("oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint", "ltx-2.3-22b-ic-lora-outpaint.safetensors", "models/loras/ltx-2.3-22b-ic-lora-outpaint.safetensors"),
]

DEFAULT_OUTPAINT_LORA = "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"
OUMOUMAD_OUTPAINT_LORA = "ltx-2.3-22b-ic-lora-outpaint.safetensors"
OUTPAINT_LORA_MODELS = {Path(model.destination).name: model for model in OUTPAINT_MODELS[len(LTX_BASE_MODELS):]}

OUTPAINT_DISTILLED_LORA_MODELS = [
    HfModel("Lightricks/LTX-2.3", "ltx-2.3-22b-distilled-lora-384-1.1.safetensors", "models/loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"),
]

CLEANUP_MODELS = [
    *LTX_BASE_MODELS,
    HfModel("oumoumad/ltx-2.3-dearchive-lora", "lora_weights_step_05000.safetensors", "models/loras/ltx-2.3-dearchive-lora.safetensors"),
]

QWEN_IMAGE_EDIT_MODELS = [
    HfModel("unsloth/Qwen-Image-Edit-2511-GGUF", QWEN_IMAGE_EDIT_MODEL, f"models/diffusion_models/{QWEN_IMAGE_EDIT_MODEL}"),
    HfModel("Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    HfModel("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", "models/vae/qwen_image_vae.safetensors"),
    HfModel("lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"),
]

# Music score: Stable Audio Open, loaded by ComfyUI core audio nodes from models/checkpoints.
# NOTE: stable-audio-open-1.0 is a *gated* repo - the user must accept its licence on Hugging
# Face and authenticate (`hf auth login` / HF_TOKEN) for the download to succeed. Audio models
# are therefore fetched in soft mode (see ensure_audio_models): a failure logs guidance and
# continues instead of aborting the run.
MUSIC_MODELS = [
    HfModel("stabilityai/stable-audio-open-1.0", "model.safetensors", "models/checkpoints/stable_audio_open_1.0.safetensors"),
    HfModel("google-t5/t5-base", "model.safetensors", "models/text_encoders/t5_base.safetensors"),
]

# Sound effects: MMAudio (video -> synchronized audio), kijai's ComfyUI-ready safetensors.
# These land in models/mmaudio/, where ComfyUI-MMAudio's loaders look by default.
SFX_MODELS = [
    HfModel("Kijai/MMAudio_safetensors", "mmaudio_large_44k_v2_fp16.safetensors", "models/mmaudio/mmaudio_large_44k_v2_fp16.safetensors"),
    HfModel("Kijai/MMAudio_safetensors", "mmaudio_vae_44k_fp16.safetensors", "models/mmaudio/mmaudio_vae_44k_fp16.safetensors"),
    HfModel("Kijai/MMAudio_safetensors", "mmaudio_synchformer_fp16.safetensors", "models/mmaudio/mmaudio_synchformer_fp16.safetensors"),
    HfModel("Kijai/MMAudio_safetensors", "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors", "models/mmaudio/apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors"),
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


def ensure_hf_models(comfy_dir: Path, models: list[HfModel], required: bool = True) -> None:
    """Download each model to comfy_dir/destination if missing.

    When ``required`` is False, a failed download (e.g. a gated repo without an access token)
    logs actionable guidance and continues instead of raising, so an optional/gated model does
    not abort the whole stage.
    """
    ensure_huggingface_hub()

    cache_root = huggingface_cache_dir(ROOT / ".cache" / "huggingface")
    cache_root.mkdir(parents=True, exist_ok=True)
    old_python_utf8 = os.environ.get("PYTHONUTF8")
    old_python_io = os.environ.get("PYTHONIOENCODING")
    old_progress = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    old_symlink_warning = os.environ.get("HF_HUB_DISABLE_SYMLINKS_WARNING")
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    try:
        for model in models:
            destination = resolve_comfy_model_path(comfy_dir, model.destination)
            print(f"Checking model: {model.repo}/{model.file}", flush=True)
            if destination.exists():
                print(f"Model already exists: {destination}", flush=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                size = remote_file_size(model.repo, model.file)
                size_text = f" ({format_bytes(size)})" if size else ""
                print(f"Downloading model: {model.repo}/{model.file}{size_text}", flush=True)
                downloaded = download_hf_file(model.repo, model.file, cache_root, size)
                copy_model_file(downloaded, destination, size or downloaded.stat().st_size)
                print(f"Downloaded: {destination}", flush=True)
            except Exception as exc:
                if required:
                    if huggingface_access_denied(exc):
                        model_url = f"https://huggingface.co/{model.repo}"
                        raise HuggingFaceAccessError(
                            "Approve the official LTX outpainting model download on Hugging Face. "
                            f"ARP will open {model_url}; approve access there, then run Outpainting again."
                        ) from None
                    raise
                print(f"Warning: could not auto-download {model.repo}/{model.file}: {exc}", flush=True)
                print(
                    f"Warning: place the file manually at {destination}, or accept the model "
                    f"licence on Hugging Face and authenticate (hf auth login / HF_TOKEN), then "
                    f"retry. See docs/installer-model-sources.md. Continuing without it.",
                    flush=True,
                )
    finally:
        restore_env("PYTHONUTF8", old_python_utf8)
        restore_env("PYTHONIOENCODING", old_python_io)
        restore_env("HF_HUB_DISABLE_PROGRESS_BARS", old_progress)
        restore_env("HF_HUB_DISABLE_SYMLINKS_WARNING", old_symlink_warning)


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def remote_file_size(repo: str, filename: str) -> int:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id=repo, files_metadata=True)
        for sibling in info.siblings or []:
            if sibling.rfilename == filename and sibling.size:
                return int(sibling.size)
    except Exception:
        pass

    try:
        from huggingface_hub import hf_hub_url

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


def download_hf_file(repo: str, filename: str, cache_root: Path, total_size: int = 0) -> Path:
    from huggingface_hub import hf_hub_download

    kwargs = {"repo_id": repo, "filename": filename, "cache_dir": cache_root}
    stop_progress = threading.Event()
    progress_thread = None
    if total_size > 0:
        baseline = cache_file_sizes(cache_root)
        progress_thread = threading.Thread(
            target=report_hf_download_progress,
            args=(cache_root, total_size, baseline, stop_progress),
            daemon=True,
        )
        progress_thread.start()
    try:
        downloaded = Path(hf_hub_download(**kwargs))
        if total_size > 0:
            print("Download progress: 100%", flush=True)
        return downloaded
    finally:
        stop_progress.set()
        if progress_thread:
            progress_thread.join(timeout=1)


def cache_file_sizes(cache_root: Path) -> dict[Path, int]:
    sizes: dict[Path, int] = {}
    if not cache_root.exists():
        return sizes
    for path in cache_root.rglob("*"):
        if path.is_file():
            try:
                sizes[path] = path.stat().st_size
            except OSError:
                pass
    return sizes


def report_hf_download_progress(cache_root: Path, total_size: int, baseline: dict[Path, int], stop: threading.Event) -> None:
    last_percent = -1
    last_downloaded = 0
    last_update_at = time.monotonic()
    last_heartbeat_at = 0.0
    started_at = time.monotonic()
    while not stop.wait(2):
        downloaded = estimate_downloaded_bytes(cache_root, baseline)
        if downloaded <= 0:
            continue
        now = time.monotonic()
        percent = max(0, min(99, int((downloaded / total_size) * 100)))
        moved = downloaded > last_downloaded
        if moved:
            last_update_at = now
            last_downloaded = downloaded
        if percent > last_percent:
            eta = download_eta(started_at, downloaded, total_size)
            eta_text = f", ETA {eta}" if eta else ""
            print(f"Download progress: {percent}%{eta_text}", flush=True)
            last_percent = percent
            last_heartbeat_at = now
        elif now - last_heartbeat_at >= 30:
            note = "still working" if now - last_update_at < 120 else "waiting for network or Hugging Face"
            print(f"Download progress: {percent}%, {note}", flush=True)
            last_heartbeat_at = now


def download_eta(started_at: float, downloaded: int, total_size: int) -> str:
    elapsed = max(0.0, time.monotonic() - started_at)
    if elapsed <= 0 or downloaded <= 0 or downloaded >= total_size:
        return ""
    bytes_per_second = downloaded / elapsed
    if bytes_per_second <= 0:
        return ""
    remaining = max(0.0, (total_size - downloaded) / bytes_per_second)
    return format_duration(remaining)


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def copy_model_file(source: Path, destination: Path, total_size: int) -> None:
    print(f"Installing model: {destination}", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_destination = destination.with_name(destination.name + ".partial")
    copied = 0
    last_percent = -1
    last_heartbeat_at = time.monotonic()
    with source.open("rb") as src, temp_destination.open("wb") as dst:
        while True:
            chunk = src.read(16 * 1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            copied += len(chunk)
            if total_size > 0:
                percent = max(0, min(99, int((copied / total_size) * 100)))
                now = time.monotonic()
                if percent > last_percent or now - last_heartbeat_at >= 30:
                    print(f"Install progress: {percent}%", flush=True)
                    last_percent = percent
                    last_heartbeat_at = now
    shutil.copystat(source, temp_destination)
    if destination.exists():
        destination.unlink()
    temp_destination.replace(destination)
    print("Install progress: 100%", flush=True)


def estimate_downloaded_bytes(cache_root: Path, baseline: dict[Path, int]) -> int:
    best = 0
    newest_incomplete: tuple[float, int] | None = None
    if not cache_root.exists():
        return best
    for path in cache_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            current = path.stat().st_size
            mtime = path.stat().st_mtime
        except OSError:
            continue
        previous = baseline.get(path, 0)
        growth = max(0, current - previous)
        if path.name.endswith(".incomplete"):
            downloaded = max(growth, current)
            if newest_incomplete is None or mtime > newest_incomplete[0]:
                newest_incomplete = (mtime, downloaded)
            continue
        best = max(best, growth)
    if newest_incomplete is not None:
        return newest_incomplete[1]
    return best


def ensure_outpaint_models(comfy_dir: Path, include_distilled_lora: bool = False, outpaint_lora: str = DEFAULT_OUTPAINT_LORA) -> None:
    models = [*LTX_BASE_MODELS]
    selected = OUTPAINT_LORA_MODELS.get(Path(outpaint_lora.replace("\\", "/")).name)
    if selected is not None:
        models.append(selected)
    if include_distilled_lora:
        models.extend(OUTPAINT_DISTILLED_LORA_MODELS)
    ensure_hf_models(comfy_dir, models)


def ensure_cleanup_models(comfy_dir: Path) -> None:
    ensure_hf_models(comfy_dir, CLEANUP_MODELS)


def ensure_qwen_image_edit_models(comfy_dir: Path) -> None:
    ensure_hf_models(comfy_dir, QWEN_IMAGE_EDIT_MODELS)


def ensure_audio_models(comfy_dir: Path, music: bool = True, sfx: bool = True) -> None:
    """Fetch the soundtrack models on first use. Soft mode: failures (e.g. the gated Stable
    Audio repo) log guidance and continue rather than aborting the Create Audio Track phase."""
    models: list[HfModel] = []
    if music:
        models += MUSIC_MODELS
    if sfx:
        models += SFX_MODELS
    if models:
        ensure_hf_models(comfy_dir, models, required=False)
