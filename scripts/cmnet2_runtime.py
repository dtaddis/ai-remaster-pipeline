from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


MODEL_NAME = "DINOv2FeatureV6_LocalAtten_s2_154000.pth"


def resolve_model(runtime_dir: Path, explicit: str = "", comfy_dir: str = "") -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(runtime_dir / "weights" / MODEL_NAME)
    if comfy_dir:
        candidates.append(Path(comfy_dir) / "custom_nodes" / "reference-video-colorization" / "checkpoints" / MODEL_NAME)
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        if resolved.is_file():
            return resolved
    checked = "\n  - ".join(str(path.expanduser().resolve(strict=False)) for path in candidates)
    raise FileNotFoundError(
        "CMNET2's ColorMNet checkpoint is missing. Rerun install_windows.bat with model downloads enabled."
        + (f"\nChecked:\n  - {checked}" if checked else "")
    )


def ordered_references(references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order real reference anchors by source position without losing manifest order ties."""
    indexed = list(enumerate(references))
    indexed.sort(
        key=lambda pair: (
            pair[1].get("selected_frame") is None,
            int(pair[1].get("selected_frame") or 0),
            pair[0],
        )
    )
    return [item for _index, item in indexed]


class CMNet2Session:
    """One loaded CMNET2 network reused across independently reset shots."""

    def __init__(
        self,
        runtime_dir: Path,
        model_path: Path,
        *,
        top_k: int = 30,
        mem_every: int = 5,
        max_frames: int = 1000,
    ) -> None:
        runtime_dir = runtime_dir.resolve(strict=False)
        if not (runtime_dir / "colormnet" / "colormnet_render.py").is_file():
            raise FileNotFoundError(f"CMNET2 runtime is incomplete: {runtime_dir}")
        if str(runtime_dir) not in sys.path:
            sys.path.insert(0, str(runtime_dir))

        try:
            import torch
            from colormnet.colormnet_render import ColorMNetRender
        except ImportError as exc:
            raise RuntimeError(
                "CMNET2 dependencies are missing. Rerun install_windows.bat to install PyTorch, torchvision, "
                "OpenCV, Pillow, scikit-image, einops and tqdm."
            ) from exc

        hub_dir = runtime_dir / "models"
        if hub_dir.is_dir():
            torch.hub.set_dir(str(hub_dir))
        if not torch.cuda.is_available():
            raise RuntimeError("CMNET2 requires a CUDA-capable NVIDIA GPU, but PyTorch cannot access CUDA.")

        torch.backends.cudnn.benchmark = True
        torch.set_grad_enabled(False)
        print(f"Loading CMNET2 model: {model_path}", flush=True)
        self._torch = torch
        self._colorizer = ColorMNetRender(
            vid_length=max(1, int(max_frames)),
            enable_resize=False,
            encode_mode=1,
            max_memory_frames=max(1, int(max_frames)),
            reset_on_ref_update=False,
            top_k=max(1, int(top_k)),
            mem_every=max(1, int(mem_every)),
            project_dir=str(runtime_dir),
            model_path=str(model_path),
        )

    def colorize_segment(
        self,
        source_video: Path,
        output: Path,
        references: list[dict[str, Any]],
        *,
        start_frame: int,
        end_frame: int,
        width: int,
        height: int,
        fps: float,
        ffmpeg: str,
        crf: int = 18,
    ) -> None:
        try:
            import cv2
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("CMNET2 runtime dependencies are missing; rerun install_windows.bat.") from exc

        anchors = ordered_references(references)
        if not anchors:
            raise RuntimeError("CMNET2 needs at least one colour reference for every enabled shot.")
        frame_count = max(1, int(end_frame) - int(start_frame))
        self._colorizer.reset_memory(frame_count)

        resampling = getattr(Image, "Resampling", Image).LANCZOS
        prepared = []
        print(f"Preloading {len(anchors)} CMNET2 reference anchor(s)...", flush=True)
        for anchor in anchors:
            path = Path(anchor["path"])
            with Image.open(path) as image:
                reference = image.convert("RGB")
                if reference.size != (width, height):
                    reference = reference.resize((width, height), resampling)
                reference = reference.copy()
            self._colorizer.preload_reference(reference)
            prepared.append(reference)
        print(f"CMNET2 permanent memory contains {self._colorizer.get_perm_mem_frame_count()} reference(s).", flush=True)

        capture = cv2.VideoCapture(str(source_video))
        if not capture.isOpened():
            raise RuntimeError(f"CMNET2 could not open source video: {source_video}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start_frame)))

        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:.12g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            str(max(0, min(51, int(crf)))),
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            str(partial),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        written = 0
        try:
            for local_index in range(frame_count):
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(
                        f"CMNET2 source ended after {written} of {frame_count} requested frames "
                        f"(source frame {start_frame + local_index})."
                    )
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
                rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self._colorizer.set_ref_frame(prepared[0] if local_index == 0 else None)
                result = self._colorizer.colorize_frame(ti=local_index, frame_i=rgb, lab_mode="gpu")
                bgr = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)
                if process.stdin is None:
                    raise RuntimeError("CMNET2 FFmpeg encoder pipe closed unexpectedly.")
                process.stdin.write(bgr.tobytes())
                written += 1
                percent = (written * 100) // frame_count
                print(
                    f"\rCMNET2 frames {written}/{frame_count} [{percent:3d}%]",
                    end="" if written < frame_count else "\n",
                    flush=True,
                )
        except BaseException:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            partial.unlink(missing_ok=True)
            raise
        finally:
            capture.release()
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass

        error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"CMNET2 FFmpeg encoding failed ({return_code}): {error.strip()}")
        if written != frame_count:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"CMNET2 wrote {written} frames; expected {frame_count}.")
        os.replace(partial, output)
