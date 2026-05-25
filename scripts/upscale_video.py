from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import ROOT, file_fingerprint, resolve_path, root_relative, safe_stem, resumable_output, video_info, write_signature


def default_output(source: Path, method: str, scale: int) -> Path:
    return ROOT / "output" / "upscaled" / f"{safe_stem(source.name)}_{method}_x{scale}.mp4"


def signature(args: argparse.Namespace, source: Path, output_width: int, output_height: int) -> dict[str, Any]:
    repo_dir = resolve_path(args.realbasicvsr_repo) if args.realbasicvsr_repo else default_repo_dir()
    return {
        "version": 1,
        "tool": "upscale_video.py",
        "method": args.method,
        "source": root_relative(source),
        "source_fingerprint": file_fingerprint(source),
        "scale": args.scale,
        "output_width": output_width,
        "output_height": output_height,
        "realbasicvsr_repo": root_relative(repo_dir),
        "python_executable": resolve_upscaler_python(args.python_executable),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "max_seq_len": args.max_seq_len,
        "fps": args.fps,
    }


def default_repo_dir() -> Path:
    return ROOT / "tools" / "realbasicvsr"


def default_config(repo: Path) -> Path:
    return repo / "configs" / "realbasicvsr_x4.py"


def default_checkpoint(repo: Path) -> Path:
    checkpoint_dir = repo / "checkpoints"
    matches = sorted(checkpoint_dir.glob("*.pth")) if checkpoint_dir.exists() else []
    return matches[0] if matches else checkpoint_dir / "RealBasicVSR_x4.pth"


def default_realbasicvsr_python() -> Path:
    if sys.platform == "win32":
        return ROOT / ".venv-realbasicvsr" / "Scripts" / "python.exe"
    return ROOT / ".venv-realbasicvsr" / "bin" / "python"


def resolve_upscaler_python(value: str) -> str:
    if value:
        return str(resolve_path(value))
    candidate = default_realbasicvsr_python()
    return str(candidate) if candidate.exists() else sys.executable


def ensure_default_checkpoint(path: Path) -> None:
    if path.exists():
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise FileNotFoundError(
            f"RealBasicVSR checkpoint was not found: {path}. Install huggingface_hub or download "
            "akhaliq/RealBasicVSR_x4/RealBasicVSR_x4.pth into that folder."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading RealBasicVSR checkpoint: {path}", flush=True)
    downloaded = hf_hub_download(
        repo_id="akhaliq/RealBasicVSR_x4",
        filename="RealBasicVSR_x4.pth",
        local_dir=str(path.parent),
    )
    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != path.resolve():
        shutil.copy2(downloaded_path, path)


def ensure_realbasicvsr_runtime(python_executable: str) -> None:
    probe = subprocess.run(
        [
            python_executable,
            "-c",
            "import mmcv, mmedit",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        return
    details = (probe.stderr or probe.stdout or "").strip()
    raise RuntimeError(
        "RealBasicVSR Python dependencies are missing from the selected Python environment. "
        "Run scripts/setup_realbasicvsr_env.py to create .venv-realbasicvsr, or set python_executable in Settings > "
        f"Upscaling Backend to a Python environment that can import mmcv and mmedit. Details: {details}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upscale a recomposited ARP video.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--method", choices=["realbasicvsr"], default="realbasicvsr")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--realbasicvsr-repo", default="")
    parser.add_argument("--python-executable", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--max-seq-len", type=int, default=0)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def realbasicvsr_command(args: argparse.Namespace, source: Path, output: Path) -> list[str]:
    repo = resolve_path(args.realbasicvsr_repo) if args.realbasicvsr_repo else default_repo_dir()
    script = repo / "inference_realbasicvsr.py"
    if not script.exists():
        raise FileNotFoundError(
            f"RealBasicVSR was not found at {script}. Clone https://github.com/ckkelvinchan/RealBasicVSR "
            "into tools/realbasicvsr or set the RealBasicVSR repo path in the Upscaling tab."
        )

    config = resolve_path(args.config) if args.config else default_config(repo)
    checkpoint = resolve_path(args.checkpoint) if args.checkpoint else default_checkpoint(repo)
    if not config.exists():
        raise FileNotFoundError(f"RealBasicVSR config was not found: {config}")
    if not args.checkpoint:
        ensure_default_checkpoint(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"RealBasicVSR checkpoint was not found: {checkpoint}. Download the x4 checkpoint into {checkpoint.parent} "
            "or set the checkpoint path in the Upscaling tab."
        )

    fps = args.fps or float(video_info(source)["fps"])
    python_executable = resolve_upscaler_python(args.python_executable)
    ensure_realbasicvsr_runtime(python_executable)

    command = [
        python_executable,
        str(script),
        str(config),
        str(checkpoint),
        str(source),
        str(output),
        f"--fps={fps:.8f}",
    ]
    if args.max_seq_len:
        command.append(f"--max-seq-len={args.max_seq_len}")
    return command


def run(args: argparse.Namespace) -> int:
    source = resolve_path(args.input)
    if not source.exists():
        raise FileNotFoundError(f"Input video not found for upscaling: {source}")

    info = video_info(source)
    width = int(info["width"]) * max(1, int(args.scale))
    height = int(info["height"]) * max(1, int(args.scale))
    output = resolve_path(args.output) if args.output else default_output(source, args.method, args.scale)
    sig = signature(args, source, width, height)

    if not args.force and resumable_output(output, sig, video_like=source, width=width, height=height):
        print(f"Reuse upscaled video: {output}", flush=True)
        return 0
    if args.dry_run:
        print(f"Would upscale {source} -> {output} using {args.method}", flush=True)
        print(" ".join(realbasicvsr_command(args, source, output)), flush=True)
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    if partial.exists():
        partial.unlink()

    command = realbasicvsr_command(args, source, partial)
    print(" ".join(command), flush=True)
    result = subprocess.run(command, check=False, cwd=resolve_path(args.realbasicvsr_repo) if args.realbasicvsr_repo else default_repo_dir())
    if result.returncode != 0:
        if partial.exists():
            partial.unlink()
        raise subprocess.CalledProcessError(result.returncode, command)

    if not partial.exists():
        raise RuntimeError(f"RealBasicVSR finished but did not create expected output: {partial}")
    if output.exists():
        output.unlink()
    shutil.move(str(partial), str(output))
    write_signature(output, sig)
    print(f"Wrote upscaled video: {output}", flush=True)
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
