from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import ROOT


def run(command: list[str], cwd: Path = ROOT) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def uv_executable() -> str:
    candidate = Path(sys.executable).with_name("uv.exe" if os.name == "nt" else "uv")
    if candidate.exists():
        return str(candidate)
    return "uv"


def venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def ensure_pip(python: Path) -> None:
    probe = subprocess.run([str(python), "-m", "pip", "--version"], check=False, capture_output=True, text=True)
    if probe.returncode == 0:
        return
    run([str(python), "-m", "ensurepip", "--upgrade"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a Python environment for RealBasicVSR.")
    parser.add_argument("--venv", default=str(ROOT / ".venv-realbasicvsr"))
    parser.add_argument("--python", default="3.10")
    parser.add_argument("--torch-index-url", default="https://download.pytorch.org/whl/cu117")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    venv = Path(args.venv)
    python = venv_python(venv)
    uv = uv_executable()

    if args.force and venv.exists():
        raise RuntimeError(f"Refusing to delete existing environment automatically: {venv}")

    if not python.exists():
        run([uv, "python", "install", args.python])
        run([uv, "venv", "--seed", "--python", args.python, str(venv)])

    ensure_pip(python)
    run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "numpy<2",
            "torch==1.13.1+cu117",
            "torchvision==0.14.1+cu117",
            "--extra-index-url",
            args.torch_index_url,
        ]
    )
    run([str(python), "-m", "pip", "install", "openmim==0.3.9"])
    run([str(python), "-m", "mim", "install", "mmcv-full==1.7.1"])
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "mmedit==0.16.0",
            "opencv-python<4.9",
            "imageio-ffmpeg",
            "scikit-image",
            "tqdm",
        ]
    )
    run([str(python), "-m", "pip", "install", "--force-reinstall", "numpy<2"])
    run([str(python), "-c", "import mmcv, mmedit; print('RealBasicVSR runtime OK')"])
    print(f"RealBasicVSR Python: {python}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
