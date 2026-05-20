from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from comfy_api import extract_output_files, node_by_id, queue_prompt, set_widget, wait_for_comfy, wait_for_prompt, workflow_to_prompt
from common import ROOT, file_fingerprint, resolve_path, root_relative, resumable_output, write_signature, safe_stem
from dependency_manager import ensure_outpaint_models
from prepare_outpaint_input import default_output as default_prepared_output
from prepare_outpaint_input import even, parse_aspect, probe_video


DEFAULT_WORKFLOW = ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json"
DEFAULT_COMFY_DIR = ROOT / "tools" / "comfyui"


def aspect_slug(value: str) -> str:
    return value.replace(":", "x").replace(".", "_")


def load_local_config() -> dict[str, str]:
    path = ROOT / ".ai_remaster_config.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return {str(key): str(value) for key, value in data.items() if value is not None}


def default_output(source: Path, aspect: str) -> Path:
    return ROOT / "intermediate" / "outpainted" / f"{safe_stem(source.name)}_{aspect_slug(aspect)}_outpainted.mp4"


def default_raw_output(source: Path, aspect: str) -> Path:
    return ROOT / "intermediate" / "outpainted" / f"{safe_stem(source.name)}_{aspect_slug(aspect)}_raw_comfy.mp4"


def prepared_for(source: Path, aspect: str, target_height: int | None) -> Path:
    info = probe_video(source)
    height = even(target_height or info["height"])
    width = even(height * parse_aspect(aspect))
    return default_prepared_output(source, width, height)


def run_command(command: list[str], dry_run: bool) -> None:
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def copy_to_comfy_input(source: Path, comfy_dir: Path) -> str:
    target_dir = comfy_dir / "input" / "arp_outpaint"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if not target.exists() or source.stat().st_mtime_ns != target.stat().st_mtime_ns or source.stat().st_size != target.stat().st_size:
        shutil.copy2(source, target)
    return f"arp_outpaint/{target.name}"


def set_widget_if_node(workflow: dict[str, Any], node_id: str | None, widget: str | int, value: Any) -> None:
    if not node_id:
        return
    set_widget(node_by_id(workflow, node_id), widget, value)


def patch_workflow(args, workflow: dict[str, Any], prepared: Path, comfy_dir: Path, output_prefix: str) -> dict[str, Any]:
    video_name = copy_to_comfy_input(prepared, comfy_dir)
    set_widget_if_node(workflow, args.load_video_node_id, args.video_widget, video_name)
    set_widget_if_node(workflow, args.positive_node_id, args.prompt_widget, args.prompt)
    set_widget_if_node(workflow, args.negative_node_id, args.prompt_widget, args.negative_prompt)
    set_widget_if_node(workflow, args.save_node_id, args.save_prefix_widget, output_prefix)

    for node_id in args.extra_save_node_id:
        set_widget_if_node(workflow, node_id, args.save_prefix_widget, output_prefix)

    model_patches = {
        "3940": ("0", "ltx-2.3-22b-dev-fp8.safetensors"),
        "4010": ("0", "ltx-2.3-22b-dev-fp8.safetensors"),
        "5023": ("0", "gemma_3_12B_it_fp8_scaled.safetensors"),
        "5011": ("0", "ltx-2.3-22b-ic-lora-outpaint.safetensors"),
        "4922": ("0", "ltx-2.3-22b-distilled-lora-384.safetensors"),
    }
    for node_id, (widget, value) in model_patches.items():
        try:
            set_widget(node_by_id(workflow, node_id), widget, value)
        except KeyError:
            pass

    # ARP prepares the black target canvas itself, so disable the workflow's demo padding node if present.
    try:
        pad_node = node_by_id(workflow, "5086")
        if isinstance(pad_node.get("widgets_values"), list) and len(pad_node["widgets_values"]) >= 4:
            pad_node["widgets_values"][0:4] = [0, 0, 0, 0]
    except KeyError:
        pass

    return workflow_to_prompt(workflow, args.output_node_id)


def raw_signature(args, workflow_path: Path, prepared: Path) -> dict[str, Any]:
    return {
        "version": 1,
        "tool": "outpaint_video.py/raw_comfy",
        "prepared": root_relative(prepared),
        "prepared_fingerprint": file_fingerprint(prepared),
        "workflow": root_relative(workflow_path),
        "workflow_fingerprint": file_fingerprint(workflow_path),
        "target_aspect": args.target_aspect,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "load_video_node_id": args.load_video_node_id,
        "save_node_id": args.save_node_id,
        "extra_save_node_id": args.extra_save_node_id,
        "output_node_id": args.output_node_id,
    }


def newest_output(files: list[Path]) -> Path:
    videos = [path for path in files if path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}]
    candidates = videos or files
    if not candidates:
        raise RuntimeError("ComfyUI completed but did not report an output file.")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def build_parser() -> argparse.ArgumentParser:
    config = load_local_config()
    parser = argparse.ArgumentParser(description="Run the LTX IC-LoRA outpainting stage end to end.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target-aspect", default="16:9")
    parser.add_argument("--target-height", type=int)
    parser.add_argument("--output")
    parser.add_argument("--raw-output")
    parser.add_argument("--workflow", default=str(DEFAULT_WORKFLOW))
    parser.add_argument("--comfy-dir", default=config.get("comfy_dir", str(DEFAULT_COMFY_DIR)))
    parser.add_argument("--comfy-url", default=config.get("comfy_url", "http://127.0.0.1:8188"))
    parser.add_argument("--comfy-output-root")
    parser.add_argument("--load-video-node-id", default="5060")
    parser.add_argument("--video-widget", default="video")
    parser.add_argument("--save-node-id", default="5076")
    parser.add_argument("--extra-save-node-id", action="append", default=["5069"])
    parser.add_argument("--save-prefix-widget", default="filename_prefix")
    parser.add_argument("--output-node-id", default="5076")
    parser.add_argument("--positive-node-id", default="2483")
    parser.add_argument("--negative-node-id", default="2612")
    parser.add_argument("--prompt-widget", default="0")
    parser.add_argument("--prompt", default="naturalistic period film footage, coherent background extension, preserve camera motion, realistic cinematic lighting")
    parser.add_argument("--negative-prompt", default="cartoon, game, 3d render, still image, static, warped geometry, flicker, smeared details")
    parser.add_argument("--black-lift", type=float, default=0.018)
    parser.add_argument("--gamma", type=float, default=1.06)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = resolve_path(args.source)
    workflow_path = resolve_path(args.workflow)
    comfy_dir = resolve_path(args.comfy_dir)
    comfy_output_root = resolve_path(args.comfy_output_root) if args.comfy_output_root else comfy_dir / "output"
    output = resolve_path(args.output) if args.output else default_output(source, args.target_aspect)
    raw_output = resolve_path(args.raw_output) if args.raw_output else default_raw_output(source, args.target_aspect)
    prepared = prepared_for(source, args.target_aspect, args.target_height)

    if not source.exists():
        raise FileNotFoundError(f"Source video not found: {source}")
    if not workflow_path.exists():
        raise FileNotFoundError(f"Outpainting workflow not found: {workflow_path}")
    if not (comfy_dir / "main.py").exists():
        raise FileNotFoundError(f"ComfyUI main.py not found: {comfy_dir / 'main.py'}")

    if not args.dry_run:
        ensure_outpaint_models(comfy_dir)

    prepare_command = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_outpaint_input.py"),
        "--source",
        str(source),
        "--target-aspect",
        args.target_aspect,
        "--black-lift",
        str(args.black_lift),
        "--gamma",
        str(args.gamma),
    ]
    if args.target_height:
        prepare_command += ["--target-height", str(args.target_height)]
    if args.force:
        prepare_command.append("--force")
    if args.dry_run:
        prepare_command.append("--dry-run")
    run_command(prepare_command, False)

    output_prefix = f"arp_outpaint/{safe_stem(source.name)}_{aspect_slug(args.target_aspect)}"
    print(f"Prepared Comfy input: {prepared}")
    if not args.dry_run:
        raw_sig = raw_signature(args, workflow_path, prepared)
        if not args.force and resumable_output(raw_output, raw_sig, video_like=prepared):
            print(f"Reuse raw Comfy render: {raw_output}")
        else:
            workflow = json.loads(workflow_path.read_text(encoding="utf-8-sig"))
            prompt = patch_workflow(args, workflow, prepared, comfy_dir, output_prefix)
            print(f"Waiting for ComfyUI at {args.comfy_url}...")
            wait_for_comfy(args.comfy_url, timeout_seconds=180, poll_seconds=args.poll_seconds)
            prompt_id = queue_prompt(args.comfy_url, prompt)
            print(f"Queued ComfyUI prompt: {prompt_id}")
            history = wait_for_prompt(args.comfy_url, prompt_id, args.poll_seconds)
            produced = newest_output(extract_output_files(history, comfy_output_root))
            raw_output.parent.mkdir(parents=True, exist_ok=True)
            tmp = raw_output.with_suffix(raw_output.suffix + ".partial")
            shutil.copy2(produced, tmp)
            tmp.replace(raw_output)
            write_signature(raw_output, raw_sig)
            print(f"Wrote raw Comfy render: {raw_output}")

    finalize_command = [
        sys.executable,
        str(ROOT / "scripts" / "finalize_outpaint_output.py"),
        "--source",
        str(raw_output),
        "--output",
        str(output),
        "--black-lift",
        str(args.black_lift),
        "--gamma",
        str(args.gamma),
    ]
    if args.force:
        finalize_command.append("--force")
    if args.dry_run:
        finalize_command.append("--dry-run")
    run_command(finalize_command, args.dry_run)
    print(f"Wrote outpainted video: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
