from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
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


def target_size(source: Path, aspect: str, target_height: int | None) -> tuple[int, int]:
    info = probe_video(source)
    height = even(target_height or 720)
    width = even(height * parse_aspect(aspect))
    return width, height


def default_output(source: Path, aspect: str, target_height: int | None) -> Path:
    width, height = target_size(source, aspect, target_height)
    return ROOT / "intermediate" / "outpainted" / f"{safe_stem(source.name)}_{aspect_slug(aspect)}_{width}x{height}_outpainted.mp4"


def default_raw_output(source: Path, aspect: str, target_height: int | None) -> Path:
    width, height = target_size(source, aspect, target_height)
    return ROOT / "intermediate" / "outpainted" / f"{safe_stem(source.name)}_{aspect_slug(aspect)}_{width}x{height}_raw_comfy.mp4"


def prepared_for(source: Path, aspect: str, target_height: int | None) -> Path:
    info = probe_video(source)
    height = even(target_height or info["height"])
    width = even(height * parse_aspect(aspect))
    return default_prepared_output(source, width, height)


def run_command(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def find_ffmpeg() -> str:
    candidates = [
        ROOT / ".cache" / "tools" / "ffmpeg" / "ffmpeg.exe",
        Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
        Path("ffmpeg"),
    ]
    for candidate in candidates:
        try:
            subprocess.run([str(candidate), "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return str(candidate)
        except Exception:
            continue
    raise FileNotFoundError("ffmpeg was not found. Re-run install_windows.bat to install ARP's local FFmpeg copy.")


def copy_to_comfy_input(source: Path, comfy_dir: Path) -> str:
    target_dir = comfy_dir / "input" / "arp_outpaint"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if not target.exists() or source.stat().st_mtime_ns != target.stat().st_mtime_ns or source.stat().st_size != target.stat().st_size:
        shutil.copy2(source, target)
    return f"arp_outpaint/{target.name}"


def copy_reference_frame_to_comfy_input(source: Path, comfy_dir: Path) -> str:
    import cv2

    target_dir = comfy_dir / "input" / "arp_outpaint"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{source.stem}_reference.png"
    if not target.exists() or source.stat().st_mtime_ns > target.stat().st_mtime_ns:
        cap = cv2.VideoCapture(str(source))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Could not extract reference frame from: {source}")
        cv2.imwrite(str(target), frame)
    return f"arp_outpaint/{target.name}"


def set_widget_if_node(workflow: dict[str, Any], node_id: str | None, widget: str | int, value: Any) -> None:
    if not node_id:
        return
    set_widget(node_by_id(workflow, node_id), widget, value)


def add_or_replace_node(workflow: dict[str, Any], node: dict[str, Any]) -> None:
    nodes = workflow.setdefault("nodes", [])
    node_id = str(node["id"])
    for index, existing in enumerate(nodes):
        if str(existing.get("id")) == node_id:
            nodes[index] = node
            return
    nodes.append(node)


def patch_link(workflow: dict[str, Any], link_id: int, source_id: int, source_slot: int, target_id: int, target_slot: int, link_type: str) -> None:
    links = workflow.setdefault("links", [])
    for link in links:
        if int(link[0]) == link_id:
            link[1:6] = [source_id, source_slot, target_id, target_slot, link_type]
            return
    links.append([link_id, source_id, source_slot, target_id, target_slot, link_type])


def set_input_link(workflow: dict[str, Any], node_id: str, input_name: str, link_id: int) -> None:
    node = node_by_id(workflow, node_id)
    for item in node.get("inputs", []):
        if item.get("name") == input_name:
            item["link"] = link_id
            return
    node.setdefault("inputs", []).append({"name": input_name, "link": link_id})


def clear_input_link(workflow: dict[str, Any], node_id: str, input_name: str) -> None:
    node = node_by_id(workflow, node_id)
    for item in node.get("inputs", []):
        if item.get("name") == input_name:
            item["link"] = None
            return


def ensure_widget_input(node: dict[str, Any], name: str, input_type: str = "COMBO") -> None:
    for item in node.setdefault("inputs", []):
        if item.get("name") == name:
            item.setdefault("widget", {"name": name})
            return
    node["inputs"].append({"name": name, "type": input_type, "widget": {"name": name}})


def patch_lightweight_gguf(workflow: dict[str, Any], args) -> None:
    model_node = node_by_id(workflow, "3940")
    model_node["type"] = "UnetLoaderGGUF"
    model_node["title"] = "Unet Loader (GGUF)"
    model_node["inputs"] = [{"name": "unet_name", "type": "COMBO", "widget": {"name": "unet_name"}}]
    model_node["widgets_values"] = [args.gguf_model]
    model_node["outputs"] = [{"name": "MODEL", "type": "MODEL", "links": [13217]}]

    add_or_replace_node(
        workflow,
        {
            "id": 9001,
            "type": "VAELoader",
            "title": "LTX 2.3 Video VAE",
            "mode": 0,
            "inputs": [{"name": "vae_name", "type": "COMBO", "widget": {"name": "vae_name"}}],
            "outputs": [{"name": "VAE", "type": "VAE", "links": [13279, 13348, 13405]}],
            "widgets_values": [args.video_vae],
        },
    )
    patch_link(workflow, 13217, 3940, 0, 5011, 0, "MODEL")
    patch_link(workflow, 13279, 9001, 0, 3159, 0, "VAE")
    patch_link(workflow, 13348, 9001, 0, 4851, 1, "VAE")
    patch_link(workflow, 13405, 9001, 0, 5012, 2, "VAE")
    set_input_link(workflow, "5011", "model", 13217)
    lora_node = node_by_id(workflow, "5011")
    ensure_widget_input(lora_node, "lora_name")
    ensure_widget_input(lora_node, "strength_model", "FLOAT")
    set_widget(lora_node, "0", args.outpaint_lora)
    set_widget(lora_node, "1", 1.0)
    audio_vae_node = node_by_id(workflow, "4010")
    ensure_widget_input(audio_vae_node, "ckpt_name")
    set_widget(audio_vae_node, "0", args.audio_vae_checkpoint)
    text_node = node_by_id(workflow, "5023")
    ensure_widget_input(text_node, "text_encoder")
    ensure_widget_input(text_node, "ckpt_name")
    ensure_widget_input(text_node, "device")
    set_widget(text_node, "0", args.text_encoder)
    set_widget(text_node, "1", args.text_encoder_checkpoint)


def patch_workflow(args, workflow: dict[str, Any], prepared: Path, comfy_dir: Path, output_prefix: str) -> dict[str, Any]:
    video_name = copy_to_comfy_input(prepared, comfy_dir)
    image_name = copy_reference_frame_to_comfy_input(prepared, comfy_dir)
    prepared_info = probe_video(prepared)
    set_widget_if_node(workflow, args.load_video_node_id, args.video_widget, video_name)
    try:
        image_node = node_by_id(workflow, "2004")
        ensure_widget_input(image_node, "image")
        set_widget(image_node, "0", image_name)
    except KeyError:
        pass
    set_widget_if_node(workflow, args.positive_node_id, args.prompt_widget, args.prompt)
    set_widget_if_node(workflow, args.negative_node_id, args.prompt_widget, args.negative_prompt)
    set_widget_if_node(workflow, args.save_node_id, args.save_prefix_widget, output_prefix)

    for node_id in args.extra_save_node_id:
        set_widget_if_node(workflow, node_id, args.save_prefix_widget, output_prefix)

    # Avoid depending on the optional ComfyMath CM_FloatToInt node; the audio latent node accepts
    # a normal integer widget value when its frame_rate link is cleared.
    try:
        clear_input_link(workflow, "3980", "frame_rate")
        audio_latent_node = node_by_id(workflow, "3980")
        set_widget(audio_latent_node, "1", int(round(float(prepared_info.get("fps") or 24))))
    except KeyError:
        pass

    try:
        latent_video_node = node_by_id(workflow, "3059")
        for input_name in ("width", "height", "length"):
            clear_input_link(workflow, "3059", input_name)
        set_widget(latent_video_node, "0", int(prepared_info["width"]))
        set_widget(latent_video_node, "1", int(prepared_info["height"]))
        set_widget(latent_video_node, "2", int(prepared_info["frames"]))
        set_widget(latent_video_node, "3", 1)
    except KeyError:
        pass

    # ARP already prepares an exact target-size canvas, so bypass the demo workflow's
    # pad/resize/reference-image branch. This avoids optional/dynamic helper nodes and
    # feeds the prepared video frames directly into LTX conditioning.
    try:
        set_input_link(workflow, "3336", "image", 13586)
        set_input_link(workflow, "5012", "image", 13586)
        set_input_link(workflow, args.save_node_id, "images", 13594)
    except KeyError:
        pass

    if args.model_backend == "gguf":
        patch_lightweight_gguf(workflow, args)
    else:
        model_patches = {
            "3940": ("0", "ltx-2.3-22b-dev-fp8.safetensors"),
            "4010": ("0", "ltx-2.3-22b-dev-fp8.safetensors"),
            "5023": ("0", args.text_encoder),
            "5011": ("0", args.outpaint_lora),
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
        "model_backend": args.model_backend,
        "gguf_model": args.gguf_model,
        "video_vae": args.video_vae,
        "outpaint_lora": args.outpaint_lora,
        "chunk_seconds": args.chunk_seconds,
        "overlap_frames": args.overlap_frames,
    }


def newest_output(files: list[Path]) -> Path:
    videos = [path for path in files if path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}]
    candidates = videos or files
    if not candidates:
        raise RuntimeError("ComfyUI completed but did not report an output file.")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def chunk_ranges(prepared: Path, chunk_seconds: float, overlap_frames: int) -> list[tuple[int, int, int]]:
    info = probe_video(prepared)
    total_frames = int(info["frames"])
    if chunk_seconds <= 0 or total_frames <= 0:
        return [(0, 0, total_frames)]
    chunk_frames = max(1, int(round(chunk_seconds * info["fps"])))
    if chunk_frames >= total_frames:
        return [(0, 0, total_frames)]
    overlap = max(0, min(int(overlap_frames), chunk_frames - 1))
    step = max(1, chunk_frames - overlap)
    ranges: list[tuple[int, int, int]] = []
    start = 0
    while start < total_frames:
        end = min(total_frames, start + chunk_frames)
        ranges.append((len(ranges), start, end))
        if end >= total_frames:
            break
        start += step
    return ranges


def split_chunk(ffmpeg: str, prepared: Path, chunk_path: Path, start_frame: int, end_frame: int, force: bool) -> None:
    if chunk_path.exists() and not force:
        return
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    partial = chunk_path.with_suffix(chunk_path.suffix + ".partial" + chunk_path.suffix)
    vf = f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS"
    subprocess.run([ffmpeg, "-y", "-i", str(prepared), "-vf", vf, "-an", "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", str(partial)], check=True)
    partial.replace(chunk_path)


def stitch_chunks(ffmpeg: str, chunks: list[Path], output: Path, overlap_frames: int, force: bool) -> None:
    if output.exists() and not force:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="arp_stitch_") as tmp_text:
        tmp = Path(tmp_text)
        list_file = tmp / "chunks.txt"
        trimmed_paths: list[Path] = []
        for index, chunk in enumerate(chunks):
            trimmed = tmp / f"trimmed_{index:04d}.mp4"
            # LTX 2.3 can return fewer frames than were supplied for each chunk. In
            # that case trimming the nominal overlap again shortens the final clip.
            start_frame = 0
            vf = f"trim=start_frame={start_frame},setpts=PTS-STARTPTS"
            subprocess.run([ffmpeg, "-y", "-i", str(chunk), "-vf", vf, "-an", "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", str(trimmed)], check=True)
            trimmed_paths.append(trimmed)
        list_file.write_text("".join(f"file '{path.as_posix()}'\n" for path in trimmed_paths), encoding="utf-8")
        partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
        subprocess.run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(partial)], check=True)
        partial.replace(output)


def build_parser() -> argparse.ArgumentParser:
    config = load_local_config()
    parser = argparse.ArgumentParser(description="Run the LTX IC-LoRA outpainting stage end to end.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target-aspect", default="16:9")
    parser.add_argument("--target-height", type=int, default=720)
    parser.add_argument("--chunk-seconds", type=float, default=20.0, help="Outpaint in chunks of roughly this many seconds. Use 0 to send the full clip.")
    parser.add_argument("--overlap-frames", type=int, default=8, help="Frames repeated between neighbouring chunks before stitching.")
    parser.add_argument("--model-backend", choices=["gguf", "checkpoint"], default="gguf")
    parser.add_argument("--gguf-model", default="LTX-2.3-distilled-Q4_K_M.gguf")
    parser.add_argument("--video-vae", default="LTX23_video_vae_bf16.safetensors")
    parser.add_argument("--audio-vae-checkpoint", default="ltx-2.3-22b-dev-fp8.safetensors")
    parser.add_argument("--text-encoder", default="gemma_3_12B_it_fp8_scaled.safetensors")
    parser.add_argument("--text-encoder-checkpoint", default="ltx-2.3-22b-dev-fp8.safetensors")
    parser.add_argument("--outpaint-lora", default="ltx-2.3-22b-ic-lora-outpaint.safetensors")
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

    if not source.exists():
        raise FileNotFoundError(f"Source video not found: {source}")
    if not workflow_path.exists():
        raise FileNotFoundError(f"Outpainting workflow not found: {workflow_path}")
    if not (comfy_dir / "main.py").exists():
        raise FileNotFoundError(f"ComfyUI main.py not found: {comfy_dir / 'main.py'}")
    if args.model_backend == "gguf" and not (comfy_dir / "custom_nodes" / "ComfyUI-GGUF").exists():
        raise FileNotFoundError(f"ComfyUI-GGUF is required for lightweight outpainting. Re-run install_windows.bat, then restart ComfyUI: {comfy_dir / 'custom_nodes' / 'ComfyUI-GGUF'}")

    output = resolve_path(args.output) if args.output else default_output(source, args.target_aspect, args.target_height)
    raw_output = resolve_path(args.raw_output) if args.raw_output else default_raw_output(source, args.target_aspect, args.target_height)
    prepared = prepared_for(source, args.target_aspect, args.target_height)

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
    width, height = target_size(source, args.target_aspect, args.target_height)
    print(f"Preparing expanded outpaint canvas: {width}x{height}, aspect {args.target_aspect}, black_lift={args.black_lift}, gamma={args.gamma}", flush=True)
    run_command(prepare_command, False)

    output_prefix = f"arp_outpaint/{safe_stem(source.name)}_{aspect_slug(args.target_aspect)}_{width}x{height}"
    print(f"Prepared expanded canvas for ComfyUI: {prepared}", flush=True)
    if not args.dry_run:
        raw_sig = raw_signature(args, workflow_path, prepared)
        if not args.force and resumable_output(raw_output, raw_sig, video_like=prepared):
            print(f"Reuse raw Comfy render: {raw_output}", flush=True)
        else:
            print(f"Waiting for ComfyUI at {args.comfy_url}...", flush=True)
            wait_for_comfy(args.comfy_url, timeout_seconds=180, poll_seconds=args.poll_seconds)
            ffmpeg = find_ffmpeg()
            ranges = chunk_ranges(prepared, args.chunk_seconds, args.overlap_frames)
            print(f"Splitting prepared canvas into {len(ranges)} chunk(s): {args.chunk_seconds:g}s chunks, {args.overlap_frames} overlap frame(s)", flush=True)
            chunk_dir = ROOT / ".cache" / "outpaint_chunks" / f"{safe_stem(source.name)}_{aspect_slug(args.target_aspect)}_{width}x{height}"
            raw_chunks: list[Path] = []
            for chunk_index, start_frame, end_frame in ranges:
                chunk_prepared = chunk_dir / f"prepared_{chunk_index:04d}_{start_frame:06d}_{end_frame:06d}.mp4"
                chunk_raw = chunk_dir / f"raw_{chunk_index:04d}_{start_frame:06d}_{end_frame:06d}.mp4"
                print(f"Outpaint chunk {chunk_index + 1}/{len(ranges)}: frames {start_frame}-{end_frame}", flush=True)
                split_chunk(ffmpeg, prepared, chunk_prepared, start_frame, end_frame, args.force)
                chunk_sig = raw_signature(args, workflow_path, chunk_prepared)
                if not args.force and resumable_output(chunk_raw, chunk_sig, video_like=chunk_prepared):
                    print(f"Reuse raw Comfy chunk: {chunk_raw}", flush=True)
                    raw_chunks.append(chunk_raw)
                    continue
                workflow = json.loads(workflow_path.read_text(encoding="utf-8-sig"))
                chunk_prefix = f"{output_prefix}_chunk_{chunk_index:04d}"
                prompt = patch_workflow(args, workflow, chunk_prepared, comfy_dir, chunk_prefix)
                prompt_id = queue_prompt(args.comfy_url, prompt)
                print(f"Queued ComfyUI prompt: {prompt_id}", flush=True)
                history = wait_for_prompt(args.comfy_url, prompt_id, args.poll_seconds)
                produced = newest_output(extract_output_files(history, comfy_output_root))
                chunk_raw.parent.mkdir(parents=True, exist_ok=True)
                chunk_tmp = chunk_raw.with_suffix(chunk_raw.suffix + ".partial")
                shutil.copy2(produced, chunk_tmp)
                chunk_tmp.replace(chunk_raw)
                write_signature(chunk_raw, chunk_sig)
                print(f"Wrote raw Comfy chunk: {chunk_raw}", flush=True)
                raw_chunks.append(chunk_raw)
            stitch_chunks(ffmpeg, raw_chunks, raw_output, args.overlap_frames, True)
            write_signature(raw_output, raw_sig)
            print(f"Wrote raw Comfy render: {raw_output}", flush=True)

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
    print(f"Wrote outpainted video: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
