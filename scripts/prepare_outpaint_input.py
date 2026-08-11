from __future__ import annotations

import argparse
import os
import subprocess
import time
import uuid
from pathlib import Path

import cv2

from common import file_fingerprint, format_time, resolve_path, root_relative, resumable_output, write_signature
from outpaint_geometry import crop_box, source_placement


def parse_aspect(value: str) -> float:
    if ':' in value:
        left, right = value.split(':', 1)
        return float(left) / float(right)
    return float(value)


def even(value: float) -> int:
    number = int(round(value))
    return number if number % 2 == 0 else number + 1


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {path}')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f'Could not probe video dimensions: {path}')
    return {'width': width, 'height': height, 'fps': fps, 'frames': frames, 'duration': frames / fps if fps else 0}


def find_ffmpeg(explicit: str | None) -> str:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([Path(__file__).resolve().parents[1] / '.cache' / 'tools' / 'ffmpeg' / 'ffmpeg.exe', Path('C:/Program Files/ffmpeg/bin/ffmpeg.exe'), Path('ffmpeg')])
    for candidate in candidates:
        try:
            subprocess.run([str(candidate), '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return str(candidate)
        except Exception:
            continue
    raise FileNotFoundError('ffmpeg was not found. Install it or pass --ffmpeg.')


def encoder_args(args):
    if args.encoder == 'prores':
        return ['-c:v', 'prores_ks', '-profile:v', '3', '-pix_fmt', 'yuv422p10le']
    return ['-c:v', 'libx264', '-crf', str(args.crf), '-preset', args.preset, '-pix_fmt', 'yuv420p']


def crop_values(args, info: dict) -> tuple[int, int, int, int, int, int]:
    return crop_box(
        int(info["width"]),
        int(info["height"]),
        int(args.crop_left),
        int(args.crop_right),
        int(args.crop_top),
        int(args.crop_bottom),
    )


def source_placement_size(args, info: dict, target_width: int, target_height: int) -> tuple[int, int, int, int]:
    delivery_w = int(args.delivery_width or target_width)
    delivery_h = int(args.delivery_height or target_height)
    crops = tuple(int(getattr(args, key)) for key in ("crop_left", "crop_right", "crop_top", "crop_bottom"))
    delivery = source_placement(int(info["width"]), int(info["height"]), delivery_w, delivery_h, crops)
    target = source_placement(int(info["width"]), int(info["height"]), target_width, target_height, crops, delivery_w, delivery_h)
    return delivery.width, delivery.height, target.width, target.height


def signature(args, source: Path, info: dict, target_width: int, target_height: int) -> dict:
    return {
        'version': 10,
        'tool': 'prepare_outpaint_input.py',
        'geometry': 'crop_then_fit_v1',
        'source': root_relative(source),
        'source_fingerprint': file_fingerprint(source),
        'source_width': info['width'],
        'source_height': info['height'],
        'target_width': target_width,
        'target_height': target_height,
        'delivery_width': int(args.delivery_width or target_width),
        'delivery_height': int(args.delivery_height or target_height),
        'target_aspect': args.target_aspect,
        'crop_left': max(0, int(args.crop_left)),
        'crop_right': max(0, int(args.crop_right)),
        'crop_top': max(0, int(args.crop_top)),
        'crop_bottom': max(0, int(args.crop_bottom)),
        'black_lift': args.black_lift,
        'gamma': args.gamma,
        'outpaint_all_black_regions': bool(getattr(args, 'outpaint_all_black_regions', False)),
        'encoder': args.encoder,
        'crf': args.crf,
        'preset': args.preset,
    }


def build_filter(args, info: dict, target_width: int, target_height: int) -> str:
    lift = max(0.0, min(0.25, args.black_lift))
    gamma = max(0.1, args.gamma)
    left, _right, top, _bottom, crop_width, crop_height = crop_values(args, info)
    # By default the source image is lifted away from exact 0 before padding. The synthetic
    # margins stay exact black. In all-black-regions mode, source blacks are left alone so
    # embedded matte bars/black restoration gaps can be treated like outpaintable padding.
    lut = f"r=255*({lift}+(1-{lift})*pow(val/255\\,1/{gamma})):g=255*({lift}+(1-{lift})*pow(val/255\\,1/{gamma})):b=255*({lift}+(1-{lift})*pow(val/255\\,1/{gamma}))"
    # Rebuild timestamps from decoded frame numbers before any frame-rate conversion. Some MKV
    # sources expose 23.976 fps frames on a millisecond time base, so frame 8 can be timestamped
    # at 0.334s instead of 8/23.976 (0.333667s). Sampling by those rounded timestamps randomly
    # picks the previous frame at cuts; frame-index timing keeps the outpaint canvas aligned.
    #
    # Crop first and treat the remaining frame as the complete source. Fit that
    # post-trim geometry into the delivery canvas, then map it proportionally
    # into the model-safe canvas. This leaves exactly one simple pillarbox or
    # letterbox for LTX; removed trim pixels never become generation bands.
    delivery_w = int(args.delivery_width or target_width)
    delivery_h = int(args.delivery_height or target_height)
    crops = tuple(int(getattr(args, key)) for key in ("crop_left", "crop_right", "crop_top", "crop_bottom"))
    delivery = source_placement(int(info["width"]), int(info["height"]), delivery_w, delivery_h, crops)
    target = source_placement(int(info["width"]), int(info["height"]), target_width, target_height, crops, delivery_w, delivery_h)
    delivery_source_w, delivery_source_h = delivery.width, delivery.height
    target_source_w, target_source_h = target.width, target.height
    scale_steps = f"scale=w={delivery_source_w}:h={delivery_source_h}:flags=lanczos"
    if (delivery_source_w, delivery_source_h) != (target_source_w, target_source_h):
        scale_steps += f",scale=w={target_source_w}:h={target_source_h}:flags=lanczos"
    source_filters = f"[0:v]trim=start_frame=0,setpts=N/({info['fps']:.8f}*TB),fps={info['fps']:.8f},crop=w={crop_width}:h={crop_height}:x={left}:y={top},{scale_steps},setsar=1,format=rgb24"
    if not getattr(args, 'outpaint_all_black_regions', False):
        source_filters += f",lutrgb={lut}"
    source_filters += "[src]"
    return ';'.join([
        f"color=c=black:s={target_width}x{target_height}:r={info['fps']:.8f}[bg]",
        source_filters,
        f'[bg][src]overlay=x={target.x}:y={target.y}:shortest=1:format=auto,format=yuv420p[v]',
    ])


def default_output(source: Path, target_width: int, target_height: int, delivery_width: int = 0, delivery_height: int = 0) -> Path:
    delivery_tag = f'_from{delivery_width}x{delivery_height}' if (delivery_width and delivery_height and (delivery_width != target_width or delivery_height != target_height)) else ''
    return resolve_path(Path('intermediate') / 'outpaint_prepared' / f'{source.stem}_{target_width}x{target_height}{delivery_tag}_lifted.mp4')


def partial_output_path(output: Path) -> Path:
    return output.with_name(f'{output.stem}.partial.{os.getpid()}.{uuid.uuid4().hex}{output.suffix}')


def cleanup_stale_partials(output: Path, max_age_seconds: float = 24 * 60 * 60) -> None:
    now = time.time()
    patterns = [
        f'{output.stem}.partial.*{output.suffix}',
        f'{output.name}.partial{output.suffix}',
    ]
    for pattern in patterns:
        try:
            candidates = list(output.parent.glob(pattern))
        except OSError:
            continue
        for candidate in candidates:
            try:
                if candidate == output or now - candidate.stat().st_mtime < max_age_seconds:
                    continue
                candidate.unlink()
            except OSError:
                pass


def replace_with_retry(partial: Path, output: Path, attempts: int = 60, delay: float = 0.5) -> None:
    last_exc: PermissionError | None = None
    for attempt in range(1, attempts + 1):
        try:
            partial.replace(output)
            return
        except PermissionError as exc:
            last_exc = exc
            if attempt == 1:
                print(f'Waiting for file lock to clear before replacing prepared input: {partial} -> {output}', flush=True)
            time.sleep(delay)
    raise PermissionError(
        f'Could not replace prepared input because the temporary or final file is open in another process: {partial} -> {output}. '
        'Close any ARP preview, media player, Resolve bin/timeline item, Explorer preview, or antivirus scan using this file and try again.'
    ) from last_exc


def build_parser():
    parser = argparse.ArgumentParser(description='Crop a source clip, then fit it into an exact-black LTX outpainting canvas.')
    parser.add_argument('--source', required=True, help='Input 4:3 or source-aspect clip.')
    parser.add_argument('--output', help='Prepared clip to write. Defaults to intermediate/outpaint_prepared/<stem>_<size>_lifted.mp4')
    parser.add_argument('--target-aspect', default='16:9')
    parser.add_argument('--target-width', type=int, help='Output width (model-safe, e.g. 1280). Defaults to target-height * target-aspect.')
    parser.add_argument('--target-height', type=int, help='Output height (model-safe, e.g. 704). Defaults to the source height.')
    parser.add_argument('--delivery-width', type=int, default=0, help='Delivery width for AR-correct scaling (step 1). Source is scaled to fit delivery dims first, then the changed axis is squished to model-safe. Defaults to --target-width.')
    parser.add_argument('--delivery-height', type=int, default=0, help='Delivery height for AR-correct scaling (step 1). Defaults to --target-height.')
    parser.add_argument('--crop-left', type=int, default=0, help='Pixels to crop from the source before padding.')
    parser.add_argument('--crop-right', type=int, default=0, help='Pixels to crop from the source before padding.')
    parser.add_argument('--crop-top', type=int, default=0, help='Pixels to crop from the source before padding.')
    parser.add_argument('--crop-bottom', type=int, default=0, help='Pixels to crop from the source before padding.')
    parser.add_argument('--black-lift', type=float, default=0.018, help='Raise source pixels away from pure black before padding. 0.018 is about 5/255.')
    parser.add_argument('--gamma', type=float, default=1.06, help='Additional source gamma lift before padding. Values above 1 brighten shadows.')
    parser.add_argument('--outpaint-all-black-regions', action='store_true', help='Do not lift source blacks. Pure black regions inside the source can be outpainted like padded margins.')
    parser.add_argument('--encoder', choices=['h264', 'prores'], default='h264')
    parser.add_argument('--crf', type=int, default=12)
    parser.add_argument('--preset', default='medium')
    parser.add_argument('--ffmpeg')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    return parser


def main():
    args = build_parser().parse_args()
    source = resolve_path(args.source)
    if not source.exists():
        raise FileNotFoundError(f'Source video not found: {source}')
    info = probe_video(source)
    target_height = even(args.target_height or info['height'])
    target_width = even(args.target_width or (target_height * parse_aspect(args.target_aspect)))
    output = resolve_path(args.output) if args.output else default_output(source, target_width, target_height, int(args.delivery_width or 0), int(args.delivery_height or 0))
    sig = signature(args, source, info, target_width, target_height)
    if not args.force and resumable_output(output, sig, video_like=source, width=target_width, height=target_height):
        print(f'Reuse prepared outpaint input: {output}', flush=True)
        return 0
    ffmpeg = find_ffmpeg(args.ffmpeg)
    partial = partial_output_path(output)
    fps = float(info["fps"] or 24.0)
    command = [
        ffmpeg,
        '-y',
        '-i',
        str(source),
        '-filter_complex',
        build_filter(args, info, target_width, target_height),
        '-map',
        '[v]',
        '-map',
        '0:a?',
        '-r',
        f'{fps:.8f}',
        '-fps_mode',
        'cfr',
        *encoder_args(args),
        '-c:a',
        'copy',
        str(partial),
    ]
    print(f"Source: {info['width']}x{info['height']} {info['fps']:.6g}fps {format_time(info['duration'])}", flush=True)
    mode = 'all black regions' if args.outpaint_all_black_regions else 'protected source blacks'
    print(f'Prepared canvas: {target_width}x{target_height}, crop LRTB={args.crop_left},{args.crop_right},{args.crop_top},{args.crop_bottom}, black_lift={args.black_lift}, gamma={args.gamma}, mode={mode}', flush=True)
    print(' '.join(command), flush=True)
    if args.dry_run:
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    cleanup_stale_partials(output)
    subprocess.run(command, check=True)
    replace_with_retry(partial, output)
    write_signature(output, sig)
    print(f'Wrote prepared outpaint input: {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

