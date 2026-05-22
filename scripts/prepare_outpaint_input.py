from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2

from common import file_fingerprint, format_time, resolve_path, root_relative, resumable_output, write_signature


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
    left = max(0, int(args.crop_left))
    right = max(0, int(args.crop_right))
    top = max(0, int(args.crop_top))
    bottom = max(0, int(args.crop_bottom))
    width = max(2, info["width"] - left - right)
    height = max(2, info["height"] - top - bottom)
    width = width if width % 2 == 0 else width - 1
    height = height if height % 2 == 0 else height - 1
    return left, right, top, bottom, width, height


def signature(args, source: Path, info: dict, target_width: int, target_height: int) -> dict:
    return {
        'version': 2,
        'tool': 'prepare_outpaint_input.py',
        'source': root_relative(source),
        'source_fingerprint': file_fingerprint(source),
        'source_width': info['width'],
        'source_height': info['height'],
        'target_width': target_width,
        'target_height': target_height,
        'target_aspect': args.target_aspect,
        'crop_left': max(0, int(args.crop_left)),
        'crop_right': max(0, int(args.crop_right)),
        'crop_top': max(0, int(args.crop_top)),
        'crop_bottom': max(0, int(args.crop_bottom)),
        'black_lift': args.black_lift,
        'gamma': args.gamma,
        'encoder': args.encoder,
        'crf': args.crf,
        'preset': args.preset,
    }


def build_filter(args, info: dict, target_width: int, target_height: int) -> str:
    lift = max(0.0, min(0.25, args.black_lift))
    gamma = max(0.1, args.gamma)
    left, _right, top, _bottom, crop_width, crop_height = crop_values(args, info)
    # The source image is lifted away from exact 0 before padding. The synthetic 16:9 margins stay exact black.
    lut = f"r=255*({lift}+(1-{lift})*pow(val/255\\,1/{gamma})):g=255*({lift}+(1-{lift})*pow(val/255\\,1/{gamma})):b=255*({lift}+(1-{lift})*pow(val/255\\,1/{gamma}))"
    return ';'.join([
        f"color=c=black:s={target_width}x{target_height}:r={info['fps']:.8f}[bg]",
        f"[0:v]crop=w={crop_width}:h={crop_height}:x={left}:y={top},scale=w={target_width}:h={target_height}:force_original_aspect_ratio=decrease:flags=lanczos,setsar=1,format=rgb24,lutrgb={lut}[src]",
        '[bg][src]overlay=x=(W-w)/2:y=(H-h)/2:shortest=1:format=auto,format=yuv420p[v]',
    ])


def default_output(source: Path, target_width: int, target_height: int) -> Path:
    return resolve_path(Path('intermediate') / 'outpaint_prepared' / f'{source.stem}_{target_width}x{target_height}_lifted.mp4')


def build_parser():
    parser = argparse.ArgumentParser(description='Prepare a source clip for LTX IC-LoRA outpainting by lifting source blacks and padding exact-black 16:9 margins.')
    parser.add_argument('--source', required=True, help='Input 4:3 or source-aspect clip.')
    parser.add_argument('--output', help='Prepared clip to write. Defaults to intermediate/outpaint_prepared/<stem>_<size>_lifted.mp4')
    parser.add_argument('--target-aspect', default='16:9')
    parser.add_argument('--target-height', type=int, help='Output height. Defaults to the source height.')
    parser.add_argument('--crop-left', type=int, default=0, help='Pixels to crop from the source before padding.')
    parser.add_argument('--crop-right', type=int, default=0, help='Pixels to crop from the source before padding.')
    parser.add_argument('--crop-top', type=int, default=0, help='Pixels to crop from the source before padding.')
    parser.add_argument('--crop-bottom', type=int, default=0, help='Pixels to crop from the source before padding.')
    parser.add_argument('--black-lift', type=float, default=0.018, help='Raise source pixels away from pure black before padding. 0.018 is about 5/255.')
    parser.add_argument('--gamma', type=float, default=1.06, help='Additional source gamma lift before padding. Values above 1 brighten shadows.')
    parser.add_argument('--encoder', choices=['h264', 'prores'], default='h264')
    parser.add_argument('--crf', type=int, default=12)
    parser.add_argument('--preset', default='slow')
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
    target_width = even(target_height * parse_aspect(args.target_aspect))
    output = resolve_path(args.output) if args.output else default_output(source, target_width, target_height)
    sig = signature(args, source, info, target_width, target_height)
    if not args.force and resumable_output(output, sig, video_like=source, width=target_width, height=target_height):
        print(f'Reuse prepared outpaint input: {output}', flush=True)
        return 0
    ffmpeg = find_ffmpeg(args.ffmpeg)
    partial = output.with_suffix(output.suffix + '.partial' + output.suffix)
    command = [ffmpeg, '-y', '-i', str(source), '-filter_complex', build_filter(args, info, target_width, target_height), '-map', '[v]', '-map', '0:a?', *encoder_args(args), '-c:a', 'copy', str(partial)]
    print(f"Source: {info['width']}x{info['height']} {info['fps']:.6g}fps {format_time(info['duration'])}", flush=True)
    print(f'Prepared canvas: {target_width}x{target_height}, crop LRTB={args.crop_left},{args.crop_right},{args.crop_top},{args.crop_bottom}, black_lift={args.black_lift}, gamma={args.gamma}', flush=True)
    print(' '.join(command), flush=True)
    if args.dry_run:
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    partial.replace(output)
    write_signature(output, sig)
    print(f'Wrote prepared outpaint input: {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

