from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from common import file_fingerprint, resolve_path, root_relative, resumable_output, video_info, write_signature


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


def source_rectangle(args, info: dict) -> tuple[int, int, int, int] | None:
    values = (args.source_x, args.source_y, args.source_width, args.source_height)
    if any(value is None for value in values):
        return None
    x, y, width, height = (int(value) for value in values)
    frame_width, frame_height = int(info['width']), int(info['height'])
    x = max(0, min(frame_width - 1, x))
    y = max(0, min(frame_height - 1, y))
    width = max(1, min(frame_width - x, width))
    height = max(1, min(frame_height - y, height))
    return x, y, width, height


def inverse_filter(args, info: dict) -> str:
    target_width = int(args.target_width or info["width"])
    target_height = int(args.target_height or info["height"])
    scale = ""
    if target_width != int(info["width"]) or target_height != int(info["height"]):
        scale = f",scale={target_width}:{target_height}:flags=lanczos"
    monochrome = ",hue=s=0" if args.monochrome else ""
    rectangle = source_rectangle(args, info)
    restore_tone = bool(args.restore_tone and not args.skip_restore)
    if rectangle is None:
        if restore_tone:
            lift = max(0.0, min(0.25, args.black_lift))
            gamma = max(0.1, args.gamma)
            expr = f"if(lt(val/255\\,{lift})\\,0\\,255*pow((val/255-{lift})/(1-{lift})\\,{gamma}))"
            return f"[0:v]format=rgb24,lutrgb=r='{expr}':g='{expr}':b='{expr}'{monochrome},format=yuv420p{scale}[v]"
        return f"[0:v]format=yuv420p{monochrome}{scale}[v]"

    x, y, width, height = rectangle
    right, bottom = x + width - 1, y + height - 1
    feather = max(0.0, min(64.0, float(args.edge_feather)))
    sharpen = max(0.0, min(1.5, float(args.edge_sharpen)))
    blur = f',gblur=sigma={feather:.4f}' if feather else ''
    sharpen_filter = f'unsharp=5:5:{sharpen:.4f}:5:5:0.0' if sharpen else 'null'
    if restore_tone:
        lift = max(0.0, min(0.25, args.black_lift))
        gamma = max(0.1, args.gamma)
        expr = f"if(lt(val/255\\,{lift})\\,0\\,255*pow((val/255-{lift})/(1-{lift})\\,{gamma}))"
        centre_filter = f"lutrgb=r='{expr}':g='{expr}':b='{expr}'"
    else:
        centre_filter = 'null'
    mask_expr = f"if(between(X\\,{x}\\,{right})*between(Y\\,{y}\\,{bottom})\\,255\\,0)"
    return (
        '[0:v]format=rgb24,split=3[raw][original][maskbase];'
        f'[raw]{sharpen_filter}[generated];'
        f'[original]{centre_filter}[centre];'
        f"[maskbase]format=gray,geq=lum='{mask_expr}'{blur}[sourcemask];"
        f'[generated][centre][sourcemask]maskedmerge{monochrome},format=yuv420p{scale}[v]'
    )


def signature(args, source: Path, info: dict) -> dict:
    return {
        'version': 8,
        'tool': 'finalize_outpaint_output.py',
        'source': root_relative(source),
        'source_fingerprint': file_fingerprint(source),
        'skip_restore': args.skip_restore,
        'restore_tone': args.restore_tone,
        'black_lift': args.black_lift if args.restore_tone else 0.0,
        'gamma': args.gamma if args.restore_tone else 1.0,
        'monochrome': args.monochrome,
        'target_width': args.target_width,
        'target_height': args.target_height,
        'source_rectangle': source_rectangle(args, info),
        'edge_feather': args.edge_feather,
        'edge_sharpen': args.edge_sharpen,
        'encoder': args.encoder,
        'crf': args.crf,
        'preset': args.preset,
    }


def default_output(source: Path) -> Path:
    return resolve_path(Path('intermediate') / 'outpainted' / f'{source.stem}_restored.mp4')


def replace_with_retry(partial: Path, output: Path, attempts: int = 20, delay: float = 0.5) -> None:
    last_exc: PermissionError | None = None
    for attempt in range(1, attempts + 1):
        try:
            partial.replace(output)
            return
        except PermissionError as exc:
            last_exc = exc
            if attempt == 1:
                print(f'Waiting for file lock to clear before replacing outpaint output: {output}', flush=True)
            time.sleep(delay)
    raise PermissionError(
        f'Could not replace outpaint output because it is open in another process: {output}. '
        'Close any ARP preview, media player, Resolve bin/timeline item, or Explorer preview using this file and try again.'
    ) from last_exc


def build_parser():
    parser = argparse.ArgumentParser(description='Finish an LTX IC-LoRA outpaint render without changing its source tonality.')
    parser.add_argument('--source', required=True, help='ComfyUI/LTX outpainted render made from prepare_outpaint_input.py output.')
    parser.add_argument('--output', help='Restored clip to write. Defaults to intermediate/outpainted/<stem>_restored.mp4')
    parser.add_argument('--black-lift', type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument('--gamma', type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument('--skip-restore', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--restore-tone', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--monochrome', action='store_true', help='Remove residual chroma from monochrome-source outpainting.')
    parser.add_argument('--target-width', type=int, help='Scale the restored clip to this delivery width.')
    parser.add_argument('--target-height', type=int, help='Scale the restored clip to this delivery height.')
    parser.add_argument('--source-x', type=int, help='Left edge of the protected source region in the rendered canvas.')
    parser.add_argument('--source-y', type=int, help='Top edge of the protected source region in the rendered canvas.')
    parser.add_argument('--source-width', type=int, help='Width of the protected source region in the rendered canvas.')
    parser.add_argument('--source-height', type=int, help='Height of the protected source region in the rendered canvas.')
    parser.add_argument('--edge-feather', type=float, default=6.0, help='Soft transition, in pixels, between restored source and generated outpaint.')
    parser.add_argument('--edge-sharpen', type=float, default=0.35, help='Modest unsharp amount applied only to generated outpaint regions; 0 disables it.')
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
        raise FileNotFoundError(f'Outpainted source not found: {source}')
    output = resolve_path(args.output) if args.output else default_output(source)
    info = video_info(source)
    target_width = int(args.target_width or info["width"])
    target_height = int(args.target_height or info["height"])
    sig = signature(args, source, info)
    if not args.force and resumable_output(output, sig, width=target_width, height=target_height):
        print(f'Reuse restored outpaint: {output}')
        return 0
    ffmpeg = find_ffmpeg(args.ffmpeg)
    fps = float(info["fps"] or 24.0)
    partial = output.with_suffix(output.suffix + '.partial' + output.suffix)
    command = [
        ffmpeg,
        '-y',
        '-i',
        str(source),
        '-filter_complex',
        inverse_filter(args, info),
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
    print(' '.join(command))
    if args.dry_run:
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    replace_with_retry(partial, output)
    write_signature(output, sig)
    print(f'Wrote restored outpaint: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
