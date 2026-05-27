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


def inverse_filter(args) -> str:
    lift = max(0.0, min(0.25, args.black_lift))
    gamma = max(0.1, args.gamma)
    if args.skip_restore:
        return '[0:v]format=yuv420p[v]'
    expr = f"if(lt(val/255\\,{lift})\\,0\\,255*pow((val/255-{lift})/(1-{lift})\\,{gamma}))"
    return f"[0:v]format=rgb24,lutrgb=r='{expr}':g='{expr}':b='{expr}',format=yuv420p[v]"


def signature(args, source: Path) -> dict:
    return {
        'version': 2,
        'tool': 'finalize_outpaint_output.py',
        'source': root_relative(source),
        'source_fingerprint': file_fingerprint(source),
        'black_lift': args.black_lift,
        'gamma': args.gamma,
        'skip_restore': args.skip_restore,
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
    parser = argparse.ArgumentParser(description='Restore the black/gamma lift after an LTX IC-LoRA outpaint render.')
    parser.add_argument('--source', required=True, help='ComfyUI/LTX outpainted render made from prepare_outpaint_input.py output.')
    parser.add_argument('--output', help='Restored clip to write. Defaults to intermediate/outpainted/<stem>_restored.mp4')
    parser.add_argument('--black-lift', type=float, default=0.018, help='Must match prepare_outpaint_input.py.')
    parser.add_argument('--gamma', type=float, default=1.06, help='Must match prepare_outpaint_input.py.')
    parser.add_argument('--skip-restore', action='store_true', help='Only remux/re-encode. Useful for comparisons.')
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
    sig = signature(args, source)
    if not args.force and resumable_output(output, sig, video_like=source):
        print(f'Reuse restored outpaint: {output}')
        return 0
    ffmpeg = find_ffmpeg(args.ffmpeg)
    fps = float(video_info(source)["fps"] or 24.0)
    partial = output.with_suffix(output.suffix + '.partial' + output.suffix)
    command = [
        ffmpeg,
        '-y',
        '-i',
        str(source),
        '-filter_complex',
        inverse_filter(args),
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
