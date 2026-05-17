from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import file_fingerprint, resolve_path, root_relative, signature_matches, write_signature


def find_ffmpeg(explicit: str | None):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([
        Path('C:/Program Files/ffmpeg/bin/ffmpeg.exe'),
        Path('ffmpeg'),
    ])
    for candidate in candidates:
        try:
            subprocess.run([str(candidate), '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return str(candidate)
        except Exception:
            continue
    raise FileNotFoundError('ffmpeg was not found. Install it or pass --ffmpeg.')


def signature(args):
    values = vars(args).copy()
    for key in ['outpainted', 'source', 'colorized']:
        value = values.get(key)
        if value:
            path = resolve_path(value)
            values[key] = root_relative(path)
            values[key + '_fingerprint'] = file_fingerprint(path)
    values.pop('ffmpeg', None)
    values['tool'] = 'final_composite.py'
    values['version'] = 1
    return values


def encoder_args(args):
    if args.encoder == 'prores':
        return ['-c:v', 'prores_ks', '-profile:v', '3', '-pix_fmt', 'yuv422p10le']
    return ['-c:v', 'libx264', '-crf', str(args.crf), '-preset', args.preset, '-pix_fmt', 'yuv420p']


def build_filter(args, has_color):
    feather = max(1, int(args.feather_pixels))
    sat = max(0.0, args.saturation)
    temp = args.temperature
    color_opacity = max(0.0, min(1.0, args.color_opacity))
    filters = [
        '[1:v][0:v]scale2ref=w=oh*mdar:h=ih[src][base]',
        f"[src]format=rgba,split[src_rgb][src_a]",
        f"[src_a]alphaextract,geq=lum='if(lt(X,{feather}),255*X/{feather},if(gt(X,W-{feather}),255*(W-X)/{feather},255))'[mask]",
        '[src_rgb][mask]alphamerge[srcm]',
        '[base][srcm]overlay=x=(W-w)/2:y=(H-h)/2[merged]',
    ]
    if has_color:
        red = max(temp, 0.0)
        blue = max(-temp, 0.0)
        filters.append(f'[2:v]eq=saturation={sat}:brightness=0:contrast=1,colorbalance=rs={red:.4f}:bs={blue:.4f}[col]')
        filters.append(f'[merged][col]blend=all_mode=overlay:all_opacity={color_opacity}[vout]')
    else:
        filters.append('[merged]copy[vout]')
    return ';'.join(filters)


def run(args):
    outpainted = resolve_path(args.outpainted)
    source = resolve_path(args.source)
    colorized = resolve_path(args.colorized) if args.colorized else None
    output = resolve_path(args.output)
    sig = signature(args)
    if not args.force and signature_matches(output, sig):
        print(f'Reuse composite: {output}')
        return 0
    ffmpeg = find_ffmpeg(args.ffmpeg)
    cmd = [ffmpeg, '-y', '-i', str(outpainted), '-i', str(source)]
    if colorized:
        cmd += ['-i', str(colorized)]
    cmd += ['-filter_complex', build_filter(args, bool(colorized)), '-map', '[vout]', '-map', '1:a?', '-shortest']
    cmd += encoder_args(args)
    cmd += ['-c:a', 'copy', str(output.with_suffix(output.suffix + '.partial' + output.suffix))]
    print(' '.join(cmd))
    if args.dry_run:
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + '.partial' + output.suffix)
    subprocess.run(cmd, check=True)
    partial.replace(output)
    write_signature(output, sig)
    print(f'Wrote composite: {output}')
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description='Composite outpainted, original-source centre, and optional color layer into a final master.')
    parser.add_argument('--outpainted', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--colorized')
    parser.add_argument('--output', required=True)
    parser.add_argument('--feather-pixels', type=int, default=80)
    parser.add_argument('--saturation', type=float, default=0.82)
    parser.add_argument('--temperature', type=float, default=-0.015, help='Negative cools the color overlay; positive warms it.')
    parser.add_argument('--color-opacity', type=float, default=1.0)
    parser.add_argument('--encoder', choices=['h264', 'prores'], default='h264')
    parser.add_argument('--crf', type=int, default=16)
    parser.add_argument('--preset', default='slow')
    parser.add_argument('--ffmpeg')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    return parser


def main():
    return run(build_parser().parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
