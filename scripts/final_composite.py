from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from common import file_fingerprint, resolve_path, root_relative, resumable_output, write_signature


def find_ffmpeg(explicit: str | None):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([
        Path(__file__).resolve().parents[1] / '.cache' / 'tools' / 'ffmpeg' / 'ffmpeg.exe',
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
    values['version'] = 3
    return values


def encoder_args(args):
    if args.encoder == 'prores':
        return ['-c:v', 'prores_ks', '-profile:v', '3', '-pix_fmt', 'yuv422p10le']
    return ['-c:v', 'libx264', '-crf', str(args.crf), '-preset', args.preset, '-pix_fmt', 'yuv420p']


def replace_with_retry(source: Path, target: Path, attempts: int = 30, delay: float = 0.5) -> None:
    last_exc: PermissionError | None = None
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError as exc:
            last_exc = exc
            print(f"Composited output is locked by another process; retrying in {delay:g}s ({attempt + 1}/{attempts})...", flush=True)
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def parse_rate(value: str) -> float:
    if not value or value == "0/0":
        return 24.0
    if "/" in value:
        left, right = value.split("/", 1)
        return float(left) / float(right)
    return float(value)


def probe_fps(ffmpeg: str, source: Path) -> float:
    ffprobe = Path(ffmpeg).with_name("ffprobe.exe") if Path(ffmpeg).suffix.lower() == ".exe" else Path("ffprobe")
    try:
        result = subprocess.run(
            [str(ffprobe), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate,r_frame_rate", "-of", "json", str(source)],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(result.stdout).get("streams", [{}])[0]
        return parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "24")
    except Exception:
        return 24.0


def source_crop_filter(args) -> str:
    left = max(0, int(args.crop_left))
    right = max(0, int(args.crop_right))
    top = max(0, int(args.crop_top))
    bottom = max(0, int(args.crop_bottom))
    if not any((left, right, top, bottom)):
        return ""
    return f"crop=w=iw-{left}-{right}:h=ih-{top}-{bottom}:x={left}:y={top},"


def build_filter(args, has_color, fps: float):
    feather = max(1, int(args.feather_pixels))
    sat = max(0.0, args.saturation)
    temp = args.temperature
    color_opacity = max(0.0, min(1.0, args.color_opacity))
    fps_text = f"{fps:.8f}"
    crop = source_crop_filter(args)
    filters = [
        f'[0:v]fps=fps={fps_text},setpts=N/({fps_text}*TB)[base0]',
        f'[1:v]fps=fps={fps_text},{crop}setpts=N/({fps_text}*TB)[src0]',
        '[src0][base0]scale2ref=w=oh*mdar:h=ih[src][base]',
        f"[src]format=rgba,split[src_rgb][src_a]",
        f"[src_a]alphaextract,geq=lum='if(lt(X,{feather}),255*X/{feather},if(gt(X,W-{feather}),255*(W-X)/{feather},255))'[mask]",
        '[src_rgb][mask]alphamerge[srcm]',
        '[base][srcm]overlay=x=(W-w)/2:y=(H-h)/2[merged]',
    ]
    if has_color:
        red = max(temp, 0.0)
        blue = max(-temp, 0.0)
        filters.append(f'[2:v]fps=fps={fps_text},setpts=N/({fps_text}*TB)[col0]')
        filters.append('[col0][merged]scale2ref=w=iw:h=ih[colscaled][mergedref]')
        filters.append(f'[colscaled]eq=saturation={sat}:brightness=0:contrast=1,colorbalance=rs={red:.4f}:bs={blue:.4f},format=yuv444p[colfmt]')
        filters.append('[mergedref]format=yuv444p[basefmt]')
        if color_opacity < 1.0:
            filters.append(f'[basefmt][colfmt]blend=all_expr=A*(1-{color_opacity:.6f})+B*{color_opacity:.6f},format=yuv444p[colblend]')
            color_source = 'colblend'
        else:
            color_source = 'colfmt'
        filters.append(f'[basefmt]extractplanes=y,setsar=1[basey];[{color_source}]extractplanes=u+v[colu0][colv0]')
        filters.append('[colu0]setsar=1[colu];[colv0]setsar=1[colv]')
        filters.append('[basey][colu][colv]mergeplanes=0x001020:yuv444p,setsar=1,format=yuv420p[vout]')
    else:
        filters.append('[merged]copy[vout]')
    return ';'.join(filters)


def run(args):
    outpainted = resolve_path(args.outpainted)
    source = resolve_path(args.source)
    colorized = resolve_path(args.colorized) if args.colorized else None
    output = resolve_path(args.output)
    sig = signature(args)
    if not args.force and resumable_output(output, sig, video_like=outpainted):
        print(f'Reuse composite: {output}')
        return 0
    ffmpeg = find_ffmpeg(args.ffmpeg)
    fps = probe_fps(ffmpeg, source)
    cmd = [ffmpeg, '-y', '-i', str(outpainted), '-i', str(source)]
    if colorized:
        cmd += ['-i', str(colorized)]
    cmd += ['-filter_complex', build_filter(args, bool(colorized), fps), '-map', '[vout]', '-map', '1:a?', '-shortest', '-r', f'{fps:.8f}', '-fps_mode', 'cfr']
    partial = output.with_name(f"{output.stem}.partial.{os_safe_pid()}{output.suffix}")
    cmd += encoder_args(args)
    cmd += ['-c:a', 'copy', str(partial)]
    print(' '.join(cmd))
    if args.dry_run:
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    replace_with_retry(partial, output)
    write_signature(output, sig)
    print(f'Wrote composite: {output}')
    return 0


def os_safe_pid() -> str:
    try:
        import os

        return str(os.getpid())
    except Exception:
        return str(int(time.time()))


def build_parser():
    parser = argparse.ArgumentParser(description='Composite outpainted, original-source centre, and optional color layer into a composited render.')
    parser.add_argument('--outpainted', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--colorized')
    parser.add_argument('--output', required=True)
    parser.add_argument('--feather-pixels', type=int, default=80)
    parser.add_argument('--saturation', type=float, default=0.82)
    parser.add_argument('--temperature', type=float, default=-0.015, help='Negative cools the color overlay; positive warms it.')
    parser.add_argument('--color-opacity', type=float, default=1.0)
    parser.add_argument('--crop-left', type=int, default=0)
    parser.add_argument('--crop-right', type=int, default=0)
    parser.add_argument('--crop-top', type=int, default=0)
    parser.add_argument('--crop-bottom', type=int, default=0)
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
