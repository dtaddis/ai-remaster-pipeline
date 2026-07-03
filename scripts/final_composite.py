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
    values['version'] = 8
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
            print(f"Final output is locked by another process; retrying in {delay:g}s ({attempt + 1}/{attempts})...", flush=True)
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


def nested_expr(fn: str, values: list[str]) -> str:
    if not values:
        return "0"
    expr = values[0]
    for value in values[1:]:
        expr = f"{fn}({expr},{value})"
    return expr


def source_rgb_max_expr(x_expr: str = "X", y_expr: str = "Y") -> str:
    return f"max(max(r({x_expr},{y_expr}),g({x_expr},{y_expr})),b({x_expr},{y_expr}))"


def source_black_matte_expr(threshold: int, shrink: int) -> str:
    radius = max(0, min(8, int(shrink)))
    offsets = [(0, 0)]
    if radius:
        offsets.extend([
            (-radius, 0),
            (radius, 0),
            (0, -radius),
            (0, radius),
            (-radius, -radius),
            (radius, -radius),
            (-radius, radius),
            (radius, radius),
        ])
    samples = []
    for dx, dy in offsets:
        x = "X" if dx == 0 else f"min(max(X{dx:+d},0),W-1)"
        y = "Y" if dy == 0 else f"min(max(Y{dy:+d},0),H-1)"
        samples.append(source_rgb_max_expr(x, y))
    return f"lte({nested_expr('min', samples)},{threshold})"


def source_alpha_expr(args, feather: int) -> str:
    edge_alpha = f"if(lt(X,{feather}),255*X/{feather},if(gt(X,W-{feather}),255*(W-X)/{feather},255))"
    if not getattr(args, "source_black_transparent", False):
        return edge_alpha
    threshold = max(0, min(255, int(getattr(args, "source_black_threshold", 24))))
    shrink = max(0, min(8, int(getattr(args, "source_black_matte_shrink_pixels", 2))))
    return f"if({source_black_matte_expr(threshold, shrink)},0,{edge_alpha})"


def normalized_percent(value: float, default: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number / 100.0 if number > 4.0 else number


def temperature_balance(value: float) -> tuple[float, float]:
    """Return FFmpeg colorbalance red/blue shadow strengths from Kelvin.

    6500K is treated as neutral. Lower Kelvin warms the color layer, higher
    Kelvin cools it. Small legacy values are accepted by mapping negative to
    blue and positive to red.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 6500.0
    if abs(number) <= 20.0:
        return max(number, 0.0), max(-number, 0.0)
    delta = max(-4000.0, min(4000.0, number - 6500.0))
    strength = abs(delta) / 4000.0 * 0.12
    return (strength, 0.0) if delta < 0 else (0.0, strength)


def build_filter(args, has_color, fps: float, has_outpainted: bool = True):
    feather = max(1, int(args.feather_pixels))
    sat = max(0.0, normalized_percent(args.saturation, 0.82))
    color_opacity = max(0.0, min(1.0, normalized_percent(args.color_opacity, 1.0)))
    fps_text = f"{fps:.8f}"
    crop = source_crop_filter(args)
    color_input = 2 if has_outpainted else 1
    # Optionally scale the outpainted video to the delivery output dimensions.
    # This corrects for LTX's model-safe quantisation (e.g. 704p → 720p) so the
    # final composite is at the user's intended resolution.
    out_w = int(args.output_width) if args.output_width else 0
    out_h = int(args.output_height) if args.output_height else 0
    scale_base = f",scale={out_w}:{out_h}:flags=lanczos" if (out_w and out_h) else ""
    if has_outpainted:
        filters = [
            f'[0:v]setpts=N/({fps_text}*TB),fps=fps={fps_text}{scale_base}[base0]',
            f'[1:v]setpts=N/({fps_text}*TB),fps=fps={fps_text},{crop}setsar=1[src0]',
            '[src0][base0]scale2ref=w=trunc(oh*mdar/2)*2:h=ih[src][base]',
            f"[src]format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='{source_alpha_expr(args, feather)}'[srcm]",
            '[base][srcm]overlay=x=(W-w)/2:y=(H-h)/2[merged]',
        ]
    else:
        filters = [
            f'[0:v]setpts=N/({fps_text}*TB),fps=fps={fps_text},{crop}setsar=1,format=yuv444p[merged]',
        ]
    if has_color:
        red, blue = temperature_balance(args.temperature)
        filters.append(f'[{color_input}:v]setpts=N/({fps_text}*TB),fps=fps={fps_text}[col0]')
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
    outpainted = resolve_path(args.outpainted) if args.outpainted else None
    source = resolve_path(args.source)
    colorized = resolve_path(args.colorized) if args.colorized else None
    output = resolve_path(args.output)
    sig = signature(args)
    video_like = outpainted or source
    if not args.force and resumable_output(output, sig, video_like=video_like):
        print(f'Reuse composite: {output}')
        return 0
    ffmpeg = find_ffmpeg(args.ffmpeg)
    fps = probe_fps(ffmpeg, source)
    cmd = [ffmpeg, '-y']
    if outpainted:
        cmd += ['-i', str(outpainted), '-i', str(source)]
        audio_input = '1:a?'
    else:
        cmd += ['-i', str(source)]
        audio_input = '0:a?'
    if colorized:
        cmd += ['-i', str(colorized)]
    cmd += ['-filter_complex', build_filter(args, bool(colorized), fps, bool(outpainted)), '-map', '[vout]', '-map', audio_input, '-shortest', '-r', f'{fps:.8f}', '-fps_mode', 'cfr']
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
    parser = argparse.ArgumentParser(description='Composite outpainted, original-source centre, and optional color layer into a final master.')
    parser.add_argument('--outpainted')
    parser.add_argument('--source', required=True)
    parser.add_argument('--colorized')
    parser.add_argument('--output', required=True)
    parser.add_argument('--feather-pixels', type=int, default=80)
    parser.add_argument('--saturation', type=float, default=82.0, help='Color layer saturation. Values above 4 are treated as percentages.')
    parser.add_argument('--temperature', type=float, default=6500.0, help='Color temperature in Kelvin. 6500 is neutral; lower warms, higher cools.')
    parser.add_argument('--color-opacity', type=float, default=100.0, help='Color layer opacity. Values above 4 are treated as percentages.')
    parser.add_argument('--output-width', type=int, default=0, help='Scale outpainted video to this width before compositing (delivery upscale, e.g. 1280 to correct 704→720).')
    parser.add_argument('--output-height', type=int, default=0, help='Scale outpainted video to this height before compositing (delivery upscale, e.g. 720 to correct 704→720).')
    parser.add_argument('--source-black-transparent', action='store_true', help='Treat near-black source pixels as transparent so outpainted regions remain visible in the final composite.')
    parser.add_argument('--source-black-threshold', type=int, default=24, help='Maximum RGB channel value considered source black when --source-black-transparent is enabled.')
    parser.add_argument('--source-black-matte-shrink-pixels', type=int, default=2, help='Shrink the source matte by this many pixels around detected black regions to avoid dark resampling halos.')
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
