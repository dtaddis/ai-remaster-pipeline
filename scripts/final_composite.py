from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from common import file_fingerprint, resolve_path, root_relative, resumable_output, write_signature
from outpaint_geometry import source_placement
from reference_luminance import ffmpeg_curve, identity_curve, reference_luminance_plan


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
    for key in ['outpainted', 'source', 'colorized', 'custom_mask']:
        value = values.get(key)
        if value:
            path = resolve_path(value)
            values[key] = root_relative(path)
            values[key + '_fingerprint'] = file_fingerprint(path)
    values.pop('ffmpeg', None)
    values['tool'] = 'final_composite.py'
    values['version'] = 13
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


def probe_duration(ffmpeg: str, source: Path) -> float:
    ffprobe = Path(ffmpeg).with_name("ffprobe.exe") if Path(ffmpeg).suffix.lower() == ".exe" else Path("ffprobe")
    try:
        result = subprocess.run(
            [str(ffprobe), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(source)],
            check=True,
            capture_output=True,
            text=True,
        )
        return max(0.0, float(json.loads(result.stdout).get("format", {}).get("duration") or 0.0))
    except Exception:
        return 0.0


def probe_dimensions(ffmpeg: str, source: Path) -> tuple[int, int]:
    ffprobe = Path(ffmpeg).with_name("ffprobe.exe") if Path(ffmpeg).suffix.lower() == ".exe" else Path("ffprobe")
    result = subprocess.run(
        [str(ffprobe), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout).get("streams", [{}])[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not probe video dimensions: {source}")
    return width, height


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


def source_luma_max_expr(x_expr: str = "X", y_expr: str = "Y") -> str:
    """Sample a gray plane that already holds max(r,g,b) for every pixel.

    Folding the three channels down with native filters first lets the matte run
    as a one-plane expression instead of an rgba one, which is the difference
    between three pixel fetches per sample and one.
    """
    return f"lum({x_expr},{y_expr})"


def source_black_matte_expr(threshold: int, shrink: int, sampler=source_rgb_max_expr) -> str:
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
        samples.append(sampler(x, y))
    return f"lte({nested_expr('min', samples)},{threshold})"


def source_alpha_expr(args, feather: int, *, horizontal: bool, vertical: bool, sampler=source_rgb_max_expr) -> str:
    edge_alphas = []
    if horizontal:
        edge_alphas.append(
            f"if(lt(X,{feather}),255*X/{feather},if(gt(X,W-{feather}),255*(W-X)/{feather},255))"
        )
    if vertical:
        edge_alphas.append(
            f"if(lt(Y,{feather}),255*Y/{feather},if(gt(Y,H-{feather}),255*(H-Y)/{feather},255))"
        )
    edge_alpha = "255" if not edge_alphas else edge_alphas[0]
    if len(edge_alphas) == 2:
        edge_alpha = f"min({edge_alphas[0]},{edge_alphas[1]})"
    if not getattr(args, "source_black_transparent", False):
        return edge_alpha
    threshold = max(0, min(255, int(getattr(args, "source_black_threshold", 24))))
    shrink = max(0, min(8, int(getattr(args, "source_black_matte_shrink_pixels", 2))))
    return f"if({source_black_matte_expr(threshold, shrink, sampler)},0,{edge_alpha})"


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


def append_reference_luminance_filter(filters: list[str], input_label: str, plan: list[dict], fps: float) -> str:
    spans = [item for item in plan if item.get('end_frame') is None or int(item['end_frame']) > int(item['start_frame'])]
    if not spans:
        return input_label
    if len(spans) == 1 and int(spans[0].get('start_frame', 0)) == 0 and spans[0].get('end_frame') is None:
        curve = spans[0].get('curve') or [[0.0, 0.0], [1.0, 1.0]]
        if identity_curve(curve):
            return input_label
        filters.append(f"[{input_label}]curves=all='{ffmpeg_curve(curve)}'[lumamerged]")
        return 'lumamerged'

    split_labels = ''.join(f'[lumasrc{index}]' for index in range(len(spans)))
    filters.append(f'[{input_label}]split={len(spans)}{split_labels}')
    output_labels = []
    for index, item in enumerate(spans):
        start = max(0, int(item.get('start_frame', 0)))
        end = item.get('end_frame')
        trim = f'trim=start_frame={start}' + (f':end_frame={max(start + 1, int(end))}' if end is not None else '')
        curve = item.get('curve') or [[0.0, 0.0], [1.0, 1.0]]
        tone = '' if identity_curve(curve) else f",curves=all='{ffmpeg_curve(curve)}'"
        filters.append(f'[lumasrc{index}]{trim},setpts=PTS-STARTPTS{tone}[lumaseg{index}]')
        output_labels.append(f'[lumaseg{index}]')
    filters.append(''.join(output_labels) + f'concat=n={len(output_labels)}:v=1:a=0[lumaconcat]')
    filters.append(f'[lumaconcat]setpts=N/({fps:.8f}*TB),fps=fps={fps:.8f}[lumamerged]')
    return 'lumamerged'


def append_source_alpha_mask(filters: list[str], args, feather: int, *, horizontal: bool, vertical: bool, width: int, height: int, source_label: str) -> tuple[str, str]:
    """Build the source overlay's alpha as its own gray stream.

    Computing the alpha inside an rgba geq costs one interpreted expression per
    pixel per plane for every frame, which dominates the whole composite. Two
    cheaper shapes replace it:

    * The feather ramp depends only on X/Y, so it is evaluated once on a still
      frame and reused for the clip by alphamerge's frame sync. The still needs
      its own source -- a one-frame branch taken off a split of the source
      stream loses a frame at the head of the clip.
    * The black-region matte does depend on picture content, but its per-sample
      max(r,g,b) can be folded down by native filters first, leaving geq a
      single plane to walk instead of four.

    Returns the labels of the colour stream and of the gray alpha mask.
    """
    if getattr(args, 'source_black_transparent', False):
        expr = source_alpha_expr(args, feather, horizontal=horizontal, vertical=vertical, sampler=source_luma_max_expr)
        filters.extend([
            f'[{source_label}]split[srcrgb][srckey]',
            '[srckey]format=gbrp,extractplanes=g+b+r[srckeyg][srckeyb][srckeyr]',
            '[srckeyg][srckeyb]blend=all_mode=lighten[srckeygb]',
            '[srckeygb][srckeyr]blend=all_mode=lighten,format=gray[srcluma]',
            f"[srcluma]geq=lum='{expr}'[srcalphamask]",
        ])
        return 'srcrgb', 'srcalphamask'
    expr = source_alpha_expr(args, feather, horizontal=horizontal, vertical=vertical)
    filters.append(
        f"color=c=black:s={width}x{height}:d=1,trim=end_frame=1,format=gray,geq=lum='{expr}'[srcalphamask]"
    )
    return source_label, 'srcalphamask'


def build_filter(args, has_color, fps: float, has_outpainted: bool = True, source_size: tuple[int, int] | None = None, base_size: tuple[int, int] | None = None, luminance_plan: list[dict] | None = None):
    feather = max(1, int(args.feather_pixels))
    sat = max(0.0, normalized_percent(args.saturation, 0.82))
    color_opacity = max(0.0, min(1.0, normalized_percent(args.color_opacity, 1.0)))
    fps_text = f"{fps:.8f}"
    crop = source_crop_filter(args)
    color_input = 2 if has_outpainted else 1
    custom_mask_input = (3 if has_color else 2) if has_outpainted and getattr(args, 'custom_mask', None) else None
    # Optionally scale the outpainted video to the delivery output dimensions.
    # This corrects for LTX's model-safe quantisation (e.g. 704p → 720p) so the
    # final composite is at the user's intended resolution.
    out_w = int(args.output_width) if args.output_width else 0
    out_h = int(args.output_height) if args.output_height else 0
    scale_base = f",scale={out_w}:{out_h}:flags=lanczos" if (out_w and out_h) else ""
    if has_outpainted:
        if source_size and base_size:
            crops = tuple(int(getattr(args, key)) for key in ("crop_left", "crop_right", "crop_top", "crop_bottom"))
            placement = source_placement(source_size[0], source_size[1], base_size[0], base_size[1], crops)
            feather_horizontal = placement.x > 0 or placement.x + placement.width < base_size[0]
            feather_vertical = placement.y > 0 or placement.y + placement.height < base_size[1]
            filters = [
                f'[0:v]setpts=N/({fps_text}*TB),fps=fps={fps_text}{scale_base}[base]',
                f'[1:v]setpts=N/({fps_text}*TB),fps=fps={fps_text},{crop}scale={placement.width}:{placement.height}:flags=lanczos,setsar=1[src]',
            ]
            rgb_label, alpha_label = append_source_alpha_mask(
                filters,
                args,
                feather,
                horizontal=feather_horizontal,
                vertical=feather_vertical,
                width=placement.width,
                height=placement.height,
                source_label='src',
            )
            if custom_mask_input is not None:
                # Fold the custom mask into the alpha before it is merged, so the
                # source never has to be split and re-alphaextracted per frame.
                filters.extend([
                    f'[{custom_mask_input}:v]scale={base_size[0]}:{base_size[1]}:flags=neighbor,crop=w={placement.width}:h={placement.height}:x={placement.x}:y={placement.y},format=gray,negate[customkeep]',
                    f'[{alpha_label}][customkeep]blend=all_mode=multiply[srcalphacustom]',
                ])
                alpha_label = 'srcalphacustom'
            filters.append(f'[{rgb_label}]format=rgba[srcrgba]')
            filters.append(f'[srcrgba][{alpha_label}]alphamerge[srcm]')
            filters.append(f'[base][srcm]overlay=x={placement.x}:y={placement.y}[merged]')
        else:
            filters = [
                f'[0:v]setpts=N/({fps_text}*TB),fps=fps={fps_text}{scale_base}[base0]',
                f'[1:v]setpts=N/({fps_text}*TB),fps=fps={fps_text},{crop}setsar=1[src0]',
                '[src0][base0]scale2ref=w=trunc(oh*mdar/2)*2:h=ih[src][base]',
                f"[src]format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='{source_alpha_expr(args, feather, horizontal=True, vertical=True)}'[srcm]",
                '[base][srcm]overlay=x=(W-w)/2:y=(H-h)/2[merged]',
            ]
    else:
        filters = [
            f'[0:v]setpts=N/({fps_text}*TB),fps=fps={fps_text},{crop}setsar=1,format=yuv444p[merged]',
        ]
    base_label = append_reference_luminance_filter(filters, 'merged', luminance_plan or [], fps) if has_color else 'merged'
    if has_color:
        red, blue = temperature_balance(args.temperature)
        filters.append(f'[{color_input}:v]setpts=N/({fps_text}*TB),fps=fps={fps_text}[col0]')
        filters.append(f'[col0][{base_label}]scale2ref=w=iw:h=ih[colscaled][mergedref]')
        filters.append(f'[colscaled]eq=saturation={sat}:brightness=0:contrast=1,colorbalance=rs={red:.4f}:bs={blue:.4f},format=yuv444p[colfmt]')
        filters.append('[mergedref]format=yuv444p[basefmt]')
        if color_opacity < 1.0:
            filters.append(f'[basefmt][colfmt]blend=all_expr=A*(1-{color_opacity:.6f})+B*{color_opacity:.6f},format=yuv444p[colblend]')
            color_source = 'colblend'
        else:
            color_source = 'colfmt'
        filters.append(f'[basefmt]extractplanes=y,setsar=1[basey];[{color_source}]extractplanes=u+v[colu0][colv0]')
        filters.append('[colu0]setsar=1[colu];[colv0]setsar=1[colv]')
        final_frame = max((int(item['end_frame']) for item in (luminance_plan or []) if item.get('end_frame') is not None), default=0)
        output_label = 'vouttimed' if final_frame else 'vout'
        output_format = 'yuv444p10le' if getattr(args, 'intermediate_profile', '') == 'lossless' else 'yuv420p'
        filters.append(f'[basey][colu][colv]mergeplanes=0x001020:yuv444p,setsar=1,format={output_format}[{output_label}]')
        if final_frame:
            filters.append(f'[{output_label}]trim=end_frame={final_frame},setpts=N/({fps_text}*TB),fps=fps={fps_text}[vout]')
    else:
        filters.append('[merged]copy[vout]')
    return ';'.join(filters)


def input_args(outpainted: Path | None, source: Path, colorized: Path | None, custom_mask: Path | None) -> tuple[list[str], str]:
    """Build ffmpeg's input list for the composite, plus the audio stream to map.

    The custom mask is a single still frame and is deliberately not passed with
    -loop 1. An endless image input keeps blend's frame sync emitting frames long
    after the video streams have ended, so the encode never terminates and the
    output file grows without bound -- and -shortest cannot rescue it when the
    source carries no audio track to bound the output against. Frame sync already
    repeats the last mask frame for every video frame, which is what we want.
    """
    inputs: list[str] = []
    if outpainted:
        inputs += ['-i', str(outpainted), '-i', str(source)]
        audio_input = '1:a?'
    else:
        inputs += ['-i', str(source)]
        audio_input = '0:a?'
    if colorized:
        inputs += ['-i', str(colorized)]
    if custom_mask:
        inputs += ['-i', str(custom_mask)]
    return inputs, audio_input


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
    luminance_plan = []
    if colorized and args.reference_luminance_match:
        if args.manifest:
            manifest = resolve_path(args.manifest)
            luminance_plan = reference_luminance_plan(manifest, fps, args.reference_luminance_strength)
            matched = sum(1 for item in luminance_plan if item.get('matched'))
            print(f'Reference luminance matching: {matched}/{len(luminance_plan)} shot span(s), strength {args.reference_luminance_strength:g}%')
        else:
            print('Reference luminance matching requested without a manifest; using original source luminance.')
    source_size = probe_dimensions(ffmpeg, source)
    base_size = None
    if outpainted:
        base_size = (int(args.output_width), int(args.output_height)) if args.output_width and args.output_height else probe_dimensions(ffmpeg, outpainted)
    cmd = [ffmpeg, '-y']
    inputs, audio_input = input_args(outpainted, source, colorized, resolve_path(args.custom_mask) if args.custom_mask else None)
    cmd += inputs
    cmd += ['-filter_complex', build_filter(args, bool(colorized), fps, bool(outpainted), source_size, base_size, luminance_plan), '-map', '[vout]', '-map', audio_input, '-shortest', '-r', f'{fps:.8f}', '-fps_mode', 'cfr']
    partial = output.with_name(f"{output.stem}.partial.{os_safe_pid()}{output.suffix}")
    cmd += encoder_args(args)
    if args.encoder == 'prores':
        cmd += ['-c:a', 'pcm_s16le']
    else:
        cmd += ['-c:a', 'aac', '-b:a', '320k', '-movflags', '+faststart']
    cmd += [str(partial)]
    print(' '.join(cmd))
    if args.dry_run:
        return 0
    # The composite is one long ffmpeg pass with no natural milestones, so publish the
    # frame count the encode is working towards. ARP pairs it with ffmpeg's own
    # "frame=" counter to drive the Recomposition progress bar.
    duration = probe_duration(ffmpeg, outpainted or source)
    if duration > 0:
        print(f'Composite frames: {max(1, int(round(duration * fps)))} at {fps:.6f} fps', flush=True)
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
    parser.add_argument('--manifest', help='Shot manifest containing source_reference and color_reference pairs.')
    parser.add_argument('--reference-luminance-match', action=argparse.BooleanOptionalAction, default=False, help='Match each shot luminance to its colour reference using one stable tonal curve per shot.')
    parser.add_argument('--reference-luminance-strength', type=float, default=70.0, help='Strength of the shot-level reference luminance curve, from 0 to 100 percent.')
    parser.add_argument('--output-width', type=int, default=0, help='Scale outpainted video to this width before compositing (delivery upscale, e.g. 1280 to correct 704→720).')
    parser.add_argument('--output-height', type=int, default=0, help='Scale outpainted video to this height before compositing (delivery upscale, e.g. 720 to correct 704→720).')
    parser.add_argument('--source-black-transparent', action='store_true', help='Treat near-black source pixels as transparent so outpainted regions remain visible in the final composite.')
    parser.add_argument('--source-black-threshold', type=int, default=24, help='Maximum RGB channel value considered source black when --source-black-transparent is enabled.')
    parser.add_argument('--source-black-matte-shrink-pixels', type=int, default=2, help='Shrink the source matte by this many pixels around detected black regions to avoid dark resampling halos.')
    parser.add_argument('--custom-mask', help='Additive full-canvas mask whose selected pixels remain transparent in the source overlay.')
    parser.add_argument('--crop-left', type=int, default=0)
    parser.add_argument('--crop-right', type=int, default=0)
    parser.add_argument('--crop-top', type=int, default=0)
    parser.add_argument('--crop-bottom', type=int, default=0)
    parser.add_argument('--encoder', choices=['h264', 'prores'], default='h264')
    parser.add_argument('--crf', type=int, default=16)
    parser.add_argument('--preset', default='slow')
    parser.add_argument('--intermediate-profile', choices=['low', 'medium', 'high', 'lossless'], default='')
    parser.add_argument('--ffmpeg')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    return parser


def main():
    return run(build_parser().parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
