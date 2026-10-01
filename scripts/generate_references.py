#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, io, json
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from common import ROOT, file_fingerprint, format_time, resolve_path, root_relative, safe_stem, resumable_output, signature_path, write_signature

DEFAULT_REFERENCE_ROOT = ROOT / 'intermediate' / 'outpainted_references'
DEFAULT_COLOR_REFERENCE_ROOT = ROOT / 'intermediate' / 'outpainted_references_color'
DEFAULT_MANIFEST_ROOT = ROOT / 'manifests' / 'references'

@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float


@dataclass
class Sample:
    frame: int
    time: float
    mean_luma: float
    black_ratio: float
    sharpness: float
    hist: np.ndarray
    color_hist: np.ndarray
    edge_hist: np.ndarray
    dhash: np.ndarray
    gray_small: np.ndarray


@dataclass
class Shot:
    index: int
    start_frame: int
    end_frame: int
    samples: list[Sample]


@dataclass
class ReferenceRow:
    index: int
    start_frame: int
    end_frame: int
    selected_frame: int
    selected_time: float
    source_reference: Path
    color_reference: Path
    reused_color_from: Path | None = None


def probe_video(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f'Could not read video metadata: {path}')
    return VideoInfo(width, height, fps, frame_count, frame_count / fps)

def frame_hist(gray):
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    return cv2.normalize(hist, hist).flatten().astype(np.float32)


def color_hist(frame):
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten().astype(np.float32)


def edge_hist(gray):
    edges = cv2.Canny(gray, 60, 140)
    hist = cv2.calcHist([edges], [0], None, [16], [0, 256])
    return cv2.normalize(hist, hist).flatten().astype(np.float32)


def dhash(gray):
    tiny = cv2.resize(gray, (17, 16), interpolation=cv2.INTER_AREA)
    return (tiny[:, 1:] > tiny[:, :-1]).astype(np.uint8).flatten()


def hist_distance(a, b):
    corr = cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CORREL)
    return 1.0 if np.isnan(corr) else float(max(0, min(2, 1 - corr)))


def array_distance(a, b):
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def hash_distance(a, b):
    return float(np.mean(a != b))


def analyze_frame(frame, idx, fps):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
    return Sample(
        idx,
        idx / fps,
        float(np.mean(small)),
        float(np.mean(small < 12)),
        float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        frame_hist(small),
        color_hist(frame),
        edge_hist(small),
        dhash(small),
        small,
    )


def analyze_image(path: Path) -> Sample:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f'Could not read image: {path}')
    return analyze_frame(frame, 0, 1.0)
def transition_score(a, b):
    return (
        0.28 * hist_distance(a.color_hist, b.color_hist)
        + 0.22 * array_distance(a.gray_small, b.gray_small)
        + 0.18 * hash_distance(a.dhash, b.dhash)
        + 0.14 * hist_distance(a.hist, b.hist)
        + 0.08 * hist_distance(a.edge_hist, b.edge_hist)
        + 0.06 * abs(a.mean_luma - b.mean_luma) / 255
        + 0.04 * abs(a.black_ratio - b.black_ratio)
    )


def reuse_similarity_score(a, b):
    return (
        0.26 * array_distance(a.gray_small, b.gray_small)
        + 0.22 * hash_distance(a.dhash, b.dhash)
        + 0.18 * hist_distance(a.hist, b.hist)
        + 0.14 * hist_distance(a.color_hist, b.color_hist)
        + 0.10 * hist_distance(a.edge_hist, b.edge_hist)
        + 0.06 * abs(a.mean_luma - b.mean_luma) / 255
        + 0.04 * abs(a.black_ratio - b.black_ratio)
    )


def is_fade(sample, args):
    return sample.black_ratio >= args.fade_black_ratio or sample.mean_luma <= args.fade_luma


def near(boundaries, frame, gap):
    return any(abs(frame - boundary) < gap for boundary in boundaries)

def sample_video(path, info, args):
    step = 1 if args.sample_seconds <= 0 else max(1, int(round(args.sample_seconds * info.fps)))
    cap = cv2.VideoCapture(str(path))
    out = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            out.append(analyze_frame(frame, idx, info.fps))
        idx += 1
    cap.release()
    if not out:
        raise RuntimeError(f'No frames sampled from {path}')
    return out


def dissolve_score(samples, i, window):
    before = max(0, i - window)
    after = min(len(samples) - 1, i + window)
    return 0.0 if before == i or after == i else transition_score(samples[before], samples[after])


def refine_boundary(samples, start, end):
    if end <= start:
        return samples[end].frame
    strongest_index = max(range(start + 1, end + 1), key=lambda i: transition_score(samples[i - 1], samples[i]))
    return samples[strongest_index].frame

def detect_shots(samples, info, args):
    min_frames = max(1, int(round(args.min_shot_seconds * info.fps)))
    dedupe = max(1, int(round(args.boundary_dedupe_seconds * info.fps)))
    dissolve_window = max(1, int(round(args.dissolve_window_seconds * info.fps)))
    dissolve_gap = max(min_frames, int(round(args.dissolve_min_gap_seconds * info.fps)))
    scores = [0.0] + [transition_score(samples[i - 1], samples[i]) for i in range(1, len(samples))]
    nz = np.array([score for score in scores[1:] if score > 0], dtype=np.float32)
    if nz.size:
        median = float(np.median(nz))
        mad = float(np.median(np.abs(nz - median)))
        threshold = max(args.shot_threshold, median + args.dynamic_threshold_scale * max(mad * 1.4826, 0.001))
    else:
        threshold = args.shot_threshold
    dissolve_scores = [dissolve_score(samples, i, dissolve_window) for i in range(len(samples))]
    boundaries = [0]
    last = 0
    anchor = samples[0]
    fade_start = None
    for i in range(1, len(samples)):
        prev, cur = samples[i - 1], samples[i]
        far = cur.frame - last >= min_frames
        prev_fade = is_fade(prev, args)
        cur_fade = is_fade(cur, args)
        adjacent = scores[i]
        prev_score = scores[i - 1] if i > 1 else 0.0
        next_score = scores[i + 1] if i + 1 < len(scores) else 0.0
        local_peak = adjacent >= prev_score and adjacent >= next_score
        peak_margin = adjacent - max(prev_score, next_score)
        if cur_fade and not prev_fade and fade_start is None:
            fade_start = i
        if fade_start is not None and prev_fade and not cur_fade:
            boundary = refine_boundary(samples, fade_start, i)
            if boundary - last >= min_frames and not near(boundaries, boundary, dedupe):
                boundaries.append(boundary)
                last = boundary
                anchor = cur
            fade_start = None
            continue

        dissolve = dissolve_scores[i]
        previous_dissolve = dissolve_scores[i - 1] if i > 1 else 0.0
        next_dissolve = dissolve_scores[i + 1] if i + 1 < len(dissolve_scores) else 0.0
        dissolve_peak = dissolve >= previous_dissolve and dissolve >= next_dissolve
        hard = (local_peak and adjacent >= threshold and peak_margin >= args.peak_margin) or adjacent >= threshold * 1.8
        gradual = (
            args.anchor_threshold > 0
            and cur.frame - last >= int(round(args.anchor_min_seconds * info.fps))
            and adjacent >= args.anchor_adjacent_floor
            and transition_score(anchor, cur) >= args.anchor_threshold
        )
        cross = (
            args.dissolve_threshold > 0
            and dissolve_peak
            and dissolve >= args.dissolve_threshold
            and cur.frame - last >= dissolve_gap
            and not near(boundaries, cur.frame, dissolve_gap)
        )
        if far and (hard or gradual or cross):
            if hard:
                boundary = cur.frame
                this_dedupe = min_frames
            elif cross:
                boundary = cur.frame
                this_dedupe = dedupe
            else:
                boundary = refine_boundary(samples, max(0, i - 3), i)
                this_dedupe = dedupe
            if boundary - last >= min_frames and not near(boundaries, boundary, this_dedupe):
                boundaries.append(boundary)
                last = boundary
                anchor = cur
                fade_start = None
    boundaries.append(info.frame_count)
    return [
        Shot(idx, start, end, [sample for sample in samples if start <= sample.frame < end])
        for idx, (start, end) in enumerate(zip(boundaries, boundaries[1:]))
    ]

def representative_sample(samples,start,end,fps):
    usable=[s for s in samples if s.black_ratio<0.82 and 14<=s.mean_luma<=242]; candidates=usable or samples
    if not candidates:
        gray=np.full((90,160),128,dtype=np.uint8); frame=(start+end)//2; return Sample(frame,frame/fps,128,0,0,frame_hist(gray),np.zeros(24*16,dtype=np.float32),edge_hist(gray),dhash(gray),gray)
    midpoint=(start+end)/2; max_sharp=max(s.sharpness for s in candidates) or 1.0
    return max(candidates,key=lambda s:(s.sharpness/max_sharp)-1.8*s.black_ratio-.35*abs(s.mean_luma-92)/255-.65*abs(s.frame-midpoint)/max(1,end-start))
def read_frame(path,frame_index):
    cap=cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,frame_index)); ok,frame=cap.read(); cap.release()
    if not ok: raise RuntimeError(f'Could not read frame {frame_index} from {path}')
    return frame
def write_png(path,frame):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),frame): raise RuntimeError(f'Could not write image: {path}')

def existing_color_candidates(color_root,bw_root):
    out=[]
    if not color_root.exists() or not bw_root.exists(): return out
    for source in bw_root.rglob('*.png'):
        rel=source.relative_to(bw_root); color=color_root/rel
        if color.exists():
            try: out.append((color,analyze_image(source)))
            except RuntimeError: pass
    return out

def build_rows(args,source_path,info,shots):
    clip_id=args.reference_set or safe_stem(source_path.name); src_dir=args.reference_root/clip_id; col_dir=args.color_reference_root/clip_id
    candidates=existing_color_candidates(args.color_reference_root,args.reference_root) if args.reuse_existing_references else []
    rows=[]
    for index,shot in enumerate(shots):
        selected=representative_sample(shot.samples,shot.start_frame,shot.end_frame,info.fps); name=f"cut_{index:04d}_{format_time(selected.time).replace(':','.')}.png"; src=src_dir/name; color=col_dir/name; reused=None; best=999
        for cpath,csample in candidates:
            score=reuse_similarity_score(selected,csample)
            if score<best: best=score; reused=cpath
        if reused is not None and best<=args.existing_reuse_threshold: color=reused
        rows.append(ReferenceRow(index,shot.start_frame,shot.end_frame,selected.frame,selected.time,src,color,reused if color==reused else None))
        if args.limit is not None and len(rows)>=args.limit: break
    return rows

MANIFEST_FIELDS=['enabled','start_frame','end_frame','selected_frame','end','source_reference','color_reference','prompt','fade_to_next','crossfade_seconds']
# Everything that decides where the cuts fall and which frames are extracted.
DETECTION_ARGS=('sample_seconds','shot_threshold','dynamic_threshold_scale','peak_margin','anchor_threshold','anchor_min_seconds','anchor_adjacent_floor','dissolve_threshold','dissolve_window_seconds','dissolve_min_gap_seconds','boundary_dedupe_seconds','min_shot_seconds','fade_black_ratio','fade_luma','reuse_existing_references','existing_reuse_threshold','frame_width','frame_height','reference_set','limit')

def read_manifest(path):
    """(source_video, fieldnames, rows) of an existing shot manifest; empty when there is none."""
    if not path.is_file(): return '',[],[]
    with path.open('r',encoding='utf-8-sig',newline='') as h: lines=h.readlines()
    source=next((line.split('=',1)[1].strip() for line in lines if line.startswith('# source_video=')),'')
    reader=csv.DictReader(line for line in lines if not line.startswith('#'))
    rows=list(reader); return source,list(reader.fieldnames or []),rows

def write_manifest(path,source_path,rows,info,kept=None,fields=None):
    """Write the manifest. kept maps a row index to a previous manifest row (every column) that replaces it."""
    kept=kept or {}; fieldnames=MANIFEST_FIELDS+[f for f in (fields or []) if f not in MANIFEST_FIELDS]
    out=[]
    for i,row in enumerate(rows):
        fresh={'enabled':'true','start_frame':row.start_frame,'end_frame':row.end_frame,'selected_frame':row.selected_frame,'end':format_time(min(row.end_frame/info.fps,info.duration)),'source_reference':root_relative(row.source_reference) if row.source_reference else '','color_reference':root_relative(row.color_reference),'prompt':'','fade_to_next':'false','crossfade_seconds':''}
        out.append({**fresh,**{k:v for k,v in kept.get(i,{}).items() if k in fieldnames and v is not None}})
    buf=io.StringIO(newline=''); buf.write(f'# source_video={root_relative(source_path)}\n')
    w=csv.DictWriter(buf,fieldnames=fieldnames,lineterminator='\n',extrasaction='ignore'); w.writeheader(); w.writerows(out); text=buf.getvalue()
    # Unchanged bytes keep the file's mtime, so later phases do not see a changed input.
    if path.is_file() and path.read_text(encoding='utf-8-sig')==text: return
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as h: h.write(text)
    tmp.replace(path)

def carry_over_rows(rows,previous,fps):
    """Keep a previous manifest row, edits and all, wherever re-detection found the same shot span.

    A re-rendered outpaint (one regenerated chunk, say) changes the video's bytes but rarely its
    cuts, so prompts, fades, disabled shots and chosen references survive. Returns {index: row}."""
    by_span={}
    for old in previous:
        try: by_span[(int(old.get('start_frame') or ''),int(old.get('end_frame') or ''))]=old
        except ValueError: pass
    kept={}
    for i,row in enumerate(rows):
        old=by_span.get((row.start_frame,row.end_frame))
        if old is None: continue
        try: frame=int(old.get('selected_frame') or row.selected_frame)
        except ValueError: frame=row.selected_frame
        src=resolve_path(old['source_reference']) if old.get('source_reference') else None
        color=resolve_path(old['color_reference']) if old.get('color_reference') else row.color_reference
        rows[i]=ReferenceRow(row.index,row.start_frame,row.end_frame,frame,frame/fps,src,color); kept[i]=old
    return kept

def manifest_signature(args,source_path,fingerprint):
    return {'version':1,'generator':'generate_references.py/manifest','source_video':root_relative(source_path),'source_fingerprint':fingerprint,'reference_root':root_relative(args.reference_root),'color_reference_root':root_relative(args.color_reference_root),**{k:getattr(args,k) for k in DETECTION_ARGS}}

def frames_present(manifest):
    _source,_fields,rows=read_manifest(manifest)
    return bool(rows) and all(resolve_path(r['source_reference']).is_file() for r in rows if r.get('source_reference'))

def manifest_reference_rows(rows,fps):
    """The manifest's own rows as ReferenceRows, or None if any row lacks integer frame spans."""
    out=[]
    for index,row in enumerate(rows):
        try: start=int(row['start_frame']); end=int(row['end_frame']); frame=int(row.get('selected_frame') or start)
        except (KeyError,TypeError,ValueError): return None
        src=resolve_path(row['source_reference']) if row.get('source_reference') else None
        color=resolve_path(row['color_reference']) if row.get('color_reference') else None
        out.append(ReferenceRow(index,start,end,frame,frame/fps,src,color))
    return out

def cuts_still_fit(manifest,source_path,info,args):
    """The existing shots still describe this video: it is the manifest's own video, with the same
    frame count, and the detection settings are unchanged (or unknown, for a manifest written
    before manifests were signed). Outpainting repaints only the edges, so a re-render changes
    the video's bytes, not where its shots fall; re-cutting it would only throw away edits."""
    source,_fields,rows=read_manifest(manifest)
    if not rows or not source or resolve_path(source).resolve()!=source_path.resolve(): return False
    spans=manifest_reference_rows(rows,info.fps)
    if not spans or max(row.end_frame for row in spans)!=info.frame_count: return False
    try: stored=json.loads(signature_path(manifest).read_text(encoding='utf-8-sig'))
    except (OSError,ValueError): return True
    now=json.loads(json.dumps({k:getattr(args,k) for k in DETECTION_ARGS}))
    return all(stored.get(k)==v for k,v in now.items())

def source_signature(source_path,row,out_w=0,out_h=0,fingerprint=None): return {'version':3,'source_video':root_relative(source_path),'source_fingerprint':fingerprint or file_fingerprint(source_path),'selected_frame':row.selected_frame,'selected_time':row.selected_time,'output_width':out_w or 0,'output_height':out_h or 0,'grayscale':True,'generator':'generate_references.py'}
def extract_frames(args,source_path,info,rows,fingerprint=None):
    out_w=int(args.frame_width or 0); out_h=int(args.frame_height or 0)
    write_w=out_w or info.width; write_h=out_h or info.height
    fingerprint=fingerprint or file_fingerprint(source_path)
    rows=[row for row in rows if row.source_reference]
    expected={row.source_reference for row in rows}
    if args.prune_source_frames:
        for folder in {p.parent for p in expected}:
            if folder.exists():
                for png in folder.glob('cut_*.png'):
                    if png not in expected: png.unlink(missing_ok=True); png.with_suffix(png.suffix+'.sig.json').unlink(missing_ok=True); print(f'Removed orphan source frame: {png}')
    for row in rows:
        sig=source_signature(source_path,row,out_w,out_h,fingerprint)
        if not args.force and not args.regenerate_source_frames and resumable_output(row.source_reference,sig,width=write_w,height=write_h): print(f'Reuse source frame {row.index:04d}: {row.source_reference}'); continue
        if not args.force and row.source_reference.exists() and resumable_output(row.source_reference,sig,width=write_w,height=write_h): print(f'Reuse source frame {row.index:04d}: {row.source_reference}'); continue
        frame=read_frame(source_path,row.selected_frame)
        # Dearchive can introduce colour. Reference generation is a separate, user-directed
        # colourization pass, so always restore a neutral B&W source still before Qwen creates
        # the colour guide used by Deep Exemplar/ColorMNet.
        frame=cv2.cvtColor(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
        if out_w and out_h and (frame.shape[1]!=out_w or frame.shape[0]!=out_h):
            frame=cv2.resize(frame,(out_w,out_h),interpolation=cv2.INTER_LANCZOS4)
        write_png(row.source_reference,frame); write_signature(row.source_reference,sig); print(f'Wrote source frame {row.index:04d} ({write_w}x{write_h}): {row.source_reference}')
def default_manifest_path(source_path): return DEFAULT_MANIFEST_ROOT/f'colorize_manifest_{safe_stem(source_path.name)}_shots_auto.csv'


def build_parser():
    parser=argparse.ArgumentParser(description='Detect cuts in an outpainted clip and write reference-image manifests.')
    parser.add_argument('--source-video',required=True,help='Video to analyse. Relative paths are resolved from the repo root.')
    parser.add_argument('--output-manifest',type=Path,help='Manifest to write. Defaults to manifests/references/colorize_manifest_<video>_shots_auto.csv')
    parser.add_argument('--reference-root',type=Path,default=DEFAULT_REFERENCE_ROOT,help='Root for extracted black-and-white reference frames.')
    parser.add_argument('--color-reference-root',type=Path,default=DEFAULT_COLOR_REFERENCE_ROOT,help='Root for colorized reference frames.')
    parser.add_argument('--reference-set',help='Folder name under the reference roots. Defaults to the source-video stem.')
    parser.add_argument('--sample-seconds',type=float,default=0.0,help='Analysis interval. 0 means every frame.')
    parser.add_argument('--shot-threshold',type=float,default=0.075)
    parser.add_argument('--dynamic-threshold-scale',type=float,default=2.4)
    parser.add_argument('--peak-margin',type=float,default=0.01)
    parser.add_argument('--anchor-threshold',type=float,default=0.50)
    parser.add_argument('--anchor-min-seconds',type=float,default=8.0)
    parser.add_argument('--anchor-adjacent-floor',type=float,default=0.004)
    parser.add_argument('--dissolve-threshold',type=float,default=0.20)
    parser.add_argument('--dissolve-window-seconds',type=float,default=1.5)
    parser.add_argument('--dissolve-min-gap-seconds',type=float,default=4.0)
    parser.add_argument('--boundary-dedupe-seconds',type=float,default=1.5)
    parser.add_argument('--min-shot-seconds',type=float,default=1.0)
    parser.add_argument('--fade-black-ratio',type=float,default=0.72)
    parser.add_argument('--fade-luma',type=float,default=18.0)
    parser.add_argument('--reuse-existing-references',dest='reuse_existing_references',action='store_true',default=True,help='Reuse existing color references by source-frame similarity.')
    parser.add_argument('--no-reuse-existing-references',dest='reuse_existing_references',action='store_false')
    parser.add_argument('--existing-reuse-threshold',type=float,default=0.025)
    parser.add_argument('--regenerate-source-frames',dest='regenerate_source_frames',action='store_true',default=True,help='Rewrite source screenshots by default so stale cut data is flushed out.')
    parser.add_argument('--keep-existing-source-frames',dest='regenerate_source_frames',action='store_false')
    parser.add_argument('--no-prune-source-frames',dest='prune_source_frames',action='store_false',default=True)
    parser.add_argument('--frame-width',type=int,default=0,help='Resize extracted reference frames to this width (e.g. delivery width to correct model-safe LTX output).')
    parser.add_argument('--frame-height',type=int,default=0,help='Resize extracted reference frames to this height (e.g. delivery height to correct model-safe LTX output).')
    parser.add_argument('--limit',type=int,help='Limit rows for smoke tests.')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--force',action='store_true')
    return parser

def main():
    args=build_parser().parse_args()
    source_path=resolve_path(args.source_video)
    if not source_path.exists():
        raise FileNotFoundError(f'Source video not found: {source_path}')
    args.reference_root=resolve_path(args.reference_root)
    args.color_reference_root=resolve_path(args.color_reference_root)
    manifest=resolve_path(args.output_manifest) if args.output_manifest else default_manifest_path(source_path)
    fingerprint=file_fingerprint(source_path)
    sig=manifest_signature(args,source_path,fingerprint)
    if not args.force and resumable_output(manifest,sig) and frames_present(manifest):
        # Re-detecting rewrites every row, so an up-to-date manifest (and its edits) is left alone.
        print(f'Reuse shot manifest (same video and detection settings): {manifest}'); return 0
    info=probe_video(source_path)
    if not args.force and cuts_still_fit(manifest,source_path,info,args):
        _source,_fields,rows=read_manifest(manifest)
        print(f'Keeping all {len(rows)} shots and their edits: the video changed but its length and the detection settings did not. Refreshing the reference frames only.')
        if args.dry_run: return 0
        extract_frames(args,source_path,info,manifest_reference_rows(rows,info.fps),fingerprint)
        write_signature(manifest,sig)
        print(f'Reuse shot manifest: {manifest}'); return 0
    samples=sample_video(source_path,info,args)
    shots=detect_shots(samples,info,args)
    rows=build_rows(args,source_path,info,shots)
    _previous_source,previous_fields,previous_rows=read_manifest(manifest) if not args.force else ('',[],[])
    kept=carry_over_rows(rows,previous_rows,info.fps)
    if previous_rows: print(f'Kept {len(kept)} of {len(rows)} shot(s), with their edits, from the previous manifest (same frame span).')
    reused=sum(1 for row in rows if row.reused_color_from)
    print(f'Source: {source_path}')
    print(f'Video: {info.width}x{info.height}, {info.fps:.6g} fps, {info.frame_count} frames, {format_time(info.duration)}')
    print(f'Detected {len(shots)} cut spans; writing {len(rows)} manifest rows, {len(rows)-reused} unique/new reference targets, {reused} reused.')
    for row in rows:
        note=f' reuse={root_relative(row.reused_color_from)}' if row.reused_color_from else ''
        print(f'cut {row.index:04d} selected={format_time(row.selected_time)} end={format_time(row.end_frame/info.fps)} ref={root_relative(row.color_reference)}{note}')
    if args.dry_run:
        print(f'Dry run; would write {manifest}')
        return 0
    extract_frames(args,source_path,info,rows,fingerprint)
    write_manifest(manifest,source_path,rows,info,kept,previous_fields)
    write_signature(manifest,sig)
    print(f'Wrote manifest: {manifest}')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
