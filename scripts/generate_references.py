#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from common import ROOT, file_fingerprint, format_time, resolve_path, root_relative, safe_stem, resumable_output, write_signature

DEFAULT_REFERENCE_ROOT = ROOT / 'intermediate' / 'outpainted_references'
DEFAULT_COLOR_REFERENCE_ROOT = ROOT / 'intermediate' / 'outpainted_references_color'
DEFAULT_MANIFEST_ROOT = ROOT / 'manifests' / 'references'

@dataclass
class VideoInfo:
    width:int; height:int; fps:float; frame_count:int; duration:float
@dataclass
class Sample:
    frame:int; time:float; mean_luma:float; black_ratio:float; sharpness:float; hist:np.ndarray; color_hist:np.ndarray; edge_hist:np.ndarray; dhash:np.ndarray; gray_small:np.ndarray
@dataclass
class Shot:
    index:int; start_frame:int; end_frame:int; samples:list[Sample]
@dataclass
class ReferenceRow:
    index:int; end_frame:int; selected_frame:int; selected_time:float; source_reference:Path; color_reference:Path; reused_color_from:Path|None=None

def probe_video(path:Path)->VideoInfo:
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise RuntimeError(f'Could not open video: {path}')
    fps=cap.get(cv2.CAP_PROP_FPS) or 24.0; fc=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0); cap.release()
    if fc<=0 or w<=0 or h<=0: raise RuntimeError(f'Could not read video metadata: {path}')
    return VideoInfo(w,h,fps,fc,fc/fps)

def frame_hist(gray):
    hist=cv2.calcHist([gray],[0],None,[32],[0,256]); return cv2.normalize(hist,hist).flatten().astype(np.float32)
def color_hist(frame):
    small=cv2.resize(frame,(160,90),interpolation=cv2.INTER_AREA); hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV); hist=cv2.calcHist([hsv],[0,1],None,[24,16],[0,180,0,256]); return cv2.normalize(hist,hist).flatten().astype(np.float32)
def edge_hist(gray):
    edges=cv2.Canny(gray,60,140); hist=cv2.calcHist([edges],[0],None,[16],[0,256]); return cv2.normalize(hist,hist).flatten().astype(np.float32)
def dhash(gray):
    tiny=cv2.resize(gray,(17,16),interpolation=cv2.INTER_AREA); return (tiny[:,1:]>tiny[:,:-1]).astype(np.uint8).flatten()
def hist_distance(a,b):
    corr=cv2.compareHist(a.astype(np.float32),b.astype(np.float32),cv2.HISTCMP_CORREL)
    return 1.0 if np.isnan(corr) else float(max(0,min(2,1-corr)))
def array_distance(a,b): return float(np.mean(np.abs(a.astype(np.float32)-b.astype(np.float32)))/255.0)
def hash_distance(a,b): return float(np.mean(a!=b))

def analyze_frame(frame,idx,fps):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); small=cv2.resize(gray,(160,90),interpolation=cv2.INTER_AREA)
    return Sample(idx,idx/fps,float(np.mean(small)),float(np.mean(small<12)),float(cv2.Laplacian(gray,cv2.CV_64F).var()),frame_hist(small),color_hist(frame),edge_hist(small),dhash(small),small)
def analyze_image(path:Path)->Sample:
    frame=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if frame is None: raise RuntimeError(f'Could not read image: {path}')
    return analyze_frame(frame,0,1.0)
def transition_score(a,b):
    return .28*hist_distance(a.color_hist,b.color_hist)+.22*array_distance(a.gray_small,b.gray_small)+.18*hash_distance(a.dhash,b.dhash)+.14*hist_distance(a.hist,b.hist)+.08*hist_distance(a.edge_hist,b.edge_hist)+.06*abs(a.mean_luma-b.mean_luma)/255+.04*abs(a.black_ratio-b.black_ratio)
def reuse_similarity_score(a,b):
    return .26*array_distance(a.gray_small,b.gray_small)+.22*hash_distance(a.dhash,b.dhash)+.18*hist_distance(a.hist,b.hist)+.14*hist_distance(a.color_hist,b.color_hist)+.10*hist_distance(a.edge_hist,b.edge_hist)+.06*abs(a.mean_luma-b.mean_luma)/255+.04*abs(a.black_ratio-b.black_ratio)
def is_fade(s,args): return s.black_ratio>=args.fade_black_ratio or s.mean_luma<=args.fade_luma
def near(boundaries,frame,gap): return any(abs(frame-b)<gap for b in boundaries)

def sample_video(path,info,args):
    step=1 if args.sample_seconds<=0 else max(1,int(round(args.sample_seconds*info.fps))); cap=cv2.VideoCapture(str(path)); out=[]; idx=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if idx%step==0: out.append(analyze_frame(frame,idx,info.fps))
        idx+=1
    cap.release()
    if not out: raise RuntimeError(f'No frames sampled from {path}')
    return out

def dissolve_score(samples,i,window):
    before=max(0,i-window); after=min(len(samples)-1,i+window)
    return 0.0 if before==i or after==i else transition_score(samples[before],samples[after])
def refine_boundary(samples,start,end):
    if end<=start: return samples[end].frame
    return samples[max(range(start+1,end+1),key=lambda i:transition_score(samples[i-1],samples[i]))].frame

def detect_shots(samples,info,args):
    min_frames=max(1,int(round(args.min_shot_seconds*info.fps))); dedupe=max(1,int(round(args.boundary_dedupe_seconds*info.fps))); dw=max(1,int(round(args.dissolve_window_seconds*info.fps))); dgap=max(min_frames,int(round(args.dissolve_min_gap_seconds*info.fps)))
    scores=[0.0]+[transition_score(samples[i-1],samples[i]) for i in range(1,len(samples))]
    nz=np.array([s for s in scores[1:] if s>0],dtype=np.float32)
    if nz.size:
        med=float(np.median(nz)); mad=float(np.median(np.abs(nz-med))); threshold=max(args.shot_threshold,med+args.dynamic_threshold_scale*max(mad*1.4826,0.001))
    else: threshold=args.shot_threshold
    ds=[dissolve_score(samples,i,dw) for i in range(len(samples))]
    boundaries=[0]; last=0; anchor=samples[0]; fade_start=None
    for i in range(1,len(samples)):
        prev,cur=samples[i-1],samples[i]; far=cur.frame-last>=min_frames; prev_f=is_fade(prev,args); cur_f=is_fade(cur,args)
        adjacent=scores[i]; prev_score=scores[i-1] if i>1 else 0.0; next_score=scores[i+1] if i+1<len(scores) else 0.0
        local_peak=adjacent>=prev_score and adjacent>=next_score; peak_margin=adjacent-max(prev_score,next_score)
        if cur_f and not prev_f and fade_start is None: fade_start=i
        if fade_start is not None and prev_f and not cur_f:
            boundary=refine_boundary(samples,fade_start,i)
            if boundary-last>=min_frames and not near(boundaries,boundary,dedupe): boundaries.append(boundary); last=boundary; anchor=cur
            fade_start=None; continue
        dissolve=ds[i]; pd=ds[i-1] if i>1 else 0.0; nd=ds[i+1] if i+1<len(ds) else 0.0; dissolve_peak=dissolve>=pd and dissolve>=nd
        hard=local_peak and adjacent>=threshold and peak_margin>=args.peak_margin
        gradual=args.anchor_threshold>0 and cur.frame-last>=int(round(args.anchor_min_seconds*info.fps)) and adjacent>=args.anchor_adjacent_floor and transition_score(anchor,cur)>=args.anchor_threshold
        cross=args.dissolve_threshold>0 and dissolve_peak and dissolve>=args.dissolve_threshold and cur.frame-last>=dgap and not near(boundaries,cur.frame,dgap)
        if far and (hard or gradual or cross):
            if hard: boundary=cur.frame; this_dedupe=min_frames
            elif cross: boundary=cur.frame; this_dedupe=dedupe
            else: boundary=refine_boundary(samples,max(0,i-3),i); this_dedupe=dedupe
            if boundary-last>=min_frames and not near(boundaries,boundary,this_dedupe): boundaries.append(boundary); last=boundary; anchor=cur; fade_start=None
    boundaries.append(info.frame_count)
    return [Shot(idx,start,end,[s for s in samples if start<=s.frame<end]) for idx,(start,end) in enumerate(zip(boundaries,boundaries[1:]))]

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
        rows.append(ReferenceRow(index,shot.end_frame,selected.frame,selected.time,src,color,reused if color==reused else None))
        if args.limit is not None and len(rows)>=args.limit: break
    return rows

def write_manifest(path,source_path,rows,info):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as h:
        h.write(f'# source_video={root_relative(source_path)}\n'); w=csv.writer(h,lineterminator='\n'); w.writerow(['enabled','end','source_reference','color_reference'])
        for row in rows: w.writerow(['true',format_time(min(row.end_frame/info.fps,info.duration)),root_relative(row.source_reference),root_relative(row.color_reference)])
    tmp.replace(path)
def source_signature(source_path,row): return {'version':1,'source_video':root_relative(source_path),'source_fingerprint':file_fingerprint(source_path),'selected_frame':row.selected_frame,'selected_time':row.selected_time,'generator':'generate_references.py'}
def extract_frames(args,source_path,rows):
    expected={row.source_reference for row in rows}
    if args.prune_source_frames:
        for folder in {p.parent for p in expected}:
            if folder.exists():
                for png in folder.glob('cut_*.png'):
                    if png not in expected: png.unlink(missing_ok=True); png.with_suffix(png.suffix+'.sig.json').unlink(missing_ok=True); print(f'Removed orphan source frame: {png}')
    for row in rows:
        sig=source_signature(source_path,row)
        if not args.force and not args.regenerate_source_frames and resumable_output(row.source_reference,sig,width=info.width,height=info.height): print(f'Reuse source frame {row.index:04d}: {row.source_reference}'); continue
        if not args.force and row.source_reference.exists() and resumable_output(row.source_reference,sig,width=info.width,height=info.height): print(f'Reuse source frame {row.index:04d}: {row.source_reference}'); continue
        write_png(row.source_reference,read_frame(source_path,row.selected_frame)); write_signature(row.source_reference,sig); print(f'Wrote source frame {row.index:04d}: {row.source_reference}')
def default_manifest_path(source_path): return DEFAULT_MANIFEST_ROOT/f'colorize_manifest_{safe_stem(source_path.name)}_shots_auto.csv'


def build_parser():
    parser=argparse.ArgumentParser(description='Detect cuts in an outpainted clip and write reference-image manifests.')
    parser.add_argument('--source-video',required=True,help='Video to analyse. Relative paths are resolved from the repo root.')
    parser.add_argument('--output-manifest',type=Path,help='Manifest to write. Defaults to manifests/references/colorize_manifest_<video>_shots_auto.csv')
    parser.add_argument('--reference-root',type=Path,default=DEFAULT_REFERENCE_ROOT,help='Root for extracted black-and-white reference frames.')
    parser.add_argument('--color-reference-root',type=Path,default=DEFAULT_COLOR_REFERENCE_ROOT,help='Root for colorized reference frames.')
    parser.add_argument('--reference-set',help='Folder name under the reference roots. Defaults to the source-video stem.')
    parser.add_argument('--sample-seconds',type=float,default=0.0,help='Analysis interval. 0 means every frame.')
    parser.add_argument('--shot-threshold',type=float,default=0.09)
    parser.add_argument('--dynamic-threshold-scale',type=float,default=3.2)
    parser.add_argument('--peak-margin',type=float,default=0.01)
    parser.add_argument('--anchor-threshold',type=float,default=0.36)
    parser.add_argument('--anchor-min-seconds',type=float,default=4.0)
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
    info=probe_video(source_path)
    samples=sample_video(source_path,info,args)
    shots=detect_shots(samples,info,args)
    rows=build_rows(args,source_path,info,shots)
    reused=sum(1 for row in rows if row.reused_from)
    print(f'Source: {source_path}')
    print(f'Video: {info.width}x{info.height}, {info.fps:.6g} fps, {info.frame_count} frames, {format_time(info.duration)}')
    print(f'Detected {len(shots)} cut spans; writing {len(rows)} manifest rows, {len(rows)-reused} unique/new reference targets, {reused} reused.')
    for row in rows:
        note=f' reuse={root_relative(row.reused_from)}' if row.reused_from else ''
        print(f'cut {row.index:04d} selected={format_time(row.selected_time)} end={format_time(row.end_frame/info.fps)} ref={root_relative(row.color_reference)}{note}')
    if args.dry_run:
        print(f'Dry run; would write {manifest}')
        return 0
    extract_frames(args,source_path,rows)
    write_manifest(manifest,source_path,rows,info)
    print(f'Wrote manifest: {manifest}')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
