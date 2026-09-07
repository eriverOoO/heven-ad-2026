"""Pure-Python experimental post-NMS replay helpers (not production code)."""
from __future__ import annotations
import math

def _corners(b):
    l, w, yaw = b["length"] / 2, b["width"] / 2, b.get("yaw", 0.0)
    c, s = math.cos(yaw), math.sin(yaw)
    return [(b["x"] + c*x-s*y, b["y"] + s*x+c*y) for x,y in ((l,w),(-l,w),(-l,-w),(l,-w))]

def _area(p):
    return abs(sum(x1*y2-x2*y1 for (x1,y1),(x2,y2) in zip(p,p[1:]+p[:1])))/2 if len(p)>2 else 0.

def _clip(subject, clip):
    out = subject
    for a,b in zip(clip, clip[1:]+clip[:1]):
        inp, out = out, []
        if not inp: break
        def inside(p): return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]) > 0
        def cross(p,q):
            dx,dy=p[0]-q[0],p[1]-q[1]; ex,ey=a[0]-b[0],a[1]-b[1]
            d=dx*ey-dy*ex
            if abs(d)<1e-12: return p
            t=((p[0]-a[0])*ey-(p[1]-a[1])*ex)/d
            return (p[0]+t*(q[0]-p[0]),p[1]+t*(q[1]-p[1]))
        prev=inp[-1]
        for cur in inp:
            if inside(cur):
                if not inside(prev): out.append(cross(cur,prev))
                out.append(cur)
            elif inside(prev): out.append(cross(cur,prev))
            prev=cur
    return out

def bev_iou(a, b):
    inter=_area(_clip(_corners(a),_corners(b)))
    union=a["length"]*a["width"]+b["length"]*b["width"]-inter
    return inter/union if union else 0.

def score_filter(items, threshold):
    return [x for x in items if x["score"] >= threshold]

def greedy_class_agnostic_nms(items, threshold):
    """Score-descending BEV-IoU suppression matching OpenPCDet NMS semantics."""
    kept=[]
    for item in sorted(items, key=lambda x: x["score"], reverse=True):
        if all(bev_iou(item, prior) <= threshold for prior in kept): kept.append(item)
    return kept
