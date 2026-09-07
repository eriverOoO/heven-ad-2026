#!/usr/bin/env python3
"""Controlled secondary-NMS + fixed historical tracker replay.

This consumes *already OpenPCDet-post-NMS* records.  It is consequently an
experimental second suppression pass, not a replacement for raw-model NMS.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from stage_postprocess import greedy_class_agnostic_nms, score_filter

def parse():
    p=argparse.ArgumentParser(); p.add_argument('--predictions',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--thresholds',type=float,nargs='+',default=[.7,.5,.3,.1]); p.add_argument('--score',type=float,default=.1); p.add_argument('--no-secondary-nms',action='store_true'); return p.parse_args()

def det(d):
    b=d['box_lidar']; return dict(x=b[0],y=b[1],z=b[2],length=b[3],width=b[4],height=b[5],yaw=b[6],score=d['score'],class_label=d['class_name'])

def main():
    a=parse(); a.output_dir.mkdir(parents=True,exist_ok=True)
    records=[json.loads(x) for x in a.predictions.open()]
    repo=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(repo/'ad_lidar_perception')); sys.path.insert(0,'/home/didgang1203/py-ab3dmot-deps')
    # The audit worktree intentionally does not initialise third-party
    # submodules.  Reuse the pinned, read-only main checkout submodule rather
    # than mutating either worktree.
    ab3dmot_root=Path('/home/didgang1203/projects/heven-ad-2026/references/ab3dmot')
    from ad_lidar_perception.ab3dmot_config import AB3DMOTConfig
    from ad_lidar_perception.ab3dmot_core import AB3DMOTTracker, Detection
    sys.path.insert(0,'/home/didgang1203/heven_presentation_assets/gt_mot_eval')
    from trackeval_adapter import evaluate
    gt=json.load(open('/home/didgang1203/heven_presentation_assets/motion_gt_expansion/canonical/gt/gt_sequences_full.json'))['frames']
    valid={13,48,46,15,36,1,35,44,18,34,38,40,23,31,21,27,16,20,30}
    results=[]
    for t in a.thresholds:
        frames=[]; tracker=AB3DMOTTracker(AB3DMOTConfig(association_metric='euclidean',matcher='hungarian',euclidean_gate_m=3.0,state_estimator='linear_kf',yaw_measurement_mode='detector',min_hits=1,max_age=2),ab3dmot_root=ab3dmot_root)
        gt_seq=[]; trk_seq=[]; prev=None
        for idx,(record,g) in enumerate(zip(records,gt)):
            scored=score_filter([det(x) for x in record['detections']],a.score)
            items=scored if a.no_secondary_nms else greedy_class_agnostic_nms(scored,t)
            timestamp=g['header_stamp_ns']/1e9; timestamp=timestamp if prev is None or timestamp>prev else prev+.001; prev=timestamp
            states=tracker.step([Detection(x=x['x'],y=x['y'],z=x['z'],yaw=x['yaw'],length=x['length'],width=x['width'],height=x['height'],label={'vehicle':1,'pedestrian':2,'obstacle':0}.get(x['class_label'],0),label_probability=x['score'],existence_probability=x['score']) for x in items],timestamp)
            frames.append({'frame_idx':idx,'detections':items,'tracks':[{'track_id':s.track_id,'x':s.x,'y':s.y} for s in states]})
            gt_seq.append({'objects':[{'actor_id':o['actor_id'],'x':o['x'],'y':o['y']} for o in g['objects'] if o['actor_id'] in valid]})
            trk_seq.append({'tracks':frames[-1]['tracks']})
        provenance='fresh in-model NMS output; no secondary suppression' if a.no_secondary_nms else 'secondary NMS applied to historical OpenPCDet post-NMS output'
        mot=evaluate(gt_seq,trk_seq,threshold_m=2.0); mot.update({'secondary_nms_iou_threshold':None if a.no_secondary_nms else t,'score_threshold':a.score,'frames':len(frames),'detections':sum(len(f['detections']) for f in frames),'stage_provenance':provenance})
        tag=str(t).replace('.','_'); (a.output_dir/f'secondary_nms_{tag}_metrics.json').write_text(json.dumps(mot,indent=2)+'\n')
        with (a.output_dir/f'secondary_nms_{tag}.jsonl').open('w') as out:
            for f in frames: out.write(json.dumps(f,separators=(',',':'))+'\n')
        results.append(mot)
    (a.output_dir/'secondary_nms_summary.json').write_text(json.dumps(results,indent=2)+'\n'); print(json.dumps(results,indent=2))
if __name__=='__main__': main()
