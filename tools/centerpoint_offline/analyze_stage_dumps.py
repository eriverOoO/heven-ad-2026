#!/usr/bin/env python3
"""Summarize streamed stage dumps against MORAI GT without retaining tensors."""
from __future__ import annotations
import argparse,json,math
from collections import defaultdict
from pathlib import Path
from stability_metrics import candidate_counts, describe, one_to_one
from stage_postprocess import bev_iou

def box(x):
    b=x['box_lidar']; return dict(x=b[0],y=b[1],z=b[2],length=b[3],width=b[4],height=b[5],yaw=b[6],score=x['score'],class_label=x['class_name'])
def labels(root,sid):
    p=json.loads((root/'labels'/f'{sid}.json').read_text())
    return [dict(x,class_name='vehicle') for x in p['ground_truth']['boxes'] if x.get('class_name')=='vehicle' and x.get('actor_id') not in {2,3}]
def main():
    p=argparse.ArgumentParser();p.add_argument('--dump',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    aggregates=defaultdict(lambda:{'frames':0,'candidates':0,'gt':0,'matched':0,'counts':[],'errors':[],'unmatched':0}); geometry=[]; per_group=defaultdict(lambda:defaultdict(list))
    for line in a.dump.open():
        record=json.loads(line);gt=labels(a.labels,record['sample_id'])
        for stage,items in record['stages'].items():
            ps=[box(x) for x in items]; d=aggregates[stage];cs=candidate_counts(gt,ps);ms=one_to_one(gt,ps);d['frames']+=1;d['candidates']+=len(ps);d['gt']+=len(gt);d['matched']+=len(ms);d['counts']+=cs;d['errors'] += [dist for _,_,dist in ms];d['unmatched']+=len(ps)-len(ms);per_group[record['group']][stage].append(len(ps))
            if stage=='post_nms':
                for g in gt:
                    near=[q for q in ps if q['class_label']=='vehicle' and math.hypot(g['x']-q['x'],g['y']-q['y'])<=3]
                    for i in range(len(near)):
                        for j in range(i+1,len(near)):
                            geometry.append({'center_distance_m':math.hypot(near[i]['x']-near[j]['x'],near[i]['y']-near[j]['y']),'bev_iou':bev_iou(near[i],near[j]),'yaw_difference_rad':abs(math.atan2(math.sin(near[i]['yaw']-near[j]['yaw']),math.cos(near[i]['yaw']-near[j]['yaw']))),'score_difference':abs(near[i]['score']-near[j]['score']),'class_conflict':near[i]['class_label']!=near[j]['class_label']})
    out={'stages':{},'groups':{g:{s:describe(v) for s,v in d.items()} for g,d in per_group.items()},'surviving_duplicate_pair_geometry':{'pairs':len(geometry),**{k:describe([x[k] for x in geometry]) for k in ('center_distance_m','bev_iou','yaw_difference_rad','score_difference')},'class_conflict_fraction':sum(x['class_conflict'] for x in geometry)/len(geometry) if geometry else None}}
    for stage,d in aggregates.items():
        out['stages'][stage]={'frames':d['frames'],'candidates_per_frame':d['candidates']/d['frames'],'predictions_per_gt':describe(d['counts']),'duplicate_rate_ge2':sum(x>=2 for x in d['counts'])/d['gt'] if d['gt'] else None,'recall':d['matched']/d['gt'] if d['gt'] else None,'position_error_m':describe(d['errors']),'unmatched_per_frame':d['unmatched']/d['frames']}
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
