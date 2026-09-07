#!/usr/bin/env python3
"""Opt-in, streaming OpenPCDet decoded→score→NMS stage dumper.

It monkey-patches only the in-process NMS call to observe decoded candidates;
no OpenPCDet tracked source or production ROS behavior is changed.
"""
from __future__ import annotations
import argparse, csv, inspect, json, sys
from pathlib import Path
import numpy as np

STAGE_NAMES=('raw_decoded','score_filtered','post_nms','ros_final')

def parse():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--openpcdet-root',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--object-csv',type=Path,required=True); p.add_argument('--per-group',type=int,default=10); return p.parse_args()

def select_frames(path, n):
    rows=list(csv.DictReader(path.open())); groups={'severe':lambda v:v>=12,'typical':lambda v:v==8,'clean':lambda v:v==1}; selected=[]
    for name,pred in groups.items():
        choices=[]
        for r in rows:
            i=int(r['frame_idx'])
            if pred(int(r['predictions_per_gt'])) and i not in choices: choices.append(i)
        selected.extend((name,i) for i in choices[:n])
    return selected

def serial(boxes,scores,labels,names):
    out=[]
    for i,(b,s,l) in enumerate(zip(boxes,scores,labels)):
        label=int(l); out.append({'candidate_id':i,'class_name':names[label] if 0<=label<len(names) else f'unknown_{label}','score':float(s),'box_lidar':[float(x) for x in b[:7]]})
    return out

def main():
    a=parse(); sys.path.insert(0,str(a.openpcdet_root)); sys.path.insert(0,str(Path(__file__).parent))
    import torch
    from easydict import EasyDict
    from morai_dataset import CLASS_NAMES, make_openpcdet_dataset
    from openpcdet_runtime import import_smoke_components, load_batch_to_cuda
    from train_morai_centerpoint import DEFAULT_DATA_CONFIG, DEFAULT_MODEL_CONFIG, load_yaml_config
    data_cfg=EasyDict(load_yaml_config(DEFAULT_DATA_CONFIG)); data_cfg.DATA_SPLIT.test='train'; model_cfg=EasyDict(load_yaml_config(DEFAULT_MODEL_CONFIG))
    template, cp=import_smoke_components(a.openpcdet_root)
    # import_smoke_components installs namespace shells first; importing this
    # module earlier would execute OpenPCDet's broad package initializer.
    from pcdet.models.model_utils import model_nms_utils
    dataset=make_openpcdet_dataset(template)(data_cfg,list(CLASS_NAMES),False,a.dataset.resolve(),None)
    model=cp(model_cfg.MODEL,len(CLASS_NAMES),dataset).cuda().eval(); model.load_state_dict(torch.load(a.checkpoint,map_location='cuda')['model_state'],strict=True)
    # Decode every top-K geometrically valid candidate; score filtering is
    # explicitly replayed below. NMS remains the historical 0.70 implementation.
    model.model_cfg.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH=0.0
    original=model_nms_utils.class_agnostic_nms; captured=[]
    def observe(*,box_scores,box_preds,nms_config,score_thresh=None):
        # CenterHead calls this immediately after assigning mapped labels to
        # local final_dict. Capturing that local avoids modifying dependency
        # code while retaining class provenance.
        labels=inspect.currentframe().f_back.f_locals['final_dict']['pred_labels']
        captured.append((box_preds.detach().cpu().numpy(),box_scores.detach().cpu().numpy(),labels.detach().cpu().numpy()))
        return original(box_scores=box_scores,box_preds=box_preds,nms_config=nms_config,score_thresh=score_thresh)
    model_nms_utils.class_agnostic_nms=observe
    id_to_index={str(x):i for i,x in enumerate(dataset.morai_core.sample_ids)}; selected=select_frames(a.object_csv,a.per_group); a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w') as out, torch.no_grad():
        for group,idx in selected:
            sample=dataset[id_to_index[str(dataset.morai_core.sample_ids[idx])]]; batch=dataset.collate_batch([sample]); load_batch_to_cuda(batch); captured.clear(); predictions,_=model(batch)
            if len(captured)!=1: raise RuntimeError(f'expected one CenterHead NMS call, got {len(captured)}')
            raw_boxes,raw_scores,raw_labels=captured[0]
            score_mask=raw_scores>=.1; score_boxes,score_scores=raw_boxes[score_mask],raw_scores[score_mask]
            # Re-run the exact OpenPCDet CUDA NMS over the score stage.
            b=torch.as_tensor(score_boxes,device='cuda'); s=torch.as_tensor(score_scores,device='cuda'); cfg=model.model_cfg.DENSE_HEAD.POST_PROCESSING.NMS_CONFIG
            keep,keep_scores=original(box_scores=s,box_preds=b,nms_config=cfg,score_thresh=None); keep=keep.detach().cpu().numpy(); keep_scores=keep_scores.detach().cpu().numpy()
            stage={'raw_decoded':serial(raw_boxes,raw_scores,raw_labels,CLASS_NAMES),'score_filtered':serial(score_boxes,score_scores,raw_labels[score_mask],CLASS_NAMES),'post_nms':serial(score_boxes[keep],keep_scores,raw_labels[score_mask][keep],CLASS_NAMES)}
            # The ROS wrapper applies the same score threshold and max=500;
            # NMS output here is <=500, therefore this is identical final set.
            stage['ros_final']=stage['post_nms']
            if tuple(stage) != STAGE_NAMES: raise RuntimeError('stage provenance order changed')
            out.write(json.dumps({'frame_idx':idx,'sample_id':str(dataset.morai_core.sample_ids[idx]),'group':group,'stages':stage},separators=(',',':'))+'\n'); out.flush()
    model_nms_utils.class_agnostic_nms=original
if __name__=='__main__': main()
