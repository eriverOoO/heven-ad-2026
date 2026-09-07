#!/usr/bin/env python3
"""Fresh historical-model inference with independent in-model NMS settings."""
from __future__ import annotations
import argparse,json,os,random,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from nms_override import apply_nms_threshold
from morai_dataset import CLASS_NAMES,make_openpcdet_dataset
from openpcdet_runtime import import_smoke_components,load_batch_to_cuda
from prediction_bridge import convert_openpcdet_prediction
from train_morai_centerpoint import DEFAULT_DATA_CONFIG,DEFAULT_MODEL_CONFIG,load_yaml_config

def parse():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--openpcdet-root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--nms-thresholds',type=float,nargs='+',default=[.7,.5,.3,.2,.1]);p.add_argument('--max-frames',type=int);return p.parse_args()
def tag(t): return f'nms_{t:.2f}'.replace('.','_')
def main():
 a=parse();import torch;from easydict import EasyDict
 random.seed(2026);np.random.seed(2026);torch.manual_seed(2026)
 data=EasyDict(load_yaml_config(DEFAULT_DATA_CONFIG));data.DATA_SPLIT.test='train';model_cfg=EasyDict(load_yaml_config(DEFAULT_MODEL_CONFIG));template,cp=import_smoke_components(a.openpcdet_root);dataset=make_openpcdet_dataset(template)(data,list(CLASS_NAMES),False,a.dataset.resolve(),None)
 model=cp(model_cfg.MODEL,len(CLASS_NAMES),dataset).cuda().eval();model.load_state_dict(torch.load(a.checkpoint,map_location='cuda')['model_state'],strict=True)
 a.output_dir.mkdir(parents=True,exist_ok=True);limit=len(dataset) if a.max_frames is None else min(len(dataset),a.max_frames)
 for requested in a.nms_thresholds:
  actual=apply_nms_threshold(model,requested);out=a.output_dir/f'{tag(actual)}.jsonl';meta=a.output_dir/f'{tag(actual)}.metadata.json';start=time.perf_counter()
  with out.open('w') as stream,torch.no_grad():
   for index in range(limit):
    sample=dataset[index];batch=dataset.collate_batch([sample]);load_batch_to_cuda(batch);preds,_=model(batch);source=dataset.morai_core[index]
    frame=convert_openpcdet_prediction(sample_id=str(batch['frame_id'][0]),source_header_stamp_ns=source['source_header_stamp_ns'],prediction=preds[0],class_names=CLASS_NAMES,frame_id=source['coordinate_frame'],inference_time_ms=0.)
    stream.write(json.dumps(asdict(frame),separators=(',',':'))+'\n')
  nms=model.dense_head.model_cfg.POST_PROCESSING.NMS_CONFIG
  meta.write_text(json.dumps({'checkpoint':str(a.checkpoint.resolve()),'config':str(DEFAULT_MODEL_CONFIG),'score_threshold':float(model.dense_head.model_cfg.POST_PROCESSING.SCORE_THRESH),'nms_type':str(nms.NMS_TYPE),'nms_threshold':actual,'nms_pre_maxsize':int(nms.NMS_PRE_MAXSIZE),'nms_post_maxsize':int(nms.get('NMS_POST_MAXSIZE',model.dense_head.model_cfg.POST_PROCESSING.MAX_OBJ_PER_SAMPLE)),'frames':limit,'seconds':time.perf_counter()-start},indent=2)+'\n')
if __name__=='__main__':main()
