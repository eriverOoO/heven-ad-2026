# CenterPoint AV2 VLP-16-like inference audit v1

## Executive summary

This is a first-frame sensor-sparsity audit, not a MORAI performance claim.
AV2 feather decoding and ROS publishing were separated by canonical NPZ, so
the ROS Python environment imports no `pyarrow` or OpenPCDet environment.
For timestamp `315969904460072000`, native and VLP-16-like clouds both passed
the same Autoware CenterPoint 0.51 TensorRT runtime in single-sweep mode.

The adapted cloud retained 14.43% of points. Both branches produced 11 final
detections, but supported-vehicle recall under a documented 3 m one-to-one
center gate fell from 90.9% to 72.7%. This is preliminary evidence of strong
structured-sparsity sensitivity for this frame; it is not evidence about
MORAI or generalization.

## Coordinate and timestamp contract

LiDAR xyz and cuboids are both AV2 egovehicle coordinates: rear-axle origin,
+x forward, +y left, +z up. No LiDAR extrinsic or city pose is applied for
single-frame evaluation. All 10 downloaded filenames have exact annotation
timestamps. On the first sweep, direct point-in-cuboid counts for five checked
regular vehicles exactly equal `num_interior_pts`, supporting the contract.
The ROS `frame_id` is `av2_egovehicle`; an identity static TF only satisfies
the node interface and does not transform coordinates. Densification uses an
experimental `num_past_frames=0` config.

## Python bridge and input density

`prepare_av2_xyzirc.py` runs in the pyarrow-capable offline environment and
writes canonical NPZ. `publish_av2_xyzirc.py` runs under sourced ROS
`/usr/bin/python3` and imports only NumPy and ROS messages. `return_type=0` is
an explicit placeholder. Native AV2 `laser_number` 0--63 is preserved as
uint16 `channel`; pinned `InputPointType` accepts that field, while
`generateSweepPoints_kernel` constructs model features from x/y/z,
intensity, and time lag without reading or modulo-reducing `channel`.

| Metric | Native | VLP-16-like |
| --- | ---: | ---: |
| Input points | 95,231 | 13,738 |
| Model-ROI points | 87,792 | 12,469 |
| Retention | 100% | 14.43% |
| Final detections | 11 | 11 |
| Mean final score | 0.722 | 0.629 |

The adapter defines 16 target elevations with IDs 0--15, but channel 0
(-15 degrees) has no surviving returns in any of the 10 sweeps. Aggregate
observed IDs are 1--15; per-frame occupancy ranges from 12 to 15. Aggregate
counts for IDs 1--15 are respectively 13,302, 23, 12,961, 92, 13,708, 96,
13,616, 1,044, 12,816, 4,125, 11,640, 12,663, 10,549, 13,176, and 8,464.
This is zero occupancy of one defined logical beam, not an indexing or `%16`
operation. It limits any claim that the observed output is a fully occupied
16-beam scan and should be calibrated against real MORAI data later.

## First paired detection result

Only `REGULAR_VEHICLE` cuboids inside the model ROI are included (11 GT).
Matching is one-to-one Hungarian assignment with a 3 m BEV center gate.

| Metric | Native | VLP-16-like |
| --- | ---: | ---: |
| Vehicle detections evaluated | 11 | 10 |
| Other-class final detections | 0 | 1 |
| Matched GT | 10 | 8 |
| Recall | 90.91% | 72.73% |
| Unmatched vehicle detections | 1 | 2 |
| Mean matched center error | 0.287 m | 0.245 m |
| Mean final score | 0.722 | 0.629 |

The lower adapted center error is conditional on only eight surviving matches
and must not be interpreted as improved localization. A forced 11-to-11
native/adapted assignment has 13.54 m mean center shift, showing that equal
final counts hide major candidate replacement.

The table is reproducible with `evaluate_av2_xyzirc_pair.py`; its JSON output
is stored outside Git under `av2_vlp16_inference_v1/metrics/`. ROS echo files
contain a trailing YAML document separator, so the evaluator explicitly uses
the first complete response and never combines repeated callbacks.

The paired run used the provisioned full CenterPoint v3 engines, the isolated
Autoware 0.51 install, `ROS_DOMAIN_ID=77`, and the exact same timestamp,
thresholds, engine, frame label, and identity interface TF for both branches.
Only structured point selection differs.

## Stage visibility and limitations

The verified runtime order is head output, CUDA decode plus score gate,
positive compaction/sort, circle NMS, object conversion, IoU NMS, class
remap, final output. Only final output is currently observable. Post-score,
circle-NMS and pre-IoU counts are therefore `not measured`; they are not
inferred from final messages. A default-off isolated source patch remains
required.

Only one frame has been inferred so far. No MORAI calibration, multi-frame
densification, tracker metric, or target-domain conclusion is available.
Retraining decision remains **NOT YET**.
