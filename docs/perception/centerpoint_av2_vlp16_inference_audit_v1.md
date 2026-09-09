# CenterPoint AV2 structured-sparsity inference audit v1

## 1. Executive summary

The current Autoware CenterPoint 0.51 runtime was instrumented at five real
post-processing stages and run on paired native and sparsified versions of 10
AV2 Sensor sweeps. The first observable internal loss is already present at
the post-score compacted stage: native produces 598 candidates and the tested
structured sparsifier produces 247. Circle NMS and IoU NMS suppress a smaller,
not larger, fraction of sparse-input candidates. NMS is therefore not the
source of the native-to-sparse loss in this sample.

Across 89 supported vehicle actor-frames, native micro recall is 78.65% and
sparse-input recall is 44.94%. The drop is strongly range-dependent and starts
between 20 and 40 m; recall is unchanged at 0--20 m but falls from 80.65% to
29.03% at 40--60 m. These are descriptive results from 10 sweeps of one AV2
log, not a MORAI performance or generalization claim.

The current adapter is classified **C: requires correction**. It selects beams
by elevation about the AV2 rear-axle ego origin, whereas physical beam angle is
defined about each LiDAR origin. This creates zero or near-zero occupancy in
several nominal beams before azimuth reduction. Consequently, this run is a
useful structured sparsity stress-test but is not a physically representative
VLP-16 emulation. The retraining decision remains **NOT YET**.

## 2. Scope and coordinate contract

The source is AV2 Sensor validation log
`02678d04-cc9f-3148-9f95-1ba66347dff9`. Only the 10 already-downloaded sweeps
were used. AV2 xyz and annotation cuboids are both in the egovehicle frame:
rear-axle origin, +x forward, +y left, +z up. Annotation timestamps match all
10 LiDAR filename timestamps exactly. Direct oriented point-in-cuboid counts
on five checked regular vehicles equal AV2 `num_interior_pts`.

No sensor extrinsic is reapplied to inference points and no city pose is used
for this single-sweep experiment. `frame_id=av2_egovehicle` and an identity
static TF satisfy the node interface without changing xyz. Densification is
disabled with the experimental `num_past_frames=0` configuration.

The [official AV2 Sensor contract](https://argoverse.github.io/user-guide/datasets/sensor.html)
states that returns from two stacked 32-beam LiDARs are aggregated, ego-motion
compensated, and supplied in the egovehicle frame.
The local AV2 API `Sweep.from_feather` independently confirms this contract.
Calibration extrinsics are used below only to diagnose physical ray angles,
never to transform the detector or GT inputs.

## 3. Runtime and stage instrumentation

The pinned source order is:

```text
TensorRT head
  -> CUDA decode + class/distance score gate
  -> positive candidate compaction and score sort
  -> circle NMS
  -> Box3D to DetectedObject
  -> BEV IoU NMS
  -> class remap
  -> final DetectedObjects
```

Source symbols are `CenterPointTRT::postProcess`,
`PostProcessCUDA::generateDetectedBoxes3D_launch`, `circleNMS`,
`LidarCenterPointNode::pointCloudCallback`, `NonMaximumSuppression::apply`, and
`DetectionClassRemapper::mapClasses`. The repository patch is applied only to
an ignored isolated source copy under `.autoware_runtime/src`; the pinned
Autoware checkout and shared workspace are unchanged.

`enable_stage_dump` defaults to `false`. When enabled it publishes:

| Stage | Topic | Meaning |
| --- | --- | --- |
| S1 | `~/debug/stages/post_score` | compacted, score-sorted candidates before circle NMS |
| S2 | `~/debug/stages/post_circle_nms` | surviving `Box3D` candidates |
| S3 | `~/debug/stages/pre_iou` | S2 converted to `DetectedObject` |
| S4 | `~/debug/stages/post_iou` | BEV IoU NMS survivors before remap |
| S5 | `~/debug/stages/final` | final post-remap objects |

Messages retain timestamp, class, score, center, dimensions, yaw, and velocity.
No stable source index exists after CUDA compaction. Raw pre-score tensor output
is not instrumented, so S1 cannot distinguish head/decode loss from score-gate
loss.

Instrumentation does not alter inference logic. In the same callback, normal
final output and S5 were exactly equal for all 11 native and 12 sparse-input
first-frame objects.
Independent dump-on and dump-off process runs had equal counts/classes and
were numerically equivalent: maximum BEV center distance 0.005173 m,
z difference 0.002441 m, yaw difference 0.002482 rad, and score difference
0.002089. The small independent-run variation is TensorRT numeric variation,
not a stage-dump output change.

## 4. First-frame stage flow

Timestamp: `315969904460072000`.

| Stage | Native | Structured sparse | Delta |
| --- | ---: | ---: | ---: |
| Input points | 95,231 | 13,738 | -81,493 |
| ROI points | 87,792 | 12,469 | -75,323 |
| S1 post-score | 48 | 39 | -9 |
| S2 circle NMS | 11 | 12 | +1 |
| S3 pre-IoU | 11 | 12 | +1 |
| S4 post-IoU | 11 | 12 | +1 |
| S5 final | 11 | 12 | +1 |

The earliest observable post-processing divergence is S1. First-frame S1
score distributions are:

| Input | n | mean | median | p10 | p25 | p75 | p90 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 48 | .555 | .535 | .401 | .441 | .655 | .779 | .861 |
| Sparse | 39 | .543 | .518 | .369 | .433 | .627 | .760 | .818 |

Circle-NMS reduction is 77.08% native versus 69.23% sparse; IoU-NMS reduction
is zero for both. Despite similar final counts, class-correct vehicle recall is
90.91% native and 72.73% sparse. Final count alone therefore hides candidate
replacement and missed GT.

The earlier independent sparse run had 11 final objects. The exact-timestamp
stage rerun has one additional label-7 object at score .362; its other 11
objects match within 5 cm. Vehicle matches and recall are unchanged. This is a
borderline inter-process TensorRT/score-gate variation and is retained rather
than silently forcing the earlier count.

## 5. Ten-sweep stage aggregate

All 20 independent runs (10 timestamps x two input modes) produced non-empty
S1--S5 captures, and all 100 stage headers retain their exact source AV2
timestamp. Counts below are per-frame means; totals are in parentheses.

| Stage | Native | Structured sparse | Relative change |
| --- | ---: | ---: | ---: |
| Input points | 91,103.9 (911,039) | 12,827.5 (128,275) | -85.92% |
| ROI points | 84,355.3 (843,553) | 11,950.9 (119,509) | -85.83% |
| S1 post-score | 59.8 (598) | 24.7 (247) | -58.70% |
| S2 circle NMS | 16.3 (163) | 7.6 (76) | -53.37% |
| S3 pre-IoU | 16.3 (163) | 7.6 (76) | -53.37% |
| S4 post-IoU | 15.6 (156) | 7.4 (74) | -52.56% |
| S5 final | 15.6 (156) | 7.4 (74) | -52.56% |

Aggregate S1 score statistics:

| Input | n | mean | median | p10 | p25 | p75 | p90 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 598 | .509 | .481 | .371 | .408 | .580 | .703 | .861 |
| Sparse | 247 | .496 | .459 | .365 | .390 | .572 | .681 | .856 |

The dominant S1 effect is candidate-count loss rather than a large global
shift of the surviving score distribution. Circle-NMS reduction is 72.74%
native and 69.23% sparse. IoU-NMS reduction is 4.29% native and 2.63% sparse.
Thus neither NMS stage preferentially removes sparse-input candidates.

## 6. Ten-sweep detection aggregate

Evaluation includes only `REGULAR_VEHICLE` GT centers inside the CenterPoint
ROI. Matching is maximum-cardinality, minimum-distance Hungarian assignment
under a 3 m BEV gate. Vehicle recall/FP are class-correct; other predicted
classes are not silently counted as vehicle matches.

| Metric | Native | Structured sparse |
| --- | ---: | ---: |
| Frames | 10 | 10 |
| Supported vehicle GT | 89 | 89 |
| Vehicle matches | 70 | 40 |
| Micro recall | 78.65% | 44.94% |
| Mean per-frame recall | 77.49% | 42.85% |
| Final detections/frame | 15.6 | 7.4 |
| Vehicle detections/frame | 12.1 | 5.6 |
| Vehicle FP/frame | 5.1 | 1.6 |
| Matched center error, mean | 0.272 m | 0.416 m |
| Matched center error, median | 0.235 m | 0.258 m |
| Matched score, mean | .660 | .631 |

The lower first-frame sparse center error is survivor-conditioned and is not a
localization improvement. On all 10 sweeps, sparse matched error is worse and
has a 2.236 m maximum versus 0.959 m native.

## 7. GT-conditioned paired stability

Native and sparse detections are never paired by array index. Each branch is
first matched independently to GT; metrics below use the 38 GT objects found
in both branches. Unmatched detections remain unmatched.

| Paired metric | Mean | Median | p90 | Max |
| --- | ---: | ---: | ---: | ---: |
| Native-to-sparse center shift | 0.280 m | 0.064 m | 0.685 m | 1.947 m |
| Native center error | 0.223 m | 0.236 m | 0.328 m | 0.404 m |
| Sparse center error | 0.434 m | 0.280 m | 0.832 m | 2.236 m |
| Size-vector shift | 0.220 m | 0.181 m | 0.431 m | 0.777 m |
| Score shift (sparse - native) | -.091 | -.062 | +.019 | +.223 |
| Raw directional yaw shift | 0.920 rad | 0.045 rad | 3.110 rad | 3.131 rad |
| Pi-symmetric box-axis shift | 0.115 rad | 0.036 rad | 0.080 rad | 1.519 rad |

There are no class changes among these 38 jointly detected objects. Twelve
have a directional yaw change above pi/2. The raw yaw mean is therefore
dominated by near-pi front/back flips; the axis-symmetric statistic
is included to prevent those flips from being mistaken for a general 0.92 rad
box-axis rotation. Front/back direction remains semantically relevant and is
therefore not discarded from the raw result.

## 8. Distance sensitivity

These bins are descriptive only because all observations come from one log.
Point counts are direct full-3D oriented-cuboid counts in the same AV2 ego
frame.

| Range | GT | Native recall | Sparse recall | Delta | Native pts/object median | Sparse pts/object median | Native score | Sparse score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0--20 m | 16 | 93.75% | 93.75% | 0.00 pp | 564 | 115 | .777 | .754 |
| 20--40 m | 19 | 89.47% | 78.95% | -10.53 pp | 107 | 17 | .709 | .608 |
| 40--60 m | 31 | 80.65% | 29.03% | -51.61 pp | 30 | 0 | .630 | .466 |
| 60--80 m | 23 | 56.52% | 4.35% | -52.17 pp | 9 | 0 | .521 | .617* |

`*` The 60--80 m sparse score has only one matched object and is not a stable
estimate. The sharp 40 m transition tracks the median sparse points per object
falling to zero.

## 9. Beam-occupancy diagnosis

The existing adapter uses target angles `[-15, +1, -13, +3, ..., -1, +15]`
degrees and a +/-0.7 degree ego-origin elevation gate, followed by nearest
return per beam/azimuth bin.

| Ch | Nominal | Accepted ego elevation | Before azimuth | After azimuth | Occupied frames |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | -15 | none | 0 | 0 | 0/10 |
| 1 | +1 | +0.30..+1.70 | 60,899 | 13,302 | 10/10 |
| 2 | -13 | -12.71..-12.31 | 23 | 23 | 4/10 |
| 3 | +3 | +2.30..+3.70 | 68,685 | 12,961 | 10/10 |
| 4 | -11 | -11.70..-10.30 | 109 | 92 | 4/10 |
| 5 | +5 | +4.30..+5.70 | 68,967 | 13,708 | 10/10 |
| 6 | -9 | -9.02..-8.30 | 98 | 96 | 6/10 |
| 7 | +7 | +6.30..+7.70 | 63,115 | 13,616 | 10/10 |
| 8 | -7 | -7.69..-6.30 | 1,105 | 1,044 | 10/10 |
| 9 | +9 | +8.30..+9.70 | 54,185 | 12,816 | 10/10 |
| 10 | -5 | -5.70..-4.30 | 5,020 | 4,125 | 10/10 |
| 11 | +11 | +10.30..+11.70 | 47,415 | 11,640 | 10/10 |
| 12 | -3 | -3.70..-2.30 | 32,166 | 12,663 | 10/10 |
| 13 | +13 | +12.30..+13.70 | 37,208 | 10,549 | 10/10 |
| 14 | -1 | -1.70..-0.30 | 58,847 | 13,176 | 10/10 |
| 15 | +15 | +14.30..+15.70 | 28,139 | 8,464 | 10/10 |

Channels 0, 2, 4, and 6 are absent or nearly empty before azimuth reduction,
so azimuth binning is not the cause. In ego coordinates, individual
`laser_number` elevation distributions are broad because the rear-axle origin
is translated from both physical LiDAR origins.

As a diagnostic only, points 0--31 were inverse-transformed with `up_lidar`
calibration and points 32--63 with `down_lidar`. Corresponding ring pairs
`n`/`n+32` then have near-identical narrow local elevation bands: median pair
difference 0.0027 degrees, mean 0.0044 degrees, maximum 0.0244 degrees. This is
strong empirical support for the source split and proves that ego-origin
elevation is not the physical beam angle. It does not alter the ego-frame
coordinates used for inference or GT.

Applying the same +/-0.7 degree target gate diagnostically in sensor-local
coordinates gives pre-azimuth target counts:

```text
[17585, 135404, 2185, 50730, 23096, 28784, 26193, 27819,
 30271, 2715, 58167, 17070, 133666, 2532, 137995, 19197]
```

All targets gain support, but channels 2, 9, and 13 remain much smaller because
some VLP-16 target angles lie between the native VLP-32 ring angles under the
strict tolerance. A correct adapter should select/source-match physical rings
in each LiDAR frame, then retain the original ego-frame xyz; it should not
reapply an extrinsic to the detector input.

**Adapter classification: C -- adapter requires correction.** The current
output remains useful as a deterministic structured sparsity stress-test, but
must not be called a physically representative VLP-16 or MORAI emulation.

## 10. Root interpretation

| Hypothesis | Evidence | Assessment |
| --- | --- | --- |
| Confidence degradation | S1 scores shift slightly overall; jointly detected GT loses .091 mean score | Moderate |
| Candidate-generation/score-gate degradation | S1 count falls 598 to 247 before either NMS | Strong, but raw-vs-gate unresolved |
| NMS interaction causes loss | Sparse circle/IoU reduction ratios are lower than native | Unsupported as primary cause |
| Point-count sensitivity | 85.9% input loss coincides with 33.7 pp micro-recall loss | Strong for this tested input |
| Beam-structure sensitivity | No count-matched random control; current beam proxy is geometrically flawed | Unresolved |
| General pretrained-model robustness | Large range-dependent loss in one small log | Moderate diagnostic evidence only |

The earliest measurable divergence is S1. Raw pre-score preservation would be
needed to decide whether the network produces fewer viable peaks or the
class/distance score gate removes them. The current data are sufficient to
exclude circle and IoU NMS as the first cause.

## 11. Reproduction and external artifacts

The isolated overlay is prepared with
`prepare_autoware_stage_overlay.sh`, built with
`build_autoware_centerpoint_isolated.sh`, and one NPZ is captured with
`run_autoware_stage_frame.sh --stage-dump on`. The aggregate is generated by
`summarize_av2_stage_audit.py`.

Large outputs remain outside Git:

```text
/home/didgang1203/datasets/centerpoint/av2_vlp16_inference_v1/
  stage_dumps/ten_sweep/<timestamp>/{native,vlp16_like}/
  metrics/ten_sweep_stage_summary.json
  metrics/ten_sweep_stage_scores.csv
  metrics/stage_dump_on_off_equivalence.json
```

## 12. Limitations and next step

- Ten sweeps from one AV2 validation log are a small correlated diagnostic
  sample, not an AV2 benchmark.
- The current sparse branch is not a valid physical VLP-16 proxy.
- Raw pre-score candidates are not observable.
- There is no count-matched random control.
- There is no MORAI VLP-16 calibration or sequence-disjoint actor-GT bag.
- Single-sweep mode differs from HEVEN's production densification contract.

Retraining decision: **NOT YET**. This experiment proves sensitivity to the
tested sparsification, not a MORAI domain gap or a need for fine-tuning.

The single highest-value next task is to correct the AV2 adapter using
source-LiDAR-local ray geometry and deterministic source-ring selection, then
rerun these same 10 paired sweeps. That removes the current confound before
adding a random count-matched control or using the result to design training.
