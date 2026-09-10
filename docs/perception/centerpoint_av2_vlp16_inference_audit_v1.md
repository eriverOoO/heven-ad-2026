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

The follow-up `source_ring_vlp16_v2` experiment in Sections 13--18 removes
that geometric defect by selecting 16 unique physical rings from `up_lidar`
after motion-aware source-local ring recovery. It restores uniform beam
occupancy and improves survivor geometry, but it does not restore detection
recall: corrected micro recall is 42.70%, versus 44.94% for the old stress
adapter and 78.65% native. The corrected loss still begins at S1, before NMS.
The corrected adapter is therefore classified **B: a useful one-lidar 16-ring
sparsity proxy, but not close enough to exact VLP-16 geometry**.

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

That next task is completed in Sections 13--18. The historical result remains
here as the reproducible control rather than being overwritten.

## 13. Corrected source-ring geometry

The previous `vlp16_like` output is preserved as the
`ego_elevation_stress_v1` historical control. Its behavior and files were not
rewritten. The corrected mode is `source_ring_vlp16_v2`.

AV2's aggregate `laser_number` split was validated as follows:

```text
0..31  -> up_lidar,   local ring = laser_number
32..63 -> down_lidar, local ring = laser_number - 32
```

The official aggregate schema alone does not name this numeric split. The
assignment is supported empirically by applying the two calibration
extrinsics and recovering 32 fixed local elevation bands per source. Matching
up/down local-ring median elevations differ by 0.0070 degrees on average and
0.0126 degrees at most after acquisition-time reconstruction.

Static inverse extrinsics at the sweep reference time are not sufficiently
accurate for recovering physical ray elevation from the motion-compensated
cloud. The corrected diagnostic uses each point's acquisition time
`sweep_timestamp + offset_ns`:

```text
p_city = city_SE3_egovehicle(reference) * p_ego_compensated
p_ego(acquisition) = inverse(city_SE3_egovehicle(acquisition)) * p_city
p_sensor = inverse(egovehicle_SE3_sensor) * p_ego(acquisition)
```

Pose translation is linearly interpolated and rotation uses quaternion SLERP.
Across the 64 rings, mean per-ring elevation MAD improves from 0.0556 degrees
with the reference-time approximation to 0.00992 degrees. The worst p05--p95
ring span improves from 6.009 degrees to 0.110 degrees. Motion undo is
therefore material for estimating ring geometry.

This reconstruction is used only to estimate fixed ring angles and choose
ring IDs. Selected output points retain their original AV2 compensated
egovehicle XYZ exactly; calibration and city poses are not applied to detector
input coordinates.

## 14. Unique VLP-16 target mapping

The primary proxy uses only `up_lidar`. It applies a deterministic,
order-preserving minimum-total-angular-error assignment from the sorted VLP-16
targets to 16 unique recovered source rings. Output channel is the sorted
elevation index 0--15, not the interleaved VLP-16 firing ID.

| Channel | Target deg | Up local ring | Measured deg | Abs. error deg |
| ---: | ---: | ---: | ---: | ---: |
| 0 | -15 | 17 | -15.648 | .648 |
| 1 | -13 | 29 | -11.316 | 1.684 |
| 2 | -11 | 21 | -8.849 | 2.151 |
| 3 | -9 | 27 | -7.259 | 1.741 |
| 4 | -7 | 16 | -6.152 | .848 |
| 5 | -5 | 22 | -4.670 | .330 |
| 6 | -3 | 18 | -3.003 | .003 |
| 7 | -1 | 13 | -1.003 | .003 |
| 8 | +1 | 10 | +.998 | .002 |
| 9 | +3 | 2 | +1.664 | 1.336 |
| 10 | +5 | 11 | +2.331 | 2.669 |
| 11 | +7 | 6 | +3.330 | 3.670 |
| 12 | +9 | 14 | +4.664 | 4.336 |
| 13 | +11 | 0 | +6.998 | 4.002 |
| 14 | +13 | 15 | +10.331 | 2.669 |
| 15 | +15 | 4 | +14.997 | .003 |

Mean absolute angular mismatch is 1.631 degrees and maximum mismatch is
4.336 degrees. The mismatch is a real limitation of selecting 16 unique
VLP-32C rings: the recovered upper source elevations do not populate every
2-degree VLP-16 target. No cross-sensor mixing, angular tolerance gate, range
crop, or beam-by-azimuth nearest-return reduction is applied.

All 16 channels are occupied in every one of the 10 sweeps. Aggregate points
per channel are:

```text
[17333, 16605, 16260, 15972, 15879, 15540, 15075, 15396,
 15336, 15069, 14829, 15103, 14693, 13337, 10905, 8919]
```

The alternating zero/near-zero beam pathology is eliminated. Proxy point
count is 22,476--24,915 per frame, roughly one quarter of the two-source native
aggregate, while preserving each selected physical source ring's original
azimuth returns.

## 15. Corrected object-density check

The table uses the same 89 supported regular-vehicle GT actor-frames and
direct oriented point-in-box counts.

| Range | GT | Native median | Old ego-angle median | Corrected source-ring median | Corrected zero rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0--20 m | 16 | 564 | 115 | 264 | 0.0% |
| 20--40 m | 19 | 107 | 17 | 17 | 0.0% |
| 40--60 m | 31 | 30 | 0 | 5 | 25.8% |
| 60--80 m | 23 | 9 | 0 | 0 | 60.9% |

The invalid ego-angle adapter caused avoidable collapse at 40--60 m; the
source-ring proxy restores a nonzero median. Far-range sparsity remains severe
at 60--80 m, where a one-source 16-ring subset has only 1.13 points/object on
average and zero points for 60.9% of GT boxes. That residual is a property of
this ring-subset proxy and scene sample, not evidence about MORAI.

## 16. Corrected CenterPoint stage replay

The same 10 timestamps were replayed with the same model, TensorRT engines,
single-sweep configuration, score gates, circle NMS, IoU NMS, ROI, and identity
interface TF. Only point selection differs. Native stage dumps from the
original controlled run are reused; corrected runs were captured into a new
mode directory.

| Stage | Native mean (total) | Old stress mean (total) | Corrected mean (total) |
| --- | ---: | ---: | ---: |
| Input | 91,103.9 (911,039) | 12,827.5 (128,275) | 23,625.1 (236,251) |
| ROI | 84,355.3 (843,553) | 11,950.9 (119,509) | 21,550.2 (215,502) |
| S1 post-score | 59.8 (598) | 24.7 (247) | 24.4 (244) |
| S2 circle NMS | 16.3 (163) | 7.6 (76) | 7.3 (73) |
| S3 pre-IoU | 16.3 (163) | 7.6 (76) | 7.3 (73) |
| S4 post-IoU | 15.6 (156) | 7.4 (74) | 7.3 (73) |
| S5 final | 15.6 (156) | 7.4 (74) | 7.3 (73) |

The corrected proxy retains 25.93% of native input points and 25.55% of native
ROI points, versus 14.08% and 14.17% for the old adapter. Despite nearly
doubling retained points, corrected S1 count is effectively unchanged from
the old stress input (244 versus 247) and remains 59.2% below native.

S1 mean/median scores are .509/.481 native, .496/.459 old, and .499/.465
corrected. The main S1 difference remains candidate-count loss, not a large
global score shift among survivors. Circle-NMS reduction is 72.74% native,
69.23% old, and 70.08% corrected. IoU-NMS reduction is 4.29%, 2.63%, and 0%,
respectively. NMS does not preferentially remove corrected sparse candidates.

The earliest measurable divergence remains **S1 post-score**. Because raw
pre-score tensors are not retained, network peak generation versus the
class/distance score gate remains unresolved. Circle NMS and IoU NMS occur too
late to explain the initial loss.

## 17. Corrected detection and paired stability

| Metric | Native | Old stress | Corrected source-ring |
| --- | ---: | ---: | ---: |
| Supported vehicle GT | 89 | 89 | 89 |
| Vehicle matches | 70 | 40 | 38 |
| Micro recall | 78.65% | 44.94% | 42.70% |
| Final detections/frame | 15.6 | 7.4 | 7.3 |
| Vehicle FP/frame | 5.1 | 1.6 | 1.8 |
| Matched center error mean | .272 m | .416 m | .278 m |
| Matched score mean | .660 | .631 | .642 |

Corrected survivor-conditioned center error is close to native, unlike the
old stress adapter. That is improved geometry among surviving detections, not
an overall localization improvement: corrected recall is lower and 51 of 89
GT actor-frames are missed.

GT-conditioned paired metrics use only the 37 actors detected in both native
and corrected branches. No array-index or forced detector pairing is used.

| Metric | Mean | Median | p90 | Max |
| --- | ---: | ---: | ---: | ---: |
| Native-to-corrected center shift | .178 m | .176 m | .348 m | .617 m |
| Corrected center error | .285 m | .248 m | .545 m | 1.003 m |
| Size-vector shift | .207 m | .155 m | .404 m | .738 m |
| Score shift (corrected - native) | -.087 | -.052 | +.024 | +.122 |
| Pi-symmetric yaw-axis shift | .049 rad | .029 rad | .115 rad | .211 rad |

There are no class changes among the 37 jointly detected actors. Six have a
directional front/back yaw flip; the Pi-symmetric axis metric prevents those
from inflating box-axis instability.

| Range | GT | Native recall | Old stress recall | Corrected recall | Corrected pts/object median |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0--20 m | 16 | 93.75% | 93.75% | 93.75% | 264 |
| 20--40 m | 19 | 89.47% | 78.95% | 78.95% | 17 |
| 40--60 m | 31 | 80.65% | 29.03% | 25.81% | 5 |
| 60--80 m | 23 | 56.52% | 4.35% | 0.00% | 0 |

All bins are descriptive, correlated observations from one log. The corrected
proxy proves that the old alternating-empty-channel bug was not the sole cause
of range loss. A single 16-ring source still supplies too few returns on many
far GT boxes in this sample.

## 18. Final interpretation and decision

| Finding | Assessment | Evidence |
| --- | --- | --- |
| Candidate/score-stage sensitivity | Strong | Native S1 598 versus corrected 244; loss precedes NMS |
| Confidence degradation | Moderate | jointly detected score shift -.087; survivor S1 distribution changes modestly |
| NMS interaction as primary loss | Unsupported | sparse circle/IoU reduction is no stronger than native |
| Point-count sensitivity | Strong | one-source 16-ring input retains 25.9%; recall loses 36.0 pp |
| Beam-structure sensitivity | Moderate | corrected and old yield similar S1/recall despite very different retained counts and geometry |
| MORAI domain conclusion | Unresolved | no MORAI VLP-16 calibration or held-out actor-GT sequence |

**Corrected adapter classification: B -- useful one-lidar 16-ring sparsity
proxy, but not close enough to VLP-16 geometry.** It is physically grounded in
one source origin, unique fixed rings, preserved azimuth returns, and original
ego coordinates. It is not exact VLP-16 emulation because mean/max target
angle mismatch is 1.631/4.336 degrees, its source is VLP-32C, and no raycasting
or VLP-16 firing/return physics is modeled.

Retraining decision: **NOT YET**. The corrected experiment establishes strong
input-sparsity sensitivity before NMS, but it is neither MORAI target-domain
evidence nor a general AV2 benchmark. The next highest-value task that does
not require MORAI data is **raw pre-score instrumentation**: preserve decoded
head candidates before the class/distance score gate, then replay this fixed
native/corrected sample to separate model peak loss from threshold-gate loss.

Corrected external artifacts:

```text
/home/didgang1203/datasets/centerpoint/av2_sensor_sample_v1/derived/
  source_ring_vlp16_v2/<timestamp>.npz
  manifests/02678d04-cc9f-3148-9f95-1ba66347dff9_source_ring_vlp16_v2.json

/home/didgang1203/datasets/centerpoint/av2_vlp16_inference_v1/
  stage_dumps/ten_sweep/<timestamp>/source_ring_vlp16_v2/
  metrics/source_ring_v2_stage_summary.json
  metrics/source_ring_v2_stage_scores.csv
```

## 19. Raw head and score-gate instrumentation

### 19.1 Pinned head contract

The runtime source remains Autoware Universe `autoware_lidar_centerpoint`
0.51.0. `CenterPointTRT::inference()` binds the head tensors in this exact
order; `CenterPointTRT::postProcess()` passes them to
`PostProcessCUDA::generateDetectedBoxes3D_launch()`.

| Binding | Runtime shape | Buffer type | Meaning in the pinned decoder |
| --- | --- | --- | --- |
| `spatial_features` | `[1, 32, 480, 480]` | FP32 input buffer | scattered pillar features |
| `heatmap` | `[1, 5, 480, 480]` | FP32 output buffer | per-class logits |
| `reg` | `[1, 2, 480, 480]` | FP32 output buffer | x/y cell offsets |
| `height` | `[1, 1, 480, 480]` | FP32 output buffer | box z |
| `dim` | `[1, 3, 480, 480]` | FP32 output buffer | log width/length/height |
| `rot` | `[1, 2, 480, 480]` | FP32 output buffer | yaw sine/cosine |
| `vel` | `[1, 2, 480, 480]` | FP32 output buffer | x/y velocity |

The binding names are taken from the pinned TensorRT engine metadata. The
engine uses FP16 precision internally, but this implementation supplies FP32
input/output buffers.

`generateBoxes3D_kernel()` visits every 480x480 cell. It applies sigmoid to
all five class logits and selects the highest-scoring class. There is no
heatmap local-maximum operation and no top-K stage. It decodes x/y from the
winning cell and `reg`, selects the first radial distance bin whose upper
bound exceeds the decoded radius, then applies that class/bin threshold. The
active model package defines four bounds `[50, 90, 121, 200]` m and a score
threshold of `0.35` for every class in every bin. A separate yaw-vector norm
gate is then applied (`0.3` for CAR/TRUCK/BUS/BICYCLE and `0.0` for
PEDESTRIAN). Positive cells are compacted and score-sorted to form S1.

Consequently there is no independently materialized R1 candidate list in the
pinned architecture:

```text
R0 dense head tensors
  -> per-cell sigmoid/max-class + decode + distance/class score gate + yaw gate
  -> S1 compacted, score-sorted boxes
```

The source locations are:

- `lib/centerpoint_trt.cpp`: `CenterPointTRT::inference`,
  `CenterPointTRT::postProcess`
- `lib/postprocess/postprocess_kernel.cu`: `generateBoxes3D_kernel`,
  `PostProcessCUDA::generateDetectedBoxes3D_launch`
- `src/node.cpp`: `LidarCenterPointNode::pointCloudCallback`

### 19.2 Opt-in probe design

The isolated overlay adds `enable_raw_head_dump`, default `false`. When it is
enabled, the six output tensors are copied read-only to bounded FP32 binary
snapshots after the unchanged production postprocess call. Metadata records
the timestamp, shapes, grid geometry, score thresholds, distance bounds, and
yaw thresholds. The metadata file is written last and is the completion
sentinel. Dumps live outside Git; dense tensors are never written as CSV.

Offline analysis reproduces the pinned kernel's gate, verifies reproduced
accepted scores/count against S1, and then emits only bounded summaries:
per-class distributions, top 100 raw-class and decoder-winner cells, rejection
reasons, threshold margins, and GT-neighborhood evidence. A five-cell radius
equals 1.6 m at the 0.32 m output-grid resolution; it is deliberately smaller
than the 3 m final-output association gate to reduce contamination from nearby
actors.

The reserved KalmanNet training was allowed to finish before the isolated
overlay rebuild. A later CPU-only KalmanNet evaluation briefly pushed
available memory below 4 GiB during replay; the Codex-owned replay was stopped,
its partial output was preserved, and the complete run was restarted only
after that evaluation ended and available memory recovered to 14 GiB.

### 19.3 Resume checkpoint

No build/replay job is left running in the background. Resume only after the
KalmanNet CPU training PID has ended and memory has recovered:

```bash
cd /tmp/heven-worktrees/centerpoint-stability-audit-v1
free -h
nvidia-smi

export TENSORRT_ROOT=/home/didgang1203/opt/tensorrt/10.8.0.43-cuda11.8
export ROS_DOMAIN_ID=77
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2 CMAKE_BUILD_PARALLEL_LEVEL=2

bash tools/centerpoint_offline/prepare_autoware_stage_overlay.sh
nice -n 10 bash tools/centerpoint_offline/build_autoware_centerpoint_isolated.sh
```

After the build succeeds, use a previously unused external output directory
for the fixed 10-frame, 20-run paired replay. The completed audit used
`raw_head_ten_sweep_v3`; substitute another suffix when reproducing it:

```bash
bash tools/centerpoint_offline/run_centerpoint_raw_head_sample.sh \
  --derived-root /home/didgang1203/datasets/centerpoint/av2_sensor_sample_v1/derived \
  --output-root /home/didgang1203/datasets/centerpoint/av2_vlp16_inference_v1/stage_dumps/raw_head_ten_sweep_repro

/home/didgang1203/venvs/heven-centerpoint/bin/python \
  tools/centerpoint_offline/summarize_centerpoint_raw_head_audit.py \
  --annotations /home/didgang1203/datasets/centerpoint/av2_sensor_sample_v1/val/02678d04-cc9f-3148-9f95-1ba66347dff9/annotations.feather \
  --derived-root /home/didgang1203/datasets/centerpoint/av2_sensor_sample_v1/derived \
  --run-root /home/didgang1203/datasets/centerpoint/av2_vlp16_inference_v1/stage_dumps/raw_head_ten_sweep_repro \
  --reference-run-root /home/didgang1203/datasets/centerpoint/av2_vlp16_inference_v1/stage_dumps/ten_sweep \
  --output /home/didgang1203/datasets/centerpoint/av2_vlp16_inference_v1/metrics/raw_head_gate_summary.json \
  --gt-csv /home/didgang1203/datasets/centerpoint/av2_vlp16_inference_v1/metrics/raw_head_gt_evidence.csv \
  --top-k-jsonl /home/didgang1203/datasets/centerpoint/av2_vlp16_inference_v1/metrics/raw_head_top100.jsonl
```

The summarizer deliberately aborts if reproduced R0 gate counts/scores differ
from S1. It reports strict and behavioral S1--S5 comparisons against the prior
raw-dump OFF capture, but does not treat separate-process numeric drift as an
instrumentation failure: the pinned node seeds a preprocessing shuffle offset
from wall-clock time. The probe itself runs only after the unchanged
postprocess has produced its output.

## 20. GT-conditioned heatmap evidence

The complete replay contains the same 10 sweeps and 89 supported
`REGULAR_VEHICLE` actor-frames as the corrected source-ring experiment. At
each GT center, the strongest CAR heatmap value in a 5-cell (1.6 m) radius was
sampled from R0. This is a GT-conditioned measurement, not forced detector
pairing.

| R0 CAR evidence near GT | Native | Corrected | Delta |
| --- | ---: | ---: | ---: |
| Count | 89 | 89 | 0 |
| Mean score | .554 | .328 | -.226 |
| Median score | .669 | .253 | -.416 |
| p25 | .383 | .039 | -.344 |
| p75 | .750 | .603 | -.147 |
| Maximum | .862 | .856 | -.006 |
| GT neighborhoods at/above .35 | 70 | 38 | -32 |

The maximum remains high because near, well-observed actors survive. The
distribution, especially its median and lower quartile, moves sharply down.
This is direct evidence that the TensorRT head heatmap signal weakens under
the corrected one-source 16-ring input before NMS.

| GT range | GT | Native mean R0 score | Corrected mean | Native >=.35 | Corrected >=.35 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0--20 m | 16 | .739 | .739 | 15 | 15 |
| 20--40 m | 19 | .649 | .503 | 17 | 15 |
| 40--60 m | 31 | .544 | .207 | 25 | 8 |
| 60--80 m | 23 | .362 | .063 | 13 | 0 |

The divergence is negligible at 0--20 m, appears at 20--40 m, and is severe
beyond 40 m. These are descriptive correlated observations from one AV2 log.

## 21. Score-gate rejection

The CPU reproduction of `generateBoxes3D_kernel()` matched each run's actual
S1 accepted count and sorted scores. The one-candidate differences from the
earlier 598/244 totals are consistent with the pinned node's time-seeded
preprocessing shuffle; the new paired run produced 599/243.

| Gate result over 10 x 480 x 480 cells | Native | Corrected | Delta |
| --- | ---: | ---: | ---: |
| Considered cells | 2,304,000 | 2,304,000 | 0 |
| Accepted into S1 | 599 | 243 | -356 |
| Below score threshold | 2,303,392 | 2,303,755 | +363 |
| Rejected by yaw norm after score pass | 9 | 2 | -7 |
| Outside configured distance bins | 0 | 0 | 0 |

CAR-winner accepted cells account for most of the change: 525 native versus
219 corrected. The remaining accepted counts are TRUCK 2/1, BUS 0/0,
BICYCLE 20/3, and PEDESTRIAN 52/20. The configured score threshold is exactly
0.35 for every class and every distance bin, so no larger far-range threshold
is present in this experiment.

The surviving S1 score distribution changes much less than its population:

| S1 score | Native | Corrected |
| --- | ---: | ---: |
| Count | 599 | 243 |
| Mean | .509 | .500 |
| Median | .482 | .465 |
| p10 | .369 | .365 |
| p90 | .709 | .698 |

This survivor conditioning is why S1 score summaries alone understated the
degradation. R0 GT-neighborhood evidence exposes the large population shifted
below the gate.

The raw probe is default-off and copies R0 only after the unchanged
postprocess has already produced S1/S2. Across separate OFF and ON processes,
all final-stage counts and all circle/pre-IoU/post-IoU counts matched in 10/10
frames for both conditions; S1 count matched 9/10. Strict numeric comparison
is confounded by the pinned node's wall-clock-seeded preprocessing offset.
The first frame's final output passed the original 5 mm/0.005 strict
tolerance; its native S1 maximum center/score changes were 4.3 mm/.0116.
These small cross-process differences are recorded rather than attributed to
the post-output read-only tensor copy.

## 22. Native-hit / corrected-miss decomposition

Final detection outcomes are based on independent GT-to-detection Hungarian
matching with the existing 3 m BEV gate.

| Outcome | Actor-frames |
| --- | ---: |
| Detected native and corrected | 37 |
| Detected native, missed corrected | 33 |
| Missed native, detected corrected | 1 |
| Missed both | 18 |

All 33 native-hit/corrected-miss cases are already explained at R0-to-S1:

| B category | Count | Meaning |
| --- | ---: | --- |
| B1 | 26 | corrected CAR peak is more than .10 below threshold after a native above-threshold peak |
| B2 | 7 | corrected CAR peak remains within .10 below threshold |
| B3 | 0 | peak above threshold but no GT-gated S1 candidate |
| B4 | 0 | corrected S1 exists but later postprocessing loses the object |

Thus there is no evidence in this sample that circle NMS, IoU NMS, yaw
validity, class competition, or another decoded-candidate condition explains
the native-hit/sparse-miss population. The continuous heatmap confidence loss
is converted into a binary miss by the 0.35 gate.

## 23. Far-range failure decomposition

There are 23 supported GT vehicles at 60--80 m. Fourteen contain zero points
after source-ring selection and nine retain nonzero points. Nevertheless, all
23 corrected GT-neighborhood peaks are below 0.35; their maximum is .329 and
their median is .017. Of the 13 actors detected natively, eight lose every
selected point and five retain nonzero corrected points, but all 13 fall below
the score threshold.

Therefore the 0% corrected recall at 60--80 m is not solely an empty-cuboid
artifact. Complete point loss explains much of it, while the remaining sparse
evidence is also insufficient for an above-threshold network response.

Across all 89 actors, corrected points/object versus corrected local heatmap
score has Spearman rho `.848` (`p=9.88e-26`). Point-retention ratio versus
native-to-corrected score change has rho `.471` (`p=3.25e-6`). These are strong
descriptive associations, not independent causal estimates.

## 24. Causal interpretation

| Hypothesis | Assessment | Evidence |
| --- | --- | --- |
| H1: sparse input weakens/disappears raw vehicle heatmap peaks | Strong | GT-local mean .554 -> .328; above-threshold 70 -> 38; range-dependent collapse starts in R0 |
| H2: score threshold amplifies continuous confidence loss | Strong | all 33 native-hit/sparse-miss objects are below .35; 7 sit within .10 below it |
| H3: distance-dependent gate amplifies far loss | Unsupported | all class/bin thresholds are .35 and no cell is outside the configured bins |
| H4: other decode/validity logic contributes materially | Unsupported | B3/B4 are zero; yaw rejection is 9 native versus 2 corrected |

The primary root cause in this bounded proxy experiment is **network-head
confidence degradation caused by reduced object evidence**. The fixed score
gate is the immediate binary mechanism that turns that degradation into S1
candidate loss. NMS is downstream and is not the source of the 599-to-243
candidate reduction.

## 25. Decision and artifacts

Retraining remains **NOT YET**. The head is demonstrably sparse-input
sensitive, which raises the priority of MORAI-domain validation and possible
fine-tuning, but this is a 10-sweep, one-log AV2 source-ring proxy rather than
MORAI target-domain evidence. Threshold changes are also not promoted to
production from this proxy.

The next highest-value MORAI-independent task was a **cached-R0 score-gate
sensitivity experiment**. Sections 26--30 report its completion. It varied
only the offline gate, measured recovered GT evidence together with
candidate/FP growth, and does not claim a deployable threshold.

Completed external artifacts:

```text
/home/didgang1203/datasets/centerpoint/av2_vlp16_inference_v1/
  stage_dumps/raw_head_ten_sweep_v3/
  metrics/raw_head_v1/summary.json
  metrics/raw_head_v1/gt_evidence.csv
  metrics/raw_head_v1/top100.jsonl
  metrics/raw_head_v1/analyzer.log
  metrics/raw_head_v1/stage_summary.json
  metrics/raw_head_v1/stage_scores.csv
  metrics/raw_head_v1/stage_analyzer.log
```

## 26. Cached-R0 score-gate sensitivity

No TensorRT inference was executed for this experiment. Each threshold reused
the exact six R0 tensors in `raw_head_ten_sweep_v3`; only a fresh in-memory
copy of the score-threshold matrix changed. The offline path faithfully
reproduces the pinned 0.51.0 sequence:

```text
R0 sigmoid / winner class / box decode
  -> class-distance score and yaw-validity gates
  -> score-ordered S1
  -> class-agnostic 0.5 m circle NMS
  -> ROS box/yaw convention
  -> 10 m search / IoU > 0.1 suppression
  -> area-based class remapping
```

At the 0.35 baseline, every cached frame reproduced every captured S1--S5
stage, including counts, classes, scores, centers, dimensions and yaw. The
largest scalar/geometry discrepancy was `8.53e-6`.

| Baseline stage total | Native captured/offline | Corrected captured/offline |
| --- | ---: | ---: |
| S1 post-score | 599 / 599 | 243 / 243 |
| S2 circle NMS | 163 / 163 | 73 / 73 |
| S3 pre-IoU | 163 / 163 | 73 / 73 |
| S4 post-IoU | 156 / 156 | 73 / 73 |
| S5 final | 156 / 156 | 73 / 73 |

The older 598/244 totals belong to a different inference capture. The pinned
node's wall-clock-seeded preprocessing order can move borderline cells across
0.35 between separate inference processes. This cached study consistently
uses the newer 599/243 artifacts and introduces no such cross-run confound.

## 27. Threshold-recovery frontier

The primary global sweep was bounded to 0.20--0.45. A small 0.01 fine sweep
between 0.20 and 0.35 located the discrete transition points; no threshold
below 0.20 was evaluated.

| Threshold | Recall | Vehicle FP/frame | Precision | F1 | S1/frame | S5/frame | Mean matched error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| .20 | 52.81% | 4.9 | 48.96% | .508 | 75.1 | 14.8 | .356 m |
| .225 | 50.56% | 4.0 | 52.94% | .517 | 58.0 | 12.8 | .299 m |
| .25 | 50.56% | 3.4 | 56.96% | .536 | 48.9 | 11.1 | .299 m |
| .275 | 49.44% | 2.6 | 62.86% | .553 | 40.1 | 9.9 | .295 m |
| **.29** | **49.44%** | **2.4** | **64.71%** | **.561** | **36.4** | **9.5** | **.295 m** |
| .30 | 47.19% | 2.4 | 63.64% | .542 | 34.5 | 9.2 | .286 m |
| .325 | 44.94% | 1.9 | 67.80% | .541 | 27.7 | 8.1 | .288 m |
| .35 baseline | 42.70% | 1.8 | 67.86% | .524 | 24.3 | 7.3 | .278 m |
| .375 | 40.45% | 1.3 | 73.47% | .522 | 20.1 | 6.1 | .278 m |
| .40 | 39.33% | 1.2 | 74.47% | .515 | 17.4 | 5.6 | .262 m |
| .45 | 34.83% | .9 | 77.50% | .481 | 13.2 | 4.6 | .283 m |

The `.29` neighborhood is the most informative diagnostic trade-off region,
not an optimal or deployable threshold. Relative to sparse 0.35 it recovers
six GT, adds six vehicle false positives and 121 S1 candidates: 1.0 extra FP
and 20.2 extra S1 candidates per recovered GT. At 0.20, nine GT are recovered
at a cost of 31 extra vehicle FP and 508 extra S1 candidates (3.44 and 56.4
per recovery respectively).

The largest corrected S1 load in the bounded sweep is 751 total / 147 maximum
per frame at .20, versus 243 total / 49 maximum at .35: approximately 3.1x
aggregate and 3.0x peak growth. The configured 5x/10x explosion guards were
not reached.

Native 0.35 remains far ahead: 78.65% recall, 5.1 vehicle FP/frame, 57.85%
precision, 59.9 S1/frame and 15.6 final detections/frame. Sparse 0.20 already
has more S1 load than native (75.1 versus 59.9) and nearly the same vehicle FP
load (4.9 versus 5.1), yet reaches only 52.81% recall. Threshold reduction
therefore cannot recover most of the missing representation.

New detections are also lower quality. The six actors newly recovered at .29
have mean/median center error `.402/.403 m`; the nine at .20 have `.687/.476
m`. The latter is substantially worse than the baseline survivor-conditioned
`.278 m`, so the extra recall is not free localization improvement.

## 28. Distance-specific recoverability

| Threshold | 0--20 m (16) | 20--40 m (19) | 40--60 m (31) | 60--80 m (23) |
| ---: | ---: | ---: | ---: | ---: |
| .35 | 93.75% | 78.95% | 25.81% | 0.00% |
| .325 | 93.75% | 84.21% | 25.81% | 4.35% |
| .30 | 93.75% | 84.21% | 29.03% | 8.70% |
| .29 | 93.75% | 84.21% | 35.48% | 8.70% |
| .275 | 93.75% | 84.21% | 35.48% | 8.70% |
| .25 | 93.75% | 89.47% | 35.48% | 8.70% |
| .20 | 93.75% | 89.47% | 38.71% | 13.04% |

Of the 23 far-range GT, all 14 zero-point cuboids remain unrecovered throughout
the sweep. Three of nine nonzero-point cuboids recover at 0.20; six remain
unrecovered. No zero-point cuboid even obtains a GT-local CAR threshold
crossing. This supports a representation/evidence limit rather than a gate-only
explanation of far-range loss.

The native-hit/corrected-miss cohort contains 33 actors. Final recoveries are:

| Threshold | Local evidence crosses | S1 match | Final match | Still unrecovered |
| ---: | ---: | ---: | ---: | ---: |
| .325 | 2 | 2 | 2 | 31 |
| .30 | 4 | 4 | 4 | 29 |
| .275 | 6 | 6 | 6 | 27 |
| .25 | 7 | 7 | 7 | 26 |
| .225 | 8 | 7 | 7 | 26 |
| .20 | 9 | 9 | 9 | 24 |

The one `.225` evidence crossing without an S1 match is retained explicitly;
a CAR-local value crossing the threshold is not necessarily the cell's winning
class. No forced recovery or detector pairing is used.

## 29. Global versus CAR-only lowering

CAR-only lowering holds the other four head-class thresholds at 0.35. Vehicle
recall and vehicle FP are identical to the global policy at every tested value,
while avoiding low-score non-vehicle candidates and outputs.

| Policy | Vehicle recall | Vehicle FP/frame | All final | S1 total |
| --- | ---: | ---: | ---: | ---: |
| Global .35 | 42.70% | 1.8 | 73 | 243 |
| Global .30 | 47.19% | 2.4 | 92 | 345 |
| CAR .30 / others .35 | 47.19% | 2.4 | 83 | 321 |
| Global .25 | 50.56% | 3.4 | 111 | 489 |
| CAR .25 / others .35 | 50.56% | 3.4 | 96 | 439 |
| Global .20 | 52.81% | 4.9 | 148 | 751 |
| CAR .20 / others .35 | 52.81% | 4.9 | 114 | 629 |

At global .29 the unmatched/non-supported output composition is 24 CAR, 24
PEDESTRIAN, one TRUCK and two BICYCLE boxes over ten frames. Most occur at
20--40 m (31/51). CAR-only lowering is therefore the more efficient diagnostic
policy for vehicle recovery, but it does not fix the fundamental recall limit.

## 30. Runtime calibration versus representation

The outcome is **Mixed**: modest CAR score calibration can recover a small
near/mid-range subset at manageable load, but threshold adjustment is
insufficient overall and especially at long range.

| Hypothesis | Assessment | Evidence |
| --- | --- | --- |
| H1: 0.35 materially amplifies sparse recall loss | Moderate | lowering to .29 recovers 6/33 native-hit misses, but 27 remain |
| H2: modest lowering recovers useful GT without disproportionate FP | Moderate | .29 adds 6 GT and 6 vehicle FP; new error is .402 m |
| H3: large recovery requires unacceptable growth | Strong | even .20 adds 508 S1 and 31 FP but recovers only 9 GT |
| H4: CAR-only dominates global lowering | Strong | same vehicle metrics with fewer S1 and final non-vehicle outputs |
| H5: far-range loss is largely gate-unrecoverable | Strong | only 3/23 recover at .20; all 14 zero-point GT remain missed |

Retraining remains **NOT YET** because this is still one AV2 log and a
source-ring proxy, not target-domain MORAI evidence. The result does raise the
priority of future MORAI-domain adaptation: runtime threshold calibration can
only provide partial compensation for a raw representation loss.

The next highest-value MORAI-independent task is a **cached-R0 regression-head
geometry audit**. It should compare GT-conditioned center, size and yaw
regression for native versus corrected inputs, including below-threshold
actors, to determine whether sparsification damages geometry as well as
confidence. The already completed MORAI converter/training preflight should
not be duplicated before target data arrives.

External outputs:

```text
/home/didgang1203/datasets/centerpoint/av2_vlp16_inference_v1/
  metrics/score_gate_sensitivity_v1/
    summary.csv
    per_threshold.json
    per_gt_recovery.csv
    fp_breakdown.csv
    distance_tradeoff.csv
    recall_vs_threshold.png
    fp_per_frame_vs_threshold.png
    s1_per_frame_vs_threshold.png
    precision_recall.png
    recovered_gt_vs_extra_fp.png
    distance_recall_vs_threshold.png
```

## 31. Cached-R0 Regression Geometry Audit

This offline-only audit reuses the exact ten-sweep TensorRT R0 snapshots used
by the raw-confidence and score-gate audits. It neither invokes TensorRT nor
changes a runtime threshold. The input remains the corrected one-source,
16-ring proxy; it is not a MORAI result or an exact VLP-16 simulation.

### 31.1 Decoder contract

The pinned Autoware 0.51 CUDA decoder was used directly in the audit helper:

| Head | Pinned decode semantics |
| --- | --- |
| `reg` | `x/y = range_min + voxel_size * downsample * (grid + reg)` |
| `height` | raw value is decoded box `z` |
| `dim` | raw order is `[width, length, height]`; every component is `exp(raw)` |
| `rot` | raw yaw is `atan2(rot[0]=sin, rot[1]=cos)`; AV2 comparison yaw is `-raw_yaw - pi/2` |
| `vel` | raw `vx/vy`, reported only as a native-to-sparse diagnostic |

For every one of the 89 supported `REGULAR_VEHICLE` GT cuboids, two probes
were retained without score filtering:

- **Same-cell:** native and sparse heads decoded at the identical GT-projected
  feature-grid cell. This isolates regression-head change from peak movement.
- **Winner-cell:** each condition decoded at its own highest CAR heatmap cell
  in the same bounded GT neighborhood. This measures the geometry of the
  candidate the detector is actually most likely to select.

Yaw uses a pi-symmetric box-axis metric. Thus a front/rear flip is not counted
as a box-axis geometry failure.

### 31.2 Same-cell versus winner-cell geometry

| Metric, all 89 GT | Native same/winner | Sparse same/winner | Interpretation |
| --- | ---: | ---: | --- |
| Same-cell XY error | .098 m | .109 m | Within-cell center regression is largely retained |
| Same-cell native-to-sparse XY shift | \- | .051 m | Small relative to a .32 m feature stride |
| Winner-cell XY error | .379 m | .756 m | Strongest sparse candidate is materially less localised |
| Winner grid displacement | \- | .631 m | Most of the winner-center change is peak movement |
| Winner residual shift after grid movement | \- | .058 m | Remaining within-cell regression change is small |
| Winner relative dimension error | .092 | .133 | Sparse dimensions have a worse tail |
| Winner box-axis yaw error | .071 rad | .162 rad | Sparse yaw is less stable |
| Winner absolute Z error | .237 m | .474 m | Sparse height/Z regression is less stable |

The same-cell dimension, yaw, and Z distributions also develop long sparse
tails (same-cell dimension relative-error p90 `.159 -> .493`, yaw-axis shift
p90 `.468 rad`, Z-shift p90 `.944 m`). Consequently, this is not a purely
confidence-only result. However, the dominant change in selected box location
is the heatmap winner moving, rather than a uniform failure of center-offset
regression at the same cell.

### 31.3 Cohorts and range

The survivor cohort (37 detected by both conditions) remains relatively stable:
native/sparse winner center error is `.233/.285 m` and mean peak displacement
is `.184 m`. The critical native-hit/sparse-miss cohort (33 GT) has a mean
score change of `-.461`, native/sparse winner center error `.306/.929 m`, and
mean peak displacement `.833 m`. Its winner dimension error rises
`.101 -> .144`, yaw-axis error `.057 -> .160 rad`, and Z error `.298 -> .569 m`.
Thus these misses are not explained only by a binary score gate: the sparse
winner often moves to a geometrically worse local maximum.

| Distance | GT | Native winner XY | Sparse winner XY | Sparse peak drift | Native/sparse yaw-axis |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0--20 m | 16 | .234 m | .273 m | .140 m | .038/.041 rad |
| 20--40 m | 19 | .295 m | .479 m | .355 m | .039/.062 rad |
| 40--60 m | 31 | .355 m | .690 m | .571 m | .050/.227 rad |
| 60--80 m | 23 | .582 m | 1.412 m | 1.283 m | .150/.243 rad |

For 60--80 m, the 14 zero-sparse-point cuboids have sparse winner error
`1.735 m` and drift `1.796 m`; the nine nonzero-point cuboids improve to
`.909 m` and `.485 m`, respectively. A same-cell value can still look smooth
when an object has no points, but its sparse winner is usually a background
maximum and is not useful detector geometry.

### 31.4 Confidence and geometry relationship

Across all 89 GT, sparse CAR score has weak association with same-cell center
error (Spearman rho `-.150`) but strong descriptive association with
winner-cell center error (`-.683`), dimension error (`-.508`), and yaw-axis
error (`-.537`). Sparse points per object likewise associates with winner
center error (`rho=-.564`) but not same-cell center error (`rho=-.080`). These
are descriptive one-log correlations, not causal estimates.

### 31.5 Causal classification and implication

| Component | Assessment | Evidence |
| --- | --- | --- |
| Heatmap confidence | Strong degradation | prior R0 audit: mean CAR score `.554 -> .328`; threshold survivors `70 -> 38` |
| Peak location | Strong degradation | winner drift `.631 m` overall, `1.283 m` at 60--80 m |
| Same-cell center regression | Weak degradation | center shift `.051 m`; XY error changes `.098 -> .109 m` |
| Dimension regression | Moderate degradation | winner relative error `.092 -> .133` with sparse long tails |
| Yaw regression | Moderate degradation | winner axis error `.071 -> .162 rad`, especially beyond 40 m |
| Height/Z regression | Moderate degradation | winner Z error `.237 -> .474 m` |

The representation failure is therefore **Mixed**: confidence weakening and
peak-localisation instability dominate, while Z, dimensions, and yaw also
degrade for a meaningful subset. Score calibration alone cannot restore a
candidate whose heatmap maximum has moved to background geometry.

The retraining decision remains **NOT YET**. This AV2 source-ring proxy raises
the priority of eventual target-domain detector adaptation, potentially beyond
a classification-only adjustment, but it cannot establish a MORAI fine-tuning
requirement without a sequence-disjoint MORAI actor-GT bag.

External outputs (not committed):

```text
/home/didgang1203/datasets/centerpoint/av2_vlp16_inference_v1/
  metrics/regression_head_geometry_v1/
    per_gt_geometry.csv
    cohort_summary.csv
    correlations.csv
    head_summary.json
    native_vs_sparse_center_error.png
    score_delta_vs_center_error_delta.png
    sparse_points_vs_center_error.png
    winner_grid_displacement_by_distance.png
```

The next highest-value MORAI-independent task is a **deterministic
point-count-matched random-downsample control replay** of the same ten AV2
sweeps. It would distinguish simple point-count loss from the one-source
ring-structure effect before interpreting any future MORAI result.

## 32. Point-Count versus Ring-Structure Controlled Replay

This experiment resolves the prior count-versus-structure limitation with ten
identical timestamps and 89 supported vehicle GT. Threshold remains `.35`,
densification remains disabled, and all runs use the same Autoware 0.51 model,
TensorRT engines, ROI, stage instrumentation, and Hungarian 3 m GT matching.
No production parameter was changed.

| Arm | Input |
| --- | --- |
| A | Native AV2 aggregate: dual LiDAR / 64 `laser_number` values |
| B | `up_lidar` only, all physical rings 0--31, no subsampling |
| C | Up-only without-replacement random subset, exactly count-matched to D per frame; seeds 11/29/47 |
| D | Existing corrected up-only physical source-ring 16-ring proxy |

C and D therefore have the same source LiDAR, timestamp, AV2 egovehicle
coordinates, input count, model and threshold. They differ only in whether
the retained up-LiDAR points are randomly distributed or constrained to the
16 selected physical vertical rings. This is a controlled contrast, not an
additive causal decomposition of a nonlinear detector.

### 32.1 Aggregate result

| Arm | Recall | Vehicle FP/frame | S1/frame | Final/frame | GT-local R0 CAR | Winner XY | Peak drift | Zero-point GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A dual64 | 78.65% | 5.1 | 59.9 | 15.6 | .554 | .379 m | .396 m | 4.5% |
| B up32 | 70.79% | 5.2 | 58.5 | 14.9 | .531 | .391 m | .414 m | 5.6% |
| C random matched | 58.05% ± 1.06% | 3.73 ± .21 | 43.6 ± .22 | 12.1 ± .05 | .442 ± .004 | .480 ± .004 m | .503 ± .003 m | 8.25% ± 1.06% |
| D ring16 | 42.70% | 1.8 | 24.3 | 7.3 | .328 | .756 m | .760 m | 24.7% |

The random controls are stable: recall ranges only 57.30--59.55%, R0 score
.437--.447, and winner XY error .475--.484 m. D is substantially below this
entire random-replay range.

### 32.2 Distance contrast

| Distance | A dual64 | B up32 | C random matched | D ring16 |
| --- | ---: | ---: | ---: | ---: |
| 0--20 m | 93.75% | 93.75% | 93.75% | 93.75% |
| 20--40 m | 89.47% | 84.21% | 84.21--89.47% | 78.95% |
| 40--60 m | 80.65% | 77.42% | 51.61--64.52% | 25.81% |
| 60--80 m | 56.52% | 34.78% | 4.35--17.39% | 0.00% |

At 60--80 m, median GT points are 7 for B, 2--3 for C, and 0 for D;
zero-point fractions are 8.7%, 13.0--17.4%, and 60.9%, respectively. Random
count matching preserves some far-object evidence that structured rings miss.

### 32.3 Controlled contrasts and assessment

- **A → B, second-source / density contribution:** recall falls 7.87 points
  and far-range recall 56.52% → 34.78%. This is a moderate effect; it may
  include FOV and source-placement interactions, not merely a “second lidar”
  scalar contribution.
- **B → C, point-count contribution:** matching Up32 to Ring16 point counts
  reduces recall by 12.74 points and R0 CAR score .531 → .442. Point count is
  a strong contributor.
- **C → D, ring-structure contribution:** with the same source and exact
  per-frame count, structured selection further reduces recall by 15.35
  points, S1/frame 43.6 → 24.3, R0 .442 → .328, and increases zero-point GT
  8.25% → 24.7%. This is strong evidence that vertical sampling structure has
  an additional effect beyond count alone.

| Effect | Assessment |
| --- | --- |
| Point count | Strong |
| Ring structure | Strong |
| Second-source / source-density | Moderate |
| Range interaction | Strong; effects concentrate beyond 40 m |

The conclusion is **Mixed, with both point-count and structured vertical-ring
effects material**. It does not establish expected MORAI accuracy: the data is
one AV2 log and D is a source-ring proxy rather than a physical VLP-16 scan.
It does show that interpreting the prior ring16 result as merely random
point-count loss would be incorrect.

Retraining remains **NOT YET**. These results strengthen the priority of
sequence-disjoint MORAI actor-GT acquisition and subsequent target-domain
evaluation, but do not justify a production threshold change or MORAI
fine-tuning without target-domain evidence.

External outputs (not committed):

```text
/home/didgang1203/datasets/centerpoint/av2_vlp16_inference_v1/
  metrics/sampling_control_v1/
    summary.csv
    summary.json
    per_seed.csv
    per_frame.csv
    per_gt.csv
    distance_summary.csv
```
