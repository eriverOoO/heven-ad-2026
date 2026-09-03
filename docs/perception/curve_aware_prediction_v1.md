# Curve-Aware Prediction v1 — motion-history yaw rate

**Scope:** one bounded, opt-in change in the prediction adapter/runtime layer
(`ad_lidar_perception/src/tracking/autoware_prediction_node.cpp`). No new
predictor, no lane/map-aware prediction, no change to `imm_predictor.cpp`, no
change to the Dynamic Occupancy Grid.

## Problem

On curved roads and the roundabout, predicted object trajectories do not bend:
the IMM coordinated-turn (CT) hypothesis is never selected and predicted
headings stay flat.

Root cause (from the `morai_cam4_20260813_163222` prediction audit): the IMM
CT model's only curvature input is the yaw-rate measurement
`TrackMeasurement2D::yaw_rate_radps`, which the adapter fills from
`tracked_objects.twist.twist.angular.z`. The AB3DMOT tracker's Linear KF has a
10-state model `[x, y, z, yaw, l, w, h, vx, vy, vz]` with **no angular-velocity
state**, so it publishes `twist.angular.z == 0` on every frame, and CenterPoint's
real yaw is discarded upstream (`yaw_measurement_mode="unobserved"`). The IMM
therefore sees `omega == 0` forever, `update_model()` applies its
`turn_evidence < 0.03` penalty to CT every cycle, and `propagate()` takes the
straight-line branch. Audit numbers over a ~60 s window: predicted heading
change exactly `0.0` on 1981 predictions; CT selected 0 / 1704 frames; CT
probability max 0.084 — while the turn rate implied by consecutive tracked
velocity headings had median 0.078 rad/s, p95 1.26 rad/s.

## Fix

A new opt-in yaw-rate source. `AutowarePredictionAdapterConfig` gains
`yaw_rate_source` (`tracker` | `motion_history`) and a `motion_history` block.

* `tracker` (**default**) — byte-for-byte the previous behaviour:
  `measurement.yaw_rate_radps = twist.twist.angular.z`,
  `measurement.yaw_rate_variance_rad2ps2 = positive_variance(twist.covariance[35], 0.04)`.
* `motion_history` — the adapter keeps a small bounded FIFO of recent
  world-frame velocity samples per track (added to the existing per-UUID
  `TrackHistory`, capped at `history_samples`), derives a robust turn rate from
  the velocity-heading history, and overrides `measurement.yaw_rate_radps` +
  `.yaw_rate_variance_rad2ps2` **only when the validity gates pass**. On any
  gate failure it leaves the tracker value untouched (CV fallback).

The estimate feeds the *existing, unchanged* IMM. TEST B
(`ConstantLeftTurnBendsPredictionAndReachesCt`) confirms the current IMM logic
responds correctly to a valid non-zero yaw-rate (fed omega 0.30 -> CT
probability ~0.88, fused yaw-rate ~0.27, predicted heading bends left), so
`imm_predictor.cpp` was not modified.

### Algorithm

Per track, each frame, in `adapt_with_diagnostics()` loop 2:

1. Push `{stamp_ns, vx_world, vy_world}` onto the track's `velocity_history`;
   drop the oldest while `size() > history_samples`.
2. If `yaw_rate_source == motion_history`, call
   `estimate_yaw_rate_from_motion_history(velocity_history, cfg, max_gap_s)`:
   * **Gates (any failure -> `std::nullopt`, keep tracker rate):**
     `size() < history_samples`; any sample speed `< min_speed_mps`; any
     inter-sample `dt <= 0` (duplicate / non-monotonic) or `dt > max_gap_s`
     (history discontinuity — `max_gap_s = imm.maximum_update_interval_s`,
     2.0 s); any non-finite input or slope.
   * **Angle unwrap:** each adjacent pair contributes
     `wrap(atan2(vy_i, vx_i) - atan2(vy_{i-1}, vx_{i-1})) / dt_i`, wrapping the
     *difference* into `(-pi, pi]`. `+179 deg -> -178 deg` yields `+3 deg`, never
     a `2*pi` spike.
   * **Estimator:** median of the `history_samples - 1` adjacent slopes
     (outlier-robust once `history_samples >= 4`).
   * **Clamp:** `omega_est = clamp(median, -max_yaw_rate_radps, +max_yaw_rate_radps)`.
3. On a valid estimate: `measurement.yaw_rate_radps = omega_est`,
   `measurement.yaw_rate_variance_rad2ps2 = yaw_rate_variance`, then
   re-`validate_imm_measurement()`.

`velocity_history` is part of `TrackHistory`, so it is staged, committed,
expired, and `reset()` exactly with the owning track (no separate lifetime).

### Parameters — `config/tracking/prediction.yaml`

```yaml
yaw_rate_source: tracker           # tracker | motion_history
motion_history:
  history_samples: 4               # median of 3 slopes
  min_speed_mps: 2.0
  max_yaw_rate_radps: 1.5          # safety clamp (audited windowed estimates <= 0.76)
  yaw_rate_variance: 0.10          # reported variance for the derived rate
```

`history_samples` default is **4**, not the ~5 first considered: the audit's
median tracked-object lifetime is ~2 frames at ~8.7 Hz, and
`history_samples = 5` left <13 % of moving track-frames eligible with 0 of 5
sampled turning windows above 0.10 rad/s. 4 still requires 3 consecutive
inter-sample intervals (a median-of-3, outlier-robust) while roughly doubling
activation.

Launch: `prediction.launch.py yaw_rate_source:=…`, forwarded as
`prediction_yaw_rate_source:=…` through `lidar_perception.launch.py` (both
tracker backends), `lidar_bag_replay.launch.py`, and
`study_pipeline_rviz.launch.py`.

### Yaw-rate variance — what it means

`yaw_rate_variance` (0.10 (rad/s)^2) is **not** a measured angular velocity. It
is a fixed, conservative variance reflecting that the rate is a finite
difference over ~`history_samples / tracking_rate` s (~0.4 s) of noisy KF
velocity. It is deliberately larger than the 0.04 (rad/s)^2 tracker-twist
fallback. It is load-bearing: it sets how much authority the IMM gives the CT
hypothesis. It is not tuned to maximise CT selection — straight objects still
produce `omega_est ~ 0` and stay CV.

## Fallback / safety behaviour

`motion_history` can only ever *widen* the set of yaw rates the IMM sees; every
failure path returns to the exact `tracker` behaviour for that frame:

| condition | result |
| --- | --- |
| `yaw_rate_source: tracker` (default) | tracker twist verbatim; identical to pre-change |
| history shorter than `history_samples` | tracker fallback |
| any sample below `min_speed_mps` | tracker fallback |
| duplicate / non-monotonic / >2 s-gap timestamps | tracker fallback |
| non-finite velocity or slope | tracker fallback |
| `|omega_est|` beyond `max_yaw_rate_radps` | clamped to the ceiling |
| straight motion | `omega_est ~ 0`, IMM stays CV |

The diagnostic topic (`/ad/perception/objects/prediction_debug`) reports
`yaw_rate_source_used` (`tracker` / `motion_history` / `tracker_fallback`) and
`yaw_rate_radps` per track so activation is directly measurable.

## Limitations

* Activation is bounded by tracker fragmentation, which this PR does not touch:
  a track must survive `history_samples` consecutive above-speed frames.
* The reported yaw-rate variance is a fixed constant, not derived from the
  fit residual.
* `min_speed_mps` blinds the estimator to slow turners (parked-car pull-outs,
  pedestrians) — deliberate; their velocity heading is noise-dominated.
* No off-road / drivable-mask metric in this PR.
* **Dynamic OGM future-sweep integration remains a separate next task**: the
  predicted trajectory keyframes are now curved, but the Dynamic Occupancy Grid
  still rasterises only each object's current footprint.

## Validation

Unit tests: `test_autoware_prediction_adapter.cpp` (11 new `CurveAwarePrediction`
cases: straight / left turn / right turn / short history / low speed / wraparound
/ clamp / tracker-mode compatibility / parse / config validation, plus the
straight-object parity check). `test_imm_predictor.cpp` unchanged and passing.

Bag A/B (`morai_cam4_20260813_163222`): the captured
`/ad/perception/objects/tracked` stream replayed twice through a standalone
`ad_autoware_prediction_node` (`yaw_rate_source:=tracker` vs `:=motion_history`,
identical input). See the PR description for the before/after table.
