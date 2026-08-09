# kalman-filter-localization-ros2 overlay

This directory carries the ordered ESKF overlays required by the HEVEN
localization profile.

- Upstream: `https://github.com/rsasaki0109/kalman_filter_localization_ros2.git`
- Base revision: `fc1f4d39c942813ea83dc4f017eb0892756ea94d`
- Patches, in application order:
  1. `0001-large-imu-gap-recovery.patch`
  2. `0002-stationary-accel-bias-initialization.patch`
  3. `0003-wheel-confirmed-zupt.patch`
  4. `0004-gate-preinitialization-output.patch`
- License: the upstream BSD license is preserved in `LICENSE.upstream`.

The first patch classifies an oversized IMU interval without mutating filter state,
starts an isolated replay epoch, gates fresh localization output, and permits a
position-only re-anchor only after multiple time-ordered GNSS fixes are mutually
consistent. Normal GNSS innovation and robust-loss gates remain active outside
that recovery state.

The second patch optionally initializes the gravity-observable accelerometer
bias from the same stationary IMU window used for roll and pitch. The option is
disabled by default upstream. HEVEN production also leaves it disabled after
one short same-topic simultaneous stationary 2x2 A/B found no additional body-z benefit
beyond non-zero initial bias covariance. Three short final-candidate
closed-loop A/B repeats then reduced mean body-z velocity RMSE by `63.3%`
without a repeatable z-position change. The covariance-only configuration
remains a provisional production candidate because bias covariance and
estimator counters are not yet persisted and three-minute mixed driven
validation is still pending. Another deployment may opt in only after
independent sensor validation.

The third patch optionally makes zero-velocity updates fail closed unless the
stationary detector continuously receives fresh wheel-speed samples below a
configured threshold. Upstream-compatible defaults leave this wheel gate
disabled; deployments can opt in with `zupt_require_wheel_speed` when the wheel
speed input and timestamps are trustworthy. The implementation rejects stale,
future-dated, non-finite, or above-threshold evidence and resets the stationary
duration whenever confirmation is lost. It also provides a separately disabled
one-shot standstill mode that reinitializes only the velocity covariance and
clears its cross-covariances before the ordinary Joseph-form ZUPT. The nominal
velocity is not reset directly. `zupt_reinitialize_velocity_covariance` requires
wheel-gated ZUPT, applies once per continuously confirmed stationary episode,
uses `zupt_reinitialization_max_speed_mps` as a false-positive guard, and stores
the decision in the immutable IMU replay plan so rewind follows the same event
timeline. `var_zupt_reinitialized_velocity` is intentionally distinct from the
ZUPT measurement variance.

The fourth patch adds a publication-only stationary-initialization readiness
gate. When stationary initialization is enabled, pose, odometry, and bias
outputs remain suppressed while the initializer collects samples, on the
initializer-completing sample, and while the first post-initialization IMU
seeds replay or direct prediction. Publication becomes ready only after replay
applies a valid propagation or direct prediction returns an updated state.
Profiles with stationary initialization disabled retain first-IMU publication.
Estimator state, covariance, noise, initializer thresholds, and replay math are
unchanged.

Apply it immediately after `vcs import` and before `rosdep install`:

```bash
src/heven_ad_2026/scripts/apply_dependency_patches.sh src
```

The script is fail-closed: it requires the exact base revision, refuses an
unpatched dirty checkout, uses `git apply --check`, and recognizes the fully
applied ordered overlay by reversing it in a private temporary Git index.
