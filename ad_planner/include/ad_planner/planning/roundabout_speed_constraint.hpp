#ifndef AD_PLANNER__PLANNING__ROUNDABOUT_SPEED_CONSTRAINT_HPP_
#define AD_PLANNER__PLANNING__ROUNDABOUT_SPEED_CONSTRAINT_HPP_

#include <optional>

namespace ad_planner {

// Backend-agnostic, policy-free translation of one RoundaboutGapResponse frame
// into an external upper speed limit for the planner's existing longitudinal
// path. It owns no thresholds, no deceleration constants, and no roundabout
// policy - those belong to Roundabout Gap Response. It only maps the already
// classified advisory (RELEASE / YIELD / HOLD) onto a speed cap this tick.
//
// action mirrors ad_interfaces::msg::RoundaboutGapResponse::ACTION_* (0 RELEASE
// / 1 YIELD / 2 HOLD); any other value is treated as "no constraint".
//
// The YIELD cap is derived entirely from the response's own published policy
// facts, so no constant is duplicated in the planner. The response reports
//   comfortable_stop_distance_m = ego_speed_mps^2 / (2 * a_comfortable)
// where a_comfortable is its own canonical comfortable deceleration. Therefore
//   ego_speed_mps * sqrt(available_distance_m / comfortable_stop_distance_m)
//     == sqrt(2 * a_comfortable * available_distance_m)
// (the ego_speed_mps terms cancel exactly). That is the maximum speed from
// which the ego can still stop comfortably over the remaining pre-entry
// distance under the same constant-deceleration envelope the response used. As
// the ego approaches, available_distance_m shrinks and the cap shrinks with it;
// once the comfortable margin is consumed the response classifies HOLD and the
// cap becomes 0.
struct RoundaboutResponseConstraintInput {
  bool received{false};
  bool fresh{false};
  bool active{false};
  int action{0};
  double ego_speed_mps{0.0};
  double available_distance_m{0.0};
  double comfortable_stop_distance_m{0.0};
};

// The external maximum speed (m/s) the longitudinal controller should honour
// this tick for the roundabout advisory, or std::nullopt for "no roundabout
// longitudinal constraint" (the planner keeps its own desired speed unchanged).
//
// Guarantees:
//   * the return value is never negative and never NaN/Inf;
//   * a missing, stale, inactive, RELEASE, or in any way malformed / contradictory
//     response yields std::nullopt - never 0.0 and never a pass-through - so a
//     bad message can only ever be a no-op, never a spurious brake and never a
//     speed increase;
//   * HOLD yields exactly 0.0;
//   * YIELD yields the comfortable-stop speed cap described above.
//
// This function never applies the planner's own nominal cruise target; the
// caller must still clamp with std::min against the existing desired speed so
// the constraint can only ever lower it.
std::optional<double> roundabout_response_speed_limit(
    const RoundaboutResponseConstraintInput &input);

} // namespace ad_planner

#endif // AD_PLANNER__PLANNING__ROUNDABOUT_SPEED_CONSTRAINT_HPP_
