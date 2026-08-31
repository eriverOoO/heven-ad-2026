#ifndef AD_PLANNER__PLANNING__HIGHWAY_MERGE_SPEED_CONSTRAINT_HPP_
#define AD_PLANNER__PLANNING__HIGHWAY_MERGE_SPEED_CONSTRAINT_HPP_

#include <optional>

namespace ad_planner {

// Backend-agnostic, policy-free translation of one HighwayMergeGapResponse frame
// into (a) an external upper speed limit for the planner's existing longitudinal
// path and (b) a revocable merge-authorization fact for a future mission layer.
//
// It owns no thresholds, no deceleration constants, and no merge policy - those
// belong to Highway Merge Gap Response. It only maps the already classified
// advisory (MERGE_READY / WAIT / HOLD) onto this tick's speed cap and a boolean
// authorization. Policy is never re-decided here: the pair (active, action) is
// the whole input; `reason` is diagnostic and is never branched on (an inactive
// frame carries REASON_NONE = 0, the same enum-0 trap as ACTION_MERGE_READY = 0).
//
// action mirrors ad_interfaces::msg::HighwayMergeGapResponse::ACTION_* (0
// MERGE_READY / 1 WAIT / 2 HOLD); any other value is treated as "no constraint"
// and "not authorized".
//
// zone_matches is computed by the caller: the response carries a merge_zone_id
// string and the consumer knows which zone it is integrating. A response for a
// different zone is not this consumer's message - it can neither authorize a
// merge nor contribute a speed cap (the upstream response node already rejects a
// wrong-zone risk frame wholesale; this mirrors that for the wrong-zone response
// case).
//
// The WAIT cap is derived entirely from the response's own published policy
// facts, so no constant is duplicated in the planner. The response reports
//   comfortable_stop_distance_m = ego_speed_mps^2 / (2 * a_comfortable)
// where a_comfortable is its own canonical comfortable deceleration. Therefore
//   ego_speed_mps * sqrt(available_distance_m / comfortable_stop_distance_m)
//     == sqrt(2 * a_comfortable * available_distance_m)
// (the ego_speed_mps terms cancel exactly). That is the maximum speed from which
// the ego can still stop comfortably over the remaining pre-decision-boundary
// distance under the same constant-deceleration envelope the response used. As
// the ego approaches, available_distance_m shrinks and the cap shrinks with it;
// once the comfortable margin is consumed the response classifies HOLD and the
// cap becomes 0.
struct HighwayMergeResponseConstraintInput {
  bool received{false};
  bool fresh{false};
  bool active{false};
  bool zone_matches{false};
  int action{0};
  double ego_speed_mps{0.0};
  double available_distance_m{0.0};
  double comfortable_stop_distance_m{0.0};
};

// The external maximum speed (m/s) the longitudinal controller should honour
// this tick for the highway-merge advisory, or std::nullopt for "no
// highway-merge longitudinal constraint" (the planner keeps its own desired
// speed unchanged).
//
// Guarantees:
//   * the return value is never negative and never NaN/Inf;
//   * a missing, stale, inactive, wrong-zone, MERGE_READY, or in any way
//     malformed / contradictory response yields std::nullopt - never 0.0 and
//     never a pass-through - so a bad message can only ever be a no-op, never a
//     spurious brake and never a speed increase;
//   * HOLD yields exactly 0.0;
//   * WAIT yields the comfortable-stop speed cap described above.
//
// This function never applies the planner's own nominal cruise target; the
// caller must still clamp with std::min against the existing desired speed so
// the constraint can only ever lower it.
std::optional<double> highway_merge_response_speed_limit(
    const HighwayMergeResponseConstraintInput &input);

// True only when a fresh, active, matching-zone response classifies
// ACTION_MERGE_READY. ACTION_MERGE_READY = 0 is the enum default, so `active`
// must be checked: an inactive frame also carries the non-restrictive
// MERGE_READY enum and must never authorize a merge. This is a fact for a future
// mission layer; it is never latched and is recomputed from the freshest
// response every tick, so a following WAIT / HOLD / stale / missing / inactive
// / wrong-zone frame immediately revokes it.
bool highway_merge_response_merge_authorized(
    const HighwayMergeResponseConstraintInput &input);

} // namespace ad_planner

#endif // AD_PLANNER__PLANNING__HIGHWAY_MERGE_SPEED_CONSTRAINT_HPP_
