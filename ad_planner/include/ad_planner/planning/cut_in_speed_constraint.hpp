#ifndef AD_PLANNER__PLANNING__CUT_IN_SPEED_CONSTRAINT_HPP_
#define AD_PLANNER__PLANNING__CUT_IN_SPEED_CONSTRAINT_HPP_

#include <optional>

namespace ad_planner {

// Backend-agnostic, policy-free translation of one CutInResponse frame into an
// external upper speed limit for the planner's existing longitudinal path. It
// owns no thresholds, no deceleration constants, and no cut-in equations - those
// belong to Cut-in Response. It only decides whether the already-computed
// request should cap the planner's own desired speed this tick.
//
// action mirrors ad_interfaces::msg::CutInResponse::ACTION_* (0 NONE / 1
// SLOWDOWN / 2 HOLD); any other value is treated as "no constraint".
struct CutInResponseConstraintInput {
  bool received{false};
  bool fresh{false};
  bool active{false};
  int action{0};
  bool requested_max_speed_valid{false};
  double requested_max_speed_mps{0.0};
};

// The external maximum speed (m/s) the longitudinal controller should honour
// this tick, or std::nullopt for "no cut-in longitudinal constraint" (the
// planner keeps its own desired speed unchanged).
//
// Guarantees:
//   * the return value is never negative and never NaN/Inf;
//   * a stale, missing, inactive, NONE, or in any way malformed / contradictory
//     response yields std::nullopt - never 0.0 and never a pass-through - so a
//     bad message can only ever be a no-op, never a spurious brake or a speed
//     increase;
//   * SLOWDOWN yields the requested upper bound (strictly positive by contract);
//   * HOLD yields exactly 0.0, and only when the response says so explicitly.
//
// This function never applies the planner's own nominal cruise target; the
// caller must still clamp with std::min against the existing desired speed so
// the constraint can only ever lower it.
std::optional<double>
cut_in_response_speed_limit(const CutInResponseConstraintInput &input);

} // namespace ad_planner

#endif // AD_PLANNER__PLANNING__CUT_IN_SPEED_CONSTRAINT_HPP_
