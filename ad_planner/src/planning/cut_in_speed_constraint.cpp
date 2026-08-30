#include "ad_planner/planning/cut_in_speed_constraint.hpp"

#include <cmath>

namespace ad_planner {
namespace {

// Mirrors ad_interfaces/msg/CutInResponse.msg. Kept local so the pure core has
// no ROS message dependency, matching the other planning cores.
constexpr int kActionNone = 0;
constexpr int kActionSlowdown = 1;
constexpr int kActionHold = 2;

} // namespace

std::optional<double>
cut_in_response_speed_limit(const CutInResponseConstraintInput &input) {
  if (!input.received || !input.fresh || !input.active) {
    return std::nullopt;
  }
  if (!input.requested_max_speed_valid ||
      !std::isfinite(input.requested_max_speed_mps)) {
    return std::nullopt;
  }

  switch (input.action) {
  case kActionSlowdown:
    if (input.requested_max_speed_mps <= 0.0) {
      // SLOWDOWN below the current speed but not a full hold; a non-positive
      // value contradicts the contract and is rejected rather than clamped.
      return std::nullopt;
    }
    return input.requested_max_speed_mps;
  case kActionHold:
    if (input.requested_max_speed_mps != 0.0) {
      return std::nullopt;
    }
    return 0.0;
  case kActionNone:
  default:
    return std::nullopt;
  }
}

} // namespace ad_planner
