#include "ad_planner/planning/highway_merge_speed_constraint.hpp"

#include <algorithm>
#include <cmath>

namespace ad_planner {
namespace {

// Mirrors ad_interfaces/msg/HighwayMergeGapResponse.msg. Kept local so the pure
// core has no ROS message dependency, matching the other planning cores.
constexpr int kActionMergeReady = 0;
constexpr int kActionWait = 1;
constexpr int kActionHold = 2;

bool usable_frame(const HighwayMergeResponseConstraintInput &input) {
  return input.received && input.fresh && input.active && input.zone_matches;
}

} // namespace

std::optional<double> highway_merge_response_speed_limit(
    const HighwayMergeResponseConstraintInput &input) {
  if (!usable_frame(input)) {
    return std::nullopt;
  }

  switch (input.action) {
  case kActionMergeReady:
    // MERGE_READY removes only the highway-merge cap; it never raises speed and
    // never overrides another active limit.
    return std::nullopt;
  case kActionHold:
    return 0.0;
  case kActionWait: {
    if (!std::isfinite(input.ego_speed_mps) || input.ego_speed_mps < 0.0 ||
        !std::isfinite(input.available_distance_m) ||
        input.available_distance_m <= 0.0 ||
        !std::isfinite(input.comfortable_stop_distance_m) ||
        input.comfortable_stop_distance_m <= 0.0) {
      // The response classified WAIT, so by its own policy the ego is moving and
      // has more than the comfortable stopping margin to the decision boundary.
      // Facts that contradict that are a malformed frame: no highway-merge
      // constraint rather than a fabricated one.
      return std::nullopt;
    }
    const double cap =
        input.ego_speed_mps * std::sqrt(input.available_distance_m /
                                        input.comfortable_stop_distance_m);
    if (!std::isfinite(cap)) {
      return std::nullopt;
    }
    return std::max(0.0, cap);
  }
  default:
    return std::nullopt;
  }
}

bool highway_merge_response_merge_authorized(
    const HighwayMergeResponseConstraintInput &input) {
  return usable_frame(input) && input.action == kActionMergeReady;
}

} // namespace ad_planner
