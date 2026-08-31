#include "ad_planner/planning/roundabout_speed_constraint.hpp"

#include <algorithm>
#include <cmath>

namespace ad_planner {
namespace {

// Mirrors ad_interfaces/msg/RoundaboutGapResponse.msg. Kept local so the pure
// core has no ROS message dependency, matching the other planning cores.
constexpr int kActionRelease = 0;
constexpr int kActionYield = 1;
constexpr int kActionHold = 2;

} // namespace

std::optional<double> roundabout_response_speed_limit(
    const RoundaboutResponseConstraintInput &input) {
  if (!input.received || !input.fresh || !input.active) {
    return std::nullopt;
  }

  switch (input.action) {
  case kActionRelease:
    // RELEASE removes only the roundabout constraint; it never raises speed and
    // never overrides another active limit.
    return std::nullopt;
  case kActionHold:
    return 0.0;
  case kActionYield: {
    if (!std::isfinite(input.ego_speed_mps) || input.ego_speed_mps < 0.0 ||
        !std::isfinite(input.available_distance_m) ||
        input.available_distance_m <= 0.0 ||
        !std::isfinite(input.comfortable_stop_distance_m) ||
        input.comfortable_stop_distance_m <= 0.0) {
      // The response classified YIELD, so by its own policy the ego is moving
      // and has room to stop comfortably. Facts that contradict that are a
      // malformed frame: no roundabout constraint rather than a fabricated one.
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

} // namespace ad_planner
