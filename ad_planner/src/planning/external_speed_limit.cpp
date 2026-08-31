#include "ad_planner/planning/external_speed_limit.hpp"

#include <algorithm>
#include <cmath>

namespace ad_planner {

std::optional<double> combine_speed_limits(std::optional<double> a,
                                           std::optional<double> b) {
  const auto usable = [](const std::optional<double> &value) {
    return value.has_value() && std::isfinite(*value);
  };
  if (usable(a) && usable(b)) {
    return std::min(*a, *b);
  }
  if (usable(a)) {
    return a;
  }
  if (usable(b)) {
    return b;
  }
  return std::nullopt;
}

} // namespace ad_planner
