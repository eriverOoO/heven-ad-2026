#ifndef AD_PLANNER__COMMON__VEHICLE_OBSERVATION_HPP_
#define AD_PLANNER__COMMON__VEHICLE_OBSERVATION_HPP_

#include <cmath>
#include <optional>

namespace ad_planner
{

inline std::optional<double> validated_speed_mps(double signed_velocity_mps)
{
  if (!std::isfinite(signed_velocity_mps)) {
    return std::nullopt;
  }
  return std::abs(signed_velocity_mps);
}

}  // namespace ad_planner

#endif  // AD_PLANNER__COMMON__VEHICLE_OBSERVATION_HPP_
