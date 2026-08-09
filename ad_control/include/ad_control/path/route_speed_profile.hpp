#ifndef AD_CONTROL__PATH__ROUTE_SPEED_PROFILE_HPP_
#define AD_CONTROL__PATH__ROUTE_SPEED_PROFILE_HPP_

#include <cstddef>
#include <vector>

#include "ad_control/common/types.hpp"

namespace ad_control
{

struct LongitudinalProfile
{
  std::vector<double> speed_mps;
  std::vector<double> acceleration_mps2;
  std::vector<double> deceleration_mps2;
  double braking_delay_s{0.0};

  bool empty() const noexcept
  {
    return speed_mps.empty() && acceleration_mps2.empty() &&
           deceleration_mps2.empty();
  }
};

struct RouteSpeedZone
{
  Point3 start;
  Point3 end;
  double maximum_speed_mps;
  double maximum_projection_distance_m{5.0};
};

struct RouteSpeedProfileConfig
{
  double minimum_speed_mps;
  double maximum_speed_mps;
  double lateral_acceleration_mps2;
  double acceleration_mps2;
  double deceleration_mps2;
  std::size_t curvature_window_radius;
  double curvature_lookahead_m{1.0};
  LongitudinalProfile longitudinal_profile;
  std::vector<RouteSpeedZone> speed_zones;
};

struct RouteSpeedProfile
{
  std::vector<double> speed_mps;
  std::vector<double> curvature_inv_m;
  bool uses_longitudinal_profile{false};
};

double longitudinal_acceleration_limit(
  const RouteSpeedProfileConfig & config, double speed_mps, bool braking);

RouteSpeedProfile build_route_speed_profile(
  const Route & route, const RouteSpeedProfileConfig & config);

std::vector<double> build_route_speed_profile(
  const Route & route,
  double lateral_acceleration_mps2);

}  // namespace ad_control

#endif  // AD_CONTROL__PATH__ROUTE_SPEED_PROFILE_HPP_
