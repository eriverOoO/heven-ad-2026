#ifndef AD_PLANNER__LOCAL_PLANNING__OCCUPANCY_GRID_REPROJECTOR_HPP_
#define AD_PLANNER__LOCAL_PLANNING__OCCUPANCY_GRID_REPROJECTOR_HPP_

#include <cstdint>
#include <optional>

#include "ad_planner/common/types.hpp"
#include "ad_planner/local_planning/common/local_motion_frame.hpp"

namespace ad_planner
{

struct Point2
{
  double x{0.0};
  double y{0.0};
};

struct QuaternionComponents
{
  double x{0.0};
  double y{0.0};
  double z{0.0};
  double w{0.0};
};

struct GridReprojectionConfig
{
  double width_m{54.0};
  double height_m{54.0};
  double resolution_m{0.1};
  std::int8_t outside_value{-1};
};

// Returns yaw only for finite, nonzero, near-unit planar quaternions.
std::optional<double> planar_yaw_from_quaternion(
  const QuaternionComponents & quaternion);

// FrameTransform2 follows p_odom = translation + R(yaw) * p_source.
// The output grid is axis-aligned in odom and centered on ego_in_odom.
std::optional<OccupancyGrid> reproject_occupancy_grid(
  const OccupancyGrid & source,
  const FrameTransform2 & odom_from_source,
  const Point2 & ego_in_odom,
  const GridReprojectionConfig & config);

}  // namespace ad_planner

#endif  // AD_PLANNER__LOCAL_PLANNING__OCCUPANCY_GRID_REPROJECTOR_HPP_
