#ifndef AD_PLANNER__LOCAL_PLANNING__LOCAL_MOTION_FRAME_HPP_
#define AD_PLANNER__LOCAL_PLANNING__LOCAL_MOTION_FRAME_HPP_

#include <string>

#include "ad_planner/local_planning/common/local_motion.hpp"

namespace ad_planner
{

struct FrameTransform2
{
  double x_m{0.0};
  double y_m{0.0};
  double yaw_rad{0.0};
};

struct PrimaryRouteProjection
{
  double route_s_m{0.0};
  double lateral_distance_m{0.0};
  double heading_error_rad{0.0};
};

// Returned yaw values are normalized to the [-pi, pi] interval.
Pose2 transform_pose(const FrameTransform2 & transform, const Pose2 & pose);

ReferenceCorridor transform_reference_corridor(
  const FrameTransform2 & transform, const ReferenceCorridor & corridor,
  const std::string & target_frame_id);

OccupancyGrid transform_occupancy_grid_origin(
  const FrameTransform2 & transform, const OccupancyGrid & grid);

PrimaryRouteProjection project_primary_route(
  const ReferenceCorridor & corridor, const Pose2 & ego_pose);

ReferenceCorridor window_reference_corridor(
  const ReferenceCorridor & corridor, const Pose2 & ego_pose,
  double behind_m, double ahead_m);

}  // namespace ad_planner

#endif  // AD_PLANNER__LOCAL_PLANNING__LOCAL_MOTION_FRAME_HPP_
