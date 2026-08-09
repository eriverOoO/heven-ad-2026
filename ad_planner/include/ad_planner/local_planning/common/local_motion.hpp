#ifndef AD_PLANNER__LOCAL_PLANNING__LOCAL_MOTION_HPP_
#define AD_PLANNER__LOCAL_PLANNING__LOCAL_MOTION_HPP_

#include <cstdint>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "ad_planner/common/types.hpp"

namespace ad_planner
{

struct TimedTrajectoryPoint
{
  Pose2 pose;
  double time_from_start_s{0.0};
  double speed_mps{0.0};
  double curvature_inv_m{0.0};
};

struct TimedTrajectory
{
  std::string frame_id{"odom"};
  std::vector<TimedTrajectoryPoint> points;
};

struct ReferencePoint
{
  Pose2 pose;
  double route_s_m{0.0};
  double curvature_inv_m{0.0};
  double left_width_m{0.0};
  double right_width_m{0.0};
  double speed_limit_mps{0.0};
};

struct ReferenceLane
{
  std::string lane_sequence_id;
  std::vector<std::string> source_link_ids;
  std::vector<ReferencePoint> points;
  std::vector<std::size_t> left_lane_indices;
  std::vector<std::size_t> right_lane_indices;
};

struct ReferenceCorridor
{
  std::string frame_id{"odom"};
  std::vector<ReferenceLane> lanes;
  std::size_t primary_lane_index{0};
};

struct EgoState
{
  Pose2 pose;
  double speed_mps{0.0};
  double yaw_rate_radps{0.0};
};

struct PredictedFootprint
{
  double time_from_start_s{0.0};
  Pose2 pose;
  double length_m{0.0};
  double width_m{0.0};
  double covariance_xx{0.0};
  double covariance_yy{0.0};
  double covariance_xy{0.0};
};

struct PredictedObject
{
  std::string object_id;
  std::vector<PredictedFootprint> footprints;
};

using PredictedObjectSet = std::vector<PredictedObject>;

struct VehicleConstraints
{
  double wheelbase_m{0.0};
  double maximum_steering_rad{0.0};
  double maximum_speed_mps{0.0};
  double maximum_acceleration_mps2{0.0};
  double maximum_deceleration_mps2{0.0};
  double maximum_lateral_acceleration_mps2{0.0};
  double maximum_jerk_mps3{0.0};
  double footprint_front_m{0.0};
  double footprint_rear_m{0.0};
  double footprint_half_width_m{0.0};
};

struct ExternalVelocityCommand
{
  double vx_mps{0.0};
  double wz_rad_s{0.0};
  std::int64_t receipt_steady_ns{0};
};

struct PlannerCost
{
  std::string name;
  double value{0.0};
};

struct LocalPlanningRequest
{
  ReferenceCorridor reference_corridor;
  EgoState ego;
  OccupancyGrid occupancy_grid;
  std::optional<OccupancyGrid> drivable_mask;
  PredictedObjectSet predicted_objects;
  std::optional<TimedTrajectory> previous_trajectory;
  PhysicalCommand previous_command;
  VehicleConstraints constraints;
  std::int64_t stamp_ns{0};
  std::int64_t steady_time_ns{0};
  double dt_s{0.0};
  int behavior_id{0};
  int gear_id{4};
};

struct LocalPlanningResult
{
  bool valid{false};
  std::string reason{"invalid"};
  TimedTrajectory trajectory;
  std::vector<TimedTrajectory> candidate_trajectories;
  double desired_speed_mps{0.0};
  double desired_curvature_inv_m{0.0};
  std::optional<PhysicalCommand> direct_command;
  std::vector<PlannerCost> costs;
};

class LocalMotionBackend
{
public:
  virtual ~LocalMotionBackend() = default;
  virtual bool observe_external_velocity_command(
    const ExternalVelocityCommand &)
  {
    return false;
  }
  virtual LocalPlanningResult plan(const LocalPlanningRequest & request) = 0;
};

}  // namespace ad_planner

#endif  // AD_PLANNER__LOCAL_PLANNING__LOCAL_MOTION_HPP_
