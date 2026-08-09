#include "ad_planner/local_planning/dwa/dwa_backend.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ad_planner
{
namespace
{

TimedTrajectory timed_trajectory(
  const Trajectory & trajectory, const Pose2 & initial_pose,
  const std::string & frame_id, double dt_s)
{
  TimedTrajectory result;
  result.frame_id = frame_id;
  Pose2 previous_pose = initial_pose;
  result.points.reserve(trajectory.poses.size());
  for (std::size_t index = 0; index < trajectory.poses.size(); ++index) {
    const auto & pose = trajectory.poses[index];
    const double distance_m = std::hypot(
      pose.x - previous_pose.x, pose.y - previous_pose.y);
    const double speed_mps = distance_m / dt_s;
    const double curvature_inv_m = distance_m == 0.0 ? 0.0 :
      (pose.yaw_rad - previous_pose.yaw_rad) / distance_m;
    result.points.push_back(TimedTrajectoryPoint{
      pose, dt_s * static_cast<double>(index + 1U),
      speed_mps, curvature_inv_m});
    previous_pose = pose;
  }
  return result;
}

}  // namespace

DwaBackend::DwaBackend(DwaConfig config)
: controller_(config), dt_s_(config.dt)
{
}

LocalPlanningResult DwaBackend::plan(const LocalPlanningRequest & request)
{
  if (request.reference_corridor.lanes.empty() ||
    request.reference_corridor.primary_lane_index >=
    request.reference_corridor.lanes.size())
  {
    return LocalPlanningResult{};
  }
  const auto & lane = request.reference_corridor.lanes.at(
    request.reference_corridor.primary_lane_index);
  if (lane.points.empty()) {
    return LocalPlanningResult{};
  }
  const auto & target_pose = lane.points.back().pose;
  std::vector<Point3> reference_path;
  reference_path.reserve(lane.points.size());
  for (const auto & point : lane.points) {
    reference_path.push_back(Point3{
      point.pose.x, point.pose.y, 0.0});
  }
  try {
    controller_.apply_vehicle_constraints(request.constraints);
  } catch (const std::invalid_argument & error) {
    LocalPlanningResult result;
    result.reason = error.what();
    return result;
  }
  const auto dwa_result = controller_.plan_with_reference(
    request.occupancy_grid, request.ego.pose,
    Point3{target_pose.x, target_pose.y, 0.0},
    reference_path,
    request.ego.speed_mps, request.previous_command.steering_rad,
    request.behavior_id, request.gear_id,
    request.drivable_mask ? &*request.drivable_mask : nullptr,
    &request.predicted_objects);
  if (!dwa_result.valid || !dwa_result.local_trajectory) {
    LocalPlanningResult result;
    result.reason = dwa_result.reason;
    return result;
  }

  LocalPlanningResult result;
  result.valid = true;
  result.reason = dwa_result.reason;
  result.direct_command = dwa_result.command;
  result.trajectory = timed_trajectory(
    *dwa_result.local_trajectory, request.ego.pose,
    request.reference_corridor.frame_id, dt_s_);
  result.candidate_trajectories.reserve(
    controller_.candidate_trajectories().size());
  for (const auto & candidate : controller_.candidate_trajectories()) {
    result.candidate_trajectories.push_back(timed_trajectory(
        candidate, request.ego.pose,
        request.reference_corridor.frame_id, dt_s_));
  }
  if (!result.trajectory.points.empty()) {
    result.desired_speed_mps =
      dwa_result.target_speed_mps.value_or(
      result.trajectory.points.front().speed_mps);
    result.desired_curvature_inv_m = result.trajectory.points.front().curvature_inv_m;
  }
  result.costs = {
    PlannerCost{"candidate_count", static_cast<double>(
        result.candidate_trajectories.size())}};
  return result;
}

}  // namespace ad_planner
