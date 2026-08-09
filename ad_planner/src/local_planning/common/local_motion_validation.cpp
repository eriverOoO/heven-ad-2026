#include "ad_planner/local_planning/common/local_motion_validation.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace ad_planner
{

std::optional<std::int64_t> valid_ros_stamp_nanoseconds(
  const std::int32_t seconds, const std::uint32_t nanoseconds)
{
  constexpr std::uint32_t kNanosecondsPerSecond = 1'000'000'000U;
  if (seconds < 0 || nanoseconds >= kNanosecondsPerSecond ||
    (seconds == 0 && nanoseconds == 0U))
  {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(seconds) * kNanosecondsPerSecond +
         static_cast<std::int64_t>(nanoseconds);
}

bool valid_direct_command(
  const PhysicalCommand & command, const double maximum_steering_rad)
{
  return std::isfinite(maximum_steering_rad) && maximum_steering_rad > 0.0 &&
         std::isfinite(command.accel) && std::isfinite(command.brake) &&
         std::isfinite(command.steering_rad) &&
         command.accel >= 0.0 && command.accel <= 1.0 &&
         command.brake >= 0.0 && command.brake <= 1.0 &&
         !(command.accel > 0.0 && command.brake > 0.0) &&
         std::abs(command.steering_rad) <= maximum_steering_rad;
}

bool valid_timed_trajectory(
  const TimedTrajectory & trajectory, const std::string & expected_frame)
{
  if (trajectory.frame_id != expected_frame || trajectory.points.empty()) {
    return false;
  }
  double previous_time_s = -std::numeric_limits<double>::infinity();
  for (const auto & point : trajectory.points) {
    if (!std::isfinite(point.pose.x) || !std::isfinite(point.pose.y) ||
      !std::isfinite(point.pose.yaw_rad) ||
      !std::isfinite(point.time_from_start_s) ||
      !std::isfinite(point.speed_mps) ||
      !std::isfinite(point.curvature_inv_m) ||
      !(point.time_from_start_s > previous_time_s))
    {
      return false;
    }
    previous_time_s = point.time_from_start_s;
  }
  return true;
}

}  // namespace ad_planner
