#ifndef AD_PLANNER__LOCAL_PLANNING__LOCAL_MOTION_VALIDATION_HPP_
#define AD_PLANNER__LOCAL_PLANNING__LOCAL_MOTION_VALIDATION_HPP_

#include <cstdint>
#include <optional>
#include <string>

#include "ad_planner/common/types.hpp"
#include "ad_planner/local_planning/common/local_motion.hpp"

namespace ad_planner
{

std::optional<std::int64_t> valid_ros_stamp_nanoseconds(
  std::int32_t seconds, std::uint32_t nanoseconds);

bool valid_direct_command(
  const PhysicalCommand & command, double maximum_steering_rad);

bool valid_timed_trajectory(
  const TimedTrajectory & trajectory, const std::string & expected_frame);

}  // namespace ad_planner

#endif  // AD_PLANNER__LOCAL_PLANNING__LOCAL_MOTION_VALIDATION_HPP_
