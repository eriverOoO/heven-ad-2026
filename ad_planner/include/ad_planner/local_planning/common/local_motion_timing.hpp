#ifndef AD_PLANNER__LOCAL_PLANNING__LOCAL_MOTION_TIMING_HPP_
#define AD_PLANNER__LOCAL_PLANNING__LOCAL_MOTION_TIMING_HPP_

#include <cstdint>
#include <string>

namespace ad_planner
{

struct LocalMotionTimingLimits
{
  double maximum_odometry_age_s{0.50};
  double maximum_grid_age_s{0.50};
  double maximum_grid_odometry_skew_s{0.10};
};

struct LocalMotionTimingValidation
{
  bool valid{false};
  std::string reason;
};

LocalMotionTimingValidation validate_local_motion_timing(
  std::int64_t now_ns,
  std::int64_t odometry_stamp_ns,
  std::int64_t grid_stamp_ns,
  const LocalMotionTimingLimits & limits);

}  // namespace ad_planner

#endif  // AD_PLANNER__LOCAL_PLANNING__LOCAL_MOTION_TIMING_HPP_
