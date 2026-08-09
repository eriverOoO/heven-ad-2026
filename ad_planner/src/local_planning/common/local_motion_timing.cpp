#include "ad_planner/local_planning/common/local_motion_timing.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace ad_planner
{
namespace
{

using WideNanoseconds = __int128;
constexpr long double kNanosecondsPerSecond = 1'000'000'000.0L;

bool seconds_to_nanoseconds(double seconds, std::int64_t & nanoseconds)
{
  if (!std::isfinite(seconds) || !(seconds > 0.0)) {
    return false;
  }
  const long double scaled =
    static_cast<long double>(seconds) * kNanosecondsPerSecond;
  const long double rounded = std::round(scaled);
  if (!std::isfinite(rounded) || rounded < 1.0L ||
    rounded > static_cast<long double>(std::numeric_limits<std::int64_t>::max()))
  {
    return false;
  }
  nanoseconds = static_cast<std::int64_t>(rounded);
  return true;
}

LocalMotionTimingValidation invalid(const char * reason)
{
  return LocalMotionTimingValidation{false, reason};
}

}  // namespace

LocalMotionTimingValidation validate_local_motion_timing(
  std::int64_t now_ns,
  std::int64_t odometry_stamp_ns,
  std::int64_t grid_stamp_ns,
  const LocalMotionTimingLimits & limits)
{
  std::int64_t maximum_odometry_age_ns = 0;
  std::int64_t maximum_grid_age_ns = 0;
  std::int64_t maximum_skew_ns = 0;
  if (!seconds_to_nanoseconds(
      limits.maximum_odometry_age_s, maximum_odometry_age_ns) ||
    !seconds_to_nanoseconds(limits.maximum_grid_age_s, maximum_grid_age_ns) ||
    !seconds_to_nanoseconds(
      limits.maximum_grid_odometry_skew_s, maximum_skew_ns))
  {
    return invalid("timing limits must be finite, positive, and representable");
  }
  if (odometry_stamp_ns <= 0) {
    return invalid("odometry stamp must be positive");
  }
  if (grid_stamp_ns <= 0) {
    return invalid("grid stamp must be positive");
  }

  const WideNanoseconds odometry_age =
    static_cast<WideNanoseconds>(now_ns) -
    static_cast<WideNanoseconds>(odometry_stamp_ns);
  const WideNanoseconds grid_age =
    static_cast<WideNanoseconds>(now_ns) -
    static_cast<WideNanoseconds>(grid_stamp_ns);
  if (odometry_age < 0) {
    return invalid("odometry stamp is in the future");
  }
  if (grid_age < 0) {
    return invalid("grid stamp is in the future");
  }
  if (odometry_age > static_cast<WideNanoseconds>(maximum_odometry_age_ns)) {
    return invalid("odometry stamp is stale");
  }
  if (grid_age > static_cast<WideNanoseconds>(maximum_grid_age_ns)) {
    return invalid("grid stamp is stale");
  }

  WideNanoseconds skew =
    static_cast<WideNanoseconds>(odometry_stamp_ns) -
    static_cast<WideNanoseconds>(grid_stamp_ns);
  if (skew < 0) {
    skew = -skew;
  }
  if (skew > static_cast<WideNanoseconds>(maximum_skew_ns)) {
    return invalid("odometry and grid stamps exceed maximum skew");
  }
  return LocalMotionTimingValidation{true, ""};
}

}  // namespace ad_planner
