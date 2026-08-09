#include "ad_control/lateral/stanley.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

#include "ad_control/path/route_speed_profile.hpp"

namespace ad_control
{
namespace
{
constexpr long double kPi = 3.14159265358979323846L;
constexpr double kSteeringRateRadPerSec = 120.0 * 3.14159265358979323846 / 180.0;
constexpr double kJumpReseedDistanceM = 20.0;
constexpr long double kWrongWayHeadingErrorRad = 2.0L * kPi / 3.0L;

bool finite_pose(const Pose2 & pose)
{
  return std::isfinite(pose.x) && std::isfinite(pose.y) && std::isfinite(pose.yaw_rad);
}

bool checked_difference(double lhs, double rhs, long double & difference)
{
  difference = static_cast<long double>(lhs) - static_cast<long double>(rhs);
  return std::isfinite(difference) &&
         std::abs(difference) <=
         static_cast<long double>(std::numeric_limits<double>::max());
}

long double distance_xy(const Pose2 & lhs, const Pose2 & rhs)
{
  return std::hypot(
    static_cast<long double>(lhs.x) - static_cast<long double>(rhs.x),
    static_cast<long double>(lhs.y) - static_cast<long double>(rhs.y));
}

std::pair<std::size_t, std::size_t> path_segment(
  const Route & route, std::size_t index)
{
  if (index + 1 < route.points.size()) {
    return {index, index + 1};
  }
  if (route.closed) {
    return {index, 0};
  }
  return {index - 1, index};
}

StanleyConfig with_longitudinal_profile(ProfileStanleyConfig config)
{
  config.stanley.speed_profile.longitudinal_profile =
    std::move(config.longitudinal_profile);
  return std::move(config.stanley);
}
}  // namespace

StanleyController::StanleyController(Route route, StanleyConfig config)
: route_(std::move(route)), config_(config),
  progress_(route_, config_.forward_window, config_.max_laps), pid_(config_.pid)
{
  if (!std::isfinite(config_.target_speed_mps) || config_.target_speed_mps < 0.0 ||
    !std::isfinite(config_.cross_track_gain) || config_.cross_track_gain < 0.0 ||
    !std::isfinite(config_.speed_softening_mps) || config_.speed_softening_mps <= 0.0 ||
    !std::isfinite(config_.lookahead_time_s) || config_.lookahead_time_s < 0.0 ||
    !std::isfinite(config_.lookahead_min_m) || config_.lookahead_min_m < 0.0 ||
    !std::isfinite(config_.lookahead_max_m) ||
    config_.lookahead_max_m < config_.lookahead_min_m ||
    !std::isfinite(config_.control_point_x_m) ||
    !std::isfinite(config_.heading_error_gain) ||
    config_.heading_error_gain < 0.0 ||
    !std::isfinite(config_.launch_speed_mps) || config_.launch_speed_mps < 0.0 ||
    !std::isfinite(config_.launch_ramp_s) || config_.launch_ramp_s <= 0.0 ||
    !std::isfinite(config_.max_steer_rad) || config_.max_steer_rad <= 0.0)
  {
    throw std::invalid_argument("Stanley configuration must be finite and physically valid");
  }
  route_speed_profile_ = build_route_speed_profile(route_, config_.speed_profile);
}

StanleyController::StanleyController(Route route, ProfileStanleyConfig config)
: StanleyController(std::move(route), with_longitudinal_profile(std::move(config))) {}

ProfileStanleyController::ProfileStanleyController(
  Route route, ProfileStanleyConfig config)
: StanleyController(std::move(route), std::move(config))
{
}

ControllerResult StanleyController::update(
  const Pose2 & pose, double speed_mps, double dt, int behavior_id, int gear_id,
  std::optional<double> target_speed_mps)
{
  const double selected_target_speed = target_speed_mps.value_or(config_.target_speed_mps);
  if (!finite_pose(pose) || !std::isfinite(speed_mps) || speed_mps < 0.0 ||
    !std::isfinite(dt) || dt <= 0.0 || !std::isfinite(selected_target_speed) ||
    selected_target_speed < 0.0)
  {
    return ControllerResult{};
  }

  const Pose2 control_pose{
    pose.x + config_.control_point_x_m * std::cos(pose.yaw_rad),
    pose.y + config_.control_point_x_m * std::sin(pose.yaw_rad),
    pose.yaw_rad};
  if (!finite_pose(control_pose)) {
    return ControllerResult{};
  }

  RouteProgressTracker next_progress = progress_;
  PidController next_pid = pid_;
  double next_steering = previous_steering_rad_;
  double next_launch_elapsed = launch_elapsed_s_;
  bool reseed = previous_control_pose_.has_value() &&
    distance_xy(control_pose, *previous_control_pose_) > kJumpReseedDistanceM;
  if (reseed) {
    next_progress = RouteProgressTracker(
      route_, config_.forward_window, config_.max_laps);
    next_pid = PidController(config_.pid);
    next_steering = 0.0;
    next_launch_elapsed = 0.0;
  }

  const Point3 progress_position{control_pose.x, control_pose.y, 0.0};
  RouteProgressState progress =
    next_progress.update(progress_position, control_pose.yaw_rad);
  auto route_distance = [&]() {
      const Point3 & nearest_point = route_.points[progress.target_index];
      return std::hypot(
        static_cast<long double>(control_pose.x) - nearest_point.x,
        static_cast<long double>(control_pose.y) - nearest_point.y);
    };
  long double distance_from_route = route_distance();
  if (!std::isfinite(distance_from_route) ||
    distance_from_route >
    static_cast<long double>(std::numeric_limits<double>::max()))
  {
    ControllerResult result;
    result.reason = "nonrepresentable route localization distance";
    return result;
  }
  if (!reseed && distance_from_route > kJumpReseedDistanceM) {
    next_progress = RouteProgressTracker(
      route_, config_.forward_window, config_.max_laps);
    next_pid = PidController(config_.pid);
    next_steering = 0.0;
    next_launch_elapsed = 0.0;
    progress = next_progress.update(progress_position, control_pose.yaw_rad);
    distance_from_route = route_distance();
  }
  if (!std::isfinite(distance_from_route) ||
    distance_from_route >
    static_cast<long double>(std::numeric_limits<double>::max()))
  {
    ControllerResult result;
    result.reason = "nonrepresentable route localization distance";
    return result;
  }
  if (distance_from_route > kJumpReseedDistanceM) {
    ControllerResult result;
    result.valid = true;
    result.reason = "route localization mismatch";
    previous_control_pose_ = control_pose;
    last_valid_ = result;
    return result;
  }
  if (progress.terminal_full_brake) {
    ControllerResult result;
    result.valid = true;
    result.reason = "route terminal";
    result.target_speed_mps = 0.0;
    progress_ = std::move(next_progress);
    previous_control_pose_ = control_pose;
    last_valid_ = result;
    return result;
  }

  const std::size_t nearest_index = progress.target_index;
  std::size_t target_index = nearest_index;
  const double lookahead_m = std::clamp(
    speed_mps * config_.lookahead_time_s,
    config_.lookahead_min_m,
    config_.lookahead_max_m);
  long double accumulated = 0.0L;
  std::size_t preview_steps = 0;
  const std::size_t maximum_preview_steps = route_.closed ?
    route_.points.size() : route_.points.size() - 1 - target_index;
  while (preview_steps < maximum_preview_steps &&
    accumulated < static_cast<long double>(lookahead_m))
  {
    const Point3 & current = route_.points[target_index];
    const std::size_t next_index = (target_index + 1) % route_.points.size();
    const Point3 & next = route_.points[next_index];
    long double dx = 0.0L;
    long double dy = 0.0L;
    if (!checked_difference(next.x, current.x, dx) ||
      !checked_difference(next.y, current.y, dy))
    {
      ControllerResult result;
      result.reason = "nonrepresentable Stanley lookahead geometry";
      return result;
    }
    accumulated += std::hypot(dx, dy);
    if (!std::isfinite(accumulated)) {
      ControllerResult result;
      result.reason = "non-finite Stanley lookahead";
      return result;
    }
    target_index = next_index;
    ++preview_steps;
  }
  const auto [segment_start, segment_end] = path_segment(route_, target_index);
  const Point3 & start = route_.points[segment_start];
  const Point3 & end = route_.points[segment_end];
  long double segment_dx = 0.0L;
  long double segment_dy = 0.0L;
  if (!checked_difference(end.x, start.x, segment_dx) ||
    !checked_difference(end.y, start.y, segment_dy))
  {
    ControllerResult result;
    result.reason = "nonrepresentable Stanley segment geometry";
    return result;
  }
  const long double path_heading = std::atan2(segment_dy, segment_dx);
  const auto [nearest_start, nearest_end] = path_segment(route_, nearest_index);
  const Point3 & cte_start = route_.points[nearest_start];
  const Point3 & cte_end = route_.points[nearest_end];
  long double cte_dx = 0.0L;
  long double cte_dy = 0.0L;
  long double pose_dx = 0.0L;
  long double pose_dy = 0.0L;
  if (!checked_difference(cte_end.x, cte_start.x, cte_dx) ||
    !checked_difference(cte_end.y, cte_start.y, cte_dy) ||
    !checked_difference(control_pose.x, cte_start.x, pose_dx) ||
    !checked_difference(control_pose.y, cte_start.y, pose_dy))
  {
    ControllerResult result;
    result.reason = "nonrepresentable Stanley cross-track geometry";
    return result;
  }
  const long double cte_segment_length = std::hypot(cte_dx, cte_dy);
  if (cte_segment_length <= 0.0L || !std::isfinite(cte_segment_length)) {
    ControllerResult result;
    result.reason = "invalid Stanley cross-track segment";
    return result;
  }
  const long double nearest_path_heading = std::atan2(cte_dy, cte_dx);
  const long double nearest_heading_error = std::remainder(
    nearest_path_heading - static_cast<long double>(control_pose.yaw_rad),
    2.0L * kPi);
  if (std::abs(nearest_heading_error) > kWrongWayHeadingErrorRad) {
    ControllerResult result;
    result.valid = true;
    result.reason = "route heading mismatch";
    previous_control_pose_ = control_pose;
    last_valid_ = result;
    return result;
  }
  const long double heading_error = std::remainder(
    path_heading - static_cast<long double>(control_pose.yaw_rad), 2.0L * kPi);
  const long double cross_track_error =
    (cte_dy * pose_dx - cte_dx * pose_dy) / cte_segment_length;
  const long double raw_steering =
    static_cast<long double>(config_.heading_error_gain) * heading_error +
    std::atan2(
    static_cast<long double>(config_.cross_track_gain) * cross_track_error,
    static_cast<long double>(speed_mps) + config_.speed_softening_mps);
  if (!std::isfinite(path_heading) || !std::isfinite(heading_error) ||
    !std::isfinite(cross_track_error) || !std::isfinite(raw_steering))
  {
    ControllerResult result;
    result.reason = "non-finite Stanley calculation";
    return result;
  }

  const long double physical_steering = std::clamp(
    raw_steering, -static_cast<long double>(config_.max_steer_rad),
    static_cast<long double>(config_.max_steer_rad));
  const double maximum_steering_change = kSteeringRateRadPerSec * dt;
  const double steering = std::clamp(
    static_cast<double>(physical_steering),
    next_steering - maximum_steering_change,
    next_steering + maximum_steering_change);

  const double ramp_fraction =
    std::min(next_launch_elapsed / config_.launch_ramp_s, 1.0);
  const double launch_target =
    config_.launch_speed_mps +
    (selected_target_speed - config_.launch_speed_mps) * ramp_fraction;
  double governed_target_speed = std::min(selected_target_speed, launch_target);
  governed_target_speed = std::min(
    governed_target_speed, route_speed_profile_.speed_mps[nearest_index]);
  ControllerResult result = next_pid.update(
    speed_mps, governed_target_speed, dt, behavior_id, gear_id);
  if (!result.valid) {
    return ControllerResult{};
  }
  result.command.steering_rad = steering;
  result.target = route_.points[target_index];
  result.target_speed_mps = governed_target_speed;
  result.reason = "ok";
  progress_ = std::move(next_progress);
  pid_ = std::move(next_pid);
  previous_steering_rad_ = steering;
  launch_elapsed_s_ = next_launch_elapsed + dt;
  previous_control_pose_ = control_pose;
  last_valid_ = result;
  return result;
}

const RouteProgressState & StanleyController::progress_state() const noexcept
{
  return progress_.state();
}
const PidState & StanleyController::pid_state() const noexcept {return pid_.state();}
const ControllerResult & StanleyController::last_valid_result() const noexcept
{
  return last_valid_;
}
const RouteSpeedProfile * StanleyController::route_speed_profile() const noexcept
{
  return &route_speed_profile_;
}
}  // namespace ad_control
