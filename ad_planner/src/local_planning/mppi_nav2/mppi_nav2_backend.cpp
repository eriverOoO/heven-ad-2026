#include "ad_planner/local_planning/mppi_nav2/mppi_nav2_backend.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "ad_control/command/command_adapter.hpp"
#include "ad_planner/local_planning/common/local_motion_validation.hpp"

namespace ad_planner
{
namespace
{

constexpr long double kNanosecondsPerSecond = 1'000'000'000.0L;
constexpr long double kPi = 3.1415926535897932384626433832795L;
constexpr long double kTwoPi = 2.0L * kPi;
constexpr std::size_t kMaximumRolloutPoints = 10'001U;

LocalPlanningResult invalid_result(const std::string & reason)
{
  LocalPlanningResult result;
  result.reason = reason;
  return result;
}

bool finite_positive(const double value)
{
  return std::isfinite(value) && value > 0.0;
}

bool approximately_equal(const double lhs, const double rhs)
{
  const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
  const double tolerance =
    64.0 * std::numeric_limits<double>::epsilon() * scale;
  return std::abs(lhs - rhs) <= tolerance;
}

std::optional<std::string> request_validation_error(
  const LocalPlanningRequest & request,
  const MppiNav2BackendConfig & config)
{
  if (!std::isfinite(request.ego.pose.x) ||
    !std::isfinite(request.ego.pose.y) ||
    !std::isfinite(request.ego.pose.yaw_rad) ||
    !std::isfinite(request.ego.speed_mps) ||
    !std::isfinite(request.ego.yaw_rate_radps) ||
    request.ego.speed_mps < 0.0)
  {
    return "mppi request ego state is invalid";
  }
  if (request.stamp_ns <= 0) {
    return "mppi request ROS stamp is invalid";
  }
  if (!finite_positive(request.dt_s)) {
    return "mppi request dt is invalid";
  }

  const auto & constraints = request.constraints;
  const std::array<double, 9> positive_constraints{
    constraints.wheelbase_m,
    constraints.maximum_steering_rad,
    constraints.maximum_speed_mps,
    constraints.maximum_acceleration_mps2,
    constraints.maximum_deceleration_mps2,
    constraints.maximum_lateral_acceleration_mps2,
    constraints.maximum_jerk_mps3,
    constraints.footprint_front_m,
    constraints.footprint_rear_m};
  if (std::any_of(
      positive_constraints.begin(), positive_constraints.end(),
    [](const double value) {return !finite_positive(value);}) ||
    !finite_positive(constraints.footprint_half_width_m) ||
    !(constraints.maximum_steering_rad < static_cast<double>(kPi / 2.0L)))
  {
    return "mppi request constraints are invalid";
  }
  if (request.ego.speed_mps > constraints.maximum_speed_mps) {
    return "mppi ego speed exceeds maximum speed";
  }
  if (!valid_direct_command(
      request.previous_command, constraints.maximum_steering_rad))
  {
    return "mppi previous command is invalid";
  }
  if (request.behavior_id < 0) {
    return "mppi behavior id is invalid";
  }
  if (request.gear_id != ad_control::kGearDriveCode) {
    return "mppi requires acknowledged drive gear";
  }
  if (!approximately_equal(
      config.command.wheelbase_m, constraints.wheelbase_m))
  {
    return "mppi adapter wheelbase does not match request constraint";
  }
  if (config.command.maximum_road_wheel_angle_rad >
    constraints.maximum_steering_rad)
  {
    return "mppi adapter steering limit exceeds request constraint";
  }
  if (std::abs(request.previous_command.steering_rad) >
    config.command.maximum_road_wheel_angle_rad)
  {
    return "mppi previous steering exceeds adapter limit";
  }
  const long double steering_delta =
    static_cast<long double>(config.command.steering_rate_limit_rad_s) *
    static_cast<long double>(request.dt_s);
  if (!std::isfinite(steering_delta) ||
    std::abs(steering_delta) >
    static_cast<long double>(std::numeric_limits<double>::max()))
  {
    return "mppi steering rollout arithmetic is invalid";
  }
  return std::nullopt;
}

bool representable_double(const long double value)
{
  return std::isfinite(value) &&
         std::abs(value) <=
         static_cast<long double>(std::numeric_limits<double>::max());
}

std::optional<TimedTrajectory> make_command_rollout(
  const LocalPlanningRequest & request,
  const ad_control::MppiVelocityTarget & target,
  const double rollout_dt_s,
  const double rollout_horizon_s,
  const std::size_t rollout_step_count)
{
  TimedTrajectory trajectory;
  trajectory.frame_id = "odom";
  trajectory.points.reserve(rollout_step_count + 1U);
  trajectory.points.push_back(TimedTrajectoryPoint{
    request.ego.pose, 0.0, target.desired_speed_mps,
    target.desired_curvature_inv_m});

  long double x = static_cast<long double>(request.ego.pose.x);
  long double y = static_cast<long double>(request.ego.pose.y);
  long double yaw = static_cast<long double>(request.ego.pose.yaw_rad);
  const long double speed =
    static_cast<long double>(target.desired_speed_mps);
  const long double angular_velocity =
    speed * static_cast<long double>(target.desired_curvature_inv_m);
  const long double dt = static_cast<long double>(rollout_dt_s);
  if (!std::isfinite(angular_velocity)) {
    return std::nullopt;
  }

  double previous_time_s = 0.0;
  for (std::size_t index = 1U; index <= rollout_step_count; ++index) {
    const long double cosine = std::cos(yaw);
    const long double sine = std::sin(yaw);
    const long double next_x = x + speed * cosine * dt;
    const long double next_y = y + speed * sine * dt;
    const long double next_yaw = std::remainder(
      yaw + angular_velocity * dt, kTwoPi);
    const long double multiple_time =
      static_cast<long double>(index) * dt;
    if (!representable_double(next_x) ||
      !representable_double(next_y) ||
      !representable_double(next_yaw) ||
      !representable_double(multiple_time))
    {
      return std::nullopt;
    }

    const double time_s = index == rollout_step_count ?
      rollout_horizon_s : static_cast<double>(multiple_time);
    if (!std::isfinite(time_s) || !(time_s > previous_time_s)) {
      return std::nullopt;
    }
    trajectory.points.push_back(TimedTrajectoryPoint{
      Pose2{
        static_cast<double>(next_x),
        static_cast<double>(next_y),
        static_cast<double>(next_yaw)},
      time_s,
      target.desired_speed_mps,
      target.desired_curvature_inv_m});
    x = next_x;
    y = next_y;
    yaw = next_yaw;
    previous_time_s = time_s;
  }
  return trajectory;
}

}  // namespace

MppiNav2Backend::MppiNav2Backend(MppiNav2BackendConfig config)
: config_(std::move(config)),
  command_adapter_(config_.command)
{
  if (!finite_positive(config_.command_timeout_s)) {
    throw std::invalid_argument(
            "MPPI command timeout must be finite and positive");
  }
  if (!finite_positive(config_.diagnostic_rollout_dt_s)) {
    throw std::invalid_argument(
            "MPPI diagnostic rollout dt must be finite and positive");
  }
  if (!finite_positive(config_.diagnostic_rollout_horizon_s)) {
    throw std::invalid_argument(
            "MPPI diagnostic rollout horizon must be finite and positive");
  }

  const long double ratio =
    static_cast<long double>(config_.diagnostic_rollout_horizon_s) /
    static_cast<long double>(config_.diagnostic_rollout_dt_s);
  if (!std::isfinite(ratio)) {
    throw std::invalid_argument(
            "MPPI diagnostic rollout horizon/dt is not finite");
  }
  const long double rounded_ratio = std::round(ratio);
  const long double tolerance =
    64.0L * static_cast<long double>(
    std::numeric_limits<double>::epsilon()) *
    std::max(1.0L, std::abs(ratio));
  if (std::abs(ratio - rounded_ratio) > tolerance ||
    !(rounded_ratio >= 1.0L))
  {
    throw std::invalid_argument(
            "MPPI diagnostic rollout horizon must be an integral multiple of dt");
  }
  if (rounded_ratio >
    static_cast<long double>(kMaximumRolloutPoints - 1U))
  {
    throw std::invalid_argument(
            "MPPI diagnostic rollout has too many points");
  }
  rollout_step_count_ = static_cast<std::size_t>(rounded_ratio);
}

bool MppiNav2Backend::observe_external_velocity_command(
  const ExternalVelocityCommand & command)
{
  if (!std::isfinite(command.vx_mps) ||
    !std::isfinite(command.wz_rad_s) ||
    command.receipt_steady_ns <= 0)
  {
    return false;
  }

  std::lock_guard<std::mutex> lock(state_mutex_);
  if (command_ &&
    command.receipt_steady_ns <= command_->receipt_steady_ns)
  {
    return false;
  }
  if (!command_adapter_.target_from_twist(
      command.vx_mps, command.wz_rad_s).valid)
  {
    return false;
  }
  command_ = command;
  return true;
}

LocalPlanningResult MppiNav2Backend::plan(
  const LocalPlanningRequest & request)
{
  if (request.steady_time_ns <= 0) {
    return invalid_result("mppi request steady time is invalid");
  }
  if (const auto error = request_validation_error(request, config_)) {
    return invalid_result(*error);
  }

  std::lock_guard<std::mutex> lock(state_mutex_);
  if (last_successful_plan_steady_ns_ > 0 &&
    request.steady_time_ns < last_successful_plan_steady_ns_)
  {
    return invalid_result("mppi request steady time is not monotonic");
  }
  if (!command_) {
    return invalid_result("mppi command unavailable");
  }
  if (command_->receipt_steady_ns > request.steady_time_ns) {
    return invalid_result("mppi command receipt is in the future");
  }
  // Both operands are positive and receipt <= request, so this signed
  // subtraction is representable in int64_t.
  const std::int64_t age_ns =
    request.steady_time_ns - command_->receipt_steady_ns;
  const long double age_s =
    static_cast<long double>(age_ns) / kNanosecondsPerSecond;
  if (age_s > static_cast<long double>(config_.command_timeout_s)) {
    return invalid_result("mppi command is stale");
  }

  const auto target = command_adapter_.target_from_twist(
    command_->vx_mps, command_->wz_rad_s);
  if (!target.valid) {
    return invalid_result(target.reason);
  }
  if (target.desired_speed_mps > request.constraints.maximum_speed_mps) {
    return invalid_result("mppi command exceeds maximum speed");
  }
  const long double lateral_acceleration =
    static_cast<long double>(target.desired_speed_mps) *
    static_cast<long double>(target.desired_speed_mps) *
    std::abs(static_cast<long double>(
      target.desired_curvature_inv_m));
  if (!std::isfinite(lateral_acceleration) ||
    lateral_acceleration >
    static_cast<long double>(
      request.constraints.maximum_lateral_acceleration_mps2))
  {
    return invalid_result(
      "mppi command exceeds maximum lateral acceleration");
  }

  std::optional<TimedTrajectory> rollout;
  try {
    rollout = make_command_rollout(
      request, target, config_.diagnostic_rollout_dt_s,
      config_.diagnostic_rollout_horizon_s, rollout_step_count_);
  } catch (const std::exception &) {
    return invalid_result("mppi command rollout allocation failed");
  }
  if (!rollout) {
    return invalid_result("mppi command rollout is not finite");
  }

  LocalPlanningResult result;
  result.reason = "ok";
  result.trajectory = std::move(*rollout);
  result.costs = {
    PlannerCost{"trajectory_kind.command_rollout", 1.0}};

  const auto command_result = command_adapter_.update(
    command_->vx_mps, command_->wz_rad_s, request.ego.speed_mps,
    request.dt_s, request.previous_command.steering_rad,
    request.behavior_id, request.gear_id);
  if (!command_result.valid) {
    return invalid_result(command_result.reason);
  }

  result.valid = true;
  result.desired_speed_mps = command_result.desired_speed_mps;
  result.desired_curvature_inv_m =
    command_result.desired_curvature_inv_m;
  result.direct_command = command_result.command;
  last_successful_plan_steady_ns_ = request.steady_time_ns;
  return result;
}

}  // namespace ad_planner
