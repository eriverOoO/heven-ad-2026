#include "ad_control/command/mppi_command_adapter.hpp"

#include <cmath>
#include <stdexcept>

namespace ad_control
{
namespace
{

MppiCommandResult adapt_valid_result(
  const ControllerResult & controller_result,
  const double desired_speed_mps,
  const double desired_curvature_inv_m)
{
  if (!controller_result.valid) {
    return MppiCommandResult{};
  }

  MppiCommandResult result;
  result.valid = true;
  result.reason = controller_result.reason;
  result.command = controller_result.command;
  result.desired_speed_mps = desired_speed_mps;
  result.desired_curvature_inv_m = desired_curvature_inv_m;
  return result;
}

}  // namespace

MppiCommandAdapter::MppiCommandAdapter(MppiCommandConfig config)
: config_(config),
  command_adapter_(CurvatureCommandAdapterConfig{
    config_.longitudinal_pid,
    config_.wheelbase_m,
    config_.maximum_road_wheel_angle_rad,
    config_.steering_rate_limit_rad_s})
{
  if (
    !std::isfinite(config_.near_zero_speed_mps) ||
    config_.near_zero_speed_mps < 0.0)
  {
    throw std::invalid_argument(
            "near-zero MPPI speed must be finite and nonnegative");
  }
  if (config_.allow_reverse) {
    throw std::invalid_argument("reverse MPPI commands are not supported");
  }
}

MppiVelocityTarget MppiCommandAdapter::target_from_twist(
  const double vx_mps, const double wz_rad_s) const
{
  if (!std::isfinite(vx_mps) || !std::isfinite(wz_rad_s)) {
    return MppiVelocityTarget{};
  }

  if (std::abs(vx_mps) <= config_.near_zero_speed_mps) {
    return MppiVelocityTarget{true, "ok", 0.0, 0.0};
  }

  if (vx_mps < 0.0) {
    MppiVelocityTarget result;
    result.reason = "reverse_disabled";
    return result;
  }

  const double curvature = wz_rad_s / vx_mps;
  if (!std::isfinite(curvature)) {
    return MppiVelocityTarget{};
  }
  return MppiVelocityTarget{true, "ok", vx_mps, curvature};
}

MppiCommandResult MppiCommandAdapter::update(
  const double vx_mps, const double wz_rad_s, const double actual_speed_mps,
  const double dt_s, const double previous_steering_rad,
  const int behavior_id, const int gear_id)
{
  const auto target = target_from_twist(vx_mps, wz_rad_s);
  if (!target.valid) {
    MppiCommandResult result;
    result.reason = target.reason;
    return result;
  }

  const auto controller_result = command_adapter_.update(
    CurvatureCommandInput{
      actual_speed_mps,
      target.desired_speed_mps,
      target.desired_curvature_inv_m,
      previous_steering_rad,
      dt_s,
      behavior_id,
      gear_id,
    });
  return adapt_valid_result(
    controller_result, target.desired_speed_mps,
    target.desired_curvature_inv_m);
}

const PidState & MppiCommandAdapter::pid_state() const noexcept
{
  return command_adapter_.pid_state();
}

}  // namespace ad_control
