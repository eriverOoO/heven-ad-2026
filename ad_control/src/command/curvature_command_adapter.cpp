#include "ad_control/command/curvature_command_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ad_control
{
namespace
{

bool finite(const double value)
{
  return std::isfinite(value);
}

void require_positive_finite(const double value, const char * const name)
{
  if (!finite(value) || !(value > 0.0)) {
    throw std::invalid_argument(std::string(name) + " must be finite and positive");
  }
}

bool valid_input(
  const CurvatureCommandInput & input, const CurvatureCommandAdapterConfig & config)
{
  if (!finite(input.current_speed_mps) || !finite(input.target_speed_mps) ||
    !finite(input.desired_curvature_inv_m) || !finite(input.previous_steering_rad) ||
    !finite(input.dt_s) || input.current_speed_mps < 0.0 || input.target_speed_mps < 0.0 ||
    !(input.dt_s > 0.0) ||
    std::abs(input.previous_steering_rad) > config.maximum_road_wheel_steering_rad)
  {
    return false;
  }

  const long double raw_product = static_cast<long double>(config.wheelbase_m) *
    static_cast<long double>(input.desired_curvature_inv_m);
  const long double slew_delta = static_cast<long double>(config.maximum_steering_rate_radps) *
    static_cast<long double>(input.dt_s);
  return std::isfinite(raw_product) && std::isfinite(static_cast<double>(raw_product)) &&
         std::isfinite(slew_delta) && std::isfinite(static_cast<double>(slew_delta));
}

}  // namespace

CurvatureCommandAdapter::CurvatureCommandAdapter(CurvatureCommandAdapterConfig config)
: config_(config), pid_(config_.pid)
{
  require_positive_finite(config_.wheelbase_m, "wheelbase");
  require_positive_finite(
    config_.maximum_road_wheel_steering_rad, "maximum road-wheel steering");
  require_positive_finite(config_.maximum_steering_rate_radps, "maximum steering rate");
}

ControllerResult CurvatureCommandAdapter::update(const CurvatureCommandInput & input)
{
  if (!valid_input(input, config_)) {
    return ControllerResult{};
  }

  const double raw_steering = std::atan(
    static_cast<double>(static_cast<long double>(config_.wheelbase_m) *
    static_cast<long double>(input.desired_curvature_inv_m)));
  if (!finite(raw_steering)) {
    return ControllerResult{};
  }
  const double limited_steering = std::clamp(
    raw_steering, -config_.maximum_road_wheel_steering_rad,
    config_.maximum_road_wheel_steering_rad);
  const double maximum_delta = static_cast<double>(
    static_cast<long double>(config_.maximum_steering_rate_radps) *
    static_cast<long double>(input.dt_s));
  const double lower = input.previous_steering_rad - maximum_delta;
  const double upper = input.previous_steering_rad + maximum_delta;
  if (!finite(lower) || !finite(upper)) {
    return ControllerResult{};
  }
  const double safe_steering = std::clamp(limited_steering, lower, upper);
  if (!finite(safe_steering)) {
    return ControllerResult{};
  }

  ControllerResult result = pid_.update(
    input.current_speed_mps, input.target_speed_mps, input.dt_s,
    input.behavior_id, input.gear_id);
  if (result.valid) {
    result.command.steering_rad = safe_steering;
    last_valid_ = result;
  }
  return result;
}

const PidState & CurvatureCommandAdapter::pid_state() const noexcept
{
  return pid_.state();
}

const ControllerResult & CurvatureCommandAdapter::last_valid_result() const noexcept
{
  return last_valid_;
}

}  // namespace ad_control
