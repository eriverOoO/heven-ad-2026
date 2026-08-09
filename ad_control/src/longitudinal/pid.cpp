#include "ad_control/longitudinal/pid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ad_control
{

PidController::PidController(PidConfig config)
: config_(config)
{
  if (!std::isfinite(config_.kp) || !std::isfinite(config_.ki) ||
    !std::isfinite(config_.kd) || !std::isfinite(config_.integral_limit) ||
    !std::isfinite(config_.derivative_limit) || !std::isfinite(config_.error_scale) ||
    !std::isfinite(config_.brake_deadband) || !std::isfinite(config_.brake_kp) ||
    !std::isfinite(config_.brake_ki) || !std::isfinite(config_.brake_kd) ||
    !std::isfinite(config_.derivative_filter_time_constant_s) ||
    config_.integral_limit < 0.0 || config_.derivative_limit < 0.0 ||
    config_.error_scale <= 0.0 || config_.brake_deadband < 0.0 ||
    config_.derivative_filter_time_constant_s < 0.0 ||
    (config_.use_separate_brake_gains &&
    (config_.brake_kp < 0.0 || config_.brake_ki < 0.0 || config_.brake_kd < 0.0)))
  {
    throw std::invalid_argument("PID configuration must be finite with nonnegative limits");
  }
}

ControllerResult PidController::update(
  double current_speed_mps, double target_speed_mps, double dt,
  int behavior_id, int gear_id)
{
  if (!std::isfinite(current_speed_mps) || !std::isfinite(target_speed_mps) ||
    !std::isfinite(dt) || dt <= 0.0)
  {
    return ControllerResult{};
  }

  PidState next = state_;
  const long double raw_error =
    (static_cast<long double>(target_speed_mps) -
    static_cast<long double>(current_speed_mps)) * config_.error_scale;
  if (std::abs(raw_error) > static_cast<long double>(std::numeric_limits<double>::max())) {
    return ControllerResult{};
  }
  const long double deadband = static_cast<long double>(config_.brake_deadband);
  const long double switching_tolerance =
    32.0L * std::numeric_limits<double>::epsilon() *
    std::max({1.0L, std::abs(raw_error), deadband});
  const int actuation_mode =
    raw_error > switching_tolerance ? 1 :
    (raw_error < -deadband - switching_tolerance ? -1 : 0);
  const long double wide_error =
    actuation_mode > 0 ? raw_error :
    (actuation_mode < 0 ? raw_error + deadband : 0.0L);
  const double error = static_cast<double>(wide_error);
  const bool transition = !next.initialized || next.behavior_id != behavior_id ||
    next.gear_id != gear_id || next.actuation_mode != actuation_mode;
  if (transition) {
    next.initialized = true;
    next.integral = 0.0;
    next.previous_error = error;
    next.filtered_derivative = 0.0;
    next.actuation_mode = actuation_mode;
    next.behavior_id = behavior_id;
    next.gear_id = gear_id;
  }
  if ((wide_error < 0.0L && next.integral > 0.0) ||
    (wide_error > 0.0L && next.integral < 0.0))
  {
    next.integral = 0.0;
  }
  const double integral_before_update = next.integral;
  const long double wide_integral = static_cast<long double>(next.integral) +
    wide_error * static_cast<long double>(dt);
  const double candidate_integral = static_cast<double>(std::clamp(
      wide_integral, -static_cast<long double>(config_.integral_limit),
      static_cast<long double>(config_.integral_limit)));
  const long double raw_derivative = transition ? 0.0L : std::clamp(
    (wide_error - static_cast<long double>(next.previous_error)) /
    static_cast<long double>(dt),
    -static_cast<long double>(config_.derivative_limit),
    static_cast<long double>(config_.derivative_limit));
  const long double filter_alpha =
    config_.derivative_filter_time_constant_s <= 0.0 ? 1.0L :
    static_cast<long double>(dt) /
    (static_cast<long double>(config_.derivative_filter_time_constant_s) +
    static_cast<long double>(dt));
  const long double wide_derivative =
    static_cast<long double>(next.filtered_derivative) +
    filter_alpha *
    (raw_derivative - static_cast<long double>(next.filtered_derivative));
  next.filtered_derivative = static_cast<double>(wide_derivative);
  next.previous_error = error;

  const double kp = actuation_mode < 0 && config_.use_separate_brake_gains ?
    config_.brake_kp : config_.kp;
  const double ki = actuation_mode < 0 && config_.use_separate_brake_gains ?
    config_.brake_ki : config_.ki;
  const double kd = actuation_mode < 0 && config_.use_separate_brake_gains ?
    config_.brake_kd : config_.kd;
  const auto output_for_integral = [&](double integral) {
      return static_cast<long double>(kp) * wide_error +
             static_cast<long double>(ki) * static_cast<long double>(integral) +
             static_cast<long double>(kd) * wide_derivative;
    };
  long double output = output_for_integral(candidate_integral);
  const long double integral_output_change = static_cast<long double>(ki) *
    (static_cast<long double>(candidate_integral) -
    static_cast<long double>(integral_before_update));
  const bool winds_further_positive = output > 1.0L && integral_output_change > 0.0L;
  const bool winds_further_negative = output < -1.0L && integral_output_change < 0.0L;
  if (winds_further_positive || winds_further_negative) {
    next.integral = integral_before_update;
    output = output_for_integral(next.integral);
  } else {
    next.integral = candidate_integral;
  }
  ControllerResult result;
  result.valid = true;
  result.reason = "ok";
  if (actuation_mode > 0 && output > 0.0) {
    result.command.accel = static_cast<double>(std::clamp(output, 0.0L, 1.0L));
    result.command.brake = 0.0;
  } else if (actuation_mode < 0 && output < 0.0) {
    result.command.accel = 0.0;
    result.command.brake = static_cast<double>(std::clamp(-output, 0.0L, 1.0L));
  } else {
    result.command.accel = 0.0;
    result.command.brake = 0.0;
  }
  state_ = next;
  last_valid_ = result;
  return result;
}

const PidState & PidController::state() const noexcept {return state_;}
const ControllerResult & PidController::last_valid_result() const noexcept {return last_valid_;}

}  // namespace ad_control
