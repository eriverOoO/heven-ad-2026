#ifndef AD_CONTROL__LONGITUDINAL__PID_HPP_
#define AD_CONTROL__LONGITUDINAL__PID_HPP_

#include "ad_control/common/types.hpp"

namespace ad_control
{

struct PidConfig
{
  double kp;
  double ki;
  double kd;
  double integral_limit;
  double derivative_limit;
  double error_scale{1.0};
  double brake_deadband{0.0};
  bool use_separate_brake_gains{false};
  double brake_kp{0.0};
  double brake_ki{0.0};
  double brake_kd{0.0};
  double derivative_filter_time_constant_s{0.1};
};

struct PidState
{
  bool initialized{false};
  double integral{0.0};
  double previous_error{0.0};
  double filtered_derivative{0.0};
  int actuation_mode{0};
  int behavior_id{0};
  int gear_id{0};
};

inline bool operator==(const PidState & lhs, const PidState & rhs)
{
  return lhs.initialized == rhs.initialized && lhs.integral == rhs.integral &&
         lhs.previous_error == rhs.previous_error &&
         lhs.filtered_derivative == rhs.filtered_derivative &&
         lhs.actuation_mode == rhs.actuation_mode &&
         lhs.behavior_id == rhs.behavior_id && lhs.gear_id == rhs.gear_id;
}

class PidController
{
public:
  explicit PidController(PidConfig config);
  ControllerResult update(
    double current_speed_mps, double target_speed_mps, double dt,
    int behavior_id, int gear_id);
  const PidState & state() const noexcept;
  const ControllerResult & last_valid_result() const noexcept;

private:
  PidConfig config_;
  PidState state_;
  ControllerResult last_valid_;
};

}  // namespace ad_control
#endif  // AD_CONTROL__LONGITUDINAL__PID_HPP_
