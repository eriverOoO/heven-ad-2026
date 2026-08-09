#ifndef AD_CONTROL__COMMAND__CURVATURE_COMMAND_ADAPTER_HPP_
#define AD_CONTROL__COMMAND__CURVATURE_COMMAND_ADAPTER_HPP_

#include "ad_control/longitudinal/pid.hpp"

namespace ad_control
{

struct CurvatureCommandAdapterConfig
{
  PidConfig pid;
  double wheelbase_m{0.0};
  double maximum_road_wheel_steering_rad{0.0};
  double maximum_steering_rate_radps{0.0};
};

struct CurvatureCommandInput
{
  double current_speed_mps{0.0};
  double target_speed_mps{0.0};
  double desired_curvature_inv_m{0.0};
  double previous_steering_rad{0.0};
  double dt_s{0.0};
  int behavior_id{0};
  int gear_id{0};
};

class CurvatureCommandAdapter
{
public:
  explicit CurvatureCommandAdapter(CurvatureCommandAdapterConfig config);

  ControllerResult update(const CurvatureCommandInput & input);
  const PidState & pid_state() const noexcept;
  const ControllerResult & last_valid_result() const noexcept;

private:
  CurvatureCommandAdapterConfig config_;
  PidController pid_;
  ControllerResult last_valid_;
};

}  // namespace ad_control

#endif  // AD_CONTROL__COMMAND__CURVATURE_COMMAND_ADAPTER_HPP_
