#ifndef AD_CONTROL__COMMAND__MPPI_COMMAND_ADAPTER_HPP_
#define AD_CONTROL__COMMAND__MPPI_COMMAND_ADAPTER_HPP_

#include <string>

#include "ad_control/command/curvature_command_adapter.hpp"

namespace ad_control
{

struct MppiCommandConfig
{
  double wheelbase_m{3.0};
  double maximum_road_wheel_angle_rad{0.588};
  double steering_rate_limit_rad_s{0.35};
  double near_zero_speed_mps{0.05};
  bool allow_reverse{false};
  PidConfig longitudinal_pid{0.0, 0.0, 0.0, 0.0, 0.0};
};

struct MppiCommandResult
{
  bool valid{false};
  std::string reason{"invalid"};
  PhysicalCommand command{};
  double desired_speed_mps{0.0};
  double desired_curvature_inv_m{0.0};
};

struct MppiVelocityTarget
{
  bool valid{false};
  std::string reason{"invalid"};
  double desired_speed_mps{0.0};
  double desired_curvature_inv_m{0.0};
};

class MppiCommandAdapter
{
public:
  explicit MppiCommandAdapter(MppiCommandConfig config);

  MppiVelocityTarget target_from_twist(
    double vx_mps, double wz_rad_s) const;

  MppiCommandResult update(
    double vx_mps, double wz_rad_s, double actual_speed_mps,
    double dt_s, double previous_steering_rad, int behavior_id, int gear_id);

  const PidState & pid_state() const noexcept;

private:
  MppiCommandConfig config_;
  CurvatureCommandAdapter command_adapter_;
};

}  // namespace ad_control

#endif  // AD_CONTROL__COMMAND__MPPI_COMMAND_ADAPTER_HPP_
