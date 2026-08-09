#include "ad_control/command/command_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ad_control
{

ad_morai_interfaces::msg::CtrlCmd make_ctrl_cmd(
  const PhysicalCommand & command,
  GearRequest gear_request,
  const AcknowledgedGear & acknowledged_gear,
  double maximum_steering_rad,
  const std_msgs::msg::Header & header)
{
  if (!std::isfinite(maximum_steering_rad) || maximum_steering_rad <= 0.0) {
    throw std::invalid_argument("maximum steering angle must be finite and positive");
  }

  ad_morai_interfaces::msg::CtrlCmd message;
  message.header = header;
  message.ctrl_mode = ad_morai_interfaces::msg::CtrlCmd::CTRL_MODE_AUTO;
  message.gear = acknowledged_gear.resolve(gear_request);
  message.long_cmd_type = ad_morai_interfaces::msg::CtrlCmd::LONG_CMD_THROTTLE;
  message.velocity = 0.0F;
  message.acceleration = 0.0F;
  message.accel = static_cast<float>(command.accel);
  message.brake = static_cast<float>(command.brake);
  message.steering = static_cast<float>(std::clamp(
      command.steering_rad / maximum_steering_rad, -1.0, 1.0));
  return message;
}

}  // namespace ad_control
