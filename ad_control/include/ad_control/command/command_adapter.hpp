#ifndef AD_CONTROL__COMMAND__COMMAND_ADAPTER_HPP_
#define AD_CONTROL__COMMAND__COMMAND_ADAPTER_HPP_

#include <cstdint>
#include <optional>

#include "ad_control/common/types.hpp"
#include "ad_morai_interfaces/msg/ctrl_cmd.hpp"
#include "std_msgs/msg/header.hpp"

namespace ad_control
{

inline constexpr std::int8_t kGearReverseCode = 2;
inline constexpr std::int8_t kGearDriveCode = 4;

enum class GearRequest
{
  kKeep,
  kReverse,
  kDrive,
};

class AcknowledgedGear
{
public:
  bool update(std::int8_t gear)
  {
    if (gear < 0 || gear > 5) {
      return false;
    }
    value_ = gear;
    return true;
  }

  std::int8_t resolve(GearRequest request) const noexcept
  {
    if (request == GearRequest::kReverse) {
      return kGearReverseCode;
    }
    if (request == GearRequest::kDrive) {
      return kGearDriveCode;
    }
    return value_.value_or(kGearDriveCode);
  }

private:
  std::optional<std::int8_t> value_;
};

ad_morai_interfaces::msg::CtrlCmd make_ctrl_cmd(
  const PhysicalCommand & command,
  GearRequest gear_request,
  const AcknowledgedGear & acknowledged_gear,
  double maximum_steering_rad,
  const std_msgs::msg::Header & header);

}  // namespace ad_control

#endif  // AD_CONTROL__COMMAND__COMMAND_ADAPTER_HPP_
