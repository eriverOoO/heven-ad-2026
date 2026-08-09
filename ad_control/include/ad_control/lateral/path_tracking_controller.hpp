#ifndef AD_CONTROL__LATERAL__PATH_TRACKING_CONTROLLER_HPP_
#define AD_CONTROL__LATERAL__PATH_TRACKING_CONTROLLER_HPP_

#include <optional>

#include "ad_control/common/types.hpp"

namespace ad_control
{

struct RouteSpeedProfile;

class PathTrackingController
{
public:
  virtual ~PathTrackingController() = default;

  virtual ControllerResult update(
    const Pose2 & pose, double speed_mps, double dt, int behavior_id,
    int gear_id,
    std::optional<double> target_speed_mps = std::nullopt) = 0;

  virtual const RouteSpeedProfile * route_speed_profile() const noexcept
  {
    return nullptr;
  }
};

}  // namespace ad_control

#endif  // AD_CONTROL__LATERAL__PATH_TRACKING_CONTROLLER_HPP_
