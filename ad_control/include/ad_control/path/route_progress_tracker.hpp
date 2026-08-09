#ifndef AD_CONTROL__PATH__ROUTE_PROGRESS_TRACKER_HPP_
#define AD_CONTROL__PATH__ROUTE_PROGRESS_TRACKER_HPP_

#include <cstddef>

#include "ad_control/common/types.hpp"

namespace ad_control
{

class RouteProgressTracker
{
public:
  explicit RouteProgressTracker(
    Route route, std::size_t forward_window, std::size_t max_laps = 1);

  RouteProgressState update(const Point3 & position);
  RouteProgressState update(const Point3 & position, double yaw_rad);
  const RouteProgressState & state() const noexcept;

private:
  Route route_;
  std::size_t forward_window_;
  std::size_t max_laps_;
  RouteProgressState state_;
};

}  // namespace ad_control

#endif  // AD_CONTROL__PATH__ROUTE_PROGRESS_TRACKER_HPP_
