#ifndef AD_PLANNER__VISUALIZATION__PATH_TRACKING_MARKERS_HPP_
#define AD_PLANNER__VISUALIZATION__PATH_TRACKING_MARKERS_HPP_

#include <builtin_interfaces/msg/time.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <cstddef>
#include <string>

#include "ad_control/common/types.hpp"
#include "ad_control/path/route_speed_profile.hpp"

namespace ad_planner {

visualization_msgs::msg::MarkerArray make_route_profile_markers(
    const ad_control::Route &route,
    const ad_control::RouteSpeedProfile *profile, const std::string &frame_id,
    const builtin_interfaces::msg::Time &stamp, std::size_t sample_stride);

} // namespace ad_planner

#endif // AD_PLANNER__VISUALIZATION__PATH_TRACKING_MARKERS_HPP_
