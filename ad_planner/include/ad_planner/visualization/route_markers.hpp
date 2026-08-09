#ifndef AD_PLANNER__VISUALIZATION__ROUTE_MARKERS_HPP_
#define AD_PLANNER__VISUALIZATION__ROUTE_MARKERS_HPP_

#include <cstddef>
#include <cstdint>
#include <string>

#include <visualization_msgs/msg/marker_array.hpp>

#include "ad_planner/local_planning/common/local_motion.hpp"

namespace ad_planner {

struct RouteMarkerLimits {
  std::size_t maximum_occupancy_points_per_class{2048U};
  std::size_t maximum_prediction_segments_per_class{4096U};
};

visualization_msgs::msg::MarkerArray build_occupancy_relevance_markers(
    const std::string &frame_id, const OccupancyGrid &grid,
    const ReferenceCorridor &corridor, std::int8_t occupied_threshold,
    const RouteMarkerLimits &limits = {});

visualization_msgs::msg::MarkerArray build_predicted_relevance_markers(
    const std::string &frame_id, const PredictedObjectSet &objects,
    const ReferenceCorridor &corridor, const RouteMarkerLimits &limits = {});

} // namespace ad_planner

#endif // AD_PLANNER__VISUALIZATION__ROUTE_MARKERS_HPP_
