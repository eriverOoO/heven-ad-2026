#pragma once

#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace ad_viz::perception
{

struct ObjectMarkerConfig
{
  double marker_lifetime_sec{0.5};
  double box_line_width_m{0.06};
  double label_height_m{0.45};
  double label_z_offset_m{0.25};
  double velocity_scale_sec{1.0};
  double minimum_velocity_mps{0.01};
};

visualization_msgs::msg::MarkerArray build_detection_markers(
  const autoware_perception_msgs::msg::DetectedObjects & input,
  const ObjectMarkerConfig & config = {});

visualization_msgs::msg::MarkerArray build_tracked_markers(
  const autoware_perception_msgs::msg::TrackedObjects & input,
  const ObjectMarkerConfig & config = {});

}  // namespace ad_viz::perception
