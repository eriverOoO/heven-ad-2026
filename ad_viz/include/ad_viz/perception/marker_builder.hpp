#pragma once

#include <ad_interfaces/msg/predicted_object_array.hpp>
#include <std_msgs/msg/header.hpp>
#include <unique_identifier_msgs/msg/uuid.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <cstdint>
#include <string>

namespace ad_viz::perception
{

inline constexpr std::int32_t kCurrentBoxMarkerId = 0;
inline constexpr std::int32_t kHalfSecondBoxMarkerId = 1;
inline constexpr std::int32_t kOneSecondBoxMarkerId = 2;
inline constexpr std::int32_t kMotionLineMarkerId = 3;
inline constexpr std::int32_t kVelocityArrowMarkerId = 4;
inline constexpr std::int32_t kTrackLabelMarkerId = 5;

struct MarkerBuilderConfig
{
  double marker_lifetime_sec{0.5};
  double stale_timeout_sec{1.0};
  double current_alpha{0.80};
  double half_second_alpha{0.50};
  double one_second_alpha{0.25};
  double line_alpha{0.90};
  double arrow_alpha{0.95};
  double box_line_width_m{0.06};
  double line_width_m{0.10};
  double arrow_shaft_diameter_m{0.10};
  double arrow_head_diameter_m{0.22};
  double arrow_head_length_m{0.30};
  double label_height_m{0.45};
  double label_z_offset_m{0.25};
};

std::int64_t prediction_stamp_ns(const std_msgs::msg::Header & header);

std::string uuid_namespace(
  const unique_identifier_msgs::msg::UUID & object_id);

visualization_msgs::msg::MarkerArray build_delete_all(
  const std_msgs::msg::Header & header,
  const MarkerBuilderConfig & config);

visualization_msgs::msg::MarkerArray build_prediction_markers(
  const ad_interfaces::msg::PredictedObjectArray & input,
  const MarkerBuilderConfig & config);

}  // namespace ad_viz::perception
