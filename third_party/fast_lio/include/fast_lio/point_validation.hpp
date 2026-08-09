// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

namespace fast_lio {

inline bool finite_xyz(float x, float y, float z) {
  return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

inline bool has_exact_field(
    const sensor_msgs::msg::PointCloud2 & cloud,
    const std::string_view name,
    const std::uint8_t datatype,
    const std::uint32_t byte_width)
{
  std::size_t matches = 0;
  for (const auto & field : cloud.fields) {
    if (field.name != name) {
      continue;
    }
    ++matches;
    if (field.datatype != datatype || field.count != 1 ||
        field.offset > cloud.point_step ||
        byte_width > cloud.point_step - field.offset)
    {
      return false;
    }
  }
  return matches == 1;
}

inline bool valid_morai_pointcloud_layout(
    const sensor_msgs::msg::PointCloud2 & cloud)
{
  using Field = sensor_msgs::msg::PointField;
  return
    has_exact_field(cloud, "x", Field::FLOAT32, 4) &&
    has_exact_field(cloud, "y", Field::FLOAT32, 4) &&
    has_exact_field(cloud, "z", Field::FLOAT32, 4) &&
    has_exact_field(cloud, "intensity", Field::FLOAT32, 4) &&
    has_exact_field(cloud, "time", Field::FLOAT32, 4) &&
    has_exact_field(cloud, "ring", Field::UINT16, 2);
}

inline bool valid_pointcloud_shape(
    const sensor_msgs::msg::PointCloud2 & cloud,
    const std::size_t maximum_points)
{
  if (cloud.height == 0 || cloud.width == 0 || cloud.point_step == 0) {
    return false;
  }
  const std::uint64_t point_count =
    static_cast<std::uint64_t>(cloud.height) * cloud.width;
  const std::uint64_t minimum_row_step =
    static_cast<std::uint64_t>(cloud.point_step) * cloud.width;
  const std::uint64_t required_bytes =
    static_cast<std::uint64_t>(cloud.row_step) * cloud.height;
  return point_count <= maximum_points &&
         cloud.row_step >= minimum_row_step &&
         required_bytes == cloud.data.size();
}

inline bool valid_relative_point_time(
    const float point_time_sec,
    const double maximum_scan_duration_sec)
{
  return maximum_scan_duration_sec > 0.0 &&
         std::isfinite(point_time_sec) &&
         point_time_sec >= 0.0 &&
         point_time_sec <= maximum_scan_duration_sec;
}

inline bool valid_relative_frame(const std::string_view frame)
{
  return !frame.empty() && frame.front() != '/' &&
         frame.find("//") == std::string_view::npos &&
         frame.find_first_of(" \t\r\n") == std::string_view::npos;
}

inline bool matches_lidar_frame(
    const std::string_view incoming_frame,
    const std::string_view configured_lidar_frame)
{
  return valid_relative_frame(configured_lidar_frame) &&
         incoming_frame == configured_lidar_frame;
}

}  // namespace fast_lio
