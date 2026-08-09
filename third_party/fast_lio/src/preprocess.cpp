// SPDX-License-Identifier: GPL-2.0-or-later
// Derived from FAST_LIO_ROS2 at Ericsii/FAST_LIO_ROS2@2fffc570.
#include "preprocess.hpp"

#include <fast_lio/point_validation.hpp>

#include <pcl_conversions/pcl_conversions.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace fast_lio {
namespace velodyne_ros {
struct EIGEN_ALIGN16 Point {
  PCL_ADD_POINT4D;
  float intensity;
  float time;
  std::uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace velodyne_ros
}  // namespace fast_lio

POINT_CLOUD_REGISTER_POINT_STRUCT(fast_lio::velodyne_ros::Point,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
  (float, time, time)(std::uint16_t, ring, ring))

namespace fast_lio {

Preprocess::Preprocess() = default;

void Preprocess::set(bool enabled, int scan_lines, double blind_range, int filter_num) {
  feature_enabled = enabled;
  N_SCANS = scan_lines;
  blind = blind_range;
  point_filter_num = filter_num;
}

void Preprocess::process(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &message,
                         PointCloudXYZI::Ptr &output) {
  pcl::PointCloud<velodyne_ros::Point> source;
  pcl::fromROSMsg(*message, source);
  output->clear();
  output->reserve(source.size());
  const double maximum_scan_duration_sec = 2.0 / static_cast<double>(SCAN_RATE);
  for (std::size_t i = 0; i < source.size(); ++i) {
    const auto &point = source.points[i];
    if (point.ring >= N_SCANS || i % static_cast<std::size_t>(point_filter_num) != 0) continue;
    if (!finite_xyz(point.x, point.y, point.z)) continue;
    if (!std::isfinite(point.intensity) ||
        !valid_relative_point_time(point.time, maximum_scan_duration_sec)) continue;
    if (point.x * point.x + point.y * point.y + point.z * point.z <= blind * blind) continue;
    PointType added;
    added.x = point.x;
    added.y = point.y;
    added.z = point.z;
    added.intensity = point.intensity;
    added.normal_x = added.normal_y = added.normal_z = 0.0F;
    added.curvature = point.time * time_unit_scale_;
    output->push_back(added);
  }
}

}  // namespace fast_lio
