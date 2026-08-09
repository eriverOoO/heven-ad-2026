// SPDX-License-Identifier: GPL-2.0-or-later
// Derived from FAST_LIO_ROS2 at Ericsii/FAST_LIO_ROS2@2fffc570.
// This MORAI port intentionally retains only the standard PointCloud2
// Velodyne-style path; Livox, Ouster and MID-360 message paths are removed.
#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

namespace fast_lio {

enum TimeUnit { SEC = 0, MS = 1, US = 2, NS = 3 };

class Preprocess {
 public:
  Preprocess();
  void set(bool feature_enabled, int scan_lines, double blind, int point_filter_num);
  void process(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &message,
               PointCloudXYZI::Ptr &output);

  int point_filter_num{2};
  int N_SCANS{16};
  int SCAN_RATE{10};
  int time_unit{SEC};
  double blind{0.1};
  bool feature_enabled{false};

 private:
  float time_unit_scale_{1000.0F};  // FastLIO stores per-point offsets in ms.
};

}  // namespace fast_lio
