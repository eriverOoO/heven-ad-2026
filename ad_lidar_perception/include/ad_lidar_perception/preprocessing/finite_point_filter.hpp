#ifndef AD_LIDAR_PERCEPTION__PREPROCESSING__FINITE_POINT_FILTER_HPP_
#define AD_LIDAR_PERCEPTION__PREPROCESSING__FINITE_POINT_FILTER_HPP_

#include <cstddef>

#include <sensor_msgs/msg/point_cloud2.hpp>

namespace ad_lidar_perception::preprocessing
{

struct FinitePointFilterStats
{
  std::size_t input_points{0U};
  std::size_t output_points{0U};
  std::size_t removed_nonfinite{0U};
};

struct FinitePointFilterResult
{
  sensor_msgs::msg::PointCloud2 cloud;
  FinitePointFilterStats stats;
};

FinitePointFilterResult filter_finite_xyz(
  const sensor_msgs::msg::PointCloud2 & input);

}  // namespace ad_lidar_perception::preprocessing

#endif  // AD_LIDAR_PERCEPTION__PREPROCESSING__FINITE_POINT_FILTER_HPP_
