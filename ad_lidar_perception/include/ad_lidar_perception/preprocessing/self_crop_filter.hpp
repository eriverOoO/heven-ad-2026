#ifndef AD_LIDAR_PERCEPTION__PREPROCESSING__SELF_CROP_FILTER_HPP_
#define AD_LIDAR_PERCEPTION__PREPROCESSING__SELF_CROP_FILTER_HPP_

#include <cstddef>
#include <optional>

#include <sensor_msgs/msg/point_cloud2.hpp>

namespace ad_lidar_perception::preprocessing
{

struct SelfCropBounds
{
  double min_x_m{-0.990};
  double max_x_m{4.045};
  double min_y_m{-1.145};
  double max_y_m{1.145};
  double min_z_m{-0.200};
  double max_z_m{1.805};
};

struct RigidTransform3
{
  double translation_x_m{0.0};
  double translation_y_m{0.0};
  double translation_z_m{0.0};
  double quaternion_x{0.0};
  double quaternion_y{0.0};
  double quaternion_z{0.0};
  double quaternion_w{1.0};
};

struct SelfCropResult
{
  sensor_msgs::msg::PointCloud2 cloud;
  std::size_t input_points{0U};
  std::size_t removed_points{0U};
  std::size_t nonfinite_points{0U};
};

SelfCropResult crop_self_points(
  const sensor_msgs::msg::PointCloud2 & input,
  const SelfCropBounds & bounds,
  const std::optional<RigidTransform3> & base_from_input);

}  // namespace ad_lidar_perception::preprocessing

#endif  // AD_LIDAR_PERCEPTION__PREPROCESSING__SELF_CROP_FILTER_HPP_
