#ifndef AD_LIDAR_PERCEPTION__PREPROCESSING__GRAVITY_LEVELER_HPP_
#define AD_LIDAR_PERCEPTION__PREPROCESSING__GRAVITY_LEVELER_HPP_

#include <array>
#include <string>

#include <sensor_msgs/msg/point_cloud2.hpp>

namespace ad_lidar_perception::preprocessing
{

/// A rigid transform using the convention ^A T_B: coordinates in B map into A.
struct GravityLevelingTransform
{
  std::array<double, 3> translation{0.0, 0.0, 0.0};
  std::array<double, 4> quaternion_xyzw{0.0, 0.0, 0.0, 1.0};
};

struct GravityLevelingResult
{
  sensor_msgs::msg::PointCloud2 cloud;
  /// Dynamic transform ^B T_V for base_link -> <sensor_name>_leveled_frame.
  GravityLevelingTransform base_from_level;
};

/// Return the exact generic level-frame child for a valid relative sensor frame.
std::string derive_leveled_frame(const std::string & input_frame);

///
/// Level a completed scan at one timestamp (this is not deskew).
///
/// Given ^O T_B and ^B T_L, this constructs ^O T_L = ^O T_B * ^B T_L,
/// places V at the same physical origin with ^O R_V = Rz(yaw(^O R_L)), and
/// maps each point with ^V T_L = inverse(^O T_V) * ^O T_L. Only XYZ bytes
/// are changed. The returned transform is ^B T_V = inverse(^O T_B) * ^O T_V.
///
GravityLevelingResult level_xyzirt_cloud(
  const sensor_msgs::msg::PointCloud2 & input,
  const GravityLevelingTransform & odom_from_base,
  const GravityLevelingTransform & base_from_lidar,
  const std::string & leveled_frame);

}  // namespace ad_lidar_perception::preprocessing

#endif  // AD_LIDAR_PERCEPTION__PREPROCESSING__GRAVITY_LEVELER_HPP_
