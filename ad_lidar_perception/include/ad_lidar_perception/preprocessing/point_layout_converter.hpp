#ifndef AD_LIDAR_PERCEPTION__PREPROCESSING__POINT_LAYOUT_CONVERTER_HPP_
#define AD_LIDAR_PERCEPTION__PREPROCESSING__POINT_LAYOUT_CONVERTER_HPP_

#include <cstddef>
#include <cstdint>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace ad_lidar_perception::preprocessing
{

struct ConverterConfig
{
  double intensity_scale{1.0};
  double intensity_offset{0.0};
  std::uint8_t nonfinite_intensity{0};
  std::uint8_t return_type{0};
};

struct ConversionStats
{
  std::size_t input_points{0};
  std::size_t output_points{0};
  std::size_t clamped_low{0};
  std::size_t clamped_high{0};
  std::size_t nonfinite_intensity{0};
};

struct ConversionResult
{
  sensor_msgs::msg::PointCloud2 cloud;
  ConversionStats stats;
};

ConversionResult convert_morai_xyzirt_to_point_xyzirc(
  const sensor_msgs::msg::PointCloud2 & input,
  const ConverterConfig & config);

}  // namespace ad_lidar_perception::preprocessing

#endif  // AD_LIDAR_PERCEPTION__PREPROCESSING__POINT_LAYOUT_CONVERTER_HPP_
