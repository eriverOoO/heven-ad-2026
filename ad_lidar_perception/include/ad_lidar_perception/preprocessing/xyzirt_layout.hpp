#ifndef AD_LIDAR_PERCEPTION__PREPROCESSING__XYZIRT_LAYOUT_HPP_
#define AD_LIDAR_PERCEPTION__PREPROCESSING__XYZIRT_LAYOUT_HPP_

#include <cstddef>
#include <cstdint>

#include <sensor_msgs/msg/point_cloud2.hpp>

namespace ad_lidar_perception::preprocessing
{

struct XyzirtPoint
{
  float x;
  float y;
  float z;
  float intensity;
  std::uint16_t ring;
  float time;
};

class XyzirtCloudView
{
public:
  static constexpr std::uint32_t kPointStep = 22U;
  static constexpr std::uint32_t kXOffset = 0U;
  static constexpr std::uint32_t kYOffset = 4U;
  static constexpr std::uint32_t kZOffset = 8U;
  static constexpr std::uint32_t kIntensityOffset = 12U;
  static constexpr std::uint32_t kRingOffset = 16U;
  static constexpr std::uint32_t kTimeOffset = 18U;

  explicit XyzirtCloudView(const sensor_msgs::msg::PointCloud2 & cloud);

  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] std::size_t point_offset(std::size_t index) const;
  [[nodiscard]] XyzirtPoint point(std::size_t index) const;

private:
  const sensor_msgs::msg::PointCloud2 * cloud_;
  std::size_t size_;
};

}  // namespace ad_lidar_perception::preprocessing

#endif  // AD_LIDAR_PERCEPTION__PREPROCESSING__XYZIRT_LAYOUT_HPP_
