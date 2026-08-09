#include "ad_lidar_perception/preprocessing/xyzirt_layout.hpp"

#include <sensor_msgs/msg/point_field.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace ad_lidar_perception::preprocessing
{
namespace
{

using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

struct ExpectedField
{
  const char * name;
  std::uint32_t offset;
  std::uint8_t datatype;
};

constexpr std::array<ExpectedField, 6> kFields{{
  {"x", XyzirtCloudView::kXOffset, PointField::FLOAT32},
  {"y", XyzirtCloudView::kYOffset, PointField::FLOAT32},
  {"z", XyzirtCloudView::kZOffset, PointField::FLOAT32},
  {"intensity", XyzirtCloudView::kIntensityOffset, PointField::FLOAT32},
  {"ring", XyzirtCloudView::kRingOffset, PointField::UINT16},
  {"time", XyzirtCloudView::kTimeOffset, PointField::FLOAT32},
}};

std::uint64_t checked_multiply(
  const std::uint64_t lhs, const std::uint64_t rhs, const char * label)
{
  if (lhs != 0U && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
    throw std::invalid_argument(std::string(label) + " arithmetic overflow");
  }
  return lhs * rhs;
}

template<typename T>
T read_value(const std::uint8_t * data)
{
  T value{};
  std::memcpy(&value, data, sizeof(T));
  return value;
}

void validate(const PointCloud2 & cloud)
{
  if (cloud.is_bigendian) {
    throw std::invalid_argument("XYZIRT cloud must be little-endian");
  }
  if (cloud.fields.size() != kFields.size()) {
    throw std::invalid_argument("XYZIRT cloud must contain exactly six fields");
  }
  for (std::size_t index = 0U; index < kFields.size(); ++index) {
    const auto & actual = cloud.fields[index];
    const auto & expected = kFields[index];
    if (
      actual.name != expected.name || actual.offset != expected.offset ||
      actual.datatype != expected.datatype || actual.count != 1U)
    {
      throw std::invalid_argument(
              "XYZIRT field " + std::to_string(index) + " must be " +
              expected.name + " with the strict XYZIRT layout");
    }
  }
  if (cloud.point_step != XyzirtCloudView::kPointStep) {
    throw std::invalid_argument("XYZIRT point_step must be 22");
  }
  if (cloud.width > 0U && cloud.height == 0U) {
    throw std::invalid_argument("XYZIRT height must be nonzero when width is nonzero");
  }
  const auto minimum_row_step =
    checked_multiply(cloud.width, XyzirtCloudView::kPointStep, "row_step");
  if (static_cast<std::uint64_t>(cloud.row_step) < minimum_row_step) {
    throw std::invalid_argument("XYZIRT row_step is smaller than width * 22");
  }
  const auto point_count = checked_multiply(cloud.width, cloud.height, "point count");
  if (point_count > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("XYZIRT point count exceeds addressable memory");
  }
  const auto expected_size = checked_multiply(cloud.row_step, cloud.height, "data size");
  if (
    expected_size > std::numeric_limits<std::size_t>::max() ||
    cloud.data.size() != static_cast<std::size_t>(expected_size))
  {
    throw std::invalid_argument("XYZIRT data size must equal row_step * height");
  }
}

}  // namespace

XyzirtCloudView::XyzirtCloudView(const PointCloud2 & cloud)
: cloud_(&cloud), size_(0U)
{
  validate(cloud);
  size_ = static_cast<std::size_t>(cloud.width) * cloud.height;
}

std::size_t XyzirtCloudView::size() const noexcept
{
  return size_;
}

std::size_t XyzirtCloudView::point_offset(const std::size_t index) const
{
  if (index >= size_) {
    throw std::out_of_range("XYZIRT point index is out of range");
  }
  const auto row = index / cloud_->width;
  const auto column = index % cloud_->width;
  return row * cloud_->row_step + column * cloud_->point_step;
}

XyzirtPoint XyzirtCloudView::point(const std::size_t index) const
{
  const auto * record = cloud_->data.data() + point_offset(index);
  return XyzirtPoint{
    read_value<float>(record + kXOffset),
    read_value<float>(record + kYOffset),
    read_value<float>(record + kZOffset),
    read_value<float>(record + kIntensityOffset),
    read_value<std::uint16_t>(record + kRingOffset),
    read_value<float>(record + kTimeOffset),
  };
}

}  // namespace ad_lidar_perception::preprocessing
