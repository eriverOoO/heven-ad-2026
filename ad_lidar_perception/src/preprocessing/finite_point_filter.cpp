#include "ad_lidar_perception/preprocessing/finite_point_filter.hpp"

#include <sensor_msgs/msg/point_field.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace ad_lidar_perception::preprocessing
{
namespace
{

using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

std::size_t checked_product(
  const std::size_t lhs, const std::size_t rhs, const char * label)
{
  if (lhs != 0U && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::invalid_argument(std::string(label) + " overflows");
  }
  return lhs * rhs;
}

std::array<std::size_t, 3U> xyz_offsets(const PointCloud2 & input)
{
  constexpr std::array<const char *, 3U> names{"x", "y", "z"};
  std::array<std::optional<std::size_t>, 3U> offsets;
  for (const auto & field : input.fields) {
    for (std::size_t index = 0U; index < names.size(); ++index) {
      if (field.name != names[index]) {
        continue;
      }
      if (offsets[index].has_value()) {
        throw std::invalid_argument(
                std::string("duplicate PointCloud2 field: ") + names[index]);
      }
      if (
        field.datatype != PointField::FLOAT32 || field.count != 1U ||
        field.offset > input.point_step ||
        input.point_step - field.offset < sizeof(float))
      {
        throw std::invalid_argument(
                std::string("PointCloud2 field must be FLOAT32: ") + names[index]);
      }
      offsets[index] = field.offset;
    }
  }
  std::array<std::size_t, 3U> result{};
  for (std::size_t index = 0U; index < names.size(); ++index) {
    if (!offsets[index].has_value()) {
      throw std::invalid_argument(
              std::string("PointCloud2 is missing field: ") + names[index]);
    }
    result[index] = offsets[index].value();
  }
  return result;
}

float read_float(const std::uint8_t * bytes, const std::size_t offset)
{
  float value{};
  std::memcpy(&value, bytes + offset, sizeof(value));
  return value;
}

}  // namespace

FinitePointFilterResult filter_finite_xyz(const PointCloud2 & input)
{
  if (input.is_bigendian) {
    throw std::invalid_argument("big-endian PointCloud2 is unsupported");
  }
  if (input.point_step == 0U) {
    throw std::invalid_argument("PointCloud2 point_step must be positive");
  }
  if (input.width > 0U && input.height == 0U) {
    throw std::invalid_argument(
            "PointCloud2 height must be positive when width is positive");
  }
  const auto minimum_row_step = checked_product(
    static_cast<std::size_t>(input.width),
    static_cast<std::size_t>(input.point_step), "PointCloud2 row_step");
  if (input.row_step < minimum_row_step) {
    throw std::invalid_argument(
            "PointCloud2 row_step is smaller than width * point_step");
  }
  const auto expected_size = checked_product(
    static_cast<std::size_t>(input.row_step),
    static_cast<std::size_t>(input.height), "PointCloud2 data size");
  if (input.data.size() != expected_size) {
    throw std::invalid_argument(
            "PointCloud2 data size does not match row_step * height");
  }
  const auto point_count = checked_product(
    static_cast<std::size_t>(input.width),
    static_cast<std::size_t>(input.height), "PointCloud2 point count");
  if (point_count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(
            "compacted PointCloud2 width is not representable");
  }
  const auto offsets = xyz_offsets(input);

  FinitePointFilterResult result;
  result.cloud.header = input.header;
  result.cloud.height = 1U;
  result.cloud.width = 0U;
  result.cloud.fields = input.fields;
  result.cloud.is_bigendian = false;
  result.cloud.point_step = input.point_step;
  result.cloud.row_step = 0U;
  result.cloud.is_dense = true;
  result.stats.input_points = point_count;
  result.cloud.data.reserve(checked_product(
    point_count, static_cast<std::size_t>(input.point_step),
    "PointCloud2 compact data size"));

  for (std::uint32_t row = 0U; row < input.height; ++row) {
    for (std::uint32_t column = 0U; column < input.width; ++column) {
      const auto input_offset =
        static_cast<std::size_t>(row) * input.row_step +
        static_cast<std::size_t>(column) * input.point_step;
      const auto * record = input.data.data() + input_offset;
      if (
        !std::isfinite(read_float(record, offsets[0])) ||
        !std::isfinite(read_float(record, offsets[1])) ||
        !std::isfinite(read_float(record, offsets[2])))
      {
        ++result.stats.removed_nonfinite;
        continue;
      }
      result.cloud.data.insert(
        result.cloud.data.end(), record, record + input.point_step);
      ++result.stats.output_points;
    }
  }

  result.cloud.width = static_cast<std::uint32_t>(result.stats.output_points);
  result.cloud.row_step = result.cloud.width * result.cloud.point_step;
  return result;
}

}  // namespace ad_lidar_perception::preprocessing
