#include "ad_lidar_perception/preprocessing/point_layout_converter.hpp"

#include "ad_lidar_perception/preprocessing/xyzirt_layout.hpp"

#include <sensor_msgs/msg/point_field.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ad_lidar_perception::preprocessing
{
namespace
{

using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

constexpr std::uint32_t kOutputPointStep = 16U;

PointField make_field(
  const char * name, const std::uint32_t offset, const std::uint8_t datatype)
{
  PointField field;
  field.name = name;
  field.offset = offset;
  field.datatype = datatype;
  field.count = 1U;
  return field;
}

std::vector<PointField> output_fields()
{
  return {
    make_field("x", 0U, PointField::FLOAT32),
    make_field("y", 4U, PointField::FLOAT32),
    make_field("z", 8U, PointField::FLOAT32),
    make_field("intensity", 12U, PointField::UINT8),
    make_field("return_type", 13U, PointField::UINT8),
    make_field("channel", 14U, PointField::UINT16),
  };
}

std::uint64_t checked_multiply(
  const std::uint64_t lhs, const std::uint64_t rhs, const char * label)
{
  if (lhs != 0U && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
    throw std::invalid_argument(std::string(label) + " arithmetic overflow");
  }
  return lhs * rhs;
}

template<typename T>
void write_value(std::vector<std::uint8_t> & data, const std::size_t offset, const T & value)
{
  std::memcpy(data.data() + offset, &value, sizeof(T));
}

void validate_input_layout(const PointCloud2 & input)
{
  const auto point_count = checked_multiply(input.width, input.height, "point count");
  if (
    point_count >
    std::numeric_limits<std::uint32_t>::max() / kOutputPointStep ||
    point_count > std::numeric_limits<std::size_t>::max() / kOutputPointStep)
  {
    throw std::invalid_argument("input point count exceeds compact output limits");
  }
  static_cast<void>(XyzirtCloudView(input));
}

void validate_config(const ConverterConfig & config)
{
  if (!std::isfinite(config.intensity_scale)) {
    throw std::invalid_argument("intensity scale must be finite");
  }
  if (!std::isfinite(config.intensity_offset)) {
    throw std::invalid_argument("intensity offset must be finite");
  }
}

std::uint8_t convert_intensity(
  const float input, const ConverterConfig & config, ConversionStats & stats)
{
  if (!std::isfinite(input)) {
    ++stats.nonfinite_intensity;
    return config.nonfinite_intensity;
  }
  const auto scaled =
    config.intensity_scale * static_cast<double>(input) + config.intensity_offset;
  if (std::isfinite(scaled) && scaled < 0.0) {
    ++stats.clamped_low;
  } else if (std::isfinite(scaled) && scaled > 255.0) {
    ++stats.clamped_high;
  }
  return static_cast<std::uint8_t>(
    std::round(std::clamp(scaled, 0.0, 255.0)));
}

}  // namespace

ConversionResult convert_morai_xyzirt_to_point_xyzirc(
  const PointCloud2 & input,
  const ConverterConfig & config)
{
  validate_input_layout(input);
  validate_config(config);
  const XyzirtCloudView input_view(input);

  ConversionResult result;
  result.cloud.header = input.header;
  result.cloud.height = 1U;
  result.cloud.width = 0U;
  result.cloud.fields = output_fields();
  result.cloud.is_bigendian = false;
  result.cloud.point_step = kOutputPointStep;
  result.cloud.row_step = 0U;
  result.cloud.is_dense = true;
  result.stats.input_points =
    static_cast<std::size_t>(input.width) * input.height;
  result.cloud.data.reserve(result.stats.input_points * kOutputPointStep);

  for (std::uint32_t row = 0; row < input.height; ++row) {
    for (std::uint32_t column = 0; column < input.width; ++column) {
      const auto point_index =
        static_cast<std::size_t>(row) * input.width + column;
      const auto point = input_view.point(point_index);
      const auto x = point.x;
      const auto y = point.y;
      const auto z = point.z;
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        throw std::invalid_argument("input contains a nonfinite coordinate");
      }
      const auto intensity = point.intensity;
      const auto channel = point.ring;
      const auto output_offset = result.cloud.data.size();
      result.cloud.data.resize(output_offset + kOutputPointStep);
      write_value(result.cloud.data, output_offset + 0U, x);
      write_value(result.cloud.data, output_offset + 4U, y);
      write_value(result.cloud.data, output_offset + 8U, z);
      result.cloud.data[output_offset + 12U] =
        convert_intensity(intensity, config, result.stats);
      result.cloud.data[output_offset + 13U] = config.return_type;
      write_value(result.cloud.data, output_offset + 14U, channel);
      ++result.stats.output_points;
    }
  }

  result.cloud.width = static_cast<std::uint32_t>(result.stats.output_points);
  result.cloud.row_step =
    static_cast<std::uint32_t>(result.stats.output_points * kOutputPointStep);
  return result;
}

}  // namespace ad_lidar_perception::preprocessing
