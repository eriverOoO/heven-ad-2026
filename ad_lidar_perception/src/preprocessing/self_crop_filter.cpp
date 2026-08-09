#include "ad_lidar_perception/preprocessing/self_crop_filter.hpp"

#include "ad_lidar_perception/preprocessing/xyzirt_layout.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ad_lidar_perception::preprocessing
{
namespace
{

void validate_bounds(const SelfCropBounds & bounds)
{
  const std::array<double, 6> values{
    bounds.min_x_m, bounds.max_x_m, bounds.min_y_m,
    bounds.max_y_m, bounds.min_z_m, bounds.max_z_m};
  if (!std::all_of(values.begin(), values.end(), [](const double value) {
      return std::isfinite(value);
    }))
  {
    throw std::invalid_argument("self crop bounds must be finite");
  }
  if (bounds.min_x_m > bounds.max_x_m) {
    throw std::invalid_argument("self crop x bounds require min <= max");
  }
  if (bounds.min_y_m > bounds.max_y_m) {
    throw std::invalid_argument("self crop y bounds require min <= max");
  }
  if (bounds.min_z_m > bounds.max_z_m) {
    throw std::invalid_argument("self crop z bounds require min <= max");
  }
}

RigidTransform3 normalized_transform(
  const std::optional<RigidTransform3> & base_from_input)
{
  if (!base_from_input.has_value()) {
    throw std::invalid_argument("self crop requires a timestamped base-from-input transform");
  }
  auto transform = base_from_input.value();
  if (
    !std::isfinite(transform.translation_x_m) ||
    !std::isfinite(transform.translation_y_m) ||
    !std::isfinite(transform.translation_z_m))
  {
    throw std::invalid_argument("self crop transform translation must be finite");
  }
  const std::array<double, 4> quaternion{
    transform.quaternion_x, transform.quaternion_y,
    transform.quaternion_z, transform.quaternion_w};
  if (!std::all_of(quaternion.begin(), quaternion.end(), [](const double value) {
      return std::isfinite(value);
    }))
  {
    throw std::invalid_argument("self crop transform quaternion must be finite");
  }
  const auto scale = std::max({
    std::abs(transform.quaternion_x), std::abs(transform.quaternion_y),
    std::abs(transform.quaternion_z), std::abs(transform.quaternion_w)});
  if (scale == 0.0) {
    throw std::invalid_argument("self crop transform quaternion must be nonzero");
  }
  transform.quaternion_x /= scale;
  transform.quaternion_y /= scale;
  transform.quaternion_z /= scale;
  transform.quaternion_w /= scale;
  const auto norm = std::sqrt(
    transform.quaternion_x * transform.quaternion_x +
    transform.quaternion_y * transform.quaternion_y +
    transform.quaternion_z * transform.quaternion_z +
    transform.quaternion_w * transform.quaternion_w);
  transform.quaternion_x /= norm;
  transform.quaternion_y /= norm;
  transform.quaternion_z /= norm;
  transform.quaternion_w /= norm;
  return transform;
}

std::array<double, 3> transform_point(
  const RigidTransform3 & transform, const float x, const float y, const float z)
{
  const auto qx = transform.quaternion_x;
  const auto qy = transform.quaternion_y;
  const auto qz = transform.quaternion_z;
  const auto qw = transform.quaternion_w;
  const auto source_x = static_cast<double>(x);
  const auto source_y = static_cast<double>(y);
  const auto source_z = static_cast<double>(z);
  return {
    (1.0 - 2.0 * (qy * qy + qz * qz)) * source_x +
    2.0 * (qx * qy - qz * qw) * source_y +
    2.0 * (qx * qz + qy * qw) * source_z + transform.translation_x_m,
    2.0 * (qx * qy + qz * qw) * source_x +
    (1.0 - 2.0 * (qx * qx + qz * qz)) * source_y +
    2.0 * (qy * qz - qx * qw) * source_z + transform.translation_y_m,
    2.0 * (qx * qz - qy * qw) * source_x +
    2.0 * (qy * qz + qx * qw) * source_y +
    (1.0 - 2.0 * (qx * qx + qy * qy)) * source_z + transform.translation_z_m,
  };
}

bool inside_inclusive(
  const std::array<double, 3> & point, const SelfCropBounds & bounds)
{
  return point[0] >= bounds.min_x_m && point[0] <= bounds.max_x_m &&
         point[1] >= bounds.min_y_m && point[1] <= bounds.max_y_m &&
         point[2] >= bounds.min_z_m && point[2] <= bounds.max_z_m;
}

}  // namespace

SelfCropResult crop_self_points(
  const sensor_msgs::msg::PointCloud2 & input,
  const SelfCropBounds & bounds,
  const std::optional<RigidTransform3> & base_from_input)
{
  validate_bounds(bounds);
  const auto transform = normalized_transform(base_from_input);
  const XyzirtCloudView view(input);
  if (
    view.size() >
    static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) /
    XyzirtCloudView::kPointStep)
  {
    throw std::invalid_argument("self crop compact output exceeds row_step limits");
  }

  SelfCropResult result;
  result.input_points = view.size();
  result.cloud.header = input.header;
  result.cloud.height = 1U;
  result.cloud.width = 0U;
  result.cloud.fields = input.fields;
  result.cloud.is_bigendian = input.is_bigendian;
  result.cloud.point_step = input.point_step;
  result.cloud.row_step = 0U;
  result.cloud.is_dense = true;
  result.cloud.data.reserve(view.size() * input.point_step);

  for (std::size_t index = 0U; index < view.size(); ++index) {
    const auto point = view.point(index);
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      ++result.nonfinite_points;
      continue;
    }
    if (inside_inclusive(transform_point(transform, point.x, point.y, point.z), bounds)) {
      ++result.removed_points;
      continue;
    }
    const auto source_offset = view.point_offset(index);
    const auto output_offset = result.cloud.data.size();
    result.cloud.data.resize(output_offset + input.point_step);
    std::memcpy(
      result.cloud.data.data() + output_offset,
      input.data.data() + source_offset, input.point_step);
  }

  const auto survivor_count =
    result.input_points - result.removed_points - result.nonfinite_points;
  result.cloud.width = static_cast<std::uint32_t>(survivor_count);
  result.cloud.row_step = static_cast<std::uint32_t>(survivor_count * input.point_step);
  return result;
}

}  // namespace ad_lidar_perception::preprocessing
