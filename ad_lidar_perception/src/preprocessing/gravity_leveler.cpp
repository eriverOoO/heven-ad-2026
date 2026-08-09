#include "ad_lidar_perception/preprocessing/gravity_leveler.hpp"

#include "ad_lidar_perception/preprocessing/xyzirt_layout.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ad_lidar_perception::preprocessing
{
namespace
{

using Matrix3 = std::array<double, 9>;
using Quaternion = std::array<double, 4>;
using Vector3 = std::array<double, 3>;

bool valid_relative_frame(const std::string & frame)
{
  if (frame.empty() || frame.front() == '/') {
    return false;
  }
  return std::none_of(frame.begin(), frame.end(), [](const char character) {
    return std::isspace(static_cast<unsigned char>(character)) != 0;
  });
}

Quaternion normalized_quaternion(
  const Quaternion & quaternion, const char * transform_name)
{
  if (!std::all_of(
      quaternion.begin(), quaternion.end(), [](const double value) {
        return std::isfinite(value);
      }))
  {
    throw std::invalid_argument(
            std::string(transform_name) + " quaternion must be finite");
  }
  const auto scale = std::max({
      std::abs(quaternion[0]), std::abs(quaternion[1]),
      std::abs(quaternion[2]), std::abs(quaternion[3])});
  if (scale == 0.0) {
    throw std::invalid_argument(
            std::string(transform_name) + " quaternion must be nonzero");
  }
  Quaternion result{
    quaternion[0] / scale, quaternion[1] / scale,
    quaternion[2] / scale, quaternion[3] / scale};
  const auto norm = std::sqrt(
    result[0] * result[0] + result[1] * result[1] +
    result[2] * result[2] + result[3] * result[3]);
  for (auto & value : result) {
    value /= norm;
  }
  return result;
}

void validate_translation(const Vector3 & translation, const char * transform_name)
{
  if (!std::all_of(
      translation.begin(), translation.end(), [](const double value) {
        return std::isfinite(value);
      }))
  {
    throw std::invalid_argument(
            std::string(transform_name) + " translation must be finite");
  }
}

Quaternion multiply(const Quaternion & left, const Quaternion & right)
{
  const auto lx = left[0];
  const auto ly = left[1];
  const auto lz = left[2];
  const auto lw = left[3];
  const auto rx = right[0];
  const auto ry = right[1];
  const auto rz = right[2];
  const auto rw = right[3];
  return {
    lw * rx + lx * rw + ly * rz - lz * ry,
    lw * ry - lx * rz + ly * rw + lz * rx,
    lw * rz + lx * ry - ly * rx + lz * rw,
    lw * rw - lx * rx - ly * ry - lz * rz,
  };
}

Quaternion conjugate(const Quaternion & quaternion)
{
  return {-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]};
}

Matrix3 rotation_matrix(const Quaternion & quaternion)
{
  const auto x = quaternion[0];
  const auto y = quaternion[1];
  const auto z = quaternion[2];
  const auto w = quaternion[3];
  return {
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
    2.0 * (x * z + y * w), 2.0 * (x * y + z * w),
    1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
    2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
    1.0 - 2.0 * (x * x + y * y),
  };
}

Vector3 rotate(const Matrix3 & rotation, const Vector3 & point)
{
  return {
    rotation[0] * point[0] + rotation[1] * point[1] + rotation[2] * point[2],
    rotation[3] * point[0] + rotation[4] * point[1] + rotation[5] * point[2],
    rotation[6] * point[0] + rotation[7] * point[1] + rotation[8] * point[2],
  };
}

bool is_identity(const Matrix3 & rotation)
{
  constexpr Matrix3 identity{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  for (std::size_t index = 0U; index < rotation.size(); ++index) {
    if (std::abs(rotation[index] - identity[index]) > 1.0e-14) {
      return false;
    }
  }
  return true;
}

void write_float(
  std::vector<std::uint8_t> & bytes, const std::size_t offset, const float value)
{
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

}  // namespace

std::string derive_leveled_frame(const std::string & input_frame)
{
  if (!valid_relative_frame(input_frame)) {
    throw std::invalid_argument("gravity leveler input frame must be a valid relative frame");
  }
  const std::string link_suffix{"_link"};
  if (
    input_frame.size() >= link_suffix.size() &&
    input_frame.compare(
      input_frame.size() - link_suffix.size(), link_suffix.size(), link_suffix) == 0)
  {
    return input_frame.substr(0, input_frame.size() - link_suffix.size()) + "_leveled_frame";
  }
  return input_frame + "_leveled_frame";
}

GravityLevelingResult level_xyzirt_cloud(
  const sensor_msgs::msg::PointCloud2 & input,
  const GravityLevelingTransform & odom_from_base,
  const GravityLevelingTransform & base_from_lidar,
  const std::string & leveled_frame)
{
  const auto derived_frame = derive_leveled_frame(input.header.frame_id);
  if (!valid_relative_frame(leveled_frame) || leveled_frame != derived_frame) {
    throw std::invalid_argument(
            "gravity leveler output frame must exactly match the derived input frame");
  }
  validate_translation(odom_from_base.translation, "odom-from-base");
  validate_translation(base_from_lidar.translation, "base-from-lidar");
  const auto q_odom_base = normalized_quaternion(
    odom_from_base.quaternion_xyzw, "odom-from-base");
  const auto q_base_lidar = normalized_quaternion(
    base_from_lidar.quaternion_xyzw, "base-from-lidar");
  const auto q_odom_lidar = normalized_quaternion(
    multiply(q_odom_base, q_base_lidar), "odom-from-lidar");
  const auto odom_lidar_rotation = rotation_matrix(q_odom_lidar);
  const auto lidar_yaw = std::atan2(
    odom_lidar_rotation[3], odom_lidar_rotation[0]);
  const Quaternion q_odom_level{
    0.0, 0.0, std::sin(lidar_yaw * 0.5), std::cos(lidar_yaw * 0.5)};
  const auto q_level_lidar = normalized_quaternion(
    multiply(conjugate(q_odom_level), q_odom_lidar), "level-from-lidar");
  const auto q_base_level = normalized_quaternion(
    multiply(conjugate(q_odom_base), q_odom_level), "base-from-level");
  const auto level_lidar_rotation = rotation_matrix(q_level_lidar);

  const XyzirtCloudView view(input);
  std::vector<Vector3> transformed_points;
  transformed_points.reserve(view.size());
  const auto maximum_float = static_cast<double>(std::numeric_limits<float>::max());
  for (std::size_t index = 0U; index < view.size(); ++index) {
    const auto point = view.point(index);
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      throw std::invalid_argument("gravity leveler input contains a nonfinite XYZ coordinate");
    }
    const auto transformed = rotate(
      level_lidar_rotation,
      {static_cast<double>(point.x), static_cast<double>(point.y),
        static_cast<double>(point.z)});
    if (!std::all_of(
        transformed.begin(), transformed.end(),
        [maximum_float](const double value) {
          return std::isfinite(value) && std::abs(value) <= maximum_float;
        }))
    {
      throw std::invalid_argument(
              "gravity leveler transformed XYZ does not fit finite float32");
    }
    transformed_points.push_back(transformed);
  }

  GravityLevelingResult result;
  result.cloud = input;
  result.cloud.header.frame_id = leveled_frame;
  // V and L share their physical origin. Thus ^V t_L is exactly zero and,
  // after inverse(^O T_B) * ^O T_V, ^B t_V is algebraically ^B t_L.
  // Use the simplified values to avoid subtracting nearly equal world positions.
  result.base_from_level.translation = base_from_lidar.translation;
  result.base_from_level.quaternion_xyzw = q_base_level;
  if (is_identity(level_lidar_rotation)) {
    return result;
  }
  for (std::size_t index = 0U; index < view.size(); ++index) {
    const auto offset = view.point_offset(index);
    write_float(
      result.cloud.data, offset + XyzirtCloudView::kXOffset,
      static_cast<float>(transformed_points[index][0]));
    write_float(
      result.cloud.data, offset + XyzirtCloudView::kYOffset,
      static_cast<float>(transformed_points[index][1]));
    write_float(
      result.cloud.data, offset + XyzirtCloudView::kZOffset,
      static_cast<float>(transformed_points[index][2]));
  }
  return result;
}

}  // namespace ad_lidar_perception::preprocessing
