#include "ad_lidar_perception/preprocessing/pointcloud_densifier.hpp"

#include <sensor_msgs/msg/point_field.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ad_lidar_perception::preprocessing
{
namespace
{

using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

struct CloudLayout
{
  std::size_t point_count;
  std::uint32_t x_offset;
  std::uint32_t y_offset;
  std::uint32_t z_offset;
};

struct VoxelKey
{
  std::int64_t x;
  std::int64_t y;
  std::int64_t z;

  bool operator==(const VoxelKey & other) const noexcept
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelHash
{
  std::size_t operator()(const VoxelKey & key) const noexcept
  {
    auto hash = std::hash<std::int64_t>{}(key.x);
    hash ^= std::hash<std::int64_t>{}(key.y) + 0x9e3779b9U + (hash << 6U) +
      (hash >> 2U);
    hash ^= std::hash<std::int64_t>{}(key.z) + 0x9e3779b9U + (hash << 6U) +
      (hash >> 2U);
    return hash;
  }
};

struct HistoricalRecord
{
  std::size_t source_offset;
  std::array<float, 3> xyz;
};

struct DeclaredFieldRange
{
  std::uint64_t begin;
  std::uint64_t end;
  bool is_xyz;
};

std::uint64_t datatype_size(const std::uint8_t datatype)
{
  switch (datatype) {
    case PointField::INT8:
    case PointField::UINT8:
      return 1U;
    case PointField::INT16:
    case PointField::UINT16:
      return 2U;
    case PointField::INT32:
    case PointField::UINT32:
    case PointField::FLOAT32:
      return 4U;
    case PointField::FLOAT64:
      return 8U;
    default:
      throw std::invalid_argument("point field datatype is unsupported");
  }
}

std::uint64_t checked_multiply(
  const std::uint64_t lhs, const std::uint64_t rhs, const char * label)
{
  if (lhs != 0U && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
    throw std::invalid_argument(std::string(label) + " arithmetic overflow");
  }
  return lhs * rhs;
}

bool valid_relative_frame(const std::string & frame)
{
  if (frame.empty() || frame.front() == '/') {
    return false;
  }
  return std::none_of(frame.begin(), frame.end(), [](const char character) {
    return std::isspace(static_cast<unsigned char>(character)) != 0;
  });
}

CloudLayout validate_cloud(const PointCloud2 & cloud)
{
  if (!valid_relative_frame(cloud.header.frame_id)) {
    throw std::invalid_argument("cloud frame must be a valid relative frame");
  }
  if (cloud.header.stamp.sec < 0 || cloud.header.stamp.nanosec >= 1000000000U) {
    throw std::invalid_argument("cloud stamp must be a valid nonnegative ROS time");
  }
  if (cloud.is_bigendian) {
    throw std::invalid_argument("densifier requires a little-endian cloud");
  }
  if (cloud.point_step == 0U) {
    throw std::invalid_argument("point_step must be positive");
  }
  if (cloud.height == 0U && cloud.width != 0U) {
    throw std::invalid_argument("nonempty width requires nonzero height");
  }

  std::array<std::optional<std::uint32_t>, 3> offsets;
  std::vector<DeclaredFieldRange> declared_ranges;
  declared_ranges.reserve(cloud.fields.size());
  std::unordered_set<std::string> declared_names;
  declared_names.reserve(cloud.fields.size());
  for (const auto & field : cloud.fields) {
    if (field.name.empty() || !declared_names.insert(field.name).second) {
      throw std::invalid_argument("point field names must be nonempty and unique");
    }
    std::size_t axis = offsets.size();
    if (field.name == "x") {
      axis = 0U;
    } else if (field.name == "y") {
      axis = 1U;
    } else if (field.name == "z") {
      axis = 2U;
    }
    const bool is_xyz = axis != offsets.size();

    if (field.count == 0U) {
      throw std::invalid_argument("point field count must be positive");
    }
    const auto element_size = datatype_size(field.datatype);
    const auto byte_count = checked_multiply(
      element_size, field.count, "point field byte count");
    const auto begin = static_cast<std::uint64_t>(field.offset);
    if (byte_count > std::numeric_limits<std::uint64_t>::max() - begin) {
      throw std::invalid_argument("point field byte range arithmetic overflow");
    }
    const auto end = begin + byte_count;
    if (end > cloud.point_step) {
      throw std::invalid_argument("point field byte range exceeds point_step");
    }
    for (const auto & existing : declared_ranges) {
      if (
        (is_xyz || existing.is_xyz) && begin < existing.end &&
        existing.begin < end)
      {
        throw std::invalid_argument(
                "x/y/z byte ranges must not overlap any declared field");
      }
    }
    declared_ranges.push_back({begin, end, is_xyz});

    if (!is_xyz) {
      continue;
    }
    if (offsets[axis]) {
      throw std::invalid_argument("x/y/z fields must be unique");
    }
    if (field.datatype != PointField::FLOAT32 || field.count != 1U) {
      throw std::invalid_argument("x/y/z fields must each be one FLOAT32");
    }
    offsets[axis] = field.offset;
  }
  if (!offsets[0] || !offsets[1] || !offsets[2]) {
    throw std::invalid_argument("cloud must contain unique x/y/z FLOAT32 fields");
  }

  const auto minimum_row_step = checked_multiply(
    cloud.width, cloud.point_step, "minimum row_step");
  if (cloud.row_step < minimum_row_step) {
    throw std::invalid_argument("row_step is smaller than width * point_step");
  }
  const auto point_count = checked_multiply(cloud.width, cloud.height, "point count");
  const auto expected_data_size = checked_multiply(
    cloud.row_step, cloud.height, "cloud data size");
  if (
    point_count > std::numeric_limits<std::size_t>::max() ||
    expected_data_size > std::numeric_limits<std::size_t>::max() ||
    cloud.data.size() != static_cast<std::size_t>(expected_data_size))
  {
    throw std::invalid_argument("cloud dimensions and data size are inconsistent");
  }
  return {
    static_cast<std::size_t>(point_count), *offsets[0], *offsets[1], *offsets[2]};
}

std::size_t point_offset(
  const PointCloud2 & cloud, const std::size_t index)
{
  const auto row = index / cloud.width;
  const auto column = index % cloud.width;
  return row * cloud.row_step + column * cloud.point_step;
}

float read_float(const PointCloud2 & cloud, const std::size_t offset)
{
  float value{};
  std::memcpy(&value, cloud.data.data() + offset, sizeof(value));
  return value;
}

std::array<double, 3> read_xyz(
  const PointCloud2 & cloud, const CloudLayout & layout,
  const std::size_t index)
{
  const auto offset = point_offset(cloud, index);
  const std::array<double, 3> point{
    read_float(cloud, offset + layout.x_offset),
    read_float(cloud, offset + layout.y_offset),
    read_float(cloud, offset + layout.z_offset),
  };
  if (!std::all_of(point.begin(), point.end(), [](const double value) {
      return std::isfinite(value);
    }))
  {
    throw std::invalid_argument("cloud contains nonfinite XYZ");
  }
  return point;
}

bool same_schema(const PointCloud2 & left, const PointCloud2 & right)
{
  if (
    left.point_step != right.point_step ||
    left.is_bigendian != right.is_bigendian ||
    left.fields.size() != right.fields.size())
  {
    return false;
  }
  for (std::size_t index = 0U; index < left.fields.size(); ++index) {
    const auto & lhs = left.fields[index];
    const auto & rhs = right.fields[index];
    if (
      lhs.name != rhs.name || lhs.offset != rhs.offset ||
      lhs.datatype != rhs.datatype || lhs.count != rhs.count)
    {
      return false;
    }
  }
  return true;
}

std::int64_t stamp_nanoseconds(const PointCloud2 & cloud)
{
  return static_cast<std::int64_t>(cloud.header.stamp.sec) * 1000000000LL +
         cloud.header.stamp.nanosec;
}

VoxelKey voxel_key(const std::array<double, 3> & point, const double voxel_size)
{
  std::array<std::int64_t, 3> indices{};
  constexpr auto minimum = static_cast<long double>(
    std::numeric_limits<std::int64_t>::lowest());
  constexpr auto maximum = static_cast<long double>(
    std::numeric_limits<std::int64_t>::max());
  for (std::size_t axis = 0U; axis < point.size(); ++axis) {
    const auto quotient = static_cast<long double>(point[axis]) / voxel_size;
    const auto floored = std::floor(quotient);
    if (!std::isfinite(quotient) || floored < minimum || floored > maximum) {
      throw std::overflow_error("voxel index exceeds int64 range");
    }
    indices[axis] = static_cast<std::int64_t>(floored);
  }
  return {indices[0], indices[1], indices[2]};
}

std::array<double, 4> normalized_quaternion(
  const std::array<double, 4> & input)
{
  if (!std::all_of(input.begin(), input.end(), [](const double value) {
      return std::isfinite(value);
    }))
  {
    throw std::invalid_argument("transform quaternion must be finite");
  }
  const auto scale = std::max({
      std::abs(input[0]), std::abs(input[1]), std::abs(input[2]),
      std::abs(input[3])});
  if (scale == 0.0) {
    throw std::invalid_argument("transform quaternion must be nonzero");
  }
  std::array<double, 4> result{
    input[0] / scale, input[1] / scale, input[2] / scale, input[3] / scale};
  const auto norm = std::sqrt(
    result[0] * result[0] + result[1] * result[1] +
    result[2] * result[2] + result[3] * result[3]);
  if (!std::isfinite(norm) || norm == 0.0) {
    throw std::invalid_argument("transform quaternion cannot be normalized");
  }
  for (auto & value : result) {
    value /= norm;
  }
  return result;
}

std::array<double, 3> transform_point(
  const std::array<double, 3> & point,
  const std::array<double, 3> & translation,
  const std::array<double, 4> & quaternion)
{
  const auto x = quaternion[0];
  const auto y = quaternion[1];
  const auto z = quaternion[2];
  const auto w = quaternion[3];
  const std::array<double, 9> rotation{
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
    2.0 * (x * z + y * w), 2.0 * (x * y + z * w),
    1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
    2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
    1.0 - 2.0 * (x * x + y * y),
  };
  return {
    rotation[0] * point[0] + rotation[1] * point[1] +
      rotation[2] * point[2] + translation[0],
    rotation[3] * point[0] + rotation[4] * point[1] +
      rotation[5] * point[2] + translation[1],
    rotation[6] * point[0] + rotation[7] * point[1] +
      rotation[8] * point[2] + translation[2],
  };
}

void validate_config(const DensifierConfig & config)
{
  const std::array<double, 7> values{
    config.voxel_size_m, config.roi_min_x_m, config.roi_max_x_m,
    config.roi_min_y_m, config.roi_max_y_m, config.maximum_history_age_sec,
    config.maximum_translation_jump_m};
  if (!std::all_of(values.begin(), values.end(), [](const double value) {
      return std::isfinite(value);
    }) || !std::isfinite(config.maximum_rotation_jump_rad))
  {
    throw std::invalid_argument("densifier configuration must be finite");
  }
  if (
    config.voxel_size_m <= 0.0 || config.roi_min_x_m > config.roi_max_x_m ||
    config.roi_min_y_m > config.roi_max_y_m ||
    config.maximum_history_age_sec <= 0.0 ||
    config.maximum_translation_jump_m < 0.0 ||
    config.maximum_rotation_jump_rad < 0.0 ||
    config.maximum_rotation_jump_rad > 3.14159265358979323846)
  {
    throw std::invalid_argument("densifier configuration bounds are invalid");
  }
}

}  // namespace

PointcloudDensifier::PointcloudDensifier(DensifierConfig config)
: config_(config)
{
  validate_config(config_);
}

bool PointcloudDensifier::has_history() const noexcept
{
  return history_.has_value();
}

DensifierResult PointcloudDensifier::process(
  const sensor_msgs::msg::PointCloud2 & current,
  const std::optional<DensifierTransform> & current_from_previous)
{
  const auto fallback = [this, &current](const DensifierStatus status) {
      history_ = current;
      return DensifierResult{current, status, 0U};
    };

  CloudLayout current_layout{};
  try {
    current_layout = validate_cloud(current);
    for (std::size_t index = 0U; index < current_layout.point_count; ++index) {
      static_cast<void>(read_xyz(current, current_layout, index));
    }
  } catch (const std::exception &) {
    history_.reset();
    return {current, DensifierStatus::kMalformedCurrent, 0U};
  }
  if (!history_) {
    history_ = current;
    return {current, DensifierStatus::kFirstFrame, 0U};
  }

  CloudLayout history_layout{};
  try {
    history_layout = validate_cloud(*history_);
    for (std::size_t index = 0U; index < history_layout.point_count; ++index) {
      static_cast<void>(read_xyz(*history_, history_layout, index));
    }
  } catch (const std::exception &) {
    return fallback(DensifierStatus::kMalformedHistory);
  }

  const auto current_stamp = stamp_nanoseconds(current);
  const auto history_stamp = stamp_nanoseconds(*history_);
  if (current_stamp <= history_stamp) {
    return fallback(DensifierStatus::kNonIncreasingStamp);
  }
  const auto age_sec = static_cast<double>(current_stamp - history_stamp) * 1.0e-9;
  if (!std::isfinite(age_sec) || age_sec > config_.maximum_history_age_sec) {
    return fallback(DensifierStatus::kStaleHistory);
  }
  if (!same_schema(current, *history_)) {
    return fallback(DensifierStatus::kSchemaMismatch);
  }
  if (current.header.frame_id != history_->header.frame_id) {
    return fallback(DensifierStatus::kFrameMismatch);
  }
  if (!current_from_previous) {
    return fallback(DensifierStatus::kTransformUnavailable);
  }

  std::array<double, 4> quaternion{};
  try {
    if (!std::all_of(
        current_from_previous->translation.begin(),
        current_from_previous->translation.end(), [](const double value) {
          return std::isfinite(value);
        }))
    {
      throw std::invalid_argument("transform translation must be finite");
    }
    quaternion = normalized_quaternion(current_from_previous->quaternion_xyzw);
  } catch (const std::exception &) {
    return fallback(DensifierStatus::kNumericalFailure);
  }

  const auto translation_norm = std::hypot(
    current_from_previous->translation[0],
    current_from_previous->translation[1],
    current_from_previous->translation[2]);
  if (
    !std::isfinite(translation_norm) ||
    translation_norm > config_.maximum_translation_jump_m)
  {
    return fallback(DensifierStatus::kTransformTranslationJump);
  }
  const auto rotation_angle = 2.0 * std::acos(
    std::clamp(std::abs(quaternion[3]), 0.0, 1.0));
  if (!std::isfinite(rotation_angle)) {
    return fallback(DensifierStatus::kNumericalFailure);
  }
  if (rotation_angle > config_.maximum_rotation_jump_rad) {
    return fallback(DensifierStatus::kTransformRotationJump);
  }

  std::unordered_set<VoxelKey, VoxelHash> claimed_voxels;
  std::vector<HistoricalRecord> accepted_history;
  try {
    claimed_voxels.reserve(
      current_layout.point_count + history_layout.point_count);
    for (std::size_t index = 0U; index < current_layout.point_count; ++index) {
      claimed_voxels.insert(
        voxel_key(
          read_xyz(current, current_layout, index), config_.voxel_size_m));
    }
    accepted_history.reserve(history_layout.point_count);
    const auto maximum_float = static_cast<double>(std::numeric_limits<float>::max());
    for (std::size_t index = 0U; index < history_layout.point_count; ++index) {
      const auto transformed = transform_point(
        read_xyz(*history_, history_layout, index),
        current_from_previous->translation, quaternion);
      if (!std::all_of(
          transformed.begin(), transformed.end(),
          [maximum_float](const double value) {
            return std::isfinite(value) && std::abs(value) <= maximum_float;
          }))
      {
        throw std::overflow_error("transformed XYZ does not fit finite float32");
      }
      if (
        transformed[0] < config_.roi_min_x_m ||
        transformed[0] > config_.roi_max_x_m ||
        transformed[1] < config_.roi_min_y_m ||
        transformed[1] > config_.roi_max_y_m)
      {
        continue;
      }
      if (
        claimed_voxels.insert(
          voxel_key(transformed, config_.voxel_size_m)).second)
      {
        accepted_history.push_back({
            point_offset(*history_, index),
            {static_cast<float>(transformed[0]), static_cast<float>(transformed[1]),
              static_cast<float>(transformed[2])}});
      }
    }
  } catch (const std::exception &) {
    return fallback(DensifierStatus::kNumericalFailure);
  }

  if (accepted_history.empty()) {
    return fallback(DensifierStatus::kNoEligibleHistory);
  }

  const auto output_count = current_layout.point_count + accepted_history.size();
  if (
    output_count < current_layout.point_count ||
    output_count > std::numeric_limits<std::uint32_t>::max() ||
    output_count > std::numeric_limits<std::uint32_t>::max() / current.point_step ||
    (current.point_step != 0U &&
    output_count > std::numeric_limits<std::size_t>::max() / current.point_step))
  {
    return fallback(DensifierStatus::kNumericalFailure);
  }

  try {
    auto output = current;
    output.height = 1U;
    output.width = static_cast<std::uint32_t>(output_count);
    output.row_step = static_cast<std::uint32_t>(
      output_count * current.point_step);
    output.data.clear();
    output.data.reserve(static_cast<std::size_t>(output.row_step));
    for (std::size_t index = 0U; index < current_layout.point_count; ++index) {
      const auto offset = point_offset(current, index);
      output.data.insert(
        output.data.end(), current.data.begin() + offset,
        current.data.begin() + offset + current.point_step);
    }
    for (const auto & record : accepted_history) {
      const auto destination = output.data.size();
      output.data.insert(
        output.data.end(), history_->data.begin() + record.source_offset,
        history_->data.begin() + record.source_offset + history_->point_step);
      std::memcpy(
        output.data.data() + destination + current_layout.x_offset,
        &record.xyz[0], sizeof(float));
      std::memcpy(
        output.data.data() + destination + current_layout.y_offset,
        &record.xyz[1], sizeof(float));
      std::memcpy(
        output.data.data() + destination + current_layout.z_offset,
        &record.xyz[2], sizeof(float));
    }
    history_ = current;
    return {
      std::move(output), DensifierStatus::kFused, accepted_history.size()};
  } catch (const std::exception &) {
    return fallback(DensifierStatus::kNumericalFailure);
  }
}

}  // namespace ad_lidar_perception::preprocessing
