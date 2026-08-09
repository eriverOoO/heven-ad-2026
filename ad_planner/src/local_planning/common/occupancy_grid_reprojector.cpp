#include "ad_planner/local_planning/common/occupancy_grid_reprojector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <vector>

#include "ad_planner/local_planning/common/occupancy.hpp"

namespace ad_planner
{
namespace
{

constexpr double kQuaternionNormTolerance = 1.0e-6;
constexpr double kPlanarAngleToleranceRad = 0.05;
constexpr long double kCellRatioToleranceScale = 64.0L;

struct OutputShape
{
  std::size_t width{0U};
  std::size_t height{0U};
  std::size_t cell_count{0U};
  double resolution{0.0};
};

struct OutputGeometry
{
  double origin_x{0.0};
  double origin_y{0.0};
  double first_center_x{0.0};
  double first_center_y{0.0};
  double last_center_x{0.0};
  double last_center_y{0.0};
};

bool finite(const double value)
{
  return std::isfinite(value);
}

bool checked_double(const long double value, double & output)
{
  if (!std::isfinite(value) ||
    value > static_cast<long double>(std::numeric_limits<double>::max()) ||
    value < -static_cast<long double>(std::numeric_limits<double>::max()))
  {
    return false;
  }
  output = static_cast<double>(value);
  return std::isfinite(output);
}

bool checked_offset(const double base, const long double offset, double & output)
{
  if (!finite(base) || !std::isfinite(offset)) {
    return false;
  }
  const long double base_wide = static_cast<long double>(base);
  const long double maximum =
    static_cast<long double>(std::numeric_limits<double>::max());
  if ((offset > 0.0L && offset > maximum - base_wide) ||
    (offset < 0.0L && -offset > base_wide + maximum))
  {
    return false;
  }
  return checked_double(base_wide + offset, output);
}

std::optional<std::size_t> checked_cell_dimension(
  const double extent_m, const double resolution_m)
{
  if (!finite(extent_m) || !finite(resolution_m) ||
    extent_m <= 0.0 || resolution_m <= 0.0)
  {
    return std::nullopt;
  }
  const long double ratio =
    static_cast<long double>(extent_m) / static_cast<long double>(resolution_m);
  if (!std::isfinite(ratio)) {
    return std::nullopt;
  }
  const long double rounded = std::round(ratio);
  const long double tolerance =
    kCellRatioToleranceScale *
    static_cast<long double>(std::numeric_limits<double>::epsilon()) *
    std::max(1.0L, std::abs(ratio));
  if (std::abs(ratio - rounded) > tolerance || rounded < 1.0L ||
    rounded > static_cast<long double>(std::numeric_limits<std::uint32_t>::max()))
  {
    return std::nullopt;
  }
  return static_cast<std::size_t>(rounded);
}

std::optional<OutputShape> checked_output_shape(
  const GridReprojectionConfig & config)
{
  if (config.outside_value != -1 || !finite(config.resolution_m) ||
    config.resolution_m <= 0.0)
  {
    return std::nullopt;
  }
  const float ros_resolution = static_cast<float>(config.resolution_m);
  if (!std::isfinite(ros_resolution) || ros_resolution <= 0.0F) {
    return std::nullopt;
  }
  // Cell counts follow the requested double ratio. Geometry then uses the
  // exact effective float32 resolution that the ROS message can advertise.
  const auto width = checked_cell_dimension(config.width_m, config.resolution_m);
  const auto height = checked_cell_dimension(config.height_m, config.resolution_m);
  if (!width || !height ||
    *width > std::numeric_limits<std::size_t>::max() / *height)
  {
    return std::nullopt;
  }
  const std::size_t cell_count = *width * *height;
  const std::vector<std::int8_t> allocation_probe;
  if (cell_count > allocation_probe.max_size()) {
    return std::nullopt;
  }
  return OutputShape{
    *width, *height, cell_count, static_cast<double>(ros_resolution)};
}

std::optional<OutputGeometry> checked_output_geometry(
  const Point2 & ego, const OutputShape & shape)
{
  if (!finite(ego.x) || !finite(ego.y)) {
    return std::nullopt;
  }
  const long double resolution_wide =
    static_cast<long double>(shape.resolution);
  const long double width_span =
    static_cast<long double>(shape.width) * resolution_wide;
  const long double height_span =
    static_cast<long double>(shape.height) * resolution_wide;
  if (!std::isfinite(width_span) || !std::isfinite(height_span) ||
    width_span <= 0.0L || height_span <= 0.0L)
  {
    return std::nullopt;
  }

  OutputGeometry geometry;
  if (!checked_offset(ego.x, -0.5L * width_span, geometry.origin_x) ||
    !checked_offset(ego.y, -0.5L * height_span, geometry.origin_y) ||
    !checked_offset(
      ego.x, -0.5L * width_span + 0.5L * resolution_wide,
      geometry.first_center_x) ||
    !checked_offset(
      ego.y, -0.5L * height_span + 0.5L * resolution_wide,
      geometry.first_center_y) ||
    !checked_offset(
      ego.x, 0.5L * width_span - 0.5L * resolution_wide,
      geometry.last_center_x) ||
    !checked_offset(
      ego.y, 0.5L * height_span - 0.5L * resolution_wide,
      geometry.last_center_y))
  {
    return std::nullopt;
  }
  return geometry;
}

bool valid_source_grid(const OccupancyGrid & source)
{
  if (!validate_occupancy_grid(source).valid) {
    return false;
  }
  if (source.width > std::numeric_limits<std::uint32_t>::max() ||
    source.height > std::numeric_limits<std::uint32_t>::max())
  {
    return false;
  }
  for (const std::int8_t value : source.cells) {
    if (value < -1 || value > 100) {
      return false;
    }
  }

  const long double span_x =
    static_cast<long double>(source.width) *
    static_cast<long double>(source.resolution);
  const long double span_y =
    static_cast<long double>(source.height) *
    static_cast<long double>(source.resolution);
  if (!std::isfinite(span_x) || !std::isfinite(span_y) ||
    span_x <= 0.0L || span_y <= 0.0L)
  {
    return false;
  }
  const long double cosine = std::cos(
    static_cast<long double>(source.origin.yaw_rad));
  const long double sine = std::sin(
    static_cast<long double>(source.origin.yaw_rad));
  for (const long double x : {0.0L, span_x}) {
    for (const long double y : {0.0L, span_y}) {
      double corner_x = 0.0;
      double corner_y = 0.0;
      if (!checked_double(
          static_cast<long double>(source.origin.x) + cosine * x - sine * y,
          corner_x) ||
        !checked_double(
          static_cast<long double>(source.origin.y) + sine * x + cosine * y,
          corner_y))
      {
        return false;
      }
    }
  }
  return true;
}

bool inverse_sample_coordinates(
  const double output_x, const double output_y,
  const FrameTransform2 & odom_from_source,
  const long double transform_cosine, const long double transform_sine,
  const OccupancyGrid & source,
  const long double source_cosine, const long double source_sine,
  long double & local_x, long double & local_y)
{
  const long double dx =
    static_cast<long double>(output_x) -
    static_cast<long double>(odom_from_source.x_m);
  const long double dy =
    static_cast<long double>(output_y) -
    static_cast<long double>(odom_from_source.y_m);
  const long double source_x = transform_cosine * dx + transform_sine * dy;
  const long double source_y = -transform_sine * dx + transform_cosine * dy;
  const long double origin_dx =
    source_x - static_cast<long double>(source.origin.x);
  const long double origin_dy =
    source_y - static_cast<long double>(source.origin.y);
  local_x = source_cosine * origin_dx + source_sine * origin_dy;
  local_y = -source_sine * origin_dx + source_cosine * origin_dy;
  double checked_x = 0.0;
  double checked_y = 0.0;
  return checked_double(local_x, checked_x) && checked_double(local_y, checked_y);
}

bool preflight_inverse_coordinates(
  const OutputGeometry & geometry,
  const FrameTransform2 & odom_from_source,
  const OccupancyGrid & source,
  const long double transform_cosine, const long double transform_sine,
  const long double source_cosine, const long double source_sine)
{
  for (const double x : {geometry.first_center_x, geometry.last_center_x}) {
    for (const double y : {geometry.first_center_y, geometry.last_center_y}) {
      long double local_x = 0.0L;
      long double local_y = 0.0L;
      if (!inverse_sample_coordinates(
          x, y, odom_from_source, transform_cosine, transform_sine,
          source, source_cosine, source_sine, local_x, local_y))
      {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

std::optional<double> planar_yaw_from_quaternion(
  const QuaternionComponents & quaternion)
{
  if (!finite(quaternion.x) || !finite(quaternion.y) ||
    !finite(quaternion.z) || !finite(quaternion.w))
  {
    return std::nullopt;
  }
  const double norm = std::hypot(
    std::hypot(quaternion.x, quaternion.y),
    std::hypot(quaternion.z, quaternion.w));
  if (!finite(norm) || norm == 0.0 ||
    std::abs(norm - 1.0) > kQuaternionNormTolerance)
  {
    return std::nullopt;
  }
  const double x = quaternion.x / norm;
  const double y = quaternion.y / norm;
  const double z = quaternion.z / norm;
  const double w = quaternion.w / norm;
  const double roll = std::atan2(
    2.0 * (w * x + y * z),
    1.0 - 2.0 * (x * x + y * y));
  const double pitch_sine = std::clamp(2.0 * (w * y - z * x), -1.0, 1.0);
  const double pitch = std::asin(pitch_sine);
  if (!finite(roll) || !finite(pitch) ||
    std::abs(roll) > kPlanarAngleToleranceRad ||
    std::abs(pitch) > kPlanarAngleToleranceRad)
  {
    return std::nullopt;
  }
  const double yaw = std::atan2(
    2.0 * (w * z + x * y),
    1.0 - 2.0 * (y * y + z * z));
  return finite(yaw) ? std::optional<double>{yaw} : std::nullopt;
}

std::optional<OccupancyGrid> reproject_occupancy_grid(
  const OccupancyGrid & source,
  const FrameTransform2 & odom_from_source,
  const Point2 & ego_in_odom,
  const GridReprojectionConfig & config)
{
  if (!valid_source_grid(source) ||
    !finite(odom_from_source.x_m) || !finite(odom_from_source.y_m) ||
    !finite(odom_from_source.yaw_rad))
  {
    return std::nullopt;
  }
  const auto shape = checked_output_shape(config);
  if (!shape) {
    return std::nullopt;
  }
  const auto geometry = checked_output_geometry(
    ego_in_odom, *shape);
  if (!geometry) {
    return std::nullopt;
  }

  const long double transform_cosine = std::cos(
    static_cast<long double>(odom_from_source.yaw_rad));
  const long double transform_sine = std::sin(
    static_cast<long double>(odom_from_source.yaw_rad));
  const long double source_cosine = std::cos(
    static_cast<long double>(source.origin.yaw_rad));
  const long double source_sine = std::sin(
    static_cast<long double>(source.origin.yaw_rad));
  if (!std::isfinite(transform_cosine) || !std::isfinite(transform_sine) ||
    !std::isfinite(source_cosine) || !std::isfinite(source_sine) ||
    !preflight_inverse_coordinates(
      *geometry, odom_from_source, source, transform_cosine, transform_sine,
      source_cosine, source_sine))
  {
    return std::nullopt;
  }

  const long double source_span_x =
    static_cast<long double>(source.width) *
    static_cast<long double>(source.resolution);
  const long double source_span_y =
    static_cast<long double>(source.height) *
    static_cast<long double>(source.resolution);
  OccupancyGrid output;
  output.origin = Pose2{geometry->origin_x, geometry->origin_y, 0.0};
  output.resolution = shape->resolution;
  output.width = shape->width;
  output.height = shape->height;
  output.valid = true;
  output.fresh = true;
  try {
    output.cells.assign(shape->cell_count, config.outside_value);
  } catch (const std::bad_alloc &) {
    return std::nullopt;
  } catch (const std::length_error &) {
    return std::nullopt;
  }

  const long double output_resolution =
    static_cast<long double>(shape->resolution);
  for (std::size_t y = 0U; y < shape->height; ++y) {
    double output_y = 0.0;
    if (!checked_double(
        static_cast<long double>(geometry->origin_y) +
        (static_cast<long double>(y) + 0.5L) * output_resolution,
        output_y))
    {
      return std::nullopt;
    }
    for (std::size_t x = 0U; x < shape->width; ++x) {
      double output_x = 0.0;
      if (!checked_double(
          static_cast<long double>(geometry->origin_x) +
          (static_cast<long double>(x) + 0.5L) * output_resolution,
          output_x))
      {
        return std::nullopt;
      }
      long double local_x = 0.0L;
      long double local_y = 0.0L;
      if (!inverse_sample_coordinates(
          output_x, output_y, odom_from_source,
          transform_cosine, transform_sine, source, source_cosine, source_sine,
          local_x, local_y))
      {
        return std::nullopt;
      }
      if (local_x < 0.0L || local_y < 0.0L ||
        local_x >= source_span_x || local_y >= source_span_y)
      {
        continue;
      }
      const long double source_x = std::floor(
        local_x / static_cast<long double>(source.resolution));
      const long double source_y = std::floor(
        local_y / static_cast<long double>(source.resolution));
      if (!std::isfinite(source_x) || !std::isfinite(source_y) ||
        source_x < 0.0L || source_y < 0.0L ||
        source_x >= static_cast<long double>(source.width) ||
        source_y >= static_cast<long double>(source.height))
      {
        continue;
      }
      const std::size_t source_index =
        static_cast<std::size_t>(source_y) * source.width +
        static_cast<std::size_t>(source_x);
      output.cells[y * shape->width + x] = source.cells[source_index];
    }
  }
  return output;
}

}  // namespace ad_planner
