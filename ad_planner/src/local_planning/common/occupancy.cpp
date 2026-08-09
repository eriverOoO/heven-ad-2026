#include "ad_planner/local_planning/common/occupancy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ad_planner
{
namespace
{

bool intervals_intersect(
  long double projection, long double footprint_radius, long double cell_radius)
{
  const long double scale = std::max(
    {1.0L, std::abs(projection), footprint_radius, cell_radius});
  const long double tolerance =
    32.0L * std::numeric_limits<long double>::epsilon() * scale;
  return std::abs(projection) <= footprint_radius + cell_radius + tolerance;
}

bool rectangle_intersects_cell(
  long double center_x, long double center_y,
  long double length_axis_x, long double length_axis_y,
  long double width_axis_x, long double width_axis_y,
  long double half_length, long double half_width,
  long double cell_center_x, long double cell_center_y, long double cell_half)
{
  const long double dx = cell_center_x - center_x;
  const long double dy = cell_center_y - center_y;
  if (!intervals_intersect(
      dx, half_length * std::abs(length_axis_x) +
      half_width * std::abs(width_axis_x), cell_half))
  {
    return false;
  }
  if (!intervals_intersect(
      dy, half_length * std::abs(length_axis_y) +
      half_width * std::abs(width_axis_y), cell_half))
  {
    return false;
  }
  if (!intervals_intersect(
      dx * length_axis_x + dy * length_axis_y, half_length,
      cell_half * (std::abs(length_axis_x) + std::abs(length_axis_y))))
  {
    return false;
  }
  return intervals_intersect(
    dx * width_axis_x + dy * width_axis_y, half_width,
    cell_half * (std::abs(width_axis_x) + std::abs(width_axis_y)));
}

}  // namespace

GridValidation validate_occupancy_grid(const OccupancyGrid & grid)
{
  if (!grid.valid || !grid.fresh) {return {false, "grid invalid or stale"};}
  if (!std::isfinite(grid.origin.x) || !std::isfinite(grid.origin.y) ||
    !std::isfinite(grid.origin.yaw_rad) || !std::isfinite(grid.resolution) ||
    grid.resolution <= 0.0 || grid.width == 0 || grid.height == 0)
  {
    return {false, "invalid grid metadata"};
  }
  if (grid.width > std::numeric_limits<std::size_t>::max() / grid.height) {
    return {false, "grid dimensions overflow"};
  }
  if (grid.cells.size() != grid.width * grid.height) {
    return {false, "grid data size mismatch"};
  }
  return {true, "ok"};
}

std::optional<GridCell> world_to_cell(const OccupancyGrid & grid, const Point3 & point)
{
  if (!validate_occupancy_grid(grid).valid || !std::isfinite(point.x) ||
    !std::isfinite(point.y))
  {
    return std::nullopt;
  }
  const double dx = point.x - grid.origin.x;
  const double dy = point.y - grid.origin.y;
  const double cosine = std::cos(grid.origin.yaw_rad);
  const double sine = std::sin(grid.origin.yaw_rad);
  const double local_x = cosine * dx + sine * dy;
  const double local_y = -sine * dx + cosine * dy;
  if (!std::isfinite(local_x) || !std::isfinite(local_y)) {return std::nullopt;}
  if (local_x < 0.0 || local_y < 0.0) {return std::nullopt;}
  const double grid_x = std::floor(local_x / grid.resolution);
  const double grid_y = std::floor(local_y / grid.resolution);
  if (grid_x >= static_cast<double>(grid.width) ||
    grid_y >= static_cast<double>(grid.height))
  {
    return std::nullopt;
  }
  return GridCell{static_cast<std::size_t>(grid_x), static_cast<std::size_t>(grid_y)};
}

std::int8_t cell_value(const OccupancyGrid & grid, const Point3 & point)
{
  const auto cell = world_to_cell(grid, point);
  if (!cell) {return -1;}
  return grid.cells[cell->y * grid.width + cell->x];
}

bool footprint_is_safe(
  const OccupancyGrid & grid, const Pose2 & pose, const FootprintConfig & footprint)
{
  if (!validate_occupancy_grid(grid).valid || !std::isfinite(pose.x) ||
    !std::isfinite(pose.y) || !std::isfinite(pose.yaw_rad) ||
    !std::isfinite(footprint.half_length_m) || !std::isfinite(footprint.half_width_m) ||
    !std::isfinite(footprint.clearance_m) ||
    !std::isfinite(footprint.center_offset_x_m) || footprint.half_length_m < 0.0 ||
    footprint.half_width_m < 0.0 || footprint.clearance_m < 0.0 ||
    footprint.occupied_threshold < 0 ||
    footprint.occupied_threshold > 100 ||
    footprint.maximum_cells_to_check == 0)
  {
    return false;
  }
  const long double half_length = static_cast<long double>(footprint.half_length_m) +
    static_cast<long double>(footprint.clearance_m);
  const long double half_width = static_cast<long double>(footprint.half_width_m) +
    static_cast<long double>(footprint.clearance_m);
  const long double resolution = static_cast<long double>(grid.resolution);
  const long double grid_yaw = static_cast<long double>(grid.origin.yaw_rad);
  const long double relative_yaw = static_cast<long double>(pose.yaw_rad) - grid_yaw;
  const long double length_axis_x = std::cos(relative_yaw);
  const long double length_axis_y = std::sin(relative_yaw);
  const long double width_axis_x = -length_axis_y;
  const long double width_axis_y = length_axis_x;
  const long double dx = static_cast<long double>(pose.x) - grid.origin.x;
  const long double dy = static_cast<long double>(pose.y) - grid.origin.y;
  const long double grid_cosine = std::cos(grid_yaw);
  const long double grid_sine = std::sin(grid_yaw);
  const long double center_offset = static_cast<long double>(footprint.center_offset_x_m);
  const long double center_x =
    grid_cosine * dx + grid_sine * dy + center_offset * length_axis_x;
  const long double center_y =
    -grid_sine * dx + grid_cosine * dy + center_offset * length_axis_y;
  const long double extent_x = half_length * std::abs(length_axis_x) +
    half_width * std::abs(width_axis_x);
  const long double extent_y = half_length * std::abs(length_axis_y) +
    half_width * std::abs(width_axis_y);
  const long double minimum_x = center_x - extent_x;
  const long double maximum_x = center_x + extent_x;
  const long double minimum_y = center_y - extent_y;
  const long double maximum_y = center_y + extent_y;
  const long double map_width = static_cast<long double>(grid.width) * resolution;
  const long double map_height = static_cast<long double>(grid.height) * resolution;
  const long double geometry[] = {
    half_length, half_width, center_offset, length_axis_x, length_axis_y, center_x, center_y,
    extent_x, extent_y, minimum_x, maximum_x, minimum_y, maximum_y,
    map_width, map_height};
  for (const long double value : geometry) {
    if (!std::isfinite(value)) {return false;}
  }
  if (minimum_x <= 0.0L || minimum_y <= 0.0L ||
    maximum_x >= map_width || maximum_y >= map_height)
  {
    return false;
  }

  const long double minimum_grid_x = std::floor(std::nextafter(
      minimum_x / resolution, -std::numeric_limits<long double>::infinity()));
  const long double maximum_grid_x = std::floor(std::nextafter(
      maximum_x / resolution, std::numeric_limits<long double>::infinity()));
  const long double minimum_grid_y = std::floor(std::nextafter(
      minimum_y / resolution, -std::numeric_limits<long double>::infinity()));
  const long double maximum_grid_y = std::floor(std::nextafter(
      maximum_y / resolution, std::numeric_limits<long double>::infinity()));
  if (!std::isfinite(minimum_grid_x) || !std::isfinite(maximum_grid_x) ||
    !std::isfinite(minimum_grid_y) || !std::isfinite(maximum_grid_y) ||
    minimum_grid_x < 0.0L || minimum_grid_y < 0.0L ||
    maximum_grid_x >= static_cast<long double>(grid.width) ||
    maximum_grid_y >= static_cast<long double>(grid.height))
  {
    return false;
  }
  const std::size_t first_x = static_cast<std::size_t>(minimum_grid_x);
  const std::size_t last_x = static_cast<std::size_t>(maximum_grid_x);
  const std::size_t first_y = static_cast<std::size_t>(minimum_grid_y);
  const std::size_t last_y = static_cast<std::size_t>(maximum_grid_y);
  const std::size_t columns = last_x - first_x + 1U;
  const std::size_t rows = last_y - first_y + 1U;
  if (columns > footprint.maximum_cells_to_check / rows) {return false;}
  const std::size_t candidate_cells = columns * rows;
  if (candidate_cells > footprint.maximum_cells_to_check) {return false;}

  const long double cell_half = resolution * 0.5L;
  for (std::size_t y = first_y;; ++y) {
    for (std::size_t x = first_x;; ++x) {
      const long double cell_center_x =
        (static_cast<long double>(x) + 0.5L) * resolution;
      const long double cell_center_y =
        (static_cast<long double>(y) + 0.5L) * resolution;
      if (rectangle_intersects_cell(
          center_x, center_y, length_axis_x, length_axis_y,
          width_axis_x, width_axis_y, half_length, half_width,
          cell_center_x, cell_center_y, cell_half))
      {
        const std::int8_t value = grid.cells[y * grid.width + x];
        if (value < 0 || value >= footprint.occupied_threshold) {return false;}
      }
      if (x == last_x) {break;}
    }
    if (y == last_y) {break;}
  }
  return true;
}

double nearest_unsafe_clearance(
  const OccupancyGrid & grid, const Point3 & point, std::int8_t occupied_threshold,
  double maximum_distance_m)
{
  if (!validate_occupancy_grid(grid).valid || !std::isfinite(point.x) ||
    !std::isfinite(point.y) || !std::isfinite(maximum_distance_m) ||
    maximum_distance_m < 0.0 || occupied_threshold < 0 ||
    occupied_threshold > 100)
  {
    return 0.0;
  }
  double best = maximum_distance_m;
  const double cosine = std::cos(grid.origin.yaw_rad);
  const double sine = std::sin(grid.origin.yaw_rad);
  for (std::size_t y = 0; y < grid.height; ++y) {
    for (std::size_t x = 0; x < grid.width; ++x) {
      const std::int8_t value = grid.cells[y * grid.width + x];
      if (value >= 0 && value < occupied_threshold) {continue;}
      const double local_x = (static_cast<double>(x) + 0.5) * grid.resolution;
      const double local_y = (static_cast<double>(y) + 0.5) * grid.resolution;
      const double world_x = grid.origin.x + cosine * local_x - sine * local_y;
      const double world_y = grid.origin.y + sine * local_x + cosine * local_y;
      best = std::min(best, std::hypot(world_x - point.x, world_y - point.y));
    }
  }
  return best;
}

}  // namespace ad_planner
