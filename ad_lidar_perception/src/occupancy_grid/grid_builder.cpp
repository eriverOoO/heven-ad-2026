#include "ad_lidar_perception/occupancy_grid/grid_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ad_lidar_perception::occupancy_grid
{

GridBuilder::GridBuilder(GridConfig config)
: config_(std::move(config)), width_(0U), height_(0U)
{
  if (!std::isfinite(config_.x_min) || !std::isfinite(config_.x_max) ||
    !std::isfinite(config_.y_min) || !std::isfinite(config_.y_max) ||
    !std::isfinite(config_.z_min) || !std::isfinite(config_.z_max) ||
    !std::isfinite(config_.resolution) ||
    !std::isfinite(config_.inflation_radius_m) ||
    !std::isfinite(config_.inflation_cost_scaling_factor) ||
    !std::isfinite(config_.ego_clear_x_min) ||
    !std::isfinite(config_.ego_clear_x_max) ||
    !std::isfinite(config_.ego_clear_y_min) ||
    !std::isfinite(config_.ego_clear_y_max) ||
    config_.resolution <= 0.0 || config_.x_max <= config_.x_min ||
    config_.y_max <= config_.y_min || config_.z_max < config_.z_min ||
    config_.ego_clear_x_max < config_.ego_clear_x_min ||
    config_.ego_clear_y_max < config_.ego_clear_y_min ||
    config_.inflation_radius_m < 0.0 ||
    config_.inflation_cost_scaling_factor <= 0.0)
  {
    throw std::invalid_argument("invalid occupancy-grid configuration");
  }
  config_.resolution = static_cast<double>(static_cast<float>(config_.resolution));
  const double width_cells =
    (config_.x_max - config_.x_min) / config_.resolution;
  const double height_cells =
    (config_.y_max - config_.y_min) / config_.resolution;
  const double inflation_cells =
    config_.inflation_radius_m / config_.resolution;
  const double rounded_width_cells = std::round(width_cells);
  const double rounded_height_cells = std::round(height_cells);
  const double width_tolerance =
    8.0 * static_cast<double>(std::numeric_limits<float>::epsilon()) *
    std::max(1.0, std::abs(width_cells));
  const double height_tolerance =
    8.0 * static_cast<double>(std::numeric_limits<float>::epsilon()) *
    std::max(1.0, std::abs(height_cells));
  const double maximum_dimension =
    static_cast<double>(std::numeric_limits<int>::max());
  if (!std::isfinite(config_.resolution) || config_.resolution <= 0.0 ||
    !std::isfinite(width_cells) || !std::isfinite(height_cells) ||
    !std::isfinite(inflation_cells) ||
    width_cells < 0.5 || height_cells < 0.5 ||
    width_cells > maximum_dimension || height_cells > maximum_dimension ||
    std::abs(width_cells - rounded_width_cells) > width_tolerance ||
    std::abs(height_cells - rounded_height_cells) > height_tolerance ||
    inflation_cells >= static_cast<double>(std::numeric_limits<int>::max()))
  {
    throw std::invalid_argument(
            "occupancy-grid dimensions or inflation are not representable");
  }
  width_ = static_cast<std::size_t>(rounded_width_cells);
  height_ = static_cast<std::size_t>(rounded_height_cells);
  if (width_ == 0U || height_ == 0U ||
    width_ > std::numeric_limits<std::size_t>::max() / height_)
  {
    throw std::invalid_argument("occupancy grid has an invalid cell count");
  }
  const auto cell_count = width_ * height_;
  if (cell_count > std::vector<std::int8_t>{}.max_size() ||
    cell_count > std::vector<bool>{}.max_size())
  {
    throw std::invalid_argument(
            "occupancy-grid allocation size is not representable");
  }

  const std::int64_t radius = static_cast<std::int64_t>(
    std::ceil(config_.inflation_radius_m / config_.resolution));
  const std::int64_t maximum_dx = std::min(
    radius, static_cast<std::int64_t>(width_ - 1U));
  const std::int64_t maximum_dy = std::min(
    radius, static_cast<std::int64_t>(height_ - 1U));
  for (std::int64_t dy = -maximum_dy; dy <= maximum_dy; ++dy) {
    for (std::int64_t dx = -maximum_dx; dx <= maximum_dx; ++dx) {
      const double distance_m =
        std::hypot(static_cast<double>(dx), static_cast<double>(dy)) *
        config_.resolution;
      if (distance_m > config_.inflation_radius_m) {
        continue;
      }
      const double decayed_cost = 100.0 * std::exp(
        -config_.inflation_cost_scaling_factor * distance_m);
      const auto cost = std::clamp<std::int64_t>(
        std::llround(decayed_cost), 1, 100);
      inflation_kernel_.push_back(
        InflationOffset{dx, dy, static_cast<std::int8_t>(cost)});
    }
  }
}

std::vector<std::int8_t> GridBuilder::build(const std::vector<Point3> & points) const
{
  return build_impl(points, nullptr);
}

std::vector<std::int8_t> GridBuilder::build(
  const std::vector<Point3> & points,
  const DrivableMask & drivable_mask) const
{
  validate_mask(drivable_mask);
  return build_impl(points, &drivable_mask);
}

std::vector<std::int8_t> GridBuilder::build_impl(
  const std::vector<Point3> & points,
  const DrivableMask * const drivable_mask) const
{
  std::vector<std::int8_t> grid(width_ * height_, 0);
  std::vector<bool> occupied(grid.size(), false);
  std::vector<std::size_t> occupied_indices;
  occupied_indices.reserve(std::min(points.size(), grid.size()));

  for (const auto & point : points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z) ||
      point.x < config_.x_min || point.x >= config_.x_max ||
      point.y < config_.y_min || point.y >= config_.y_max ||
      point.z < config_.z_min || point.z > config_.z_max)
    {
      continue;
    }
    if (point.x >= config_.ego_clear_x_min && point.x <= config_.ego_clear_x_max &&
      point.y >= config_.ego_clear_y_min && point.y <= config_.ego_clear_y_max)
    {
      continue;
    }
    const auto x = static_cast<std::size_t>(
      std::floor((point.x - config_.x_min) / config_.resolution));
    const auto y = static_cast<std::size_t>(
      std::floor((point.y - config_.y_min) / config_.resolution));
    if (x >= width_ || y >= height_) {
      continue;
    }
    const std::size_t index = y * width_ + x;
    if (drivable_mask != nullptr && drivable_mask->data[index] != 0) {
      continue;
    }
    if (!occupied[index]) {
      occupied[index] = true;
      occupied_indices.push_back(index);
    }
  }

  for (const std::size_t index : occupied_indices) {
    const auto y = static_cast<std::int64_t>(index / width_);
    const auto x = static_cast<std::int64_t>(index % width_);
    for (const auto & offset : inflation_kernel_) {
      const std::int64_t cell_x = x + offset.dx;
      const std::int64_t cell_y = y + offset.dy;
      if (cell_x < 0 || cell_y < 0 ||
        cell_x >= static_cast<std::int64_t>(width_) ||
        cell_y >= static_cast<std::int64_t>(height_))
      {
        continue;
      }
      auto & cell = grid[static_cast<std::size_t>(cell_y) * width_ +
          static_cast<std::size_t>(cell_x)];
      cell = std::max(cell, offset.cost);
    }
  }

  // Clear the ego footprint after inflation as well.  Filtering only the raw
  // returns is insufficient: returns just outside the body inflate back into
  // the footprint and make DWA reject its initial pose.  The collision topic
  // remains responsible for detecting an object already touching the body;
  // obstacles outside this small rectangle remain available to trajectory
  // collision checks.
  for (std::size_t y = 0U; y < height_; ++y) {
    const double cell_y =
      config_.y_min + (static_cast<double>(y) + 0.5) * config_.resolution;
    if (cell_y < config_.ego_clear_y_min ||
      cell_y > config_.ego_clear_y_max)
    {
      continue;
    }
    for (std::size_t x = 0U; x < width_; ++x) {
      const double cell_x =
        config_.x_min + (static_cast<double>(x) + 0.5) * config_.resolution;
      if (cell_x >= config_.ego_clear_x_min &&
        cell_x <= config_.ego_clear_x_max)
      {
        grid[y * width_ + x] = 0;
      }
    }
  }
  return grid;
}

void GridBuilder::validate_mask(const DrivableMask & drivable_mask) const
{
  if (drivable_mask.x_min != config_.x_min ||
    drivable_mask.y_min != config_.y_min ||
    drivable_mask.resolution != config_.resolution ||
    drivable_mask.width != width_ ||
    drivable_mask.height != height_ ||
    drivable_mask.data.size() != width_ * height_)
  {
    throw std::invalid_argument(
            "drivable-mask geometry does not match occupancy grid");
  }
}

std::size_t GridBuilder::width() const
{
  return width_;
}

std::size_t GridBuilder::height() const
{
  return height_;
}

const GridConfig & GridBuilder::config() const
{
  return config_;
}

}  // namespace ad_lidar_perception::occupancy_grid
