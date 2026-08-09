#include "ad_lidar_perception/occupancy_grid/grid_combiner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ad_lidar_perception::occupancy_grid
{
namespace
{

bool valid_geometry(const GridGeometry & geometry)
{
  if (!std::isfinite(geometry.x_min_m) ||
    !std::isfinite(geometry.y_min_m) ||
    !std::isfinite(geometry.resolution_m) ||
    geometry.resolution_m <= 0.0 ||
    geometry.width == 0U || geometry.height == 0U ||
    geometry.width > std::numeric_limits<std::size_t>::max() / geometry.height)
  {
    return false;
  }
  const float normalized_resolution =
    static_cast<float>(geometry.resolution_m);
  return std::isfinite(normalized_resolution) && normalized_resolution > 0.0F;
}

bool valid_cell(const std::int8_t cell)
{
  return cell >= -1 && cell <= 100;
}

}  // namespace

std::vector<std::int8_t> combine_cost_layers(
  const std::vector<std::int8_t> & static_cells,
  const std::vector<std::int8_t> & dynamic_cells)
{
  if (static_cells.size() != dynamic_cells.size()) {
    throw std::invalid_argument("occupancy layers have unequal sizes");
  }
  std::vector<std::int8_t> result;
  result.reserve(static_cells.size());
  for (std::size_t index = 0U; index < static_cells.size(); ++index) {
    const auto static_cell = static_cells[index];
    const auto dynamic_cell = dynamic_cells[index];
    if (!valid_cell(static_cell) || !valid_cell(dynamic_cell)) {
      throw std::invalid_argument("occupancy layer contains an invalid cost");
    }
    if (static_cell < 0 && dynamic_cell < 0) {
      result.push_back(-1);
    } else if (static_cell < 0) {
      result.push_back(dynamic_cell);
    } else if (dynamic_cell < 0) {
      result.push_back(static_cell);
    } else {
      result.push_back(std::max(static_cell, dynamic_cell));
    }
  }
  return result;
}

bool layers_are_compatible(
  const GridLayerMetadata & static_metadata,
  const GridLayerMetadata & dynamic_metadata)
{
  if (!valid_geometry(static_metadata.geometry) ||
    !valid_geometry(dynamic_metadata.geometry) ||
    static_metadata.frame_id.empty() ||
    dynamic_metadata.frame_id.empty() ||
    static_metadata.frame_id != dynamic_metadata.frame_id ||
    static_metadata.stamp_ns <= 0 || dynamic_metadata.stamp_ns <= 0 ||
    static_metadata.stamp_ns != dynamic_metadata.stamp_ns)
  {
    return false;
  }
  const auto & static_geometry = static_metadata.geometry;
  const auto & dynamic_geometry = dynamic_metadata.geometry;
  if (static_geometry.x_min_m != dynamic_geometry.x_min_m ||
    static_geometry.y_min_m != dynamic_geometry.y_min_m ||
    static_cast<float>(static_geometry.resolution_m) !=
    static_cast<float>(dynamic_geometry.resolution_m) ||
    static_geometry.width != dynamic_geometry.width ||
    static_geometry.height != dynamic_geometry.height)
  {
    return false;
  }

  return true;
}

}  // namespace ad_lidar_perception::occupancy_grid
