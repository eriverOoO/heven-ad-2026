#ifndef AD_LIDAR_PERCEPTION__OCCUPANCY_GRID__GRID_COMBINER_HPP_
#define AD_LIDAR_PERCEPTION__OCCUPANCY_GRID__GRID_COMBINER_HPP_

#include "ad_lidar_perception/occupancy_grid/dynamic_grid_builder.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace ad_lidar_perception::occupancy_grid
{

struct GridLayerMetadata
{
  GridGeometry geometry;
  std::string frame_id;
  std::int64_t stamp_ns{0};
};

std::vector<std::int8_t> combine_cost_layers(
  const std::vector<std::int8_t> & static_cells,
  const std::vector<std::int8_t> & dynamic_cells);

bool layers_are_compatible(
  const GridLayerMetadata & static_metadata,
  const GridLayerMetadata & dynamic_metadata);

}  // namespace ad_lidar_perception::occupancy_grid

#endif  // AD_LIDAR_PERCEPTION__OCCUPANCY_GRID__GRID_COMBINER_HPP_
