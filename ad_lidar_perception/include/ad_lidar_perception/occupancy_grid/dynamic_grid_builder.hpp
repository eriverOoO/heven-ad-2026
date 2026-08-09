#ifndef AD_LIDAR_PERCEPTION__OCCUPANCY_GRID__DYNAMIC_GRID_BUILDER_HPP_
#define AD_LIDAR_PERCEPTION__OCCUPANCY_GRID__DYNAMIC_GRID_BUILDER_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ad_lidar_perception::occupancy_grid
{

struct GridGeometry
{
  double x_min_m{0.0};
  double y_min_m{0.0};
  double resolution_m{0.0};
  std::size_t width{0U};
  std::size_t height{0U};
};

struct DynamicBox
{
  double x_m{0.0};
  double y_m{0.0};
  double yaw_rad{0.0};
  double length_m{0.0};
  double width_m{0.0};
  double covariance_xx{0.0};
  double covariance_xy{0.0};
  double covariance_yy{0.0};
};

struct DynamicGridConfig
{
  double covariance_sigma{2.0};
  double minimum_inflation_m{0.20};
  std::int8_t occupied_cost{100};
  std::size_t maximum_cells_per_object{20000U};
};

std::vector<DynamicBox> interpolate_dynamic_trajectory(
  const std::vector<DynamicBox> & keyframes,
  double maximum_center_spacing_m,
  std::size_t maximum_output_samples);

std::vector<std::int8_t> build_dynamic_grid(
  const GridGeometry & geometry,
  const std::vector<DynamicBox> & objects,
  const DynamicGridConfig & config);

std::vector<std::int8_t> build_dynamic_grid(
  const GridGeometry & geometry,
  const std::vector<DynamicBox> & objects,
  const DynamicGridConfig & config,
  const std::vector<std::int8_t> & drivable_mask);

}  // namespace ad_lidar_perception::occupancy_grid

#endif  // AD_LIDAR_PERCEPTION__OCCUPANCY_GRID__DYNAMIC_GRID_BUILDER_HPP_
