#ifndef AD_PLANNER__OCCUPANCY_HPP_
#define AD_PLANNER__OCCUPANCY_HPP_

#include <optional>
#include <string>

#include "ad_planner/common/types.hpp"

namespace ad_planner
{

struct GridValidation
{
  bool valid{false};
  std::string reason;
};

struct GridCell
{
  std::size_t x;
  std::size_t y;
};

struct FootprintConfig
{
  double half_length_m;
  double half_width_m;
  double clearance_m;
  std::int8_t occupied_threshold;
  std::size_t maximum_cells_to_check;
  double center_offset_x_m{0.0};
};

GridValidation validate_occupancy_grid(const OccupancyGrid & grid);
std::optional<GridCell> world_to_cell(const OccupancyGrid & grid, const Point3 & point);
std::int8_t cell_value(const OccupancyGrid & grid, const Point3 & point);
bool footprint_is_safe(
  const OccupancyGrid & grid, const Pose2 & pose, const FootprintConfig & footprint);
double nearest_unsafe_clearance(
  const OccupancyGrid & grid, const Point3 & point, std::int8_t occupied_threshold,
  double maximum_distance_m);

}  // namespace ad_planner
#endif  // AD_PLANNER__OCCUPANCY_HPP_
