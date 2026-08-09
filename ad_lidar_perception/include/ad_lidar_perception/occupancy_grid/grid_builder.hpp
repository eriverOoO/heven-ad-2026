#ifndef AD_LIDAR_PERCEPTION__OCCUPANCY_GRID__GRID_BUILDER_HPP_
#define AD_LIDAR_PERCEPTION__OCCUPANCY_GRID__GRID_BUILDER_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ad_lidar_perception::occupancy_grid
{

struct Point3
{
  double x;
  double y;
  double z;
};

struct GridConfig
{
  double x_min{-4.0};
  double x_max{100.0};
  double y_min{-10.0};
  double y_max{10.0};
  double z_min{0.1};
  double z_max{2.0};
  double resolution{0.1};
  double inflation_radius_m{1.8};
  double inflation_cost_scaling_factor{2.0};
  double ego_clear_x_min{-1.0};
  double ego_clear_x_max{4.05};
  double ego_clear_y_min{-1.15};
  double ego_clear_y_max{1.15};
};

struct DrivableMask
{
  double x_min;
  double y_min;
  double resolution;
  std::size_t width;
  std::size_t height;
  // Row-major occupancy-style mask: 0 is drivable, positive values are
  // non-drivable, and nav_msgs/OccupancyGrid unknown=-1 fails closed.
  std::vector<std::int8_t> data;
};

class GridBuilder
{
public:
  explicit GridBuilder(GridConfig config);

  std::vector<std::int8_t> build(const std::vector<Point3> & points) const;
  std::vector<std::int8_t> build(
    const std::vector<Point3> & points,
    const DrivableMask & drivable_mask) const;
  std::size_t width() const;
  std::size_t height() const;
  const GridConfig & config() const;

private:
  struct InflationOffset
  {
    std::int64_t dx;
    std::int64_t dy;
    std::int8_t cost;
  };

  std::vector<std::int8_t> build_impl(
    const std::vector<Point3> & points,
    const DrivableMask * drivable_mask) const;
  void validate_mask(const DrivableMask & drivable_mask) const;

  GridConfig config_;
  std::size_t width_;
  std::size_t height_;
  std::vector<InflationOffset> inflation_kernel_;
};

}  // namespace ad_lidar_perception::occupancy_grid

#endif  // AD_LIDAR_PERCEPTION__OCCUPANCY_GRID__GRID_BUILDER_HPP_
