#ifndef AD_PLANNER__LOCAL_PLANNING__ROAD_CORRIDOR_GRID_HPP_
#define AD_PLANNER__LOCAL_PLANNING__ROAD_CORRIDOR_GRID_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <builtin_interfaces/msg/time.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

#include "ad_planner/local_planning/common/local_motion.hpp"

namespace ad_planner
{

struct RouteSliceOccupancy
{
  std::size_t occupied_cell_count{0U};
  std::size_t unknown_cell_count{0U};
  std::optional<double> nearest_occupied_s_m;
};

struct RoadCorridorGridWindow
{
  double minimum_x_m{-4.0};
  double maximum_x_m{100.0};
  double minimum_y_m{-10.0};
  double maximum_y_m{10.0};
  double resolution_m{0.1};
};

struct RoadCorridorGridWork
{
  std::size_t total_segment_count{0U};
  std::size_t candidate_segment_count{0U};
  std::size_t candidate_cell_visit_count{0U};
};

// Owns a validated, immutable corridor geometry and its spatial index.
// Construct this once when a route is loaded and reuse it for every rolling
// mask/query stamp.
class PreparedRoadCorridor
{
public:
  struct Implementation;

  explicit PreparedRoadCorridor(const ReferenceCorridor & corridor);
  PreparedRoadCorridor(const PreparedRoadCorridor &) noexcept = default;
  PreparedRoadCorridor(PreparedRoadCorridor &&) noexcept = default;
  PreparedRoadCorridor & operator=(
    const PreparedRoadCorridor &) noexcept = default;
  PreparedRoadCorridor & operator=(PreparedRoadCorridor &&) noexcept = default;
  ~PreparedRoadCorridor();

  const std::string & frame_id() const noexcept;
  std::size_t segment_count() const noexcept;

private:
  std::shared_ptr<const Implementation> implementation_;

  friend nav_msgs::msg::OccupancyGrid rasterize_road_corridor(
    const PreparedRoadCorridor & corridor,
    const nav_msgs::msg::OccupancyGrid & grid_template,
    RoadCorridorGridWork * work);
  friend RouteSliceOccupancy query_route_slice_occupancy(
    const PreparedRoadCorridor & corridor,
    const nav_msgs::msg::OccupancyGrid & occupancy_grid,
    double near_s_m, double far_s_m,
    std::int8_t occupied_threshold,
    RoadCorridorGridWork * work);
};

// Builds grid geometry in the route frame for a base-relative rectangular
// window. route_from_base is the planar base pose expressed in route_frame.
nav_msgs::msg::OccupancyGrid make_route_aligned_grid_template(
  const RoadCorridorGridWindow & window,
  const std::string & route_frame,
  const builtin_interfaces::msg::Time & stamp,
  const Pose2 & route_from_base);

// Re-expresses an already-rasterized route-frame mask in base_frame without
// resampling. The data order is unchanged and the grid origin becomes the
// base-relative window minimum with identity orientation.
nav_msgs::msg::OccupancyGrid express_road_corridor_mask_in_base_frame(
  const nav_msgs::msg::OccupancyGrid & route_frame_mask,
  const RoadCorridorGridWindow & window,
  const std::string & base_frame);

// Copies the template header and geometry and replaces its data with a
// conservative road mask: 0 is drivable and 100 is non-drivable.
// Invalid or frame-incompatible geometry throws instead of returning a
// potentially permissive mask.
nav_msgs::msg::OccupancyGrid rasterize_road_corridor(
  const ReferenceCorridor & corridor,
  const nav_msgs::msg::OccupancyGrid & grid_template);

nav_msgs::msg::OccupancyGrid rasterize_road_corridor(
  const PreparedRoadCorridor & corridor,
  const nav_msgs::msg::OccupancyGrid & grid_template,
  RoadCorridorGridWork * work = nullptr);

// Counts occupied and unknown cell centers that lie inside the corridor and
// within the inclusive absolute route-progress interval [near_s_m, far_s_m].
// Invalid route/grid geometry or cell costs throw std::invalid_argument.
RouteSliceOccupancy query_route_slice_occupancy(
  const ReferenceCorridor & corridor,
  const nav_msgs::msg::OccupancyGrid & occupancy_grid,
  double near_s_m, double far_s_m,
  std::int8_t occupied_threshold = 50);

RouteSliceOccupancy query_route_slice_occupancy(
  const PreparedRoadCorridor & corridor,
  const nav_msgs::msg::OccupancyGrid & occupancy_grid,
  double near_s_m, double far_s_m,
  std::int8_t occupied_threshold = 50,
  RoadCorridorGridWork * work = nullptr);

}  // namespace ad_planner

#endif  // AD_PLANNER__LOCAL_PLANNING__ROAD_CORRIDOR_GRID_HPP_
