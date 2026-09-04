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

// Expand an ordered footprint sequence -- [current_footprint, future_footprint,
// ...], already transformed into the grid frame and already clipped to the
// caller's forward horizon -- into a gap-free sweep of footprints covering the
// space the object is predicted to occupy up to that horizon. The centre
// spacing is derived from the first footprint's own dimensions so a fast object
// with sparse predicted keyframes still yields contiguous occupancy at the
// given grid resolution; it never re-predicts motion. A single-footprint input
// (stationary object, or no in-horizon future state) is returned unchanged, so
// a stationary object can never sweep a larger area than its current footprint.
// Delegates the actual interpolation to interpolate_dynamic_trajectory; throws
// the same exceptions on non-finite geometry or on exceeding
// maximum_output_samples, which the caller treats as a per-object fallback.
std::vector<DynamicBox> sweep_object_footprints(
  const std::vector<DynamicBox> & footprints,
  double grid_resolution_m,
  std::size_t maximum_output_samples);

// A single predicted object whose grid-clipped, uncertainty-inflated footprint
// would exceed config.maximum_cells_per_object is skipped (not rasterized)
// rather than aborting the whole grid. When oversized_objects_skipped is not
// null it receives the count of such objects for this call so the caller can
// surface a diagnostic. Genuinely malformed objects (non-finite fields,
// non-positive dimensions, non-positive-semidefinite covariance) and invalid
// geometry/config/mask still throw and are the caller's fail-safe concern.
std::vector<std::int8_t> build_dynamic_grid(
  const GridGeometry & geometry,
  const std::vector<DynamicBox> & objects,
  const DynamicGridConfig & config,
  std::size_t * oversized_objects_skipped = nullptr);

std::vector<std::int8_t> build_dynamic_grid(
  const GridGeometry & geometry,
  const std::vector<DynamicBox> & objects,
  const DynamicGridConfig & config,
  const std::vector<std::int8_t> & drivable_mask,
  std::size_t * oversized_objects_skipped = nullptr);

}  // namespace ad_lidar_perception::occupancy_grid

#endif  // AD_LIDAR_PERCEPTION__OCCUPANCY_GRID__DYNAMIC_GRID_BUILDER_HPP_
