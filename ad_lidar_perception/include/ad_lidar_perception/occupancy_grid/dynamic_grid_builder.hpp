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

// Why one physical predicted object's future sweep was or was not rasterized in
// full. Reported by budget_object_sweep so the caller can keep per-object
// diagnostics (oversized_objects_skipped stays one-per-object, not
// one-per-temporal-footprint).
enum class SweepOutcome
{
  // The full gap-free interpolated sweep is returned.
  kFullSweep,
  // The current footprint's own inflated bounding box already exceeds
  // config.maximum_cells_per_object; only it is returned and build_dynamic_grid
  // skip-counts it exactly once, exactly as in the pre-sweep path.
  kCurrentOversized,
  // A single swept footprint exceeds config.maximum_cells_per_object, or the
  // swept group's total grid-clipped candidate-cell work exceeds one full grid.
  // Only the (in-budget) current footprint is returned; the future expansion is
  // dropped whole.
  kBudgetCapped,
  // sweep_object_footprints could not expand the trajectory into a bounded
  // interpolation. Only the current footprint is returned.
  kExpansionFailed,
};

// Budget one predicted object's future sweep as a single group. footprints[0]
// is the object's current footprint; footprints[1..] are its in-horizon
// predicted keyframes, already transformed into the grid frame and already
// clipped to the caller's forward horizon (>= 2 entries). Returns the footprint
// list to rasterize for this one object: the gap-free interpolated sweep when
// it fits, otherwise just the current footprint. Never returns a
// partially-rasterized sweep. *outcome, when not null, receives which case
// applied. Throws exactly as footprint_candidate_cells / sweep_object_footprints
// do on a malformed current footprint (the caller's fail-safe-clear concern);
// a malformed *future* keyframe instead yields kExpansionFailed.
std::vector<DynamicBox> budget_object_sweep(
  const GridGeometry & geometry,
  const std::vector<DynamicBox> & footprints,
  const DynamicGridConfig & config,
  std::size_t maximum_sweep_samples,
  SweepOutcome * outcome = nullptr);

// The grid-clipped, uncertainty-inflated bounding-box cell count of one
// footprint -- the quantity config.maximum_cells_per_object is checked against.
// Returns 0 when the inflated footprint lies entirely outside the grid (the
// builder skips such a footprint before the budget check), and
// std::numeric_limits<std::size_t>::max() when the clipped width x height would
// overflow. Throws exactly as the builder's per-object prologue does on a
// malformed footprint (non-finite fields, non-positive dimensions,
// non-positive-semidefinite covariance, overflowed inflation/extent). The
// dynamic-occupancy node uses this to budget a predicted object's whole future
// sweep as one group before flattening it into individual footprints.
std::size_t footprint_candidate_cells(
  const GridGeometry & geometry,
  const DynamicBox & footprint,
  const DynamicGridConfig & config);

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
