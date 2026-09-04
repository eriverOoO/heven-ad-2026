#include "ad_lidar_perception/occupancy_grid/dynamic_grid_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ad_lidar_perception::occupancy_grid
{
namespace
{

// Covariance values within this scaled tolerance are treated as numerical
// round-off at the positive-semidefinite boundary.
constexpr double kCovarianceEpsilon = 1.0e-9;
constexpr double kIntersectionEpsilon = 1.0e-12;

bool finite_geometry(const GridGeometry & geometry)
{
  return std::isfinite(geometry.x_min_m) &&
         std::isfinite(geometry.y_min_m) &&
         std::isfinite(geometry.resolution_m);
}

void validate_geometry(const GridGeometry & geometry)
{
  if (!finite_geometry(geometry) || geometry.resolution_m <= 0.0 ||
    geometry.width == 0U || geometry.height == 0U ||
    geometry.width > std::numeric_limits<std::size_t>::max() / geometry.height)
  {
    throw std::invalid_argument("invalid dynamic-grid geometry");
  }
  const auto cell_count = geometry.width * geometry.height;
  if (cell_count > std::vector<std::int8_t>{}.max_size()) {
    throw std::invalid_argument("dynamic-grid cell count is not representable");
  }
  const double x_max =
    geometry.x_min_m + geometry.resolution_m * static_cast<double>(geometry.width);
  const double y_max =
    geometry.y_min_m + geometry.resolution_m * static_cast<double>(geometry.height);
  if (!std::isfinite(x_max) || !std::isfinite(y_max) ||
    x_max <= geometry.x_min_m || y_max <= geometry.y_min_m)
  {
    throw std::invalid_argument("invalid dynamic-grid extent");
  }
}

void validate_config(const DynamicGridConfig & config)
{
  const int occupied_cost = static_cast<int>(config.occupied_cost);
  if (!std::isfinite(config.covariance_sigma) ||
    !std::isfinite(config.minimum_inflation_m) ||
    config.covariance_sigma < 0.0 || config.minimum_inflation_m < 0.0 ||
    occupied_cost < 1 || occupied_cost > 100 ||
    config.maximum_cells_per_object == 0U)
  {
    throw std::invalid_argument("invalid dynamic-grid configuration");
  }
}

double maximum_covariance_eigenvalue(const DynamicBox & object)
{
  const double scale = std::max(
    {1.0, std::abs(object.covariance_xx),
      std::abs(object.covariance_xy), std::abs(object.covariance_yy)});
  const double tolerance = kCovarianceEpsilon * scale;
  if (object.covariance_xx < -tolerance ||
    object.covariance_yy < -tolerance)
  {
    throw std::invalid_argument("dynamic-object covariance has a negative diagonal");
  }

  const double xx = std::max(0.0, object.covariance_xx);
  const double yy = std::max(0.0, object.covariance_yy);
  const double midpoint = xx * 0.5 + yy * 0.5;
  const double radius = std::hypot(
    xx * 0.5 - yy * 0.5, object.covariance_xy);
  const double minimum_eigenvalue = midpoint - radius;
  const double maximum_eigenvalue = midpoint + radius;
  if (!std::isfinite(minimum_eigenvalue) ||
    !std::isfinite(maximum_eigenvalue) ||
    minimum_eigenvalue < -tolerance)
  {
    throw std::invalid_argument(
            "dynamic-object covariance is not positive semidefinite");
  }
  return std::max(0.0, maximum_eigenvalue);
}

void validate_object(const DynamicBox & object)
{
  if (!std::isfinite(object.x_m) || !std::isfinite(object.y_m) ||
    !std::isfinite(object.yaw_rad) || !std::isfinite(object.length_m) ||
    !std::isfinite(object.width_m) ||
    !std::isfinite(object.covariance_xx) ||
    !std::isfinite(object.covariance_xy) ||
    !std::isfinite(object.covariance_yy) ||
    object.length_m <= 0.0 || object.width_m <= 0.0)
  {
    throw std::invalid_argument("invalid dynamic object");
  }
}

bool intersects(
  double object_x, double object_y,
  double cosine, double sine,
  double half_length, double half_width,
  double cell_x, double cell_y, double cell_half)
{
  const double dx = cell_x - object_x;
  const double dy = cell_y - object_y;
  const double abs_cosine = std::abs(cosine);
  const double abs_sine = std::abs(sine);

  const auto overlaps = [](double distance, double limit) {
      const double tolerance =
        kIntersectionEpsilon * std::max({1.0, std::abs(distance), limit});
      return std::abs(distance) <= limit + tolerance;
    };

  if (!overlaps(
      dx * cosine + dy * sine,
      half_length + cell_half * (abs_cosine + abs_sine)))
  {
    return false;
  }
  if (!overlaps(
      -dx * sine + dy * cosine,
      half_width + cell_half * (abs_sine + abs_cosine)))
  {
    return false;
  }
  if (!overlaps(
      dx, cell_half + half_length * abs_cosine + half_width * abs_sine))
  {
    return false;
  }
  return overlaps(
    dy, cell_half + half_length * abs_sine + half_width * abs_cosine);
}

std::size_t clipped_min_index(
  double coordinate, double grid_min, double grid_max,
  double resolution, std::size_t cell_count)
{
  if (coordinate <= grid_min) {
    return 0U;
  }
  if (coordinate >= grid_max) {
    return cell_count - 1U;
  }
  const double normalized_coordinate =
    (coordinate - grid_min) / resolution;
  const double lower_index = std::ceil(
    std::nextafter(
      normalized_coordinate, -std::numeric_limits<double>::infinity())) - 1.0;
  if (lower_index <= 0.0) {
    return 0U;
  }
  return std::min(
    cell_count - 1U,
    static_cast<std::size_t>(lower_index));
}

std::size_t clipped_max_index(
  double coordinate, double grid_min, double grid_max,
  double resolution, std::size_t cell_count)
{
  if (coordinate >= grid_max) {
    return cell_count - 1U;
  }
  if (coordinate <= grid_min) {
    return 0U;
  }
  return std::min(
    cell_count - 1U,
    static_cast<std::size_t>(std::floor((coordinate - grid_min) / resolution)));
}

// The grid-clipped inflated bounding box of one footprint plus the rotation
// terms the SAT rasterization needs. off_grid is set when the footprint lies
// entirely outside the grid; the index fields are then unspecified.
struct FootprintExtent
{
  bool off_grid{true};
  double cosine{1.0};
  double sine{0.0};
  double half_length{0.0};
  double half_width{0.0};
  std::size_t minimum_x{0U};
  std::size_t maximum_x{0U};
  std::size_t minimum_y{0U};
  std::size_t maximum_y{0U};
};

FootprintExtent footprint_extent(
  const GridGeometry & geometry, const DynamicBox & object,
  const DynamicGridConfig & config)
{
  validate_object(object);
  const double maximum_eigenvalue = maximum_covariance_eigenvalue(object);
  const double covariance_inflation =
    config.covariance_sigma * std::sqrt(maximum_eigenvalue);
  if (!std::isfinite(covariance_inflation)) {
    throw std::invalid_argument("dynamic-object covariance inflation overflowed");
  }
  const double inflation =
    std::max(config.minimum_inflation_m, covariance_inflation);

  FootprintExtent extent;
  extent.half_length = object.length_m * 0.5 + inflation;
  extent.half_width = object.width_m * 0.5 + inflation;
  extent.cosine = std::cos(object.yaw_rad);
  extent.sine = std::sin(object.yaw_rad);
  const double extent_x =
    std::abs(extent.cosine) * extent.half_length +
    std::abs(extent.sine) * extent.half_width;
  const double extent_y =
    std::abs(extent.sine) * extent.half_length +
    std::abs(extent.cosine) * extent.half_width;
  if (!std::isfinite(extent.half_length) || !std::isfinite(extent.half_width) ||
    !std::isfinite(extent_x) || !std::isfinite(extent_y))
  {
    throw std::invalid_argument("dynamic-object footprint overflowed");
  }

  const double grid_x_max = geometry.x_min_m +
    geometry.resolution_m * static_cast<double>(geometry.width);
  const double grid_y_max = geometry.y_min_m +
    geometry.resolution_m * static_cast<double>(geometry.height);
  const double object_x_min = object.x_m - extent_x;
  const double object_x_max = object.x_m + extent_x;
  const double object_y_min = object.y_m - extent_y;
  const double object_y_max = object.y_m + extent_y;
  if (!std::isfinite(object_x_min) || !std::isfinite(object_x_max) ||
    !std::isfinite(object_y_min) || !std::isfinite(object_y_max))
  {
    throw std::invalid_argument("dynamic-object bounds overflowed");
  }
  if (object_x_max < geometry.x_min_m || object_x_min > grid_x_max ||
    object_y_max < geometry.y_min_m || object_y_min > grid_y_max)
  {
    return extent;
  }

  extent.off_grid = false;
  extent.minimum_x = clipped_min_index(
    object_x_min, geometry.x_min_m, grid_x_max,
    geometry.resolution_m, geometry.width);
  extent.maximum_x = clipped_max_index(
    object_x_max, geometry.x_min_m, grid_x_max,
    geometry.resolution_m, geometry.width);
  extent.minimum_y = clipped_min_index(
    object_y_min, geometry.y_min_m, grid_y_max,
    geometry.resolution_m, geometry.height);
  extent.maximum_y = clipped_max_index(
    object_y_max, geometry.y_min_m, grid_y_max,
    geometry.resolution_m, geometry.height);
  return extent;
}

// Grid-clipped candidate cell count for an on-grid footprint extent, saturating
// to size_t max on overflow so a budget comparison still rejects it (this
// reproduces the builder's original width > budget/height overflow guard).
std::size_t candidate_cell_count(const FootprintExtent & extent)
{
  const std::size_t candidate_width = extent.maximum_x - extent.minimum_x + 1U;
  const std::size_t candidate_height = extent.maximum_y - extent.minimum_y + 1U;
  if (candidate_width >
    std::numeric_limits<std::size_t>::max() / candidate_height)
  {
    return std::numeric_limits<std::size_t>::max();
  }
  return candidate_width * candidate_height;
}

}  // namespace

std::size_t footprint_candidate_cells(
  const GridGeometry & geometry, const DynamicBox & footprint,
  const DynamicGridConfig & config)
{
  validate_geometry(geometry);
  validate_config(config);
  const auto extent = footprint_extent(geometry, footprint, config);
  if (extent.off_grid) {
    return 0U;
  }
  return candidate_cell_count(extent);
}

std::vector<DynamicBox> interpolate_dynamic_trajectory(
  const std::vector<DynamicBox> & keyframes,
  const double maximum_center_spacing_m,
  const std::size_t maximum_output_samples)
{
  if (keyframes.empty() || !std::isfinite(maximum_center_spacing_m) ||
    maximum_center_spacing_m <= 0.0 || maximum_output_samples == 0U)
  {
    throw std::invalid_argument(
            "invalid dynamic-trajectory interpolation input");
  }
  for (const auto & keyframe : keyframes) {
    validate_object(keyframe);
    (void)maximum_covariance_eigenvalue(keyframe);
  }
  if (keyframes.size() > maximum_output_samples) {
    throw std::length_error(
            "dynamic-trajectory keyframes exceed maximum output samples");
  }

  std::vector<DynamicBox> output;
  output.reserve(keyframes.size());
  output.push_back(keyframes.front());
  constexpr long double kTwoPi =
    2.0L * 3.141592653589793238462643383279502884L;
  const auto interpolate = [](const double first, const double second,
      const long double ratio)
    {
      const long double value =
        static_cast<long double>(first) +
        ratio * (
        static_cast<long double>(second) -
        static_cast<long double>(first));
      if (!std::isfinite(value) ||
        value > static_cast<long double>(
          std::numeric_limits<double>::max()) ||
        value < -static_cast<long double>(
          std::numeric_limits<double>::max()))
      {
        throw std::invalid_argument(
                "dynamic-trajectory interpolation overflowed");
      }
      return static_cast<double>(value);
    };

  for (std::size_t index = 1U; index < keyframes.size(); ++index) {
    const auto & first = keyframes[index - 1U];
    const auto & second = keyframes[index];
    const double distance =
      std::hypot(second.x_m - first.x_m, second.y_m - first.y_m);
    if (!std::isfinite(distance)) {
      throw std::invalid_argument(
              "dynamic-trajectory segment length is invalid");
    }
    const double interval_value =
      std::max(1.0, std::ceil(distance / maximum_center_spacing_m));
    if (!std::isfinite(interval_value) ||
      interval_value >
      static_cast<double>(std::numeric_limits<std::size_t>::max()))
    {
      throw std::length_error(
              "dynamic-trajectory segment sample count is not representable");
    }
    const auto intervals = static_cast<std::size_t>(interval_value);
    if (intervals > maximum_output_samples - output.size()) {
      throw std::length_error(
              "dynamic trajectory exceeds maximum output samples");
    }
    const long double yaw_delta = std::remainder(
      static_cast<long double>(second.yaw_rad) -
      static_cast<long double>(first.yaw_rad), kTwoPi);
    if (!std::isfinite(yaw_delta)) {
      throw std::invalid_argument(
              "dynamic-trajectory yaw interpolation is invalid");
    }
    for (std::size_t sample = 1U; sample <= intervals; ++sample) {
      const long double ratio =
        static_cast<long double>(sample) /
        static_cast<long double>(intervals);
      const long double yaw = std::remainder(
        static_cast<long double>(first.yaw_rad) + ratio * yaw_delta,
        kTwoPi);
      if (!std::isfinite(yaw)) {
        throw std::invalid_argument(
                "dynamic-trajectory yaw interpolation overflowed");
      }
      output.push_back(
        DynamicBox{
          interpolate(first.x_m, second.x_m, ratio),
          interpolate(first.y_m, second.y_m, ratio),
          static_cast<double>(yaw),
          interpolate(first.length_m, second.length_m, ratio),
          interpolate(first.width_m, second.width_m, ratio),
          interpolate(
            first.covariance_xx, second.covariance_xx, ratio),
          interpolate(
            first.covariance_xy, second.covariance_xy, ratio),
          interpolate(
            first.covariance_yy, second.covariance_yy, ratio)});
    }
  }
  return output;
}

std::vector<DynamicBox> sweep_object_footprints(
  const std::vector<DynamicBox> & footprints,
  const double grid_resolution_m,
  const std::size_t maximum_output_samples)
{
  if (footprints.empty() || !std::isfinite(grid_resolution_m) ||
    grid_resolution_m <= 0.0 || maximum_output_samples == 0U)
  {
    throw std::invalid_argument("invalid object-footprint sweep input");
  }
  if (footprints.size() == 1U) {
    return footprints;
  }
  const auto & current = footprints.front();
  validate_object(current);
  // Sample densely enough that consecutive interpolated footprints overlap:
  // half the smaller footprint edge guarantees no gap between rasterized boxes,
  // and never finer than one grid cell so the sample count stays bounded.
  const double spacing = std::max(
    grid_resolution_m,
    0.5 * std::min(current.length_m, current.width_m));
  if (!std::isfinite(spacing) || spacing <= 0.0) {
    throw std::invalid_argument("object-footprint sweep spacing is invalid");
  }
  return interpolate_dynamic_trajectory(
    footprints, spacing, maximum_output_samples);
}

std::vector<DynamicBox> budget_object_sweep(
  const GridGeometry & geometry,
  const std::vector<DynamicBox> & footprints,
  const DynamicGridConfig & config,
  const std::size_t maximum_sweep_samples,
  SweepOutcome * const outcome)
{
  validate_geometry(geometry);
  validate_config(config);
  if (footprints.size() < 2U) {
    throw std::invalid_argument(
            "budget_object_sweep needs a current footprint and >= 1 keyframe");
  }
  const auto set_outcome = [outcome](const SweepOutcome value) {
      if (outcome != nullptr) {
        *outcome = value;
      }
      return value;
    };

  std::vector<DynamicBox> current_only{footprints.front()};

  // The current footprint is kept verbatim. When its own inflated bounding box
  // already exceeds the per-object budget the object is oversized in the
  // pre-sweep sense: keep only that footprint (build_dynamic_grid skip-counts
  // it once) and never expand an already-pathological footprint.
  if (footprint_candidate_cells(geometry, footprints.front(), config) >
    config.maximum_cells_per_object)
  {
    set_outcome(SweepOutcome::kCurrentOversized);
    return current_only;
  }

  std::vector<DynamicBox> swept;
  try {
    swept = sweep_object_footprints(
      footprints, geometry.resolution_m, maximum_sweep_samples);
  } catch (const std::exception &) {
    // Degenerate spacing / sample budget / interpolation overflow / a malformed
    // future keyframe: drop the future expansion, keep the current footprint.
    set_outcome(SweepOutcome::kExpansionFailed);
    return current_only;
  }

  // Budget the swept group as one object: no individual swept footprint may
  // exceed the per-object cell budget (so build_dynamic_grid never skip-counts
  // a sweep footprint), and the group's total grid-clipped candidate-cell
  // rasterization work may not exceed one full grid. Failing either check drops
  // the whole future expansion -- deterministic and all-or-nothing.
  const std::size_t grid_cell_budget = geometry.width * geometry.height;
  std::size_t group_candidate_cells = 0U;
  for (const auto & swept_box : swept) {
    const std::size_t cells =
      footprint_candidate_cells(geometry, swept_box, config);
    if (cells > config.maximum_cells_per_object) {
      set_outcome(SweepOutcome::kBudgetCapped);
      return current_only;
    }
    group_candidate_cells += cells;
    if (group_candidate_cells > grid_cell_budget) {
      set_outcome(SweepOutcome::kBudgetCapped);
      return current_only;
    }
  }

  set_outcome(SweepOutcome::kFullSweep);
  return swept;
}

std::vector<std::int8_t> build_dynamic_grid_impl(
  const GridGeometry & geometry,
  const std::vector<DynamicBox> & objects,
  const DynamicGridConfig & config,
  const std::vector<std::int8_t> * const drivable_mask,
  std::size_t * const oversized_objects_skipped)
{
  if (oversized_objects_skipped != nullptr) {
    *oversized_objects_skipped = 0U;
  }
  validate_geometry(geometry);
  validate_config(config);

  const std::size_t cell_count = geometry.width * geometry.height;
  if (drivable_mask != nullptr) {
    if (drivable_mask->size() != cell_count) {
      throw std::invalid_argument(
              "drivable-mask size does not match dynamic-grid geometry");
    }
    for (const std::int8_t value : *drivable_mask) {
      if (value < -1 || value > 100) {
        throw std::invalid_argument(
                "drivable mask contains an invalid occupancy value");
      }
    }
  }

  std::vector<std::int8_t> grid(cell_count, 0);
  const double cell_half = geometry.resolution_m * 0.5;

  for (const auto & object : objects) {
    const auto extent = footprint_extent(geometry, object, config);
    if (extent.off_grid) {
      continue;
    }
    const double cosine = extent.cosine;
    const double sine = extent.sine;
    const double half_length = extent.half_length;
    const double half_width = extent.half_width;
    const std::size_t minimum_x = extent.minimum_x;
    const std::size_t maximum_x = extent.maximum_x;
    const std::size_t minimum_y = extent.minimum_y;
    const std::size_t maximum_y = extent.maximum_y;
    // A single object whose grid-clipped inflated footprint would exceed the
    // per-object cell budget is skipped, not fatal: one pathologically
    // uncertain or coarse object must not erase every other valid object in
    // the frame. Per-object rasterization work stays bounded by this same
    // budget, so total per-frame work is unchanged. After Future Sweep v1 the
    // dynamic-occupancy node budgets a predicted object's whole swept
    // footprint group up front (using footprint_candidate_cells) so anything
    // reaching this loop from the sweep path already fits; the only footprints
    // this guard now skips are the single current footprint of a genuinely
    // oversized predicted object, keeping the counter one-per-object.
    if (candidate_cell_count(extent) > config.maximum_cells_per_object) {
      if (oversized_objects_skipped != nullptr) {
        ++*oversized_objects_skipped;
      }
      continue;
    }

    for (std::size_t y = minimum_y; y <= maximum_y; ++y) {
      const double cell_y =
        geometry.y_min_m + (static_cast<double>(y) + 0.5) * geometry.resolution_m;
      for (std::size_t x = minimum_x; x <= maximum_x; ++x) {
        const double cell_x =
          geometry.x_min_m + (static_cast<double>(x) + 0.5) * geometry.resolution_m;
        if (intersects(
            object.x_m, object.y_m, cosine, sine, half_length, half_width,
            cell_x, cell_y, cell_half))
        {
          const std::size_t index = y * geometry.width + x;
          if (drivable_mask == nullptr || (*drivable_mask)[index] == 0) {
            grid[index] = config.occupied_cost;
          }
        }
      }
    }
  }
  return grid;
}

std::vector<std::int8_t> build_dynamic_grid(
  const GridGeometry & geometry,
  const std::vector<DynamicBox> & objects,
  const DynamicGridConfig & config,
  std::size_t * const oversized_objects_skipped)
{
  return build_dynamic_grid_impl(
    geometry, objects, config, nullptr, oversized_objects_skipped);
}

std::vector<std::int8_t> build_dynamic_grid(
  const GridGeometry & geometry,
  const std::vector<DynamicBox> & objects,
  const DynamicGridConfig & config,
  const std::vector<std::int8_t> & drivable_mask,
  std::size_t * const oversized_objects_skipped)
{
  return build_dynamic_grid_impl(
    geometry, objects, config, &drivable_mask, oversized_objects_skipped);
}

}  // namespace ad_lidar_perception::occupancy_grid
