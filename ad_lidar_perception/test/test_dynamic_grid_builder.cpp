#include "ad_lidar_perception/occupancy_grid/dynamic_grid_builder.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

using ad_lidar_perception::occupancy_grid::DynamicBox;
using ad_lidar_perception::occupancy_grid::DynamicGridConfig;
using ad_lidar_perception::occupancy_grid::GridGeometry;
using ad_lidar_perception::occupancy_grid::build_dynamic_grid;
using ad_lidar_perception::occupancy_grid::interpolate_dynamic_trajectory;

GridGeometry geometry(
  std::size_t width = 6U, std::size_t height = 4U, double resolution = 1.0)
{
  return GridGeometry{0.0, 0.0, resolution, width, height};
}

DynamicGridConfig config()
{
  return DynamicGridConfig{0.0, 0.0, 100, 10000U};
}

DynamicBox box(
  double x, double y, double yaw, double length, double width,
  double covariance_xx = 0.0, double covariance_xy = 0.0,
  double covariance_yy = 0.0)
{
  return DynamicBox{
    x, y, yaw, length, width,
    covariance_xx, covariance_xy, covariance_yy};
}

std::int8_t at(
  const std::vector<std::int8_t> & grid,
  const GridGeometry & grid_geometry,
  std::size_t x,
  std::size_t y)
{
  return grid.at(y * grid_geometry.width + x);
}

TEST(DynamicGridBuilder, EmptyInputProducesExactFreeRowMajorGrid)
{
  const auto grid_geometry = GridGeometry{-4.0, -10.0, 0.1, 280U, 200U};
  const auto grid = build_dynamic_grid(grid_geometry, {}, config());

  ASSERT_EQ(grid.size(), 56000U);
  for (const auto cell : grid) {
    EXPECT_EQ(cell, 0);
  }
}

TEST(DynamicGridBuilder, AxisAlignedBoundaryContactCountsAsOccupied)
{
  const auto grid_geometry = geometry(3U, 3U);
  const auto grid = build_dynamic_grid(
    grid_geometry, {box(0.5, 0.5, 0.0, 1.0, 1.0)}, config());

  const std::vector<std::int8_t> expected{
    100, 100, 0,
    100, 100, 0,
    0, 0, 0};
  EXPECT_EQ(grid, expected);
}

TEST(DynamicGridBuilder, InternalLowerGridBoundaryContactCountsAsOccupied)
{
  const auto grid_geometry = geometry(4U, 1U);
  const auto grid = build_dynamic_grid(
    grid_geometry, {box(1.5, 0.5, 0.0, 1.0, 0.2)}, config());

  EXPECT_EQ(grid, (std::vector<std::int8_t>{100, 100, 100, 0}));
}

TEST(DynamicGridBuilder, RotatedFootprintUsesSatAndClipsAtGridBoundary)
{
  const auto grid_geometry = geometry(4U, 4U);
  const auto grid = build_dynamic_grid(
    grid_geometry, {box(0.0, 1.0, std::acos(-1.0) / 2.0, 2.0, 1.0)},
    config());

  EXPECT_EQ(at(grid, grid_geometry, 0U, 0U), 100);
  EXPECT_EQ(at(grid, grid_geometry, 0U, 1U), 100);
  EXPECT_EQ(at(grid, grid_geometry, 0U, 2U), 100);
  EXPECT_EQ(at(grid, grid_geometry, 1U, 1U), 0);
  EXPECT_EQ(at(grid, grid_geometry, 0U, 3U), 0);
}

TEST(DynamicGridBuilder, ZeroCovarianceStillUsesMinimumInflation)
{
  const auto grid_geometry = geometry(6U, 4U);
  auto grid_config = config();
  grid_config.covariance_sigma = 2.0;
  grid_config.minimum_inflation_m = 0.5;

  const auto grid = build_dynamic_grid(
    grid_geometry, {box(2.5, 1.5, 0.0, 1.0, 1.0)}, grid_config);

  EXPECT_EQ(at(grid, grid_geometry, 1U, 0U), 100);
  EXPECT_EQ(at(grid, grid_geometry, 4U, 1U), 0);
}

TEST(DynamicGridBuilder, LargestCovarianceEigenvalueControlsIsotropicInflation)
{
  const auto grid_geometry = geometry(12U, 12U);
  auto grid_config = config();
  grid_config.covariance_sigma = 2.0;

  const auto grid = build_dynamic_grid(
    grid_geometry,
    {box(5.5, 5.5, 0.0, 1.0, 1.0, 4.0, 0.0, 1.0)},
    grid_config);

  // sqrt(lambda_max=4) * sigma=2 expands the box to x=10 exactly.
  // Closed-set SAT therefore occupies the cell beginning at x=10.
  EXPECT_EQ(at(grid, grid_geometry, 10U, 5U), 100);
  EXPECT_EQ(at(grid, grid_geometry, 11U, 5U), 0);
  EXPECT_EQ(at(grid, grid_geometry, 5U, 10U), 100);
}

TEST(DynamicGridBuilder, UnionsCurrentHalfSecondAndOneSecondFootprints)
{
  const auto grid_geometry = geometry(6U, 1U);
  const auto grid = build_dynamic_grid(
    grid_geometry,
    {
      box(0.5, 0.5, 0.0, 0.2, 0.2),
      box(2.5, 0.5, 0.0, 0.2, 0.2),
      box(4.5, 0.5, 0.0, 0.2, 0.2),
    },
    config());

  EXPECT_EQ(grid, (std::vector<std::int8_t>{100, 0, 100, 0, 100, 0}));
}

TEST(DynamicGridBuilder, InterpolatesContinuousPredictionBetweenKeyframes)
{
  const auto samples = interpolate_dynamic_trajectory(
    {
      box(0.5, 0.5, 0.0, 0.2, 0.2),
      box(4.5, 0.5, 0.0, 0.2, 0.2),
    },
    1.0, 16U);
  ASSERT_EQ(samples.size(), 5U);

  const auto grid = build_dynamic_grid(
    geometry(6U, 1U), samples, config(),
    std::vector<std::int8_t>{100, 100, 0, 100, 100, 100});

  EXPECT_EQ(grid, (std::vector<std::int8_t>{0, 0, 100, 0, 0, 0}));
}

TEST(DynamicGridBuilder, PredictionInterpolationUsesShortestYawAndBoundsWork)
{
  const double pi = std::acos(-1.0);
  auto first = box(0.0, 0.0, pi - 0.1, 1.0, 1.0);
  auto second = box(1.0, 0.0, -pi + 0.1, 3.0, 2.0);
  second.covariance_xx = 2.0;

  const auto samples = interpolate_dynamic_trajectory(
    {first, second}, 0.5, 3U);
  ASSERT_EQ(samples.size(), 3U);
  EXPECT_NEAR(std::abs(samples[1].yaw_rad), pi, 1.0e-9);
  EXPECT_NEAR(samples[1].length_m, 2.0, 1.0e-9);
  EXPECT_NEAR(samples[1].width_m, 1.5, 1.0e-9);
  EXPECT_NEAR(samples[1].covariance_xx, 1.0, 1.0e-9);

  EXPECT_THROW(
    interpolate_dynamic_trajectory({first, second}, 0.5, 2U),
    std::length_error);
  EXPECT_THROW(
    interpolate_dynamic_trajectory({first, second}, 0.0, 3U),
    std::invalid_argument);
  EXPECT_THROW(
    interpolate_dynamic_trajectory({}, 0.5, 3U),
    std::invalid_argument);
}

TEST(DynamicGridBuilder, KeepsOnlyPredictedFootprintCellsInsideDrivableMask)
{
  const auto grid_geometry = geometry(6U, 1U);
  const std::vector<std::int8_t> drivable_mask{100, 100, 0, 100, 0, 100};

  const auto grid = build_dynamic_grid(
    grid_geometry,
    {
      box(0.5, 0.5, 0.0, 0.2, 0.2),
      box(2.5, 0.5, 0.0, 0.2, 0.2),
      box(4.5, 0.5, 0.0, 0.2, 0.2),
    },
    config(), drivable_mask);

  EXPECT_EQ(grid, (std::vector<std::int8_t>{0, 0, 100, 0, 100, 0}));
}

TEST(DynamicGridBuilder, DrivableMaskUnknownAndNonzeroCellsFailClosed)
{
  const auto grid_geometry = geometry(3U, 1U);
  const std::vector<std::int8_t> drivable_mask{0, -1, 50};

  const auto grid = build_dynamic_grid(
    grid_geometry,
    {
      box(0.5, 0.5, 0.0, 0.2, 0.2),
      box(1.5, 0.5, 0.0, 0.2, 0.2),
      box(2.5, 0.5, 0.0, 0.2, 0.2),
    },
    config(), drivable_mask);

  EXPECT_EQ(grid, (std::vector<std::int8_t>{100, 0, 0}));
}

TEST(DynamicGridBuilder, RejectsMalformedDrivableMask)
{
  const auto grid_geometry = geometry(3U, 1U);

  EXPECT_THROW(
    build_dynamic_grid(
      grid_geometry, {}, config(), std::vector<std::int8_t>{0, 0}),
    std::invalid_argument);
  EXPECT_THROW(
    build_dynamic_grid(
      grid_geometry, {}, config(), std::vector<std::int8_t>{0, 0, 101}),
    std::invalid_argument);
  EXPECT_THROW(
    build_dynamic_grid(
      grid_geometry, {}, config(), std::vector<std::int8_t>{0, 0, -2}),
    std::invalid_argument);
}

TEST(DynamicGridBuilder, RejectsMalformedGeometryBeforeAllocation)
{
  const auto objects = std::vector<DynamicBox>{};

  auto invalid = geometry();
  invalid.x_min_m = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(build_dynamic_grid(invalid, objects, config()), std::invalid_argument);

  invalid = geometry();
  invalid.resolution_m = 0.0;
  EXPECT_THROW(build_dynamic_grid(invalid, objects, config()), std::invalid_argument);

  invalid = geometry();
  invalid.width = 0U;
  EXPECT_THROW(build_dynamic_grid(invalid, objects, config()), std::invalid_argument);

  invalid = geometry();
  invalid.height = 0U;
  EXPECT_THROW(build_dynamic_grid(invalid, objects, config()), std::invalid_argument);

  invalid = geometry();
  invalid.width = std::numeric_limits<std::size_t>::max();
  invalid.height = 2U;
  EXPECT_THROW(build_dynamic_grid(invalid, objects, config()), std::invalid_argument);
}

TEST(DynamicGridBuilder, RejectsInvalidConfiguration)
{
  auto invalid = config();
  invalid.covariance_sigma = -1.0;
  EXPECT_THROW(build_dynamic_grid(geometry(), {}, invalid), std::invalid_argument);

  invalid = config();
  invalid.minimum_inflation_m = std::numeric_limits<double>::infinity();
  EXPECT_THROW(build_dynamic_grid(geometry(), {}, invalid), std::invalid_argument);

  invalid = config();
  invalid.occupied_cost = 0;
  EXPECT_THROW(build_dynamic_grid(geometry(), {}, invalid), std::invalid_argument);

  invalid = config();
  invalid.occupied_cost = 101;
  EXPECT_THROW(build_dynamic_grid(geometry(), {}, invalid), std::invalid_argument);

  invalid = config();
  invalid.maximum_cells_per_object = 0U;
  EXPECT_THROW(build_dynamic_grid(geometry(), {}, invalid), std::invalid_argument);
}

TEST(DynamicGridBuilder, RejectsInvalidObjectsAndNonPsdCovariance)
{
  auto invalid = box(1.0, 1.0, 0.0, 1.0, 1.0);
  invalid.x_m = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(
    build_dynamic_grid(geometry(), {invalid}, config()), std::invalid_argument);

  invalid = box(1.0, 1.0, 0.0, 1.0, 1.0);
  invalid.yaw_rad = std::numeric_limits<double>::infinity();
  EXPECT_THROW(
    build_dynamic_grid(geometry(), {invalid}, config()), std::invalid_argument);

  invalid = box(1.0, 1.0, 0.0, 0.0, 1.0);
  EXPECT_THROW(
    build_dynamic_grid(geometry(), {invalid}, config()), std::invalid_argument);

  invalid = box(1.0, 1.0, 0.0, 1.0, -1.0);
  EXPECT_THROW(
    build_dynamic_grid(geometry(), {invalid}, config()), std::invalid_argument);

  invalid = box(1.0, 1.0, 0.0, 1.0, 1.0, -0.1, 0.0, 1.0);
  EXPECT_THROW(
    build_dynamic_grid(geometry(), {invalid}, config()), std::invalid_argument);

  invalid = box(1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0);
  EXPECT_THROW(
    build_dynamic_grid(geometry(), {invalid}, config()), std::invalid_argument);
}

TEST(DynamicGridBuilder, SkipsOversizedPerObjectCandidateWithoutRasterizing)
{
  auto guarded = config();
  guarded.maximum_cells_per_object = 10U;

  std::size_t skipped = 99U;
  const auto grid = build_dynamic_grid(
    geometry(100U, 100U), {box(50.0, 50.0, 0.0, 100.0, 100.0)}, guarded,
    &skipped);

  EXPECT_EQ(skipped, 1U);
  EXPECT_EQ(grid.size(), 100U * 100U);
  EXPECT_TRUE(
    std::all_of(
      grid.begin(), grid.end(),
      [](const std::int8_t value) {return value == 0;}));
}

TEST(DynamicGridBuilder, OversizedObjectDoesNotEraseOtherValidObjects)
{
  auto guarded = config();
  guarded.maximum_cells_per_object = 10U;
  const auto grid_geometry = geometry(100U, 100U);

  std::size_t skipped = 0U;
  const auto grid = build_dynamic_grid(
    grid_geometry,
    {
      box(2.0, 2.0, 0.0, 1.0, 1.0),                 // small, valid
      box(50.0, 50.0, 0.0, 100.0, 100.0),           // oversized -> skipped
      box(90.0, 90.0, 0.0, 1.0, 1.0),               // small, valid
    },
    guarded, &skipped);

  EXPECT_EQ(skipped, 1U);
  EXPECT_EQ(at(grid, grid_geometry, 2U, 2U), 100);
  EXPECT_EQ(at(grid, grid_geometry, 90U, 90U), 100);
  const auto occupied = std::count_if(
    grid.begin(), grid.end(),
    [](const std::int8_t value) {return value > 0;});
  EXPECT_GT(occupied, 0);
  EXPECT_LT(occupied, 100 * 100);  // the 100x100 oversized box was not painted
}

TEST(DynamicGridBuilder, CountsEveryOversizedObjectSeparately)
{
  auto guarded = config();
  guarded.maximum_cells_per_object = 10U;

  std::size_t skipped = 0U;
  const auto grid = build_dynamic_grid(
    geometry(100U, 100U),
    {
      box(20.0, 20.0, 0.0, 100.0, 100.0),
      box(60.0, 60.0, 0.0, 100.0, 100.0),
      box(80.0, 80.0, 0.0, 1.0, 1.0),
    },
    guarded, &skipped);

  EXPECT_EQ(skipped, 2U);
  EXPECT_GT(
    std::count_if(
      grid.begin(), grid.end(),
      [](const std::int8_t value) {return value > 0;}),
    0);
}

TEST(DynamicGridBuilder, HighButFiniteCovarianceIsSkippedNotFatal)
{
  auto guarded = config();
  guarded.covariance_sigma = 2.0;
  guarded.maximum_cells_per_object = 20000U;

  std::size_t skipped = 0U;
  // 359 m^2 position variance -> ~38 m inflation, the observed AB3DMOT case.
  const auto grid = build_dynamic_grid(
    GridGeometry{-4.0, -10.0, 0.1, 1040U, 200U},
    {box(40.0, 0.0, 0.0, 1.0, 1.0, 359.0, 0.0, 359.0)},
    guarded, &skipped);

  EXPECT_EQ(skipped, 1U);
  EXPECT_EQ(grid.size(), 1040U * 200U);
  EXPECT_TRUE(
    std::all_of(
      grid.begin(), grid.end(),
      [](const std::int8_t value) {return value == 0;}));
}

TEST(DynamicGridBuilder, MalformedObjectStillThrowsForFailSafeClear)
{
  std::size_t skipped = 0U;
  // NaN position: genuine upstream corruption, not merely large -> fatal.
  EXPECT_THROW(
    build_dynamic_grid(
      geometry(), {box(std::nan(""), 0.0, 0.0, 1.0, 1.0)}, config(), &skipped),
    std::invalid_argument);
  // Negative covariance diagonal: not positive-semidefinite -> fatal.
  EXPECT_THROW(
    build_dynamic_grid(
      geometry(), {box(1.0, 1.0, 0.0, 1.0, 1.0, -5.0, 0.0, 1.0)}, config(),
      &skipped),
    std::invalid_argument);
}

TEST(DynamicGridBuilder, OversizedSkipIsDeterministicAndOutParamResets)
{
  auto guarded = config();
  guarded.maximum_cells_per_object = 10U;
  const std::vector<DynamicBox> objects{
    box(2.0, 2.0, 0.0, 1.0, 1.0),
    box(50.0, 50.0, 0.0, 100.0, 100.0)};

  std::size_t skipped_a = 7U;
  const auto grid_a = build_dynamic_grid(
    geometry(100U, 100U), objects, guarded, &skipped_a);
  std::size_t skipped_b = 0U;
  const auto grid_b = build_dynamic_grid(
    geometry(100U, 100U), objects, guarded, &skipped_b);

  EXPECT_EQ(skipped_a, 1U);
  EXPECT_EQ(skipped_b, 1U);
  EXPECT_EQ(grid_a, grid_b);

  // No oversized object -> counter resets to 0, not left stale.
  std::size_t skipped_c = 5U;
  (void)build_dynamic_grid(
    geometry(100U, 100U), {box(2.0, 2.0, 0.0, 1.0, 1.0)}, guarded, &skipped_c);
  EXPECT_EQ(skipped_c, 0U);
}

using ad_lidar_perception::occupancy_grid::sweep_object_footprints;

// Centroid (mean cell centre, grid units) of every occupied cell.
std::pair<double, double> occupied_centroid(
  const std::vector<std::int8_t> & grid, const GridGeometry & grid_geometry)
{
  double sum_x = 0.0;
  double sum_y = 0.0;
  std::size_t count = 0U;
  for (std::size_t y = 0U; y < grid_geometry.height; ++y) {
    for (std::size_t x = 0U; x < grid_geometry.width; ++x) {
      if (grid.at(y * grid_geometry.width + x) > 0) {
        sum_x += static_cast<double>(x);
        sum_y += static_cast<double>(y);
        ++count;
      }
    }
  }
  if (count == 0U) {
    return {0.0, 0.0};
  }
  return {sum_x / static_cast<double>(count), sum_y / static_cast<double>(count)};
}

// A: with no in-horizon predicted state the sweep is exactly the legacy single
// current footprint (this is also the node's feature-disabled / empty-future
// fallback path; the node-level flag-off contract is locked by
// test_occupancy_layer_launch.py's "must not collapse future predictions").
TEST(DynamicGridFutureSweep, NoFutureKeyframeReproducesLegacySingleFootprint)
{
  const auto current = box(1.5, 0.5, 0.0, 0.2, 0.2);
  const auto swept = sweep_object_footprints({current}, 0.1, 64U);

  ASSERT_EQ(swept.size(), 1U);
  const auto grid_geometry = geometry(6U, 1U);
  EXPECT_EQ(
    build_dynamic_grid(grid_geometry, swept, config()),
    build_dynamic_grid(grid_geometry, {current}, config()));
}

// H: a stationary object (identical keyframes) never sweeps more than its
// current footprint.
TEST(DynamicGridFutureSweep, StationaryObjectDoesNotExpandArea)
{
  const auto current = box(2.5, 0.5, 0.0, 0.2, 0.2);
  const auto swept = sweep_object_footprints({current, current, current}, 0.1, 64U);

  const auto grid_geometry = geometry(6U, 1U);
  EXPECT_EQ(
    build_dynamic_grid(grid_geometry, swept, config()),
    build_dynamic_grid(grid_geometry, {current}, config()));
}

// B: a straight predicted future produces one continuous occupied corridor with
// no gap between the sparse keyframes.
TEST(DynamicGridFutureSweep, StraightFutureProducesGapFreeCorridor)
{
  const auto swept = sweep_object_footprints(
    {box(0.5, 0.5, 0.0, 0.4, 0.4), box(9.5, 0.5, 0.0, 0.4, 0.4)},
    0.1, 512U);

  const auto grid_geometry = geometry(10U, 1U);
  const auto grid = build_dynamic_grid(grid_geometry, swept, config());
  for (std::size_t x = 0U; x < 10U; ++x) {
    EXPECT_EQ(at(grid, grid_geometry, x, 0U), 100)
      << "gap in swept corridor at x=" << x;
  }
}

// E: the sweep never extends past the last keyframe it was given (the node
// clips keyframes to future_sweep_horizon_s before calling this).
TEST(DynamicGridFutureSweep, SweepIsBoundedByTheLastKeyframe)
{
  const auto swept = sweep_object_footprints(
    {box(0.5, 0.5, 0.0, 0.4, 0.4), box(3.5, 0.5, 0.0, 0.4, 0.4)},
    0.1, 512U);

  const auto grid_geometry = geometry(12U, 1U);
  const auto grid = build_dynamic_grid(grid_geometry, swept, config());
  // Everything beyond the last keyframe (x=3.5) plus its half-width stays free.
  for (std::size_t x = 5U; x < 12U; ++x) {
    EXPECT_EQ(at(grid, grid_geometry, x, 0U), 0) << "occupancy past horizon at x=" << x;
  }
}

// C: a future that bends +y (left) sweeps a curved region -- its occupied
// centroid is displaced toward +y relative to a straight sweep from the same
// start pose and travel distance.
TEST(DynamicGridFutureSweep, LeftBendingFutureSweepsCurvedRegion)
{
  const auto grid_geometry = geometry(14U, 14U);

  const auto straight = build_dynamic_grid(
    grid_geometry,
    sweep_object_footprints(
      {box(1.0, 1.0, 0.0, 0.4, 0.4), box(8.0, 1.0, 0.0, 0.4, 0.4)},
      0.1, 512U),
    config());
  const auto curved = build_dynamic_grid(
    grid_geometry,
    sweep_object_footprints(
      {
        box(1.0, 1.0, 0.0, 0.4, 0.4),
        box(5.0, 4.0, 0.6, 0.4, 0.4),
        box(7.0, 9.0, 1.2, 0.4, 0.4),
        box(7.5, 12.0, 1.5, 0.4, 0.4),
      },
      0.1, 512U),
    config());

  const auto straight_centroid = occupied_centroid(straight, grid_geometry);
  const auto curved_centroid = occupied_centroid(curved, grid_geometry);
  // Straight sweep centroid sits near the start row; the curved sweep pulls the
  // occupied mass strongly toward +y.
  EXPECT_GT(curved_centroid.second, straight_centroid.second + 3.0);
}

// D: the opposite turn sweeps the mirror region -- symmetric lateral
// displacement, opposite sign.
TEST(DynamicGridFutureSweep, RightBendingFutureIsTheMirrorOfLeftBending)
{
  const auto grid_geometry = geometry(14U, 28U);
  const double y0 = 13.5;

  const auto sweep_arm = [&](const double s) {
      return build_dynamic_grid(
        grid_geometry,
        sweep_object_footprints(
          {
            box(1.0, y0, 0.0, 0.4, 0.4),
            box(5.0, y0 + s * 3.0, s * 0.6, 0.4, 0.4),
            box(7.0, y0 + s * 8.0, s * 1.2, 0.4, 0.4),
            box(7.5, y0 + s * 11.0, s * 1.5, 0.4, 0.4),
          },
          0.1, 512U),
        config());
    };

  const auto left_centroid = occupied_centroid(sweep_arm(1.0), grid_geometry);
  const auto right_centroid = occupied_centroid(sweep_arm(-1.0), grid_geometry);

  EXPECT_GT(left_centroid.second, y0 + 2.0);
  EXPECT_LT(right_centroid.second, y0 - 2.0);
  EXPECT_NEAR(
    left_centroid.second - y0, y0 - right_centroid.second, 1.0);
  EXPECT_NEAR(left_centroid.first, right_centroid.first, 1.0);
}

// F: a fast object with sparse keyframes still fills every grid cell along the
// path -- spacing derives from the footprint size, not the keyframe spacing.
TEST(DynamicGridFutureSweep, FastObjectSparseKeyframesHasNoGridHoles)
{
  const auto swept = sweep_object_footprints(
    {box(0.5, 0.5, 0.0, 0.3, 0.3), box(59.5, 0.5, 0.0, 0.3, 0.3)},
    0.1, 2048U);

  const auto grid_geometry = geometry(60U, 1U);
  const auto grid = build_dynamic_grid(grid_geometry, swept, config());
  EXPECT_TRUE(
    std::all_of(
      grid.begin(), grid.end(),
      [](const std::int8_t value) {return value == 100;}));
}

// G: the drivable-mask gate still applies to every swept footprint -- swept
// cells outside the mask stay free, exactly as for the current footprint.
TEST(DynamicGridFutureSweep, DrivableMaskGatesEverySweptFootprint)
{
  const auto swept = sweep_object_footprints(
    {box(0.5, 0.5, 0.0, 0.4, 0.4), box(5.5, 0.5, 0.0, 0.4, 0.4)},
    0.1, 128U);

  const auto grid_geometry = geometry(6U, 1U);
  //             x=0  1    2  3    4  5
  const std::vector<std::int8_t> drivable_mask{0, 100, 0, 100, 0, 0};
  const auto grid = build_dynamic_grid(
    grid_geometry, swept, config(), drivable_mask);

  EXPECT_EQ(grid, (std::vector<std::int8_t>{100, 0, 100, 0, 100, 100}));
}

// I: a non-finite predicted keyframe and an unbounded expansion both throw --
// the node catches these and falls back to the current footprint, never a
// crash, NaN, or grid explosion.
TEST(DynamicGridFutureSweep, RejectsNonFiniteKeyframeAndUnboundedExpansion)
{
  auto bad = box(3.0, 0.0, 0.0, 0.4, 0.4);
  bad.x_m = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(
    sweep_object_footprints({box(0.0, 0.0, 0.0, 0.4, 0.4), bad}, 0.1, 128U),
    std::invalid_argument);

  // 400 m at 0.2 m spacing = 2000 samples > the caller's budget.
  EXPECT_THROW(
    sweep_object_footprints(
      {box(0.0, 0.0, 0.0, 0.4, 0.4), box(400.0, 0.0, 0.0, 0.4, 0.4)},
      0.1, 64U),
    std::length_error);

  EXPECT_THROW(
    sweep_object_footprints({box(0.0, 0.0, 0.0, 0.4, 0.4)}, 0.0, 64U),
    std::invalid_argument);
}

// The keyframe budget comfortably exceeds a realistic in-horizon prediction:
// current + ~7 states at 0.5 s spacing over a 3 s horizon is 8 keyframes.
TEST(DynamicGridFutureSweep, KeyframeBudgetExceedsRealisticPredictionDepth)
{
  std::vector<DynamicBox> footprints;
  for (int i = 0; i < 8; ++i) {
    footprints.push_back(
      box(static_cast<double>(i) * 4.0, 0.0, 0.0, 4.5, 2.0));
  }
  EXPECT_NO_THROW(sweep_object_footprints(footprints, 0.1, 2048U));
}

}  // namespace
