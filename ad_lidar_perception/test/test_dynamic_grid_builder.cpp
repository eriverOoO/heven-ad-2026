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

// ---------------------------------------------------------------------------
// Per-object future-sweep budget (footprint_candidate_cells + budget_object_sweep)
// ---------------------------------------------------------------------------
using ad_lidar_perception::occupancy_grid::SweepOutcome;
using ad_lidar_perception::occupancy_grid::budget_object_sweep;
using ad_lidar_perception::occupancy_grid::footprint_candidate_cells;

// footprint_candidate_cells reports exactly the grid-clipped inflated
// bounding-box size the builder's per-object guard is checked against: a box
// just under the budget is rasterized, a box just over is skip-counted, and an
// entirely off-grid box reports 0.
TEST(FootprintCandidateCells, PredictsBuilderOversizedDecisionExactly)
{
  auto guarded = config();
  guarded.maximum_cells_per_object = 100U;
  const auto grid_geometry = geometry(50U, 50U, 1.0);

  const auto within = box(25.0, 25.0, 0.0, 8.0, 8.0);   // 10 x 10 clipped cells
  EXPECT_EQ(footprint_candidate_cells(grid_geometry, within, guarded), 100U);
  std::size_t within_skipped = 5U;
  (void)build_dynamic_grid(grid_geometry, {within}, guarded, &within_skipped);
  EXPECT_EQ(within_skipped, 0U);

  const auto over = box(25.0, 25.0, 0.0, 12.0, 12.0);   // 14 x 14 clipped cells
  EXPECT_GT(footprint_candidate_cells(grid_geometry, over, guarded), 100U);
  std::size_t over_skipped = 0U;
  (void)build_dynamic_grid(grid_geometry, {over}, guarded, &over_skipped);
  EXPECT_EQ(over_skipped, 1U);

  EXPECT_EQ(
    footprint_candidate_cells(
      grid_geometry, box(1000.0, 1000.0, 0.0, 1.0, 1.0), guarded),
    0U);
}

// A: with the sweep disabled the node feeds a single current footprint and
// build_dynamic_grid rasterizes it exactly as before -- locked by
// DynamicGridFutureSweep.NoFutureKeyframeReproducesLegacySingleFootprint and by
// test_occupancy_layer_launch.py's flag-off contract. budget_object_sweep is
// only ever reached with the sweep enabled and >= 2 footprints; it rejects a
// degenerate call.
TEST(BudgetObjectSweep, RejectsFewerThanTwoFootprints)
{
  EXPECT_THROW(
    budget_object_sweep(
      geometry(20U, 20U, 1.0), {box(1.0, 1.0, 0.0, 1.0, 1.0)}, config(), 2048U),
    std::invalid_argument);
}

// B: a normal moving object with a reasonable total returns the full gap-free
// sweep and no footprint is skip-counted.
TEST(BudgetObjectSweep, NormalMovingObjectReturnsFullSweepWithNoSkip)
{
  const auto grid_geometry = geometry(400U, 20U, 0.1);
  const std::vector<DynamicBox> keyframes{
    box(0.5, 1.0, 0.0, 0.4, 0.4), box(30.5, 1.0, 0.0, 0.4, 0.4)};

  SweepOutcome outcome{};
  const auto group = budget_object_sweep(
    grid_geometry, keyframes, config(), 2048U, &outcome);

  EXPECT_EQ(outcome, SweepOutcome::kFullSweep);
  EXPECT_GT(group.size(), 50U);
  std::size_t skipped = 9U;
  const auto grid = build_dynamic_grid(grid_geometry, group, config(), &skipped);
  EXPECT_EQ(skipped, 0U);
  EXPECT_GT(
    std::count_if(
      grid.begin(), grid.end(), [](std::int8_t v) {return v > 0;}),
    0);
}

// C: a long valid sweep contributes many interpolated footprints, none of which
// is counted as an oversized object.
TEST(BudgetObjectSweep, ManyValidSweepFootprintsAreNeverCountedOversized)
{
  const auto grid_geometry = geometry(600U, 20U, 0.1);
  const std::vector<DynamicBox> keyframes{
    box(0.5, 1.0, 0.0, 0.3, 0.3), box(50.5, 1.0, 0.0, 0.3, 0.3)};

  SweepOutcome outcome{};
  const auto group = budget_object_sweep(
    grid_geometry, keyframes, config(), 2048U, &outcome);

  EXPECT_EQ(outcome, SweepOutcome::kFullSweep);
  EXPECT_GT(group.size(), 150U);
  std::size_t skipped = 3U;
  (void)build_dynamic_grid(grid_geometry, group, config(), &skipped);
  EXPECT_EQ(skipped, 0U);
}

// D: a single pathological future footprint (one interpolated box over the
// per-footprint budget) drops the whole future expansion; the in-budget current
// footprint is kept and rasterizes with no skip.
TEST(BudgetObjectSweep, SinglePathologicalFutureFootprintDropsWholeSweep)
{
  auto guarded = config();  // maximum_cells_per_object = 10000
  const auto grid_geometry = geometry(200U, 200U, 1.0);  // grid budget 40000
  const std::vector<DynamicBox> keyframes{
    box(20.0, 20.0, 0.0, 2.0, 2.0),        // current: tiny, in budget
    box(60.0, 60.0, 0.0, 250.0, 250.0)};   // future: clips to ~200 x 200 cells

  SweepOutcome outcome{};
  const auto group = budget_object_sweep(
    grid_geometry, keyframes, guarded, 2048U, &outcome);

  EXPECT_EQ(outcome, SweepOutcome::kBudgetCapped);
  ASSERT_EQ(group.size(), 1U);
  EXPECT_DOUBLE_EQ(group.front().x_m, 20.0);
  EXPECT_DOUBLE_EQ(group.front().length_m, 2.0);
  std::size_t skipped = 4U;
  const auto grid = build_dynamic_grid(
    grid_geometry, group, guarded, &skipped);
  EXPECT_EQ(skipped, 0U);
  EXPECT_GT(
    std::count_if(
      grid.begin(), grid.end(), [](std::int8_t v) {return v > 0;}),
    0);
}

// E: an aggregate-pathological sweep -- every individual footprint fits, but
// the group's total grid-clipped candidate-cell work exceeds one full grid --
// is dropped whole (deterministic, all-or-nothing), current footprint kept.
TEST(BudgetObjectSweep, AggregatePathologicalSweepIsDroppedWhole)
{
  auto guarded = config();  // maximum_cells_per_object = 10000
  const auto grid_geometry = geometry(200U, 200U, 1.0);  // grid budget 40000
  std::vector<DynamicBox> keyframes;
  // 10 footprints, each ~91 x 91 = 8281 candidate cells (< 10000), 5 m apart:
  // one interpolation interval each -> ~10 footprints, sum ~82000 > 40000.
  for (int i = 0; i < 10; ++i) {
    keyframes.push_back(
      box(60.0 + static_cast<double>(i) * 5.0, 100.0, 0.0, 90.0, 90.0));
  }

  SweepOutcome outcome{};
  const auto group = budget_object_sweep(
    grid_geometry, keyframes, guarded, 2048U, &outcome);

  EXPECT_EQ(outcome, SweepOutcome::kBudgetCapped);
  ASSERT_EQ(group.size(), 1U);
  EXPECT_DOUBLE_EQ(group.front().x_m, 60.0);

  // Deterministic: a second identical call gives the identical result.
  SweepOutcome repeat_outcome{};
  const auto repeat = budget_object_sweep(
    grid_geometry, keyframes, guarded, 2048U, &repeat_outcome);
  EXPECT_EQ(repeat_outcome, SweepOutcome::kBudgetCapped);
  EXPECT_EQ(repeat.size(), 1U);
}

// F: when the current footprint itself is pathological the object keeps the
// pre-sweep legacy semantics exactly -- only the current footprint is emitted
// and build_dynamic_grid skip-counts it once.
TEST(BudgetObjectSweep, PathologicalCurrentFootprintKeepsLegacySkipSemantics)
{
  auto guarded = config();  // maximum_cells_per_object = 10000
  const auto grid_geometry = geometry(200U, 200U, 1.0);
  const std::vector<DynamicBox> keyframes{
    box(100.0, 100.0, 0.0, 300.0, 300.0),   // current: clips to ~200 x 200
    box(150.0, 100.0, 0.0, 300.0, 300.0)};

  SweepOutcome outcome{};
  const auto group = budget_object_sweep(
    grid_geometry, keyframes, guarded, 2048U, &outcome);

  EXPECT_EQ(outcome, SweepOutcome::kCurrentOversized);
  ASSERT_EQ(group.size(), 1U);
  EXPECT_DOUBLE_EQ(group.front().length_m, 300.0);
  std::size_t skipped = 0U;
  const auto grid = build_dynamic_grid(
    grid_geometry, group, guarded, &skipped);
  EXPECT_EQ(skipped, 1U);
  EXPECT_TRUE(
    std::all_of(
      grid.begin(), grid.end(), [](std::int8_t v) {return v == 0;}));
}

// G: one physical object with many swept footprints that fails the budget once
// increments the oversized counter by 1, never by the footprint count. Contrast
// against feeding the raw (un-budgeted) sweep, which multi-counts.
TEST(BudgetObjectSweep, OversizedSweptGroupIncrementsTheCounterByOneNotMany)
{
  auto guarded = config();  // maximum_cells_per_object = 10000
  const auto grid_geometry = geometry(200U, 200U, 1.0);
  const std::vector<DynamicBox> keyframes{
    box(20.0, 100.0, 0.0, 2.0, 2.0),        // current: in budget
    box(120.0, 100.0, 0.0, 180.0, 180.0)};  // grows past budget mid-sweep

  SweepOutcome outcome{};
  const auto group = budget_object_sweep(
    grid_geometry, keyframes, guarded, 2048U, &outcome);
  EXPECT_EQ(outcome, SweepOutcome::kBudgetCapped);
  ASSERT_EQ(group.size(), 1U);
  std::size_t budgeted_skipped = 0U;
  (void)build_dynamic_grid(grid_geometry, group, guarded, &budgeted_skipped);
  EXPECT_EQ(budgeted_skipped, 0U);  // current footprint is in budget

  const auto raw_sweep = sweep_object_footprints(
    keyframes, grid_geometry.resolution_m, 2048U);
  std::size_t raw_skipped = 0U;
  (void)build_dynamic_grid(
    grid_geometry, raw_sweep, guarded, &raw_skipped);
  EXPECT_GT(raw_skipped, 1U);  // the pre-fix per-temporal-footprint multi-count
}

// H: two objects, one valid and one oversized -- the valid object's complete
// sweep is still rasterized; only the oversized object is affected/counted.
TEST(BudgetObjectSweep, ValidObjectSweepSurvivesAlongsideOversizedObject)
{
  auto guarded = config();  // maximum_cells_per_object = 10000
  const auto grid_geometry = geometry(200U, 60U, 1.0);

  SweepOutcome valid_outcome{};
  auto valid_group = budget_object_sweep(
    grid_geometry,
    {box(10.0, 30.0, 0.0, 2.0, 2.0), box(40.0, 30.0, 0.0, 2.0, 2.0)},
    guarded, 2048U, &valid_outcome);
  EXPECT_EQ(valid_outcome, SweepOutcome::kFullSweep);

  SweepOutcome oversized_outcome{};
  auto oversized_group = budget_object_sweep(
    grid_geometry,
    {box(120.0, 30.0, 0.0, 300.0, 300.0), box(140.0, 30.0, 0.0, 300.0, 300.0)},
    guarded, 2048U, &oversized_outcome);
  EXPECT_EQ(oversized_outcome, SweepOutcome::kCurrentOversized);

  std::vector<DynamicBox> all;
  all.insert(all.end(), valid_group.begin(), valid_group.end());
  all.insert(all.end(), oversized_group.begin(), oversized_group.end());
  std::size_t skipped = 0U;
  const auto grid = build_dynamic_grid(
    grid_geometry, all, guarded, &skipped);

  EXPECT_EQ(skipped, 1U);
  EXPECT_EQ(at(grid, grid_geometry, 25U, 30U), 100);  // valid object's corridor
}

// I: budgeting must not straighten or otherwise alter a curved sweep -- an
// under-budget curved object rasterizes to exactly the raw sweep.
TEST(BudgetObjectSweep, CurvedSweepGeometryIsUnchangedByBudgeting)
{
  const auto grid_geometry = geometry(40U, 40U, 1.0);
  const std::vector<DynamicBox> keyframes{
    box(2.0, 2.0, 0.0, 0.4, 0.4),
    box(10.0, 8.0, 0.6, 0.4, 0.4),
    box(16.0, 18.0, 1.2, 0.4, 0.4),
    box(18.0, 26.0, 1.5, 0.4, 0.4)};

  const auto raw = sweep_object_footprints(
    keyframes, grid_geometry.resolution_m, 2048U);
  SweepOutcome outcome{};
  const auto budgeted = budget_object_sweep(
    grid_geometry, keyframes, config(), 2048U, &outcome);

  EXPECT_EQ(outcome, SweepOutcome::kFullSweep);
  ASSERT_EQ(raw.size(), budgeted.size());
  for (std::size_t i = 0U; i < raw.size(); ++i) {
    EXPECT_DOUBLE_EQ(raw[i].x_m, budgeted[i].x_m);
    EXPECT_DOUBLE_EQ(raw[i].y_m, budgeted[i].y_m);
    EXPECT_DOUBLE_EQ(raw[i].yaw_rad, budgeted[i].yaw_rad);
  }
  EXPECT_EQ(
    build_dynamic_grid(grid_geometry, raw, config()),
    build_dynamic_grid(grid_geometry, budgeted, config()));
}

// J: a stationary object neither expands its occupied area nor inflates the
// budget -- its swept group rasterizes to exactly its current footprint.
TEST(BudgetObjectSweep, StationaryObjectStaysCurrentFootprintOnly)
{
  const auto grid_geometry = geometry(20U, 20U, 1.0);
  const auto current = box(10.0, 10.0, 0.0, 1.0, 1.0);

  SweepOutcome outcome{};
  const auto group = budget_object_sweep(
    grid_geometry, {current, current, current}, config(), 2048U, &outcome);

  EXPECT_EQ(outcome, SweepOutcome::kFullSweep);
  EXPECT_EQ(
    build_dynamic_grid(grid_geometry, group, config()),
    build_dynamic_grid(grid_geometry, {current}, config()));
}

// A malformed current footprint still throws (fail-safe clear), a malformed
// future keyframe instead falls back to the current footprint.
TEST(BudgetObjectSweep, MalformedCurrentThrowsMalformedFutureFallsBack)
{
  auto non_psd = box(1.0, 1.0, 0.0, 1.0, 1.0);
  non_psd.covariance_xx = -5.0;
  EXPECT_THROW(
    budget_object_sweep(
      geometry(20U, 20U, 1.0), {non_psd, box(2.0, 1.0, 0.0, 1.0, 1.0)},
      config(), 2048U),
    std::invalid_argument);

  auto nan_future = box(5.0, 0.0, 0.0, 0.4, 0.4);
  nan_future.x_m = std::numeric_limits<double>::quiet_NaN();
  SweepOutcome outcome{};
  const auto group = budget_object_sweep(
    geometry(60U, 10U, 1.0), {box(0.0, 5.0, 0.0, 0.4, 0.4), nan_future},
    config(), 2048U, &outcome);
  EXPECT_EQ(outcome, SweepOutcome::kExpansionFailed);
  ASSERT_EQ(group.size(), 1U);
  EXPECT_DOUBLE_EQ(group.front().x_m, 0.0);
}

}  // namespace
