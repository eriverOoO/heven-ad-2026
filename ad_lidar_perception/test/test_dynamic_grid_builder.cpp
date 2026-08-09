#include "ad_lidar_perception/occupancy_grid/dynamic_grid_builder.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
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

TEST(DynamicGridBuilder, RejectsOversizedPerObjectCandidateBeforeRasterizing)
{
  auto guarded = config();
  guarded.maximum_cells_per_object = 10U;

  EXPECT_THROW(
    build_dynamic_grid(
      geometry(100U, 100U), {box(50.0, 50.0, 0.0, 100.0, 100.0)}, guarded),
    std::length_error);
}

}  // namespace
