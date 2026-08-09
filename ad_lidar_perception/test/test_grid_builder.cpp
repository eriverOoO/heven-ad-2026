#include "ad_lidar_perception/occupancy_grid/grid_builder.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace
{

ad_lidar_perception::occupancy_grid::GridConfig test_config()
{
  return ad_lidar_perception::occupancy_grid::GridConfig{
    -2.0, 4.0, -3.0, 3.0, 0.1, 0.7, 1.0, 1.0, std::log(2.0),
    -1.0, 0.5, -0.5, 0.5};
}

std::int8_t at(
  const ad_lidar_perception::occupancy_grid::GridBuilder & builder,
  const std::vector<std::int8_t> & grid, double x, double y)
{
  const auto & config = builder.config();
  const auto grid_x = static_cast<std::size_t>((x - config.x_min) / config.resolution);
  const auto grid_y = static_cast<std::size_t>((y - config.y_min) / config.resolution);
  return grid[grid_y * builder.width() + grid_x];
}

ad_lidar_perception::occupancy_grid::DrivableMask matching_mask(
  const ad_lidar_perception::occupancy_grid::GridBuilder & builder,
  std::int8_t fill = 100)
{
  return ad_lidar_perception::occupancy_grid::DrivableMask{
    builder.config().x_min,
    builder.config().y_min,
    builder.config().resolution,
    builder.width(),
    builder.height(),
    std::vector<std::int8_t>(builder.width() * builder.height(), fill)};
}

void set_mask(
  const ad_lidar_perception::occupancy_grid::GridBuilder & builder,
  ad_lidar_perception::occupancy_grid::DrivableMask & mask,
  double x, double y, std::int8_t value)
{
  const auto & config = builder.config();
  const auto grid_x = static_cast<std::size_t>((x - config.x_min) / config.resolution);
  const auto grid_y = static_cast<std::size_t>((y - config.y_min) / config.resolution);
  mask.data[grid_y * builder.width() + grid_x] = value;
}

TEST(GridBuilder, BuildsFreeGridAndPreservesConfiguredGeometry)
{
  const ad_lidar_perception::occupancy_grid::GridBuilder builder(test_config());
  const auto grid = builder.build({});

  EXPECT_EQ(builder.width(), 6U);
  EXPECT_EQ(builder.height(), 6U);
  EXPECT_EQ(grid.size(), 36U);
  for (const auto cell : grid) {
    EXPECT_EQ(cell, 0);
  }
}

TEST(GridBuilder, DefaultGeometryCoversIoniq5BodyAndDwaForwardHorizon)
{
  const ad_lidar_perception::occupancy_grid::GridConfig config;
  const ad_lidar_perception::occupancy_grid::GridBuilder builder(config);

  EXPECT_DOUBLE_EQ(config.x_min, -4.0);
  EXPECT_DOUBLE_EQ(config.x_max, 100.0);
  EXPECT_DOUBLE_EQ(config.y_min, -10.0);
  EXPECT_DOUBLE_EQ(config.y_max, 10.0);
  EXPECT_DOUBLE_EQ(config.z_min, 0.1);
  EXPECT_DOUBLE_EQ(config.z_max, 2.0);
  EXPECT_DOUBLE_EQ(config.inflation_radius_m, 1.8);
  EXPECT_DOUBLE_EQ(config.inflation_cost_scaling_factor, 2.0);
  EXPECT_DOUBLE_EQ(config.ego_clear_x_min, -1.0);
  EXPECT_DOUBLE_EQ(config.ego_clear_x_max, 4.05);
  EXPECT_DOUBLE_EQ(config.ego_clear_y_min, -1.15);
  EXPECT_DOUBLE_EQ(config.ego_clear_y_max, 1.15);
  EXPECT_EQ(builder.width(), 1040U);
  EXPECT_EQ(builder.height(), 200U);
  EXPECT_DOUBLE_EQ(
    builder.config().resolution,
    static_cast<double>(static_cast<float>(0.1)));
}

TEST(GridBuilder, PreservesUpperBoundExclusionAndUsesExponentialMetricDecay)
{
  auto config = test_config();
  config.inflation_radius_m = 2.0;
  config.ego_clear_x_min = -100.0;
  config.ego_clear_x_max = -99.0;
  const ad_lidar_perception::occupancy_grid::GridBuilder builder(config);
  const auto grid = builder.build(
    {
      {3.999, 2.999, 0.4},
      {4.0, 2.0, 0.4},
      {3.0, 3.0, 0.4},
    });

  EXPECT_EQ(at(builder, grid, 3.2, 2.2), 100);
  EXPECT_EQ(at(builder, grid, 2.2, 2.2), 50);
  EXPECT_EQ(at(builder, grid, 1.2, 2.2), 25);
}

TEST(GridBuilder, FiltersHeightAndEgoThenInflatesUniqueObstacleCells)
{
  const ad_lidar_perception::occupancy_grid::GridBuilder builder(test_config());
  const auto grid = builder.build(
    {
      {2.2, 1.2, 0.4},
      {2.2, 1.2, 0.4},
      {2.2, -1.2, 2.0},
      {0.0, 0.0, 0.4},
    });

  EXPECT_EQ(at(builder, grid, 2.2, 1.2), 100);
  EXPECT_EQ(at(builder, grid, 1.2, 1.2), 50);
  EXPECT_EQ(at(builder, grid, 2.2, 0.2), 50);
  EXPECT_EQ(at(builder, grid, 0.2, 0.2), 0);
  EXPECT_EQ(at(builder, grid, 2.2, -1.2), 0);
}

TEST(GridBuilder, ClearsEgoFootprintAgainAfterObstacleInflation)
{
  auto config = test_config();
  config.resolution = 0.1;
  config.inflation_radius_m = 1.0;
  config.ego_clear_x_min = -0.5;
  config.ego_clear_x_max = 0.5;
  config.ego_clear_y_min = -0.5;
  config.ego_clear_y_max = 0.5;
  const ad_lidar_perception::occupancy_grid::GridBuilder builder(config);
  const auto grid = builder.build({{0.8, 0.0, 0.4}});

  EXPECT_EQ(at(builder, grid, 0.4, 0.0), 0);
  EXPECT_GT(at(builder, grid, 0.6, 0.0), 0);
  EXPECT_EQ(at(builder, grid, 0.8, 0.0), 100);
}

TEST(GridBuilder, DrivableMaskDiscardsNonDrivableAndUnknownSeedsBeforeInflation)
{
  auto config = test_config();
  config.ego_clear_x_min = -100.0;
  config.ego_clear_x_max = -99.0;
  const ad_lidar_perception::occupancy_grid::GridBuilder builder(config);
  auto mask = matching_mask(builder);
  set_mask(builder, mask, 2.2, 1.2, 0);
  set_mask(builder, mask, 2.2, -1.2, -1);

  const auto grid = builder.build(
    {
      {2.2, 1.2, 0.4},
      {2.2, -1.2, 0.4},
      {-1.2, 1.2, 0.4},
    },
    mask);

  EXPECT_EQ(at(builder, grid, 2.2, 1.2), 100);
  EXPECT_EQ(at(builder, grid, 1.2, 1.2), 50);
  EXPECT_EQ(at(builder, grid, 2.2, -1.2), 0);
  EXPECT_EQ(at(builder, grid, 1.2, -1.2), 0);
  EXPECT_EQ(at(builder, grid, -1.2, 1.2), 0);
  EXPECT_EQ(at(builder, grid, -0.2, 1.2), 0);
}

TEST(GridBuilder, MaskedBuildDoesNotMutateOffRoadCellsDuringEgoClearing)
{
  auto config = test_config();
  config.resolution = 0.1;
  config.inflation_radius_m = 1.0;
  config.ego_clear_x_min = -0.5;
  config.ego_clear_x_max = 0.5;
  config.ego_clear_y_min = -0.5;
  config.ego_clear_y_max = 0.5;
  const ad_lidar_perception::occupancy_grid::GridBuilder builder(config);
  auto mask = matching_mask(builder, 0);
  set_mask(builder, mask, 0.4, 0.0, 100);
  const auto original_mask = mask.data;

  const auto grid = builder.build({{0.8, 0.0, 0.4}}, mask);

  EXPECT_EQ(at(builder, grid, 0.4, 0.0), 0);
  EXPECT_EQ(mask.data, original_mask);
}

TEST(GridBuilder, DrivableMaskRejectsEveryGeometryMismatch)
{
  const ad_lidar_perception::occupancy_grid::GridBuilder builder(test_config());
  const auto expect_invalid = [&builder](const auto & mutate) {
      auto mask = matching_mask(builder, 0);
      mutate(mask);
      EXPECT_THROW((void)builder.build({}, mask), std::invalid_argument);
    };

  expect_invalid([](auto & mask) {mask.x_min += mask.resolution;});
  expect_invalid([](auto & mask) {mask.y_min += mask.resolution;});
  expect_invalid([](auto & mask) {mask.resolution *= 2.0;});
  expect_invalid([](auto & mask) {--mask.width;});
  expect_invalid([](auto & mask) {--mask.height;});
  expect_invalid([](auto & mask) {mask.data.pop_back();});
}

TEST(GridBuilder, LargeInflationScalingFactorSaturatesWithoutOverflow)
{
  auto config = test_config();
  config.x_min = 0.0;
  config.x_max = 5.0;
  config.y_min = 0.0;
  config.y_max = 1.0;
  config.resolution = 1.0;
  config.inflation_radius_m = 2.0;
  config.inflation_cost_scaling_factor = std::numeric_limits<double>::max();
  config.ego_clear_x_min = -100.0;
  config.ego_clear_x_max = -99.0;
  const ad_lidar_perception::occupancy_grid::GridBuilder builder(config);
  const auto grid = builder.build({{2.5, 0.5, 0.4}});

  EXPECT_EQ(grid, (std::vector<std::int8_t>{1, 1, 100, 1, 1}));
}

TEST(GridBuilder, HugeFiniteInflationClipsWorkToGridWithoutChangingCosts)
{
  auto config = test_config();
  config.x_min = 0.0;
  config.x_max = 3.0;
  config.y_min = 0.0;
  config.y_max = 3.0;
  config.resolution = 1.0;
  config.inflation_radius_m = 100000.0;
  config.inflation_cost_scaling_factor = std::log(2.0);
  config.ego_clear_x_min = -100.0;
  config.ego_clear_x_max = -99.0;
  const ad_lidar_perception::occupancy_grid::GridBuilder builder(config);
  const auto grid = builder.build({{1.5, 1.5, 0.4}});

  EXPECT_EQ(
    grid,
    (std::vector<std::int8_t>{
      38, 50, 38,
      50, 100, 50,
      38, 50, 38}));
}

TEST(GridBuilder, RejectsInvalidGeometry)
{
  auto config = test_config();
  config.resolution = 0.0;
  EXPECT_THROW(
    (void)ad_lidar_perception::occupancy_grid::GridBuilder{config},
    std::invalid_argument);

  config = test_config();
  config.inflation_cost_scaling_factor = 0.0;
  EXPECT_THROW(
    (void)ad_lidar_perception::occupancy_grid::GridBuilder{config},
    std::invalid_argument);

  config = test_config();
  config.inflation_cost_scaling_factor = -1.0;
  EXPECT_THROW(
    (void)ad_lidar_perception::occupancy_grid::GridBuilder{config},
    std::invalid_argument);

  config = test_config();
  config.inflation_radius_m = std::numeric_limits<double>::max();
  EXPECT_THROW(
    (void)ad_lidar_perception::occupancy_grid::GridBuilder{config},
    std::invalid_argument);

  config = test_config();
  config.resolution = 1.0;
  config.inflation_radius_m =
    static_cast<double>(std::numeric_limits<int>::max());
  EXPECT_THROW(
    (void)ad_lidar_perception::occupancy_grid::GridBuilder{config},
    std::invalid_argument);

  config = test_config();
  config.x_min = 0.0;
  config.x_max = 1.4;
  config.y_min = 0.0;
  config.y_max = 1.0;
  config.resolution = 1.0;
  EXPECT_THROW(
    (void)ad_lidar_perception::occupancy_grid::GridBuilder{config},
    std::invalid_argument);
}

TEST(GridBuilder, RejectsEveryNonfiniteConfigurationValue)
{
  auto expect_invalid = [](const auto & mutate) {
      auto config = test_config();
      mutate(config);
      EXPECT_THROW(
        (void)ad_lidar_perception::occupancy_grid::GridBuilder{config},
        std::invalid_argument);
    };
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();

  expect_invalid([&](auto & config) {config.x_min = nan;});
  expect_invalid([&](auto & config) {config.x_max = infinity;});
  expect_invalid([&](auto & config) {config.y_min = -infinity;});
  expect_invalid([&](auto & config) {config.y_max = nan;});
  expect_invalid([&](auto & config) {config.z_min = nan;});
  expect_invalid([&](auto & config) {config.z_max = infinity;});
  expect_invalid([&](auto & config) {config.resolution = nan;});
  expect_invalid([&](auto & config) {config.inflation_radius_m = infinity;});
  expect_invalid(
    [&](auto & config) {config.inflation_cost_scaling_factor = nan;});
  expect_invalid(
    [&](auto & config) {config.inflation_cost_scaling_factor = infinity;});
  expect_invalid([&](auto & config) {config.ego_clear_x_min = nan;});
  expect_invalid([&](auto & config) {config.ego_clear_x_max = infinity;});
  expect_invalid([&](auto & config) {config.ego_clear_y_min = -infinity;});
  expect_invalid([&](auto & config) {config.ego_clear_y_max = nan;});
  expect_invalid(
    [&](auto & config) {config.ego_clear_x_max = config.ego_clear_x_min - 1.0;});
  expect_invalid(
    [&](auto & config) {config.ego_clear_y_max = config.ego_clear_y_min - 1.0;});
}

TEST(GridBuilder, RejectsUnrepresentableAndOverflowingCellCounts)
{
  auto unrepresentable = test_config();
  unrepresentable.resolution = std::numeric_limits<double>::min();
  EXPECT_THROW(
    (void)ad_lidar_perception::occupancy_grid::GridBuilder{unrepresentable},
    std::invalid_argument);

  auto overflowing = test_config();
  overflowing.x_min = 0.0;
  overflowing.x_max = 4.0e9;
  overflowing.y_min = 0.0;
  overflowing.y_max = 4.0e9;
  overflowing.resolution = 1.0;
  EXPECT_THROW(
    (void)ad_lidar_perception::occupancy_grid::GridBuilder{overflowing},
    std::invalid_argument);

  auto signed_index_overflow = test_config();
  signed_index_overflow.x_min = 0.0;
  signed_index_overflow.x_max =
    static_cast<double>(std::numeric_limits<int>::max()) + 1.0;
  signed_index_overflow.y_min = 0.0;
  signed_index_overflow.y_max = 1.0;
  signed_index_overflow.resolution = 1.0;
  EXPECT_THROW(
    (void)ad_lidar_perception::occupancy_grid::GridBuilder{
      signed_index_overflow},
    std::invalid_argument);
}

}  // namespace
