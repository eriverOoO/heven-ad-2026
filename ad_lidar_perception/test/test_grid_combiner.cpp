#include "ad_lidar_perception/occupancy_grid/grid_combiner.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using ad_lidar_perception::occupancy_grid::GridGeometry;
using ad_lidar_perception::occupancy_grid::GridLayerMetadata;
using ad_lidar_perception::occupancy_grid::combine_cost_layers;
using ad_lidar_perception::occupancy_grid::layers_are_compatible;

GridLayerMetadata metadata()
{
  return GridLayerMetadata{
    GridGeometry{
      -4.0,
      -10.0,
      static_cast<double>(static_cast<float>(0.1)),
      280U,
      200U},
    "base_link",
    1'000'000'000LL};
}

TEST(GridCombiner, AppliesKnownMaximumAndUnknownTruthTable)
{
  const std::vector<std::int8_t> static_cells{-1, -1, 0, -1, 20, 100};
  const std::vector<std::int8_t> dynamic_cells{-1, 0, -1, 60, 60, 60};

  EXPECT_EQ(
    combine_cost_layers(static_cells, dynamic_cells),
    (std::vector<std::int8_t>{-1, 0, 0, 60, 60, 100}));
}

TEST(GridCombiner, RejectsUnequalSizesAndOutOfRangeCosts)
{
  EXPECT_THROW(
    combine_cost_layers({0}, {0, 0}), std::invalid_argument);
  EXPECT_THROW(
    combine_cost_layers({-2}, {0}), std::invalid_argument);
  EXPECT_THROW(
    combine_cost_layers({0}, {101}), std::invalid_argument);
}

TEST(GridCombiner, AcceptsExactGeometryFrameAndStampBoundary)
{
  const auto static_layer = metadata();
  auto dynamic_layer = static_layer;

  EXPECT_TRUE(layers_are_compatible(static_layer, dynamic_layer));
}

TEST(GridCombiner, RejectsEvenOneNanosecondStampDifference)
{
  const auto static_layer = metadata();
  auto dynamic_layer = static_layer;
  dynamic_layer.stamp_ns += 1LL;

  EXPECT_FALSE(layers_are_compatible(static_layer, dynamic_layer));
}

TEST(GridCombiner, NormalizesResolutionThroughOccupancyGridFloatStorage)
{
  const auto static_layer = metadata();
  auto dynamic_layer = static_layer;
  dynamic_layer.geometry.resolution_m = 0.1;

  EXPECT_TRUE(layers_are_compatible(static_layer, dynamic_layer));
}

TEST(GridCombiner, RejectsOriginResolutionDimensionsAndFrameMismatch)
{
  const auto static_layer = metadata();
  auto dynamic_layer = static_layer;

  dynamic_layer.geometry.x_min_m += 0.01;
  EXPECT_FALSE(layers_are_compatible(static_layer, dynamic_layer));
  dynamic_layer = static_layer;

  dynamic_layer.geometry.y_min_m += 0.01;
  EXPECT_FALSE(layers_are_compatible(static_layer, dynamic_layer));
  dynamic_layer = static_layer;

  dynamic_layer.geometry.resolution_m = 0.2;
  EXPECT_FALSE(layers_are_compatible(static_layer, dynamic_layer));
  dynamic_layer = static_layer;

  dynamic_layer.geometry.width += 1U;
  EXPECT_FALSE(layers_are_compatible(static_layer, dynamic_layer));
  dynamic_layer = static_layer;

  dynamic_layer.geometry.height += 1U;
  EXPECT_FALSE(layers_are_compatible(static_layer, dynamic_layer));
  dynamic_layer = static_layer;

  dynamic_layer.frame_id = "odom";
  EXPECT_FALSE(layers_are_compatible(static_layer, dynamic_layer));
}

TEST(GridCombiner, RejectsLargeStampDifference)
{
  const auto static_layer = metadata();
  auto dynamic_layer = static_layer;
  dynamic_layer.stamp_ns += 150'000'000LL;

  EXPECT_FALSE(layers_are_compatible(static_layer, dynamic_layer));
}

TEST(GridCombiner, RejectsZeroAndOverflowEdgeTimestamps)
{
  auto static_layer = metadata();
  auto dynamic_layer = static_layer;

  static_layer.stamp_ns = 0;
  EXPECT_FALSE(layers_are_compatible(static_layer, dynamic_layer));
  static_layer = metadata();

  dynamic_layer.stamp_ns = 0;
  EXPECT_FALSE(layers_are_compatible(static_layer, dynamic_layer));
  dynamic_layer = static_layer;

  static_layer.stamp_ns = 1;
  dynamic_layer.stamp_ns = std::numeric_limits<std::int64_t>::max();
  EXPECT_FALSE(layers_are_compatible(static_layer, dynamic_layer));
}

TEST(GridCombiner, RejectsMalformedMetadata)
{
  auto static_layer = metadata();
  auto dynamic_layer = static_layer;

  static_layer.frame_id.clear();
  EXPECT_FALSE(layers_are_compatible(static_layer, dynamic_layer));
  static_layer = metadata();

  static_layer.geometry.resolution_m = 0.0;
  EXPECT_FALSE(layers_are_compatible(static_layer, dynamic_layer));
  static_layer = metadata();

  dynamic_layer.geometry.x_min_m = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(layers_are_compatible(static_layer, dynamic_layer));
}

}  // namespace
