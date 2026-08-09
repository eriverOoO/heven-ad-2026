#include <cmath>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "ad_planner/local_planning/common/occupancy.hpp"

namespace {
using namespace ad_planner;

OccupancyGrid grid(std::size_t width = 4, std::size_t height = 3) {
  OccupancyGrid value;
  value.origin = Pose2{10.0, 20.0, std::acos(-1.0) / 2.0};
  value.resolution = 0.5;
  value.width = width;
  value.height = height;
  value.cells.assign(width * height, 0);
  value.valid = true;
  value.fresh = true;
  return value;
}

OccupancyGrid rear_axle_grid() {
  OccupancyGrid value;
  value.origin = Pose2{-3.0, -3.0, 0.0};
  value.resolution = 0.1;
  value.width = 90;
  value.height = 90;
  value.cells.assign(value.width * value.height, 0);
  value.valid = true;
  value.fresh = true;
  return value;
}

void occupy(OccupancyGrid &map, double x, double y) {
  const auto cell = world_to_cell(map, Point3{x, y, 0.0});
  ASSERT_TRUE(cell.has_value());
  map.cells[cell->y * map.width + cell->x] = 100;
}

TEST(Occupancy, UsesOriginYawResolutionAndDimensionsForWorldCoordinates) {
  auto map = grid();
  map.cells[1] = 77;
  const auto cell = world_to_cell(map, Point3{9.75, 20.75, 0.0});
  ASSERT_TRUE(cell.has_value());
  EXPECT_EQ(cell->x, 1U);
  EXPECT_EQ(cell->y, 0U);
  EXPECT_EQ(cell_value(map, Point3{9.75, 20.75, 0.0}), 77);
  EXPECT_FALSE(world_to_cell(map, Point3{11.0, 19.5, 0.0}).has_value());
}

TEST(Occupancy, RejectsMalformedMetadataOverflowAndDataSize) {
  auto map = grid();
  map.cells.pop_back();
  EXPECT_FALSE(validate_occupancy_grid(map).valid);
  map.width = std::numeric_limits<std::size_t>::max();
  map.height = 2;
  EXPECT_FALSE(validate_occupancy_grid(map).valid);
  map = grid();
  map.resolution = 0.0;
  EXPECT_FALSE(validate_occupancy_grid(map).valid);
}

TEST(Occupancy, UnknownOutOfBoundsAndOccupiedFootprintCellsAreUnsafe) {
  OccupancyGrid map;
  map.origin = Pose2{-2.0, -2.0, 0.0};
  map.resolution = 1.0;
  map.width = 5;
  map.height = 5;
  map.cells.assign(25, 0);
  map.valid = true;
  map.fresh = true;
  const FootprintConfig footprint{0.6, 0.6, 0.0, 50, 64};

  map.cells[2 * 5 + 1] = 100;
  EXPECT_FALSE(footprint_is_safe(map, Pose2{0.0, 0.0, 0.0}, footprint));
  map.cells[2 * 5 + 1] = -1;
  EXPECT_FALSE(footprint_is_safe(map, Pose2{0.0, 0.0, 0.0}, footprint));
  map.cells[2 * 5 + 1] = 0;
  EXPECT_TRUE(footprint_is_safe(map, Pose2{0.0, 0.0, 0.0}, footprint));
  EXPECT_FALSE(footprint_is_safe(map, Pose2{-1.8, -1.8, 0.0}, footprint));
}

TEST(Occupancy, InvalidGridIsNeverTreatedAsClear) {
  auto map = grid();
  map.fresh = false;
  EXPECT_FALSE(
      footprint_is_safe(map, Pose2{10.0, 20.0, 0.0}, {0.1, 0.1, 0.0, 50, 16}));
}

TEST(Occupancy, RejectsFootprintExtentOverflowWithoutLooping) {
  auto map = grid();
  const double maximum = std::numeric_limits<double>::max();
  EXPECT_FALSE(
      footprint_is_safe(map, Pose2{10.0, 20.0, 0.0},
                        FootprintConfig{maximum, 0.1, maximum, 50, 16}));
}

TEST(Occupancy, ChecksEveryCellAreaIntersectingSmallFootprint) {
  OccupancyGrid map;
  map.origin = Pose2{0.0, 0.0, 0.0};
  map.resolution = 1.0;
  map.width = 3;
  map.height = 3;
  map.cells.assign(9, 0);
  map.valid = true;
  map.fresh = true;
  FootprintConfig footprint{0.1, 0.1, 0.0, 50, 16};

  map.cells[1 * map.width + 1] = 100;
  EXPECT_FALSE(footprint_is_safe(map, Pose2{0.95, 0.95, 0.0}, footprint));
  map.cells[1 * map.width + 1] = -1;
  EXPECT_FALSE(footprint_is_safe(map, Pose2{0.95, 0.95, 0.0}, footprint));
}

TEST(Occupancy, ChecksRotatedFootprintAgainstEveryIntersectingCell) {
  OccupancyGrid map;
  map.origin = Pose2{0.0, 0.0, 0.0};
  map.resolution = 1.0;
  map.width = 4;
  map.height = 3;
  map.cells.assign(12, 0);
  map.cells[1 * map.width + 2] = 100;
  map.valid = true;
  map.fresh = true;
  FootprintConfig footprint{0.8, 0.05, 0.0, 50, 16};

  EXPECT_FALSE(footprint_is_safe(map, Pose2{1.4, 1.0, std::acos(-1.0) / 4.0},
                                 footprint));
}

TEST(Occupancy, UsesRearAxleOffsetForIoniq5FrontAndRearExtents) {
  FootprintConfig footprint{2.3175, 0.945, 0.0, 100, 4096};
  footprint.center_offset_x_m = 1.5275;

  auto map = rear_axle_grid();
  occupy(map, 3.75, 0.0);
  EXPECT_FALSE(footprint_is_safe(map, Pose2{0.0, 0.0, 0.0}, footprint));

  map = rear_axle_grid();
  occupy(map, 4.0, 0.0);
  EXPECT_TRUE(footprint_is_safe(map, Pose2{0.0, 0.0, 0.0}, footprint));

  map = rear_axle_grid();
  occupy(map, -0.7, 0.0);
  EXPECT_FALSE(footprint_is_safe(map, Pose2{0.0, 0.0, 0.0}, footprint));

  map = rear_axle_grid();
  occupy(map, -1.0, 0.0);
  EXPECT_TRUE(footprint_is_safe(map, Pose2{0.0, 0.0, 0.0}, footprint));
}

TEST(Occupancy, RotatesRearAxleOffsetWithPoseYaw) {
  FootprintConfig footprint{2.3175, 0.945, 0.0, 100, 4096};
  footprint.center_offset_x_m = 1.5275;

  auto map = rear_axle_grid();
  occupy(map, 0.0, 3.75);
  EXPECT_FALSE(footprint_is_safe(map, Pose2{0.0, 0.0, std::acos(-1.0) / 2.0},
                                 footprint));

  map = rear_axle_grid();
  occupy(map, 0.0, -1.0);
  EXPECT_TRUE(footprint_is_safe(map, Pose2{0.0, 0.0, std::acos(-1.0) / 2.0},
                                footprint));
}

TEST(Occupancy, RejectsNonfiniteFootprintCenterOffset) {
  auto map = rear_axle_grid();
  FootprintConfig footprint{2.3175, 0.945, 0.2, 100, 4096};
  footprint.center_offset_x_m = std::numeric_limits<double>::quiet_NaN();

  EXPECT_FALSE(footprint_is_safe(map, Pose2{0.0, 0.0, 0.0}, footprint));
}

TEST(Occupancy, RejectsOccupiedThresholdOutsideOccupancyGridRange) {
  auto map = rear_axle_grid();
  FootprintConfig footprint{2.3175, 0.945, 0.2, 101, 4096};

  EXPECT_FALSE(footprint_is_safe(map, Pose2{0.0, 0.0, 0.0}, footprint));
  EXPECT_DOUBLE_EQ(
      nearest_unsafe_clearance(map, Point3{0.0, 0.0, 0.0}, 101, 10.0), 0.0);
}

TEST(Occupancy, RejectsNonprogressingOrExcessiveCellEnumeration) {
  OccupancyGrid map;
  map.origin = Pose2{-1.0e300, 0.0, 0.0};
  map.resolution = 1.0;
  map.width = 4;
  map.height = 4;
  map.cells.assign(16, 0);
  map.valid = true;
  map.fresh = true;
  FootprintConfig nonprogressing{1.0e300, 0.1, 0.0, 50, 16};
  EXPECT_FALSE(footprint_is_safe(map, Pose2{0.0, 1.0, 0.0}, nonprogressing));

  FootprintConfig bounded{1.1, 1.1, 0.0, 50, 1};
  map.origin = Pose2{-2.0, -2.0, 0.0};
  EXPECT_FALSE(footprint_is_safe(map, Pose2{0.0, 0.0, 0.0}, bounded));
}
} // namespace
