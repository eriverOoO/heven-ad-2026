#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "ad_planner/local_planning/common/occupancy.hpp"
#include "ad_planner/local_planning/common/occupancy_grid_reprojector.hpp"

namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;

using ad_planner::cell_value;
using ad_planner::FrameTransform2;
using ad_planner::GridReprojectionConfig;
using ad_planner::OccupancyGrid;
using ad_planner::planar_yaw_from_quaternion;
using ad_planner::Point2;
using ad_planner::Point3;
using ad_planner::Pose2;
using ad_planner::QuaternionComponents;
using ad_planner::reproject_occupancy_grid;

OccupancyGrid make_grid(const std::size_t width, const std::size_t height,
                        const double resolution = 1.0,
                        const Pose2 &origin = Pose2{}) {
  OccupancyGrid grid;
  grid.origin = origin;
  grid.resolution = resolution;
  grid.width = width;
  grid.height = height;
  grid.cells.resize(width * height);
  for (std::size_t index = 0U; index < grid.cells.size(); ++index) {
    grid.cells[index] = static_cast<std::int8_t>(index % 101U);
  }
  grid.valid = true;
  grid.fresh = true;
  return grid;
}

GridReprojectionConfig config(const double width_m, const double height_m,
                              const double resolution_m = 1.0) {
  GridReprojectionConfig result;
  result.width_m = width_m;
  result.height_m = height_m;
  result.resolution_m = resolution_m;
  result.outside_value = -1;
  return result;
}

Point2 transformed_source_center(const OccupancyGrid &source,
                                 const FrameTransform2 &odom_from_source) {
  const double source_cosine = std::cos(source.origin.yaw_rad);
  const double source_sine = std::sin(source.origin.yaw_rad);
  const double local_x =
      static_cast<double>(source.width) * source.resolution * 0.5;
  const double local_y =
      static_cast<double>(source.height) * source.resolution * 0.5;
  const double center_source_x =
      source.origin.x + source_cosine * local_x - source_sine * local_y;
  const double center_source_y =
      source.origin.y + source_sine * local_x + source_cosine * local_y;
  const double transform_cosine = std::cos(odom_from_source.yaw_rad);
  const double transform_sine = std::sin(odom_from_source.yaw_rad);
  return Point2{odom_from_source.x_m + transform_cosine * center_source_x -
                    transform_sine * center_source_y,
                odom_from_source.y_m + transform_sine * center_source_x +
                    transform_cosine * center_source_y};
}

TEST(OccupancyGridReprojector, IdentityTransformPreservesEveryCellCost) {
  auto source = make_grid(3U, 2U);
  source.cells = {-1, 0, 100, 17, 42, 81};

  const auto output = reproject_occupancy_grid(
      source, FrameTransform2{}, Point2{1.5, 1.0}, config(3.0, 2.0));

  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->cells, source.cells);
  EXPECT_TRUE(output->valid);
  EXPECT_TRUE(output->fresh);
}

TEST(OccupancyGridReprojector, NinetyDegreeTransformRotatesCellLocation) {
  auto source = make_grid(2U, 2U);
  source.cells = {0, 100, 0, 0};

  const auto output =
      reproject_occupancy_grid(source, FrameTransform2{0.0, 0.0, kPi / 2.0},
                               Point2{0.0, 1.0}, config(2.0, 2.0));

  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->cells, (std::vector<std::int8_t>{0, -1, 100, -1}));
}

TEST(OccupancyGridReprojector,
     OutputIsAxisAlignedAndGeometricallyCenteredOnEgo) {
  const auto source = make_grid(1U, 1U);

  const auto output = reproject_occupancy_grid(
      source, FrameTransform2{}, Point2{12.5, -4.0}, config(6.0, 4.0, 0.5));

  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->width, 12U);
  EXPECT_EQ(output->height, 8U);
  EXPECT_DOUBLE_EQ(output->resolution, 0.5);
  EXPECT_DOUBLE_EQ(output->origin.x, 9.5);
  EXPECT_DOUBLE_EQ(output->origin.y, -6.0);
  EXPECT_DOUBLE_EQ(output->origin.yaw_rad, 0.0);
}

TEST(OccupancyGridReprojector,
     CanonicalizesFloat32ResolutionAcrossPublishedGeometry) {
  const auto source = make_grid(1U, 1U);
  const auto output = reproject_occupancy_grid(
      source, FrameTransform2{}, Point2{3.0, -2.0}, config(54.0, 54.0, 0.1));

  ASSERT_TRUE(output.has_value());
  ASSERT_EQ(output->width, 540U);
  ASSERT_EQ(output->height, 540U);
  const double effective_resolution =
      static_cast<double>(static_cast<float>(0.1));
  EXPECT_DOUBLE_EQ(output->resolution, effective_resolution);
  EXPECT_DOUBLE_EQ(output->origin.x, 3.0 - 0.5 * 540.0 * effective_resolution);
  EXPECT_DOUBLE_EQ(output->origin.y, -2.0 - 0.5 * 540.0 * effective_resolution);
  EXPECT_DOUBLE_EQ(output->origin.x +
                       static_cast<double>(output->width) * output->resolution,
                   3.0 + 0.5 * 540.0 * effective_resolution);
  EXPECT_DOUBLE_EQ(output->origin.y +
                       static_cast<double>(output->height) * output->resolution,
                   -2.0 + 0.5 * 540.0 * effective_resolution);
}

TEST(OccupancyGridReprojector,
     CombinesSourceOriginAndFrameRotationByReverseSampling) {
  auto source = make_grid(2U, 2U, 1.0, Pose2{2.0, 1.0, kPi / 2.0});
  source.cells = {10, 20, 30, 40};

  const auto output =
      reproject_occupancy_grid(source, FrameTransform2{10.0, -3.0, kPi / 2.0},
                               Point2{8.0, -2.0}, config(2.0, 2.0));

  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->cells, (std::vector<std::int8_t>{40, 30, 20, 10}));
}

TEST(OccupancyGridReprojector, UsesHalfOpenSourceBoundsAndUnknownOutside) {
  auto source = make_grid(2U, 2U);
  source.cells = {1, 2, 3, 4};

  const auto output = reproject_occupancy_grid(
      source, FrameTransform2{}, Point2{1.0, 1.0}, config(3.0, 3.0));

  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->cells,
            (std::vector<std::int8_t>{1, 2, -1, 3, 4, -1, -1, -1, -1}));
}

TEST(OccupancyGridReprojector, FullyOutsideCoverageRemainsUnknown) {
  const auto source = make_grid(2U, 2U);

  const auto output = reproject_occupancy_grid(
      source, FrameTransform2{}, Point2{100.0, -100.0}, config(3.0, 2.0));

  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->cells, (std::vector<std::int8_t>(6U, -1)));
}

TEST(OccupancyGridReprojector, EveryOutsideEdgeAndCornerRemainsUnknown) {
  auto source = make_grid(2U, 2U);
  source.cells = {1, 2, 3, 4};

  const auto output = reproject_occupancy_grid(
      source, FrameTransform2{}, Point2{1.0, 1.0}, config(4.0, 4.0));

  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->cells,
            (std::vector<std::int8_t>{-1, -1, -1, -1, -1, 1, 2, -1, -1, 3, 4,
                                      -1, -1, -1, -1, -1}));
}

TEST(OccupancyGridReprojector,
     PreservesAllowedCellRangeAndRejectsInvalidCosts) {
  auto source = make_grid(3U, 1U);
  source.cells = {-1, 0, 100};
  const auto output = reproject_occupancy_grid(
      source, FrameTransform2{}, Point2{1.5, 0.5}, config(3.0, 1.0));
  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->cells, source.cells);

  source.cells[0] = -2;
  EXPECT_FALSE(reproject_occupancy_grid(source, FrameTransform2{},
                                        Point2{1.5, 0.5}, config(3.0, 1.0)));
  source.cells[0] = 101;
  EXPECT_FALSE(reproject_occupancy_grid(source, FrameTransform2{},
                                        Point2{1.5, 0.5}, config(3.0, 1.0)));
}

TEST(OccupancyGridReprojector,
     RejectsMalformedGridTransformPointAndOutsidePolicy) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  auto source = make_grid(2U, 2U);

  auto invalid = source;
  invalid.valid = false;
  EXPECT_FALSE(reproject_occupancy_grid(invalid, FrameTransform2{},
                                        Point2{1.0, 1.0}, config(2.0, 2.0)));
  invalid = source;
  invalid.fresh = false;
  EXPECT_FALSE(reproject_occupancy_grid(invalid, FrameTransform2{},
                                        Point2{1.0, 1.0}, config(2.0, 2.0)));
  invalid = source;
  invalid.cells.pop_back();
  EXPECT_FALSE(reproject_occupancy_grid(invalid, FrameTransform2{},
                                        Point2{1.0, 1.0}, config(2.0, 2.0)));
  invalid = source;
  invalid.origin.yaw_rad = nan;
  EXPECT_FALSE(reproject_occupancy_grid(invalid, FrameTransform2{},
                                        Point2{1.0, 1.0}, config(2.0, 2.0)));

  EXPECT_FALSE(reproject_occupancy_grid(source, FrameTransform2{nan, 0.0, 0.0},
                                        Point2{1.0, 1.0}, config(2.0, 2.0)));
  EXPECT_FALSE(reproject_occupancy_grid(source, FrameTransform2{},
                                        Point2{nan, 1.0}, config(2.0, 2.0)));

  auto invalid_config = config(2.0, 2.0);
  invalid_config.outside_value = 0;
  EXPECT_FALSE(reproject_occupancy_grid(source, FrameTransform2{},
                                        Point2{1.0, 1.0}, invalid_config));
  invalid_config = config(2.0, 2.0);
  invalid_config.width_m = 0.0;
  EXPECT_FALSE(reproject_occupancy_grid(source, FrameTransform2{},
                                        Point2{1.0, 1.0}, invalid_config));
  invalid_config = config(2.0, 2.0);
  invalid_config.height_m = nan;
  EXPECT_FALSE(reproject_occupancy_grid(source, FrameTransform2{},
                                        Point2{1.0, 1.0}, invalid_config));
}

TEST(OccupancyGridReprojector, AppliesScaleAwareIntegralCellRatioTolerance) {
  const auto source = make_grid(4U, 1U);
  const double inside_delta = 2.0e-14;
  auto inside = config(3.0 + inside_delta, 1.0);
  const auto output = reproject_occupancy_grid(source, FrameTransform2{},
                                               Point2{1.5, 0.5}, inside);
  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->width, 3U);

  auto outside = config(3.0 + 1.0e-12, 1.0);
  EXPECT_FALSE(reproject_occupancy_grid(source, FrameTransform2{},
                                        Point2{1.5, 0.5}, outside));
}

TEST(OccupancyGridReprojector,
     RejectsCellCountAndAllocationOverflowBeforeAllocating) {
  const auto source = make_grid(1U, 1U);
  const double uint32_max =
      static_cast<double>(std::numeric_limits<std::uint32_t>::max());

  EXPECT_FALSE(reproject_occupancy_grid(source, FrameTransform2{}, Point2{},
                                        config(uint32_max + 1.0, 1.0)));
  EXPECT_FALSE(reproject_occupancy_grid(source, FrameTransform2{}, Point2{},
                                        config(uint32_max, uint32_max)));
}

TEST(OccupancyGridReprojector,
     RejectsResolutionThatCannotNarrowToPositiveFiniteFloat32) {
  const auto source = make_grid(1U, 1U);
  const double too_small =
      static_cast<double>(std::numeric_limits<float>::denorm_min()) * 0.25;
  const double too_large =
      static_cast<double>(std::numeric_limits<float>::max()) * 2.0;

  EXPECT_FALSE(
      reproject_occupancy_grid(source, FrameTransform2{}, Point2{},
                               config(too_small, too_small, too_small)));
  EXPECT_FALSE(
      reproject_occupancy_grid(source, FrameTransform2{}, Point2{},
                               config(too_large, too_large, too_large)));
}

TEST(OccupancyGridReprojector,
     RejectsNonfiniteCoordinateArithmeticBeforeAllocation) {
  const auto source = make_grid(2U, 2U);
  const double maximum = std::numeric_limits<double>::max();

  EXPECT_FALSE(reproject_occupancy_grid(
      source, FrameTransform2{}, Point2{maximum, maximum}, config(2.0, 2.0)));
  EXPECT_FALSE(
      reproject_occupancy_grid(source, FrameTransform2{-maximum, 0.0, 0.0},
                               Point2{maximum, 0.0}, config(1.0, 1.0)));
}

TEST(OccupancyGridReprojector,
     OptimizedSamplerMatchesWorldToCellAtRandomizedPoints) {
  std::mt19937 generator(0x4d505049U);
  std::uniform_real_distribution<double> position(-4.0, 4.0);
  std::uniform_real_distribution<double> yaw(-kPi, kPi);
  auto source = make_grid(7U, 5U, 0.4);
  for (std::size_t index = 0U; index < source.cells.size(); ++index) {
    source.cells[index] =
        index % 7U == 0U ? -1 : static_cast<std::int8_t>((index * 17U) % 101U);
  }

  for (std::size_t iteration = 0U; iteration < 25U; ++iteration) {
    source.origin =
        Pose2{position(generator), position(generator), yaw(generator)};
    const FrameTransform2 transform{position(generator), position(generator),
                                    yaw(generator)};
    const Point2 ego = transformed_source_center(source, transform);
    const auto output = reproject_occupancy_grid(source, transform, ego,
                                                 config(3.0, 2.0, 0.25));
    ASSERT_TRUE(output.has_value());

    const double transform_cosine = std::cos(transform.yaw_rad);
    const double transform_sine = std::sin(transform.yaw_rad);
    for (std::size_t y = 0U; y < output->height; ++y) {
      for (std::size_t x = 0U; x < output->width; ++x) {
        const double output_x =
            output->origin.x +
            (static_cast<double>(x) + 0.5) * output->resolution;
        const double output_y =
            output->origin.y +
            (static_cast<double>(y) + 0.5) * output->resolution;
        const double dx = output_x - transform.x_m;
        const double dy = output_y - transform.y_m;
        const Point3 point_in_source{
            transform_cosine * dx + transform_sine * dy,
            -transform_sine * dx + transform_cosine * dy, 0.0};
        EXPECT_EQ(output->cells[y * output->width + x],
                  cell_value(source, point_in_source));
      }
    }
  }
}

TEST(OccupancyGridReprojector, AcceptsOnlyFiniteNearUnitPlanarQuaternions) {
  const double yaw = 0.7;
  const QuaternionComponents valid{0.0, 0.0, std::sin(yaw * 0.5),
                                   std::cos(yaw * 0.5)};
  const auto extracted = planar_yaw_from_quaternion(valid);
  ASSERT_TRUE(extracted.has_value());
  EXPECT_NEAR(*extracted, yaw, 1.0e-12);

  const double near_unit_scale = 1.0 + 5.0e-8;
  const auto near_unit = planar_yaw_from_quaternion(QuaternionComponents{
      0.0, 0.0, valid.z * near_unit_scale, valid.w * near_unit_scale});
  ASSERT_TRUE(near_unit.has_value());
  EXPECT_NEAR(*near_unit, yaw, 1.0e-12);

  const double road_roll = 0.01;
  const double road_pitch = -0.01;
  const double cr = std::cos(road_roll * 0.5);
  const double sr = std::sin(road_roll * 0.5);
  const double cp = std::cos(road_pitch * 0.5);
  const double sp = std::sin(road_pitch * 0.5);
  const double cy = std::cos(yaw * 0.5);
  const double sy = std::sin(yaw * 0.5);
  const auto road_tilt = planar_yaw_from_quaternion(QuaternionComponents{
      sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy});
  ASSERT_TRUE(road_tilt.has_value());
  EXPECT_NEAR(*road_tilt, yaw, 1.0e-12);

  EXPECT_FALSE(planar_yaw_from_quaternion(QuaternionComponents{}));
  EXPECT_FALSE(
      planar_yaw_from_quaternion(QuaternionComponents{0.0, 0.0, 0.0, 1.01}));
  EXPECT_FALSE(planar_yaw_from_quaternion(QuaternionComponents{
      std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 1.0}));
  EXPECT_FALSE(planar_yaw_from_quaternion(QuaternionComponents{
      0.0, 0.0, std::numeric_limits<double>::infinity(), 1.0}));
  EXPECT_FALSE(planar_yaw_from_quaternion(
      QuaternionComponents{std::sin(0.1), 0.0, 0.0, std::cos(0.1)}));
  EXPECT_FALSE(planar_yaw_from_quaternion(
      QuaternionComponents{0.0, std::sin(0.1), 0.0, std::cos(0.1)}));
}
} // namespace
