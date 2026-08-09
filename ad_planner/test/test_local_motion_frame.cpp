#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ad_planner/local_planning/common/local_motion_frame.hpp"

namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;

using ad_planner::FrameTransform2;
using ad_planner::OccupancyGrid;
using ad_planner::Pose2;
using ad_planner::project_primary_route;
using ad_planner::ReferenceCorridor;
using ad_planner::ReferenceLane;
using ad_planner::ReferencePoint;
using ad_planner::transform_occupancy_grid_origin;
using ad_planner::transform_pose;
using ad_planner::transform_reference_corridor;
using ad_planner::window_reference_corridor;

ReferencePoint point(const double x, const double y, const double yaw,
                     const double route_s) {
  return ReferencePoint{Pose2{x, y, yaw}, route_s, 0.01, 1.5, 1.7, 12.0};
}

ReferenceLane lane(const std::string &id, const std::vector<double> &stations,
                   const double y = 0.0) {
  ReferenceLane result;
  result.lane_sequence_id = id;
  result.source_link_ids = {"source-" + id};
  for (const double station : stations) {
    result.points.push_back(point(station, y, 0.0, station));
  }
  return result;
}

ReferenceCorridor corridor() {
  ReferenceCorridor result;
  result.frame_id = "map";
  result.primary_lane_index = 0U;
  result.lanes = {
      lane("primary", {0.0, 10.0, 20.0, 30.0}),
      lane("left", {0.0, 10.0, 20.0, 30.0}, 3.0),
      lane("disjoint", {100.0, 110.0, 120.0}, -3.0),
  };
  result.lanes[0].left_lane_indices = {1U};
  result.lanes[0].right_lane_indices = {2U};
  result.lanes[1].right_lane_indices = {0U};
  result.lanes[2].left_lane_indices = {0U};
  return result;
}

TEST(LocalMotionFrame, IdentityTransformPreservesPose) {
  const Pose2 pose{1.0, -2.0, 0.25};
  const auto transformed = transform_pose(FrameTransform2{}, pose);

  EXPECT_EQ(transformed, pose);
}

TEST(LocalMotionFrame, RotatesAndTranslatesPoseByNinetyDegrees) {
  const auto transformed = transform_pose(
      FrameTransform2{10.0, -3.0, kPi / 2.0}, Pose2{2.0, 1.0, 0.0});

  EXPECT_NEAR(transformed.x, 9.0, 1e-12);
  EXPECT_NEAR(transformed.y, -1.0, 1e-12);
  EXPECT_NEAR(transformed.yaw_rad, kPi / 2.0, 1e-12);
}

TEST(LocalMotionFrame, WrapsTransformedYawToMinusPiThroughPi) {
  const auto transformed =
      transform_pose(FrameTransform2{0.0, 0.0, 0.2}, Pose2{0.0, 0.0, kPi});

  EXPECT_GE(transformed.yaw_rad, -kPi);
  EXPECT_LE(transformed.yaw_rad, kPi);
  EXPECT_NEAR(transformed.yaw_rad, -kPi + 0.2, 1e-12);
}

TEST(LocalMotionFrame, TransformPreservesCorridorMetadataAndTopology) {
  const auto original = corridor();
  const auto transformed = transform_reference_corridor(
      FrameTransform2{2.0, 4.0, kPi / 2.0}, original, "odom");

  EXPECT_EQ(transformed.frame_id, "odom");
  EXPECT_EQ(transformed.primary_lane_index, original.primary_lane_index);
  ASSERT_EQ(transformed.lanes.size(), original.lanes.size());
  for (std::size_t lane_index = 0; lane_index < original.lanes.size();
       ++lane_index) {
    const auto &before = original.lanes[lane_index];
    const auto &after = transformed.lanes[lane_index];
    EXPECT_EQ(after.lane_sequence_id, before.lane_sequence_id);
    EXPECT_EQ(after.source_link_ids, before.source_link_ids);
    EXPECT_EQ(after.left_lane_indices, before.left_lane_indices);
    EXPECT_EQ(after.right_lane_indices, before.right_lane_indices);
    ASSERT_EQ(after.points.size(), before.points.size());
    for (std::size_t point_index = 0; point_index < before.points.size();
         ++point_index) {
      EXPECT_DOUBLE_EQ(after.points[point_index].route_s_m,
                       before.points[point_index].route_s_m);
      EXPECT_DOUBLE_EQ(after.points[point_index].curvature_inv_m,
                       before.points[point_index].curvature_inv_m);
      EXPECT_DOUBLE_EQ(after.points[point_index].left_width_m,
                       before.points[point_index].left_width_m);
      EXPECT_DOUBLE_EQ(after.points[point_index].right_width_m,
                       before.points[point_index].right_width_m);
      EXPECT_DOUBLE_EQ(after.points[point_index].speed_limit_mps,
                       before.points[point_index].speed_limit_mps);
    }
  }
  EXPECT_NEAR(transformed.lanes[0].points[0].pose.x, 2.0, 1e-12);
  EXPECT_NEAR(transformed.lanes[0].points[0].pose.y, 4.0, 1e-12);
}

TEST(LocalMotionFrame, TransformsOnlyRotatedOccupancyGridOrigin) {
  OccupancyGrid grid;
  grid.origin = Pose2{1.0, 2.0, 0.1};
  grid.resolution = 0.25;
  grid.width = 4U;
  grid.height = 2U;
  grid.cells = {0, 1, -1, 100};
  grid.valid = true;
  grid.fresh = true;

  const auto transformed = transform_occupancy_grid_origin(
      FrameTransform2{3.0, 4.0, kPi / 2.0}, grid);

  EXPECT_NEAR(transformed.origin.x, 1.0, 1e-12);
  EXPECT_NEAR(transformed.origin.y, 5.0, 1e-12);
  EXPECT_NEAR(transformed.origin.yaw_rad, kPi / 2.0 + 0.1, 1e-12);
  EXPECT_DOUBLE_EQ(transformed.resolution, grid.resolution);
  EXPECT_EQ(transformed.width, grid.width);
  EXPECT_EQ(transformed.height, grid.height);
  EXPECT_EQ(transformed.cells, grid.cells);
  EXPECT_EQ(transformed.valid, grid.valid);
  EXPECT_EQ(transformed.fresh, grid.fresh);
}

TEST(LocalMotionFrame, MiddleWindowIncludesBoundaryBracketingPoints) {
  const auto windowed =
      window_reference_corridor(corridor(), Pose2{15.0, 1.0, 0.0}, 3.0, 3.0);

  const auto &points = windowed.lanes[0].points;
  ASSERT_EQ(points.size(), 2U);
  EXPECT_DOUBLE_EQ(points[0].route_s_m, 10.0);
  EXPECT_DOUBLE_EQ(points[1].route_s_m, 20.0);
}

TEST(LocalMotionFrame, WindowClampsAtStartAndEnd) {
  const auto start =
      window_reference_corridor(corridor(), Pose2{1.0, 0.0, 0.0}, 10.0, 2.0);
  ASSERT_EQ(start.lanes[0].points.size(), 2U);
  EXPECT_DOUBLE_EQ(start.lanes[0].points[0].route_s_m, 0.0);
  EXPECT_DOUBLE_EQ(start.lanes[0].points[1].route_s_m, 10.0);

  const auto end =
      window_reference_corridor(corridor(), Pose2{29.0, 0.0, 0.0}, 2.0, 10.0);
  ASSERT_EQ(end.lanes[0].points.size(), 2U);
  EXPECT_DOUBLE_EQ(end.lanes[0].points[0].route_s_m, 20.0);
  EXPECT_DOUBLE_EQ(end.lanes[0].points[1].route_s_m, 30.0);
}

TEST(LocalMotionFrame, KeepsDisjointAdjacentSequenceSeparateWithNearestPair) {
  const auto windowed =
      window_reference_corridor(corridor(), Pose2{15.0, 0.0, 0.0}, 2.0, 2.0);

  ASSERT_EQ(windowed.lanes.size(), 3U);
  EXPECT_EQ(windowed.lanes[2].lane_sequence_id, "disjoint");
  EXPECT_EQ(windowed.lanes[2].source_link_ids,
            (std::vector<std::string>{"source-disjoint"}));
  EXPECT_EQ(windowed.lanes[2].left_lane_indices,
            (std::vector<std::size_t>{0U}));
  ASSERT_EQ(windowed.lanes[2].points.size(), 2U);
  EXPECT_DOUBLE_EQ(windowed.lanes[2].points[0].route_s_m, 100.0);
  EXPECT_DOUBLE_EQ(windowed.lanes[2].points[1].route_s_m, 110.0);
}

TEST(LocalMotionFrame, PrimaryProjectionRejectsOverlappingReverseBranch) {
  auto overlapping = corridor();
  overlapping.lanes[0].points = {
      point(0.0, 0.0, 0.0, 0.0),    point(10.0, 0.0, 0.0, 10.0),
      point(20.0, 10.0, 0.0, 30.0), point(10.0, 0.0, kPi, 50.0),
      point(0.0, 0.0, kPi, 60.0),
  };

  const auto projection =
      project_primary_route(overlapping, Pose2{5.0, 0.0, 0.0});

  EXPECT_NEAR(projection.route_s_m, 5.0, 1.0e-12);
  EXPECT_NEAR(projection.lateral_distance_m, 0.0, 1.0e-12);
  EXPECT_NEAR(projection.heading_error_rad, 0.0, 1.0e-12);
}

TEST(LocalMotionFrame, PrimaryProjectionRejectsWrongWayPose) {
  EXPECT_THROW(project_primary_route(corridor(), Pose2{5.0, 0.0, kPi}),
               std::invalid_argument);
}

TEST(LocalMotionFrame, ProjectsFiniteNonzeroPrimarySegmentBelowOnePicometer) {
  auto tiny = corridor();
  tiny.lanes[0].points = {
      point(0.0, 0.0, 0.0, 0.0),
      point(5e-13, 0.0, 0.0, 1.0),
  };

  const auto windowed =
      window_reference_corridor(tiny, Pose2{2.5e-13, 0.0, 0.0}, 0.0, 0.0);

  ASSERT_EQ(windowed.lanes[0].points.size(), 2U);
  EXPECT_DOUBLE_EQ(windowed.lanes[0].points[0].route_s_m, 0.0);
  EXPECT_DOUBLE_EQ(windowed.lanes[0].points[1].route_s_m, 1.0);
}

TEST(LocalMotionFrame, RejectsInvalidAndOverflowingInputWithoutPartialOutput) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(transform_pose(FrameTransform2{nan, 0.0, 0.0}, Pose2{}),
               std::invalid_argument);
  EXPECT_THROW(transform_pose(FrameTransform2{}, Pose2{0.0, 0.0, nan}),
               std::invalid_argument);
  EXPECT_THROW(transform_reference_corridor(FrameTransform2{}, corridor(), ""),
               std::invalid_argument);

  auto invalid_primary = corridor();
  invalid_primary.primary_lane_index = invalid_primary.lanes.size();
  EXPECT_THROW(window_reference_corridor(invalid_primary, Pose2{}, 1.0, 1.0),
               std::invalid_argument);

  auto zero_segment = corridor();
  zero_segment.lanes[0].points[1].pose = zero_segment.lanes[0].points[0].pose;
  EXPECT_THROW(window_reference_corridor(zero_segment, Pose2{}, 1.0, 1.0),
               std::invalid_argument);

  auto nonmonotonic = corridor();
  nonmonotonic.lanes[1].points[1].route_s_m = 0.0;
  EXPECT_THROW(
      transform_reference_corridor(FrameTransform2{}, nonmonotonic, "odom"),
      std::invalid_argument);

  auto overflowing = corridor();
  overflowing.lanes[0].points[0].pose.x = std::numeric_limits<double>::max();
  overflowing.lanes[0].points[1].pose.x = -std::numeric_limits<double>::max();
  EXPECT_THROW(window_reference_corridor(overflowing, Pose2{}, 1.0, 1.0),
               std::invalid_argument);

  EXPECT_THROW(window_reference_corridor(corridor(), Pose2{}, -1.0, 1.0),
               std::invalid_argument);
  EXPECT_THROW(window_reference_corridor(corridor(), Pose2{}, 1.0, nan),
               std::invalid_argument);
}
} // namespace
