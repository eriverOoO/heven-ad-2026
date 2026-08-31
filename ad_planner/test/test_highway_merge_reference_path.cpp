// Pure tests for the online highway-merge lateral reference path builder.
//
// The builder produces the real route:0:left:1 acceleration-lane centerline
// followed by the real route:0 mainline centerline past the merge point -- no
// synthetic lane-change curve. These tests lock the join / continuity / station
// contract and the reject paths, and (via test_highway_merge_reference_path.py)
// the equivalence to the committed golden fixture.

#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "ad_planner/planning/highway_merge_reference_path.hpp"

namespace {

using ad_planner::build_highway_merge_reference_path;
using ad_planner::HighwayMergeReferencePathConfig;
using ad_planner::ReferenceLane;
using ad_planner::ReferencePoint;

// A straight lane heading -y (like the real K-City acceleration lane / mainline
// through the merge region), sampled every `spacing` metres.
ReferenceLane straight_lane(const std::string &id, double x0, double y0,
                            double route_s0, int count, double spacing = 0.5,
                            double lateral = 0.0) {
  ReferenceLane lane;
  lane.lane_sequence_id = id;
  for (int i = 0; i < count; ++i) {
    ReferencePoint p;
    p.pose.x = x0 + lateral;
    p.pose.y = y0 - spacing * i;
    p.pose.yaw_rad = -M_PI / 2.0;
    p.route_s_m = route_s0 + spacing * i;
    lane.points.push_back(p);
  }
  return lane;
}

TEST(HighwayMergeReferencePath, BuildsSourceThenTargetWithoutDuplicatingTheJoin) {
  // source spans route_s [1118.74, 1286.24] (336 pts @ 0.5 m); the coincident
  // merge-complete point is the source's last.
  const double merge_complete = 1118.7418 + 0.5 * 335;
  ReferenceLane source =
      straight_lane("route:0:left:1", 66.0, 256.5, 1118.7418, 336);
  // target continues straight down from the source's last point.
  ReferenceLane target = straight_lane(
      "route:0", 66.0, 256.5 - 0.5 * 335, merge_complete, 800);

  HighwayMergeReferencePathConfig cfg;
  const auto built =
      build_highway_merge_reference_path(source, target, merge_complete, cfg);

  EXPECT_EQ(built.source_point_count, 336U);
  EXPECT_GT(built.target_point_count, 300U);  // 200 m / 0.5 m minus the join
  EXPECT_EQ(built.route.points.size(),
            built.source_point_count + built.target_point_count);
  EXPECT_FALSE(built.route.closed);

  // source section preserved verbatim
  for (std::size_t i = 0; i < source.points.size(); ++i) {
    EXPECT_DOUBLE_EQ(built.route.points[i].x, source.points[i].pose.x);
    EXPECT_DOUBLE_EQ(built.route.points[i].y, source.points[i].pose.y);
  }
  // the join point is the source last, contributed once: the first target
  // point is strictly past merge-complete
  const auto &join = built.route.points[built.source_point_count - 1U];
  const auto &after = built.route.points[built.source_point_count];
  EXPECT_DOUBLE_EQ(join.y, source.points.back().pose.y);
  EXPECT_LT(after.y, join.y);  // continues down -y, no repeat
  EXPECT_NEAR(built.splice_join_gap_m, 0.5, 1e-6);
  EXPECT_LT(built.splice_join_heading_delta_rad, 1e-6);
  EXPECT_DOUBLE_EQ(built.splice_route_s_m, merge_complete);
}

TEST(HighwayMergeReferencePath, StationsAreMonotonicAndSpacingBounded) {
  const double merge_complete = 1000.0 + 0.5 * 199;
  ReferenceLane source = straight_lane("s", 0.0, 0.0, 1000.0, 200);
  ReferenceLane target = straight_lane("t", 0.0, -0.5 * 199, merge_complete, 600);
  const auto built = build_highway_merge_reference_path(
      source, target, merge_complete, {});
  for (std::size_t i = 1; i < built.route.points.size(); ++i) {
    const double d = std::hypot(built.route.points[i].x - built.route.points[i - 1].x,
                                built.route.points[i].y - built.route.points[i - 1].y);
    EXPECT_GT(d, 1e-6);
    EXPECT_LT(d, 1.0);
  }
}

TEST(HighwayMergeReferencePath, DeterministicAcrossRepeatedBuilds) {
  const double merge_complete = 500.0 + 0.5 * 149;
  ReferenceLane source = straight_lane("s", 10.0, 5.0, 500.0, 150);
  ReferenceLane target = straight_lane("t", 10.0, 5.0 - 0.5 * 149, merge_complete, 500);
  const auto a = build_highway_merge_reference_path(source, target, merge_complete, {});
  const auto b = build_highway_merge_reference_path(source, target, merge_complete, {});
  ASSERT_EQ(a.route.points.size(), b.route.points.size());
  for (std::size_t i = 0; i < a.route.points.size(); ++i) {
    EXPECT_DOUBLE_EQ(a.route.points[i].x, b.route.points[i].x);
    EXPECT_DOUBLE_EQ(a.route.points[i].y, b.route.points[i].y);
  }
}

TEST(HighwayMergeReferencePath, RejectsTooFewPoints) {
  ReferenceLane source = straight_lane("s", 0, 0, 100, 1);
  ReferenceLane target = straight_lane("t", 0, 0, 100, 400);
  EXPECT_THROW(build_highway_merge_reference_path(source, target, 100.0, {}),
               std::invalid_argument);
}

TEST(HighwayMergeReferencePath, RejectsSourceNotEndingAtMergeComplete) {
  const double merge_complete = 200.0 + 0.5 * 99;
  ReferenceLane source = straight_lane("s", 0, 0, 200.0, 100);
  ReferenceLane target = straight_lane("t", 0, -50, merge_complete, 400);
  // declare a merge-complete 10 m away from the source's actual end
  EXPECT_THROW(
      build_highway_merge_reference_path(source, target, merge_complete + 10.0, {}),
      std::invalid_argument);
}

TEST(HighwayMergeReferencePath, RejectsInsufficientTargetContinuation) {
  const double merge_complete = 100.0 + 0.5 * 99;
  ReferenceLane source = straight_lane("s", 0, 0, 100.0, 100);
  // target has only one point past merge-complete
  ReferenceLane target = straight_lane("t", 0, -0.5 * 99, merge_complete, 2);
  EXPECT_THROW(
      build_highway_merge_reference_path(source, target, merge_complete, {}),
      std::invalid_argument);
}

TEST(HighwayMergeReferencePath, RejectsNonFiniteSourcePoint) {
  const double merge_complete = 100.0 + 0.5 * 99;
  ReferenceLane source = straight_lane("s", 0, 0, 100.0, 100);
  source.points[40].pose.x = std::numeric_limits<double>::quiet_NaN();
  ReferenceLane target = straight_lane("t", 0, -0.5 * 99, merge_complete, 400);
  EXPECT_THROW(
      build_highway_merge_reference_path(source, target, merge_complete, {}),
      std::invalid_argument);
}

TEST(HighwayMergeReferencePath, RejectsDiscontinuousJoin) {
  const double merge_complete = 100.0 + 0.5 * 99;
  ReferenceLane source = straight_lane("s", 0, 0, 100.0, 100);
  // target starts 5 m sideways from where the source ends -> teleport
  ReferenceLane target =
      straight_lane("t", 5.0, -0.5 * 99, merge_complete, 400);
  EXPECT_THROW(
      build_highway_merge_reference_path(source, target, merge_complete, {}),
      std::invalid_argument);
}

TEST(HighwayMergeReferencePath, RejectsNonMonotonicSourceStation) {
  const double merge_complete = 100.0 + 0.5 * 99;
  ReferenceLane source = straight_lane("s", 0, 0, 100.0, 100);
  source.points[50].route_s_m = source.points[10].route_s_m;  // jump backwards
  ReferenceLane target = straight_lane("t", 0, -0.5 * 99, merge_complete, 400);
  EXPECT_THROW(
      build_highway_merge_reference_path(source, target, merge_complete, {}),
      std::invalid_argument);
}

}  // namespace
