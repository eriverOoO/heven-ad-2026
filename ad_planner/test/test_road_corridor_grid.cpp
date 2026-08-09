#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "ad_planner/local_planning/common/road_corridor_grid.hpp"

namespace {

constexpr double kHalfPi = 1.57079632679489661923;

using ad_planner::express_road_corridor_mask_in_base_frame;
using ad_planner::make_route_aligned_grid_template;
using ad_planner::Pose2;
using ad_planner::PreparedRoadCorridor;
using ad_planner::query_route_slice_occupancy;
using ad_planner::rasterize_road_corridor;
using ad_planner::ReferenceCorridor;
using ad_planner::ReferenceLane;
using ad_planner::ReferencePoint;
using ad_planner::RoadCorridorGridWindow;
using ad_planner::RoadCorridorGridWork;

ReferencePoint point(const double x, const double y, const double yaw,
                     const double route_s, const double left_width,
                     const double right_width) {
  ReferencePoint result;
  result.pose = Pose2{x, y, yaw};
  result.route_s_m = route_s;
  result.left_width_m = left_width;
  result.right_width_m = right_width;
  result.speed_limit_mps = 20.0;
  return result;
}

ReferenceLane straight_lane(const double y = 0.0, const double left_width = 1.0,
                            const double right_width = 1.0) {
  ReferenceLane lane;
  lane.lane_sequence_id = "straight";
  lane.points = {point(0.0, y, 0.0, 0.0, left_width, right_width),
                 point(10.0, y, 0.0, 10.0, left_width, right_width)};
  return lane;
}

ReferenceCorridor corridor_with(ReferenceLane lane) {
  ReferenceCorridor corridor;
  corridor.frame_id = "map";
  corridor.lanes.push_back(std::move(lane));
  return corridor;
}

nav_msgs::msg::OccupancyGrid grid(const std::uint32_t width,
                                  const std::uint32_t height,
                                  const double origin_x, const double origin_y,
                                  const float resolution = 1.0F,
                                  const double yaw = 0.0) {
  nav_msgs::msg::OccupancyGrid result;
  result.header.frame_id = "map";
  result.info.resolution = resolution;
  result.info.width = width;
  result.info.height = height;
  result.info.origin.position.x = origin_x;
  result.info.origin.position.y = origin_y;
  result.info.origin.orientation.z = std::sin(0.5 * yaw);
  result.info.origin.orientation.w = std::cos(0.5 * yaw);
  result.data.assign(static_cast<std::size_t>(width) * height, 0);
  return result;
}

TEST(RoadCorridorGrid, RasterizesOnlyWholeCellsInsideStraightCorridor) {
  const auto route = corridor_with(straight_lane());
  const auto geometry = grid(4U, 4U, 0.0, -2.0);

  const auto mask = rasterize_road_corridor(route, geometry);

  EXPECT_EQ(mask.header.frame_id, "map");
  EXPECT_EQ(mask.info.width, 4U);
  EXPECT_EQ(mask.info.height, 4U);
  EXPECT_EQ(mask.data,
            (std::vector<std::int8_t>{100, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0,
                                      0, 100, 100, 100, 100}));
}

TEST(RoadCorridorGrid, InterpolatesLaneWidthAlongRoute) {
  auto lane = straight_lane();
  lane.points.front().left_width_m = 0.5;
  lane.points.back().left_width_m = 2.5;
  const auto route = corridor_with(std::move(lane));
  const auto geometry = grid(10U, 1U, 0.0, 1.0);

  const auto mask = rasterize_road_corridor(route, geometry);

  EXPECT_EQ(mask.data, (std::vector<std::int8_t>{100, 100, 100, 100, 100, 100,
                                                 100, 100, 0, 0}));
}

TEST(RoadCorridorGrid, HonorsRotatedTemplateGridGeometry) {
  ReferenceLane lane;
  lane.lane_sequence_id = "north";
  lane.points = {point(0.0, 0.0, kHalfPi, 0.0, 1.0, 1.0),
                 point(0.0, 4.0, kHalfPi, 4.0, 1.0, 1.0)};
  const auto route = corridor_with(std::move(lane));
  const auto geometry = grid(4U, 4U, 2.0, 0.0, 1.0F, kHalfPi);

  const auto mask = rasterize_road_corridor(route, geometry);

  EXPECT_EQ(mask.data,
            (std::vector<std::int8_t>{100, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0,
                                      0, 100, 100, 100, 100}));
}

TEST(RoadCorridorGrid, KeepsCellsAcrossAdjacentLaneSeamDrivable) {
  ReferenceCorridor route;
  route.frame_id = "map";
  route.lanes = {straight_lane(-1.0), straight_lane(1.0)};
  route.lanes[0].left_lane_indices.push_back(1U);
  route.lanes[1].right_lane_indices.push_back(0U);
  const auto geometry = grid(2U, 1U, 0.0, -0.5);

  const auto mask = rasterize_road_corridor(route, geometry);

  EXPECT_EQ(mask.data, (std::vector<std::int8_t>{0, 0}));
}

TEST(RoadCorridorGrid, RejectsInvalidRouteAndTemplateGeometry) {
  const auto valid_route = corridor_with(straight_lane());
  const auto valid_grid = grid(2U, 2U, 0.0, -1.0);

  auto bad_route = valid_route;
  bad_route.primary_lane_index = 1U;
  EXPECT_THROW(rasterize_road_corridor(bad_route, valid_grid),
               std::invalid_argument);

  bad_route = valid_route;
  bad_route.lanes.front().points.front().left_width_m = 0.0;
  EXPECT_THROW(rasterize_road_corridor(bad_route, valid_grid),
               std::invalid_argument);

  auto bad_grid = valid_grid;
  bad_grid.header.frame_id = "odom";
  EXPECT_THROW(rasterize_road_corridor(valid_route, bad_grid),
               std::invalid_argument);

  bad_grid = valid_grid;
  bad_grid.info.origin.orientation.w = 0.0;
  EXPECT_THROW(rasterize_road_corridor(valid_route, bad_grid),
               std::invalid_argument);

  bad_grid = valid_grid;
  bad_grid.info.resolution = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(rasterize_road_corridor(valid_route, bad_grid),
               std::invalid_argument);
}

TEST(RoadCorridorGrid, QueriesOccupiedAndUnknownCellsOnlyInRouteSlice) {
  const auto route = corridor_with(straight_lane());
  auto occupancy = grid(10U, 4U, 0.0, -2.0);
  occupancy.data[1U * 10U + 3U] = 80;
  occupancy.data[3U * 10U + 4U] = 100;
  occupancy.data[2U * 10U + 5U] = -1;
  occupancy.data[1U * 10U + 8U] = 90;

  const auto result =
      query_route_slice_occupancy(route, occupancy, 2.0, 6.0, 50);

  EXPECT_EQ(result.occupied_cell_count, 1U);
  ASSERT_TRUE(result.nearest_occupied_s_m.has_value());
  EXPECT_NEAR(*result.nearest_occupied_s_m, 3.5, 1e-9);
  EXPECT_EQ(result.unknown_cell_count, 1U);
}

TEST(RoadCorridorGrid, QueryKeepsInWindowProjectionAtSelfIntersection) {
  ReferenceCorridor route;
  route.frame_id = "map";
  ReferenceLane lane;
  lane.lane_sequence_id = "self-intersection";
  lane.points = {point(-2.0, 0.0, 0.0, 0.0, 1.0, 1.0),
                 point(2.0, 0.0, 0.0, 4.0, 1.0, 1.0),
                 point(0.0, -2.0, kHalfPi, 8.0, 1.0, 1.0),
                 point(0.0, 2.0, kHalfPi, 12.0, 1.0, 1.0)};
  route.lanes.push_back(std::move(lane));
  auto occupancy = grid(1U, 1U, -0.5, -0.5);
  occupancy.data.front() = 100;

  const auto result =
      query_route_slice_occupancy(route, occupancy, 9.0, 11.0, 50);

  EXPECT_EQ(result.occupied_cell_count, 1U);
  ASSERT_TRUE(result.nearest_occupied_s_m.has_value());
  EXPECT_NEAR(*result.nearest_occupied_s_m, 10.0, 1e-9);
}

TEST(RoadCorridorGrid, QueryRejectsMalformedCostsAndRouteInterval) {
  const auto route = corridor_with(straight_lane());
  auto occupancy = grid(2U, 2U, 0.0, -1.0);
  occupancy.data.front() = 101;

  EXPECT_THROW(query_route_slice_occupancy(route, occupancy, 0.0, 2.0, 50),
               std::invalid_argument);

  occupancy.data.front() = 0;
  EXPECT_THROW(query_route_slice_occupancy(route, occupancy, 2.0, 1.0, 50),
               std::invalid_argument);
  EXPECT_THROW(query_route_slice_occupancy(route, occupancy, 0.0, 2.0, -1),
               std::invalid_argument);
}

TEST(RoadCorridorGrid, RasterizesCpSizedGridWithinLivePlanningBudget) {
  ReferenceLane lane;
  lane.lane_sequence_id = "cp3-cp6-sized";
  lane.points.reserve(704U);
  for (std::size_t index = 0U; index < 704U; ++index) {
    const double x = 0.1 * static_cast<double>(index);
    lane.points.push_back(point(x, 0.0, 0.0, x, 1.75, 1.75));
  }
  const auto route = corridor_with(std::move(lane));
  const auto geometry = grid(1040U, 200U, 0.0, -10.0, 0.1F);

  const auto started = std::chrono::steady_clock::now();
  const auto mask = rasterize_road_corridor(route, geometry);
  const std::chrono::duration<double> elapsed =
      std::chrono::steady_clock::now() - started;

  EXPECT_EQ(mask.data.size(), 208000U);
  EXPECT_LT(elapsed.count(), 0.5);
  RecordProperty("elapsed_ms", elapsed.count() * 1000.0);
}

TEST(RoadCorridorGrid, PreparedProductionRouteBoundsWorkToLocalWindow) {
  ReferenceLane lane;
  lane.lane_sequence_id = "production-sized";
  lane.points.reserve(6441U);
  for (std::size_t index = 0U; index < 6441U; ++index) {
    const double x = static_cast<double>(index);
    lane.points.push_back(point(x, 0.0, 0.0, x, 1.75, 1.75));
  }
  auto route = corridor_with(std::move(lane));
  const PreparedRoadCorridor prepared(route);
  const auto local_grid = grid(1040U, 200U, -4.0, -10.0, 0.1F);

  RoadCorridorGridWork raster_work;
  const auto mask = rasterize_road_corridor(prepared, local_grid, &raster_work);

  EXPECT_EQ(mask.data.size(), 208000U);
  EXPECT_EQ(raster_work.total_segment_count, 6440U);
  EXPECT_LE(raster_work.candidate_segment_count, 160U);
  EXPECT_LT(raster_work.candidate_segment_count,
            raster_work.total_segment_count / 10U);

  auto occupancy = local_grid;
  occupancy.data[100U * 1040U + 500U] = 100;
  RoadCorridorGridWork query_work;
  const auto result = query_route_slice_occupancy(prepared, occupancy, 0.0,
                                                  100.0, 50, &query_work);

  EXPECT_EQ(result.occupied_cell_count, 1U);
  EXPECT_EQ(query_work.total_segment_count, 6440U);
  EXPECT_LE(query_work.candidate_segment_count, 160U);
  EXPECT_LT(query_work.candidate_segment_count,
            query_work.total_segment_count / 10U);
}

TEST(RoadCorridorGrid, PreparedCorridorOwnsValidatedImmutableGeometry) {
  auto route = corridor_with(straight_lane());
  const PreparedRoadCorridor prepared(route);
  route.lanes.front().points.front().left_width_m = 0.0;
  const auto local_grid = grid(4U, 4U, 0.0, -2.0);

  EXPECT_NO_THROW(rasterize_road_corridor(prepared, local_grid));
  EXPECT_THROW(static_cast<void>(PreparedRoadCorridor{route}),
               std::invalid_argument);
}

TEST(RoadCorridorGrid, BuildsRouteTemplateAndEquivalentBaseFrameMask) {
  const RoadCorridorGridWindow window{-4.0, 100.0, -10.0, 10.0, 0.1};
  builtin_interfaces::msg::Time stamp;
  stamp.sec = 123;
  stamp.nanosec = 456U;

  auto route_template = make_route_aligned_grid_template(
      window, "map", stamp, Pose2{10.0, 20.0, kHalfPi});

  EXPECT_EQ(route_template.header.frame_id, "map");
  EXPECT_EQ(route_template.header.stamp, stamp);
  EXPECT_EQ(route_template.info.map_load_time, stamp);
  EXPECT_EQ(route_template.info.width, 1040U);
  EXPECT_EQ(route_template.info.height, 200U);
  EXPECT_FLOAT_EQ(route_template.info.resolution, 0.1F);
  EXPECT_NEAR(route_template.info.origin.position.x, 20.0, 1e-9);
  EXPECT_NEAR(route_template.info.origin.position.y, 16.0, 1e-9);
  EXPECT_NEAR(route_template.info.origin.orientation.z, std::sin(kHalfPi / 2.0),
              1e-9);
  EXPECT_NEAR(route_template.info.origin.orientation.w, std::cos(kHalfPi / 2.0),
              1e-9);
  ASSERT_EQ(route_template.data.size(), 208000U);

  route_template.data[1234U] = 0;
  const auto base_mask = express_road_corridor_mask_in_base_frame(
      route_template, window, "base_link");

  EXPECT_EQ(base_mask.header.frame_id, "base_link");
  EXPECT_EQ(base_mask.header.stamp, stamp);
  EXPECT_EQ(base_mask.info.map_load_time, stamp);
  EXPECT_EQ(base_mask.info.width, 1040U);
  EXPECT_EQ(base_mask.info.height, 200U);
  EXPECT_DOUBLE_EQ(base_mask.info.origin.position.x, -4.0);
  EXPECT_DOUBLE_EQ(base_mask.info.origin.position.y, -10.0);
  EXPECT_DOUBLE_EQ(base_mask.info.origin.position.z, 0.0);
  EXPECT_DOUBLE_EQ(base_mask.info.origin.orientation.x, 0.0);
  EXPECT_DOUBLE_EQ(base_mask.info.origin.orientation.y, 0.0);
  EXPECT_DOUBLE_EQ(base_mask.info.origin.orientation.z, 0.0);
  EXPECT_DOUBLE_EQ(base_mask.info.origin.orientation.w, 1.0);
  EXPECT_EQ(base_mask.data, route_template.data);
}

TEST(RoadCorridorGrid, RejectsUnrepresentableGridWindow) {
  builtin_interfaces::msg::Time stamp;
  stamp.sec = 1;
  const Pose2 identity{};

  EXPECT_THROW(make_route_aligned_grid_template(
                   RoadCorridorGridWindow{-4.0, 100.05, -10.0, 10.0, 0.1},
                   "map", stamp, identity),
               std::invalid_argument);
  EXPECT_THROW(make_route_aligned_grid_template(
                   RoadCorridorGridWindow{-4.0, 100.0, -10.0, 10.0, 0.0}, "map",
                   stamp, identity),
               std::invalid_argument);
  EXPECT_THROW(make_route_aligned_grid_template(
                   RoadCorridorGridWindow{-4.0, 100.0, -10.0, 10.0, 0.1}, "",
                   stamp, identity),
               std::invalid_argument);
}

} // namespace
