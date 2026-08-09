#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ad_planner/visualization/route_markers.hpp"

namespace
{

constexpr double kPi = 3.14159265358979323846;

ad_planner::ReferenceLane straight_lane(
  const std::string & id, const double y_m,
  const double left_width_m = 1.0, const double right_width_m = 1.0)
{
  ad_planner::ReferenceLane lane;
  lane.lane_sequence_id = id;
  lane.points = {
    {{-1.0, y_m, 0.0}, 0.0, 0.0, left_width_m, right_width_m, 10.0},
    {{10.0, y_m, 0.0}, 11.0, 0.0, left_width_m, right_width_m, 10.0},
  };
  return lane;
}

ad_planner::ReferenceCorridor corridor_with_lanes(
  std::vector<ad_planner::ReferenceLane> lanes)
{
  ad_planner::ReferenceCorridor corridor;
  corridor.frame_id = "map";
  corridor.lanes = std::move(lanes);
  corridor.primary_lane_index = 0U;
  return corridor;
}

ad_planner::OccupancyGrid occupied_grid(
  const ad_planner::Pose2 & origin, const double resolution,
  const std::size_t width, const std::size_t height)
{
  ad_planner::OccupancyGrid grid;
  grid.origin = origin;
  grid.resolution = resolution;
  grid.width = width;
  grid.height = height;
  grid.cells.assign(width * height, std::int8_t{100});
  grid.valid = true;
  grid.fresh = true;
  return grid;
}

const visualization_msgs::msg::Marker & red_marker(
  const visualization_msgs::msg::MarkerArray & markers)
{
  return markers.markers.at(1U);
}

const visualization_msgs::msg::Marker & blue_marker(
  const visualization_msgs::msg::MarkerArray & markers)
{
  return markers.markers.at(2U);
}

TEST(RouteMarkers, ColorsOccupiedCellsInsideAnyCorridorLaneRed)
{
  const auto corridor = corridor_with_lanes(
    {straight_lane("primary", 0.0), straight_lane("adjacent", 4.0)});
  const auto grid = occupied_grid({0.0, -2.0, 0.0}, 1.0, 1U, 8U);

  const auto markers = ad_planner::build_occupancy_relevance_markers(
    "map", grid, corridor, 50);

  ASSERT_EQ(markers.markers.size(), 3U);
  EXPECT_EQ(
    markers.markers.front().action,
    visualization_msgs::msg::Marker::DELETEALL);
  EXPECT_EQ(red_marker(markers).type, visualization_msgs::msg::Marker::CUBE_LIST);
  EXPECT_EQ(red_marker(markers).points.size(), 4U);
  EXPECT_FLOAT_EQ(red_marker(markers).color.r, 1.0F);
  EXPECT_EQ(blue_marker(markers).type, visualization_msgs::msg::Marker::CUBE_LIST);
  EXPECT_EQ(blue_marker(markers).points.size(), 4U);
  EXPECT_FLOAT_EQ(blue_marker(markers).color.b, 1.0F);
}

TEST(RouteMarkers, PreservesRotatedOccupancyGridMarkerPose)
{
  const auto corridor = corridor_with_lanes({straight_lane("primary", 0.0)});
  const auto grid = occupied_grid({2.0, -0.5, kPi * 0.25}, 1.0, 1U, 1U);

  const auto markers = ad_planner::build_occupancy_relevance_markers(
    "map", grid, corridor, 50);

  ASSERT_EQ(red_marker(markers).points.size(), 1U);
  EXPECT_NEAR(red_marker(markers).pose.position.x, 2.0, 1.0e-9);
  EXPECT_NEAR(red_marker(markers).pose.position.y, -0.5, 1.0e-9);
  EXPECT_NEAR(red_marker(markers).pose.orientation.z, std::sin(kPi * 0.125), 1.0e-9);
  EXPECT_NEAR(red_marker(markers).pose.orientation.w, std::cos(kPi * 0.125), 1.0e-9);
  EXPECT_NEAR(red_marker(markers).points[0].x, 0.5, 1.0e-9);
  EXPECT_NEAR(red_marker(markers).points[0].y, 0.5, 1.0e-9);
  EXPECT_TRUE(blue_marker(markers).points.empty());
}

TEST(RouteMarkers, CapsDenseOccupancyPayloadDeterministically)
{
  const auto corridor = corridor_with_lanes(
    {straight_lane("wide", 200.0, 201.0, 201.0)});
  const auto grid = occupied_grid({0.0, 0.0, 0.0}, 1.0, 520U, 400U);
  ad_planner::RouteMarkerLimits limits;
  limits.maximum_occupancy_points_per_class = 16U;

  const auto first = ad_planner::build_occupancy_relevance_markers(
    "map", grid, corridor, 50, limits);
  const auto second = ad_planner::build_occupancy_relevance_markers(
    "map", grid, corridor, 50, limits);

  ASSERT_EQ(first.markers.size(), 3U);
  ASSERT_EQ(red_marker(first).points.size(), 16U);
  EXPECT_LE(blue_marker(first).points.size(), 16U);
  ASSERT_EQ(red_marker(first).points.size(), red_marker(second).points.size());
  for (std::size_t index = 0U; index < red_marker(first).points.size(); ++index) {
    EXPECT_EQ(red_marker(first).points[index], red_marker(second).points[index]);
  }
}

TEST(RouteMarkers, ColorsFootprintPerimeterPortionsByCorridorMembership)
{
  const auto corridor = corridor_with_lanes({straight_lane("primary", 0.0)});
  ad_planner::PredictedObject object;
  object.object_id = "crossing_box";
  object.footprints = {
    {0.0, {5.0, 1.0, 0.0}, 2.0, 4.0, 0.0, 0.0},
  };

  const auto markers = ad_planner::build_predicted_relevance_markers(
    "map", {object}, corridor);

  ASSERT_EQ(markers.markers.size(), 3U);
  EXPECT_EQ(
    markers.markers.front().action,
    visualization_msgs::msg::Marker::DELETEALL);
  EXPECT_EQ(red_marker(markers).type, visualization_msgs::msg::Marker::LINE_LIST);
  EXPECT_EQ(blue_marker(markers).type, visualization_msgs::msg::Marker::LINE_LIST);
  EXPECT_FALSE(red_marker(markers).points.empty());
  EXPECT_FALSE(blue_marker(markers).points.empty());
  EXPECT_EQ(red_marker(markers).points.size() % 2U, 0U);
  EXPECT_EQ(blue_marker(markers).points.size() % 2U, 0U);
  EXPECT_FLOAT_EQ(red_marker(markers).color.r, 1.0F);
  EXPECT_FLOAT_EQ(blue_marker(markers).color.b, 1.0F);
}

TEST(RouteMarkers, ColorsCrossingPredictionTrajectoryRedAndBlue)
{
  const auto corridor = corridor_with_lanes({straight_lane("primary", 0.0)});
  ad_planner::PredictedObject object;
  object.object_id = "crossing_path";
  object.footprints = {
    {0.0, {5.0, -3.0, kPi * 0.5}, 0.2, 0.2, 0.0, 0.0},
    {1.0, {5.0, 3.0, kPi * 0.5}, 0.2, 0.2, 0.0, 0.0},
  };

  const auto markers = ad_planner::build_predicted_relevance_markers(
    "map", {object}, corridor);

  EXPECT_FALSE(red_marker(markers).points.empty());
  EXPECT_FALSE(blue_marker(markers).points.empty());
}

TEST(RouteMarkers, CapsPredictionSegmentsDeterministically)
{
  const auto corridor = corridor_with_lanes(
    {straight_lane("wide", 0.0, 10.0, 10.0)});
  ad_planner::PredictedObject object;
  object.object_id = "long_prediction";
  for (std::size_t index = 0U; index < 50U; ++index) {
    object.footprints.push_back(
      {static_cast<double>(index),
        {static_cast<double>(index) * 0.1, 0.0, 0.0},
        4.0, 2.0, 0.0, 0.0});
  }
  ad_planner::RouteMarkerLimits limits;
  limits.maximum_prediction_segments_per_class = 7U;

  const auto first = ad_planner::build_predicted_relevance_markers(
    "map", {object}, corridor, limits);
  const auto second = ad_planner::build_predicted_relevance_markers(
    "map", {object}, corridor, limits);

  ASSERT_EQ(red_marker(first).points.size(), 14U);
  EXPECT_LE(blue_marker(first).points.size(), 14U);
  ASSERT_EQ(red_marker(first).points.size(), red_marker(second).points.size());
  for (std::size_t index = 0U; index < red_marker(first).points.size(); ++index) {
    EXPECT_EQ(red_marker(first).points[index], red_marker(second).points[index]);
  }
}

TEST(RouteMarkers, RejectsPredictionGeometryThatWouldRequireUnboundedSampling)
{
  const auto corridor = corridor_with_lanes({straight_lane("primary", 0.0)});
  ad_planner::PredictedObject object;
  object.object_id = "oversized";
  object.footprints = {
    {0.0, {0.0, 0.0, 0.0}, 20000.0, 1.0, 0.0, 0.0},
  };
  ad_planner::RouteMarkerLimits limits;
  limits.maximum_prediction_segments_per_class = 1U;

  EXPECT_THROW(
    ad_planner::build_predicted_relevance_markers(
      "map", {object}, corridor, limits),
    std::invalid_argument);
}

TEST(RouteMarkers, RejectsMismatchedFramesAndMalformedInputs)
{
  const auto corridor = corridor_with_lanes({straight_lane("primary", 0.0)});
  auto grid = occupied_grid({0.0, 0.0, 0.0}, 1.0, 1U, 1U);

  EXPECT_THROW(
    ad_planner::build_occupancy_relevance_markers(
      "base_link", grid, corridor, 50),
    std::invalid_argument);
  EXPECT_THROW(
    ad_planner::build_occupancy_relevance_markers(
      "map", grid, corridor, -1),
    std::invalid_argument);

  grid.fresh = false;
  EXPECT_THROW(
    ad_planner::build_occupancy_relevance_markers(
      "map", grid, corridor, 50),
    std::invalid_argument);

  grid.fresh = true;
  grid.cells.front() = 101;
  EXPECT_THROW(
    ad_planner::build_occupancy_relevance_markers(
      "map", grid, corridor, 50),
    std::invalid_argument);

  ad_planner::RouteMarkerLimits limits;
  limits.maximum_prediction_segments_per_class = 0U;
  EXPECT_THROW(
    ad_planner::build_predicted_relevance_markers(
      "map", {}, corridor, limits),
    std::invalid_argument);

  ad_planner::PredictedObject object;
  object.object_id = "invalid";
  object.footprints = {
    {0.0, {0.0, 0.0, 0.0}, 1.0, 1.0, 0.0, 0.0},
    {0.5, {1.0, 0.0, 0.0}, 1.0, 1.0, 0.0, 0.0},
    {0.25, {2.0, 0.0, 0.0}, 1.0, 1.0, 0.0, 0.0},
  };
  EXPECT_THROW(
    ad_planner::build_predicted_relevance_markers(
      "map", {object}, corridor),
    std::invalid_argument);

  object.footprints.resize(1U);
  object.footprints.front().length_m =
    std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(
    ad_planner::build_predicted_relevance_markers(
      "map", {object}, corridor),
    std::invalid_argument);

  object.footprints.front().length_m = 1.0;
  object.footprints.front().covariance_xx = 1.0e200;
  object.footprints.front().covariance_yy = 1.0e200;
  object.footprints.front().covariance_xy =
    std::numeric_limits<double>::max();
  EXPECT_THROW(
    ad_planner::build_predicted_relevance_markers(
      "map", {object}, corridor),
    std::invalid_argument);
}

}  // namespace
