#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "ad_planner/visualization/path_tracking_markers.hpp"

namespace {

ad_control::Route three_point_route() {
  return ad_control::Route{{
                               {0.0, 0.0, 0.0},
                               {1.0, 0.0, 0.0},
                               {2.0, 1.0, 0.0},
                           },
                           false};
}

ad_control::RouteSpeedProfile three_point_profile() {
  return ad_control::RouteSpeedProfile{{2.0, 4.0, 6.0}, {0.0, 0.25, 0.5}};
}

void expect_finite_marker(const visualization_msgs::msg::Marker &marker) {
  EXPECT_TRUE(std::isfinite(marker.pose.position.x));
  EXPECT_TRUE(std::isfinite(marker.pose.position.y));
  EXPECT_TRUE(std::isfinite(marker.pose.position.z));
  EXPECT_TRUE(std::isfinite(marker.pose.orientation.x));
  EXPECT_TRUE(std::isfinite(marker.pose.orientation.y));
  EXPECT_TRUE(std::isfinite(marker.pose.orientation.z));
  EXPECT_TRUE(std::isfinite(marker.pose.orientation.w));
  EXPECT_TRUE(std::isfinite(marker.scale.x));
  EXPECT_TRUE(std::isfinite(marker.scale.y));
  EXPECT_TRUE(std::isfinite(marker.scale.z));
  EXPECT_TRUE(std::isfinite(marker.color.r));
  EXPECT_TRUE(std::isfinite(marker.color.g));
  EXPECT_TRUE(std::isfinite(marker.color.b));
  EXPECT_TRUE(std::isfinite(marker.color.a));
  for (const auto &point : marker.points) {
    EXPECT_TRUE(std::isfinite(point.x));
    EXPECT_TRUE(std::isfinite(point.y));
    EXPECT_TRUE(std::isfinite(point.z));
  }
  for (const auto &color : marker.colors) {
    EXPECT_TRUE(std::isfinite(color.r));
    EXPECT_TRUE(std::isfinite(color.g));
    EXPECT_TRUE(std::isfinite(color.b));
    EXPECT_TRUE(std::isfinite(color.a));
    EXPECT_GE(color.r, 0.0F);
    EXPECT_LE(color.r, 1.0F);
    EXPECT_GE(color.g, 0.0F);
    EXPECT_LE(color.g, 1.0F);
    EXPECT_GE(color.b, 0.0F);
    EXPECT_LE(color.b, 1.0F);
  }
}

TEST(PathTrackingMarkers, ColorsSampledRouteByActualProfileSpeed) {
  const auto route = three_point_route();
  const auto profile = three_point_profile();
  builtin_interfaces::msg::Time stamp;
  stamp.sec = 12;

  const auto markers =
      ad_planner::make_route_profile_markers(route, &profile, "map", stamp, 1U);

  ASSERT_EQ(markers.markers.size(), 2U);
  const auto &line = markers.markers[0];
  EXPECT_EQ(line.header.frame_id, "map");
  EXPECT_EQ(line.header.stamp, stamp);
  EXPECT_EQ(line.ns, "path_tracking_profile");
  EXPECT_EQ(line.id, 0);
  EXPECT_EQ(line.type, visualization_msgs::msg::Marker::LINE_STRIP);
  ASSERT_EQ(line.points.size(), 3U);
  ASSERT_EQ(line.colors.size(), line.points.size());
  EXPECT_GT(line.colors.front().b, line.colors.front().r);
  EXPECT_GT(line.colors.back().r, line.colors.back().b);
  EXPECT_NE(line.colors.front(), line.colors.back());
  expect_finite_marker(line);
}

TEST(PathTrackingMarkers, EmitsFiniteCurvatureIndicators) {
  const auto route = three_point_route();
  const auto profile = three_point_profile();

  const auto markers = ad_planner::make_route_profile_markers(
      route, &profile, "map", builtin_interfaces::msg::Time{}, 1U);

  ASSERT_EQ(markers.markers.size(), 2U);
  const auto &curvature = markers.markers[1];
  EXPECT_EQ(curvature.ns, "path_tracking_curvature");
  EXPECT_EQ(curvature.id, 1);
  EXPECT_EQ(curvature.type, visualization_msgs::msg::Marker::LINE_LIST);
  ASSERT_EQ(curvature.points.size(), 6U);
  expect_finite_marker(curvature);
}

TEST(PathTrackingMarkers, RejectsMismatchedProfilesAndZeroStride) {
  const auto route = three_point_route();
  auto profile = three_point_profile();
  profile.speed_mps.pop_back();
  EXPECT_THROW(ad_planner::make_route_profile_markers(
                   route, &profile, "map", builtin_interfaces::msg::Time{}, 1U),
               std::invalid_argument);

  profile = three_point_profile();
  profile.curvature_inv_m.pop_back();
  EXPECT_THROW(ad_planner::make_route_profile_markers(
                   route, &profile, "map", builtin_interfaces::msg::Time{}, 1U),
               std::invalid_argument);

  profile = three_point_profile();
  EXPECT_THROW(ad_planner::make_route_profile_markers(
                   route, &profile, "map", builtin_interfaces::msg::Time{}, 0U),
               std::invalid_argument);
}

TEST(PathTrackingMarkers, ClassicBackendsDoNotInventProfileMarkers) {
  const auto markers = ad_planner::make_route_profile_markers(
      three_point_route(), nullptr, "map", builtin_interfaces::msg::Time{}, 1U);
  EXPECT_TRUE(markers.markers.empty());
}

} // namespace
