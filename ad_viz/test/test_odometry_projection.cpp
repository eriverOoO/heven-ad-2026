#include <gtest/gtest.h>

#include <array>
#include <initializer_list>
#include <limits>

#include "ad_viz/localization/odometry_projection.hpp"
#include "ad_viz/localization/route_elevation_projection.hpp"

namespace
{

TEST(OdometryProjection, FlattensOnlyPositionZ)
{
  nav_msgs::msg::Odometry input;
  input.header.stamp.sec = 123;
  input.header.stamp.nanosec = 456U;
  input.header.frame_id = "map";
  input.child_frame_id = "base_link";
  input.pose.pose.position.x = 10.5;
  input.pose.pose.position.y = -7.25;
  input.pose.pose.position.z = 42.0;
  input.pose.pose.orientation.x = 0.1;
  input.pose.pose.orientation.y = 0.2;
  input.pose.pose.orientation.z = 0.3;
  input.pose.pose.orientation.w = 0.9;
  input.twist.twist.linear.x = 12.0;
  input.twist.twist.linear.y = 0.25;
  input.twist.twist.linear.z = -0.5;
  input.twist.twist.angular.z = 0.75;
  for (std::size_t index = 0; index < input.pose.covariance.size(); ++index) {
    input.pose.covariance[index] = static_cast<double>(index) + 0.125;
    input.twist.covariance[index] = static_cast<double>(index) + 100.125;
  }

  const auto projected = ad_viz::localization::project_odometry_to_ground(input);

  EXPECT_DOUBLE_EQ(input.pose.pose.position.z, 42.0);
  EXPECT_DOUBLE_EQ(projected.pose.pose.position.z, 0.0);
  EXPECT_EQ(projected.header, input.header);
  EXPECT_EQ(projected.child_frame_id, input.child_frame_id);
  EXPECT_DOUBLE_EQ(projected.pose.pose.position.x, input.pose.pose.position.x);
  EXPECT_DOUBLE_EQ(projected.pose.pose.position.y, input.pose.pose.position.y);
  EXPECT_EQ(projected.pose.pose.orientation, input.pose.pose.orientation);
  EXPECT_EQ(projected.pose.covariance, input.pose.covariance);
  EXPECT_EQ(projected.twist.twist, input.twist.twist);
  EXPECT_EQ(projected.twist.covariance, input.twist.covariance);
}

nav_msgs::msg::Path route(
  const std::initializer_list<std::array<double, 3>> points)
{
  nav_msgs::msg::Path path;
  path.header.frame_id = "map";
  for (const auto & point : points) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header = path.header;
    pose.pose.position.x = point[0];
    pose.pose.position.y = point[1];
    pose.pose.position.z = point[2];
    pose.pose.orientation.w = 1.0;
    path.poses.push_back(pose);
  }
  return path;
}

nav_msgs::msg::Odometry route_input(const double x, const double y)
{
  nav_msgs::msg::Odometry input;
  input.header.stamp.sec = 10;
  input.header.frame_id = "odom";
  input.child_frame_id = "base_link";
  input.pose.pose.position.x = x;
  input.pose.pose.position.y = y;
  input.pose.pose.position.z = -25.0;
  input.pose.pose.orientation.z = 0.25;
  input.pose.pose.orientation.w = 0.9682458365518543;
  input.pose.covariance[0] = 4.0;
  input.twist.twist.linear.x = 7.0;
  input.twist.covariance[0] = 9.0;
  return input;
}

TEST(RouteElevationProjection, InterpolatesClosestSegmentAndChangesOnlyZ)
{
  const auto input = route_input(2.5, 1.0);
  const auto path = route({{0.0, 0.0, 10.0}, {10.0, 0.0, 20.0}});

  const auto projected =
    ad_viz::localization::project_odometry_to_route_elevation(
    input, path, 2.0);

  ASSERT_TRUE(projected.has_value());
  EXPECT_DOUBLE_EQ(projected->pose.pose.position.z, 12.5);
  EXPECT_EQ(projected->header, input.header);
  EXPECT_EQ(projected->child_frame_id, input.child_frame_id);
  EXPECT_DOUBLE_EQ(projected->pose.pose.position.x, input.pose.pose.position.x);
  EXPECT_DOUBLE_EQ(projected->pose.pose.position.y, input.pose.pose.position.y);
  EXPECT_EQ(projected->pose.pose.orientation, input.pose.pose.orientation);
  EXPECT_EQ(projected->pose.covariance, input.pose.covariance);
  EXPECT_EQ(projected->twist.twist, input.twist.twist);
  EXPECT_EQ(projected->twist.covariance, input.twist.covariance);
  EXPECT_DOUBLE_EQ(input.pose.pose.position.z, -25.0);
}

TEST(RouteElevationProjection, ClampsAtEndpointAndRejectsFarPosition)
{
  const auto path = route({{0.0, 0.0, 10.0}, {10.0, 0.0, 20.0}});

  const auto endpoint =
    ad_viz::localization::project_odometry_to_route_elevation(
    route_input(-2.0, 0.0), path, 3.0);
  const auto far = ad_viz::localization::project_odometry_to_route_elevation(
    route_input(5.0, 6.0), path, 5.0);

  ASSERT_TRUE(endpoint.has_value());
  EXPECT_DOUBLE_EQ(endpoint->pose.pose.position.z, 10.0);
  EXPECT_FALSE(far.has_value());
}

TEST(RouteElevationProjection, HandlesDegenerateSegmentAndRejectsInvalidData)
{
  const auto duplicate = route({{3.0, 4.0, 7.0}, {3.0, 4.0, 9.0}});
  const auto projected =
    ad_viz::localization::project_odometry_to_route_elevation(
    route_input(3.0, 4.0), duplicate, 1.0);
  ASSERT_TRUE(projected.has_value());
  EXPECT_DOUBLE_EQ(projected->pose.pose.position.z, 7.0);

  auto invalid = duplicate;
  invalid.poses.front().pose.position.z =
    std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(
    ad_viz::localization::project_odometry_to_route_elevation(
      route_input(3.0, 4.0), invalid, 1.0).has_value());
  EXPECT_FALSE(
    ad_viz::localization::project_odometry_to_route_elevation(
      route_input(3.0, 4.0), nav_msgs::msg::Path{}, 1.0).has_value());
  EXPECT_FALSE(
    ad_viz::localization::project_odometry_to_route_elevation(
      route_input(3.0, 4.0), duplicate, -1.0).has_value());
}

TEST(RouteElevationHold, KeepsCurrentOdometryAndLastValidZOutsideRoute)
{
  ad_viz::localization::RouteElevationHold hold;
  auto near_input = route_input(2.5, 1.0);
  auto near_projected = near_input;
  near_projected.pose.pose.position.z = 12.5;
  const auto near_output = hold.apply(near_input, near_projected);
  EXPECT_DOUBLE_EQ(near_output.pose.pose.position.z, 12.5);

  auto far_input = route_input(50.0, 60.0);
  far_input.header.stamp.sec = 20;
  far_input.pose.pose.orientation.z = 0.5;
  far_input.pose.pose.orientation.w = 0.8660254037844386;
  far_input.twist.twist.linear.x = 11.0;
  far_input.pose.covariance[0] = 16.0;
  const auto far_output = hold.apply(far_input, std::nullopt);

  EXPECT_EQ(far_output.header, far_input.header);
  EXPECT_EQ(far_output.child_frame_id, far_input.child_frame_id);
  EXPECT_DOUBLE_EQ(far_output.pose.pose.position.x, 50.0);
  EXPECT_DOUBLE_EQ(far_output.pose.pose.position.y, 60.0);
  EXPECT_DOUBLE_EQ(far_output.pose.pose.position.z, 12.5);
  EXPECT_EQ(far_output.pose.pose.orientation, far_input.pose.pose.orientation);
  EXPECT_EQ(far_output.pose.covariance, far_input.pose.covariance);
  EXPECT_EQ(far_output.twist.twist, far_input.twist.twist);
  EXPECT_EQ(far_output.twist.covariance, far_input.twist.covariance);
}

TEST(RouteElevationHold, PreservesInputZBeforeFirstValidProjection)
{
  ad_viz::localization::RouteElevationHold hold;
  const auto input = route_input(50.0, 60.0);

  const auto output = hold.apply(input, std::nullopt);

  EXPECT_EQ(output, input);
  EXPECT_DOUBLE_EQ(output.pose.pose.position.z, -25.0);
}

TEST(RouteElevationHold, UpdatesHeldZAfterReturningNearRoute)
{
  ad_viz::localization::RouteElevationHold hold;
  const auto input = route_input(2.5, 1.0);
  auto first_projection = input;
  first_projection.pose.pose.position.z = 12.5;
  hold.apply(input, first_projection);

  auto second_projection = input;
  second_projection.pose.pose.position.z = 31.25;
  hold.apply(input, second_projection);

  const auto far_output = hold.apply(route_input(100.0, 100.0), std::nullopt);
  EXPECT_DOUBLE_EQ(far_output.pose.pose.position.z, 31.25);
}

}  // namespace
