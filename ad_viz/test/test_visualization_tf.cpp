#include <gtest/gtest.h>

#include <limits>

#include "ad_viz/localization/visualization_tf.hpp"

namespace
{

geometry_msgs::msg::TransformStamped transform(
  const std::string & parent, const std::string & child, const double z)
{
  geometry_msgs::msg::TransformStamped output;
  output.header.stamp.sec = 10;
  output.header.frame_id = parent;
  output.child_frame_id = child;
  output.transform.translation.z = z;
  output.transform.rotation.w = 1.0;
  return output;
}

TEST(VisualizationTf, RemovesOnlyCanonicalOdomToBaseLinkEdge)
{
  tf2_msgs::msg::TFMessage input;
  input.transforms = {
    transform("odom", "base_link", 0.0),
    transform("base_link", "rear_axle_link", 0.2),
    transform("map", "other_dynamic_frame", 3.0)};

  const auto output = ad_viz::localization::without_transform_edge(
    input, "odom", "base_link");

  ASSERT_EQ(output.transforms.size(), 2U);
  EXPECT_EQ(output.transforms[0], input.transforms[1]);
  EXPECT_EQ(output.transforms[1], input.transforms[2]);
}

TEST(VisualizationTf, BuildsVisualizationBaseTransformFromRouteOdometry)
{
  nav_msgs::msg::Odometry odometry;
  odometry.header.stamp.sec = 42;
  odometry.header.frame_id = "odom";
  odometry.child_frame_id = "base_link";
  odometry.pose.pose.position.x = 1.0;
  odometry.pose.pose.position.y = 2.0;
  odometry.pose.pose.position.z = 28.5;
  odometry.pose.pose.orientation.z = 0.5;
  odometry.pose.pose.orientation.w = 0.8660254037844386;

  const auto output =
    ad_viz::localization::visualization_transform_from_odometry(
    odometry, "odom", "base_link");

  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->header, odometry.header);
  EXPECT_EQ(output->child_frame_id, "base_link");
  EXPECT_DOUBLE_EQ(output->transform.translation.x, 1.0);
  EXPECT_DOUBLE_EQ(output->transform.translation.y, 2.0);
  EXPECT_DOUBLE_EQ(output->transform.translation.z, 28.5);
  EXPECT_EQ(output->transform.rotation, odometry.pose.pose.orientation);
}

TEST(VisualizationTf, RejectsWrongFramesAndNonFinitePose)
{
  nav_msgs::msg::Odometry odometry;
  odometry.header.frame_id = "map";
  odometry.child_frame_id = "base_link";
  odometry.pose.pose.orientation.w = 1.0;
  EXPECT_FALSE(
    ad_viz::localization::visualization_transform_from_odometry(
      odometry, "odom", "base_link").has_value());

  odometry.header.frame_id = "odom";
  odometry.pose.pose.position.z =
    std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(
    ad_viz::localization::visualization_transform_from_odometry(
      odometry, "odom", "base_link").has_value());
}

}  // namespace
