#include <gtest/gtest.h>

#include <limits>

#include "ad_planner/visualization/planner_visualization.hpp"

namespace ad_planner
{
namespace
{

TEST(PlannerVisualization, BuildsGlobalPathWithStableMapFrame)
{
  const Route route{{Point3{1.0, 2.0, 3.0}, Point3{4.0, 5.0, 6.0}}, false};
  const rclcpp::Time stamp(12, 34, RCL_ROS_TIME);

  const auto path = make_global_path_message(route, "map", stamp);

  EXPECT_EQ(path.header.frame_id, "map");
  EXPECT_EQ(path.header.stamp, stamp);
  ASSERT_EQ(path.poses.size(), 2U);
  EXPECT_DOUBLE_EQ(path.poses[0].pose.position.x, 1.0);
  EXPECT_DOUBLE_EQ(path.poses[1].pose.position.z, 6.0);
  EXPECT_DOUBLE_EQ(path.poses[0].pose.orientation.w, 1.0);
}

TEST(PlannerVisualization, BuildsSelectedAndCandidateLocalMotionMessages)
{
  LocalPlanningResult result;
  result.valid = true;
  result.trajectory.frame_id = "odom";
  result.trajectory.points = {
    TimedTrajectoryPoint{Pose2{1.0, 2.0, 0.4}, 0.1, 3.0, 0.0},
    TimedTrajectoryPoint{Pose2{2.0, 3.0, 0.5}, 0.2, 3.0, 0.0}};
  result.candidate_trajectories = {result.trajectory};
  const rclcpp::Time stamp(21, 0, RCL_ROS_TIME);

  const auto messages = make_local_motion_visualization(
    &result, "odom", stamp);

  EXPECT_EQ(messages.selected_path.header.frame_id, "odom");
  ASSERT_EQ(messages.selected_path.poses.size(), 2U);
  EXPECT_NE(messages.selected_path.poses[0].pose.orientation.z, 0.0);
  ASSERT_EQ(messages.candidates.markers.size(), 2U);
  EXPECT_EQ(
    messages.candidates.markers[0].action,
    visualization_msgs::msg::Marker::DELETEALL);
  EXPECT_EQ(
    messages.candidates.markers[1].type,
    visualization_msgs::msg::Marker::LINE_STRIP);
  EXPECT_EQ(messages.candidates.markers[1].points.size(), 2U);
}

TEST(PlannerVisualization, EmptyLocalMotionClearsStaleMarkersAndPath)
{
  const rclcpp::Time stamp(21, 0, RCL_ROS_TIME);

  const auto messages = make_local_motion_visualization(
    nullptr, "odom", stamp);

  EXPECT_TRUE(messages.selected_path.poses.empty());
  ASSERT_EQ(messages.candidates.markers.size(), 1U);
  EXPECT_EQ(
    messages.candidates.markers.front().action,
    visualization_msgs::msg::Marker::DELETEALL);
}

TEST(PlannerVisualization, InvalidCandidateIsNotPublished)
{
  LocalPlanningResult result;
  result.trajectory.frame_id = "odom";
  result.trajectory.points = {
    TimedTrajectoryPoint{Pose2{0.0, 0.0, 0.0}, 0.1, 1.0, 0.0}};
  auto malformed = result.trajectory;
  malformed.points.front().pose.x =
    std::numeric_limits<double>::quiet_NaN();
  result.candidate_trajectories = {malformed};

  const auto messages = make_local_motion_visualization(
    &result, "odom", rclcpp::Time(1, 0, RCL_ROS_TIME));

  ASSERT_EQ(messages.candidates.markers.size(), 1U);
  EXPECT_EQ(
    messages.candidates.markers.front().action,
    visualization_msgs::msg::Marker::DELETEALL);
}

TEST(PlannerVisualization, BuildsControllerTargetPathAndSpeed)
{
  ControllerResult result;
  result.valid = true;
  result.target_speed_mps = 7.5;
  result.target = Point3{3.0, 4.0, 0.0};
  result.local_trajectory = Trajectory{
    {Pose2{1.0, 2.0, 0.0}, Pose2{2.0, 3.0, 0.0}}};

  const auto messages = make_controller_visualization(
    result, "map", rclcpp::Time(2, 0, RCL_ROS_TIME));

  ASSERT_TRUE(messages.target_speed.has_value());
  EXPECT_FLOAT_EQ(messages.target_speed->data, 7.5F);
  ASSERT_TRUE(messages.target.has_value());
  EXPECT_DOUBLE_EQ(messages.target->pose.position.x, 3.0);
  ASSERT_TRUE(messages.local_path.has_value());
  EXPECT_EQ(messages.local_path->poses.size(), 2U);
}

TEST(PlannerVisualization, NamesPathTrackingPublisherByArtifact)
{
  PlannerVisualizationTopics topics;
  topics.path_tracking = "/ad/viz/planner/path_tracking";
  EXPECT_EQ(topics.path_tracking, "/ad/viz/planner/path_tracking");

  const auto publish = &PlannerVisualization::publish_path_tracking;
  EXPECT_NE(publish, nullptr);
}

}  // namespace
}  // namespace ad_planner
