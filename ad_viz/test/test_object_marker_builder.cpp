#include <ad_viz/perception/object_marker_builder.hpp>

#include <gtest/gtest.h>

#include <autoware_perception_msgs/msg/detected_object.hpp>
#include <autoware_perception_msgs/msg/object_classification.hpp>
#include <autoware_perception_msgs/msg/shape.hpp>
#include <autoware_perception_msgs/msg/tracked_object.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using autoware_perception_msgs::msg::DetectedObject;
using autoware_perception_msgs::msg::DetectedObjects;
using autoware_perception_msgs::msg::ObjectClassification;
using autoware_perception_msgs::msg::Shape;
using autoware_perception_msgs::msg::TrackedObject;
using autoware_perception_msgs::msg::TrackedObjects;
using visualization_msgs::msg::Marker;

ObjectClassification classification(const std::uint8_t label, const float probability)
{
  ObjectClassification result;
  result.label = label;
  result.probability = probability;
  return result;
}

DetectedObject detected_object()
{
  DetectedObject result;
  result.existence_probability = 0.83F;
  result.classification = {
    classification(ObjectClassification::UNKNOWN, 0.10F),
    classification(ObjectClassification::CAR, 0.90F)};
  result.kinematics.pose_with_covariance.pose.position.x = 12.0;
  result.kinematics.pose_with_covariance.pose.position.y = -2.0;
  result.kinematics.pose_with_covariance.pose.position.z = 0.7;
  result.kinematics.pose_with_covariance.pose.orientation.z = std::sin(0.3);
  result.kinematics.pose_with_covariance.pose.orientation.w = std::cos(0.3);
  result.shape.type = Shape::BOUNDING_BOX;
  result.shape.dimensions.x = 4.6;
  result.shape.dimensions.y = 1.9;
  result.shape.dimensions.z = 1.6;
  return result;
}

DetectedObjects detections()
{
  DetectedObjects result;
  result.header.frame_id = "lidar_link";
  result.header.stamp.sec = 10;
  result.objects.push_back(detected_object());
  return result;
}

TrackedObjects tracks(const bool moving = true)
{
  TrackedObjects result;
  result.header.frame_id = "odom";
  result.header.stamp.sec = 10;
  TrackedObject object;
  for (std::size_t index = 0; index < object.object_id.uuid.size(); ++index) {
    object.object_id.uuid[index] = static_cast<std::uint8_t>(index + 1U);
  }
  object.existence_probability = 0.91F;
  object.classification = {classification(ObjectClassification::TRUCK, 0.95F)};
  object.kinematics.pose_with_covariance.pose.position.x = 20.0;
  object.kinematics.pose_with_covariance.pose.position.y = 3.0;
  object.kinematics.pose_with_covariance.pose.position.z = 1.0;
  object.kinematics.pose_with_covariance.pose.orientation.w = 1.0;
  object.kinematics.is_stationary = !moving;
  object.kinematics.twist_with_covariance.twist.linear.x = moving ? 4.0 : 0.0;
  object.shape.type = Shape::BOUNDING_BOX;
  object.shape.dimensions.x = 6.0;
  object.shape.dimensions.y = 2.4;
  object.shape.dimensions.z = 2.8;
  result.objects.push_back(object);
  return result;
}

TEST(ObjectMarkerBuilder, DetectionPreservesPoseDimensionsAndAddsClassLabel)
{
  const auto input = detections();
  const auto output = ad_viz::perception::build_detection_markers(input);

  ASSERT_EQ(output.markers.size(), 3U);
  EXPECT_EQ(output.markers[0].action, Marker::DELETEALL);
  const auto & box = output.markers[1];
  EXPECT_EQ(box.header, input.header);
  EXPECT_EQ(box.ns, "detected/0000");
  EXPECT_EQ(box.type, Marker::LINE_LIST);
  EXPECT_EQ(box.pose, input.objects[0].kinematics.pose_with_covariance.pose);
  ASSERT_EQ(box.points.size(), 24U);
  EXPECT_DOUBLE_EQ(box.points[0].x, -2.3);
  EXPECT_DOUBLE_EQ(box.points[0].y, -0.95);
  EXPECT_DOUBLE_EQ(box.points[0].z, -0.8);
  EXPECT_FLOAT_EQ(box.color.r, 1.0F);

  const auto & label = output.markers[2];
  EXPECT_EQ(label.type, Marker::TEXT_VIEW_FACING);
  EXPECT_EQ(label.text, "CAR 0.83");
  EXPECT_DOUBLE_EQ(label.pose.position.z, 1.75);
}

TEST(ObjectMarkerBuilder, TrackUsesUuidAndOnlyPublishesAvailableVelocity)
{
  const auto moving = ad_viz::perception::build_tracked_markers(tracks(true));
  ASSERT_EQ(moving.markers.size(), 4U);
  EXPECT_EQ(moving.markers[1].type, Marker::LINE_LIST);
  EXPECT_EQ(
    moving.markers[1].ns,
    "tracked/0102030405060708090a0b0c0d0e0f10");
  EXPECT_EQ(moving.markers[2].text, "TRUCK 0.91 090a0b0c0d0e0f10");
  EXPECT_EQ(moving.markers[3].type, Marker::ARROW);
  ASSERT_EQ(moving.markers[3].points.size(), 2U);
  EXPECT_DOUBLE_EQ(moving.markers[3].points[1].x, 4.0);

  const auto stationary = ad_viz::perception::build_tracked_markers(tracks(false));
  ASSERT_EQ(stationary.markers.size(), 3U);
  EXPECT_EQ(stationary.markers[0].action, Marker::DELETEALL);
}

TEST(ObjectMarkerBuilder, TrackLabelUsesConfiguredIdPrefix)
{
  ad_viz::perception::ObjectMarkerConfig config;
  config.id_prefix = "B-";
  const auto output = ad_viz::perception::build_tracked_markers(tracks(true), config);
  EXPECT_EQ(output.markers[2].text, "TRUCK 0.91 B-090a0b0c0d0e0f10");
}

TEST(ObjectMarkerBuilder, TrajectoryMarkersSkipShortHistoriesAndUsePrefixedNamespace)
{
  std_msgs::msg::Header header;
  header.frame_id = "odom";
  header.stamp.sec = 10;

  geometry_msgs::msg::Point point_a;
  point_a.x = 1.0;
  geometry_msgs::msg::Point point_b;
  point_b.x = 2.0;

  std::vector<std::pair<std::string, std::vector<geometry_msgs::msg::Point>>> histories{
    {"single", {point_a}},
    {"pair", {point_a, point_b}}};

  ad_viz::perception::ObjectMarkerConfig config;
  config.id_prefix = "B-";
  const auto output = ad_viz::perception::build_trajectory_markers(histories, header, config);

  ASSERT_EQ(output.markers.size(), 1U);
  EXPECT_EQ(output.markers[0].type, Marker::LINE_STRIP);
  EXPECT_EQ(output.markers[0].ns, "trajectory/B-pair");
  ASSERT_EQ(output.markers[0].points.size(), 2U);
}

TEST(ObjectMarkerBuilder, EmptyArraysDeleteAllAndUnsupportedGeometryIsRejected)
{
  DetectedObjects empty;
  empty.header.frame_id = "lidar_link";
  ASSERT_EQ(ad_viz::perception::build_detection_markers(empty).markers.size(), 1U);

  auto unsupported = detections();
  unsupported.objects[0].shape.type = Shape::CYLINDER;
  EXPECT_THROW(
    ad_viz::perception::build_detection_markers(unsupported),
    std::invalid_argument);
}

}  // namespace
