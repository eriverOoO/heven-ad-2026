#pragma once

#include "ad_viz/perception/object_marker_builder.hpp"
#include "ad_viz/perception/trajectory_history.hpp"

#include <ad_interfaces/msg/predicted_object_array.hpp>
#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <memory>

namespace ad_viz::perception
{

class PerceptionVisualizerNode : public rclcpp::Node
{
public:
  explicit PerceptionVisualizerNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void on_detections(
    const autoware_perception_msgs::msg::DetectedObjects::ConstSharedPtr input);
  void on_tracks(
    const autoware_perception_msgs::msg::TrackedObjects::ConstSharedPtr input);
  void on_predictions(
    const ad_interfaces::msg::PredictedObjectArray::ConstSharedPtr input);

  ObjectMarkerConfig config_;
  std::unique_ptr<TrajectoryHistory> trajectory_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr detection_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr tracked_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr prediction_publisher_;
  rclcpp::Subscription<autoware_perception_msgs::msg::DetectedObjects>::SharedPtr
    detection_subscription_;
  rclcpp::Subscription<autoware_perception_msgs::msg::TrackedObjects>::SharedPtr
    tracked_subscription_;
  rclcpp::Subscription<ad_interfaces::msg::PredictedObjectArray>::SharedPtr
    prediction_subscription_;
};

}  // namespace ad_viz::perception
