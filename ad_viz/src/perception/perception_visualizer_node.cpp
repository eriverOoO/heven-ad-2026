#include "ad_viz/perception/perception_visualizer_node.hpp"

#include "ad_viz/perception/marker_builder.hpp"

#include <algorithm>
#include <exception>
#include <functional>
#include <string>

namespace ad_viz::perception
{

PerceptionVisualizerNode::PerceptionVisualizerNode(
  const rclcpp::NodeOptions & options)
: Node("perception_visualizer", options),
  config_{
    declare_parameter<double>("marker_lifetime_sec", 0.5),
    declare_parameter<double>("box_line_width_m", 0.06),
    declare_parameter<double>("label_height_m", 0.45),
    declare_parameter<double>("label_z_offset_m", 0.25),
    declare_parameter<double>("velocity_scale_sec", 1.0),
    declare_parameter<double>("minimum_velocity_mps", 0.01)}
{
  const auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
  const auto marker_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
  const std::string detected_input = declare_parameter<std::string>(
    "detected_input_topic", "/ad/perception/objects/detected");
  const std::string tracked_input = declare_parameter<std::string>(
    "tracked_input_topic", "/ad/perception/objects/tracked");
  const std::string predicted_input = declare_parameter<std::string>(
    "predicted_input_topic", "/ad/perception/objects/predicted");
  const std::string detected_output = declare_parameter<std::string>(
    "detected_output_topic", "/ad/visualization/detected_objects");
  const std::string tracked_output = declare_parameter<std::string>(
    "tracked_output_topic", "/ad/visualization/tracked_objects");
  const std::string predicted_output = declare_parameter<std::string>(
    "predicted_output_topic", "/ad/visualization/predicted_objects");
  const bool visualize_predictions = declare_parameter<bool>(
    "visualize_predictions", true);

  detection_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>(
    detected_output, marker_qos);
  tracked_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>(
    tracked_output, marker_qos);
  detection_subscription_ =
    create_subscription<autoware_perception_msgs::msg::DetectedObjects>(
    detected_input, qos,
    std::bind(&PerceptionVisualizerNode::on_detections, this, std::placeholders::_1));
  tracked_subscription_ =
    create_subscription<autoware_perception_msgs::msg::TrackedObjects>(
    tracked_input, qos,
    std::bind(&PerceptionVisualizerNode::on_tracks, this, std::placeholders::_1));

  if (visualize_predictions) {
    prediction_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      predicted_output, marker_qos);
    prediction_subscription_ = create_subscription<ad_interfaces::msg::PredictedObjectArray>(
      predicted_input, qos,
      std::bind(&PerceptionVisualizerNode::on_predictions, this, std::placeholders::_1));
  }
}

void PerceptionVisualizerNode::on_detections(
  const autoware_perception_msgs::msg::DetectedObjects::ConstSharedPtr input)
{
  try {
    detection_publisher_->publish(build_detection_markers(*input, config_));
  } catch (const std::exception & error) {
    RCLCPP_WARN(get_logger(), "Rejected detected objects for visualization: %s", error.what());
  }
}

void PerceptionVisualizerNode::on_tracks(
  const autoware_perception_msgs::msg::TrackedObjects::ConstSharedPtr input)
{
  try {
    tracked_publisher_->publish(build_tracked_markers(*input, config_));
  } catch (const std::exception & error) {
    RCLCPP_WARN(get_logger(), "Rejected tracked objects for visualization: %s", error.what());
  }
}

void PerceptionVisualizerNode::on_predictions(
  const ad_interfaces::msg::PredictedObjectArray::ConstSharedPtr input)
{
  try {
    MarkerBuilderConfig prediction_config;
    prediction_config.marker_lifetime_sec = config_.marker_lifetime_sec;
    prediction_config.stale_timeout_sec = std::max(1.0, config_.marker_lifetime_sec * 2.0);
    prediction_publisher_->publish(build_prediction_markers(*input, prediction_config));
  } catch (const std::exception & error) {
    RCLCPP_WARN(get_logger(), "Rejected predicted objects for visualization: %s", error.what());
  }
}

}  // namespace ad_viz::perception
