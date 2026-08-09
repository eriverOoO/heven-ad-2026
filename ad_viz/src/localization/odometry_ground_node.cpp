#include "ad_viz/localization/odometry_ground_node.hpp"

#include <stdexcept>
#include <string>

#include "ad_viz/localization/odometry_projection.hpp"

namespace ad_viz::localization
{

OdometryGroundNode::OdometryGroundNode(const rclcpp::NodeOptions & options)
: Node("ad_localization_ground_odometry", options)
{
  const std::string input_topic = declare_parameter<std::string>(
    "input_odometry_topic", "/ad/localization/odometry");
  const std::string output_topic = declare_parameter<std::string>(
    "output_odometry_topic", "/ad/viz/localization/odometry_ground");
  if (input_topic.empty() || output_topic.empty() ||
    get_node_topics_interface()->resolve_topic_name(input_topic) ==
    get_node_topics_interface()->resolve_topic_name(output_topic))
  {
    throw std::invalid_argument(
            "odometry input and ground visualization output topics must be nonempty and distinct");
  }

  const auto qos =
    rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
  publisher_ = create_publisher<nav_msgs::msg::Odometry>(output_topic, qos);
  subscription_ = create_subscription<nav_msgs::msg::Odometry>(
    input_topic, qos,
    [this](const nav_msgs::msg::Odometry::ConstSharedPtr input) {
      publisher_->publish(project_odometry_to_ground(*input));
    });
}

}  // namespace ad_viz::localization
