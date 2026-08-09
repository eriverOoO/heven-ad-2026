#include "ad_localization/manager/localization_manager_node.hpp"

#include <functional>
#include <memory>
#include <string>

#include "ad_localization/manager/localization_manager.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_ros/transform_broadcaster.h"

namespace ad_localization
{
namespace
{

class LocalizationManagerNode final : public rclcpp::Node
{
public:
  explicit LocalizationManagerNode(const rclcpp::NodeOptions & options)
  : rclcpp::Node("localization_manager", options)
  {
    declare_parameter<std::string>(
      "input_odometry_topic", "/ad/localization/backends/gnss_imu/odometry");
    declare_parameter<std::string>(
      "canonical_odometry_topic", "/ad/localization/odometry");
    declare_parameter<std::string>("map_frame", "map");
    declare_parameter<std::string>("odom_frame", "odom");
    declare_parameter<std::string>("base_frame", "base_link");
    declare_parameter<bool>("publish_tf", true);

    config_ = {
      get_parameter("map_frame").as_string(),
      get_parameter("odom_frame").as_string(),
      get_parameter("base_frame").as_string()};
    manager_ = std::make_unique<LocalizationManager>(config_);
    const auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
    publisher_ = create_publisher<nav_msgs::msg::Odometry>(
      get_parameter("canonical_odometry_topic").as_string(), qos);
    publish_tf_ = get_parameter("publish_tf").as_bool();
    if (publish_tf_) {
      transform_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
      static_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
      const builtin_interfaces::msg::Time static_stamp = now();
      static_broadcaster_->sendTransform(map_to_odom_transform(config_, static_stamp));
    }
    subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      get_parameter("input_odometry_topic").as_string(), qos,
      std::bind(&LocalizationManagerNode::candidate_callback, this, std::placeholders::_1));
    reset_service_ = create_service<std_srvs::srv::Trigger>(
      "/ad/localization/reset_manager",
      [this](
        const std_srvs::srv::Trigger::Request::SharedPtr,
        std_srvs::srv::Trigger::Response::SharedPtr response)
      {
        manager_->reset();
        response->success = true;
        response->message = "localization manager timestamp epoch reset";
      });
  }

private:
  void candidate_callback(const nav_msgs::msg::Odometry::SharedPtr candidate)
  {
    const auto accepted = manager_->accept(*candidate);
    if (!accepted) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "dropping invalid, duplicate, or regressing backend odometry");
      return;
    }
    publisher_->publish(*accepted);
    if (publish_tf_ && transform_broadcaster_) {
      transform_broadcaster_->sendTransform(odometry_transform(*accepted));
    }
  }

  LocalizationManagerConfig config_;
  std::unique_ptr<LocalizationManager> manager_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr publisher_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subscription_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> transform_broadcaster_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
  bool publish_tf_{true};
};

}  // namespace

std::shared_ptr<rclcpp::Node> make_localization_manager_node(
  const rclcpp::NodeOptions & options)
{
  return std::make_shared<LocalizationManagerNode>(options);
}

}  // namespace ad_localization
