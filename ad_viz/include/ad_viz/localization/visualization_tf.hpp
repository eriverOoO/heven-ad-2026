#pragma once

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include <optional>
#include <string>

namespace ad_viz::localization
{

tf2_msgs::msg::TFMessage without_transform_edge(
  const tf2_msgs::msg::TFMessage & input,
  const std::string & parent_frame,
  const std::string & child_frame);

std::optional<geometry_msgs::msg::TransformStamped>
visualization_transform_from_odometry(
  const nav_msgs::msg::Odometry & odometry,
  const std::string & expected_parent_frame,
  const std::string & expected_child_frame);

}  // namespace ad_viz::localization
