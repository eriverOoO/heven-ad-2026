#pragma once

#include <nav_msgs/msg/odometry.hpp>

namespace ad_viz::localization
{

nav_msgs::msg::Odometry project_odometry_to_ground(
  const nav_msgs::msg::Odometry & input);

}  // namespace ad_viz::localization
