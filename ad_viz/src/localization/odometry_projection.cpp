#include "ad_viz/localization/odometry_projection.hpp"

namespace ad_viz::localization
{

nav_msgs::msg::Odometry project_odometry_to_ground(
  const nav_msgs::msg::Odometry & input)
{
  auto output = input;
  output.pose.pose.position.z = 0.0;
  return output;
}

}  // namespace ad_viz::localization
