#include "ad_viz/localization/odometry_route_elevation_node.hpp"

#include <rclcpp/rclcpp.hpp>

#include <memory>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    std::make_shared<ad_viz::localization::OdometryRouteElevationNode>());
  rclcpp::shutdown();
  return 0;
}
