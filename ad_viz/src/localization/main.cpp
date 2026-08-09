#include "ad_viz/localization/odometry_ground_node.hpp"

#include <rclcpp/rclcpp.hpp>

#include <memory>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    std::make_shared<ad_viz::localization::OdometryGroundNode>());
  rclcpp::shutdown();
  return 0;
}
