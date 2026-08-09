#include "ad_viz/perception/perception_marker_node.hpp"

#include <rclcpp/rclcpp.hpp>

#include <memory>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    std::make_shared<ad_viz::perception::PerceptionMarkerNode>());
  rclcpp::shutdown();
  return 0;
}
