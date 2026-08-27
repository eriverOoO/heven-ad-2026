#include "ad_viz/perception/perception_visualizer_node.hpp"

#include <rclcpp/rclcpp.hpp>

#include <memory>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ad_viz::perception::PerceptionVisualizerNode>());
  rclcpp::shutdown();
  return 0;
}
