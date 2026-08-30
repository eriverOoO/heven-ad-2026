#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "cut_in_response_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    std::make_shared<ad_planner::CutInResponseNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
