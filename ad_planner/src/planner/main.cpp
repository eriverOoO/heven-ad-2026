#include "ad_planner/planner/planner_node.hpp"

#include <exception>

#include "rclcpp/rclcpp.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(ad_planner::make_planner_node());
  } catch (const std::exception & error) {
    RCLCPP_FATAL(rclcpp::get_logger("ad_planner"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
