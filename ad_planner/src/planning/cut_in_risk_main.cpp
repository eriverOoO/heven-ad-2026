#include "cut_in_risk_node.hpp"

#include <memory>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ad_planner::CutInRiskNode>(rclcpp::NodeOptions{}));
  rclcpp::shutdown();
  return 0;
}
