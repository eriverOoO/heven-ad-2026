#include "highway_merge_gap_risk_node.hpp"

#include <memory>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    std::make_shared<ad_planner::HighwayMergeGapRiskNode>(rclcpp::NodeOptions{}));
  rclcpp::shutdown();
  return 0;
}
