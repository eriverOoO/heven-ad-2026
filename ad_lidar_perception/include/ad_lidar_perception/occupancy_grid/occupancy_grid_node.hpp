#ifndef AD_LIDAR_PERCEPTION__OCCUPANCY_GRID__OCCUPANCY_GRID_NODE_HPP_
#define AD_LIDAR_PERCEPTION__OCCUPANCY_GRID__OCCUPANCY_GRID_NODE_HPP_

#include <memory>

#include "rclcpp/node.hpp"

namespace ad_lidar_perception::occupancy_grid
{

std::shared_ptr<rclcpp::Node> make_occupancy_grid_node();

}  // namespace ad_lidar_perception::occupancy_grid

#endif  // AD_LIDAR_PERCEPTION__OCCUPANCY_GRID__OCCUPANCY_GRID_NODE_HPP_
