#ifndef AD_LIDAR_PERCEPTION__PREPROCESSING__FINITE_POINT_FILTER_NODE_HPP_
#define AD_LIDAR_PERCEPTION__PREPROCESSING__FINITE_POINT_FILTER_NODE_HPP_

#include <memory>

#include <rclcpp/node.hpp>

namespace ad_lidar_perception::preprocessing
{

std::shared_ptr<rclcpp::Node> make_finite_point_filter_node();

}  // namespace ad_lidar_perception::preprocessing

#endif  // AD_LIDAR_PERCEPTION__PREPROCESSING__FINITE_POINT_FILTER_NODE_HPP_
