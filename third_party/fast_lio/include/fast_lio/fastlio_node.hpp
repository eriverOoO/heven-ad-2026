// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <rclcpp/rclcpp.hpp>

namespace fast_lio {
rclcpp::Node::SharedPtr make_fastlio_node(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
}  // namespace fast_lio
