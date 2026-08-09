#include <chrono>
#include <csignal>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "mppi_follow_path_proxy.hpp"

namespace
{

volatile std::sig_atomic_t shutdown_requested = 0;

extern "C" void request_shutdown(const int)
{
  shutdown_requested = 1;
}

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(
    argc, argv, rclcpp::InitOptions(),
    rclcpp::SignalHandlerOptions::None);
  static_cast<void>(std::signal(SIGINT, request_shutdown));
  static_cast<void>(std::signal(SIGTERM, request_shutdown));

  const auto node =
    std::make_shared<ad_planner::MppiFollowPathProxy>();
  rclcpp::executors::SingleThreadedExecutor executor;
  const auto ros_node = std::static_pointer_cast<rclcpp::Node>(node);
  executor.add_node(ros_node);
  while (rclcpp::ok() && shutdown_requested == 0) {
    executor.spin_once(std::chrono::milliseconds(50));
  }

  node->deactivate();
  const auto cancel_drain_deadline =
    std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
  while (rclcpp::ok() &&
    std::chrono::steady_clock::now() < cancel_drain_deadline)
  {
    executor.spin_once(std::chrono::milliseconds(10));
  }
  executor.remove_node(ros_node);
  rclcpp::shutdown();
  return 0;
}
