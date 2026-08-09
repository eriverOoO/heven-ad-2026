#ifndef AD_PLANNER__PLANNER__PLANNER_ROS_INTERFACES_HPP_
#define AD_PLANNER__PLANNER__PLANNER_ROS_INTERFACES_HPP_

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include <ad_interfaces/msg/planner_status.hpp>
#include <ad_interfaces/msg/predicted_object_array.hpp>
#include <ad_morai_interfaces/msg/collision_array.hpp>
#include <ad_morai_interfaces/msg/ctrl_cmd.hpp>
#include <ad_morai_interfaces/msg/ego_vehicle_status.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/node.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/int8.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace ad_planner {

struct PlannerRosInterfaceConfig {
  bool road_gate_enabled{true};
  bool stop_line_enabled{false};
  bool external_velocity_enabled{false};
};

struct PlannerRosCallbacks {
  std::function<void(const ad_morai_interfaces::msg::EgoVehicleStatus &)>
      vehicle_status;
  std::function<void(const nav_msgs::msg::Odometry &)> odometry;
  std::function<void(const ad_morai_interfaces::msg::CollisionArray &)>
      collisions;
  std::function<void(const nav_msgs::msg::OccupancyGrid &)> occupancy_grid;
  std::function<void(const nav_msgs::msg::OccupancyGrid &)>
      static_ungated;
  std::function<void(const nav_msgs::msg::OccupancyGrid &)> drivable_mask;
  std::function<void(const ad_interfaces::msg::PredictedObjectArray &)>
      predicted_objects;
  std::function<void(const std_msgs::msg::Int8 &)> traffic_signal;
  std::function<void(const std_msgs::msg::Bool &)> stop_line;
  std::function<void()> tuning_lease;
  std::function<std::pair<bool, std::string>(bool)> hold_control;
  std::function<std::pair<bool, std::string>()> reset_controllers;
  std::function<void(const geometry_msgs::msg::Twist &)> external_velocity;
};

class PlannerRosInterfaces {
public:
  PlannerRosInterfaces(rclcpp::Node &node, PlannerRosInterfaceConfig config,
                       PlannerRosCallbacks callbacks);

  void publish_command(const ad_morai_interfaces::msg::CtrlCmd &message);

  void publish_status(const ad_interfaces::msg::PlannerStatus &message);

private:
  PlannerRosCallbacks callbacks_;

  rclcpp::Publisher<ad_morai_interfaces::msg::CtrlCmd>::SharedPtr
      command_publisher_;
  rclcpp::Publisher<ad_interfaces::msg::PlannerStatus>::SharedPtr
      status_publisher_;
  rclcpp::Subscription<ad_morai_interfaces::msg::EgoVehicleStatus>::SharedPtr
      status_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
      odometry_subscription_;
  rclcpp::Subscription<ad_morai_interfaces::msg::CollisionArray>::SharedPtr
      collision_subscription_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr
      grid_subscription_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr
      static_ungated_subscription_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr
      drivable_mask_subscription_;
  rclcpp::Subscription<ad_interfaces::msg::PredictedObjectArray>::SharedPtr
      predicted_objects_subscription_;
  rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr traffic_subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr stop_line_subscription_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr
      tuning_lease_subscription_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr tuning_hold_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_controllers_service_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr
      external_velocity_subscription_;
};

} // namespace ad_planner

#endif // AD_PLANNER__PLANNER__PLANNER_ROS_INTERFACES_HPP_
