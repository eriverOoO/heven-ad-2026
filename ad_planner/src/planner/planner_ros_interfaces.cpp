#include "planner_ros_interfaces.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace ad_planner {

PlannerRosInterfaces::PlannerRosInterfaces(rclcpp::Node &node,
                                           PlannerRosInterfaceConfig config,
                                           PlannerRosCallbacks callbacks)
    : callbacks_(std::move(callbacks)) {
  const auto sensor_qos = rclcpp::SensorDataQoS();
  const auto reliable_qos = rclcpp::QoS(1).reliable();

  command_publisher_ = node.create_publisher<ad_morai_interfaces::msg::CtrlCmd>(
      node.declare_parameter<std::string>("topics.command",
                                          "/ad/control/command"),
      reliable_qos);
  status_publisher_ = node.create_publisher<ad_interfaces::msg::PlannerStatus>(
      node.declare_parameter<std::string>("topics.planner_status",
                                          "/ad/planner/status"),
      reliable_qos);
  tuning_lease_subscription_ = node.create_subscription<std_msgs::msg::Empty>(
      node.declare_parameter<std::string>("topics.tuning_lease",
                                          "/ad/tuning/lease"),
      reliable_qos, [this](std_msgs::msg::Empty::ConstSharedPtr) {
        callbacks_.tuning_lease();
      });

  status_subscription_ =
      node.create_subscription<ad_morai_interfaces::msg::EgoVehicleStatus>(
          node.declare_parameter<std::string>("topics.vehicle_status",
                                              "/ad/vehicle/status"),
          sensor_qos,
          [this](ad_morai_interfaces::msg::EgoVehicleStatus::ConstSharedPtr
                     message) { callbacks_.vehicle_status(*message); });
  odometry_subscription_ = node.create_subscription<nav_msgs::msg::Odometry>(
      node.declare_parameter<std::string>("topics.odometry",
                                          "/ad/localization/odometry"),
      sensor_qos, [this](nav_msgs::msg::Odometry::ConstSharedPtr message) {
        callbacks_.odometry(*message);
      });
  collision_subscription_ =
      node.create_subscription<ad_morai_interfaces::msg::CollisionArray>(
          node.declare_parameter<std::string>("topics.collisions",
                                              "/ad/safety/collisions"),
          sensor_qos,
          [this](ad_morai_interfaces::msg::CollisionArray::ConstSharedPtr
                     message) { callbacks_.collisions(*message); });
  grid_subscription_ = node.create_subscription<nav_msgs::msg::OccupancyGrid>(
      node.declare_parameter<std::string>("topics.occupancy_grid",
                                          "/ad/perception/occupancy_grid"),
      sensor_qos, [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr message) {
        callbacks_.occupancy_grid(*message);
      });
  static_ungated_subscription_ =
      node.create_subscription<nav_msgs::msg::OccupancyGrid>(
          node.declare_parameter<std::string>(
              "visualization.topics.static_ungated",
              "/ad/viz/perception/occupancy/static_ungated"),
          sensor_qos,
          [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr message) {
            callbacks_.static_ungated(*message);
          });
  if (config.road_gate_enabled) {
    drivable_mask_subscription_ =
        node.create_subscription<nav_msgs::msg::OccupancyGrid>(
            node.declare_parameter<std::string>("topics.drivable_mask",
                                                "/ad/planning/drivable_mask"),
            sensor_qos,
            [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr message) {
              callbacks_.drivable_mask(*message);
            });
  }
  predicted_objects_subscription_ =
      node.create_subscription<ad_interfaces::msg::PredictedObjectArray>(
          node.declare_parameter<std::string>(
              "topics.predicted_objects", "/ad/perception/objects/predicted"),
          rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
          [this](ad_interfaces::msg::PredictedObjectArray::ConstSharedPtr
                     message) { callbacks_.predicted_objects(*message); });
  traffic_subscription_ = node.create_subscription<std_msgs::msg::Int8>(
      node.declare_parameter<std::string>("topics.traffic_signal",
                                          "/ad/perception/traffic_signal"),
      sensor_qos, [this](std_msgs::msg::Int8::ConstSharedPtr message) {
        callbacks_.traffic_signal(*message);
      });
  if (config.stop_line_enabled) {
    stop_line_subscription_ = node.create_subscription<std_msgs::msg::Bool>(
        node.declare_parameter<std::string>("topics.stop_line",
                                            "/ad/perception/stop_line"),
        sensor_qos, [this](std_msgs::msg::Bool::ConstSharedPtr message) {
          callbacks_.stop_line(*message);
        });
  }

  if (config.cut_in_response_constraint_enabled) {
    cut_in_response_subscription_ =
        node.create_subscription<ad_interfaces::msg::CutInResponse>(
            node.declare_parameter<std::string>(
                "topics.cut_in_response", "/ad/planning/cut_in_response"),
            rclcpp::QoS(1).reliable(),
            [this](ad_interfaces::msg::CutInResponse::ConstSharedPtr message) {
              callbacks_.cut_in_response(*message);
            });
    cut_in_speed_limit_publisher_ =
        node.create_publisher<std_msgs::msg::Float32>(
            node.declare_parameter<std::string>(
                "topics.cut_in_speed_limit", "/ad/planner/cut_in_speed_limit"),
            reliable_qos);
  }

  if (config.roundabout_response_constraint_enabled) {
    roundabout_response_subscription_ =
        node.create_subscription<ad_interfaces::msg::RoundaboutGapResponse>(
            node.declare_parameter<std::string>(
                "topics.roundabout_gap_response",
                "/ad/planning/roundabout_gap_response"),
            rclcpp::QoS(1).reliable(),
            [this](
                ad_interfaces::msg::RoundaboutGapResponse::ConstSharedPtr
                    message) { callbacks_.roundabout_response(*message); });
    roundabout_speed_limit_publisher_ =
        node.create_publisher<std_msgs::msg::Float32>(
            node.declare_parameter<std::string>(
                "topics.roundabout_speed_limit",
                "/ad/planner/roundabout_speed_limit"),
            reliable_qos);
  }

  if (config.highway_merge_response_integration_enabled) {
    highway_merge_response_subscription_ =
        node.create_subscription<ad_interfaces::msg::HighwayMergeGapResponse>(
            node.declare_parameter<std::string>(
                "topics.highway_merge_gap_response",
                "/ad/planning/highway_merge_gap_response"),
            rclcpp::QoS(1).reliable(),
            [this](ad_interfaces::msg::HighwayMergeGapResponse::ConstSharedPtr
                       message) {
              callbacks_.highway_merge_response(*message);
            });
    highway_merge_speed_limit_publisher_ =
        node.create_publisher<std_msgs::msg::Float32>(
            node.declare_parameter<std::string>(
                "topics.highway_merge_speed_limit",
                "/ad/planner/highway_merge_speed_limit"),
            reliable_qos);
    highway_merge_authorized_publisher_ =
        node.create_publisher<std_msgs::msg::Bool>(
            node.declare_parameter<std::string>(
                "topics.highway_merge_authorized",
                "/ad/planner/highway_merge_authorized"),
            reliable_qos);
  }

  if (config.highway_merge_mission_enabled) {
    highway_merge_mission_state_publisher_ =
        node.create_publisher<std_msgs::msg::UInt8>(
            node.declare_parameter<std::string>(
                "topics.highway_merge_mission_state",
                "/ad/planner/highway_merge_mission_state"),
            reliable_qos);
  }

  if (config.highway_merge_reference_path_enabled) {
    highway_merge_reference_active_publisher_ =
        node.create_publisher<std_msgs::msg::Bool>(
            node.declare_parameter<std::string>(
                "topics.highway_merge_reference_active",
                "/ad/planner/highway_merge_reference_active"),
            reliable_qos);
  }

  tuning_hold_service_ = node.create_service<std_srvs::srv::SetBool>(
      "/ad/planner/hold_control",
      [this](const std_srvs::srv::SetBool::Request::SharedPtr request,
             std_srvs::srv::SetBool::Response::SharedPtr response) {
        const auto result = callbacks_.hold_control(request->data);
        response->success = result.first;
        response->message = result.second;
      });
  reset_controllers_service_ = node.create_service<std_srvs::srv::Trigger>(
      "/ad/planner/reset_path_tracking",
      [this](const std_srvs::srv::Trigger::Request::SharedPtr,
             std_srvs::srv::Trigger::Response::SharedPtr response) {
        const auto result = callbacks_.reset_controllers();
        response->success = result.first;
        response->message = result.second;
      });

  if (config.external_velocity_enabled) {
    const std::string command_topic = node.declare_parameter<std::string>(
        "mppi_nav2.cmd_vel_topic", "/ad/planner/mppi/cmd_vel");
    if (command_topic.empty()) {
      throw std::invalid_argument("mppi_nav2.cmd_vel_topic must not be empty");
    }
    const auto command_qos =
        rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
    external_velocity_subscription_ =
        node.create_subscription<geometry_msgs::msg::Twist>(
            command_topic, command_qos,
            [this](geometry_msgs::msg::Twist::ConstSharedPtr message) {
              callbacks_.external_velocity(*message);
            });
  }
}

void PlannerRosInterfaces::publish_command(
    const ad_morai_interfaces::msg::CtrlCmd &message) {
  command_publisher_->publish(message);
}

void PlannerRosInterfaces::publish_status(
    const ad_interfaces::msg::PlannerStatus &message) {
  status_publisher_->publish(message);
}

void PlannerRosInterfaces::publish_cut_in_speed_limit(float value) {
  if (!cut_in_speed_limit_publisher_) {
    return;
  }
  std_msgs::msg::Float32 message;
  message.data = value;
  cut_in_speed_limit_publisher_->publish(message);
}

void PlannerRosInterfaces::publish_roundabout_speed_limit(float value) {
  if (!roundabout_speed_limit_publisher_) {
    return;
  }
  std_msgs::msg::Float32 message;
  message.data = value;
  roundabout_speed_limit_publisher_->publish(message);
}

void PlannerRosInterfaces::publish_highway_merge_speed_limit(float value) {
  if (!highway_merge_speed_limit_publisher_) {
    return;
  }
  std_msgs::msg::Float32 message;
  message.data = value;
  highway_merge_speed_limit_publisher_->publish(message);
}

void PlannerRosInterfaces::publish_highway_merge_authorized(bool value) {
  if (!highway_merge_authorized_publisher_) {
    return;
  }
  std_msgs::msg::Bool message;
  message.data = value;
  highway_merge_authorized_publisher_->publish(message);
}

void PlannerRosInterfaces::publish_highway_merge_mission_state(
    std::uint8_t value) {
  if (!highway_merge_mission_state_publisher_) {
    return;
  }
  std_msgs::msg::UInt8 message;
  message.data = value;
  highway_merge_mission_state_publisher_->publish(message);
}

void PlannerRosInterfaces::publish_highway_merge_reference_active(bool value) {
  if (!highway_merge_reference_active_publisher_) {
    return;
  }
  std_msgs::msg::Bool message;
  message.data = value;
  highway_merge_reference_active_publisher_->publish(message);
}

} // namespace ad_planner
