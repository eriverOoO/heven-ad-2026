#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>

#include <ad_interfaces/msg/planner_status.hpp>
#include <ad_interfaces/msg/predicted_object.hpp>
#include <ad_interfaces/msg/predicted_object_array.hpp>
#include <ad_interfaces/msg/predicted_state.hpp>
#include <ad_morai_interfaces/msg/collision_array.hpp>
#include <ad_morai_interfaces/msg/ctrl_cmd.hpp>
#include <ad_morai_interfaces/msg/ego_vehicle_status.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/int8.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2/exceptions.h>
#include <tf2/utils.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "ad_control/command/curvature_command_adapter.hpp"
#include "ad_control/lateral/path_tracking_controller_factory.hpp"
#include "ad_planner/behavior/bt_nodes.hpp"
#include "ad_planner/behavior/planner_supervisor.hpp"
#include "ad_planner/common/exact_stamp_pairer.hpp"
#include "ad_planner/common/parameter_validation.hpp"
#include "ad_planner/common/vehicle_observation.hpp"
#include "ad_planner/io/data_loader.hpp"
#include "ad_planner/io/route_corridor_loader.hpp"
#include "ad_planner/local_planning/common/future_road_risk.hpp"
#include "ad_planner/local_planning/common/local_motion_frame.hpp"
#include "ad_planner/local_planning/common/local_motion_runtime.hpp"
#include "ad_planner/local_planning/common/local_motion_timing.hpp"
#include "ad_planner/local_planning/common/local_motion_validation.hpp"
#include "ad_planner/local_planning/common/occupancy.hpp"
#include "ad_planner/local_planning/common/prediction_admission.hpp"
#include "ad_planner/local_planning/common/road_corridor_grid.hpp"
#include "ad_planner/local_planning/local_motion_factory.hpp"
#include "ad_planner/planner/planner_node.hpp"
#include "ad_planner/visualization/path_tracking_markers.hpp"
#include "ad_planner/visualization/planner_visualization.hpp"
#include "ad_planner/visualization/route_markers.hpp"
#include "planner_ros_interfaces.hpp"

namespace ad_planner {

using ad_control::make_path_tracking_controller;
using ad_control::parse_path_tracking_backend;
using ad_control::PathTrackingBackend;
using ad_control::PathTrackingController;
using ad_control::PathTrackingParameterProvider;
using ad_control::PidConfig;

namespace {

std::string read_text(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open behavior tree: " + path.string());
  }
  std::ostringstream text;
  text << input.rdbuf();
  return text.str();
}

bool finite_quaternion(const geometry_msgs::msg::Quaternion &quaternion) {
  return std::isfinite(quaternion.x) && std::isfinite(quaternion.y) &&
         std::isfinite(quaternion.z) && std::isfinite(quaternion.w);
}

bool valid_sha256(const std::string &digest) {
  return digest.size() == 64U &&
         std::all_of(digest.begin(), digest.end(), [](const char value) {
           return (value >= '0' && value <= '9') ||
                  (value >= 'a' && value <= 'f');
         });
}

bool same_grid_geometry(const OccupancyGrid &lhs, const OccupancyGrid &rhs) {
  return lhs.origin == rhs.origin && lhs.resolution == rhs.resolution &&
         lhs.width == rhs.width && lhs.height == rhs.height;
}

double
prediction_duration_seconds(const builtin_interfaces::msg::Duration &duration) {
  if (duration.sec < 0 || duration.nanosec >= 1'000'000'000U) {
    throw std::invalid_argument(
        "prediction duration must be nonnegative and normalized");
  }
  return static_cast<double>(duration.sec) +
         static_cast<double>(duration.nanosec) * 1.0e-9;
}

std::string
prediction_object_id(const unique_identifier_msgs::msg::UUID &uuid) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const std::uint8_t byte : uuid.uuid) {
    output << std::setw(2) << static_cast<unsigned int>(byte);
  }
  return output.str();
}

PredictedFootprint
make_predicted_footprint(const geometry_msgs::msg::PoseWithCovariance &pose,
                         const geometry_msgs::msg::Vector3 &dimensions,
                         const double time_from_start_s) {
  if (!std::isfinite(time_from_start_s) || !std::isfinite(dimensions.x) ||
      !std::isfinite(dimensions.y) || dimensions.x <= 0.0 ||
      dimensions.y <= 0.0 || !std::isfinite(pose.pose.position.x) ||
      !std::isfinite(pose.pose.position.y) ||
      !finite_quaternion(pose.pose.orientation) ||
      !std::isfinite(pose.covariance[0]) ||
      !std::isfinite(pose.covariance[1]) ||
      !std::isfinite(pose.covariance[6]) ||
      !std::isfinite(pose.covariance[7]) || pose.covariance[0] < 0.0 ||
      pose.covariance[7] < 0.0) {
    throw std::invalid_argument("predicted footprint is malformed");
  }
  const double covariance_scale = std::max(
      {1.0, std::abs(pose.covariance[1]), std::abs(pose.covariance[6])});
  if (std::abs(pose.covariance[1] - pose.covariance[6]) >
      1.0e-9 * covariance_scale) {
    throw std::invalid_argument(
        "predicted footprint covariance must be symmetric");
  }
  const double covariance_xy =
      0.5 * pose.covariance[1] + 0.5 * pose.covariance[6];
  const double covariance_midpoint =
      0.5 * pose.covariance[0] + 0.5 * pose.covariance[7];
  const double covariance_radius = std::hypot(
      0.5 * pose.covariance[0] - 0.5 * pose.covariance[7], covariance_xy);
  if (!std::isfinite(covariance_midpoint) ||
      !std::isfinite(covariance_radius) ||
      covariance_midpoint - covariance_radius < -1.0e-9 * covariance_scale) {
    throw std::invalid_argument(
        "predicted footprint covariance must be positive semidefinite");
  }
  const double quaternion_norm_squared =
      pose.pose.orientation.x * pose.pose.orientation.x +
      pose.pose.orientation.y * pose.pose.orientation.y +
      pose.pose.orientation.z * pose.pose.orientation.z +
      pose.pose.orientation.w * pose.pose.orientation.w;
  if (!std::isfinite(quaternion_norm_squared) ||
      std::abs(quaternion_norm_squared - 1.0) > 1.0e-6) {
    throw std::invalid_argument(
        "predicted footprint quaternion must be normalized");
  }
  const double yaw = tf2::getYaw(pose.pose.orientation);
  if (!std::isfinite(yaw)) {
    throw std::invalid_argument("predicted footprint yaw is invalid");
  }
  return PredictedFootprint{
      time_from_start_s,
      Pose2{pose.pose.position.x, pose.pose.position.y, yaw},
      dimensions.x,
      dimensions.y,
      pose.covariance[0],
      pose.covariance[7],
      covariance_xy};
}

PredictedObjectSet adapt_predicted_objects(
    const ad_interfaces::msg::PredictedObjectArray &message) {
  PredictedObjectSet output;
  output.reserve(message.objects.size());
  for (const auto &input : message.objects) {
    if (!std::isfinite(input.existence_probability) ||
        input.existence_probability < 0.0F ||
        input.existence_probability > 1.0F || input.states.empty()) {
      throw std::invalid_argument(
          "predicted object probability or horizon count is invalid");
    }
    PredictedObject object;
    object.object_id = prediction_object_id(input.object_id);
    object.footprints.reserve(input.states.size() + 1U);
    object.footprints.push_back(
        make_predicted_footprint(input.initial_pose, input.dimensions, 0.0));
    double previous_time_s = 0.0;
    for (const auto &state : input.states) {
      const double time_s = prediction_duration_seconds(state.time_from_start);
      if (!(time_s > previous_time_s)) {
        throw std::invalid_argument(
            "prediction horizons must be strictly increasing and positive");
      }
      object.footprints.push_back(
          make_predicted_footprint(state.pose, input.dimensions, time_s));
      previous_time_s = time_s;
    }
    output.push_back(std::move(object));
  }
  return output;
}

PredictedObjectSet age_predictions(const PredictedObjectSet &predictions,
                                   const double age_s) {
  if (!std::isfinite(age_s)) {
    throw std::invalid_argument("prediction age must be finite");
  }
  PredictedObjectSet output = predictions;
  for (auto &object : output) {
    for (auto &footprint : object.footprints) {
      footprint.time_from_start_s -= age_s;
    }
  }
  return output;
}

nav_msgs::msg::OccupancyGrid make_ros_grid(const OccupancyGrid &grid,
                                           const std::string &frame_id,
                                           const rclcpp::Time &stamp) {
  nav_msgs::msg::OccupancyGrid message;
  message.header.frame_id = frame_id;
  message.header.stamp = stamp;
  message.info.resolution = static_cast<float>(grid.resolution);
  message.info.width = static_cast<std::uint32_t>(grid.width);
  message.info.height = static_cast<std::uint32_t>(grid.height);
  message.info.origin.position.x = grid.origin.x;
  message.info.origin.position.y = grid.origin.y;
  message.info.origin.orientation.z = std::sin(grid.origin.yaw_rad * 0.5);
  message.info.origin.orientation.w = std::cos(grid.origin.yaw_rad * 0.5);
  message.data = grid.cells;
  return message;
}

class RosControllerParameterProvider final
    : public LocalMotionParameterProvider,
      public PathTrackingParameterProvider {
public:
  explicit RosControllerParameterProvider(rclcpp::Node &node) : node_(node) {}

  double get_double(const std::string &name, double default_value) override {
    if (node_.has_parameter(name)) {
      return node_.get_parameter(name).as_double();
    }
    return node_.declare_parameter<double>(name, default_value);
  }

  int get_int(const std::string &name, int default_value) override {
    if (node_.has_parameter(name)) {
      return static_cast<int>(node_.get_parameter(name).as_int());
    }
    return node_.declare_parameter<int>(name, default_value);
  }

  std::vector<double>
  get_double_array(const std::string &name,
                   const std::vector<double> &default_value) override {
    if (node_.has_parameter(name)) {
      return node_.get_parameter(name).as_double_array();
    }
    return node_.declare_parameter<std::vector<double>>(name, default_value);
  }

private:
  rclcpp::Node &node_;
};

struct TimedOccupancyGridMessage {
  nav_msgs::msg::OccupancyGrid message;
  double receipt_time_s{0.0};
};

struct TimedVisualizationGrid {
  OccupancyGrid grid;
  std::string frame_id;
  double receipt_time_s{0.0};
};

TimedVisualizationGrid
adapt_visualization_grid(const nav_msgs::msg::OccupancyGrid &message,
                         const double receipt_time_s) {
  if (message.header.frame_id.empty() || !std::isfinite(receipt_time_s) ||
      !std::isfinite(message.info.origin.position.x) ||
      !std::isfinite(message.info.origin.position.y) ||
      message.info.origin.position.z != 0.0 ||
      !finite_quaternion(message.info.origin.orientation)) {
    throw std::invalid_argument(
        "ungated visualization grid has invalid metadata");
  }
  const auto &orientation = message.info.origin.orientation;
  const double quaternion_norm =
      std::hypot(std::hypot(orientation.x, orientation.y),
                 std::hypot(orientation.z, orientation.w));
  if (!std::isfinite(quaternion_norm) ||
      std::abs(quaternion_norm - 1.0) > 1.0e-6) {
    throw std::invalid_argument(
        "ungated visualization grid quaternion must be normalized");
  }

  OccupancyGrid grid;
  grid.origin =
      Pose2{message.info.origin.position.x, message.info.origin.position.y,
            tf2::getYaw(message.info.origin.orientation)};
  grid.resolution = message.info.resolution;
  grid.width = message.info.width;
  grid.height = message.info.height;
  grid.cells = message.data;
  grid.valid = true;
  grid.fresh = true;
  if (!validate_occupancy_grid(grid).valid ||
      !std::all_of(grid.cells.begin(), grid.cells.end(),
                   [](const std::int8_t value) {
                     return value >= -1 && value <= 100;
                   })) {
    throw std::invalid_argument(
        "ungated visualization grid has invalid geometry or cells");
  }
  return TimedVisualizationGrid{std::move(grid), message.header.frame_id,
                                receipt_time_s};
}

} // namespace

class AdPlannerNode final : public rclcpp::Node {
public:
  AdPlannerNode() : Node("ad_planner"), steady_clock_(RCL_STEADY_TIME) {
    local_motion_backend_kind_ = parse_local_motion_backend(
        declare_parameter<std::string>("local_motion.backend", "dwa"));
    path_tracking_backend_name_ =
        declare_parameter<std::string>("path_tracking.backend", "stanley");
    path_tracking_backend_ =
        parse_path_tracking_backend(path_tracking_backend_name_);
    const std::string data_dir =
        resolve_data_dir(declare_parameter<std::string>("data_dir", ""));
    const auto route_path =
        std::filesystem::path(data_dir) /
        declare_parameter<std::string>("path_file",
                                       "path/2026_molit_comp_global_path.txt");
    route_ = DataLoader::load_path(route_path);

    control_period_s_ = positive_finite_parameter(
        declare_parameter<double>("control_period_sec", 0.05),
        "control_period_sec");
    tuning_lease_required_ =
        declare_parameter<bool>("tuning.lease_required", false);
    tuning_lease_timeout_s_ = positive_finite_parameter(
        declare_parameter<double>("tuning.lease_timeout_sec", 1.0),
        "tuning.lease_timeout_sec");

    const PidConfig pid{
        declare_parameter<double>("speed_pid.kp", 0.3),
        declare_parameter<double>("speed_pid.ki", 0.0),
        declare_parameter<double>("speed_pid.kd", 0.01),
        declare_parameter<double>("speed_pid.integral_limit", 10.0),
        declare_parameter<double>("speed_pid.derivative_limit", 10.0)};
    const double steering_limit = declare_parameter<double>(
        "maximum_steering_rad", kMaximumAllowedSteeringRad);
    RosControllerParameterProvider controller_parameters(*this);
    path_tracking_ = make_path_tracking_controller(
        path_tracking_backend_, route_, controller_parameters);
    visualization_profile_sample_stride_ = positive_size_parameter(
        declare_parameter<int>("visualization.profile_sample_stride", 10),
        "visualization.profile_sample_stride");
    route_relevance_period_s_ = positive_finite_parameter(
        declare_parameter<double>("visualization.route_relevance_period_sec",
                                  0.20),
        "visualization.route_relevance_period_sec");
    route_profile_markers_ = make_route_profile_markers(
        route_, path_tracking_->route_speed_profile(), "map", now(),
        visualization_profile_sample_stride_);
    configure_local_motion(data_dir, controller_parameters, pid,
                           steering_limit);

    planner_config_ = load_planner_config(steering_limit);
    context_.inputs.route_ready = true;
    context_.callbacks.follow_global_path = [this]() {
      return run_path_tracking();
    };
    context_.callbacks.perception_local_planner = [this]() {
      return run_local_planner();
    };

    const auto tree_path =
        std::filesystem::path(
            ament_index_cpp::get_package_share_directory("ad_planner")) /
        "behavior_trees" / "ad_planner.xml";
    supervisor_ = std::make_unique<PlannerSupervisor>(context_, planner_config_,
                                                      read_text(tree_path));

    create_ros_interfaces();
    publish_global_path();
    visualization_->publish_route_profile(route_profile_markers_, now());
    timer_ = create_wall_timer(std::chrono::duration<double>(control_period_s_),
                               [this]() { tick(); });
    if (local_motion_backend_kind_ == LocalMotionBackendKind::kMppiNav2) {
      global_path_timer_ = create_wall_timer(
          std::chrono::duration<double>(mppi_path_refresh_period_s_),
          [this]() { publish_global_path(); });
    }

    RCLCPP_INFO(get_logger(), "loaded %zu route points from %s",
                route_.points.size(), route_path.c_str());
  }

private:
  std::string resolve_data_dir(const std::string &parameter) const {
    if (!parameter.empty()) {
      return parameter;
    }
    const char *environment = std::getenv("AD_DATA_DIR");
    if (environment && *environment != '\0') {
      return environment;
    }
    throw std::runtime_error("set data_dir or AD_DATA_DIR");
  }

  void
  configure_local_motion(const std::string &data_dir,
                         RosControllerParameterProvider &controller_parameters,
                         const PidConfig &pid,
                         const double output_maximum_steering_rad) {
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    try {
      odom_frame_ =
          declare_parameter<std::string>("local_motion.odom_frame", "odom");
      map_frame_ =
          declare_parameter<std::string>("local_motion.map_frame", "map");
      if (odom_frame_.empty() || map_frame_.empty()) {
        throw std::invalid_argument(
            "local motion frame names must not be empty");
      }
      local_motion_timing_limits_ = LocalMotionTimingLimits{
          positive_finite_parameter(
              declare_parameter<double>("local_motion.max_odometry_age_sec",
                                        0.50),
              "local_motion.max_odometry_age_sec"),
          positive_finite_parameter(
              declare_parameter<double>("local_motion.max_grid_age_sec", 0.50),
              "local_motion.max_grid_age_sec"),
          positive_finite_parameter(
              declare_parameter<double>("local_motion.max_grid_odom_skew_sec",
                                        0.10),
              "local_motion.max_grid_odom_skew_sec")};
      corridor_window_behind_m_ = positive_finite_parameter(
          declare_parameter<double>("local_motion.corridor_window.behind_m",
                                    10.0),
          "local_motion.corridor_window.behind_m");
      corridor_window_ahead_m_ = positive_finite_parameter(
          declare_parameter<double>("local_motion.corridor_window.ahead_m",
                                    100.0),
          "local_motion.corridor_window.ahead_m");
      road_gate_enabled_ = declare_parameter<bool>("road_gate.enabled", true);
      prediction_mode_ = parse_prediction_mode(declare_parameter<std::string>(
          "local_motion.prediction.mode", "disabled"));
      prediction_timeout_s_ = positive_finite_parameter(
          declare_parameter<double>("local_motion.prediction_timeout_sec",
                                    0.50),
          "local_motion.prediction_timeout_sec");

      const double wheelbase_m = positive_finite_parameter(
          declare_parameter<double>("local_motion.wheelbase_m", 3.0),
          "local_motion.wheelbase_m");
      const double maximum_steering_rad = positive_finite_parameter(
          declare_parameter<double>("local_motion.maximum_steering_rad",
                                    kMaximumAllowedSteeringRad),
          "local_motion.maximum_steering_rad");
      const double maximum_steering_rate_radps = positive_finite_parameter(
          declare_parameter<double>("local_motion.maximum_steering_rate_radps",
                                    2.0943951023931953),
          "local_motion.maximum_steering_rate_radps");
      vehicle_constraints_ = VehicleConstraints{
          wheelbase_m,
          maximum_steering_rad,
          positive_finite_parameter(
              declare_parameter<double>("local_motion.maximum_speed_mps",
                                        16.25),
              "local_motion.maximum_speed_mps"),
          positive_finite_parameter(
              declare_parameter<double>(
                  "local_motion.maximum_acceleration_mps2", 5.0),
              "local_motion.maximum_acceleration_mps2"),
          positive_finite_parameter(
              declare_parameter<double>(
                  "local_motion.maximum_deceleration_mps2", 6.0),
              "local_motion.maximum_deceleration_mps2"),
          positive_finite_parameter(
              declare_parameter<double>(
                  "local_motion.maximum_lateral_acceleration_mps2", 6.0),
              "local_motion.maximum_lateral_acceleration_mps2"),
          positive_finite_parameter(
              declare_parameter<double>("local_motion.maximum_jerk_mps3", 5.0),
              "local_motion.maximum_jerk_mps3"),
          positive_finite_parameter(
              declare_parameter<double>("local_motion.footprint_front_m",
                                        3.845),
              "local_motion.footprint_front_m"),
          positive_finite_parameter(
              declare_parameter<double>("local_motion.footprint_rear_m", 0.79),
              "local_motion.footprint_rear_m"),
          positive_finite_parameter(
              declare_parameter<double>("local_motion.footprint_half_width_m",
                                        0.945),
              "local_motion.footprint_half_width_m")};
      auto curvature_adapter =
          std::make_unique<ad_control::CurvatureCommandAdapter>(
              ad_control::CurvatureCommandAdapterConfig{
                  pid, wheelbase_m, maximum_steering_rad,
                  maximum_steering_rate_radps});

      auto local_motion_backend = make_local_motion_backend(
          local_motion_backend_kind_, controller_parameters);
      local_motion_runtime_ = std::make_unique<LocalMotionRuntime>(
          std::move(local_motion_backend), std::move(curvature_adapter),
          LocalMotionRuntimeConfig{maximum_steering_rad,
                                   output_maximum_steering_rad});
      if (local_motion_backend_kind_ == LocalMotionBackendKind::kMppiNav2) {
        mppi_path_refresh_period_s_ = positive_finite_parameter(
            declare_parameter<double>("mppi_nav2.path_refresh_period_s", 0.25),
            "mppi_nav2.path_refresh_period_s");
      }

      std::filesystem::path corridor_path = declare_parameter<std::string>(
          "route_corridor_file", "map/route_corridor.json");
      if (corridor_path.empty()) {
        throw std::invalid_argument("route_corridor_file must not be empty");
      }
      if (corridor_path.is_relative()) {
        corridor_path = std::filesystem::path(data_dir) / corridor_path;
      }
      const std::string expected_global_path_sha256 =
          declare_parameter<std::string>(
              "route_corridor.expected_global_path_sha256", "");
      if (!valid_sha256(expected_global_path_sha256)) {
        throw std::invalid_argument("route corridor global path SHA-256 must "
                                    "be 64 lowercase hexadecimal characters");
      }
      route_corridor_ = load_route_corridor(
          corridor_path, {{"global_path", expected_global_path_sha256}});
      if (route_corridor_->corridor.frame_id.empty()) {
        throw std::invalid_argument("route corridor frame must not be empty");
      }
      prepared_route_corridor_.emplace(route_corridor_->corridor);
    } catch (const std::exception &error) {
      local_motion_activation_error_ =
          std::string("local motion activation failed: ") + error.what();
      local_motion_runtime_.reset();
      prepared_route_corridor_.reset();
      route_corridor_.reset();
      RCLCPP_ERROR(get_logger(), "%s", local_motion_activation_error_.c_str());
    }
  }

  PlannerConfig load_planner_config(double steering_limit) {
    PlannerConfig config;
    config.maximum_steering_rad = steering_limit;
    config.freshness.status_timeout_s =
        declare_parameter<double>("freshness.status_sec", 0.5);
    config.freshness.collision_timeout_s =
        declare_parameter<double>("freshness.collision_sec", 0.5);
    config.freshness.grid_timeout_s =
        declare_parameter<double>("freshness.grid_sec", 0.5);
    config.freshness.traffic_timeout_s =
        declare_parameter<double>("freshness.traffic_sec", 0.5);
    config.freshness.stop_line_timeout_s =
        declare_parameter<double>("freshness.stop_line_sec", 0.5);
    config.collision.reverse_accel =
        declare_parameter<double>("collision.reverse_accel", 0.2);
    config.collision.reverse_duration_s =
        declare_parameter<double>("collision.reverse_duration_sec", 2.0);
    config.collision.reverse_brake_duration_s =
        declare_parameter<double>("collision.reverse_brake_sec", 1.0);
    config.collision.cooldown_s =
        declare_parameter<double>("collision.cooldown_sec", 2.0);
    config.traffic.stop_line_enabled =
        declare_parameter<bool>("traffic.stop_line_enabled", false);
    config.traffic.zone_entry_radius_m = positive_finite_parameter(
        declare_parameter<double>("traffic.lateral_tolerance_m", 2.0),
        "traffic.lateral_tolerance_m");
    config.traffic.zone_exit_radius_m = positive_finite_parameter(
        declare_parameter<double>("traffic.zone_exit_radius_m", 3.0),
        "traffic.zone_exit_radius_m");
    config.traffic.front_bumper_x_m = positive_finite_parameter(
        declare_parameter<double>("traffic.front_bumper_x_m", 3.845),
        "traffic.front_bumper_x_m");
    config.traffic.stopping_margin_m =
        declare_parameter<double>("traffic.stopping_margin_m", 1.0);
    config.traffic.reaction_time_s =
        declare_parameter<double>("traffic.reaction_time_sec", 0.11589156);
    config.traffic.braking_deceleration_mps2 = positive_finite_parameter(
        declare_parameter<double>("traffic.braking_deceleration_mps2", 1.8),
        "traffic.braking_deceleration_mps2");
    config.traffic.brake_command = positive_finite_parameter(
        declare_parameter<double>("traffic.brake_command", 0.2),
        "traffic.brake_command");
    config.traffic.late_brake_deceleration_ratio = positive_finite_parameter(
        declare_parameter<double>("traffic.late_brake_deceleration_ratio",
                                  1.15),
        "traffic.late_brake_deceleration_ratio");
    config.traffic.red_confirmation_duration_s = positive_finite_parameter(
        declare_parameter<double>("traffic.red_confirmation_duration_sec", 0.2),
        "traffic.red_confirmation_duration_sec");
    config.traffic.green_release_duration_s = positive_finite_parameter(
        declare_parameter<double>("traffic.green_release_duration_sec", 0.5),
        "traffic.green_release_duration_sec");
    const int traffic_zone_count =
        declare_parameter<int>("traffic.zones.count", 0);
    if (traffic_zone_count < 0) {
      throw std::invalid_argument("traffic.zones.count must be nonnegative");
    }
    config.traffic.zones.reserve(static_cast<std::size_t>(traffic_zone_count));
    for (int index = 0; index < traffic_zone_count; ++index) {
      const std::string name =
          "traffic.zones." + std::to_string(index) + ".xy_m";
      const auto xy =
          declare_parameter<std::vector<double>>(name, std::vector<double>{});
      if (xy.size() != 2U || !std::isfinite(xy[0]) ||
          !std::isfinite(xy[1])) {
        throw std::invalid_argument(name + " must contain finite x and y");
      }
      config.traffic.zones.push_back(Point3{xy[0], xy[1], 0.0});
    }
    config.perception.enabled =
        declare_parameter<bool>("perception.enabled", true);
    config.perception.route_aligned_activation =
        declare_parameter<bool>("perception.route_aligned_activation", true);
    config.perception.clear_release_duration_s = positive_finite_parameter(
        declare_parameter<double>("perception.clear_release_duration_sec", 2.0),
        "perception.clear_release_duration_sec");
    config.perception.speed_aware_lookahead =
        declare_parameter<bool>("perception.speed_aware_lookahead", true);
    config.perception.near_x_m =
        declare_parameter<double>("perception.near_x_m", 1.0);
    config.perception.minimum_lookahead_m =
        declare_parameter<double>("perception.minimum_lookahead_m", 20.0);
    config.perception.maximum_lookahead_m =
        declare_parameter<double>("perception.maximum_lookahead_m", 99.0);
    config.perception.front_bumper_x_m =
        declare_parameter<double>("perception.front_bumper_x_m", 3.845);
    config.perception.reaction_time_s =
        declare_parameter<double>("perception.reaction_time_sec", 1.0);
    config.perception.braking_deceleration_mps2 =
        declare_parameter<double>("perception.braking_deceleration_mps2", 1.8);
    config.perception.stopping_margin_m =
        declare_parameter<double>("perception.stopping_margin_m", 5.0);
    config.perception.future_road_risk_required =
        prediction_mode_ == PredictionMode::kRequired;
    config.perception.future_road_risk_timeout_s = prediction_timeout_s_;
    config.perception.future_road_risk_limits = FutureRoadRiskLimits{
        positive_finite_parameter(
            declare_parameter<double>("perception.future_road_risk.horizon_sec",
                                      6.0),
            "perception.future_road_risk.horizon_sec"),
        positive_size_parameter(
            declare_parameter<int>(
                "perception.future_road_risk.maximum_objects", 128),
            "perception.future_road_risk.maximum_objects"),
        positive_size_parameter(
            declare_parameter<int>(
                "perception.future_road_risk.maximum_footprints", 2048),
            "perception.future_road_risk.maximum_footprints"),
        positive_size_parameter(
            declare_parameter<int>(
                "perception.future_road_risk.maximum_corridor_segments", 4096),
            "perception.future_road_risk.maximum_corridor_segments"),
        declare_parameter<double>(
            "perception.future_road_risk.covariance_sigma", 2.0),
        declare_parameter<double>(
            "perception.future_road_risk.minimum_margin_m", 0.20)};
    config.perception.forward_check_pose =
        Pose2{declare_parameter<double>("perception.forward_check_pose.x", 5.5),
              declare_parameter<double>("perception.forward_check_pose.y", 0.0),
              declare_parameter<double>("perception.forward_check_pose.yaw_rad",
                                        0.0)};
    const int perception_occupied_threshold = declare_parameter<int>(
        "perception.forward_check_footprint.occupied_threshold", 50);
    if (perception_occupied_threshold < 0 ||
        perception_occupied_threshold > 100) {
      throw std::invalid_argument("perception.forward_check_footprint.occupied_"
                                  "threshold must be between 0 and 100");
    }
    config.perception.forward_check_footprint = FootprintConfig{
        positive_finite_parameter(
            declare_parameter<double>(
                "perception.forward_check_footprint.half_length_m", 4.2),
            "perception.forward_check_footprint.half_length_m"),
        positive_finite_parameter(
            declare_parameter<double>(
                "perception.forward_check_footprint.half_width_m", 1.2),
            "perception.forward_check_footprint.half_width_m"),
        declare_parameter<double>(
            "perception.forward_check_footprint.clearance_m", 0.1),
        static_cast<std::int8_t>(perception_occupied_threshold),
        positive_size_parameter(
            declare_parameter<int>(
                "perception.forward_check_footprint.maximum_cells_to_check",
                32768),
            "perception.forward_check_footprint.maximum_cells_to_check")};
    return config;
  }

  void create_ros_interfaces() {
    visualization_ = std::make_unique<PlannerVisualization>(
        *this,
        PlannerVisualizationTopics{
            declare_parameter<std::string>("topics.path", "/ad/planner/path"),
            declare_parameter<std::string>(
                "visualization.topics.local_path",
                "/ad/viz/planner/local_path"),
            declare_parameter<std::string>(
                "visualization.topics.candidate_paths",
                "/ad/viz/planner/candidate_paths"),
            declare_parameter<std::string>(
                "visualization.topics.path_tracking",
                "/ad/viz/planner/path_tracking"),
            declare_parameter<std::string>(
                "visualization.topics.occupancy_relevance",
                "/ad/viz/planner/occupancy_relevance"),
            declare_parameter<std::string>(
                "visualization.topics.relevant_objects",
                "/ad/viz/planner/relevant_objects"),
            declare_parameter<std::string>(
                "visualization.topics.target", "/ad/viz/planner/target"),
            declare_parameter<std::string>("topics.target_speed",
                                           "/ad/planner/target_speed")});

    PlannerRosCallbacks callbacks;
    callbacks.vehicle_status = [this](const auto &message) {
      on_vehicle_status(message);
    };
    callbacks.odometry = [this](const auto &message) { on_odometry(message); };
    callbacks.collisions = [this](const auto &message) {
      on_collisions(message);
    };
    callbacks.occupancy_grid = [this](const auto &message) {
      on_grid(message);
    };
    callbacks.static_ungated = [this](const auto &message) {
      on_static_ungated(message);
    };
    callbacks.drivable_mask = [this](const auto &message) {
      on_drivable_mask(message);
    };
    callbacks.predicted_objects = [this](const auto &message) {
      on_predicted_objects(message);
    };
    callbacks.traffic_signal = [this](const auto &message) {
      on_traffic(message);
    };
    callbacks.stop_line = [this](const auto &message) {
      on_stop_line(message);
    };
    callbacks.tuning_lease = [this]() {
      tuning_lease_received_ = true;
      tuning_lease_receipt_s_ = steady_now();
    };
    callbacks.hold_control = [this](const bool hold) {
      tuning_hold_control_ = hold;
      if (hold) {
        supervisor_->halt();
      }
      return std::make_pair(true,
                            hold ? std::string("planner held at full brake")
                                 : std::string("planner control released"));
    };
    callbacks.reset_controllers = [this]() { return reset_controllers(); };
    callbacks.external_velocity = [this](const auto &message) {
      const std::int64_t receipt_steady_ns = steady_clock_.now().nanoseconds();
      if (receipt_steady_ns <= 0 || !local_motion_runtime_) {
        return;
      }
      static_cast<void>(
          local_motion_runtime_->observe_external_velocity_command(
              ExternalVelocityCommand{message.linear.x, message.angular.z,
                                      receipt_steady_ns}));
    };

    ros_interfaces_ = std::make_unique<PlannerRosInterfaces>(
        *this,
        PlannerRosInterfaceConfig{
            road_gate_enabled_, planner_config_.traffic.stop_line_enabled,
            local_motion_backend_kind_ == LocalMotionBackendKind::kMppiNav2 &&
                local_motion_runtime_ != nullptr},
        std::move(callbacks));
  }

  std::pair<bool, std::string> reset_controllers() {
    if (!tuning_hold_control_) {
      return {false, "hold planner control before resetting path tracking"};
    }
    try {
      RosControllerParameterProvider parameters(*this);
      auto next_path_tracking = make_path_tracking_controller(
          path_tracking_backend_, route_, parameters);
      auto next_local_motion =
          make_local_motion_backend(local_motion_backend_kind_, parameters);
      auto next_route_profile_markers = make_route_profile_markers(
          route_, next_path_tracking->route_speed_profile(), "map", now(),
          visualization_profile_sample_stride_);
      if (!local_motion_runtime_) {
        throw std::runtime_error("local motion runtime is unavailable");
      }
      path_tracking_ = std::move(next_path_tracking);
      local_motion_runtime_->replace_backend(std::move(next_local_motion));
      route_profile_markers_ = std::move(next_route_profile_markers);
      previous_local_trajectory_.reset();
      previous_physical_command_ = PhysicalCommand{};
      previous_steering_rad_ = 0.0;
      last_controller_result_.reset();
      supervisor_->halt();
      visualization_->publish_route_profile(route_profile_markers_, now());
      return {true, "path tracking and local motion controllers reset"};
    } catch (const std::exception &error) {
      return {false, error.what()};
    }
  }

  double steady_now() { return steady_clock_.now().seconds(); }

  void refresh_vehicle_observation() {
    auto &input = context_.inputs.status;
    input.received = status_received_ && odometry_received_;
    const bool effective_odometry_valid =
        odometry_valid_ && local_motion_pose_valid_;
    input.valid = status_valid_ && effective_odometry_valid;
    input.receipt_time_s = std::min(status_receipt_s_, odometry_receipt_s_);
  }

  void
  on_vehicle_status(const ad_morai_interfaces::msg::EgoVehicleStatus &message) {
    status_received_ = true;
    status_receipt_s_ = steady_now();
    // The competition contract exposes longitudinal VelocityX plus gear.  The
    // bridge has already converted the wire value from km/h to m/s.
    const auto speed_mps = validated_speed_mps(message.velocity.x);
    status_valid_ =
        acknowledged_gear_.update(message.gear) && speed_mps.has_value();
    context_.inputs.status.value.gear =
        acknowledged_gear_.resolve(GearRequest::kKeep);
    if (speed_mps) {
      context_.inputs.status.value.speed_mps = *speed_mps;
    }
    refresh_vehicle_observation();
  }

  void on_odometry(const nav_msgs::msg::Odometry &message) {
    const auto &position = message.pose.pose.position;
    const double yaw = tf2::getYaw(message.pose.pose.orientation);
    odometry_received_ = true;
    odometry_receipt_s_ = steady_now();
    odometry_valid_ = std::isfinite(position.x) && std::isfinite(position.y) &&
                      std::isfinite(yaw);
    odometry_frame_id_ = message.header.frame_id;
    const auto odometry_stamp_ns = valid_ros_stamp_nanoseconds(
        message.header.stamp.sec, message.header.stamp.nanosec);
    odometry_stamp_ = odometry_stamp_ns
                          ? rclcpp::Time(*odometry_stamp_ns, RCL_ROS_TIME)
                          : rclcpp::Time(0, 0, RCL_ROS_TIME);
    odometry_pose_ = Pose2{position.x, position.y, yaw};
    odometry_metadata_valid_ = !odometry_frame_id_.empty() &&
                               odometry_stamp_ns.has_value() &&
                               finite_quaternion(message.pose.pose.orientation);
    local_motion_pose_valid_ = false;
    context_.inputs.status.value.pose = odometry_pose_;
    refresh_vehicle_observation();
  }

  void on_collisions(const ad_morai_interfaces::msg::CollisionArray &message) {
    auto &input = context_.inputs.collisions;
    input.received = true;
    input.valid = true;
    input.receipt_time_s = steady_now();
    input.object_types.clear();
    for (const auto &collision : message.collisions) {
      input.object_types.push_back(collision.object_type);
    }
  }

  void commit_grid(const nav_msgs::msg::OccupancyGrid &message,
                   const double receipt_time_s,
                   const std::optional<std::int64_t> &grid_stamp_ns) {
    auto &input = context_.inputs.grid;
    input.received = true;
    input.receipt_time_s = receipt_time_s;
    auto &grid = input.value;
    grid_frame_id_ = message.header.frame_id;
    grid_stamp_ = grid_stamp_ns ? rclcpp::Time(*grid_stamp_ns, RCL_ROS_TIME)
                                : rclcpp::Time(0, 0, RCL_ROS_TIME);
    grid.origin =
        Pose2{message.info.origin.position.x, message.info.origin.position.y,
              tf2::getYaw(message.info.origin.orientation)};
    grid.resolution = message.info.resolution;
    grid.width = message.info.width;
    grid.height = message.info.height;
    grid.cells = message.data;
    grid.valid = grid.cells.size() == grid.width * grid.height;
    grid.fresh = true;
    grid_metadata_valid_ = !grid_frame_id_.empty() &&
                           grid_stamp_ns.has_value() &&
                           finite_quaternion(message.info.origin.orientation) &&
                           validate_occupancy_grid(grid).valid;
    input.valid = grid_metadata_valid_;
  }

  void commit_drivable_mask(const nav_msgs::msg::OccupancyGrid &message,
                            const std::int64_t stamp_ns) {
    drivable_mask_received_ = true;
    drivable_mask_frame_id_ = message.header.frame_id;
    drivable_mask_stamp_ = rclcpp::Time(stamp_ns, RCL_ROS_TIME);

    OccupancyGrid mask;
    mask.origin =
        Pose2{message.info.origin.position.x, message.info.origin.position.y,
              tf2::getYaw(message.info.origin.orientation)};
    mask.resolution = message.info.resolution;
    mask.width = message.info.width;
    mask.height = message.info.height;
    mask.cells = message.data;
    mask.valid = mask.cells.size() == mask.width * mask.height;
    mask.fresh = true;
    const auto &origin = message.info.origin;
    const bool exact_base_geometry =
        origin.position.z == 0.0 && origin.orientation.x == 0.0 &&
        origin.orientation.y == 0.0 && origin.orientation.z == 0.0 &&
        origin.orientation.w == 1.0;
    const bool binary_values = std::all_of(
        mask.cells.begin(), mask.cells.end(),
        [](const std::int8_t value) { return value == 0 || value == 100; });
    drivable_mask_metadata_valid_ =
        !drivable_mask_frame_id_.empty() && exact_base_geometry &&
        validate_occupancy_grid(mask).valid && binary_values;
    if (drivable_mask_metadata_valid_) {
      drivable_mask_ = std::move(mask);
    } else {
      drivable_mask_.reset();
    }
  }

  void clear_static_ungated() {
    static_ungated_.reset();
    static_ungated_metadata_valid_ = false;
    static_ungated_frame_id_.clear();
    static_ungated_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    static_ungated_receipt_s_ = 0.0;
  }

  void commit_occupancy_visualization_pair(
      ExactStampPairer<std::int64_t, TimedVisualizationGrid>::Pair pair) {
    if (!grid_metadata_valid_ || grid_stamp_.nanoseconds() != pair.stamp_ns ||
        pair.left != pair.stamp_ns || pair.right.frame_id != grid_frame_id_ ||
        !same_grid_geometry(pair.right.grid, context_.inputs.grid.value)) {
      clear_static_ungated();
      return;
    }
    static_ungated_ = std::move(pair.right.grid);
    static_ungated_frame_id_ = std::move(pair.right.frame_id);
    static_ungated_stamp_ = rclcpp::Time(pair.stamp_ns, RCL_ROS_TIME);
    static_ungated_receipt_s_ = pair.right.receipt_time_s;
    static_ungated_metadata_valid_ = true;
  }

  void pair_planning_grid_for_visualization(const std::int64_t stamp_ns) {
    if (static_ungated_stamp_.nanoseconds() != stamp_ns) {
      clear_static_ungated();
    }
    const auto pair =
        occupancy_visualization_pairer_.add_left(stamp_ns, stamp_ns);
    if (pair.has_value()) {
      commit_occupancy_visualization_pair(std::move(*pair));
    }
  }

  void commit_road_gate_pair(
      ExactStampPairer<TimedOccupancyGridMessage,
                       nav_msgs::msg::OccupancyGrid>::Pair pair) {
    commit_grid(pair.left.message, pair.left.receipt_time_s, pair.stamp_ns);
    commit_drivable_mask(pair.right, pair.stamp_ns);
    if (grid_metadata_valid_ && drivable_mask_metadata_valid_) {
      pair_planning_grid_for_visualization(pair.stamp_ns);
    } else {
      clear_static_ungated();
    }
  }

  void on_grid(const nav_msgs::msg::OccupancyGrid &message) {
    const double receipt_time_s = steady_now();
    const auto stamp_ns = valid_ros_stamp_nanoseconds(
        message.header.stamp.sec, message.header.stamp.nanosec);
    if (!road_gate_enabled_) {
      commit_grid(message, receipt_time_s, stamp_ns);
      if (stamp_ns.has_value() && grid_metadata_valid_) {
        pair_planning_grid_for_visualization(*stamp_ns);
      } else {
        clear_static_ungated();
      }
      return;
    }
    if (!stamp_ns.has_value()) {
      auto &input = context_.inputs.grid;
      input.received = true;
      input.receipt_time_s = receipt_time_s;
      input.valid = false;
      grid_metadata_valid_ = false;
      return;
    }
    const auto pair = road_gate_pairer_.add_left(
        *stamp_ns, TimedOccupancyGridMessage{message, receipt_time_s});
    if (pair.has_value()) {
      commit_road_gate_pair(std::move(*pair));
    }
  }

  void on_static_ungated(const nav_msgs::msg::OccupancyGrid &message) {
    const double receipt_time_s = steady_now();
    const auto stamp_ns = valid_ros_stamp_nanoseconds(
        message.header.stamp.sec, message.header.stamp.nanosec);
    if (!stamp_ns.has_value()) {
      clear_static_ungated();
      return;
    }
    try {
      auto visualization_grid =
          adapt_visualization_grid(message, receipt_time_s);
      const auto pair = occupancy_visualization_pairer_.add_right(
          *stamp_ns, std::move(visualization_grid));
      if (pair.has_value()) {
        commit_occupancy_visualization_pair(std::move(*pair));
      }
    } catch (const std::exception &error) {
      if (grid_stamp_.nanoseconds() == *stamp_ns) {
        clear_static_ungated();
      }
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "discarding ungated visualization grid: %s",
                           error.what());
    }
  }

  void on_drivable_mask(const nav_msgs::msg::OccupancyGrid &message) {
    const auto stamp_ns = valid_ros_stamp_nanoseconds(
        message.header.stamp.sec, message.header.stamp.nanosec);
    if (!stamp_ns.has_value()) {
      drivable_mask_received_ = true;
      drivable_mask_metadata_valid_ = false;
      drivable_mask_.reset();
      return;
    }
    const auto pair = road_gate_pairer_.add_right(*stamp_ns, message);
    if (pair.has_value()) {
      commit_road_gate_pair(std::move(*pair));
    }
  }

  void on_predicted_objects(
      const ad_interfaces::msg::PredictedObjectArray &message) {
    predicted_objects_received_ = true;
    predicted_objects_receipt_s_ = steady_now();
    predicted_objects_valid_ = false;
    predicted_objects_.clear();
    const auto stamp_ns = valid_ros_stamp_nanoseconds(
        message.header.stamp.sec, message.header.stamp.nanosec);
    if (!stamp_ns.has_value() || message.header.frame_id != odom_frame_) {
      prediction_history_.clear();
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "discarding predicted objects with invalid stamp or frame");
      return;
    }
    try {
      auto objects = adapt_predicted_objects(message);
      const auto insert_result = prediction_history_.insert(
          PredictionSnapshot{*stamp_ns, predicted_objects_receipt_s_, objects});
      predicted_objects_ = std::move(objects);
      predicted_objects_stamp_ = rclcpp::Time(*stamp_ns, RCL_ROS_TIME);
      predicted_objects_valid_ = true;
      if (insert_result == PredictionSnapshotInsertResult::kClockRollback) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "prediction clock rolled back; waiting for a "
                             "second new-epoch sample");
      }
    } catch (const std::exception &error) {
      prediction_history_.clear();
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "discarding malformed predicted objects: %s",
                           error.what());
    }
  }

  void on_traffic(const std_msgs::msg::Int8 &message) {
    context_.inputs.traffic_signal = {true, true, steady_now(), message.data};
  }

  void on_stop_line(const std_msgs::msg::Bool &message) {
    context_.inputs.stop_line = {true, true, steady_now(), message.data};
  }

  ControllerResult remember(ControllerResult result, std::string frame) {
    last_controller_result_ = result;
    last_result_frame_ = std::move(frame);
    return result;
  }

  ControllerResult run_path_tracking() {
    return remember(
        path_tracking_->update(context_.inputs.status.value.pose,
                               context_.inputs.status.value.speed_mps,
                               control_period_s_, 0,
                               context_.inputs.status.value.gear),
        "map");
  }

  FrameTransform2 lookup_planar_transform(const std::string &target_frame,
                                          const std::string &source_frame,
                                          const rclcpp::Time &stamp) {
    if (target_frame == source_frame) {
      return FrameTransform2{};
    }
    if (!tf_buffer_) {
      throw std::runtime_error("local motion TF buffer is unavailable");
    }
    const auto stamped = tf_buffer_->lookupTransform(
        target_frame, source_frame, stamp,
        rclcpp::Duration::from_seconds(control_period_s_));
    const double yaw = tf2::getYaw(stamped.transform.rotation);
    if (!std::isfinite(stamped.transform.translation.x) ||
        !std::isfinite(stamped.transform.translation.y) ||
        !finite_quaternion(stamped.transform.rotation) || !std::isfinite(yaw)) {
      throw std::runtime_error("local motion transform is not finite");
    }
    return FrameTransform2{stamped.transform.translation.x,
                           stamped.transform.translation.y, yaw};
  }

  void prepare_local_motion_poses() {
    local_motion_pose_valid_ = false;
    if (!odometry_received_ || !odometry_valid_ || !odometry_metadata_valid_) {
      local_motion_pose_error_ = "local motion odometry metadata is invalid";
      refresh_vehicle_observation();
      return;
    }

    try {
      const auto odom_transform = lookup_planar_transform(
          odom_frame_, odometry_frame_id_, odometry_stamp_);
      const auto map_transform = lookup_planar_transform(
          map_frame_, odometry_frame_id_, odometry_stamp_);
      local_motion_ego_pose_ = transform_pose(odom_transform, odometry_pose_);
      context_.inputs.status.value.pose =
          transform_pose(map_transform, odometry_pose_);
      local_motion_pose_valid_ = true;
      local_motion_pose_error_.clear();
    } catch (const tf2::TransformException &error) {
      local_motion_pose_error_ =
          std::string("stamped local motion transform failed: ") + error.what();
    } catch (const std::exception &error) {
      local_motion_pose_error_ =
          std::string("local motion pose preparation failed: ") + error.what();
    }
    refresh_vehicle_observation();
  }

  void refresh_route_occupancy_observation() {
    auto &output = context_.inputs.route_occupancy;
    output.received = context_.inputs.grid.received;
    output.valid = false;
    output.receipt_time_s = context_.inputs.grid.receipt_time_s;
    if (!planner_config_.perception.route_aligned_activation) {
      return;
    }
    if (!route_corridor_.has_value() || !prepared_route_corridor_.has_value() ||
        !grid_metadata_valid_ || !odometry_metadata_valid_ ||
        !local_motion_pose_valid_) {
      return;
    }

    try {
      const auto route_from_grid = lookup_planar_transform(
          route_corridor_->corridor.frame_id, grid_frame_id_, grid_stamp_);
      const auto grid_in_route = transform_occupancy_grid_origin(
          route_from_grid, context_.inputs.grid.value);
      const auto route_from_odometry =
          lookup_planar_transform(route_corridor_->corridor.frame_id,
                                  odometry_frame_id_, odometry_stamp_);
      const auto ego_in_route =
          transform_pose(route_from_odometry, odometry_pose_);
      const auto projection =
          project_primary_route(route_corridor_->corridor, ego_in_route);
      const double retained_lookahead =
          supervisor_ ? supervisor_->state().perception_latched_lookahead_m
                      : 0.0;
      const auto forward_check = make_perception_forward_check(
          planner_config_.perception, context_.inputs.status.value.speed_mps,
          retained_lookahead);
      const double near_s =
          projection.route_s_m + planner_config_.perception.near_x_m;
      const double far_s = projection.route_s_m + forward_check.far_x_m;
      const auto occupancy = query_route_slice_occupancy(
          *prepared_route_corridor_,
          make_ros_grid(grid_in_route, route_corridor_->corridor.frame_id,
                        grid_stamp_),
          near_s, far_s,
          planner_config_.perception.forward_check_footprint
              .occupied_threshold);
      const bool clear = occupancy.occupied_cell_count == 0U &&
                         occupancy.unknown_cell_count == 0U;
      std::optional<double> nearest_unsafe_distance_m;
      if (occupancy.nearest_occupied_s_m.has_value()) {
        nearest_unsafe_distance_m =
            *occupancy.nearest_occupied_s_m - projection.route_s_m;
      } else if (occupancy.unknown_cell_count > 0U) {
        // Unknown road cells are fail-closed. The exact nearest unknown
        // distance is not needed for activation, so report the evaluated
        // interval start as the conservative diagnostic distance.
        nearest_unsafe_distance_m = planner_config_.perception.near_x_m;
      }
      output.value = RouteOccupancyState{clear,
                                         planner_config_.perception.near_x_m,
                                         forward_check.far_x_m,
                                         occupancy.occupied_cell_count,
                                         occupancy.unknown_cell_count,
                                         nearest_unsafe_distance_m};
      output.valid = true;
    } catch (const tf2::TransformException &error) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "route occupancy transform unavailable: %s",
                           error.what());
    } catch (const std::exception &error) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "route occupancy query failed: %s", error.what());
    }
  }

  void refresh_future_road_risk_observation() {
    auto &output = context_.inputs.future_road_risk;
    output = FutureRoadRiskObservation{};
    if (prediction_mode_ == PredictionMode::kDisabled) {
      output.received = true;
      output.valid = true;
      output.receipt_time_s = context_.steady_time_s;
      return;
    }

    // Do not mask a more specific, already fail-closed local-motion error.
    // The controller path below will report route-cache and timing failures
    // while still commanding full brake.
    if (!local_motion_activation_error_.empty()) {
      output.received = true;
      output.valid = true;
      output.receipt_time_s = context_.steady_time_s;
      return;
    }
    if (odometry_metadata_valid_ && grid_metadata_valid_) {
      const auto timing = validate_local_motion_timing(
          now().nanoseconds(), odometry_stamp_.nanoseconds(),
          grid_stamp_.nanoseconds(), local_motion_timing_limits_);
      if (!timing.valid) {
        output.received = true;
        output.valid = true;
        output.receipt_time_s = context_.steady_time_s;
        return;
      }
    }

    output.received = predicted_objects_received_;
    output.receipt_time_s = predicted_objects_receipt_s_;
    if (!predicted_objects_received_ || !predicted_objects_valid_ ||
        !route_corridor_.has_value() || !odometry_metadata_valid_ ||
        !local_motion_pose_valid_) {
      return;
    }

    const auto selection = prediction_history_.select(
        prediction_mode_, odometry_stamp_.nanoseconds(), context_.steady_time_s,
        prediction_timeout_s_);
    if (!selection.admitted || !selection.use_predictions) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "future road risk prediction unavailable: %s",
                           selection.reason.c_str());
      return;
    }

    output.receipt_time_s = selection.snapshot.receipt_time_s;
    try {
      const auto odom_from_route = lookup_planar_transform(
          odom_frame_, route_corridor_->corridor.frame_id, odometry_stamp_);
      const auto corridor_in_odom = transform_reference_corridor(
          odom_from_route, route_corridor_->corridor, odom_frame_);
      const auto local_corridor = window_reference_corridor(
          corridor_in_odom, local_motion_ego_pose_, corridor_window_behind_m_,
          corridor_window_ahead_m_);
      const auto aged_predictions =
          age_predictions(selection.snapshot.objects, selection.age_s);
      output.value = evaluate_future_road_risk(
          local_corridor, aged_predictions,
          planner_config_.perception.future_road_risk_limits);
      output.valid = true;
    } catch (const tf2::TransformException &error) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "future road risk transform unavailable: %s",
                           error.what());
    } catch (const std::exception &error) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "future road risk evaluation failed: %s",
                           error.what());
    }
  }

  static visualization_msgs::msg::MarkerArray
  delete_all_markers(const std::string &frame_id, const rclcpp::Time &stamp) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = stamp;
    marker.action = visualization_msgs::msg::Marker::DELETEALL;
    visualization_msgs::msg::MarkerArray markers;
    markers.markers.push_back(std::move(marker));
    return markers;
  }

  void clear_occupancy_relevance() {
    if (!occupancy_relevance_visible_) {
      return;
    }
    visualization_->publish_occupancy_relevance(delete_all_markers(
        route_corridor_ ? route_corridor_->corridor.frame_id : map_frame_,
        now()));
    occupancy_relevance_visible_ = false;
    last_occupancy_relevance_stamp_ns_.reset();
  }

  void clear_prediction_relevance() {
    if (!prediction_relevance_visible_) {
      return;
    }
    visualization_->publish_planner_relevant_objects(
        delete_all_markers(odom_frame_, now()));
    prediction_relevance_visible_ = false;
    last_prediction_relevance_stamp_ns_.reset();
  }

  void publish_route_relevance_visualization() {
    const bool route_ready = route_corridor_.has_value() && tf_buffer_;
    const bool occupancy_ready =
        route_ready && grid_metadata_valid_ && context_.inputs.grid.valid &&
        static_ungated_metadata_valid_ && static_ungated_.has_value() &&
        static_ungated_stamp_ == grid_stamp_ &&
        static_ungated_frame_id_ == grid_frame_id_ &&
        same_grid_geometry(*static_ungated_, context_.inputs.grid.value) &&
        context_.steady_time_s - context_.inputs.grid.receipt_time_s <=
            planner_config_.freshness.grid_timeout_s &&
        context_.steady_time_s - static_ungated_receipt_s_ <=
            planner_config_.freshness.grid_timeout_s;
    if (!occupancy_ready) {
      clear_occupancy_relevance();
    } else if ((!last_occupancy_relevance_stamp_ns_.has_value() ||
                *last_occupancy_relevance_stamp_ns_ !=
                    grid_stamp_.nanoseconds()) &&
               context_.steady_time_s -
                       last_occupancy_relevance_publication_s_ >=
                   route_relevance_period_s_) {
      try {
        const auto route_from_grid = lookup_planar_transform(
            route_corridor_->corridor.frame_id, static_ungated_frame_id_,
            static_ungated_stamp_);
        const auto grid_in_route = transform_occupancy_grid_origin(
            route_from_grid, *static_ungated_);
        auto markers = build_occupancy_relevance_markers(
            route_corridor_->corridor.frame_id, grid_in_route,
            route_corridor_->corridor,
            planner_config_.perception.forward_check_footprint
                .occupied_threshold);
        for (auto &marker : markers.markers) {
          marker.header.stamp = grid_stamp_;
        }
        visualization_->publish_occupancy_relevance(markers);
        occupancy_relevance_visible_ = true;
        last_occupancy_relevance_stamp_ns_ = grid_stamp_.nanoseconds();
        last_occupancy_relevance_publication_s_ = context_.steady_time_s;
      } catch (const std::exception &error) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "cannot publish occupancy relevance: %s",
                             error.what());
        clear_occupancy_relevance();
      }
    }

    const bool prediction_ready =
        route_ready && predicted_objects_received_ &&
        predicted_objects_valid_ &&
        context_.steady_time_s - predicted_objects_receipt_s_ <=
            prediction_timeout_s_;
    if (!prediction_ready) {
      clear_prediction_relevance();
    } else if ((!last_prediction_relevance_stamp_ns_.has_value() ||
                *last_prediction_relevance_stamp_ns_ !=
                    predicted_objects_stamp_.nanoseconds()) &&
               context_.steady_time_s -
                       last_prediction_relevance_publication_s_ >=
                   route_relevance_period_s_) {
      try {
        const auto odom_from_route = lookup_planar_transform(
            odom_frame_, route_corridor_->corridor.frame_id,
            predicted_objects_stamp_);
        const auto corridor_in_odom = transform_reference_corridor(
            odom_from_route, route_corridor_->corridor, odom_frame_);
        auto markers = build_predicted_relevance_markers(
            odom_frame_, predicted_objects_, corridor_in_odom);
        for (auto &marker : markers.markers) {
          marker.header.stamp = predicted_objects_stamp_;
        }
        visualization_->publish_planner_relevant_objects(markers);
        prediction_relevance_visible_ = true;
        last_prediction_relevance_stamp_ns_ =
            predicted_objects_stamp_.nanoseconds();
        last_prediction_relevance_publication_s_ = context_.steady_time_s;
      } catch (const std::exception &error) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "cannot publish prediction relevance: %s",
                             error.what());
        clear_prediction_relevance();
      }
    }
  }

  ControllerResult fail_local_motion(const std::string &reason) {
    publish_local_motion_visualization(nullptr);
    return remember(ControllerResult{false, PhysicalCommand{}, reason},
                    odom_frame_);
  }

  ControllerResult run_local_motion() {
    if (!local_motion_activation_error_.empty()) {
      return fail_local_motion(local_motion_activation_error_);
    }
    if (!local_motion_runtime_ || !route_corridor_) {
      return fail_local_motion("local motion backend is unavailable");
    }
    if (!odometry_metadata_valid_ || !grid_metadata_valid_ ||
        !local_motion_pose_valid_) {
      return fail_local_motion(
          local_motion_pose_error_.empty()
              ? "local motion odometry or grid metadata is invalid"
              : local_motion_pose_error_);
    }

    const auto timing = validate_local_motion_timing(
        now().nanoseconds(), odometry_stamp_.nanoseconds(),
        grid_stamp_.nanoseconds(), local_motion_timing_limits_);
    if (!timing.valid) {
      return fail_local_motion(timing.reason);
    }

    try {
      const auto corridor_transform = lookup_planar_transform(
          odom_frame_, route_corridor_->corridor.frame_id, odometry_stamp_);
      const auto transformed_corridor = transform_reference_corridor(
          corridor_transform, route_corridor_->corridor, odom_frame_);
      const auto windowed_corridor = window_reference_corridor(
          transformed_corridor, local_motion_ego_pose_,
          corridor_window_behind_m_, corridor_window_ahead_m_);

      const auto grid_transform =
          lookup_planar_transform(odom_frame_, grid_frame_id_, grid_stamp_);
      const auto transformed_grid = transform_occupancy_grid_origin(
          grid_transform, context_.inputs.grid.value);
      std::optional<OccupancyGrid> transformed_drivable_mask;
      if (road_gate_enabled_) {
        if (!drivable_mask_received_ || !drivable_mask_metadata_valid_ ||
            !drivable_mask_.has_value() ||
            drivable_mask_frame_id_ != grid_frame_id_ ||
            drivable_mask_stamp_ != grid_stamp_ ||
            !same_grid_geometry(*drivable_mask_, context_.inputs.grid.value)) {
          return fail_local_motion(
              "exact-stamp drivable mask is unavailable or mismatched");
        }
        transformed_drivable_mask =
            transform_occupancy_grid_origin(grid_transform, *drivable_mask_);
      }

      LocalPlanningRequest request;
      request.reference_corridor = windowed_corridor;
      request.ego = EgoState{local_motion_ego_pose_,
                             context_.inputs.status.value.speed_mps, 0.0};
      request.occupancy_grid = transformed_grid;
      request.drivable_mask = std::move(transformed_drivable_mask);
      request.predicted_objects = {};
      if (prediction_mode_ == PredictionMode::kRequired &&
          predicted_objects_received_ && !predicted_objects_valid_) {
        return fail_local_motion(
            "required predicted-object input is malformed");
      }
      const auto prediction_selection = prediction_history_.select(
          prediction_mode_, odometry_stamp_.nanoseconds(),
          context_.steady_time_s, prediction_timeout_s_);
      if (!prediction_selection.admitted) {
        return fail_local_motion(prediction_selection.reason);
      }
      if (prediction_selection.use_predictions) {
        request.predicted_objects = age_predictions(
            prediction_selection.snapshot.objects, prediction_selection.age_s);
      }
      request.previous_trajectory = previous_local_trajectory_;
      request.previous_command = previous_physical_command_;
      request.constraints = vehicle_constraints_;
      request.stamp_ns = odometry_stamp_.nanoseconds();
      request.dt_s = control_period_s_;
      request.behavior_id = 1;
      request.gear_id = context_.inputs.status.value.gear;

      request.steady_time_ns = steady_clock_.now().nanoseconds();
      const auto runtime_result =
          local_motion_runtime_->plan(request, previous_steering_rad_);
      if (!runtime_result.valid) {
        return fail_local_motion(runtime_result.reason);
      }

      previous_local_trajectory_ = runtime_result.planning.trajectory;
      publish_local_motion_visualization(&runtime_result.planning);
      return remember(runtime_result.controller, odom_frame_);
    } catch (const tf2::TransformException &error) {
      return fail_local_motion(
          std::string("stamped local motion transform failed: ") +
          error.what());
    } catch (const std::exception &error) {
      return fail_local_motion(std::string("local motion planning failed: ") +
                               error.what());
    }
  }

  ControllerResult run_local_planner() { return run_local_motion(); }

  bool mandatory_inputs_ready(double now) const {
    const auto &status = context_.inputs.status;
    const auto &collision = context_.inputs.collisions;
    const auto &future_risk = context_.inputs.future_road_risk;
    const bool future_risk_ready =
        !planner_config_.perception.future_road_risk_required ||
        (future_risk.received && future_risk.valid &&
         std::isfinite(future_risk.receipt_time_s) &&
         future_risk.receipt_time_s <= now &&
         now - future_risk.receipt_time_s <=
             planner_config_.perception.future_road_risk_timeout_s);
    return context_.inputs.route_ready && status.received && status.valid &&
           collision.received && collision.valid && future_risk_ready &&
           now - status.receipt_time_s <=
               planner_config_.freshness.status_timeout_s &&
           now - collision.receipt_time_s <=
               planner_config_.freshness.collision_timeout_s;
  }

  void tick() {
    context_.steady_time_s = steady_now();
    last_controller_result_.reset();
    // A tuning hold suppresses actuation but must not suppress read-only TF
    // and pose preparation. The tuner waits for inputs_ready before resetting
    // and releasing control, so skipping this work while held creates a
    // startup deadlock for DWA/Frenet/MPPI backends.
    local_motion_visualization_published_this_tick_ = false;
    prepare_local_motion_poses();
    refresh_route_occupancy_observation();
    refresh_future_road_risk_observation();
    publish_route_relevance_visualization();
    const bool local_motion_pose_preparation_failed = !local_motion_pose_valid_;
    if (tuning_hold_control_) {
      PlannerTickResult held;
      held.command = PlannerContext::full_brake_command();
      held.status = SupervisorStatus::kRunning;
      held.active_behavior = "tuning_hold";
      held.failsafe_reason = "controller tuning reset in progress";
      publish_command(held.command);
      publish_status(held);
      publish_path_tracking();
      return;
    }
    if (tuning_lease_required_ &&
        (!tuning_lease_received_ ||
         context_.steady_time_s - tuning_lease_receipt_s_ >
             tuning_lease_timeout_s_)) {
      supervisor_->halt();
      PlannerTickResult held;
      held.command = PlannerContext::full_brake_command();
      held.status = SupervisorStatus::kRunning;
      held.active_behavior = "tuning_lease_timeout";
      held.failsafe_reason = "controller tuning lease missing or expired";
      publish_command(held.command);
      publish_status(held);
      publish_path_tracking();
      return;
    }
    auto result = supervisor_->tick();
    if (!local_motion_visualization_published_this_tick_) {
      publish_local_motion_visualization(nullptr);
    }
    if (local_motion_pose_preparation_failed &&
        !local_motion_pose_error_.empty()) {
      result.failsafe_reason = local_motion_pose_error_;
    }
    publish_command(result.command);
    publish_status(result);
    publish_path_tracking();
    if (last_controller_result_) {
      publish_visualization(*last_controller_result_, last_result_frame_);
    }
  }

  void publish_command(const PlannerCommand &command) {
    std_msgs::msg::Header header;
    header.stamp = now();
    header.frame_id = "base_link";
    const auto message = ad_control::make_ctrl_cmd(
        command.motion, command.gear_request, acknowledged_gear_,
        planner_config_.maximum_steering_rad, header);
    previous_steering_rad_ = command.motion.steering_rad;
    previous_physical_command_ = command.motion;
    ros_interfaces_->publish_command(message);
  }

  void publish_status(const PlannerTickResult &result) {
    ad_interfaces::msg::PlannerStatus message;
    message.header.stamp = now();
    message.header.frame_id = "map";
    message.inputs_ready = mandatory_inputs_ready(context_.steady_time_s);
    message.active_behavior = result.active_behavior;
    message.failsafe_reason = result.failsafe_reason;
    if (local_motion_backend_kind_ == LocalMotionBackendKind::kDwa &&
        last_controller_result_ && !last_controller_result_->valid) {
      message.dwa_failure_reason = last_controller_result_->reason;
    }
    ros_interfaces_->publish_status(message);
  }

  void publish_path_tracking() {
    const auto stamp = now();
    visualization_->publish_path_tracking(route_profile_markers_, stamp);
  }

  void publish_global_path() {
    visualization_->publish_global_path(route_, "map", now());
  }

  void publish_local_motion_visualization(const LocalPlanningResult *result) {
    visualization_->publish_local_motion(result, odom_frame_, now());
    local_motion_visualization_published_this_tick_ = true;
  }

  void publish_visualization(const ControllerResult &result,
                             const std::string &frame) {
    visualization_->publish_controller(result, frame, now());
  }

  rclcpp::Clock steady_clock_;
  LocalMotionBackendKind local_motion_backend_kind_{
      LocalMotionBackendKind::kDwa};
  PathTrackingBackend path_tracking_backend_{PathTrackingBackend::kStanley};
  std::unique_ptr<LocalMotionRuntime> local_motion_runtime_;
  std::optional<LoadedRouteCorridor> route_corridor_;
  std::optional<PreparedRoadCorridor> prepared_route_corridor_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  LocalMotionTimingLimits local_motion_timing_limits_;
  VehicleConstraints vehicle_constraints_;
  std::string local_motion_activation_error_;
  std::string odom_frame_{"odom"};
  std::string map_frame_{"map"};
  double corridor_window_behind_m_{10.0};
  double corridor_window_ahead_m_{100.0};
  double prediction_timeout_s_{0.50};
  PredictionMode prediction_mode_{PredictionMode::kDisabled};
  bool road_gate_enabled_{true};
  std::string odometry_frame_id_;
  std::string grid_frame_id_;
  std::string drivable_mask_frame_id_;
  std::string static_ungated_frame_id_;
  rclcpp::Time odometry_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time grid_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time drivable_mask_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time static_ungated_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time predicted_objects_stamp_{0, 0, RCL_ROS_TIME};
  Pose2 odometry_pose_;
  Pose2 local_motion_ego_pose_;
  bool odometry_metadata_valid_{false};
  bool grid_metadata_valid_{false};
  bool drivable_mask_received_{false};
  bool drivable_mask_metadata_valid_{false};
  bool static_ungated_metadata_valid_{false};
  bool predicted_objects_received_{false};
  bool predicted_objects_valid_{false};
  bool local_motion_pose_valid_{false};
  bool local_motion_visualization_published_this_tick_{false};
  std::string local_motion_pose_error_;
  std::optional<TimedTrajectory> previous_local_trajectory_;
  std::optional<OccupancyGrid> drivable_mask_;
  std::optional<OccupancyGrid> static_ungated_;
  ExactStampPairer<TimedOccupancyGridMessage, nav_msgs::msg::OccupancyGrid>
      road_gate_pairer_{16U};
  // Only the visualization payload is retained, and the queue is bounded.
  // Planning never consumes this pairer or its ungated grid.
  ExactStampPairer<std::int64_t, TimedVisualizationGrid>
      occupancy_visualization_pairer_{4U};
  // Keep several timestamped prediction snapshots so a prediction callback
  // slightly ahead of the latest odometry cannot replace the newest usable
  // snapshot. A >1 s backward jump is treated as a simulator clock epoch reset.
  PredictionSnapshotHistory prediction_history_{16U, 1'000'000'000LL};
  PredictedObjectSet predicted_objects_;
  PhysicalCommand previous_physical_command_;
  double control_period_s_{0.05};
  double mppi_path_refresh_period_s_{0.25};
  std::size_t visualization_profile_sample_stride_{10U};
  double route_relevance_period_s_{0.20};
  double previous_steering_rad_{0.0};
  bool status_received_{false};
  bool status_valid_{false};
  bool odometry_received_{false};
  bool odometry_valid_{false};
  bool tuning_hold_control_{false};
  bool tuning_lease_required_{false};
  bool tuning_lease_received_{false};
  double tuning_lease_timeout_s_{1.0};
  double tuning_lease_receipt_s_{0.0};
  double status_receipt_s_{0.0};
  double odometry_receipt_s_{0.0};
  double predicted_objects_receipt_s_{0.0};
  double static_ungated_receipt_s_{0.0};
  double last_occupancy_relevance_publication_s_{
      -std::numeric_limits<double>::infinity()};
  double last_prediction_relevance_publication_s_{
      -std::numeric_limits<double>::infinity()};
  std::optional<std::int64_t> last_occupancy_relevance_stamp_ns_;
  std::optional<std::int64_t> last_prediction_relevance_stamp_ns_;
  bool occupancy_relevance_visible_{false};
  bool prediction_relevance_visible_{false};
  AcknowledgedGear acknowledged_gear_;

  Route route_;
  PlannerConfig planner_config_;
  PlannerContext context_;
  std::unique_ptr<PathTrackingController> path_tracking_;
  std::unique_ptr<PlannerSupervisor> supervisor_;
  std::optional<ControllerResult> last_controller_result_;
  std::string last_result_frame_{"map"};
  std::string path_tracking_backend_name_{"stanley"};
  visualization_msgs::msg::MarkerArray route_profile_markers_;

  std::unique_ptr<PlannerRosInterfaces> ros_interfaces_;
  std::unique_ptr<PlannerVisualization> visualization_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr global_path_timer_;
};

std::shared_ptr<rclcpp::Node> make_planner_node() {
  return std::make_shared<AdPlannerNode>();
}

} // namespace ad_planner
