#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ad_morai_interfaces/msg/ego_vehicle_status.hpp"
#include "ad_morai_interfaces/msg/gps_rmc.hpp"
#include "ad_localization/adapter/localization_adapter.hpp"
#include "ad_localization/adapter/localization_node.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "lifecycle_msgs/msg/state.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/expand_topic_or_service_name.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "rclcpp_lifecycle/lifecycle_publisher.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2_ros/static_transform_broadcaster.h"

namespace ad_localization
{
namespace
{

using CallbackReturn =
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

bool valid_relative_frame(const std::string & frame)
{
  return !frame.empty() && frame.front() != '/' && frame.find("//") == std::string::npos &&
         frame.find_first_of(" \t\r\n") == std::string::npos;
}

bool valid_imu_orientation(const geometry_msgs::msg::Quaternion & orientation)
{
  const double norm_squared =
    orientation.x * orientation.x + orientation.y * orientation.y +
    orientation.z * orientation.z + orientation.w * orientation.w;
  return std::isfinite(orientation.x) && std::isfinite(orientation.y) &&
         std::isfinite(orientation.z) && std::isfinite(orientation.w) &&
         std::isfinite(norm_squared) && norm_squared >= 1.0e-12;
}

diagnostic_msgs::msg::KeyValue diagnostic_value(
  const std::string & key, std::size_t value)
{
  diagnostic_msgs::msg::KeyValue result;
  result.key = key;
  result.value = std::to_string(value);
  return result;
}

diagnostic_msgs::msg::KeyValue diagnostic_text_value(
  const std::string & key, const std::string & value)
{
  diagnostic_msgs::msg::KeyValue result;
  result.key = key;
  result.value = value;
  return result;
}

}  // namespace

class AdLocalizationNode final : public rclcpp_lifecycle::LifecycleNode
{
public:
  explicit AdLocalizationNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp_lifecycle::LifecycleNode("ad_localization", options)
  {
    declare_parameter<std::string>("gps_topic", "/ad/sensors/gps/fix");
    declare_parameter<std::string>("gps_rmc_topic", "/ad/sensors/gps/rmc");
    declare_parameter<std::string>("imu_topic", "/ad/sensors/imu/data");
    declare_parameter<std::string>("eskf_imu_topic", "/ad/localization/input/eskf_imu");
    declare_parameter<std::string>("status_topic", "/ad/vehicle/status");
    declare_parameter<std::string>("filtered_odometry_topic", "/ad/localization/odometry");
    declare_parameter<std::string>("gnss_pose_topic", "/ad/localization/input/gnss_pose");
    declare_parameter<std::string>("initial_pose_topic", "/ad/localization/input/initial_pose");
    declare_parameter<std::string>("wheel_speed_topic", "/ad/localization/input/wheel_speed");
    declare_parameter<std::string>("pose2d_topic", "/ad/localization/pose2d");
    declare_parameter<std::string>("diagnostics_topic", "/diagnostics");
    declare_parameter<std::string>("localization_backend", "gnss_imu");
    declare_parameter<std::string>("map_frame", "map");
    declare_parameter<std::string>("reference_frame", "odom");
    declare_parameter<std::string>("base_frame", "base_link");
    declare_parameter<bool>("publish_map_to_odom_tf", true);
    declare_parameter<int>("utm_epsg", 32652);
    declare_parameter<double>("utm_easting_offset_m", 302595.0);
    declare_parameter<double>("utm_northing_offset_m", 4124145.0);
    declare_parameter<double>("map_altitude_offset_m", 0.0);
    declare_parameter<double>("maximum_map_radius_m", 100000.0);
    declare_parameter<std::vector<double>>("gnss_lever_arm_m", {0.0, 0.0, 1.5685});
    declare_parameter<double>("wheel_standstill_threshold_mps", 0.05);
    declare_parameter<double>("wheel_speed_variance", 0.04);
    declare_parameter<double>("wheel_lateral_speed_variance", 0.04);
    declare_parameter<bool>("wheel_use_device_timestamp", false);
    declare_parameter<bool>("gps_course_enabled", false);
    declare_parameter<double>("gps_course_maximum_age_sec", 0.25);
    declare_parameter<double>("gps_course_yaw_offset_rad", 0.0);
    declare_parameter<double>("maximum_abs_sideslip_rad", 0.35);
    declare_parameter<double>("initialization_sync_tolerance_sec", 0.25);
    declare_parameter<std::string>("initial_orientation_source", "imu");
    declare_parameter<double>("initial_orientation_yaw_offset_rad", 0.0);
    declare_parameter<double>("eskf_world_yaw_offset_rad", -0.02350724531030645);
    declare_parameter<int>("initial_position_sample_count", 1);
    declare_parameter<std::vector<double>>(
      "initial_position_override_xy_m", std::vector<double>{});
    declare_parameter<double>("initial_pose_retry_period_sec", 0.2);
    declare_parameter<double>("input_timeout_sec", 0.5);
    declare_parameter<double>("input_future_tolerance_sec", 0.25);
    declare_parameter<double>("diagnostics_period_sec", 1.0);
  }

protected:
  CallbackReturn on_configure(const rclcpp_lifecycle::State &) override
  {
    reset_configured_entities();
    try {
      gps_topic_ = get_parameter("gps_topic").as_string();
      gps_rmc_topic_ = get_parameter("gps_rmc_topic").as_string();
      imu_topic_ = get_parameter("imu_topic").as_string();
      eskf_imu_topic_ = get_parameter("eskf_imu_topic").as_string();
      status_topic_ = get_parameter("status_topic").as_string();
      filtered_odometry_topic_ = get_parameter("filtered_odometry_topic").as_string();
      gnss_pose_topic_ = get_parameter("gnss_pose_topic").as_string();
      initial_pose_topic_ = get_parameter("initial_pose_topic").as_string();
      wheel_speed_topic_ = get_parameter("wheel_speed_topic").as_string();
      pose2d_topic_ = get_parameter("pose2d_topic").as_string();
      diagnostics_topic_ = get_parameter("diagnostics_topic").as_string();
      localization_backend_ = get_parameter("localization_backend").as_string();
      map_frame_ = get_parameter("map_frame").as_string();
      publish_map_to_odom_tf_ =
        get_parameter("publish_map_to_odom_tf").as_bool();
      eskf_world_yaw_offset_rad_ =
        get_parameter("eskf_world_yaw_offset_rad").as_double();

      AdapterConfig config;
      config.reference_frame = get_parameter("reference_frame").as_string();
      config.base_frame = get_parameter("base_frame").as_string();
      config.utm_epsg = static_cast<int>(get_parameter("utm_epsg").as_int());
      config.easting_offset_m = get_parameter("utm_easting_offset_m").as_double();
      config.northing_offset_m = get_parameter("utm_northing_offset_m").as_double();
      config.altitude_offset_m = get_parameter("map_altitude_offset_m").as_double();
      config.maximum_map_radius_m = get_parameter("maximum_map_radius_m").as_double();
      const auto lever_arm = get_parameter("gnss_lever_arm_m").as_double_array();
      if (lever_arm.size() != 3) {
        throw std::invalid_argument("gnss_lever_arm_m must contain exactly three values");
      }
      if (!std::all_of(
          lever_arm.begin(), lever_arm.end(),
          [](double value) {return std::isfinite(value);}))
      {
        throw std::invalid_argument("gnss_lever_arm_m values must be finite");
      }
      std::copy(lever_arm.begin(), lever_arm.end(), config.gnss_lever_arm_m.begin());
      config.wheel_standstill_threshold_mps =
        get_parameter("wheel_standstill_threshold_mps").as_double();
      config.wheel_speed_variance = get_parameter("wheel_speed_variance").as_double();
      config.wheel_lateral_speed_variance =
        get_parameter("wheel_lateral_speed_variance").as_double();
      config.wheel_use_device_timestamp =
        get_parameter("wheel_use_device_timestamp").as_bool();
      config.gps_course_enabled = get_parameter("gps_course_enabled").as_bool();
      config.gps_course_maximum_age_sec =
        get_parameter("gps_course_maximum_age_sec").as_double();
      config.gps_course_yaw_offset_rad =
        get_parameter("gps_course_yaw_offset_rad").as_double();
      config.maximum_abs_sideslip_rad =
        get_parameter("maximum_abs_sideslip_rad").as_double();
      config.initialization_sync_tolerance_sec =
        get_parameter("initialization_sync_tolerance_sec").as_double();
      config.initial_orientation_source =
        get_parameter("initial_orientation_source").as_string();
      config.initial_orientation_yaw_offset_rad =
        get_parameter("initial_orientation_yaw_offset_rad").as_double();
      const auto initial_position_sample_count =
        get_parameter("initial_position_sample_count").as_int();
      if (initial_position_sample_count <= 0 ||
        initial_position_sample_count > std::numeric_limits<int>::max())
      {
        throw std::invalid_argument("initial_position_sample_count must be a positive int");
      }
      config.initial_position_sample_count = static_cast<int>(initial_position_sample_count);
      const auto initial_position_override_xy =
        get_parameter("initial_position_override_xy_m").as_double_array();
      if (!initial_position_override_xy.empty()) {
        if (initial_position_override_xy.size() != 2 ||
          !std::all_of(
            initial_position_override_xy.begin(), initial_position_override_xy.end(),
            [](double value) {return std::isfinite(value);}))
        {
          throw std::invalid_argument(
                  "initial_position_override_xy_m must be empty or contain two finite values");
        }
        config.initial_position_override_xy_m = std::array<double, 2>{
          initial_position_override_xy[0], initial_position_override_xy[1]};
      }

      initial_pose_retry_period_sec_ =
        get_parameter("initial_pose_retry_period_sec").as_double();
      input_timeout_sec_ = get_parameter("input_timeout_sec").as_double();
      input_future_tolerance_sec_ =
        get_parameter("input_future_tolerance_sec").as_double();
      diagnostics_period_sec_ = get_parameter("diagnostics_period_sec").as_double();
      validate_configuration(config);
      adapter_config_ = config;
      adapter_ = std::make_unique<LocalizationAdapter>(adapter_config_);
      reset_adapter_service_ = create_service<std_srvs::srv::Trigger>(
        "/ad/localization/reset_adapter",
        [this](
          const std_srvs::srv::Trigger::Request::SharedPtr,
          std_srvs::srv::Trigger::Response::SharedPtr response)
        {
          if (!adapter_) {
            response->success = false;
            response->message = "localization adapter is not configured";
            return;
          }
          adapter_->reset();
          reset_health_state();
          response->success = true;
          response->message = "localization adapter state reset";
        });

      const auto reliable_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
      const auto initial_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
      gnss_pose_publisher_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        gnss_pose_topic_, reliable_qos);
      initial_pose_publisher_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        initial_pose_topic_, initial_qos);
      if (localization_backend_ == "eskf") {
        eskf_imu_publisher_ = create_publisher<sensor_msgs::msg::Imu>(
          eskf_imu_topic_, rclcpp::SensorDataQoS());
      }
      wheel_speed_publisher_ =
        create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>(
        wheel_speed_topic_, reliable_qos);
      pose2d_publisher_ = create_publisher<geometry_msgs::msg::Pose2D>(
        pose2d_topic_, reliable_qos);
      diagnostics_publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
        diagnostics_topic_, reliable_qos);
      if (publish_map_to_odom_tf_) {
        static_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
      }

      initial_pose_timer_ = create_wall_timer(
        std::chrono::duration<double>(initial_pose_retry_period_sec_),
        std::bind(&AdLocalizationNode::publish_pending_initial_pose, this));
      diagnostics_timer_ = create_wall_timer(
        std::chrono::duration<double>(diagnostics_period_sec_),
        std::bind(&AdLocalizationNode::publish_diagnostics, this));
      reset_health_state();
      RCLCPP_INFO(
        get_logger(), "configured GNSS, IMU, and wheel adapter for %s backend",
        localization_backend_.c_str());
      return CallbackReturn::SUCCESS;
    } catch (const std::exception & error) {
      RCLCPP_ERROR(get_logger(), "localization configuration rejected: %s", error.what());
      reset_configured_entities();
      return CallbackReturn::FAILURE;
    }
  }

  CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override
  {
    if (!adapter_ || !gnss_pose_publisher_ || !initial_pose_publisher_ ||
      !wheel_speed_publisher_ || !pose2d_publisher_ || !diagnostics_publisher_ ||
      (publish_map_to_odom_tf_ && !static_broadcaster_) ||
      (localization_backend_ == "eskf" && !eskf_imu_publisher_))
    {
      RCLCPP_ERROR(get_logger(), "localization activation rejected: node is not configured");
      return CallbackReturn::FAILURE;
    }

    try {
      gps_subscription_ = create_subscription<sensor_msgs::msg::NavSatFix>(
        gps_topic_, rclcpp::SensorDataQoS(),
        std::bind(&AdLocalizationNode::gps_callback, this, std::placeholders::_1));
      if (adapter_config_.gps_course_enabled) {
        gps_rmc_subscription_ = create_subscription<ad_morai_interfaces::msg::GpsRmc>(
          gps_rmc_topic_, rclcpp::SensorDataQoS(),
          std::bind(&AdLocalizationNode::gps_rmc_callback, this, std::placeholders::_1));
      }
      imu_subscription_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, rclcpp::SensorDataQoS(),
        std::bind(&AdLocalizationNode::imu_callback, this, std::placeholders::_1));
      status_subscription_ = create_subscription<ad_morai_interfaces::msg::EgoVehicleStatus>(
        status_topic_, rclcpp::SensorDataQoS(),
        std::bind(&AdLocalizationNode::status_callback, this, std::placeholders::_1));
      filtered_odometry_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
        filtered_odometry_topic_, rclcpp::QoS(rclcpp::KeepLast(10)).reliable(),
        std::bind(&AdLocalizationNode::filtered_odometry_callback, this, std::placeholders::_1));

      const auto result = rclcpp_lifecycle::LifecycleNode::on_activate(state);
      if (result != CallbackReturn::SUCCESS) {
        reset_subscriptions();
        return result;
      }
      if (publish_map_to_odom_tf_ && !static_transform_published_) {
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = now();
        transform.header.frame_id = map_frame_;
        transform.child_frame_id = adapter_config_.reference_frame;
        transform.transform.rotation.w = 1.0;
        static_broadcaster_->sendTransform(transform);
        static_transform_published_ = true;
      }
      RCLCPP_INFO(get_logger(), "activated localization input adapter");
      return CallbackReturn::SUCCESS;
    } catch (const std::exception & error) {
      reset_subscriptions();
      deactivate_publishers();
      RCLCPP_ERROR(get_logger(), "localization activation failed: %s", error.what());
      return CallbackReturn::FAILURE;
    }
  }

  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override
  {
    reset_subscriptions();
    const auto result = rclcpp_lifecycle::LifecycleNode::on_deactivate(state);
    RCLCPP_INFO(get_logger(), "deactivated localization input adapter");
    return result;
  }

  CallbackReturn on_cleanup(const rclcpp_lifecycle::State &) override
  {
    reset_configured_entities();
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_shutdown(const rclcpp_lifecycle::State &) override
  {
    reset_configured_entities();
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_error(const rclcpp_lifecycle::State &) override
  {
    reset_subscriptions();
    deactivate_publishers();
    RCLCPP_ERROR(get_logger(), "localization adapter entered the error state");
    return CallbackReturn::SUCCESS;
  }

private:
  bool active()
  {
    return get_current_state().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE;
  }

  void validate_project_topic(const std::string & topic, const std::string & parameter_name) const
  {
    if (topic.rfind("/ad/", 0) != 0) {
      throw std::invalid_argument(parameter_name + " must remain under /ad/");
    }
    const auto expanded = rclcpp::expand_topic_or_service_name(
      topic, get_name(), get_namespace(), false);
    if (expanded != topic) {
      throw std::invalid_argument(parameter_name + " must be an absolute normalized topic");
    }
  }

  void validate_absolute_topic(const std::string & topic, const std::string & parameter_name) const
  {
    if (topic.empty() || topic.front() != '/') {
      throw std::invalid_argument(parameter_name + " must be an absolute topic");
    }
    const auto expanded = rclcpp::expand_topic_or_service_name(
      topic, get_name(), get_namespace(), false);
    if (expanded != topic) {
      throw std::invalid_argument(parameter_name + " must be an absolute normalized topic");
    }
  }

  void validate_configuration(const AdapterConfig & config) const
  {
    if (
      localization_backend_ != "gnss_imu" && localization_backend_ != "eskf" &&
      localization_backend_ != "fastlio" && localization_backend_ != "hybrid" &&
      localization_backend_ != "imu_quaternion_encoder" &&
      localization_backend_ != "quaternion_wheel_gnss_ekf")
    {
      throw std::invalid_argument(
              "localization_backend must be exactly gnss_imu, eskf, imu_quaternion_encoder, quaternion_wheel_gnss_ekf, fastlio, or hybrid");
    }
    if (localization_backend_ == "eskf") {
      if (config.initial_orientation_source != "imu") {
        throw std::invalid_argument(
                "eskf backend requires initial_orientation_source=imu");
      }
      if (config.initial_orientation_yaw_offset_rad != 0.0) {
        throw std::invalid_argument(
                "eskf backend requires initial_orientation_yaw_offset_rad=0.0");
      }
    }
    const std::array<std::pair<std::string, std::string>, 10> project_topics{{
      {gps_topic_, "gps_topic"},
      {gps_rmc_topic_, "gps_rmc_topic"},
      {imu_topic_, "imu_topic"},
      {eskf_imu_topic_, "eskf_imu_topic"},
      {status_topic_, "status_topic"},
      {filtered_odometry_topic_, "filtered_odometry_topic"},
      {gnss_pose_topic_, "gnss_pose_topic"},
      {initial_pose_topic_, "initial_pose_topic"},
      {wheel_speed_topic_, "wheel_speed_topic"},
      {pose2d_topic_, "pose2d_topic"},
    }};
    for (const auto & [topic, name] : project_topics) {
      validate_project_topic(topic, name);
    }
    validate_absolute_topic(diagnostics_topic_, "diagnostics_topic");

    std::vector<std::string> all_topics;
    all_topics.reserve(project_topics.size() + 1);
    for (const auto & item : project_topics) {
      all_topics.push_back(item.first);
    }
    all_topics.push_back(diagnostics_topic_);
    std::sort(all_topics.begin(), all_topics.end());
    if (std::adjacent_find(all_topics.begin(), all_topics.end()) != all_topics.end()) {
      throw std::invalid_argument("localization topics must be distinct");
    }

    if (!valid_relative_frame(map_frame_) || !valid_relative_frame(config.reference_frame) ||
      !valid_relative_frame(config.base_frame))
    {
      throw std::invalid_argument("map, reference, and base must be relative TF frames");
    }
    if (map_frame_ == config.reference_frame || map_frame_ == config.base_frame ||
      config.reference_frame == config.base_frame)
    {
      throw std::invalid_argument("map, reference, and base frames must be distinct");
    }
    if (config.utm_epsg != 32652) {
      throw std::invalid_argument("utm_epsg must be 32652");
    }
    if (!std::isfinite(eskf_world_yaw_offset_rad_)) {
      throw std::invalid_argument("eskf_world_yaw_offset_rad must be finite");
    }
    const std::array<double, 3> positive_durations{
      initial_pose_retry_period_sec_, input_timeout_sec_, diagnostics_period_sec_};
    if (!std::all_of(
        positive_durations.begin(), positive_durations.end(),
        [](double value) {return std::isfinite(value) && value > 0.0;}))
    {
      throw std::invalid_argument("retry, input timeout, and diagnostics periods must be positive");
    }
    if (!std::isfinite(input_future_tolerance_sec_) || input_future_tolerance_sec_ < 0.0) {
      throw std::invalid_argument("input future tolerance must be finite and nonnegative");
    }
  }

  bool fresh(const builtin_interfaces::msg::Time & stamp) const
  {
    if (stamp.sec < 0 || stamp.nanosec >= 1000000000U ||
      (stamp.sec == 0 && stamp.nanosec == 0))
    {
      return false;
    }
    const rclcpp::Time message_time(stamp, get_clock()->get_clock_type());
    const double age = (now() - message_time).seconds();
    return std::isfinite(age) && age >= -input_future_tolerance_sec_ &&
           age <= input_timeout_sec_;
  }

  void gps_callback(const sensor_msgs::msg::NavSatFix::SharedPtr fix)
  {
    if (!active() || !fresh(fix->header.stamp) || !adapter_) {
      ++gnss_dropped_;
      return;
    }
    auto pose = adapter_->convert_gnss(*fix);
    if (!pose) {
      ++gnss_dropped_;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "dropping invalid GNSS fix");
      return;
    }
    ++gnss_accepted_;
    last_valid_gnss_time_ = now();
    gnss_pose_publisher_->publish(*pose);
    if (auto initial = adapter_->observe_gnss_for_initialization(*pose)) {
      initial_pose_publisher_->publish(*initial);
    }
  }

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr imu)
  {
    if (!active() || !fresh(imu->header.stamp) || !valid_imu_orientation(imu->orientation) ||
      !adapter_)
    {
      ++imu_dropped_;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "dropping invalid IMU sample");
      return;
    }
    const sensor_msgs::msg::Imu * initialization_imu = imu.get();
    std::optional<sensor_msgs::msg::Imu> corrected_imu;
    if (localization_backend_ == "eskf") {
      corrected_imu = apply_world_yaw_correction(*imu, eskf_world_yaw_offset_rad_);
      if (!corrected_imu || !eskf_imu_publisher_ || !eskf_imu_publisher_->is_activated()) {
        ++imu_dropped_;
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000, "dropping uncorrectable ESKF IMU sample");
        return;
      }
      eskf_imu_publisher_->publish(*corrected_imu);
      initialization_imu = &*corrected_imu;
    }
    if (auto initial = adapter_->observe_imu_for_initialization(*initialization_imu)) {
      initial_pose_publisher_->publish(*initial);
      ++imu_accepted_;
      return;
    }
    ++imu_accepted_;
  }

  void gps_rmc_callback(const ad_morai_interfaces::msg::GpsRmc::SharedPtr rmc)
  {
    if (!active() || !fresh(rmc->header.stamp) || !adapter_ ||
      !adapter_->observe_gps_rmc(*rmc))
    {
      ++gps_course_dropped_;
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "dropping invalid GPS RMC course sample");
      return;
    }
    ++gps_course_accepted_;
  }

  void status_callback(const ad_morai_interfaces::msg::EgoVehicleStatus::SharedPtr status)
  {
    if (!active() || !fresh(status->header.stamp) || !adapter_) {
      ++wheel_dropped_;
      return;
    }
    if (auto initial = adapter_->observe_vehicle_status_for_initialization(*status)) {
      initial_pose_publisher_->publish(*initial);
    }
    auto wheel = adapter_->convert_wheel_speed(*status);
    if (!wheel) {
      ++wheel_dropped_;
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "dropping invalid or ambiguous wheel-speed sample");
      return;
    }
    ++wheel_accepted_;
    wheel_speed_publisher_->publish(*wheel);
  }

  void filtered_odometry_callback(const nav_msgs::msg::Odometry::SharedPtr odometry)
  {
    if (!active() || !fresh(odometry->header.stamp) || !adapter_) {
      ++odometry_dropped_;
      return;
    }
    auto pose = adapter_->convert_filtered_odometry(*odometry);
    if (!pose) {
      ++odometry_dropped_;
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "dropping invalid selected-localizer odometry");
      return;
    }
    adapter_->acknowledge_filtered_odometry();
    filtered_output_received_ = true;
    last_filtered_odometry_time_ = now();
    ++odometry_accepted_;
    pose2d_publisher_->publish(*pose);
  }

  void publish_pending_initial_pose()
  {
    if (!active() || !adapter_ || !initial_pose_publisher_->is_activated()) {
      return;
    }
    if (auto initial = adapter_->pending_initial_pose()) {
      initial_pose_publisher_->publish(*initial);
    }
  }

  void publish_diagnostics()
  {
    if (!active() || !diagnostics_publisher_ || !diagnostics_publisher_->is_activated()) {
      return;
    }
    diagnostic_msgs::msg::DiagnosticArray array;
    array.header.stamp = now();
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "ad_localization/input_adapter";
    status.hardware_id = "morai_competition_inputs";

    const double gnss_age = last_valid_gnss_time_ ?
      (now() - *last_valid_gnss_time_).seconds() : INFINITY;
    const double odometry_age = last_filtered_odometry_time_ ?
      (now() - *last_filtered_odometry_time_).seconds() : INFINITY;
    const bool gnss_unavailable =
      !std::isfinite(gnss_age) || gnss_age > input_timeout_sec_;
    const bool odometry_unavailable =
      !std::isfinite(odometry_age) || odometry_age > input_timeout_sec_;

    if (localization_backend_ == "gnss_imu" && gnss_unavailable) {
      status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
      status.message = "GNSS unavailable; direct localization output will stop";
    } else if (!filtered_output_received_) {
      status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
      status.message = "waiting for " + localization_backend_ + " localization output";
    } else if (odometry_unavailable) {
      status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
      status.message = "filtered odometry is stale";
    } else if (localization_backend_ == "eskf" && gnss_unavailable) {
      status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
      status.message = "GNSS unavailable; ESKF output may be propagation-only";
    } else {
      status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
      status.message = "localization inputs and filtered output are fresh";
    }
    status.values = {
      diagnostic_text_value("localization_backend", localization_backend_),
      diagnostic_value("gnss_accepted", gnss_accepted_),
      diagnostic_value("gnss_dropped", gnss_dropped_),
      diagnostic_value("gps_course_accepted", gps_course_accepted_),
      diagnostic_value("gps_course_dropped", gps_course_dropped_),
      diagnostic_value("imu_accepted", imu_accepted_),
      diagnostic_value("imu_dropped", imu_dropped_),
      diagnostic_value("wheel_accepted", wheel_accepted_),
      diagnostic_value("wheel_dropped", wheel_dropped_),
      diagnostic_value("odometry_accepted", odometry_accepted_),
      diagnostic_value("odometry_dropped", odometry_dropped_),
    };
    array.status.push_back(std::move(status));
    diagnostics_publisher_->publish(array);
  }

  void reset_health_state()
  {
    gnss_accepted_ = 0;
    gnss_dropped_ = 0;
    gps_course_accepted_ = 0;
    gps_course_dropped_ = 0;
    imu_accepted_ = 0;
    imu_dropped_ = 0;
    wheel_accepted_ = 0;
    wheel_dropped_ = 0;
    odometry_accepted_ = 0;
    odometry_dropped_ = 0;
    filtered_output_received_ = false;
    last_valid_gnss_time_.reset();
    last_filtered_odometry_time_.reset();
  }

  void reset_subscriptions()
  {
    gps_subscription_.reset();
    gps_rmc_subscription_.reset();
    imu_subscription_.reset();
    status_subscription_.reset();
    filtered_odometry_subscription_.reset();
  }

  void deactivate_publishers()
  {
    if (gnss_pose_publisher_ && gnss_pose_publisher_->is_activated()) {
      gnss_pose_publisher_->on_deactivate();
    }
    if (initial_pose_publisher_ && initial_pose_publisher_->is_activated()) {
      initial_pose_publisher_->on_deactivate();
    }
    if (eskf_imu_publisher_ && eskf_imu_publisher_->is_activated()) {
      eskf_imu_publisher_->on_deactivate();
    }
    if (wheel_speed_publisher_ && wheel_speed_publisher_->is_activated()) {
      wheel_speed_publisher_->on_deactivate();
    }
    if (pose2d_publisher_ && pose2d_publisher_->is_activated()) {
      pose2d_publisher_->on_deactivate();
    }
    if (diagnostics_publisher_ && diagnostics_publisher_->is_activated()) {
      diagnostics_publisher_->on_deactivate();
    }
  }

  void reset_configured_entities()
  {
    reset_subscriptions();
    deactivate_publishers();
    initial_pose_timer_.reset();
    diagnostics_timer_.reset();
    gnss_pose_publisher_.reset();
    initial_pose_publisher_.reset();
    eskf_imu_publisher_.reset();
    wheel_speed_publisher_.reset();
    pose2d_publisher_.reset();
    diagnostics_publisher_.reset();
    static_broadcaster_.reset();
    reset_adapter_service_.reset();
    adapter_.reset();
    static_transform_published_ = false;
    reset_health_state();
  }

  AdapterConfig adapter_config_;
  std::unique_ptr<LocalizationAdapter> adapter_;

  std::string gps_topic_;
  std::string gps_rmc_topic_;
  std::string imu_topic_;
  std::string eskf_imu_topic_;
  std::string status_topic_;
  std::string filtered_odometry_topic_;
  std::string gnss_pose_topic_;
  std::string initial_pose_topic_;
  std::string wheel_speed_topic_;
  std::string pose2d_topic_;
  std::string diagnostics_topic_;
  std::string localization_backend_;
  std::string map_frame_;
  double initial_pose_retry_period_sec_{0.0};
  double input_timeout_sec_{0.0};
  double input_future_tolerance_sec_{0.0};
  double diagnostics_period_sec_{0.0};
  double eskf_world_yaw_offset_rad_{0.0};
  bool static_transform_published_{false};
  bool publish_map_to_odom_tf_{true};
  bool filtered_output_received_{false};

  std::size_t gnss_accepted_{0};
  std::size_t gnss_dropped_{0};
  std::size_t gps_course_accepted_{0};
  std::size_t gps_course_dropped_{0};
  std::size_t imu_accepted_{0};
  std::size_t imu_dropped_{0};
  std::size_t wheel_accepted_{0};
  std::size_t wheel_dropped_{0};
  std::size_t odometry_accepted_{0};
  std::size_t odometry_dropped_{0};
  std::optional<rclcpp::Time> last_valid_gnss_time_;
  std::optional<rclcpp::Time> last_filtered_odometry_time_;

  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_subscription_;
  rclcpp::Subscription<ad_morai_interfaces::msg::GpsRmc>::SharedPtr gps_rmc_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
  rclcpp::Subscription<ad_morai_interfaces::msg::EgoVehicleStatus>::SharedPtr status_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr filtered_odometry_subscription_;
  rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::PoseStamped>::SharedPtr
    gnss_pose_publisher_;
  rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::PoseStamped>::SharedPtr
    initial_pose_publisher_;
  rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Imu>::SharedPtr eskf_imu_publisher_;
  rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr
    wheel_speed_publisher_;
  rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::Pose2D>::SharedPtr pose2d_publisher_;
  rclcpp_lifecycle::LifecyclePublisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
    diagnostics_publisher_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
  rclcpp::TimerBase::SharedPtr initial_pose_timer_;
  rclcpp::TimerBase::SharedPtr diagnostics_timer_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_adapter_service_;
};

std::shared_ptr<rclcpp_lifecycle::LifecycleNode> make_localization_node(
  const rclcpp::NodeOptions & options)
{
  return std::make_shared<AdLocalizationNode>(options);
}

}  // namespace ad_localization
