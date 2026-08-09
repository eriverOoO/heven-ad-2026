#include "ad_localization/quaternion_wheel_gnss_ekf/quaternion_wheel_gnss_ekf_node.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ad_localization/quaternion_wheel_gnss_ekf/quaternion_wheel_gnss_ekf.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "std_srvs/srv/trigger.hpp"

namespace ad_localization
{
namespace
{

geometry_msgs::msg::Quaternion quaternion_from_rpy(
  double roll, double pitch, double yaw)
{
  const double cr = std::cos(roll * 0.5);
  const double sr = std::sin(roll * 0.5);
  const double cp = std::cos(pitch * 0.5);
  const double sp = std::sin(pitch * 0.5);
  const double cy = std::cos(yaw * 0.5);
  const double sy = std::sin(yaw * 0.5);
  geometry_msgs::msg::Quaternion result;
  result.x = sr * cp * cy - cr * sp * sy;
  result.y = cr * sp * cy + sr * cp * sy;
  result.z = cr * cp * sy - sr * sp * cy;
  result.w = cr * cp * cy + sr * sp * sy;
  return result;
}

class QuaternionWheelGnssEkfNode final : public rclcpp::Node
{
public:
  explicit QuaternionWheelGnssEkfNode(const rclcpp::NodeOptions & options)
  : rclcpp::Node("quaternion_wheel_gnss_ekf", options)
  {
    declare_parameter<std::string>("imu_topic", "/ad/sensors/imu/data");
    declare_parameter<std::string>(
      "wheel_topic", "/ad/localization/input/wheel_speed");
    declare_parameter<std::string>(
      "gnss_topic", "/ad/localization/input/gnss_pose");
    declare_parameter<std::string>(
      "output_odometry_topic",
      "/ad/localization/backends/quaternion_wheel_gnss_ekf/odometry");
    declare_parameter<std::string>("reference_frame", "odom");
    declare_parameter<std::string>("base_frame", "base_link");
    declare_parameter<std::string>("imu_frame", "imu_link");
    declare_parameter<std::vector<double>>(
      "gnss_lever_arm_m", {0.0, 0.0, 0.0});
    declare_parameter<std::vector<double>>(
      "imu_mount_rpy_rad", {0.0, 0.0, 0.0});
    declare_parameter<double>("world_yaw_offset_rad", 0.0);
    declare_parameter<double>("maximum_imu_age_sec", 0.1);
    declare_parameter<double>("maximum_prediction_dt_sec", 0.5);
    declare_parameter<int>("initialization.sample_count", 20);
    declare_parameter<double>("initial_position_variance_m2", 1.0);
    declare_parameter<double>("initial_wheel_bias_mps", 0.0);
    declare_parameter<double>(
      "initial_wheel_bias_variance_m2ps2", 0.25);
    declare_parameter<double>(
      "wheel_speed_variance_floor_m2ps2", 0.001);
    declare_parameter<double>(
      "wheel_bias_random_walk_variance_m2ps3", 0.01);
    declare_parameter<double>("gnss_variance_m2", 9.0);
    declare_parameter<double>("gnss_mahalanobis_threshold", 9.21);
    declare_parameter<double>("teleport.distance_m", 8.0);
    declare_parameter<int>("teleport.confirmation_samples", 3);
    declare_parameter<double>("teleport.candidate_radius_m", 4.0);
    declare_parameter<double>("teleport.max_interval_sec", 0.5);
    declare_parameter<double>("fixed_output_z_m", 0.0);
    declare_parameter<double>("unobserved_variance", 1.0e6);
    declare_parameter<double>("orientation_variance_rad2", 0.0001);
    declare_parameter<bool>("publish_tf", false);
    if (get_parameter("publish_tf").as_bool()) {
      throw std::invalid_argument(
              "quaternion_wheel_gnss_ekf publish_tf must remain false; localization_manager owns TF");
    }

    const auto lever_arm =
      get_parameter("gnss_lever_arm_m").as_double_array();
    if (lever_arm.size() != 3) {
      throw std::invalid_argument(
              "gnss_lever_arm_m must contain exactly three values");
    }
    const auto imu_mount_rpy =
      get_parameter("imu_mount_rpy_rad").as_double_array();
    if (imu_mount_rpy.size() != 3) {
      throw std::invalid_argument(
              "imu_mount_rpy_rad must contain exactly three values");
    }
    const auto initialization_sample_count =
      get_parameter("initialization.sample_count").as_int();
    const auto teleport_confirmation_samples =
      get_parameter("teleport.confirmation_samples").as_int();
    if (initialization_sample_count <= 0 ||
      initialization_sample_count > std::numeric_limits<int>::max() ||
      teleport_confirmation_samples <= 0 ||
      teleport_confirmation_samples > std::numeric_limits<int>::max())
    {
      throw std::invalid_argument("sample counts must be positive int values");
    }

    QuaternionWheelGnssEkfConfig config;
    config.reference_frame = get_parameter("reference_frame").as_string();
    config.base_frame = get_parameter("base_frame").as_string();
    config.imu_frame = get_parameter("imu_frame").as_string();
    std::copy(
      lever_arm.begin(), lever_arm.end(), config.gnss_lever_arm_m.begin());
    config.base_to_imu_orientation = quaternion_from_rpy(
      imu_mount_rpy[0], imu_mount_rpy[1], imu_mount_rpy[2]);
    config.world_yaw_offset_rad =
      get_parameter("world_yaw_offset_rad").as_double();
    config.maximum_imu_age_sec =
      get_parameter("maximum_imu_age_sec").as_double();
    config.maximum_prediction_dt_sec =
      get_parameter("maximum_prediction_dt_sec").as_double();
    config.initialization_sample_count =
      static_cast<int>(initialization_sample_count);
    config.initial_position_variance_m2 =
      get_parameter("initial_position_variance_m2").as_double();
    config.initial_wheel_bias_mps =
      get_parameter("initial_wheel_bias_mps").as_double();
    config.initial_wheel_bias_variance_m2ps2 =
      get_parameter("initial_wheel_bias_variance_m2ps2").as_double();
    config.wheel_speed_variance_floor_m2ps2 =
      get_parameter("wheel_speed_variance_floor_m2ps2").as_double();
    config.wheel_bias_random_walk_variance_m2ps3 =
      get_parameter("wheel_bias_random_walk_variance_m2ps3").as_double();
    config.gnss_variance_m2 =
      get_parameter("gnss_variance_m2").as_double();
    config.gnss_mahalanobis_threshold =
      get_parameter("gnss_mahalanobis_threshold").as_double();
    config.teleport_distance_m =
      get_parameter("teleport.distance_m").as_double();
    config.teleport_confirmation_samples =
      static_cast<int>(teleport_confirmation_samples);
    config.teleport_candidate_radius_m =
      get_parameter("teleport.candidate_radius_m").as_double();
    config.teleport_max_interval_sec =
      get_parameter("teleport.max_interval_sec").as_double();
    config.fixed_output_z_m = get_parameter("fixed_output_z_m").as_double();
    config.unobserved_variance =
      get_parameter("unobserved_variance").as_double();
    config.orientation_variance_rad2 =
      get_parameter("orientation_variance_rad2").as_double();
    filter_ = std::make_unique<QuaternionWheelGnssEkf>(std::move(config));

    const auto reliable_qos =
      rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
    publisher_ = create_publisher<nav_msgs::msg::Odometry>(
      get_parameter("output_odometry_topic").as_string(), reliable_qos);
    imu_subscription_ = create_subscription<sensor_msgs::msg::Imu>(
      get_parameter("imu_topic").as_string(), rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::Imu::SharedPtr message) {
        filter_->observe_imu(*message);
      });
    wheel_subscription_ = create_subscription<
      geometry_msgs::msg::TwistWithCovarianceStamped>(
      get_parameter("wheel_topic").as_string(), reliable_qos,
      [this](
        const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr message)
      {
        publish(filter_->observe_wheel_speed(*message));
      });
    gnss_subscription_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      get_parameter("gnss_topic").as_string(), reliable_qos,
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr message) {
        publish(filter_->observe_gnss(*message));
      });
    reset_service_ = create_service<std_srvs::srv::Trigger>(
      "/ad/localization/reset_quaternion_wheel_gnss_ekf",
      [this](
        const std_srvs::srv::Trigger::Request::SharedPtr,
        std_srvs::srv::Trigger::Response::SharedPtr response)
      {
        filter_->reset();
        response->success = true;
        response->message = "quaternion_wheel_gnss_ekf reset";
      });
  }

private:
  void publish(const std::optional<nav_msgs::msg::Odometry> & output)
  {
    if (output) {
      publisher_->publish(*output);
    }
  }

  std::unique_ptr<QuaternionWheelGnssEkf> filter_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr
    wheel_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
    gnss_subscription_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;
};

}  // namespace

std::shared_ptr<rclcpp::Node> make_quaternion_wheel_gnss_ekf_node(
  const rclcpp::NodeOptions & options)
{
  return std::make_shared<QuaternionWheelGnssEkfNode>(options);
}

}  // namespace ad_localization
