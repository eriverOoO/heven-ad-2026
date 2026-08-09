#include "ad_localization/imu_quaternion_encoder/imu_quaternion_encoder_node.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ad_localization/imu_quaternion_encoder/imu_quaternion_encoder.hpp"
#include "ad_morai_interfaces/msg/ego_vehicle_status.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
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

class ImuQuaternionEncoderNode final : public rclcpp::Node
{
public:
  explicit ImuQuaternionEncoderNode(const rclcpp::NodeOptions & options)
  : rclcpp::Node("imu_quaternion_encoder", options)
  {
    declare_parameter<std::string>("mode", "status_pose");
    declare_parameter<std::string>("status_topic", "/ad/vehicle/status");
    declare_parameter<std::string>("imu_topic", "/ad/sensors/imu/data");
    declare_parameter<std::string>(
      "gnss_seed_topic", "/ad/localization/input/gnss_pose");
    declare_parameter<std::string>(
      "output_odometry_topic",
      "/ad/localization/backends/imu_quaternion_encoder/odometry");
    declare_parameter<std::string>("status_frame", "map");
    declare_parameter<std::string>("reference_frame", "odom");
    declare_parameter<std::string>("base_frame", "base_link");
    declare_parameter<std::string>("imu_frame", "imu_link");
    declare_parameter<std::vector<double>>(
      "status_origin_to_base_m", {0.0, 0.0, 0.0});
    declare_parameter<std::vector<double>>(
      "gnss_lever_arm_m", {0.0, 0.0, 0.0});
    declare_parameter<bool>("reject_zero_status_position", true);
    declare_parameter<std::vector<double>>(
      "imu_mount_rpy_rad", {0.0, 0.0, 0.0});
    declare_parameter<double>("world_yaw_offset_rad", 0.0);
    declare_parameter<double>("maximum_imu_age_sec", 0.1);
    declare_parameter<double>("maximum_integration_dt_sec", 0.5);
    declare_parameter<int>("initial_seed.sample_count", 1);
    declare_parameter<bool>("automatic_reseed.enabled", false);
    declare_parameter<double>("automatic_reseed.distance_m", 8.0);
    declare_parameter<int>("automatic_reseed.confirmation_samples", 3);
    declare_parameter<double>("automatic_reseed.candidate_radius_m", 4.0);
    declare_parameter<double>("automatic_reseed.max_interval_sec", 0.5);
    declare_parameter<double>("position_variance_m2", 0.01);
    declare_parameter<double>("orientation_variance_rad2", 0.0001);
    declare_parameter<double>("speed_variance_m2ps2", 0.04);
    declare_parameter<bool>("publish_tf", false);
    if (get_parameter("publish_tf").as_bool()) {
      throw std::invalid_argument(
              "imu_quaternion_encoder publish_tf must remain false; localization_manager owns TF");
    }

    const auto offset = get_parameter("status_origin_to_base_m").as_double_array();
    if (offset.size() != 3) {
      throw std::invalid_argument("status_origin_to_base_m must contain exactly three values");
    }
    const auto lever_arm = get_parameter("gnss_lever_arm_m").as_double_array();
    if (lever_arm.size() != 3) {
      throw std::invalid_argument("gnss_lever_arm_m must contain exactly three values");
    }
    const auto imu_mount_rpy = get_parameter("imu_mount_rpy_rad").as_double_array();
    if (imu_mount_rpy.size() != 3) {
      throw std::invalid_argument("imu_mount_rpy_rad must contain exactly three values");
    }
    ImuQuaternionEncoderConfig config;
    config.mode = parse_imu_quaternion_encoder_mode(get_parameter("mode").as_string());
    config.status_frame = get_parameter("status_frame").as_string();
    config.reference_frame = get_parameter("reference_frame").as_string();
    config.base_frame = get_parameter("base_frame").as_string();
    config.imu_frame = get_parameter("imu_frame").as_string();
    std::copy(offset.begin(), offset.end(), config.status_origin_to_base_m.begin());
    std::copy(lever_arm.begin(), lever_arm.end(), config.gnss_lever_arm_m.begin());
    config.reject_zero_status_position =
      get_parameter("reject_zero_status_position").as_bool();
    config.base_to_imu_orientation = quaternion_from_rpy(
      imu_mount_rpy[0], imu_mount_rpy[1], imu_mount_rpy[2]);
    config.world_yaw_offset_rad = get_parameter("world_yaw_offset_rad").as_double();
    config.maximum_imu_age_sec = get_parameter("maximum_imu_age_sec").as_double();
    config.maximum_integration_dt_sec =
      get_parameter("maximum_integration_dt_sec").as_double();
    config.initial_seed_sample_count = static_cast<int>(
      get_parameter("initial_seed.sample_count").as_int());
    config.automatic_reseed_enabled =
      get_parameter("automatic_reseed.enabled").as_bool();
    config.automatic_reseed_distance_m =
      get_parameter("automatic_reseed.distance_m").as_double();
    config.automatic_reseed_confirmation_samples = static_cast<int>(
      get_parameter("automatic_reseed.confirmation_samples").as_int());
    config.automatic_reseed_candidate_radius_m =
      get_parameter("automatic_reseed.candidate_radius_m").as_double();
    config.automatic_reseed_max_interval_sec =
      get_parameter("automatic_reseed.max_interval_sec").as_double();
    config.position_variance_m2 = get_parameter("position_variance_m2").as_double();
    config.orientation_variance_rad2 =
      get_parameter("orientation_variance_rad2").as_double();
    config.speed_variance_m2ps2 = get_parameter("speed_variance_m2ps2").as_double();
    encoder_ = std::make_unique<ImuQuaternionEncoder>(std::move(config));

    const auto reliable_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    publisher_ = create_publisher<nav_msgs::msg::Odometry>(
      get_parameter("output_odometry_topic").as_string(), reliable_qos);
    status_subscription_ = create_subscription<ad_morai_interfaces::msg::EgoVehicleStatus>(
      get_parameter("status_topic").as_string(), rclcpp::SensorDataQoS(),
      std::bind(&ImuQuaternionEncoderNode::status_callback, this, std::placeholders::_1));
    imu_subscription_ = create_subscription<sensor_msgs::msg::Imu>(
      get_parameter("imu_topic").as_string(), rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::Imu::SharedPtr message) {
        encoder_->observe_imu(*message);
      });
    seed_subscription_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      get_parameter("gnss_seed_topic").as_string(), reliable_qos,
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr message) {
        encoder_->observe_gnss_seed(*message);
      });
    reset_service_ = create_service<std_srvs::srv::Trigger>(
      "/ad/localization/reset_imu_quaternion_encoder",
      [this](
        const std_srvs::srv::Trigger::Request::SharedPtr,
        std_srvs::srv::Trigger::Response::SharedPtr response)
      {
        encoder_->reset();
        response->success = true;
        response->message = "imu_quaternion_encoder reset";
      });
  }

private:
  void status_callback(
    const ad_morai_interfaces::msg::EgoVehicleStatus::SharedPtr status)
  {
    const auto output = encoder_->observe_status(*status);
    if (output) {
      publisher_->publish(*output);
    }
  }

  std::unique_ptr<ImuQuaternionEncoder> encoder_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr publisher_;
  rclcpp::Subscription<ad_morai_interfaces::msg::EgoVehicleStatus>::SharedPtr
    status_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr seed_subscription_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;
};

}  // namespace

std::shared_ptr<rclcpp::Node> make_imu_quaternion_encoder_node(
  const rclcpp::NodeOptions & options)
{
  return std::make_shared<ImuQuaternionEncoderNode>(options);
}

}  // namespace ad_localization
